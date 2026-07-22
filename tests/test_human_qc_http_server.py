from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread

import pytest

from human_qc.http_server import create_http_server
from human_qc.warn_service import WarnRevisionError
from qc_common.reviewer_lease import Lease, LeaseConflictError, LeaseStore, LeaseTokenError


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 7, 14, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


def test_lease_expiry_renewal_and_release() -> None:
    clock = Clock()
    store = LeaseStore(clock=clock)
    first = store.acquire("asset-1", "alice", ttl_seconds=10)
    with pytest.raises(LeaseConflictError):
        store.acquire("asset-1", "bob", ttl_seconds=10)
    renewed = store.renew("asset-1", first.token, ttl_seconds=20)
    assert renewed.token == first.token
    released = store.release("asset-1", first.token)
    assert released.token == first.token
    with pytest.raises(LeaseTokenError):
        store.validate("asset-1", first.token)


class FakeFacade:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.task = {
            "asset_id": "asset-1",
            "report_revision": 3,
            "manual_review_state": "queued",
            "completion_mode": None,
            "failure_reason": None,
            "video": {
                "url": "/media/assets/asset-1/source",
                "fps": 30.0,
                "total_frames": 1800,
            },
            "issues": [],
            "reason_options": [],
            "lease": {
                "read_only": False,
                "token": "lease-1",
                "expires_at": "2026-07-22T01:00:00+00:00",
                "code": None,
            },
        }
        self.raise_conflict = False
        self.raise_stale = False

    def list_assets(self):
        return ("asset-1",)

    def get_asset_task(self, asset_id: str, *, lease_token: str | None = None):
        self.calls.append(("get_asset_task", asset_id, lease_token))
        if asset_id != "asset-1":
            raise KeyError("/private/batch/missing.json")
        return self.task

    def current_revision(self, asset_id: str):
        self.calls.append(("current_revision", asset_id))
        return 3 if asset_id == "asset-1" else None

    def acquire_lease(self, asset_id: str, ttl_seconds: int | None = None):
        self.calls.append(("acquire_lease", asset_id, ttl_seconds))
        if self.raise_conflict:
            raise LeaseConflictError("asset held by reviewer=bob token=secret")
        return Lease(asset_id, "alice", "lease-1", "2026-07-22T01:00:00+00:00")

    def renew_lease(self, asset_id: str, token: str, ttl_seconds: int | None = None):
        self.calls.append(("renew_lease", asset_id, token, ttl_seconds))
        return Lease(asset_id, "alice", token, "2026-07-22T01:00:00+00:00")

    def release_lease(self, asset_id: str, token: str):
        self.calls.append(("release_lease", asset_id, token))
        return Lease(asset_id, "alice", token, "2026-07-22T01:00:00+00:00")

    def validate_lease(self, asset_id: str, token: str):
        self.calls.append(("validate_lease", asset_id, token))
        if token != "lease-1":
            raise LeaseTokenError("invalid token secret-token")
        return Lease(asset_id, "alice", token, "2026-07-22T01:00:00+00:00")

    def warn_verdict(self, asset_id: str, **payload):
        self.calls.append(("warn_verdict", asset_id, payload))
        if self.raise_stale:
            raise WarnRevisionError(
                "expected revision from /private/batch; command=cat secret; traceback"
            )
        return self.task

    def warn_complete(self, asset_id: str, **payload):
        self.calls.append(("warn_complete", asset_id, payload))
        return self.task


def _request_raw(
    server,
    method: str,
    path: str,
    body: dict | None = None,
    *,
    headers: dict[str, str] | None = None,
):
    connection = HTTPConnection("127.0.0.1", server.server_port)
    payload = None if body is None else json.dumps(body).encode("utf-8")
    request_headers = dict(headers or {})
    if payload is not None:
        request_headers["Content-Type"] = "application/json"
    connection.request(method, path, body=payload, headers=request_headers)
    response = connection.getresponse()
    value = response.read()
    result_headers = {name: value for name, value in response.getheaders()}
    status = response.status
    connection.close()
    return status, result_headers, value


def _request_json(server, method: str, path: str, body: dict | None = None, **kwargs):
    status, headers, raw = _request_raw(server, method, path, body, **kwargs)
    return status, headers, json.loads(raw.decode("utf-8"))


def _running_server(facade):
    server = create_http_server("127.0.0.1", 0, facade)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _stop(server, thread: Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def test_warn_routes_are_canonical_and_task_header_auto_renews() -> None:
    facade = FakeFacade()
    server, thread = _running_server(facade)
    try:
        status, _, assets = _request_json(server, "GET", "/api/warn/assets")
        assert status == 200 and assets == {"assets": ["asset-1"]}

        status, _, value = _request_json(
            server,
            "GET",
            "/api/warn/assets/asset-1/task",
            headers={"X-Reviewer-Lease": "lease-1"},
        )
        assert status == 200
        assert value["task"]["video"]["fps"] == 30.0
        assert ("get_asset_task", "asset-1", "lease-1") in facade.calls

        for old_path in (
            "/api/assets/asset-1/task",
            "/api/semantic/assets/asset-1/task",
            "/evidence/sam3/frame.png",
            "/api/warn/assets/asset-1/overlays/warn-1/status",
        ):
            status, _, value = _request_json(server, "GET", old_path)
            assert status == 404
            assert value["error"]["code"] == "not_found"
    finally:
        _stop(server, thread)


def test_warn_mutation_routes_forward_complete_payload_and_release() -> None:
    facade = FakeFacade()
    server, thread = _running_server(facade)
    try:
        status, _, lease = _request_json(
            server,
            "POST",
            "/api/warn/assets/asset-1/lease/acquire",
            {"ttl_seconds": 60, "reviewer": "mallory"},
        )
        assert status == 200 and lease["lease"]["token"] == "lease-1"
        assert ("acquire_lease", "asset-1", 60) in facade.calls

        status, _, renewed = _request_json(
            server,
            "POST",
            "/api/warn/assets/asset-1/lease/renew",
            {"expected_revision": 3, "lease_token": "lease-1", "ttl_seconds": 60},
        )
        assert status == 200 and renewed["lease"]["token"] == "lease-1"

        failure_reason = {"reason_codes": ["other"], "other_text": "动作不可辨"}
        status, _, value = _request_json(
            server,
            "POST",
            "/api/warn/assets/asset-1/issues/warn-1/verdict",
            {
                "expected_revision": 3,
                "lease_token": "lease-1",
                "verdict": "fail",
                "reason": "occlusion",
                "failure_reason": failure_reason,
            },
        )
        assert status == 200 and value["task"]["asset_id"] == "asset-1"
        verdict = [call for call in facade.calls if call[0] == "warn_verdict"][-1]
        assert verdict[2] == {
            "expected_revision": 3,
            "lease_token": "lease-1",
            "issue_id": "warn-1",
            "verdict": "fail",
            "reason": "occlusion",
            "failure_reason": failure_reason,
        }

        status, _, _ = _request_json(
            server,
            "POST",
            "/api/warn/assets/asset-1/complete",
            {
                "expected_revision": 3,
                "lease_token": "lease-1",
                "completion_mode": "early_fail",
                "failure_reason": failure_reason,
            },
        )
        assert status == 200
        complete = [call for call in facade.calls if call[0] == "warn_complete"][-1]
        assert complete[2]["completion_mode"] == "early_fail"
        assert complete[2]["failure_reason"] == failure_reason

        status, _, released = _request_json(
            server,
            "POST",
            "/api/warn/assets/asset-1/lease/release",
            {"expected_revision": 3, "lease_token": "lease-1"},
        )
        assert status == 200
        assert released == {"released": True}
        assert ("release_lease", "asset-1", "lease-1") in facade.calls
    finally:
        _stop(server, thread)


def test_issue_id_mismatch_is_rejected_before_mutation() -> None:
    facade = FakeFacade()
    server, thread = _running_server(facade)
    try:
        status, _, value = _request_json(
            server,
            "POST",
            "/api/warn/assets/asset-1/issues/warn-1/verdict",
            {
                "issue_id": "warn-2",
                "expected_revision": 3,
                "lease_token": "lease-1",
                "verdict": "pass",
            },
        )
        assert status == 400 and value["error"]["code"] == "bad_request"
        assert not any(call[0] == "warn_verdict" for call in facade.calls)
    finally:
        _stop(server, thread)


def test_stale_and_lease_errors_are_stable_and_sanitized() -> None:
    facade = FakeFacade()
    server, thread = _running_server(facade)
    try:
        facade.raise_stale = True
        status, _, value = _request_json(
            server,
            "POST",
            "/api/warn/assets/asset-1/issues/warn-1/verdict",
            {
                "expected_revision": 3,
                "lease_token": "lease-1",
                "verdict": "pass",
            },
        )
        assert status == 409
        assert value["error"] == {
            "code": "stale_revision",
            "message": "the report revision is stale",
            "current_revision": 3,
        }
        serialized = json.dumps(value)
        assert "/private/" not in serialized
        assert "command" not in serialized
        assert "traceback" not in serialized

        facade.raise_conflict = True
        status, _, value = _request_json(
            server,
            "POST",
            "/api/warn/assets/asset-1/lease/acquire",
            {},
        )
        assert status == 423
        assert value["error"]["code"] == "lease_held"
        assert "bob" not in json.dumps(value)
        assert "secret" not in json.dumps(value)

        status, _, value = _request_json(
            server,
            "POST",
            "/api/warn/assets/asset-1/complete",
            {"expected_revision": 3, "lease_token": "wrong"},
        )
        assert status == 423
        assert value["error"]["code"] == "lease_invalid"
        assert "wrong" not in json.dumps(value)
    finally:
        _stop(server, thread)


def test_source_media_supports_full_and_strict_single_ranges(tmp_path: Path) -> None:
    from tests.test_human_qc_workbench import _service

    service, _, _, context = _service(tmp_path)
    source = context.batch_root / context.source_files["video"]["path"]
    expected = source.read_bytes()
    server, thread = _running_server(service)
    try:
        status, headers, body = _request_raw(
            server, "GET", "/media/assets/asset-1/source"
        )
        assert status == 200
        assert body == expected
        assert headers["Content-Length"] == str(len(expected))
        assert headers["Accept-Ranges"] == "bytes"

        status, headers, body = _request_raw(
            server,
            "GET",
            "/media/assets/asset-1/source",
            headers={"Range": "bytes=0-99"},
        )
        assert status == 206
        assert body == expected[:100]
        assert headers["Content-Range"] == f"bytes 0-99/{len(expected)}"
        assert headers["Content-Length"] == "100"

        status, headers, body = _request_raw(
            server,
            "GET",
            "/media/assets/asset-1/source",
            headers={"Range": "bytes=-10"},
        )
        assert status == 206 and body == expected[-10:]

        status, headers, body = _request_raw(
            server,
            "GET",
            "/media/assets/asset-1/source",
            headers={"Range": "bytes=0-1,4-5"},
        )
        assert status == 416
        assert headers["Content-Range"] == f"bytes */{len(expected)}"
        assert json.loads(body)["error"]["code"] == "range_not_satisfiable"
    finally:
        _stop(server, thread)


def test_source_media_head_is_bodyless_for_full_range_and_416(tmp_path: Path) -> None:
    from tests.test_human_qc_workbench import _service

    service, _, _, context = _service(tmp_path)
    source = context.batch_root / context.source_files["video"]["path"]
    size = source.stat().st_size
    server, thread = _running_server(service)
    try:
        status, headers, body = _request_raw(
            server, "HEAD", "/media/assets/asset-1/source"
        )
        assert status == 200
        assert body == b""
        assert headers["Content-Length"] == str(size)
        assert headers["Accept-Ranges"] == "bytes"

        status, headers, body = _request_raw(
            server,
            "HEAD",
            "/media/assets/asset-1/source",
            headers={"Range": "bytes=0-99"},
        )
        assert status == 206
        assert body == b""
        assert headers["Content-Length"] == "100"
        assert headers["Content-Range"] == f"bytes 0-99/{size}"

        status, headers, body = _request_raw(
            server,
            "HEAD",
            "/media/assets/asset-1/source",
            headers={"Range": "bytes=0-1,4-5"},
        )
        assert status == 416
        assert body == b""
        assert headers["Content-Range"] == f"bytes */{size}"
        assert headers["Content-Type"].startswith("application/json")
    finally:
        _stop(server, thread)


def test_known_route_unsupported_method_is_json_405_and_unknown_is_404() -> None:
    facade = FakeFacade()
    server, thread = _running_server(facade)
    try:
        status, headers, value = _request_json(
            server, "PUT", "/api/warn/assets/asset-1/complete", {}
        )
        assert status == 405
        assert headers["Content-Type"].startswith("application/json")
        assert headers["Allow"] == "POST"
        assert value["error"]["code"] == "method_not_allowed"

        status, headers, value = _request_json(
            server, "BREW", "/api/warn/assets/asset-1/complete", {}
        )
        assert status == 405
        assert headers["Content-Type"].startswith("application/json")
        assert value["error"]["code"] == "method_not_allowed"

        status, headers, value = _request_json(
            server, "PUT", "/api/warn/assets/asset-1/not-a-route", {}
        )
        assert status == 404
        assert headers["Content-Type"].startswith("application/json")
        assert value["error"]["code"] == "not_found"
    finally:
        _stop(server, thread)


def test_hard_reload_recovers_fixed_reviewer_lease_via_opaque_session(
    tmp_path: Path,
) -> None:
    from tests.test_human_qc_workbench import _service

    service, _, _, _ = _service(tmp_path)
    server, thread = _running_server(service)
    try:
        fixed_session = "a" * 32
        status, headers, first = _request_json(
            server,
            "GET",
            "/api/warn/assets/asset-1/task",
            headers={"Cookie": f"warn_reviewer_session={fixed_session}"},
        )
        assert status == 200
        token = first["task"]["lease"]["token"]
        cookie = headers["Set-Cookie"].split(";", 1)[0]
        assert token not in cookie
        assert "alice" not in cookie
        assert fixed_session not in cookie

        status, _, reloaded = _request_json(
            server,
            "GET",
            "/api/warn/assets/asset-1/task",
            headers={"Cookie": cookie},
        )
        assert status == 200
        assert reloaded["task"]["lease"]["read_only"] is False
        assert reloaded["task"]["lease"]["token"] == token
    finally:
        _stop(server, thread)


def test_source_replacement_between_catalog_and_stream_is_rejected(tmp_path: Path) -> None:
    from tests.test_human_qc_workbench import _service

    service, _, _, context = _service(tmp_path)
    source = context.batch_root / context.source_files["video"]["path"]

    class ReplacingSourceFacade:
        def __getattr__(self, name: str):
            return getattr(service, name)

        def source_media(self, asset_id: str):
            resource = service.source_media(asset_id)
            replacement = source.with_suffix(".replacement")
            replacement.write_bytes(b"z" * resource.size)
            replacement.replace(source)
            return resource

    server, thread = _running_server(ReplacingSourceFacade())
    try:
        status, headers, value = _request_json(
            server, "GET", "/media/assets/asset-1/source"
        )
        assert status == 404
        assert headers["Content-Type"].startswith("application/json")
        assert value["error"]["code"] == "source_video_unavailable"
        assert "z" * 32 not in json.dumps(value)
    finally:
        _stop(server, thread)


def test_overlay_media_is_allowlisted_per_asset(tmp_path: Path) -> None:
    from tests.test_human_qc_workbench import _service

    service, _, _, _ = _service(tmp_path)
    overlay = tmp_path / "overlays" / "ready.mp4"
    overlay.parent.mkdir()
    overlay.write_bytes(b"ready-overlay")
    unlisted = overlay.parent / "unlisted.mp4"
    unlisted.write_bytes(b"secret-overlay")
    service.media_catalog.allow_overlay("asset-1", "overlay-opaque-1", overlay)
    server, thread = _running_server(service)
    try:
        status, _, body = _request_raw(
            server,
            "GET",
            "/media/assets/asset-1/overlays/overlay-opaque-1",
        )
        assert status == 200 and body == b"ready-overlay"
        for path in (
            "/media/assets/asset-1/overlays/unlisted.mp4",
            "/media/assets/other-asset/overlays/overlay-opaque-1",
            "/media/assets/asset-1/overlays/..%2Funlisted.mp4",
        ):
            status, _, value = _request_json(server, "GET", path)
            assert status == 404
            assert value["error"]["code"] == "not_found"
            assert "secret-overlay" not in json.dumps(value)
    finally:
        _stop(server, thread)
