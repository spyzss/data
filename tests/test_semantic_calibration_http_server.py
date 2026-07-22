from __future__ import annotations

import json
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread

import pytest

from qc_common.reviewer_lease import LeaseConflictError, LeaseTokenError
from semantic_calibration.http_server import SEMANTIC_ROUTES, VIDEO_CHUNK_SIZE, create_http_server
from semantic_calibration.service import SemanticEligibilityError


class FakeSemanticApplication:
    def __init__(self) -> None:
        self.revision = 7
        self.token = "lease-token"
        self.calls: list[tuple[str, str | None, dict]] = []

    def list_assets(self):
        return [{"asset_id": "asset-1", "state": "ready"}]

    def get_task(self, asset_id: str):
        if asset_id == "missing":
            raise KeyError(asset_id)
        if asset_id == "blocked":
            raise SemanticEligibilityError("semantic_not_ready")
        if asset_id == "explode":
            raise RuntimeError("/private/secret/report.json traceback detail")
        return {"asset_id": asset_id, "revision": self.revision, "task_type": "semantic_calibration"}

    def acquire_lease(self, asset_id: str, reviewer: str, ttl_seconds: int | None = None):
        if reviewer == "busy":
            raise LeaseConflictError("asset is already leased")
        self.calls.append(("lease_acquire", asset_id, {"reviewer": reviewer}))
        return {"asset_id": asset_id, "reviewer": reviewer, "token": self.token, "expires_at": "later"}

    def renew_lease(self, asset_id: str, lease_token: str, ttl_seconds: int | None = None):
        if lease_token != self.token:
            raise LeaseTokenError("stale lease")
        self.calls.append(("lease_renew", asset_id, {}))
        return {"asset_id": asset_id, "reviewer": "alice", "token": self.token, "expires_at": "later"}

    def release_lease(self, asset_id: str, lease_token: str):
        if lease_token != self.token:
            raise LeaseTokenError("stale lease")
        self.calls.append(("lease_release", asset_id, {}))
        return {"released": True}

    def mutate(self, operation: str, asset_id: str, payload: dict):
        self.calls.append((operation, asset_id, payload))
        return {"asset_id": asset_id, "revision": self.revision + 1, "operation": operation}

    def semantic_video_path(self, asset_id: str):
        if not hasattr(self, "video_path"):
            raise KeyError(asset_id)
        return self.video_path


def _request(server, method: str, path: str, body: dict | None = None):
    connection = HTTPConnection("127.0.0.1", server.server_port)
    payload = None if body is None else json.dumps(body).encode("utf-8")
    connection.request(
        method,
        path,
        body=payload,
        headers={"Content-Type": "application/json"} if payload is not None else {},
    )
    response = connection.getresponse()
    raw = response.read()
    connection.close()
    value = json.loads(raw.decode("utf-8")) if raw else None
    return response.status, value


def _raw_request(server, method: str, path: str, body: bytes = b"", headers: dict | None = None):
    connection = HTTPConnection("127.0.0.1", server.server_port)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    raw = response.read()
    content_type = response.getheader("Content-Type")
    connection.close()
    return response.status, content_type, raw


def _raw_request_details(server, method: str, path: str, body: bytes = b"", headers: dict | None = None):
    connection = HTTPConnection("127.0.0.1", server.server_port)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    raw = response.read()
    details = {
        "status": response.status,
        "content_type": response.getheader("Content-Type"),
        "content_range": response.getheader("Content-Range"),
        "content_length": response.getheader("Content-Length"),
        "allow": response.getheader("Allow"),
        "body": raw,
    }
    connection.close()
    return details


@pytest.fixture
def running_server():
    application = FakeSemanticApplication()
    server = create_http_server("127.0.0.1", 0, application, static_root=None)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield application, server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_route_table_is_explicit_and_semantic_only() -> None:
    assert SEMANTIC_ROUTES == {
        "GET /api/semantic/assets",
        "GET /api/semantic/assets/{asset_id}/task",
        "GET /api/semantic/assets/{asset_id}/video",
        "POST /api/semantic/assets/{asset_id}/lease/acquire",
        "POST /api/semantic/assets/{asset_id}/lease/renew",
        "POST /api/semantic/assets/{asset_id}/lease/release",
        "POST /api/semantic/assets/{asset_id}/boundary/pending",
        "POST /api/semantic/assets/{asset_id}/text/pending",
        "POST /api/semantic/assets/{asset_id}/pending/confirm",
        "POST /api/semantic/assets/{asset_id}/pending/cancel",
        "POST /api/semantic/assets/{asset_id}/complete",
    }


def test_list_task_lease_release_and_mutation_routes(running_server) -> None:
    application, server = running_server
    status, value = _request(server, "GET", "/api/semantic/assets")
    assert status == 200 and value["assets"][0]["asset_id"] == "asset-1"
    status, value = _request(server, "GET", "/api/semantic/assets/asset-1/task")
    assert status == 200 and value["task"]["revision"] == 7

    status, lease = _request(
        server,
        "POST",
        "/api/semantic/assets/asset-1/lease/acquire",
        {"reviewer": "alice", "ttl_seconds": 60},
    )
    assert status == 200 and lease["lease"]["token"] == application.token
    common = {"expected_revision": 7, "lease_token": application.token}
    status, _ = _request(server, "POST", "/api/semantic/assets/asset-1/lease/renew", common)
    assert status == 200
    status, _ = _request(
        server,
        "POST",
        "/api/semantic/assets/asset-1/lease/release",
        {"lease_token": application.token},
    )
    assert status == 200

    payloads = {
        "boundary/pending": {**common, "boundary_index": 1, "new_frame_exclusive": 42},
        "text/pending": {**common, "segment_id": "s1", "text_cn": "甲", "text_en": "a"},
        "pending/confirm": common,
        "pending/cancel": common,
        "complete": common,
    }
    for route, payload in payloads.items():
        status, value = _request(
            server,
            "POST",
            f"/api/semantic/assets/asset-1/{route}",
            payload,
        )
        assert status == 200, (route, value)


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/api/warn/assets"),
        ("POST", "/api/warn/assets/asset-1/complete"),
        ("GET", "/api/assets/asset-1/task"),
        ("GET", "/evidence/private/file.png"),
        ("GET", "/api/semantic/unknown"),
    ],
)
def test_non_semantic_routes_are_not_exposed(running_server, method: str, path: str) -> None:
    _, server = running_server
    status, value = _request(server, method, path, {})
    assert status == 404
    assert value["error"]["code"] == "not_found"


def test_stable_conflicts_do_not_leak_internal_details(running_server) -> None:
    _, server = running_server
    status, value = _request(server, "GET", "/api/semantic/assets/blocked/task")
    assert status == 409 and value["error"]["code"] == "semantic_not_ready"

    status, value = _request(server, "GET", "/api/semantic/assets/explode/task")
    encoded = json.dumps(value)
    assert status == 500 and value["error"]["code"] == "internal_error"
    assert "/private/" not in encoded and "traceback" not in encoded.lower()

    status, value = _request(
        server,
        "POST",
        "/api/semantic/assets/asset-1/lease/acquire",
        {"reviewer": "busy"},
    )
    assert status == 423 and value["error"]["code"] == "lease_held"


@pytest.mark.parametrize("method", ["OPTIONS", "PUT", "PATCH", "DELETE", "TRACE", "BREW"])
def test_unsupported_methods_return_stable_json_405(running_server, method: str) -> None:
    _, server = running_server
    status, content_type, raw = _raw_request(
        server,
        method,
        "/api/semantic/assets/asset-1/task",
        b"{}",
        {"Content-Type": "application/json"},
    )
    value = json.loads(raw)
    assert status == 405
    assert content_type.startswith("application/json")
    assert value["error"]["code"] == "method_not_allowed"


def test_head_returns_stable_json_media_type_and_405_without_a_body(running_server) -> None:
    _, server = running_server
    response = _raw_request_details(server, "HEAD", "/api/semantic/assets/asset-1/task")
    assert response["status"] == 405
    assert response["content_type"].startswith("application/json")
    assert response["body"] == b""


def test_post_requires_json_media_type_before_any_mutation(running_server) -> None:
    application, server = running_server
    before = list(application.calls)
    status, _, raw = _raw_request(
        server,
        "POST",
        "/api/semantic/assets/asset-1/lease/acquire",
        b'{"reviewer":"alice"}',
        {"Content-Type": "text/plain"},
    )
    assert status == 415
    assert json.loads(raw)["error"]["code"] == "unsupported_media_type"
    assert application.calls == before

    status, _, _ = _raw_request(
        server,
        "POST",
        "/api/semantic/assets/asset-1/lease/acquire",
        b'{"reviewer":"alice"}',
        {"Content-Type": "application/json; charset=utf-8"},
    )
    assert status == 200


def test_unknown_post_is_404_before_content_type_validation(running_server) -> None:
    _, server = running_server
    status, _, raw = _raw_request(
        server,
        "POST",
        "/api/semantic/assets/asset-1/not-a-route",
        b"plain",
        {"Content-Type": "text/plain"},
    )
    assert status == 404
    assert json.loads(raw)["error"]["code"] == "not_found"


def test_semantic_video_route_is_contained_and_supports_browser_ranges(
    running_server, tmp_path: Path
) -> None:
    application, server = running_server
    video = tmp_path / "asset.mp4"
    video.write_bytes(b"0123456789")
    application.video_path = video

    status, content_type, body = _raw_request(
        server,
        "GET",
        "/api/semantic/assets/asset-1/video",
        headers={"Range": "bytes=2-5"},
    )
    assert status == 206
    assert content_type == "video/mp4"
    assert body == b"2345"


@pytest.mark.parametrize(
    "header",
    ["bytes=0", "bytes=+1-", "bytes=0-1,3-4", "items=0-1", "bytes=-0", "bytes=20-30"],
)
def test_invalid_or_unsatisfiable_ranges_return_416_with_standard_size_header(
    running_server, tmp_path: Path, header: str
) -> None:
    application, server = running_server
    video = tmp_path / "asset.mp4"
    video.write_bytes(b"0123456789")
    application.video_path = video
    response = _raw_request_details(
        server,
        "GET",
        "/api/semantic/assets/asset-1/video",
        headers={"Range": header},
    )
    assert response["status"] == 416
    assert response["content_range"] == "bytes */10"
    assert json.loads(response["body"])["error"]["code"] == "invalid_range"


def test_suffix_and_long_end_ranges_are_normalized(running_server, tmp_path: Path) -> None:
    application, server = running_server
    video = tmp_path / "asset.mp4"
    video.write_bytes(b"0123456789")
    application.video_path = video
    suffix = _raw_request_details(
        server, "GET", "/api/semantic/assets/asset-1/video", headers={"Range": "bytes=-3"}
    )
    assert suffix["status"] == 206
    assert suffix["content_range"] == "bytes 7-9/10"
    assert suffix["body"] == b"789"
    long_end = _raw_request_details(
        server, "GET", "/api/semantic/assets/asset-1/video", headers={"Range": "bytes=2-999"}
    )
    assert long_end["content_range"] == "bytes 2-9/10"
    assert long_end["body"] == b"23456789"


def test_zero_byte_video_is_safe_and_any_range_is_unsatisfiable(running_server, tmp_path: Path) -> None:
    application, server = running_server
    video = tmp_path / "empty.mp4"
    video.write_bytes(b"")
    application.video_path = video
    full = _raw_request_details(server, "GET", "/api/semantic/assets/asset-1/video")
    assert full["status"] == 200
    assert full["content_length"] == "0"
    assert full["body"] == b""
    ranged = _raw_request_details(
        server, "GET", "/api/semantic/assets/asset-1/video", headers={"Range": "bytes=0-0"}
    )
    assert ranged["status"] == 416
    assert ranged["content_range"] == "bytes */0"


def test_video_response_reads_in_bounded_chunks(running_server, tmp_path: Path, monkeypatch) -> None:
    application, server = running_server
    video = tmp_path / "large.mp4"
    video.write_bytes(b"x" * (VIDEO_CHUNK_SIZE * 2 + 17))
    application.video_path = video
    original_open = Path.open
    read_sizes: list[int] = []

    class TrackingReader:
        def __init__(self, wrapped) -> None:
            self.wrapped = wrapped

        def __enter__(self):
            self.wrapped.__enter__()
            return self

        def __exit__(self, *args):
            return self.wrapped.__exit__(*args)

        def seek(self, *args):
            return self.wrapped.seek(*args)

        def read(self, size=-1):
            read_sizes.append(size)
            return self.wrapped.read(size)

    def tracking_open(path: Path, *args, **kwargs):
        wrapped = original_open(path, *args, **kwargs)
        return TrackingReader(wrapped) if path == video else wrapped

    monkeypatch.setattr(Path, "open", tracking_open)
    response = _raw_request_details(server, "GET", "/api/semantic/assets/asset-1/video")
    assert response["status"] == 200
    assert len(response["body"]) == VIDEO_CHUNK_SIZE * 2 + 17
    assert len(read_sizes) >= 3
    assert max(read_sizes) <= VIDEO_CHUNK_SIZE
