from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from http.client import HTTPConnection
from threading import Thread

import pytest

from human_qc.http_server import create_http_server
from human_qc.lease import LeaseConflictError, LeaseStore, LeaseTokenError
from human_qc.workbench_service import WorkbenchService


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 7, 14, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


def test_lease_expiry_and_renewal() -> None:
    clock = Clock()
    store = LeaseStore(clock=clock)
    first = store.acquire("asset-1", "alice", ttl_seconds=10)
    with pytest.raises(LeaseConflictError):
        store.acquire("asset-1", "bob", ttl_seconds=10)
    renewed = store.renew("asset-1", first.token, ttl_seconds=20)
    assert renewed.token == first.token
    clock.advance(21)
    with pytest.raises(LeaseTokenError):
        store.renew("asset-1", first.token, ttl_seconds=10)
    second = store.acquire("asset-1", "bob", ttl_seconds=10)
    assert second.reviewer == "bob"


class FakeFacade:
    def __init__(self) -> None:
        self.calls = []
        self.task = {"asset_id": "asset-1", "revision": 3, "state": "ready"}

    def get_asset_task(self, asset_id: str):
        if asset_id != "asset-1":
            raise KeyError(asset_id)
        return self.task

    def current_revision(self, asset_id: str):
        return self.task["revision"] if asset_id == "asset-1" else None

    def acquire_lease(self, asset_id: str, reviewer: str, ttl_seconds: int):
        return self.lease_store.acquire(asset_id, reviewer, ttl_seconds)

    def renew_lease(self, asset_id: str, token: str, ttl_seconds: int):
        return self.lease_store.renew(asset_id, token, ttl_seconds)

    def mutate(self, name: str, asset_id: str, payload: dict):
        self.calls.append((name, asset_id, payload))
        return {**self.task, "revision": self.task["revision"] + 1, "operation": name}


def _server(facade: FakeFacade):
    facade.lease_store = LeaseStore()
    return create_http_server("127.0.0.1", 0, facade)


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
    value = json.loads(response.read().decode("utf-8"))
    connection.close()
    return response.status, value


def test_http_success_and_bad_payload_statuses() -> None:
    facade = FakeFacade()
    server = _server(facade)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, value = _request(server, "GET", "/api/assets/asset-1/task")
        assert status == 200 and value["task"]["revision"] == 3
        status, value = _request(server, "POST", "/api/assets/asset-1/semantic/complete", {})
        assert status == 400 and value["error"]["code"] == "bad_request"
        status, value = _request(server, "GET", "/api/assets/missing/task")
        assert status == 404 and value["error"]["code"] == "not_found"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_serves_sampled_sam3_overlay_evidence(tmp_path) -> None:
    overlay = tmp_path / "sam3" / "combined_overlays" / "frame-10.png"
    overlay.parent.mkdir(parents=True)
    overlay.write_bytes(b"sampled-sam3-overlay")
    facade = FakeFacade()
    facade.lease_store = LeaseStore()
    server = create_http_server(
        "127.0.0.1",
        0,
        facade,
        evidence_root=tmp_path,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection("127.0.0.1", server.server_port)
        connection.request("GET", "/evidence/sam3/combined_overlays/frame-10.png")
        response = connection.getresponse()
        body = response.read()
        connection.close()
        assert response.status == 200
        assert response.getheader("Content-Type") == "image/png"
        assert body == b"sampled-sam3-overlay"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_lease_and_mutation_delegate() -> None:
    facade = FakeFacade()
    server = _server(facade)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, lease = _request(
            server,
            "POST",
            "/api/assets/asset-1/lease/acquire",
            {"reviewer": "alice", "ttl_seconds": 60},
        )
        assert status == 200 and lease["lease"]["token"]
        token = lease["lease"]["token"]
        status, renewed = _request(
            server,
            "POST",
            "/api/assets/asset-1/lease/renew",
            {"expected_revision": 3, "lease_token": token, "ttl_seconds": 60},
        )
        assert status == 200 and renewed["lease"]["token"] == token
        status, value = _request(
            server,
            "POST",
            "/api/assets/asset-1/semantic/complete",
            {"expected_revision": 3, "lease_token": token},
        )
        assert status == 200 and value["task"]["operation"] == "semantic_complete"
        assert facade.calls[-1][0] == "semantic_complete"
        status, value = _request(
            server,
            "POST",
            "/api/assets/asset-1/warn/warn-1/verdict",
            {
                "expected_revision": 3,
                "lease_token": token,
                "verdict": "pass",
            },
        )
        assert status == 200 and value["task"]["operation"] == "warn_verdict"
        assert facade.calls[-1][2]["issue_id"] == "warn-1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
