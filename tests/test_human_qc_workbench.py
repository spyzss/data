from __future__ import annotations

import json
from http.client import HTTPConnection
from threading import Thread

from human_qc.http_server import create_http_server


class _Facade:
    def __init__(self) -> None:
        self.task = {
            "asset_id": "asset-1",
            "revision": 1,
            "task_type": "semantic_calibration",
            "semantic": {
                "report_revision": 1,
                "report_state": "in_progress",
                "timeline": {
                    "frame_count": 90,
                    "fps": 30,
                    "segments": [
                        {"internal_id": "s1", "start_frame": 0, "end_frame_exclusive": 30},
                        {"internal_id": "s2", "start_frame": 30, "end_frame_exclusive": 60},
                        {"internal_id": "s3", "start_frame": 60, "end_frame_exclusive": 90},
                    ],
                },
                "pending_edit": None,
            },
        }

    def get_asset_task(self, asset_id: str):
        if asset_id != "asset-1":
            raise KeyError(asset_id)
        return self.task

    def current_revision(self, asset_id: str):
        return self.task["revision"] if asset_id == "asset-1" else None


def _request(server, method: str, path: str):
    connection = HTTPConnection("127.0.0.1", server.server_port)
    connection.request(method, path)
    response = connection.getresponse()
    body = response.read()
    connection.close()
    return response.status, body


def test_static_workbench_is_served_as_warn_only() -> None:
    server = create_http_server("127.0.0.1", 0, _Facade())
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _request(server, "GET", "/")
        assert status == 200
        text = body.decode("utf-8")
        assert "warn_adapter.js" in text
        assert "workbench.css" in text
        assert "data-workbench-stage" in text
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_static_modules_expose_warn_only_contracts() -> None:
    from pathlib import Path

    root = Path(__file__).parents[1] / "human_qc" / "static"
    app = (root / "app.js").read_text(encoding="utf-8")
    warn_adapter = (root / "warn_adapter.js").read_text(encoding="utf-8")
    assert "semantic_adapter" not in app
    assert "submitBoundary" not in app
    assert "/warn/${encodeURIComponent(issueId)}/verdict" in app
    assert "data-machine-reason" in warn_adapter
    assert "data-machine-metrics" in warn_adapter
    assert "data-machine-threshold" in warn_adapter
    assert "data-evidence-window" in warn_adapter
    assert "data-overlay-error" in warn_adapter
    assert "videoPlaceholder" in warn_adapter
    assert '[data-video-placeholder]' in app
