import http.client
import json
import threading
from pathlib import Path

from tools.serve_manual_review import create_server


def post_json(port: int, path: str, payload: dict) -> tuple[int, dict]:
    body = json.dumps(payload).encode("utf-8")
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request(
            "POST",
            path,
            body,
            {
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
        )
        response = conn.getresponse()
        response_body = response.read().decode("utf-8")
    finally:
        conn.close()
    return response.status, json.loads(response_body)


def run_server_in_thread(review_dir: Path, save_dir: Path):
    server = create_server(
        host="127.0.0.1",
        port=0,
        review_dir=review_dir,
        save_dir=save_dir,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_manual_review_server_post_writes_autosave_files(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    save_dir = tmp_path / "manual_review"
    review_dir.mkdir()
    payload = {
        "run_label": "smoke20",
        "reviewer": "nathan",
        "manual_labels_csv": "review_id,segment_id\nrq_001,rq_001_seg_001\n",
        "progress_json": {"segmentsByReviewId": {"rq_001": [{"segment_id": "rq_001_seg_001"}]}},
        "source": "autosave",
    }
    server, thread = run_server_in_thread(review_dir, save_dir)
    try:
        status, response = post_json(server.server_address[1], "/api/manual-review/save", payload)
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert status == 200
    assert response["ok"] is True
    assert (save_dir / "manual_labels_autosave.csv").read_text(encoding="utf-8") == payload[
        "manual_labels_csv"
    ]
    assert json.loads((save_dir / "manual_review_progress_autosave.json").read_text()) == payload[
        "progress_json"
    ]
    meta = json.loads((save_dir / "manual_review_autosave_meta.json").read_text())
    assert meta["reviewer"] == "nathan"
    assert meta["run_label"] == "smoke20"
    assert meta["source"] == "autosave"
    assert meta["row_count"] == 1


def test_manual_review_server_explicit_save_writes_timestamped_backups(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    save_dir = tmp_path / "manual_review"
    review_dir.mkdir()
    payload = {
        "run_label": "smoke20",
        "reviewer": "nathan",
        "manual_labels_csv": "review_id,segment_id\nrq_001,rq_001_seg_001\n",
        "progress_json": {"saved": True},
        "source": "explicit_save",
    }
    server, thread = run_server_in_thread(review_dir, save_dir)
    try:
        status, response = post_json(server.server_address[1], "/api/manual-review/save", payload)
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert status == 200
    assert response["ok"] is True
    assert list(save_dir.glob("manual_labels_*.csv"))
    assert list(save_dir.glob("manual_review_progress_*.json"))


def test_manual_review_server_rejects_other_post_paths_and_never_uses_payload_paths(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    save_dir = tmp_path / "manual_review"
    review_dir.mkdir()
    payload = {
        "run_label": "../../escape",
        "reviewer": "nathan",
        "manual_labels_csv": "review_id,segment_id\nrq_001,rq_001_seg_001\n",
        "progress_json": {"path": "../escape"},
        "source": "explicit_save",
    }
    server, thread = run_server_in_thread(review_dir, save_dir)
    try:
        bad_status, _bad_response = post_json(server.server_address[1], "/api/manual-review/../save", payload)
        good_status, _good_response = post_json(
            server.server_address[1], "/api/manual-review/save", payload
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert bad_status == 404
    assert good_status == 200
    assert (save_dir / "manual_labels_autosave.csv").exists()
    assert not (tmp_path / "escape").exists()
