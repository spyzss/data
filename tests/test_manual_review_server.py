import csv
import http.client
import io
import json
import threading
from pathlib import Path

import pytest

import tools.serve_manual_review as manual_review_server
from tools.build_video_review_clips import VIDEO_MANUAL_LABEL_COLUMNS
from tools.serve_manual_review import create_server, save_manual_review_payload


def test_server_manual_label_schema_matches_frontend_export_schema() -> None:
    assert manual_review_server.MANUAL_LABEL_COLUMNS == tuple(
        VIDEO_MANUAL_LABEL_COLUMNS
    )


def manual_labels_csv(review_id: str = "rq_001") -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=VIDEO_MANUAL_LABEL_COLUMNS)
    writer.writeheader()
    writer.writerow(
        {
            "review_id": review_id,
            "segment_id": f"{review_id}_seg_001",
            "supplier_id": "supplier_a",
            "asset_id": "asset_a",
            "window_start_frame": 1,
            "window_end_frame": 10,
            "affected_start_frame": 2,
            "affected_end_frame": 4,
            "manual_outcome": "true_positive",
            "failure_mode": "severe_keypoint_offset",
            "severity": "high",
            "confidence": "medium",
            "acceptance_status": "rejected",
            "reviewer": "nathan",
        }
    )
    return output.getvalue()


def progress_json(review_id: str = "rq_001") -> dict:
    return {
        "version": 2,
        "segmentsByReviewId": {
            review_id: [{"segment_id": f"{review_id}_seg_001"}]
        },
        "sampledFrameIndexByReviewId": {review_id: 0},
    }


def valid_payload(review_id: str = "rq_001", source: str = "autosave") -> dict:
    return {
        "run_label": "smoke20",
        "reviewer": "nathan",
        "manual_labels_csv": manual_labels_csv(review_id),
        "progress_json": progress_json(review_id),
        "source": source,
        "source_page": (
            "https://gateway.example/ide/proxy/8899/"
            "video_review/review_index_video.html"
        ),
        "schema_version": "manual_review_progress.v2",
        "review_item_count": 1,
    }


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


def post_raw(
    port: int,
    path: str,
    body: bytes,
    *,
    content_length: int | None = None,
) -> tuple[int, dict]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request(
            "POST",
            path,
            body,
            {
                "Content-Type": "application/json",
                "Content-Length": str(
                    len(body) if content_length is None else content_length
                ),
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
    payload = valid_payload()
    server, thread = run_server_in_thread(review_dir, save_dir)
    try:
        status, response = post_json(
            server.server_address[1],
            "/video_review/__manual_review_save__",
            payload,
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert status == 200
    assert response["ok"] is True
    assert response["manual_label_row_count"] == 1
    assert response["saved_at_utc"]
    assert set(response["written_files"]) >= {
        "manual_labels.csv",
        "manual_review_progress.json",
        "manual_review_save_metadata.json",
    }
    with (save_dir / "manual_labels.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        assert handle.read() == payload["manual_labels_csv"]
    assert json.loads((save_dir / "manual_review_progress.json").read_text()) == payload[
        "progress_json"
    ]
    meta = json.loads((save_dir / "manual_review_save_metadata.json").read_text())
    assert meta["reviewer"] == "nathan"
    assert meta["run_label"] == "smoke20"
    assert meta["source"] == "autosave"
    assert meta["manual_label_row_count"] == 1
    assert meta["review_item_count"] == 1
    assert meta["source_page"].endswith("video_review/review_index_video.html")
    assert meta["schema_version"] == "manual_review_progress.v2"


def test_manual_review_server_explicit_save_writes_timestamped_backups(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    save_dir = tmp_path / "manual_review"
    review_dir.mkdir()
    payload = valid_payload(source="explicit_save")
    server, thread = run_server_in_thread(review_dir, save_dir)
    try:
        status, response = post_json(
            server.server_address[1],
            "/video_review_full/__manual_review_save__",
            payload,
        )
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
    payload = valid_payload(source="explicit_save")
    payload["run_label"] = "../../escape"
    server, thread = run_server_in_thread(review_dir, save_dir)
    try:
        bad_status, _bad_response = post_json(
            server.server_address[1], "/video_review/not_the_save_route", payload
        )
        good_status, _good_response = post_json(
            server.server_address[1], "/video_review/__manual_review_save__", payload
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert bad_status == 404
    assert good_status == 200
    assert (save_dir / "manual_labels_autosave.csv").exists()
    assert not (tmp_path / "escape").exists()


def test_manual_review_server_rejects_malformed_and_invalid_payloads(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    save_dir = tmp_path / "manual_review"
    review_dir.mkdir()
    server, thread = run_server_in_thread(review_dir, save_dir)
    try:
        malformed_status, _ = post_raw(
            server.server_address[1],
            "/video_review/__manual_review_save__",
            b"{not-json",
        )
        missing_status, missing_response = post_json(
            server.server_address[1],
            "/video_review/__manual_review_save__",
            {"progress_json": progress_json()},
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert malformed_status == 400
    assert missing_status == 400
    assert "manual_labels_csv" in missing_response["error"]
    assert not save_dir.exists()


def test_manual_review_server_rejects_excessive_request_body(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    save_dir = tmp_path / "manual_review"
    review_dir.mkdir()
    server, thread = run_server_in_thread(review_dir, save_dir)
    try:
        status, response = post_raw(
            server.server_address[1],
            "/video_review/__manual_review_save__",
            b"{}",
            content_length=manual_review_server.MAX_REQUEST_BYTES + 1,
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert status == 413
    assert response["error"] == "request_body_too_large"
    assert not save_dir.exists()


def test_failed_staging_preserves_previous_authoritative_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_dir = tmp_path / "manual_review"
    first = valid_payload("rq_first")
    second = valid_payload("rq_second")
    save_manual_review_payload(first, save_dir)
    previous_csv = (save_dir / "manual_labels.csv").read_text(encoding="utf-8")
    previous_progress = (save_dir / "manual_review_progress.json").read_text(
        encoding="utf-8"
    )
    real_named_temporary_file = manual_review_server.tempfile.NamedTemporaryFile
    calls = 0

    def fail_second_stage(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated staging failure")
        return real_named_temporary_file(*args, **kwargs)

    monkeypatch.setattr(
        manual_review_server.tempfile,
        "NamedTemporaryFile",
        fail_second_stage,
    )

    with pytest.raises(OSError, match="simulated staging failure"):
        save_manual_review_payload(second, save_dir)

    assert (save_dir / "manual_labels.csv").read_text(encoding="utf-8") == previous_csv
    assert (
        save_dir / "manual_review_progress.json"
    ).read_text(encoding="utf-8") == previous_progress
