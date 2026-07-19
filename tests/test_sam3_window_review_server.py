from __future__ import annotations

import csv
from http.client import HTTPConnection
import json
from pathlib import Path
from threading import Thread
from urllib.parse import quote

import pandas as pd
import pytest

from human_qc.sam3_window_review import (
    ReviewConflictError,
    ReviewValidationError,
    Sam3WindowReviewStore,
    load_review_bundle,
)
from human_qc.sam3_window_review_server import create_sam3_window_review_server


PNG_BYTES = b"\x89PNG\r\n\x1a\nfixture"


def _fixture_bundle(
    tmp_path: Path,
    *,
    missing_first: bool = False,
    asset_id: str = "episode_000001",
    windows: tuple[tuple[int, int], ...] = ((0, 20), (30, 50)),
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest_path = tmp_path / "manifest.csv"
    queue_path = tmp_path / "queue.csv"
    evidence_path = tmp_path / "evidence.csv"
    pd.DataFrame([{"supplier_id": "jdt", "asset_id": asset_id}]).to_csv(
        manifest_path, index=False
    )
    queue = []
    evidence = []
    review_asset_id = asset_id if asset_id.startswith("jdt__") else f"jdt__{asset_id}"
    for window_index, (start, end) in enumerate(windows):
        review_id = f"{review_asset_id}__window_{start}_{end}"
        queue.append(
            {
                "review_id": review_id,
                "supplier_id": "jdt",
                "asset_id": asset_id,
                "window_start_frame": start,
                "window_end_frame": end,
                "source_start_frame": start,
                "source_end_frame": end,
                "frame_coordinate_system": "source_inclusive",
                "sam3_window_state": "review" if window_index == 0 else "fail",
                "left_window_containment_verdict": "fail",
                "right_window_containment_verdict": "review",
                "fps": 30.0,
                "trigger_reason_json": '["containment_mismatch"]',
                "trigger_metrics_json": '{"outside_ratio": 0.4}',
            }
        )
        span = end - start
        for frame_idx in (start, start + span // 4, start + span // 2, end - span // 4, end):
            source = tmp_path / "source evidence" / review_id / f"frame {frame_idx}.png"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(PNG_BYTES)
            evidence.append(
                {
                    "review_id": review_id,
                    "supplier_id": "jdt",
                    "asset_id": asset_id,
                    "window_start_frame": start,
                    "window_end_frame": end,
                    "frame_idx": frame_idx,
                    "source_module": "sam3_containment",
                    "evidence_type": "combined_overlay",
                    "hand_side": "both",
                    "source_path": str(source),
                    "metadata_json": json.dumps({"source_frame_idx": frame_idx}),
                }
            )
    if missing_first:
        Path(evidence[0]["source_path"]).unlink()
    pd.DataFrame(queue).to_csv(queue_path, index=False)
    pd.DataFrame(evidence).to_csv(evidence_path, index=False)
    return load_review_bundle(
        manifest_path=manifest_path,
        queue_path=queue_path,
        evidence_path=evidence_path,
        review_dir=tmp_path / "review",
    )


def test_store_saves_same_asset_windows_independently_and_recovers_after_refresh(
    tmp_path: Path,
) -> None:
    bundle = _fixture_bundle(tmp_path)
    save_dir = tmp_path / "saved"
    store = Sam3WindowReviewStore(bundle, save_dir)
    first, second = [item["review_id"] for item in bundle.items]

    first_saved = store.save(first, "pass", "alice", expected_revision=0)
    second_saved = store.save(second, "fail", "bob", expected_revision=0)

    assert first_saved["revision"] == 1
    assert second_saved["revision"] == 1
    snapshot = store.snapshot()
    assert snapshot["reviews"][first]["verdict"] == "pass"
    assert snapshot["reviews"][second]["verdict"] == "fail"
    assert snapshot["reviews"][first]["reviewer"] == "alice"
    assert snapshot["reviews"][second]["reviewer"] == "bob"
    assert snapshot["reviews"][second]["duration_resolution_status"] == "not_annotated"
    assert snapshot["reviews"][second]["review_id"] == second
    assert snapshot["reviews"][second]["source_queue_path"] == str(bundle.queue_path)
    assert snapshot["reviews"][second]["source_queue_sha256"] == bundle.queue_sha256
    assert len(snapshot["reviews"][second]["evidence_provenance"]) == 5

    restored = Sam3WindowReviewStore(bundle, save_dir).snapshot()
    assert restored == snapshot

    with (save_dir / "sam3_window_review_results.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert {row["review_id"]: row["verdict"] for row in rows} == {
        first: "pass",
        second: "fail",
    }
    assert all(row["duration_resolution_status"] == "not_annotated" for row in rows)
    forbidden = {"affected_start_frame", "affected_end_frame", "rejected_duration"}
    assert forbidden.isdisjoint(rows[0])


def test_prefixed_jdt_asset_id_is_preserved_in_state_and_csv(tmp_path: Path) -> None:
    asset_id = "jdt__episode_000001"
    review_id = "jdt__episode_000001__window_0_5760"
    bundle = _fixture_bundle(
        tmp_path,
        asset_id=asset_id,
        windows=((0, 5760),),
    )
    save_dir = tmp_path / "saved"
    store = Sam3WindowReviewStore(bundle, save_dir)

    assert bundle.items[0]["asset_id"] == asset_id
    assert bundle.items[0]["review_id"] == review_id
    saved = store.save(review_id, "fail", "alice", expected_revision=0)
    assert saved["asset_id"] == asset_id
    assert saved["review_id"] == review_id
    assert store.snapshot()["reviews"][review_id]["asset_id"] == asset_id

    with (save_dir / "sam3_window_review_results.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["asset_id"] == asset_id
    assert rows[0]["review_id"] == review_id


def test_saved_state_is_not_reused_after_queue_content_changes(tmp_path: Path) -> None:
    bundle = _fixture_bundle(tmp_path)
    save_dir = tmp_path / "saved"
    store = Sam3WindowReviewStore(bundle, save_dir)
    store.save(bundle.items[0]["review_id"], "pass", "alice", expected_revision=0)

    queue = pd.read_csv(bundle.queue_path)
    queue.loc[0, "trigger_reason_json"] = '["different_queue_contract"]'
    queue.to_csv(bundle.queue_path, index=False)
    changed_bundle = load_review_bundle(
        manifest_path=bundle.manifest_path,
        queue_path=bundle.queue_path,
        evidence_path=bundle.evidence_path,
        review_dir=bundle.review_dir,
    )

    with pytest.raises(ReviewValidationError, match="queue.*identity|sha256"):
        Sam3WindowReviewStore(changed_bundle, save_dir)


def test_unresolved_is_not_exported_as_completed_or_pass(tmp_path: Path) -> None:
    bundle = _fixture_bundle(tmp_path)
    save_dir = tmp_path / "saved"
    store = Sam3WindowReviewStore(bundle, save_dir)

    assert store.snapshot()["reviews"] == {}
    assert not (save_dir / "sam3_window_review_results.csv").exists()
    assert all(item["manual_review"]["verdict"] is None for item in bundle.items)


def test_store_rejects_invalid_verdict_stale_revision_and_missing_evidence(
    tmp_path: Path,
) -> None:
    bundle = _fixture_bundle(tmp_path)
    store = Sam3WindowReviewStore(bundle, tmp_path / "saved")
    review_id = bundle.items[0]["review_id"]

    with pytest.raises(ReviewValidationError, match="pass or fail"):
        store.save(review_id, "false", "alice", expected_revision=0)
    store.save(review_id, "pass", "alice", expected_revision=0)
    with pytest.raises(ReviewConflictError, match="revision"):
        store.save(review_id, "fail", "alice", expected_revision=0)

    missing_bundle = _fixture_bundle(tmp_path / "missing", missing_first=True)
    missing_store = Sam3WindowReviewStore(missing_bundle, tmp_path / "missing-save")
    with pytest.raises(ReviewConflictError, match="evidence"):
        missing_store.save(
            missing_bundle.items[0]["review_id"], "pass", "alice", expected_revision=0
        )


def test_store_rechecks_staged_evidence_before_completing_review(tmp_path: Path) -> None:
    bundle = _fixture_bundle(tmp_path)
    store = Sam3WindowReviewStore(bundle, tmp_path / "saved")
    item = bundle.items[0]
    relative = item["evidence"][0]["url"].removeprefix("/assets/")
    (bundle.assets_root / relative).unlink()

    with pytest.raises(ReviewConflictError, match="evidence"):
        store.save(item["review_id"], "pass", "alice", expected_revision=0)


def _request(server, method: str, path: str, payload: dict | bytes | None = None):
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    if isinstance(payload, dict):
        body = json.dumps(payload).encode("utf-8")
    else:
        body = payload
    headers = {"Content-Type": "application/json"} if body is not None else {}
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    response_body = response.read()
    headers_out = {key.lower(): value for key, value in response.getheaders()}
    connection.close()
    return response.status, headers_out, response_body


def _running_server(bundle, store):
    server = create_sam3_window_review_server(
        "127.0.0.1", 0, bundle=bundle, store=store
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_http_serves_only_whitelisted_images_with_real_content_type(tmp_path: Path) -> None:
    bundle = _fixture_bundle(tmp_path)
    store = Sam3WindowReviewStore(bundle, tmp_path / "saved")
    secret = bundle.assets_root / "secret.png"
    secret.write_bytes(PNG_BYTES)
    server, thread = _running_server(bundle, store)
    try:
        image_responses = [
            _request(server, "GET", evidence["url"])
            for evidence in bundle.items[0]["evidence"]
        ]
        unlisted, _, _ = _request(server, "GET", "/assets/secret.png")
        traversal, _, _ = _request(server, "GET", "/assets/%2e%2e/secret.png")
        encoded_traversal, _, _ = _request(
            server, "GET", "/assets/%252e%252e/secret.png"
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert len(image_responses) == 5
    assert all(status == 200 for status, _headers, _body in image_responses)
    assert all(headers["content-type"] == "image/png" for _status, headers, _body in image_responses)
    assert all(body == PNG_BYTES for _status, _headers, body in image_responses)
    assert unlisted == 404
    assert traversal == 404
    assert encoded_traversal == 404


def test_http_autosave_and_refresh_recover_authoritative_server_state(
    tmp_path: Path,
) -> None:
    bundle = _fixture_bundle(tmp_path)
    store = Sam3WindowReviewStore(bundle, tmp_path / "saved")
    review_id = bundle.items[0]["review_id"]
    server, thread = _running_server(bundle, store)
    try:
        before_status, _, before_body = _request(server, "GET", "/api/review-bundle")
        save_status, _, save_body = _request(
            server,
            "POST",
            "/api/reviews/" + quote(review_id, safe=""),
            {"verdict": "pass", "reviewer": "alice", "expected_revision": 0},
        )
        after_status, _, after_body = _request(server, "GET", "/api/review-bundle")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    before = json.loads(before_body)
    saved = json.loads(save_body)
    after = json.loads(after_body)
    assert before_status == save_status == after_status == 200
    assert before["bundle"]["items"][0]["manual_review"]["status"] == "unresolved"
    assert saved["review"]["review_id"] == review_id
    assert after["bundle"]["items"][0]["manual_review"]["verdict"] == "pass"
    assert after["bundle"]["items"][0]["manual_review"]["reviewer"] == "alice"


def test_http_rejects_malformed_stale_and_missing_evidence_mutations(tmp_path: Path) -> None:
    bundle = _fixture_bundle(tmp_path)
    store = Sam3WindowReviewStore(bundle, tmp_path / "saved")
    review_id = bundle.items[0]["review_id"]
    server, thread = _running_server(bundle, store)
    try:
        malformed, _, _ = _request(
            server, "POST", "/api/reviews/" + quote(review_id, safe=""), b"{bad"
        )
        first, _, _ = _request(
            server,
            "POST",
            "/api/reviews/" + quote(review_id, safe=""),
            {"verdict": "fail", "reviewer": "alice", "expected_revision": 0},
        )
        stale, _, _ = _request(
            server,
            "POST",
            "/api/reviews/" + quote(review_id, safe=""),
            {"verdict": "pass", "reviewer": "alice", "expected_revision": 0},
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert malformed == 400
    assert first == 200
    assert stale == 409

    missing_bundle = _fixture_bundle(tmp_path / "missing", missing_first=True)
    missing_store = Sam3WindowReviewStore(missing_bundle, tmp_path / "missing-save")
    missing_id = missing_bundle.items[0]["review_id"]
    server, thread = _running_server(missing_bundle, missing_store)
    try:
        missing, _, body = _request(
            server,
            "POST",
            "/api/reviews/" + quote(missing_id, safe=""),
            {"verdict": "pass", "reviewer": "alice", "expected_revision": 0},
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert missing == 409
    assert "evidence" in json.loads(body)["error"]["message"]


def test_http_refresh_marks_deleted_staged_evidence_unavailable(tmp_path: Path) -> None:
    bundle = _fixture_bundle(tmp_path)
    store = Sam3WindowReviewStore(bundle, tmp_path / "saved")
    item = bundle.items[0]
    relative = item["evidence"][0]["url"].removeprefix("/assets/")
    (bundle.assets_root / relative).unlink()
    server, thread = _running_server(bundle, store)
    try:
        status, _, body = _request(server, "GET", "/api/review-bundle")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    refreshed = json.loads(body)["bundle"]["items"][0]
    assert status == 200
    assert refreshed["can_review"] is False
    assert refreshed["evidence_status"] == "error"
    assert "missing" in refreshed["evidence_error"]
