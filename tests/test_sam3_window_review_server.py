from __future__ import annotations

import csv
from http.client import HTTPConnection
import json
import multiprocessing
from pathlib import Path
from threading import Thread
from urllib.parse import quote

import pandas as pd
import pytest

import human_qc.sam3_window_review as sam3_window_review_module
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


def _server_process_save(
    manifest_path: str,
    queue_path: str,
    evidence_path: str,
    review_dir: str,
    save_dir: str,
    review_index: int,
    verdict: str,
    reviewer: str,
    ready,
    proceed,
    results,
) -> None:
    server = None
    try:
        bundle = load_review_bundle(
            manifest_path=Path(manifest_path),
            queue_path=Path(queue_path),
            evidence_path=Path(evidence_path),
            review_dir=Path(review_dir),
        )
        store = Sam3WindowReviewStore(bundle, Path(save_dir))
        server = create_sam3_window_review_server(
            "127.0.0.1", 0, bundle=bundle, store=store
        )
        ready.set()
        if not proceed.wait(timeout=20):
            raise TimeoutError("timed out waiting to save")
        review_id = bundle.items[review_index]["review_id"]
        record = server.store.save(
            review_id,
            verdict,
            reviewer,
            expected_revision=0,
        )
        results.put(
            {
                "status": "saved",
                "review_id": review_id,
                "revision": record["revision"],
            }
        )
    except Exception as exc:  # pragma: no cover - asserted in the parent process.
        results.put(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        )
    finally:
        if server is not None:
            server.server_close()


def _join_process(process) -> None:
    process.join(timeout=20)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        pytest.fail("server process did not exit")
    assert process.exitcode == 0


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


def test_two_server_processes_preserve_independent_window_writes(tmp_path: Path) -> None:
    bundle = _fixture_bundle(tmp_path)
    save_dir = tmp_path / "saved"
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    first_ready = context.Event()
    first_proceed = context.Event()
    second_ready = context.Event()
    second_proceed = context.Event()
    second_proceed.set()
    common = (
        str(bundle.manifest_path),
        str(bundle.queue_path),
        str(bundle.evidence_path),
        str(bundle.review_dir),
        str(save_dir),
    )
    first = context.Process(
        target=_server_process_save,
        args=(*common, 1, "fail", "alice", first_ready, first_proceed, results),
    )
    second = context.Process(
        target=_server_process_save,
        args=(*common, 0, "pass", "bob", second_ready, second_proceed, results),
    )
    try:
        first.start()
        assert first_ready.wait(timeout=20)
        second.start()
        assert second_ready.wait(timeout=20)
        second_result = results.get(timeout=20)
        assert second_result["status"] == "saved"
        first_proceed.set()
        first_result = results.get(timeout=20)
        assert first_result["status"] == "saved"
    finally:
        first_proceed.set()
        if first.pid is not None:
            _join_process(first)
        if second.pid is not None:
            _join_process(second)

    restored = Sam3WindowReviewStore(bundle, save_dir).snapshot()
    assert set(restored["reviews"]) == {
        bundle.items[0]["review_id"],
        bundle.items[1]["review_id"],
    }


def test_stale_revision_is_rejected_across_server_processes(tmp_path: Path) -> None:
    bundle = _fixture_bundle(tmp_path, windows=((0, 20),))
    save_dir = tmp_path / "saved"
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    stale_ready = context.Event()
    stale_proceed = context.Event()
    current_ready = context.Event()
    current_proceed = context.Event()
    current_proceed.set()
    common = (
        str(bundle.manifest_path),
        str(bundle.queue_path),
        str(bundle.evidence_path),
        str(bundle.review_dir),
        str(save_dir),
    )
    stale = context.Process(
        target=_server_process_save,
        args=(*common, 0, "fail", "alice", stale_ready, stale_proceed, results),
    )
    current = context.Process(
        target=_server_process_save,
        args=(*common, 0, "pass", "bob", current_ready, current_proceed, results),
    )
    try:
        stale.start()
        assert stale_ready.wait(timeout=20)
        current.start()
        assert current_ready.wait(timeout=20)
        current_result = results.get(timeout=20)
        assert current_result["status"] == "saved"
        stale_proceed.set()
        stale_result = results.get(timeout=20)
        assert stale_result["status"] == "error"
        assert stale_result["error_type"] == "ReviewConflictError"
        assert "revision" in stale_result["message"]
    finally:
        stale_proceed.set()
        if stale.pid is not None:
            _join_process(stale)
        if current.pid is not None:
            _join_process(current)

    restored = Sam3WindowReviewStore(bundle, save_dir).snapshot()
    review = restored["reviews"][bundle.items[0]["review_id"]]
    assert review["verdict"] == "pass"
    assert review["reviewer"] == "bob"


@pytest.mark.parametrize("damage", ["missing", "inconsistent"])
def test_startup_rebuilds_derived_csv_from_authoritative_state(
    tmp_path: Path, damage: str
) -> None:
    bundle = _fixture_bundle(tmp_path)
    save_dir = tmp_path / "saved"
    store = Sam3WindowReviewStore(bundle, save_dir)
    store.save(bundle.items[0]["review_id"], "pass", "alice", expected_revision=0)
    results_path = save_dir / "sam3_window_review_results.csv"
    expected = results_path.read_text(encoding="utf-8")
    if damage == "missing":
        results_path.unlink()
    else:
        results_path.write_text("stale,csv\n", encoding="utf-8")

    restored = Sam3WindowReviewStore(bundle, save_dir)

    assert restored.snapshot()["reviews"]
    assert results_path.read_text(encoding="utf-8") == expected


def test_derived_csv_write_failure_does_not_roll_back_authoritative_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _fixture_bundle(tmp_path)
    save_dir = tmp_path / "saved"
    store = Sam3WindowReviewStore(bundle, save_dir)
    review_id = bundle.items[0]["review_id"]
    real_write = sam3_window_review_module._write_text_files_atomically

    def fail_csv(save_root: Path, files: dict[str, str]) -> None:
        if "sam3_window_review_results.csv" in files:
            raise OSError("derived CSV unavailable")
        real_write(save_root, files)

    monkeypatch.setattr(
        sam3_window_review_module,
        "_write_text_files_atomically",
        fail_csv,
    )
    saved = store.save(review_id, "fail", "alice", expected_revision=0)

    assert saved["verdict"] == "fail"
    authoritative = json.loads(
        (save_dir / "sam3_window_review_state.json").read_text(encoding="utf-8")
    )
    assert authoritative["reviews"][review_id]["verdict"] == "fail"
    assert not (save_dir / "sam3_window_review_results.csv").exists()

    monkeypatch.setattr(
        sam3_window_review_module,
        "_write_text_files_atomically",
        real_write,
    )
    Sam3WindowReviewStore(bundle, save_dir)
    assert (save_dir / "sam3_window_review_results.csv").is_file()


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
