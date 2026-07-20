from __future__ import annotations

from datetime import datetime, timedelta, timezone
from http.client import HTTPConnection
import json
from pathlib import Path
from threading import Thread

import pytest

from human_qc.evidence import EvidenceService
from human_qc.hdf5_commit import Hdf5CommitError
from human_qc.http_server import create_http_server
from human_qc.lease import LeaseStore, LeaseTokenError
from human_qc.semantic_service import (
    BoundaryEditRequest,
    SemanticCalibrationService,
    StaleSemanticRevisionError,
)
from human_qc.warn_service import WarnReviewService
from human_qc.workbench_service import WorkbenchService
from qc_common.report import load_asset_qc_report
from tests.test_human_qc_end_to_end import NOW, build_file_asset


LEASE = "lease-recovery"


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 7, 15, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


def _semantic(asset, *, lease: str = LEASE) -> SemanticCalibrationService:
    return SemanticCalibrationService(
        assets={asset.asset_id: asset.hdf5_path},
        reports={asset.asset_id: asset.report_path},
        leases={asset.asset_id: lease},
    )


def _boundary_request(
    revision: int,
    *,
    frame: int = 35,
    lease: str = LEASE,
    reviewer: str = "alice",
) -> BoundaryEditRequest:
    return BoundaryEditRequest(1, frame, None, revision, lease, reviewer)


def _http_task(server, asset_id: str) -> dict:
    connection = HTTPConnection("127.0.0.1", server.server_port)
    connection.request("GET", f"/api/assets/{asset_id}/task")
    response = connection.getresponse()
    body = json.loads(response.read().decode("utf-8"))
    connection.close()
    assert response.status == 200
    return body["task"]


def _managed_artifacts(path: Path) -> list[Path]:
    return sorted(
        candidate
        for candidate in path.parent.iterdir()
        if candidate.name.endswith(".bak")
        or candidate.name.startswith(f".{path.name}.human-qc-")
    )


def test_browser_refresh_recovers_pending_edit_from_server_report(tmp_path: Path) -> None:
    asset = build_file_asset(tmp_path, asset_id="asset-refresh")
    first = _semantic(asset)
    pending = first.begin_boundary_edit(asset.asset_id, _boundary_request(1))
    restarted_semantic = _semantic(asset)
    restarted = WorkbenchService(
        restarted_semantic,
        WarnReviewService(reports={asset.asset_id: asset.report_path}),
        asset_contexts={asset.asset_id: asset.context},
    )
    server = create_http_server("127.0.0.1", 0, restarted)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        browser_task = _http_task(server, asset.asset_id)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    restored = browser_task["semantic"]["pending_edit"]
    assert browser_task["revision"] == pending.report_revision
    assert restored is not None
    assert restored["affected_segment_ids"] == list(pending.pending_edit.affected_segment_ids)
    assert len(restored["before"]) == len(restored["after"]) == 2


def test_two_reviewers_with_same_snapshot_cannot_overwrite_newer_revision(tmp_path: Path) -> None:
    asset = build_file_asset(tmp_path, asset_id="asset-stale")
    alice = _semantic(asset, lease="lease-alice")
    bob = _semantic(asset, lease="lease-bob")
    snapshot_revision = alice.get_task(asset.asset_id).report_revision
    alice.begin_boundary_edit(
        asset.asset_id,
        _boundary_request(snapshot_revision, lease="lease-alice", reviewer="alice"),
    )

    before = asset.report_path.read_bytes()
    with pytest.raises(StaleSemanticRevisionError, match="expected revision"):
        bob.begin_boundary_edit(
            asset.asset_id,
            _boundary_request(
                snapshot_revision, frame=36, lease="lease-bob", reviewer="bob"
            ),
        )
    assert asset.report_path.read_bytes() == before


def test_expired_workbench_lease_blocks_mutation_without_report_write(tmp_path: Path) -> None:
    asset = build_file_asset(tmp_path, asset_id="asset-expired")
    clock = Clock()
    semantic = SemanticCalibrationService(
        assets={asset.asset_id: asset.hdf5_path}, reports={asset.asset_id: asset.report_path}
    )
    warn = WarnReviewService(reports={asset.asset_id: asset.report_path}, clock=lambda: NOW)
    workbench = WorkbenchService(
        semantic,
        warn,
        lease_store=LeaseStore(clock=clock),
        asset_contexts={asset.asset_id: asset.context},
    )
    task = workbench.get_asset_task(asset.asset_id)
    lease = workbench.acquire_lease(asset.asset_id, "alice", 1)
    clock.advance(2)
    before = asset.report_path.read_bytes()

    with pytest.raises(LeaseTokenError, match="expired"):
        workbench.semantic_boundary_pending(
            asset.asset_id,
            boundary_index=1,
            new_frame_exclusive=35,
            expected_revision=task["revision"],
            lease_token=lease.token,
        )
    assert asset.report_path.read_bytes() == before


def test_overlay_failure_degrades_in_integrated_task_but_keeps_clip(tmp_path: Path) -> None:
    asset = build_file_asset(tmp_path, asset_id="asset-overlay", hard_fail=False)
    semantic = _semantic(asset)
    semantic.complete(asset.asset_id, expected_revision=1, lease_token=LEASE)

    def generate_clip(command, output: Path) -> None:
        output.write_bytes(b"clip")

    def fail_overlay(*args, **kwargs) -> None:
        raise RuntimeError("overlay renderer unavailable")

    evidence = EvidenceService(
        asset.root / "cache",
        ffmpeg_runner=generate_clip,
        overlay_renderer=fail_overlay,
    )
    workbench = WorkbenchService(
        semantic,
        WarnReviewService(reports={asset.asset_id: asset.report_path}, leases={asset.asset_id: LEASE}),
        evidence,
        asset_contexts={asset.asset_id: asset.context},
    )
    task = workbench.get_asset_task(asset.asset_id)

    assert task["task_type"] == "warn_review"
    assert task["evidence"][0]["clip_url"]
    assert task["evidence"][0]["overlay_url"] is None
    assert task["evidence"][0]["generation_error"] == "overlay_unavailable"
    assert "renderer unavailable" not in json.dumps(task["evidence"])


def test_clip_failure_exposes_stable_code_without_internal_exception(tmp_path: Path) -> None:
    asset = build_file_asset(tmp_path, asset_id="asset-clip-error", hard_fail=False)
    semantic = _semantic(asset)
    semantic.complete(asset.asset_id, expected_revision=1, lease_token=LEASE)

    def fail_clip(*_args: object) -> None:
        raise RuntimeError(f"ffmpeg failed for {asset.video_path}")

    workbench = WorkbenchService(
        semantic,
        WarnReviewService(reports={asset.asset_id: asset.report_path}, leases={asset.asset_id: LEASE}),
        EvidenceService(asset.root / "cache", ffmpeg_runner=fail_clip),
        asset_contexts={asset.asset_id: asset.context},
    )
    task = workbench.get_asset_task(asset.asset_id)

    assert task["evidence"][0]["generation_error"] == "clip_unavailable"
    assert str(asset.video_path) not in json.dumps(task["evidence"])


def test_prepare_phase_failure_leaves_original_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset = build_file_asset(tmp_path, asset_id="asset-prepare")
    service = _semantic(asset)
    before_hdf5 = asset.hdf5_path.read_bytes()
    before_report = asset.report_path.read_bytes()

    def fail_prepare(*args, **kwargs):
        raise Hdf5CommitError("prepare failed")

    monkeypatch.setattr("human_qc.semantic_service.prepare_hdf5_replacement", fail_prepare)
    with pytest.raises(Hdf5CommitError, match="prepare"):
        service.complete(asset.asset_id, expected_revision=1, lease_token=LEASE)

    assert asset.hdf5_path.read_bytes() == before_hdf5
    assert asset.report_path.read_bytes() == before_report
    assert _managed_artifacts(asset.hdf5_path) == []


def test_replace_failure_keeps_old_bytes_then_restart_completes_and_cleans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset = build_file_asset(tmp_path, asset_id="asset-replace")
    service = _semantic(asset)
    before_hdf5 = asset.hdf5_path.read_bytes()

    monkeypatch.setattr(
        "human_qc.semantic_service.commit_hdf5_replacement",
        lambda prepared: (_ for _ in ()).throw(RuntimeError("replace failed")),
    )
    with pytest.raises(RuntimeError, match="replace failed"):
        service.complete(asset.asset_id, expected_revision=1, lease_token=LEASE)
    assert asset.hdf5_path.read_bytes() == before_hdf5
    report = load_asset_qc_report(asset.report_path)
    assert report is not None and report["semantic_calibration"]["state"] == "finalizing"
    assert _managed_artifacts(asset.hdf5_path)

    monkeypatch.undo()
    recovered = _semantic(asset).get_task(asset.asset_id)
    assert recovered.report_state == "completed"
    assert asset.hdf5_path.read_bytes() != before_hdf5
    assert _managed_artifacts(asset.hdf5_path) == []


def test_replace_success_report_failure_recovers_after_restart_without_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset = build_file_asset(tmp_path, asset_id="asset-report-recovery")
    service = _semantic(asset)
    module = __import__("human_qc.semantic_service", fromlist=["update_human_state"])
    real_update = module.update_human_state
    calls = 0

    def fail_second_update(path, expected_revision, mutate):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("report completion failed")
        return real_update(path, expected_revision, mutate)

    monkeypatch.setattr("human_qc.semantic_service.update_human_state", fail_second_update)
    with pytest.raises(RuntimeError, match="report completion failed"):
        service.complete(asset.asset_id, expected_revision=1, lease_token=LEASE)
    report = load_asset_qc_report(asset.report_path)
    assert report is not None and report["semantic_calibration"]["state"] == "finalizing"

    monkeypatch.undo()
    recovered = _semantic(asset).get_task(asset.asset_id)
    assert recovered.report_state == "completed"
    assert _managed_artifacts(asset.hdf5_path) == []


def test_unknown_current_hash_refuses_recovery_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    asset = build_file_asset(tmp_path, asset_id="asset-hash-conflict")
    service = _semantic(asset)
    monkeypatch.setattr(
        "human_qc.semantic_service.commit_hdf5_replacement",
        lambda prepared: (_ for _ in ()).throw(RuntimeError("pause before replace")),
    )
    with pytest.raises(RuntimeError, match="pause before replace"):
        service.complete(asset.asset_id, expected_revision=1, lease_token=LEASE)
    monkeypatch.undo()

    with asset.hdf5_path.open("ab") as handle:
        handle.write(b"unknown-current-hash")
    unknown_bytes = asset.hdf5_path.read_bytes()
    with pytest.raises(Hdf5CommitError, match="hash|conflict"):
        _semantic(asset).get_task(asset.asset_id)
    assert asset.hdf5_path.read_bytes() == unknown_bytes
