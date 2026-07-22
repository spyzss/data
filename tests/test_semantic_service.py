from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import h5py
import pytest

from semantic_calibration.contracts import SubtaskSegment
from semantic_calibration.hdf5_commit import Hdf5CommitError
from semantic_calibration.service import (
    BoundaryEditRequest,
    SemanticCalibrationService,
    SemanticTaskView,
    TaskStateError,
    TextEditRequest,
)
from semantic_calibration.source_adapters import Hdf5ScalarJsonSubtaskAdapter
from semantic_calibration.timeline import SharedBoundaryTimeline
from qc_common.report import load_asset_qc_report, write_asset_qc_report
from tests.qc_report_fixtures import make_v2_report


ASSET_ID = "617856"
LEASE = "lease-alice"
DATASET_PATH = "/label/subtask_label"


def _payload() -> dict:
    rows = []
    for index, (start, end, cn, en) in enumerate(
        [(0, 50, "第一步", "first"), (51, 122, "第二步", "second"), (123, 194, "第三步", "third")]
    ):
        rows.append(
            {
                "start_frame": start,
                "end_frame": end,
                "start_time_sec": start / 30,
                "end_time_sec": end / 30,
                "subtask_cn": cn,
                "subtask_en": en,
                "verb": "act",
                "object": "thing",
                "target": "thing",
                "hand": "right",
                "phase": f"p{index}",
                "evidence_frames": [start, end],
                "confidence": 0.9,
                "status": "confirmed",
            }
        )
    return {
        "id": ASSET_ID,
        "scene": "kitchen",
        "task": "demo",
        "fps": 30.0,
        "frame_count": 195,
        "annotations": rows,
    }


def _write_asset(tmp_path: Path, *, report: dict | None = None) -> tuple[Path, Path]:
    hdf5_path = tmp_path / f"{ASSET_ID}.hdf5"
    with h5py.File(hdf5_path, "w") as handle:
        label = handle.create_group("label")
        dataset = label.create_dataset(DATASET_PATH.rsplit("/", 1)[1], shape=(), dtype="S65536")
        dataset[()] = json.dumps(_payload(), ensure_ascii=False).encode("utf-8")

    report_path = tmp_path / "quality_archive" / f"{ASSET_ID}.json"
    value = make_v2_report(status="awaiting_external") if report is None else copy.deepcopy(report)
    value["asset_id"] = ASSET_ID
    # A semantic workbench task is only valid when the orchestrator cursor is
    # paused at this external module; other cursors must be rejected.
    if report is None:
        value["pipeline_state"]["next_module"] = "semantic_consistency"
    if value["pipeline_state"].get("next_module") == "semantic_consistency" and value[
        "manual_review"
    ].get("state") == "not_evaluated":
        value["manual_review"].update(
            {
                "required": False,
                "state": "not_required",
                "selected_issue_ids": [],
                "selected_issue_id": None,
                "issue_reviews": {},
                "completed_at": None,
            }
        )
    value["source_files"] = {"hdf5": {"path": str(hdf5_path)}}
    write_asset_qc_report(report_path, value, expected_revision=0, profile="acceptance")
    return hdf5_path, report_path


def _mutate_external_immutable_annotation(path: Path) -> None:
    with h5py.File(path, "r+") as handle:
        dataset = handle[DATASET_PATH]
        payload = json.loads(bytes(dataset[()]).decode("utf-8"))
        payload["annotations"][0]["verb"] = "externally-changed"
        dataset[()] = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        handle.flush()


def _service(tmp_path: Path, *, report: dict | None = None, **kwargs) -> SemanticCalibrationService:
    hdf5_path, report_path = _write_asset(tmp_path, report=report)
    return SemanticCalibrationService(
        assets={ASSET_ID: hdf5_path},
        reports={ASSET_ID: report_path},
        leases={ASSET_ID: LEASE},
        **kwargs,
    )


def boundary_request(index: int, new_frame: int, *, revision: int = 3) -> BoundaryEditRequest:
    return BoundaryEditRequest(
        boundary_index=index,
        new_frame_exclusive=new_frame,
        actor_segment_id=None,
        expected_revision=revision,
        lease_token=LEASE,
        reviewer="alice",
    )


def test_confirm_boundary_edit_commits_both_segments_once(tmp_path: Path) -> None:
    service = _service(tmp_path)
    pending = service.begin_boundary_edit(ASSET_ID, boundary_request(1, 60, revision=1))
    assert len(pending.pending_edit.before) == 2
    assert len(pending.pending_edit.after) == 2

    confirmed = service.confirm_pending(ASSET_ID, expected_revision=2, lease_token=LEASE)
    assert confirmed.timeline.boundaries == (0, 60, 123, 195)
    assert confirmed.timeline_edit_count == 1
    assert confirmed.pending_edit is None


def test_cancel_restores_both_segments_without_increment(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.begin_boundary_edit(ASSET_ID, boundary_request(2, 140, revision=1))
    cancelled = service.cancel_pending(ASSET_ID, expected_revision=2, lease_token=LEASE)
    assert cancelled.timeline.boundaries == (0, 51, 123, 195)
    assert cancelled.timeline_edit_count == 0


def test_text_pending_contains_one_segment_and_confirm_changes_text(tmp_path: Path) -> None:
    service = _service(tmp_path)
    pending = service.begin_text_edit(
        ASSET_ID,
        TextEditRequest(
            segment_id=service.get_task(ASSET_ID).timeline.segments[1].internal_id,
            text_cn="新文本",
            text_en="new text",
            expected_revision=1,
            lease_token=LEASE,
            reviewer="alice",
        ),
    )
    assert len(pending.pending_edit.before) == 1
    assert len(pending.pending_edit.after) == 1
    confirmed = service.confirm_pending(ASSET_ID, expected_revision=2, lease_token=LEASE)
    assert confirmed.timeline.segments[1].text_cn == "新文本"
    assert confirmed.timeline.segments[1].text_en == "new text"
    assert confirmed.subtask_text_edit_count == 1


@pytest.mark.parametrize("method", ["begin_boundary_edit", "begin_text_edit"])
def test_pending_rejects_other_edits_and_completion(tmp_path: Path, method: str) -> None:
    service = _service(tmp_path)
    service.begin_boundary_edit(ASSET_ID, boundary_request(1, 60, revision=1))
    request = boundary_request(2, 140, revision=2)
    if method == "begin_text_edit":
        segment_id = service.get_task(ASSET_ID).timeline.segments[0].internal_id
        request = TextEditRequest(segment_id, "x", "x", 2, LEASE, "alice")
    with pytest.raises(RuntimeError, match="pending"):
        getattr(service, method)(ASSET_ID, request)
    with pytest.raises(RuntimeError, match="pending"):
        service.complete(ASSET_ID, expected_revision=2, lease_token=LEASE)


def test_assets_only_service_honors_pending_navigation_lock_for_second_asset(
    tmp_path: Path,
) -> None:
    first_hdf5, _ = _write_asset(tmp_path)
    other_id = "617857"
    other_root = tmp_path / "other"
    other_hdf5 = other_root / f"{other_id}.hdf5"
    other_root.mkdir()
    payload = _payload()
    payload["id"] = other_id
    with h5py.File(other_hdf5, "w") as handle:
        dataset = handle.create_group("label").create_dataset(
            "subtask_label", shape=(), dtype="S65536"
        )
        dataset[()] = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    other_report = other_root / "quality_archive" / f"{other_id}.json"
    other_value = make_v2_report(status="awaiting_external")
    other_value["asset_id"] = other_id
    other_value["pipeline_state"]["next_module"] = "semantic_consistency"
    write_asset_qc_report(other_report, other_value, expected_revision=0, profile="acceptance")

    service = SemanticCalibrationService(
        assets={ASSET_ID: first_hdf5, other_id: other_hdf5},
        leases={ASSET_ID: LEASE, other_id: LEASE},
    )
    service.begin_boundary_edit(ASSET_ID, boundary_request(1, 60, revision=1))
    restarted = SemanticCalibrationService(
        assets={ASSET_ID: first_hdf5, other_id: other_hdf5},
        leases={ASSET_ID: LEASE, other_id: LEASE},
    )
    with pytest.raises(RuntimeError, match="navigation|pending"):
        restarted.get_task(other_id)


def test_stale_revision_and_wrong_lease_are_rejected_without_writes(tmp_path: Path) -> None:
    service = _service(tmp_path)
    before = service.get_task(ASSET_ID)
    with pytest.raises(Exception, match="revision"):
        service.begin_boundary_edit(ASSET_ID, boundary_request(1, 60, revision=0))
    with pytest.raises(Exception, match="lease"):
        service.begin_boundary_edit(ASSET_ID, BoundaryEditRequest(1, 60, None, 1, "wrong", "alice"))
    assert service.get_task(ASSET_ID) == before


def test_boundary_drag_rejects_outer_or_crossing_positions(tmp_path: Path) -> None:
    service = _service(tmp_path)
    with pytest.raises(ValueError):
        service.begin_boundary_edit(ASSET_ID, boundary_request(1, 0, revision=1))
    with pytest.raises(ValueError):
        service.begin_boundary_edit(ASSET_ID, boundary_request(1, 123, revision=1))
    assert service.get_task(ASSET_ID).pending_edit is None


def test_boundary_drag_rejects_noop_at_current_boundary(tmp_path: Path) -> None:
    service = _service(tmp_path)
    with pytest.raises(ValueError, match="change|boundary"):
        service.begin_boundary_edit(ASSET_ID, boundary_request(1, 51, revision=1))


def test_complete_prepare_failure_leaves_report_and_hdf5_unpublished(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(tmp_path)
    before_hdf5 = service.get_task(ASSET_ID).hdf5_sha256
    monkeypatch.setattr("semantic_calibration.service.prepare_hdf5_replacement", lambda *a, **k: (_ for _ in ()).throw(Hdf5CommitError("prepare failed")))
    with pytest.raises(Hdf5CommitError):
        service.complete(ASSET_ID, expected_revision=1, lease_token=LEASE)
    task = service.get_task(ASSET_ID)
    assert task.hdf5_sha256 == before_hdf5
    assert task.report_state != "completed"


def test_complete_publishes_hdf5_then_marks_report_completed(tmp_path: Path) -> None:
    service = _service(tmp_path)
    completed = service.complete(ASSET_ID, expected_revision=1, lease_token=LEASE)
    assert completed.report_state == "completed"
    assert completed.semantic_consistency_state == "completed"
    assert completed.pipeline_state == "completed"
    persisted = load_asset_qc_report(service.report_path(ASSET_ID))
    assert persisted is not None
    assert persisted["semantic_calibration"]["state"] == "completed"
    assert persisted["semantic_calibration"]["final_hdf5_sha256"] == completed.hdf5_sha256


def test_completed_boundary_timeline_survives_service_restart_with_stable_ids(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    service.begin_boundary_edit(ASSET_ID, boundary_request(1, 60, revision=1))
    confirmed = service.confirm_pending(ASSET_ID, expected_revision=2, lease_token=LEASE)
    original_ids = tuple(segment.internal_id for segment in confirmed.timeline.segments)

    service.complete(ASSET_ID, expected_revision=3, lease_token=LEASE)
    restarted = SemanticCalibrationService(
        assets={ASSET_ID: service._assets[ASSET_ID]},
        reports={ASSET_ID: service.report_path(ASSET_ID)},
        leases={ASSET_ID: LEASE},
    ).get_task(ASSET_ID)

    assert restarted.timeline.boundaries == (0, 60, 123, 195)
    assert tuple(segment.internal_id for segment in restarted.timeline.segments) == original_ids
    assert tuple(segment.text_cn for segment in restarted.timeline.segments) == (
        "第一步",
        "第二步",
        "第三步",
    )


def test_complete_rejects_pipeline_cursor_outside_semantic_stage(tmp_path: Path) -> None:
    report = make_v2_report(status="awaiting_external")
    report["pipeline_state"]["next_module"] = "video_quality"
    service = _service(tmp_path, report=report)
    before_hdf5 = service._assets[ASSET_ID].read_bytes()
    before_report = service.report_path(ASSET_ID).read_bytes()

    with pytest.raises(TaskStateError, match="pipeline|semantic|next_module"):
        service.complete(ASSET_ID, expected_revision=1, lease_token=LEASE)

    assert service._assets[ASSET_ID].read_bytes() == before_hdf5
    assert service.report_path(ASSET_ID).read_bytes() == before_report


def test_semantic_task_is_blocked_while_manual_review_is_incomplete(tmp_path: Path) -> None:
    report = make_v2_report(status="awaiting_external")
    report["pipeline_state"]["next_module"] = "semantic_consistency"
    report["manual_review"].update(
        {
            "state": "queued",
            "candidate_issue_ids": ["warn-1"],
            "required": True,
            "selected_issue_ids": [],
            "selected_issue_id": None,
            "issue_reviews": {},
            "completed_at": None,
        }
    )
    service = _service(tmp_path, report=report)
    before_hdf5 = service._assets[ASSET_ID].read_bytes()
    before_report = service.report_path(ASSET_ID).read_bytes()

    with pytest.raises(TaskStateError, match="manual|eligible|blocked"):
        service.get_task(ASSET_ID)
    with pytest.raises(TaskStateError, match="manual|eligible|blocked"):
        service.complete(ASSET_ID, expected_revision=1, lease_token=LEASE)

    assert service._assets[ASSET_ID].read_bytes() == before_hdf5
    assert service.report_path(ASSET_ID).read_bytes() == before_report


def test_assets_only_service_gates_derived_persisted_manual_review(tmp_path: Path) -> None:
    report = make_v2_report(status="awaiting_external")
    report["pipeline_state"]["next_module"] = "semantic_consistency"
    report["manual_review"].update(
        {
            "state": "queued",
            "candidate_issue_ids": ["warn-1"],
            "required": True,
            "selected_issue_ids": [],
            "selected_issue_id": None,
            "issue_reviews": {},
            "completed_at": None,
        }
    )
    hdf5_path, _ = _write_asset(tmp_path, report=report)
    service = SemanticCalibrationService(
        assets={ASSET_ID: hdf5_path},
        leases={ASSET_ID: LEASE},
    )

    with pytest.raises(TaskStateError, match="manual|eligible|blocked"):
        service.get_task(ASSET_ID)


def test_assets_only_service_allows_reportless_read_only_preview(tmp_path: Path) -> None:
    hdf5_path, report_path = _write_asset(tmp_path)
    report_path.unlink()
    service = SemanticCalibrationService(
        assets={ASSET_ID: hdf5_path},
        leases={ASSET_ID: LEASE},
    )

    task = service.get_task(ASSET_ID)

    assert task.asset_id == ASSET_ID
    assert task.report_state == "not_started"
    assert task.timeline.boundaries == (0, 51, 123, 195)


def test_completed_semantic_task_read_rechecks_manual_terminal_state(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.complete(ASSET_ID, expected_revision=1, lease_token=LEASE)
    report_path = service.report_path(ASSET_ID)
    report = load_asset_qc_report(report_path)
    assert report is not None
    report["manual_review"].update(
        {
            "state": "queued",
            "required": True,
            "candidate_issue_ids": ["warn-1"],
            "selected_issue_ids": ["warn-1"],
            "selected_issue_id": "warn-1",
            "issue_reviews": {},
            "completed_at": None,
        }
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

    restarted = SemanticCalibrationService(
        assets={ASSET_ID: service._assets[ASSET_ID]},
        reports={ASSET_ID: report_path},
        leases={ASSET_ID: LEASE},
    )
    with pytest.raises(TaskStateError, match="manual|eligible|blocked"):
        restarted.get_task(ASSET_ID)


@pytest.mark.parametrize("pipeline_status", ["running", "awaiting_external", "stopped"])
def test_completed_semantic_history_is_readable_after_pipeline_moves_downstream(
    tmp_path: Path,
    pipeline_status: str,
) -> None:
    service = _service(tmp_path)
    service.complete(
        ASSET_ID,
        expected_revision=1,
        lease_token=LEASE,
        advance_pipeline=False,
    )
    report_path = service.report_path(ASSET_ID)
    report = load_asset_qc_report(report_path)
    assert report is not None
    report["pipeline_state"].update(
        {
            "status": pipeline_status,
            "last_completed_module": "semantic_consistency",
            "next_module": None if pipeline_status == "stopped" else "duplicate_check",
            "stop_reason": "downstream_failure" if pipeline_status == "stopped" else None,
        }
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    restarted = SemanticCalibrationService(
        assets={ASSET_ID: service._assets[ASSET_ID]},
        reports={ASSET_ID: report_path},
        leases={ASSET_ID: LEASE},
    )

    task = restarted.get_task(ASSET_ID)

    assert task.report_state == "completed"
    assert task.pipeline_state == pipeline_status
    idempotent = restarted.complete(
        ASSET_ID,
        expected_revision=task.report_revision,
        lease_token=LEASE,
    )
    assert idempotent.report_state == "completed"
    with pytest.raises(TaskStateError, match="pipeline|semantic|editable"):
        restarted.begin_boundary_edit(
            ASSET_ID,
            boundary_request(1, 60, revision=task.report_revision),
        )


@pytest.mark.parametrize("action", ["confirm", "cancel"])
def test_pending_semantic_mutation_rechecks_persisted_manual_gate(
    tmp_path: Path,
    action: str,
) -> None:
    service = _service(tmp_path)
    service.begin_boundary_edit(ASSET_ID, boundary_request(1, 60, revision=1))
    report_path = service.report_path(ASSET_ID)
    report = load_asset_qc_report(report_path)
    assert report is not None
    report["manual_review"].update(
        {
            "state": "in_progress",
            "required": True,
            "candidate_issue_ids": ["warn-1"],
            "selected_issue_ids": ["warn-1"],
            "selected_issue_id": "warn-1",
            "issue_reviews": {},
            "completed_at": None,
        }
    )
    report["pipeline_state"].update(
        {"status": "awaiting_external", "next_module": "manual_review"}
    )
    report["report_revision"] = 3
    report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    before = report_path.read_bytes()

    with pytest.raises(TaskStateError, match="manual|semantic|next_module|blocked"):
        if action == "confirm":
            service.confirm_pending(ASSET_ID, expected_revision=3, lease_token=LEASE)
        else:
            service.cancel_pending(ASSET_ID, expected_revision=3, lease_token=LEASE)

    assert report_path.read_bytes() == before


@pytest.mark.parametrize("operation", ["get", "confirm", "complete"])
def test_external_immutable_canonical_change_fails_closed(
    tmp_path: Path, operation: str
) -> None:
    service = _service(tmp_path)
    service.begin_boundary_edit(ASSET_ID, boundary_request(1, 60, revision=1))
    if operation == "complete":
        service.confirm_pending(ASSET_ID, expected_revision=2, lease_token=LEASE)
    source_path = service._assets[ASSET_ID]
    _mutate_external_immutable_annotation(source_path)
    if operation == "complete":
        report_path = service.report_path(ASSET_ID)
        report = load_asset_qc_report(report_path)
        assert report is not None
        report["semantic_calibration"]["base_hdf5_sha256"] = hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest()
        report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(TaskStateError, match="source identity|canonical|immutable|verb"):
        if operation == "get":
            service.get_task(ASSET_ID)
        elif operation == "confirm":
            service.confirm_pending(ASSET_ID, expected_revision=2, lease_token=LEASE)
        else:
            service.complete(ASSET_ID, expected_revision=3, lease_token=LEASE)


def test_corrupt_working_timeline_rejects_boolean_frame_values(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.begin_boundary_edit(ASSET_ID, boundary_request(1, 60, revision=1))
    report_path = service.report_path(ASSET_ID)
    report = load_asset_qc_report(report_path)
    assert report is not None
    report["semantic_calibration"]["working_timeline"][0]["start_frame"] = False
    report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(TaskStateError, match="integer|frame|working_timeline"):
        service.get_task(ASSET_ID)


def test_replace_then_report_failure_recovers_finalizing_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(tmp_path)
    real_update = __import__("semantic_calibration.service", fromlist=["update_human_state"]).update_human_state
    calls = 0

    def fail_second_update(path, expected_revision, mutate):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("report commit failed after replace")
        return real_update(path, expected_revision, mutate)

    monkeypatch.setattr("semantic_calibration.service.update_human_state", fail_second_update)
    with pytest.raises(RuntimeError, match="after replace"):
        service.complete(ASSET_ID, expected_revision=1, lease_token=LEASE)

    report = load_asset_qc_report(service.report_path(ASSET_ID))
    assert report is not None
    assert report["semantic_calibration"]["state"] == "finalizing"

    recovered = SemanticCalibrationService(
        assets={ASSET_ID: service._assets[ASSET_ID]},
        reports={ASSET_ID: service.report_path(ASSET_ID)},
        leases={ASSET_ID: LEASE},
    ).get_task(ASSET_ID)
    assert recovered.report_state == "completed"
    assert recovered.semantic_consistency_state == "completed"


def test_finalizing_record_source_path_tamper_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(tmp_path)

    def fail_replace(prepared) -> None:
        raise RuntimeError("replace interrupted")

    monkeypatch.setattr("semantic_calibration.service.commit_hdf5_replacement", fail_replace)
    with pytest.raises(RuntimeError, match="interrupted"):
        service.complete(ASSET_ID, expected_revision=1, lease_token=LEASE)
    monkeypatch.undo()

    report_path = service.report_path(ASSET_ID)
    report = load_asset_qc_report(report_path)
    assert report is not None
    foreign_path = tmp_path / "foreign" / f"{ASSET_ID}.hdf5"
    foreign_path.parent.mkdir()
    foreign_path.write_bytes(service._assets[ASSET_ID].read_bytes())
    report["semantic_calibration"]["finalizing_record"]["source_path"] = str(foreign_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    foreign_before = foreign_path.read_bytes()

    restarted = SemanticCalibrationService(
        assets={ASSET_ID: service._assets[ASSET_ID]},
        reports={ASSET_ID: report_path},
        leases={ASSET_ID: LEASE},
    )
    with pytest.raises(Hdf5CommitError, match="source_path"):
        restarted.get_task(ASSET_ID)
    assert foreign_path.read_bytes() == foreign_before


def test_finalizing_wrong_pipeline_cursor_fails_closed_before_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(tmp_path)

    monkeypatch.setattr(
        "semantic_calibration.service.commit_hdf5_replacement",
        lambda prepared: (_ for _ in ()).throw(RuntimeError("replace interrupted")),
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        service.complete(ASSET_ID, expected_revision=1, lease_token=LEASE)

    report_path = service.report_path(ASSET_ID)
    report = load_asset_qc_report(report_path)
    assert report is not None
    report["pipeline_state"]["next_module"] = "video_quality"
    report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

    restarted = SemanticCalibrationService(
        assets={ASSET_ID: service._assets[ASSET_ID]},
        reports={ASSET_ID: report_path},
        leases={ASSET_ID: LEASE},
    )
    with pytest.raises(TaskStateError, match="pipeline|semantic|next_module"):
        restarted.get_task(ASSET_ID)


def test_finalizing_task_is_recovered_before_read(tmp_path: Path) -> None:
    report = make_v2_report(status="awaiting_external")
    report["pipeline_state"]["next_module"] = "semantic_consistency"
    report["manual_review"].update(
        {
            "required": False,
            "state": "not_required",
            "selected_issue_ids": [],
            "selected_issue_id": None,
            "issue_reviews": {},
            "completed_at": None,
        }
    )
    report["semantic_calibration"] = {
        "state": "finalizing",
        "source_dataset_path": DATASET_PATH,
        "base_hdf5_sha256": "sha256:" + "0" * 64,
        "final_hdf5_sha256": None,
        "timeline_edit_count": 0,
        "subtask_text_edit_count": 0,
        "pending_edit": None,
        "audit": [],
    }
    service = _service(tmp_path, report=report)
    with pytest.raises(Hdf5CommitError, match="finalizing|staging|record"):
        service.get_task(ASSET_ID)
