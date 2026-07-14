from __future__ import annotations

import copy
import json
from pathlib import Path

import h5py
import pytest

from human_qc.contracts import SubtaskSegment
from human_qc.hdf5_commit import Hdf5CommitError
from human_qc.semantic_service import (
    BoundaryEditRequest,
    SemanticCalibrationService,
    SemanticTaskView,
    TextEditRequest,
)
from human_qc.source_adapters import Hdf5ScalarJsonSubtaskAdapter
from human_qc.timeline import SharedBoundaryTimeline
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
    value["source_files"] = {"hdf5": {"path": str(hdf5_path)}}
    write_asset_qc_report(report_path, value, expected_revision=0, profile="acceptance")
    return hdf5_path, report_path


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


def test_complete_prepare_failure_leaves_report_and_hdf5_unpublished(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(tmp_path)
    before_hdf5 = service.get_task(ASSET_ID).hdf5_sha256
    monkeypatch.setattr("human_qc.semantic_service.prepare_hdf5_replacement", lambda *a, **k: (_ for _ in ()).throw(Hdf5CommitError("prepare failed")))
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


def test_replace_then_report_failure_recovers_finalizing_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(tmp_path)
    real_update = __import__("human_qc.semantic_service", fromlist=["update_human_state"]).update_human_state
    calls = 0

    def fail_second_update(path, expected_revision, mutate):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("report commit failed after replace")
        return real_update(path, expected_revision, mutate)

    monkeypatch.setattr("human_qc.semantic_service.update_human_state", fail_second_update)
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


def test_finalizing_task_is_recovered_before_read(tmp_path: Path) -> None:
    report = make_v2_report(status="awaiting_external")
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
