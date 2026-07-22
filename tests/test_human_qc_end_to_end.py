from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from human_qc.evidence import EvidenceService
from human_qc.semantic_service import PendingEditError, SemanticCalibrationService
from human_qc.warn_service import (
    WarnReviewService,
    WarnStateError,
    reduce_overall_decision,
)
from human_qc.workbench_service import WorkbenchService
from qc_common.report import load_asset_qc_report, write_asset_qc_report
from qc_pipeline.context import AssetContext
from qc_reporting.aggregate import aggregate_projection
from qc_reporting.projection import project_quality_archive
from tests.qc_report_fixtures import make_v2_report


DATASET_PATH = "/label/subtask_label"
NOW = "2026-07-15T08:00:00+00:00"


@dataclass(frozen=True)
class FileAsset:
    asset_id: str
    root: Path
    hdf5_path: Path
    report_path: Path
    video_path: Path
    context: AssetContext


def _annotations() -> list[dict[str, Any]]:
    result = []
    for index, (start, end) in enumerate(((0, 29), (30, 59), (60, 89))):
        result.append(
            {
                "start_frame": start,
                "end_frame": end,
                "start_time_sec": start / 30,
                "end_time_sec": end / 30,
                "subtask_cn": f"步骤{index + 1}",
                "subtask_en": f"step {index + 1}",
                "verb": "move",
                "object": "block",
                "target": "tray",
                "hand": "right",
                "phase": f"phase-{index + 1}",
                "evidence_frames": [start, end],
                "confidence": 0.95,
                "status": "confirmed",
            }
        )
    return result


def _payload(asset_id: str) -> dict[str, Any]:
    return {
        "id": asset_id,
        "scene": "fixture-kitchen",
        "task": "move blocks",
        "fps": 30.0,
        "frame_count": 90,
        "annotations": _annotations(),
    }


def _warn_issue(issue_id: str, start: int, end: int) -> dict[str, Any]:
    skeleton = {
        "frames": [
            {"frame_idx": frame, "points": [[point, point + 0.5] for point in range(21)]}
            for frame in range(90)
        ]
    }
    return {
        "issue_id": issue_id,
        "code": "machine_warn",
        "severity": "warn",
        "module": "video_quality",
        "issue_type": "metric_threshold",
        "metric": "blur_score",
        "observed_value": 0.2,
        "operator": "<",
        "boundary_value": 0.5,
        "rule_id": f"rule-{issue_id}",
        "needs_manual_review": True,
        "start_frame": start,
        "end_frame_exclusive": end,
        "skeleton": skeleton,
        "context": {"reason": "machine-selected warning"},
    }


def _hard_fail_issue() -> dict[str, Any]:
    return {
        "issue_id": "hard-fail-1",
        "code": "missing_required_signal",
        "severity": "fail",
        "module": "precheck",
        "issue_type": "missing_input",
        "metric": "finite_keypoints",
        "observed_value": 20,
        "operator": "<",
        "boundary_value": 21,
        "rule_id": "hard-rule-1",
        "needs_manual_review": False,
        "context": {},
    }


def build_file_asset(
    tmp_path: Path,
    *,
    asset_id: str = "asset-e2e",
    selected_warn_ids: tuple[str, ...] = ("warn-left", "warn-right"),
    candidate_warn_ids: tuple[str, ...] | None = None,
    machine_warn_ids: tuple[str, ...] = ("warn-left", "warn-right"),
    hard_fail: bool = True,
    profile: str = "supplier_evaluation",
) -> FileAsset:
    root = tmp_path / asset_id
    root.mkdir(parents=True)
    hdf5_path = root / "source" / f"{asset_id}.hdf5"
    hdf5_path.parent.mkdir()
    with h5py.File(hdf5_path, "w") as handle:
        handle.attrs["supplier"] = "fixture-supplier"
        handle.attrs["matrix"] = np.arange(4, dtype=np.int16).reshape(2, 2)
        label = handle.create_group("label")
        label.attrs["owner"] = "supplier"
        target = label.create_dataset("subtask_label", shape=(), dtype="S65536")
        target.attrs["schema"] = "closed-subtask-json"
        target[()] = json.dumps(_payload(asset_id), ensure_ascii=False).encode("utf-8")
        sensors = handle.create_group("sensors")
        sensors.attrs["rate_hz"] = np.float32(30)
        frames = sensors.create_dataset(
            "frames",
            data=np.arange(72, dtype=np.int32).reshape(6, 3, 4),
            chunks=(2, 3, 4),
            compression="gzip",
        )
        frames.attrs["units"] = "raw"
        nested = sensors.create_group("nested")
        nested.create_dataset("names", data=np.asarray([b"left", b"right"], dtype="S8"))

    video_path = root / "video" / f"{asset_id}.mp4"
    video_path.parent.mkdir()
    video_path.write_bytes(b"fixture-video-bytes")

    windows = {"warn-left": (5, 15), "warn-right": (45, 55)}
    warn_issues = [
        _warn_issue(issue_id, *windows[issue_id]) for issue_id in machine_warn_ids
    ]
    issues = warn_issues + ([_hard_fail_issue()] if hard_fail else [])
    candidates = list(selected_warn_ids if candidate_warn_ids is None else candidate_warn_ids)
    report = make_v2_report(status="awaiting_external")
    report["asset_id"] = asset_id
    report["execution"]["profile"] = profile
    report["execution"]["module_states"] = {
        "machine_pass": {"state": "completed", "verdict": "pass"},
        "machine_warn": {
            "state": "completed",
            "verdict": "warn" if warn_issues else "pass",
        },
        "machine_fail": {"state": "completed", "verdict": "fail" if hard_fail else "pass"},
    }
    report["pipeline_state"] = {
        "status": "awaiting_external",
        "last_completed_module": "video_quality",
        "next_module": "manual_review" if selected_warn_ids else "semantic_consistency",
        "stop_reason": None,
    }
    report["source_files"] = {"hdf5": {"path": str(hdf5_path)}}
    report["issues"] = issues
    report["manual_review"] = {
        "required": bool(selected_warn_ids),
        "state": "queued" if selected_warn_ids else "not_required",
        "candidate_issue_ids": candidates,
        "failures_for_batch_stats_issue_ids": [],
        "selected_issue_ids": list(selected_warn_ids),
        "selected_issue_id": selected_warn_ids[0] if selected_warn_ids else None,
        "issue_reviews": {},
        "completed_at": None,
    }
    report_path = root / "quality_archive" / f"{asset_id}.json"
    write_asset_qc_report(report_path, report, expected_revision=0, profile=profile)
    context = AssetContext(
        asset_id=asset_id,
        batch_root=root,
        report_path=report_path,
        source_files={
            "video": {"path": video_path.relative_to(root).as_posix()},
            "hdf5": {"path": hdf5_path.relative_to(root).as_posix()},
        },
        source_range=(0, 90),
        metadata={"fps": 30},
    )
    return FileAsset(asset_id, root, hdf5_path, report_path, video_path, context)


def make_workbench(asset: FileAsset, *, evidence: EvidenceService | None = None) -> WorkbenchService:
    semantic = SemanticCalibrationService(
        assets={asset.asset_id: asset.hdf5_path},
        reports={asset.asset_id: asset.report_path},
    )
    warn = WarnReviewService(
        reports={asset.asset_id: asset.report_path}, reviewer="alice", clock=lambda: NOW
    )
    return WorkbenchService(
        semantic,
        warn,
        evidence,
        asset_contexts={asset.asset_id: asset.context},
        profile="supplier_evaluation",
    )


def _encoded(value: object) -> tuple[str, tuple[int, ...], bytes | str]:
    array = np.asarray(value)
    if array.dtype.kind in {"O", "U"}:
        payload: bytes | str = repr(array.tolist())
    else:
        payload = array.tobytes()
    return array.dtype.str, array.shape, payload


def snapshot_hdf5(path: Path) -> dict[str, tuple[Any, ...]]:
    snapshot: dict[str, tuple[Any, ...]] = {}
    with h5py.File(path, "r") as handle:
        snapshot["/"] = ("group", tuple(sorted((key, _encoded(value)) for key, value in handle.attrs.items())))

        def visit(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            attrs = tuple(sorted((key, _encoded(value)) for key, value in obj.attrs.items()))
            if isinstance(obj, h5py.Group):
                snapshot[f"/{name}"] = ("group", attrs)
                return
            snapshot[f"/{name}"] = (
                "dataset",
                obj.dtype.str,
                obj.shape,
                obj.chunks,
                obj.compression,
                attrs,
                _encoded(obj[()]),
            )

        handle.visititems(visit)
    return snapshot


def _assert_only_subtask_data_changed(before: dict[str, tuple[Any, ...]], after: dict[str, tuple[Any, ...]]) -> None:
    assert before.keys() == after.keys()
    for path in before:
        if path != DATASET_PATH:
            assert after[path] == before[path], path
            continue
        assert after[path][:-1] == before[path][:-1]
        assert after[path][-1] != before[path][-1]


def _assert_half_open_coverage(annotations: list[dict[str, Any]], frame_count: int) -> None:
    half_open = [(row["start_frame"], row["end_frame"] + 1) for row in annotations]
    assert half_open[0][0] == 0
    assert half_open[-1][1] == frame_count
    assert all(left[1] == right[0] for left, right in zip(half_open, half_open[1:]))


def test_file_level_warn_then_semantic_workflow_preserves_source_fidelity(tmp_path: Path) -> None:
    asset = build_file_asset(tmp_path)
    before_hdf5 = snapshot_hdf5(asset.hdf5_path)
    before_report = load_asset_qc_report(asset.report_path)
    assert before_report is not None
    machine_issues = copy.deepcopy(before_report["issues"])
    workbench = make_workbench(asset)
    lease = workbench.acquire_lease(asset.asset_id, "alice", 120)

    task = workbench.get_asset_task(asset.asset_id)
    assert task["task_type"] == "warn_review"
    assert task["warn"]["selected_issue_ids"] == ["warn-left", "warn-right"]

    first_review = workbench.warn_verdict(
        asset.asset_id,
        issue_id="warn-left",
        verdict="pass",
        reason="false positive",
        expected_revision=task["revision"],
        lease_token=lease.token,
    )
    with pytest.raises(WarnStateError, match="every selected issue"):
        workbench.warn_complete(
            asset.asset_id,
            expected_revision=first_review["revision"],
            lease_token=lease.token,
        )
    incomplete = load_asset_qc_report(asset.report_path)
    assert incomplete is not None
    assert incomplete["manual_review"]["state"] == "in_progress"
    assert incomplete["overall_decision"] is None
    second_review = workbench.warn_verdict(
        asset.asset_id,
        issue_id="warn-right",
        verdict="pass",
        reason="false positive",
        expected_revision=first_review["revision"],
        lease_token=lease.token,
    )
    semantic_task = workbench.warn_complete(
        asset.asset_id,
        expected_revision=second_review["revision"],
        lease_token=lease.token,
    )
    assert semantic_task["task_type"] == "semantic_calibration"
    assert len(semantic_task["semantic"]["timeline"]["segments"]) == 3

    left = workbench.semantic_boundary_pending(
        asset.asset_id,
        boundary_index=1,
        new_frame_exclusive=35,
        expected_revision=semantic_task["revision"],
        lease_token=lease.token,
        reviewer="alice",
    )
    pending = left["semantic"]["pending_edit"]
    assert len(pending["affected_segment_ids"]) == 2
    assert len(pending["before"]) == len(pending["after"]) == 2
    assert [(row["start_frame"], row["end_frame_exclusive"]) for row in pending["after"]] == [
        (0, 35),
        (35, 60),
    ]
    with pytest.raises(PendingEditError, match="pending"):
        workbench.semantic_complete(
            asset.asset_id, expected_revision=left["revision"], lease_token=lease.token
        )
    confirmed = workbench.semantic_pending_confirm(
        asset.asset_id, expected_revision=left["revision"], lease_token=lease.token
    )
    assert confirmed["semantic"]["timeline_edit_count"] == 1
    assert [row["end_frame_exclusive"] for row in confirmed["semantic"]["timeline"]["segments"]] == [35, 60, 90]

    right = workbench.semantic_boundary_pending(
        asset.asset_id,
        boundary_index=2,
        new_frame_exclusive=65,
        expected_revision=confirmed["revision"],
        lease_token=lease.token,
        reviewer="alice",
    )
    cancelled = workbench.semantic_pending_cancel(
        asset.asset_id, expected_revision=right["revision"], lease_token=lease.token
    )
    assert cancelled["semantic"]["timeline_edit_count"] == 1
    assert [row["end_frame_exclusive"] for row in cancelled["semantic"]["timeline"]["segments"]] == [35, 60, 90]

    segment_id = cancelled["semantic"]["timeline"]["segments"][1]["internal_id"]
    text_pending = workbench.semantic_text_pending(
        asset.asset_id,
        segment_id=segment_id,
        text_cn="新的第二步",
        text_en="new second step",
        expected_revision=cancelled["revision"],
        lease_token=lease.token,
        reviewer="alice",
    )
    text_confirmed = workbench.semantic_pending_confirm(
        asset.asset_id,
        expected_revision=text_pending["revision"],
        lease_token=lease.token,
    )
    assert text_confirmed["semantic"]["subtask_text_edit_count"] == 1
    completed = workbench.semantic_complete(
        asset.asset_id,
        expected_revision=text_confirmed["revision"],
        lease_token=lease.token,
    )
    assert completed["task_type"] == "completed"

    persisted = load_asset_qc_report(asset.report_path)
    assert persisted is not None
    assert persisted["issues"] == machine_issues
    assert persisted["manual_review"]["issue_reviews"]["warn-left"]["effective_verdict"] == "pass"
    assert persisted["manual_review"]["issue_reviews"]["warn-right"]["effective_verdict"] == "pass"
    assert persisted["manual_review"]["completion_mode"] == "all_reviewed"
    assert persisted["manual_review"]["failure_reason"] is None
    assert persisted["overall_decision"] == "fail"
    assert persisted["pipeline_state"]["status"] == "completed"
    assert persisted["pipeline_state"]["next_module"] is None
    assert persisted["semantic_calibration"]["state"] == "completed"
    assert persisted["semantic_calibration"]["timeline_edit_count"] == 1
    assert persisted["semantic_calibration"]["subtask_text_edit_count"] == 1

    projection = project_quality_archive(asset.report_path.parent)
    asset_row = next(row for row in projection.asset_rows if row["asset_id"] == asset.asset_id)
    assert asset_row["status"] == "completed"
    assert asset_row["decision"] == "fail"
    batch = aggregate_projection(projection)
    assert batch["overall"]["final_fail_assets"] == 1
    assert batch["overall"]["timeline_edit_count"] == 1
    assert batch["overall"]["subtask_text_edit_count"] == 1

    after_hdf5 = snapshot_hdf5(asset.hdf5_path)
    _assert_only_subtask_data_changed(before_hdf5, after_hdf5)
    with h5py.File(asset.hdf5_path, "r") as handle:
        raw = handle[DATASET_PATH][()]
        canonical = json.loads(bytes(raw).decode("utf-8"))
    assert set(canonical) == {"id", "scene", "task", "fps", "frame_count", "annotations"}
    expected_annotation_fields = set(_annotations()[0])
    assert all(set(row) == expected_annotation_fields for row in canonical["annotations"])
    assert canonical["annotations"][1]["subtask_cn"] == "新的第二步"
    _assert_half_open_coverage(canonical["annotations"], canonical["frame_count"])
    assert not list(asset.hdf5_path.parent.glob("*.bak"))
    assert not list(asset.hdf5_path.parent.glob(f".{asset.hdf5_path.name}.human-qc-*"))


def test_only_selected_warn_is_reviewable_and_all_pass_cannot_override_hard_fail(
    tmp_path: Path,
) -> None:
    asset = build_file_asset(
        tmp_path,
        asset_id="asset-hard-fail",
        selected_warn_ids=("warn-left",),
        candidate_warn_ids=("warn-left", "warn-right"),
        hard_fail=True,
        profile="supplier_evaluation",
    )
    workbench = make_workbench(asset)
    lease = workbench.acquire_lease(asset.asset_id, "alice", 120)
    warn_task = workbench.get_asset_task(asset.asset_id)

    assert warn_task["warn"]["selected_issue_ids"] == ["warn-left"]
    assert [issue["issue_id"] for issue in warn_task["warn"]["issues"]] == ["warn-left"]
    with pytest.raises(WarnStateError, match="selected"):
        workbench.warn_verdict(
            asset.asset_id,
            issue_id="warn-right",
            verdict="pass",
            expected_revision=warn_task["revision"],
            lease_token=lease.token,
        )
    reviewed = workbench.warn_verdict(
        asset.asset_id,
        issue_id="warn-left",
        verdict="pass",
        expected_revision=warn_task["revision"],
        lease_token=lease.token,
    )
    semantic_task = workbench.warn_complete(
        asset.asset_id,
        expected_revision=reviewed["revision"],
        lease_token=lease.token,
    )
    workbench.semantic_complete(
        asset.asset_id,
        expected_revision=semantic_task["revision"],
        lease_token=lease.token,
    )

    report = load_asset_qc_report(asset.report_path)
    assert report is not None
    assert report["manual_review"]["issue_reviews"]["warn-left"]["effective_verdict"] == "pass"
    assert "warn-right" not in report["manual_review"]["issue_reviews"]
    assert report["manual_review"]["completion_mode"] == "all_reviewed"
    assert report["manual_review"]["failure_reason"] is None
    assert report["pipeline_state"]["status"] == "completed"
    assert report["pipeline_state"]["next_module"] is None
    assert report["overall_decision"] == "fail"
    assert reduce_overall_decision(report) == "fail"


def test_no_selected_warns_skip_manual_review_and_finish_pass(tmp_path: Path) -> None:
    asset = build_file_asset(
        tmp_path,
        asset_id="asset-no-warn",
        selected_warn_ids=(),
        machine_warn_ids=(),
        hard_fail=False,
        profile="acceptance",
    )
    workbench = make_workbench(asset)
    lease = workbench.acquire_lease(asset.asset_id, "alice", 120)
    task = workbench.get_asset_task(asset.asset_id)
    completed = workbench.semantic_complete(
        asset.asset_id, expected_revision=task["revision"], lease_token=lease.token
    )
    assert completed["task_type"] == "completed"
    report = load_asset_qc_report(asset.report_path)
    assert report is not None
    assert report["issues"] == []
    assert report["execution"]["module_states"]["machine_warn"]["verdict"] == "pass"
    assert report["manual_review"]["state"] == "not_required"
    assert report["overall_decision"] == "pass"


def test_evidence_fixture_contains_half_open_video_window_and_21_point_skeleton(tmp_path: Path) -> None:
    asset = build_file_asset(tmp_path, asset_id="asset-evidence", hard_fail=False)
    calls: list[list[str]] = []

    def fake_ffmpeg(command: Any, output: Path) -> None:
        calls.append(list(command))
        output.write_bytes(b"clip")

    captured: dict[str, Any] = {}

    def render_overlay(issue: dict[str, Any], output: Path, start: int, end: int) -> None:
        captured.update(issue=issue, start=start, end=end)
        output.write_bytes(b"overlay")

    evidence = EvidenceService(
        asset.root / "cache", ffmpeg_runner=fake_ffmpeg, overlay_renderer=render_overlay
    )
    report = load_asset_qc_report(asset.report_path)
    assert report is not None
    view = evidence.resolve(report["issues"][0], asset.context)
    assert (view.start_frame, view.end_frame_exclusive) == (5, 15)
    assert len(captured["issue"]["skeleton"]["frames"]) == 10
    assert all(len(frame["points"]) == 21 for frame in captured["issue"]["skeleton"]["frames"])
    assert "start_frame=5" in " ".join(calls[0])
    assert "end_frame=15" in " ".join(calls[0])
