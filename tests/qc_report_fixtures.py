from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from qc_common.config import LoadedQcConfig, load_qc_acceptance_config
from qc_pipeline.context import AssetContext


def make_asset_context(tmp_path: Path, asset_id: str) -> AssetContext:
    return AssetContext(
        asset_id=asset_id,
        batch_root=tmp_path,
        report_path=tmp_path / "quality_archive" / f"{asset_id}.json",
        source_files={},
    )


def loaded_test_config() -> LoadedQcConfig:
    return load_qc_acceptance_config()


def make_v1_video_report() -> dict[str, Any]:
    return {
        "schema_version": "asset_qc_report.v1",
        "qc_config": {
            "schema_version": "qc_acceptance_config_schema.v1",
            "config_version": "qc_acceptance_v1.1.0",
            "config_name": "acceptance_gate",
            "config_path": "configs/qc_acceptance.yaml",
            "config_hash": "sha256:" + "0" * 64,
        },
        "asset_id": "408817",
        "report_revision": 1,
        "pipeline_state": {
            "status": "running",
            "last_completed_module": "video_quality",
            "next_module": "sam3_containment",
        },
        "overall_decision": None,
        "issues": [],
        "manual_review": {
            "required": None,
            "state": "not_evaluated",
            "candidate_issue_ids": [],
            "failures_for_batch_stats_issue_ids": [],
        },
        "video_quality": {
            "flow": {
                "entry_gate": {
                    "state": "ready",
                    "eligible": True,
                    "blocked_by_module": None,
                    "required_inputs": ["source_files.video.path"],
                    "missing_inputs": [],
                    "upstream_continue": True,
                },
                "result_gate": {
                    "verdict": "pass",
                    "has_fail": False,
                    "has_warn": False,
                },
                "exit_gate": {
                    "state": "continue",
                    "continue_to_next_module": True,
                    "next_module": "sam3_containment",
                },
            },
            "evaluation": {
                "decision": "pass",
                "reasons": [],
                "warn_reasons": [],
                "issue_ids": [],
                "should_run_mask_qc": True,
            },
            "metrics": {},
        },
    }


def make_v2_report(
    *,
    status: str = "running",
    overall_decision: str | None = None,
    pipeline_status: str | None = None,
) -> dict[str, Any]:
    if pipeline_status is not None:
        status = pipeline_status
    report = make_v1_video_report()
    report.update(
        {
            "schema_version": "asset_qc_report.v2",
            "qc_config": {
                "schema_version": "qc_acceptance_config_schema.v2",
                "config_version": "qc_acceptance_v2.0.0",
                "config_name": "acceptance_gate",
                "config_path": "configs/qc_acceptance.yaml",
                "config_hash": "sha256:" + "1" * 64,
            },
            "execution": {
                "profile": "acceptance",
                "started_at": None,
                "updated_at": None,
            },
            "pipeline_state": {
                "status": status,
                "last_completed_module": "video_quality",
                "next_module": "sam3_containment",
                "stop_reason": None,
            },
            "overall_decision": overall_decision,
            "source_files": {},
            "runtime_errors": [],
        }
    )
    return copy.deepcopy(report)


def make_boundary_edit(
    *,
    affected_segment_ids: list[str] | tuple[str, ...] = ("segment-0", "segment-1"),
) -> dict[str, Any]:
    """Return a compact boundary pending-edit fixture for human-report tests."""

    ids = list(affected_segment_ids)
    before = [
        {
            "internal_id": segment_id,
            "start_frame": index * 10,
            "end_frame_exclusive": (index + 1) * 10,
            "text_cn": f"before-{index}",
            "text_en": f"before-{index}",
        }
        for index, segment_id in enumerate(ids)
    ]
    after = copy.deepcopy(before)
    return {
        "edit_type": "boundary",
        "boundary_id": "b1",
        "boundary_index": 1,
        "actor_segment_id": ids[0] if ids else "segment-0",
        "affected_segment_ids": ids,
        "before": before,
        "after": after,
        "reviewer": "alice",
        "created_at": "2026-07-15T00:00:00Z",
    }


def make_text_edit(*, affected_segment_ids: list[str] | tuple[str, ...] = ("segment-0",)) -> dict[str, Any]:
    ids = list(affected_segment_ids)
    segment_id = ids[0] if ids else "segment-0"
    before = {
        "internal_id": segment_id,
        "start_frame": 0,
        "end_frame_exclusive": 10,
        "text_cn": "旧文本",
        "text_en": "old text",
    }
    after = copy.deepcopy(before)
    after.update({"text_cn": "新文本", "text_en": "new text"})
    return {
        "edit_type": "text",
        "segment_id": segment_id,
        "affected_segment_ids": ids,
        "before": before,
        "after": after,
        "reviewer": "alice",
        "created_at": "2026-07-15T00:00:00Z",
    }


def make_semantic_block(
    *,
    state: str = "not_started",
    pending_edit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "state": state,
        "source_dataset_path": "/label/subtask_label",
        "base_hdf5_sha256": "sha256:" + "a" * 64,
        "final_hdf5_sha256": None,
        "timeline_edit_count": 0,
        "subtask_text_edit_count": 0,
        "pending_edit": pending_edit,
        "audit": [],
    }


def make_review(verdict: str = "pass") -> dict[str, Any]:
    return {
        "verdict": verdict,
        "reviewer": "alice",
        "reviewed_at": "2026-07-15T00:00:00Z",
    }


def make_manual_block(
    *,
    state: str = "not_evaluated",
    candidate_issue_ids: list[str] | tuple[str, ...] = (),
    selected_issue_ids: list[str] | tuple[str, ...] = (),
    issue_reviews: dict[str, Any] | None = None,
    completed_at: str | None = None,
) -> dict[str, Any]:
    if state == "completed" and completed_at is None:
        completed_at = "2026-07-15T00:00:00Z"
    return {
        "required": None,
        "state": state,
        "candidate_issue_ids": list(candidate_issue_ids),
        "failures_for_batch_stats_issue_ids": [],
        "selected_issue_ids": list(selected_issue_ids),
        "selected_issue_id": None,
        "issue_reviews": {} if issue_reviews is None else copy.deepcopy(issue_reviews),
        "completed_at": completed_at,
    }
