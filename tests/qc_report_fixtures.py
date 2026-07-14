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
) -> dict[str, Any]:
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
