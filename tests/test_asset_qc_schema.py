from typing import Any

import pytest

from qc_common.schema import validate_asset_qc_report


def make_valid_running_report() -> dict[str, Any]:
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
                "result_gate": {"verdict": "pass", "has_fail": False, "has_warn": False},
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


def test_running_report_requires_null_overall_decision() -> None:
    report = make_valid_running_report()
    report["overall_decision"] = "warn"

    with pytest.raises(ValueError, match="overall_decision"):
        validate_asset_qc_report(report)


def test_stopped_report_accepts_fail_decision() -> None:
    report = make_valid_running_report()
    report["pipeline_state"] = {
        "status": "stopped",
        "last_completed_module": "video_quality",
        "next_module": "batch_statistics",
    }
    report["overall_decision"] = "fail"
    report["video_quality"]["flow"]["result_gate"] = {
        "verdict": "fail",
        "has_fail": True,
        "has_warn": False,
    }
    report["video_quality"]["flow"]["exit_gate"] = {
        "state": "stop_qc",
        "continue_to_next_module": False,
        "next_module": "batch_statistics",
    }

    validate_asset_qc_report(report)


def test_pre_video_report_does_not_require_video_block() -> None:
    report = make_valid_running_report()
    report["pipeline_state"] = {
        "status": "pending",
        "last_completed_module": "keypoint_temporal",
        "next_module": "video_quality",
    }
    del report["video_quality"]

    validate_asset_qc_report(report)


def test_completed_video_gate_requires_video_block() -> None:
    report = make_valid_running_report()
    del report["video_quality"]

    with pytest.raises(ValueError, match="video_quality"):
        validate_asset_qc_report(report)
