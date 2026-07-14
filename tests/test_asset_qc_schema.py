import pytest

from qc_common.schema import validate_asset_qc_report
from tests.qc_report_fixtures import make_v1_video_report


def test_running_report_requires_null_overall_decision() -> None:
    report = make_v1_video_report()
    report["overall_decision"] = "warn"

    with pytest.raises(ValueError, match="overall_decision"):
        validate_asset_qc_report(report)


def test_stopped_report_accepts_fail_decision() -> None:
    report = make_v1_video_report()
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
    report = make_v1_video_report()
    report["pipeline_state"] = {
        "status": "pending",
        "last_completed_module": "keypoint_temporal",
        "next_module": "video_quality",
    }
    del report["video_quality"]

    validate_asset_qc_report(report)


def test_completed_video_gate_requires_video_block() -> None:
    report = make_v1_video_report()
    del report["video_quality"]

    with pytest.raises(ValueError, match="video_quality"):
        validate_asset_qc_report(report)
