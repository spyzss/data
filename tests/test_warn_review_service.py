from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from human_qc.warn_service import (
    WarnLeaseError,
    WarnReviewService,
    WarnRevisionError,
    WarnStateError,
    effective_issue_verdict,
    reduce_overall_decision,
)
from qc_common.report import load_asset_qc_report, write_asset_qc_report
from tests.qc_report_fixtures import make_v2_report


ASSET_ID = "617856"
LEASE = "lease-alice"
NOW = "2026-07-14T12:00:00+00:00"


def _issue(issue_id: str = "warn-1", *, severity: str = "warn") -> dict:
    return {
        "issue_id": issue_id,
        "code": "warn_code" if severity == "warn" else "fail_code",
        "severity": severity,
        "module": "video_quality",
        "issue_type": "metric_threshold",
        "metric": "blur_score",
        "observed_value": 0.2,
        "operator": "<",
        "boundary_value": 0.5,
        "rule_id": "rule-1",
        "needs_manual_review": severity == "warn",
        "context": {},
    }


def _report(
    tmp_path: Path,
    *,
    selected: list[str] | None = None,
    candidates: list[str] | None = None,
    issues: list[dict] | None = None,
    pipeline_status: str = "awaiting_external",
    next_module: str = "manual_review",
) -> Path:
    candidates = ["warn-1"] if candidates is None else candidates
    selected = list(candidates) if selected is None else selected
    report = make_v2_report(status=pipeline_status)
    report["asset_id"] = ASSET_ID
    report["report_revision"] = 1
    report["pipeline_state"] = {
        "status": pipeline_status,
        "last_completed_module": "semantic_consistency",
        "next_module": next_module,
        "stop_reason": None,
    }
    report["semantic_calibration"] = {
        "state": "completed",
        "source_dataset_path": "/label/subtask_label",
        "base_hdf5_sha256": "sha256:" + "0" * 64,
        "final_hdf5_sha256": "sha256:" + "1" * 64,
        "timeline_edit_count": 0,
        "subtask_text_edit_count": 0,
        "pending_edit": None,
        "audit": [],
    }
    report["semantic_consistency"] = {
        "state": "completed",
        "execution_kind": "external",
    }
    report["issues"] = copy.deepcopy(issues or [_issue()])
    report["manual_review"] = {
        "required": bool(selected),
        "state": "queued" if selected else "not_evaluated",
        "candidate_issue_ids": list(candidates),
        "failures_for_batch_stats_issue_ids": [],
        "selected_issue_ids": list(selected),
        "selected_issue_id": selected[0] if selected else None,
        "issue_reviews": {},
        "completed_at": None,
    }
    path = tmp_path / "quality_archive" / f"{ASSET_ID}.json"
    write_asset_qc_report(path, report, expected_revision=0, profile="acceptance")
    return path


def _service(tmp_path: Path, **kwargs) -> WarnReviewService:
    return WarnReviewService(
        reports={ASSET_ID: _report(tmp_path, **kwargs)},
        leases={ASSET_ID: LEASE},
        reviewer="alice",
        clock=lambda: NOW,
    )


def test_warn_module_is_missing_before_implementation() -> None:
    # This test is intentionally collected before the production module exists
    # so the first focused run records the expected RED import failure.
    assert WarnReviewService is not None


def test_human_pass_does_not_mutate_machine_issue(tmp_path: Path) -> None:
    service = _service(tmp_path)
    before = load_asset_qc_report(service.report_path(ASSET_ID))
    assert before is not None
    before_issue = copy.deepcopy(before["issues"][0])

    view = service.submit_verdict(
        ASSET_ID, "warn-1", "pass", None, expected_revision=1, lease_token=LEASE
    )
    after = load_asset_qc_report(service.report_path(ASSET_ID))
    assert after is not None
    assert after["issues"][0] == before_issue
    assert view.issue_reviews["warn-1"]["effective_verdict"] == "pass"
    assert view.issue_reviews["warn-1"]["machine_verdict"] == "warn"


def test_selected_issue_requires_verdict_before_complete(tmp_path: Path) -> None:
    service = _service(tmp_path)
    with pytest.raises(WarnStateError, match="verdict|selected"):
        service.complete(ASSET_ID, expected_revision=1, lease_token=LEASE)

    reviewed = service.submit_verdict(
        ASSET_ID, "warn-1", "fail", "not clear", expected_revision=1, lease_token=LEASE
    )
    assert reviewed.report_revision == 2
    completed = service.complete(ASSET_ID, expected_revision=2, lease_token=LEASE)
    assert completed.state == "completed"
    assert completed.overall_decision == "fail"


def test_resubmission_audits_previous_review_and_increments_once(tmp_path: Path) -> None:
    service = _service(tmp_path)
    first = service.submit_verdict(
        ASSET_ID, "warn-1", "pass", "first", expected_revision=1, lease_token=LEASE
    )
    second = service.submit_verdict(
        ASSET_ID, "warn-1", "fail", "second", expected_revision=2, lease_token=LEASE
    )
    assert first.report_revision == 2
    assert second.report_revision == 3
    report = load_asset_qc_report(service.report_path(ASSET_ID))
    assert report is not None
    assert report["manual_review"]["issue_reviews"]["warn-1"]["verdict"] == "fail"
    assert report["manual_review"]["review_audit"][0]["previous"]["verdict"] == "pass"


@pytest.mark.parametrize("issue_id", ["unknown", "candidate-not-selected"])
def test_unknown_or_unselected_issue_is_rejected_without_write(
    tmp_path: Path, issue_id: str
) -> None:
    service = _service(
        tmp_path,
        candidates=["warn-1", "candidate-not-selected"],
        selected=["warn-1"],
        issues=[_issue("warn-1"), _issue("candidate-not-selected")],
    )
    before = service.get_task(ASSET_ID)
    with pytest.raises(WarnStateError, match="selected|candidate|unknown"):
        service.submit_verdict(
            ASSET_ID, issue_id, "pass", None, expected_revision=1, lease_token=LEASE
        )
    assert service.get_task(ASSET_ID) == before


def test_no_selected_candidates_are_not_required(tmp_path: Path) -> None:
    service = _service(tmp_path, candidates=["warn-1"], selected=[])
    assert service.get_task(ASSET_ID) is None
    completed = service.complete(ASSET_ID, expected_revision=1, lease_token=LEASE)
    assert completed.state == "not_required"
    assert completed.pipeline_state == "completed"
    assert completed.overall_decision == "pass"


def test_terminal_warn_completion_is_rejected(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.submit_verdict(
        ASSET_ID, "warn-1", "pass", None, expected_revision=1, lease_token=LEASE
    )
    service.complete(ASSET_ID, expected_revision=2, lease_token=LEASE)
    with pytest.raises(WarnStateError, match="already completed|terminal"):
        service.complete(ASSET_ID, expected_revision=3, lease_token=LEASE)


def test_stale_revision_and_wrong_lease_are_rejected(tmp_path: Path) -> None:
    service = _service(tmp_path)
    with pytest.raises(WarnRevisionError):
        service.submit_verdict(
            ASSET_ID, "warn-1", "pass", None, expected_revision=0, lease_token=LEASE
        )
    with pytest.raises(WarnLeaseError):
        service.submit_verdict(
            ASSET_ID, "warn-1", "pass", None, expected_revision=1, lease_token="bad"
        )
    assert service.get_task(ASSET_ID).report_revision == 1


def test_pending_semantic_edit_blocks_warn_mutation(tmp_path: Path) -> None:
    report_path = _report(tmp_path)
    report = load_asset_qc_report(report_path)
    assert report is not None
    report["semantic_calibration"]["pending_edit"] = {"edit_type": "text"}
    report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    service = WarnReviewService(
        reports={ASSET_ID: report_path}, leases={ASSET_ID: LEASE}, reviewer="alice"
    )
    with pytest.raises(WarnStateError, match="pending"):
        service.submit_verdict(
            ASSET_ID, "warn-1", "pass", None, expected_revision=1, lease_token=LEASE
        )


@pytest.mark.parametrize(
    ("machine", "review", "expected"),
    [
        ({"severity": "warn"}, None, "warn"),
        ({"severity": "warn"}, {"verdict": "pass"}, "pass"),
        ({"severity": "warn"}, {"verdict": "fail"}, "fail"),
        ({"severity": "fail"}, {"verdict": "pass"}, "fail"),
        ({"severity": "fail", "verdict": "warn"}, {"verdict": "pass"}, "fail"),
    ],
)
def test_effective_verdict_preserves_machine_hard_fail(machine, review, expected) -> None:
    assert effective_issue_verdict(machine, review) == expected


@pytest.mark.parametrize(
    ("status", "machine_fail", "manual_state", "manual_verdict", "expected"),
    [
        ("completed", False, "not_required", None, "pass"),
        ("completed", False, "completed", "pass", "pass"),
        ("completed", False, "completed", "fail", "fail"),
        ("completed", True, "completed", "pass", "fail"),
        ("awaiting_external", False, "queued", None, None),
        ("error", False, "completed", "pass", None),
    ],
)
def test_final_reducer_precedence(
    status, machine_fail, manual_state, manual_verdict, expected
) -> None:
    report = make_v2_report(status=status)
    report["pipeline_state"]["status"] = status
    report["pipeline_state"]["next_module"] = "manual_review"
    report["semantic_calibration"] = {
        "state": "completed",
        "source_dataset_path": "/label/subtask_label",
        "base_hdf5_sha256": "sha256:" + "0" * 64,
        "final_hdf5_sha256": "sha256:" + "1" * 64,
        "timeline_edit_count": 0,
        "subtask_text_edit_count": 0,
        "pending_edit": None,
        "audit": [],
    }
    report["issues"] = [_issue("hard-fail", severity="fail")] if machine_fail else []
    report["manual_review"] = {
        "required": manual_state != "not_required",
        "state": manual_state,
        "candidate_issue_ids": ["warn-1"] if manual_state != "not_required" else [],
        "failures_for_batch_stats_issue_ids": [],
        "selected_issue_ids": ["warn-1"] if manual_state != "not_required" else [],
        "selected_issue_id": "warn-1" if manual_state != "not_required" else None,
        "issue_reviews": (
            {"warn-1": {"verdict": manual_verdict}}
            if manual_verdict
            else {}
        ),
        "completed_at": NOW if manual_state == "completed" else None,
    }
    assert reduce_overall_decision(report) == expected


def test_reducer_does_not_skip_queued_candidates_without_selection() -> None:
    report = make_v2_report(status="completed")
    report["pipeline_state"]["status"] = "completed"
    report["semantic_calibration"] = {
        "state": "completed",
        "source_dataset_path": "/label/subtask_label",
        "base_hdf5_sha256": "sha256:" + "0" * 64,
        "final_hdf5_sha256": "sha256:" + "1" * 64,
        "timeline_edit_count": 0,
        "subtask_text_edit_count": 0,
        "pending_edit": None,
        "audit": [],
    }
    report["manual_review"] = {
        "required": True,
        "state": "queued",
        "candidate_issue_ids": ["warn-1"],
        "failures_for_batch_stats_issue_ids": [],
        "selected_issue_ids": [],
        "selected_issue_id": None,
        "issue_reviews": {},
        "completed_at": None,
    }
    assert reduce_overall_decision(report) is None


def test_machine_hard_fail_precedes_unfinished_semantic_stage() -> None:
    report = make_v2_report(status="completed")
    report["pipeline_state"]["status"] = "completed"
    report["semantic_calibration"] = {
        "state": "in_progress",
        "source_dataset_path": "/label/subtask_label",
        "base_hdf5_sha256": "sha256:" + "0" * 64,
        "final_hdf5_sha256": None,
        "timeline_edit_count": 0,
        "subtask_text_edit_count": 0,
        "pending_edit": None,
        "audit": [],
    }
    report["issues"] = [_issue("hard-fail", severity="fail")]
    report["manual_review"] = {
        "required": False,
        "state": "not_required",
        "candidate_issue_ids": [],
        "failures_for_batch_stats_issue_ids": [],
        "selected_issue_ids": [],
        "selected_issue_id": None,
        "issue_reviews": {},
        "completed_at": None,
    }
    assert reduce_overall_decision(report) == "fail"
