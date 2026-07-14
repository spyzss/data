from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from qc_common.schema import validate_asset_qc_report
from qc_common.report import StaleReportRevisionError, load_asset_qc_report, write_asset_qc_report
from human_qc.report_updates import (
    initialize_manual_review,
    initialize_semantic_calibration,
    reduce_overall_decision,
    update_human_state,
)
from tests.qc_report_fixtures import (
    make_boundary_edit,
    make_manual_block,
    make_review,
    make_semantic_block,
    make_text_edit,
    make_v1_video_report,
    make_v2_report,
)


def test_semantic_pending_boundary_requires_two_affected_segments() -> None:
    report = make_v2_report(pipeline_status="awaiting_external")
    report["semantic_calibration"] = make_semantic_block(
        state="in_progress",
        pending_edit=make_boundary_edit(affected_segment_ids=["only-one"]),
    )

    with pytest.raises(ValueError):
        validate_asset_qc_report(report)


def test_semantic_pending_boundary_requires_two_distinct_segment_ids() -> None:
    report = make_v2_report()
    report["semantic_calibration"] = make_semantic_block(
        state="in_progress",
        pending_edit=make_boundary_edit(
            affected_segment_ids=["segment-0", "segment-0"]
        ),
    )

    with pytest.raises(ValueError):
        validate_asset_qc_report(report)


def test_semantic_pending_text_requires_one_affected_segment() -> None:
    report = make_v2_report()
    report["semantic_calibration"] = make_semantic_block(
        state="in_progress",
        pending_edit=make_text_edit(affected_segment_ids=["segment-0", "segment-1"]),
    )

    with pytest.raises(ValueError):
        validate_asset_qc_report(report)


def test_completed_manual_review_requires_every_selected_verdict() -> None:
    report = make_v2_report()
    report["manual_review"] = make_manual_block(
        state="completed",
        candidate_issue_ids=["warn-1", "warn-2"],
        selected_issue_ids=["warn-1", "warn-2"],
        issue_reviews={"warn-1": make_review("pass")},
        completed_at="2026-07-15T00:00:00Z",
    )

    with pytest.raises(ValueError):
        validate_asset_qc_report(report)


def test_manual_selected_ids_must_be_candidate_subset() -> None:
    report = make_v2_report()
    report["manual_review"] = make_manual_block(
        state="in_progress",
        candidate_issue_ids=["warn-1"],
        selected_issue_ids=["warn-1", "warn-2"],
    )

    with pytest.raises(ValueError):
        validate_asset_qc_report(report)


def test_valid_human_blocks_validate() -> None:
    report = make_v2_report()
    report["semantic_calibration"] = make_semantic_block(
        state="in_progress",
        pending_edit=make_boundary_edit(),
    )
    report["manual_review"] = make_manual_block(
        state="completed",
        candidate_issue_ids=["warn-1", "warn-2"],
        selected_issue_ids=["warn-1", "warn-2"],
        issue_reviews={
            "warn-1": make_review("pass"),
            "warn-2": make_review("fail"),
        },
        completed_at="2026-07-15T00:00:00Z",
    )

    validate_asset_qc_report(report)


def test_v1_migration_adds_human_defaults_and_preserves_unknown_extensions() -> None:
    from qc_common.report_migration import migrate_v1_to_v2

    old = make_v1_video_report()
    old["future_extension"] = {"keep": [1, 2, 3]}
    old["manual_review"]["legacy_extension"] = {"preserve": True}
    frozen = copy.deepcopy(old)

    migrated = migrate_v1_to_v2(
        old,
        config_reference={
            "schema_version": "qc_acceptance_config_schema.v2",
            "config_version": "qc_acceptance_v2.0.0",
            "config_name": "acceptance_gate",
            "config_path": "configs/qc_acceptance.yaml",
            "config_hash": "sha256:" + "b" * 64,
        },
    )

    assert old == frozen
    assert migrated["future_extension"] == old["future_extension"]
    assert migrated["manual_review"]["legacy_extension"] == {"preserve": True}
    assert migrated["manual_review"]["selected_issue_ids"] == []
    assert migrated["manual_review"]["selected_issue_id"] is None
    assert migrated["manual_review"]["issue_reviews"] == {}
    assert migrated["manual_review"]["completed_at"] is None
    assert migrated["semantic_calibration"]["state"] == "not_started"
    validate_asset_qc_report(migrated)


def test_v2_migration_returns_defensive_copy_with_human_blocks() -> None:
    from qc_common.report_migration import migrate_v1_to_v2

    report = make_v2_report()
    report["semantic_calibration"] = make_semantic_block()
    migrated = migrate_v1_to_v2(report, config_reference={})

    assert migrated == report
    assert migrated is not report
    assert migrated["semantic_calibration"] is not report["semantic_calibration"]


def _persist_report(tmp_path: Path, report: dict | None = None) -> tuple[Path, dict]:
    path = tmp_path / "quality_archive" / "asset.json"
    value = make_v2_report() if report is None else report
    write_asset_qc_report(path, value, expected_revision=0, profile="acceptance")
    persisted = load_asset_qc_report(path)
    assert persisted is not None
    return path, persisted


def test_initialize_human_blocks_and_update_once_preserve_machine_payload(
    tmp_path: Path,
) -> None:
    report = make_v2_report()
    report["future_extension"] = {"keep": True}
    report["issues"] = [
        {
            "issue_id": "warn-1",
            "severity": "warn",
            "module": "video_quality",
            "needs_manual_review": True,
        }
    ]
    path, persisted = _persist_report(tmp_path, report)
    machine_before = {
        field: copy.deepcopy(persisted[field])
        for field in ("qc_config", "source_files", "issues", "runtime_errors")
    }

    def mutate(candidate: dict) -> None:
        initialize_semantic_calibration(
            candidate,
            "/label/subtask_label",
            "sha256:" + "c" * 64,
        )
        initialize_manual_review(candidate, ["warn-1"])
        candidate["semantic_calibration"]["state"] = "completed"
        candidate["manual_review"].update(
            {
                "selected_issue_ids": ["warn-1"],
                "issue_reviews": {"warn-1": make_review("pass")},
                "state": "completed",
                "completed_at": "2026-07-15T00:00:00Z",
            }
        )
        candidate["pipeline_state"].update(
            {
                "status": "completed",
                "next_module": None,
                "stop_reason": None,
            }
        )
        candidate["overall_decision"] = reduce_overall_decision(candidate)

    updated = update_human_state(path, persisted["report_revision"], mutate)

    assert updated["report_revision"] == persisted["report_revision"] + 1
    assert updated["overall_decision"] == "pass"
    assert updated["future_extension"] == {"keep": True}
    for field, value in machine_before.items():
        assert updated[field] == value
    assert json.loads(path.read_text(encoding="utf-8")) == updated
    validate_asset_qc_report(updated)


def test_human_update_rejects_stale_revision_without_writing(tmp_path: Path) -> None:
    path, persisted = _persist_report(tmp_path)
    before = path.read_bytes()

    with pytest.raises(StaleReportRevisionError):
        update_human_state(
            path,
            persisted["report_revision"] - 1,
            lambda candidate: candidate["overall_decision"].__class__,
        )

    assert path.read_bytes() == before


@pytest.mark.parametrize("field", ["qc_config", "source_files", "issues", "runtime_errors"])
def test_human_update_rejects_machine_owned_top_level_changes(
    tmp_path: Path,
    field: str,
) -> None:
    path, persisted = _persist_report(tmp_path)
    before = path.read_bytes()

    def mutate(candidate: dict) -> None:
        if isinstance(candidate[field], list):
            candidate[field].append({"changed": True})
        elif isinstance(candidate[field], dict):
            candidate[field]["changed"] = True

    with pytest.raises(ValueError, match="machine-owned|immutable|allowed"):
        update_human_state(path, persisted["report_revision"], mutate)

    assert path.read_bytes() == before


def test_human_update_rejects_unknown_top_level_extension_change(tmp_path: Path) -> None:
    report = make_v2_report()
    report["future_extension"] = {"keep": True}
    path, persisted = _persist_report(tmp_path, report)
    before = path.read_bytes()

    with pytest.raises(ValueError, match="allowed|immutable"):
        update_human_state(
            path,
            persisted["report_revision"],
            lambda candidate: candidate["future_extension"].update({"bad": True}),
        )

    assert path.read_bytes() == before


def test_human_update_rejects_invalid_selected_ids_before_atomic_write(
    tmp_path: Path,
) -> None:
    path, persisted = _persist_report(tmp_path)
    before = path.read_bytes()

    def mutate(candidate: dict) -> None:
        initialize_manual_review(candidate, ["warn-1"])
        candidate["manual_review"]["selected_issue_ids"] = ["not-a-candidate"]

    with pytest.raises(ValueError):
        update_human_state(path, persisted["report_revision"], mutate)

    assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("machine_fail", "review_verdict", "expected"),
    [(False, "pass", "pass"), (False, "fail", "fail"), (True, "pass", "fail")],
)
def test_reduce_overall_decision_preserves_machine_fail_and_human_warn_verdict(
    machine_fail: bool,
    review_verdict: str,
    expected: str,
) -> None:
    report = make_v2_report(status="completed", overall_decision=None)
    report["semantic_calibration"] = make_semantic_block(state="completed")
    report["manual_review"] = make_manual_block(
        state="completed",
        candidate_issue_ids=["warn-1"],
        selected_issue_ids=["warn-1"],
        issue_reviews={"warn-1": make_review(review_verdict)},
        completed_at="2026-07-15T00:00:00Z",
    )
    if machine_fail:
        report["issues"] = [
            {
                "issue_id": "fail-1",
                "severity": "fail",
                "module": "video_quality",
            }
        ]

    assert reduce_overall_decision(report) == expected


def test_reduce_overall_decision_is_null_until_human_review_completes() -> None:
    report = make_v2_report(status="completed", overall_decision=None)
    report["semantic_calibration"] = make_semantic_block(state="completed")
    report["manual_review"] = make_manual_block(
        state="in_progress",
        candidate_issue_ids=["warn-1"],
        selected_issue_ids=["warn-1"],
    )

    assert reduce_overall_decision(report) is None


def test_reduce_overall_decision_keeps_acceptance_hard_fail_after_human_skip() -> None:
    report = make_v2_report(status="stopped", overall_decision="fail")
    report["semantic_calibration"] = make_semantic_block(state="skipped_due_to_fail")
    report["manual_review"] = make_manual_block(state="skipped_due_to_fail")
    report["issues"] = [
        {"issue_id": "fail-1", "severity": "fail", "module": "video_quality"}
    ]

    assert reduce_overall_decision(report) == "fail"
