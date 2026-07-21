from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from qc_common.schema import ReportValidationError, validate_asset_qc_report
from qc_common.manual_review import select_pending_manual_review_candidates
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


@pytest.mark.parametrize(
    "malformed_snapshot",
    [
        7,
        {"start_frame": 0, "end_frame_exclusive": 10, "text_cn": "a", "text_en": "b"},
        {"internal_id": "segment-0", "end_frame_exclusive": 10, "text_cn": "a", "text_en": "b"},
        {"internal_id": "segment-0", "start_frame": 0, "text_cn": "a", "text_en": "b"},
        {"internal_id": "segment-0", "start_frame": 0, "end_frame_exclusive": 10},
    ],
)
def test_pending_snapshot_requires_identity_range_and_text(
    malformed_snapshot: object,
) -> None:
    report = make_v2_report()
    pending = make_text_edit()
    pending["before"] = malformed_snapshot
    report["semantic_calibration"] = make_semantic_block(
        state="in_progress",
        pending_edit=pending,
    )

    with pytest.raises(ValueError, match="pending_edit|snapshot|before"):
        validate_asset_qc_report(report)


def test_json_schema_rejects_primitive_pending_snapshots() -> None:
    report = make_v2_report()
    pending = make_boundary_edit()
    pending["after"] = ["segment-0", "segment-1"]
    report["semantic_calibration"] = make_semantic_block(
        state="in_progress",
        pending_edit=pending,
    )
    schema = json.loads(
        (Path(__file__).parents[1] / "schemas" / "asset_qc_report.v2.schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert list(Draft202012Validator(schema).iter_errors(report))


@pytest.mark.parametrize("pending", [make_boundary_edit(), make_text_edit()])
def test_valid_boundary_and_text_pending_snapshots_validate(pending: dict) -> None:
    report = make_v2_report()
    report["semantic_calibration"] = make_semantic_block(
        state="in_progress",
        pending_edit=pending,
    )

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
    report["manual_review"]["completion_mode"] = "all_reviewed"

    with pytest.raises(ValueError):
        validate_asset_qc_report(report)


def test_completed_all_reviewed_rejects_a_fail_verdict() -> None:
    report = make_v2_report()
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
    report["manual_review"]["completion_mode"] = "all_reviewed"

    with pytest.raises(ReportValidationError, match="completion_mode"):
        validate_asset_qc_report(report)


@pytest.mark.parametrize(
    ("field", "value", "error_path"),
    [
        ("selected_issue_ids", ["warn-2"], "selected_issue_ids"),
        ("issue_reviews", {"warn-1": {"verdict": "unknown"}}, "verdict"),
        ("completed_at", None, "completed_at"),
    ],
)
def test_legacy_completed_review_still_enforces_existing_structure(
    field: str,
    value: object,
    error_path: str,
) -> None:
    report = make_v2_report()
    report["manual_review"] = make_manual_block(
        state="completed",
        candidate_issue_ids=["warn-1"],
        selected_issue_ids=["warn-1"],
        issue_reviews={"warn-1": make_review("pass")},
        completed_at="2026-07-15T00:00:00Z",
    )
    report["manual_review"][field] = value

    with pytest.raises(ReportValidationError, match=error_path):
        validate_asset_qc_report(report)


@pytest.mark.parametrize("missing_field", ["completion_mode", "failure_reason"])
def test_write_rejects_legacy_completed_v2_without_canonical_fields(
    tmp_path: Path,
    missing_field: str,
) -> None:
    report = make_v2_report()
    report["manual_review"] = make_manual_block(
        state="completed",
        candidate_issue_ids=["warn-1"],
        selected_issue_ids=["warn-1"],
        issue_reviews={"warn-1": make_review("pass")},
        completed_at="2026-07-15T00:00:00Z",
    )
    report["manual_review"].update(
        {"completion_mode": "all_reviewed", "failure_reason": None}
    )
    del report["manual_review"][missing_field]

    with pytest.raises(ReportValidationError, match=missing_field):
        write_asset_qc_report(tmp_path / "quality_archive" / "legacy.json", report, 0)


def test_completed_early_fail_records_canonical_failure_reason() -> None:
    report = make_v2_report()
    report["manual_review"] = make_manual_block(
        state="completed",
        candidate_issue_ids=["warn-1", "warn-2", "warn-3"],
        selected_issue_ids=["warn-1", "warn-2", "warn-3"],
        issue_reviews={"warn-1": make_review("fail")},
        completed_at="2026-07-15T00:00:00Z",
    )
    report["manual_review"].update(
        {
            "completion_mode": "early_fail",
            "failure_reason": {
                "mode": "manual",
                "reason_codes": ["occlusion", "other"],
                "other_text": "手被工具完全遮挡",
            },
        }
    )

    validate_asset_qc_report(report)

    assert report["manual_review"]["completion_mode"] == "early_fail"
    assert report["manual_review"]["failure_reason"] == {
        "mode": "manual",
        "reason_codes": ["occlusion", "other"],
        "other_text": "手被工具完全遮挡",
    }


@pytest.mark.parametrize(
    ("completion_mode", "failure_reason", "issue_reviews", "error_path"),
    [
        (
            "early_fail",
            {"mode": "manual", "reason_codes": ["other"], "other_text": "  "},
            {"warn-1": make_review("fail")},
            "failure_reason.other_text",
        ),
        (
            "early_fail",
            None,
            {"warn-1": make_review("pass")},
            "completion_mode",
        ),
        (
            "all_reviewed",
            None,
            {"warn-1": make_review("pass")},
            "issue_reviews",
        ),
    ],
)
def test_completed_manual_review_rejects_invalid_completion_contract(
    completion_mode: str,
    failure_reason: dict | None,
    issue_reviews: dict,
    error_path: str,
) -> None:
    report = make_v2_report()
    report["manual_review"] = make_manual_block(
        state="completed",
        candidate_issue_ids=["warn-1", "warn-2"],
        selected_issue_ids=["warn-1", "warn-2"],
        issue_reviews=issue_reviews,
        completed_at="2026-07-15T00:00:00Z",
    )
    report["manual_review"].update(
        {
            "completion_mode": completion_mode,
            "failure_reason": failure_reason,
        }
    )

    with pytest.raises(ReportValidationError, match=error_path):
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


def test_manual_selection_policy_must_be_supported() -> None:
    report = make_v2_report()
    report["manual_review"] = make_manual_block(
        state="queued",
        candidate_issue_ids=["warn-1"],
        selected_issue_ids=["warn-1"],
    )
    report["manual_review"]["selection_policy"] = "unknown_selector"

    with pytest.raises(ValueError, match="selection_policy"):
        validate_asset_qc_report(report)


def test_pending_selection_preserves_existing_explicit_snapshot() -> None:
    report = make_v2_report(status="running")
    report["pipeline_state"]["next_module"] = "manual_review"
    report["manual_review"] = make_manual_block(
        state="queued",
        candidate_issue_ids=["warn-1", "warn-2"],
        selected_issue_ids=["warn-2"],
    )

    changed = select_pending_manual_review_candidates(report)

    assert changed is False
    assert report["manual_review"]["candidate_issue_ids"] == ["warn-1", "warn-2"]
    assert report["manual_review"]["selected_issue_ids"] == ["warn-2"]
    assert "selection_policy" not in report["manual_review"]


def test_pending_all_candidates_selection_keeps_full_candidate_pool() -> None:
    report = make_v2_report(status="running")
    report["pipeline_state"]["next_module"] = "manual_review"
    report["manual_review"] = make_manual_block(
        state="not_evaluated",
        candidate_issue_ids=["warn-2", "warn-1"],
        selected_issue_ids=[],
    )

    changed = select_pending_manual_review_candidates(report)

    assert changed is True
    assert report["manual_review"]["candidate_issue_ids"] == ["warn-2", "warn-1"]
    assert report["manual_review"]["selected_issue_ids"] == ["warn-2", "warn-1"]
    assert report["manual_review"]["selection_policy"] == "all_candidates"
    assert report["manual_review"]["required"] is True
    assert report["manual_review"]["state"] == "queued"


def test_selection_does_not_mutate_completed_review() -> None:
    report = make_v2_report(status="completed")
    report["pipeline_state"]["next_module"] = None
    report["manual_review"] = make_manual_block(
        state="not_required",
        candidate_issue_ids=["warn-1"],
        selected_issue_ids=[],
    )

    changed = select_pending_manual_review_candidates(report)

    assert changed is False
    assert report["manual_review"]["selected_issue_ids"] == []


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


def test_v1_migration_merges_semantic_defaults_into_opaque_block() -> None:
    from qc_common.report_migration import migrate_v1_to_v2

    old = make_v1_video_report()
    old["semantic_calibration"] = {
        "legacy_extension": {"keep": ["before", "after"]},
        "revision_id": "semantic-r4",
    }

    migrated = migrate_v1_to_v2(old, config_reference=old["qc_config"])
    semantic = migrated["semantic_calibration"]

    assert semantic["legacy_extension"] == {"keep": ["before", "after"]}
    assert semantic["revision_id"] == "semantic-r4"
    assert semantic["state"] == "not_started"
    assert semantic["source_dataset_path"] is None
    assert semantic["base_hdf5_sha256"] is None
    assert semantic["final_hdf5_sha256"] is None
    assert semantic["timeline_edit_count"] == 0
    assert semantic["subtask_text_edit_count"] == 0
    assert semantic["pending_edit"] is None
    assert semantic["audit"] == []
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
                "completion_mode": "all_reviewed",
                "failure_reason": None,
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
