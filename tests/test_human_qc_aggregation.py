from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from qc_reporting.aggregate import aggregate_projection
from qc_common.projection import project_manual_review_counts
from qc_reporting.projection import (
    BatchProjection,
    project_human_review_rows,
    project_quality_archive,
)
from tests.qc_report_fixtures import (
    make_manual_block,
    make_semantic_block,
    make_v2_report,
)


def _issue(issue_id: str, severity: str = "warn") -> dict[str, Any]:
    return {
        "issue_id": issue_id,
        "code": f"test.{issue_id}",
        "severity": severity,
        "module": "video_quality",
        "issue_type": "test_issue",
        "metric": "video.metric",
        "observed_value": 1,
        "operator": ">",
        "boundary_value": 0,
        "rule_id": f"video_quality.{issue_id}",
        "needs_manual_review": severity == "warn",
        "context": {},
        "evidence_ids": [],
    }


def _review(verdict: str) -> dict[str, Any]:
    return {
        "verdict": verdict,
        "effective_verdict": verdict,
        "machine_verdict": "warn",
        "reason": None,
        "reviewer": "alice",
        "reviewed_at": "2026-07-15T00:00:00Z",
    }


def _report(
    asset_id: str,
    *,
    revision: int,
    profile: str = "supplier_evaluation",
    status: str = "completed",
    decision: str | None = "fail",
    warn_verdicts: tuple[str, ...] = ("pass", "fail"),
    fail_issues: int = 0,
    timeline_edits: int = 1,
    text_edits: int = 0,
) -> dict[str, Any]:
    report = make_v2_report(status=status, overall_decision=decision)
    report["asset_id"] = asset_id
    report["report_revision"] = revision
    report["execution"]["profile"] = profile
    report["pipeline_state"] = {
        "status": status,
        "last_completed_module": "manual_review" if status == "completed" else "video_quality",
        "next_module": None if status == "completed" else "semantic_consistency",
        "stop_reason": None,
    }
    warn_ids = [f"warn-{index}" for index in range(len(warn_verdicts))]
    fail_ids = [f"fail-{index}" for index in range(fail_issues)]
    report["issues"] = [*(_issue(item) for item in warn_ids), *(_issue(item, "fail") for item in fail_ids)]
    semantic = make_semantic_block(state="completed")
    semantic["final_hdf5_sha256"] = "sha256:" + "b" * 64
    semantic["timeline_edit_count"] = timeline_edits
    semantic["subtask_text_edit_count"] = text_edits
    report["semantic_calibration"] = semantic
    report["manual_review"] = make_manual_block(
        state="completed" if warn_ids else "not_required",
        candidate_issue_ids=warn_ids,
        selected_issue_ids=warn_ids,
        issue_reviews={
            issue_id: _review(verdict)
            for issue_id, verdict in zip(warn_ids, warn_verdicts, strict=True)
        },
    )
    report["manual_review"]["failures_for_batch_stats_issue_ids"] = fail_ids
    return report


def _write_report(archive: Path, report: dict[str, Any]) -> None:
    archive.mkdir(parents=True, exist_ok=True)
    path = archive / f"{report['asset_id']}.json"
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")


def test_human_projection_reads_only_canonical_human_blocks() -> None:
    report = _report("asset-a", revision=7, timeline_edits=1, text_edits=2)

    rows = tuple(project_human_review_rows(report))

    assert rows == (
        {
            "asset_id": "asset-a",
            "supplier_id": "unknown",
            "profile": "supplier_evaluation",
            "report_revision": 7,
            "semantic_state": "completed",
            "manual_review_state": "completed",
            "timeline_edit_count": 1,
            "subtask_text_edit_count": 2,
            "issue_reviews": {
                "warn-0": {"human_verdict": "pass", "effective_verdict": "pass"},
                "warn-1": {"human_verdict": "fail", "effective_verdict": "fail"},
            },
        },
    )


def test_manual_review_projection_counts_unreviewed_early_fail_warns() -> None:
    projected = project_manual_review_counts(
        {
            "selected_issue_ids": ["warn-1", "warn-2", "warn-3"],
            "issue_reviews": {"warn-1": _review("fail")},
        }
    )

    assert projected["human_reviewed_warn_count"] == 1
    assert projected["human_confirmed_fail_count"] == 1
    assert projected["unreviewed_selected_warn_count"] == 2


def test_quality_archive_projection_preserves_machine_warn_and_human_rows(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "quality_archive"
    _write_report(archive, _report("asset-a", revision=7))
    # Legacy exports are deliberately irrelevant to the formal projection.
    (tmp_path / "manual_review.csv").write_text("issue_id,verdict\nwarn-0,fail\n")
    (tmp_path / "progress.json").write_text('{"warn-0": "fail"}')

    projection = project_quality_archive(archive)

    assert isinstance(projection, BatchProjection)
    assert [row["machine_severity"] for row in projection.issue_rows] == ["warn", "warn"]
    assert projection.human_review_rows[0]["timeline_edit_count"] == 1
    assert projection.human_review_rows[0]["issue_reviews"]["warn-0"]["human_verdict"] == "pass"


def test_aggregate_counts_two_warn_outcomes_and_one_boundary_transaction() -> None:
    projection = SimpleNamespace(
        asset_rows=(
            {
                "asset_id": "asset-a",
                "profile": "supplier_evaluation",
                "status": "completed",
                "decision": "fail",
                "report_revision": 4,
            },
        ),
        issue_rows=(
            {"asset_id": "asset-a", "profile": "supplier_evaluation", "issue_id": "w-pass", "machine_severity": "warn", "report_revision": 4},
            {"asset_id": "asset-a", "profile": "supplier_evaluation", "issue_id": "w-fail", "machine_severity": "warn", "report_revision": 4},
        ),
        execution_rows=(),
        human_review_rows=(
            {
                "asset_id": "asset-a",
                "profile": "supplier_evaluation",
                "report_revision": 4,
                "timeline_edit_count": 1,
                "subtask_text_edit_count": 0,
                "issue_reviews": {
                    "w-pass": {"human_verdict": "pass", "effective_verdict": "pass"},
                    "w-fail": {"human_verdict": "fail", "effective_verdict": "fail"},
                },
            },
        ),
    )

    stats = aggregate_projection(projection)["overall"]

    assert stats["auto_fail_assets"] == 0
    assert stats["auto_fail_issues"] == 0
    assert stats["machine_warn_issues"] == 2
    assert stats["human_checked_warn_issues"] == 2
    assert stats["human_resolved_warn_issues"] == 1
    assert stats["human_confirmed_fail_issues"] == 1
    assert stats["timeline_edit_count"] == 1
    assert stats["subtask_text_edit_count"] == 0
    assert stats["final_pass_assets"] == 0
    assert stats["final_fail_assets"] == 1


def test_aggregate_keeps_only_rows_from_latest_asset_revision() -> None:
    projection = SimpleNamespace(
        asset_rows=(
            {"asset_id": "a", "profile": "acceptance", "status": "completed", "decision": "fail", "report_revision": 1},
            {"asset_id": "a", "profile": "acceptance", "status": "completed", "decision": "pass", "report_revision": 2},
        ),
        issue_rows=(
            {"asset_id": "a", "profile": "acceptance", "issue_id": "old", "machine_severity": "fail", "report_revision": 1},
            {"asset_id": "a", "profile": "acceptance", "issue_id": "current", "machine_severity": "warn", "report_revision": 2},
        ),
        execution_rows=(),
        human_review_rows=(
            {"asset_id": "a", "profile": "acceptance", "report_revision": 1, "timeline_edit_count": 9, "subtask_text_edit_count": 8, "issue_reviews": {}},
            {"asset_id": "a", "profile": "acceptance", "report_revision": 2, "timeline_edit_count": 1, "subtask_text_edit_count": 2, "issue_reviews": {"current": {"human_verdict": "pass", "effective_verdict": "pass"}}},
        ),
    )

    stats = aggregate_projection(projection)["overall"]

    assert stats["auto_fail_issues"] == 0
    assert stats["machine_warn_issues"] == 1
    assert stats["timeline_edit_count"] == 1
    assert stats["subtask_text_edit_count"] == 2
    assert stats["final_pass_assets"] == 1
    assert stats["final_fail_assets"] == 0


def test_incomplete_assets_do_not_enter_final_pass_rate_denominator() -> None:
    completed = _report(
        "done",
        revision=1,
        profile="acceptance",
        decision="pass",
        warn_verdicts=(),
        timeline_edits=0,
    )
    pending = _report(
        "pending",
        revision=1,
        profile="acceptance",
        status="running",
        decision=None,
        warn_verdicts=(),
        timeline_edits=0,
    )
    assets = tuple(
        {
            "asset_id": report["asset_id"],
            "profile": "acceptance",
            "status": report["pipeline_state"]["status"],
            "decision": report["overall_decision"],
            "report_revision": report["report_revision"],
        }
        for report in (completed, pending)
    )
    projection = SimpleNamespace(
        asset_rows=assets,
        issue_rows=(),
        execution_rows=(),
        human_review_rows=tuple(
            row
            for report in (completed, pending)
            for row in project_human_review_rows(report)
        ),
    )

    stats = aggregate_projection(projection)["overall"]

    assert stats["final_pass_assets"] == 1
    assert stats["final_fail_assets"] == 0
    assert stats["final_asset_count"] == 1
    assert stats["pass_rate"] == 1.0


def test_profile_groups_keep_human_metrics_separate() -> None:
    acceptance = _report("a", revision=1, profile="acceptance", decision="pass", warn_verdicts=("pass",), timeline_edits=1)
    supplier = _report("s", revision=1, profile="supplier_evaluation", decision="fail", warn_verdicts=("fail",), timeline_edits=2)
    projection = SimpleNamespace(
        asset_rows=tuple(
            {
                "asset_id": report["asset_id"],
                "profile": report["execution"]["profile"],
                "status": "completed",
                "decision": report["overall_decision"],
                "report_revision": 1,
            }
            for report in (acceptance, supplier)
        ),
        issue_rows=tuple(
            {
                "asset_id": report["asset_id"],
                "profile": report["execution"]["profile"],
                "issue_id": issue["issue_id"],
                "machine_severity": issue["severity"],
                "report_revision": 1,
            }
            for report in (acceptance, supplier)
            for issue in report["issues"]
        ),
        execution_rows=(),
        human_review_rows=tuple(
            row
            for report in (acceptance, supplier)
            for row in project_human_review_rows(report)
        ),
    )

    stats = aggregate_projection(projection)["by_profile"]

    assert stats["acceptance"]["human_resolved_warn_issues"] == 1
    assert stats["acceptance"]["human_confirmed_fail_issues"] == 0
    assert stats["supplier_evaluation"]["human_resolved_warn_issues"] == 0
    assert stats["supplier_evaluation"]["human_confirmed_fail_issues"] == 1
    assert stats["supplier_evaluation"]["timeline_edit_count"] == 2
