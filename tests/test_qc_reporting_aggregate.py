from __future__ import annotations

from pathlib import Path
from typing import Any

from qc_reporting.aggregate import aggregate_projection
from qc_reporting.projection import project_quality_archive
from tests.test_qc_reporting_projection import _issue, _write_report


def test_aggregation_counts_assets_and_issues_separately(tmp_path: Path) -> None:
    archive = tmp_path / "quality_archive"
    _write_report(
        archive,
        "a",
        "acceptance",
        "fail",
        issues=[_issue("f1", severity="fail"), _issue("f2", severity="fail")],
    )
    _write_report(archive, "b", "supplier_evaluation", "pass", issues=[])

    stats = aggregate_projection(project_quality_archive(archive))

    assert stats["overall"]["asset_count"] == 2
    assert stats["overall"]["automatic_hard_fail_asset_count"] == 1
    assert stats["overall"]["automatic_hard_fail_issue_count"] == 2
    assert stats["overall"]["final_fail_asset_count"] == 1
    assert stats["by_profile"]["acceptance"]["asset_count"] == 1
    assert stats["by_profile"]["supplier_evaluation"]["module_coverage"]["sam3_containment"] == 1.0


def test_aggregation_deduplicates_asset_rows_and_issues() -> None:
    projection_type = type("Projection", (), {})
    projection = projection_type()
    projection.asset_rows = (
        {"asset_id": "a", "profile": "acceptance", "status": "completed", "decision": "pass"},
        {"asset_id": "a", "profile": "acceptance", "status": "completed", "decision": "pass"},
    )
    projection.issue_rows = (
        {"asset_id": "a", "profile": "acceptance", "issue_id": "w1", "machine_severity": "warn"},
        {"asset_id": "a", "profile": "acceptance", "issue_id": "w1", "machine_severity": "warn"},
    )
    projection.execution_rows = ()

    stats = aggregate_projection(projection)

    assert stats["overall"]["asset_count"] == 1
    assert stats["overall"]["machine_warn_issue_count"] == 1


def test_profile_stats_do_not_mix_issue_or_execution_rows(tmp_path: Path) -> None:
    archive = tmp_path / "quality_archive"
    _write_report(archive, "a", "acceptance", "pass", issues=[_issue("w1")])
    _write_report(archive, "s", "supplier_evaluation", "pass", issues=[])

    stats = aggregate_projection(project_quality_archive(archive))

    assert stats["by_profile"]["acceptance"]["machine_warn_issue_count"] == 1
    assert stats["by_profile"]["supplier_evaluation"]["machine_warn_issue_count"] == 0
    assert stats["by_profile"]["supplier_evaluation"]["module_coverage"]["sam3_containment"] == 1.0


def test_aggregation_reports_incomplete_assets_and_pass_rate() -> None:
    projection_type = type("Projection", (), {})
    projection = projection_type()
    projection.asset_rows = (
        {"asset_id": "done", "profile": "acceptance", "status": "completed", "decision": "pass"},
        {"asset_id": "pending", "profile": "acceptance", "status": "running", "decision": None},
    )
    projection.issue_rows = ()
    projection.execution_rows = ()

    stats = aggregate_projection(projection)["overall"]

    assert stats["completed_asset_count"] == 1
    assert stats["incomplete_asset_count"] == 1
    assert stats["final_pass_asset_count"] == 1
    assert stats["final_fail_asset_count"] == 0
    assert stats["pass_rate"] == 1.0
