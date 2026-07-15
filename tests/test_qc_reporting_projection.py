from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from qc_reporting.projection import BatchProjection, project_quality_archive
from tests.qc_report_fixtures import make_manual_block, make_v2_report


def _issue(issue_id: str, severity: str = "warn", module: str = "video_quality") -> dict[str, Any]:
    return {
        "issue_id": issue_id,
        "code": f"{module}.issue",
        "severity": severity,
        "module": module,
        "issue_type": "test_issue",
        "metric": "metric",
        "observed_value": 1,
        "operator": ">",
        "boundary_value": 0,
        "rule_id": f"{module}.rule",
        "needs_manual_review": severity == "warn",
        "context": {"start_frame": 10, "end_frame": 20},
        "evidence_ids": [],
    }


def _write_report(
    archive: Path,
    asset_id: str,
    profile: str,
    decision: str,
    *,
    issues: list[dict[str, Any]] | None = None,
) -> Path:
    report = make_v2_report(status="completed", overall_decision=decision)
    report["asset_id"] = asset_id
    report["supplier_id"] = f"{profile}-supplier"
    report["execution"]["profile"] = profile
    report["execution"]["module_states"] = {
        "hdf5_text_info": {"state": "completed"},
        "sam3_containment": {"state": "completed"},
    }
    report["pipeline_state"] = {
        "status": "completed",
        "last_completed_module": "sam3_containment",
        "next_module": None,
        "stop_reason": None,
    }
    report["issues"] = copy.deepcopy(issues or [])
    candidate_ids = [
        item["issue_id"] for item in report["issues"] if item["severity"] == "warn"
    ]
    report["manual_review"] = make_manual_block(
        state="queued" if candidate_ids else "not_required",
        candidate_issue_ids=candidate_ids,
    )
    report["manual_review"]["required"] = bool(candidate_ids)
    report["manual_review"]["failures_for_batch_stats_issue_ids"] = [
        item["issue_id"] for item in report["issues"] if item["severity"] == "fail"
    ]
    path = archive / f"{asset_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    return path


def test_project_quality_archive_returns_three_normalized_tables(tmp_path: Path) -> None:
    archive = tmp_path / "quality_archive"
    _write_report(archive, "asset-a", "acceptance", "pass", issues=[_issue("w1")])

    projection = project_quality_archive(archive)

    assert isinstance(projection, BatchProjection)
    assert isinstance(projection.asset_rows, tuple)
    assert len(projection.asset_rows) == 1
    assert projection.asset_rows[0]["asset_id"] == "asset-a"
    assert projection.asset_rows[0]["profile"] == "acceptance"
    assert projection.issue_rows[0]["issue_id"] == "w1"
    assert projection.issue_rows[0]["machine_severity"] == "warn"
    assert projection.issue_rows[0]["human_verdict"] is None
    assert projection.execution_rows[0]["module"] == "hdf5_text_info"
    assert projection.source_manifest[0]["path"].endswith("asset-a.json")


def test_project_quality_archive_validates_every_report_and_includes_path(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "quality_archive"
    _write_report(archive, "valid", "acceptance", "pass")
    (archive / "bad.json").write_text(json.dumps({"schema_version": "future"}), encoding="utf-8")

    with pytest.raises(ValueError, match=r"bad\.json"):
        project_quality_archive(archive)


def test_projection_keeps_all_machine_issues_independent_of_warn_candidates(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "quality_archive"
    report_path = _write_report(
        archive,
        "asset-a",
        "acceptance",
        "fail",
        issues=[_issue("w1"), _issue("f1", severity="fail")],
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["manual_review"]["candidate_issue_ids"] = []
    report_path.write_text(json.dumps(report), encoding="utf-8")

    projection = project_quality_archive(archive)

    assert {row["issue_id"] for row in projection.issue_rows} == {"w1", "f1"}
