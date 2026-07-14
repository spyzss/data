from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from qc_reporting.projection import (
    iter_asset_reports,
    project_quality_archive_review_rows,
    project_warn_review_rows,
)
from qc_common.report_mutation import initialize_v2_report
from qc_pipeline.context import AssetContext
from tests.qc_report_fixtures import make_v2_report
from tests.qc_report_fixtures import loaded_test_config


def warn_issue(
    issue_id: str,
    start: int = 10,
    end: int = 20,
    *,
    module: str = "keypoint_temporal",
    severity: str = "warn",
    needs_manual_review: bool = True,
) -> dict[str, Any]:
    return {
        "issue_id": issue_id,
        "code": "temporal_jump",
        "severity": severity,
        "module": module,
        "issue_type": "temporal_jump",
        "metric": "joint_displacement_m_max",
        "observed_value": 0.08,
        "operator": ">",
        "boundary_value": 0.05,
        "rule_id": f"{module}.temporal_jump",
        "needs_manual_review": needs_manual_review,
        "context": {
            "coordinate_system": "source_inclusive",
            "start_frame": start,
            "end_frame": end,
            "hand_side": "left",
            "reason": "temporal jump needs human review",
        },
        "evidence_ids": [f"{issue_id}:evidence"],
    }


def fail_issue(issue_id: str) -> dict[str, Any]:
    issue = warn_issue(issue_id, severity="fail", needs_manual_review=False)
    issue["code"] = "hard_failure"
    return issue


def write_report(
    root: Path,
    *,
    asset: str,
    issues: list[dict[str, Any]],
    candidates: list[str],
    state: str,
    supplier: str = "supplier-a",
) -> Path:
    report = make_v2_report(status="completed", overall_decision="pass")
    report["asset_id"] = asset
    report["supplier_id"] = supplier
    report["pipeline_state"] = {
        "status": "completed",
        "last_completed_module": "keypoint_temporal",
        "next_module": None,
        "stop_reason": None,
    }
    report["issues"] = copy.deepcopy(issues)
    report["manual_review"] = {
        "required": bool(candidates),
        "state": state,
        "candidate_issue_ids": candidates,
        "failures_for_batch_stats_issue_ids": [
            item["issue_id"] for item in issues if item["severity"] == "fail"
        ],
    }
    report["keypoint_temporal"] = {
        "flow": {
            "result_gate": {"verdict": "warn", "has_fail": False, "has_warn": True}
        },
        "evaluation": {"decision": "warn"},
        "metrics": {"joint_displacement_m_max": 0.08, "window_count": 1},
        "evidence": [
            {
                "evidence_id": f"{issues[0]['issue_id']}:evidence"
                if issues
                else "unused",
                "kind": "clip",
                "path": "evidence/temporal_jump.mp4",
                "coordinate_system": "source_inclusive",
                "start_frame": 10,
                "end_frame": 20,
                "hand_side": "left",
                "overlay_path": "evidence/temporal_jump_overlay.mp4",
            }
        ]
        if issues
        else [],
    }
    path = root / "quality_archive" / f"{asset}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    return path


def test_review_queue_comes_only_from_candidate_issue_ids(tmp_path: Path) -> None:
    write_report(
        tmp_path,
        asset="warn",
        issues=[warn_issue("w1")],
        candidates=["w1"],
        state="queued",
    )
    write_report(
        tmp_path,
        asset="pass",
        issues=[],
        candidates=[],
        state="not_required",
    )
    write_report(
        tmp_path,
        asset="fail",
        issues=[fail_issue("f1")],
        candidates=[],
        state="skipped_due_to_fail",
    )

    rows = list(project_quality_archive_review_rows(tmp_path / "quality_archive"))

    assert [(row["asset_id"], row["issue_id"]) for row in rows] == [("warn", "w1")]
    assert rows[0]["window_start_frame"] == 10
    assert rows[0]["window_end_frame"] == 20
    assert rows[0]["review_id"] == "w1"


def test_projection_keeps_machine_context_metrics_and_relative_evidence() -> None:
    report = make_v2_report(status="completed", overall_decision="pass")
    report["asset_id"] = "asset-1"
    report["supplier_id"] = "supplier-a"
    issue = warn_issue("w1", 11, 21)
    report["issues"] = [issue]
    report["manual_review"] = {
        "required": True,
        "state": "queued",
        "candidate_issue_ids": ["w1"],
        "failures_for_batch_stats_issue_ids": [],
    }
    report["keypoint_temporal"] = {
        "metrics": {"joint_displacement_m_max": 0.08, "window_count": 1},
        "evidence": [
            {
                "evidence_id": "w1:evidence",
                "kind": "clip",
                "path": "evidence/temporal_jump.mp4",
                "coordinate_system": "source_inclusive",
                "start_frame": 11,
                "end_frame": 21,
                "hand_side": "left",
                "overlay_path": "evidence/temporal_jump_overlay.mp4",
            }
        ],
    }

    row = project_warn_review_rows(report)[0]

    assert row["review_id"] == "w1"
    assert row["supplier_id"] == "supplier-a"
    assert row["module"] == "keypoint_temporal"
    assert row["rule_id"] == "keypoint_temporal.temporal_jump"
    assert row["reason"] == "temporal jump needs human review"
    assert row["window_start_frame"] == 11
    assert row["window_end_frame"] == 21
    assert row["hand_side"] == "left"
    assert row["auto_verdict"] == "warn"
    assert json.loads(row["key_metrics_json"])["joint_displacement_m_max"] == 0.08
    assert row["evidence_path"] == "evidence/temporal_jump.mp4"
    assert row["overlay_path"] == "evidence/temporal_jump_overlay.mp4"


@pytest.mark.parametrize(
    ("candidate_ids", "message"),
    [
        (["missing"], "manual_review.candidate_issue_ids"),
        (["f1"], "severity"),
        (["w1", "w1"], "candidate_issue_ids"),
    ],
)
def test_invalid_candidate_reference_fails_asset_projection(
    candidate_ids: list[str],
    message: str,
) -> None:
    report = make_v2_report(status="completed", overall_decision="pass")
    report["asset_id"] = "asset-1"
    report["issues"] = [warn_issue("w1"), fail_issue("f1")]
    report["manual_review"] = {
        "required": True,
        "state": "queued",
        "candidate_issue_ids": candidate_ids,
        "failures_for_batch_stats_issue_ids": ["f1"],
    }

    with pytest.raises(ValueError, match=message):
        project_warn_review_rows(report)


def test_iter_asset_reports_validates_each_report_and_includes_json_path(
    tmp_path: Path,
) -> None:
    write_report(
        tmp_path,
        asset="valid",
        issues=[warn_issue("w1")],
        candidates=["w1"],
        state="queued",
    )
    bad = tmp_path / "quality_archive" / "bad.json"
    bad.write_text(json.dumps({"schema_version": "future"}), encoding="utf-8")

    with pytest.raises(ValueError, match=r"bad\.json"):
        list(iter_asset_reports(tmp_path / "quality_archive"))


def test_quality_archive_cli_writes_warn_rows_without_pass_sampling(tmp_path: Path) -> None:
    write_report(
        tmp_path,
        asset="warn",
        issues=[warn_issue("w1")],
        candidates=["w1"],
        state="queued",
    )
    write_report(
        tmp_path,
        asset="pass",
        issues=[],
        candidates=[],
        state="not_required",
    )
    output_dir = tmp_path / "review"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.build_manual_review_queue",
            "--quality-archive",
            str(tmp_path / "quality_archive"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    queue = pd.read_csv(output_dir / "review_queue.csv")
    assert queue["asset_id"].tolist() == ["warn"]
    assert "pass_sample" not in queue["auto_verdict"].tolist()
    assert pd.read_csv(output_dir / "manual_labels_template.csv")["review_id"].tolist() == ["w1"]


def test_quality_archive_cli_resolves_overlay_paths_from_batch_root(
    tmp_path: Path,
) -> None:
    batch_root = tmp_path / "batch"
    archive = batch_root / "quality_archive"
    overlay = batch_root / "evidence" / "temporal_jump_overlay.mp4"
    overlay.parent.mkdir(parents=True)
    overlay.write_bytes(b"batch-root-overlay")
    write_report(
        batch_root,
        asset="warn",
        issues=[warn_issue("w1")],
        candidates=["w1"],
        state="queued",
    )
    output_dir = batch_root / "review"
    repo_root = Path(__file__).resolve().parents[1]
    process_cwd = tmp_path / "process-cwd"
    process_cwd.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "build_manual_review_queue.py"),
            "--quality-archive",
            str(archive),
            "--output-dir",
            str(output_dir),
        ],
        cwd=process_cwd,
        check=True,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    queue = pd.read_csv(output_dir / "review_queue.csv")
    assert queue.loc[0, "overlay_path"] == "evidence/temporal_jump_overlay.mp4"
    assert queue.loc[0, "display_overlay_path"] == "assets/overlays/temporal_jump_overlay.mp4"
    copied = output_dir / "assets" / "overlays" / "temporal_jump_overlay.mp4"
    assert copied.read_bytes() == b"batch-root-overlay"
    assert "assets/overlays/temporal_jump_overlay.mp4" in (
        output_dir / "review_index.html"
    ).read_text(encoding="utf-8")


def test_initialize_report_persists_supplier_for_canonical_projection(
    tmp_path: Path,
) -> None:
    context = AssetContext(
        asset_id="asset-meta",
        batch_root=tmp_path,
        report_path=tmp_path / "quality_archive" / "asset-meta.json",
        source_files={},
        metadata={"supplier_id": "supplier-from-metadata"},
    )
    report = initialize_v2_report(
        context,
        loaded_test_config(),
        "acceptance",
        "2026-07-15T00:00:00Z",
    )
    report.update(
        {
            "report_revision": 1,
            "pipeline_state": {
                "status": "completed",
                "last_completed_module": "keypoint_temporal",
                "next_module": None,
                "stop_reason": None,
            },
            "overall_decision": "pass",
            "issues": [warn_issue("w1")],
            "manual_review": {
                "required": True,
                "state": "queued",
                "candidate_issue_ids": ["w1"],
                "failures_for_batch_stats_issue_ids": [],
            },
            "keypoint_temporal": {
                "evidence": [
                    {
                        "evidence_id": "w1:evidence",
                        "path": "evidence/temporal_jump.mp4",
                    }
                ]
            },
        }
    )
    context.report_path.parent.mkdir(parents=True, exist_ok=True)
    context.report_path.write_text(json.dumps(report), encoding="utf-8")

    loaded = json.loads(context.report_path.read_text(encoding="utf-8"))
    rows = project_warn_review_rows(loaded)

    assert loaded["supplier_id"] == "supplier-from-metadata"
    assert rows[0]["supplier_id"] == "supplier-from-metadata"


def test_projection_uses_metadata_supplier_fallback_for_legacy_report() -> None:
    report = make_v2_report(status="completed", overall_decision="pass")
    report["asset_id"] = "asset-legacy"
    report["metadata"] = {"supplier": "supplier-from-report-metadata"}
    issue = warn_issue("w1")
    report["issues"] = [issue]
    report["manual_review"] = {
        "required": True,
        "state": "queued",
        "candidate_issue_ids": ["w1"],
        "failures_for_batch_stats_issue_ids": [],
    }

    rows = project_warn_review_rows(report)

    assert rows[0]["supplier_id"] == "supplier-from-report-metadata"
