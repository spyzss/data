from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

import pytest

from qc_reporting.projection import project_quality_archive
from tools.build_batch_qc_ledger import main as batch_main
from tools.build_batch_qc_ledger import parse_args as parse_batch_args
from tools.build_qc_json_projection import (
    main as projection_main,
    run_projection_cli,
    write_reconciliation_only,
)
from tools.build_xjgt_acceptance_report import parse_args as parse_xjgt_args
from tests.test_qc_reporting_projection import _issue, _write_report


def test_projection_cli_prefers_qc_json_over_conflicting_legacy_sidecar(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "quality_archive"
    _write_report(archive, "a", "acceptance", "pass", issues=[])
    sidecar = tmp_path / "candidate_windows.json"
    sidecar.write_text(
        json.dumps([{"asset_id": "a", "auto_verdict": "fail"}]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    assert (
        projection_main(
            [
                "--quality-archive",
                str(archive),
                "--output-dir",
                str(output_dir),
                "--formats",
                "csv",
                "xlsx",
                "markdown",
                "--legacy-reconciliation-candidate-windows",
                str(sidecar),
            ]
        )
        == 0
    )

    ledger = pd.read_csv(output_dir / "assets.csv")
    assert ledger.loc[0, "overall_decision"] == "pass"
    reconciliation = pd.read_csv(output_dir / "reconciliation.csv")
    assert reconciliation.loc[0, "difference_type"] == "legacy_conflicts_with_qc_json"
    workbook = load_workbook(output_dir / "qc_projection.xlsx", read_only=True)
    assert workbook.sheetnames == [
        "Summary",
        "Assets",
        "Issues",
        "Execution",
        "Data_Dictionary",
    ]


def test_projection_script_cli_help_runs_from_repository_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "build_qc_json_projection.py"),
            "--help",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Project canonical QC JSON reports" in result.stdout


def test_projection_cli_writes_parquet_tables_from_same_projection(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "quality_archive"
    _write_report(archive, "a", "acceptance", "pass", issues=[])
    output_dir = tmp_path / "out"

    assert (
        projection_main(
            [
                "--quality-archive",
                str(archive),
                "--output-dir",
                str(output_dir),
                "--formats",
                "parquet",
            ]
        )
        == 0
    )

    parquet = pd.read_parquet(output_dir / "assets.parquet")
    assert parquet.loc[0, "overall_decision"] == "pass"


def test_projection_cli_preserves_empty_issue_and_execution_table_headers(
    tmp_path: Path,
) -> None:
    all_pass_archive = tmp_path / "all_pass" / "quality_archive"
    nonempty_archive = tmp_path / "nonempty" / "quality_archive"
    _write_report(all_pass_archive, "pass-asset", "acceptance", "pass", issues=[])
    _write_report(
        nonempty_archive,
        "warn-asset",
        "acceptance",
        "pass",
        issues=[_issue("warn-1")],
    )

    all_pass_output = tmp_path / "all_pass" / "projection"
    nonempty_output = tmp_path / "nonempty" / "projection"
    for archive, output in (
        (all_pass_archive, all_pass_output),
        (nonempty_archive, nonempty_output),
    ):
        assert (
            projection_main(
                [
                    "--quality-archive",
                    str(archive),
                    "--output-dir",
                    str(output),
                    "--formats",
                    "csv",
                    "parquet",
                    "xlsx",
                ]
            )
            == 0
        )

    def table_headers(output: Path, table: str) -> tuple[str, ...]:
        csv_headers = tuple(pd.read_csv(output / f"{table}s.csv", nrows=0).columns)
        parquet_headers = tuple(pd.read_parquet(output / f"{table}s.parquet").columns)
        workbook = load_workbook(output / "qc_projection.xlsx", read_only=True)
        sheet_name = {"issue": "Issues", "execution": "Execution"}[table]
        worksheet = workbook[sheet_name]
        xlsx_headers = tuple(next(worksheet.iter_rows(values_only=True)))
        return csv_headers, parquet_headers, xlsx_headers

    all_pass_issues = table_headers(all_pass_output, "issue")
    nonempty_issues = table_headers(nonempty_output, "issue")
    all_pass_execution = table_headers(all_pass_output, "execution")
    nonempty_execution = table_headers(nonempty_output, "execution")

    assert all_pass_issues[0]
    assert all_pass_issues == nonempty_issues
    assert all_pass_execution[0]
    assert all_pass_execution == nonempty_execution


def test_generic_legacy_sidecars_are_all_reconciled_as_evidence(tmp_path: Path) -> None:
    archive = tmp_path / "quality_archive"
    _write_report(archive, "a", "acceptance", "pass", issues=[])
    sidecars = {}
    for source in (
        "manifest",
        "precheck_check_results",
        "precheck_clip_aggregates",
        "precheck_candidate_windows",
        "sam3_window_summary",
        "video_quality_results",
        "manual_review_labels",
    ):
        path = tmp_path / f"{source}.json"
        path.write_text(json.dumps([{"asset_id": "a", "verdict": "pass"}]), encoding="utf-8")
        sidecars[source] = path

    paths = run_projection_cli(
        archive,
        tmp_path / "out",
        formats=("csv",),
        legacy_sidecars=sidecars,
    )

    reconciliation = pd.read_csv(paths["reconciliation_csv"])
    assert set(reconciliation["source"]) == set(sidecars)
    assert set(reconciliation["difference_type"]) == {"match"}


def test_reconciliation_csv_has_fixed_headers_for_empty_rows(tmp_path: Path) -> None:
    archive = tmp_path / "quality_archive"
    _write_report(archive, "a", "acceptance", "pass", issues=[])
    projection = project_quality_archive(archive)
    output = tmp_path / "reconciliation.csv"

    write_reconciliation_only(projection, {}, output)

    assert tuple(pd.read_csv(output, nrows=0).columns) == (
        "source",
        "legacy_path",
        "legacy_row_index",
        "asset_id",
        "legacy_verdict",
        "qc_json_verdict",
        "difference_type",
    )


def test_reconciliation_rejects_unsupported_sidecar_format(tmp_path: Path) -> None:
    archive = tmp_path / "quality_archive"
    _write_report(archive, "a", "acceptance", "pass", issues=[])
    unsupported = tmp_path / "manifest.yaml"
    unsupported.write_text("asset_id: a\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported legacy sidecar extension"):
        run_projection_cli(
            archive,
            tmp_path / "out",
            formats=("csv",),
            legacy_sidecars={"manifest": unsupported},
        )


def test_batch_and_xjgt_accept_legacy_reconciliation_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_batch_qc_ledger.py",
            "--quality-archive",
            "archive",
            "--output-dir",
            "out",
            "--legacy-reconciliation-manifest",
            "manifest.json",
            "--legacy-reconciliation-precheck-clip-aggregates",
            "clip.json",
            "--legacy-reconciliation-precheck-check-results",
            "check.json",
            "--legacy-reconciliation-candidate-windows",
            "candidate.json",
            "--legacy-reconciliation-sam3-window-summary",
            "sam3.json",
            "--legacy-reconciliation-video-quality-results",
            "video.json",
            "--legacy-reconciliation-manual-review-labels",
            "manual.json",
        ],
    )
    batch = parse_batch_args()
    assert batch.supplier_sample_manifest == Path("manifest.json")
    assert batch.legacy_reconciliation_precheck_check_results == Path("check.json")

    xjgt = parse_xjgt_args(
        [
            "--quality-archive",
            "archive",
            "--output-dir",
            "out",
            "--legacy-reconciliation-sam3-window-summary",
            "sam3.json",
            "--legacy-reconciliation-manual-review-labels",
            "manual.json",
            "--legacy-reconciliation-video-quality-summary",
            "video-summary.json",
        ]
    )
    assert xjgt.sam3_window_summary == Path("sam3.json")
    assert xjgt.manual_review_labels == Path("manual.json")
    assert xjgt.video_quality_summary == Path("video-summary.json")


def test_batch_cli_reconciles_every_legacy_input_without_changing_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "quality_archive"
    _write_report(archive, "a", "acceptance", "pass", issues=[])
    sidecar_names = {
        "manifest": "manifest.json",
        "precheck_check_results": "check.json",
        "precheck_clip_aggregates": "clip.json",
        "precheck_candidate_windows": "candidate.json",
        "sam3_window_summary": "sam3.json",
        "video_quality_results": "video.json",
        "manual_review_labels": "manual.json",
    }
    args = [
        "build_batch_qc_ledger.py",
        "--quality-archive",
        str(archive),
        "--output-dir",
        str(tmp_path / "out"),
        "--formats",
        "csv",
    ]
    for source, name in sidecar_names.items():
        path = tmp_path / name
        path.write_text(json.dumps([{"asset_id": "a", "verdict": "pass"}]), encoding="utf-8")
        option = {
            "manifest": "--legacy-reconciliation-manifest",
            "precheck_check_results": "--legacy-reconciliation-precheck-check-results",
            "precheck_clip_aggregates": "--legacy-reconciliation-precheck-clip-aggregates",
            "precheck_candidate_windows": "--legacy-reconciliation-candidate-windows",
            "sam3_window_summary": "--legacy-reconciliation-sam3-window-summary",
            "video_quality_results": "--legacy-reconciliation-video-quality-results",
            "manual_review_labels": "--legacy-reconciliation-manual-review-labels",
        }[source]
        args.extend([option, str(path)])
    monkeypatch.setattr(sys, "argv", args)

    assert batch_main() == 0
    reconciliation = pd.read_csv(tmp_path / "out" / "reconciliation.csv")
    assert set(reconciliation["source"]) == set(sidecar_names)
    assert pd.read_csv(tmp_path / "out" / "batch_qc_ledger.csv").loc[0, "decision"] == "pass"
