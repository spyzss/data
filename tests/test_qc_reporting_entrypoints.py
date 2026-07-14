from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from tools.build_qc_json_projection import main as projection_main
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
