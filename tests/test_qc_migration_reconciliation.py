from __future__ import annotations

import json
from pathlib import Path

from qc_reporting.migration import reconcile_legacy_outputs
from qc_reporting.projection import project_quality_archive
from tests.qc_report_fixtures import make_v1_video_report, make_v2_report


def _write_migration_fixture(tmp_path: Path) -> tuple[Path, Path]:
    archive = tmp_path / "quality_archive"
    archive.mkdir()
    report_path = archive / "408817.json"
    report_path.write_text(
        json.dumps(make_v1_video_report(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    legacy_path = tmp_path / "legacy_result.json"
    legacy_path.write_text(
        json.dumps([{"asset_id": "408817", "verdict": "fail"}]) + "\n",
        encoding="utf-8",
    )
    return report_path, legacy_path


def test_migration_reconciliation_never_rewrites_source_or_uses_legacy_verdict(
    tmp_path: Path,
) -> None:
    v1_path, legacy = _write_migration_fixture(tmp_path)
    before = v1_path.read_bytes()

    projected = project_quality_archive(v1_path.parent)
    differences = reconcile_legacy_outputs(
        quality_archive=v1_path.parent,
        legacy_inputs=[legacy],
    )

    assert v1_path.read_bytes() == before
    assert projected.asset_rows[0]["overall_decision"] is None
    assert differences == [
        {
            "source": "legacy_result",
            "legacy_path": str(legacy),
            "legacy_row_index": 0,
            "asset_id": "408817",
            "legacy_verdict": "fail",
            "qc_json_verdict": None,
            "difference_type": "legacy_conflicts_with_qc_json",
            "authoritative_source": "asset_qc_json",
        }
    ]


def test_migration_reconciliation_returns_only_differences_in_stable_order(
    tmp_path: Path,
) -> None:
    v1_path, _legacy = _write_migration_fixture(tmp_path)
    v2_report = make_v2_report(status="completed", overall_decision="pass")
    v2_report["asset_id"] = "passed"
    v2_report["execution"]["profile"] = "acceptance"
    (v1_path.parent / "passed.json").write_text(
        json.dumps(v2_report) + "\n",
        encoding="utf-8",
    )
    matching = tmp_path / "matching.json"
    matching.write_text(
        json.dumps([{"asset_id": "passed", "verdict": "pass"}]) + "\n",
        encoding="utf-8",
    )
    missing = tmp_path / "missing.json"
    missing.write_text(
        json.dumps([{"asset_id": "unknown", "verdict": "pass"}]) + "\n",
        encoding="utf-8",
    )

    differences = reconcile_legacy_outputs(
        quality_archive=v1_path.parent,
        legacy_inputs=[missing, matching],
    )

    assert [row["legacy_path"] for row in differences] == [str(missing)]
    assert differences[0]["difference_type"] == "legacy_asset_missing_in_qc_json"
    assert differences[0]["authoritative_source"] == "asset_qc_json"
