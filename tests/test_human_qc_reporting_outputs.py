from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from qc_reporting.aggregate import aggregate_projection
from qc_reporting.export import FORMAL_AGGREGATE_METRICS, write_aggregate_outputs


def _statistics() -> dict[str, object]:
    projection = SimpleNamespace(
        asset_rows=(
            {
                "asset_id": "asset-a",
                "profile": "supplier_evaluation",
                "status": "completed",
                "decision": "fail",
                "report_revision": 3,
            },
        ),
        issue_rows=(
            {
                "asset_id": "asset-a",
                "profile": "supplier_evaluation",
                "issue_id": "machine-fail",
                "machine_severity": "fail",
                "report_revision": 3,
            },
            {
                "asset_id": "asset-a",
                "profile": "supplier_evaluation",
                "issue_id": "warn-pass",
                "machine_severity": "warn",
                "report_revision": 3,
            },
            {
                "asset_id": "asset-a",
                "profile": "supplier_evaluation",
                "issue_id": "warn-fail",
                "machine_severity": "warn",
                "report_revision": 3,
            },
        ),
        execution_rows=(),
        human_review_rows=(
            {
                "asset_id": "asset-a",
                "profile": "supplier_evaluation",
                "report_revision": 3,
                "timeline_edit_count": 1,
                "subtask_text_edit_count": 2,
                "issue_reviews": {
                    "warn-pass": {
                        "human_verdict": "pass",
                        "effective_verdict": "pass",
                    },
                    "warn-fail": {
                        "human_verdict": "fail",
                        "effective_verdict": "fail",
                    },
                },
            },
        ),
    )
    return aggregate_projection(projection)


EXPECTED_OVERALL = {
    "auto_fail_assets": "1",
    "auto_fail_issues": "1",
    "machine_warn_issues": "2",
    "human_checked_warn_issues": "2",
    "human_resolved_warn_issues": "1",
    "human_confirmed_fail_issues": "1",
    "unreviewed_selected_warn_issues": "0",
    "timeline_edit_count": "1",
    "subtask_text_edit_count": "2",
    "final_pass_assets": "0",
    "final_fail_assets": "1",
    "pass_rate": "0.0",
}


def _metric_map(frame: pd.DataFrame, *, profile: str = "overall") -> dict[str, str]:
    selected = frame[
        (frame["scope"] == ("overall" if profile == "overall" else "profile"))
        & (frame["profile"] == profile)
    ]
    return dict(zip(selected["metric"], selected["value_json"], strict=True))


def _read_markdown(path: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or line.startswith("| scope ") or line.startswith("|---"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows.append(dict(zip(("scope", "profile", "metric", "value_json"), cells, strict=True)))
    return pd.DataFrame(rows)


def test_all_formal_formats_publish_identical_human_metrics(tmp_path: Path) -> None:
    paths = write_aggregate_outputs(_statistics(), tmp_path)

    assert set(paths) == {"csv", "parquet", "xlsx", "markdown"}
    frames = {
        "csv": pd.read_csv(paths["csv"], dtype=str, keep_default_na=False),
        "parquet": pd.read_parquet(paths["parquet"]).astype(str),
        "xlsx": pd.read_excel(paths["xlsx"], sheet_name="Metrics", dtype=str).fillna(""),
        "markdown": _read_markdown(paths["markdown"]),
    }

    for name, frame in frames.items():
        assert tuple(frame.columns) == ("scope", "profile", "metric", "value_json"), name
        assert _metric_map(frame) == EXPECTED_OVERALL, name
        assert _metric_map(frame, profile="supplier_evaluation") == EXPECTED_OVERALL, name


def test_formal_metric_contract_lists_every_required_human_field() -> None:
    assert FORMAL_AGGREGATE_METRICS == (
        "auto_fail_assets",
        "auto_fail_issues",
        "machine_warn_issues",
        "human_checked_warn_issues",
        "human_resolved_warn_issues",
        "human_confirmed_fail_issues",
        "unreviewed_selected_warn_issues",
        "timeline_edit_count",
        "subtask_text_edit_count",
        "final_pass_assets",
        "final_fail_assets",
        "pass_rate",
    )


def test_export_fails_closed_when_statistics_are_missing_a_formal_metric(
    tmp_path: Path,
) -> None:
    statistics = _statistics()
    del statistics["overall"]["human_resolved_warn_issues"]  # type: ignore[index]

    with pytest.raises(ValueError, match="human_resolved_warn_issues"):
        write_aggregate_outputs(statistics, tmp_path)
