"""Canonical tabular exports for formal QC aggregate statistics.

All supported formats are rendered from the same long-form rows.  Keeping one
normalized representation prevents CSV, Parquet, XLSX, and Markdown from
silently drifting to different metric names or denominators.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd


FORMAL_AGGREGATE_METRICS = (
    "auto_fail_assets",
    "auto_fail_issues",
    "machine_warn_issues",
    "human_checked_warn_issues",
    "human_resolved_warn_issues",
    "human_confirmed_fail_issues",
    "timeline_edit_count",
    "subtask_text_edit_count",
    "final_pass_assets",
    "final_fail_assets",
    "pass_rate",
)

SUPPORTED_AGGREGATE_FORMATS = ("csv", "parquet", "xlsx", "markdown")
_COLUMNS = ("scope", "profile", "metric", "value_json")


def aggregate_metric_rows(statistics: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    """Return the one canonical row set used by every formal output format."""

    if not isinstance(statistics, Mapping):
        raise TypeError("statistics must be a mapping")
    overall = statistics.get("overall")
    by_profile = statistics.get("by_profile")
    if not isinstance(overall, Mapping):
        raise ValueError("statistics.overall must be a mapping")
    if not isinstance(by_profile, Mapping):
        raise ValueError("statistics.by_profile must be a mapping")

    rows = list(_group_rows("overall", "overall", overall))
    for raw_profile in sorted(by_profile, key=str):
        profile = str(raw_profile)
        values = by_profile[raw_profile]
        if not isinstance(values, Mapping):
            raise ValueError(f"statistics.by_profile.{profile} must be a mapping")
        rows.extend(_group_rows("profile", profile, values))
    return tuple(rows)


def _group_rows(
    scope: str,
    profile: str,
    values: Mapping[str, Any],
) -> tuple[dict[str, str], ...]:
    missing = [metric for metric in FORMAL_AGGREGATE_METRICS if metric not in values]
    if missing:
        raise ValueError(
            f"statistics {scope}/{profile} is missing formal metric: {missing[0]}"
        )
    return tuple(
        {
            "scope": scope,
            "profile": profile,
            "metric": metric,
            "value_json": json.dumps(
                values[metric], ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        }
        for metric in FORMAL_AGGREGATE_METRICS
    )


def write_aggregate_outputs(
    statistics: Mapping[str, Any],
    output_dir: Path,
    formats: Sequence[str] = SUPPORTED_AGGREGATE_FORMATS,
) -> dict[str, Path]:
    """Write identical formal aggregate rows in each requested format."""

    requested = tuple(dict.fromkeys(str(item).lower() for item in formats))
    unsupported = [item for item in requested if item not in SUPPORTED_AGGREGATE_FORMATS]
    if unsupported:
        raise ValueError(f"unsupported aggregate output format: {unsupported[0]}")

    rows = aggregate_metric_rows(statistics)
    frame = pd.DataFrame(rows, columns=_COLUMNS)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for format_name in requested:
        path = root / f"human_qc_aggregate.{_suffix(format_name)}"
        if format_name == "csv":
            frame.to_csv(path, index=False)
        elif format_name == "parquet":
            frame.to_parquet(path, index=False)
        elif format_name == "xlsx":
            frame.to_excel(path, sheet_name="Metrics", index=False)
        else:
            path.write_text(_markdown(rows), encoding="utf-8")
        paths[format_name] = path
    return paths


def _suffix(format_name: str) -> str:
    return "md" if format_name == "markdown" else format_name


def _markdown(rows: Sequence[Mapping[str, str]]) -> str:
    lines = [
        "# Human QC aggregate metrics",
        "",
        "| scope | profile | metric | value_json |",
        "|---|---|---|---|",
    ]
    lines.extend(
        "| " + " | ".join(_markdown_cell(row[column]) for column in _COLUMNS) + " |"
        for row in rows
    )
    return "\n".join(lines) + "\n"


def _markdown_cell(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


__all__ = [
    "FORMAL_AGGREGATE_METRICS",
    "SUPPORTED_AGGREGATE_FORMATS",
    "aggregate_metric_rows",
    "write_aggregate_outputs",
]
