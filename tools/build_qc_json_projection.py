#!/usr/bin/env python3
"""Publish the canonical asset QC JSON reports as batch projections.

The JSON files under ``quality_archive`` are the only source of formal
verdicts.  Legacy sidecars can be supplied to this command for reconciliation
evidence, but they never participate in the asset/issue/execution tables or in
the aggregate statistics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

# Keep direct script invocation equivalent to module invocation before loading
# repository-local ``qc_reporting`` modules.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from openpyxl import Workbook

from qc_reporting.aggregate import aggregate_projection
from qc_reporting.cache import (
    build_source_manifest,
    load_projection_cache,
    write_projection_cache,
)
from qc_reporting.projection import BatchProjection, project_quality_archive


LOGGER = logging.getLogger("build_qc_json_projection")
SUPPORTED_FORMATS = ("csv", "parquet", "xlsx", "markdown")
RECONCILIATION_COLUMNS = (
    "source",
    "legacy_path",
    "legacy_row_index",
    "asset_id",
    "legacy_verdict",
    "qc_json_verdict",
    "difference_type",
    "authoritative_source",
)
_TABLE_ROWS = {
    "asset": "asset_rows",
    "issue": "issue_rows",
    "execution": "execution_rows",
}
# Projection tables are a public interchange format.  Keep their columns
# stable even when a valid batch has no rows for a table (for example, an
# all-pass batch has no issue rows).
_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "asset": (
        "asset_id",
        "supplier_id",
        "profile",
        "schema_version",
        "status",
        "pipeline_status",
        "decision",
        "overall_decision",
        "report_revision",
        "config_hash",
        "config_version",
        "config_path",
        "module_coverage",
        "stop_position",
        "stop_reason",
        "finalizable",
    ),
    "issue": (
        "asset_id",
        "supplier_id",
        "profile",
        "report_revision",
        "issue_id",
        "module",
        "rule_id",
        "code",
        "issue_type",
        "machine_severity",
        "machine_verdict",
        "severity",
        "human_verdict",
        "effective_verdict",
        "needs_manual_review",
        "window_start_frame",
        "window_end_frame",
        "source_level",
        "review",
    ),
    "execution": (
        "asset_id",
        "supplier_id",
        "profile",
        "report_revision",
        "module",
        "state",
        "duration_sec",
        "duration_ms",
        "duration",
        "continued_after_fail",
        "runtime_error",
        "runtime_errors",
    ),
}


def _json_safe(value: Any) -> Any:
    """Convert projection extension values into deterministic JSON values."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if value == value and value not in {float("inf"), float("-inf")} else None
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _rows_for_table(projection: BatchProjection, table: str) -> tuple[Mapping[str, Any], ...]:
    if table not in _TABLE_ROWS:
        raise ValueError(f"unsupported projection table: {table}")
    rows = getattr(projection, _TABLE_ROWS[table])
    return tuple(row for row in rows if isinstance(row, Mapping))


def _dataframe(
    rows: Iterable[Mapping[str, Any]],
    *,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append(
            {
                str(key): (
                    json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (Mapping, tuple, list))
                    else _json_safe(value)
                )
                for key, value in row.items()
            }
        )
    if columns is None:
        return pd.DataFrame(normalized)
    return pd.DataFrame(normalized, columns=tuple(str(column) for column in columns))


def _table_dataframe(projection: BatchProjection, table: str) -> pd.DataFrame:
    if table not in _TABLE_COLUMNS:
        raise ValueError(f"unsupported projection table: {table}")
    return _dataframe(
        _rows_for_table(projection, table),
        columns=_TABLE_COLUMNS[table],
    )


def _write_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Parquet is a formal output in this command.  Do not silently replace it
    # with a sidecar CSV: a missing parquet engine is an actionable error.
    df.to_parquet(path, index=False)
    return path


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _summary_rows(statistics: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    overall = statistics.get("overall", {})
    if isinstance(overall, Mapping):
        for metric, value in sorted(overall.items()):
            rows.append(
                {
                    "scope": "overall",
                    "profile": "",
                    "metric": metric,
                    "value": json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (Mapping, list, tuple))
                    else _json_safe(value),
                }
            )
    by_profile = statistics.get("by_profile", {})
    if isinstance(by_profile, Mapping):
        for profile, profile_stats in sorted(by_profile.items(), key=lambda item: str(item[0])):
            if not isinstance(profile_stats, Mapping):
                continue
            for metric, value in sorted(profile_stats.items()):
                rows.append(
                    {
                        "scope": "profile",
                        "profile": str(profile),
                        "metric": metric,
                        "value": json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)
                        if isinstance(value, (Mapping, list, tuple))
                        else _json_safe(value),
                    }
                )
    return rows


def _data_dictionary_rows() -> list[dict[str, str]]:
    return [
        {"sheet": "Summary", "column": "scope", "definition": "overall or execution profile scope"},
        {"sheet": "Summary", "column": "profile", "definition": "execution profile when scope=profile"},
        {"sheet": "Summary", "column": "metric", "definition": "aggregate_projection statistic name"},
        {"sheet": "Summary", "column": "value", "definition": "aggregate statistic value"},
        {"sheet": "Assets", "column": "overall_decision", "definition": "formal decision from QC JSON"},
        {"sheet": "Issues", "column": "machine_severity", "definition": "machine severity from QC JSON issue"},
        {"sheet": "Issues", "column": "human_verdict", "definition": "optional human issue verdict from QC JSON"},
        {"sheet": "Execution", "column": "state", "definition": "module state from QC JSON execution"},
        {"sheet": "Execution", "column": "runtime_error", "definition": "structured runtime error evidence"},
    ]


def _write_xlsx(
    projection: BatchProjection,
    statistics: Mapping[str, Any],
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    workbook.remove(workbook.active)
    tables: list[tuple[str, pd.DataFrame]] = [
        ("Summary", _dataframe(_summary_rows(statistics))),
        ("Assets", _table_dataframe(projection, "asset")),
        ("Issues", _table_dataframe(projection, "issue")),
        ("Execution", _table_dataframe(projection, "execution")),
        ("Data_Dictionary", _dataframe(_data_dictionary_rows())),
    ]
    for sheet_name, frame in tables:
        sheet = workbook.create_sheet(sheet_name)
        sheet.append([str(column) for column in frame.columns])
        for row in frame.itertuples(index=False, name=None):
            sheet.append([None if pd.isna(value) else value for value in row])
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
    workbook.save(path)
    return path


def _markdown(statistics: Mapping[str, Any]) -> str:
    overall = statistics.get("overall", {})
    lines = ["# QC JSON Projection", "", "## Overall", ""]
    if isinstance(overall, Mapping):
        lines.extend([f"- **{key}**: `{_markdown_value(value)}`" for key, value in sorted(overall.items())])
    by_profile = statistics.get("by_profile", {})
    if isinstance(by_profile, Mapping) and by_profile:
        lines.extend(["", "## By profile", ""])
        for profile, profile_stats in sorted(by_profile.items(), key=lambda item: str(item[0])):
            lines.extend([f"### {profile}", ""])
            if isinstance(profile_stats, Mapping):
                lines.extend(
                    [f"- **{key}**: `{_markdown_value(value)}`" for key, value in sorted(profile_stats.items())]
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _markdown_value(value: Any) -> str:
    safe = _json_safe(value)
    if isinstance(safe, (Mapping, list)):
        return json.dumps(safe, ensure_ascii=False, sort_keys=True)
    return str(safe)


def write_projection_outputs(
    projection: BatchProjection,
    statistics: Mapping[str, Any],
    output_dir: Path,
    formats: Iterable[str] = SUPPORTED_FORMATS,
) -> dict[str, Path]:
    """Write all requested tables from one projection/statistics snapshot.

    The returned keys are stable and intentionally use singular table names
    (``asset_csv``, ``issue_csv`` and ``execution_csv``).  The on-disk names
    stay plural for compatibility with existing batch tooling.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = tuple(dict.fromkeys(str(fmt).lower() for fmt in formats))
    unsupported = sorted(set(requested) - set(SUPPORTED_FORMATS))
    if unsupported:
        raise ValueError(f"unsupported projection formats: {', '.join(unsupported)}")
    paths: dict[str, Path] = {}
    frames = {
        "asset": _table_dataframe(projection, "asset"),
        "issue": _table_dataframe(projection, "issue"),
        "execution": _table_dataframe(projection, "execution"),
    }
    if "csv" in requested:
        for table, frame in frames.items():
            paths[f"{table}_csv"] = _write_csv(frame, output_dir / f"{table}s.csv")
    if "parquet" in requested:
        for table, frame in frames.items():
            paths[f"{table}_parquet"] = _write_parquet(frame, output_dir / f"{table}s.parquet")
    if "xlsx" in requested:
        paths["xlsx"] = _write_xlsx(projection, statistics, output_dir / "qc_projection.xlsx")
    if "markdown" in requested:
        markdown_path = output_dir / "qc_projection.md"
        markdown_path.write_text(_markdown(statistics), encoding="utf-8")
        paths["markdown"] = markdown_path
    # Keep the exact aggregate snapshot available to consumers regardless of
    # presentation format; this is the source used to render Markdown/XLSX.
    statistics_path = output_dir / "statistics.json"
    statistics_path.write_text(
        json.dumps(_json_safe(statistics), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["statistics_json"] = statistics_path
    return paths


def _read_legacy_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(row) for row in payload if isinstance(row, Mapping)]
        if isinstance(payload, Mapping):
            for key in ("rows", "records", "items", "windows", "results"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [dict(row) for row in value if isinstance(row, Mapping)]
            return [dict(payload)]
        raise ValueError(f"legacy sidecar JSON must contain an object or list: {path}")
    if suffix == ".csv":
        frame = pd.read_csv(path, dtype={"asset_id": str})
    elif suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        raise ValueError(f"unsupported legacy sidecar extension: {path}")
    frame = frame.where(pd.notna(frame), None)
    return [dict(row) for row in frame.to_dict(orient="records")]


def _legacy_verdict(row: Mapping[str, Any]) -> str | None:
    for key in (
        "auto_verdict",
        "overall_decision",
        "final_verdict",
        "verdict",
        "source_verdict",
        "status",
        "decision",
    ):
        value = row.get(key)
        if value is None or value == "":
            continue
        return str(value).strip().lower()
    return None


def build_reconciliation_rows(
    projection: BatchProjection,
    sidecars: Mapping[str, Path | None],
) -> list[dict[str, Any]]:
    """Compare legacy sidecar evidence to QC JSON without changing verdicts."""

    sidecars = _normalize_legacy_sidecars(sidecars)

    assets = {
        str(row.get("asset_id")): row
        for row in projection.asset_rows
        if isinstance(row, Mapping) and row.get("asset_id") not in {None, ""}
    }
    rows: list[dict[str, Any]] = []
    for source_name, raw_path in sidecars.items():
        if raw_path is None:
            continue
        path = Path(raw_path)
        for index, legacy in enumerate(_read_legacy_records(path)):
            asset_id = str(
                legacy.get("asset_id")
                or legacy.get("clip_id")
                or legacy.get("content_id")
                or legacy.get("id")
                or ""
            )
            qc_row = assets.get(asset_id)
            qc_verdict = None if qc_row is None else qc_row.get("overall_decision")
            legacy_verdict = _legacy_verdict(legacy)
            if qc_row is None:
                difference_type = "legacy_asset_missing_in_qc_json"
            elif legacy_verdict is None:
                difference_type = "legacy_verdict_missing"
            elif str(qc_verdict or "").lower() != legacy_verdict:
                difference_type = "legacy_conflicts_with_qc_json"
            else:
                difference_type = "match"
            rows.append(
                {
                    "source": source_name,
                    "legacy_path": str(path),
                    "legacy_row_index": index,
                    "asset_id": asset_id,
                    "legacy_verdict": legacy_verdict,
                    "qc_json_verdict": qc_verdict,
                    "difference_type": difference_type,
                    # Legacy rows are evidence only, including matching rows;
                    # canonical QC JSON remains the source of truth.
                    "authoritative_source": "asset_qc_json",
                }
            )
    return rows


def write_reconciliation_only(
    projection: BatchProjection,
    sidecars: Mapping[str, Path | None],
    output_path: Path,
) -> Path:
    """Write sidecar differences as evidence-only CSV."""

    rows = build_reconciliation_rows(projection, sidecars)
    frame = _dataframe(rows, columns=RECONCILIATION_COLUMNS)
    return _write_csv(frame, Path(output_path))


def _normalize_legacy_sidecars(
    sidecars: Mapping[str, Path | None],
) -> dict[str, Path | None]:
    """Validate and normalize a source-name to legacy path mapping.

    ``None`` is retained as an explicit omitted input so callers can build a
    mapping directly from optional CLI arguments.  Every non-empty source is
    preserved; unsupported file types are rejected later by
    :func:`_read_legacy_records` instead of being silently ignored.
    """

    if not isinstance(sidecars, Mapping):
        raise TypeError("legacy_sidecars must be a mapping of source names to paths")
    normalized: dict[str, Path | None] = {}
    for raw_source, raw_path in sidecars.items():
        source = str(raw_source).strip() if isinstance(raw_source, str) else ""
        if not source:
            raise ValueError("legacy reconciliation source names must be non-empty strings")
        if raw_path is None:
            normalized[source] = None
            continue
        try:
            normalized[source] = Path(raw_path)
        except TypeError as exc:
            raise TypeError(
                f"legacy reconciliation path for {source!r} must be path-like or None"
            ) from exc
    return normalized


def run_projection_cli(
    quality_archive: Path,
    output_dir: Path,
    *,
    formats: Iterable[str] = SUPPORTED_FORMATS,
    cache_dir: Path | None = None,
    legacy_sidecars: Mapping[str, Path | None] | None = None,
    legacy_candidate_windows: Path | None = None,
    legacy_sam3_window_summary: Path | None = None,
    legacy_video_quality: Path | None = None,
    legacy_issue_events: Path | None = None,
) -> dict[str, Path]:
    archive_path = Path(quality_archive)
    if cache_dir is None:
        projection = project_quality_archive(archive_path)
    else:
        cache_path = Path(cache_dir)
        expected_manifest = build_source_manifest(archive_path)
        projection = load_projection_cache(cache_path, expected_manifest)
        if projection is None:
            projection = project_quality_archive(archive_path)
            write_projection_cache(projection, cache_path)
    statistics = aggregate_projection(projection)
    paths = write_projection_outputs(projection, statistics, Path(output_dir), formats)
    sidecars = _normalize_legacy_sidecars(legacy_sidecars or {})
    compatibility_sidecars = {
        "candidate_windows": legacy_candidate_windows,
        "sam3_window_summary": legacy_sam3_window_summary,
        "video_quality": legacy_video_quality,
        "issue_events": legacy_issue_events,
    }
    for source, raw_path in compatibility_sidecars.items():
        if raw_path is None:
            continue
        path = Path(raw_path)
        if source in sidecars and sidecars[source] != path:
            raise ValueError(
                f"legacy reconciliation source {source!r} was supplied with conflicting paths"
            )
        sidecars[source] = path
    if any(path is not None for path in sidecars.values()):
        paths["reconciliation_csv"] = write_reconciliation_only(
            projection,
            sidecars,
            Path(output_dir) / "reconciliation.csv",
        )
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project canonical QC JSON reports into batch tables and statistics."
    )
    parser.add_argument("--quality-archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=SUPPORTED_FORMATS,
        default=list(SUPPORTED_FORMATS),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional rebuildable parquet cache directory for the projection.",
    )
    parser.add_argument("--legacy-reconciliation-candidate-windows", type=Path)
    parser.add_argument("--legacy-reconciliation-sam3-window-summary", type=Path)
    parser.add_argument("--legacy-reconciliation-video-quality", type=Path)
    parser.add_argument("--legacy-reconciliation-issue-events", type=Path)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    paths = run_projection_cli(
        args.quality_archive,
        args.output_dir,
        formats=args.formats,
        cache_dir=args.cache_dir,
        legacy_candidate_windows=args.legacy_reconciliation_candidate_windows,
        legacy_sam3_window_summary=args.legacy_reconciliation_sam3_window_summary,
        legacy_video_quality=args.legacy_reconciliation_video_quality,
        legacy_issue_events=args.legacy_reconciliation_issue_events,
    )
    for path in paths.values():
        LOGGER.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RECONCILIATION_COLUMNS",
    "build_reconciliation_rows",
    "main",
    "parse_args",
    "run_projection_cli",
    "write_projection_outputs",
    "write_reconciliation_only",
]
