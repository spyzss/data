"""Storage helpers for QC runner outputs."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .types import CheckResult

logger = logging.getLogger(__name__)


def results_to_dataframe(results: list[CheckResult]) -> pd.DataFrame:
    records = [result.to_record() for result in results]
    df = pd.DataFrame(records)
    if not df.empty:
        df["metrics"] = df["metrics"].apply(json.dumps)
    return df


def write_dataframe(df: pd.DataFrame, path: Path) -> Path:
    """
    Write a dataframe, preferring parquet and falling back to CSV.

    The fallback keeps laptop smoke tests useful in environments without a
    parquet engine installed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception as exc:
        fallback = path.with_suffix(".csv")
        logger.warning("Could not write parquet %s (%s); writing %s", path, exc, fallback)
        df.to_csv(fallback, index=False)
        return fallback


def dataframe_to_json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a result dataframe to JSON records with metrics as objects."""
    records = df.to_dict(orient="records")
    for record in records:
        metrics = record.get("metrics")
        if isinstance(metrics, str):
            try:
                record["metrics"] = json.loads(metrics)
            except json.JSONDecodeError:
                record["metrics"] = {}
    return records


def write_json_records(records: list[dict[str, Any]], path: Path) -> Path:
    """Write JSON records for interface consumers that do not read parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_safe(records),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_existing_dataframe(path: Path) -> pd.DataFrame | None:
    """Read an existing parquet result, or its CSV fallback, if present."""
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            logger.warning("Could not read parquet %s: %s", path, exc)
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def aggregate_results(results: list[CheckResult]) -> list[dict[str, Any]]:
    """Build per-clip/per-check aggregate rows from per-frame results."""
    grouped: dict[tuple[int, str], list[CheckResult]] = {}
    for result in results:
        grouped.setdefault((result.episode_idx, result.check), []).append(result)

    aggregates: list[dict[str, Any]] = []
    for (episode_idx, check), rows in sorted(grouped.items()):
        calibrated = [row.flag for row in rows if row.flag is not None]
        clip_flag = any(calibrated) if calibrated else None
        aggregates.append(
            {
                "episode_idx": episode_idx,
                "check": check,
                "checked_frames": len(rows),
                "flagged_frames": sum(row.flag is True for row in rows),
                "uncalibrated_frames": sum(row.flag is None for row in rows),
                "clip_flag": clip_flag,
            }
        )
    return aggregates
