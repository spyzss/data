#!/usr/bin/env python3
"""Read-only native-versus-standardized temporal metric audit."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precheck.checks.keypoint_temporal import KeypointTemporalCheck  # noqa: E402
from precheck.adapters.supplier_hdf5 import load_supplier_hdf5_clip  # noqa: E402
from qc_common.config import load_qc_acceptance_config  # noqa: E402
from qc_common.types import CheckResult, ClipInputs  # noqa: E402
from tools.run_manifest_precheck import (  # noqa: E402
    load_deepreach_clip,
    load_jdt_clip,
    read_manifest,
)


AUDIT_SCHEMA_VERSION = "temporal_timebase_ab_audit.v1"
_NATIVE_FIELDS = {
    "acceleration": "joint_acceleration_m_s2_max",
    "displacement": "joint_displacement_m_max",
    "angle": "joint_angle_change_deg_max",
    "rotation": "rotation_delta_max",
}
_STANDARDIZED_FIELDS = {
    "acceleration": "joint_acceleration_standardized_m_s2_max",
    "displacement": "joint_displacement_standardized_m_max",
    "angle": "joint_angle_change_standardized_deg_max",
    "rotation": "rotation_delta_standardized_max",
}
_THRESHOLD_BY_LOGICAL_NAME = {
    "acceleration": "joint_acceleration_m_s2_max_threshold",
    "displacement": "joint_displacement_m_max_threshold",
    "angle": "joint_angle_change_deg_max_threshold",
    "rotation": "rotation_delta_max_threshold",
}


def _finite_values(rows: Sequence[CheckResult], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.metrics.get(field)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            values.append(numeric)
    return values


def _percentiles(values: Sequence[float], prefix: str) -> dict[str, float | None]:
    if not values:
        return {
            f"{prefix}_p50": None,
            f"{prefix}_p90": None,
            f"{prefix}_p95": None,
            f"{prefix}_p99": None,
            f"{prefix}_max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_p50": float(np.percentile(array, 50)),
        f"{prefix}_p90": float(np.percentile(array, 90)),
        f"{prefix}_p95": float(np.percentile(array, 95)),
        f"{prefix}_p99": float(np.percentile(array, 99)),
        f"{prefix}_max": float(np.max(array)),
    }


def _seed_and_exceed_counts(
    rows: Sequence[CheckResult],
    *,
    fields: Mapping[str, str],
    eligibility_field: str,
    parameters: Mapping[str, Any],
) -> tuple[int, int, int]:
    eligible_count = 0
    exceed_count = 0
    seed_count = 0
    for row in rows:
        if row.metrics.get(eligibility_field) is not True:
            continue
        values: dict[str, float] = {}
        for logical_name, field in fields.items():
            try:
                value = float(row.metrics.get(field))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values[logical_name] = value
        if not values:
            continue
        eligible_count += 1
        exceeded = {
            name
            for name, value in values.items()
            if value > float(parameters[_THRESHOLD_BY_LOGICAL_NAME[name]])
        }
        if exceeded:
            exceed_count += 1
        if (
            "acceleration" in exceeded
            or "displacement" in exceeded
            or len(exceeded) >= 2
        ):
            seed_count += 1
    return eligible_count, exceed_count, seed_count


def _summarize_rows(
    rows: Sequence[CheckResult],
    *,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for label, field in (
        ("native_acceleration", _NATIVE_FIELDS["acceleration"]),
        ("native_displacement", _NATIVE_FIELDS["displacement"]),
        (
            "standardized_acceleration",
            _STANDARDIZED_FIELDS["acceleration"],
        ),
        (
            "standardized_displacement",
            _STANDARDIZED_FIELDS["displacement"],
        ),
    ):
        summary.update(_percentiles(_finite_values(rows, field), label))

    native_eligible, native_exceeded, native_seeds = _seed_and_exceed_counts(
        rows,
        fields=_NATIVE_FIELDS,
        eligibility_field="temporal_pair_eligible",
        parameters=parameters,
    )
    standardized_eligible, standardized_exceeded, standardized_seeds = (
        _seed_and_exceed_counts(
            rows,
            fields=_STANDARDIZED_FIELDS,
            eligibility_field="standardized_temporal_pair_eligible",
            parameters=parameters,
        )
    )
    summary.update(
        {
            "native_eligible_frame_count": native_eligible,
            "standardized_eligible_frame_count": standardized_eligible,
            "native_threshold_exceed_rate": (
                native_exceeded / native_eligible if native_eligible else None
            ),
            "standardized_threshold_exceed_rate": (
                standardized_exceeded / standardized_eligible
                if standardized_eligible
                else None
            ),
            "native_candidate_seed_rate": (
                native_seeds / native_eligible if native_eligible else None
            ),
            "standardized_candidate_seed_rate": (
                standardized_seeds / standardized_eligible
                if standardized_eligible
                else None
            ),
            "native_candidate_seed_count": native_seeds,
            "standardized_candidate_seed_count": standardized_seeds,
            "standardized_sample_count": sum(
                row.metrics.get("standardized_sample_selected") is True
                for row in rows
            ),
        }
    )
    return summary


def _default_clip_loader(
    supplier: str,
) -> Callable[[dict[str, Any], int], ClipInputs]:
    jdt_cache: dict[Path, pd.DataFrame] = {}

    def load(row: dict[str, Any], row_index: int) -> ClipInputs:
        if supplier in {"dr", "deepreach"}:
            return load_deepreach_clip(row, episode_idx=row_index)
        if supplier == "xjgt":
            source_text = row.get("hdf5_path") or row.get("source_path")
            if not source_text:
                raise ValueError("xjgt manifest row requires hdf5_path or source_path")
            full_clip = load_supplier_hdf5_clip(
                Path(str(source_text)).expanduser(),
                episode_idx=row_index,
                fps=row.get("fps"),
            )
            start_value = row.get("start_frame")
            end_value = row.get("end_frame")
            start = int(start_value) if start_value not in {None, ""} else 0
            end = (
                int(end_value)
                if end_value not in {None, ""}
                else max(0, full_clip.num_frames - 1)
            )
            if start < 0 or end < start or end >= full_clip.num_frames:
                raise ValueError("xjgt manifest source frame range is invalid")
            stop = end + 1
            return ClipInputs(
                episode_idx=row_index,
                frame_indices=list(range(start, stop)),
                keypoints={
                    name: values[start:stop]
                    for name, values in (full_clip.keypoints or {}).items()
                },
                rotations={
                    name: values[start:stop]
                    for name, values in (full_clip.rotations or {}).items()
                },
                timestamps_ns=(
                    None
                    if full_clip.timestamps_ns is None
                    else full_clip.timestamps_ns[start:stop]
                ),
                fps=full_clip.fps,
            )
        parquet_path = Path(str(row["parquet_path"])).expanduser()
        if parquet_path not in jdt_cache:
            jdt_cache[parquet_path] = pd.read_parquet(parquet_path)
        return load_jdt_clip(
            row,
            episode_idx=row_index,
            source_frame=jdt_cache[parquet_path],
        )

    return load


def audit_temporal_timebase(
    *,
    manifest: Path,
    supplier: str,
    output_dir: Path,
    asset_ids: Sequence[str] | None,
    max_clips: int | None,
    config_path: Path | None,
    clip_loader: Callable[[dict[str, Any], int], ClipInputs] | None = None,
    temporal_config_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit temporal timebases without mutating any existing QC artifacts."""
    normalized_supplier = supplier.lower()
    if normalized_supplier not in {"jdt", "xjgt", "dr", "deepreach"}:
        raise ValueError("supplier must be jdt, xjgt, dr, or deepreach")
    if max_clips is not None and max_clips < 0:
        raise ValueError("max_clips must be >= 0")
    rows = read_manifest(Path(manifest))
    selected_asset_ids = set(asset_ids or ())
    if selected_asset_ids:
        rows = [
            row for row in rows if str(row.get("asset_id")) in selected_asset_ids
        ]
    if max_clips is not None:
        rows = rows[:max_clips]

    loaded_config = load_qc_acceptance_config(config_path)
    parameters = loaded_config.module_parameters("keypoint_temporal")
    parameters.update(dict(temporal_config_overrides or {}))
    load_clip = clip_loader or _default_clip_loader(normalized_supplier)
    output_rows: list[dict[str, Any]] = []
    all_temporal_rows: list[CheckResult] = []
    mapping_by_asset: dict[str, list[int]] = {}
    failures: list[dict[str, Any]] = []
    for row_index, manifest_row in enumerate(rows):
        asset_id = str(manifest_row.get("asset_id") or f"row-{row_index}")
        try:
            clip = load_clip(manifest_row, row_index)
            temporal_rows = KeypointTemporalCheck(parameters).run(clip)
            all_temporal_rows.extend(temporal_rows)
            audit = next(
                (
                    item.metrics["temporal_sampling_audit"]
                    for item in temporal_rows
                    if isinstance(
                        item.metrics.get("temporal_sampling_audit"), Mapping
                    )
                ),
                {},
            )
            mapping = [int(value) for value in audit.get("source_frame_mapping", [])]
            mapping_by_asset[asset_id] = mapping
            output_rows.append(
                {
                    "schema_version": AUDIT_SCHEMA_VERSION,
                    "row_type": "asset",
                    "supplier": normalized_supplier,
                    "asset_id": asset_id,
                    "status": "completed",
                    **_summarize_rows(temporal_rows, parameters=parameters),
                    "source_fps": audit.get("source_fps"),
                    "source_fps_scope": "asset",
                    "temporal_target_hz": audit.get("temporal_target_hz"),
                    "sampling_method": audit.get("sampling_method"),
                    "source_frame_count": int(audit.get("source_frame_count", 0)),
                    "timestamp_source": audit.get("timestamp_source"),
                    "duplicate_source_frame_drop_count": int(
                        audit.get("duplicate_source_frame_drop_count", 0)
                    ),
                    "invalid_timestamp_count": int(
                        audit.get("invalid_timestamp_count", 0)
                    ),
                    "non_monotonic_timestamp_count": int(
                        audit.get("non_monotonic_timestamp_count", 0)
                    ),
                    "temporal_gap_break_count": int(
                        audit.get("temporal_gap_break_count", 0)
                    ),
                    "source_frame_mapping_json": json.dumps(mapping),
                }
            )
        except Exception as exc:
            failures.append({"asset_id": asset_id, "error": str(exc)})

    overall = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "row_type": "overall",
        "supplier": normalized_supplier,
        "asset_id": "__overall__",
        "status": "completed" if not failures else "partial",
        **_summarize_rows(all_temporal_rows, parameters=parameters),
        "source_fps": None,
        "source_fps_scope": "per_asset",
        "temporal_target_hz": parameters["temporal_target_hz"],
        "sampling_method": "nearest_monotonic_no_reuse",
        "source_frame_count": sum(
            int(row.get("source_frame_count", 0)) for row in output_rows
        ),
        "timestamp_source": "mixed_or_per_asset",
        "duplicate_source_frame_drop_count": sum(
            int(row.get("duplicate_source_frame_drop_count", 0))
            for row in output_rows
        ),
        "invalid_timestamp_count": sum(
            int(row.get("invalid_timestamp_count", 0)) for row in output_rows
        ),
        "non_monotonic_timestamp_count": sum(
            int(row.get("non_monotonic_timestamp_count", 0))
            for row in output_rows
        ),
        "temporal_gap_break_count": sum(
            int(row.get("temporal_gap_break_count", 0)) for row in output_rows
        ),
        "source_frame_mapping_json": json.dumps(mapping_by_asset, sort_keys=True),
    }
    output_rows.append(overall)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(output_rows)
    frame.to_csv(output_dir / "temporal_timebase_ab_audit.csv", index=False)
    frame.to_parquet(
        output_dir / "temporal_timebase_ab_audit.parquet",
        index=False,
    )
    (output_dir / "temporal_timebase_ab_audit.json").write_text(
        json.dumps(output_rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "manifest": str(Path(manifest)),
                "supplier": normalized_supplier,
                "asset_ids": list(asset_ids or ()),
                "max_clips": max_clips,
                "config_reference": loaded_config.json_reference(),
                "temporal_target_hz": parameters["temporal_target_hz"],
                "temporal_timestamp_source": parameters[
                    "temporal_timestamp_source"
                ],
                "sampling_method": "nearest_monotonic_no_reuse",
                "decision_metric_source": "standardized_30hz",
                "models_loaded": [],
                "mutates_precheck_outputs": False,
                "failures": failures,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "manifest_row_count": len(rows),
        "completed_asset_count": len(output_rows) - 1,
        "failed_asset_count": len(failures),
        "output_dir": str(output_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit native versus standardized temporal metrics"
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--supplier",
        required=True,
        choices=("jdt", "xjgt", "dr", "deepreach"),
    )
    parser.add_argument("--asset-ids", nargs="+")
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = audit_temporal_timebase(
        manifest=args.manifest,
        supplier=args.supplier,
        output_dir=args.output_dir,
        asset_ids=args.asset_ids,
        max_clips=args.max_clips,
        config_path=args.config,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["failed_asset_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
