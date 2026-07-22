#!/usr/bin/env python3
"""Read-only native-versus-standardized temporal metric audit."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precheck.checks.keypoint_temporal import KeypointTemporalCheck  # noqa: E402
from precheck.checks.skeleton_quality_score import (  # noqa: E402
    GEOMETRY_METRIC_NAMES,
    SkeletonQualityScoreCheck,
)
from precheck.adapters.supplier_hdf5 import load_supplier_hdf5_clip  # noqa: E402
from qc_common.config import load_qc_acceptance_config  # noqa: E402
from qc_common.frame_survival import source_frame_at  # noqa: E402
from qc_common.keypoints import (  # noqa: E402
    EGODATA_HAND21_INDEX_TO_ACCEPTANCE_BASE,
)
from qc_common.types import CheckResult, ClipInputs  # noqa: E402
from tools.run_manifest_precheck import (  # noqa: E402
    load_deepreach_clip,
    load_jdt_clip,
    read_manifest,
)


AUDIT_SCHEMA_VERSION = "temporal_timebase_ab_audit.v2"
AUDIT_PRODUCER_VERSION = "temporal-timebase-audit-v2"
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
_STRONG_RATIO_BY_LOGICAL_NAME = {
    "acceleration": "strong_acceleration_ratio",
    "displacement": "strong_displacement_ratio",
}
_PROJECTION_EVIDENCE_FIELDS = (
    "needs_projection_review",
    "needs_out_of_frame_review",
)
_SEED_REASON_CATEGORIES = (
    "acceleration_only",
    "displacement_only",
    "acceleration_and_displacement",
    "angle_related",
    "rotation_related",
    "side_view_hand_orientation",
    "multiple_temporal_metrics",
    "hard_invalid_or_ineligible",
    "no_temporal_seed",
)
_EXTREME_COLUMNS = (
    "schema_version",
    "asset_id",
    "source_frame",
    "local_frame",
    "side",
    "hand",
    "joint_index",
    "joint_name",
    "evidence_source_frames",
    "evidence_timestamps",
    "previous_position",
    "current_position",
    "next_position",
    "position_abs_m",
    "native_displacement_m",
    "standardized_displacement_m",
    "native_acceleration_m_s2",
    "standardized_acceleration_m_s2",
    "morphology_status",
    "morphology_reason",
    "presence_status",
    "presence_reason",
    "projection_sam3_status",
    "anomaly_classification",
    "anomaly_source",
    "evidence_provenance",
    "supplier_acknowledged_issue_pattern",
    "frame_level_supplier_confirmed",
    "ranking_metric",
    "ranking_value",
)


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
    if len(values) == 0:
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


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _numeric(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if math.isfinite(numeric) else None


def _metric_breakdown_rows(
    rows: Sequence[CheckResult],
    *,
    asset_id: str,
    parameters: Mapping[str, Any],
) -> list[dict[str, Any]]:
    values_by_metric: dict[tuple[str, str], list[float]] = {}
    for timebase, fields, eligibility_field in (
        ("native", _NATIVE_FIELDS, "temporal_pair_eligible"),
        (
            "standardized",
            _STANDARDIZED_FIELDS,
            "standardized_temporal_pair_eligible",
        ),
    ):
        for metric, field in fields.items():
            values_by_metric[(timebase, metric)] = [
                value
                for row in rows
                if row.metrics.get(eligibility_field) is True
                for value in [_numeric(row.metrics.get(field))]
                if value is not None
            ]
    return _metric_breakdown_from_values(
        values_by_metric,
        asset_id=asset_id,
        parameters=parameters,
    )


def _metric_breakdown_from_values(
    values_by_metric: Mapping[tuple[str, str], Sequence[float]],
    *,
    asset_id: str,
    parameters: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for timebase, fields, eligibility_field in (
        ("native", _NATIVE_FIELDS, "temporal_pair_eligible"),
        (
            "standardized",
            _STANDARDIZED_FIELDS,
            "standardized_temporal_pair_eligible",
        ),
    ):
        for metric, field in fields.items():
            threshold = float(parameters[_THRESHOLD_BY_LOGICAL_NAME[metric]])
            strong_ratio_name = _STRONG_RATIO_BY_LOGICAL_NAME.get(metric)
            strong_threshold = (
                threshold * float(parameters[strong_ratio_name])
                if strong_ratio_name is not None
                else None
            )
            values = list(values_by_metric.get((timebase, metric), ()))
            exceed_count = int(sum(value > threshold for value in values))
            strong_exceed_count = (
                int(sum(value >= strong_threshold for value in values))
                if strong_threshold is not None
                else None
            )
            percentile_values = _percentiles(values, "metric")
            records.append(
                {
                    "schema_version": "temporal_metric_exceedance_breakdown.v1",
                    "asset_id": asset_id,
                    "timebase": timebase,
                    "metric": metric,
                    "metric_field": field,
                    "eligibility_field": eligibility_field,
                    "threshold": threshold,
                    "threshold_source": "runtime_qc_config",
                    "strong_threshold": strong_threshold,
                    "strong_threshold_source": (
                        f"runtime_qc_config:{strong_ratio_name}"
                        if strong_ratio_name is not None
                        else None
                    ),
                    "eligible_frame_count": len(values),
                    "exceed_frame_count": exceed_count,
                    "strong_exceed_frame_count": strong_exceed_count,
                    "exceed_rate": _safe_ratio(exceed_count, len(values)),
                    "strong_exceed_rate": (
                        _safe_ratio(int(strong_exceed_count), len(values))
                        if strong_exceed_count is not None
                        else None
                    ),
                    "p50": percentile_values["metric_p50"],
                    "p90": percentile_values["metric_p90"],
                    "p95": percentile_values["metric_p95"],
                    "p99": percentile_values["metric_p99"],
                    "max": percentile_values["metric_max"],
                }
            )
    return records


def _seed_reason_category(
    result: CheckResult,
    *,
    seed: Mapping[str, Any] | None,
) -> str:
    metrics = result.metrics
    if (
        metrics.get("temporal_output_valid") is not True
        or bool(metrics.get("keypoint_presence_invalid", 0.0))
    ):
        return "hard_invalid_or_ineligible"
    if seed is not None:
        production_reasons = {
            str(value) for value in seed.get("trigger_reason", [])
        }
        geometry_seed_reasons = production_reasons - {"multi_signal_seed"}
        if geometry_seed_reasons == {"acceleration_seed"} and (
            "multi_signal_seed" not in production_reasons
        ):
            return "acceleration_only"
        if geometry_seed_reasons == {"displacement_seed"} and (
            "multi_signal_seed" not in production_reasons
        ):
            return "displacement_only"
        if geometry_seed_reasons == {
            "acceleration_seed",
            "displacement_seed",
        }:
            return "acceleration_and_displacement"
        if production_reasons == {"extreme_rotation_delta"}:
            return "rotation_related"
        if production_reasons == {"side_view_hand_orientation"}:
            return "side_view_hand_orientation"
        if production_reasons:
            return "multiple_temporal_metrics"
    exceeded = set(metrics.get("which_thresholds_exceeded", []))
    acceleration = "joint_acceleration_m_s2_max"
    displacement = "joint_displacement_m_max"
    angle = "joint_angle_change_deg_max"
    rotation = "rotation_delta_max"
    if exceeded == {acceleration}:
        return "acceleration_only"
    if exceeded == {displacement}:
        return "displacement_only"
    if exceeded == {acceleration, displacement}:
        return "acceleration_and_displacement"
    if exceeded == {angle}:
        return "angle_related"
    if exceeded == {rotation}:
        return "rotation_related"
    if len(exceeded.intersection(GEOMETRY_METRIC_NAMES)) >= 2:
        return "multiple_temporal_metrics"
    return "no_temporal_seed"


def _seed_reason_breakdown_rows(
    results: Sequence[CheckResult],
    *,
    check: SkeletonQualityScoreCheck,
    asset_id: str,
    timebase: str,
) -> list[dict[str, Any]]:
    counts = {
        category: {"frame_count": 0, "candidate_seed_count": 0}
        for category in _SEED_REASON_CATEGORIES
    }
    production_reason_counts: dict[str, int] = {}
    for result in results:
        if result.frame_idx < 0:
            continue
        seed = check.temporal_seed_record(result, asset_id)
        category = _seed_reason_category(result, seed=seed)
        counts[category]["frame_count"] += 1
        counts[category]["candidate_seed_count"] += int(seed is not None)
        if seed is not None:
            combination = "+".join(
                sorted(str(value) for value in seed.get("trigger_reason", []))
            )
            production_reason_counts[combination] = (
                production_reason_counts.get(combination, 0) + 1
            )
    exclusive_rows = [
        {
            "schema_version": "temporal_candidate_seed_reason_breakdown.v1",
            "asset_id": asset_id,
            "timebase": timebase,
            "breakdown_kind": "exclusive_audit_category",
            "reason_category": category,
            "production_trigger_reason": None,
            "frame_count": values["frame_count"],
            "candidate_seed_count": values["candidate_seed_count"],
            "non_candidate_seed_count": (
                values["frame_count"] - values["candidate_seed_count"]
            ),
            "categories_are_mutually_exclusive": True,
            "angle_or_rotation_exceedance_is_not_automatically_a_seed": True,
        }
        for category, values in counts.items()
    ]
    production_rows = [
        {
            "schema_version": "temporal_candidate_seed_reason_breakdown.v1",
            "asset_id": asset_id,
            "timebase": timebase,
            "breakdown_kind": "production_trigger_reason_combination",
            "reason_category": "production_trigger_reason",
            "production_trigger_reason": combination,
            "frame_count": count,
            "candidate_seed_count": count,
            "non_candidate_seed_count": 0,
            "categories_are_mutually_exclusive": True,
            "angle_or_rotation_exceedance_is_not_automatically_a_seed": True,
        }
        for combination, count in sorted(production_reason_counts.items())
    ]
    return exclusive_rows + production_rows


def _merge_inclusive_intervals(
    intervals: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted((int(start), int(end)) for start, end in intervals):
        if end < start:
            continue
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _candidate_coverage_row(
    *,
    clip: ClipInputs,
    asset_id: str,
    timebase: str,
    check: SkeletonQualityScoreCheck,
    results: Sequence[CheckResult],
) -> dict[str, Any]:
    seeds = [
        seed
        for result in results
        if result.frame_idx >= 0
        for seed in (check.temporal_seed_record(result, asset_id),)
        if seed is not None
    ]
    eligible_count = sum(
        result.frame_idx >= 0
        and result.metrics.get("temporal_output_valid") is True
        and not bool(result.metrics.get("keypoint_presence_invalid", 0.0))
        for result in results
    )
    seed_runs = check.build_seed_runs(seeds)
    kept_runs = [
        run
        for run in seed_runs
        if len(run["seeds"]) >= check.candidate_min_seed_run_frames
    ]
    expanded = [check.expand_seed_run_window(clip, run) for run in kept_runs]
    windows = list(check.candidate_windows)
    union = _merge_inclusive_intervals(
        [(int(row["start_frame"]), int(row["end_frame"])) for row in windows]
    )
    union_count = sum(end - start + 1 for start, end in union)
    total_frame_count = int(clip.num_frames)
    fps = _numeric(clip.fps)
    largest_window_frame_count = max(
        (end - start + 1 for start, end in union),
        default=0,
    )
    duration_seconds = union_count / fps if fps is not None else None
    total_duration_seconds = (
        total_frame_count / fps
        if fps is not None and total_frame_count > 0
        else None
    )
    return {
        "schema_version": "temporal_candidate_window_coverage.v1",
        "asset_id": asset_id,
        "timebase": timebase,
        "source_fps": fps,
        "source_frame_count": total_frame_count,
        "candidate_seed_count": len(seeds),
        "candidate_seed_ratio": _safe_ratio(len(seeds), eligible_count),
        "eligible_frame_count": eligible_count,
        "candidate_window_count": len(windows),
        "premerge_window_count": len(expanded),
        "merged_window_count": len(windows),
        "window_merge_reduction_count": max(0, len(expanded) - len(windows)),
        "candidate_source_frame_union": [list(item) for item in union],
        "candidate_source_frame_union_interval_count": len(union),
        "candidate_source_frame_union_count": union_count,
        "candidate_source_frame_union_ratio": _safe_ratio(
            union_count,
            total_frame_count,
        ),
        "candidate_duration_seconds": duration_seconds,
        "candidate_duration_ratio": (
            duration_seconds / total_duration_seconds
            if duration_seconds is not None
            and total_duration_seconds is not None
            and total_duration_seconds > 0.0
            else None
        ),
        "largest_window_frame_count": largest_window_frame_count,
        "largest_window_duration_seconds": (
            largest_window_frame_count / fps if fps is not None else None
        ),
        "pre_context_frames": check.candidate_pre_context_frames,
        "pre_context_seconds": (
            check.candidate_pre_context_frames / fps if fps is not None else None
        ),
        "post_context_frames": check.candidate_post_context_frames,
        "post_context_seconds": (
            check.candidate_post_context_frames / fps if fps is not None else None
        ),
        "merge_gap_frames": 0,
        "merge_gap_seconds": 0.0 if fps is not None else None,
        "candidate_gap_close_frames": check.candidate_gap_close_frames,
        "candidate_gap_close_seconds": (
            check.candidate_gap_close_frames / fps if fps is not None else None
        ),
        "min_seed_run": check.candidate_min_seed_run_frames,
        "source_frame_interval_semantics": "inclusive",
        "standardized_anchor_coordinate_system": "source_frame",
    }


def _clip_source_frames(clip: ClipInputs) -> list[int]:
    source_indices = getattr(clip, "source_frame_indices", clip.frame_indices)
    fallback_start = int(getattr(clip, "clip_start_frame", 0))
    return [
        source_frame_at(
            source_indices,
            offset,
            fallback_start_frame=fallback_start,
        )
        for offset in range(clip.num_frames)
    ]


def _joint_identity(joint_name: str) -> tuple[str | None, int | None]:
    side = (
        "left"
        if joint_name.startswith("left")
        else "right"
        if joint_name.startswith("right")
        else None
    )
    base_name = joint_name[len(side) :] if side is not None else joint_name
    joint_index = next(
        (
            index
            for index, expected_name in EGODATA_HAND21_INDEX_TO_ACCEPTANCE_BASE.items()
            if expected_name == base_name
        ),
        None,
    )
    if joint_index is None and base_name.rsplit("_", 1)[-1].isdigit():
        joint_index = int(base_name.rsplit("_", 1)[-1])
    return side, joint_index


def _position_vector(values: np.ndarray, offset: int) -> list[float | None] | None:
    if offset < 0 or offset >= values.shape[0]:
        return None
    return [
        float(value) if math.isfinite(float(value)) else None
        for value in np.asarray(values[offset]).reshape(-1)[:3]
    ]


_EXTREME_RANKING_ORDER = (
    "position_abs_m",
    "native_displacement_m",
    "standardized_displacement_m",
    "native_acceleration_m_s2",
    "standardized_acceleration_m_s2",
    "nonfinite_coordinate",
)


def _select_extreme_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    top_n: int,
) -> list[dict[str, Any]]:
    """Select top records round-robin without comparing unlike physical units."""

    if top_n <= 0:
        return []
    grouped: dict[str, list[dict[str, Any]]] = {
        metric: [] for metric in _EXTREME_RANKING_ORDER
    }
    for raw in rows:
        row = dict(raw)
        metric = str(row.get("ranking_metric", ""))
        if metric in grouped:
            grouped[metric].append(row)
    for metric, records in grouped.items():
        records.sort(
            key=lambda row: (
                float(row.get("ranking_value") or 0.0),
                str(row.get("asset_id", "")),
                int(row.get("source_frame", -1)),
            ),
            reverse=True,
        )
    selected: list[dict[str, Any]] = []
    rank = 0
    while len(selected) < top_n:
        added = False
        for metric in _EXTREME_RANKING_ORDER:
            records = grouped[metric]
            if rank < len(records):
                selected.append(records[rank])
                added = True
                if len(selected) == top_n:
                    break
        if not added:
            break
        rank += 1
    return selected


def _joint_standardized_metrics(
    values: np.ndarray,
    *,
    evidence_offsets: Sequence[int],
    evidence_timestamps: Sequence[float],
) -> tuple[float | None, float | None]:
    displacement: float | None = None
    acceleration: float | None = None
    if len(evidence_offsets) >= 2 and len(evidence_timestamps) >= 2:
        previous, current = int(evidence_offsets[-2]), int(evidence_offsets[-1])
        dt = float(evidence_timestamps[-1]) - float(evidence_timestamps[-2])
        pair = np.asarray(values[[previous, current]], dtype=np.float64)
        if dt > 0.0 and np.all(np.isfinite(pair)):
            displacement = float(np.linalg.norm(pair[1] - pair[0]))
    if len(evidence_offsets) >= 3 and len(evidence_timestamps) >= 3:
        first, middle, current = (int(value) for value in evidence_offsets[-3:])
        t0, t1, t2 = (float(value) for value in evidence_timestamps[-3:])
        positions = np.asarray(values[[first, middle, current]], dtype=np.float64)
        dt0 = t1 - t0
        dt1 = t2 - t1
        velocity_midpoint_dt = (t2 - t0) / 2.0
        if (
            dt0 > 0.0
            and dt1 > 0.0
            and velocity_midpoint_dt > 0.0
            and np.all(np.isfinite(positions))
        ):
            velocity0 = (positions[1] - positions[0]) / dt0
            velocity1 = (positions[2] - positions[1]) / dt1
            acceleration = float(
                np.linalg.norm(velocity1 - velocity0) / velocity_midpoint_dt
            )
    return displacement, acceleration


def _extreme_records_for_clip(
    *,
    clip: ClipInputs,
    asset_id: str,
    temporal_rows: Sequence[CheckResult],
    top_n: int,
    finite_extreme_displacement_m: float | None,
    finite_extreme_acceleration_m_s2: float | None,
    finite_extreme_position_abs_m: float | None,
    supplier_acknowledged_issue_pattern: bool,
    score_rows: Sequence[CheckResult] = (),
) -> list[dict[str, Any]]:
    if top_n <= 0 or not clip.keypoints:
        return []
    source_frames = _clip_source_frames(clip)
    fps = _numeric(clip.fps)
    temporal_by_local = {
        int(row.metrics["local_frame_idx"]): row
        for row in temporal_rows
        if "local_frame_idx" in row.metrics
    }
    score_by_source = {
        int(row.frame_idx): row for row in score_rows if row.frame_idx >= 0
    }
    heaps: dict[str, list[tuple[float, int, dict[str, Any]]]] = {
        metric: [] for metric in _EXTREME_RANKING_ORDER
    }
    serial = 0
    for joint_name, raw_values in clip.keypoints.items():
        values = np.asarray(raw_values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] < 3:
            continue
        side, joint_index = _joint_identity(str(joint_name))
        frame_count = min(clip.num_frames, values.shape[0])
        for offset in range(frame_count):
            current = np.asarray(values[offset, :3], dtype=np.float64)
            finite_current = bool(np.all(np.isfinite(current)))
            position_abs = (
                float(np.max(np.abs(current))) if finite_current else None
            )
            native_displacement: float | None = None
            native_acceleration: float | None = None
            if offset > 0:
                pair = np.asarray(values[offset - 1 : offset + 1, :3])
                if np.all(np.isfinite(pair)):
                    native_displacement = float(np.linalg.norm(pair[1] - pair[0]))
            if offset > 1 and fps is not None:
                triple = np.asarray(values[offset - 2 : offset + 1, :3])
                if np.all(np.isfinite(triple)):
                    native_acceleration = float(
                        np.linalg.norm(triple[2] - 2.0 * triple[1] + triple[0])
                        * fps
                        * fps
                    )
            temporal_row = temporal_by_local.get(offset)
            evidence_offsets: list[int] = []
            evidence_source_frames: list[int] = []
            evidence_timestamps: list[float] = []
            if temporal_row is not None:
                evidence_offsets = [
                    int(value)
                    for value in temporal_row.metrics.get(
                        "evidence_local_frames",
                        [],
                    )
                ]
                evidence_source_frames = [
                    int(value)
                    for value in temporal_row.metrics.get(
                        "evidence_source_frames",
                        [],
                    )
                ]
                evidence_timestamps = [
                    float(value)
                    for value in temporal_row.metrics.get(
                        "evidence_timestamps",
                        [],
                    )
                ]
            standardized_displacement, standardized_acceleration = (
                _joint_standardized_metrics(
                    values,
                    evidence_offsets=evidence_offsets,
                    evidence_timestamps=evidence_timestamps,
                )
                if evidence_offsets
                else (None, None)
            )
            if not evidence_source_frames:
                start = max(0, offset - (2 if native_acceleration is not None else 1))
                evidence_source_frames = source_frames[start : offset + 1]
                if fps is not None:
                    evidence_timestamps = [
                        (frame - source_frames[0]) / fps
                        for frame in evidence_source_frames
                    ]
            score_row = score_by_source.get(int(source_frames[offset]))
            score_metrics = score_row.metrics if score_row is not None else {}
            presence_status: str | None = None
            presence_reason: str | None = None
            if "keypoint_presence_invalid" in score_metrics:
                presence_invalid = bool(
                    score_metrics.get("keypoint_presence_invalid")
                )
                presence_status = "invalid" if presence_invalid else "valid"
                presence_reason = (
                    "keypoint_presence_invalid"
                    if presence_invalid
                    else "keypoint_presence_valid"
                )
            projection_status: str | None = None
            if any(name in score_metrics for name in _PROJECTION_EVIDENCE_FIELDS):
                if bool(score_metrics.get("needs_out_of_frame_review", False)):
                    projection_status = "projection_out_of_frame_review"
                elif bool(score_metrics.get("needs_projection_review", False)):
                    projection_status = "projection_review"
                else:
                    projection_status = "projection_no_review"

            anomaly_sources: list[str] = []
            if (
                finite_extreme_position_abs_m is not None
                and position_abs is not None
                and position_abs >= finite_extreme_position_abs_m
            ):
                anomaly_sources.append("finite_extreme_position_abs_m")
            if finite_extreme_displacement_m is not None and any(
                value is not None and value >= finite_extreme_displacement_m
                for value in (native_displacement, standardized_displacement)
            ):
                anomaly_sources.append("finite_extreme_displacement_m")
            if finite_extreme_acceleration_m_s2 is not None and any(
                value is not None and value >= finite_extreme_acceleration_m_s2
                for value in (native_acceleration, standardized_acceleration)
            ):
                anomaly_sources.append("finite_extreme_acceleration_m_s2")
            if not finite_current:
                classification = "nonfinite_or_missing"
            elif anomaly_sources:
                classification = "finite_extreme_coordinate_anomaly_candidate"
            else:
                classification = "ordinary_temporal_exceedance"

            ranking_values = {
                "position_abs_m": position_abs,
                "native_displacement_m": native_displacement,
                "standardized_displacement_m": standardized_displacement,
                "native_acceleration_m_s2": native_acceleration,
                "standardized_acceleration_m_s2": standardized_acceleration,
            }
            finite_ranking: dict[str, float] = {
                name: value
                for name, value in ranking_values.items()
                if value is not None and math.isfinite(value)
            }
            record = {
                "schema_version": "temporal_extreme_coordinate_record.v1",
                "asset_id": asset_id,
                "source_frame": int(source_frames[offset]),
                "local_frame": offset,
                "side": side,
                "hand": side,
                "joint_index": joint_index,
                "joint_name": str(joint_name),
                "evidence_source_frames": evidence_source_frames,
                "evidence_timestamps": evidence_timestamps,
                "previous_position": _position_vector(values, offset - 1),
                "current_position": _position_vector(values, offset),
                "next_position": _position_vector(values, offset + 1),
                "position_abs_m": position_abs,
                "native_displacement_m": native_displacement,
                "standardized_displacement_m": standardized_displacement,
                "native_acceleration_m_s2": native_acceleration,
                "standardized_acceleration_m_s2": standardized_acceleration,
                "morphology_status": None,
                "morphology_reason": None,
                "presence_status": presence_status,
                "presence_reason": presence_reason,
                "projection_sam3_status": projection_status,
                "anomaly_classification": classification,
                "anomaly_source": (
                    anomaly_sources or ["raw_value_top_n_ranking"]
                ),
                "evidence_provenance": (
                    "raw_supplier_keypoints+read_only_temporal_recomputation"
                ),
                "supplier_acknowledged_issue_pattern": (
                    supplier_acknowledged_issue_pattern
                ),
                "frame_level_supplier_confirmed": False,
            }
            candidates = (
                {"nonfinite_coordinate": 0.0}
                if classification == "nonfinite_or_missing"
                else finite_ranking
            )
            for ranking_metric, ranking_value in candidates.items():
                ranked_record = {
                    **record,
                    "ranking_metric": ranking_metric,
                    "ranking_value": (
                        None
                        if ranking_metric == "nonfinite_coordinate"
                        else ranking_value
                    ),
                }
                item = (float(ranking_value), serial, ranked_record)
                serial += 1
                heap = heaps[ranking_metric]
                if len(heap) < top_n:
                    heapq.heappush(heap, item)
                elif item[:2] > heap[0][:2]:
                    heapq.heapreplace(heap, item)
    candidates = [
        item[2]
        for heap in heaps.values()
        for item in heap
    ]
    return _select_extreme_records(candidates, top_n=top_n)


def _write_json_records(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        json.dumps(list(rows), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_table(
    *,
    output_dir: Path,
    stem: str,
    rows: Sequence[Mapping[str, Any]],
    parquet: bool,
    columns: Sequence[str] | None = None,
) -> None:
    frame = pd.DataFrame.from_records(rows, columns=columns)
    frame.to_csv(output_dir / f"{stem}.csv", index=False)
    _write_json_records(output_dir / f"{stem}.json", rows)
    if parquet:
        frame.to_parquet(output_dir / f"{stem}.parquet", index=False)


def _path_identity(path: Path, *, content_hash: bool) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    identity: dict[str, Any] = {"resolved_path": str(resolved)}
    if not resolved.is_file():
        identity["missing"] = True
        return identity
    stat = resolved.stat()
    identity.update({"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
    if content_hash:
        identity["sha256"] = "sha256:" + hashlib.sha256(
            resolved.read_bytes()
        ).hexdigest()
    return identity


def _mapping_summary(mapping: Sequence[int], edge_count: int = 5) -> dict[str, Any]:
    frames = [int(value) for value in mapping]
    gaps = [right - left for left, right in zip(frames[:-1], frames[1:])]
    histogram: dict[str, int] = {}
    for gap in gaps:
        histogram[str(gap)] = histogram.get(str(gap), 0) + 1
    return {
        "sample_count": len(frames),
        "first_frames": frames[:edge_count],
        "last_frames": frames[-edge_count:] if frames else [],
        "gap_histogram": histogram,
        "duplicate_count": len(frames) - len(set(frames)),
        "monotonic": all(right > left for left, right in zip(frames[:-1], frames[1:])),
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
    values_by_field = {
        field: _finite_values(rows, field)
        for field in (*_NATIVE_FIELDS.values(), *_STANDARDIZED_FIELDS.values())
    }
    return _summarize_value_collections(
        values_by_field,
        native_eligible=native_eligible,
        native_exceeded=native_exceeded,
        native_seeds=native_seeds,
        standardized_eligible=standardized_eligible,
        standardized_exceeded=standardized_exceeded,
        standardized_seeds=standardized_seeds,
        standardized_sample_count=sum(
            row.metrics.get("standardized_sample_selected") is True
            for row in rows
        ),
    )


def _summarize_value_collections(
    values_by_field: Mapping[str, Sequence[float]],
    *,
    native_eligible: int,
    native_exceeded: int,
    native_seeds: int,
    standardized_eligible: int,
    standardized_exceeded: int,
    standardized_seeds: int,
    standardized_sample_count: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for timebase, fields in (
        ("native", _NATIVE_FIELDS),
        ("standardized", _STANDARDIZED_FIELDS),
    ):
        for metric, field in fields.items():
            summary.update(
                _percentiles(
                    values_by_field.get(field, ()),
                    f"{timebase}_{metric}",
                )
            )
    summary.update(
        {
            "native_eligible_frame_count": native_eligible,
            "standardized_eligible_frame_count": standardized_eligible,
            "native_threshold_exceed_count": native_exceeded,
            "standardized_threshold_exceed_count": standardized_exceeded,
            "native_threshold_exceed_rate": _safe_ratio(
                native_exceeded,
                native_eligible,
            ),
            "standardized_threshold_exceed_rate": _safe_ratio(
                standardized_exceeded,
                standardized_eligible,
            ),
            "native_candidate_seed_rate": _safe_ratio(
                native_seeds,
                native_eligible,
            ),
            "standardized_candidate_seed_rate": _safe_ratio(
                standardized_seeds,
                standardized_eligible,
            ),
            "native_candidate_seed_count": native_seeds,
            "standardized_candidate_seed_count": standardized_seeds,
            "standardized_sample_count": standardized_sample_count,
        }
    )
    return summary


def _supplier_source_reference(
    row: Mapping[str, Any],
    supplier: str,
) -> tuple[str, str | None]:
    fields = (
        ("parquet_path",)
        if supplier == "jdt"
        else ("hdf5_path", "source_path")
        if supplier == "xjgt"
        else ("hdf5_path",)
    )
    for field in fields:
        value = row.get(field)
        if value is None:
            continue
        if isinstance(value, (float, np.floating)) and not math.isfinite(
            float(value)
        ):
            continue
        text = str(value).strip()
        if text:
            return field, text
    return fields[0], None


def _resolve_supplier_path(
    raw_path: str | None,
    *,
    manifest: Path,
) -> Path | None:
    if raw_path is None:
        return None
    source_path = Path(raw_path).expanduser()
    if source_path.is_absolute():
        return source_path
    manifest_path = Path(os.path.abspath(Path(manifest).expanduser()))
    return Path(os.path.abspath(manifest_path.parent / source_path))


def _declared_manifest_frame_count(row: Mapping[str, Any]) -> int | None:
    try:
        start = int(row.get("start_frame"))
        end = int(row.get("end_frame"))
    except (TypeError, ValueError, OverflowError):
        return None
    return end - start + 1 if end >= start else None


def _default_clip_loader(
    supplier: str,
    manifest: Path,
) -> Callable[[dict[str, Any], int], ClipInputs]:
    cached_jdt_path: Path | None = None
    cached_jdt_frame: pd.DataFrame | None = None

    def load(row: dict[str, Any], row_index: int) -> ClipInputs:
        nonlocal cached_jdt_path, cached_jdt_frame
        source_field, raw_source_path = _supplier_source_reference(row, supplier)
        source_path = _resolve_supplier_path(
            raw_source_path,
            manifest=manifest,
        )
        if supplier in {"dr", "deepreach"}:
            resolved_row = dict(row)
            if source_path is not None:
                resolved_row[source_field] = str(source_path)
            return load_deepreach_clip(resolved_row, episode_idx=row_index)
        if supplier == "xjgt":
            if source_path is None:
                raise ValueError("xjgt manifest row requires hdf5_path or source_path")
            full_clip = load_supplier_hdf5_clip(
                source_path,
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
        if source_path is None:
            raise ValueError("jdt manifest row requires parquet_path")
        parquet_path = source_path
        if cached_jdt_path != parquet_path or cached_jdt_frame is None:
            loaded_jdt_frame = pd.read_parquet(parquet_path)
            cached_jdt_path = parquet_path
            cached_jdt_frame = loaded_jdt_frame
        resolved_row = dict(row)
        resolved_row[source_field] = str(parquet_path)
        return load_jdt_clip(
            resolved_row,
            episode_idx=row_index,
            source_frame=cached_jdt_frame,
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
    finite_extreme_displacement_m: float | None = None,
    finite_extreme_acceleration_m_s2: float | None = None,
    finite_extreme_position_abs_m: float | None = None,
    top_extreme_records: int = 100,
    audit_parameter_sources: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Audit temporal timebases without mutating any existing QC artifacts."""
    normalized_supplier = supplier.lower()
    if normalized_supplier not in {"jdt", "xjgt", "dr", "deepreach"}:
        raise ValueError("supplier must be jdt, xjgt, dr, or deepreach")
    if max_clips is not None and max_clips < 0:
        raise ValueError("max_clips must be >= 0")
    if top_extreme_records < 0:
        raise ValueError("top_extreme_records must be >= 0")
    for name, value in (
        ("finite_extreme_displacement_m", finite_extreme_displacement_m),
        ("finite_extreme_acceleration_m_s2", finite_extreme_acceleration_m_s2),
        ("finite_extreme_position_abs_m", finite_extreme_position_abs_m),
    ):
        if value is not None and (not math.isfinite(value) or value <= 0.0):
            raise ValueError(f"{name} must be finite and positive when provided")
    manifest_path = Path(manifest).expanduser()
    rows = read_manifest(manifest_path)
    selected_asset_ids = set(asset_ids or ())
    if selected_asset_ids:
        rows = [
            row for row in rows if str(row.get("asset_id")) in selected_asset_ids
        ]
    if max_clips is not None:
        rows = rows[:max_clips]
    declared_frame_counts = [
        _declared_manifest_frame_count(row) for row in rows
    ]
    declared_source_frame_universe_count = sum(
        value for value in declared_frame_counts if value is not None
    )
    undeclared_frame_range_row_count = sum(
        value is None for value in declared_frame_counts
    )

    loaded_config = load_qc_acceptance_config(config_path)
    parameters = loaded_config.module_parameters("keypoint_temporal")
    parameters.update(dict(temporal_config_overrides or {}))
    load_clip = clip_loader or _default_clip_loader(
        normalized_supplier,
        manifest_path,
    )
    output_rows: list[dict[str, Any]] = []
    breakdown_value_buffers: dict[tuple[str, str], list[np.ndarray]] = {
        (timebase, metric): []
        for timebase, fields in (
            ("native", _NATIVE_FIELDS),
            ("standardized", _STANDARDIZED_FIELDS),
        )
        for metric in fields
    }
    mapping_by_asset: dict[str, list[int]] = {}
    metric_breakdown_rows: list[dict[str, Any]] = []
    seed_breakdown_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    extreme_rows: list[dict[str, Any]] = []
    source_input_identities: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row_index, manifest_row in enumerate(rows):
        asset_id = str(manifest_row.get("asset_id") or f"row-{row_index}")
        source_field, raw_source_path = _supplier_source_reference(
            manifest_row,
            normalized_supplier,
        )
        attempted_source_path = _resolve_supplier_path(
            raw_source_path,
            manifest=manifest_path,
        )
        source_input_identities.append(
            {
                "asset_id": asset_id,
                "source_path_field": source_field,
                "raw_path": raw_source_path,
                "manifest_parent": str(
                    Path(os.path.abspath(manifest_path)).parent
                ),
                "resolved_identity": (
                    _path_identity(attempted_source_path, content_hash=False)
                    if attempted_source_path is not None
                    else None
                ),
            }
        )
        try:
            clip = load_clip(manifest_row, row_index)
            if getattr(clip, "asset_id", None) is None:
                setattr(clip, "asset_id", asset_id)
            temporal_rows = KeypointTemporalCheck(parameters).run(clip)
            for timebase, fields, eligibility_field in (
                ("native", _NATIVE_FIELDS, "temporal_pair_eligible"),
                (
                    "standardized",
                    _STANDARDIZED_FIELDS,
                    "standardized_temporal_pair_eligible",
                ),
            ):
                for metric, field in fields.items():
                    values = [
                        value
                        for result in temporal_rows
                        if result.metrics.get(eligibility_field) is True
                        for value in [_numeric(result.metrics.get(field))]
                        if value is not None
                    ]
                    if values:
                        breakdown_value_buffers[(timebase, metric)].append(
                            np.asarray(values, dtype=np.float64)
                        )
            asset_metric_breakdown = _metric_breakdown_rows(
                temporal_rows,
                asset_id=asset_id,
                parameters=parameters,
            )
            metric_breakdown_rows.extend(asset_metric_breakdown)
            score_by_timebase: dict[
                str,
                tuple[SkeletonQualityScoreCheck, list[CheckResult]],
            ] = {}
            for timebase in ("native", "standardized"):
                score_parameters = dict(parameters)
                score_parameters["temporal_decision_timebase"] = timebase
                score_check = SkeletonQualityScoreCheck(score_parameters)
                score_results = [
                    result
                    for result in score_check.run(clip)
                    if result.frame_idx >= 0
                ]
                score_by_timebase[timebase] = (score_check, score_results)
                seed_breakdown_rows.extend(
                    _seed_reason_breakdown_rows(
                        score_results,
                        check=score_check,
                        asset_id=asset_id,
                        timebase=timebase,
                    )
                )
                coverage_rows.append(
                    _candidate_coverage_row(
                        clip=clip,
                        asset_id=asset_id,
                        timebase=timebase,
                        check=score_check,
                        results=score_results,
                    )
                )
            clip_extreme_rows = _extreme_records_for_clip(
                    clip=clip,
                    asset_id=asset_id,
                    temporal_rows=temporal_rows,
                    score_rows=score_by_timebase["standardized"][1],
                    top_n=top_extreme_records,
                    finite_extreme_displacement_m=finite_extreme_displacement_m,
                    finite_extreme_acceleration_m_s2=(
                        finite_extreme_acceleration_m_s2
                    ),
                    finite_extreme_position_abs_m=finite_extreme_position_abs_m,
                    supplier_acknowledged_issue_pattern=(
                        normalized_supplier == "jdt"
                    ),
                )
            extreme_rows = _select_extreme_records(
                [*extreme_rows, *clip_extreme_rows],
                top_n=top_extreme_records,
            )
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
            asset_summary = {
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
            for timebase, (score_check, score_results) in score_by_timebase.items():
                seeds = sum(
                    score_check.temporal_seed_record(result, asset_id) is not None
                    for result in score_results
                )
                eligible = sum(
                    result.metrics.get("temporal_output_valid") is True
                    and not bool(
                        result.metrics.get("keypoint_presence_invalid", 0.0)
                    )
                    for result in score_results
                )
                asset_summary[f"{timebase}_candidate_seed_count"] = seeds
                asset_summary[f"{timebase}_candidate_seed_rate"] = _safe_ratio(
                    seeds,
                    eligible,
                )
            output_rows.append(asset_summary)
        except Exception as exc:
            failures.append(
                {
                    "asset_id": asset_id,
                    "source_path_field": source_field,
                    "raw_path": raw_source_path,
                    "manifest_parent": str(
                        Path(os.path.abspath(manifest_path)).parent
                    ),
                    "attempted_resolved_path": (
                        str(attempted_source_path)
                        if attempted_source_path is not None
                        else None
                    ),
                    "error": str(exc),
                }
            )

    native_eligible = sum(
        int(row.get("native_eligible_frame_count", 0)) for row in output_rows
    )
    standardized_eligible = sum(
        int(row.get("standardized_eligible_frame_count", 0))
        for row in output_rows
    )
    native_exceeded = sum(
        int(row.get("native_threshold_exceed_count", 0)) for row in output_rows
    )
    standardized_exceeded = sum(
        int(row.get("standardized_threshold_exceed_count", 0))
        for row in output_rows
    )
    overall_metric_values = {
        key: (
            np.concatenate(chunks)
            if chunks
            else np.asarray([], dtype=np.float64)
        )
        for key, chunks in breakdown_value_buffers.items()
    }
    overall = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "row_type": "overall",
        "supplier": normalized_supplier,
        "asset_id": "__overall__",
        "status": "completed" if not failures else "partial",
        **_summarize_value_collections(
            {
                field: overall_metric_values[(timebase, metric)]
                for timebase, fields in (
                    ("native", _NATIVE_FIELDS),
                    ("standardized", _STANDARDIZED_FIELDS),
                )
                for metric, field in fields.items()
            },
            native_eligible=native_eligible,
            native_exceeded=native_exceeded,
            native_seeds=sum(
                int(row.get("native_candidate_seed_count", 0))
                for row in output_rows
            ),
            standardized_eligible=standardized_eligible,
            standardized_exceeded=standardized_exceeded,
            standardized_seeds=sum(
                int(row.get("standardized_candidate_seed_count", 0))
                for row in output_rows
            ),
            standardized_sample_count=sum(
                int(row.get("standardized_sample_count", 0))
                for row in output_rows
            ),
        ),
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
    for timebase in ("native", "standardized"):
        seed_count = sum(
            int(row["candidate_seed_count"])
            for row in coverage_rows
            if row["timebase"] == timebase
        )
        eligible_count = sum(
            int(row["eligible_frame_count"])
            for row in coverage_rows
            if row["timebase"] == timebase
        )
        overall[f"{timebase}_candidate_seed_count"] = seed_count
        overall[f"{timebase}_candidate_seed_rate"] = _safe_ratio(
            seed_count,
            eligible_count,
        )
    output_rows.append(overall)
    overall_breakdown = _metric_breakdown_from_values(
        overall_metric_values,
        asset_id="__overall__",
        parameters=parameters,
    )
    metric_breakdown_rows.extend(overall_breakdown)
    extreme_rows = _select_extreme_records(
        extreme_rows,
        top_n=top_extreme_records,
    )
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
    _write_table(
        output_dir=output_dir,
        stem="temporal_metric_exceedance_breakdown",
        rows=metric_breakdown_rows,
        parquet=False,
    )
    _write_table(
        output_dir=output_dir,
        stem="temporal_candidate_seed_reason_breakdown",
        rows=seed_breakdown_rows,
        parquet=False,
    )
    _write_table(
        output_dir=output_dir,
        stem="temporal_candidate_window_coverage",
        rows=coverage_rows,
        parquet=True,
    )
    _write_table(
        output_dir=output_dir,
        stem="temporal_extreme_coordinate_records",
        rows=extreme_rows,
        parquet=True,
        columns=_EXTREME_COLUMNS,
    )
    _write_json_records(output_dir / "failures.json", failures)
    source_mapping_summaries = {
        asset_id: _mapping_summary(mapping)
        for asset_id, mapping in sorted(mapping_by_asset.items())
    }
    thresholds = {
        name: parameters[config_name]
        for name, config_name in _THRESHOLD_BY_LOGICAL_NAME.items()
    }
    selected_source_frame_universe_count = (
        max(
            declared_source_frame_universe_count,
            int(overall["source_frame_count"]),
        )
        if undeclared_frame_range_row_count
        else declared_source_frame_universe_count
    )
    candidate_config = {
        name: parameters.get(name)
        for name in (
            "candidate_gap_close_frames",
            "candidate_min_seed_run_frames",
            "candidate_pre_context_frames",
            "candidate_post_context_frames",
            "candidate_merge_overlapping_only",
        )
    }
    extreme_audit_parameters = {
        "finite_extreme_displacement_m": {
            "value": finite_extreme_displacement_m,
            "source": (audit_parameter_sources or {}).get(
                "finite_extreme_displacement_m",
                "cli" if finite_extreme_displacement_m is not None else "default_none",
            ),
            "enabled": finite_extreme_displacement_m is not None,
        },
        "finite_extreme_acceleration_m_s2": {
            "value": finite_extreme_acceleration_m_s2,
            "source": (audit_parameter_sources or {}).get(
                "finite_extreme_acceleration_m_s2",
                "cli" if finite_extreme_acceleration_m_s2 is not None else "default_none",
            ),
            "enabled": finite_extreme_acceleration_m_s2 is not None,
        },
        "finite_extreme_position_abs_m": {
            "value": finite_extreme_position_abs_m,
            "source": (audit_parameter_sources or {}).get(
                "finite_extreme_position_abs_m",
                "cli" if finite_extreme_position_abs_m is not None else "default_none",
            ),
            "enabled": finite_extreme_position_abs_m is not None,
        },
        "top_extreme_records": {
            "value": top_extreme_records,
            "source": (audit_parameter_sources or {}).get(
                "top_extreme_records",
                "caller_or_default",
            ),
            "enabled": top_extreme_records > 0,
        },
        "affects_acceptance_decisions": False,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "producer_version": AUDIT_PRODUCER_VERSION,
                "manifest": str(Path(manifest)),
                "input_identities": {
                    "manifest": _path_identity(manifest_path, content_hash=True),
                    "qc_config": _path_identity(
                        loaded_config.path,
                        content_hash=True,
                    ),
                    "supplier_sources": source_input_identities,
                },
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
                "native_metric_source": "native_source_fps",
                "standardized_metric_source": "standardized_30hz",
                "aggregation_memory_policy": {
                    "source_parquet_cache_entries": 1,
                    "retains_full_cross_clip_check_results": False,
                    "percentile_storage": "exact_scalar_metric_buffers",
                    "counts": "streaming_per_asset_accumulation",
                },
                "sam3_comparison_mode": "not_applicable",
                "frame_universe_definition": (
                    "selected_manifest_inclusive_source_frame_ranges"
                ),
                "thresholds": thresholds,
                "candidate_window_config": candidate_config,
                "finite_extreme_audit_parameters": extreme_audit_parameters,
                "schema_versions": {
                    "temporal_output": "keypoint_temporal.output.v3",
                    "ab_audit": AUDIT_SCHEMA_VERSION,
                    "metric_breakdown": "temporal_metric_exceedance_breakdown.v1",
                    "candidate_coverage": "temporal_candidate_window_coverage.v1",
                    "extreme_records": "temporal_extreme_coordinate_record.v1",
                },
                "source_frame_mapping_summary_by_asset": source_mapping_summaries,
                "evaluated_counts": {
                    "source_frame_universe_count": int(
                        selected_source_frame_universe_count
                    ),
                    "successfully_loaded_source_frame_count": int(
                        overall["source_frame_count"]
                    ),
                    "undeclared_frame_range_row_count": int(
                        undeclared_frame_range_row_count
                    ),
                    **{
                        f"{timebase}_{suffix}": value
                        for timebase in ("native", "standardized")
                        for suffix, value in (
                            (
                                "evaluated_frame_count",
                                sum(
                                    int(item["eligible_frame_count"])
                                    for item in coverage_rows
                                    if item["timebase"] == timebase
                                ),
                            ),
                            (
                                "unevaluable_frame_count",
                                max(
                                    0,
                                    int(selected_source_frame_universe_count)
                                    - sum(
                                        int(item["eligible_frame_count"])
                                        for item in coverage_rows
                                        if item["timebase"] == timebase
                                    ),
                                ),
                            ),
                        )
                    },
                },
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
        "source_frame_mapping_summary_by_asset": source_mapping_summaries,
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
    parser.add_argument("--finite-extreme-displacement-m", type=float)
    parser.add_argument("--finite-extreme-acceleration-m-s2", type=float)
    parser.add_argument("--finite-extreme-position-abs-m", type=float)
    parser.add_argument("--top-extreme-records", type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    top_extreme_records = (
        100 if args.top_extreme_records is None else args.top_extreme_records
    )
    summary = audit_temporal_timebase(
        manifest=args.manifest,
        supplier=args.supplier,
        output_dir=args.output_dir,
        asset_ids=args.asset_ids,
        max_clips=args.max_clips,
        config_path=args.config,
        finite_extreme_displacement_m=args.finite_extreme_displacement_m,
        finite_extreme_acceleration_m_s2=args.finite_extreme_acceleration_m_s2,
        finite_extreme_position_abs_m=args.finite_extreme_position_abs_m,
        top_extreme_records=top_extreme_records,
        audit_parameter_sources={
            "finite_extreme_displacement_m": (
                "cli"
                if args.finite_extreme_displacement_m is not None
                else "default_none"
            ),
            "finite_extreme_acceleration_m_s2": (
                "cli"
                if args.finite_extreme_acceleration_m_s2 is not None
                else "default_none"
            ),
            "finite_extreme_position_abs_m": (
                "cli"
                if args.finite_extreme_position_abs_m is not None
                else "default_none"
            ),
            "top_extreme_records": (
                "cli" if args.top_extreme_records is not None else "default"
            ),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["failed_asset_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
