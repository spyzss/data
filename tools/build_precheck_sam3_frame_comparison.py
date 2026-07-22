#!/usr/bin/env python3
"""Compare raw Precheck frame flags with candidate-independent SAM3 results."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from qc_common.config import load_qc_acceptance_config
from tools.run_manifest_sam3_containment import read_records


_PRIMARY_CHECKS = {
    "keypoint_missing": "keypoint_presence",
    "keypoint_presence": "keypoint_presence",
    "keypoint_morphology": "keypoint_morphology",
    "keypoint_temporal": "keypoint_temporal",
}
_FLAG_COLUMNS = (
    "keypoint_presence_flag",
    "keypoint_morphology_flag",
    "keypoint_temporal_flag",
    "projection_flag",
    "composite_skeleton_flag",
)
_PROJECTION_KEYS = (
    "needs_projection_review",
    "needs_out_of_frame_review",
    "left_needs_projection_review",
    "right_needs_projection_review",
    "left_needs_out_of_frame_review",
    "right_needs_out_of_frame_review",
)
_TEMPORAL_POSITIVE_DEFINITIONS = (
    "any_precheck_flag",
    "temporal_only_flag",
    "standardized_temporal_flag",
    "hard_existence_morphology_flag",
    "finite_extreme_coordinate_anomaly",
    "candidate_window_membership",
)


def _artifact_files(paths: Sequence[Path], *, artifact_name: str) -> list[Path]:
    files: list[Path] = []
    for value in paths:
        path = Path(value)
        if path.is_dir():
            files.extend(path.rglob(artifact_name))
        else:
            files.append(path)
    return sorted({path.resolve() for path in files}, key=str)


def read_precheck_artifacts(
    paths: Sequence[Path], *, artifact_name: str
) -> list[dict[str, Any]]:
    """Read explicit artifacts or roots and recover per-asset directory identity."""

    unique_files = _artifact_files(paths, artifact_name=artifact_name)
    if not unique_files:
        raise ValueError(f"no {artifact_name} artifacts found")
    rows: list[dict[str, Any]] = []
    for path in unique_files:
        inferred_asset_id = (
            path.parent.parent.name if path.parent.name == "precheck" else None
        )
        for raw in read_records(path):
            row = dict(raw)
            existing = row.get("asset_id")
            if inferred_asset_id is not None:
                if existing is not None and str(existing) != inferred_asset_id:
                    raise ValueError(
                        "asset_id disagrees with artifact path: "
                        f"{existing!r} != {inferred_asset_id!r} ({path})"
                    )
                row["asset_id"] = inferred_asset_id
            elif existing is None or not str(existing).strip():
                raise ValueError(
                    f"artifact row lacks asset_id and path has no precheck identity: {path}"
                )
            row.setdefault("source_artifact_path", str(path))
            rows.append(row)
    return rows


def _json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        return _json_list(decoded)
    if isinstance(value, Mapping):
        return [json.dumps(dict(value), sort_keys=True)]
    if isinstance(value, Iterable):
        return [str(item) for item in value if item is not None and str(item)]
    return [str(value)]


def _metrics(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, Mapping) else {}
    return {}


def _truthy(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1"}
    return False


def _source_frame(row: Mapping[str, Any]) -> int | None:
    value = row.get("source_frame")
    if value is None:
        value = row.get("source_frame_idx", row.get("frame_idx"))
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def _stable_json(values: Iterable[str]) -> str:
    return json.dumps(sorted(set(values)), separators=(",", ":"))


def _candidate_identity(row: Mapping[str, Any]) -> str:
    for key in ("review_id", "candidate_id", "window_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    asset_id = str(row.get("asset_id", ""))
    start = row.get("start_frame", row.get("window_start_frame"))
    end = row.get("end_frame", row.get("window_end_frame"))
    return f"{asset_id}:{start}:{end}"


def _candidate_bounds(row: Mapping[str, Any]) -> tuple[int, int] | None:
    start = row.get("start_frame", row.get("window_start_frame"))
    end = row.get("end_frame", row.get("window_end_frame"))
    try:
        return int(start), int(end)
    except (TypeError, ValueError, OverflowError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if math.isfinite(numeric) else None


def _manifest_frame_universe(
    manifest_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for row in manifest_rows:
        asset_id = str(row.get("asset_id", "")).strip()
        if not asset_id:
            raise ValueError("manifest row requires asset_id")
        start = row.get("start_frame")
        end = row.get("end_frame")
        try:
            start_frame = int(start)
            end_frame = int(end)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"manifest row requires inclusive start_frame/end_frame: {asset_id}"
            ) from exc
        if start_frame < 0 or end_frame < start_frame:
            raise ValueError(f"invalid manifest source-frame range: {asset_id}")
        for source_frame in range(start_frame, end_frame + 1):
            key = (asset_id, source_frame)
            if key in seen:
                raise ValueError(f"duplicate manifest frame key: {key}")
            seen.add(key)
            records.append(
                {
                    "asset_id": asset_id,
                    "source_frame": source_frame,
                    "manifest_start_frame": start_frame,
                    "manifest_end_frame": end_frame,
                }
            )
    return records


def _sam3_by_key(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        asset_id = str(row.get("asset_id", "")).strip()
        source_frame = _source_frame(row)
        if not asset_id or source_frame is None:
            raise ValueError("SAM3 frame row requires asset_id and source_frame")
        key = (asset_id, source_frame)
        if key in by_key:
            raise ValueError(f"duplicate SAM3 frame key: {key}")
        row["source_frame"] = source_frame
        by_key[key] = row
    return by_key


def _tri_or(*values: bool | None) -> bool | None:
    if any(value is True for value in values):
        return True
    if any(value is None for value in values):
        return None
    return False


def _candidate_membership_by_key(
    universe_keys: Sequence[tuple[str, int]],
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], bool]:
    by_asset: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        bounds = _candidate_bounds(row)
        if bounds is not None:
            by_asset[str(row.get("asset_id", ""))].append(bounds)
    return {
        key: any(start <= key[1] <= end for start, end in by_asset.get(key[0], ()))
        for key in universe_keys
    }


def _extreme_state(
    metrics: Mapping[str, Any],
    *,
    timebase: str,
    thresholds: Mapping[str, float | None],
) -> bool | None:
    configured = {
        name: _finite_float(thresholds.get(name))
        for name in (
            "finite_extreme_displacement_m",
            "finite_extreme_acceleration_m_s2",
            "finite_extreme_position_abs_m",
        )
    }
    configured = {name: value for name, value in configured.items() if value is not None}
    if not configured:
        return None
    metric_fields = {
        "finite_extreme_position_abs_m": "joint_position_abs_m_max",
        "finite_extreme_displacement_m": (
            "joint_displacement_standardized_m_max"
            if timebase == "standardized"
            else "joint_displacement_m_max"
        ),
        "finite_extreme_acceleration_m_s2": (
            "joint_acceleration_standardized_m_s2_max"
            if timebase == "standardized"
            else "joint_acceleration_m_s2_max"
        ),
    }
    missing_configured_metric = False
    for threshold_name, threshold in configured.items():
        value = _finite_float(metrics.get(metric_fields[threshold_name]))
        if value is None:
            missing_configured_metric = True
            continue
        if value >= threshold:
            return True
    return None if missing_configured_metric else False


def _standardized_temporal_state(
    metrics: Mapping[str, Any],
    *,
    temporal_thresholds: Mapping[str, float] | None,
) -> bool | None:
    if metrics.get("standardized_sample_selected") is not True:
        return None
    if metrics.get("standardized_temporal_pair_eligible") is not True:
        return None
    if not temporal_thresholds:
        return None
    fields = {
        "joint_acceleration_m_s2_max_threshold": (
            "joint_acceleration_standardized_m_s2_max"
        ),
        "joint_displacement_m_max_threshold": (
            "joint_displacement_standardized_m_max"
        ),
        "joint_angle_change_deg_max_threshold": (
            "joint_angle_change_standardized_deg_max"
        ),
        "rotation_delta_max_threshold": "rotation_delta_standardized_max",
    }
    observed = False
    for threshold_name, field in fields.items():
        threshold = _finite_float(temporal_thresholds.get(threshold_name))
        value = _finite_float(metrics.get(field))
        if threshold is None or value is None:
            continue
        observed = True
        if value > threshold:
            return True
    return False if observed else None


def _native_temporal_state(
    metrics: Mapping[str, Any],
    *,
    temporal_thresholds: Mapping[str, float] | None,
    legacy_flag: bool | None,
    hard_flag: bool | None,
) -> bool | None:
    """Recover native temporal state without reusing a presence-only score flag."""

    if metrics.get("temporal_output_valid") is not True:
        return None
    exceeded = metrics.get("which_thresholds_exceeded")
    if isinstance(exceeded, (list, tuple, set)):
        temporal_names = {
            "joint_acceleration_m_s2_max",
            "joint_displacement_m_max",
            "joint_angle_change_deg_max",
            "rotation_delta_max",
        }
        return bool(temporal_names.intersection(str(value) for value in exceeded))
    if temporal_thresholds:
        fields = {
            "joint_acceleration_m_s2_max_threshold": (
                "joint_acceleration_m_s2_max"
            ),
            "joint_displacement_m_max_threshold": "joint_displacement_m_max",
            "joint_angle_change_deg_max_threshold": (
                "joint_angle_change_deg_max"
            ),
            "rotation_delta_max_threshold": "rotation_delta_max",
        }
        observed = False
        for threshold_name, field in fields.items():
            threshold = _finite_float(temporal_thresholds.get(threshold_name))
            value = _finite_float(metrics.get(field))
            if threshold is None or value is None:
                continue
            observed = True
            if value > threshold:
                return True
        if observed:
            return False
    if bool(metrics.get("keypoint_presence_invalid", False)) or hard_flag is True:
        return None
    return legacy_flag


def _project_temporal_variant(
    *,
    universe_keys: Sequence[tuple[str, int]],
    check_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    candidate_inputs_provided: bool,
    timebase: str,
    thresholds: Mapping[str, float | None],
    temporal_thresholds: Mapping[str, float] | None,
) -> dict[tuple[str, int], dict[str, bool | None]]:
    states: dict[tuple[str, int], dict[str, bool | None]] = {
        key: {
            "hard_existence_morphology_flag": None,
            "temporal_only_flag": None,
            "standardized_temporal_flag": None,
            "finite_extreme_coordinate_anomaly": None,
            "projection_flag": None,
            "formal_precheck_flag": None,
        }
        for key in universe_keys
    }
    for row in check_rows:
        asset_id = str(row.get("asset_id", "")).strip()
        source_frame = _source_frame(row)
        key = (asset_id, source_frame) if source_frame is not None else None
        if key not in states:
            continue
        state = states[key]
        check = str(row.get("check", ""))
        flag = _optional_bool(row.get("flag"))
        metrics = _metrics(row.get("metrics"))
        if check in {"keypoint_missing", "keypoint_presence", "keypoint_morphology"}:
            if flag is not None:
                current = state["hard_existence_morphology_flag"]
                state["hard_existence_morphology_flag"] = (
                    flag if current is None else bool(current or flag)
                )
            continue
        if check == "keypoint_temporal":
            native_state = _native_temporal_state(
                metrics,
                temporal_thresholds=temporal_thresholds,
                legacy_flag=flag,
                hard_flag=state["hard_existence_morphology_flag"],
            )
            if timebase == "native" and native_state is not None:
                state["temporal_only_flag"] = native_state
            standardized_state = _standardized_temporal_state(
                metrics,
                temporal_thresholds=temporal_thresholds,
            )
            if standardized_state is not None:
                state["standardized_temporal_flag"] = standardized_state
            continue
        if check != "skeleton_quality_score":
            continue
        if flag is not None:
            current_formal = state["formal_precheck_flag"]
            state["formal_precheck_flag"] = (
                flag if current_formal is None else bool(current_formal or flag)
            )
        projection_values = [
            _optional_bool(metrics.get(name))
            for name in _PROJECTION_KEYS
            if name in metrics
        ]
        if projection_values:
            state["projection_flag"] = any(value is True for value in projection_values)
        temporal_valid = metrics.get("temporal_output_valid") is True
        sample_selected = metrics.get("standardized_sample_selected") is True
        decision_source = str(metrics.get("decision_metric_source", ""))
        timebase_matches = (
            decision_source == "standardized_30hz"
            if timebase == "standardized"
            else decision_source in {"", "native_source_fps"}
        )
        if temporal_valid and timebase_matches:
            if timebase == "native":
                native_state = _native_temporal_state(
                    metrics,
                    temporal_thresholds=temporal_thresholds,
                    legacy_flag=flag,
                    hard_flag=state["hard_existence_morphology_flag"],
                )
                if native_state is not None:
                    state["temporal_only_flag"] = native_state
            elif sample_selected:
                standardized_temporal_state = _standardized_temporal_state(
                    metrics,
                    temporal_thresholds=temporal_thresholds,
                )
                if (
                    standardized_temporal_state is None
                    and not bool(metrics.get("keypoint_presence_invalid", False))
                    and state["hard_existence_morphology_flag"] is not True
                ):
                    standardized_temporal_state = flag
                state["temporal_only_flag"] = standardized_temporal_state
                if standardized_temporal_state is not None:
                    state["standardized_temporal_flag"] = (
                        standardized_temporal_state
                    )
        standardized_state = _standardized_temporal_state(
            metrics,
            temporal_thresholds=temporal_thresholds,
        )
        if standardized_state is not None:
            state["standardized_temporal_flag"] = standardized_state
        extreme = _extreme_state(metrics, timebase=timebase, thresholds=thresholds)
        current_extreme = state["finite_extreme_coordinate_anomaly"]
        state["finite_extreme_coordinate_anomaly"] = (
            extreme
            if current_extreme is None
            else _tri_or(current_extreme, extreme)
        )

    candidate_membership = _candidate_membership_by_key(
        universe_keys,
        candidate_rows,
    )
    for key, state in states.items():
        state["candidate_window_membership"] = (
            candidate_membership[key] if candidate_inputs_provided else None
        )
        state["any_precheck_flag"] = _tri_or(
            state["hard_existence_morphology_flag"],
            state["formal_precheck_flag"],
            False if state["projection_flag"] is None else state["projection_flag"],
        )
    return states


def _sam3_proxy_label(verdict: Any, *, mode: str) -> str:
    normalized = str(verdict or "missing").strip().lower()
    if mode == "broad":
        if normalized in {"fail", "review"}:
            return "positive"
        if normalized == "pass":
            return "negative"
        return "unevaluable"
    if mode != "strict":
        raise ValueError("SAM3 proxy mode must be broad or strict")
    if normalized == "fail":
        return "positive"
    if normalized == "pass":
        return "negative"
    if normalized == "review":
        return "review"
    return "unevaluable"


def build_temporal_frame_comparison(
    *,
    manifest_rows: Sequence[Mapping[str, Any]],
    sam3_frame_rows: Sequence[Mapping[str, Any]],
    native_check_rows: Sequence[Mapping[str, Any]],
    standardized_check_rows: Sequence[Mapping[str, Any]],
    native_candidate_rows: Sequence[Mapping[str, Any]],
    standardized_candidate_rows: Sequence[Mapping[str, Any]],
    candidate_inputs_provided: bool,
    thresholds: Mapping[str, float | None],
    temporal_thresholds: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Project two Precheck variants onto one manifest-inclusive universe."""
    universe = _manifest_frame_universe(manifest_rows)
    universe_keys = [
        (str(row["asset_id"]), int(row["source_frame"])) for row in universe
    ]
    sam3 = _sam3_by_key(sam3_frame_rows)
    native = _project_temporal_variant(
        universe_keys=universe_keys,
        check_rows=native_check_rows,
        candidate_rows=native_candidate_rows,
        candidate_inputs_provided=candidate_inputs_provided,
        timebase="native",
        thresholds=thresholds,
        temporal_thresholds=temporal_thresholds,
    )
    standardized = _project_temporal_variant(
        universe_keys=universe_keys,
        check_rows=standardized_check_rows,
        candidate_rows=standardized_candidate_rows,
        candidate_inputs_provided=candidate_inputs_provided,
        timebase="standardized",
        thresholds=thresholds,
        temporal_thresholds=temporal_thresholds,
    )
    records: list[dict[str, Any]] = []
    for base, key in zip(universe, universe_keys):
        sam3_row = sam3.get(key, {})
        verdict = sam3_row.get("frame_verdict", "missing")
        record = {
            **base,
            **{
                f"sam3_{name}": value
                for name, value in sam3_row.items()
                if name not in {"asset_id", "source_frame"}
            },
            "sam3_frame_verdict": verdict,
            "sam3_proxy_broad": _sam3_proxy_label(verdict, mode="broad"),
            "sam3_proxy_strict": _sam3_proxy_label(verdict, mode="strict"),
        }
        for baseline, projected in (
            ("native", native[key]),
            ("standardized", standardized[key]),
        ):
            for definition in _TEMPORAL_POSITIVE_DEFINITIONS:
                value = projected.get(definition)
                record[f"{baseline}_{definition}"] = value
                record[f"{baseline}_{definition}_evaluable"] = value is not None
        records.append(record)
    return pd.DataFrame.from_records(records)


def _plain_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def confusion_metrics(
    *,
    visual_labels: Sequence[bool],
    precheck_labels: Sequence[bool],
    total_universe_count: int,
    review_count: int,
) -> dict[str, Any]:
    if len(visual_labels) != len(precheck_labels):
        raise ValueError("visual and Precheck labels must have equal length")
    tp = fn = fp = tn = 0
    for visual, precheck in zip(visual_labels, precheck_labels):
        if visual and precheck:
            tp += 1
        elif visual:
            fn += 1
        elif precheck:
            fp += 1
        else:
            tn += 1
    evaluated = tp + fn + fp + tn
    precision = _plain_ratio(tp, tp + fp)
    recall = _plain_ratio(tp, tp + fn)
    false_negative_rate = _plain_ratio(fn, tp + fn)
    false_positive_rate = _plain_ratio(fp, fp + tn)
    specificity = _plain_ratio(tn, fp + tn)
    negative_predictive_value = _plain_ratio(tn, tn + fn)
    balanced_accuracy = (
        (recall + specificity) / 2.0
        if recall is not None and specificity is not None
        else None
    )
    f1 = _plain_ratio(2 * tp, 2 * tp + fp + fn)
    mcc_denominator = math.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    return {
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "unevaluable_count": max(0, int(total_universe_count) - evaluated),
        "evaluated_frame_count": evaluated,
        "precheck_positive_rate": _plain_ratio(tp + fp, evaluated),
        "visual_positive_rate": _plain_ratio(tp + fn, evaluated),
        "precision": precision,
        "recall": recall,
        "false_negative_rate": false_negative_rate,
        "false_positive_rate": false_positive_rate,
        "specificity": specificity,
        "negative_predictive_value": negative_predictive_value,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "mcc": (
            float((tp * tn - fp * fn) / mcc_denominator)
            if mcc_denominator > 0.0
            else None
        ),
        "review_count": int(review_count),
        "review_rate": _plain_ratio(int(review_count), int(total_universe_count)),
    }


def _reference_summary_rows(
    frame: pd.DataFrame,
    *,
    reference_column: str,
    comparison_reference: str,
    is_final_ground_truth: bool,
    review_column: str | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    total = len(frame)
    for definition in _TEMPORAL_POSITIVE_DEFINITIONS:
        native_column = f"native_{definition}"
        standardized_column = f"standardized_{definition}"
        common_precheck = frame[native_column].notna() & frame[
            standardized_column
        ].notna()
        reference = frame[reference_column]
        visual_evaluable = reference.isin(["positive", "negative"])
        evaluated_mask = common_precheck & visual_evaluable
        review_values = (
            frame[review_column] if review_column is not None else reference
        )
        review_count = int(
            (
                common_precheck
                & review_values.astype(str).str.strip().str.lower().eq("review")
            ).sum()
        )
        for baseline, column in (
            ("native", native_column),
            ("standardized", standardized_column),
        ):
            selected = frame.loc[evaluated_mask]
            metrics = confusion_metrics(
                visual_labels=[
                    value == "positive"
                    for value in selected[reference_column].tolist()
                ],
                precheck_labels=[bool(value) for value in selected[column].tolist()],
                total_universe_count=total,
                review_count=review_count,
            )
            records.append(
                {
                    "schema_version": "temporal_precheck_confusion_summary.v1",
                    "comparison_reference": comparison_reference,
                    "baseline": baseline,
                    "precheck_positive_definition": definition,
                    "frame_universe_definition": (
                        "manifest_inclusive_source_frames_with_common_precheck_evaluability"
                    ),
                    "common_precheck_evaluable_count": int(common_precheck.sum()),
                    "is_final_acceptance_ground_truth": is_final_ground_truth,
                    **metrics,
                }
            )
    return records


def build_temporal_confusion_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = _reference_summary_rows(
        frame,
        reference_column="sam3_proxy_broad",
        comparison_reference="sam3_proxy_broad",
        is_final_ground_truth=False,
        review_column="sam3_frame_verdict",
    )
    records.extend(
        _reference_summary_rows(
            frame,
            reference_column="sam3_proxy_strict",
            comparison_reference="sam3_proxy_strict",
            is_final_ground_truth=False,
            review_column="sam3_frame_verdict",
        )
    )
    return records


def _manual_segments(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, Mapping)]
    if not isinstance(payload, Mapping):
        return []
    for key in ("segments", "records", "labels"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, Mapping)]
    return []


def _manual_label_class(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {
        "positive",
        "true_positive",
        "fail",
        "invalid",
        "abnormal",
    }:
        return "positive"
    if normalized in {
        "acceptable_flagged",
        "acceptable",
        "negative",
        "pass",
        "clean",
    }:
        return "negative"
    if normalized in {"review", "partial", "uncertain"}:
        return "review"
    return "unevaluable"


def _attach_manual_labels(frame: pd.DataFrame, payload: Any) -> pd.DataFrame:
    labeled = frame.copy()
    labels: dict[tuple[str, int], set[str]] = defaultdict(set)
    universe_keys = set(
        zip(
            labeled["asset_id"].astype(str),
            labeled["source_frame"].astype(int),
        )
    )
    for row in _manual_segments(payload):
        asset_id = str(row.get("asset_id", "")).strip()
        start = row.get("frame_start", row.get("start"))
        end = row.get("frame_end", row.get("end"))
        verdict = row.get("manual_verdict", row.get("label"))
        try:
            start_frame = int(start)
            end_frame = int(end)
        except (TypeError, ValueError, OverflowError):
            continue
        label = _manual_label_class(verdict)
        for source_frame in range(start_frame, end_frame + 1):
            key = (asset_id, source_frame)
            if key in universe_keys:
                labels[key].add(label)
    values: list[str] = []
    for asset_id, source_frame in zip(
        labeled["asset_id"].astype(str),
        labeled["source_frame"].astype(int),
    ):
        frame_labels = labels.get((asset_id, source_frame), set())
        values.append(next(iter(frame_labels)) if len(frame_labels) == 1 else "unevaluable")
    labeled["manual_ground_truth"] = values
    return labeled


def _attach_supplier_confirmed_intervals(
    frame: pd.DataFrame,
    payload: Any,
) -> pd.DataFrame:
    labeled = frame.copy()
    confirmed: set[tuple[str, int]] = set()
    universe_keys = set(
        zip(
            labeled["asset_id"].astype(str),
            labeled["source_frame"].astype(int),
        )
    )
    for row in _manual_segments(payload):
        asset_id = str(row.get("asset_id", "")).strip()
        start = row.get("frame_start", row.get("start"))
        end = row.get("frame_end", row.get("end"))
        try:
            start_frame = int(start)
            end_frame = int(end)
        except (TypeError, ValueError, OverflowError):
            continue
        for source_frame in range(start_frame, end_frame + 1):
            key = (asset_id, source_frame)
            if key in universe_keys:
                confirmed.add(key)
    labeled["supplier_confirmed_ground_truth"] = [
        "positive" if (asset_id, source_frame) in confirmed else "unevaluable"
        for asset_id, source_frame in zip(
            labeled["asset_id"].astype(str),
            labeled["source_frame"].astype(int),
        )
    ]
    labeled["supplier_acknowledged_issue_pattern"] = True
    labeled["frame_level_supplier_confirmed"] = [
        (asset_id, source_frame) in confirmed
        for asset_id, source_frame in zip(
            labeled["asset_id"].astype(str),
            labeled["source_frame"].astype(int),
        )
    ]
    return labeled


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_temporal_comparison_outputs(
    *,
    manifest_rows: Sequence[Mapping[str, Any]],
    sam3_frame_rows: Sequence[Mapping[str, Any]],
    native_check_rows: Sequence[Mapping[str, Any]],
    standardized_check_rows: Sequence[Mapping[str, Any]],
    native_candidate_rows: Sequence[Mapping[str, Any]],
    standardized_candidate_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    config_reference: Mapping[str, Any],
    thresholds: Mapping[str, float | None],
    manual_labels: Any | None,
    run_metadata: Mapping[str, Any],
    candidate_inputs_provided: bool = True,
    supplier_confirmed_intervals: Any | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"comparison output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = build_temporal_frame_comparison(
        manifest_rows=manifest_rows,
        sam3_frame_rows=sam3_frame_rows,
        native_check_rows=native_check_rows,
        standardized_check_rows=standardized_check_rows,
        native_candidate_rows=native_candidate_rows,
        standardized_candidate_rows=standardized_candidate_rows,
        candidate_inputs_provided=candidate_inputs_provided,
        thresholds=thresholds,
        temporal_thresholds=(
            dict(run_metadata.get("acceptance_thresholds", {})) or None
        ),
    )
    frame.to_csv(
        output_dir / "temporal_precheck_vs_sam3_frame_comparison.csv",
        index=False,
    )
    frame.to_parquet(
        output_dir / "temporal_precheck_vs_sam3_frame_comparison.parquet",
        index=False,
    )
    _json_dump(
        output_dir / "temporal_precheck_vs_sam3_frame_comparison.json",
        json.loads(frame.to_json(orient="records")),
    )
    summary = build_temporal_confusion_summary(frame)
    summary_frame = pd.DataFrame.from_records(summary)
    summary_frame.to_csv(
        output_dir / "temporal_precheck_vs_sam3_confusion_summary.csv",
        index=False,
    )
    summary_frame.to_parquet(
        output_dir / "temporal_precheck_vs_sam3_confusion_summary.parquet",
        index=False,
    )
    _json_dump(
        output_dir / "temporal_precheck_vs_sam3_confusion_summary.json",
        summary,
    )
    manual_summary: list[dict[str, Any]] = []
    if manual_labels is not None:
        manual_frame = _attach_manual_labels(frame, manual_labels)
        manual_frame.to_csv(
            output_dir / "temporal_precheck_vs_manual_frame_comparison.csv",
            index=False,
        )
        manual_frame.to_parquet(
            output_dir / "temporal_precheck_vs_manual_frame_comparison.parquet",
            index=False,
        )
        manual_summary = _reference_summary_rows(
            manual_frame,
            reference_column="manual_ground_truth",
            comparison_reference="manual_ground_truth",
            is_final_ground_truth=True,
        )
        pd.DataFrame.from_records(manual_summary).to_csv(
            output_dir / "temporal_precheck_vs_manual_confusion_summary.csv",
            index=False,
        )
        _json_dump(
            output_dir / "temporal_precheck_vs_manual_confusion_summary.json",
            manual_summary,
        )
    supplier_summary: list[dict[str, Any]] = []
    if supplier_confirmed_intervals is not None:
        supplier_frame = _attach_supplier_confirmed_intervals(
            frame,
            supplier_confirmed_intervals,
        )
        supplier_frame.to_csv(
            output_dir / "temporal_precheck_vs_supplier_confirmed_frame_comparison.csv",
            index=False,
        )
        supplier_frame.to_parquet(
            output_dir / "temporal_precheck_vs_supplier_confirmed_frame_comparison.parquet",
            index=False,
        )
        supplier_summary = _reference_summary_rows(
            supplier_frame,
            reference_column="supplier_confirmed_ground_truth",
            comparison_reference="supplier_confirmed_ground_truth",
            is_final_ground_truth=False,
        )
        pd.DataFrame.from_records(supplier_summary).to_csv(
            output_dir / "temporal_precheck_vs_supplier_confirmed_confusion_summary.csv",
            index=False,
        )
        _json_dump(
            output_dir / "temporal_precheck_vs_supplier_confirmed_confusion_summary.json",
            supplier_summary,
        )
    verdict_counts = frame["sam3_frame_verdict"].value_counts().to_dict()
    run_config = {
        "schema_version": "temporal_precheck_vs_sam3_run_config.v1",
        "producer_version": "temporal-precheck-vs-sam3-comparison-v2",
        "config_reference": dict(config_reference),
        "input_metadata": dict(run_metadata),
        "native_metric_source": "native_source_fps",
        "standardized_metric_source": "standardized_30hz",
        "temporal_target_hz": run_metadata.get("temporal_target_hz", 30.0),
        "sam3_comparison_modes": ["broad", "strict"],
        "frame_universe_definition": "manifest_inclusive_source_frames",
        "frame_universe_count": len(frame),
        "sam3_verdict_counts": {str(key): int(value) for key, value in verdict_counts.items()},
        "thresholds": dict(thresholds),
        "finite_extreme_audit_parameters": {
            name: {
                "value": value,
                "source": dict(
                    run_metadata.get("audit_only_threshold_sources", {})
                ).get(name, "caller_or_default"),
                "enabled": value is not None,
                "affects_acceptance_decisions": False,
            }
            for name, value in thresholds.items()
        },
        "acceptance_thresholds": dict(
            run_metadata.get("acceptance_thresholds", {})
        ),
        "candidate_window_config": dict(
            run_metadata.get("candidate_window_config", {})
        ),
        "candidate_window_inputs_provided": candidate_inputs_provided,
        "schema_versions": {
            "temporal_output": "keypoint_temporal.output.v3",
            "frame_comparison": "temporal_precheck_vs_sam3_frame_comparison.v1",
            "confusion_summary": "temporal_precheck_confusion_summary.v1",
        },
        "comparison_semantics": {
            "sam3_proxy_broad": "fail_or_review_positive; pass_negative",
            "sam3_proxy_strict": "fail_positive; pass_negative; review_separate",
            "blocked_error_missing": "unevaluable",
            "sam3_is_manual_ground_truth": False,
            "sam3_limitations": [
                "occlusion",
                "side_view",
                "mask_under_segmentation",
                "3d_coordinate_errors_may_be_detected_or_missed",
            ],
            "supplier_acknowledgement_is_pattern_evidence_not_frame_ground_truth": True,
        },
        "historical_sanity_reference_is_not_hardcoded": True,
        "reproduction_diagnostic_dimensions": [
            "frame_universe",
            "schema",
            "eligibility",
            "baseline_config",
            "cached_artifact_version",
            "flag_definition",
            "sam3_verdict_definition",
            "source_coordinate_mapping",
        ],
        "manual_ground_truth_output_written": manual_labels is not None,
        "supplier_confirmed_intervals_output_written": (
            supplier_confirmed_intervals is not None
        ),
        "comparison_counts": summary,
        "standardized_unsampled_source_frame_policy": (
            "unknown_not_clean_no_forward_fill"
        ),
        "source_coordinate_mapping_policy": (
            "source_frame_keys_joined_once_no_local_offset_conversion"
        ),
        "models_loaded": [],
        "mutates_precheck_outputs": False,
    }
    _json_dump(output_dir / "run_config.json", run_config)
    _json_dump(output_dir / "failures.json", [])
    return {
        "frame_universe_count": len(frame),
        "sam3_summary_row_count": len(summary),
        "manual_summary_row_count": len(manual_summary),
        "supplier_confirmed_summary_row_count": len(supplier_summary),
        "output_dir": str(output_dir),
    }


def project_precheck_frames(
    *,
    sam3_frame_rows: Sequence[Mapping[str, Any]],
    check_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Project raw check rows onto the exact exhaustive SAM3 frame universe."""

    universe: dict[tuple[str, int], dict[str, Any]] = {}
    for row in sam3_frame_rows:
        asset_id = str(row.get("asset_id", ""))
        frame = _source_frame(row)
        if not asset_id or frame is None:
            raise ValueError("SAM3 frame row requires asset_id and source_frame")
        key = (asset_id, frame)
        if key in universe:
            raise ValueError(f"duplicate SAM3 frame key: {key}")
        universe[key] = {
            "asset_id": asset_id,
            "source_frame": frame,
            **{column: False for column in _FLAG_COLUMNS},
            "precheck_modules": set(),
            "precheck_reason_codes": set(),
            "precheck_module_reasons": defaultdict(set),
            "candidate_window_membership": False,
            "candidate_review_ids": set(),
        }

    for check_row in check_rows:
        asset_id = str(check_row.get("asset_id", ""))
        frame = _source_frame(check_row)
        projected = universe.get((asset_id, frame)) if frame is not None else None
        if projected is None:
            continue
        check = str(check_row.get("check", ""))
        metrics = _metrics(check_row.get("metrics"))
        flag = _truthy(check_row.get("flag"))
        module = _PRIMARY_CHECKS.get(check)
        module_flag = False
        if module is not None and flag:
            projected[f"{module}_flag"] = True
            module_flag = True
        elif check == "skeleton_quality_score":
            projection_flag = any(_truthy(metrics.get(key)) for key in _PROJECTION_KEYS)
            if projection_flag:
                projected["projection_flag"] = True
                module_flag = True
            if flag:
                projected["composite_skeleton_flag"] = True
                module_flag = True
            module = "skeleton_quality_score"
        if not module_flag:
            continue
        projected["precheck_modules"].add(module)
        reason = check_row.get("reason")
        if reason is not None and str(reason).strip():
            projected["precheck_reason_codes"].add(str(reason))
            projected["precheck_module_reasons"][module].add(str(reason))

    candidates_by_asset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidate_rows:
        candidates_by_asset[str(candidate.get("asset_id", ""))].append(candidate)
    for (asset_id, frame), projected in universe.items():
        for candidate in candidates_by_asset.get(asset_id, ()):
            bounds = _candidate_bounds(candidate)
            if bounds is None:
                continue
            start, end = bounds
            if start <= frame <= end:
                projected["candidate_window_membership"] = True
                projected["candidate_review_ids"].add(_candidate_identity(candidate))

    records: list[dict[str, Any]] = []
    for key in sorted(universe):
        row = universe[key]
        row["precheck_problem_any"] = any(bool(row[column]) for column in _FLAG_COLUMNS)
        row["precheck_modules"] = _stable_json(row["precheck_modules"])
        row["precheck_reason_codes"] = _stable_json(row["precheck_reason_codes"])
        row["precheck_module_reasons"] = json.dumps(
            {
                module: sorted(reasons)
                for module, reasons in sorted(row["precheck_module_reasons"].items())
            },
            separators=(",", ":"),
        )
        row["candidate_review_ids"] = _stable_json(row["candidate_review_ids"])
        records.append(row)
    return pd.DataFrame.from_records(records)


def compare_frames(
    *,
    sam3_frame_rows: Sequence[Mapping[str, Any]],
    precheck_frame_rows: pd.DataFrame | Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Build an exact source-frame comparison with explicit unevaluable rows."""

    sam3 = pd.DataFrame.from_records(sam3_frame_rows)
    if sam3.empty:
        raise ValueError("SAM3 frame results are empty")
    if "source_frame" not in sam3.columns:
        raise ValueError("SAM3 frame results require source_frame")
    keys = ["asset_id", "source_frame"]
    if sam3.duplicated(keys).any():
        raise ValueError("SAM3 frame results contain duplicate asset/source-frame keys")
    precheck = (
        precheck_frame_rows.copy()
        if isinstance(precheck_frame_rows, pd.DataFrame)
        else pd.DataFrame.from_records(precheck_frame_rows)
    )
    if precheck.duplicated(keys).any():
        raise ValueError("Precheck projection contains duplicate asset/source-frame keys")
    joined = sam3.merge(precheck, on=keys, how="left", validate="one_to_one")
    for column in (*_FLAG_COLUMNS, "precheck_problem_any", "candidate_window_membership"):
        if column in joined:
            joined[column] = joined[column].fillna(False).astype(bool)
    for column in (
        "precheck_modules",
        "precheck_reason_codes",
        "candidate_review_ids",
    ):
        if column in joined:
            joined[column] = joined[column].fillna("[]")
    if "precheck_module_reasons" in joined:
        joined["precheck_module_reasons"] = joined[
            "precheck_module_reasons"
        ].fillna("{}")

    labels: list[str] = []
    classes: list[str] = []
    exclusions: list[str | None] = []
    for row in joined.to_dict(orient="records"):
        verdict = str(row.get("frame_verdict", "")).strip().lower()
        precheck_problem = bool(row.get("precheck_problem_any", False))
        if verdict in {"fail", "review"}:
            label = "problem"
            comparison = "TP" if precheck_problem else "FN"
            exclusion = None
        elif verdict == "pass":
            label = "clean"
            comparison = "FP" if precheck_problem else "TN"
            exclusion = None
        else:
            label = "unevaluable"
            comparison = "UNEVALUABLE"
            exclusion = f"sam3_{verdict or 'unknown'}"
        labels.append(label)
        classes.append(comparison)
        exclusions.append(exclusion)
    joined["sam3_binary_label"] = labels
    joined["comparison_class"] = classes
    joined["exclusion_reason"] = exclusions
    joined["evaluable"] = joined["comparison_class"] != "UNEVALUABLE"
    joined["sam3_frame_verdict"] = joined["frame_verdict"]
    joined["sam3_reason_codes"] = joined["frame_reason_codes"]
    return joined.sort_values(keys, kind="stable").reset_index(drop=True)


def _ratio(numerator: int, denominator: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "value": float(numerator / denominator) if denominator else None,
    }
    if not denominator:
        result["status"] = "not_applicable"
    return result


def build_comparison_summary(comparison: pd.DataFrame) -> dict[str, Any]:
    counts = comparison["comparison_class"].value_counts().to_dict()
    tp = int(counts.get("TP", 0))
    fn = int(counts.get("FN", 0))
    fp = int(counts.get("FP", 0))
    tn = int(counts.get("TN", 0))
    evaluable = tp + fn + fp + tn
    unevaluable = int(counts.get("UNEVALUABLE", 0))
    return {
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "total_evaluable_frames": evaluable,
        "total_unevaluable_frames": unevaluable,
        "detection_rate": _ratio(tp, tp + fn),
        "raw_frame_recall": _ratio(tp, tp + fn),
        "miss_rate": _ratio(fn, tp + fn),
        "precision": _ratio(tp, tp + fp),
        "false_discovery_rate": _ratio(fp, tp + fp),
        "false_positive_rate": _ratio(fp, fp + tn),
        "accuracy": _ratio(tp + tn, evaluable),
        "sam3_problem_frame_rate": _ratio(tp + fn, evaluable),
        "precheck_flagged_frame_rate": _ratio(tp + fp, evaluable),
        "evaluable_coverage": _ratio(evaluable, evaluable + unevaluable),
    }


def contiguous_intervals(rows: pd.DataFrame, *, class_name: str) -> pd.DataFrame:
    selected = rows.loc[rows["comparison_class"] == class_name, ["asset_id", "source_frame"]]
    records: list[dict[str, Any]] = []
    for asset_id, group in selected.groupby("asset_id", sort=True):
        frames = sorted({int(value) for value in group["source_frame"].tolist()})
        if not frames:
            continue
        start = previous = frames[0]
        for frame in frames[1:]:
            if frame != previous + 1:
                records.append(
                    {"asset_id": asset_id, "start_frame": start, "end_frame": previous, "frame_count": previous - start + 1}
                )
                start = frame
            previous = frame
        records.append(
            {"asset_id": asset_id, "start_frame": start, "end_frame": previous, "frame_count": previous - start + 1}
        )
    return pd.DataFrame.from_records(
        records, columns=["asset_id", "start_frame", "end_frame", "frame_count"]
    )


def _write_table_pair(frame: pd.DataFrame, output_dir: Path, stem: str) -> None:
    frame.to_parquet(output_dir / f"{stem}.parquet", index=False)
    frame.to_csv(output_dir / f"{stem}.csv", index=False)


def _per_asset(comparison: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for asset_id, group in comparison.groupby("asset_id", sort=True):
        summary = build_comparison_summary(group)
        verdicts = group["sam3_frame_verdict"].value_counts().to_dict()
        fn_frames = sorted(
            int(value)
            for value in group.loc[group["comparison_class"] == "FN", "source_frame"]
        )
        fn_intervals = contiguous_intervals(group, class_name="FN")
        candidate_problem = group[
            (group["sam3_binary_label"] == "problem")
            & group["candidate_window_membership"]
        ]
        sam3_problem_count = int((group["sam3_binary_label"] == "problem").sum())
        candidate_window_recall = _ratio(
            len(candidate_problem), sam3_problem_count
        )
        processing_column = next(
            (
                column
                for column in ("processing_seconds", "elapsed_seconds")
                if column in group.columns
            ),
            None,
        )
        records.append(
            {
                "asset_id": asset_id,
                "source_frame_count": len(group),
                "evaluable_count": int(group["evaluable"].sum()),
                "sam3_fail_count": int(verdicts.get("fail", 0)),
                "sam3_review_count": int(verdicts.get("review", 0)),
                "sam3_pass_count": int(verdicts.get("pass", 0)),
                "sam3_blocked_count": int(
                    len(group) - verdicts.get("fail", 0) - verdicts.get("review", 0) - verdicts.get("pass", 0)
                ),
                "TP": summary["TP"],
                "FN": summary["FN"],
                "FP": summary["FP"],
                "TN": summary["TN"],
                "recall": summary["detection_rate"]["value"],
                "recall_numerator": summary["detection_rate"]["numerator"],
                "recall_denominator": summary["detection_rate"]["denominator"],
                "miss_rate": summary["miss_rate"]["value"],
                "miss_rate_numerator": summary["miss_rate"]["numerator"],
                "miss_rate_denominator": summary["miss_rate"]["denominator"],
                "precision": summary["precision"]["value"],
                "precision_numerator": summary["precision"]["numerator"],
                "precision_denominator": summary["precision"]["denominator"],
                "false_positive_rate": summary["false_positive_rate"]["value"],
                "false_positive_rate_numerator": summary["false_positive_rate"][
                    "numerator"
                ],
                "false_positive_rate_denominator": summary["false_positive_rate"][
                    "denominator"
                ],
                "candidate_window_recall": candidate_window_recall["value"],
                "candidate_window_recall_numerator": candidate_window_recall[
                    "numerator"
                ],
                "candidate_window_recall_denominator": candidate_window_recall[
                    "denominator"
                ],
                "first_fn_frame": fn_frames[0] if fn_frames else None,
                "last_fn_frame": fn_frames[-1] if fn_frames else None,
                "fn_interval_count": len(fn_intervals),
                "processing_seconds": (
                    float(group[processing_column].fillna(0.0).sum())
                    if processing_column
                    else None
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _module_coverage(comparison: pd.DataFrame) -> pd.DataFrame:
    sam3_problem = comparison["sam3_binary_label"] == "problem"
    flags = comparison[list(_FLAG_COLUMNS)].astype(bool)
    flag_count = flags.sum(axis=1)
    records: list[dict[str, Any]] = []
    for column in _FLAG_COLUMNS:
        module_flag = flags[column]
        caught = int((module_flag & sam3_problem).sum())
        problem_count = int(sam3_problem.sum())
        recall = _ratio(caught, problem_count)
        false_positive = int((module_flag & (comparison["sam3_binary_label"] == "clean")).sum())
        records.append(
            {
                "module": column.removesuffix("_flag"),
                "flagged_frame_count": int(module_flag.sum()),
                "sam3_problem_caught_count": caught,
                "sam3_problem_frames_uniquely_caught": int(
                    (module_flag & sam3_problem & (flag_count == 1)).sum()
                ),
                "sam3_problem_frame_count": problem_count,
                "recall": recall["value"],
                "recall_numerator": recall["numerator"],
                "recall_denominator": recall["denominator"],
                "overlap_with_other_modules_count": int(
                    (module_flag & (flag_count > 1)).sum()
                ),
                "false_positive_count": false_positive,
            }
        )
    return pd.DataFrame.from_records(records)


def _reason_matrix(comparison: pd.DataFrame) -> pd.DataFrame:
    counts: dict[tuple[str, str, str, str, str], int] = defaultdict(int)
    for row in comparison.to_dict(orient="records"):
        raw_pairs = row.get("precheck_module_reasons")
        try:
            module_reasons = (
                json.loads(raw_pairs)
                if isinstance(raw_pairs, str) and raw_pairs.strip()
                else raw_pairs
            )
        except json.JSONDecodeError:
            module_reasons = None
        if not isinstance(module_reasons, Mapping) or not module_reasons:
            module_reasons = {"none": ["none"]}
        sam3_reasons = _json_list(row.get("frame_reason_codes")) or ["none"]
        for module, reasons in sorted(module_reasons.items()):
            for precheck_reason in _json_list(reasons) or ["none"]:
                for sam3_reason in sam3_reasons:
                    counts[
                        (
                            module,
                            precheck_reason,
                            str(row.get("sam3_frame_verdict", "")),
                            sam3_reason,
                            str(row.get("comparison_class", "")),
                        )
                    ] += 1
    return pd.DataFrame.from_records(
        [
            {
                "precheck_module": key[0],
                "precheck_reason": key[1],
                "sam3_verdict": key[2],
                "sam3_reason": key[3],
                "comparison_class": key[4],
                "frame_count": count,
            }
            for key, count in sorted(counts.items())
        ],
        columns=[
            "precheck_module",
            "precheck_reason",
            "sam3_verdict",
            "sam3_reason",
            "comparison_class",
            "frame_count",
        ],
    )


def _evidence_manifest(
    comparison: pd.DataFrame, *, require_positive_evidence: bool
) -> pd.DataFrame:
    positives = comparison.loc[comparison["sam3_binary_label"] == "problem"].copy()
    if "evidence_path" not in positives:
        positives["evidence_path"] = None
    missing = positives["evidence_path"].isna() | (positives["evidence_path"].astype(str).str.strip() == "")
    if require_positive_evidence and bool(missing.any()):
        missing_keys = positives.loc[missing, ["asset_id", "source_frame"]].to_dict(orient="records")
        raise ValueError(f"SAM3 problem frames missing evidence: {missing_keys}")
    positives["overlay_path"] = positives["evidence_path"]
    positives["source_video"] = positives.get("source_video_path")
    for column in (
        "supplier",
        "sam3_frame_verdict",
        "sam3_reason_codes",
        "precheck_modules",
        "precheck_reason_codes",
        "left_inside_ratio",
        "right_inside_ratio",
        "model_config_identity",
        "source_video",
    ):
        if column not in positives:
            positives[column] = None
    columns = [
        "asset_id",
        "supplier",
        "source_frame",
        "video_frame",
        "comparison_class",
        "sam3_frame_verdict",
        "sam3_reason_codes",
        "precheck_modules",
        "precheck_reason_codes",
        "source_video",
        "overlay_path",
        "left_inside_ratio",
        "right_inside_ratio",
        "model_config_identity",
    ]
    return positives.loc[~missing, columns].reset_index(drop=True)


def build_comparison_outputs(
    *,
    sam3_frame_rows: Sequence[Mapping[str, Any]],
    check_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    manifest_asset_ids: Sequence[str],
    model_run_lineage: Mapping[str, Any],
    require_positive_evidence: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"comparison output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    precheck = project_precheck_frames(
        sam3_frame_rows=sam3_frame_rows,
        check_rows=check_rows,
        candidate_rows=candidate_rows,
    )
    comparison = compare_frames(
        sam3_frame_rows=sam3_frame_rows,
        precheck_frame_rows=precheck,
    )
    evidence = _evidence_manifest(
        comparison, require_positive_evidence=require_positive_evidence
    )
    summary = build_comparison_summary(comparison)
    producer_summary = model_run_lineage.get("summary", {})
    if not isinstance(producer_summary, Mapping):
        producer_summary = {}
    compared_assets = int(comparison["asset_id"].nunique())
    summary.update(
        {
            "schema_version": "precheck_vs_sam3_summary.v1",
            "total_manifest_assets": len(set(manifest_asset_ids)),
            "compared_asset_count": compared_assets,
            "completed_assets": int(
                producer_summary.get("completed_assets", compared_assets)
            ),
            "failed_assets": int(
                producer_summary.get(
                    "failed_assets",
                    max(0, len(set(manifest_asset_ids)) - compared_assets),
                )
            ),
            "total_source_frames": len(comparison),
            "lineage": dict(model_run_lineage),
        }
    )
    verdict_counts = comparison["sam3_frame_verdict"].value_counts().to_dict()
    summary.update(
        {
            "sam3_fail_count": int(verdict_counts.get("fail", 0)),
            "sam3_review_count": int(verdict_counts.get("review", 0)),
            "sam3_pass_count": int(verdict_counts.get("pass", 0)),
            "sam3_blocked_count": int(
                len(comparison)
                - verdict_counts.get("fail", 0)
                - verdict_counts.get("review", 0)
                - verdict_counts.get("pass", 0)
            ),
            "sam3_error_count": int(
                comparison["sam3_reason_codes"]
                .astype(str)
                .str.contains("runtime_error|input_invalid|input_missing", regex=True)
                .sum()
            ),
            "sam3_unevaluable_verdict_counts": {
                str(key): int(value)
                for key, value in sorted(
                    comparison.loc[
                        comparison["comparison_class"] == "UNEVALUABLE",
                        "sam3_frame_verdict",
                    ]
                    .value_counts()
                    .to_dict()
                    .items()
                )
            },
        }
    )
    candidate_problem = comparison[
        (comparison["sam3_binary_label"] == "problem")
        & comparison["candidate_window_membership"]
    ]
    problem_count = int((comparison["sam3_binary_label"] == "problem").sum())
    summary["candidate_frame_recall"] = _ratio(len(candidate_problem), problem_count)

    _write_table_pair(precheck, output_dir, "precheck_frame_flags")
    _write_table_pair(comparison, output_dir, "precheck_vs_sam3_frame_comparison")
    for class_name, stem in (
        ("FN", "false_negative_frames"),
        ("FP", "false_positive_frames"),
        ("TP", "true_positive_frames"),
    ):
        _write_table_pair(
            comparison.loc[comparison["comparison_class"] == class_name].reset_index(drop=True),
            output_dir,
            stem,
        )
        contiguous_intervals(comparison, class_name=class_name).to_csv(
            output_dir / f"{stem.removesuffix('_frames')}_intervals.csv", index=False
        )
    sam3_problem_rows = comparison.loc[
        comparison["sam3_binary_label"] == "problem"
    ].copy()
    sam3_problem_rows["comparison_class"] = "SAM3_PROBLEM"
    contiguous_intervals(sam3_problem_rows, class_name="SAM3_PROBLEM").to_csv(
        output_dir / "sam3_problem_intervals.csv", index=False
    )
    for verdict, filename in (
        ("fail", "sam3_fail_intervals.csv"),
        ("review", "sam3_review_intervals.csv"),
    ):
        selected = comparison.loc[
            comparison["sam3_frame_verdict"] == verdict
        ].copy()
        selected["comparison_class"] = verdict.upper()
        contiguous_intervals(selected, class_name=verdict.upper()).to_csv(
            output_dir / filename, index=False
        )
    unevaluable = comparison.loc[~comparison["evaluable"]].copy()
    unevaluable["comparison_class"] = "UNEVALUABLE"
    contiguous_intervals(unevaluable, class_name="UNEVALUABLE").to_csv(
        output_dir / "unevaluable_intervals.csv", index=False
    )
    per_asset = _per_asset(comparison)
    per_asset.to_csv(output_dir / "precheck_vs_sam3_per_asset.csv", index=False)
    ranking = per_asset.sort_values(
        ["FN", "asset_id"], ascending=[False, True], kind="stable"
    ) if not per_asset.empty else per_asset
    ranking.to_csv(output_dir / "assets_ranked_by_false_negatives.csv", index=False)
    _module_coverage(comparison).to_csv(
        output_dir / "precheck_module_coverage.csv", index=False
    )
    _reason_matrix(comparison).to_csv(
        output_dir / "precheck_sam3_reason_matrix.csv", index=False
    )
    evidence.to_csv(output_dir / "review_evidence_manifest.csv", index=False)
    (output_dir / "precheck_vs_sam3_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare raw Precheck frame flags with exhaustive SAM3 results."
    )
    parser.add_argument("--sam3-frame-results", type=Path, required=True)
    parser.add_argument("--precheck-results", type=Path, action="append")
    parser.add_argument("--candidate-windows", type=Path, action="append")
    parser.add_argument("--native-precheck-results", type=Path, action="append")
    parser.add_argument(
        "--standardized-precheck-results",
        type=Path,
        action="append",
    )
    parser.add_argument("--native-candidate-windows", type=Path, action="append")
    parser.add_argument(
        "--standardized-candidate-windows",
        type=Path,
        action="append",
    )
    parser.add_argument("--manual-labels", type=Path)
    parser.add_argument("--supplier-confirmed-affected-intervals", type=Path)
    parser.add_argument("--asset-ids", nargs="+")
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--finite-extreme-displacement-m", type=float)
    parser.add_argument("--finite-extreme-acceleration-m-s2", type=float)
    parser.add_argument("--finite-extreme-position-abs-m", type=float)
    parser.add_argument(
        "--allow-config-hash-mismatch",
        action="store_true",
        help=(
            "allow an explicitly non-isolated historical comparison when "
            "producer config hashes differ from --config"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--producer-run-config", type=Path)
    parser.add_argument("--allow-missing-positive-evidence", action="store_true")
    return parser


def _selected_manifest_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    asset_ids: Sequence[str] | None,
    max_clips: int | None,
) -> list[dict[str, Any]]:
    if max_clips is not None and max_clips < 0:
        raise ValueError("max_clips must be >= 0")
    selected_ids = set(asset_ids or ())
    selected = [
        dict(row)
        for row in rows
        if not selected_ids or str(row.get("asset_id")) in selected_ids
    ]
    return selected if max_clips is None else selected[:max_clips]


def _cli_path_metadata(paths: Sequence[Path] | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_path in paths or ():
        path = Path(raw_path).expanduser().resolve()
        record: dict[str, Any] = {"resolved_path": str(path)}
        if path.is_file():
            stat = path.stat()
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            record.update(
                {
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "sha256": f"sha256:{digest.hexdigest()}",
                }
            )
        elif path.is_dir():
            record["kind"] = "artifact_root"
        else:
            record["missing"] = True
        records.append(record)
    return records


def _discovered_artifact_metadata(
    paths: Sequence[Path] | None,
    *,
    artifact_name: str,
) -> list[dict[str, Any]]:
    if not paths:
        return []
    return _cli_path_metadata(
        _artifact_files(paths, artifact_name=artifact_name)
    )


def _producer_config_hashes(payload: Mapping[str, Any]) -> list[str]:
    hashes: set[str] = set()
    direct = payload.get("config_hash")
    if isinstance(direct, str) and direct:
        hashes.add(direct)
    for key in ("config_reference", "qc_config"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            config_hash = value.get("config_hash")
            if isinstance(config_hash, str) and config_hash:
                hashes.add(config_hash)
    fingerprint = payload.get("fingerprint")
    if isinstance(fingerprint, Mapping):
        fingerprint_config = fingerprint.get("config")
        if isinstance(fingerprint_config, Mapping):
            config_hash = fingerprint_config.get("config_hash")
            if isinstance(config_hash, str) and config_hash:
                hashes.add(config_hash)
    return sorted(hashes)


def _producer_run_config_records(
    paths: Sequence[Path] | None,
    *,
    artifact_name: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    artifact_files = (
        _artifact_files(paths, artifact_name=artifact_name) if paths else []
    )
    for artifact_path in artifact_files:
        path = artifact_path.parent / "run_config.json"
        artifact_identity = _cli_path_metadata([artifact_path])[0]
        if not path.is_file():
            records.append(
                {
                    "consumed_artifact": artifact_identity,
                    "run_config": None,
                    "run_config_missing": True,
                    "decision_metric_source": None,
                    "temporal_output_schema_version": None,
                    "config_hashes": [],
                }
            )
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"producer run_config must be a mapping: {path}")
        temporal_sampling = payload.get("temporal_sampling")
        temporal_sampling = (
            temporal_sampling if isinstance(temporal_sampling, Mapping) else {}
        )
        fingerprint = payload.get("fingerprint")
        fingerprint = fingerprint if isinstance(fingerprint, Mapping) else {}
        records.append(
            {
                "consumed_artifact": artifact_identity,
                "run_config": _cli_path_metadata([path])[0],
                "run_config_missing": False,
                "decision_metric_source": (
                    payload.get("decision_metric_source")
                    or temporal_sampling.get("decision_metric_source")
                ),
                "temporal_output_schema_version": (
                    payload.get("temporal_output_schema_version")
                    or fingerprint.get("temporal_output_schema_version")
                    or temporal_sampling.get("schema_version")
                ),
                "config_hashes": _producer_config_hashes(payload),
            }
        )
    return records


def _validate_variant_timebase(
    rows: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    expected_source: str,
    producer_run_configs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    row_sources = sorted(
        {
            str(source)
            for row in rows
            if str(row.get("check", "")) == "skeleton_quality_score"
            for source in [_metrics(row.get("metrics")).get("decision_metric_source")]
            if source is not None and str(source)
        }
    )
    config_sources = sorted(
        {
            str(record.get("decision_metric_source"))
            for record in producer_run_configs
            if record.get("decision_metric_source")
        }
    )
    artifact_count = len(producer_run_configs)
    decision_lineage_count = sum(
        bool(record.get("decision_metric_source"))
        for record in producer_run_configs
    )
    observed = sorted(set(row_sources).union(config_sources))
    mismatched = [source for source in observed if source != expected_source]
    if mismatched:
        raise ValueError(
            f"{variant} artifacts declare {mismatched}; expected {expected_source}"
        )
    if observed and decision_lineage_count < artifact_count:
        validation_status = "verified_source_with_partial_artifact_lineage"
    elif observed:
        validation_status = "verified"
    else:
        validation_status = "unverified_missing_lineage"
    schema_versions = sorted(
        {
            str(record.get("temporal_output_schema_version"))
            for record in producer_run_configs
            if record.get("temporal_output_schema_version")
        }
    )
    if not schema_versions:
        schema_status = "unverified_missing_lineage"
    elif schema_versions == ["keypoint_temporal.output.v3"]:
        schema_lineage_count = sum(
            bool(record.get("temporal_output_schema_version"))
            for record in producer_run_configs
        )
        schema_status = (
            "verified"
            if schema_lineage_count == artifact_count
            else "partially_unverified_missing_lineage"
        )
    else:
        schema_status = "mismatch_reported_not_rewritten"
    return {
        "status": validation_status,
        "expected_decision_metric_source": expected_source,
        "observed_row_sources": row_sources,
        "observed_run_config_sources": config_sources,
        "consumed_artifact_count": artifact_count,
        "decision_lineage_artifact_count": decision_lineage_count,
        "decision_lineage_coverage": _plain_ratio(
            decision_lineage_count,
            artifact_count,
        ),
        "expected_temporal_output_schema_version": (
            "keypoint_temporal.output.v3"
        ),
        "observed_temporal_output_schema_versions": schema_versions,
        "temporal_output_schema_status": schema_status,
    }


def _config_compatibility(
    producer_run_configs: Sequence[Mapping[str, Any]],
    *,
    runtime_config_hash: str,
) -> dict[str, Any]:
    artifact_count = len(producer_run_configs)
    hash_lineage_count = sum(
        bool(record.get("config_hashes")) for record in producer_run_configs
    )
    observed = sorted(
        {
            str(value)
            for record in producer_run_configs
            for value in record.get("config_hashes", [])
        }
    )
    if not observed:
        status = "unverified_missing_config_hash"
    elif observed != [runtime_config_hash]:
        status = "mismatch_reported_not_rewritten"
    elif hash_lineage_count < artifact_count:
        status = "partially_unverified_missing_config_hash"
    else:
        status = "verified"
    return {
        "status": status,
        "runtime_config_hash": runtime_config_hash,
        "producer_config_hashes": observed,
        "consumed_artifact_count": artifact_count,
        "config_hash_artifact_count": hash_lineage_count,
        "artifact_lineage_coverage": _plain_ratio(
            hash_lineage_count,
            artifact_count,
        ),
    }


def _validate_config_compatibility(
    *,
    native_run_configs: Sequence[Mapping[str, Any]],
    standardized_run_configs: Sequence[Mapping[str, Any]],
    runtime_config_hash: str,
    allow_mismatch: bool,
) -> dict[str, Any]:
    result = {
        "native": _config_compatibility(
            native_run_configs,
            runtime_config_hash=runtime_config_hash,
        ),
        "standardized": _config_compatibility(
            standardized_run_configs,
            runtime_config_hash=runtime_config_hash,
        ),
        "mismatch_override_enabled": bool(allow_mismatch),
    }
    mismatched = [
        variant
        for variant in ("native", "standardized")
        if result[variant]["status"] == "mismatch_reported_not_rewritten"
    ]
    if mismatched and not allow_mismatch:
        details = {
            variant: result[variant]["producer_config_hashes"]
            for variant in mismatched
        }
        raise ValueError(
            "producer/runtime config hash mismatch for "
            f"{details}; pass --allow-config-hash-mismatch only for an "
            "intentional non-isolated historical comparison"
        )
    return result


def _read_manual_payload(path: Path | None) -> Any | None:
    if path is None:
        return None
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return read_records(path)


def _run_dual_cli(args: argparse.Namespace) -> int:
    if not args.native_precheck_results or not args.standardized_precheck_results:
        raise ValueError(
            "dual comparison requires both --native-precheck-results and "
            "--standardized-precheck-results"
        )
    if args.manifest is None:
        raise ValueError("dual comparison requires --manifest")
    native_candidates_provided = args.native_candidate_windows is not None
    standardized_candidates_provided = args.standardized_candidate_windows is not None
    if native_candidates_provided != standardized_candidates_provided:
        raise ValueError(
            "candidate comparison requires both native and standardized candidate inputs"
        )
    loaded_config = load_qc_acceptance_config(args.config)
    temporal_parameters = loaded_config.module_parameters("keypoint_temporal")
    manifest_rows = _selected_manifest_rows(
        read_records(args.manifest),
        asset_ids=args.asset_ids,
        max_clips=args.max_clips,
    )
    selected_asset_ids = {str(row.get("asset_id")) for row in manifest_rows}
    sam3_rows = [
        row
        for row in read_records(args.sam3_frame_results)
        if str(row.get("asset_id")) in selected_asset_ids
    ]
    native_rows = read_precheck_artifacts(
        args.native_precheck_results,
        artifact_name="check_results.json",
    )
    standardized_rows = read_precheck_artifacts(
        args.standardized_precheck_results,
        artifact_name="check_results.json",
    )
    native_run_configs = [
        *_producer_run_config_records(
            args.native_precheck_results,
            artifact_name="check_results.json",
        ),
        *_producer_run_config_records(
            args.native_candidate_windows,
            artifact_name="candidate_windows.json",
        ),
    ]
    standardized_run_configs = [
        *_producer_run_config_records(
            args.standardized_precheck_results,
            artifact_name="check_results.json",
        ),
        *_producer_run_config_records(
            args.standardized_candidate_windows,
            artifact_name="candidate_windows.json",
        ),
    ]
    variant_validation = {
        "native": _validate_variant_timebase(
            native_rows,
            variant="native",
            expected_source="native_source_fps",
            producer_run_configs=native_run_configs,
        ),
        "standardized": _validate_variant_timebase(
            standardized_rows,
            variant="standardized",
            expected_source="standardized_30hz",
            producer_run_configs=standardized_run_configs,
        ),
    }
    native_candidates = (
        read_precheck_artifacts(
            args.native_candidate_windows,
            artifact_name="candidate_windows.json",
        )
        if native_candidates_provided
        else []
    )
    standardized_candidates = (
        read_precheck_artifacts(
            args.standardized_candidate_windows,
            artifact_name="candidate_windows.json",
        )
        if standardized_candidates_provided
        else []
    )
    audit_thresholds = {
        "finite_extreme_displacement_m": args.finite_extreme_displacement_m,
        "finite_extreme_acceleration_m_s2": args.finite_extreme_acceleration_m_s2,
        "finite_extreme_position_abs_m": args.finite_extreme_position_abs_m,
    }
    for name, value in audit_thresholds.items():
        if value is not None and (not math.isfinite(value) or value <= 0.0):
            raise ValueError(f"{name} must be finite and > 0 when configured")
    candidate_config = {
        name: temporal_parameters.get(name)
        for name in (
            "candidate_gap_close_frames",
            "candidate_min_seed_run_frames",
            "candidate_pre_context_frames",
            "candidate_post_context_frames",
            "candidate_merge_overlapping_only",
        )
    }
    config_compatibility = _validate_config_compatibility(
        native_run_configs=native_run_configs,
        standardized_run_configs=standardized_run_configs,
        runtime_config_hash=loaded_config.sha256,
        allow_mismatch=args.allow_config_hash_mismatch,
    )
    run_metadata = {
        "input_paths": {
            "manifest": _cli_path_metadata([args.manifest]),
            "sam3_frame_results": _cli_path_metadata([args.sam3_frame_results]),
            "native_precheck_results": _cli_path_metadata(
                args.native_precheck_results
            ),
            "standardized_precheck_results": _cli_path_metadata(
                args.standardized_precheck_results
            ),
            "native_candidate_windows": _cli_path_metadata(
                args.native_candidate_windows
            ),
            "standardized_candidate_windows": _cli_path_metadata(
                args.standardized_candidate_windows
            ),
            "manual_labels": _cli_path_metadata(
                [args.manual_labels] if args.manual_labels else []
            ),
            "supplier_confirmed_affected_intervals": _cli_path_metadata(
                [args.supplier_confirmed_affected_intervals]
                if args.supplier_confirmed_affected_intervals
                else []
            ),
        },
        "consumed_artifacts": {
            "native_check_results": _discovered_artifact_metadata(
                args.native_precheck_results,
                artifact_name="check_results.json",
            ),
            "standardized_check_results": _discovered_artifact_metadata(
                args.standardized_precheck_results,
                artifact_name="check_results.json",
            ),
            "native_candidate_windows": _discovered_artifact_metadata(
                args.native_candidate_windows,
                artifact_name="candidate_windows.json",
            ),
            "standardized_candidate_windows": _discovered_artifact_metadata(
                args.standardized_candidate_windows,
                artifact_name="candidate_windows.json",
            ),
        },
        "producer_run_configs": {
            "native": native_run_configs,
            "standardized": standardized_run_configs,
        },
        "variant_validation": variant_validation,
        "config_compatibility": config_compatibility,
        "asset_ids": list(args.asset_ids or ()),
        "max_clips": args.max_clips,
        "temporal_target_hz": temporal_parameters["temporal_target_hz"],
        "acceptance_thresholds": {
            name: temporal_parameters[name]
            for name in (
                "joint_acceleration_m_s2_max_threshold",
                "joint_displacement_m_max_threshold",
                "joint_angle_change_deg_max_threshold",
                "rotation_delta_max_threshold",
            )
        },
        "candidate_window_config": candidate_config,
        "audit_only_threshold_sources": {
            name: "cli" if value is not None else "default_none"
            for name, value in audit_thresholds.items()
        },
    }
    summary = build_temporal_comparison_outputs(
        manifest_rows=manifest_rows,
        sam3_frame_rows=sam3_rows,
        native_check_rows=native_rows,
        standardized_check_rows=standardized_rows,
        native_candidate_rows=native_candidates,
        standardized_candidate_rows=standardized_candidates,
        output_dir=args.output_dir,
        config_reference=loaded_config.json_reference(),
        thresholds=audit_thresholds,
        manual_labels=_read_manual_payload(args.manual_labels),
        run_metadata=run_metadata,
        candidate_inputs_provided=native_candidates_provided,
        supplier_confirmed_intervals=_read_manual_payload(
            args.supplier_confirmed_affected_intervals
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dual_requested = bool(
        args.native_precheck_results or args.standardized_precheck_results
    )
    if dual_requested:
        try:
            return _run_dual_cli(args)
        except Exception as exc:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            failure_path = args.output_dir / "failures.json"
            if not failure_path.exists():
                _json_dump(
                    failure_path,
                    [{"error_type": type(exc).__name__, "error": str(exc)}],
                )
            print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
            return 1
    if not args.precheck_results:
        raise ValueError(
            "legacy comparison requires --precheck-results, or use both dual inputs"
        )
    sam3_rows = read_records(args.sam3_frame_results)
    check_rows = read_precheck_artifacts(
        args.precheck_results, artifact_name="check_results.json"
    )
    candidate_rows = (
        read_precheck_artifacts(
            args.candidate_windows, artifact_name="candidate_windows.json"
        )
        if args.candidate_windows
        else []
    )
    manifest_rows = read_records(args.manifest) if args.manifest else sam3_rows
    lineage = (
        json.loads(args.producer_run_config.read_text(encoding="utf-8"))
        if args.producer_run_config
        else {}
    )
    build_comparison_outputs(
        sam3_frame_rows=sam3_rows,
        check_rows=check_rows,
        candidate_rows=candidate_rows,
        output_dir=args.output_dir,
        manifest_asset_ids=tuple(str(row.get("asset_id", "")) for row in manifest_rows),
        model_run_lineage=lineage,
        require_positive_evidence=not args.allow_missing_positive_evidence,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
