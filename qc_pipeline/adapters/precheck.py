"""Translate legacy precheck outputs into unified QC module contracts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import fields, replace
from numbers import Real
from pathlib import Path
from typing import Any, NamedTuple, TypeVar, cast

from precheck.config import (
    CompositeFrameVerdictConfig,
    KeypointMissingConfig,
    KeypointMorphologyConfig,
    KeypointTemporalConfig,
    PrecheckConfig,
    QualityScoreConfig,
    SkeletonQualityScoreConfig,
    TextIntegrityConfig,
)
from qc_common.config import LoadedQcConfig
from qc_common.contracts import EvidenceRef, Issue, ModuleResult, Verdict, build_issue_id
from qc_common.frame_survival import FrameExclusion, temporal_transition_lineage
from qc_common.types import CheckResult


_VERDICT_ORDER = {"skipped": -1, "pass": 0, "warn": 1, "fail": 2}
_PRECHECK_NAME_BY_MODULE = {
    "hdf5_text_info": "text_integrity",
    "quality_hand": "quality_score",
    "keypoint_presence": "keypoint_missing",
    "keypoint_morphology": "keypoint_morphology",
    "keypoint_temporal": "keypoint_temporal",
    "text_integrity": "text_integrity",
    "quality_score": "quality_score",
    "keypoint_missing": "keypoint_missing",
    "skeleton_quality_score": "skeleton_quality_score",
    "composite_frame_verdict": "composite_frame_verdict",
}
_ConfigType = TypeVar("_ConfigType")

MORPHOLOGY_REASON_TO_RULE = {
    "palm_scale_too_small": "keypoint_morphology.palm_scale_too_small",
    "bone_length_ratio_spread": "keypoint_morphology.bone_length_ratio_spread",
    "max_normalized_bone_length": (
        "keypoint_morphology.max_normalized_bone_length"
    ),
    "zero_length_bone_count": "keypoint_morphology.zero_length_bone_count",
    "duplicate_joint_pair_count": (
        "keypoint_morphology.duplicate_joint_pair_count"
    ),
    "collapsed_finger_count": "keypoint_morphology.collapsed_finger_count",
    "joint_angle_min_deg": "keypoint_morphology.joint_angle_min_deg",
    "joint_angle_violation_fraction": (
        "keypoint_morphology.joint_angle_violation_fraction"
    ),
}

TEMPORAL_RULES = {
    "candidate": "keypoint_temporal.composite_frame_verdict",
    "threshold": "keypoint_temporal.skeleton_quality_score",
    "projection": "keypoint_temporal.projection_review",
    "strong": "keypoint_temporal.strong_temporal_failure",
    "no_valid": "keypoint_temporal.no_valid_output",
}

_CORE_TEMPORAL_METRICS = (
    "joint_angle_change_deg_max",
    "rotation_delta_max",
    "joint_acceleration_m_s2_max",
    "joint_displacement_m_max",
)
_STANDARDIZED_TEMPORAL_METRIC_BY_NATIVE = {
    "joint_angle_change_deg_max": "joint_angle_change_standardized_deg_max",
    "rotation_delta_max": "rotation_delta_standardized_max",
    "joint_acceleration_m_s2_max": "joint_acceleration_standardized_m_s2_max",
    "joint_displacement_m_max": "joint_displacement_standardized_m_max",
}

_MORPHOLOGY_OBSERVED_METRIC = {
    "palm_scale_too_small": "palm_scale_m",
    "bone_length_ratio_spread": "bone_length_ratio_spread",
    "max_normalized_bone_length": "normalized_bone_length_max",
    "zero_length_bone_count": "zero_length_bone_count",
    "duplicate_joint_pair_count": "duplicate_joint_pair_count",
    "collapsed_finger_count": "collapsed_finger_count",
    "joint_angle_min_deg": "joint_angle_min_deg",
    "joint_angle_violation_fraction": "joint_angle_violation_fraction",
}

_MORPHOLOGY_THRESHOLD_PARAMETER = {
    ("palm_scale_too_small", "fail"): "min_palm_scale_m",
    ("bone_length_ratio_spread", "review"): (
        "max_bone_length_ratio_spread_review"
    ),
    ("bone_length_ratio_spread", "fail"): "max_bone_length_ratio_spread_fail",
    ("max_normalized_bone_length", "review"): (
        "max_normalized_bone_length_review"
    ),
    ("max_normalized_bone_length", "fail"): (
        "max_normalized_bone_length_fail"
    ),
    ("zero_length_bone_count", "review"): (
        "max_zero_length_bone_count_review"
    ),
    ("zero_length_bone_count", "fail"): "max_zero_length_bone_count_fail",
    ("duplicate_joint_pair_count", "review"): (
        "max_duplicate_joint_pair_count_review"
    ),
    ("duplicate_joint_pair_count", "fail"): (
        "max_duplicate_joint_pair_count_fail"
    ),
    ("joint_angle_min_deg", "review"): "min_joint_angle_deg_review",
    ("joint_angle_min_deg", "fail"): "min_joint_angle_deg_fail",
    ("joint_angle_violation_fraction", "review"): (
        "max_joint_angle_violation_fraction_review"
    ),
    ("joint_angle_violation_fraction", "fail"): (
        "max_joint_angle_violation_fraction_fail"
    ),
}


class _NormalizedFailure(NamedTuple):
    side: str
    frame: int
    severity: Verdict
    rule_id: str
    metric: str
    observed: Any
    operator: str
    boundary: Any
    temporal_pair_start_frame: int | None = None
    temporal_pair_end_frame: int | None = None
    temporal_transition_attribution: str | None = None


_CompactedFailure = tuple[_NormalizedFailure, tuple[int, int]]


def _worst(*verdicts: Verdict) -> Verdict:
    if not verdicts:
        return "pass"
    return max(verdicts, key=_VERDICT_ORDER.__getitem__)


def _summary(results: Sequence[CheckResult], check: str) -> CheckResult:
    summaries = [
        row for row in results if row.check == check and row.frame_idx == -1
    ]
    if len(summaries) != 1:
        raise ValueError(
            f"expected one {check} summary row, found {len(summaries)}"
        )
    return summaries[0]


def _rule(
    config: LoadedQcConfig,
    module: str,
    name: str,
) -> Mapping[str, Any]:
    rule = config.module_rules(module).get(name)
    if not isinstance(rule, Mapping):
        raise ValueError(f"missing configured rule: {module}.{name}")
    rule_id = rule.get("rule_id")
    if not isinstance(rule_id, str) or not rule_id:
        raise ValueError(f"configured rule has no rule_id: {module}.{name}")
    return rule


def _rule_verdict(rule: Mapping[str, Any]) -> Verdict:
    verdict = rule.get("verdict")
    if verdict not in _VERDICT_ORDER:
        raise ValueError(f"configured rule has invalid verdict: {verdict!r}")
    return cast(Verdict, verdict)


def _json_safe_observed(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, Real):
        numeric = float(value)
        if math.isfinite(numeric):
            return int(value) if isinstance(value, int) else numeric
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe_observed(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_json_safe_observed(item) for item in value]
    if isinstance(value, list):
        return [_json_safe_observed(item) for item in value]
    return str(value)


def _issue_from_row(
    *,
    asset_id: str,
    module: str,
    rule_id: str,
    row: CheckResult,
    source_relative_path: str,
    severity: str,
    needs_manual_review: bool,
    metric: str,
    observed_value: Any,
    operator: str,
    boundary_value: Any,
    hand_side: str | None = None,
    frame_range: tuple[int | None, int | None] | None = None,
    evidence_kind: str = "check_result",
) -> Issue:
    if severity not in {"warn", "fail"}:
        raise ValueError(f"issue severity must be warn or fail, got {severity!r}")
    if frame_range is None:
        start_frame = row.frame_idx if row.frame_idx >= 0 else None
        end_frame = start_frame
    else:
        start_frame, end_frame = frame_range
    issue_type = rule_id.rsplit(".", 1)[-1]
    coordinate_system = "source_inclusive"
    return Issue(
        issue_id=build_issue_id(
            asset_id=asset_id,
            module=module,
            rule_id=rule_id,
            source_relative_path=source_relative_path,
            coordinate_system=coordinate_system,
            start_frame=start_frame,
            end_frame=end_frame,
            hand_side=hand_side,
            evidence_kind=evidence_kind,
        ),
        code=issue_type,
        severity=cast(Any, severity),
        module=module,
        issue_type=issue_type,
        metric=metric,
        observed_value=_json_safe_observed(observed_value),
        operator=operator,
        boundary_value=_json_safe_observed(boundary_value),
        rule_id=rule_id,
        needs_manual_review=needs_manual_review,
        context={
            "coordinate_system": coordinate_system,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "hand_side": hand_side,
        },
    )


def adapt_hdf5_text_info(
    *,
    asset_id: str,
    source_relative_path: str,
    results: Sequence[CheckResult],
    config: LoadedQcConfig,
) -> ModuleResult:
    """Adapt the legacy text_integrity summary without rereading HDF5."""
    summary = _summary(results, "text_integrity")
    missing = int(summary.metrics.get("missing_field_count", 0))
    empty = int(summary.metrics.get("empty_field_count", 0))
    invalid_text = (
        "not valid JSON" in summary.reason or "no text_label" in summary.reason
    )
    issue: Issue | None = None
    if missing or empty:
        configured_rule = _rule(config, "hdf5_text_info", "missing_required_field")
        verdict = _rule_verdict(configured_rule)
        metric = "missing_field_count" if missing else "empty_field_count"
        observed = missing if missing else empty
        issue = _issue_from_row(
            asset_id=asset_id,
            module="hdf5_text_info",
            rule_id=str(configured_rule["rule_id"]),
            row=summary,
            source_relative_path=source_relative_path,
            severity=verdict,
            needs_manual_review=False,
            metric=metric,
            observed_value=observed,
            operator=">",
            boundary_value=0,
        )
    elif invalid_text or summary.flag is True:
        configured_rule = _rule(config, "hdf5_text_info", "missing_text_field")
        verdict = _rule_verdict(configured_rule)
        issue = _issue_from_row(
            asset_id=asset_id,
            module="hdf5_text_info",
            rule_id=str(configured_rule["rule_id"]),
            row=summary,
            source_relative_path=source_relative_path,
            severity=verdict,
            needs_manual_review=False,
            metric="text_label",
            observed_value=summary.reason,
            operator="valid_json",
            boundary_value=True,
        )
    else:
        verdict = "pass"

    return ModuleResult(
        module="hdf5_text_info",
        verdict=verdict,
        evaluation={"decision": verdict, "reason": summary.reason},
        metrics=dict(summary.metrics),
        issues=() if issue is None else (issue,),
    )


def _quality_issue(
    *,
    asset_id: str,
    source_relative_path: str,
    row: CheckResult,
    rule: Mapping[str, Any],
    needs_manual_review: bool,
    metric: str,
    observed_value: Any,
    operator: str,
    boundary_value: Any,
    hand_side: str | None,
) -> Issue:
    verdict = _rule_verdict(rule)
    return _issue_from_row(
        asset_id=asset_id,
        module="quality_hand",
        rule_id=str(rule["rule_id"]),
        row=row,
        source_relative_path=source_relative_path,
        severity=verdict,
        needs_manual_review=needs_manual_review,
        metric=metric,
        observed_value=observed_value,
        operator=operator,
        boundary_value=boundary_value,
        hand_side=hand_side,
    )


def _is_configured_value(value: Any, valid_values: Sequence[Any]) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    numeric = float(value)
    return math.isfinite(numeric) and any(numeric == float(item) for item in valid_values)


def adapt_quality_hand(
    *,
    asset_id: str,
    source_relative_path: str,
    results: Sequence[CheckResult],
    config: LoadedQcConfig,
) -> ModuleResult:
    """Adapt quality_score rows using the configured per-hand semantics."""
    rows = [row for row in results if row.check == "quality_score"]
    frame_rows = sorted(
        (row for row in rows if row.frame_idx >= 0),
        key=lambda row: row.frame_idx,
    )
    summary_rows = [row for row in rows if row.frame_idx == -1]
    summary = summary_rows[0] if len(summary_rows) == 1 else None
    metrics = dict(summary.metrics) if summary is not None else {}
    if not frame_rows:
        return ModuleResult(
            module="quality_hand",
            verdict="skipped",
            evaluation={
                "decision": "skipped",
                "reason": "source_signal_not_provided",
            },
            metrics=metrics,
        )

    supplier_enum_rows = [
        row
        for row in frame_rows
        if "supplier_status_left" in row.metrics
        or "supplier_status_right" in row.metrics
    ]
    if supplier_enum_rows:
        if len(supplier_enum_rows) != len(frame_rows):
            raise ValueError("mixed numeric and enum supplier hand quality rows")
        return ModuleResult(
            module="quality_hand",
            verdict="pass",
            evaluation={
                "decision": "pass",
                "reason": "optional_supplier_evidence_validated",
            },
            metrics=metrics,
        )

    parameters = config.module_parameters("quality_hand")
    legacy = parameters.get("legacy_numeric_contract", parameters)
    if not isinstance(legacy, Mapping):
        raise ValueError("quality_hand legacy_numeric_contract must be a mapping")
    expected_shape = list(legacy["expected_shape"])
    valid_values = list(legacy["valid_values"])
    low_value = legacy["low_quality_hand_value"]
    issues: list[Issue] = []
    verdicts: list[Verdict] = ["pass"]

    for row in frame_rows:
        present = [name for name in ("quality_left", "quality_right") if name in row.metrics]
        if len(present) != 2 or expected_shape != [2]:
            rule = _rule(config, "quality_hand", "invalid_quality_hand_shape")
            issues.append(
                _quality_issue(
                    asset_id=asset_id,
                    source_relative_path=source_relative_path,
                    row=row,
                    rule=rule,
                    needs_manual_review=False,
                    metric="quality_hand_shape",
                    observed_value=len(present),
                    operator="!=",
                    boundary_value=expected_shape[0] if expected_shape else None,
                    hand_side=None,
                )
            )
            verdicts.append(_rule_verdict(rule))
            continue

        left = row.metrics["quality_left"]
        right = row.metrics["quality_right"]
        invalid_sides = [
            side
            for side, value in (("left", left), ("right", right))
            if not _is_configured_value(value, valid_values)
        ]
        if invalid_sides:
            rule = _rule(config, "quality_hand", "invalid_quality_hand_value")
            for side in invalid_sides:
                observed = left if side == "left" else right
                issues.append(
                    _quality_issue(
                        asset_id=asset_id,
                        source_relative_path=source_relative_path,
                        row=row,
                        rule=rule,
                        needs_manual_review=False,
                        metric=f"quality_{side}",
                        observed_value=observed,
                        operator="not in",
                        boundary_value=valid_values,
                        hand_side=side,
                    )
                )
            verdicts.append(_rule_verdict(rule))
            continue

        low_sides = [
            side
            for side, value in (("left", left), ("right", right))
            if float(value) == float(low_value)
        ]
        if len(low_sides) == 2:
            rule = _rule(config, "quality_hand", "low_quality_both_hands")
            issues.append(
                _quality_issue(
                    asset_id=asset_id,
                    source_relative_path=source_relative_path,
                    row=row,
                    rule=rule,
                    needs_manual_review=False,
                    metric="quality_hand",
                    observed_value=[left, right],
                    operator="==",
                    boundary_value=[low_value, low_value],
                    hand_side="both",
                )
            )
            verdicts.append(_rule_verdict(rule))
        elif low_sides:
            rule = _rule(config, "quality_hand", "low_quality_single_hand")
            side = low_sides[0]
            issues.append(
                _quality_issue(
                    asset_id=asset_id,
                    source_relative_path=source_relative_path,
                    row=row,
                    rule=rule,
                    needs_manual_review=True,
                    metric=f"quality_{side}",
                    observed_value=left if side == "left" else right,
                    operator="==",
                    boundary_value=low_value,
                    hand_side=side,
                )
            )
            verdicts.append(_rule_verdict(rule))

    verdict = _worst(*verdicts)
    reason = (
        next((row.reason for row in frame_rows if row.reason), "")
        if issues
        else (summary.reason if summary is not None else frame_rows[0].reason)
    )
    return ModuleResult(
        module="quality_hand",
        verdict=verdict,
        evaluation={"decision": verdict, "reason": reason},
        metrics=metrics,
        issues=tuple(issues),
    )


def _contiguous_ranges(frames: Sequence[int]) -> tuple[tuple[int, int], ...]:
    ranges: list[list[int]] = []
    for frame in sorted(set(int(value) for value in frames)):
        if not ranges or frame > ranges[-1][1] + 1:
            ranges.append([frame, frame])
        else:
            ranges[-1][1] = frame
    return tuple((start, end) for start, end in ranges)


def _compact_failures(
    failures: Sequence[_NormalizedFailure],
) -> tuple[_CompactedFailure, ...]:
    """Compact adjacent normalized failures with identical issue semantics."""
    def canonical(value: Any) -> str:
        return json.dumps(
            _json_safe_observed(value), sort_keys=True, separators=(",", ":")
        )

    grouped: dict[tuple[Any, ...], list[_NormalizedFailure]] = {}
    for failure in failures:
        compatibility = (
            failure.side, failure.severity, failure.rule_id,
            failure.metric, failure.operator, canonical(failure.boundary),
            failure.temporal_transition_attribution,
        )
        grouped.setdefault(compatibility, []).append(failure)

    compacted: list[_CompactedFailure] = []
    for compatible in grouped.values():
        example = compatible[0]
        ranges = _contiguous_ranges(tuple(item.frame for item in compatible))
        for start_frame, end_frame in ranges:
            observed = [
                item.observed for item in compatible
                if start_frame <= item.frame <= end_frame
            ]
            temporal_pair_starts = [
                item.temporal_pair_start_frame
                for item in compatible
                if start_frame <= item.frame <= end_frame
                and item.temporal_pair_start_frame is not None
            ]
            temporal_pair_ends = [
                item.temporal_pair_end_frame
                for item in compatible
                if start_frame <= item.frame <= end_frame
                and item.temporal_pair_end_frame is not None
            ]
            if example.operator in {"<", "<="}:
                extreme: Any = min(observed)
            elif example.operator in {">", ">="}:
                extreme = max(observed)
            else:
                unique = {canonical(value): _json_safe_observed(value) for value in observed}
                ordered = [unique[key] for key in sorted(unique)]
                extreme = ordered[0] if len(ordered) == 1 else ordered
            compacted.append(
                (
                    example._replace(
                        frame=start_frame,
                        observed=extreme,
                        temporal_pair_start_frame=(
                            min(temporal_pair_starts) if temporal_pair_starts else None
                        ),
                        temporal_pair_end_frame=(
                            max(temporal_pair_ends) if temporal_pair_ends else None
                        ),
                    ),
                    (start_frame, end_frame),
                )
            )

    return tuple(
        sorted(
            compacted,
            key=lambda item: (
                item[1][0], item[1][1], item[0].side,
                item[0].rule_id, item[0].severity, item[0].metric,
                item[0].operator, repr(item[0].boundary),
            ),
        )
    )


def adapt_keypoint_presence(
    *,
    asset_id: str,
    source_relative_path: str,
    results: Sequence[CheckResult],
    config: LoadedQcConfig,
) -> ModuleResult:
    """Adapt legacy per-frame keypoint-presence observations."""
    relevant_rows = [
        row
        for row in results
        if row.check in {"keypoint_missing", "skeleton_quality_score"}
    ]

    def explicitly_missing_keypoint_field(row: CheckResult) -> bool:
        metrics = row.metrics
        if "keypoint_field_present" in metrics:
            return metrics["keypoint_field_present"] is False or metrics[
                "keypoint_field_present"
            ] == 0
        return bool(
            metrics.get("missing_keypoint_field")
            or metrics.get("keypoint_field_missing")
        )

    missing_row = next(
        (row for row in relevant_rows if explicitly_missing_keypoint_field(row)),
        None,
    )
    if missing_row is not None:
        missing_rule = _rule(
            config,
            "keypoint_presence",
            "missing_keypoint_field",
        )
        issue = _issue_from_row(
            asset_id=asset_id,
            module="keypoint_presence",
            source_relative_path=source_relative_path,
            rule_id=str(missing_rule["rule_id"]),
            row=missing_row,
            severity=_rule_verdict(missing_rule),
            needs_manual_review=False,
            metric="keypoint_field_present",
            observed_value=False,
            operator="!=",
            boundary_value=True,
            hand_side=None,
            frame_range=(None, None),
        )
        return ModuleResult(
            module="keypoint_presence",
            verdict=issue.severity,
            evaluation={
                "decision": issue.severity,
                "checked_frame_count": 0,
                "invalid_frame_count": 0,
                "invalid_frame_ratio": 0.0,
            },
            metrics={"invalid_frame_ranges": ()},
            issues=(issue,),
        )

    rows = [row for row in relevant_rows if row.frame_idx >= 0]
    if not rows:
        return ModuleResult(
            module="keypoint_presence",
            verdict="skipped",
            evaluation={
                "decision": "skipped",
                "reason": "source_signal_not_provided",
            },
            metrics={},
        )

    parameters = config.module_parameters("keypoint_presence")
    expected_count = float(parameters["expected_keypoints_per_hand"])
    count_warn_boundary = float(parameters["min_valid_points_per_hand_warn"])
    count_fail_boundary = float(parameters["min_valid_points_per_hand_fail"])
    ratio_warn_boundary = float(parameters["missing_frame_ratio_warn"])
    ratio_fail_boundary = float(parameters["missing_frame_ratio_fail"])
    count_rule = _rule(config, "keypoint_presence", "too_few_valid_points")
    ratio_rule = _rule(config, "keypoint_presence", "high_missing_frame_ratio")
    nonfinite_rule = _rule(config, "keypoint_presence", "nan_or_inf")
    observations: dict[tuple[str, int], dict[str, Any]] = {}
    invalid_frames_by_hand = {"left": set(), "right": set()}
    minimum_counts: dict[str, float] = {}

    for row in rows:
        for side in ("left", "right"):
            observed = observations.setdefault(
                (side, row.frame_idx),
                {
                    "counts": [],
                    "finite_counts": [],
                    "ratios": [],
                    "missing": False,
                    "quality_low": False,
                    "explicit_invalid": False,
                    "invalid_reasons": [],
                    "all_zero": False,
                    "all_identical": False,
                },
            )
            count = row.metrics.get(f"valid_keypoint_count_{side}")
            if isinstance(count, Real):
                observed["counts"].append(float(count))
            finite_count = row.metrics.get(f"finite_keypoint_count_{side}")
            if isinstance(finite_count, Real):
                observed["finite_counts"].append(float(finite_count))
            ratio = row.metrics.get(f"missing_fraction_in_10s_window_{side}")
            if isinstance(ratio, Real):
                observed["ratios"].append(float(ratio))
            missing = row.metrics.get(f"missing_keypoint_count_{side}")
            observed["missing"] |= isinstance(missing, Real) and float(missing) > 0
            observed["quality_low"] |= bool(
                row.metrics.get(f"quality_low_{side}", 0.0)
            )
            observed["explicit_invalid"] |= bool(
                row.metrics.get(f"keypoint_existence_invalid_{side}", False)
            )
            reasons = row.metrics.get(f"invalid_reasons_{side}", ())
            if isinstance(reasons, (list, tuple)):
                observed["invalid_reasons"].extend(
                    str(reason) for reason in reasons if str(reason)
                )
            observed["all_zero"] |= bool(
                row.metrics.get(f"all_zero_{side}", False)
            )
            observed["all_identical"] |= bool(
                row.metrics.get(f"all_identical_{side}", False)
            )

    selected: dict[tuple[str, int], _NormalizedFailure] = {}
    for side in ("left", "right"):
        count_metric = f"valid_keypoint_count_{side}"
        ratio_metric = f"missing_fraction_in_10s_window_{side}"
        side_observations = {
            frame: observed
            for (observed_side, frame), observed in observations.items()
            if observed_side == side
        }
        finite_counts = [
            value
            for observed in side_observations.values()
            for value in observed["counts"]
            if math.isfinite(value)
        ]
        if finite_counts:
            minimum_counts[side] = min(finite_counts)
        side_structured_frames = {
            frame
            for frame, observed in side_observations.items()
            if observed["counts"] or observed["missing"]
        }
        side_detector_invalid_frames = {
            frame
            for frame, observed in side_observations.items()
            if observed["explicit_invalid"]
            or observed["missing"]
            or any(
                not math.isfinite(value) or value < expected_count
                for value in observed["counts"]
            )
        }
        side_invalid_ratio = (
            len(side_detector_invalid_frames) / len(side_structured_frames)
            if side_structured_frames
            else 0.0
        )
        for frame, observed in side_observations.items():
            counts = observed["counts"]
            nonfinite_counts = [value for value in counts if not math.isfinite(value)]
            finite_frame_counts = [value for value in counts if math.isfinite(value)]
            count = min(finite_frame_counts, default=expected_count)
            nonfinite_ratios = [
                value for value in observed["ratios"] if not math.isfinite(value)
            ]
            explicit_ratio = max(
                (
                    value
                    for value in observed["ratios"]
                    if math.isfinite(value)
                ),
                default=-math.inf,
            )
            aggregate_ratio = (
                side_invalid_ratio
                if frame in side_detector_invalid_frames
                else -math.inf
            )
            ratio = max(explicit_ratio, aggregate_ratio)
            failure: _NormalizedFailure | None = None
            if observed["explicit_invalid"]:
                invalid_reasons = tuple(sorted(set(observed["invalid_reasons"])))
                if "nonfinite_keypoints" in invalid_reasons:
                    failure = _NormalizedFailure(
                        side,
                        frame,
                        "fail",
                        str(nonfinite_rule["rule_id"]),
                        f"invalid_reasons_{side}",
                        list(invalid_reasons),
                        "==",
                        [],
                    )
                else:
                    failure = _NormalizedFailure(
                        side,
                        frame,
                        "fail",
                        str(count_rule["rule_id"]),
                        count_metric,
                        count,
                        "<",
                        expected_count,
                    )
            elif (
                nonfinite_counts or nonfinite_ratios
            ) and parameters["nan_or_inf_fail"]:
                metric = count_metric if nonfinite_counts else ratio_metric
                invalid_value = (
                    nonfinite_counts[0]
                    if nonfinite_counts
                    else nonfinite_ratios[0]
                )
                failure = _NormalizedFailure(
                    side, frame, "fail", str(nonfinite_rule["rule_id"]), metric,
                    _json_safe_observed(invalid_value), "is_finite", True,
                )
            elif count < count_fail_boundary:
                failure = _NormalizedFailure(
                    side, frame, "fail", str(count_rule["rule_id"]), count_metric, count,
                    "<", count_fail_boundary,
                )
            elif ratio >= ratio_fail_boundary:
                metric = ratio_metric if explicit_ratio >= aggregate_ratio else f"invalid_frame_ratio_{side}"
                failure = _NormalizedFailure(
                    side, frame, "fail", str(ratio_rule["rule_id"]), metric, ratio,
                    ">=", ratio_fail_boundary,
                )
            elif count < count_warn_boundary:
                failure = _NormalizedFailure(
                    side, frame, "warn", str(count_rule["rule_id"]), count_metric, count,
                    "<", count_warn_boundary,
                )
            elif ratio >= ratio_warn_boundary:
                metric = ratio_metric if explicit_ratio >= aggregate_ratio else f"invalid_frame_ratio_{side}"
                failure = _NormalizedFailure(
                    side, frame, "warn", str(ratio_rule["rule_id"]), metric, ratio,
                    ">=", ratio_warn_boundary,
                )
            if failure is not None:
                selected[(side, frame)] = failure
            # Rolling ratios can affect the module verdict, but they do not make
            # the current source frame itself existence-invalid.
        invalid_frames_by_hand[side].update(side_detector_invalid_frames)

    checked_frames = {row.frame_idx for row in rows}
    invalid_frames = set().union(*invalid_frames_by_hand.values())
    invalid_frames.update(
        row.frame_idx
        for row in rows
        if (
            row.check == "skeleton_quality_score"
            and bool(row.metrics.get("keypoint_presence_invalid", 0.0))
        )
        or (row.check == "keypoint_missing" and row.flag is True)
    )
    affected_frame_ranges = _contiguous_ranges(tuple(invalid_frames))
    invalid_frame_details = [
        {
            "side": side,
            "frame_idx": frame,
            "invalid_reasons": sorted(set(observed["invalid_reasons"])),
            "valid_point_count": int(min(observed["counts"], default=0.0)),
            "finite_point_count": int(
                min(observed["finite_counts"], default=0.0)
            ),
            "all_zero": bool(observed["all_zero"]),
            "all_identical": bool(observed["all_identical"]),
        }
        for (side, frame), observed in sorted(observations.items())
        if observed["explicit_invalid"]
    ]

    issues = tuple(
        _issue_from_row(
            asset_id=asset_id,
            module="keypoint_presence",
            source_relative_path=source_relative_path,
            rule_id=failure.rule_id,
            row=rows[0],
            severity=failure.severity,
            needs_manual_review=failure.severity == "warn",
            metric=failure.metric,
            observed_value=failure.observed,
            operator=failure.operator,
            boundary_value=failure.boundary,
            hand_side=failure.side,
            frame_range=frame_range,
        )
        for failure, frame_range in _compact_failures(tuple(selected.values()))
    )
    verdict = _worst("pass", *(issue.severity for issue in issues))
    frame_exclusions = tuple(
        FrameExclusion(
            start_frame=row.frame_idx,
            end_frame=row.frame_idx,
            module="keypoint_presence",
            reason=reason,
            raw_severity="fail",
            hand_side=side,
            first_introduced_stage="keypoint_presence",
        )
        for row in rows
        if row.check == "keypoint_missing" and row.frame_idx >= 0
        for side in ("left", "right")
        if bool(row.metrics.get(f"keypoint_existence_invalid_{side}", False))
        for reason in (
            tuple(
                str(value)
                for value in row.metrics.get(f"invalid_reasons_{side}", ())
                if str(value)
            )
            or ("keypoint_existence_invalid",)
        )
    )
    return ModuleResult(
        module="keypoint_presence",
        verdict=verdict,
        evaluation={
            "decision": verdict,
            "checked_frame_count": len(checked_frames),
            "invalid_frame_count": len(invalid_frames),
            "invalid_frame_ratio": len(invalid_frames) / len(checked_frames),
            "affected_frame_count": len(invalid_frames),
            "affected_frame_ranges": affected_frame_ranges,
            "invalid_frame_details": invalid_frame_details,
        },
        metrics={
            **{
                f"min_valid_keypoint_count_{side}": value
                for side, value in minimum_counts.items()
            },
            "invalid_frame_ranges": affected_frame_ranges,
            "invalid_frame_ranges_by_hand": {
                side: _contiguous_ranges(tuple(side_frames))
                for side, side_frames in invalid_frames_by_hand.items()
            },
            "affected_frame_count": len(invalid_frames),
            "affected_frame_ranges": affected_frame_ranges,
            "invalid_frame_details": invalid_frame_details,
        },
        issues=issues,
        frame_exclusions=frame_exclusions,
    )


class _TemporalCandidate(NamedTuple):
    start_frame: int
    end_frame: int
    hand_side: str | None
    trigger_metrics: Mapping[str, Any]


def _temporal_rule(config: LoadedQcConfig, alias: str) -> Mapping[str, Any]:
    return _rule(
        config, "keypoint_temporal", TEMPORAL_RULES[alias].rsplit(".", 1)[-1]
    )


def _temporal_row_failures(
    row: CheckResult,
    *,
    config: LoadedQcConfig,
    parameters: Mapping[str, Any],
) -> tuple[_NormalizedFailure, ...]:
    if row.check != "skeleton_quality_score" or row.frame_idx < 0:
        return ()

    skeleton_verdict = row.metrics.get("skeleton_verdict")
    raw_tokens = row.metrics.get("which_thresholds_exceeded")
    tokens = (
        {
            token
            for token in raw_tokens
            if isinstance(token, str) and token in _CORE_TEMPORAL_METRICS
        }
        if isinstance(raw_tokens, (list, tuple))
        else set()
    )

    def observed_metric_name(metric: str) -> str:
        standardized = _STANDARDIZED_TEMPORAL_METRIC_BY_NATIVE[metric]
        return (
            standardized
            if row.metrics.get("decision_metric_source") == "standardized_30hz"
            and standardized in row.metrics
            else metric
        )

    def failure(
        alias: str,
        side: str,
        metric: str,
        observed: Any,
        operator: str,
        boundary: Any,
    ) -> _NormalizedFailure:
        rule = _temporal_rule(config, alias)
        use_standardized_lineage = (
            row.metrics.get("decision_metric_source") == "standardized_30hz"
        )
        lineage = temporal_transition_lineage(
            target_frame=row.frame_idx,
            pair_start_frame=(
                row.metrics.get("standardized_temporal_pair_start_frame")
                if use_standardized_lineage
                else row.metrics.get("temporal_pair_start_frame")
            ),
            pair_end_frame=(
                row.metrics.get("standardized_temporal_pair_end_frame")
                if use_standardized_lineage
                else row.metrics.get("temporal_pair_end_frame")
            ),
            attribution=(
                row.metrics.get("standardized_temporal_transition_attribution")
                if use_standardized_lineage
                else row.metrics.get("temporal_transition_attribution")
            ),
        )
        return _NormalizedFailure(
            side, row.frame_idx, _rule_verdict(rule), str(rule["rule_id"]),
            metric, observed, operator, boundary,
            lineage.pair_start_frame,
            lineage.pair_end_frame,
            lineage.attribution,
        )

    ratios: dict[str, float] = {}
    for metric in tokens:
        observed = row.metrics.get(observed_metric_name(metric))
        threshold = parameters.get(f"{metric}_threshold")
        if (
            isinstance(observed, Real)
            and not isinstance(observed, bool)
            and isinstance(threshold, Real)
            and not isinstance(threshold, bool)
            and math.isfinite(float(observed))
            and float(threshold) > 0.0
        ):
            ratios[metric] = float(observed) / float(threshold)

    hard_count = int(parameters["hard_exceeded_metric_count"])
    strong_acceleration = (
        ratios.get("joint_acceleration_m_s2_max", -math.inf)
        >= float(parameters["strong_acceleration_ratio"])
    )
    strong_displacement = (
        ratios.get("joint_displacement_m_max", -math.inf)
        >= float(parameters["strong_displacement_ratio"])
    )
    if skeleton_verdict == "suspect" and (
        len(tokens) >= hard_count
        or strong_acceleration
        or strong_displacement
        or row.severity == "fail"
        or row.metrics.get("temporal_severity") == "fail"
    ):
        strongest_metric = max(
            ratios,
            key=ratios.__getitem__,
            default="exceeded_temporal_metric_count",
        )
        if len(tokens) >= hard_count:
            metric = "exceeded_temporal_metric_count"
            observed: Any = len(tokens)
            operator = ">="
            boundary: Any = hard_count
        elif strongest_metric in ratios:
            metric = observed_metric_name(strongest_metric)
            observed = row.metrics.get(metric)
            operator = ">"
            boundary = parameters[f"{strongest_metric}_threshold"]
        else:
            metric = "which_thresholds_exceeded"
            observed = sorted(tokens)
            operator = "not_empty"
            boundary = []
        return (
            failure(
                "strong", "both", metric, observed, operator, boundary,
            ),
        )

    projection_sides = [
        side
        for side in parameters["sides"]
        if bool(row.metrics.get(f"{side}_needs_projection_review", 0.0))
    ]
    if projection_sides:
        return tuple(
            failure(
                "projection", str(side), f"{side}_needs_projection_review",
                row.metrics[f"{side}_needs_projection_review"], "==", 1.0,
            )
            for side in projection_sides
        )
    if bool(row.metrics.get("needs_projection_review", 0.0)):
        return (
            failure(
                "projection", "both", "needs_projection_review",
                row.metrics["needs_projection_review"], "==", 1.0,
            ),
        )

    if tokens and skeleton_verdict in {"review", "suspect"}:
        strongest_metric = max(
            tokens,
            key=lambda metric: ratios.get(metric, -math.inf),
        )
        metric = observed_metric_name(strongest_metric)
        observed = row.metrics.get(metric)
        return (
            failure(
                "threshold",
                "both",
                metric,
                observed,
                ">",
                parameters[f"{strongest_metric}_threshold"],
            ),
        )

    return ()


def _is_valid_temporal_row(row: CheckResult) -> bool:
    if row.check != "skeleton_quality_score" or row.frame_idx < 0:
        return False
    return row.metrics.get("temporal_output_valid") is True


def _source_temporal_candidates(
    asset_id: str,
    candidate_windows: Sequence[Mapping[str, Any]],
) -> tuple[_TemporalCandidate, ...]:
    def normalize(candidate: Mapping[str, Any]) -> _TemporalCandidate:
        side = candidate.get("hand_side")
        trigger_metrics = candidate.get("trigger_metrics")
        return _TemporalCandidate(
            start_frame=int(candidate["start_frame"]),
            end_frame=int(candidate["end_frame"]),
            hand_side=side if isinstance(side, str) and side else None,
            trigger_metrics=(
                trigger_metrics if isinstance(trigger_metrics, Mapping) else {}
            ),
        )

    return tuple(
        normalize(candidate)
        for candidate in candidate_windows
        if candidate.get("asset_id") == asset_id
        and candidate.get("coordinate_space") == "source"
        and candidate.get("frame_coordinate_system") in {None, "source_inclusive"}
    )


def _peak_trigger_metrics(
    candidates: Sequence[_TemporalCandidate],
) -> dict[str, float | int]:
    peaks: dict[str, float | int] = {}

    def update(name: str, value: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, Real):
            return
        numeric = float(value)
        if not math.isfinite(numeric):
            return
        previous = peaks.get(name)
        if previous is None or numeric > float(previous):
            peaks[name] = int(value) if isinstance(value, int) else numeric

    for candidate in candidates:
        for name, value in candidate.trigger_metrics.items():
            if isinstance(name, str):
                update(name, value)
    return {name: peaks[name] for name in sorted(peaks)}


def _link_issue_evidence(
    issue: Issue,
    *,
    kind: str,
    path: str,
    frame_range: tuple[int, int],
    hand_side: str | None,
) -> tuple[Issue, EvidenceRef]:
    evidence_id = f"{issue.issue_id}:{kind}"
    evidence = EvidenceRef(
        evidence_id=evidence_id,
        kind=kind,
        path=path,
        coordinate_system="source_inclusive",
        start_frame=frame_range[0],
        end_frame=frame_range[1],
        hand_side=hand_side,
    )
    return replace(issue, evidence_ids=(evidence_id,)), evidence


def adapt_keypoint_temporal(
    *,
    asset_id: str,
    source_relative_path: str,
    results: Sequence[CheckResult],
    candidate_windows: Sequence[Mapping[str, Any]],
    config: LoadedQcConfig,
) -> ModuleResult:
    """Adapt structured temporal rows and mapped source candidate windows."""
    rows = [
        row
        for row in results
        if row.check in {"keypoint_temporal", "skeleton_quality_score"}
    ]
    candidates = _source_temporal_candidates(asset_id, candidate_windows)
    frame_rows = [row for row in rows if row.frame_idx >= 0]
    valid_rows = [row for row in frame_rows if _is_valid_temporal_row(row)]
    uncalibrated_frame_count = len(frame_rows) - len(valid_rows)
    if not valid_rows:
        no_valid_rule = _temporal_rule(config, "no_valid")
        issue_row = (
            frame_rows[0]
            if frame_rows
            else CheckResult(
                "skeleton_quality_score",
                0,
                -1,
                {},
                None,
                "no valid temporal output",
                severity="uncalibrated",
            )
        )
        issue = _issue_from_row(
            asset_id=asset_id,
            module="keypoint_temporal",
            rule_id=str(no_valid_rule["rule_id"]),
            row=issue_row,
            source_relative_path=source_relative_path,
            severity=_rule_verdict(no_valid_rule),
            needs_manual_review=True,
            metric="valid_temporal_frame_count",
            observed_value=0,
            operator=">",
            boundary_value=0,
            frame_range=(None, None),
            evidence_kind="check_result",
        )
        return ModuleResult(
            module="keypoint_temporal",
            verdict="warn",
            evaluation={
                "decision": "review",
                "output_status": "no_valid_output",
                "reason": "no_valid_temporal_output",
                "checked_frame_count": len(frame_rows),
                "valid_frame_count": 0,
                "uncalibrated_frame_count": uncalibrated_frame_count,
                "candidate_window_count": len(candidates),
            },
            metrics={
                "valid_frame_count": 0,
                "uncalibrated_frame_count": uncalibrated_frame_count,
                "peak_trigger_metrics": _peak_trigger_metrics(candidates),
                "candidate_window_count": len(candidates),
                "candidate_frame_union_count": len(
                    {
                        frame
                        for candidate in candidates
                        for frame in range(
                            candidate.start_frame,
                            candidate.end_frame + 1,
                        )
                    }
                ),
            },
            issues=(issue,),
        )

    parameters = config.module_parameters("keypoint_temporal")
    failures = tuple(
        failure
        for row in valid_rows
        for failure in _temporal_row_failures(
            row,
            config=config,
            parameters=parameters,
        )
    )
    failures = tuple(
        failure
        for failure in failures
        if failure.severity == "fail"
        or not any(
            candidate.start_frame <= failure.frame <= candidate.end_frame
            and candidate.hand_side in {None, "both", failure.side}
            for candidate in candidates
        )
    )
    compacted_failures = _compact_failures(failures)
    issue_evidence: list[tuple[Issue, EvidenceRef]] = []
    row_for_issue = valid_rows[0] if valid_rows else None
    if row_for_issue is not None:
        for failure, frame_range in compacted_failures:
            issue = _issue_from_row(
                asset_id=asset_id,
                module="keypoint_temporal",
                rule_id=failure.rule_id,
                row=row_for_issue,
                source_relative_path=source_relative_path,
                severity=failure.severity,
                needs_manual_review=failure.severity == "warn",
                metric=failure.metric,
                observed_value=failure.observed,
                operator=failure.operator,
                boundary_value=failure.boundary,
                hand_side=failure.side,
                frame_range=frame_range,
                evidence_kind="frame_metrics",
            )
            issue = replace(
                issue,
                context={
                    **dict(issue.context),
                    "temporal_pair_start_frame": failure.temporal_pair_start_frame,
                    "temporal_pair_end_frame": failure.temporal_pair_end_frame,
                    "temporal_transition_attribution": failure.temporal_transition_attribution,
                },
            )
            issue_evidence.append(
                _link_issue_evidence(
                    issue,
                    kind="frame_metrics",
                    path="check_results.json",
                    frame_range=frame_range,
                    hand_side=failure.side,
                )
            )

    candidate_rule = (
        _temporal_rule(config, "candidate") if candidates else None
    )
    if candidate_rule is not None:
        candidate_row = CheckResult(
            "composite_frame_verdict", 0, -1, {}, None, "candidate window"
        )
        for candidate in candidates:
            issue = _issue_from_row(
                asset_id=asset_id,
                module="keypoint_temporal",
                rule_id=str(candidate_rule["rule_id"]),
                row=candidate_row,
                source_relative_path=source_relative_path,
                severity=_rule_verdict(candidate_rule),
                needs_manual_review=True,
                metric="trigger_metrics",
                observed_value=candidate.trigger_metrics,
                operator="exists",
                boundary_value=True,
                hand_side=candidate.hand_side,
                frame_range=(candidate.start_frame, candidate.end_frame),
                evidence_kind="candidate_window",
            )
            issue_evidence.append(
                _link_issue_evidence(
                    issue,
                    kind="candidate_window",
                    path="candidate_windows.json",
                    frame_range=(candidate.start_frame, candidate.end_frame),
                    hand_side=candidate.hand_side,
                )
            )

    issues = tuple(issue for issue, _evidence in issue_evidence)
    verdict = _worst("pass", *(issue.severity for issue in issues))
    checked_frames = {row.frame_idx for row in rows if row.frame_idx >= 0}
    return ModuleResult(
        module="keypoint_temporal",
        verdict=verdict,
        evaluation={
            "decision": verdict,
            "output_status": "valid",
            "checked_frame_count": len(checked_frames),
            "valid_frame_count": len(valid_rows),
            "uncalibrated_frame_count": uncalibrated_frame_count,
            "candidate_window_count": len(candidates),
        },
        metrics={
            "peak_trigger_metrics": _peak_trigger_metrics(candidates),
            "candidate_window_count": len(candidates),
            "candidate_frame_union_count": len(
                {
                    frame
                    for candidate in candidates
                    for frame in range(candidate.start_frame, candidate.end_frame + 1)
                }
            ),
        },
        issues=issues,
        evidence=tuple(evidence for _issue, evidence in issue_evidence),
        frame_exclusions=tuple(
            FrameExclusion(
                start_frame=frame_range[0],
                end_frame=frame_range[1],
                module="keypoint_temporal",
                reason=failure.rule_id,
                raw_severity="fail",
                hand_side=failure.side,
                first_introduced_stage="keypoint_temporal",
                temporal_pair_start_frame=failure.temporal_pair_start_frame,
                temporal_pair_end_frame=failure.temporal_pair_end_frame,
                temporal_transition_attribution=failure.temporal_transition_attribution,
            )
            for failure, frame_range in compacted_failures
            if failure.severity == "fail"
        ),
    )


def _morphology_failure(
    row: CheckResult,
    token: Any,
    parameters: Mapping[str, Any],
) -> _NormalizedFailure | None:
    if not isinstance(token, str) or ":" not in token:
        return None
    side, reason = token.split(":", 1)
    if side not in parameters["sides"]:
        return None

    if reason == "palm_scale_too_small":
        rule_name = reason
        detector_verdict = "fail"
    elif reason.endswith("_review"):
        rule_name = reason.removesuffix("_review")
        detector_verdict = "review"
    elif reason.endswith("_fail"):
        rule_name = reason.removesuffix("_fail")
        detector_verdict = "fail"
    else:
        return None

    rule_id = MORPHOLOGY_REASON_TO_RULE.get(rule_name)
    metric = _MORPHOLOGY_OBSERVED_METRIC.get(rule_name)
    if rule_id is None or metric is None:
        return None
    observed_key = f"{side}_{metric}"
    if observed_key not in row.metrics:
        return None

    if rule_name == "collapsed_finger_count":
        operator = ">"
        boundary = 0
    else:
        parameter_name = _MORPHOLOGY_THRESHOLD_PARAMETER.get(
            (rule_name, detector_verdict)
        )
        if parameter_name is None:
            return None
        boundary = parameters[parameter_name]
        operator = "<" if rule_name == "palm_scale_too_small" else ">="
        if rule_name == "joint_angle_min_deg":
            operator = "<="

    severity: Verdict = "warn" if detector_verdict == "review" else "fail"
    return _NormalizedFailure(
        side=side,
        frame=row.frame_idx,
        severity=severity,
        rule_id=rule_id,
        metric=metric,
        observed=row.metrics[observed_key],
        operator=operator,
        boundary=boundary,
    )


def adapt_keypoint_morphology(
    *,
    asset_id: str,
    source_relative_path: str,
    results: Sequence[CheckResult],
    config: LoadedQcConfig,
) -> ModuleResult:
    """Adapt structured legacy morphology thresholds into frame-range issues."""
    summary = _summary(results, "keypoint_morphology")
    detector_verdict = summary.metrics.get("morphology_verdict")
    verdict_by_detector = {
        "pass": "pass",
        "review": "warn",
        "fail": "fail",
        "not_applicable": "skipped",
    }
    if detector_verdict not in verdict_by_detector:
        raise ValueError(
            f"invalid keypoint_morphology summary verdict: {detector_verdict!r}"
        )
    verdict = cast(Verdict, verdict_by_detector[detector_verdict])
    frame_rows = [
        row
        for row in results
        if row.check == "keypoint_morphology" and row.frame_idx >= 0
    ]
    fail_frames = {
        row.frame_idx
        for row in frame_rows
        if row.metrics.get("morphology_verdict") == "fail"
    }
    review_frames = {
        row.frame_idx
        for row in frame_rows
        if row.metrics.get("morphology_verdict") == "review"
    }
    affected_frames = fail_frames | review_frames
    frame_metrics = {
        "fail_frame_count": len(fail_frames),
        "fail_frame_ranges": _contiguous_ranges(tuple(fail_frames)),
        "review_frame_count": len(review_frames),
        "review_frame_ranges": _contiguous_ranges(tuple(review_frames)),
        "affected_frame_count": len(affected_frames),
        "affected_frame_ranges": _contiguous_ranges(tuple(affected_frames)),
    }
    if verdict == "skipped":
        return ModuleResult(
            module="keypoint_morphology",
            verdict="skipped",
            evaluation={
                "decision": "skipped",
                "reason": summary.reason,
                **frame_metrics,
            },
            metrics={**dict(summary.metrics), **frame_metrics},
        )

    parameters = config.module_parameters("keypoint_morphology")
    failures = [
        failure
        for row in results
        if row.check == "keypoint_morphology" and row.frame_idx >= 0
        for token in row.metrics.get("which_thresholds_exceeded", ())
        if (failure := _morphology_failure(row, token, parameters)) is not None
    ]
    compacted_failures = _compact_failures(failures)
    issue_evidence: list[tuple[Issue, EvidenceRef]] = []
    for failure, frame_range in compacted_failures:
        issue = _issue_from_row(
            asset_id=asset_id,
            module="keypoint_morphology",
            rule_id=failure.rule_id,
            row=summary,
            source_relative_path=source_relative_path,
            severity=failure.severity,
            needs_manual_review=failure.severity == "warn",
            metric=failure.metric,
            observed_value=failure.observed,
            operator=failure.operator,
            boundary_value=failure.boundary,
            hand_side=failure.side,
            frame_range=frame_range,
            evidence_kind="frame_metrics",
        )
        issue_evidence.append(
            _link_issue_evidence(
                issue,
                kind="frame_metrics",
                path="check_results.json",
                frame_range=frame_range,
                hand_side=failure.side,
            )
        )
    return ModuleResult(
        module="keypoint_morphology",
        verdict=verdict,
        evaluation={"decision": verdict, "reason": summary.reason, **frame_metrics},
        metrics={**dict(summary.metrics), **frame_metrics},
        issues=tuple(issue for issue, _evidence in issue_evidence),
        evidence=tuple(evidence for _issue, evidence in issue_evidence),
        frame_exclusions=tuple(
            FrameExclusion(
                start_frame=frame_range[0],
                end_frame=frame_range[1],
                module="keypoint_morphology",
                reason=failure.rule_id,
                raw_severity="fail",
                hand_side=failure.side,
                first_introduced_stage="keypoint_morphology",
            )
            for failure, frame_range in compacted_failures
            if failure.severity == "fail"
        ),
    )


def _config_from_parameters(
    cls: type[_ConfigType],
    parameters: Mapping[str, Any],
) -> _ConfigType:
    values: dict[str, Any] = {}
    for item in fields(cls):
        if item.name not in parameters:
            raise ValueError(
                f"unified config has no value for {cls.__name__}.{item.name}"
            )
        values[item.name] = parameters[item.name]
    return cls(**values)


def precheck_config_from_unified(
    config: LoadedQcConfig,
    *,
    module_names: Sequence[str],
    output_dir: Path,
) -> PrecheckConfig:
    """Inject unified v2 parameters into the existing precheck config types."""
    enabled_checks: list[str] = []
    for name in module_names:
        try:
            check_name = _PRECHECK_NAME_BY_MODULE[name]
        except KeyError:
            raise ValueError(f"unsupported unified precheck module: {name}") from None
        if check_name not in enabled_checks:
            enabled_checks.append(check_name)

    text = config.module_parameters("hdf5_text_info")
    quality = config.module_parameters("quality_hand")
    presence = config.module_parameters("keypoint_presence")
    morphology = config.module_parameters("keypoint_morphology")
    temporal = config.module_parameters("keypoint_temporal")
    return PrecheckConfig(
        output_dir=Path(output_dir),
        enabled_checks=enabled_checks,
        text_integrity=_config_from_parameters(TextIntegrityConfig, text),
        quality_score=_config_from_parameters(QualityScoreConfig, quality),
        keypoint_missing=_config_from_parameters(KeypointMissingConfig, presence),
        keypoint_morphology=_config_from_parameters(
            KeypointMorphologyConfig, morphology
        ),
        keypoint_temporal=_config_from_parameters(KeypointTemporalConfig, temporal),
        skeleton_quality_score=_config_from_parameters(
            SkeletonQualityScoreConfig, temporal
        ),
        composite_frame_verdict=_config_from_parameters(
            CompositeFrameVerdictConfig, temporal
        ),
    )


__all__ = [
    "_contiguous_ranges",
    "adapt_hdf5_text_info",
    "adapt_keypoint_morphology",
    "adapt_keypoint_presence",
    "adapt_keypoint_temporal",
    "adapt_quality_hand",
    "precheck_config_from_unified",
]
