"""Translate legacy precheck outputs into unified QC module contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import fields
from numbers import Real
from pathlib import Path
from typing import Any, TypeVar, cast

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
from qc_common.contracts import Issue, ModuleResult, Verdict, build_issue_id
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
) -> Issue:
    if severity not in {"warn", "fail"}:
        raise ValueError(f"issue severity must be warn or fail, got {severity!r}")
    start_frame = row.frame_idx if row.frame_idx >= 0 else None
    end_frame = start_frame
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
            evidence_kind="check_result",
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

    parameters = config.module_parameters("quality_hand")
    expected_shape = list(parameters["expected_shape"])
    valid_values = list(parameters["valid_values"])
    low_value = parameters["low_quality_hand_value"]
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
    "adapt_hdf5_text_info",
    "adapt_quality_hand",
    "precheck_config_from_unified",
]
