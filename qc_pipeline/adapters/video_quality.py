"""Adapt legacy video-quality results to the unified QC module contract."""

from __future__ import annotations

import mimetypes
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

from acceptance_pull.video_quality import (
    VideoQualityResult,
    asset_qc_result_to_json,
    load_video_quality_config,
)
from qc_common.config import LoadedQcConfig
from qc_common.contracts import (
    EvidenceRef,
    Issue,
    ModuleResult,
    Verdict,
    build_issue_id,
    relative_evidence_path,
)
from qc_common.report_mutation import apply_module_result
from qc_pipeline.context import AssetContext


_COORDINATE_SYSTEM = "source_video_inclusive"
ReadinessCondition = Literal[
    "ready_to_write",
    "already_completed",
    "awaiting_pipeline",
    "invalid_report",
]
_READINESS_CONDITIONS = frozenset(
    {
        "ready_to_write",
        "already_completed",
        "awaiting_pipeline",
        "invalid_report",
    }
)


@dataclass(frozen=True)
class VideoQualityReportReadiness:
    """Canonical readiness result shared by both video-quality runners."""

    condition: ReadinessCondition
    reason: str
    current_next_module: str | None

    def __post_init__(self) -> None:
        if self.condition not in _READINESS_CONDITIONS:
            raise ValueError(f"unsupported readiness condition: {self.condition}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "reason": self.reason,
            "current_next_module": self.current_next_module,
            "required_module": "video_quality",
        }


def _readiness(
    condition: ReadinessCondition,
    reason: str,
    current_next_module: str | None,
) -> VideoQualityReportReadiness:
    return VideoQualityReportReadiness(
        condition=condition,
        reason=reason,
        current_next_module=current_next_module,
    )


def _completed_flow_error(
    report: Mapping[str, Any],
    module: Mapping[str, Any],
    configured_successor: str,
) -> str | None:
    flow = module.get("flow")
    result_gate = flow.get("result_gate") if isinstance(flow, Mapping) else None
    exit_gate = flow.get("exit_gate") if isinstance(flow, Mapping) else None
    evaluation = module.get("evaluation")
    if not isinstance(result_gate, Mapping) or not isinstance(exit_gate, Mapping):
        return "video_quality_exit_gate_invalid"
    verdict = result_gate.get("verdict")
    valid_verdict = verdict in {"pass", "warn", "fail", "skipped"}
    result_consistent = (
        valid_verdict
        and result_gate.get("has_fail") is (verdict == "fail")
        and result_gate.get("has_warn") is (verdict == "warn")
        and isinstance(evaluation, Mapping)
        and evaluation.get("decision") == verdict
    )
    state = exit_gate.get("state")
    continue_to_next = exit_gate.get("continue_to_next_module")
    exit_next = exit_gate.get("next_module")
    if state == "continue":
        valid_exit = continue_to_next is True and exit_next == configured_successor
    elif state == "stop_qc":
        valid_exit = (
            continue_to_next is False
            and exit_next is None
            and verdict == "fail"
        )
    else:
        valid_exit = False
    if not result_consistent or not valid_exit:
        return "video_quality_exit_gate_invalid"

    pipeline = report.get("pipeline_state")
    if not isinstance(pipeline, Mapping):
        return "video_quality_pipeline_state_invalid"
    if state == "continue":
        valid_pipeline = (
            pipeline.get("status") == "running"
            and pipeline.get("next_module") == configured_successor
            and pipeline.get("stop_reason") is None
            and report.get("overall_decision") is None
        )
    else:
        valid_pipeline = (
            pipeline.get("status") == "stopped"
            and pipeline.get("next_module") is None
            and pipeline.get("stop_reason") == "quality_fail:video_quality"
            and report.get("overall_decision") == "fail"
        )
    return None if valid_pipeline else "video_quality_pipeline_state_invalid"


def _matching_source_evidence(
    module: Mapping[str, Any],
    *,
    source_video_path: str,
    start_frame: int | None,
    end_frame: int | None,
) -> bool:
    evidence = module.get("evidence")
    return isinstance(evidence, list) and any(
        isinstance(item, Mapping)
        and item.get("kind") == "source_video"
        and item.get("path") == source_video_path
        and item.get("coordinate_system") == _COORDINATE_SYSTEM
        and item.get("start_frame") == start_frame
        and item.get("end_frame") == end_frame
        for item in evidence
    )


def inspect_video_quality_report(
    *,
    report: Mapping[str, Any] | None,
    asset_id: str,
    source_video_path: str,
    source_range: tuple[int, int] | None,
    configured_successor: str,
) -> VideoQualityReportReadiness:
    """Classify whether a report can accept or already contains this result."""
    if not configured_successor:
        raise ValueError("video_quality must have a configured successor")
    if source_range is None:
        start_frame = end_frame = None
    else:
        start_frame, exclusive_end = source_range
        if start_frame < 0 or exclusive_end <= start_frame:
            raise ValueError("source_range must be a non-empty half-open frame range")
        end_frame = exclusive_end - 1

    if report is None:
        return _readiness("awaiting_pipeline", "report_missing", None)

    if report.get("asset_id") != asset_id:
        return _readiness("invalid_report", "asset_id_mismatch", None)

    source_files = report.get("source_files")
    recorded_video = (
        source_files.get("video") if isinstance(source_files, Mapping) else None
    )
    recorded_path = (
        recorded_video.get("path")
        if isinstance(recorded_video, Mapping)
        else None
    )
    if recorded_path != source_video_path:
        return _readiness("invalid_report", "source_video_path_mismatch", None)

    pipeline_state = report.get("pipeline_state")
    if not isinstance(pipeline_state, Mapping):
        return _readiness("invalid_report", "pipeline_state_invalid", None)
    current_next = pipeline_state.get("next_module")
    last_completed = pipeline_state.get("last_completed_module")
    if current_next is not None and not isinstance(current_next, str):
        return _readiness("invalid_report", "pipeline_state_invalid", None)
    if last_completed is not None and not isinstance(last_completed, str):
        return _readiness("invalid_report", "pipeline_state_invalid", current_next)

    module = report.get("video_quality")
    if module is not None:
        if not isinstance(module, Mapping):
            return _readiness(
                "invalid_report",
                "video_quality_block_invalid",
                current_next,
            )
        flow_error = _completed_flow_error(
            report,
            module,
            configured_successor,
        )
        if flow_error is not None:
            return _readiness(
                "invalid_report",
                flow_error,
                current_next,
            )
        if not _matching_source_evidence(
            module,
            source_video_path=source_video_path,
            start_frame=start_frame,
            end_frame=end_frame,
        ):
            return _readiness(
                "invalid_report",
                "video_quality_source_range_mismatch",
                current_next,
            )
        if last_completed == "video_quality":
            return _readiness(
                "already_completed",
                "video_quality_already_completed",
                current_next,
            )

    if current_next == "video_quality":
        ready_state = (
            pipeline_state.get("status") == "running"
            and pipeline_state.get("stop_reason") is None
            and report.get("overall_decision") is None
        )
        if not ready_state:
            return _readiness(
                "invalid_report",
                "video_quality_pipeline_state_invalid",
                current_next,
            )
        return _readiness(
            "ready_to_write",
            "pipeline_ready_for_video_quality",
            current_next,
        )
    return _readiness(
        "awaiting_pipeline",
        "pipeline_not_ready_for_video_quality",
        current_next,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _source_bounds(
    result: VideoQualityResult,
    source_range: tuple[int, int] | None,
) -> tuple[int | None, int | None]:
    if source_range is not None:
        start_frame, end_frame = source_range
        if start_frame < 0 or end_frame <= start_frame:
            raise ValueError("source_range must be a non-empty half-open frame range")
        return start_frame, end_frame - 1
    if result.metrics.frame_count <= 0:
        return None, None
    return 0, result.metrics.frame_count - 1


def _source_evidence(
    *,
    result: VideoQualityResult,
    batch_root: Path,
    start_frame: int | None,
    end_frame: int | None,
    generator_version: str,
    allow_symlinked_sources: bool = False,
) -> EvidenceRef:
    path = relative_evidence_path(
        result.metrics.path,
        batch_root,
        allow_symlinked_sources=allow_symlinked_sources,
    )
    evidence_id = build_issue_id(
        asset_id=result.metrics.asset_id,
        module="video_quality",
        rule_id="video_quality.source_video",
        source_relative_path=path,
        coordinate_system=_COORDINATE_SYSTEM,
        start_frame=start_frame,
        end_frame=end_frame,
        hand_side=None,
        evidence_kind="source_video",
    )
    mime_type, _encoding = mimetypes.guess_type(result.metrics.path.name)
    return EvidenceRef(
        evidence_id=evidence_id,
        kind="source_video",
        path=path,
        coordinate_system=_COORDINATE_SYSTEM,
        start_frame=start_frame,
        end_frame=end_frame,
        mime_type=mime_type,
        generator_version=generator_version,
    )


def _adapt_issue(
    *,
    detail: Any,
    result: VideoQualityResult,
    source_path: str,
    evidence_id: str,
    start_frame: int | None,
    end_frame: int | None,
) -> Issue:
    context = dict(detail.context)
    context.update(
        {
            "coordinate_system": _COORDINATE_SYSTEM,
            "start_frame": start_frame,
            "end_frame": end_frame,
        }
    )
    issue_id = build_issue_id(
        asset_id=result.metrics.asset_id,
        module="video_quality",
        rule_id=detail.rule_id,
        source_relative_path=source_path,
        coordinate_system=_COORDINATE_SYSTEM,
        start_frame=start_frame,
        end_frame=end_frame,
        hand_side=None,
        evidence_kind="source_video",
    )
    return Issue(
        issue_id=issue_id,
        code=detail.code,
        severity=cast(Any, detail.severity),
        module="video_quality",
        issue_type=detail.issue_type,
        metric=detail.metric or detail.code,
        observed_value=detail.observed_value,
        operator=detail.operator or "triggered",
        boundary_value=detail.boundary_value,
        rule_id=detail.rule_id,
        needs_manual_review=detail.needs_manual_review,
        context=context,
        evidence_ids=(evidence_id,),
    )


def adapt_video_quality_result(
    *,
    result: VideoQualityResult,
    config: LoadedQcConfig,
    batch_root: Path,
    source_range: tuple[int, int] | None = None,
    allow_symlinked_sources: bool = False,
) -> ModuleResult:
    """Translate one batch or range result without reevaluating its metrics."""
    video_config = load_video_quality_config(config.path)
    legacy = asset_qc_result_to_json(result, video_config)
    payload = legacy["video_quality"]
    start_frame, end_frame = _source_bounds(result, source_range)
    evidence = _source_evidence(
        result=result,
        batch_root=batch_root,
        start_frame=start_frame,
        end_frame=end_frame,
        generator_version=str(payload["module_version"]),
        allow_symlinked_sources=allow_symlinked_sources,
    )
    details = (
        *result.evaluation.reason_details,
        *result.evaluation.warn_reason_details,
    )
    issues = tuple(
        _adapt_issue(
            detail=detail,
            result=result,
            source_path=evidence.path,
            evidence_id=evidence.evidence_id,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        for detail in details
    )
    metrics = {
        **dict(payload["metrics"]),
        "metadata": dict(payload["metadata"]),
        "sampling": dict(payload["sampling"]),
        "reference_quality": dict(legacy["reference_quality"]),
    }
    evaluation = dict(payload["evaluation"])
    evaluation["issue_ids"] = [issue.issue_id for issue in issues]
    return ModuleResult(
        module="video_quality",
        verdict=cast(Verdict, result.evaluation.decision),
        evaluation=evaluation,
        metrics=metrics,
        issues=issues,
        evidence=(evidence,),
        runtime={
            "stage": payload["stage"],
            "module_version": payload["module_version"],
            "errors": list(payload["errors"]),
        },
    )


def write_video_quality_result(
    *,
    context: AssetContext,
    result: VideoQualityResult,
    config: LoadedQcConfig,
    profile: str,
    expected_revision: int,
    next_module: str,
) -> dict[str, Any]:
    """Commit a video result through the shared revision-aware transaction."""
    module_result = adapt_video_quality_result(
        result=result,
        config=config,
        batch_root=context.batch_root,
        source_range=context.source_range,
    )
    return apply_module_result(
        context.report_path,
        context=context,
        config=config,
        profile=profile,
        result=module_result,
        expected_revision=expected_revision,
        next_module=next_module,
        now=_utc_now(),
    )
