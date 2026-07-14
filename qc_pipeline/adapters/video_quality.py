"""Adapt legacy video-quality results to the unified QC module contract."""

from __future__ import annotations

import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

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
) -> EvidenceRef:
    path = relative_evidence_path(result.metrics.path, batch_root)
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
