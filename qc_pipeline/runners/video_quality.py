"""Real video-quality producer and unified-adapter runner bridge."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from qc_common.config import LoadedQcConfig
from qc_common.contracts import ModuleResult
from qc_common.module_registry import ModulePrerequisiteError
from qc_pipeline.context import AssetContext


def _source_path(
    context: AssetContext,
    name: str,
    *,
    required: bool = True,
) -> Path | None:
    source = context.source_files.get(name)
    value = source.get("path") if isinstance(source, Mapping) else None
    if value is None:
        if required:
            raise ModulePrerequisiteError(
                "video_quality",
                f"source_files.{name}.path",
            )
        return None
    path = context.batch_root / str(value)
    if not path.is_file():
        raise ModulePrerequisiteError(
            "video_quality",
            f"existing source_files.{name}.path",
        )
    return path


def run(context: AssetContext, config: LoadedQcConfig) -> ModuleResult:
    from acceptance_pull.video_quality import (
        Hdf5Alignment,
        VideoQualityResult,
        analyze_video,
        analyze_video_frame_range,
        check_hdf5_alignment,
        evaluate_video_quality,
        load_video_quality_config,
    )
    from qc_pipeline.adapters.video_quality import adapt_video_quality_result

    video = _source_path(context, "video")
    hdf5 = _source_path(context, "hdf5", required=False)
    assert video is not None
    detector_config = load_video_quality_config(config.path)
    if context.source_range is None:
        metrics = analyze_video(video, detector_config, hdf5_path=hdf5)
        alignment = check_hdf5_alignment(
            video,
            context.batch_root,
            metrics,
            detector_config,
        )
    else:
        start, end = context.source_range
        analysis = analyze_video_frame_range(
            video,
            detector_config,
            start,
            end - 1,
            hdf5_path=hdf5,
        )
        metrics = analysis.metrics
        alignment = Hdf5Alignment(
            status="range_not_evaluated",
            hdf5_path=hdf5,
            hdf5_frame_count=None,
            frame_count_match=None,
            reason="logical range alignment validated by AssetContext bounds",
        )
    metrics = replace(metrics, asset_id=context.asset_id)
    evaluation = evaluate_video_quality(metrics, detector_config, alignment)
    return adapt_video_quality_result(
        result=VideoQualityResult(metrics, alignment, evaluation),
        config=config,
        batch_root=context.batch_root,
        source_range=context.source_range,
    )


__all__ = ["run"]
