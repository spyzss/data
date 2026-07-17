"""Real video-quality producer and unified-adapter runner bridge."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter
import re
from typing import Any

from qc_common.config import LoadedQcConfig
from qc_common.contracts import ModuleResult
from qc_common.module_registry import ModulePrerequisiteError
from qc_pipeline.context import AssetContext


_RANGE_DECODE_ERROR = re.compile(r"range_decode_failed:(\d+)").fullmatch


def _rebase_canonical_metrics(
    metrics: Any,
    *,
    physical_origin: int,
    logical_start: int,
    logical_end: int,
) -> Any:
    """Convert producer physical MP4 coordinates to Canonical frame numbers once."""
    intervals = []
    for interval in metrics.frozen_intervals:
        start = interval.start_frame - physical_origin
        end = interval.end_frame - physical_origin
        if start < logical_start or end < start or end >= logical_end:
            raise ValueError(
                "video producer returned a frame outside selected canonical range"
            )
        intervals.append(
            replace(
                interval,
                start_frame=start,
                end_frame=end,
                start_time_sec=start / metrics.fps if metrics.fps > 0 else 0.0,
                end_time_sec=(end + 1) / metrics.fps if metrics.fps > 0 else 0.0,
            )
        )
    errors: list[str] = []
    for error in metrics.errors:
        match = _RANGE_DECODE_ERROR(error)
        if match is None:
            errors.append(error)
            continue
        logical = int(match.group(1)) - physical_origin
        if not logical_start <= logical < logical_end:
            raise ValueError(
                "video producer error frame is outside selected canonical range"
            )
        errors.append(f"range_decode_failed:{logical}")
    return replace(metrics, frozen_intervals=tuple(intervals), errors=tuple(errors))


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


_IMPLEMENTATION_VERSION = "video-quality-producer-v1"


def _compute_video_quality(
    context: AssetContext,
    config: LoadedQcConfig,
) -> tuple[ModuleResult, object]:
    from acceptance_pull.video_quality import (
        Hdf5Alignment,
        VideoQualityResult,
        analyze_video,
        analyze_video_frame_range,
        check_hdf5_alignment,
        evaluate_video_quality,
        load_video_quality_config,
        _to_plain,
    )
    from qc_pipeline.adapters.video_quality import adapt_video_quality_result

    canonical_episode = context.metadata.get("canonical_episode")
    canonical = canonical_episode is not None
    if canonical:
        source_root = context.metadata.get("canonical_source_root")
        if not isinstance(source_root, str) or not source_root:
            raise ModulePrerequisiteError(
                "video_quality", "metadata.canonical_source_root"
            )
        from canonical_qc.bridge import CanonicalQcBridge

        bridge = CanonicalQcBridge(
            canonical_episode,
            source_root=Path(source_root),
        )
        video = None
        hdf5 = None
    else:
        video = _source_path(context, "video")
        hdf5 = _source_path(context, "hdf5", required=False)
    detector_config = load_video_quality_config(config.path)
    supplier = str(
        context.metadata.get("supplier")
        or context.metadata.get("supplier_id")
        or ""
    ).lower()
    supplier_overrides = config.module_parameters("video_quality").get(
        "supplier_overrides", {}
    )
    supplier_override = (
        supplier_overrides.get(supplier, {})
        if isinstance(supplier_overrides, Mapping)
        else {}
    )
    alignment_override = (
        supplier_override.get("hdf5_alignment", {})
        if isinstance(supplier_override, Mapping)
        else {}
    )
    if (
        not canonical
        and supplier == "potentia"
        and isinstance(alignment_override, Mapping)
        and alignment_override.get("enabled") is False
    ):
        detector_config = replace(
            detector_config,
            hdf5_alignment=replace(
                detector_config.hdf5_alignment,
                enabled=False,
            ),
        )
    if canonical:
        video = bridge.video_path()
    assert video is not None
    physical_range = (
        bridge.physical_video_range(context.source_range)
        if canonical
        else context.source_range
    )
    canonical_full_unshifted = canonical and physical_range == (
        0,
        canonical_episode.time_axis.frame_count,
    )
    if context.source_range is None and (not canonical or canonical_full_unshifted):
        metrics = analyze_video(video, detector_config, hdf5_path=hdf5)
        if canonical:
            alignment = Hdf5Alignment(
                status="matched",
                hdf5_path=None,
                hdf5_frame_count=canonical_episode.time_axis.frame_count,
                frame_count_match=True,
                frame_count_delta=0,
                frame_count_delta_ratio=0.0,
                reason="validated by CanonicalQcEpisode input boundary",
            )
        else:
            alignment = check_hdf5_alignment(
                video,
                context.batch_root,
                metrics,
                detector_config,
            )
    else:
        assert physical_range is not None
        start, end = physical_range
        analysis = analyze_video_frame_range(
            video,
            detector_config,
            start,
            end - 1,
            hdf5_path=hdf5,
        )
        metrics = analysis.metrics
        if canonical:
            logical_start, logical_end = context.source_range or (
                0,
                canonical_episode.time_axis.frame_count,
            )
            alignment = Hdf5Alignment(
                status="matched",
                hdf5_path=None,
                hdf5_frame_count=logical_end - logical_start,
                frame_count_match=True,
                frame_count_delta=0,
                frame_count_delta_ratio=0.0,
                reason="validated by CanonicalQcEpisode input boundary",
            )
        else:
            alignment = Hdf5Alignment(
                status="range_not_evaluated",
                hdf5_path=hdf5,
                hdf5_frame_count=None,
                frame_count_match=None,
                reason="logical range alignment validated by AssetContext bounds",
            )
    if canonical:
        physical_origin, _ = canonical_episode.main_video.source_frame_range
        logical_start, logical_end = context.source_range or (
            0,
            canonical_episode.time_axis.frame_count,
        )
        metrics = _rebase_canonical_metrics(
            metrics,
            physical_origin=physical_origin,
            logical_start=logical_start,
            logical_end=logical_end,
        )
        bridge.verify_sources()
    metrics = replace(metrics, asset_id=context.asset_id)
    evaluation = evaluate_video_quality(metrics, detector_config, alignment)
    raw_result = VideoQualityResult(metrics, alignment, evaluation)
    adapted = adapt_video_quality_result(
        result=raw_result,
        config=config,
        batch_root=context.batch_root,
        source_range=context.source_range,
        allow_symlinked_sources=context.allow_symlinked_sources,
    )
    return adapted, _to_plain(raw_result)


def _with_artifact_runtime(
    result: ModuleResult,
    *,
    state: str,
    elapsed_seconds: float,
    fingerprint_sha256: str,
) -> ModuleResult:
    return replace(
        result,
        runtime={
            **dict(result.runtime),
            "artifact_state": state,
            "elapsed_seconds": float(elapsed_seconds),
            "fingerprint_sha256": fingerprint_sha256,
        },
    )


def run(context: AssetContext, config: LoadedQcConfig) -> ModuleResult:
    from qc_pipeline.artifacts import (
        artifact_for,
        build_run_fingerprint,
        canonical_sha256,
        module_result_from_dict,
        promote_artifact,
        reusable_artifact,
        staged_artifact,
        write_run_config,
    )

    started = perf_counter()
    source_names = tuple(
        name for name in ("video", "hdf5") if name in context.source_files
    )
    fingerprint = build_run_fingerprint(
        context=context,
        producer="video_quality",
        config=config,
        module_names=("video_quality",),
        source_names=source_names,
        implementation_version=_IMPLEMENTATION_VERSION,
    )
    fingerprint_sha256 = canonical_sha256(fingerprint)
    artifact = artifact_for(context, "video_quality")
    if bool(context.metadata.get("reuse_artifacts", True)) and reusable_artifact(
        artifact, fingerprint
    ):
        payload = json.loads(
            (artifact.directory / "video_quality_result.json").read_text(
                encoding="utf-8"
            )
        )
        result = module_result_from_dict(payload["module_result"])
        return _with_artifact_runtime(
            result,
            state="reused",
            elapsed_seconds=perf_counter() - started,
            fingerprint_sha256=fingerprint_sha256,
        )

    result, raw_result = _compute_video_quality(context, config)
    elapsed = perf_counter() - started
    with staged_artifact(artifact) as staging:
        (staging / "video_quality_result.json").write_text(
            json.dumps(
                {"raw_result": raw_result, "module_result": result.to_dict()},
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        write_run_config(
            staging,
            producer="video_quality",
            outcome="completed",
            fingerprint=fingerprint,
            elapsed_seconds=elapsed,
        )
        promote_artifact(staging, artifact)
    return _with_artifact_runtime(
        result,
        state="computed",
        elapsed_seconds=elapsed,
        fingerprint_sha256=fingerprint_sha256,
    )


__all__ = ["run"]
