"""Real video-quality producer and unified-adapter runner bridge."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

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
    raw_result = VideoQualityResult(metrics, alignment, evaluation)
    adapted = adapt_video_quality_result(
        result=raw_result,
        config=config,
        batch_root=context.batch_root,
        source_range=context.source_range,
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
