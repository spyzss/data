"""Real precheck producer and unified-adapter runner bridges."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from qc_common.config import LoadedQcConfig
from qc_common.contracts import ModuleResult
from qc_common.module_registry import ModulePrerequisiteError, ModuleRunner
from qc_pipeline.context import AssetContext


MODULES = (
    "hdf5_text_info",
    "quality_hand",
    "keypoint_presence",
    "keypoint_morphology",
    "keypoint_temporal",
)


def _source_entry(context: AssetContext, name: str) -> Mapping[str, Any] | None:
    value = context.source_files.get(name)
    return value if isinstance(value, Mapping) else None


def _source_path(context: AssetContext, module: str, name: str) -> Path:
    entry = _source_entry(context, name)
    value = None if entry is None else entry.get("path")
    if value is None:
        raise ModulePrerequisiteError(module, f"source_files.{name}.path")
    path = context.batch_root / str(value)
    if not path.is_file():
        raise ModulePrerequisiteError(module, f"existing source_files.{name}.path")
    return path


def _source_relative_path(context: AssetContext) -> str:
    for name in ("hdf5", "parquet"):
        entry = _source_entry(context, name)
        if entry is not None and isinstance(entry.get("path"), str):
            return str(entry["path"])
    raise ModulePrerequisiteError("precheck", "hdf5 or parquet source path")


def _slice_clip(clip: Any, source_range: tuple[int, int]) -> Any:
    from qc_common.types import ClipInputs

    start, end = source_range

    def sliced(values: Any) -> Any:
        if values is None:
            return None
        if isinstance(values, Mapping):
            return {name: value[start:end] for name, value in values.items()}
        return values[start:end]

    selected = ClipInputs(
        episode_idx=clip.episode_idx,
        frame_indices=list(range(start, end)),
        keypoints=sliced(clip.keypoints),
        rotations=sliced(clip.rotations),
        confidences=sliced(clip.confidences),
        quality_hand=sliced(clip.quality_hand),
        masks=sliced(clip.masks),
        instruction=clip.instruction,
        text_label=clip.text_label,
        text_label_raw=clip.text_label_raw,
        text_label_parse_error=clip.text_label_parse_error,
        intrinsics=clip.intrinsics,
        fps=clip.fps,
    )
    for name in (
        "asset_id",
        "supplier_id",
        "source_path",
        "supplier_quality_signal",
        "morphology_status",
    ):
        if hasattr(clip, name):
            setattr(selected, name, getattr(clip, name))
    setattr(selected, "clip_start_frame", start)
    setattr(selected, "clip_end_frame", end - 1)
    return selected


def _load_clip(context: AssetContext, module: str) -> Any:
    canonical_episode = context.metadata.get("canonical_episode")
    if canonical_episode is not None:
        source_root = context.metadata.get("canonical_source_root")
        if not isinstance(source_root, str) or not source_root:
            raise ModulePrerequisiteError(module, "metadata.canonical_source_root")
        from canonical_qc.bridge import CanonicalQcBridge

        return CanonicalQcBridge(
            canonical_episode,
            source_root=Path(source_root),
        ).clip_inputs(context.source_range)

    declared_clip = context.metadata.get("clip_inputs")
    if declared_clip is not None:
        return declared_clip

    source_range = context.source_range
    supplier = str(
        context.metadata.get("supplier")
        or context.metadata.get("supplier_id")
        or ""
    ).lower()
    if supplier == "jdt" or _source_entry(context, "parquet") is not None:
        if source_range is None:
            raise ModulePrerequisiteError(module, "source_range for parquet precheck")
        from tools.run_manifest_precheck import load_jdt_clip

        start, end = source_range
        row = {
            **dict(context.metadata),
            "asset_id": context.asset_id,
            "start_frame": start,
            "end_frame": end - 1,
            "parquet_path": str(_source_path(context, module, "parquet")),
        }
        return load_jdt_clip(row, episode_idx=0)

    hdf5 = _source_path(context, module, "hdf5")
    if supplier == "deepreach":
        if source_range is None:
            raise ModulePrerequisiteError(module, "source_range for DeepReach precheck")
        from tools.run_manifest_precheck import load_deepreach_clip

        start, end = source_range
        row = {
            **dict(context.metadata),
            "asset_id": context.asset_id,
            "start_frame": start,
            "end_frame": end - 1,
            "hdf5_path": str(hdf5),
        }
        return load_deepreach_clip(row, episode_idx=0)

    from precheck.adapters import load_precheck_inputs

    clips = load_precheck_inputs(hdf5, episode_idx=0)
    if len(clips) != 1:
        raise ModulePrerequisiteError(module, "one precheck clip per asset")
    clip = clips[0]
    setattr(clip, "asset_id", context.asset_id)
    setattr(clip, "source_path", str(hdf5))
    if source_range is not None:
        clip = _slice_clip(clip, source_range)
    else:
        setattr(clip, "clip_start_frame", 0)
        setattr(clip, "clip_end_frame", max(clip.num_frames - 1, 0))
    return clip


def runner_for(module: str) -> ModuleRunner:
    if module not in MODULES:
        raise ValueError(f"unsupported precheck module: {module}")

    def run(context: AssetContext, config: LoadedQcConfig) -> ModuleResult:
        from dataclasses import replace as replace_check_result

        from precheck.runner import PrecheckRunner
        from qc_pipeline.adapters.precheck import (
            adapt_hdf5_text_info,
            adapt_keypoint_morphology,
            adapt_keypoint_presence,
            adapt_keypoint_temporal,
            adapt_quality_hand,
            precheck_config_from_unified,
        )
        from tools.run_manifest_precheck import (
            _map_candidate_window_to_source,
            _run_clip_in_local_coordinates,
        )

        output_dir = context.batch_root / ".qc_pipeline" / context.asset_id / module
        producer = PrecheckRunner(
            precheck_config_from_unified(
                config,
                module_names=[module],
                output_dir=output_dir,
            )
        )
        clip = _load_clip(context, module)
        results = _run_clip_in_local_coordinates(producer, clip)
        source_start = int(getattr(clip, "clip_start_frame", 0))
        results = [
            replace_check_result(row, frame_idx=row.frame_idx + source_start)
            if row.frame_idx >= 0
            else row
            for row in results
        ]
        source_path = _source_relative_path(context)
        common = {
            "asset_id": context.asset_id,
            "source_relative_path": source_path,
            "results": results,
            "config": config,
        }
        adapters = {
            "hdf5_text_info": adapt_hdf5_text_info,
            "quality_hand": adapt_quality_hand,
            "keypoint_presence": adapt_keypoint_presence,
            "keypoint_morphology": adapt_keypoint_morphology,
        }
        if module in adapters:
            return adapters[module](**common)

        candidates = [
            _map_candidate_window_to_source(
                candidate,
                asset_id=context.asset_id,
                supplier=str(context.metadata.get("supplier") or "unknown"),
                source_path=str(getattr(clip, "source_path", source_path)),
                clip_start_frame=source_start,
                clip_end_frame=int(getattr(clip, "clip_end_frame", source_start)),
                clip_frame_count=clip.num_frames,
            )
            for candidate in producer.candidate_window_records
        ]
        return adapt_keypoint_temporal(candidate_windows=candidates, **common)

    return run


__all__ = ["MODULES", "runner_for"]
