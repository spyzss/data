"""Real manifest SAM3 producer and unified-adapter runner bridge."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
import json
from pathlib import Path
import shutil
import tempfile
from time import perf_counter
from typing import Any, Literal

from qc_common.config import LoadedQcConfig
from qc_common.contracts import ModuleResult
from qc_common.module_registry import (
    ModuleAdapterMissingError,
    ModuleBlockedError,
    ModuleInputError,
    ModulePrerequisiteError,
    ModuleRunner,
)
from qc_pipeline.context import AssetContext
from qc_pipeline.artifacts import artifact_for
from qc_pipeline.sam3_runtime import SegmenterProvider


_IMPLEMENTATION_VERSION = "sam3-containment-producer-v2"
_MODEL_IDENTITY_FILES = ("config.json", "model.safetensors", "sam3.pt")
_MODEL_HASH_FILES = ("config.json",)


@dataclass(frozen=True)
class _QyContainmentInputs:
    video_path: Path
    observations_path: Path
    timebase_path: Path | None
    primary_camera: str
    timebase: Mapping[str, Any]
    direct_2d: Mapping[tuple[int, str], Mapping[str, Any]]
    sampled_frames: Mapping[int, tuple[int, ...]]


def _source_entry(context: AssetContext, name: str) -> Mapping[str, Any] | None:
    source = context.source_files.get(name)
    return source if isinstance(source, Mapping) else None


def _source_path(
    context: AssetContext,
    name: str,
    *,
    required: bool = True,
    expected_type: Literal["file", "directory"] = "file",
) -> Path | None:
    entry = _source_entry(context, name)
    value = entry.get("path") if entry is not None else None
    if value is None:
        if required:
            raise ModulePrerequisiteError(
                "sam3_containment",
                f"source_files.{name}.path",
            )
        return None
    path = context.batch_root / str(value)
    if expected_type == "file":
        valid = path.is_file()
    elif expected_type == "directory":
        valid = path.is_dir()
    else:
        raise ValueError(f"unsupported source path type: {expected_type}")
    if not valid:
        raise ModulePrerequisiteError(
            "sam3_containment",
            f"existing {expected_type} source_files.{name}.path",
        )
    return path


def _records_for_asset(path: Path, asset_id: str) -> list[dict[str, Any]]:
    from tools.run_manifest_sam3_containment import read_records

    return [row for row in read_records(path) if str(row.get("asset_id")) == asset_id]


def _validated_candidates(
    context: AssetContext,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    clip_start: int | None = None
    clip_end: int | None = None
    if context.source_range is not None:
        clip_start, exclusive_end = context.source_range
        clip_end = exclusive_end - 1
    for index, row in enumerate(rows):
        if row.get("sam3_eligible") is not True:
            continue
        if str(row.get("asset_id") or "") != context.asset_id:
            raise ModuleInputError(
                "sam3_containment",
                f"candidate {index} asset_id does not match {context.asset_id}",
            )
        start = row.get("start_frame")
        end = row.get("end_frame")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
        ):
            raise ModuleInputError(
                "sam3_containment",
                f"candidate {index} source bounds must be integers",
            )
        if start > end:
            raise ModuleInputError(
                "sam3_containment",
                f"candidate {index} start_frame exceeds end_frame",
            )
        coordinate_space = str(row.get("coordinate_space") or "source").lower()
        frame_coordinates = str(
            row.get("frame_coordinate_system") or "source_inclusive"
        ).lower()
        if coordinate_space != "source" or frame_coordinates != "source_inclusive":
            raise ModuleInputError(
                "sam3_containment",
                f"candidate {index} must use source_inclusive coordinates",
            )
        if (
            clip_start is not None
            and clip_end is not None
            and not (clip_start <= start <= end <= clip_end)
        ):
            raise ModuleInputError(
                "sam3_containment",
                f"candidate {index} bounds {start}..{end} outside clip "
                f"{clip_start}..{clip_end}",
            )
        if "source_start_frame" in row and row["source_start_frame"] != start:
            raise ModuleInputError(
                "sam3_containment",
                f"candidate {index} source_start_frame disagrees with start_frame",
            )
        if "source_end_frame" in row and row["source_end_frame"] != end:
            raise ModuleInputError(
                "sam3_containment",
                f"candidate {index} source_end_frame disagrees with end_frame",
            )
        validated.append(dict(row))
    return validated


def _dr_has_hard_presence_invalid(row: Mapping[str, Any]) -> bool:
    if (
        row.get("pipeline_module") != "keypoint_presence"
        or row.get("check") != "keypoint_missing"
    ):
        return False
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping):
        metrics = {}
    for side in ("left", "right"):
        if metrics.get(f"keypoint_existence_invalid_{side}") is True:
            return True
        missing_value = metrics.get(f"missing_keypoint_count_{side}")
        valid_value = metrics.get(f"valid_keypoint_count_{side}")
        try:
            if float(missing_value) > 0.0:
                return True
        except (TypeError, ValueError):
            pass
        try:
            if float(valid_value) < 21.0:
                return True
        except (TypeError, ValueError):
            pass
    return row.get("flag") is True and row.get("severity") == "fail"


def _with_artifact_runtime(
    result: ModuleResult,
    *,
    state: str,
    elapsed_seconds: float,
    fingerprint_sha256: str,
    overlay_input_recipe: Mapping[str, Any] | None = None,
) -> ModuleResult:
    overlay_runtime = (
        {}
        if overlay_input_recipe is None
        else {"overlay_input_recipe": dict(overlay_input_recipe)}
    )
    return replace(
        result,
        runtime={
            **dict(result.runtime),
            "artifact_state": state,
            "elapsed_seconds": float(elapsed_seconds),
            "fingerprint_sha256": fingerprint_sha256,
            **overlay_runtime,
        },
    )


def _candidate_source_frames(
    candidate_rows: list[dict[str, Any]],
) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                source_frame
                for candidate in candidate_rows
                for source_frame in range(
                    int(candidate["start_frame"]),
                    int(candidate["end_frame"]) + 1,
                )
            }
        )
    )


def _candidate_intervals(candidate_rows: list[dict[str, Any]]) -> tuple[tuple[int, int], ...]:
    intervals = sorted(
        (int(row["start_frame"]), int(row["end_frame"]) + 1)
        for row in candidate_rows
    )
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _linear_mapping_ranges(
    candidate_rows: list[dict[str, Any]],
    video_frame_for_source: Callable[[int], int | None],
) -> list[dict[str, int]] | None:
    ranges: list[dict[str, int]] = []
    for start, end in _candidate_intervals(candidate_rows):
        video_start = video_frame_for_source(start)
        if isinstance(video_start, bool) or not isinstance(video_start, int) or video_start < 0:
            return None
        for source_frame in range(start + 1, end):
            if video_frame_for_source(source_frame) != video_start + source_frame - start:
                return None
        ranges.append(
            {
                "start_frame": start,
                "end_frame_exclusive": end,
                "video_start_frame": video_start,
            }
        )
    return ranges or None


def _recipe_base(
    *,
    context: AssetContext,
    candidate_rows: list[dict[str, Any]],
    video_identity: str,
    producer_fingerprint_sha256: str,
    mapping_ranges: list[dict[str, int]],
    keypoint_reference: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "sam3_overlay_input.v2",
        "asset_id": context.asset_id,
        "producer_fingerprint_sha256": producer_fingerprint_sha256,
        "video_source": "video",
        "video_identity": video_identity,
        "candidate_intervals": [list(value) for value in _candidate_intervals(candidate_rows)],
        "source_mapping": {
            "schema_version": "linear_ranges.v1",
            "ranges": mapping_ranges,
        },
        "keypoints_2d_reference": dict(keypoint_reference),
    }


def _write_keypoint_sidecar(
    *,
    context: AssetContext,
    producer_fingerprint_sha256: str,
    frames: list[dict[str, Any]],
) -> dict[str, Any]:
    from qc_pipeline.artifacts import file_sha256

    root = context.batch_root / ".qc_pipeline" / context.asset_id / "sam3_overlay_inputs"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{producer_fingerprint_sha256.removeprefix('sha256:')}.json"
    payload = json.dumps(
        {
            "schema_version": "sam3_overlay_keypoints.v1",
            "asset_id": context.asset_id,
            "frames": frames,
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=root, prefix=".overlay-input-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            import os

            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(destination)
    return {
        "schema_version": "json_keypoints.v1",
        "relative_path": destination.relative_to(context.batch_root).as_posix(),
        "sha256": file_sha256(destination),
        "size_bytes": destination.stat().st_size,
    }


def _jdt_overlay_input_recipe(
    *,
    context: AssetContext,
    candidate_rows: list[dict[str, Any]],
    manifest_row: Mapping[str, Any],
    producer_fingerprint_sha256: str,
) -> dict[str, Any]:
    from qc_pipeline.artifacts import file_sha256

    video_path = _source_path(context, "video")
    parquet_path = _source_path(context, "parquet")
    assert video_path is not None
    assert parquet_path is not None
    fields: dict[str, str] = {}
    for side in ("left", "right"):
        value = manifest_row.get(f"{side}_hand_2d_field")
        if not isinstance(value, str) or not value.strip():
            raise ModuleInputError(
                "sam3_containment",
                f"manifest {side}_hand_2d_field is unavailable for overlay",
            )
        fields[side] = value.strip()
    intervals = _candidate_intervals(candidate_rows)
    if not intervals:
        raise ModuleInputError(
            "sam3_containment", "overlay source mapping has no candidate frames"
        )
    mapping_ranges = [
        {
            "start_frame": start,
            "end_frame_exclusive": end,
            "video_start_frame": start,
        }
        for start, end in intervals
    ]
    return _recipe_base(
        context=context,
        candidate_rows=candidate_rows,
        video_identity=file_sha256(video_path),
        producer_fingerprint_sha256=producer_fingerprint_sha256,
        mapping_ranges=mapping_ranges,
        keypoint_reference={
            "schema_version": "parquet_columns.v2",
            "source": "parquet",
            "sha256": file_sha256(parquet_path),
            "size_bytes": parquet_path.stat().st_size,
            "row_mapping": "source_frame_index",
            "fields": fields,
        },
    )


def _required_sides_by_source_frame(
    candidate_rows: list[dict[str, Any]],
) -> dict[int, set[str]]:
    required: dict[int, set[str]] = {}
    for candidate in candidate_rows:
        sides = _candidate_hand_sides(candidate)
        for source_frame in range(
            int(candidate["start_frame"]), int(candidate["end_frame"]) + 1
        ):
            required.setdefault(source_frame, set()).update(sides)
    return required


def _dr_overlay_input_recipe(
    *,
    context: AssetContext,
    candidate_rows: list[dict[str, Any]],
    hdf5_path: Path,
    video_path: Path,
    validation: Any,
    producer_fingerprint_sha256: str,
) -> dict[str, Any] | None:
    import numpy as np

    from acceptance_pull.supplier_adapters.deepreach_projection import (
        project_dr_hands_for_frame,
    )
    from qc_pipeline.artifacts import file_sha256

    declared_video = _source_path(context, "video", required=False)
    if (
        declared_video is None
        or declared_video.resolve() != video_path.resolve()
        or context.source_range is None
    ):
        return None
    clip_start, _ = context.source_range
    sidecar_frames: list[dict[str, Any]] = []
    for source_frame, required_sides in _required_sides_by_source_frame(
        candidate_rows
    ).items():
        projected = project_dr_hands_for_frame(
            hdf5_path,
            source_frame=source_frame,
            clip_start_frame=clip_start,
            calibration=validation.calibration,
        )
        frame_points: dict[str, list[list[float]]] = {}
        for side in required_sides:
            hand = projected.get(side)
            if not isinstance(hand, Mapping):
                return None
            pixels = np.asarray(hand.get("pixels"), dtype=np.float64)
            valid = np.asarray(hand.get("valid"), dtype=bool)
            if (
                pixels.ndim != 2
                or pixels.shape[1] != 2
                or valid.shape != pixels.shape[:1]
            ):
                return None
            usable = valid & np.isfinite(pixels).all(axis=1)
            selected = pixels[usable]
            if selected.size == 0:
                return None
            frame_points[side] = selected.tolist()
        sidecar_frames.append(
            {"source_frame": source_frame, "keypoints": frame_points}
        )
    mapping_ranges = _linear_mapping_ranges(
        candidate_rows,
        lambda source_frame: source_frame - clip_start,
    )
    if mapping_ranges is None or not sidecar_frames:
        return None
    reference = _write_keypoint_sidecar(
        context=context,
        producer_fingerprint_sha256=producer_fingerprint_sha256,
        frames=sidecar_frames,
    )
    return _recipe_base(
        context=context,
        candidate_rows=candidate_rows,
        video_identity=file_sha256(video_path),
        producer_fingerprint_sha256=producer_fingerprint_sha256,
        mapping_ranges=mapping_ranges,
        keypoint_reference=reference,
    )


def _qy_overlay_input_recipe(
    *,
    context: AssetContext,
    candidate_rows: list[dict[str, Any]],
    inputs: _QyContainmentInputs,
    producer_fingerprint_sha256: str,
) -> dict[str, Any] | None:
    import numpy as np

    from qc_pipeline.artifacts import file_sha256

    declared_video = _source_path(context, "video", required=False)
    if (
        declared_video is None
        or declared_video.resolve() != inputs.video_path.resolve()
    ):
        return None
    video_by_source: dict[int, int] = {}
    sidecar_frames: list[dict[str, Any]] = []
    for source_frame, required_sides in _required_sides_by_source_frame(
        candidate_rows
    ).items():
        video_frames: set[int] = set()
        frame_points: dict[str, list[list[float]]] = {}
        for side in required_sides:
            direct = inputs.direct_2d.get((source_frame, side))
            if not isinstance(direct, Mapping):
                return None
            raw_video_frame = direct.get("video_frame")
            if isinstance(raw_video_frame, bool) or not isinstance(
                raw_video_frame, int
            ):
                return None
            points = np.asarray(direct.get("keypoints_2d"), dtype=np.float64)
            if (
                points.ndim != 2
                or points.shape[1] != 2
                or points.size == 0
                or not np.isfinite(points).all()
            ):
                return None
            video_frames.add(raw_video_frame)
            frame_points[side] = points.tolist()
        if len(video_frames) != 1:
            return None
        video_by_source[source_frame] = next(iter(video_frames))
        sidecar_frames.append(
            {"source_frame": source_frame, "keypoints": frame_points}
        )
    mapping_ranges = _linear_mapping_ranges(
        candidate_rows,
        video_by_source.get,
    )
    if mapping_ranges is None or not sidecar_frames:
        return None
    reference = _write_keypoint_sidecar(
        context=context,
        producer_fingerprint_sha256=producer_fingerprint_sha256,
        frames=sidecar_frames,
    )
    return _recipe_base(
        context=context,
        candidate_rows=candidate_rows,
        video_identity=file_sha256(inputs.video_path),
        producer_fingerprint_sha256=producer_fingerprint_sha256,
        mapping_ranges=mapping_ranges,
        keypoint_reference=reference,
    )


def _canonical_overlay_input_recipe(
    *,
    context: AssetContext,
    candidate_rows: list[dict[str, Any]],
    episode: Any,
    video_path: Path,
    producer_fingerprint_sha256: str,
) -> dict[str, Any] | None:
    """Build a recipe only when canonical logical-to-physical mapping is proven."""
    import numpy as np

    from qc_pipeline.artifacts import file_sha256

    declared_video = _source_path(context, "video", required=False)
    if declared_video is None or declared_video.resolve() != video_path.resolve():
        return None
    try:
        logical_frame_count = int(episode.time_axis.frame_count)
        physical_start, physical_end = episode.main_video.source_frame_range
    except (AttributeError, TypeError, ValueError):
        return None
    if (
        isinstance(physical_start, bool)
        or not isinstance(physical_start, int)
        or isinstance(physical_end, bool)
        or not isinstance(physical_end, int)
        or physical_start < 0
        or physical_end - physical_start != logical_frame_count
    ):
        return None
    points = np.asarray(episode.observation.hand_keypoints_2d)
    valid = np.asarray(episode.observation.hand_joint_valid_2d)
    if (
        points.shape != (logical_frame_count, 2, 21, 2)
        or valid.shape != (logical_frame_count, 2, 21)
    ):
        return None
    required: dict[int, set[str]] = {}
    for candidate in candidate_rows:
        start = candidate.get("start_frame")
        end = candidate.get("end_frame")
        if (
            str(candidate.get("asset_id") or "") != context.asset_id
            or isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or not (0 <= start <= end < logical_frame_count)
        ):
            return None
        try:
            sides = _candidate_hand_sides(candidate)
        except ValueError:
            return None
        for source_frame in range(start, end + 1):
            required.setdefault(source_frame, set()).update(sides)
    if not required:
        return None
    sidecar_frames: list[dict[str, Any]] = []
    side_indices = {"left": 0, "right": 1}
    for source_frame, required_sides in required.items():
        frame_points: dict[str, list[list[float]]] = {}
        for side in required_sides:
            side_index = side_indices[side]
            side_points = points[source_frame, side_index]
            usable = valid[source_frame, side_index] & np.isfinite(side_points).all(
                axis=1
            )
            selected = side_points[usable]
            if selected.size == 0:
                return None
            frame_points[side] = selected.astype(np.float64, copy=False).tolist()
        sidecar_frames.append(
            {"source_frame": source_frame, "keypoints": frame_points}
        )
    mapping_ranges = _linear_mapping_ranges(
        candidate_rows,
        lambda source_frame: physical_start + source_frame,
    )
    if mapping_ranges is None:
        return None
    reference = _write_keypoint_sidecar(
        context=context,
        producer_fingerprint_sha256=producer_fingerprint_sha256,
        frames=sidecar_frames,
    )
    return _recipe_base(
        context=context,
        candidate_rows=candidate_rows,
        video_identity=file_sha256(video_path),
        producer_fingerprint_sha256=producer_fingerprint_sha256,
        mapping_ranges=mapping_ranges,
        keypoint_reference=reference,
    )


def _publish_sam3_artifact(
    *,
    context: AssetContext,
    frame_results: list[dict[str, Any]],
    window_summaries: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
    producer_run_config: Mapping[str, Any],
    producer_root: Path,
    fingerprint: Mapping[str, Any],
    elapsed_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from qc_common.io import write_json_records
    from qc_pipeline.artifacts import (
        promote_artifact,
        staged_artifact,
        write_run_config,
    )

    artifact = artifact_for(context, "sam3_containment")
    rewritten: list[dict[str, Any]] = []
    with staged_artifact(artifact) as staging:
        evidence_dir = staging / "evidence"
        for index, source_row in enumerate(evidence_rows):
            row = dict(source_row)
            value = row.get("source_path")
            if not isinstance(value, str) or not value.strip():
                raise ValueError("SAM3 evidence row has no source_path")
            source = Path(value)
            if not source.is_absolute():
                source = producer_root / source
            if not source.is_file():
                raise ValueError(f"SAM3 evidence file does not exist: {source}")
            evidence_dir.mkdir(parents=True, exist_ok=True)
            destination = evidence_dir / f"{index:04d}-{source.name}"
            shutil.copy2(source, destination)
            final_path = artifact.directory / "evidence" / destination.name
            row["source_path"] = final_path.relative_to(context.batch_root).as_posix()
            rewritten.append(row)
        write_json_records(frame_results, staging / "frame_results.json")
        write_json_records(window_summaries, staging / "window_results.json")
        write_json_records(failures, staging / "failures.json")
        write_json_records(rewritten, staging / "evidence_manifest.json")
        (staging / "producer_run_config.json").write_text(
            json.dumps(
                dict(producer_run_config),
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
            producer="sam3_containment",
            outcome="completed",
            fingerprint=fingerprint,
            elapsed_seconds=elapsed_seconds,
        )
        promote_artifact(staging, artifact)
    return window_summaries, rewritten


def _dr_projection_inputs(
    context: AssetContext,
    config: LoadedQcConfig,
) -> tuple[Path, Path, Path, Path, Any, Mapping[str, Any]]:
    from acceptance_pull.supplier_adapters.deepreach_projection import (
        validate_heuristic_head_projection_contract,
        validate_head_projection_contract,
    )

    hdf5_path = _source_path(context, "hdf5")
    video_path = _source_path(context, "video", required=False)
    if video_path is None:
        video_path = _source_path(context, "head_video")
    assert hdf5_path is not None
    assert video_path is not None
    if context.metadata.get("projection_mode") == "approx_pinhole_from_hfov":
        validation = validate_heuristic_head_projection_contract(
            hdf5_path=hdf5_path,
            video_path=video_path,
            source_range=context.source_range,
            reference_dataset=str(
                context.metadata.get("hdf5_reference_dataset") or ""
            ),
            primary_camera=str(context.metadata.get("primary_camera") or ""),
            metadata=context.metadata,
        )
        lineage = Path(validation.calibration.source)
        return hdf5_path, video_path, lineage, lineage, validation, {}
    calibration_path = _source_path(context, "calibration")
    trajectory_path = _source_path(context, "trajectory")
    assert calibration_path is not None
    assert trajectory_path is not None
    suppliers = config.module_parameters("supplier_data_audit").get("suppliers")
    dr_config = (
        suppliers.get("dr", {}) if isinstance(suppliers, Mapping) else {}
    )
    mapping = dr_config.get("mapping", {}) if isinstance(dr_config, Mapping) else {}
    if not isinstance(mapping, Mapping):
        mapping = {}
    validation = validate_head_projection_contract(
        hdf5_path=hdf5_path,
        video_path=video_path,
        calibration_path=calibration_path,
        trajectory_path=trajectory_path,
        source_range=context.source_range,
        reference_dataset=str(
            context.metadata.get("hdf5_reference_dataset") or ""
        ),
        primary_camera=str(context.metadata.get("primary_camera") or ""),
        content_id=(
            str(context.metadata.get("content_id"))
            if context.metadata.get("content_id") is not None
            else None
        ),
        calibration_mapping_status=str(
            context.metadata.get("calibration_mapping_status") or ""
        ),
        projection_validation_status=str(
            context.metadata.get("projection_validation_status") or ""
        ),
        mapping_status=str(
            dr_config.get("mapping_status")
            if isinstance(dr_config, Mapping)
            else ""
        ),
        mapping=mapping,
    )
    return (
        hdf5_path,
        video_path,
        calibration_path,
        trajectory_path,
        validation,
        dr_config if isinstance(dr_config, Mapping) else {},
    )


def _candidate_hand_sides(candidate: Mapping[str, Any]) -> tuple[str, ...]:
    side = str(candidate.get("hand_side") or "both").lower()
    if side == "both":
        return ("left", "right")
    if side in {"left", "right"}:
        return (side,)
    raise ValueError(f"invalid candidate hand_side: {side}")


def _run_dr_containment(
    *,
    context: AssetContext,
    config: LoadedQcConfig,
    candidate_rows: list[dict[str, Any]],
    hdf5_path: Path,
    video_path: Path,
    calibration_path: Path,
    validation: Any,
    segmenter: Any,
    staging_root: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    import cv2

    from acceptance_pull.supplier_adapters.deepreach_projection import (
        project_dr_hands_for_frame,
    )
    from qc_common.keypoints import acceptance_joint_names
    from tools.run_manifest_sam3_containment import (
        DEFAULT_QUERIES,
        SAM3_CONFIG,
        configured_sam3_thresholds,
        sample_manifest_window_frames,
    )
    from tools.sam3_keypoint_containment import (
        aggregate_window_containment_summaries,
        candidate_window_metadata,
        score_keypoints_against_masks,
        write_combined_overlay_image,
    )

    if context.source_range is None:
        raise ModulePrerequisiteError(
            "sam3_containment", "source_range for DR containment"
        )
    clip_start, clip_end_exclusive = context.source_range
    clip_end = clip_end_exclusive - 1
    frame_thresholds, window_thresholds = configured_sam3_thresholds(config)
    queries = [
        value.strip() for value in DEFAULT_QUERIES.split(",") if value.strip()
    ]
    joint_names = {
        side: [
            name for name in acceptance_joint_names() if name.startswith(side)
        ]
        for side in ("left", "right")
    }
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ModuleInputError("sam3_containment", "DR head video is unreadable")
    frame_cache: dict[int, Any] = {}
    mask_cache: dict[int, list[Any]] = {}
    frame_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    try:
        heuristic_projection = validation.status == "heuristic_ready"
        for window_index, candidate in enumerate(candidate_rows):
            sampled_frames = sample_manifest_window_frames(
                candidate,
                clip_start_frame=clip_start,
                clip_end_frame=clip_end,
                frames_per_window=5 if heuristic_projection else 3,
            )
            for source_frame in sampled_frames:
                local_frame = source_frame - clip_start
                if source_frame not in frame_cache:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, local_frame)
                    ok, frame_bgr = capture.read()
                    if not ok or frame_bgr is None:
                        raise ModuleInputError(
                            "sam3_containment",
                            f"DR head video frame {source_frame} is unreadable",
                        )
                    frame_cache[source_frame] = frame_bgr[..., ::-1].copy()
                frame = frame_cache[source_frame]
                if source_frame not in mask_cache:
                    mask_cache[source_frame] = segmenter.segment_frame(
                        frame,
                        queries,
                        dict(SAM3_CONFIG),
                    )
                masks = mask_cache[source_frame]
                projected = project_dr_hands_for_frame(
                    hdf5_path,
                    source_frame=source_frame,
                    clip_start_frame=clip_start,
                    calibration=validation.calibration,
                )
                combined_hands: dict[str, dict[str, Any]] = {}
                for side in ("left", "right"):
                    hand = projected[side]
                    containment, _union_mask, valid, inside = (
                        score_keypoints_against_masks(
                            frame=frame,
                            pixels=hand["pixels"],
                            joint_names=joint_names[side],
                            masks=masks,
                            valid=hand["valid"],
                            **frame_thresholds,
                        )
                    )
                    combined_hands[side] = {
                        "pixels": hand["pixels"],
                        "valid": valid,
                        "inside": inside,
                        "joint_names": joint_names[side],
                    }
                    if side not in _candidate_hand_sides(candidate):
                        continue
                    frame_rows.append(
                        {
                            "clip_id": context.asset_id,
                            "asset_id": context.asset_id,
                            "episode_idx": 0,
                            "hdf5_path": str(hdf5_path),
                            "video_path": str(video_path),
                            "calibration_path": str(validation.calibration.source),
                            "frame_idx": source_frame,
                            "source_frame_idx": source_frame,
                            "local_frame_idx": local_frame,
                            "clip_start_frame": clip_start,
                            "clip_end_frame": clip_end,
                            "candidate_start_frame": candidate["start_frame"],
                            "candidate_end_frame": candidate["end_frame"],
                            "coordinate_space": "source",
                            "projection_mode": (
                                "approx_pinhole_from_hfov"
                                if heuristic_projection
                                else "dr_head_direct_calibration"
                            ),
                            "projection_mode_used": (
                                "approx_pinhole_from_hfov"
                                if heuristic_projection
                                else "dr_head_direct_calibration"
                            ),
                            "calibration_status": (
                                "heuristic" if heuristic_projection else "verified"
                            ),
                            "projection_validation_status": (
                                "pending_visual_validation"
                                if heuristic_projection
                                else "validated"
                            ),
                            "distortion_applied": False,
                            "camera_trajectory_applied": False,
                            "projection_input_status": hand[
                                "projection_input_status"
                            ],
                            "projection_input_reason": hand[
                                "projection_input_reason"
                            ],
                            "image_width": int(frame.shape[1]),
                            "image_height": int(frame.shape[0]),
                            "camera_id": "main",
                            "camera_name": "head",
                            "hand_side": side,
                            "candidate_hand_side": candidate.get(
                                "hand_side", "both"
                            ),
                            **containment,
                            **candidate_window_metadata(candidate),
                        }
                    )
                overlay_path = write_combined_overlay_image(
                    frame=frame,
                    hands=combined_hands,
                    clip_id=(
                        f"{context.asset_id}_window_"
                        f"{candidate['start_frame']}_{candidate['end_frame']}_combined"
                    ),
                    frame_idx=source_frame,
                    output_dir=staging_root / "combined_overlays",
                )
                evidence_rows.append(
                    {
                        "review_id": "",
                        "supplier_id": "dr",
                        "asset_id": context.asset_id,
                        "window_start_frame": candidate["start_frame"],
                        "window_end_frame": candidate["end_frame"],
                        "frame_idx": source_frame,
                        "source_module": "sam3_containment",
                        "evidence_type": "combined_overlay",
                        "hand_side": "both",
                        "source_path": str(Path(overlay_path).resolve()),
                        "metadata_json": json.dumps(
                            {
                                "camera_name": "head",
                                "source_frame_idx": source_frame,
                                "local_frame_idx": local_frame,
                                "coordinate_space": "source",
                                "projection_mode": (
                                    "approx_pinhole_from_hfov"
                                    if heuristic_projection
                                    else "dr_head_direct_calibration"
                                ),
                                "calibration_status": (
                                    "heuristic"
                                    if heuristic_projection
                                    else "verified"
                                ),
                            },
                            sort_keys=True,
                        ),
                    }
                )
    finally:
        capture.release()
    window_summaries = aggregate_window_containment_summaries(
        frame_rows,
        **window_thresholds,
    )
    if heuristic_projection:
        for summary in window_summaries:
            summary["raw_window_containment_verdict"] = summary.get(
                "window_containment_verdict"
            )
            summary["window_containment_verdict"] = "projection_review"
            summary["routing_reason"] = (
                "dr_heuristic_projection_requires_human_review"
            )
            summary["projection_mode"] = "approx_pinhole_from_hfov"
            summary["calibration_status"] = "heuristic"
            summary["projection_validation_status"] = (
                "pending_visual_validation"
            )
            summary["distortion_applied"] = False
            summary["camera_trajectory_applied"] = False
    producer_run_config = {
        "supplier": "dr",
        "primary_camera": "head",
        "keypoint_source": "DR HDF5 hand/<side>/joints3d",
        "projection_status": validation.status,
        "projection_reason": validation.reason,
        "trajectory_usage": validation.trajectory_usage,
        "projection_mode": (
            "approx_pinhole_from_hfov"
            if heuristic_projection
            else "dr_head_direct_calibration"
        ),
        "calibration_status": (
            "heuristic" if heuristic_projection else "verified"
        ),
        "projection_validation_status": (
            "pending_visual_validation"
            if heuristic_projection
            else "validated"
        ),
        "distortion_applied": False,
        "camera_trajectory_applied": False,
        "candidate_window_count": len(candidate_rows),
        "sampled_source_frame_count": len(frame_cache),
        "frame_thresholds": frame_thresholds,
        "window_thresholds": window_thresholds,
    }
    return frame_rows, window_summaries, failures, evidence_rows, producer_run_config


def _qy_containment_inputs(
    context: AssetContext,
    candidate_rows: list[dict[str, Any]],
) -> _QyContainmentInputs:
    from acceptance_pull.supplier_adapters.qingyu import load_qingyu_timebase
    from acceptance_pull.supplier_adapters.qingyu_hand_pose import (
        QingyuHandPoseSession,
    )
    from tools.run_manifest_sam3_containment import sample_manifest_window_frames

    if context.source_range is None:
        raise ModuleBlockedError("sam3_containment", "frame_mapping_unverified")
    primary_camera = str(context.metadata.get("primary_camera") or "").strip()
    if not primary_camera:
        raise ModuleBlockedError("sam3_containment", "primary_camera_missing")
    if str(context.metadata.get("frame_mapping_status") or "") != "verified":
        raise ModuleBlockedError("sam3_containment", "frame_mapping_unverified")
    video_path = _source_path(context, "video")
    observations_path = _source_path(context, "observations_2d")
    assert video_path is not None
    assert observations_path is not None
    session = QingyuHandPoseSession(
        observations_path=observations_path,
    )
    timebase_source = str(context.metadata.get("timebase_source") or "").strip()
    timebase_path: Path | None = None
    if timebase_source == "derived_from_video_and_observations_2d":
        try:
            timebase = session.derive_camera_timebase(
                camera=primary_camera,
                video_path=video_path,
            )
        except ValueError as exc:
            raise ModuleInputError("sam3_containment", str(exc)) from exc
        if timebase is not None:
            expected_values = {
                "frames": context.metadata.get("video_frame_count"),
                "width": context.metadata.get("video_width"),
                "height": context.metadata.get("video_height"),
                "fps": context.metadata.get("fps"),
                "source_start_frame": context.metadata.get("start_frame"),
                "source_end_frame": context.metadata.get("end_frame"),
            }
            try:
                metadata_matches = all(
                    float(timebase[field]) == float(expected)
                    for field, expected in expected_values.items()
                    if expected not in {None, ""}
                )
            except (TypeError, ValueError):
                metadata_matches = False
            if not metadata_matches:
                raise ModuleBlockedError(
                    "sam3_containment", "frame_mapping_unverified"
                )
    else:
        timebase_path = _source_path(context, "timebase")
        assert timebase_path is not None
        episode_root = timebase_path.parent.parent
        try:
            timebase_by_camera = load_qingyu_timebase(
                timebase_path,
                episode_root=episode_root,
            )
        except ValueError as exc:
            raise ModuleInputError("sam3_containment", str(exc)) from exc
        timebase = timebase_by_camera.get(primary_camera)
    if timebase is None:
        raise ModuleBlockedError("sam3_containment", "timebase_unavailable")
    if Path(timebase["video_path"]).resolve() != video_path.resolve():
        raise ModuleBlockedError("sam3_containment", "frame_mapping_unverified")
    start_frame, end_frame_exclusive = context.source_range
    end_frame = end_frame_exclusive - 1
    if (
        int(timebase["source_start_frame"]) != start_frame
        or int(timebase["source_end_frame"]) != end_frame
    ):
        raise ModuleBlockedError("sam3_containment", "frame_mapping_unverified")
    direct_2d = session.direct_2d_index(
        camera=primary_camera,
        start_frame=start_frame,
        end_frame=end_frame,
        video_frame_count=int(timebase["frames"]),
        timestamp_start=float(timebase["source_start_timestamp"]),
        timestamp_end=float(timebase["source_end_timestamp"]),
        fps=float(timebase["fps"]),
    )
    if not direct_2d:
        raise ModuleBlockedError("sam3_containment", "frame_mapping_unverified")
    sampled_frames: dict[int, tuple[int, ...]] = {}
    for index, candidate in enumerate(candidate_rows):
        sampled = tuple(
            sample_manifest_window_frames(
                candidate,
                clip_start_frame=start_frame,
                clip_end_frame=end_frame,
                frames_per_window=3,
            )
        )
        sampled_frames[index] = sampled
        for source_frame in sampled:
            for side in _candidate_hand_sides(candidate):
                if (source_frame, side) not in direct_2d:
                    raise ModuleBlockedError(
                        "sam3_containment", "primary_camera_2d_missing"
                    )
    return _QyContainmentInputs(
        video_path=video_path,
        observations_path=observations_path,
        timebase_path=timebase_path,
        primary_camera=primary_camera,
        timebase=timebase,
        direct_2d=direct_2d,
        sampled_frames=sampled_frames,
    )


def _run_qy_containment(
    *,
    context: AssetContext,
    config: LoadedQcConfig,
    candidate_rows: list[dict[str, Any]],
    inputs: _QyContainmentInputs,
    segmenter: Any,
    staging_root: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    import cv2
    import numpy as np

    from tools.run_manifest_sam3_containment import (
        DEFAULT_QUERIES,
        SAM3_CONFIG,
        configured_sam3_thresholds,
    )
    from tools.sam3_keypoint_containment import (
        aggregate_window_containment_summaries,
        candidate_window_metadata,
        score_keypoints_against_masks,
        write_combined_overlay_image,
    )

    assert context.source_range is not None
    clip_start, clip_end_exclusive = context.source_range
    clip_end = clip_end_exclusive - 1
    frame_thresholds, window_thresholds = configured_sam3_thresholds(config)
    queries = [value.strip() for value in DEFAULT_QUERIES.split(",") if value.strip()]
    # The QY 21-point order is not yet an anatomical contract.  Containment
    # ratios are order-invariant, so use opaque labels and draw points only;
    # never attach acceptance/MANO names or inferred bones to raw QY indices.
    joint_names = {
        side: [f"qy_{side}_joint_{index:02d}" for index in range(21)]
        for side in ("left", "right")
    }
    capture = cv2.VideoCapture(str(inputs.video_path))
    if not capture.isOpened():
        capture.release()
        raise ModuleInputError("sam3_containment", "QY primary video is unreadable")
    frame_cache: dict[int, Any] = {}
    mask_cache: dict[int, list[Any]] = {}
    frame_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    try:
        for window_index, candidate in enumerate(candidate_rows):
            for source_frame in inputs.sampled_frames[window_index]:
                requested_sides = _candidate_hand_sides(candidate)
                video_frames = {
                    int(inputs.direct_2d[(source_frame, side)]["video_frame"])
                    for side in requested_sides
                }
                if len(video_frames) != 1:
                    raise ModuleBlockedError(
                        "sam3_containment", "frame_mapping_unverified"
                    )
                video_frame = next(iter(video_frames))
                if video_frame not in frame_cache:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, video_frame)
                    ok, frame_bgr = capture.read()
                    if not ok or frame_bgr is None:
                        raise ModuleInputError(
                            "sam3_containment",
                            f"QY video frame {video_frame} for source frame "
                            f"{source_frame} is unreadable",
                        )
                    frame_cache[video_frame] = frame_bgr[..., ::-1].copy()
                frame = frame_cache[video_frame]
                if video_frame not in mask_cache:
                    mask_cache[video_frame] = segmenter.segment_frame(
                        frame,
                        queries,
                        dict(SAM3_CONFIG),
                    )
                masks = mask_cache[video_frame]
                combined_hands: dict[str, dict[str, Any]] = {}
                for side in ("left", "right"):
                    direct = inputs.direct_2d.get((source_frame, side))
                    if direct is None:
                        continue
                    side_video_frame = int(direct["video_frame"])
                    if side_video_frame != video_frame:
                        raise ModuleBlockedError(
                            "sam3_containment", "frame_mapping_unverified"
                        )
                    pixels = np.asarray(direct["keypoints_2d"], dtype=np.float32)
                    containment, _union_mask, valid, inside = (
                        score_keypoints_against_masks(
                            frame=frame,
                            pixels=pixels,
                            joint_names=joint_names[side],
                            masks=masks,
                            valid=np.ones(21, dtype=bool),
                            **frame_thresholds,
                        )
                    )
                    combined_hands[side] = {
                        "pixels": pixels,
                        "valid": valid,
                        "inside": inside,
                        "joint_names": joint_names[side],
                    }
                    if side not in requested_sides:
                        continue
                    frame_rows.append(
                        {
                            "clip_id": context.asset_id,
                            "asset_id": context.asset_id,
                            "episode_idx": 0,
                            "video_path": str(inputs.video_path),
                            "observations_2d_path": str(inputs.observations_path),
                            "frame_idx": source_frame,
                            "source_frame_idx": source_frame,
                            "source_local_frame_idx": source_frame - clip_start,
                            "video_frame_idx": video_frame,
                            "clip_start_frame": clip_start,
                            "clip_end_frame": clip_end,
                            "candidate_start_frame": candidate["start_frame"],
                            "candidate_end_frame": candidate["end_frame"],
                            "coordinate_space": "source",
                            "projection_mode": "qy_direct_2d",
                            "projection_mode_used": "qy_direct_2d",
                            "projection_input_status": "direct_2d_valid",
                            "projection_input_reason": "explicit_qy_observation_mapping",
                            "image_width": int(frame.shape[1]),
                            "image_height": int(frame.shape[0]),
                            "camera_id": inputs.primary_camera,
                            "camera_name": inputs.primary_camera,
                            "joint_topology_status": "unverified_points_only",
                            "hand_side": side,
                            "candidate_hand_side": candidate.get("hand_side", "both"),
                            **containment,
                            **candidate_window_metadata(candidate),
                        }
                    )
                overlay_path = write_combined_overlay_image(
                    frame=frame,
                    hands=combined_hands,
                    clip_id=(
                        f"{context.asset_id}_window_"
                        f"{candidate['start_frame']}_{candidate['end_frame']}_combined"
                    ),
                    frame_idx=source_frame,
                    output_dir=staging_root / "combined_overlays",
                )
                evidence_rows.append(
                    {
                        "review_id": "",
                        "supplier_id": "qy",
                        "asset_id": context.asset_id,
                        "window_start_frame": candidate["start_frame"],
                        "window_end_frame": candidate["end_frame"],
                        "frame_idx": source_frame,
                        "source_module": "sam3_containment",
                        "evidence_type": "combined_overlay",
                        "hand_side": "both",
                        "source_path": str(Path(overlay_path).resolve()),
                        "metadata_json": json.dumps(
                            {
                                "camera_name": inputs.primary_camera,
                                "source_frame_idx": source_frame,
                                "video_frame_idx": video_frame,
                                "coordinate_space": "source",
                            },
                            sort_keys=True,
                        ),
                    }
                )
    finally:
        capture.release()
    window_summaries = aggregate_window_containment_summaries(
        frame_rows,
        **window_thresholds,
    )
    producer_run_config = {
        "supplier": "qy",
        "primary_camera": inputs.primary_camera,
        "keypoint_source": "QY observations_2d direct pixels",
        "frame_mapping": "explicit_source_frame_index_to_video_frame",
        "candidate_window_count": len(candidate_rows),
        "sampled_source_frame_count": len(
            {frame for values in inputs.sampled_frames.values() for frame in values}
        ),
        "sampled_video_frame_count": len(frame_cache),
        "frame_thresholds": frame_thresholds,
        "window_thresholds": window_thresholds,
    }
    return frame_rows, window_summaries, failures, evidence_rows, producer_run_config


def _run_canonical(
    context: AssetContext,
    config: LoadedQcConfig,
    segmenter_factory: Callable[..., Any] | None,
    segmenter_provider: SegmenterProvider | None,
) -> ModuleResult:
    """Run SAM3 from CanonicalEpisode without exposing supplier source layouts."""
    import pandas as pd

    from canonical_qc.bridge import CanonicalQcBridge
    from qc_pipeline.adapters.sam3_containment import adapt_sam3_containment
    from tools.run_manifest_sam3_containment import (
        ManifestSourceCache,
        SAM3_CONFIG,
        read_records,
        run_manifest_sam3_containment,
    )

    episode = context.metadata.get("canonical_episode")
    source_root = context.metadata.get("canonical_source_root")
    if not isinstance(source_root, str) or not source_root:
        raise ModulePrerequisiteError(
            "sam3_containment", "metadata.canonical_source_root"
        )
    bridge = CanonicalQcBridge(episode, source_root=Path(source_root))
    candidate_path = _source_path(context, "candidate_windows")
    assert candidate_path is not None
    candidate_rows = _records_for_asset(candidate_path, context.asset_id)

    staging_parent = context.batch_root / ".qc_pipeline" / context.asset_id / "sam3"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix="run-", dir=staging_parent))
    start, end = context.source_range or (0, episode.time_axis.frame_count)
    points = episode.observation.hand_keypoints_2d
    projection = staging_root / "canonical_keypoints_2d.parquet"
    pd.DataFrame(
        {
            "canonical_left_hand_2d": [row.reshape(-1) for row in points[:, 0]],
            "canonical_right_hand_2d": [row.reshape(-1) for row in points[:, 1]],
        }
    ).to_parquet(projection, index=False)
    video_path = bridge.video_path()
    manifest_row = {
        "asset_id": context.asset_id,
        "episode_index": 0,
        "start_frame": start,
        "end_frame": end - 1,
        "primary_video_path": str(video_path),
        "parquet_path": str(projection),
        "left_hand_2d_field": "canonical_left_hand_2d",
        "right_hand_2d_field": "canonical_right_hand_2d",
    }
    single_manifest = staging_root / "manifest.jsonl"
    single_candidates = staging_root / "candidate_windows.jsonl"
    single_manifest.write_text(
        json.dumps(manifest_row, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    single_candidates.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in candidate_rows),
        encoding="utf-8",
    )
    model = _source_path(
        context,
        "sam3_model",
        required=segmenter_factory is None,
        expected_type="directory",
    )
    if segmenter_factory is not None:
        segmenter = segmenter_factory()
    elif segmenter_provider is not None:
        assert model is not None
        segmenter = segmenter_provider(model, dict(SAM3_CONFIG))
    else:
        segmenter = None

    class CanonicalSourceCache(ManifestSourceCache):
        def read_frame(self, path: Path, frame_idx: int):  # type: ignore[no-untyped-def]
            physical_start, _ = episode.main_video.source_frame_range
            return super().read_frame(path, physical_start + frame_idx)

    source_cache = CanonicalSourceCache()
    bridge.verify_sources()
    try:
        summary = run_manifest_sam3_containment(
            manifest=single_manifest,
            candidate_windows=single_candidates,
            supplier="canonical",
            output_dir=staging_root / "output",
            max_clips=1,
            sam3_model=model,
            overwrite=True,
            segmenter=segmenter,
            source_cache=source_cache,
            config_path=config.path,
            batch_root=staging_root,
            profile=str(context.metadata.get("profile") or "acceptance"),
        )
    finally:
        source_cache.close()
    bridge.verify_sources()
    if int(summary.get("failed_asset_count", 0)):
        raise RuntimeError(f"sam3_containment producer failed: {summary}")
    output_dir = staging_root / "output"
    window_rows = read_records(
        output_dir / "window_keypoint_containment_summary.json"
    )
    frame_rows = [
        {**row, "camera_id": "main"}
        for row in read_records(output_dir / "frame_keypoint_containment.json")
    ]
    evidence_rows = read_records(output_dir / "review_evidence_manifest.csv")
    hand_quality = episode.supplier_evidence.hand_quality
    result = adapt_sam3_containment(
        asset_id=context.asset_id,
        batch_root=context.batch_root,
        window_summaries=window_rows,
        evidence_rows=evidence_rows,
        config=config,
        frame_rows=frame_rows,
        supplier_hand_quality_status=(
            hand_quality.status
            if hand_quality is not None and hand_quality.provided
            else None
        ),
    )
    from qc_pipeline.artifacts import canonical_sha256, file_sha256

    fingerprint_sha256 = canonical_sha256(
        {
            "producer": "sam3_containment",
            "implementation_version": _IMPLEMENTATION_VERSION,
            "asset_id": context.asset_id,
            "video_identity": file_sha256(video_path),
            "candidate_rows": candidate_rows,
        }
    )
    result = replace(
        result,
        runtime={
            **dict(result.runtime),
            "artifact_state": "computed",
            "fingerprint_sha256": fingerprint_sha256,
        },
    )
    try:
        recipe = _canonical_overlay_input_recipe(
            context=context,
            candidate_rows=candidate_rows,
            episode=episode,
            video_path=video_path,
            producer_fingerprint_sha256=fingerprint_sha256,
        )
    except Exception:
        return result
    if recipe is None:
        return result
    return replace(result, runtime={**dict(result.runtime), "overlay_input_recipe": recipe})


def runner(
    segmenter_factory: Callable[..., Any] | None,
    *,
    segmenter_provider: SegmenterProvider | None = None,
) -> ModuleRunner:
    def run(context: AssetContext, config: LoadedQcConfig) -> ModuleResult:
        if context.metadata.get("canonical_episode") is not None:
            return _run_canonical(
                context,
                config,
                segmenter_factory,
                segmenter_provider,
            )

        from qc_pipeline.adapters.sam3_containment import adapt_sam3_containment
        from tools.run_manifest_sam3_containment import (
            DEFAULT_QUERIES,
            SAM3_CONFIG,
            read_records,
            run_manifest_sam3_containment,
        )
        from qc_pipeline.artifacts import (
            build_run_fingerprint,
            canonical_sha256,
            directory_identity,
            file_sha256,
            reusable_artifact,
        )

        started = perf_counter()

        supplier = str(
            context.metadata.get("supplier")
            or context.metadata.get("supplier_id")
            or "jdt"
        ).lower()
        dr_inputs: tuple[Path, Path, Path, Path, Any, Mapping[str, Any]] | None = None
        qy_inputs: _QyContainmentInputs | None = None
        if supplier in {"dr", "deepreach"}:
            heuristic_projection = (
                context.metadata.get("projection_mode")
                == "approx_pinhole_from_hfov"
            )
            projection_status = str(
                context.metadata.get("projection_validation_status") or ""
            ).lower()
            if not heuristic_projection and projection_status != "validated":
                reason = (
                    "transform_ambiguous"
                    if projection_status == "transform_ambiguous"
                    else projection_status
                    if projection_status
                    in {
                        "mapping_missing",
                        "resolution_mismatch",
                        "frame_alignment_unverified",
                    }
                    else "calibration_unverified"
                )
                raise ModuleBlockedError("sam3_containment", reason)
            dr_inputs = _dr_projection_inputs(context, config)
            validation = dr_inputs[4]
            if validation.status not in {"validated", "heuristic_ready"}:
                reason = (
                    validation.reason
                    if validation.reason
                    in {
                        "mapping_missing",
                        "resolution_mismatch",
                        "calibration_video_resolution_mismatch",
                        "frame_alignment_unverified",
                        "hdf5_video_frame_count_mismatch",
                        "direct_head_transform_chain_not_explicit",
                    }
                    else validation.status
                )
                raise ModuleBlockedError("sam3_containment", reason)
        if supplier == "potentia":
            raise ModuleBlockedError("sam3_containment", "no_keypoint_input")
        if supplier in {"qy", "qingyu"}:
            if not str(context.metadata.get("primary_camera") or "").strip():
                raise ModuleBlockedError(
                    "sam3_containment", "primary_camera_missing"
                )
            if str(context.metadata.get("frame_mapping_status") or "") != "verified":
                raise ModuleBlockedError(
                    "sam3_containment", "frame_mapping_unverified"
                )

        candidate_path = artifact_for(context, "precheck").directory / "candidate_windows.json"
        if not candidate_path.is_file():
            raise ModulePrerequisiteError(
                "sam3_containment",
                "current precheck candidate_windows.json",
            )
        precheck_run_config = candidate_path.parent / "run_config.json"
        if not precheck_run_config.is_file():
            raise ModulePrerequisiteError(
                "sam3_containment",
                "current precheck run_config.json",
            )
        try:
            precheck_run = json.loads(
                precheck_run_config.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModuleInputError(
                "sam3_containment",
                f"current precheck run_config is unreadable: {exc}",
            ) from exc
        from qc_pipeline.runners.precheck import precheck_fingerprint

        expected_precheck = precheck_fingerprint(context, config)
        completed_precheck_modules = precheck_run.get("completed_modules")
        if (
            precheck_run.get("producer") != "precheck"
            or precheck_run.get("outcome") not in {"completed", "partial"}
            or precheck_run.get("fingerprint") != expected_precheck
            or not isinstance(completed_precheck_modules, list)
            or "keypoint_temporal" not in completed_precheck_modules
        ):
            raise ModuleInputError(
                "sam3_containment",
                "current precheck run does not match this asset/config or lacks temporal output",
            )
        temporal_output = precheck_run.get("temporal_output")
        qy_temporal_not_applicable = bool(
            supplier in {"qy", "qingyu"}
            and isinstance(temporal_output, Mapping)
            and temporal_output.get("status") == "not_applicable"
            and temporal_output.get("reason") == "qy_hand_topology_not_validated"
        )
        if not qy_temporal_not_applicable and (
            not isinstance(temporal_output, Mapping)
            or temporal_output.get("status") != "valid"
            or not isinstance(temporal_output.get("valid_frame_count"), int)
            or int(temporal_output["valid_frame_count"]) <= 0
        ):
            raise ModuleBlockedError(
                "sam3_containment",
                "no_valid_temporal_output",
            )
        if supplier in {"dr", "deepreach"}:
            check_results_path = candidate_path.parent / "check_results.json"
            if check_results_path.is_file():
                try:
                    check_rows = read_records(check_results_path)
                except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ModuleInputError(
                        "sam3_containment",
                        f"current precheck check_results are unreadable: {exc}",
                    ) from exc
                if any(_dr_has_hard_presence_invalid(row) for row in check_rows):
                    raise ModuleBlockedError(
                        "sam3_containment",
                        "invalid_keypoint_input",
                    )
        try:
            all_candidate_rows = read_records(candidate_path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ModuleInputError(
                "sam3_containment",
                f"candidate artifact is unreadable: {exc}",
            ) from exc
        candidate_rows = _validated_candidates(context, all_candidate_rows)
        if not candidate_rows:
            return ModuleResult(
                module="sam3_containment",
                verdict="skipped",
                evaluation={"decision": "skipped", "reason": "no_candidates"},
                metrics={"window_count": 0},
                runtime={"artifact_state": "no_candidates"},
            )

        if supplier in {"qy", "qingyu"}:
            qy_inputs = _qy_containment_inputs(context, candidate_rows)

        if supplier not in {"jdt", "dr", "deepreach", "qy", "qingyu"}:
            raise ModuleAdapterMissingError(
                "sam3_containment",
                f"supplier adapter is not implemented: {supplier}",
            )
        if supplier == "jdt":
            for source_name in ("video", "parquet"):
                _source_path(context, source_name, required=False)
            manifest_path = _source_path(context, "manifest", required=False)
            source_names = tuple(
                name
                for name in ("video", "parquet")
                if name in context.source_files
            )
        elif supplier in {"dr", "deepreach"}:
            assert dr_inputs is not None
            manifest_path = None
            source_names = tuple(
                name
                for name in ("video", "hdf5", "calibration", "trajectory")
                if name in context.source_files
            )
        else:
            assert qy_inputs is not None
            manifest_path = None
            source_names = tuple(
                name
                for name in ("video", "observations_2d", "timebase")
                if name in context.source_files
            )
        jdt_manifest_row: dict[str, Any] | None = None
        if supplier == "jdt":
            if manifest_path is not None:
                manifest_rows = _records_for_asset(manifest_path, context.asset_id)
                manifest_dir = manifest_path.parent
            else:
                declared_row = context.metadata.get("manifest_row")
                if not isinstance(declared_row, Mapping):
                    raise ModulePrerequisiteError(
                        "sam3_containment",
                        "source_files.manifest.path or metadata.manifest_row",
                    )
                manifest_rows = [dict(declared_row)]
                manifest_dir = context.batch_root
            if len(manifest_rows) != 1:
                raise ModulePrerequisiteError(
                    "sam3_containment",
                    "exactly one manifest row for asset_id",
                )
            jdt_manifest_row = dict(manifest_rows[0])
            for field, source_name in (
                ("primary_video_path", "video"),
                ("parquet_path", "parquet"),
            ):
                entry = _source_entry(context, source_name)
                if entry is not None:
                    jdt_manifest_row[field] = str(
                        context.batch_root / str(entry["path"])
                    )
                elif field in jdt_manifest_row:
                    value = Path(str(jdt_manifest_row[field]))
                    jdt_manifest_row[field] = str(
                        value if value.is_absolute() else manifest_dir / value
                    )
        model = _source_path(
            context,
            "sam3_model",
            required=segmenter_factory is None,
            expected_type="directory",
        )
        fingerprint_extra: dict[str, Any] = {
            "candidate_sha256": file_sha256(candidate_path),
            "queries": DEFAULT_QUERIES,
            "sam3_runtime": SAM3_CONFIG,
        }
        if dr_inputs is not None:
            validation = dr_inputs[4]
            fingerprint_extra["dr_projection_contract"] = {
                "adapter_version": "dr-head-hdf5-v1",
                "status": validation.status,
                "reason": validation.reason,
                "content_id": context.metadata.get("content_id"),
                "calibration_mapping_status": context.metadata.get(
                    "calibration_mapping_status"
                ),
                "projection_validation_status": context.metadata.get(
                    "projection_validation_status"
                ),
                "trajectory_usage": validation.trajectory_usage,
                "projection_mode": context.metadata.get("projection_mode"),
                "calibration_status": context.metadata.get(
                    "calibration_status"
                ),
                "head_hfov_deg": context.metadata.get("head_hfov_deg"),
                "intrinsics": {
                    name: context.metadata.get(name)
                    for name in ("fx", "fy", "cx", "cy")
                },
                "distortion_applied": context.metadata.get(
                    "distortion_applied"
                ),
                "selected_by": context.metadata.get("selected_by"),
            }
        if qy_inputs is not None:
            fingerprint_extra["qy_direct_2d_contract"] = {
                "adapter_version": "qy-direct-2d-sam3-v2",
                "primary_camera": qy_inputs.primary_camera,
                "frame_mapping": "explicit_source_frame_index_to_video_frame",
                "joint_topology": "unverified_points_only",
                "timebase_source": context.metadata.get("timebase_source"),
                "source_frame_count": int(
                    qy_inputs.timebase["source_frame_count"]
                ),
                "video_frame_count": int(qy_inputs.timebase["frames"]),
            }
        fingerprint = build_run_fingerprint(
            context=context,
            producer="sam3_containment",
            config=config,
            module_names=("sam3_containment",),
            source_names=source_names,
            implementation_version=_IMPLEMENTATION_VERSION,
            extra=fingerprint_extra,
        )
        if model is not None:
            fingerprint["sources"]["sam3_model"] = directory_identity(
                model,
                batch_root=context.batch_root,
                key_files=_MODEL_IDENTITY_FILES,
                hash_files=_MODEL_HASH_FILES,
                declared=_source_entry(context, "sam3_model"),
                allow_symlinked_sources=context.allow_symlinked_sources,
            )
        fingerprint_sha256 = canonical_sha256(fingerprint)

        def attach_overlay_recipe(result: ModuleResult) -> ModuleResult:
            """Best-effort post-processing that cannot alter machine QC."""

            try:
                if supplier == "jdt":
                    assert jdt_manifest_row is not None
                    recipe = _jdt_overlay_input_recipe(
                        context=context,
                        candidate_rows=candidate_rows,
                        manifest_row=jdt_manifest_row,
                        producer_fingerprint_sha256=fingerprint_sha256,
                    )
                elif supplier in {"dr", "deepreach"}:
                    assert dr_inputs is not None
                    recipe = _dr_overlay_input_recipe(
                        context=context,
                        candidate_rows=candidate_rows,
                        hdf5_path=dr_inputs[0],
                        video_path=dr_inputs[1],
                        validation=dr_inputs[4],
                        producer_fingerprint_sha256=fingerprint_sha256,
                    )
                else:
                    assert qy_inputs is not None
                    recipe = _qy_overlay_input_recipe(
                        context=context,
                        candidate_rows=candidate_rows,
                        inputs=qy_inputs,
                        producer_fingerprint_sha256=fingerprint_sha256,
                    )
            except Exception:
                return result
            if recipe is None:
                return result
            return replace(
                result,
                runtime={**dict(result.runtime), "overlay_input_recipe": recipe},
            )

        artifact = artifact_for(context, "sam3_containment")
        if bool(context.metadata.get("reuse_artifacts", True)) and reusable_artifact(
            artifact, fingerprint
        ):
            window_summaries = read_records(artifact.directory / "window_results.json")
            evidence_rows = read_records(artifact.directory / "evidence_manifest.json")
            result = adapt_sam3_containment(
                asset_id=context.asset_id,
                batch_root=context.batch_root,
                window_summaries=window_summaries,
                evidence_rows=evidence_rows,
                config=config,
            )
            return attach_overlay_recipe(
                _with_artifact_runtime(
                    result,
                    state="reused",
                    elapsed_seconds=perf_counter() - started,
                    fingerprint_sha256=fingerprint_sha256,
                )
            )
        staging_root = context.batch_root / ".qc_pipeline" / context.asset_id / "sam3"
        staging_root.mkdir(parents=True, exist_ok=True)
        if qy_inputs is not None:
            if segmenter_factory is not None:
                segmenter = segmenter_factory()
            elif segmenter_provider is not None:
                assert model is not None
                segmenter = segmenter_provider(model, dict(SAM3_CONFIG))
            else:
                segmenter = None
            if segmenter is None:
                raise ModulePrerequisiteError(
                    "sam3_containment", "SAM3 segmenter runtime"
                )
            (
                frame_results,
                window_summaries,
                failures,
                evidence_rows,
                producer_run_config,
            ) = _run_qy_containment(
                context=context,
                config=config,
                candidate_rows=candidate_rows,
                inputs=qy_inputs,
                segmenter=segmenter,
                staging_root=staging_root,
            )
            elapsed = perf_counter() - started
            window_summaries, evidence_rows = _publish_sam3_artifact(
                context=context,
                frame_results=frame_results,
                window_summaries=window_summaries,
                failures=failures,
                evidence_rows=evidence_rows,
                producer_run_config=producer_run_config,
                producer_root=staging_root,
                fingerprint=fingerprint,
                elapsed_seconds=elapsed,
            )
            result = adapt_sam3_containment(
                asset_id=context.asset_id,
                batch_root=context.batch_root,
                window_summaries=window_summaries,
                evidence_rows=evidence_rows,
                config=config,
            )
            return attach_overlay_recipe(
                _with_artifact_runtime(
                    result,
                    state="computed",
                    elapsed_seconds=elapsed,
                    fingerprint_sha256=fingerprint_sha256,
                )
            )
        if supplier in {"dr", "deepreach"}:
            if segmenter_factory is not None:
                segmenter = segmenter_factory()
            elif segmenter_provider is not None:
                assert model is not None
                segmenter = segmenter_provider(model, dict(SAM3_CONFIG))
            else:
                segmenter = None
            if segmenter is None:
                raise ModulePrerequisiteError(
                    "sam3_containment", "SAM3 segmenter runtime"
                )
            assert dr_inputs is not None
            (
                hdf5_path,
                video_path,
                calibration_path,
                _trajectory_path,
                validation,
                _dr_config,
            ) = dr_inputs
            (
                frame_results,
                window_summaries,
                failures,
                evidence_rows,
                producer_run_config,
            ) = _run_dr_containment(
                context=context,
                config=config,
                candidate_rows=candidate_rows,
                hdf5_path=hdf5_path,
                video_path=video_path,
                calibration_path=calibration_path,
                validation=validation,
                segmenter=segmenter,
                staging_root=staging_root,
            )
            elapsed = perf_counter() - started
            window_summaries, evidence_rows = _publish_sam3_artifact(
                context=context,
                frame_results=frame_results,
                window_summaries=window_summaries,
                failures=failures,
                evidence_rows=evidence_rows,
                producer_run_config=producer_run_config,
                producer_root=staging_root,
                fingerprint=fingerprint,
                elapsed_seconds=elapsed,
            )
            result = adapt_sam3_containment(
                asset_id=context.asset_id,
                batch_root=context.batch_root,
                window_summaries=window_summaries,
                evidence_rows=evidence_rows,
                config=config,
            )
            return attach_overlay_recipe(
                _with_artifact_runtime(
                    result,
                    state="computed",
                    elapsed_seconds=elapsed,
                    fingerprint_sha256=fingerprint_sha256,
                )
            )
        assert jdt_manifest_row is not None
        manifest_row = jdt_manifest_row
        single_manifest = staging_root / "manifest.jsonl"
        single_candidates = staging_root / "candidate_windows.jsonl"
        single_manifest.write_text(
            json.dumps(manifest_row, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        single_candidates.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for row in candidate_rows
            ),
            encoding="utf-8",
        )
        output_dir = staging_root / "output"
        if segmenter_factory is not None:
            segmenter = segmenter_factory()
        elif segmenter_provider is not None:
            assert model is not None
            segmenter = segmenter_provider(model, dict(SAM3_CONFIG))
        else:
            segmenter = None
        summary = run_manifest_sam3_containment(
            manifest=single_manifest,
            candidate_windows=single_candidates,
            supplier=supplier,
            output_dir=output_dir,
            max_clips=1,
            sam3_model=model,
            overwrite=True,
            segmenter=segmenter,
            config_path=config.path,
            batch_root=staging_root,
            profile=str(context.metadata.get("profile") or "acceptance"),
        )
        if int(summary.get("failed_asset_count", 0)):
            raise RuntimeError(f"sam3_containment producer failed: {summary}")
        window_summaries = read_records(
            output_dir / "window_keypoint_containment_summary.json"
        )
        evidence_rows = read_records(output_dir / "review_evidence_manifest.csv")
        frame_path = output_dir / "frame_keypoint_containment.json"
        failures_path = output_dir / "failures.json"
        producer_config_path = output_dir / "run_config.json"
        frame_results = read_records(frame_path) if frame_path.is_file() else []
        failures = read_records(failures_path) if failures_path.is_file() else []
        producer_run_config = (
            json.loads(producer_config_path.read_text(encoding="utf-8"))
            if producer_config_path.is_file()
            else {}
        )
        elapsed = perf_counter() - started
        window_summaries, evidence_rows = _publish_sam3_artifact(
            context=context,
            frame_results=frame_results,
            window_summaries=window_summaries,
            failures=failures,
            evidence_rows=evidence_rows,
            producer_run_config=producer_run_config,
            producer_root=staging_root,
            fingerprint=fingerprint,
            elapsed_seconds=elapsed,
        )
        result = adapt_sam3_containment(
            asset_id=context.asset_id,
            batch_root=context.batch_root,
            window_summaries=window_summaries,
            evidence_rows=evidence_rows,
            config=config,
        )
        return attach_overlay_recipe(
            _with_artifact_runtime(
                result,
                state="computed",
                elapsed_seconds=elapsed,
                fingerprint_sha256=fingerprint_sha256,
            )
        )

    return run


__all__ = ["runner"]
