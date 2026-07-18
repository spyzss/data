"""Mapping-driven DR calibration, trajectory, and projection helpers."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from acceptance_pull.supplier_adapters.structured_audit import json_path_value
from qc_common.projection import apply_rigid_transform, project_points_to_image


@dataclass(frozen=True)
class Calibration:
    status: str
    camera_name: str
    intrinsics: np.ndarray | None
    resolution: tuple[int, int] | None
    source: str
    reason: str | None = None


@dataclass(frozen=True)
class TrajectoryPose:
    source_frame: int
    rotation: np.ndarray
    translation: np.ndarray
    transform_direction: str
    timestamp: float | None = None


@dataclass(frozen=True)
class TrajectoryLoad:
    status: str
    poses: Mapping[int, TrajectoryPose]
    missing_frame_ranges: tuple[tuple[int, int], ...]
    source: str
    reason: str | None = None


@dataclass(frozen=True)
class HeadProjectionValidation:
    status: str
    reason: str
    calibration: Calibration
    video_resolution: tuple[int, int] | None
    frame_count: int | None
    trajectory_usage: str | None


def _number(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _integer(value: Any) -> int | None:
    numeric = _number(value)
    return int(numeric) if numeric is not None and numeric.is_integer() else None


def _finite_float(value: Any) -> float | None:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _mapped(payload: Any, mapping: Mapping[str, Any], name: str) -> Any:
    path = mapping.get(name)
    if not isinstance(path, str):
        return None
    value = json_path_value(payload, path)
    # The shared resolver intentionally returns an opaque sentinel for misses.
    if type(value) is object:
        return None
    return value


def load_calibration(
    path: Path,
    camera_name: str,
    mapping: Mapping[str, Any],
) -> Calibration:
    source = str(path)
    if not path.is_file():
        return Calibration(
            "calibration_unverified", camera_name, None, None, source, "missing_calibration"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return Calibration(
            "calibration_unverified", camera_name, None, None, source, f"invalid_json:{exc}"
        )
    matrix_value = _mapped(payload, mapping, "intrinsics_matrix_path")
    width = _integer(_mapped(payload, mapping, "width_path"))
    height = _integer(_mapped(payload, mapping, "height_path"))
    try:
        matrix = np.asarray(matrix_value, dtype=np.float64)
    except (TypeError, ValueError):
        matrix = np.empty((0, 0), dtype=np.float64)
    if (
        matrix.shape != (3, 3)
        or not np.all(np.isfinite(matrix))
        or width is None
        or height is None
        or width <= 0
        or height <= 0
        or matrix[0, 0] <= 0
        or matrix[1, 1] <= 0
    ):
        return Calibration(
            "calibration_unverified",
            camera_name,
            None,
            None,
            source,
            "mapped_calibration_field_missing_or_invalid",
        )
    return Calibration(
        "verified",
        camera_name,
        matrix,
        (width, height),
        source,
    )


def _quaternion_xyzw_to_rotation(values: list[float]) -> np.ndarray:
    x, y, z, w = values
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0:
        raise ValueError("zero quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _missing_ranges(frames: list[int]) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    for previous, current in zip(frames, frames[1:]):
        if current > previous + 1:
            ranges.append((previous + 1, current - 1))
    return tuple(ranges)


def load_trajectory(
    path: Path,
    mapping: Mapping[str, Any],
) -> TrajectoryLoad:
    source = str(path)
    if not path.is_file():
        return TrajectoryLoad(
            "calibration_unverified", {}, (), source, "missing_trajectory"
        )
    direction = mapping.get("transform_direction")
    if direction not in {"world_to_camera", "camera_to_world"}:
        return TrajectoryLoad(
            "transform_ambiguous", {}, (), source, "transform_direction_missing_or_ambiguous"
        )
    frame_column = mapping.get("frame_index_column")
    translation_columns = mapping.get("translation_columns")
    quaternion_columns = mapping.get("quaternion_xyzw_columns")
    timestamp_column = mapping.get("timestamp_column")
    if (
        not isinstance(frame_column, str)
        or not isinstance(translation_columns, list)
        or len(translation_columns) != 3
        or not isinstance(quaternion_columns, list)
        or len(quaternion_columns) != 4
    ):
        return TrajectoryLoad(
            "calibration_unverified", {}, (), source, "trajectory_mapping_missing"
        )
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        return TrajectoryLoad(
            "calibration_unverified", {}, (), source, f"trajectory_unreadable:{exc}"
        )
    poses: dict[int, TrajectoryPose] = {}
    for row in rows:
        frame = _integer(row.get(frame_column))
        translation = [_number(row.get(column)) for column in translation_columns]
        quaternion = [_number(row.get(column)) for column in quaternion_columns]
        if frame is None or any(value is None for value in translation + quaternion):
            return TrajectoryLoad(
                "calibration_unverified", {}, (), source, "trajectory_value_invalid"
            )
        if frame in poses:
            return TrajectoryLoad(
                "calibration_unverified", {}, (), source, "duplicate_trajectory_frame"
            )
        try:
            rotation = _quaternion_xyzw_to_rotation(
                [float(value) for value in quaternion if value is not None]
            )
        except ValueError as exc:
            return TrajectoryLoad(
                "calibration_unverified", {}, (), source, str(exc)
            )
        timestamp = (
            _number(row.get(timestamp_column))
            if isinstance(timestamp_column, str)
            else None
        )
        poses[frame] = TrajectoryPose(
            frame,
            rotation,
            np.asarray(translation, dtype=np.float64),
            str(direction),
            timestamp,
        )
    ordered = sorted(poses)
    gaps = _missing_ranges(ordered)
    return TrajectoryLoad(
        "trajectory_gap" if gaps else "verified",
        poses,
        gaps,
        source,
        "trajectory_frame_gap" if gaps else None,
    )


def _world_to_camera(points: np.ndarray, pose: TrajectoryPose) -> np.ndarray:
    if pose.transform_direction == "world_to_camera":
        return apply_rigid_transform(points, pose.rotation, pose.translation)
    inverse_rotation = pose.rotation.T
    inverse_translation = -(inverse_rotation @ pose.translation)
    return apply_rigid_transform(points, inverse_rotation, inverse_translation)


def project_hand(
    *,
    asset_id: str,
    source_frame: int,
    clip_start_frame: int,
    camera_name: str,
    hand_side: str,
    points: np.ndarray,
    calibration: Calibration,
    pose: TrajectoryPose,
) -> list[dict[str, Any]]:
    if calibration.status != "verified" or calibration.intrinsics is None or calibration.resolution is None:
        raise ValueError("calibration must be verified before projection")
    if pose.source_frame != source_frame:
        raise ValueError("trajectory pose does not match source frame")
    camera_points = _world_to_camera(np.asarray(points, dtype=np.float64), pose)
    projected = project_points_to_image(
        camera_points,
        calibration.intrinsics,
        image_width=calibration.resolution[0],
        image_height=calibration.resolution[1],
    )
    records: list[dict[str, Any]] = []
    for index in range(camera_points.shape[0]):
        projection_valid = bool(projected["projection_valid"][index])
        records.append(
            {
                "asset_id": asset_id,
                "source_frame": source_frame,
                "local_frame": source_frame - clip_start_frame,
                "camera_name": camera_name,
                "hand_side": hand_side,
                "keypoint_index": index,
                "projected_x": (
                    _finite_float(projected["u"][index]) if projection_valid else None
                ),
                "projected_y": (
                    _finite_float(projected["v"][index]) if projection_valid else None
                ),
                "source_x": _finite_float(points[index, 0]),
                "source_y": _finite_float(points[index, 1]),
                "source_z": _finite_float(points[index, 2]),
                "camera_x": _finite_float(camera_points[index, 0]),
                "camera_y": _finite_float(camera_points[index, 1]),
                "camera_z": _finite_float(camera_points[index, 2]),
                "depth_z": _finite_float(projected["z"][index]),
                "projection_valid": projection_valid,
                "in_frame": bool(projected["in_frame"][index]),
                "calibration_source": calibration.source,
                "trajectory_source_frame": pose.source_frame,
                "transform_direction": pose.transform_direction,
            }
        )
    return records


def _blocked_projection(
    *,
    status: str,
    reason: str,
    calibration_path: Path,
    video_resolution: tuple[int, int] | None = None,
    frame_count: int | None = None,
    trajectory_usage: str | None = None,
) -> HeadProjectionValidation:
    return HeadProjectionValidation(
        status=status,
        reason=reason,
        calibration=Calibration(
            "calibration_unverified",
            "head",
            None,
            None,
            str(calibration_path),
            reason,
        ),
        video_resolution=video_resolution,
        frame_count=frame_count,
        trajectory_usage=trajectory_usage,
    )


def validate_head_projection_contract(
    *,
    hdf5_path: Path,
    video_path: Path,
    calibration_path: Path,
    trajectory_path: Path,
    source_range: tuple[int, int] | None,
    reference_dataset: str,
    primary_camera: str,
    content_id: str | None,
    calibration_mapping_status: str,
    projection_validation_status: str,
    mapping_status: str,
    mapping: Mapping[str, Any],
) -> HeadProjectionValidation:
    """Validate the explicit DR head-camera projection contract without guessing."""
    if calibration_mapping_status != "mapped" or not str(content_id or "").strip():
        return _blocked_projection(
            status="calibration_unverified",
            reason="mapping_missing",
            calibration_path=calibration_path,
        )
    declared_status = str(projection_validation_status or "").lower()
    if declared_status != "validated":
        status = (
            declared_status
            if declared_status
            in {
                "calibration_unverified",
                "transform_ambiguous",
                "resolution_mismatch",
                "frame_alignment_unverified",
            }
            else "calibration_unverified"
        )
        return _blocked_projection(
            status=status,
            reason=status,
            calibration_path=calibration_path,
        )
    if mapping_status != "verified":
        return _blocked_projection(
            status="calibration_unverified",
            reason="mapping_missing",
            calibration_path=calibration_path,
        )
    if primary_camera != "head":
        return _blocked_projection(
            status="calibration_unverified",
            reason="unsupported_primary_camera",
            calibration_path=calibration_path,
        )
    projection = mapping.get("projection")
    if not isinstance(projection, Mapping):
        return _blocked_projection(
            status="transform_ambiguous",
            reason="direct_head_transform_chain_not_explicit",
            calibration_path=calibration_path,
        )
    trajectory_usage = projection.get("trajectory_usage")
    direct_chain = (
        projection.get("camera_name") == "head"
        and projection.get("joints3d_coordinate_frame") == "head_camera"
        and projection.get("joints3d_unit") == "meter"
        and projection.get("projection_direction") == "direct_camera"
        and trajectory_usage == "lineage_only"
    )
    if not direct_chain:
        return _blocked_projection(
            status="transform_ambiguous",
            reason="direct_head_transform_chain_not_explicit",
            calibration_path=calibration_path,
            trajectory_usage=(
                str(trajectory_usage) if trajectory_usage is not None else None
            ),
        )
    if not trajectory_path.is_file():
        return _blocked_projection(
            status="calibration_unverified",
            reason="missing_trajectory_lineage",
            calibration_path=calibration_path,
            trajectory_usage="lineage_only",
        )
    calibration_mapping = mapping.get("calibration")
    calibration = load_calibration(
        calibration_path,
        "head",
        calibration_mapping if isinstance(calibration_mapping, Mapping) else {},
    )
    if calibration.status != "verified":
        return HeadProjectionValidation(
            status="calibration_unverified",
            reason=calibration.reason or "calibration_unverified",
            calibration=calibration,
            video_resolution=None,
            frame_count=None,
            trajectory_usage="lineage_only",
        )

    import cv2
    import h5py

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return _blocked_projection(
                status="calibration_unverified",
                reason="head_video_unreadable",
                calibration_path=calibration_path,
                trajectory_usage="lineage_only",
            )
        video_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        video_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        video_frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        capture.release()
    video_resolution = (video_width, video_height)
    if min(video_width, video_height, video_frame_count) <= 0:
        return _blocked_projection(
            status="calibration_unverified",
            reason="head_video_metadata_invalid",
            calibration_path=calibration_path,
            video_resolution=video_resolution,
            frame_count=video_frame_count,
            trajectory_usage="lineage_only",
        )

    try:
        with h5py.File(hdf5_path, "r") as handle:
            if reference_dataset not in handle:
                raise ValueError("reference_dataset_missing")
            reference_shape = handle[reference_dataset].shape
            if not reference_shape:
                raise ValueError("reference_dataset_has_no_frame_axis")
            hdf5_frame_count = int(reference_shape[0])
            for side in ("left", "right"):
                name = f"hand/{side}/joints3d"
                if name not in handle or handle[name].shape != (hdf5_frame_count, 21, 3):
                    raise ValueError(f"{name}_shape_invalid")
            declared_frame = str(handle.attrs.get("coordinate_frame") or "").strip()
            declared_units = str(handle.attrs.get("units") or "").strip()
    except (OSError, ValueError) as exc:
        return _blocked_projection(
            status="frame_alignment_unverified",
            reason=str(exc),
            calibration_path=calibration_path,
            video_resolution=video_resolution,
            frame_count=video_frame_count,
            trajectory_usage="lineage_only",
        )
    if declared_frame and declared_frame != "head_camera":
        return _blocked_projection(
            status="transform_ambiguous",
            reason="hdf5_coordinate_frame_conflicts_with_mapping",
            calibration_path=calibration_path,
            video_resolution=video_resolution,
            frame_count=hdf5_frame_count,
            trajectory_usage="lineage_only",
        )
    if declared_units and declared_units not in {"m", "meter", "meters"}:
        return _blocked_projection(
            status="transform_ambiguous",
            reason="hdf5_units_conflict_with_mapping",
            calibration_path=calibration_path,
            video_resolution=video_resolution,
            frame_count=hdf5_frame_count,
            trajectory_usage="lineage_only",
        )
    if source_range != (0, hdf5_frame_count) or video_frame_count != hdf5_frame_count:
        return _blocked_projection(
            status="frame_alignment_unverified",
            reason="hdf5_video_frame_count_mismatch",
            calibration_path=calibration_path,
            video_resolution=video_resolution,
            frame_count=hdf5_frame_count,
            trajectory_usage="lineage_only",
        )

    resolution_policy = projection.get("resolution_policy")
    if calibration.resolution != video_resolution:
        if resolution_policy != "scale_intrinsics":
            return HeadProjectionValidation(
                status="resolution_mismatch",
                reason="calibration_video_resolution_mismatch",
                calibration=calibration,
                video_resolution=video_resolution,
                frame_count=hdf5_frame_count,
                trajectory_usage="lineage_only",
            )
        assert calibration.intrinsics is not None
        from qc_common.projection import scale_intrinsics

        calibration = Calibration(
            "verified",
            "head",
            scale_intrinsics(
                calibration.intrinsics,
                calibration.resolution,
                video_resolution,
            ),
            video_resolution,
            calibration.source,
        )
    elif resolution_policy not in {"exact", "scale_intrinsics"}:
        return HeadProjectionValidation(
            status="resolution_mismatch",
            reason="resolution_policy_missing",
            calibration=calibration,
            video_resolution=video_resolution,
            frame_count=hdf5_frame_count,
            trajectory_usage="lineage_only",
        )
    return HeadProjectionValidation(
        status="validated",
        reason="validated_direct_head_projection",
        calibration=calibration,
        video_resolution=video_resolution,
        frame_count=hdf5_frame_count,
        trajectory_usage="lineage_only",
    )


def project_dr_hands_for_frame(
    hdf5_path: Path,
    *,
    source_frame: int,
    clip_start_frame: int,
    calibration: Calibration,
) -> dict[str, dict[str, Any]]:
    """Read one DR local HDF5 row and project both hands to head pixels."""
    if (
        calibration.status != "verified"
        or calibration.intrinsics is None
        or calibration.resolution is None
    ):
        raise ValueError("verified head calibration is required")
    local_frame = int(source_frame) - int(clip_start_frame)
    if local_frame < 0:
        raise ValueError("source frame precedes clip start")

    import h5py

    projected_hands: dict[str, dict[str, Any]] = {}
    with h5py.File(hdf5_path, "r") as handle:
        for side in ("left", "right"):
            joints_path = f"hand/{side}/joints3d"
            valid_path = f"hand/{side}/valid"
            if joints_path not in handle:
                points = np.full((21, 3), np.nan, dtype=np.float64)
                hand_valid = False
                input_status = "missing_hand_input"
                input_reason = f"missing_dataset:{joints_path}"
            else:
                dataset = handle[joints_path]
                if dataset.ndim != 3 or dataset.shape[1:] != (21, 3):
                    raise ValueError(f"{joints_path} must have shape (N, 21, 3)")
                if local_frame >= dataset.shape[0]:
                    raise ValueError(
                        f"source frame {source_frame} maps outside {joints_path}"
                    )
                points = np.asarray(dataset[local_frame], dtype=np.float64)
                hand_valid = True
                if valid_path in handle:
                    valid_dataset = handle[valid_path]
                    if valid_dataset.ndim < 1 or local_frame >= valid_dataset.shape[0]:
                        raise ValueError(
                            f"source frame {source_frame} maps outside {valid_path}"
                        )
                    hand_valid = bool(
                        np.asarray(valid_dataset[local_frame]).reshape(-1)[0]
                    )
                input_status = "valid"
                input_reason = None
            projection = project_points_to_image(
                points,
                calibration.intrinsics,
                image_width=calibration.resolution[0],
                image_height=calibration.resolution[1],
            )
            valid = np.asarray(projection["projection_valid"], dtype=bool)
            if not hand_valid:
                valid[:] = False
            pixels = np.stack((projection["u"], projection["v"]), axis=1)
            pixels = np.asarray(pixels, dtype=np.float64)
            pixels[~valid] = np.nan
            valid_count = int(np.sum(valid))
            if input_status != "missing_hand_input":
                if valid_count == 21:
                    input_status = "valid"
                elif valid_count == 0:
                    input_status = "hand_invalid"
                    input_reason = "hand_valid_false_or_no_projectable_points"
                else:
                    input_status = "partial_invalid"
                    input_reason = "some_keypoints_not_projectable"
            projected_hands[side] = {
                "source_frame": int(source_frame),
                "local_frame": local_frame,
                "hand_side": side,
                "points_3d": points,
                "pixels": pixels,
                "valid": valid,
                "in_frame": np.asarray(projection["in_frame"], dtype=bool) & valid,
                "projection_input_status": input_status,
                "projection_input_reason": input_reason,
            }
    return projected_hands


__all__ = [
    "Calibration",
    "HeadProjectionValidation",
    "TrajectoryLoad",
    "TrajectoryPose",
    "load_calibration",
    "load_trajectory",
    "project_hand",
    "project_dr_hands_for_frame",
    "validate_head_projection_contract",
]
