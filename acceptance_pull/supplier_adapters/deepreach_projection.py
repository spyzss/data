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


__all__ = [
    "Calibration",
    "TrajectoryLoad",
    "TrajectoryPose",
    "load_calibration",
    "load_trajectory",
    "project_hand",
]
