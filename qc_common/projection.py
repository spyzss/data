"""Pure projection helpers for cheap visual-review prechecks."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ProjectionConfig:
    image_width: int
    image_height: int
    border_margin_px: float = 20.0


def project_manual_review_counts(
    manual_review: Mapping[str, Any],
) -> dict[str, int]:
    """Project selected Warn-review outcome counts from one report block."""

    selected_ids = manual_review.get("selected_issue_ids", ())
    reviews = manual_review.get("issue_reviews", {})
    if not isinstance(selected_ids, (list, tuple)):
        raise ValueError("manual_review.selected_issue_ids must be a sequence")
    if not isinstance(reviews, Mapping):
        raise ValueError("manual_review.issue_reviews must be an object")

    selected_set = set(selected_ids)
    reviewed_ids = selected_set.intersection(reviews)
    confirmed_fail_count = sum(
        isinstance(reviews[issue_id], Mapping)
        and reviews[issue_id].get("verdict") == "fail"
        for issue_id in reviewed_ids
    )
    return {
        "human_reviewed_warn_count": len(reviewed_ids),
        "human_confirmed_fail_count": confirmed_fail_count,
        "unreviewed_selected_warn_count": len(selected_set - reviewed_ids),
    }


def apply_rigid_transform(
    points_xyz: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Apply ``R @ p + t`` to row-major 3D points."""
    points = np.asarray(points_xyz, dtype=np.float64)
    matrix = np.asarray(rotation, dtype=np.float64)
    offset = np.asarray(translation, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape (N, 3)")
    if matrix.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    if offset.shape != (3,):
        raise ValueError("translation must have shape (3,)")
    return points @ matrix.T + offset


def scale_intrinsics(
    intrinsics: np.ndarray,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> np.ndarray:
    """Scale pixel intrinsics from one explicit resolution to another."""
    camera = np.asarray(intrinsics, dtype=np.float64)
    if camera.shape != (3, 3):
        raise ValueError("intrinsics must have shape (3, 3)")
    source_width, source_height = source_size
    target_width, target_height = target_size
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("source and target resolutions must be positive")
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    scaled = camera.copy()
    scaled[0, 0] *= scale_x
    scaled[0, 1] *= scale_x
    scaled[0, 2] *= scale_x
    scaled[1, 0] *= scale_y
    scaled[1, 1] *= scale_y
    scaled[1, 2] *= scale_y
    return scaled


def intrinsics_from_values(
    fx: float | None,
    fy: float | None,
    cx: float | None,
    cy: float | None,
) -> np.ndarray | None:
    """Build a 3x3 camera matrix from scalar fields when all are present."""
    if fx is None or fy is None or cx is None or cy is None:
        return None
    return np.asarray(
        [[float(fx), 0.0, float(cx)], [0.0, float(fy), float(cy)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def project_points_with_validity(
    points_xyz: np.ndarray,
    intrinsics: np.ndarray,
) -> dict[str, np.ndarray]:
    """Project camera-coordinate 3D points and keep validity masks."""
    points = np.asarray(points_xyz, dtype=np.float64)
    camera = np.asarray(intrinsics, dtype=np.float64)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    finite_xyz = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    positive_z = z > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        u = camera[0, 0] * x / z + camera[0, 2]
        v = camera[1, 1] * y / z + camera[1, 2]
    valid = finite_xyz & positive_z & np.isfinite(u) & np.isfinite(v)
    return {"u": u, "v": v, "z": z, "projection_valid": valid}


def project_points_to_image(
    points_xyz: np.ndarray,
    intrinsics: np.ndarray,
    *,
    image_width: int,
    image_height: int,
) -> dict[str, np.ndarray]:
    """Project points and distinguish valid depth from in-frame pixels."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image resolution must be positive")
    result = project_points_with_validity(points_xyz, intrinsics)
    valid = result["projection_valid"]
    in_frame = (
        valid
        & (result["u"] >= 0.0)
        & (result["u"] < float(image_width))
        & (result["v"] >= 0.0)
        & (result["v"] < float(image_height))
    )
    return {**result, "in_frame": in_frame}


def hand_projection_metrics(
    points_xyz: np.ndarray,
    intrinsics: np.ndarray,
    config: ProjectionConfig,
    previous_center: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Compute per-hand projected bbox, border, outside, and jump metrics."""
    projected = project_points_with_validity(points_xyz, intrinsics)
    u = projected["u"]
    v = projected["v"]
    z = projected["z"]
    valid = projected["projection_valid"]
    valid_u = u[valid]
    valid_v = v[valid]
    width = float(config.image_width)
    height = float(config.image_height)
    margin = float(config.border_margin_px)

    outside = (
        valid
        & (
            (u < 0.0)
            | (u >= width)
            | (v < 0.0)
            | (v >= height)
        )
    )
    near_border = (
        valid
        & ~outside
        & (
            (u < margin)
            | (u > width - 1.0 - margin)
            | (v < margin)
            | (v > height - 1.0 - margin)
        )
    )

    metrics: dict[str, Any] = {
        "num_projected_keypoints": float(points_xyz.shape[0]),
        "num_projection_valid": float(np.sum(valid)),
        "num_projection_invalid": float(points_xyz.shape[0] - np.sum(valid)),
        "num_points_outside_image": float(np.sum(outside)),
        "num_points_near_border": float(np.sum(near_border)),
    }

    if valid_u.size == 0:
        metrics.update(
            {
                "u_min": math.nan,
                "u_max": math.nan,
                "v_min": math.nan,
                "v_max": math.nan,
                "z_min": math.nan,
                "z_max": math.nan,
                "hand_bbox_area_2d": math.nan,
                "hand_bbox_center_u": math.nan,
                "hand_bbox_center_v": math.nan,
                "hand_bbox_center_jump_px": math.nan,
                "keypoint_bbox_touches_border": 0.0,
            }
        )
        return metrics

    u_min = float(np.min(valid_u))
    u_max = float(np.max(valid_u))
    v_min = float(np.min(valid_v))
    v_max = float(np.max(valid_v))
    center_u = (u_min + u_max) * 0.5
    center_v = (v_min + v_max) * 0.5
    if previous_center is None or not all(np.isfinite(previous_center)):
        center_jump = math.nan
    else:
        center_jump = float(
            math.hypot(center_u - previous_center[0], center_v - previous_center[1])
        )
    touches_border = (
        u_min <= margin
        or u_max >= width - 1.0 - margin
        or v_min <= margin
        or v_max >= height - 1.0 - margin
    )
    metrics.update(
        {
            "u_min": u_min,
            "u_max": u_max,
            "v_min": v_min,
            "v_max": v_max,
            "z_min": float(np.min(z[valid])),
            "z_max": float(np.max(z[valid])),
            "hand_bbox_area_2d": float(max(0.0, u_max - u_min) * max(0.0, v_max - v_min)),
            "hand_bbox_center_u": center_u,
            "hand_bbox_center_v": center_v,
            "hand_bbox_center_jump_px": center_jump,
            "keypoint_bbox_touches_border": float(touches_border),
        }
    )
    return metrics
