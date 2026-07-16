"""Pure per-hand keypoint existence and sentinel validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from qc_common.keypoints import acceptance_joint_names


@dataclass(frozen=True)
class HandKeypointValidity:
    side: str
    expected_point_count: int
    finite_point_count: int
    valid_point_count: int
    all_zero: bool
    all_identical: bool
    invalid_reasons: tuple[str, ...]
    points_by_name: Mapping[str, np.ndarray]

    @property
    def is_valid(self) -> bool:
        return not self.invalid_reasons


def inspect_hand_keypoints(
    keypoints: Mapping[str, np.ndarray] | None,
    *,
    side: str,
    frame_offset: int,
    source_value_count: int | None = None,
) -> HandKeypointValidity:
    """Validate one canonical 21-point hand without supplier quality signals."""
    expected = acceptance_joint_names([side])
    points_by_name: dict[str, np.ndarray] = {}
    has_missing = False
    has_short_coordinate = False
    has_nonfinite = False

    for name in expected:
        values = keypoints.get(name) if keypoints else None
        if values is None or values.shape[0] <= frame_offset:
            has_missing = True
            continue
        point = np.asarray(values[frame_offset], dtype=np.float64)
        if point.ndim == 0 or point.shape[0] < 3:
            has_short_coordinate = True
            continue
        if not np.all(np.isfinite(point[:3])):
            has_nonfinite = True
            continue
        points_by_name[name] = point[:3]

    finite_point_count = len(points_by_name)
    all_zero = False
    all_identical = False
    if finite_point_count == len(expected):
        points = np.stack([points_by_name[name] for name in expected])
        all_zero = bool(np.all(points == 0.0))
        all_identical = bool(np.all(points == points[0]))

    reasons: list[str] = []
    if source_value_count is not None and source_value_count < len(expected) * 3:
        reasons.append("invalid_coordinate_shape")
    if has_missing:
        reasons.append("missing_keypoints")
    if has_short_coordinate and "invalid_coordinate_shape" not in reasons:
        reasons.append("invalid_coordinate_shape")
    if has_nonfinite:
        reasons.append("nonfinite_keypoints")
    if finite_point_count < len(expected):
        reasons.append("insufficient_valid_keypoint_count")
    if all_zero:
        reasons.append("all_zero_keypoints")
    elif all_identical:
        reasons.append("all_identical_keypoints")

    valid_point_count = 0 if all_zero or all_identical else finite_point_count
    return HandKeypointValidity(
        side=side,
        expected_point_count=len(expected),
        finite_point_count=finite_point_count,
        valid_point_count=valid_point_count,
        all_zero=all_zero,
        all_identical=all_identical,
        invalid_reasons=tuple(reasons),
        points_by_name=points_by_name,
    )


__all__ = ["HandKeypointValidity", "inspect_hand_keypoints"]
