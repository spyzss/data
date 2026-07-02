"""Optional mask/keypoint containment check."""

from __future__ import annotations

import numpy as np

from precheck.base import BaseCheck
from precheck.registry import register
from qc_common.keypoints import project_points, select_hand_joints
from qc_common.types import CheckResult, ClipInputs


@register
class MaskContainmentCheck(BaseCheck):
    """Measure projected keypoint containment in an injected hand mask."""

    name = "mask_containment"
    granularity = "frame"

    def __init__(self, config: dict) -> None:
        self.sides = config.get("sides", ["left", "right"])
        self.joint_names = config.get("joint_names")

    def run(self, clip: ClipInputs) -> list[CheckResult]:
        masks = clip.masks
        keypoints = clip.keypoints
        intrinsics = clip.intrinsics
        if masks is None or not keypoints or intrinsics is None:
            return []

        joint_names = self.joint_names or select_hand_joints(sorted(keypoints), self.sides)
        joint_names = [name for name in joint_names if name in keypoints]
        if not joint_names:
            return []

        num_frames = min(clip.num_frames, *(keypoints[name].shape[0] for name in joint_names))
        results: list[CheckResult] = []
        for offset in range(num_frames):
            mask = self._mask_at(masks, clip.frame_idx_at(offset), offset)
            if mask is None:
                continue
            mask_bool = np.asarray(mask).astype(bool)
            points_xyz = np.vstack([keypoints[name][offset] for name in joint_names])
            points_uv = project_points(points_xyz, intrinsics)
            inside, distances = self._containment(points_uv, mask_bool)
            metrics = {
                "keypoints_projected": float(len(points_uv)),
                "keypoints_inside_fraction": float(np.mean(inside)) if len(inside) else 0.0,
                "out_of_mask_distance_px_mean": float(np.mean(distances)) if distances else 0.0,
                "out_of_mask_distance_px_max": float(np.max(distances)) if distances else 0.0,
            }
            results.append(
                CheckResult(
                    check=self.name,
                    episode_idx=clip.episode_idx,
                    frame_idx=clip.frame_idx_at(offset),
                    metrics=metrics,
                    flag=None,
                    reason="measured only; optional SAM3 mask containment is uncalibrated",
                )
            )
        return results

    def _mask_at(
        self,
        masks: dict[int, np.ndarray] | list[np.ndarray] | np.ndarray,
        frame_idx: int,
        offset: int,
    ) -> np.ndarray | None:
        if isinstance(masks, dict):
            return masks.get(frame_idx)
        if isinstance(masks, list):
            return masks[offset] if offset < len(masks) else None
        array = np.asarray(masks)
        if array.ndim == 2:
            return array
        if array.ndim == 3 and offset < array.shape[0]:
            return array[offset]
        return None

    def _containment(
        self,
        points_uv: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[list[bool], list[float]]:
        height, width = mask.shape
        mask_pixels = np.argwhere(mask)
        inside: list[bool] = []
        out_distances: list[float] = []
        for u, v in points_uv:
            x = int(round(u))
            y = int(round(v))
            is_inside = 0 <= x < width and 0 <= y < height and bool(mask[y, x])
            inside.append(is_inside)
            if not is_inside:
                if mask_pixels.size == 0 or not np.isfinite([u, v]).all():
                    out_distances.append(float("inf"))
                else:
                    distances = np.sqrt((mask_pixels[:, 1] - u) ** 2 + (mask_pixels[:, 0] - v) ** 2)
                    out_distances.append(float(np.min(distances)))
        return inside, [value for value in out_distances if np.isfinite(value)]
