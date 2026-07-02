"""Clip-level supplier hand-quality missing-keypoint rule."""

from __future__ import annotations

import json

import numpy as np

from precheck.base import BaseCheck
from precheck.registry import register
from qc_common.keypoints import select_hand_joints
from qc_common.types import CheckResult, ClipInputs


@register
class KeypointMissingCheck(BaseCheck):
    """Apply the 10s-window/1s-allowed hand quality acceptance rule."""

    name = "keypoint_missing"
    granularity = "clip"

    def __init__(self, config: dict) -> None:
        self.sides = config.get("sides", ["left", "right"])
        self.joint_names = config.get("joint_names")
        self.config_fps = config.get("fps")
        self.window_seconds = float(config.get("window_seconds", 10.0))
        self.allowed_missing_seconds = float(config.get("allowed_missing_seconds", 1.0))
        self.repair_records: list[dict] = []

    def run(self, clip: ClipInputs) -> list[CheckResult]:
        quality_hand = clip.quality_hand
        if quality_hand is None:
            return []

        quality = np.asarray(quality_hand, dtype=np.float64)
        if quality.ndim != 2 or quality.shape[1] < 2:
            return []

        num_frames = min(clip.num_frames, quality.shape[0])
        fps = float(self.config_fps or clip.fps or 30.0)
        window_frames = max(1, int(round(fps * self.window_seconds)))
        allowed_missing_frames = max(0, int(round(fps * self.allowed_missing_seconds)))
        low_quality = quality[:num_frames, :2] < 0.5
        constant_zero_confidence_joints = self._constant_zero_confidence_joints(
            clip.confidences,
            num_frames,
        )
        acceptance_joints = self._acceptance_joints(clip)

        self.repair_records = [
            {
                "episode_idx": clip.episode_idx,
                "frame_idx": clip.frame_idx_at(frame_offset),
                "hand": hand,
            }
            for frame_offset in range(num_frames)
            for hand_index, hand in enumerate(("left", "right"))
            if low_quality[frame_offset, hand_index]
        ]

        results: list[CheckResult] = []
        for frame_offset in range(num_frames):
            start = max(0, frame_offset - window_frames + 1)
            window = low_quality[start : frame_offset + 1]
            missing_left = int(np.sum(window[:, 0]))
            missing_right = int(np.sum(window[:, 1]))
            flag = (
                missing_left > allowed_missing_frames
                or missing_right > allowed_missing_frames
            )
            reason = {
                "rule": "quality_hand < 0.5; confidence is not an absence signal",
                "low_quality_hands": [
                    hand
                    for hand_index, hand in enumerate(("left", "right"))
                    if low_quality[frame_offset, hand_index]
                ],
                "constant_zero_confidence_joints": constant_zero_confidence_joints,
            }
            results.append(
                CheckResult(
                    check=self.name,
                    episode_idx=clip.episode_idx,
                    frame_idx=clip.frame_idx_at(frame_offset),
                    metrics={
                        "quality_low_left": float(low_quality[frame_offset, 0]),
                        "quality_low_right": float(low_quality[frame_offset, 1]),
                        "missing_frames_in_10s_window_left": float(missing_left),
                        "missing_frames_in_10s_window_right": float(missing_right),
                        "missing_fraction_in_10s_window_left": float(missing_left / len(window)),
                        "missing_fraction_in_10s_window_right": float(missing_right / len(window)),
                        "window_frames": float(window_frames),
                        "allowed_missing_frames": float(allowed_missing_frames),
                        "acceptance_joint_count": float(len(acceptance_joints)),
                        "constant_zero_confidence_joint_count": float(
                            len(constant_zero_confidence_joints)
                        ),
                    },
                    flag=flag,
                    reason=json.dumps(reason, sort_keys=True),
                )
            )
        return results

    def _acceptance_joints(self, clip: ClipInputs) -> list[str]:
        keypoints = clip.keypoints
        if not keypoints:
            return []
        if self.joint_names is not None:
            return [joint for joint in self.joint_names if joint in keypoints]
        return select_hand_joints(sorted(keypoints), self.sides)

    def _constant_zero_confidence_joints(
        self,
        confidences: dict[str, np.ndarray] | None,
        num_frames: int,
    ) -> list[str]:
        if not confidences:
            return []
        # Data-fixed constant-zero confidence is informational, not a quality signal.
        return sorted(
            joint
            for joint, values in confidences.items()
            if len(values) >= num_frames
            and np.all(np.asarray(values[:num_frames], dtype=np.float64) == 0.0)
        )
