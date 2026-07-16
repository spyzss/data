"""Per-frame canonical hand keypoint existence validation."""

from __future__ import annotations

import json

import numpy as np

from precheck.base import BaseCheck
from precheck.registry import register
from qc_common.keypoint_validity import inspect_hand_keypoints
from qc_common.keypoints import acceptance_joint_names, select_hand_joints
from qc_common.types import CheckResult, ClipInputs


@register
class KeypointMissingCheck(BaseCheck):
    """Validate canonical coordinates independently from supplier quality signals."""

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
        quality = (
            np.asarray(quality_hand, dtype=np.float64)
            if quality_hand is not None
            else None
        )
        quality_available = bool(
            quality is not None and quality.ndim == 2 and quality.shape[1] >= 2
        )
        num_frames = clip.num_frames
        fps = float(self.config_fps or clip.fps or 30.0)
        window_frames = max(1, int(round(fps * self.window_seconds)))
        allowed_missing_frames = max(0, int(round(fps * self.allowed_missing_seconds)))
        low_quality = np.zeros((num_frames, 2), dtype=bool)
        if quality_available and quality is not None:
            quality_frames = min(num_frames, quality.shape[0])
            low_quality[:quality_frames] = quality[:quality_frames, :2] < 0.5
        constant_zero_confidence_joints = self._constant_zero_confidence_joints(
            clip.confidences,
            num_frames,
        )
        acceptance_joints = self._acceptance_joints(clip)

        validity_by_side = {
            side: [
                inspect_hand_keypoints(
                    clip.keypoints,
                    side=side,
                    frame_offset=frame_offset,
                    source_value_count=self._source_value_count(
                        clip,
                        side,
                        frame_offset,
                    ),
                )
                for frame_offset in range(num_frames)
            ]
            for side in self.sides
        }
        self.repair_records = []
        for side, validity_rows in validity_by_side.items():
            for frame_offset, validity in enumerate(validity_rows):
                if validity.is_valid:
                    continue
                self.repair_records.append(
                    {
                        "episode_idx": clip.episode_idx,
                        "frame_idx": clip.frame_idx_at(frame_offset),
                        "hand": side,
                        "invalid_reasons": list(validity.invalid_reasons),
                    }
                )

        results: list[CheckResult] = []
        for frame_offset in range(num_frames):
            start = max(0, frame_offset - window_frames + 1)
            invalid_counts = {
                side: sum(
                    not validity.is_valid
                    for validity in validity_by_side[side][start : frame_offset + 1]
                )
                for side in self.sides
            }
            invalid_hands = [
                side
                for side in self.sides
                if not validity_by_side[side][frame_offset].is_valid
            ]
            flag = bool(invalid_hands)
            supplier_quality_signal = (
                "not_provided"
                if not quality_available
                else ("low" if bool(np.any(low_quality[frame_offset])) else "provided_ok")
            )
            reason = {
                "rule": "canonical 21-point coordinate existence; quality_hand is informational",
                "invalid_hands": invalid_hands,
                "invalid_reasons": {
                    side: list(validity_by_side[side][frame_offset].invalid_reasons)
                    for side in invalid_hands
                },
                "constant_zero_confidence_joints": constant_zero_confidence_joints,
                "supplier_quality_signal": supplier_quality_signal,
            }
            metrics: dict[str, object] = {
                "keypoint_presence_verdict": "fail" if flag else "pass",
                "keypoint_presence_invalid": flag,
                "quality_low_left": float(low_quality[frame_offset, 0]),
                "quality_low_right": float(low_quality[frame_offset, 1]),
                "supplier_quality_signal": supplier_quality_signal,
                "window_frames": float(window_frames),
                "allowed_missing_frames": float(allowed_missing_frames),
                "acceptance_joint_count": float(len(acceptance_joints)),
                "expected_keypoint_count_per_hand": 21.0,
                "constant_zero_confidence_joint_count": float(
                    len(constant_zero_confidence_joints)
                ),
            }
            for side in self.sides:
                validity = validity_by_side[side][frame_offset]
                metrics.update(
                    {
                        f"keypoint_existence_invalid_{side}": not validity.is_valid,
                        f"valid_keypoint_count_{side}": float(
                            validity.valid_point_count
                        ),
                        f"finite_keypoint_count_{side}": float(
                            validity.finite_point_count
                        ),
                        f"missing_keypoint_count_{side}": float(
                            validity.expected_point_count - validity.valid_point_count
                        ),
                        f"all_zero_{side}": validity.all_zero,
                        f"all_identical_{side}": validity.all_identical,
                        f"invalid_reasons_{side}": list(validity.invalid_reasons),
                        f"missing_frames_in_10s_window_{side}": float(
                            invalid_counts[side]
                        ),
                        f"missing_fraction_in_10s_window_{side}": float(
                            invalid_counts[side] / (frame_offset - start + 1)
                        ),
                    }
                )
            results.append(
                CheckResult(
                    check=self.name,
                    episode_idx=clip.episode_idx,
                    frame_idx=clip.frame_idx_at(frame_offset),
                    metrics=metrics,
                    flag=flag,
                    reason=json.dumps(reason, sort_keys=True),
                )
            )
        return results

    @staticmethod
    def _source_value_count(
        clip: ClipInputs,
        side: str,
        frame_offset: int,
    ) -> int | None:
        counts_by_side = getattr(clip, "keypoint_source_value_counts", None)
        if not isinstance(counts_by_side, dict):
            return None
        counts = counts_by_side.get(side)
        if counts is None or len(counts) <= frame_offset:
            return None
        return int(counts[frame_offset])

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
