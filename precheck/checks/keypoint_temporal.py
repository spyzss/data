"""Clip-level 3D keypoint temporal quality metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from precheck.base import BaseCheck
from precheck.registry import register
from qc_common.keypoints import (
    derive_angle_triples,
    derive_finger_bones,
    finite_stats,
    project_points,
    select_hand_joints,
)
from qc_common.frame_survival import (
    is_source_frame_eligible,
    source_frame_at,
    temporal_transition_lineage,
)
from qc_common.types import CheckResult, ClipInputs


@register
class KeypointTemporalCheck(BaseCheck):
    """Measure temporal instability in supplier 3D skeleton transforms."""

    name = "keypoint_temporal"
    granularity = "clip"

    def __init__(self, config: dict) -> None:
        self.sides = config.get("sides", ["left", "right"])
        self.joint_names = config.get("joint_names")
        self.config_fps = config.get("fps")
        self.project_2d = bool(config.get("project_2d", True))
        self.min_angle_degrees = float(config.get("min_angle_degrees", 5.0))
        self.max_angle_degrees = float(config.get("max_angle_degrees", 175.0))

    def run(self, clip: ClipInputs) -> list[CheckResult]:
        keypoints = clip.keypoints
        if not keypoints:
            return []

        all_joint_names = sorted(keypoints)
        topology_agnostic = getattr(
            clip, "topology_agnostic_joint_names", None
        )
        joint_names = (
            list(topology_agnostic)
            if topology_agnostic is not None
            else self.joint_names
            or select_hand_joints(all_joint_names, self.sides)
        )
        joint_names = [name for name in joint_names if name in keypoints]
        if not joint_names:
            return []

        fps = float(self.config_fps or clip.fps or 30.0)
        num_frames = min(clip.num_frames, *(keypoints[name].shape[0] for name in joint_names))
        bones = derive_finger_bones(joint_names)
        angle_triples = derive_angle_triples(joint_names)

        points = {
            name: np.asarray(keypoints[name], dtype=np.float64)[:num_frames]
            for name in joint_names
        }
        rotations = self._prepare_rotations(clip.rotations, joint_names, num_frames)
        projected = self._project(points, clip.intrinsics, num_frames)
        confidences = clip.confidences or {}
        quality_hand = clip.quality_hand

        results: list[CheckResult] = []
        previous_velocity: dict[str, np.ndarray] = {}
        previous_bone_lengths: dict[tuple[str, str], float] = {}
        previous_angles: dict[tuple[str, str, str], float] = {}
        source_indices = getattr(clip, "source_frame_indices", clip.frame_indices)
        eligible_ranges = getattr(clip, "eligible_frame_ranges", None)
        fallback_start = int(getattr(clip, "clip_start_frame", 0))

        def source_frame(frame_offset: int) -> int:
            return source_frame_at(
                source_indices,
                frame_offset,
                fallback_start_frame=fallback_start,
            )

        def is_eligible(frame_offset: int) -> bool:
            return is_source_frame_eligible(source_frame(frame_offset), eligible_ranges)

        for frame_offset in range(num_frames):
            current_eligible = is_eligible(frame_offset)
            if not current_eligible:
                results.append(
                    CheckResult(
                        check=self.name,
                        episode_idx=clip.episode_idx,
                        frame_idx=clip.frame_idx_at(frame_offset),
                        metrics={
                            "joint_count": float(len(joint_names)),
                            "bone_count": float(len(bones)),
                            "temporal_pair_eligible": False,
                            "skipped_pair_count": float(frame_offset > 0),
                            "temporal_pair_skip_reason": "current_frame_excluded",
                        },
                        flag=None,
                        reason="raw temporal pair skipped: current_frame_excluded",
                        severity="uncalibrated",
                    )
                )
                previous_velocity = {}
                previous_bone_lengths = {}
                previous_angles = {}
                continue
            metrics: dict[str, Any] = {
                "joint_count": float(len(joint_names)),
                "bone_count": float(len(bones)),
            }
            metrics.update(self._position_audit_metrics(points, frame_offset))
            pair_eligible = frame_offset > 0 and is_eligible(frame_offset - 1)
            metrics["temporal_pair_eligible"] = pair_eligible
            metrics["skipped_pair_count"] = float(frame_offset > 0 and not pair_eligible)
            if pair_eligible:
                metrics.update(
                    temporal_transition_lineage(
                        target_frame=source_frame(frame_offset),
                        pair_start_frame=source_frame(frame_offset - 1),
                        pair_end_frame=source_frame(frame_offset),
                    ).to_metrics()
                )
            if frame_offset > 0 and not pair_eligible:
                metrics["temporal_pair_skip_reason"] = "previous_frame_excluded"

            # Rigid-rig data has near-zero bone-length variance; keep these
            # only as a rig sanity check, not as a drift signal.
            frame_bone_lengths = self._bone_lengths(points, bones, frame_offset)
            metrics.update(finite_stats(list(frame_bone_lengths.values()), "bone_length_m"))
            metrics.update(self._bone_ratio_metrics(list(frame_bone_lengths.values())))
            frame_angles = self._angles(points, angle_triples, frame_offset)
            metrics.update(self._angle_metrics(frame_angles))
            metrics.update(self._confidence_metrics(confidences, joint_names, frame_offset))
            metrics.update(self._quality_metrics(quality_hand, frame_offset))

            velocities: dict[str, np.ndarray] = {}
            if pair_eligible:
                angle_changes = [
                    abs(angle - previous_angles[triple])
                    for triple, angle in frame_angles.items()
                    if triple in previous_angles
                ]
                metrics.update(finite_stats(angle_changes, "joint_angle_change_deg"))

                rotation_deltas = self._rotation_deltas(rotations, frame_offset)
                metrics.update(finite_stats(rotation_deltas, "rotation_delta"))

                displacements = [
                    np.linalg.norm(points[name][frame_offset] - points[name][frame_offset - 1])
                    for name in joint_names
                ]
                metrics.update(finite_stats(displacements, "joint_displacement_m"))
                metrics.update(finite_stats(np.asarray(displacements) * fps, "joint_velocity_m_s"))

                length_changes = [
                    abs(length - previous_bone_lengths[bone])
                    for bone, length in frame_bone_lengths.items()
                    if bone in previous_bone_lengths
                ]
                metrics.update(finite_stats(length_changes, "bone_length_change_m"))

                for name in joint_names:
                    velocities[name] = (points[name][frame_offset] - points[name][frame_offset - 1]) * fps

                if projected is not None:
                    pixel_displacements = [
                        np.linalg.norm(projected[name][frame_offset] - projected[name][frame_offset - 1])
                        for name in joint_names
                    ]
                    metrics.update(finite_stats(pixel_displacements, "displacement_2d_px"))

            if pair_eligible and frame_offset > 1 and previous_velocity:
                accelerations = [
                    np.linalg.norm(velocities[name] - previous_velocity[name]) * fps
                    for name in joint_names
                    if name in velocities and name in previous_velocity
                ]
                metrics.update(finite_stats(accelerations, "joint_acceleration_m_s2"))

            results.append(
                CheckResult(
                    check=self.name,
                    episode_idx=clip.episode_idx,
                    frame_idx=clip.frame_idx_at(frame_offset),
                    metrics=metrics,
                    flag=None,
                    reason="raw temporal keypoint metrics; thresholds uncalibrated",
                    severity="uncalibrated",
                )
            )
            previous_velocity = velocities
            previous_bone_lengths = frame_bone_lengths
            previous_angles = frame_angles

        return results

    @staticmethod
    def _position_audit_metrics(
        points: dict[str, np.ndarray],
        frame_offset: int,
    ) -> dict[str, float]:
        coordinates = np.concatenate(
            [
                np.asarray(values[frame_offset], dtype=np.float64).reshape(-1)
                for values in points.values()
                if values.shape[0] > frame_offset
            ]
        )
        finite = np.isfinite(coordinates)
        return {
            "joint_position_abs_m_max": (
                float(np.max(np.abs(coordinates[finite])))
                if np.any(finite)
                else math.nan
            ),
            "joint_position_finite_coordinate_count": float(np.sum(finite)),
            "joint_position_nonfinite_coordinate_count": float(
                coordinates.size - np.sum(finite)
            ),
        }

    def _prepare_rotations(
        self,
        rotations: dict[str, np.ndarray] | None,
        joint_names: list[str],
        num_frames: int,
    ) -> dict[str, np.ndarray]:
        if not rotations:
            return {}
        return {
            name: np.asarray(rotations[name], dtype=np.float64)[:num_frames]
            for name in joint_names
            if name in rotations and np.asarray(rotations[name]).ndim == 3
        }

    def _rotation_deltas(
        self,
        rotations: dict[str, np.ndarray],
        frame_offset: int,
    ) -> list[float]:
        return [
            float(np.linalg.norm(values[frame_offset] - values[frame_offset - 1], ord="fro"))
            for values in rotations.values()
            if values.shape[0] > frame_offset
        ]

    def _project(
        self,
        points: dict[str, np.ndarray],
        intrinsics: np.ndarray | None,
        num_frames: int,
    ) -> dict[str, np.ndarray] | None:
        if not self.project_2d or intrinsics is None:
            return None
        return {
            name: np.vstack(
                [project_points(values[frame_idx : frame_idx + 1], intrinsics)[0] for frame_idx in range(num_frames)]
            )
            for name, values in points.items()
        }

    def _bone_lengths(
        self,
        points: dict[str, np.ndarray],
        bones: list[tuple[str, str]],
        frame_offset: int,
    ) -> dict[tuple[str, str], float]:
        return {
            bone: float(np.linalg.norm(points[bone[0]][frame_offset] - points[bone[1]][frame_offset]))
            for bone in bones
        }

    def _bone_ratio_metrics(self, lengths: list[float]) -> dict[str, float]:
        array = np.asarray(lengths, dtype=np.float64)
        array = array[np.isfinite(array) & (array > 1e-8)]
        if array.size < 2:
            return {}
        median = float(np.median(array))
        if median <= 1e-8:
            return {}
        ratios = array / median
        return {
            "bone_length_ratio_p95": float(np.percentile(ratios, 95)),
            "bone_length_ratio_spread": float(np.max(ratios) - np.min(ratios)),
        }

    def _angles(
        self,
        points: dict[str, np.ndarray],
        triples: list[tuple[str, str, str]],
        frame_offset: int,
    ) -> dict[tuple[str, str, str], float]:
        angles: dict[tuple[str, str, str], float] = {}
        for a, b, c in triples:
            v1 = points[a][frame_offset] - points[b][frame_offset]
            v2 = points[c][frame_offset] - points[b][frame_offset]
            denom = np.linalg.norm(v1) * np.linalg.norm(v2)
            if denom <= 1e-8:
                continue
            cos_angle = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
            angles[(a, b, c)] = float(np.degrees(np.arccos(cos_angle)))
        return angles

    def _angle_metrics(
        self,
        angles: dict[tuple[str, str, str], float],
    ) -> dict[str, float]:
        if not angles:
            return {}
        angle_array = np.asarray(list(angles.values()))
        violations = (angle_array < self.min_angle_degrees) | (
            angle_array > self.max_angle_degrees
        )
        return {
            "joint_angle_degrees_mean": float(np.mean(angle_array)),
            "joint_angle_degrees_min": float(np.min(angle_array)),
            "joint_angle_degrees_max": float(np.max(angle_array)),
            "joint_angle_violation_fraction": float(np.mean(violations)),
        }

    def _confidence_metrics(
        self,
        confidences: dict[str, np.ndarray],
        joint_names: list[str],
        frame_offset: int,
    ) -> dict[str, float]:
        values = [
            float(confidences[name][frame_offset])
            for name in joint_names
            if name in confidences and len(confidences[name]) > frame_offset
        ]
        if not values:
            return {}
        # Metacarpals may have constant-0 confidence by dataset design and are
        # outside the acceptance set; confidence is not a missing/absence signal.
        return {
            "confidence_mean": float(np.mean(values)),
            "confidence_min": float(np.min(values)),
            "confidence_zero_count": float(np.sum(np.asarray(values) == 0.0)),
        }

    def _quality_metrics(
        self,
        quality_hand: np.ndarray | None,
        frame_offset: int,
    ) -> dict[str, float]:
        if quality_hand is None or quality_hand.shape[0] <= frame_offset:
            return {}
        values = np.asarray(quality_hand[frame_offset], dtype=np.float64)
        metrics = {
            "quality_hand_left": float(values[0]),
            "quality_hand_low_fraction": float(np.mean(values < 0.5)),
        }
        if values.size > 1:
            metrics["quality_hand_right"] = float(values[1])
        return metrics
