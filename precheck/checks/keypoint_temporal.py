"""Clip-level 3D keypoint temporal quality metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
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


TEMPORAL_SAMPLING_METHOD = "nearest_monotonic_no_reuse"
NATIVE_TEMPORAL_METRIC_FIELD_NAMES = (
    "joint_displacement_m_max",
    "joint_acceleration_m_s2_max",
    "joint_angle_change_deg_max",
    "rotation_delta_max",
)
STANDARDIZED_TEMPORAL_METRIC_FIELD_NAMES = (
    "joint_displacement_standardized_m_max",
    "joint_acceleration_standardized_m_s2_max",
    "joint_angle_change_standardized_deg_max",
    "rotation_delta_standardized_max",
)


@dataclass(frozen=True)
class _StandardizedSample:
    frame_offset: int
    segment_id: int
    segment_start_reason: str | None


@dataclass(frozen=True)
class _TemporalSamplingPlan:
    timestamps_seconds: np.ndarray | None
    timestamp_source: str
    samples: tuple[_StandardizedSample, ...]
    audit: dict[str, Any]


def _finite_positive(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) and numeric > 0.0 else None


def _source_frames(
    clip: ClipInputs,
    num_frames: int,
) -> list[int]:
    source_indices = getattr(clip, "source_frame_indices", clip.frame_indices)
    fallback_start = int(getattr(clip, "clip_start_frame", 0))
    return [
        source_frame_at(
            source_indices,
            frame_offset,
            fallback_start_frame=fallback_start,
        )
        for frame_offset in range(num_frames)
    ]


def _timestamp_axis(
    clip: ClipInputs,
    *,
    num_frames: int,
    source_frames: list[int],
    configured_source: str,
    source_fps: float | None,
) -> tuple[np.ndarray | None, str]:
    if configured_source not in {"auto", "timestamps_ns", "frame_index_source_fps"}:
        raise ValueError(
            "temporal_timestamp_source must be auto, timestamps_ns, "
            "or frame_index_source_fps"
        )
    if configured_source in {"auto", "timestamps_ns"}:
        raw = clip.timestamps_ns
        if raw is not None:
            timestamps = np.full(num_frames, np.nan, dtype=np.float64)
            available = min(num_frames, len(raw))
            timestamps[:available] = (
                np.asarray(raw[:available], dtype=np.float64)
                / 1_000_000_000.0
            )
            return timestamps, "timestamps_ns"
        if configured_source == "timestamps_ns":
            return None, "unavailable"
    if source_fps is None:
        return None, "unavailable"
    base = source_frames[0] if source_frames else 0
    return (
        np.asarray(
            [(frame - base) / source_fps for frame in source_frames],
            dtype=np.float64,
        ),
        "frame_index_source_fps",
    )


def _timestamp_segments(
    timestamps: np.ndarray,
    *,
    source_fps: float | None,
    max_gap_factor: float,
) -> tuple[list[tuple[list[int], str | None]], dict[str, int]]:
    invalid_count = int(np.sum(~np.isfinite(timestamps) | (timestamps < 0.0)))
    positive_deltas = np.diff(timestamps)
    finite_positive_deltas = positive_deltas[
        np.isfinite(positive_deltas) & (positive_deltas > 0.0)
    ]
    nominal_dt = (
        1.0 / source_fps
        if source_fps is not None
        else float(np.median(finite_positive_deltas))
        if finite_positive_deltas.size
        else math.nan
    )
    max_gap = (
        max_gap_factor * nominal_dt
        if math.isfinite(nominal_dt) and nominal_dt > 0.0
        else math.inf
    )
    duplicate_count = 0
    non_monotonic_count = 0
    gap_break_count = 0
    segments: list[tuple[list[int], str | None]] = []
    active: list[int] = []
    active_reason: str | None = None
    previous_valid_offset: int | None = None

    def flush() -> None:
        nonlocal active, active_reason
        if active:
            segments.append((active, active_reason))
        active = []
        active_reason = None

    for offset, timestamp in enumerate(timestamps):
        if not math.isfinite(float(timestamp)) or float(timestamp) < 0.0:
            flush()
            previous_valid_offset = None
            active_reason = "invalid_timestamp"
            continue
        break_reason: str | None = None
        if previous_valid_offset is not None:
            delta = float(timestamp - timestamps[previous_valid_offset])
            if delta == 0.0:
                duplicate_count += 1
                break_reason = "duplicate_timestamp"
            elif delta < 0.0:
                non_monotonic_count += 1
                break_reason = "non_monotonic_timestamp"
            elif delta > max_gap:
                gap_break_count += 1
                break_reason = "timestamp_gap"
        if break_reason is not None:
            flush()
            active_reason = break_reason
        active.append(offset)
        previous_valid_offset = offset
    flush()
    return segments, {
        "invalid_timestamp_count": invalid_count,
        "duplicate_timestamp_count": duplicate_count,
        "non_monotonic_timestamp_count": non_monotonic_count,
        "temporal_gap_break_count": gap_break_count,
    }


def build_standardized_temporal_sampling_plan(
    clip: ClipInputs,
    *,
    num_frames: int,
    target_hz: float,
    decision_timebase: str,
    configured_timestamp_source: str,
    source_fps: float | None,
    max_gap_factor: float,
) -> _TemporalSamplingPlan:
    """Build an auditable target-time to source-frame mapping without reuse."""
    if not math.isfinite(target_hz) or target_hz <= 0.0:
        raise ValueError("temporal_target_hz must be finite and positive")
    if not math.isfinite(max_gap_factor) or max_gap_factor <= 1.0:
        raise ValueError("temporal_max_gap_factor must be greater than 1")
    if decision_timebase not in {"standardized", "native"}:
        raise ValueError("decision_timebase must be standardized or native")
    source_frames = _source_frames(clip, num_frames)
    timestamps, timestamp_source = _timestamp_axis(
        clip,
        num_frames=num_frames,
        source_frames=source_frames,
        configured_source=configured_timestamp_source,
        source_fps=source_fps,
    )
    base_audit: dict[str, Any] = {
        "source_fps": source_fps,
        "temporal_target_hz": target_hz,
        "temporal_decision_timebase": decision_timebase,
        "timestamp_source": timestamp_source,
        "sampling_method": TEMPORAL_SAMPLING_METHOD,
        "source_frame_count": num_frames,
        "standardized_sample_count": 0,
        "duplicate_source_frame_drop_count": 0,
        "invalid_timestamp_count": 0,
        "duplicate_timestamp_count": 0,
        "non_monotonic_timestamp_count": 0,
        "temporal_gap_break_count": 0,
        "decision_metric_source": (
            "standardized_30hz"
            if decision_timebase == "standardized"
            else "native_source_fps"
        ),
        "standardized_metric_source": "standardized_30hz",
        "native_metric_source": (
            "source_fps" if source_fps is not None else "unavailable"
        ),
        "native_metric_field_names": list(NATIVE_TEMPORAL_METRIC_FIELD_NAMES),
        "standardized_metric_field_names": list(
            STANDARDIZED_TEMPORAL_METRIC_FIELD_NAMES
        ),
    }
    if timestamps is None:
        return _TemporalSamplingPlan(None, timestamp_source, (), base_audit)

    segments, timestamp_audit = _timestamp_segments(
        timestamps,
        source_fps=source_fps,
        max_gap_factor=max_gap_factor,
    )
    samples: list[_StandardizedSample] = []
    duplicate_source_drop_count = 0
    period = 1.0 / target_hz
    for segment_id, (offsets, start_reason) in enumerate(segments):
        if not offsets:
            continue
        segment_times = timestamps[offsets]
        target = float(segment_times[0])
        target_index = 0
        last_selected = -1
        segment_sample_added = False
        final_time = float(segment_times[-1])
        tolerance = max(1e-12, period * 1e-9)
        while target <= final_time + tolerance:
            position = int(np.searchsorted(segment_times, target, side="left"))
            candidates = [
                candidate
                for candidate in (position - 1, position)
                if 0 <= candidate < len(offsets)
            ]
            if not candidates:
                break
            nearest_position = min(
                candidates,
                key=lambda candidate: (
                    abs(float(segment_times[candidate]) - target),
                    candidate,
                ),
            )
            selected_offset = offsets[nearest_position]
            if selected_offset <= last_selected:
                duplicate_source_drop_count += 1
            else:
                samples.append(
                    _StandardizedSample(
                        selected_offset,
                        segment_id,
                        start_reason if not segment_sample_added else None,
                    )
                )
                last_selected = selected_offset
                segment_sample_added = True
            target_index += 1
            target = float(segment_times[0]) + target_index * period

    audit = {
        **base_audit,
        **timestamp_audit,
        "standardized_sample_count": len(samples),
        "duplicate_source_frame_drop_count": duplicate_source_drop_count,
        "source_frame_mapping": [source_frames[item.frame_offset] for item in samples],
    }
    return _TemporalSamplingPlan(timestamps, timestamp_source, tuple(samples), audit)


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
        self.temporal_decision_timebase = str(
            config.get("temporal_decision_timebase", "standardized")
        )
        if self.temporal_decision_timebase not in {"standardized", "native"}:
            raise ValueError(
                "temporal_decision_timebase must be standardized or native"
            )
        self.temporal_target_hz = float(config.get("temporal_target_hz", 30.0))
        self.temporal_timestamp_source = str(
            config.get("temporal_timestamp_source", "auto")
        )
        self.temporal_max_gap_factor = float(
            config.get("temporal_max_gap_factor", 3.0)
        )

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

        source_fps = _finite_positive(self.config_fps) or _finite_positive(clip.fps)
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
                if source_fps is not None:
                    metrics.update(
                        finite_stats(
                            np.asarray(displacements) * source_fps,
                            "joint_velocity_m_s",
                        )
                    )

                length_changes = [
                    abs(length - previous_bone_lengths[bone])
                    for bone, length in frame_bone_lengths.items()
                    if bone in previous_bone_lengths
                ]
                metrics.update(finite_stats(length_changes, "bone_length_change_m"))

                if source_fps is not None:
                    for name in joint_names:
                        velocities[name] = (
                            points[name][frame_offset]
                            - points[name][frame_offset - 1]
                        ) * source_fps

                if projected is not None:
                    pixel_displacements = [
                        np.linalg.norm(projected[name][frame_offset] - projected[name][frame_offset - 1])
                        for name in joint_names
                    ]
                    metrics.update(finite_stats(pixel_displacements, "displacement_2d_px"))

            if (
                pair_eligible
                and source_fps is not None
                and frame_offset > 1
                and previous_velocity
            ):
                accelerations = [
                    np.linalg.norm(velocities[name] - previous_velocity[name])
                    * source_fps
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

        sampling_plan = build_standardized_temporal_sampling_plan(
            clip,
            num_frames=num_frames,
            target_hz=self.temporal_target_hz,
            decision_timebase=self.temporal_decision_timebase,
            configured_timestamp_source=self.temporal_timestamp_source,
            source_fps=source_fps,
            max_gap_factor=self.temporal_max_gap_factor,
        )
        self._apply_standardized_metrics(
            results=results,
            points=points,
            rotations=rotations,
            angle_triples=angle_triples,
            sampling_plan=sampling_plan,
            source_frames=[source_frame(offset) for offset in range(num_frames)],
            eligible_ranges=eligible_ranges,
        )
        return results

    def _apply_standardized_metrics(
        self,
        *,
        results: list[CheckResult],
        points: dict[str, np.ndarray],
        rotations: dict[str, np.ndarray],
        angle_triples: list[tuple[str, str, str]],
        sampling_plan: _TemporalSamplingPlan,
        source_frames: list[int],
        eligible_ranges: Any,
    ) -> None:
        for frame_offset, result in enumerate(results):
            result.metrics.update(
                {
                    "standardized_sample_selected": False,
                    "standardized_temporal_pair_eligible": False,
                    "standardized_temporal_triple_eligible": False,
                    "standardized_temporal_pair_skip_reason": (
                        "timestamp_source_unavailable"
                        if sampling_plan.timestamps_seconds is None
                        else "not_selected_for_standardized_timebase"
                    ),
                    "temporal_target_hz": self.temporal_target_hz,
                    "timestamp_source": sampling_plan.timestamp_source,
                    "sampling_method": TEMPORAL_SAMPLING_METHOD,
                    "decision_metric_source": (
                        "standardized_30hz"
                        if self.temporal_decision_timebase == "standardized"
                        else "native_source_fps"
                    ),
                    "standardized_metric_source": "standardized_30hz",
                    "native_metric_source": (
                        "source_fps"
                        if sampling_plan.audit.get("source_fps") is not None
                        else "unavailable"
                    ),
                    "local_frame_idx": frame_offset,
                    "source_frame_idx": source_frames[frame_offset],
                }
            )
            if frame_offset == 0:
                result.metrics["temporal_sampling_audit"] = dict(
                    sampling_plan.audit
                )
        if sampling_plan.timestamps_seconds is None:
            return

        timestamps = sampling_plan.timestamps_seconds
        frame_usable = [
            self._frame_is_usable(points, frame_offset)
            for frame_offset in range(len(results))
        ]
        selected = list(sampling_plan.samples)
        pair_eligible_by_sample: list[bool] = []
        for sample_index, sample in enumerate(selected):
            offset = sample.frame_offset
            result = results[offset]
            timestamp = float(timestamps[offset])
            metrics = result.metrics
            metrics.update(
                {
                    "standardized_sample_selected": True,
                    "standardized_sample_index": sample_index,
                    "anchor_local_frame": offset,
                    "anchor_source_frame": source_frames[offset],
                    "anchor_timestamp_seconds": timestamp,
                }
            )
            previous = selected[sample_index - 1] if sample_index > 0 else None
            same_segment = (
                previous is not None and previous.segment_id == sample.segment_id
            )
            pair_reason: str | None = None
            pair_eligible = bool(same_segment)
            if not same_segment:
                pair_reason = sample.segment_start_reason or "insufficient_history"
            elif not is_source_frame_eligible(
                source_frames[offset], eligible_ranges
            ):
                pair_eligible = False
                pair_reason = "current_frame_excluded"
            elif not is_source_frame_eligible(
                source_frames[previous.frame_offset], eligible_ranges
            ):
                pair_eligible = False
                pair_reason = "previous_frame_excluded"
            elif not frame_usable[offset]:
                pair_eligible = False
                pair_reason = "current_frame_unusable"
            elif not frame_usable[previous.frame_offset]:
                pair_eligible = False
                pair_reason = "previous_frame_unusable"
            elif any(
                not is_source_frame_eligible(
                    source_frames[intermediate_offset], eligible_ranges
                )
                for intermediate_offset in range(
                    previous.frame_offset + 1,
                    offset,
                )
            ):
                pair_eligible = False
                pair_reason = "intermediate_frame_excluded"
            elif any(
                not frame_usable[intermediate_offset]
                for intermediate_offset in range(
                    previous.frame_offset + 1,
                    offset,
                )
            ):
                pair_eligible = False
                pair_reason = "intermediate_frame_unusable"

            triple_eligible = False
            previous_previous = (
                selected[sample_index - 2] if sample_index > 1 else None
            )
            if (
                pair_eligible
                and previous_previous is not None
                and previous_previous.segment_id == sample.segment_id
                and pair_eligible_by_sample[sample_index - 1]
            ):
                triple_eligible = True

            metrics["standardized_temporal_pair_eligible"] = pair_eligible
            metrics["standardized_temporal_triple_eligible"] = triple_eligible
            if pair_reason is not None:
                metrics["standardized_temporal_pair_skip_reason"] = pair_reason
            else:
                metrics.pop("standardized_temporal_pair_skip_reason", None)
            if triple_eligible:
                evidence_offsets = [
                    previous_previous.frame_offset,
                    previous.frame_offset,
                    offset,
                ]
            elif pair_eligible:
                evidence_offsets = [previous.frame_offset, offset]
            else:
                evidence_offsets = [offset]
            metrics["evidence_local_frames"] = evidence_offsets
            metrics["evidence_source_frames"] = [
                source_frames[item] for item in evidence_offsets
            ]
            metrics["evidence_timestamps"] = [
                float(timestamps[item]) for item in evidence_offsets
            ]

            current_velocity: dict[str, np.ndarray] = {}
            if pair_eligible:
                previous_offset = previous.frame_offset
                dt = timestamp - float(timestamps[previous_offset])
                metrics.update(
                    {
                        "actual_dt_seconds": dt,
                        "standardized_temporal_pair_start_frame": source_frames[
                            previous_offset
                        ],
                        "standardized_temporal_pair_end_frame": source_frames[offset],
                        "standardized_temporal_transition_attribution": "target_frame",
                    }
                )
                displacements = [
                    np.linalg.norm(points[name][offset] - points[name][previous_offset])
                    for name in points
                ]
                metrics.update(
                    finite_stats(
                        displacements,
                        "joint_displacement_standardized_m",
                    )
                )
                current_velocity = {
                    name: (points[name][offset] - points[name][previous_offset]) / dt
                    for name in points
                }
                metrics.update(
                    finite_stats(
                        [np.linalg.norm(value) for value in current_velocity.values()],
                        "joint_velocity_standardized_m_s",
                    )
                )
                previous_angles = self._angles(
                    points,
                    angle_triples,
                    previous_offset,
                )
                current_angles = self._angles(points, angle_triples, offset)
                metrics.update(
                    finite_stats(
                        [
                            abs(angle - previous_angles[triple])
                            for triple, angle in current_angles.items()
                            if triple in previous_angles
                        ],
                        "joint_angle_change_standardized_deg",
                    )
                )
                metrics.update(
                    finite_stats(
                        [
                            float(
                                np.linalg.norm(
                                    values[offset] - values[previous_offset],
                                    ord="fro",
                                )
                            )
                            for values in rotations.values()
                            if values.shape[0] > offset
                        ],
                        "rotation_delta_standardized",
                    )
                )

            if triple_eligible:
                previous_previous_offset = previous_previous.frame_offset
                previous_offset = previous.frame_offset
                previous_dt = float(
                    timestamps[previous_offset]
                    - timestamps[previous_previous_offset]
                )
                previous_velocity = {
                    name: (
                        points[name][previous_offset]
                        - points[name][previous_previous_offset]
                    )
                    / previous_dt
                    for name in points
                }
                velocity_time_delta = float(
                    (timestamps[offset] - timestamps[previous_previous_offset]) / 2.0
                )
                metrics["actual_velocity_time_delta_seconds"] = velocity_time_delta
                metrics["standardized_acceleration_time_delta_method"] = (
                    "successive_velocity_midpoint_delta"
                )
                metrics.update(
                    finite_stats(
                        [
                            np.linalg.norm(
                                current_velocity[name] - previous_velocity[name]
                            )
                            / velocity_time_delta
                            for name in points
                        ],
                        "joint_acceleration_standardized_m_s2",
                    )
                )
            pair_eligible_by_sample.append(pair_eligible)

    @staticmethod
    def _frame_is_usable(
        points: dict[str, np.ndarray],
        frame_offset: int,
    ) -> bool:
        coordinates = np.concatenate(
            [
                np.asarray(values[frame_offset], dtype=np.float64).reshape(-1)
                for values in points.values()
            ]
        )
        return bool(
            coordinates.size
            and np.all(np.isfinite(coordinates))
            and np.any(np.abs(coordinates) > 1e-12)
        )

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
