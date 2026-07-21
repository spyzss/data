from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from precheck.checks.keypoint_temporal import KeypointTemporalCheck
from precheck.checks.skeleton_quality_score import SkeletonQualityScoreCheck
from qc_common.keypoints import ACCEPTANCE_FINGER_CHAINS
from qc_common.types import CheckResult, ClipInputs


def _hand_keypoints(side: str) -> dict[str, np.ndarray]:
    points: dict[str, np.ndarray] = {
        f"{side}Hand": np.zeros((3, 3), dtype=np.float64)
    }
    for finger_index, chain in enumerate(ACCEPTANCE_FINGER_CHAINS.values()):
        for point_index, base_name in enumerate(chain):
            point = np.asarray(
                [finger_index * 0.01, 0.03 + point_index * 0.02, 0.0],
                dtype=np.float64,
            )
            points[f"{side}{base_name}"] = np.repeat(point[None, :], 3, axis=0)
    return points


def test_temporal_skips_excluded_frame_and_adjacent_pair() -> None:
    clip = ClipInputs(
        episode_idx=1,
        frame_indices=[100, 101, 102],
        keypoints={**_hand_keypoints("left"), **_hand_keypoints("right")},
        fps=30.0,
    )
    clip.eligible_frame_ranges = ((100, 100), (102, 102))

    rows = KeypointTemporalCheck({}).run(clip)
    by_frame = {row.frame_idx: row for row in rows}

    assert by_frame[101].metrics["temporal_pair_eligible"] is False
    assert by_frame[101].metrics["temporal_pair_skip_reason"] == "current_frame_excluded"
    assert by_frame[102].metrics["temporal_pair_eligible"] is False
    assert by_frame[102].metrics["temporal_pair_skip_reason"] == "previous_frame_excluded"
    assert by_frame[102].metrics["skipped_pair_count"] == 1.0
    assert "joint_displacement_m_max" not in by_frame[102].metrics


def test_temporal_transition_uses_source_target_frame_attribution() -> None:
    clip = ClipInputs(
        episode_idx=1,
        frame_indices=[40, 41, 42],
        keypoints={**_hand_keypoints("left"), **_hand_keypoints("right")},
        fps=30.0,
    )

    rows = KeypointTemporalCheck({}).run(clip)

    assert rows[1].metrics["temporal_pair_start_frame"] == 40
    assert rows[1].metrics["temporal_pair_end_frame"] == 41
    assert rows[1].metrics["temporal_transition_attribution"] == "target_frame"


def _motion_clip(
    *,
    fps: float | None,
    source_frames: list[int],
    timestamps_seconds: np.ndarray | None = None,
    acceleration: float = 0.0,
    velocity: float = 1.0,
) -> ClipInputs:
    if timestamps_seconds is None:
        if fps is None:
            physical_time = np.arange(len(source_frames), dtype=np.float64)
        else:
            physical_time = np.arange(len(source_frames), dtype=np.float64) / fps
        timestamps_ns = None
    else:
        physical_time = np.asarray(timestamps_seconds, dtype=np.float64)
        timestamps_ns = np.rint(physical_time * 1_000_000_000.0).astype(np.int64)
    x = velocity * physical_time + 0.5 * acceleration * physical_time**2
    points = np.column_stack((x, np.ones_like(x), np.ones_like(x)))
    return ClipInputs(
        episode_idx=7,
        frame_indices=source_frames,
        keypoints={"probe_joint": points},
        timestamps_ns=timestamps_ns,
        fps=fps,
    )


def _standardized_check(**overrides: object) -> KeypointTemporalCheck:
    config: dict[str, object] = {
        "joint_names": ["probe_joint"],
        "project_2d": False,
        "temporal_decision_timebase": "standardized",
        "temporal_target_hz": 30.0,
        "temporal_timestamp_source": "auto",
        "temporal_max_gap_factor": 3.0,
    }
    config.update(overrides)
    return KeypointTemporalCheck(config)


def _sampled_rows(rows: list[CheckResult]) -> list[CheckResult]:
    return [
        row
        for row in rows
        if row.metrics.get("standardized_sample_selected") is True
    ]


def test_standardized_30hz_sampling_is_identity_and_matches_native_metrics() -> None:
    clip = _motion_clip(
        fps=30.0,
        source_frames=list(range(7)),
        acceleration=2.0,
    )

    rows = _standardized_check().run(clip)
    sampled = _sampled_rows(rows)

    assert [row.frame_idx for row in sampled] == list(range(7))
    assert sampled[-1].metrics["evidence_source_frames"] == [4, 5, 6]
    assert sampled[-1].metrics["anchor_source_frame"] == 6
    assert sampled[-1].metrics["timestamp_source"] == "frame_index_source_fps"
    assert math.isclose(
        sampled[-1].metrics["joint_displacement_standardized_m_max"],
        sampled[-1].metrics["joint_displacement_m_max"],
        rel_tol=1e-9,
    )
    assert math.isclose(
        sampled[-1].metrics["joint_acceleration_standardized_m_s2_max"],
        sampled[-1].metrics["joint_acceleration_m_s2_max"],
        rel_tol=1e-7,
    )


def test_standardized_temporal_produces_angle_and_rotation_metrics() -> None:
    keypoints = _hand_keypoints("left")
    keypoints["leftIndexFingerIntermediateTip"][2, 0] += 0.01
    rotations = {
        "leftHand": np.asarray(
            [
                np.eye(3),
                np.eye(3),
                np.asarray(
                    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
                ),
            ]
        )
    }
    clip = ClipInputs(
        episode_idx=8,
        frame_indices=[0, 1, 2],
        keypoints=keypoints,
        rotations=rotations,
        fps=30.0,
    )

    sampled = _sampled_rows(KeypointTemporalCheck({"project_2d": False}).run(clip))

    assert sampled[-1].metrics["joint_angle_change_standardized_deg_max"] > 0.0
    assert sampled[-1].metrics["rotation_delta_standardized_max"] > 0.0


def test_standardized_60hz_sampling_uses_monotonic_source_frames_and_actual_dt() -> None:
    fps = 59.987
    timestamps = np.arange(13, dtype=np.float64) / fps
    clip = _motion_clip(
        fps=fps,
        source_frames=list(range(100, 113)),
        timestamps_seconds=timestamps,
        acceleration=3.0,
    )

    sampled = _sampled_rows(_standardized_check().run(clip))

    assert [row.frame_idx for row in sampled] == [100, 102, 104, 106, 108, 110, 112]
    assert sampled[-1].metrics["evidence_source_frames"] == [108, 110, 112]
    assert sampled[-1].metrics["standardized_sample_index"] == 6
    assert math.isclose(
        sampled[-1].metrics["actual_dt_seconds"],
        timestamps[12] - timestamps[10],
        rel_tol=1e-8,
    )
    assert math.isclose(
        sampled[-1].metrics["joint_acceleration_standardized_m_s2_max"],
        3.0,
        rel_tol=1e-5,
    )
    assert math.isclose(
        sampled[-1].metrics["joint_acceleration_m_s2_max"],
        sampled[-1].metrics["joint_acceleration_standardized_m_s2_max"],
        rel_tol=1e-5,
    )


def test_standardized_metrics_use_irregular_timestamps_not_fixed_target_period() -> None:
    timestamps = np.asarray([0.0, 0.030, 0.068, 0.099, 0.135], dtype=np.float64)
    clip = _motion_clip(
        fps=30.0,
        source_frames=[20, 21, 22, 23, 24],
        timestamps_seconds=timestamps,
        velocity=2.5,
    )

    sampled = _sampled_rows(_standardized_check().run(clip))
    pair_rows = [
        row
        for row in sampled
        if row.metrics.get("standardized_temporal_pair_eligible") is True
    ]

    assert pair_rows
    assert any(
        not math.isclose(row.metrics["actual_dt_seconds"], 1.0 / 30.0)
        for row in pair_rows
    )
    assert all(
        math.isclose(
            row.metrics["joint_velocity_standardized_m_s_max"],
            2.5,
            rel_tol=1e-6,
        )
        for row in pair_rows
    )


def test_standardized_30hz_reduces_alternating_60hz_sample_noise() -> None:
    fps = 60.0
    timestamps = np.arange(13, dtype=np.float64) / fps
    clip = _motion_clip(
        fps=fps,
        source_frames=list(range(13)),
        timestamps_seconds=timestamps,
        velocity=1.0,
    )
    clip.keypoints["probe_joint"][:, 0] += np.where(
        np.arange(13) % 2 == 0,
        0.0005,
        -0.0005,
    )

    rows = _standardized_check().run(clip)
    native = [
        row.metrics["joint_acceleration_m_s2_max"]
        for row in rows
        if "joint_acceleration_m_s2_max" in row.metrics
    ]
    standardized = [
        row.metrics["joint_acceleration_standardized_m_s2_max"]
        for row in rows
        if "joint_acceleration_standardized_m_s2_max" in row.metrics
    ]

    assert native
    assert standardized
    assert max(standardized) < max(native)


def test_duplicate_and_non_monotonic_timestamps_break_standardized_lineage() -> None:
    timestamps = np.asarray(
        [0.0, 1 / 60, 1 / 30, 1 / 30, 0.03, 0.08, 0.10],
        dtype=np.float64,
    )
    clip = _motion_clip(
        fps=60.0,
        source_frames=list(range(7)),
        timestamps_seconds=timestamps,
    )

    rows = _standardized_check().run(clip)
    audit = rows[0].metrics["temporal_sampling_audit"]
    sampled = _sampled_rows(rows)

    assert audit["duplicate_timestamp_count"] == 1
    assert audit["non_monotonic_timestamp_count"] == 1
    assert any(
        row.metrics.get("standardized_temporal_pair_eligible") is False
        and row.metrics.get("standardized_temporal_pair_skip_reason")
        in {"duplicate_timestamp", "non_monotonic_timestamp", "segment_start"}
        for row in sampled
    )
    assert all(
        row.metrics.get("evidence_source_frames") != [2, 4]
        for row in sampled
    )


def test_large_timestamp_gap_starts_a_new_ineligible_segment() -> None:
    timestamps = np.asarray(
        [0.0, 1 / 60, 2 / 60, 0.5, 0.5 + 1 / 60, 0.5 + 2 / 60],
        dtype=np.float64,
    )
    clip = _motion_clip(
        fps=60.0,
        source_frames=list(range(6)),
        timestamps_seconds=timestamps,
    )

    rows = _standardized_check().run(clip)
    sampled = _sampled_rows(rows)
    audit = rows[0].metrics["temporal_sampling_audit"]
    first_after_gap = next(row for row in sampled if row.frame_idx >= 3)

    assert audit["temporal_gap_break_count"] == 1
    assert first_after_gap.metrics["standardized_temporal_pair_eligible"] is False
    assert first_after_gap.metrics["standardized_temporal_pair_skip_reason"] == (
        "timestamp_gap"
    )


def test_unusable_frame_breaks_standardized_pair_without_bridge() -> None:
    clip = _motion_clip(
        fps=60.0,
        source_frames=list(range(7)),
        acceleration=1.0,
    )
    clip.keypoints["probe_joint"][2] = np.nan

    sampled = _sampled_rows(_standardized_check().run(clip))
    row_at_four = next(row for row in sampled if row.frame_idx == 4)

    assert row_at_four.metrics["standardized_temporal_pair_eligible"] is False
    assert row_at_four.metrics["standardized_temporal_pair_skip_reason"] == "previous_frame_unusable"
    assert "joint_displacement_standardized_m_max" not in row_at_four.metrics
    assert row_at_four.metrics["evidence_source_frames"] == [4]


def test_unselected_unusable_native_frame_still_breaks_standardized_pair() -> None:
    clip = _motion_clip(
        fps=60.0,
        source_frames=list(range(5)),
        acceleration=1.0,
    )
    clip.keypoints["probe_joint"][1] = np.inf

    sampled = _sampled_rows(_standardized_check().run(clip))
    row_at_two = next(row for row in sampled if row.frame_idx == 2)

    assert row_at_two.metrics["standardized_temporal_pair_eligible"] is False
    assert row_at_two.metrics["standardized_temporal_pair_skip_reason"] == (
        "intermediate_frame_unusable"
    )
    assert row_at_two.metrics["evidence_source_frames"] == [2]
    assert "joint_displacement_standardized_m_max" not in row_at_two.metrics


def test_all_zero_frame_breaks_standardized_pair_without_bridge() -> None:
    clip = _motion_clip(
        fps=60.0,
        source_frames=list(range(5)),
        acceleration=1.0,
    )
    clip.keypoints["probe_joint"][1] = 0.0

    sampled = _sampled_rows(_standardized_check().run(clip))
    row_at_two = next(row for row in sampled if row.frame_idx == 2)

    assert row_at_two.metrics["standardized_temporal_pair_eligible"] is False
    assert row_at_two.metrics["standardized_temporal_pair_skip_reason"] == (
        "intermediate_frame_unusable"
    )


def test_missing_timestamp_and_fps_stays_uncalibrated() -> None:
    clip = _motion_clip(
        fps=None,
        source_frames=[0, 1, 2],
    )

    rows = _standardized_check().run(clip)

    assert not _sampled_rows(rows)
    assert rows[0].metrics["temporal_sampling_audit"]["timestamp_source"] == "unavailable"
    assert rows[0].metrics["standardized_temporal_pair_eligible"] is False
    assert all("joint_velocity_m_s_max" not in row.metrics for row in rows)
    assert all("joint_acceleration_m_s2_max" not in row.metrics for row in rows)


def test_skeleton_decision_uses_standardized_metrics_not_high_native_metrics() -> None:
    check = SkeletonQualityScoreCheck(
        {
            "joint_names": ["probe_joint"],
            "projection_enabled": False,
            "reject_missing_keypoints": False,
            "decision_mode": "any_threshold",
        }
    )
    temporal_row = CheckResult(
        check="keypoint_temporal",
        episode_idx=1,
        frame_idx=10,
        metrics={
            "temporal_pair_eligible": True,
            "standardized_temporal_pair_eligible": True,
            "joint_acceleration_m_s2_max": 1_000.0,
            "joint_displacement_m_max": 100.0,
            "joint_angle_change_deg_max": 100.0,
            "rotation_delta_max": 100.0,
            "joint_acceleration_standardized_m_s2_max": 1.0,
            "joint_displacement_standardized_m_max": 0.001,
            "joint_angle_change_standardized_deg_max": 1.0,
            "rotation_delta_standardized_max": 0.01,
            "standardized_temporal_pair_start_frame": 8,
            "standardized_temporal_pair_end_frame": 10,
            "standardized_temporal_transition_attribution": "target_frame",
        },
        flag=None,
        reason="raw and standardized temporal metrics",
        severity="uncalibrated",
    )
    check.temporal_check = SimpleNamespace(run=lambda _clip: [temporal_row])
    clip = ClipInputs(
        episode_idx=1,
        frame_indices=[10],
        keypoints={"probe_joint": np.asarray([[0.0, 1.0, 1.0]])},
        fps=60.0,
    )

    row = check.run(clip)[0]

    assert row.metrics["decision_metric_source"] == "standardized_30hz"
    assert row.metrics["skeleton_verdict"] == "good"
    assert row.metrics["which_thresholds_exceeded"] == []
    assert row.metrics["joint_acceleration_m_s2_max"] == 1_000.0
    assert row.metrics["joint_acceleration_standardized_m_s2_max"] == 1.0
    assert row.metrics["temporal_pair_start_frame"] == 8
    assert row.metrics["temporal_pair_end_frame"] == 10


def test_candidate_windows_keep_source_coordinates_for_standardized_samples() -> None:
    clip = _motion_clip(
        fps=60.0,
        source_frames=list(range(100, 114)),
        velocity=3.0,
    )
    clip.asset_id = "jdt__episode_000001"
    check = SkeletonQualityScoreCheck(
        {
            "joint_names": ["probe_joint"],
            "project_2d": False,
            "projection_enabled": False,
            "reject_missing_keypoints": False,
            "candidate_min_seed_run_frames": 3,
            "candidate_gap_close_frames": 2,
            "candidate_pre_context_frames": 0,
            "candidate_post_context_frames": 0,
        }
    )

    check.run(clip)

    assert len(check.candidate_windows) == 1
    window = check.candidate_windows[0]
    assert window["seed_run_start"] >= 102
    assert window["seed_run_end"] <= 112
    assert window["peak_frame"] in {102, 104, 106, 108, 110, 112}
    assert window["trigger_metrics"]["decision_metric_source"] == (
        "standardized_30hz"
    )
    assert "joint_displacement_standardized_m_max" in window["trigger_metrics"]
