from __future__ import annotations

import numpy as np

from precheck.checks.keypoint_temporal import KeypointTemporalCheck
from qc_common.keypoints import ACCEPTANCE_FINGER_CHAINS
from qc_common.types import ClipInputs


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
