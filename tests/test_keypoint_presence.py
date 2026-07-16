from __future__ import annotations

import json

import numpy as np

from precheck import checks as _checks  # noqa: F401
from precheck.config import KeypointMissingConfig
from precheck.registry import create_check
from qc_common.keypoints import ACCEPTANCE_FINGER_CHAINS
from qc_common.types import ClipInputs


def _normal_hand(side: str, frame_count: int = 1) -> dict[str, np.ndarray]:
    points: dict[str, np.ndarray] = {
        f"{side}Hand": np.zeros((frame_count, 3), dtype=np.float64)
    }
    finger_x = {
        "Thumb": -0.030,
        "Index": -0.015,
        "Middle": 0.0,
        "Ring": 0.015,
        "Little": 0.030,
    }
    for finger, chain in ACCEPTANCE_FINGER_CHAINS.items():
        for point_index, base_name in enumerate(chain):
            point = np.asarray(
                [finger_x[finger], 0.030 + 0.020 * point_index, 0.0]
            )
            points[f"{side}{base_name}"] = np.repeat(
                point[None, :],
                frame_count,
                axis=0,
            )
    return points


def _run(clip: ClipInputs):
    return create_check(
        "keypoint_missing",
        KeypointMissingConfig().__dict__,
    ).run(clip)


def test_presence_runs_without_quality_hand_and_passes_normal_hands() -> None:
    clip = ClipInputs(
        episode_idx=0,
        frame_indices=[7],
        keypoints={**_normal_hand("left"), **_normal_hand("right")},
        quality_hand=None,
    )

    result = _run(clip)[0]

    assert result.frame_idx == 7
    assert result.flag is False
    assert result.metrics["keypoint_presence_verdict"] == "pass"
    assert result.metrics["valid_keypoint_count_left"] == 21
    assert result.metrics["valid_keypoint_count_right"] == 21
    assert result.metrics["keypoint_existence_invalid_left"] is False
    assert result.metrics["keypoint_existence_invalid_right"] is False


def test_presence_rejects_one_zero_hand_without_failing_normal_hand() -> None:
    keypoints = {**_normal_hand("left"), **_normal_hand("right")}
    for name, values in keypoints.items():
        if name.startswith("left"):
            values[0] = 0.0
    clip = ClipInputs(0, frame_indices=[9], keypoints=keypoints, quality_hand=None)

    result = _run(clip)[0]

    assert result.flag is True
    assert result.metrics["keypoint_presence_verdict"] == "fail"
    assert result.metrics["keypoint_existence_invalid_left"] is True
    assert result.metrics["keypoint_existence_invalid_right"] is False
    assert result.metrics["valid_keypoint_count_left"] == 0
    assert result.metrics["finite_keypoint_count_left"] == 21
    assert result.metrics["all_zero_left"] is True
    assert result.metrics["all_identical_left"] is True
    assert result.metrics["invalid_reasons_left"] == ["all_zero_keypoints"]
    assert json.loads(result.reason)["invalid_hands"] == ["left"]


def test_presence_rejects_both_zero_hands() -> None:
    keypoints = {**_normal_hand("left"), **_normal_hand("right")}
    for values in keypoints.values():
        values[0] = 0.0

    result = _run(
        ClipInputs(0, frame_indices=[4], keypoints=keypoints, quality_hand=None)
    )[0]

    assert result.flag is True
    assert result.metrics["keypoint_existence_invalid_left"] is True
    assert result.metrics["keypoint_existence_invalid_right"] is True
    assert json.loads(result.reason)["invalid_hands"] == ["left", "right"]


def test_presence_merges_invalid_frames_in_rolling_metrics() -> None:
    keypoints = {**_normal_hand("left", 5), **_normal_hand("right", 5)}
    for frame in (1, 2, 4):
        for name, values in keypoints.items():
            if name.startswith("left"):
                values[frame] = 0.0
    clip = ClipInputs(
        0,
        frame_indices=[100, 101, 102, 103, 104],
        keypoints=keypoints,
        quality_hand=None,
        fps=1.0,
    )

    results = _run(clip)

    invalid = [result.frame_idx for result in results if result.flag is True]
    assert invalid == [101, 102, 104]
    assert results[-1].metrics["missing_frames_in_10s_window_left"] == 3.0


def test_presence_reports_identical_nonzero_nan_inf_and_missing_joint() -> None:
    mutations = ("identical", "nan", "inf", "missing")
    for mutation in mutations:
        keypoints = {**_normal_hand("left"), **_normal_hand("right")}
        if mutation == "identical":
            for name, values in keypoints.items():
                if name.startswith("left"):
                    values[0] = np.asarray([1.0, 2.0, 3.0])
        elif mutation == "nan":
            keypoints["leftThumbTip"][0, 0] = np.nan
        elif mutation == "inf":
            keypoints["leftThumbTip"][0, 0] = np.inf
        else:
            del keypoints["leftThumbTip"]

        result = _run(
            ClipInputs(0, frame_indices=[0], keypoints=keypoints, quality_hand=None)
        )[0]

        assert result.flag is True
        assert result.metrics["keypoint_existence_invalid_left"] is True
        assert result.metrics["keypoint_existence_invalid_right"] is False


def test_quality_hand_is_informational_for_presence() -> None:
    clip = ClipInputs(
        0,
        frame_indices=[0],
        keypoints={**_normal_hand("left"), **_normal_hand("right")},
        quality_hand=np.asarray([[0.0, 1.0]]),
    )

    result = _run(clip)[0]

    assert result.flag is False
    assert result.metrics["quality_low_left"] == 1.0
    assert result.metrics["supplier_quality_signal"] == "low"
