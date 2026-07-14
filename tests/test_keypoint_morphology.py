from __future__ import annotations

import json
import math

import numpy as np
import pytest

from precheck import checks as _checks  # noqa: F401
from precheck.registry import create_check
from qc_common.keypoints import ACCEPTANCE_FINGER_CHAINS, acceptance_joint_names
from qc_common.types import ClipInputs


DEFAULT_CONFIG = {
    "sides": ["left", "right"],
    "duplicate_joint_distance_m": 0.00001,
    "min_palm_scale_m": 0.0001,
    "max_bone_length_ratio_spread_review": 3.0,
    "max_bone_length_ratio_spread_fail": 8.0,
    "max_normalized_bone_length_review": 3.0,
    "max_normalized_bone_length_fail": 6.0,
    "max_zero_length_bone_count_review": 1,
    "max_zero_length_bone_count_fail": 2,
    "max_duplicate_joint_pair_count_review": 1,
    "max_duplicate_joint_pair_count_fail": 3,
    "min_joint_angle_deg_review": 5.0,
    "min_joint_angle_deg_fail": 1.0,
    "max_joint_angle_violation_fraction_review": 0.15,
    "max_joint_angle_violation_fraction_fail": 0.40,
}


def _normal_hand(side: str) -> dict[str, np.ndarray]:
    points: dict[str, np.ndarray] = {
        f"{side}Hand": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)
    }
    finger_x = {
        "Thumb": -0.030,
        "Index": -0.015,
        "Middle": 0.0,
        "Ring": 0.015,
        "Little": 0.030,
    }
    for finger, chain in ACCEPTANCE_FINGER_CHAINS.items():
        x = finger_x[finger]
        for point_index, base_name in enumerate(chain):
            points[f"{side}{base_name}"] = np.asarray(
                [[x, 0.030 + 0.020 * point_index, 0.0]],
                dtype=np.float64,
            )
    return points


def _clip() -> ClipInputs:
    return ClipInputs(
        episode_idx=7,
        frame_indices=[42],
        keypoints={**_normal_hand("left"), **_normal_hand("right")},
    )


def _run(
    clip: ClipInputs,
    overrides: dict[str, object] | None = None,
) -> tuple[object, object]:
    config = {**DEFAULT_CONFIG, **(overrides or {})}
    results = create_check("keypoint_morphology", config).run(clip)
    frame = next(result for result in results if result.frame_idx == 42)
    summary = next(result for result in results if result.frame_idx == -1)
    return frame, summary


def _set_joint(clip: ClipInputs, joint_name: str, point: np.ndarray) -> None:
    assert clip.keypoints is not None
    clip.keypoints[joint_name][0] = point


def _set_finger_angle(clip: ClipInputs, side: str, finger: str, angle_deg: float) -> None:
    chain = ACCEPTANCE_FINGER_CHAINS[finger]
    knuckle_name = f"{side}{chain[0]}"
    assert clip.keypoints is not None
    knuckle = clip.keypoints[knuckle_name][0].copy()
    root_direction = clip.keypoints[f"{side}Hand"][0] - knuckle
    root_direction /= np.linalg.norm(root_direction)
    theta = math.radians(angle_deg)
    direction = np.asarray(
        [
            root_direction[0] * math.cos(theta)
            - root_direction[1] * math.sin(theta),
            root_direction[0] * math.sin(theta)
            + root_direction[1] * math.cos(theta),
            0.0,
        ]
    )
    for point_index, base_name in enumerate(chain[1:], start=1):
        _set_joint(
            clip,
            f"{side}{base_name}",
            knuckle + direction * (0.020 * point_index),
        )


def test_normal_static_hands_pass_without_temporal_or_quality_inputs() -> None:
    frame, summary = _run(_clip())

    assert frame.metrics["left_valid_keypoint_count"] == 21
    assert frame.metrics["right_valid_keypoint_count"] == 21
    assert frame.metrics["left_morphology_verdict"] == "pass"
    assert frame.metrics["right_morphology_verdict"] == "pass"
    assert frame.metrics["morphology_verdict"] == "pass"
    assert frame.flag is False
    assert summary.metrics["morphology_verdict"] == "pass"
    assert summary.metrics["thresholds"] == DEFAULT_CONFIG | {
        "sides": ["left", "right"]
    }


def test_single_duplicated_joint_is_review() -> None:
    clip = _clip()
    assert clip.keypoints is not None
    _set_joint(
        clip,
        "leftIndexFingerTip",
        clip.keypoints["leftIndexFingerIntermediateTip"][0],
    )

    frame, _summary = _run(clip)

    assert frame.metrics["left_zero_length_bone_count"] == 1
    assert frame.metrics["left_duplicate_joint_pair_count"] >= 1
    assert frame.metrics["left_morphology_verdict"] == "review"
    assert frame.metrics["morphology_verdict"] == "review"
    assert frame.flag is None
    assert "left:zero_length_bone_count_review" in frame.reason


def test_duplicated_and_collapsed_finger_is_fail() -> None:
    clip = _clip()
    assert clip.keypoints is not None
    knuckle = clip.keypoints["leftIndexFingerKnuckle"][0].copy()
    for base_name in ACCEPTANCE_FINGER_CHAINS["Index"][1:]:
        _set_joint(clip, f"left{base_name}", knuckle)

    frame, _summary = _run(clip)

    assert frame.metrics["left_collapsed_finger_count"] == 1
    assert frame.metrics["left_zero_length_bone_count"] == 3
    assert frame.metrics["left_duplicate_joint_pair_count"] >= 3
    assert frame.metrics["left_morphology_verdict"] == "fail"
    assert frame.metrics["morphology_verdict"] == "fail"
    assert frame.flag is True
    assert "left:collapsed_finger_count_fail" in frame.reason


def test_extreme_normalized_bone_length_and_spread_fail() -> None:
    clip = _clip()
    _set_joint(
        clip,
        "leftLittleFingerTip",
        np.asarray([0.030, 0.500, 0.0]),
    )

    frame, _summary = _run(clip)

    assert frame.metrics["left_normalized_bone_length_max"] >= 6.0
    assert frame.metrics["left_bone_length_ratio_spread"] >= 8.0
    assert frame.metrics["left_morphology_verdict"] == "fail"
    assert "left:max_normalized_bone_length_fail" in frame.reason
    assert "left:bone_length_ratio_spread_fail" in frame.reason


def test_max_normalized_reason_token_uses_normalized_bone_length_max_metric() -> None:
    clip = _clip()
    _set_joint(
        clip,
        "leftLittleFingerTip",
        np.asarray([0.030, 0.500, 0.0]),
    )

    frame, _summary = _run(clip)

    assert "left:max_normalized_bone_length_fail" in frame.metrics[
        "which_thresholds_exceeded"
    ]
    assert frame.metrics["left_normalized_bone_length_max"] >= 6.0
    assert "left_max_normalized_bone_length" not in frame.metrics


@pytest.mark.parametrize(
    ("angle_deg", "expected_verdict", "expected_reason"),
    [
        (3.0, "review", "left:joint_angle_min_deg_review"),
        (0.5, "fail", "left:joint_angle_min_deg_fail"),
    ],
)
def test_extreme_joint_angle_escalates(
    angle_deg: float,
    expected_verdict: str,
    expected_reason: str,
) -> None:
    clip = _clip()
    _set_finger_angle(clip, "left", "Index", angle_deg)

    frame, _summary = _run(clip)

    assert frame.metrics["left_joint_angle_min_deg"] == pytest.approx(angle_deg)
    assert frame.metrics["left_morphology_verdict"] == expected_verdict
    assert frame.metrics["morphology_verdict"] == expected_verdict
    assert expected_reason in frame.reason


@pytest.mark.parametrize("invalid_kind", ["nan", "missing"])
def test_existence_invalid_hand_is_not_applicable_not_morphology_fail(
    invalid_kind: str,
) -> None:
    clip = _clip()
    assert clip.keypoints is not None
    if invalid_kind == "nan":
        clip.keypoints["leftThumbTip"][0, 0] = np.nan
    else:
        del clip.keypoints["leftThumbTip"]

    frame, _summary = _run(clip)

    assert frame.metrics["left_valid_keypoint_count"] == 20
    assert frame.metrics["left_morphology_verdict"] == "not_applicable"
    assert frame.metrics["right_morphology_verdict"] == "pass"
    assert frame.metrics["morphology_verdict"] == "pass"
    assert "left:skipped_due_to_existence_invalid" in frame.reason
    assert frame.flag is False


def test_left_fail_and_right_pass_produces_overall_fail() -> None:
    clip = _clip()
    assert clip.keypoints is not None
    knuckle = clip.keypoints["leftRingFingerKnuckle"][0].copy()
    for base_name in ACCEPTANCE_FINGER_CHAINS["Ring"][1:]:
        _set_joint(clip, f"left{base_name}", knuckle)

    frame, _summary = _run(clip)

    assert frame.metrics["left_morphology_verdict"] == "fail"
    assert frame.metrics["right_morphology_verdict"] == "pass"
    assert frame.metrics["morphology_verdict"] == "fail"


def test_left_review_and_right_pass_produces_overall_review() -> None:
    clip = _clip()
    _set_finger_angle(clip, "left", "Middle", 3.0)

    frame, _summary = _run(clip)

    assert frame.metrics["left_morphology_verdict"] == "review"
    assert frame.metrics["right_morphology_verdict"] == "pass"
    assert frame.metrics["morphology_verdict"] == "review"


def test_clip_summary_preserves_distribution_stats_and_threshold_metadata() -> None:
    clip = _clip()
    frame, summary = _run(clip)

    for side in ("left", "right"):
        for metric in (
            "bone_length_ratio_spread",
            "normalized_bone_length_max",
            "zero_length_bone_count",
            "duplicate_joint_pair_count",
            "collapsed_finger_count",
            "joint_angle_violation_fraction",
        ):
            value = frame.metrics[f"{side}_{metric}"]
            assert summary.metrics[f"{side}_{metric}_mean"] == pytest.approx(value)
            assert summary.metrics[f"{side}_{metric}_median"] == pytest.approx(value)
            assert summary.metrics[f"{side}_{metric}_p95"] == pytest.approx(value)
            assert summary.metrics[f"{side}_{metric}_max"] == pytest.approx(value)
    assert summary.metrics["method"] == "static_per_frame_21_point_hand_geometry"
    assert summary.metrics["decision_basis"] == "fixed_config_thresholds"
    assert summary.metrics["calibration_statistics_only"] is True
    assert set(summary.metrics["thresholds"]) == set(DEFAULT_CONFIG)
    assert set(acceptance_joint_names(["left"])) == {
        name for name in clip.keypoints or {} if name.startswith("left")
    }


def test_all_morphology_verdict_metrics_are_json_serializable() -> None:
    normal = _clip()

    review = _clip()
    assert review.keypoints is not None
    _set_joint(
        review,
        "leftIndexFingerTip",
        review.keypoints["leftIndexFingerIntermediateTip"][0],
    )

    fail = _clip()
    assert fail.keypoints is not None
    collapsed = fail.keypoints["leftIndexFingerKnuckle"][0].copy()
    for base_name in ACCEPTANCE_FINGER_CHAINS["Index"][1:]:
        _set_joint(fail, f"left{base_name}", collapsed)

    not_applicable = _clip()
    assert not_applicable.keypoints is not None
    not_applicable.keypoints["leftThumbTip"][0, 0] = np.nan

    verdicts: set[str] = set()
    for clip in (normal, review, fail, not_applicable):
        results = create_check("keypoint_morphology", DEFAULT_CONFIG).run(clip)
        for result in results:
            json.dumps(result.metrics)
            assert isinstance(result.reason, str)
            assert all(
                isinstance(reason, str)
                for reason in result.metrics["which_thresholds_exceeded"]
            )
        frame = next(result for result in results if result.frame_idx == 42)
        verdicts.add(frame.metrics["left_morphology_verdict"])

    assert verdicts == {"pass", "review", "fail", "not_applicable"}
