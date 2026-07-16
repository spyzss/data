from __future__ import annotations

import numpy as np

from qc_common.keypoint_validity import inspect_hand_keypoints
from qc_common.keypoints import ACCEPTANCE_FINGER_CHAINS


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
        for point_index, base_name in enumerate(chain):
            points[f"{side}{base_name}"] = np.asarray(
                [[finger_x[finger], 0.030 + 0.020 * point_index, 0.0]],
                dtype=np.float64,
            )
    return points


def test_validity_accepts_normal_hand_and_local_duplicate() -> None:
    keypoints = _normal_hand("left")
    keypoints["leftIndexFingerTip"][0] = keypoints[
        "leftIndexFingerIntermediateTip"
    ][0]

    result = inspect_hand_keypoints(keypoints, side="left", frame_offset=0)

    assert result.is_valid is True
    assert result.valid_point_count == 21
    assert result.finite_point_count == 21
    assert result.invalid_reasons == ()
    assert result.all_zero is False
    assert result.all_identical is False


def test_validity_rejects_all_zero_and_all_identical_sentinels() -> None:
    all_zero = _normal_hand("left")
    for values in all_zero.values():
        values[0] = 0.0
    zero_result = inspect_hand_keypoints(all_zero, side="left", frame_offset=0)

    assert zero_result.is_valid is False
    assert zero_result.valid_point_count == 0
    assert zero_result.finite_point_count == 21
    assert zero_result.all_zero is True
    assert zero_result.all_identical is True
    assert zero_result.invalid_reasons == ("all_zero_keypoints",)

    identical = _normal_hand("left")
    for values in identical.values():
        values[0] = np.asarray([1.0, 2.0, 3.0])
    identical_result = inspect_hand_keypoints(
        identical,
        side="left",
        frame_offset=0,
    )

    assert identical_result.is_valid is False
    assert identical_result.valid_point_count == 0
    assert identical_result.finite_point_count == 21
    assert identical_result.all_zero is False
    assert identical_result.all_identical is True
    assert identical_result.invalid_reasons == ("all_identical_keypoints",)


def test_validity_rejects_nonfinite_missing_and_short_coordinates() -> None:
    for mutation, expected_reason in (
        ("nan", "nonfinite_keypoints"),
        ("inf", "nonfinite_keypoints"),
        ("missing", "missing_keypoints"),
        ("short", "invalid_coordinate_shape"),
    ):
        keypoints = _normal_hand("left")
        if mutation == "nan":
            keypoints["leftThumbTip"][0, 0] = np.nan
        elif mutation == "inf":
            keypoints["leftThumbTip"][0, 0] = np.inf
        elif mutation == "missing":
            del keypoints["leftThumbTip"]
        else:
            keypoints["leftThumbTip"] = np.asarray([[1.0, 2.0]])

        result = inspect_hand_keypoints(keypoints, side="left", frame_offset=0)

        assert result.is_valid is False
        assert result.valid_point_count == 20
        assert expected_reason in result.invalid_reasons
        assert "insufficient_valid_keypoint_count" in result.invalid_reasons
