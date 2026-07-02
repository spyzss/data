from pathlib import Path
import json

import numpy as np

from annotation_verify.config import AnnotationVerifyConfig
from annotation_verify.runner import AnnotationVerifyRunner
from precheck.config import PrecheckConfig, SkeletonQualityScoreConfig
from precheck.registry import available_checks
from precheck.runner import PrecheckRunner
from qc_common.keypoints import (
    ACCEPTANCE_FINGER_CHAINS,
    derive_finger_bones,
    select_hand_joints,
)
from qc_common.types import CheckResult, ClipInputs


def _synthetic_keypoints(num_frames: int = 6) -> dict[str, np.ndarray]:
    keypoints: dict[str, np.ndarray] = {}
    for side_index, side in enumerate(("left", "right")):
        keypoints[f"{side}Hand"] = np.asarray(
            [
                np.array([0.0, 0.08 * side_index, 1.0])
                + np.array([0.0005 * frame_idx, 0.0, 0.0])
                for frame_idx in range(num_frames)
            ]
        )
        for finger_index, finger in enumerate(("Thumb", "Index", "Middle", "Ring", "Little")):
            base = np.array([0.03 * finger_index, 0.08 * side_index, 1.0])
            for part_index, base_name in enumerate(ACCEPTANCE_FINGER_CHAINS[finger]):
                trajectory = []
                for frame_idx in range(num_frames):
                    jitter = np.array([0.001 * frame_idx, 0.0, 0.0])
                    bend = 0.0015 * frame_idx * max(part_index - 1, 0) ** 2
                    trajectory.append(
                        base
                        + np.array([bend, 0.01 * part_index, 0.0])
                        + jitter
                    )
                keypoints[f"{side}{base_name}"] = np.asarray(trajectory)
    return keypoints


def _rotation_z(angle_rad: float) -> np.ndarray:
    cos_value = np.cos(angle_rad)
    sin_value = np.sin(angle_rad)
    return np.asarray(
        [
            [cos_value, -sin_value, 0.0],
            [sin_value, cos_value, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _synthetic_rotations(
    keypoints: dict[str, np.ndarray],
    num_frames: int,
) -> dict[str, np.ndarray]:
    rotations: dict[str, np.ndarray] = {}
    for joint_offset, joint in enumerate(sorted(keypoints)):
        rotations[joint] = np.asarray(
            [
                _rotation_z(0.01 * frame_idx + 0.0005 * joint_offset)
                for frame_idx in range(num_frames)
            ]
        )
    return rotations


def _static_keypoints(num_frames: int) -> dict[str, np.ndarray]:
    return {
        name: np.repeat(values[:1], num_frames, axis=0)
        for name, values in _synthetic_keypoints(num_frames).items()
    }


def _rotation_jump_clip(
    episode_idx: int,
    quality_hand: np.ndarray | None = None,
) -> ClipInputs:
    num_frames = 4
    keypoints = _static_keypoints(num_frames)
    rotations = _synthetic_rotations(keypoints, num_frames)
    rotations["leftHand"][2] = _rotation_z(1.0)
    rotations["leftHand"][3] = _rotation_z(1.01)
    return ClipInputs(
        episode_idx=episode_idx,
        frame_indices=[20, 21, 22, 23],
        keypoints=keypoints,
        rotations=rotations,
        quality_hand=quality_hand,
        fps=1.0,
    )


def _displacement_jump_clip(episode_idx: int) -> ClipInputs:
    num_frames = 4
    keypoints = _static_keypoints(num_frames)
    translation = np.asarray([0.01, 0.0, 0.0])
    for values in keypoints.values():
        values[2] = values[2] + translation
        values[3] = values[3] + translation
    rotations = _synthetic_rotations(keypoints, num_frames)
    return ClipInputs(
        episode_idx=episode_idx,
        frame_indices=[20, 21, 22, 23],
        keypoints=keypoints,
        rotations=rotations,
        fps=1.0,
    )


def test_qc_runners_smoke(tmp_path: Path) -> None:
    num_frames = 12
    keypoints = _synthetic_keypoints(num_frames)
    rotations = _synthetic_rotations(keypoints, num_frames)
    confidences = {name: np.ones(num_frames) for name in keypoints}
    confidences["rightHand"] = np.zeros(num_frames)
    quality_hand = np.ones((num_frames, 2), dtype=np.float32)
    quality_hand[2:4, 0] = 0.0
    quality_hand[5, 1] = 0.0
    clip = ClipInputs(
        episode_idx=7,
        frame_indices=list(range(10, 10 + num_frames)),
        frames=[np.full((16, 16, 3), fill_value=12, dtype=np.uint8) for _ in range(num_frames)],
        keypoints=keypoints,
        rotations=rotations,
        confidences=confidences,
        quality_hand=quality_hand,
        masks=None,
        instruction="Pick up the green bottle.",
        intrinsics=np.asarray([[1000.0, 0.0, 8.0], [0.0, 1000.0, 8.0], [0.0, 0.0, 1.0]]),
        fps=1.0,
    )

    assert "keypoint_temporal" in available_checks()
    selected_joints = select_hand_joints(sorted(keypoints))
    assert "leftHand" in selected_joints
    assert "leftThumbKnuckle" in selected_joints
    assert "leftThumbTip" in selected_joints
    assert len(selected_joints) == 42
    assert all("Metacarpal" not in joint for joint in selected_joints)
    assert ("leftHand", "leftIndexFingerKnuckle") in derive_finger_bones(selected_joints)
    assert ("leftHand", "leftThumbKnuckle") in derive_finger_bones(selected_joints)

    precheck_config = PrecheckConfig(
        output_dir=tmp_path / "precheck",
        enabled_checks=[
            "overexposure",
            "keypoint_temporal",
            "keypoint_missing",
            "mask_containment",
        ],
        overwrite=True,
    )
    precheck_results = PrecheckRunner(precheck_config).run([clip])
    assert precheck_results
    assert all(isinstance(result, CheckResult) for result in precheck_results)
    assert "mask_containment" not in {result.check for result in precheck_results}
    temporal_metrics = [
        result.metrics
        for result in precheck_results
        if result.check == "keypoint_temporal" and result.frame_idx > 10
    ]
    assert any("rotation_delta_p95" in metrics for metrics in temporal_metrics)
    assert any("joint_angle_change_deg_p95" in metrics for metrics in temporal_metrics)
    assert any(metrics.get("joint_count") == 42.0 for metrics in temporal_metrics)
    assert any(metrics.get("confidence_zero_count") == 1.0 for metrics in temporal_metrics)
    missing_results = [
        result for result in precheck_results if result.check == "keypoint_missing"
    ]
    assert missing_results
    assert any(result.flag is True for result in missing_results)
    assert any(
        result.metrics["missing_frames_in_10s_window_left"] > 1.0
        for result in missing_results
    )
    assert any(
        result.metrics["acceptance_joint_count"] == 42.0
        for result in missing_results
    )
    assert (
        tmp_path / "precheck" / "keypoint_missing_repair_candidates.json"
    ).exists()
    assert (tmp_path / "precheck" / "check_results.parquet").exists() or (
        tmp_path / "precheck" / "check_results.csv"
    ).exists()
    assert (tmp_path / "precheck" / "check_results.json").exists()
    assert (tmp_path / "precheck" / "clip_aggregates.json").exists()
    result_records = json.loads(
        (tmp_path / "precheck" / "check_results.json").read_text()
    )
    assert result_records
    assert isinstance(result_records[0]["metrics"], dict)

    verify_config = AnnotationVerifyConfig(output_dir=tmp_path / "verify")
    verify_results = AnnotationVerifyRunner(verify_config).run([clip])
    assert verify_results
    assert all(result.check == "instruction_consistency" for result in verify_results)
    assert all(result.flag is None for result in verify_results)


def test_quality_score_check(tmp_path: Path) -> None:
    # frame 0: [0.0, 1.0]   -> zero on left hand      -> score 0.0
    # frame 1: [0.5, 1.0]   -> nonzero on both hands  -> score 1.0
    # frame 2: [0.5, 0.5]   -> nonzero on both hands  -> score 1.0
    # frame 3: [1.0, 1.0]   -> nonzero on both hands  -> score 1.0
    # frame 4: [0.0, 0.5]   -> zero on left hand      -> score 0.0
    quality_hand = np.asarray(
        [
            [0.0, 1.0],
            [0.5, 1.0],
            [0.5, 0.5],
            [1.0, 1.0],
            [0.0, 0.5],
        ],
        dtype=np.float32,
    )
    clip = ClipInputs(
        episode_idx=42,
        frame_indices=[0, 1, 2, 3, 4],
        quality_hand=quality_hand,
    )

    precheck_config = PrecheckConfig(
        output_dir=tmp_path / "quality_score",
        enabled_checks=["quality_score"],
        overwrite=True,
    )
    results = PrecheckRunner(precheck_config).run([clip])

    frame_rows = [result for result in results if result.frame_idx != -1]
    summary_rows = [result for result in results if result.frame_idx == -1]
    assert len(frame_rows) == 5
    assert len(summary_rows) == 1
    assert all(result.flag is None for result in frame_rows)

    frame_scores = {result.frame_idx: result.metrics["frame_score"] for result in frame_rows}
    assert frame_scores[0] == 0.0
    assert frame_scores[1] == 1.0
    assert frame_scores[2] == 1.0
    assert frame_scores[3] == 1.0
    assert frame_scores[4] == 0.0

    total_score = 0.0 + 1.0 + 1.0 + 1.0 + 0.0
    expected_pass_ratio = total_score / 5
    summary = summary_rows[0]
    assert summary.metrics["total_score"] == total_score
    assert summary.metrics["num_frames"] == 5.0
    assert summary.metrics["pass_ratio"] == expected_pass_ratio
    assert summary.metrics["pass_threshold"] == 0.90
    assert summary.flag is False
    assert expected_pass_ratio < 0.90


def test_text_integrity_check(tmp_path: Path) -> None:
    clips = [
        ClipInputs(
            episode_idx=50,
            text_label={
                "scene": "kitchen",
                "task": "pick",
                "text_en": "Pick up the green bottle.",
            },
        ),
        ClipInputs(
            episode_idx=51,
            text_label={
                "scene": "kitchen",
                "task": "pick",
                "text_en": "  ",
            },
        ),
        ClipInputs(
            episode_idx=52,
            text_label={
                "scene": "kitchen",
                "text_en": "Pick up the green bottle.",
            },
        ),
        ClipInputs(episode_idx=53),
        ClipInputs(episode_idx=54, text_label_raw="{not valid json"),
    ]

    assert "text_integrity" in available_checks()
    precheck_config = PrecheckConfig(
        output_dir=tmp_path / "text_integrity",
        enabled_checks=["text_integrity"],
        overwrite=True,
    )
    results = PrecheckRunner(precheck_config).run(clips)

    rows = {result.episode_idx: result for result in results}
    assert len(rows) == 5
    assert all(result.frame_idx == -1 for result in rows.values())

    assert rows[50].flag is None
    assert rows[50].metrics["missing_field_count"] == 0.0
    assert rows[50].metrics["empty_field_count"] == 0.0
    assert rows[50].metrics["field_present_text_en"] == 1.0
    assert rows[50].metrics["field_nonempty_text_en"] == 1.0

    assert rows[51].flag is True
    assert rows[51].metrics["missing_field_count"] == 0.0
    assert rows[51].metrics["empty_field_count"] == 1.0
    assert rows[51].metrics["field_present_text_en"] == 1.0
    assert rows[51].metrics["field_nonempty_text_en"] == 0.0
    assert "text_en" in rows[51].reason

    assert rows[52].flag is True
    assert rows[52].metrics["missing_field_count"] == 1.0
    assert rows[52].metrics["empty_field_count"] == 0.0
    assert rows[52].metrics["field_present_task"] == 0.0
    assert "task" in rows[52].reason

    assert rows[53].flag is True
    assert rows[53].metrics["missing_field_count"] == 3.0
    assert rows[53].metrics["empty_field_count"] == 0.0
    assert rows[53].reason == "no text_label"

    assert rows[54].flag is True
    assert rows[54].metrics["missing_field_count"] == 3.0
    assert rows[54].metrics["empty_field_count"] == 0.0
    assert rows[54].reason == "text_label not valid JSON"


def test_skeleton_quality_score_without_quality_hand(tmp_path: Path) -> None:
    clip = _rotation_jump_clip(episode_idx=99)

    assert "skeleton_quality_score" in available_checks()
    precheck_config = PrecheckConfig(
        output_dir=tmp_path / "skeleton_quality_score",
        enabled_checks=["skeleton_quality_score", "composite_frame_verdict"],
        overwrite=True,
    )
    results = PrecheckRunner(precheck_config).run([clip])

    skeleton_rows = {
        result.frame_idx: result
        for result in results
        if result.check == "skeleton_quality_score" and result.frame_idx != -1
    }
    composite_rows = {
        result.frame_idx: result
        for result in results
        if result.check == "composite_frame_verdict" and result.frame_idx != -1
    }
    skeleton_summary = [
        result
        for result in results
        if result.check == "skeleton_quality_score" and result.frame_idx == -1
    ]
    composite_summary = [
        result
        for result in results
        if result.check == "composite_frame_verdict" and result.frame_idx == -1
    ]
    assert set(skeleton_rows) == {20, 21, 22, 23}
    assert set(composite_rows) == {20, 21, 22, 23}
    assert len(skeleton_summary) == 1
    assert len(composite_summary) == 1

    assert skeleton_rows[21].metrics["skeleton_score"] == 1.0
    assert skeleton_rows[21].flag is None
    assert skeleton_rows[22].metrics["skeleton_score"] < 1.0
    assert skeleton_rows[22].flag is True
    assert skeleton_rows[22].metrics["which_thresholds_exceeded"] == [
        "rotation_delta_max"
    ]
    assert skeleton_rows[22].metrics["rotation_delta_max"] > 0.45

    assert all(
        row.metrics["vendor_quality_weight"] == 1.0
        for row in composite_rows.values()
    )
    assert all(
        row.metrics["supplier_label_available"] == 0.0
        for row in composite_rows.values()
    )
    assert all(row.flag is None for row in composite_rows.values())

    summary = skeleton_summary[0]
    assert summary.metrics["count_good"] == 3.0
    assert summary.metrics["count_suspect"] == 1.0
    assert summary.metrics["num_frames"] == 4.0
    assert summary.metrics["pass_ratio"] == 3.0 / 4.0
    assert summary.flag is False


def test_skeleton_quality_score_flags_displacement_metric(tmp_path: Path) -> None:
    clip = _displacement_jump_clip(episode_idx=101)
    precheck_config = PrecheckConfig(
        output_dir=tmp_path / "skeleton_displacement_score",
        enabled_checks=["skeleton_quality_score"],
        skeleton_quality_score=SkeletonQualityScoreConfig(
            joint_displacement_m_max_threshold=0.005
        ),
        overwrite=True,
    )
    results = PrecheckRunner(precheck_config).run([clip])

    frame_rows = {
        result.frame_idx: result
        for result in results
        if result.check == "skeleton_quality_score" and result.frame_idx != -1
    }
    assert frame_rows[21].flag is None
    assert frame_rows[22].flag is True
    assert frame_rows[22].metrics["which_thresholds_exceeded"] == [
        "joint_displacement_m_max"
    ]
    assert frame_rows[22].metrics["joint_displacement_m_max"] > 0.005
    assert frame_rows[22].metrics["joint_displacement_m_penalty"] == 0.25
    assert frame_rows[22].metrics["skeleton_score"] == 0.75


def test_composite_frame_verdict_audits_supplier_labels(tmp_path: Path) -> None:
    quality_hand = np.asarray(
        [
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=np.float32,
    )
    clip = _rotation_jump_clip(episode_idx=100, quality_hand=quality_hand)

    assert "composite_frame_verdict" in available_checks()
    precheck_config = PrecheckConfig(
        output_dir=tmp_path / "composite_frame_verdict",
        enabled_checks=["composite_frame_verdict"],
        overwrite=True,
    )
    results = PrecheckRunner(precheck_config).run([clip])

    frame_rows = {result.frame_idx: result for result in results if result.frame_idx != -1}
    summary_rows = [result for result in results if result.frame_idx == -1]
    assert set(frame_rows) == {20, 21, 22, 23}
    assert len(summary_rows) == 1

    flags = {frame_idx: row.flag for frame_idx, row in frame_rows.items()}
    assert flags[20] is None
    assert flags[21] is None
    assert flags[22] is True
    assert flags[23] is None
    assert frame_rows[22].metrics["which_thresholds_exceeded"] == [
        "rotation_delta_max"
    ]
    assert frame_rows[22].metrics["rotation_delta_max"] > 0.45
    assert frame_rows[21].metrics["vendor_quality_weight"] == 1.0
    assert frame_rows[21].metrics["skeleton_score"] == 1.0
    assert frame_rows[22].metrics["vendor_quality_weight"] == 1.0
    assert frame_rows[22].metrics["skeleton_score"] < 1.0
    assert frame_rows[22].metrics["audit_suspect"] == 1.0
    assert frame_rows[20].metrics["vendor_quality_weight"] == 0.0
    assert frame_rows[23].metrics["vendor_quality_weight"] == 0.5

    summary = summary_rows[0]
    assert summary.metrics["skeleton_good_count"] == 3.0
    assert summary.metrics["skeleton_suspect_count"] == 1.0
    assert summary.metrics["supplier_no_downweight_count"] == 2.0
    assert summary.metrics["supplier_downweighted_count"] == 2.0
    assert summary.metrics["audit_suspect_count"] == 1.0
    assert summary.metrics["num_frames"] == 4.0
    assert summary.metrics["pass_ratio"] == 3.0 / 4.0
    assert summary.metrics["pass_threshold"] == 0.90
    assert summary.flag is False
