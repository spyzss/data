from pathlib import Path
import json

import numpy as np

from annotation_verify.config import AnnotationVerifyConfig
from annotation_verify.runner import AnnotationVerifyRunner
from precheck.adapters.supplier_hdf5 import load_supplier_hdf5_clip
from precheck.checks.skeleton_quality_score import SkeletonQualityScoreCheck
from precheck.config import PrecheckConfig, SkeletonQualityScoreConfig
from precheck.registry import available_checks
from precheck.runner import PrecheckRunner
from qc_common.keypoints import (
    ACCEPTANCE_FINGER_CHAINS,
    acceptance_joint_names,
    derive_finger_bones,
    select_hand_joints,
)
from qc_common.projection import (
    ProjectionConfig,
    hand_projection_metrics,
    project_points_with_validity,
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


def _skeleton_candidate_result(
    frame_idx: int,
    exceeded: list[str],
    flag: bool | None = None,
    invalid: bool = False,
    temporal_output_valid: bool | None = True,
) -> CheckResult:
    metrics = {
        "joint_angle_change_deg_max": 11.0 if "joint_angle_change_deg_max" in exceeded else 1.0,
        "rotation_delta_max": 0.5 if "rotation_delta_max" in exceeded else 0.1,
        "joint_acceleration_m_s2_max": 16.0 if "joint_acceleration_m_s2_max" in exceeded else 1.0,
        "joint_displacement_m_max": 0.06 if "joint_displacement_m_max" in exceeded else 0.001,
        "joint_angle_change_deg_ratio": 1.1 if "joint_angle_change_deg_max" in exceeded else 0.1,
        "rotation_delta_ratio": 1.1 if "rotation_delta_max" in exceeded else 0.1,
        "joint_acceleration_m_s2_ratio": 1.1 if "joint_acceleration_m_s2_max" in exceeded else 0.1,
        "joint_displacement_m_ratio": 1.2 if "joint_displacement_m_max" in exceeded else 0.02,
        "which_thresholds_exceeded": exceeded,
        "keypoint_presence_invalid": 1.0 if invalid else 0.0,
        "skeleton_verdict": "invalid" if invalid else "suspect" if flag else "review",
        "temporal_output_valid": temporal_output_valid,
    }
    return CheckResult(
        check="skeleton_quality_score",
        episode_idx=12,
        frame_idx=frame_idx,
        metrics=metrics,
        flag=flag,
        reason="synthetic skeleton candidate row",
    )


def _candidate_check(**overrides: object) -> SkeletonQualityScoreCheck:
    config = {
        "candidate_gap_close_frames": 2,
        "candidate_min_seed_run_frames": 3,
        "candidate_pre_context_frames": 1,
        "candidate_post_context_frames": 1,
    }
    config.update(overrides)
    return SkeletonQualityScoreCheck(config)


def _palm_orientation_keypoints(num_frames: int, side_view: bool) -> dict[str, np.ndarray]:
    keypoints = _static_keypoints(num_frames)
    if side_view:
        wrist = np.asarray([0.0, 0.0, 1.0])
        index = np.asarray([0.0, 1.0, 1.0])
        little = np.asarray([0.0, 0.0, 2.0])
    else:
        wrist = np.asarray([0.0, 0.0, 1.0])
        index = np.asarray([1.0, 0.0, 1.0])
        little = np.asarray([0.0, 1.0, 1.0])
    for side in ("left", "right"):
        keypoints[f"{side}Hand"] = np.repeat(wrist[None, :], num_frames, axis=0)
        keypoints[f"{side}IndexFingerKnuckle"] = np.repeat(
            index[None, :],
            num_frames,
            axis=0,
        )
        keypoints[f"{side}LittleFingerKnuckle"] = np.repeat(
            little[None, :],
            num_frames,
            axis=0,
        )
    return keypoints


def _write_minimal_supplier_hdf5(path: Path, quality_hand: np.ndarray) -> None:
    import h5py

    num_frames = int(quality_hand.shape[0])
    transforms = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], num_frames, axis=0)
    transforms[:, 2, 3] = 1.0
    with h5py.File(path, "w") as handle:
        transform_group = handle.create_group("transforms")
        transform_group.create_dataset("leftHand", data=transforms)
        transform_group.create_dataset("rightHand", data=transforms)
        label_group = handle.create_group("label")
        label_group.create_dataset("quality_hand", data=quality_hand.astype(np.float32))
        handle.create_group("camera").create_dataset(
            "intrinsic",
            data=np.asarray(
                [[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
        )


def test_supplier_hdf5_loader_without_transforms(tmp_path: Path) -> None:
    import h5py

    path = tmp_path / "no_transforms.h5"
    quality_hand = np.asarray([[1.0, 1.0], [0.5, 1.0], [0.0, 1.0]], dtype=np.float32)
    with h5py.File(path, "w") as handle:
        label_group = handle.create_group("label")
        label_group.create_dataset("quality_hand", data=quality_hand)
        label_group.create_dataset(
            "text_label",
            data=json.dumps(
                {"scene": "bedroom", "task": "fold clothes", "text_en": "Fold clothes."}
            ).encode("utf-8"),
        )

    clip = load_supplier_hdf5_clip(path, episode_idx=12, fps=29.97)

    assert clip.episode_idx == 12
    assert clip.num_frames == 3
    assert clip.frame_indices == [0, 1, 2]
    assert clip.keypoints == {}
    assert clip.rotations == {}
    assert np.array_equal(clip.quality_hand, quality_hand)
    assert clip.text_label == {
        "scene": "bedroom",
        "task": "fold clothes",
        "text_en": "Fold clothes.",
    }


def test_supplier_hdf5_loader_mano_joints3d_schema(tmp_path: Path) -> None:
    import h5py

    path = tmp_path / "mano_schema.h5"
    num_frames = 4
    left_joints = np.zeros((num_frames, 21, 3), dtype=np.float32)
    right_joints = np.ones((num_frames, 21, 3), dtype=np.float32)
    for frame_idx in range(num_frames):
        left_joints[frame_idx, :, 0] = frame_idx
        right_joints[frame_idx, :, 0] = frame_idx + 10

    with h5py.File(path, "w") as handle:
        hand = handle.create_group("hand")
        left = hand.create_group("left")
        left.create_dataset("joints3d", data=left_joints)
        left.create_dataset("valid", data=np.asarray([1, 1, 0, 1], dtype=np.bool_))
        right = hand.create_group("right")
        right.create_dataset("joints3d", data=right_joints)
        right.create_dataset("valid", data=np.asarray([1, 0, 1, 1], dtype=np.bool_))

    clip = load_supplier_hdf5_clip(path, episode_idx=13, fps=29.97)

    assert clip.num_frames == num_frames
    assert len(clip.keypoints or {}) == 42
    assert np.array_equal(clip.keypoints["leftHand"], left_joints[:, 0, :])
    assert np.array_equal(clip.keypoints["leftThumbTip"], left_joints[:, 16, :])
    assert np.array_equal(clip.keypoints["leftIndexFingerTip"], left_joints[:, 17, :])
    assert np.array_equal(clip.keypoints["rightLittleFingerTip"], right_joints[:, 20, :])
    assert np.array_equal(
        clip.quality_hand,
        np.asarray([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32),
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
    assert all(result.flag is False for result in missing_results)
    assert any(result.metrics["quality_low_left"] > 0.0 for result in missing_results)
    assert all(
        result.metrics["missing_frames_in_10s_window_left"] == 0.0
        for result in missing_results
    )
    assert any(
        result.metrics["acceptance_joint_count"] == 42.0
        for result in missing_results
    )
    assert not (
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
    assert all(result.frame_idx == -1 for result in verify_results)
    assert all(result.flag is None for result in verify_results)
    assert (tmp_path / "verify" / "check_results.json").exists()
    assert (tmp_path / "verify" / "clip_aggregates.json").exists()


def test_precheck_directory_input_auto_adapter(tmp_path: Path) -> None:
    hdf5_dir = tmp_path / "supplier_hdf5"
    hdf5_dir.mkdir()
    _write_minimal_supplier_hdf5(
        hdf5_dir / "episode_000001.hdf5",
        np.asarray([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32),
    )
    _write_minimal_supplier_hdf5(
        hdf5_dir / "episode_000002.hdf5",
        np.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
    )

    config = PrecheckConfig(
        output_dir=tmp_path / "precheck_directory",
        input_paths=[hdf5_dir],
        enabled_checks=["quality_score"],
        overwrite=True,
    )
    results = PrecheckRunner(config).run()

    summaries = [result for result in results if result.frame_idx == -1]
    assert len(summaries) == 2
    assert [summary.episode_idx for summary in summaries] == [0, 1]
    assert summaries[0].metrics["pass_ratio"] == 0.5
    assert summaries[1].metrics["pass_ratio"] == 1.0


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


def test_projection_helper_metrics() -> None:
    intrinsics = np.asarray(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    points = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [-0.45, 0.0, 1.0],
            [0.60, 0.0, 1.0],
            [0.0, -0.35, 1.0],
            [0.0, 0.50, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )

    projected = project_points_with_validity(points, intrinsics)
    assert np.allclose(projected["u"][:2], [50.0, 5.0])
    assert np.allclose(projected["v"][:2], [40.0, 40.0])
    assert projected["projection_valid"].tolist() == [
        True,
        True,
        True,
        True,
        True,
        False,
    ]

    metrics = hand_projection_metrics(
        points,
        intrinsics,
        ProjectionConfig(image_width=100, image_height=80, border_margin_px=10.0),
        previous_center=(40.0, 40.0),
    )

    assert metrics["num_projected_keypoints"] == 6.0
    assert metrics["num_projection_valid"] == 5.0
    assert metrics["num_projection_invalid"] == 1.0
    assert metrics["num_points_outside_image"] == 2.0
    assert metrics["num_points_near_border"] == 2.0
    assert metrics["u_min"] == 5.0
    assert metrics["u_max"] == 110.0
    assert metrics["v_min"] == 5.0
    assert metrics["v_max"] == 90.0
    assert metrics["hand_bbox_area_2d"] == 105.0 * 85.0
    assert metrics["keypoint_bbox_touches_border"] == 1.0
    assert metrics["hand_bbox_center_jump_px"] > 10.0


def test_skeleton_projection_config_missing_skips_cleanly(tmp_path: Path) -> None:
    clip = _rotation_jump_clip(episode_idx=110)
    precheck_config = PrecheckConfig(
        output_dir=tmp_path / "skeleton_projection_skip",
        enabled_checks=["skeleton_quality_score"],
        skeleton_quality_score=SkeletonQualityScoreConfig(
            decision_mode="temporal_triage",
            projection_enabled=True,
        ),
        overwrite=True,
    )
    results = PrecheckRunner(precheck_config).run([clip])

    frame_rows = [
        result
        for result in results
        if result.check == "skeleton_quality_score" and result.frame_idx != -1
    ]
    assert frame_rows
    assert all(row.metrics["projection_enabled"] == 0.0 for row in frame_rows)
    assert frame_rows[0].flag is None
    assert all(row.severity in {"uncalibrated", "pass", "warn"} for row in frame_rows)
    assert any(row.severity == "warn" for row in frame_rows)


def test_skeleton_rotation_review_requires_projection_risk(tmp_path: Path) -> None:
    safe_clip = _rotation_jump_clip(episode_idx=111)
    projection_config = SkeletonQualityScoreConfig(
        decision_mode="temporal_triage",
        projection_image_width=1280,
        projection_image_height=720,
        projection_fx=1000.0,
        projection_fy=1000.0,
        projection_cx=640.0,
        projection_cy=360.0,
        projection_border_margin_px=20.0,
        projection_near_border_count_threshold=6,
    )
    safe_config = PrecheckConfig(
        output_dir=tmp_path / "rotation_safe_projection",
        enabled_checks=["skeleton_quality_score"],
        skeleton_quality_score=projection_config,
        overwrite=True,
    )
    safe_results = PrecheckRunner(safe_config).run([safe_clip])
    safe_row = next(
        result
        for result in safe_results
        if result.check == "skeleton_quality_score" and result.frame_idx == 22
    )
    assert safe_row.flag is True
    assert safe_row.severity == "warn"
    assert safe_row.metrics["skeleton_verdict"] == "review"
    assert safe_row.metrics["which_thresholds_exceeded"] == ["rotation_delta_max"]
    assert safe_row.metrics["needs_rotation_mask_review"] == 0.0
    assert safe_row.metrics["needs_visual_review"] == 1.0
    assert not (tmp_path / "rotation_safe_projection" / "candidate_windows.json").exists()

    edge_clip = _rotation_jump_clip(episode_idx=112)
    for joint in acceptance_joint_names(["left"]):
        values = edge_clip.keypoints[joint]
        values[:, 0] = -0.63
        values[:, 1] = 0.0
        values[:, 2] = 1.0
    edge_config = PrecheckConfig(
        output_dir=tmp_path / "rotation_edge_projection",
        enabled_checks=["skeleton_quality_score"],
        skeleton_quality_score=projection_config,
        overwrite=True,
    )
    edge_results = PrecheckRunner(edge_config).run([edge_clip])
    edge_row = next(
        result
        for result in edge_results
        if result.check == "skeleton_quality_score" and result.frame_idx == 22
    )
    assert edge_row.flag is True
    assert edge_row.severity == "warn"
    assert edge_row.metrics["skeleton_verdict"] == "review"
    assert edge_row.metrics["left_num_points_near_border"] == 21.0
    assert edge_row.metrics["left_needs_projection_review"] == 1.0
    assert edge_row.metrics["left_needs_rotation_mask_review"] == 1.0
    assert edge_row.metrics["needs_rotation_mask_review"] == 1.0
    assert edge_row.metrics["needs_visual_review"] == 1.0
    assert not (tmp_path / "rotation_edge_projection" / "candidate_windows.json").exists()


def test_skeleton_candidate_windows_use_strict_temporal_seed_runs() -> None:
    clip = ClipInputs(episode_idx=12, frame_indices=list(range(40)))
    check = _candidate_check()

    flag_only_rows = [
        _skeleton_candidate_result(frame_idx, [], flag=True)
        for frame_idx in [5, 6, 7]
    ]
    assert check.build_candidate_windows(clip, flag_only_rows) == []

    invalid_rows = [
        _skeleton_candidate_result(
            frame_idx,
            ["joint_acceleration_m_s2_max"],
            flag=True,
            invalid=True,
        )
        for frame_idx in [5, 6, 7]
    ]
    assert check.build_candidate_windows(clip, invalid_rows) == []

    rotation_only_rows = [
        _skeleton_candidate_result(frame_idx, ["rotation_delta_max"], flag=True)
        for frame_idx in [5, 6, 7]
    ]
    assert check.build_candidate_windows(clip, rotation_only_rows) == []

    acceleration_rows = [
        _skeleton_candidate_result(frame_idx, ["joint_acceleration_m_s2_max"])
        for frame_idx in [5, 6, 7]
    ]
    acceleration_windows = check.build_candidate_windows(clip, acceleration_rows)
    assert len(acceleration_windows) == 1
    assert acceleration_windows[0]["seed_run_start"] == 5
    assert acceleration_windows[0]["seed_run_end"] == 7
    assert acceleration_windows[0]["seed_run_frames"] == 3.0
    assert acceleration_windows[0]["start_frame"] == 4
    assert acceleration_windows[0]["end_frame"] == 8
    assert acceleration_windows[0]["window_source"] == "skeleton_quality_temporal_run"
    assert acceleration_windows[0]["review_type"] == ["temporal_geometry_review"]
    assert acceleration_windows[0]["sam3_eligible"] is True
    assert "acceleration_seed" in acceleration_windows[0]["trigger_reason"]

    displacement_rows = [
        _skeleton_candidate_result(frame_idx, ["joint_displacement_m_max"])
        for frame_idx in [10, 11, 12]
    ]
    displacement_windows = check.build_candidate_windows(clip, displacement_rows)
    assert len(displacement_windows) == 1
    assert "displacement_seed" in displacement_windows[0]["trigger_reason"]

    multi_signal_rows = [
        _skeleton_candidate_result(
            frame_idx,
            ["rotation_delta_max", "joint_angle_change_deg_max"],
        )
        for frame_idx in [15, 16, 17]
    ]
    multi_signal_windows = check.build_candidate_windows(clip, multi_signal_rows)
    assert len(multi_signal_windows) == 1
    assert "multi_signal_seed" in multi_signal_windows[0]["trigger_reason"]


def test_uncalibrated_temporal_rows_never_seed_candidate_windows() -> None:
    clip = ClipInputs(episode_idx=12, frame_indices=list(range(40)))
    check = _candidate_check(rotation_delta_extreme_review_threshold=0.49)

    extreme_rotation_rows = [
        _skeleton_candidate_result(
            frame_idx,
            ["rotation_delta_max"],
            flag=None,
            temporal_output_valid=False,
        )
        for frame_idx in [5, 6, 7]
    ]
    assert check.build_candidate_windows(clip, extreme_rotation_rows) == []

    side_view_rows = [
        _skeleton_candidate_result(
            frame_idx,
            [],
            flag=None,
            temporal_output_valid=None,
        )
        for frame_idx in [10, 11, 12]
    ]
    for row in side_view_rows:
        row.metrics["palm_camera_angle_deg_max"] = 90.0
        row.metrics["side_view_hand_count"] = 2.0
    side_view_check = _candidate_check(
        palm_camera_angle_review_threshold_deg=60.0,
    )
    assert side_view_check.build_candidate_windows(clip, side_view_rows) == []

    missing_validity_rows = [
        CheckResult(
            "skeleton_quality_score",
            12,
            frame_idx,
            {
                "which_thresholds_exceeded": ["joint_acceleration_m_s2_max"],
                "joint_acceleration_m_s2_max": 16.0,
                "keypoint_presence_invalid": 0.0,
            },
            True,
            "legacy row without explicit temporal validity",
            severity="warn",
        )
        for frame_idx in [15, 16, 17]
    ]
    assert check.build_candidate_windows(clip, missing_validity_rows) == []


def test_skeleton_extreme_rotation_candidate_routes_to_manual_review() -> None:
    clip = ClipInputs(episode_idx=12, frame_indices=list(range(40)))
    check = _candidate_check(rotation_delta_extreme_review_threshold=0.49)

    rotation_rows = [
        _skeleton_candidate_result(frame_idx, ["rotation_delta_max"], flag=True)
        for frame_idx in [5, 6, 7]
    ]
    windows = check.build_candidate_windows(clip, rotation_rows)

    assert len(windows) == 1
    window = windows[0]
    assert window["seed_run_start"] == 5
    assert window["seed_run_end"] == 7
    assert window["review_type"] == ["rotation_manual_review"]
    assert window["window_source"] == "skeleton_rotation_extreme"
    assert window["needs_manual_review"] is True
    assert window["sam3_eligible"] is False
    assert "extreme_rotation_delta" in window["trigger_reason"]
    assert window["trigger_metrics"]["rotation_delta_max"] == 0.5


def test_palm_orientation_front_view_does_not_trigger_side_view() -> None:
    num_frames = 4
    clip = ClipInputs(
        episode_idx=12,
        frame_indices=list(range(num_frames)),
        keypoints=_palm_orientation_keypoints(num_frames, side_view=False),
        fps=1.0,
    )
    check = _candidate_check(palm_camera_angle_review_threshold_deg=60.0)

    results = check.run(clip)
    frame_rows = [
        result
        for result in results
        if result.check == "skeleton_quality_score" and result.frame_idx >= 0
    ]

    assert frame_rows
    assert all(row.metrics["palm_camera_angle_deg_max"] < 1.0 for row in frame_rows)
    assert all(row.metrics["side_view_hand_count"] == 0.0 for row in frame_rows)
    assert check.candidate_windows == []


def test_palm_orientation_side_view_routes_to_manual_review() -> None:
    num_frames = 4
    clip = ClipInputs(
        episode_idx=12,
        frame_indices=list(range(num_frames)),
        keypoints=_palm_orientation_keypoints(num_frames, side_view=True),
        fps=1.0,
    )
    check = _candidate_check(palm_camera_angle_review_threshold_deg=60.0)

    results = check.run(clip)
    frame_rows = [
        result
        for result in results
        if result.check == "skeleton_quality_score" and result.frame_idx >= 0
    ]

    assert frame_rows
    assert all(row.metrics["palm_camera_angle_deg_max"] > 89.0 for row in frame_rows)
    assert all(row.metrics["side_view_hand_count"] == 2.0 for row in frame_rows)
    assert len(check.candidate_windows) == 1
    window = check.candidate_windows[0]
    assert window["review_type"] == ["side_view_manual_review"]
    assert window["window_source"] == "hand_absolute_orientation"
    assert window["needs_manual_review"] is True
    assert window["sam3_eligible"] is False
    assert "side_view_hand_orientation" in window["trigger_reason"]
    assert window["trigger_metrics"]["palm_camera_angle_deg_max"] > 89.0


def test_palm_orientation_threshold_none_keeps_existing_behavior() -> None:
    num_frames = 4
    clip = ClipInputs(
        episode_idx=12,
        frame_indices=list(range(num_frames)),
        keypoints=_palm_orientation_keypoints(num_frames, side_view=True),
        fps=1.0,
    )
    check = _candidate_check()

    results = check.run(clip)
    frame_rows = [
        result
        for result in results
        if result.check == "skeleton_quality_score" and result.frame_idx >= 0
    ]

    assert frame_rows
    assert all(row.metrics["palm_camera_angle_deg_max"] > 89.0 for row in frame_rows)
    assert all(row.metrics["side_view_hand_count"] == 0.0 for row in frame_rows)
    assert check.candidate_windows == []


def test_skeleton_candidate_seed_runs_before_context_expansion() -> None:
    clip = ClipInputs(episode_idx=12, frame_indices=list(range(40)))
    check = _candidate_check()

    scattered_rows = [
        _skeleton_candidate_result(frame_idx, ["joint_acceleration_m_s2_max"])
        for frame_idx in [2, 8, 14]
    ]
    assert check.build_candidate_windows(clip, scattered_rows) == []

    gap_closed_rows = [
        _skeleton_candidate_result(frame_idx, ["joint_acceleration_m_s2_max"])
        for frame_idx in [5, 8, 11]
    ]
    gap_closed_windows = check.build_candidate_windows(clip, gap_closed_rows)
    assert len(gap_closed_windows) == 1
    assert gap_closed_windows[0]["seed_run_start"] == 5
    assert gap_closed_windows[0]["seed_run_end"] == 11
    assert gap_closed_windows[0]["seed_run_frames"] == 3.0
    assert gap_closed_windows[0]["start_frame"] == 4
    assert gap_closed_windows[0]["end_frame"] == 12

    overlapping_rows = [
        _skeleton_candidate_result(frame_idx, ["joint_acceleration_m_s2_max"])
        for frame_idx in [5, 6, 7, 9, 10, 11]
    ]
    overlapping_windows = check.build_candidate_windows(clip, overlapping_rows)
    assert len(overlapping_windows) == 1
    assert overlapping_windows[0]["seed_run_start"] == 5
    assert overlapping_windows[0]["seed_run_end"] == 11

    separated_rows = [
        _skeleton_candidate_result(frame_idx, ["joint_acceleration_m_s2_max"])
        for frame_idx in [5, 6, 7, 20, 21, 22]
    ]
    separated_windows = check.build_candidate_windows(clip, separated_rows)
    assert len(separated_windows) == 2
    assert separated_windows[0]["end_frame"] < separated_windows[1]["start_frame"]


def test_skeleton_candidate_windows_are_written_by_runner(tmp_path: Path) -> None:
    num_frames = 30
    keypoints = _static_keypoints(num_frames)
    rotations = _synthetic_rotations(keypoints, num_frames)
    for joint in keypoints:
        values = keypoints[joint]
        for frame_idx in range(5, 9):
            values[frame_idx, 0] += 0.02 * (frame_idx - 4)
    input_file = tmp_path / "100030_hdf5.hdf5"
    input_file.write_bytes(b"")

    def clip_loader(
        path: Path,
        episode_idx: int | None,
        fps: float | None,
    ) -> list[ClipInputs]:
        assert path == input_file
        return [
            ClipInputs(
                episode_idx=0 if episode_idx is None else episode_idx,
                frame_indices=list(range(num_frames)),
                keypoints=keypoints,
                rotations=rotations,
                fps=fps,
            )
        ]

    config = PrecheckConfig(
        output_dir=tmp_path / "runner_candidate_windows",
        input_paths=[input_file],
        fps=1.0,
        enabled_checks=["skeleton_quality_score"],
        skeleton_quality_score=SkeletonQualityScoreConfig(
            decision_mode="temporal_triage",
            joint_displacement_m_max_threshold=0.005,
            candidate_gap_close_frames=2,
            candidate_min_seed_run_frames=3,
            candidate_pre_context_frames=1,
            candidate_post_context_frames=1,
        ),
        overwrite=True,
    )
    PrecheckRunner(config, clip_loader=clip_loader).run()

    candidate_path = tmp_path / "runner_candidate_windows" / "candidate_windows.json"
    assert candidate_path.exists()
    assert (tmp_path / "runner_candidate_windows" / "candidate_windows.parquet").exists()
    windows = json.loads(
        candidate_path.read_text()
    )
    assert windows
    assert windows[0]["asset_id"] == "100030"
    assert windows[0]["hand_side"] == "both"
    assert windows[0]["window_source"] == "skeleton_quality_temporal_run"
    assert windows[0]["seed_run_frames"] >= 3.0
    assert "displacement_seed" in windows[0]["trigger_reason"]


def test_skeleton_hard_invalid_flags_missing_keypoints(tmp_path: Path) -> None:
    clip = _rotation_jump_clip(episode_idx=114)
    clip.keypoints["leftHand"][1, 0] = np.nan
    precheck_config = PrecheckConfig(
        output_dir=tmp_path / "skeleton_hard_invalid",
        enabled_checks=["skeleton_quality_score"],
        skeleton_quality_score=SkeletonQualityScoreConfig(
            decision_mode="temporal_triage",
        ),
        overwrite=True,
    )
    results = PrecheckRunner(precheck_config).run([clip])

    invalid_row = next(
        result
        for result in results
        if result.check == "skeleton_quality_score" and result.frame_idx == 21
    )
    assert invalid_row.flag is True
    assert invalid_row.metrics["skeleton_verdict"] == "invalid"
    assert invalid_row.metrics["skeleton_score"] == 0.0
    assert invalid_row.metrics["missing_keypoint_count_left"] == 1.0
    assert invalid_row.metrics["keypoint_presence_invalid"] == 1.0
    assert not (tmp_path / "skeleton_hard_invalid" / "candidate_windows.json").exists()


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
    assert skeleton_rows[21].flag is False
    assert skeleton_rows[21].severity == "pass"
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
    assert summary.metrics["count_good"] == 2.0
    assert summary.metrics["count_suspect"] == 1.0
    assert summary.metrics["count_uncalibrated"] == 1.0
    assert summary.metrics["num_frames"] == 4.0
    assert summary.metrics["valid_frame_count"] == 3.0
    assert summary.metrics["pass_ratio"] == 2.0 / 3.0
    assert summary.flag is True


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
    assert frame_rows[21].flag is False
    assert frame_rows[21].severity == "pass"
    assert frame_rows[22].flag is True
    assert frame_rows[22].severity in {"warn", "fail"}
    assert frame_rows[22].metrics["which_thresholds_exceeded"] == [
        "joint_displacement_m_max"
    ]
    assert frame_rows[22].metrics["joint_displacement_m_max"] > 0.005
    assert frame_rows[22].metrics["joint_displacement_m_penalty"] == 0.25
    assert frame_rows[22].metrics["skeleton_score"] == 0.75

    aggregates = json.loads(
        (tmp_path / "skeleton_displacement_score" / "clip_aggregates.json")
        .read_text(encoding="utf-8")
    )
    aggregate = next(
        row for row in aggregates if row["check"] == "skeleton_quality_score"
    )
    assert aggregate == {
        "episode_idx": 101,
        "check": "skeleton_quality_score",
        "checked_frames": len(frame_rows),
        "flagged_frames": sum(row.flag is True for row in frame_rows.values()),
        "uncalibrated_frames": sum(
            row.flag is None for row in frame_rows.values()
        ),
        "clip_flag": True,
    }


def test_finite_extreme_coordinate_is_audited_without_global_clipping(
    tmp_path: Path,
) -> None:
    clip = _displacement_jump_clip(episode_idx=102)
    assert clip.keypoints is not None
    clip.keypoints["leftHand"][2, 0] = 181_819.0
    config = PrecheckConfig(
        output_dir=tmp_path / "finite_extreme_coordinate",
        enabled_checks=["skeleton_quality_score"],
        skeleton_quality_score=SkeletonQualityScoreConfig(),
        overwrite=True,
    )

    results = PrecheckRunner(config).run([clip])
    row = next(
        result
        for result in results
        if result.check == "skeleton_quality_score" and result.frame_idx == 22
    )

    assert row.metrics["joint_position_abs_m_max"] == 181_819.0
    assert row.metrics["joint_position_nonfinite_coordinate_count"] == 0.0
    assert row.metrics["joint_displacement_m_max"] > 100_000.0
    assert row.flag is True
    assert row.severity == "fail"


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
