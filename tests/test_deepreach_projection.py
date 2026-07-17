from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_rigid_transform_identity_and_known_rotation_translation() -> None:
    from qc_common.projection import apply_rigid_transform

    points = np.asarray([[1.0, 2.0, 3.0], [-1.0, 0.0, 2.0]])
    identity = apply_rigid_transform(points, np.eye(3), np.zeros(3))
    rotation_z_90 = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transformed = apply_rigid_transform(
        points,
        rotation_z_90,
        np.asarray([10.0, 20.0, 30.0]),
    )

    np.testing.assert_allclose(identity, points)
    np.testing.assert_allclose(
        transformed,
        [[8.0, 21.0, 33.0], [10.0, 19.0, 32.0]],
    )


def test_intrinsics_scaling_and_projection_validity() -> None:
    from qc_common.projection import (
        project_points_to_image,
        scale_intrinsics,
    )

    intrinsics = np.asarray([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
    scaled = scale_intrinsics(intrinsics, (1920, 1080), (960, 540))
    projected = project_points_to_image(
        np.asarray(
            [
                [0.0, 0.0, 2.0],
                [10.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        ),
        scaled,
        image_width=960,
        image_height=540,
    )

    np.testing.assert_allclose(scaled, [[500, 0, 480], [0, 500, 270], [0, 0, 1]])
    assert projected["projection_valid"].tolist() == [True, True, False, False]
    assert projected["in_frame"].tolist() == [True, False, False, False]


def test_projection_mapping_requires_explicit_transform_direction(
    tmp_path: Path,
) -> None:
    from acceptance_pull.supplier_adapters.deepreach_projection import (
        load_trajectory,
    )

    trajectory = tmp_path / "trajectory.csv"
    trajectory.write_text(
        "frame,tx,ty,tz,qx,qy,qz,qw\n0,0,0,0,0,0,0,1\n",
        encoding="utf-8",
    )
    mapping = {
        "frame_index_column": "frame",
        "translation_columns": ["tx", "ty", "tz"],
        "quaternion_xyzw_columns": ["qx", "qy", "qz", "qw"],
    }

    result = load_trajectory(trajectory, mapping)

    assert result.status == "transform_ambiguous"
    assert result.poses == {}


def test_projection_parsers_report_missing_trajectory_gap_and_calibration_field(
    tmp_path: Path,
) -> None:
    from acceptance_pull.supplier_adapters.deepreach_projection import (
        load_calibration,
        load_trajectory,
    )

    missing = load_trajectory(
        tmp_path / "missing.csv",
        {"transform_direction": "world_to_camera"},
    )
    trajectory = tmp_path / "trajectory.csv"
    trajectory.write_text(
        "frame,tx,ty,tz,qx,qy,qz,qw\n"
        "10,0,0,0,0,0,0,1\n"
        "12,0,0,0,0,0,0,1\n",
        encoding="utf-8",
    )
    mapped = load_trajectory(
        trajectory,
        {
            "frame_index_column": "frame",
            "translation_columns": ["tx", "ty", "tz"],
            "quaternion_xyzw_columns": ["qx", "qy", "qz", "qw"],
            "transform_direction": "world_to_camera",
        },
    )
    calibration_path = tmp_path / "calib.json"
    calibration_path.write_text(json.dumps({"head": {"width": 640}}), encoding="utf-8")
    calibration = load_calibration(
        calibration_path,
        "head",
        {
            "intrinsics_matrix_path": "head.K",
            "width_path": "head.width",
            "height_path": "head.height",
        },
    )

    assert missing.status == "calibration_unverified"
    assert mapped.status == "trajectory_gap"
    assert mapped.missing_frame_ranges == ((11, 11),)
    assert calibration.status == "calibration_unverified"


def test_projection_record_preserves_source_local_offset_and_hand_camera_metadata() -> None:
    from acceptance_pull.supplier_adapters.deepreach_projection import (
        Calibration,
        TrajectoryPose,
        project_hand,
    )

    calibration = Calibration(
        status="verified",
        camera_name="head",
        intrinsics=np.asarray([[100, 0, 50], [0, 100, 50], [0, 0, 1]], dtype=float),
        resolution=(100, 100),
        source="calib.json",
    )
    pose = TrajectoryPose(
        source_frame=105,
        rotation=np.eye(3),
        translation=np.zeros(3),
        transform_direction="world_to_camera",
    )

    records = project_hand(
        asset_id="task-a",
        source_frame=105,
        clip_start_frame=100,
        camera_name="head",
        hand_side="left",
        points=np.asarray([[0.0, 0.0, 1.0]]),
        calibration=calibration,
        pose=pose,
    )

    assert records == [
        {
            "asset_id": "task-a",
            "source_frame": 105,
            "local_frame": 5,
            "camera_name": "head",
            "hand_side": "left",
            "keypoint_index": 0,
            "projected_x": 50.0,
            "projected_y": 50.0,
            "source_x": 0.0,
            "source_y": 0.0,
            "source_z": 1.0,
            "camera_x": 0.0,
            "camera_y": 0.0,
            "camera_z": 1.0,
            "depth_z": 1.0,
            "projection_valid": True,
            "in_frame": True,
            "calibration_source": "calib.json",
            "trajectory_source_frame": 105,
            "transform_direction": "world_to_camera",
        }
    ]


def test_projection_record_is_strict_json_safe_for_invalid_depth() -> None:
    from acceptance_pull.supplier_adapters.deepreach_projection import (
        Calibration,
        TrajectoryPose,
        project_hand,
    )

    calibration = Calibration(
        status="verified",
        camera_name="head",
        intrinsics=np.asarray([[100, 0, 50], [0, 100, 50], [0, 0, 1]], dtype=float),
        resolution=(100, 100),
        source="calib.json",
    )
    pose = TrajectoryPose(
        source_frame=0,
        rotation=np.eye(3),
        translation=np.zeros(3),
        transform_direction="world_to_camera",
    )

    records = project_hand(
        asset_id="task-a",
        source_frame=0,
        clip_start_frame=0,
        camera_name="head",
        hand_side="left",
        points=np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0]]),
        calibration=calibration,
        pose=pose,
    )

    assert records[0]["projected_x"] is None
    assert records[0]["projected_y"] is None
    assert records[0]["depth_z"] == 0.0
    assert records[0]["projection_valid"] is False
    assert records[1]["projected_x"] is None
    assert records[1]["projected_y"] is None
    assert records[1]["depth_z"] == -1.0
    assert records[1]["projection_valid"] is False
    json.dumps(records, allow_nan=False)
