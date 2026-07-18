from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.fixtures import solid_frame, write_test_video


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


def _write_head_projection_contract_fixture(
    tmp_path: Path,
    *,
    video_frames: int = 3,
    calibration_size: tuple[int, int] = (100, 80),
) -> tuple[Path, Path, Path, Path]:
    import h5py

    hdf5_path = tmp_path / "task.h5"
    with h5py.File(hdf5_path, "w") as handle:
        handle.attrs["coordinate_frame"] = "head_camera"
        handle.attrs["units"] = "meters"
        handle.create_dataset("timestamp", data=np.arange(3, dtype=float) / 10.0)
        for side in ("left", "right"):
            group = handle.create_group(f"hand/{side}")
            points = np.zeros((3, 21, 3), dtype=np.float32)
            points[..., 2] = 1.0
            group.create_dataset("joints3d", data=points)
    video_path = tmp_path / "head.mp4"
    write_test_video(
        video_path,
        [solid_frame(50, width=100, height=80) for _ in range(video_frames)],
        fps=10.0,
    )
    width, height = calibration_size
    calibration_path = tmp_path / "calib.json"
    calibration_path.write_text(
        json.dumps(
            {
                "head": {
                    "K": [[100, 0, width / 2], [0, 100, height / 2], [0, 0, 1]],
                    "width": width,
                    "height": height,
                }
            }
        ),
        encoding="utf-8",
    )
    trajectory_path = tmp_path / "trajectory.csv"
    trajectory_path.write_text("frame\n0\n1\n2\n", encoding="utf-8")
    return hdf5_path, video_path, calibration_path, trajectory_path


def _verified_head_mapping(*, resolution_policy: str = "exact") -> dict[str, object]:
    return {
        "projection": {
            "camera_name": "head",
            "joints3d_coordinate_frame": "head_camera",
            "joints3d_unit": "meter",
            "projection_direction": "direct_camera",
            "trajectory_usage": "lineage_only",
            "resolution_policy": resolution_policy,
        },
        "calibration": {
            "intrinsics_matrix_path": "head.K",
            "width_path": "head.width",
            "height_path": "head.height",
        },
    }


def test_head_projection_contract_requires_explicit_asset_mapping(
    tmp_path: Path,
) -> None:
    from acceptance_pull.supplier_adapters.deepreach_projection import (
        validate_head_projection_contract,
    )

    hdf5, video, calibration, trajectory = _write_head_projection_contract_fixture(
        tmp_path
    )

    result = validate_head_projection_contract(
        hdf5_path=hdf5,
        video_path=video,
        calibration_path=calibration,
        trajectory_path=trajectory,
        source_range=(0, 3),
        reference_dataset="timestamp",
        primary_camera="head",
        content_id=None,
        calibration_mapping_status="mapping_missing",
        projection_validation_status="validated",
        mapping_status="verified",
        mapping=_verified_head_mapping(),
    )

    assert result.status == "calibration_unverified"
    assert result.reason == "mapping_missing"


def test_head_projection_contract_validates_direct_head_camera_chain(
    tmp_path: Path,
) -> None:
    from acceptance_pull.supplier_adapters.deepreach_projection import (
        validate_head_projection_contract,
    )

    hdf5, video, calibration, trajectory = _write_head_projection_contract_fixture(
        tmp_path
    )

    result = validate_head_projection_contract(
        hdf5_path=hdf5,
        video_path=video,
        calibration_path=calibration,
        trajectory_path=trajectory,
        source_range=(0, 3),
        reference_dataset="timestamp",
        primary_camera="head",
        content_id="content-abc",
        calibration_mapping_status="mapped",
        projection_validation_status="validated",
        mapping_status="verified",
        mapping=_verified_head_mapping(),
    )

    assert result.status == "validated"
    assert result.reason == "validated_direct_head_projection"
    assert result.calibration.intrinsics.shape == (3, 3)
    assert result.video_resolution == (100, 80)
    assert result.frame_count == 3
    assert result.trajectory_usage == "lineage_only"


def test_head_projection_contract_rejects_unexplained_resolution_mismatch(
    tmp_path: Path,
) -> None:
    from acceptance_pull.supplier_adapters.deepreach_projection import (
        validate_head_projection_contract,
    )

    hdf5, video, calibration, trajectory = _write_head_projection_contract_fixture(
        tmp_path,
        calibration_size=(200, 160),
    )

    result = validate_head_projection_contract(
        hdf5_path=hdf5,
        video_path=video,
        calibration_path=calibration,
        trajectory_path=trajectory,
        source_range=(0, 3),
        reference_dataset="timestamp",
        primary_camera="head",
        content_id="content-abc",
        calibration_mapping_status="mapped",
        projection_validation_status="validated",
        mapping_status="verified",
        mapping=_verified_head_mapping(resolution_policy="exact"),
    )

    assert result.status == "resolution_mismatch"
    assert result.reason == "calibration_video_resolution_mismatch"


def test_head_projection_contract_requires_explicit_transform_chain(
    tmp_path: Path,
) -> None:
    from acceptance_pull.supplier_adapters.deepreach_projection import (
        validate_head_projection_contract,
    )

    hdf5, video, calibration, trajectory = _write_head_projection_contract_fixture(
        tmp_path
    )
    mapping = _verified_head_mapping()
    mapping["projection"].pop("trajectory_usage")

    result = validate_head_projection_contract(
        hdf5_path=hdf5,
        video_path=video,
        calibration_path=calibration,
        trajectory_path=trajectory,
        source_range=(0, 3),
        reference_dataset="timestamp",
        primary_camera="head",
        content_id="content-abc",
        calibration_mapping_status="mapped",
        projection_validation_status="validated",
        mapping_status="verified",
        mapping=mapping,
    )

    assert result.status == "transform_ambiguous"
    assert result.reason == "direct_head_transform_chain_not_explicit"


def test_head_projection_contract_rejects_frame_alignment_mismatch(
    tmp_path: Path,
) -> None:
    from acceptance_pull.supplier_adapters.deepreach_projection import (
        validate_head_projection_contract,
    )

    hdf5, video, calibration, trajectory = _write_head_projection_contract_fixture(
        tmp_path,
        video_frames=2,
    )

    result = validate_head_projection_contract(
        hdf5_path=hdf5,
        video_path=video,
        calibration_path=calibration,
        trajectory_path=trajectory,
        source_range=(0, 3),
        reference_dataset="timestamp",
        primary_camera="head",
        content_id="content-abc",
        calibration_mapping_status="mapped",
        projection_validation_status="validated",
        mapping_status="verified",
        mapping=_verified_head_mapping(),
    )

    assert result.status == "frame_alignment_unverified"
    assert result.reason == "hdf5_video_frame_count_mismatch"


def test_dr_projection_records_missing_hand_without_fabricating_points(
    tmp_path: Path,
) -> None:
    import h5py

    from acceptance_pull.supplier_adapters.deepreach_projection import (
        Calibration,
        project_dr_hands_for_frame,
    )

    hdf5_path = tmp_path / "missing-right.h5"
    with h5py.File(hdf5_path, "w") as handle:
        left = handle.create_group("hand/left")
        points = np.zeros((1, 21, 3), dtype=np.float32)
        points[..., 2] = 1.0
        left.create_dataset("joints3d", data=points)
    calibration = Calibration(
        status="verified",
        camera_name="head",
        intrinsics=np.asarray(
            [[100, 0, 50], [0, 100, 40], [0, 0, 1]],
            dtype=float,
        ),
        resolution=(100, 80),
        source="calib.json",
    )

    projected = project_dr_hands_for_frame(
        hdf5_path,
        source_frame=0,
        clip_start_frame=0,
        calibration=calibration,
    )

    assert projected["left"]["projection_input_status"] == "valid"
    assert projected["right"]["projection_input_status"] == "missing_hand_input"
    assert projected["right"]["projection_input_reason"] == (
        "missing_dataset:hand/right/joints3d"
    )
    assert projected["right"]["pixels"].shape == (21, 2)
    assert not projected["right"]["valid"].any()
    assert np.isnan(projected["right"]["pixels"]).all()
