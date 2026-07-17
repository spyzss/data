from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from acceptance_pull.supplier_audit import audit_supplier_data
from qc_pipeline.context import AssetContext
from tests.fixtures import solid_frame, write_test_video


def dr_context(tmp_path: Path) -> AssetContext:
    source = tmp_path / "source"
    source.mkdir()
    task = source / "task"
    task.mkdir()
    for index, name in enumerate(("head.mp4", "left.mp4", "right.mp4")):
        write_test_video(
            source / name,
            [solid_frame(40 + index * 20, width=160, height=120) for _ in range(3)],
            fps=30.0,
        )
    (source / "clip.h5").write_bytes(b"source")
    (source / "calib.json").write_text(
        json.dumps(
            {
                "head": {
                    "width": 640,
                    "height": 480,
                    "K": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
                }
            }
        ),
        encoding="utf-8",
    )
    (source / "trajectory.csv").write_text(
        "frame_idx,timestamp,tx,ty,tz,qx,qy,qz,qw\n"
        "10,0.0,0,0,0,0,0,0,1\n"
        "11,0.033,0,0,0,0,0,0,1\n"
        "13,0.099,0,0,0,0,0,0,1\n",
        encoding="utf-8",
    )
    return AssetContext(
        "task-a",
        tmp_path,
        tmp_path / "quality_archive" / "task-a.json",
        source_files={
            "hdf5": {"path": "source/clip.h5"},
            "lerobot_task": {"path": "source/task"},
            "head_video": {"path": "source/head.mp4"},
            "left_wrist_video": {"path": "source/left.mp4"},
            "right_wrist_video": {"path": "source/right.mp4"},
            "calibration": {"path": "source/calib.json"},
            "trajectory": {"path": "source/trajectory.csv"},
        },
        metadata={
            "supplier": "dr",
            "primary_camera": "head",
            "task_id": "task-a",
            "task": "pick object",
            "manifest_row": {
                "supplier": "dr",
                "task_id": "task-a",
                "task": "pick object",
            },
        },
    )


def dr_mapping() -> dict[str, object]:
    return {
        "suppliers": {
            "dr": {
                "mapping_status": "verified",
                "max_trajectory_gap_frames": 1,
                "mapping": {
                    "trajectory": {
                        "frame_index_column": "frame_idx",
                        "timestamp_column": "timestamp",
                        "timestamp_unit": "s",
                        "translation_columns": ["tx", "ty", "tz"],
                        "quaternion_xyzw_columns": ["qx", "qy", "qz", "qw"],
                        "transform_direction": "world_to_camera",
                    },
                    "calibration": {
                        "intrinsics_matrix_path": "head.K",
                        "width_path": "head.width",
                        "height_path": "head.height",
                    },
                },
            }
        }
    }


def test_dr_trajectory_gap_and_calibration_metadata_are_explicit(
    tmp_path: Path,
) -> None:
    raw = audit_supplier_data(dr_context(tmp_path), dr_mapping())

    trajectory = raw["structured"]["trajectory"]
    calibration = raw["structured"]["calibration"]
    assert trajectory["frame_index_min"] == 10
    assert trajectory["frame_index_max"] == 13
    assert trajectory["frame_index_gap_count"] == 1
    assert trajectory["transform_direction"] == "world_to_camera"
    assert calibration["resolution"] == [640, 480]
    assert calibration["intrinsics_shape"] == [3, 3]
    assert raw["decision"] == "fail"
    assert any(issue["code"] == "timeline_invalid" for issue in raw["issues"])


def test_dr_audits_all_camera_metadata_duration_and_manifest_metadata(
    tmp_path: Path,
) -> None:
    raw = audit_supplier_data(dr_context(tmp_path), dr_mapping())

    cameras = raw["structured"]["video_metadata"]
    assert set(cameras) == {"head", "left_wrist", "right_wrist"}
    for camera in cameras.values():
        assert camera == {
            "status": "pass",
            "frame_count": 3,
            "width": 160,
            "height": 120,
            "fps": 30.0,
            "duration_sec": 0.1,
        }
    assert raw["manifest_metadata"] == {
        "supplier": "dr",
        "task_id": "task-a",
        "task": "pick object",
    }
    assert raw["task_metadata"] == {
        "task_id": "task-a",
        "task": "pick object",
        "task_name": None,
    }


def test_dr_missing_trajectory_is_truthful_input_failure(tmp_path: Path) -> None:
    context = dr_context(tmp_path)
    (tmp_path / "source" / "trajectory.csv").unlink()

    raw = audit_supplier_data(context, dr_mapping())

    assert raw["inventory"]["trajectory"]["status"] == "missing"
    assert raw["decision"] == "fail"
    assert "trajectory" in raw["missing_sources"]


def test_dr_trajectory_gap_threshold_comes_from_supplier_config(
    tmp_path: Path,
) -> None:
    parameters = dr_mapping()
    parameters["suppliers"]["dr"]["max_trajectory_gap_frames"] = 3

    raw = audit_supplier_data(dr_context(tmp_path), parameters)

    assert raw["structured"]["trajectory"]["frame_index_gap_count"] == 0
    assert raw["structured"]["trajectory"]["frame_index_max_step"] == 3
    assert not any(
        issue["code"] == "timeline_invalid" for issue in raw["issues"]
    )


def test_dr_trajectory_pose_columns_are_checked_for_nonfinite_values(
    tmp_path: Path,
) -> None:
    context = dr_context(tmp_path)
    (tmp_path / "source" / "trajectory.csv").write_text(
        "frame_idx,timestamp,tx,ty,tz,qx,qy,qz,qw\n"
        "10,0.0,0,0,0,0,0,0,1\n"
        "11,0.033,nan,0,0,0,0,0,1\n",
        encoding="utf-8",
    )

    raw = audit_supplier_data(context, dr_mapping())

    trajectory = raw["structured"]["trajectory"]
    assert trajectory["nonfinite_value_count"] == 1
    assert trajectory["status"] == "invalid"
    assert any(issue["code"] == "timeline_invalid" for issue in raw["issues"])


def test_deepreach_alias_audits_as_canonical_dr(tmp_path: Path) -> None:
    from qc_common.config import load_qc_acceptance_config
    from qc_pipeline.adapters.supplier_data_audit import adapt_supplier_data_audit

    base = dr_context(tmp_path)
    context = replace(base, metadata={**dict(base.metadata), "supplier": "DeepReach"})

    raw = audit_supplier_data(context, dr_mapping())
    result = adapt_supplier_data_audit(raw, load_qc_acceptance_config())

    assert raw["supplier"] == "dr"
    assert raw["supplier_id"] == "dr"
    assert raw["supplier_name"] == "DR"
    assert raw["supplier_alias"] == "DeepReach"
    assert set(raw["inventory"]) == {
        "hdf5",
        "lerobot_task",
        "head_video",
        "left_wrist_video",
        "right_wrist_video",
        "calibration",
        "trajectory",
    }
    assert result.metrics["supplier"] == "dr"
    assert result.metrics["supplier_name"] == "DR"
