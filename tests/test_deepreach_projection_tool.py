from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from tests.fixtures import solid_frame, write_test_video


def write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    hdf5 = tmp_path / "task-a.h5"
    with h5py.File(hdf5, "w") as handle:
        handle.create_dataset("timestamp", data=np.asarray([0.0, 0.1]))
        for side in ("left", "right"):
            group = handle.create_group(f"hand/{side}")
            group.create_dataset("valid", data=np.asarray([1, 1], dtype=np.uint8))
            points = np.zeros((2, 21, 3), dtype=np.float32)
            points[..., 2] = 1.0
            points[1, :, 0] = 0.1 if side == "left" else -0.1
            group.create_dataset("joints3d", data=points)
    video = tmp_path / "head.mp4"
    write_test_video(
        video,
        [solid_frame(40, width=100, height=100), solid_frame(80, width=100, height=100)],
        fps=10.0,
    )
    calibration = tmp_path / "calib.json"
    calibration.write_text(
        json.dumps(
            {
                "head": {
                    "width": 200,
                    "height": 200,
                    "K": [[200, 0, 100], [0, 200, 100], [0, 0, 1]],
                }
            }
        ),
        encoding="utf-8",
    )
    trajectory = tmp_path / "trajectory.csv"
    trajectory.write_text(
        "frame,tx,ty,tz,qx,qy,qz,qw\n"
        "100,0,0,0,0,0,0,1\n"
        "101,0,0,0,0,0,0,1\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "supplier",
                "asset_id",
                "start_frame",
                "end_frame",
                "hdf5_path",
                "head_video_path",
                "calib_path",
                "camera_trajectory_path",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "supplier": "dr",
                "asset_id": "task-a",
                "start_frame": 100,
                "end_frame": 101,
                "hdf5_path": hdf5,
                "head_video_path": video,
                "calib_path": calibration,
                "camera_trajectory_path": trajectory,
            }
        )
    mapping = tmp_path / "mapping.yaml"
    mapping.write_text(
        yaml.safe_dump(
            {
                "dr_projection": {
                    "transform_chain": {"calibration_extrinsic": "identity"},
                    "trajectory": {
                        "frame_index_column": "frame",
                        "translation_columns": ["tx", "ty", "tz"],
                        "quaternion_xyzw_columns": ["qx", "qy", "qz", "qw"],
                        "transform_direction": "world_to_camera",
                    },
                    "calibration": {
                        "cameras": {
                            "head": {
                                "intrinsics_matrix_path": "head.K",
                                "width_path": "head.width",
                                "height_path": "head.height",
                            }
                        }
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return manifest, mapping


def test_projection_tool_samples_source_frame_once_and_writes_combined_overlay(
    tmp_path: Path,
) -> None:
    from tools.audit_deepreach_projection import run_projection_audit

    manifest, mapping = write_fixture(tmp_path)
    output = tmp_path / "projection"

    summary = run_projection_audit(
        manifest=manifest,
        mapping_config=mapping,
        output_dir=output,
        asset_ids=("task-a",),
        cameras=("head",),
        frames=(101,),
        max_samples=1,
    )

    records = json.loads((output / "projection_records.json").read_text())
    audit = json.loads((output / "projection_audit_summary.json").read_text())
    assert summary == {
        "asset_count": 1,
        "sample_count": 1,
        "record_count": 42,
        "overlay_count": 1,
        "blocked_count": 0,
    }
    assert audit[0]["status"] == "projected_unverified"
    assert audit[0]["source_frame"] == 101
    assert audit[0]["local_frame"] == 1
    assert {row["hand_side"] for row in records} == {"left", "right"}
    assert {row["camera_name"] for row in records} == {"head"}
    assert {row["source_frame"] for row in records} == {101}
    assert {row["local_frame"] for row in records} == {1}
    assert {row["trajectory_source"] for row in records} == {
        str((tmp_path / "trajectory.csv"))
    }
    first_x = sorted(
        row["projected_x"] for row in records if row["keypoint_index"] == 0
    )
    assert first_x == pytest.approx([40.0, 60.0])
    assert {tuple(row["intrinsics_source_resolution"]) for row in records} == {
        (200, 200)
    }
    assert {tuple(row["projection_resolution"]) for row in records} == {(100, 100)}
    assert {row["intrinsics_scaled"] for row in records} == {True}
    assert (output / "projection_records.csv").is_file()
    assert (output / "projection_audit_summary.csv").is_file()
    run_config = json.loads((output / "run_config.json").read_text())
    assert run_config["schema_version"] == "dr_projection_audit_run.v1"
    assert run_config["max_assets"] == 5
    assert run_config["max_samples_per_asset_camera"] == 1
    overlay = output / audit[0]["overlay_path"]
    assert overlay.is_file()


def test_projection_tool_blocks_when_static_extrinsic_is_not_explicit(
    tmp_path: Path,
) -> None:
    from tools.audit_deepreach_projection import run_projection_audit

    manifest, mapping = write_fixture(tmp_path)
    payload = yaml.safe_load(mapping.read_text(encoding="utf-8"))
    payload["dr_projection"].pop("transform_chain")
    mapping.write_text(yaml.safe_dump(payload), encoding="utf-8")
    output = tmp_path / "projection"

    summary = run_projection_audit(
        manifest=manifest,
        mapping_config=mapping,
        output_dir=output,
        asset_ids=("task-a",),
        cameras=("head",),
        frames=(100,),
        max_samples=1,
    )

    audit = json.loads((output / "projection_audit_summary.json").read_text())
    assert summary["blocked_count"] == 1
    assert summary["overlay_count"] == 0
    assert audit[0]["status"] == "transform_ambiguous"
    assert audit[0]["reason"] == "calibration_extrinsic_not_explicit"


def test_projection_tool_has_an_explicit_asset_bound(tmp_path: Path) -> None:
    from tools.audit_deepreach_projection import run_projection_audit

    manifest, mapping = write_fixture(tmp_path)
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        rows = list(reader)
    second = {**rows[0], "asset_id": "task-b"}
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([rows[0], second])

    summary = run_projection_audit(
        manifest=manifest,
        mapping_config=mapping,
        output_dir=tmp_path / "bounded",
        frames=(100,),
        max_assets=1,
        max_samples=1,
    )

    assert summary["asset_count"] == 1
