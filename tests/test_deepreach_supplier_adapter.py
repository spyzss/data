from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from acceptance_pull.build_supplier_manifest import run
from acceptance_pull.supplier_adapters.deepreach import (
    LEGACY_MANIFEST_COLUMNS,
    MANIFEST_COLUMNS,
    build_deepreach_manifest,
    stage_video_quality_inputs,
)


def make_deepreach_task(
    root: Path,
    task_name: str,
    *,
    cameras: tuple[str, ...] = ("head", "left_wrist", "right_wrist"),
) -> None:
    (root / "hdf5").mkdir(parents=True, exist_ok=True)
    with h5py.File(root / "hdf5" / f"{task_name}.h5", "w") as handle:
        handle.create_dataset("timestamp", data=np.asarray([0.0, 0.1]))
        for side in ("left", "right"):
            group = handle.create_group(f"hand/{side}")
            group.create_dataset("valid", data=np.asarray([1, 1]))
            group.create_dataset("joints3d", data=np.ones((2, 21, 3)))
    task_dir = root / "lerobot_v2" / task_name
    (task_dir / "meta").mkdir(parents=True, exist_ok=True)
    (task_dir / "meta" / "info.json").write_text("{}", encoding="utf-8")
    (task_dir / "meta" / "episodes.jsonl").write_text("{}\n", encoding="utf-8")
    (task_dir / "meta" / "tasks.jsonl").write_text("{}\n", encoding="utf-8")
    (task_dir / "meta" / "wrist_calibration.json").write_text(
        "{}",
        encoding="utf-8",
    )
    parquet = task_dir / "data" / "chunk-000" / "episode_000000.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_bytes(b"parquet")
    for camera in cameras:
        video = (
            task_dir
            / "videos"
            / "chunk-000"
            / f"observation.images.{camera}"
            / "episode_000000.mp4"
        )
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"mp4")


def make_calib(calib_cache: Path, task_name: str) -> Path:
    path = calib_cache / task_name / "calib.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    (path.parent / "camera_trajectory.csv").write_text(
        "frame_idx,timestamp\n0,0.0\n",
        encoding="utf-8",
    )
    return path


def resize_deepreach_dataset(
    root: Path,
    task_name: str,
    dataset: str,
    length: int,
) -> None:
    path = root / "hdf5" / f"{task_name}.h5"
    with h5py.File(path, "a") as handle:
        del handle[dataset]
        if dataset.endswith("joints3d"):
            data = np.ones((length, 21, 3))
        elif dataset == "timestamp":
            data = np.arange(length, dtype=np.float64) * 0.1
        else:
            data = np.ones(length, dtype=np.uint8)
        handle.create_dataset(dataset, data=data)


def test_task_manifest_requires_explicit_hdf5_reference_dataset(
    tmp_path: Path,
) -> None:
    root = tmp_path / "deepreach"
    calib_cache = tmp_path / "calib"
    make_deepreach_task(root, "task_001")
    make_calib(calib_cache, "task_001")

    with pytest.raises(ValueError, match="reference_dataset"):
        build_deepreach_manifest(root=root, calib_cache=calib_cache)


def test_build_deepreach_manifest_defaults_to_head_and_canonical_columns(
    tmp_path: Path,
) -> None:
    root = tmp_path / "deepreach"
    calib_cache = tmp_path / "calib"
    make_deepreach_task(root, "task_001")
    calib = make_calib(calib_cache, "task_001")

    rows = build_deepreach_manifest(
        root=root,
        calib_cache=calib_cache,
        reference_dataset="timestamp",
    )

    assert len(rows) == 1
    assert list(rows[0]) == MANIFEST_COLUMNS
    row = rows[0]
    assert row["schema_version"] == "supplier_manifest.dr.v2"
    assert row["supplier"] == "dr"
    assert row["supplier_id"] == "dr"
    assert row["supplier_name"] == "DR"
    assert row["supplier_alias"] == "deepreach"
    assert row["asset_id"] == "task_001"
    assert row["task_name"] == "task_001"
    assert row["source_granularity"] == "task"
    assert row["start_frame"] == "0"
    assert row["end_frame"] == "1"
    assert row["source_frame_count"] == "2"
    assert row["frame_coordinate_system"] == "source_inclusive"
    assert row["primary_camera"] == "head"
    assert row["primary_video_path"] == row["head_video_path"]
    assert row["left_wrist_video_status"] == "present"
    assert row["right_wrist_video_status"] == "present"
    assert row["hdf5_path"] == str((root / "hdf5" / "task_001.h5").resolve())
    assert row["calib_path"] == str(calib.resolve())
    assert row["camera_trajectory_path"] == str(
        (calib.parent / "camera_trajectory.csv").resolve()
    )
    assert row["primary_camera_status"] == "present"
    assert row["calibration_status"] == "present_unverified"
    assert row["trajectory_status"] == "present_unverified"
    assert row["adapter_status"] == "ready_unverified"
    assert row["hdf5_reference_dataset"] == "timestamp"
    assert row["hdf5_expected_frame_count"] == "2"
    assert json.loads(row["hdf5_dataset_lengths"]) == {
        "hand/left/joints3d": 2,
        "hand/left/valid": 2,
        "hand/right/joints3d": 2,
        "hand/right/valid": 2,
        "timestamp": 2,
    }
    assert json.loads(row["hdf5_mismatch_ranges"]) == {}
    assert row["hdf5_mismatch_count"] == "0"


def test_deepreach_manifest_records_shorter_dataset_without_truncating(
    tmp_path: Path,
) -> None:
    root = tmp_path / "deepreach"
    calib_cache = tmp_path / "calib"
    make_deepreach_task(root, "task_001")
    make_calib(calib_cache, "task_001")
    resize_deepreach_dataset(root, "task_001", "hand/left/valid", 1)

    row = build_deepreach_manifest(
        root=root,
        calib_cache=calib_cache,
        reference_dataset="timestamp",
    )[0]

    assert row["source_frame_count"] == "2"
    assert row["end_frame"] == "1"
    assert row["hdf5_status"] == "inconsistent_frame_count"
    assert row["adapter_status"] == "input_invalid"
    assert json.loads(row["hdf5_dataset_lengths"])["hand/left/valid"] == 1
    assert json.loads(row["hdf5_mismatch_ranges"]) == {
        "hand/left/valid": [1, 1]
    }
    assert row["hdf5_mismatch_count"] == "1"


def test_deepreach_manifest_records_longer_and_multiple_mismatches(
    tmp_path: Path,
) -> None:
    root = tmp_path / "deepreach"
    calib_cache = tmp_path / "calib"
    make_deepreach_task(root, "task_001")
    make_calib(calib_cache, "task_001")
    resize_deepreach_dataset(root, "task_001", "hand/left/joints3d", 3)
    resize_deepreach_dataset(root, "task_001", "hand/right/valid", 1)

    row = build_deepreach_manifest(
        root=root,
        calib_cache=calib_cache,
        reference_dataset="timestamp",
    )[0]

    assert row["source_frame_count"] == "2"
    assert row["hdf5_status"] == "inconsistent_frame_count"
    assert json.loads(row["hdf5_mismatch_ranges"]) == {
        "hand/left/joints3d": [2, 2],
        "hand/right/valid": [1, 1],
    }
    assert row["hdf5_mismatch_count"] == "2"


def test_deepreach_manifest_rejects_dataset_without_frame_axis(
    tmp_path: Path,
) -> None:
    root = tmp_path / "deepreach"
    calib_cache = tmp_path / "calib"
    make_deepreach_task(root, "task_001")
    make_calib(calib_cache, "task_001")
    with h5py.File(root / "hdf5" / "task_001.h5", "a") as handle:
        del handle["hand/left/valid"]
        handle.create_dataset("hand/left/valid", data=np.asarray(1))

    row = build_deepreach_manifest(
        root=root,
        calib_cache=calib_cache,
        reference_dataset="timestamp",
    )[0]

    assert row["hdf5_status"] == "invalid_dataset_shape"
    assert row["adapter_status"] == "input_invalid"
    assert json.loads(row["hdf5_dataset_lengths"])["hand/left/valid"] is None


def test_deepreach_reference_dataset_is_explicitly_configurable(tmp_path: Path) -> None:
    root = tmp_path / "deepreach"
    calib_cache = tmp_path / "calib"
    make_deepreach_task(root, "task_001")
    make_calib(calib_cache, "task_001")
    resize_deepreach_dataset(root, "task_001", "timestamp", 3)

    row = build_deepreach_manifest(
        root=root,
        calib_cache=calib_cache,
        reference_dataset="hand/left/valid",
    )[0]

    assert row["hdf5_reference_dataset"] == "hand/left/valid"
    assert row["hdf5_expected_frame_count"] == "2"
    assert row["source_frame_count"] == "2"
    assert json.loads(row["hdf5_mismatch_ranges"]) == {"timestamp": [2, 2]}


def test_explicit_task_camera_compatibility_mode_keeps_legacy_shape(
    tmp_path: Path,
) -> None:
    root = tmp_path / "deepreach"
    calib_cache = tmp_path / "calib"
    cameras = ("head", "left_wrist", "right_wrist")
    make_deepreach_task(root, "task_001", cameras=cameras)
    make_calib(calib_cache, "task_001")

    rows = build_deepreach_manifest(
        root=root,
        calib_cache=calib_cache,
        cameras=cameras,
        granularity="task_camera",
    )

    assert all(list(row) == LEGACY_MANIFEST_COLUMNS for row in rows)
    assert [row["camera_name"] for row in rows] == list(cameras)
    assert [row["asset_id"] for row in rows] == [
        "task_001__head",
        "task_001__left_wrist",
        "task_001__right_wrist",
    ]


def test_configured_missing_primary_camera_does_not_fallback(tmp_path: Path) -> None:
    root = tmp_path / "deepreach"
    calib_cache = tmp_path / "calib"
    make_deepreach_task(root, "task_001", cameras=("head",))
    make_calib(calib_cache, "task_001")

    rows = build_deepreach_manifest(
        root=root,
        calib_cache=calib_cache,
        primary_camera="right_wrist",
        reference_dataset="timestamp",
    )

    assert len(rows) == 1
    assert rows[0]["adapter_status"] == "input_missing"
    assert rows[0]["primary_camera"] == "right_wrist"
    assert rows[0]["primary_camera_status"] == "primary_camera_missing"
    assert rows[0]["primary_video_path"] == rows[0]["right_wrist_video_path"]
    assert rows[0]["primary_video_path"].endswith(
        "observation.images.right_wrist/episode_000000.mp4"
    )


def test_stage_video_quality_inputs_creates_idempotent_symlinks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "deepreach"
    calib_cache = tmp_path / "calib"
    output_dir = tmp_path / "output"
    make_deepreach_task(root, "task_001")
    make_calib(calib_cache, "task_001")
    rows = build_deepreach_manifest(
        root=root,
        calib_cache=calib_cache,
        reference_dataset="timestamp",
    )

    first = stage_video_quality_inputs(rows, output_dir)
    second = stage_video_quality_inputs(rows, output_dir)

    hdf5_link = (
        output_dir
        / "video_quality"
        / "hdf5"
        / "task_001_hdf5.hdf5"
    )
    video_link = (
        output_dir
        / "video_quality"
        / "video"
        / "task_001_video.mp4"
    )
    assert hdf5_link.is_symlink()
    assert hdf5_link.resolve() == Path(rows[0]["hdf5_path"])
    assert video_link.is_symlink()
    assert video_link.resolve() == Path(rows[0]["primary_video_path"])
    assert first == {"created": 2, "reused": 0, "missing": 0}
    assert second == {"created": 0, "reused": 2, "missing": 0}


def test_manifest_cli_writes_manifest_calib_sidecar_and_staging(
    tmp_path: Path,
) -> None:
    root = tmp_path / "deepreach"
    calib_cache = tmp_path / "calib"
    output_dir = tmp_path / "output"
    make_deepreach_task(root, "task_001")
    make_calib(calib_cache, "task_001")

    exit_code = run(
        supplier="dr",
        root=root,
        output_dir=output_dir,
        calib_cache=calib_cache,
        hdf5_reference_dataset="timestamp",
        stage_video_quality=True,
    )

    assert exit_code == 0
    manifest_path = (
        output_dir / "manifests" / "supplier_manifest_deepreach.csv"
    )
    sidecar_path = (
        output_dir / "manifests" / "deepreach_calibration_sidecar.csv"
    )
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    with sidecar_path.open(newline="", encoding="utf-8") as handle:
        sidecar_rows = list(csv.DictReader(handle))
    assert manifest_rows[0]["supplier"] == "dr"
    assert manifest_rows[0]["asset_id"] == "task_001"
    assert manifest_rows[0]["primary_camera"] == "head"
    assert sidecar_rows == [
        {
            "asset_id": "task_001",
            "task_name": "task_001",
            "primary_camera": "head",
            "calib_path": str(
                (calib_cache / "task_001" / "calib.json").resolve()
            ),
            "camera_trajectory_path": str(
                (calib_cache / "task_001" / "camera_trajectory.csv").resolve()
            ),
            "calibration_status": "present_unverified",
            "trajectory_status": "present_unverified",
        }
    ]


def test_dr_cli_max_assets_bounds_task_level_smoke_manifest(tmp_path: Path) -> None:
    root = tmp_path / "deepreach"
    calib_cache = tmp_path / "calib"
    output_dir = tmp_path / "output"
    for task_name in ("task_001", "task_002"):
        make_deepreach_task(root, task_name)
        make_calib(calib_cache, task_name)

    run(
        supplier="dr",
        root=root,
        output_dir=output_dir,
        calib_cache=calib_cache,
        hdf5_reference_dataset="timestamp",
        max_assets=1,
    )

    with (
        output_dir / "manifests" / "supplier_manifest_deepreach.csv"
    ).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["asset_id"] for row in rows] == ["task_001"]
