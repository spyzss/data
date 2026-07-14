from __future__ import annotations

import csv
from pathlib import Path

from acceptance_pull.build_supplier_manifest import run
from acceptance_pull.supplier_adapters.deepreach import (
    MANIFEST_COLUMNS,
    build_deepreach_manifest,
    stage_video_quality_inputs,
)


def make_deepreach_task(
    root: Path,
    task_name: str,
    *,
    cameras: tuple[str, ...] = ("head",),
) -> None:
    (root / "hdf5").mkdir(parents=True, exist_ok=True)
    (root / "hdf5" / f"{task_name}.h5").write_bytes(b"hdf5")
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
    return path


def test_build_deepreach_manifest_defaults_to_head_and_canonical_columns(
    tmp_path: Path,
) -> None:
    root = tmp_path / "deepreach"
    calib_cache = tmp_path / "calib"
    make_deepreach_task(root, "task_001")
    calib = make_calib(calib_cache, "task_001")

    rows = build_deepreach_manifest(root=root, calib_cache=calib_cache)

    assert len(rows) == 1
    assert list(rows[0]) == MANIFEST_COLUMNS
    assert rows[0] == {
        "supplier_id": "deepreach",
        "supplier_name": "DeepReach",
        "asset_id": "task_001__head",
        "task_name": "task_001",
        "camera_name": "head",
        "hdf5_path": str((root / "hdf5" / "task_001.h5").resolve()),
        "video_path": str(
            (
                root
                / "lerobot_v2"
                / "task_001"
                / "videos"
                / "chunk-000"
                / "observation.images.head"
                / "episode_000000.mp4"
            ).resolve()
        ),
        "lerobot_task_dir": str((root / "lerobot_v2" / "task_001").resolve()),
        "parquet_path": str(
            (
                root
                / "lerobot_v2"
                / "task_001"
                / "data"
                / "chunk-000"
                / "episode_000000.parquet"
            ).resolve()
        ),
        "wrist_calibration_path": str(
            (
                root
                / "lerobot_v2"
                / "task_001"
                / "meta"
                / "wrist_calibration.json"
            ).resolve()
        ),
        "calib_path": str(calib.resolve()),
        "calibration_status": "present_unverified",
        "adapter_status": "ready",
    }


def test_build_deepreach_manifest_supports_all_camera_views(tmp_path: Path) -> None:
    root = tmp_path / "deepreach"
    calib_cache = tmp_path / "calib"
    cameras = ("head", "left_wrist", "right_wrist")
    make_deepreach_task(root, "task_001", cameras=cameras)
    make_calib(calib_cache, "task_001")

    rows = build_deepreach_manifest(
        root=root,
        calib_cache=calib_cache,
        cameras=cameras,
    )

    assert [row["camera_name"] for row in rows] == list(cameras)
    assert [row["asset_id"] for row in rows] == [
        "task_001__head",
        "task_001__left_wrist",
        "task_001__right_wrist",
    ]


def test_missing_video_or_calib_emits_input_missing_rows(tmp_path: Path) -> None:
    root = tmp_path / "deepreach"
    calib_cache = tmp_path / "calib"
    make_deepreach_task(root, "task_001", cameras=())

    rows = build_deepreach_manifest(root=root, calib_cache=calib_cache)

    assert len(rows) == 1
    assert rows[0]["adapter_status"] == "input_missing"
    assert rows[0]["calibration_status"] == "missing"
    assert rows[0]["video_path"].endswith(
        "observation.images.head/episode_000000.mp4"
    )


def test_stage_video_quality_inputs_creates_idempotent_symlinks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "deepreach"
    calib_cache = tmp_path / "calib"
    output_dir = tmp_path / "output"
    make_deepreach_task(root, "task_001")
    make_calib(calib_cache, "task_001")
    rows = build_deepreach_manifest(root=root, calib_cache=calib_cache)

    first = stage_video_quality_inputs(rows, output_dir)
    second = stage_video_quality_inputs(rows, output_dir)

    hdf5_link = (
        output_dir
        / "video_quality"
        / "hdf5"
        / "task_001__head_hdf5.hdf5"
    )
    video_link = (
        output_dir
        / "video_quality"
        / "video"
        / "task_001__head_video.mp4"
    )
    assert hdf5_link.is_symlink()
    assert hdf5_link.resolve() == Path(rows[0]["hdf5_path"])
    assert video_link.is_symlink()
    assert video_link.resolve() == Path(rows[0]["video_path"])
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
        supplier="deepreach",
        root=root,
        output_dir=output_dir,
        cameras=("head",),
        calib_cache=calib_cache,
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
    assert manifest_rows[0]["asset_id"] == "task_001__head"
    assert sidecar_rows == [
        {
            "asset_id": "task_001__head",
            "task_name": "task_001",
            "camera_name": "head",
            "calib_path": str(
                (calib_cache / "task_001" / "calib.json").resolve()
            ),
            "calibration_status": "present_unverified",
        }
    ]
