"""DeepReach deliverable discovery and video-quality staging."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from acceptance_pull.supplier_adapters.deepreach_hdf5 import (
    inspect_deepreach_frame_contract,
)


SUPPORTED_CAMERAS = ("head", "left_wrist", "right_wrist")
LEGACY_MANIFEST_COLUMNS = [
    "supplier_id",
    "supplier_name",
    "asset_id",
    "task_name",
    "camera_name",
    "hdf5_path",
    "video_path",
    "lerobot_task_dir",
    "parquet_path",
    "wrist_calibration_path",
    "calib_path",
    "calibration_status",
    "adapter_status",
]
MANIFEST_COLUMNS = [
    "schema_version",
    "supplier",
    "supplier_id",
    "supplier_name",
    "supplier_alias",
    "asset_id",
    "source_granularity",
    "task_name",
    "task_id",
    "start_frame",
    "end_frame",
    "source_frame_count",
    "frame_coordinate_system",
    "primary_camera",
    "primary_video_path",
    "head_video_path",
    "left_wrist_video_path",
    "right_wrist_video_path",
    "hdf5_path",
    "lerobot_task_dir",
    "parquet_path",
    "wrist_calibration_path",
    "calib_path",
    "camera_trajectory_path",
    "head_video_status",
    "left_wrist_video_status",
    "right_wrist_video_status",
    "primary_camera_status",
    "calibration_status",
    "trajectory_status",
    "hdf5_status",
    "hdf5_reference_dataset",
    "hdf5_expected_frame_count",
    "hdf5_dataset_lengths",
    "hdf5_mismatch_ranges",
    "hdf5_mismatch_count",
    "adapter_status",
]
CALIBRATION_SIDECAR_COLUMNS = [
    "asset_id",
    "task_name",
    "primary_camera",
    "calib_path",
    "camera_trajectory_path",
    "calibration_status",
    "trajectory_status",
]
LEGACY_CALIBRATION_SIDECAR_COLUMNS = [
    "asset_id",
    "task_name",
    "camera_name",
    "calib_path",
    "calibration_status",
]


def build_deepreach_manifest(
    *,
    root: Path,
    calib_cache: Path,
    cameras: Iterable[str] = ("head",),
    primary_camera: str = "head",
    granularity: str = "task",
    reference_dataset: str | None = None,
) -> list[dict[str, str]]:
    """Discover DR assets without reading large HDF5/video payloads."""
    root = root.resolve()
    calib_cache = calib_cache.resolve()
    if granularity not in {"task", "task_camera"}:
        raise ValueError(f"unsupported DeepReach granularity: {granularity}")
    if primary_camera not in SUPPORTED_CAMERAS:
        raise ValueError(f"unsupported DeepReach camera: {primary_camera}")
    if granularity == "task_camera":
        return _build_task_camera_manifest(
            root=root,
            calib_cache=calib_cache,
            cameras=cameras,
        )
    if reference_dataset is None:
        raise ValueError(
            "reference_dataset is required for task-level DeepReach manifests"
        )

    rows: list[dict[str, str]] = []
    for task_name in discover_task_names(root):
        paths = _task_paths(root, calib_cache, task_name)
        frame_contract = inspect_deepreach_frame_contract(
            paths["hdf5_path"],
            reference_dataset=reference_dataset,
        )
        expected_frame_count = frame_contract.get("expected_frame_count")
        frame_count = (
            int(expected_frame_count)
            if isinstance(expected_frame_count, int) and expected_frame_count > 0
            else None
        )
        videos = {
            camera: _video_path(paths["task_dir"], camera)
            for camera in SUPPORTED_CAMERAS
        }
        video_statuses = {
            camera: "present" if path.is_file() else "missing"
            for camera, path in videos.items()
        }
        required_paths = (
            paths["hdf5_path"],
            paths["task_dir"],
            paths["parquet_path"],
            paths["wrist_calibration_path"],
            paths["calib_path"],
            paths["trajectory_path"],
            *videos.values(),
        )
        primary_status = (
            "present"
            if videos[primary_camera].is_file()
            else "primary_camera_missing"
        )
        required_paths_present = all(path.exists() for path in required_paths)
        contract_status = str(frame_contract["status"])
        if contract_status not in {"consistent", "unreadable"}:
            adapter_status = "input_invalid"
        elif not required_paths_present or primary_status != "present":
            adapter_status = "input_missing"
        else:
            adapter_status = "ready_unverified"
        rows.append(
            {
                "schema_version": "supplier_manifest.dr.v2",
                "supplier": "dr",
                "supplier_id": "dr",
                "supplier_name": "DR",
                "supplier_alias": "deepreach",
                "asset_id": task_name,
                "source_granularity": "task",
                "task_name": task_name,
                "task_id": task_name,
                "start_frame": "0" if frame_count is not None else "",
                "end_frame": str(frame_count - 1) if frame_count is not None else "",
                "source_frame_count": str(frame_count) if frame_count is not None else "",
                "frame_coordinate_system": "source_inclusive",
                "primary_camera": primary_camera,
                "primary_video_path": resolved_text(videos[primary_camera]),
                **{
                    f"{camera}_video_path": resolved_text(videos[camera])
                    for camera in SUPPORTED_CAMERAS
                },
                "hdf5_path": resolved_text(paths["hdf5_path"]),
                "lerobot_task_dir": resolved_text(paths["task_dir"]),
                "parquet_path": resolved_text(paths["parquet_path"]),
                "wrist_calibration_path": resolved_text(
                    paths["wrist_calibration_path"]
                ),
                "calib_path": resolved_text(paths["calib_path"]),
                "camera_trajectory_path": resolved_text(paths["trajectory_path"]),
                **{
                    f"{camera}_video_status": video_statuses[camera]
                    for camera in SUPPORTED_CAMERAS
                },
                "primary_camera_status": primary_status,
                "calibration_status": (
                    "present_unverified"
                    if paths["calib_path"].is_file()
                    else "missing"
                ),
                "trajectory_status": (
                    "present_unverified"
                    if paths["trajectory_path"].is_file()
                    else "missing"
                ),
                "hdf5_status": (
                    "readable" if contract_status == "consistent" else contract_status
                ),
                "hdf5_reference_dataset": str(frame_contract["reference_dataset"]),
                "hdf5_expected_frame_count": (
                    str(expected_frame_count)
                    if isinstance(expected_frame_count, int)
                    else ""
                ),
                "hdf5_dataset_lengths": json.dumps(
                    frame_contract["dataset_lengths"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "hdf5_mismatch_ranges": json.dumps(
                    frame_contract["mismatch_ranges"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "hdf5_mismatch_count": str(frame_contract["mismatch_count"]),
                "adapter_status": adapter_status,
            }
        )
    return rows


def _task_paths(root: Path, calib_cache: Path, task_name: str) -> dict[str, Path]:
    task_dir = root / "lerobot_v2" / task_name
    calibration_dir = calib_cache / task_name
    return {
        "task_dir": task_dir,
        "hdf5_path": find_hdf5_path(root, task_name),
        "parquet_path": task_dir / "data" / "chunk-000" / "episode_000000.parquet",
        "wrist_calibration_path": task_dir / "meta" / "wrist_calibration.json",
        "calib_path": calibration_dir / "calib.json",
        "trajectory_path": calibration_dir / "camera_trajectory.csv",
    }


def _video_path(task_dir: Path, camera_name: str) -> Path:
    return (
        task_dir
        / "videos"
        / "chunk-000"
        / f"observation.images.{camera_name}"
        / "episode_000000.mp4"
    )


def _build_task_camera_manifest(
    *,
    root: Path,
    calib_cache: Path,
    cameras: Iterable[str],
) -> list[dict[str, str]]:
    selected_cameras = validate_cameras(cameras)
    rows: list[dict[str, str]] = []
    for task_name in discover_task_names(root):
        paths = _task_paths(root, calib_cache, task_name)
        task_dir = paths["task_dir"]
        hdf5_path = paths["hdf5_path"]
        parquet_path = paths["parquet_path"]
        wrist_calibration_path = paths["wrist_calibration_path"]
        calib_path = paths["calib_path"]
        calibration_status = (
            "present_unverified" if calib_path.is_file() else "missing"
        )

        for camera_name in selected_cameras:
            video_path = _video_path(task_dir, camera_name)
            required_paths = (
                hdf5_path,
                task_dir,
                parquet_path,
                wrist_calibration_path,
                video_path,
                calib_path,
            )
            adapter_status = (
                "ready"
                if all(path is not None and path.exists() for path in required_paths)
                else "input_missing"
            )
            rows.append(
                {
                    "supplier_id": "deepreach",
                    "supplier_name": "DeepReach",
                    "asset_id": asset_id_for(task_name, camera_name),
                    "task_name": task_name,
                    "camera_name": camera_name,
                    "hdf5_path": resolved_text(hdf5_path),
                    "video_path": resolved_text(video_path),
                    "lerobot_task_dir": resolved_text(task_dir),
                    "parquet_path": resolved_text(parquet_path),
                    "wrist_calibration_path": resolved_text(
                        wrist_calibration_path
                    ),
                    "calib_path": resolved_text(calib_path),
                    "calibration_status": calibration_status,
                    "adapter_status": adapter_status,
                }
            )
    return rows


def discover_task_names(root: Path) -> list[str]:
    names: set[str] = set()
    hdf5_dir = root / "hdf5"
    if hdf5_dir.is_dir():
        for suffix in ("*.h5", "*.hdf5"):
            names.update(path.stem for path in hdf5_dir.glob(suffix))
    lerobot_root = root / "lerobot_v2"
    if lerobot_root.is_dir():
        names.update(path.name for path in lerobot_root.iterdir() if path.is_dir())
    return sorted(names)


def find_hdf5_path(root: Path, task_name: str) -> Path | None:
    for suffix in (".h5", ".hdf5"):
        candidate = root / "hdf5" / f"{task_name}{suffix}"
        if candidate.is_file():
            return candidate
    return root / "hdf5" / f"{task_name}.h5"


def deepreach_frame_count(
    path: Path,
    *,
    reference_dataset: str,
) -> int | None:
    """Return the authoritative count only for a consistent frame contract."""
    contract = inspect_deepreach_frame_contract(
        path,
        reference_dataset=reference_dataset,
    )
    value = contract.get("expected_frame_count")
    return (
        int(value)
        if contract.get("status") == "consistent"
        and isinstance(value, int)
        and value > 0
        else None
    )


def validate_cameras(cameras: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(str(camera) for camera in cameras))
    if not selected:
        return ("head",)
    unsupported = sorted(set(selected) - set(SUPPORTED_CAMERAS))
    if unsupported:
        raise ValueError(f"unsupported DeepReach camera: {unsupported[0]}")
    return selected


def asset_id_for(task_name: str, camera_name: str) -> str:
    return f"{task_name}__{camera_name}"


def resolved_text(path: Path | None) -> str:
    return str(path.resolve()) if path is not None else ""


def write_manifest_outputs(
    rows: list[dict[str, str]],
    output_dir: Path,
) -> tuple[Path, Path]:
    manifest_dir = output_dir / "manifests"
    manifest_path = manifest_dir / "supplier_manifest_deepreach.csv"
    sidecar_path = manifest_dir / "deepreach_calibration_sidecar.csv"
    task_level = not rows or rows[0].get("source_granularity") == "task"
    manifest_columns = MANIFEST_COLUMNS if task_level else LEGACY_MANIFEST_COLUMNS
    sidecar_columns = (
        CALIBRATION_SIDECAR_COLUMNS
        if task_level
        else LEGACY_CALIBRATION_SIDECAR_COLUMNS
    )
    write_csv(manifest_path, manifest_columns, rows)
    sidecar_rows = [
        {column: row[column] for column in sidecar_columns}
        for row in rows
    ]
    write_csv(sidecar_path, sidecar_columns, sidecar_rows)
    return manifest_path, sidecar_path


def write_csv(
    path: Path,
    columns: list[str],
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def stage_video_quality_inputs(
    rows: list[dict[str, str]],
    output_dir: Path,
) -> dict[str, int]:
    """Create idempotent symlinks for run_acceptance_video_quality.py."""
    stage_root = output_dir / "video_quality"
    summary = {"created": 0, "reused": 0, "missing": 0}
    for row in rows:
        asset_id = row["asset_id"]
        pairs = (
            (
                Path(row["hdf5_path"]),
                stage_root / "hdf5" / f"{asset_id}_hdf5.hdf5",
            ),
            (
                Path(row.get("primary_video_path") or row.get("video_path") or ""),
                stage_root / "video" / f"{asset_id}_video.mp4",
            ),
        )
        for source, target in pairs:
            if not source.is_file():
                summary["missing"] += 1
                continue
            result = ensure_symlink(source, target)
            summary[result] += 1
    return summary


def ensure_symlink(source: Path, target: Path) -> str:
    source = source.resolve()
    if target.is_symlink():
        if target.resolve() == source:
            return "reused"
        raise FileExistsError(f"conflicting symlink: {target}")
    if target.exists():
        raise FileExistsError(f"staging target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source)
    return "created"
