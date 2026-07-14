"""DeepReach deliverable discovery and video-quality staging."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_CAMERAS = ("head", "left_wrist", "right_wrist")
MANIFEST_COLUMNS = [
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
CALIBRATION_SIDECAR_COLUMNS = [
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
) -> list[dict[str, str]]:
    """Discover DeepReach task/camera assets without reading large inputs."""
    root = root.resolve()
    calib_cache = calib_cache.resolve()
    selected_cameras = validate_cameras(cameras)
    rows: list[dict[str, str]] = []
    for task_name in discover_task_names(root):
        task_dir = root / "lerobot_v2" / task_name
        hdf5_path = find_hdf5_path(root, task_name)
        parquet_path = (
            task_dir / "data" / "chunk-000" / "episode_000000.parquet"
        )
        wrist_calibration_path = task_dir / "meta" / "wrist_calibration.json"
        calib_path = calib_cache / task_name / "calib.json"
        calibration_status = (
            "present_unverified" if calib_path.is_file() else "missing"
        )

        for camera_name in selected_cameras:
            video_path = (
                task_dir
                / "videos"
                / "chunk-000"
                / f"observation.images.{camera_name}"
                / "episode_000000.mp4"
            )
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
    write_csv(manifest_path, MANIFEST_COLUMNS, rows)
    sidecar_rows = [
        {column: row[column] for column in CALIBRATION_SIDECAR_COLUMNS}
        for row in rows
    ]
    write_csv(sidecar_path, CALIBRATION_SIDECAR_COLUMNS, sidecar_rows)
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
                Path(row["video_path"]),
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
