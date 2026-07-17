"""Potentia transport-package discovery and task-level manifests."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DELIVERABLES = {
    "video": "video.mp4",
    "meta": "meta.json",
    "frames": "frames.csv",
    "aligned": "aligned.csv",
    "imu": "imu.csv",
    "calibration": "calibration.json",
}

MANIFEST_COLUMNS = [
    "schema_version",
    "supplier",
    "supplier_id",
    "supplier_name",
    "asset_id",
    "source_partition",
    "task_id",
    "task_dir",
    "source_granularity",
    "source_coordinate_convention",
    "video_path",
    "meta_path",
    "frames_path",
    "aligned_path",
    "imu_path",
    "calibration_path",
    "video_status",
    "meta_status",
    "frames_status",
    "aligned_status",
    "imu_status",
    "calibration_status",
    "file_inventory",
    "manifest_metadata",
    "adapter_status",
]


@dataclass(frozen=True)
class PotentiaTask:
    task_id: str
    directory: Path
    source_partition: str


def discover_potentia_tasks(root: Path) -> list[PotentiaTask]:
    """Find task directories; transport packages are never emitted as assets."""
    resolved_root = root.resolve()
    candidates = sorted(
        {
            path.parent.resolve()
            for name in DELIVERABLES.values()
            for path in resolved_root.rglob(name)
            if path.is_file()
        },
        key=lambda path: path.relative_to(resolved_root).as_posix(),
    )
    tasks: list[PotentiaTask] = []
    seen: dict[str, Path] = {}
    for directory in candidates:
        task_id = directory.name
        previous = seen.get(task_id)
        if previous is not None and previous != directory:
            raise ValueError(f"duplicate Potentia task_id: {task_id}")
        seen[task_id] = directory
        relative = directory.relative_to(resolved_root)
        partition = relative.parts[0] if len(relative.parts) > 1 else "."
        tasks.append(PotentiaTask(task_id, directory, partition))
    return tasks


def build_potentia_manifest(
    root: Path,
    *,
    max_assets: int | None = None,
) -> list[dict[str, str]]:
    if max_assets is not None and max_assets < 1:
        raise ValueError("max_assets must be >= 1")
    tasks = discover_potentia_tasks(root)
    if max_assets is not None:
        tasks = tasks[:max_assets]
    rows: list[dict[str, str]] = []
    for task in tasks:
        paths = {
            key: task.directory / filename
            for key, filename in DELIVERABLES.items()
        }
        statuses = {
            key: "present" if path.is_file() else "missing"
            for key, path in paths.items()
        }
        inventory = {
            key: {"filename": DELIVERABLES[key], "status": statuses[key]}
            for key in DELIVERABLES
        }
        rows.append(
            {
                "schema_version": "supplier_manifest.potentia.v1",
                "supplier": "potentia",
                "supplier_id": "potentia",
                "supplier_name": "Potentia",
                "asset_id": f"potentia__{task.task_id}",
                "source_partition": task.source_partition,
                "task_id": task.task_id,
                "task_dir": str(task.directory),
                "source_granularity": "task",
                "source_coordinate_convention": "source_inclusive",
                **{f"{key}_path": str(path.resolve()) for key, path in paths.items()},
                **{f"{key}_status": status for key, status in statuses.items()},
                "file_inventory": json.dumps(
                    inventory,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "manifest_metadata": json.dumps(
                    {
                        "source_partition": task.source_partition,
                        "task_id": task.task_id,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "adapter_status": (
                    "ready" if all(value == "present" for value in statuses.values())
                    else "input_missing"
                ),
            }
        )
    return rows


def write_potentia_manifest(
    rows: list[dict[str, str]],
    output_dir: Path,
) -> Path:
    path = output_dir / "manifests" / "supplier_manifest_potentia.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


__all__ = [
    "DELIVERABLES",
    "MANIFEST_COLUMNS",
    "PotentiaTask",
    "build_potentia_manifest",
    "discover_potentia_tasks",
    "write_potentia_manifest",
]
