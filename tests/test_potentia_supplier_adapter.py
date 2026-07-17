from __future__ import annotations

import csv
from pathlib import Path

import pytest

from acceptance_pull.build_supplier_manifest import run
from acceptance_pull.supplier_adapters.potentia import (
    MANIFEST_COLUMNS,
    build_potentia_manifest,
)


DELIVERABLES = (
    "video.mp4",
    "meta.json",
    "frames.csv",
    "aligned.csv",
    "imu.csv",
    "calibration.json",
)


def make_task(
    root: Path,
    package: str,
    task_id: str,
    *,
    missing: tuple[str, ...] = (),
) -> Path:
    task_dir = root / package / task_id
    task_dir.mkdir(parents=True)
    for name in DELIVERABLES:
        if name in missing:
            continue
        path = task_dir / name
        if path.suffix == ".json":
            path.write_text("{}", encoding="utf-8")
        elif path.suffix == ".csv":
            path.write_text("frame_index,timestamp\n1,0.0\n", encoding="utf-8")
        else:
            path.write_bytes(b"video")
    return task_dir


def test_package_is_transport_and_each_task_directory_is_one_asset(
    tmp_path: Path,
) -> None:
    root = tmp_path / "potentia"
    first = make_task(root, "package_001", "task_a")
    second = make_task(root, "package_002", "task_b")

    rows = build_potentia_manifest(root)

    assert [row["asset_id"] for row in rows] == [
        "potentia__task_a",
        "potentia__task_b",
    ]
    assert all(list(row) == MANIFEST_COLUMNS for row in rows)
    assert [row["source_partition"] for row in rows] == [
        "package_001",
        "package_002",
    ]
    assert [row["task_dir"] for row in rows] == [
        str(first.resolve()),
        str(second.resolve()),
    ]
    assert {row["source_granularity"] for row in rows} == {"task"}
    assert {row["supplier"] for row in rows} == {"potentia"}
    assert {row["supplier_name"] for row in rows} == {"Potentia"}


def test_missing_deliverable_is_recorded_without_dropping_task(tmp_path: Path) -> None:
    root = tmp_path / "potentia"
    task_dir = make_task(root, "package_001", "task_a", missing=("imu.csv",))

    rows = build_potentia_manifest(root)

    assert len(rows) == 1
    assert rows[0]["imu_path"] == str((task_dir / "imu.csv").resolve())
    assert rows[0]["imu_status"] == "missing"
    assert rows[0]["adapter_status"] == "input_missing"


def test_duplicate_task_id_across_packages_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "potentia"
    make_task(root, "package_001", "task_a")
    make_task(root, "package_002", "task_a")

    with pytest.raises(ValueError, match="duplicate Potentia task_id: task_a"):
        build_potentia_manifest(root)


def test_potentia_cli_writes_bounded_manifest(tmp_path: Path) -> None:
    root = tmp_path / "potentia"
    make_task(root, "package_001", "task_a")
    make_task(root, "package_001", "task_b")
    output = tmp_path / "output"

    exit_code = run(
        supplier="potentia",
        root=root,
        output_dir=output,
        max_assets=1,
    )

    assert exit_code == 0
    manifest = output / "manifests" / "supplier_manifest_potentia.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["asset_id"] for row in rows] == ["potentia__task_a"]
