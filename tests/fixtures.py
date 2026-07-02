from __future__ import annotations

from pathlib import Path

import cv2
import h5py
import numpy as np
from openpyxl import Workbook


def write_manifest(path: Path, rows: list[tuple[str, str, str]]) -> None:
    wb = Workbook()
    ws = wb.active
    ws.append(["asset_id", "scene", "task"])
    for row in rows:
        ws.append(list(row))
    wb.save(path)
    wb.close()


def write_test_video(
    path: Path,
    frames: list[np.ndarray],
    fps: float = 10.0,
) -> None:
    if not frames:
        raise ValueError("frames are required")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def solid_frame(value: int, width: int = 32, height: int = 24) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def write_quality_hdf5(path: Path, frame_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        label = handle.create_group("label")
        label.create_dataset("quality_hand", data=np.ones((frame_count, 2), dtype=np.float32))


def write_quality_hdf5_with_text(path: Path, frame_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["task"] = "pick up red cup"
        meta = handle.create_group("meta")
        meta.attrs["scene"] = "kitchen"
        meta.create_dataset("instruction", data="move the cup to the tray")
        meta.create_dataset("structured_label", data='{"language":"zh","task":"整理桌面"}')
        label = handle.create_group("label")
        label.create_dataset("quality_hand", data=np.ones((frame_count, 2), dtype=np.float32))
