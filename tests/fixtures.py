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
    write_hand_keypoint_hdf5(path, frame_count=frame_count, normalized=True)


def write_hand_keypoint_hdf5(path: Path, frame_count: int, normalized: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = np.array(
        [
            [0.42, 0.50],
            [0.40, 0.47],
            [0.38, 0.44],
            [0.36, 0.41],
            [0.34, 0.38],
            [0.43, 0.44],
            [0.43, 0.39],
            [0.43, 0.34],
            [0.43, 0.29],
            [0.46, 0.44],
            [0.47, 0.39],
            [0.48, 0.34],
            [0.49, 0.29],
            [0.49, 0.45],
            [0.51, 0.41],
            [0.53, 0.37],
            [0.55, 0.33],
            [0.51, 0.48],
            [0.54, 0.45],
            [0.57, 0.42],
            [0.60, 0.39],
        ],
        dtype=np.float32,
    )
    hands = []
    for frame_idx in range(frame_count):
        shift = np.array([frame_idx * 0.002, 0.0], dtype=np.float32)
        left = base + shift
        right = base + np.array([0.18, 0.02], dtype=np.float32) + shift
        hands.append(np.stack([left, right], axis=0))
    data = np.stack(hands, axis=0) if hands else np.zeros((0, 2, 21, 2), dtype=np.float32)
    if not normalized:
        data = data * np.array([1280, 720], dtype=np.float32)
    with h5py.File(path, "w") as handle:
        label = handle.create_group("label")
        label.create_dataset("quality_hand", data=data)


def write_quality_hdf5_with_text(path: Path, frame_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["task"] = "pick up red cup"
        meta = handle.create_group("meta")
        meta.attrs["scene"] = "kitchen"
        meta.create_dataset("instruction", data="move the cup to the tray")
        meta.create_dataset("structured_label", data='{"language":"zh","task":"整理桌面"}')
        label = handle.create_group("label")
        base = np.array(
            [
                [0.42, 0.50],
                [0.40, 0.47],
                [0.38, 0.44],
                [0.36, 0.41],
                [0.34, 0.38],
                [0.43, 0.44],
                [0.43, 0.39],
                [0.43, 0.34],
                [0.43, 0.29],
                [0.46, 0.44],
                [0.47, 0.39],
                [0.48, 0.34],
                [0.49, 0.29],
                [0.49, 0.45],
                [0.51, 0.41],
                [0.53, 0.37],
                [0.55, 0.33],
                [0.51, 0.48],
                [0.54, 0.45],
                [0.57, 0.42],
                [0.60, 0.39],
            ],
            dtype=np.float32,
        )
        data = np.stack(
            [np.stack([base, base + np.array([0.18, 0.02], dtype=np.float32)], axis=0) for _ in range(frame_count)],
            axis=0,
        )
        label.create_dataset("quality_hand", data=data)
