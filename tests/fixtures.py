from __future__ import annotations

import json
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


def write_standard_hdf5_episode(
    episode_dir: Path,
    *,
    asset_id: str = "asset-001",
    frame_count: int = 3,
    fps_num: int = 10,
    fps_den: int = 1,
    hand_quality: str = "missing",
) -> tuple[Path, Path]:
    """Write a tiny real v1 Standard HDF5 episode and its external MP4."""

    if hand_quality not in {"missing", "false", "provided"}:
        raise ValueError(f"unsupported hand_quality fixture mode: {hand_quality}")
    episode_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = episode_dir / f"{asset_id}.h5"
    video_path = episode_dir / "main.mp4"
    write_test_video(
        video_path,
        [solid_frame(30 + index * 20) for index in range(frame_count)],
        fps=fps_num / fps_den,
    )

    timestamps_ns = np.arange(frame_count, dtype=np.int64) * (
        1_000_000_000 * fps_den // fps_num
    )
    keypoints_3d = np.ones((frame_count, 2, 21, 3), dtype=np.float32)
    keypoints_2d = np.ones((frame_count, 2, 21, 2), dtype=np.float32)
    valid = np.ones((frame_count, 2, 21), dtype=np.bool_)
    annotation = {
        "scene_id": "kitchen",
        "task_id": "pick-object",
        "task_category": "manipulation",
        "task_cn": "拿起物体",
        "task_en": "pick up object",
        "description_cn": "拿起物体并放到桌面中央",
        "description_en": "pick up the object and place it at the center",
        "subtask_sequence": [
            {
                "subtask_id": "subtask_001",
                "start_frame": 0,
                "end_frame_exclusive": frame_count,
                "description_cn": "拿起物体",
                "description_en": "pick up the object",
            }
        ],
    }

    with h5py.File(hdf5_path, "w") as handle:
        handle.attrs.update(
            {
                "schema_version": "egodata_hdf5_qc_input.v1",
                "asset_id": asset_id,
                "batch_id": "batch-001",
                "supplier_id": "supplier-001",
                "frame_count": np.int64(frame_count),
                "fps_num": np.int64(fps_num),
                "fps_den": np.int64(fps_den),
                "joint_topology": "egodata_hand21.v1",
                "coordinate_frame_3d": "camera:main",
                "length_unit": "meter",
                "coordinate_space_2d": "pixel",
            }
        )
        handle.create_dataset("/time/timestamps_ns", data=timestamps_ns)
        handle.create_dataset(
            "/observation/hand_keypoints_3d", data=keypoints_3d
        )
        handle.create_dataset(
            "/observation/hand_joint_valid_3d", data=valid
        )
        handle.create_dataset(
            "/observation/hand_keypoints_2d", data=keypoints_2d
        )
        handle.create_dataset(
            "/observation/hand_joint_valid_2d", data=valid.copy()
        )
        camera = handle.create_group("/camera/main")
        camera.attrs.update(
            {
                "distortion_model": "none",
                "image_width_px": np.int32(32),
                "image_height_px": np.int32(24),
                "camera_axes": "x_right_y_down_z_forward",
                "pixel_origin": "top_left",
            }
        )
        camera.create_dataset(
            "intrinsic_matrix",
            data=np.array(
                [[20.0, 0.0, 16.0], [0.0, 20.0, 12.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
        )
        camera.create_dataset(
            "distortion_coefficients", data=np.array([], dtype=np.float64)
        )
        string_dtype = h5py.string_dtype(encoding="utf-8")
        handle.create_dataset(
            "/semantics/annotation_json",
            data=json.dumps(annotation, ensure_ascii=False),
            dtype=string_dtype,
        )
        if hand_quality != "missing":
            quality = handle.create_group("/supplier/hand_quality")
            quality.attrs["provided"] = hand_quality == "provided"
            if hand_quality == "provided":
                quality.attrs["mapping_version"] = "supplier-001.hand-quality.v1"
                quality.create_dataset(
                    "raw_value",
                    data=np.arange(frame_count * 2, dtype=np.int16).reshape(
                        frame_count, 2
                    ),
                )
                quality.create_dataset(
                    "normalized_score",
                    data=np.full((frame_count, 2), 0.75, dtype=np.float32),
                )
                status_pattern = np.array(
                    [[0, 1], [2, 3], [3, 0]], dtype=np.uint8
                )
                status = status_pattern[
                    np.arange(frame_count) % len(status_pattern)
                ]
                quality.create_dataset("status", data=status)
    return hdf5_path, video_path


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
