from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


QY_CAMERAS = (
    "left_cam_left",
    "left_cam_right",
    "mid_cam_left",
    "mid_cam_right",
    "right_cam_left",
    "right_cam_right",
)


def keypoints_2d(offset: float = 0.0) -> list[list[float]]:
    return [
        [float(index) + offset, float(index * 2) + offset]
        for index in range(21)
    ]


def keypoints_3d(offset: float = 0.0) -> list[list[float]]:
    return [
        [
            float(index) / 100.0 + offset,
            float(index * 2) / 100.0 + offset,
            0.5 + float(index) / 1000.0 + offset,
        ]
        for index in range(21)
    ]


def observation_rows(
    *,
    cameras: Iterable[str] = ("mid_cam_left",),
    source_frames: Iterable[int] = (100, 101, 102),
    hands: Iterable[str] = ("left", "right"),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for camera_index, camera in enumerate(cameras):
        for frame_offset, source_frame in enumerate(source_frames):
            for hand_index, hand in enumerate(hands):
                rows.append(
                    {
                        "camera": camera,
                        # Deliberately differs from source_frame_index.
                        "video_frame": (0, 2, 3)[frame_offset],
                        "source_frame_index": source_frame,
                        "timestamp": 1_776_064_789.6437788 + frame_offset / 30.0,
                        "hand/current_hand": hand,
                        "pred_keypoints_2d": keypoints_2d(
                            camera_index + hand_index / 10.0
                        ),
                        "keypoint_scores": [0.99] * 21,
                        "visibility": 1.0,
                        "status": "ok",
                        "score": 0.99,
                    }
                )
    return rows


def trajectory_rows(
    *,
    source_steps: Iterable[int] = (100, 101, 102),
    hands: Iterable[str] = ("left", "right"),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step_offset, source_step in enumerate(source_steps):
        for hand_index, hand in enumerate(hands):
            rows.append(
                {
                    "source_step": source_step,
                    # Different origin from 2D timestamps by contract.
                    "timestamp_seconds": step_offset / 30.0,
                    "hand": hand,
                    "keypoints_3d_ref": keypoints_3d(
                        step_offset / 100.0 + hand_index / 1000.0
                    ),
                    "reference_camera": "mid_cam_left",
                    "joint_cameras_used": [["mid_cam_left"]] * 21,
                    "quality_tier": "supplier_good",
                    "trajectory_quality": "supplier_stable",
                    "joint_mean_reprojection_error_px": [0.25] * 21,
                    "joint_max_reprojection_error_px": [0.5] * 21,
                }
            )
    return rows


def make_qy_episode(
    root: Path,
    *,
    episode_id: str = "episode_000001",
    observations: list[dict[str, Any]] | None = None,
    trajectory: list[dict[str, Any]] | None = None,
    timebase_videos: list[dict[str, Any]] | None = None,
    cameras_with_video: Iterable[str] = QY_CAMERAS,
) -> Path:
    episode = root / "category__packing" / "task_pack_box" / episode_id
    (episode / "hand_pose").mkdir(parents=True, exist_ok=True)
    (episode / "timestamps").mkdir(parents=True, exist_ok=True)
    (episode / "semantic").mkdir(parents=True, exist_ok=True)
    (episode / "videos").mkdir(parents=True, exist_ok=True)
    (episode / "review").mkdir(parents=True, exist_ok=True)
    (episode / "episode_manifest.json").write_text(
        json.dumps({"episode_id": episode_id}), encoding="utf-8"
    )
    (episode / "hand_pose" / "coordinate_system.json").write_text(
        json.dumps({"coordinate_system": "camera_frame", "units": "meters"}),
        encoding="utf-8",
    )
    (episode / "hand_pose" / "quality.json").write_text(
        json.dumps({"status": "supplier_reported"}), encoding="utf-8"
    )
    (episode / "semantic" / "annotation.json").write_text(
        json.dumps({"task": "pack box"}), encoding="utf-8"
    )
    (episode / "review" / "review.mp4").write_bytes(b"review")
    for camera in cameras_with_video:
        (episode / "videos" / f"{camera}.mp4").write_bytes(b"mp4")

    if timebase_videos is None:
        timebase_videos = [
            {
                "camera": camera,
                "status": "ok",
                "video": f"videos/{camera}.mp4",
                "frames": 4,
                "width": 640,
                "height": 480,
                "fps": 30.0,
                "source_start_frame": 100,
                "source_end_frame": 102,
                "source_start_timestamp": 1_776_064_789.6437788,
                "source_end_timestamp": 1_776_064_789.7437788,
            }
            for camera in QY_CAMERAS
        ]
    (episode / "timestamps" / "episode_timebase.json").write_text(
        json.dumps({"episode_id": episode_id, "videos": timebase_videos}),
        encoding="utf-8",
    )

    pd.DataFrame(
        observation_rows() if observations is None else observations
    ).to_parquet(episode / "hand_pose" / "observations_2d.parquet", index=False)
    pd.DataFrame(
        trajectory_rows() if trajectory is None else trajectory
    ).to_parquet(episode / "hand_pose" / "trajectory_3d.parquet", index=False)
    return episode
