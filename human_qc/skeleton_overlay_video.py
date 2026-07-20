"""Render issue-bounded videos with both canonical hand skeletons overlaid."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import logging
import os
from pathlib import Path
import subprocess
from typing import Any
import uuid

import cv2
import h5py
import numpy as np

from qc_pipeline.context import AssetContext


LOGGER = logging.getLogger(__name__)

HAND_JOINT_BASE_NAMES = (
    "Hand",
    "ThumbKnuckle",
    "ThumbIntermediateBase",
    "ThumbIntermediateTip",
    "ThumbTip",
    "IndexFingerKnuckle",
    "IndexFingerIntermediateBase",
    "IndexFingerIntermediateTip",
    "IndexFingerTip",
    "MiddleFingerKnuckle",
    "MiddleFingerIntermediateBase",
    "MiddleFingerIntermediateTip",
    "MiddleFingerTip",
    "RingFingerKnuckle",
    "RingFingerIntermediateBase",
    "RingFingerIntermediateTip",
    "RingFingerTip",
    "LittleFingerKnuckle",
    "LittleFingerIntermediateBase",
    "LittleFingerIntermediateTip",
    "LittleFingerTip",
)
HAND_JOINT_NAMES = tuple(
    f"{side}{joint_name}"
    for side in ("left", "right")
    for joint_name in HAND_JOINT_BASE_NAMES
)
HAND_SKELETON_EDGES = tuple(
    edge
    for offset in (0, 21)
    for finger_start in (1, 5, 9, 13, 17)
    for edge in (
        (offset, offset + finger_start),
        (offset + finger_start, offset + finger_start + 1),
        (offset + finger_start + 1, offset + finger_start + 2),
        (offset + finger_start + 2, offset + finger_start + 3),
    )
)
SKELETON_EVIDENCE_KINDS = frozenset(
    {
        "combined_overlay",
        "skeleton",
        "skeleton_clip",
        "skeleton_overlay",
        "skeleton_video",
    }
)


class SkeletonOverlayVideoError(RuntimeError):
    """Raised when an issue-bounded overlay video cannot be rendered safely."""


def _inside_source(context: AssetContext, source_name: str) -> Path:
    source = context.source_files.get(source_name)
    if not isinstance(source, Mapping) or not isinstance(source.get("path"), str):
        raise SkeletonOverlayVideoError(f"missing {source_name} source")
    candidate = Path(source["path"])
    if not candidate.is_absolute():
        candidate = context.batch_root / candidate
    root = context.batch_root.resolve()
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SkeletonOverlayVideoError(f"invalid {source_name} source") from exc
    if not resolved.is_file():
        raise SkeletonOverlayVideoError(f"missing {source_name} source")
    return resolved


def _evidence_rows(issue: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = issue.get("evidence", [])
    if isinstance(raw, Mapping):
        raw = [raw]
    if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
        return []
    return [row for row in raw if isinstance(row, Mapping)]


def _declares_skeleton_evidence(issue: Mapping[str, Any]) -> bool:
    if issue.get("module") == "sam3_containment":
        return True
    if isinstance(issue.get("skeleton"), Mapping):
        return True
    declared_kind = str(
        issue.get("evidence_kind", issue.get("evidence_type", ""))
    ).lower()
    if declared_kind in SKELETON_EVIDENCE_KINDS:
        return True
    return any(
        str(row.get("kind", row.get("evidence_type", ""))).lower()
        in SKELETON_EVIDENCE_KINDS
        for row in _evidence_rows(issue)
    )


def _project_frame(
    handle: h5py.File,
    frame_index: int,
    image_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    camera_to_world = np.asarray(handle["transforms/camera"][frame_index], dtype=float)
    intrinsic = np.asarray(handle["camera/intrinsic"], dtype=float)
    if camera_to_world.shape != (4, 4) or intrinsic.shape != (3, 3):
        raise SkeletonOverlayVideoError("invalid camera projection data")
    positions = np.stack(
        [
            np.asarray(handle[f"transforms/{name}"][frame_index, :3, 3], dtype=float)
            for name in HAND_JOINT_NAMES
        ],
        axis=0,
    )
    homogeneous = np.concatenate(
        [positions, np.ones((len(positions), 1), dtype=float)], axis=1
    )
    points_camera = (np.linalg.inv(camera_to_world) @ homogeneous.T).T[:, :3]
    depth = points_camera[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        horizontal = intrinsic[0, 0] * points_camera[:, 0] / depth + intrinsic[0, 2]
        vertical = intrinsic[1, 1] * points_camera[:, 1] / depth + intrinsic[1, 2]
    points = np.stack([horizontal, vertical], axis=1)
    height, width = image_shape[:2]
    valid = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(depth)
        & (depth > 0)
        & (points[:, 0] >= 0)
        & (points[:, 0] < width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < height)
    )
    return points, valid


def _draw_skeleton(image: np.ndarray, points: np.ndarray, valid: np.ndarray) -> None:
    left_color = (0, 220, 0)
    right_color = (0, 140, 255)
    for start, end in HAND_SKELETON_EDGES:
        if valid[start] and valid[end]:
            color = left_color if start < 21 else right_color
            start_point = tuple(int(round(value)) for value in points[start])
            end_point = tuple(int(round(value)) for value in points[end])
            cv2.line(image, start_point, end_point, color, 2, cv2.LINE_AA)
    for joint_index, point in enumerate(points):
        if not valid[joint_index]:
            continue
        color = left_color if joint_index < 21 else right_color
        center = tuple(int(round(value)) for value in point)
        radius = 4 if joint_index in (0, 21) else 3
        cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)
        cv2.circle(image, center, radius + 1, (255, 255, 255), 1, cv2.LINE_AA)


def _encode_browser_mp4(source: Path, output: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(output),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        LOGGER.exception("Unable to start overlay video encoder")
        raise SkeletonOverlayVideoError("overlay video encoder unavailable") from exc
    if completed.returncode != 0:
        LOGGER.error("Overlay video encoder failed: %s", completed.stderr.strip())
        raise SkeletonOverlayVideoError("overlay video encoding failed")


def render_skeleton_overlay_video(
    issue: Mapping[str, Any],
    context: AssetContext,
    output: str | Path,
    start_frame: int,
    end_frame_exclusive: int,
    *,
    on_frame: Callable[[int], None] | None = None,
) -> None:
    """Write one atomic, browser-playable overlay MP4 for a half-open issue window."""

    if not isinstance(issue, Mapping) or not _declares_skeleton_evidence(issue):
        raise SkeletonOverlayVideoError("issue does not declare skeleton evidence")
    if not isinstance(context, AssetContext):
        raise SkeletonOverlayVideoError("invalid asset context")
    if (
        not isinstance(start_frame, int)
        or isinstance(start_frame, bool)
        or not isinstance(end_frame_exclusive, int)
        or isinstance(end_frame_exclusive, bool)
        or start_frame < 0
        or end_frame_exclusive <= start_frame
    ):
        raise SkeletonOverlayVideoError("invalid issue frame window")
    if context.source_range is not None:
        range_start, range_end = context.source_range
        if start_frame < range_start or end_frame_exclusive > range_end:
            raise SkeletonOverlayVideoError("issue frame window exceeds source range")

    video_path = _inside_source(context, "video")
    hdf5_path = _inside_source(context, "hdf5")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    intermediate = target.with_name(f".{target.stem}.{token}.raw.mp4")
    encoded = target.with_name(f".{target.stem}.{token}.tmp{target.suffix}")
    capture = cv2.VideoCapture(str(video_path))
    writer: cv2.VideoWriter | None = None
    try:
        if not capture.isOpened():
            raise SkeletonOverlayVideoError("source video cannot be opened")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            fps = float(context.metadata.get("fps", 0))
        if not np.isfinite(fps) or fps <= 0:
            raise SkeletonOverlayVideoError("source video has invalid frame rate")
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        with h5py.File(hdf5_path, "r") as handle:
            if len(handle["transforms/camera"]) < end_frame_exclusive:
                raise SkeletonOverlayVideoError("HDF5 projection data is too short")
            for frame_index in range(start_frame, end_frame_exclusive):
                readable, image = capture.read()
                if not readable or image is None:
                    raise SkeletonOverlayVideoError("source video frame cannot be read")
                points, valid = _project_frame(handle, frame_index, image.shape)
                _draw_skeleton(image, points, valid)
                if on_frame is not None:
                    on_frame(frame_index)
                if writer is None:
                    height, width = image.shape[:2]
                    writer = cv2.VideoWriter(
                        str(intermediate),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise SkeletonOverlayVideoError("overlay video writer unavailable")
                writer.write(image)
        if writer is None:
            raise SkeletonOverlayVideoError("issue frame window produced no video")
        writer.release()
        writer = None
        _encode_browser_mp4(intermediate, encoded)
        if not encoded.is_file():
            raise SkeletonOverlayVideoError("overlay video encoding produced no file")
        os.replace(encoded, target)
    except SkeletonOverlayVideoError:
        raise
    except Exception as exc:
        LOGGER.exception("Unexpected skeleton overlay rendering failure")
        raise SkeletonOverlayVideoError("overlay video rendering failed") from exc
    finally:
        if writer is not None:
            writer.release()
        capture.release()
        intermediate.unlink(missing_ok=True)
        encoded.unlink(missing_ok=True)


__all__ = ["SkeletonOverlayVideoError", "render_skeleton_overlay_video"]
