#!/usr/bin/env python3
"""Build short video clips and a static HTML review page for review queue items."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only on minimal installs.
    cv2 = None

try:
    import h5py
except ImportError:  # pragma: no cover - exercised only on minimal installs.
    h5py = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_manual_review_queue import (  # noqa: E402
    CONFIDENCE_ENUM,
    FAILURE_MODE_ENUM,
    MANUAL_OUTCOME_ENUM,
    SEVERITY_ENUM,
)


LOGGER = logging.getLogger("build_video_review_clips")
DEFAULT_ASSET_CLIP_SEC = 5.0
DEFAULT_FRAME_STRIDE = 3
DEFAULT_MAX_FRAMES_PER_WINDOW = 120
DEFAULT_JPEG_QUALITY = 90
ROUGH_STORAGE_BYTES_PER_FRAME = 200_000
ACCEPTANCE_STATUS_ENUM = ["accepted", "rejected", "review"]
OVERLAY_MANUAL_OUTCOME_ENUM = [
    "true_positive",
    "false_positive",
    "acceptable_flagged",
    "review",
]
OVERLAY_FAILURE_MODE_ENUM = [
    "severe_keypoint_offset",
    "visual_skeleton_presence_mismatch",
    "implausible_skeleton_pose",
    "hand_out_of_frame",
    "unknown",
]
ACCEPT_REASON_ENUM = [
    "false_alarm",
    "acceptable_side_view_projection",
    "minor_harmless_offset",
]
VIDEO_QUALITY_FAILURE_MODE_ENUM = [
    "video_blur",
    "video_exposure",
    "video_black_screen",
    "video_stutter",
]
SAM3_CONTAINMENT_FAILURE_MODE_ENUM = [
    "strong_containment_mismatch",
    "occlusion_or_mask_undersegmentation",
]
ASSET_LEVEL_FAILURE_MODE_ENUM = [
    "hdf5_text_invalid",
    "quality_hand_low",
    "keypoint_raw_invalid",
    "keypoint_low_quality_window",
]
HAND_JOINT_BASE_NAMES = [
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
]
HAND_JOINT_NAMES = [
    f"{side}{joint_name}"
    for side in ("left", "right")
    for joint_name in HAND_JOINT_BASE_NAMES
]
HAND_SKELETON_EDGES = [
    edge
    for offset in (0, 21)
    for finger_start in (1, 5, 9, 13, 17)
    for edge in (
        (offset, offset + finger_start),
        (offset + finger_start, offset + finger_start + 1),
        (offset + finger_start + 1, offset + finger_start + 2),
        (offset + finger_start + 2, offset + finger_start + 3),
    )
]
VIDEO_MANUAL_LABEL_COLUMNS = [
    "review_id",
    "segment_id",
    "supplier_id",
    "asset_id",
    "window_start_frame",
    "window_end_frame",
    "representative_frame",
    "affected_start_frame",
    "affected_end_frame",
    "auto_verdict",
    "suggested_issue_type",
    "severity_suggestion",
    "key_metrics_json",
    "reason",
    "manual_outcome",
    "failure_mode",
    "severity",
    "confidence",
    "acceptance_status",
    "reviewer",
    "comment",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build short video clips and a static HTML page for manual review."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--review-queue", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--padding-sec", type=float, default=1.0)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--frame-stride", type=int, default=DEFAULT_FRAME_STRIDE)
    parser.add_argument("--max-frames-per-window", type=int, default=DEFAULT_MAX_FRAMES_PER_WINDOW)
    parser.add_argument("--only-review-ids", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--render-overlay", dest="render_overlay", action="store_true")
    parser.add_argument("--no-render-overlay", dest="render_overlay", action="store_false")
    parser.set_defaults(render_overlay=True)
    parser.add_argument("--render-video-clips", dest="render_video_clips", action="store_true")
    parser.add_argument("--no-render-video-clips", dest="render_video_clips", action="store_false")
    parser.set_defaults(render_video_clips=False)
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    rows = build_clip_rows(
        read_manifest(args.manifest),
        read_review_queue(args.review_queue),
        output_dir=args.output_dir,
        padding_sec=args.padding_sec,
        max_items=args.max_items,
        frame_stride=args.frame_stride,
        max_frames_per_window=args.max_frames_per_window,
        only_review_ids=parse_review_ids(args.only_review_ids),
    )
    if args.dry_run:
        sys.stdout.write(json.dumps(estimate_sampled_frames(rows), indent=2, sort_keys=True) + "\n")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_video_clip_policy(rows, render_video_clips=args.render_video_clips)
    if args.render_video_clips:
        render_clips(rows, output_dir=args.output_dir, overwrite=args.overwrite)
    render_sampled_frames(
        rows,
        overwrite=args.overwrite,
        render_overlay=args.render_overlay,
        jpeg_quality=args.jpeg_quality,
    )

    csv_path = args.output_dir / "review_queue_with_clips.csv"
    html_path = args.output_dir / "review_index_video.html"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    html_path.write_text(build_review_index_video_html(rows), encoding="utf-8")
    LOGGER.info("Wrote %s", csv_path)
    LOGGER.info("Wrote %s", html_path)
    return 0


def read_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"supplier_id", "asset_id", "video_path"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"manifest missing required columns: {missing}")
    return normalize_dataframe(df)


def read_review_queue(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"supplier_id", "asset_id", "review_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"review queue missing required columns: {missing}")
    return normalize_dataframe(df)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df.where(pd.notna(df), "")


def parse_review_ids(value: str) -> set[str]:
    return {item.strip() for item in str(value or "").split(",") if item.strip()}


def build_clip_rows(
    manifest_df: pd.DataFrame,
    review_df: pd.DataFrame,
    *,
    output_dir: Path,
    padding_sec: float,
    max_items: int | None = None,
    frame_stride: int = DEFAULT_FRAME_STRIDE,
    max_frames_per_window: int = DEFAULT_MAX_FRAMES_PER_WINDOW,
    only_review_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    if only_review_ids:
        review_df = review_df[review_df["review_id"].astype(str).isin(only_review_ids)]
    if max_items is not None:
        review_df = review_df.head(max_items)
    manifest_columns = [
        column
        for column in (
            "supplier_id",
            "asset_id",
            "video_path",
            "hdf5_path",
            "hdf5",
            "fps",
            "frame_count",
            "duration_sec",
        )
        if column in manifest_df.columns
    ]
    merged = review_df.merge(
        manifest_df[manifest_columns],
        on=["supplier_id", "asset_id"],
        how="left",
        suffixes=("", "_manifest"),
    )
    rows: list[dict[str, Any]] = []
    clips_dir = output_dir / "clips"
    frames_dir = output_dir / "frames"
    for index, row in enumerate(merged.to_dict(orient="records"), start=1):
        output = dict(row)
        if not output.get("hdf5_path") and output.get("hdf5"):
            output["hdf5_path"] = output.get("hdf5")
        review_id = str(output.get("review_id") or f"review_{index:04d}")
        fps = resolve_fps(output)
        duration_sec = float_or_none(output.get("duration_sec"))
        window_start = int_or_none(output.get("window_start_frame"))
        window_end = int_or_none(output.get("window_end_frame"))
        timing = compute_clip_timing(
            window_start_frame=window_start,
            window_end_frame=window_end,
            representative_frame=int_or_none(output.get("representative_frame")),
            fps=fps,
            asset_duration_sec=duration_sec,
            padding_sec=padding_sec,
        )
        sampled_frames = sampled_frame_indices(
            window_start,
            window_end,
            frame_stride=frame_stride,
            max_frames_per_window=max_frames_per_window,
        )
        safe_review_id = safe_filename(review_id)
        sampled_frame_rows = [
            {
                "sampled_index": sampled_index,
                "frame_idx": frame_idx,
                "display_frame_path": f"frames/{safe_review_id}/frame_{frame_idx:06d}.jpg",
                "frame_path": str(frames_dir / safe_review_id / f"frame_{frame_idx:06d}.jpg"),
            }
            for sampled_index, frame_idx in enumerate(sampled_frames, start=1)
        ]
        display_clip_path = f"clips/{safe_filename(review_id)}.mp4"
        output["clip_path"] = str(clips_dir / f"{safe_filename(review_id)}.mp4")
        output["display_clip_path"] = display_clip_path
        output["clip_start_time_sec"] = round(timing["start_time_sec"], 6)
        output["clip_duration_sec"] = round(timing["duration_sec"], 6)
        output["clip_error"] = timing.get("error", "")
        output["fps"] = round(fps, 6) if fps is not None else output.get("fps", "")
        output["frame_stride"] = max(1, int(frame_stride))
        output["max_frames_per_window"] = max(1, int(max_frames_per_window))
        output["review_run_label"] = safe_filename(output_dir.name or "review")
        output["sampled_frame_count"] = len(sampled_frame_rows)
        output["sampled_frames_json"] = json.dumps(sampled_frame_rows, ensure_ascii=False)
        output["sampled_frame_error"] = ""
        rows.append(output)
    return rows


def apply_video_clip_policy(rows: list[dict[str, Any]], *, render_video_clips: bool) -> None:
    if render_video_clips:
        return
    for row in rows:
        row["clip_path"] = ""
        row["display_clip_path"] = ""
        row["clip_error"] = ""


def compute_clip_timing(
    *,
    window_start_frame: int | None,
    window_end_frame: int | None,
    representative_frame: int | None,
    fps: float | None,
    asset_duration_sec: float | None,
    padding_sec: float,
) -> dict[str, Any]:
    padding_sec = max(0.0, padding_sec)
    if window_start_frame is not None and window_end_frame is not None:
        if fps is None or fps <= 0:
            return {"start_time_sec": 0.0, "duration_sec": 0.0, "error": "missing_or_invalid_fps"}
        start_frame = min(window_start_frame, window_end_frame)
        end_frame = max(window_start_frame, window_end_frame)
        start_time = max(0.0, start_frame / fps - padding_sec)
        duration = (end_frame - start_frame + 1) / fps + (2.0 * padding_sec)
        return clamp_timing(start_time, duration, asset_duration_sec)

    if representative_frame is not None and fps is not None and fps > 0:
        center = representative_frame / fps
        start_time = max(0.0, center - (DEFAULT_ASSET_CLIP_SEC / 2.0))
    else:
        start_time = 0.0
    return clamp_timing(start_time, DEFAULT_ASSET_CLIP_SEC, asset_duration_sec)


def clamp_timing(
    start_time_sec: float,
    duration_sec: float,
    asset_duration_sec: float | None,
) -> dict[str, Any]:
    start_time_sec = max(0.0, start_time_sec)
    duration_sec = max(0.0, duration_sec)
    if asset_duration_sec is not None and asset_duration_sec > 0:
        if start_time_sec >= asset_duration_sec:
            start_time_sec = max(0.0, asset_duration_sec - min(DEFAULT_ASSET_CLIP_SEC, asset_duration_sec))
        duration_sec = min(duration_sec, max(0.0, asset_duration_sec - start_time_sec))
    return {"start_time_sec": start_time_sec, "duration_sec": duration_sec, "error": ""}


def sampled_frame_indices(
    window_start_frame: int | None,
    window_end_frame: int | None,
    *,
    frame_stride: int,
    max_frames_per_window: int,
) -> list[int]:
    if window_start_frame is None or window_end_frame is None:
        return []
    start_frame = min(window_start_frame, window_end_frame)
    end_frame = max(window_start_frame, window_end_frame)
    stride = max(1, int(frame_stride))
    cap = max(1, int(max_frames_per_window))
    frames = list(range(start_frame, end_frame + 1, stride))
    if frames and frames[-1] != end_frame and len(frames) < cap:
        frames.append(end_frame)
    return frames[:cap]


def estimate_sampled_frames(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sampled_frames = sum(int(row.get("sampled_frame_count") or 0) for row in rows)
    rough_storage_bytes = sampled_frames * ROUGH_STORAGE_BYTES_PER_FRAME
    return {
        "review_items": len(rows),
        "sampled_frames": sampled_frames,
        "rough_storage_bytes": rough_storage_bytes,
        "rough_storage_mb": round(rough_storage_bytes / 1_000_000, 2),
    }


def render_clips(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    overwrite: bool,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    for row in rows:
        if row.get("clip_error"):
            continue
        video_path = row.get("video_path")
        if not video_path:
            row["clip_error"] = "missing_video_path"
            continue
        source = Path(str(video_path))
        if not source.exists():
            row["clip_error"] = "video_path_not_found"
            continue
        target = Path(str(row["clip_path"]))
        if target.exists() and not overwrite:
            continue
        if ffmpeg is None:
            row["clip_error"] = "ffmpeg_not_found"
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        result = run_ffmpeg_clip(
            ffmpeg=ffmpeg,
            source=source,
            target=target,
            start_time_sec=float(row["clip_start_time_sec"]),
            duration_sec=float(row["clip_duration_sec"]),
            overwrite=overwrite,
        )
        if result:
            row["clip_error"] = result
            LOGGER.warning("Failed to write clip for %s: %s", row.get("review_id"), result)


def render_sampled_frames(
    rows: list[dict[str, Any]],
    *,
    overwrite: bool,
    render_overlay: bool = True,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> None:
    if cv2 is None:
        for row in rows:
            if parse_sampled_frames(row):
                row["sampled_frame_error"] = "opencv_not_installed"
        return
    jpeg_quality = min(max(int(jpeg_quality), 1), 100)
    for row in rows:
        frame_rows = parse_sampled_frames(row)
        if not frame_rows:
            continue
        video_path = row.get("video_path")
        if not video_path:
            row["sampled_frame_error"] = "missing_video_path"
            continue
        source = Path(str(video_path))
        if not source.exists():
            row["sampled_frame_error"] = "video_path_not_found"
            continue
        errors = []
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            row["sampled_frame_error"] = "video_open_failed"
            continue
        h5 = None
        hdf5_error = ""
        hdf5_path = str(row.get("hdf5_path") or "")
        if render_overlay:
            if h5py is None:
                hdf5_error = "h5py_not_installed"
            elif not hdf5_path:
                hdf5_error = "missing_hdf5_path"
            elif not Path(hdf5_path).exists():
                hdf5_error = "hdf5_path_not_found"
            else:
                try:
                    h5 = h5py.File(hdf5_path, "r")
                except OSError as exc:
                    hdf5_error = f"hdf5_open_error: {exc}"
        for frame in frame_rows:
            target = Path(str(frame["frame_path"]))
            if target.exists() and not overwrite:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            frame_idx = int(frame["frame_idx"])
            image, read_error = read_video_frame(capture, frame_idx)
            if read_error:
                image = make_error_frame(read_error)
                errors.append(f"{frame_idx}: {read_error}")
            elif render_overlay:
                if h5 is None:
                    draw_error_text(image, hdf5_error or "projection_unavailable")
                    if hdf5_error:
                        errors.append(f"{frame_idx}: {hdf5_error}")
                else:
                    projection_error = render_overlay_frame(image, h5, frame_idx, row)
                    if projection_error:
                        errors.append(f"{frame_idx}: {projection_error}")
            if not cv2.imwrite(
                str(target),
                image,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
            ):
                errors.append(f"{frame_idx}: jpeg_write_failed")
        capture.release()
        if h5 is not None:
            h5.close()
        if errors:
            row["sampled_frame_error"] = "; ".join(errors[:3])
            LOGGER.warning(
                "Failed to write sampled frames for %s: %s",
                row.get("review_id"),
                row["sampled_frame_error"],
            )


def read_video_frame(capture: Any, frame_idx: int) -> tuple[Any, str]:
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
    ok, image = capture.read()
    if not ok or image is None:
        return None, "video_frame_read_failed"
    return image, ""


def make_error_frame(message: str, *, width: int = 640, height: int = 360) -> Any:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    draw_error_text(image, message)
    return image


def draw_error_text(image: Any, message: str) -> None:
    if cv2 is None:
        return
    cv2.putText(
        image,
        f"projection error: {message}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def render_overlay_frame(image: Any, h5: Any, frame_idx: int, row: dict[str, Any]) -> str:
    try:
        uv, valid, quality_hand, status = project_hdf5_frame(h5, frame_idx, image.shape)
    except (KeyError, IndexError, ValueError, np.linalg.LinAlgError) as exc:
        message = f"projection_failed: {exc}"
        draw_error_text(image, message)
        return message
    draw_hand_overlay(image, uv, valid)
    draw_overlay_metadata(image, row, frame_idx, quality_hand, status, int(valid.sum()))
    return ""


def project_hdf5_frame(
    h5: Any,
    frame_idx: int,
    image_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, str]:
    camera_to_world = np.asarray(h5["transforms/camera"][frame_idx : frame_idx + 1], dtype=float)
    intrinsic = np.asarray(h5["camera/intrinsic"], dtype=float)
    positions = []
    for joint_name in HAND_JOINT_NAMES:
        positions.append(np.asarray(h5[f"transforms/{joint_name}"][frame_idx, :3, 3], dtype=float))
    keypoints = np.stack(positions, axis=0)[None, :, :]
    uv, _z, valid = project_world_keypoints_to_image(
        keypoints,
        camera_to_world,
        intrinsic,
        image_shape=image_shape,
    )
    quality_hand = None
    if "label/quality_hand" in h5:
        quality_hand = np.asarray(h5["label/quality_hand"][frame_idx], dtype=float)
    return uv[0], valid[0], quality_hand, "projected"


def project_world_keypoints_to_image(
    positions_world: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsic: np.ndarray,
    *,
    image_shape: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(positions_world, dtype=float)
    camera = np.asarray(camera_to_world, dtype=float)
    intrinsic = np.asarray(intrinsic, dtype=float)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions_world must have shape (T, J, 3)")
    if camera.ndim != 3 or camera.shape[-2:] != (4, 4):
        raise ValueError("camera_to_world must have shape (T, 4, 4)")
    if intrinsic.shape != (3, 3):
        raise ValueError("camera/intrinsic must have shape (3, 3)")
    if camera.shape[0] != positions.shape[0]:
        raise ValueError("positions and transforms/camera frame counts differ")

    camera_inv = np.linalg.inv(camera)
    homogeneous = np.concatenate(
        [positions, np.ones((*positions.shape[:2], 1), dtype=float)],
        axis=-1,
    )
    points_camera = np.einsum("tij,tkj->tki", camera_inv, homogeneous)[..., :3]
    z = points_camera[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u_px = intrinsic[0, 0] * points_camera[..., 0] / z + intrinsic[0, 2]
        v_px = intrinsic[1, 1] * points_camera[..., 1] / z + intrinsic[1, 2]
    uv = np.stack([u_px, v_px], axis=-1)
    valid = (
        np.isfinite(uv).all(axis=-1)
        & np.isfinite(z)
        & (z > 0)
    )
    if image_shape is not None:
        height, width = int(image_shape[0]), int(image_shape[1])
        valid &= (
            (uv[..., 0] >= 0)
            & (uv[..., 0] < width)
            & (uv[..., 1] >= 0)
            & (uv[..., 1] < height)
        )
    return uv, z, valid


def draw_hand_overlay(image: Any, uv: np.ndarray, valid: np.ndarray) -> None:
    left_color = (0, 220, 0)
    right_color = (0, 140, 255)
    for start, end in HAND_SKELETON_EDGES:
        if valid[start] and valid[end]:
            color = left_color if start < 21 else right_color
            p1 = tuple(int(round(value)) for value in uv[start])
            p2 = tuple(int(round(value)) for value in uv[end])
            cv2.line(image, p1, p2, color, 2, cv2.LINE_AA)
    for joint_index, point in enumerate(uv):
        if not valid[joint_index]:
            continue
        color = left_color if joint_index < 21 else right_color
        center = tuple(int(round(value)) for value in point)
        marker_radius = 4 if joint_index in (0, 21) else 3
        cv2.circle(image, center, marker_radius, color, -1, cv2.LINE_AA)
        cv2.circle(image, center, marker_radius + 1, (255, 255, 255), 1, cv2.LINE_AA)


def draw_overlay_metadata(
    image: Any,
    row: dict[str, Any],
    frame_idx: int,
    quality_hand: np.ndarray | None,
    status: str,
    visible_keypoints: int,
) -> None:
    if quality_hand is None or len(quality_hand) < 2:
        quality_text = "quality_hand L/R: n/a"
    else:
        quality_text = f"quality_hand L/R: {quality_hand[0]:.3f}/{quality_hand[1]:.3f}"
    lines = [
        f"asset_id: {row.get('asset_id', '')} frame_idx: {frame_idx}",
        f"candidate window: {row.get('window_start_frame', '')}-{row.get('window_end_frame', '')} stride: {row.get('frame_stride', '')}",
        quality_text,
        f"issue: {row.get('suggested_issue_type', '')}",
        f"projection: {status}; visible keypoints: {visible_keypoints}/42",
    ]
    y = 24
    for line in lines:
        cv2.putText(
            image,
            str(line),
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(line),
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 18


def run_ffmpeg_frame_extract(
    *,
    ffmpeg: str,
    source: Path,
    target: Path,
    frame_idx: int,
    overwrite: bool,
) -> str:
    cmd = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vf",
        f"select=eq(n\\,{frame_idx})",
        "-frames:v",
        "1",
        "-q:v",
        "3",
        str(target),
    ]
    try:
        completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
    except OSError as exc:
        return f"ffmpeg_error: {exc}"
    if completed.returncode != 0:
        return completed.stderr.strip() or f"ffmpeg_exit_{completed.returncode}"
    return ""


def run_ffmpeg_clip(
    *,
    ffmpeg: str,
    source: Path,
    target: Path,
    start_time_sec: float,
    duration_sec: float,
    overwrite: bool,
) -> str:
    if duration_sec <= 0:
        return "non_positive_clip_duration"
    cmd = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-loglevel",
        "error",
        "-ss",
        f"{start_time_sec:.6f}",
        "-i",
        str(source),
        "-t",
        f"{duration_sec:.6f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(target),
    ]
    try:
        completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
    except OSError as exc:
        return f"ffmpeg_error: {exc}"
    if completed.returncode != 0:
        return completed.stderr.strip() or f"ffmpeg_exit_{completed.returncode}"
    return ""


def build_review_index_video_html(rows: list[dict[str, Any]]) -> str:
    rows_json = json.dumps(json_safe(rows), ensure_ascii=False).replace("</", "<\\/")
    storage_key = build_storage_key(rows)
    enum_json = json.dumps(build_review_ui_enums(rows))
    manual_columns_json = json.dumps(VIDEO_MANUAL_LABEL_COLUMNS)
    render_video_clips = any(row.get("display_clip_path") and not row.get("clip_error") for row in rows)
    clip_js = (
        "const clip=row.display_clip_path && !row.clip_error ? "
        "`<video id=\"video-${index}\" class=\"video\" controls src=\"${escapeHtml(row.display_clip_path)}\"></video>` : '';"
        if render_video_clips
        else "const clip='';"
    )
    return (
        "<!doctype html>\n"
        "<html><head><meta charset=\"utf-8\">\n"
        "<title>Video Manual Review</title>\n"
        "<style>\n"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;line-height:1.35;color:#202124;background:#fff}\n"
        ".toolbar{position:sticky;top:0;background:#fff;border-bottom:1px solid #dadce0;padding:12px 0;margin-bottom:16px;z-index:2}.toolbar label{font-size:12px;color:#5f6368;margin-right:8px}.toolbar input{padding:6px;border:1px solid #c7cdd4;border-radius:4px}.storage-key{font-size:11px;color:#6b7280;margin-top:6px}\n"
        "button{margin-right:8px;padding:7px 10px;border:1px solid #c7cdd4;background:#f8f9fa;border-radius:4px;cursor:pointer}button.primary{background:#1a73e8;color:white;border-color:#1a73e8}\n"
        ".item{border:1px solid #dadce0;border-radius:6px;margin:14px 0;padding:12px}.grid{display:grid;grid-template-columns:minmax(280px,420px) 1fr;gap:14px}.video{width:100%;max-height:280px;background:#111}.no-clip{height:160px;border:1px dashed #c7cdd4;color:#6b7280;display:flex;align-items:center;justify-content:center}\n"
        ".frame-viewer{margin-top:10px}.frame-viewer img{width:100%;max-height:280px;object-fit:contain;background:#111;border:1px solid #dadce0}.frame-meta{font-size:13px;color:#3c4043;margin:6px 0}.warning{background:#fff7e6;border:1px solid #fbbc04;border-radius:4px;padding:8px;margin:8px 0;font-size:13px}.viewer-controls{display:flex;gap:6px;margin:6px 0}\n"
        ".meta{display:grid;grid-template-columns:repeat(3,minmax(120px,1fr));gap:6px 12px;font-size:13px}.meta b{display:block;color:#5f6368;font-size:12px}.auto,.human{border:1px solid #eceff1;border-radius:6px;padding:10px;margin-top:10px}.auto{background:#fbfcfe}.human{background:#fffdf8}.title{font-weight:700;margin-bottom:8px}.metrics,.reason{white-space:pre-wrap;background:#f8f9fa;border:1px solid #eceff1;padding:8px;margin-top:8px;font-size:12px;overflow:auto}.controls{display:grid;grid-template-columns:repeat(3,minmax(140px,1fr));gap:8px}.controls label{font-size:12px;color:#5f6368}.controls select,.controls input,.controls textarea{width:100%;box-sizing:border-box;margin-top:3px;padding:6px;border:1px solid #c7cdd4;border-radius:4px;font:inherit}.controls textarea{min-height:58px;grid-column:span 2}.status{margin-left:8px;color:#188038;font-size:13px}\n"
        ".segment-toolbar{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}.segment{border:1px solid #e0e3e7;border-radius:6px;margin-top:8px;padding:8px;background:#fff}.segment-grid{display:grid;grid-template-columns:repeat(4,minmax(120px,1fr));gap:8px}.segment-grid label{font-size:12px;color:#5f6368}.segment-grid input,.segment-grid select,.segment-grid textarea{width:100%;box-sizing:border-box;margin-top:3px;padding:6px;border:1px solid #c7cdd4;border-radius:4px;font:inherit}.segment-grid textarea{min-height:48px;grid-column:span 2}.frame-buttons{display:grid;grid-template-columns:minmax(80px,1fr) auto;gap:4px;align-items:end}.frame-buttons button{margin-right:0;padding:4px 6px;white-space:nowrap}.empty{color:#6b7280;font-size:13px;padding:8px;border:1px dashed #dadce0;border-radius:4px}.segment-invalid{border-color:#d93025!important;background:#fff4f4}.validation-error{color:#d93025;font-size:12px;margin-top:6px}.help{background:#eef5ff;border:1px solid #c7d7fe;border-radius:4px;padding:8px;margin:8px 0;font-size:13px}.micro-help{font-size:12px;color:#5f6368;margin:4px 0}\n"
        "</style></head><body>\n"
        "<h1>Video Manual Review</h1>\n"
        "<div class=\"help\">This page shows HDF5 skeleton projection overlay only. SAM3 mask containment has not been run unless a SAM3 containment input was provided upstream.</div>\n"
        "<div class=\"toolbar\"><label>Reviewer <input id=\"global-reviewer\" value=\"nathan\"></label><button class=\"primary\" onclick=\"exportManualLabelsCsv()\">Export manual_labels.csv</button><button onclick=\"saveToServer('explicit_save')\">Save to cloud</button><button onclick=\"saveProgress()\">Save progress to localStorage</button><button onclick=\"loadProgress()\">Load progress from localStorage</button><button onclick=\"exportProgressJson()\">Export progress JSON</button><button onclick=\"document.getElementById('progress-json-input').click()\">Import progress JSON</button><input id=\"progress-json-input\" type=\"file\" accept=\"application/json,.json\" style=\"display:none\" onchange=\"importProgressJson(event)\"><button onclick=\"clearProgress()\">Clear local saved progress</button><span id=\"status\" class=\"status\"></span><div class=\"storage-key\">localStorage key: <code id=\"storage-key\"></code></div></div>\n"
        "<div id=\"root\"></div>\n"
        "<script>\n"
        f"const REVIEW_ROWS = {rows_json};\n"
        f"const ENUMS = {enum_json};\n"
        f"const MANUAL_COLUMNS = {manual_columns_json};\n"
        f"const STORAGE_KEY='{storage_key}';\n"
        "const SERVER_SAVE_ENDPOINT='/api/manual-review/save';\n"
        "let segmentsByReviewId = {};\n"
        "let restoreInProgress = false;\n"
        "let serverAutosaveTimer = null;\n"
        "function escapeHtml(value){return String(value ?? '').replace(/[&<>\"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;'}[ch]));}\n"
        "function rowKey(row){return String(row.review_id ?? '');}\n"
        "function globalReviewer(){const el=document.getElementById('global-reviewer'); return el ? el.value : '';}\n"
        "function segmentFieldId(rowIndex,segmentIndex,field){return `seg-${rowIndex}-${segmentIndex}-${field}`;}\n"
        "function optionHtml(values,selected){return values.map(v=>`<option value=\"${escapeHtml(v)}\" ${v===selected?'selected':''}>${escapeHtml(v)}</option>`).join('');}\n"
        "function defaultFailureMode(row){return ENUMS.failure_mode.includes(row.suggested_issue_type) ? row.suggested_issue_type : 'unknown';}\n"
        "function defaultSeverity(row){return ENUMS.severity.includes(row.severity_suggestion) ? row.severity_suggestion : 'medium';}\n"
        "function sampledFrames(row){try{return JSON.parse(row.sampled_frames_json || '[]');}catch(_err){return [];}}\n"
        "let activeReviewIndex = 0;\n"
        "let sampledFrameIndexByReviewId = {};\n"
        "function activeSampledIndex(row){const frames=sampledFrames(row); const key=rowKey(row); const current=sampledFrameIndexByReviewId[key] ?? 0; return Math.min(Math.max(current,0), Math.max(frames.length-1,0));}\n"
        "function overlayViewerHtml(row,index){const frames=sampledFrames(row); const warning=`<div class=\"warning\">This page samples every N frames. Current sampling stride: every ${escapeHtml(row.frame_stride)} frames. For single-frame issues, regenerate this review_id with frame_stride=1.</div>`; if(!frames.length){return warning + '<div class=\"no-clip\">No sampled overlay frames</div>';} const current=activeSampledIndex(row); const frame=frames[current]; return `${warning}<div class=\"frame-viewer\" aria-label=\"Sampled overlay frame carousel\" onclick=\"activeReviewIndex=${index}\"><div class=\"frame-meta\"><b>Sampled overlay frame carousel</b></div><img id=\"sampled-frame-img-${index}\" src=\"${escapeHtml(frame.display_frame_path)}\"><div id=\"sampled-frame-meta-${index}\" class=\"frame-meta\">current sampled frame index ${current+1} / ${frames.length}; original frame_idx ${escapeHtml(frame.frame_idx)}; candidate window start/end ${escapeHtml(row.window_start_frame)}-${escapeHtml(row.window_end_frame)}; sampling stride ${escapeHtml(row.frame_stride)}</div><div class=\"viewer-controls\"><button onclick=\"moveSampledFrame(${index},-1)\">Previous sampled frame</button><button onclick=\"moveSampledFrame(${index},1)\">Next sampled frame</button></div><div class=\"viewer-controls\"><label>Go to original frame <input id=\"sampled-frame-jump-${index}\" type=\"number\"></label><button onclick=\"jumpToOriginalFrame(${index})\">Go</button><span id=\"sampled-frame-jump-note-${index}\" class=\"frame-meta\"></span></div></div>`;}\n"
        "function moveSampledFrame(rowIndex,delta){const row=REVIEW_ROWS[rowIndex]; const frames=sampledFrames(row); if(!frames.length) return; const key=rowKey(row); const current=activeSampledIndex(row); sampledFrameIndexByReviewId[key]=((current+delta)%frames.length+frames.length)%frames.length; activeReviewIndex=rowIndex; updateSampledFrame(rowIndex); autosaveProgress();}\n"
        "function updateSampledFrame(rowIndex){const row=REVIEW_ROWS[rowIndex]; const frames=sampledFrames(row); if(!frames.length) return; const current=activeSampledIndex(row); const frame=frames[current]; const img=document.getElementById(`sampled-frame-img-${rowIndex}`); const meta=document.getElementById(`sampled-frame-meta-${rowIndex}`); if(img) img.src=frame.display_frame_path; if(meta) meta.textContent=`current sampled frame index ${current+1} / ${frames.length}; original frame_idx ${frame.frame_idx}; candidate window start/end ${row.window_start_frame}-${row.window_end_frame}; sampling stride ${row.frame_stride}`;}\n"
        "function nearestSampledFrameIndex(frames, targetFrame){let bestIndex=0; let bestDistance=Infinity; frames.forEach((frame,index)=>{const distance=Math.abs(Number(frame.frame_idx)-targetFrame); if(distance < bestDistance){bestDistance=distance; bestIndex=index;}}); return bestIndex;}\n"
        "function jumpToOriginalFrame(rowIndex){const row=REVIEW_ROWS[rowIndex]; const frames=sampledFrames(row); const input=document.getElementById(`sampled-frame-jump-${rowIndex}`); const note=document.getElementById(`sampled-frame-jump-note-${rowIndex}`); if(!frames.length || !input) return; const target=Number(input.value); if(!Number.isFinite(target)){if(note) note.textContent='Enter a numeric original frame'; return;} const index=nearestSampledFrameIndex(frames,target); const selected=Number(frames[index].frame_idx); sampledFrameIndexByReviewId[rowKey(row)]=index; activeReviewIndex=rowIndex; updateSampledFrame(rowIndex); if(note){const min=Number(frames[0].frame_idx); const max=Number(frames[frames.length-1].frame_idx); note.textContent=(target < min || target > max) ? `clamped to nearest sampled frame ${selected}` : `nearest sampled frame ${selected}`;} autosaveProgress();}\n"
        "function handleKeydown(event){if(event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return; const step=event.shiftKey ? 10 : 1; moveSampledFrame(activeReviewIndex,event.key === 'ArrowRight' ? step : -step); event.preventDefault();}\n"
        "document.addEventListener('keydown', handleKeydown);\n"
        "function nextSegmentId(row){const key=rowKey(row); const count=(segmentsByReviewId[key] || []).length + 1; return `${key}_seg_${String(count).padStart(3,'0')}`;}\n"
        "function makeSegment(row,overrides={}){return {segment_id:overrides.segment_id || nextSegmentId(row),affected_start_frame:overrides.affected_start_frame ?? '',affected_end_frame:overrides.affected_end_frame ?? '',manual_outcome:overrides.manual_outcome || 'true_positive',failure_mode:overrides.failure_mode || defaultFailureMode(row),severity:overrides.severity || 'high',confidence:overrides.confidence || 'medium',acceptance_status:overrides.acceptance_status || 'rejected',reviewer:overrides.reviewer || '',comment:overrides.comment || ''};}\n"
        "function render(){const root=document.getElementById('root'); let html=''; REVIEW_ROWS.forEach((row,index)=>{__CLIP_JS__ html+=`<article class=\"item\" onclick=\"activeReviewIndex=${index}\"><div class=\"grid\"><div>${overlayViewerHtml(row,index)}${clip}</div><div><div class=\"meta\"><div><b>review_id</b>${escapeHtml(row.review_id)}</div><div><b>supplier_id</b>${escapeHtml(row.supplier_id)}</div><div><b>asset_id</b>${escapeHtml(row.asset_id)}</div><div><b>window</b>${escapeHtml(row.window_start_frame)}-${escapeHtml(row.window_end_frame)}</div><div><b>representative_frame</b>${escapeHtml(row.representative_frame)}</div><div><b>fps</b>${escapeHtml(row.fps)}</div><div><b>frame_stride</b>${escapeHtml(row.frame_stride)}</div><div><b>sampled_frame_count</b>${escapeHtml(row.sampled_frame_count)}</div><div><b>clip_start_time_sec</b>${escapeHtml(row.clip_start_time_sec)}</div><div><b>clip_duration_sec</b>${escapeHtml(row.clip_duration_sec)}</div></div><section class=\"auto\"><div class=\"title\">Auto result</div><div class=\"meta\"><div><b>suggested_issue_type</b>${escapeHtml(row.suggested_issue_type)}</div><div><b>auto_verdict</b>${escapeHtml(row.auto_verdict)}</div><div><b>severity_suggestion</b>${escapeHtml(row.severity_suggestion)}</div></div><div class=\"metrics\"><b>key_metrics_json</b>\\n${escapeHtml(row.key_metrics_json)}</div><div class=\"reason\"><b>reason</b>\\n${escapeHtml(row.reason)}</div></section><section class=\"human\"><div class=\"title\">Human affected segments</div><p class=\"empty\">Manual affected_start_frame/end_frame are original frame numbers, not sampled indexes. Use affected segments for rejected duration. Raw candidate windows are evidence only; uncovered frames are implicitly acceptable.</p><div class=\"help\">False positive = script flagged this window but human confirms there is no real issue.<br>Acceptable = flagged phenomenon exists but should not reduce usable duration.<br>Accept whole window = accepted, low severity, medium confidence; false_alarm exports false_positive, other accept reasons export acceptable_flagged.<br>Add rejected segment = true_positive, rejected, high severity, medium confidence.<br>Partial bad windows are represented by adding one rejected segment per affected range.<br>failure_mode options are limited to HDF5 skeleton projection review.<br>severe_keypoint_offset = projected skeleton/keypoints are clearly far from the hand.<br>visual_skeleton_presence_mismatch = projected skeleton presence visibly disagrees with the hand.<br>implausible_skeleton_pose = projected pose is anatomically implausible or follows the wrong structure.<br>hand_out_of_frame = hand is outside or partly outside the image.<br>unknown = issue is visible but does not fit another listed mode.</div><div class=\"segment-toolbar\"><label>accept_reason<select id=\"accept-reason-${index}\">${optionHtml(ENUMS.accept_reason,'false_alarm')}</select></label><button onclick=\"acceptWholeWindow(${index})\">Accept whole window</button><button onclick=\"addRejectedSegment(${index})\">Add rejected segment</button></div><div class=\"micro-help\">Set start/end use current overlay original frame_idx.</div><div id=\"segments-${index}\"></div></section></div></div></article>`;}); root.innerHTML=html; REVIEW_ROWS.forEach((_row,index)=>renderSegments(index));}\n"
        "function acceptReason(rowIndex){const el=document.getElementById(`accept-reason-${rowIndex}`); return el ? el.value : 'false_alarm';}\n"
        "function acceptWholeWindow(rowIndex){const row=REVIEW_ROWS[rowIndex]; const key=rowKey(row); const reason=acceptReason(rowIndex); if(reason === 'false_alarm'){segmentsByReviewId[key]=[makeSegment(row,{segment_id:`${key}_accepted`,affected_start_frame:'',affected_end_frame:'',manual_outcome:'false_positive',failure_mode:'unknown',severity:'low',confidence:'medium',acceptance_status:'accepted',comment:`accept_reason=${reason}`})];} else {segmentsByReviewId[key]=[makeSegment(row,{segment_id:`${key}_accepted`,affected_start_frame:'',affected_end_frame:'',manual_outcome:'acceptable_flagged',failure_mode:'unknown',severity:'low',confidence:'medium',acceptance_status:'accepted',comment:`accept_reason=${reason}`})];} renderSegments(rowIndex); autosaveProgress(); setStatus(`Accepted whole window: ${reason}`);}\n"
        "function addRejectedSegment(rowIndex){const row=REVIEW_ROWS[rowIndex]; const key=rowKey(row); if(!segmentsByReviewId[key]) segmentsByReviewId[key]=[]; segmentsByReviewId[key].push(makeSegment(row,{manual_outcome:'true_positive',failure_mode:'severe_keypoint_offset',severity:'high',confidence:'medium',acceptance_status:'rejected'})); renderSegments(rowIndex); autosaveProgress(); setStatus(`Rejected segments: ${segmentsByReviewId[key].length}`);}\n"
        "function deleteSegment(rowIndex,segmentIndex){const row=REVIEW_ROWS[rowIndex]; const key=rowKey(row); (segmentsByReviewId[key] || []).splice(segmentIndex,1); renderSegments(rowIndex); autosaveProgress();}\n"
        "function renderSegments(rowIndex){const row=REVIEW_ROWS[rowIndex]; const key=rowKey(row); const segments=segmentsByReviewId[key] || []; const target=document.getElementById(`segments-${rowIndex}`); if(!target) return; if(!segments.length){target.innerHTML='<div class=\"empty\">No affected segments yet.</div>'; return;} target.innerHTML=segments.map((segment,segmentIndex)=>segmentHtml(rowIndex,segmentIndex,segment)).join('');}\n"
        "function segmentHtml(rowIndex,segmentIndex,segment){const compact=(segment.acceptance_status === 'accepted') || (segment.acceptance_status === 'review' && !segment.affected_start_frame && !segment.affected_end_frame); if(compact){return `<div class=\"segment\"><div class=\"segment-grid\"><label>segment_id<input id=\"${segmentFieldId(rowIndex,segmentIndex,'segment_id')}\" value=\"${escapeHtml(segment.segment_id)}\"></label><div><b>decision</b><br>${escapeHtml(segment.manual_outcome)} / ${escapeHtml(segment.acceptance_status)}</div><label>comment<textarea id=\"${segmentFieldId(rowIndex,segmentIndex,'comment')}\">${escapeHtml(segment.comment)}</textarea></label><div><button onclick=\"deleteSegment(${rowIndex},${segmentIndex})\">Delete segment</button></div></div><div id=\"${segmentFieldId(rowIndex,segmentIndex,'validation_error')}\" class=\"validation-error\"></div></div>`;} return `<div class=\"segment\"><div class=\"segment-grid\"><label>segment_id<input id=\"${segmentFieldId(rowIndex,segmentIndex,'segment_id')}\" value=\"${escapeHtml(segment.segment_id)}\"></label><label>affected_start_frame<div class=\"frame-buttons\"><input id=\"${segmentFieldId(rowIndex,segmentIndex,'affected_start_frame')}\" value=\"${escapeHtml(segment.affected_start_frame)}\"><button title=\"Set start = current overlay frame\" onclick=\"setSegmentFrame(${rowIndex},${segmentIndex},'affected_start_frame')\">Set start</button></div></label><label>affected_end_frame<div class=\"frame-buttons\"><input id=\"${segmentFieldId(rowIndex,segmentIndex,'affected_end_frame')}\" value=\"${escapeHtml(segment.affected_end_frame)}\"><button title=\"Set end = current overlay frame\" onclick=\"setSegmentFrame(${rowIndex},${segmentIndex},'affected_end_frame')\">Set end</button></div></label><label>failure_mode<select id=\"${segmentFieldId(rowIndex,segmentIndex,'failure_mode')}\">${optionHtml(ENUMS.failure_mode,segment.failure_mode)}</select></label><label>comment<textarea id=\"${segmentFieldId(rowIndex,segmentIndex,'comment')}\">${escapeHtml(segment.comment)}</textarea></label><div><button onclick=\"deleteSegment(${rowIndex},${segmentIndex})\">Delete segment</button></div></div><div id=\"${segmentFieldId(rowIndex,segmentIndex,'validation_error')}\" class=\"validation-error\"></div></div>`;}\n"
        "function currentOverlayFrame(rowIndex){const row=REVIEW_ROWS[rowIndex]; const frames=sampledFrames(row); if(!frames.length) return ''; const current=activeSampledIndex(row); const frame=frames[current]; const value=Number(frame && frame.frame_idx); return Number.isInteger(value) ? value : '';}\n"
        "function currentVideoFrame(rowIndex){const row=REVIEW_ROWS[rowIndex]; const video=document.getElementById(`video-${rowIndex}`); const fps=Number(row.fps); const start=Number(row.clip_start_time_sec || 0); if(!video || !Number.isFinite(fps) || fps <= 0){return '';} return Math.round((start + video.currentTime) * fps);}\n"
        "function currentReviewFrame(rowIndex){const overlayFrame=currentOverlayFrame(rowIndex); if(overlayFrame !== '') return overlayFrame; return currentVideoFrame(rowIndex);}\n"
        "function setSegmentFrame(rowIndex,segmentIndex,field){const frame=currentReviewFrame(rowIndex); if(frame === ''){setStatus('No current overlay frame or video timestamp available'); return;} const el=document.getElementById(segmentFieldId(rowIndex,segmentIndex,field)); if(el){el.value=frame; validateSegment(rowIndex,segmentIndex); autosaveProgress();}}\n"
        "function getSegmentField(rowIndex,segmentIndex,field,fallback=''){const el=document.getElementById(segmentFieldId(rowIndex,segmentIndex,field)); return el ? el.value : fallback;}\n"
        "function intOrNull(value){if(value === '' || value === null || value === undefined) return null; if(!/^-?\\d+$/.test(String(value).trim())) return null; return Number(value);}\n"
        "function segmentFramesOptional(segment){return segment.manual_outcome === 'false_positive' || segment.manual_outcome === 'acceptable_flagged' || segment.acceptance_status === 'accepted' || (segment.manual_outcome === 'review' && segment.acceptance_status === 'review' && !segment.affected_start_frame && !segment.affected_end_frame);}\n"
        "function normalizeSegmentForCollection(segment){if(segment.manual_outcome === 'false_positive' || segment.manual_outcome === 'acceptable_flagged' || segment.acceptance_status === 'accepted'){return {...segment, affected_start_frame:'', affected_end_frame:''};} return segment;}\n"
        "function derivedSeverity(status,manualOutcome,source){if(source && source.severity) return source.severity; if(status === 'accepted') return 'low'; if(status === 'review' || manualOutcome === 'review') return 'medium'; return 'high';}\n"
        "function derivedConfidence(status,manualOutcome,source){if(source && source.confidence) return source.confidence; if(status === 'review' || manualOutcome === 'review') return 'low'; return 'medium';}\n"
        "function collectSegments(rowIndex){const row=REVIEW_ROWS[rowIndex]; const key=rowKey(row); const segments=segmentsByReviewId[key] || []; return segments.map((source,segmentIndex)=>{const manualOutcome=source.manual_outcome || 'true_positive'; const status=source.acceptance_status || 'rejected'; const segment=normalizeSegmentForCollection({segment_id:getSegmentField(rowIndex,segmentIndex,'segment_id',source.segment_id || nextSegmentId(row)),affected_start_frame:getSegmentField(rowIndex,segmentIndex,'affected_start_frame',source.affected_start_frame || ''),affected_end_frame:getSegmentField(rowIndex,segmentIndex,'affected_end_frame',source.affected_end_frame || ''),manual_outcome:manualOutcome,failure_mode:getSegmentField(rowIndex,segmentIndex,'failure_mode',source.failure_mode || defaultFailureMode(row)),severity:derivedSeverity(status,manualOutcome,source),confidence:derivedConfidence(status,manualOutcome,source),acceptance_status:status,comment:getSegmentField(rowIndex,segmentIndex,'comment',source.comment || '')}); segmentsByReviewId[key][segmentIndex]=segment; return segment;});}\n"
        "function setSegmentValidation(rowIndex,segmentIndex,message){const startEl=document.getElementById(segmentFieldId(rowIndex,segmentIndex,'affected_start_frame')); const endEl=document.getElementById(segmentFieldId(rowIndex,segmentIndex,'affected_end_frame')); const errorEl=document.getElementById(segmentFieldId(rowIndex,segmentIndex,'validation_error')); [startEl,endEl].forEach(el=>{if(el) el.classList.toggle('segment-invalid',Boolean(message));}); if(errorEl) errorEl.textContent=message || '';}\n"
        "function validateSegment(rowIndex,segmentIndex){const row=REVIEW_ROWS[rowIndex]; const segment=collectSegments(rowIndex)[segmentIndex]; if(!segment){return true;} if(segmentFramesOptional(segment)){setSegmentValidation(rowIndex,segmentIndex,''); return true;} const startFrame=intOrNull(segment.affected_start_frame); const endFrame=intOrNull(segment.affected_end_frame); let message=''; if(startFrame === null || endFrame === null){message='affected_start_frame and affected_end_frame must be integers';} const windowStart=intOrNull(row.window_start_frame); const windowEnd=intOrNull(row.window_end_frame); if(!message && windowStart !== null && windowEnd !== null){if(startFrame < windowStart){message='affected_start_frame is before candidate window start';} else if(endFrame > windowEnd){message='affected_end_frame is after candidate window end';} else if(startFrame > endFrame){message='affected_start_frame must be <= affected_end_frame';}} const frameCount=intOrNull(row.frame_count); if(!message && (windowStart === null || windowEnd === null) && frameCount !== null){if(startFrame < 0 || endFrame >= frameCount){message='affected frames must be inside asset frame_count';} else if(startFrame > endFrame){message='affected_start_frame must be <= affected_end_frame';}} if(!message && windowStart === null && windowEnd === null && frameCount === null && (startFrame === null || endFrame === null)){message='affected frames are required unless accepted';} setSegmentValidation(rowIndex,segmentIndex,message); return !message;}\n"
        "function validateAllSegments(){let ok=true; REVIEW_ROWS.forEach((row,rowIndex)=>{(segmentsByReviewId[rowKey(row)] || []).forEach((_segment,segmentIndex)=>{if(!validateSegment(rowIndex,segmentIndex)) ok=false;});}); return ok;}\n"
        "function collectManualRows(){return REVIEW_ROWS.flatMap((row,index)=>collectSegments(index).map(segment=>({review_id:row.review_id,segment_id:segment.segment_id,supplier_id:row.supplier_id,asset_id:row.asset_id,window_start_frame:row.window_start_frame,window_end_frame:row.window_end_frame,representative_frame:row.representative_frame,affected_start_frame:segment.affected_start_frame,affected_end_frame:segment.affected_end_frame,auto_verdict:row.auto_verdict,suggested_issue_type:row.suggested_issue_type,severity_suggestion:row.severity_suggestion,key_metrics_json:row.key_metrics_json,reason:row.reason,manual_outcome:segment.manual_outcome,failure_mode:segment.failure_mode,severity:segment.severity,confidence:segment.confidence,acceptance_status:segment.acceptance_status,reviewer:globalReviewer(),comment:segment.comment})));\n"
        "}\n"
        "function captureProgress(){const segments={}; REVIEW_ROWS.forEach((row,index)=>{segments[rowKey(row)]=collectSegments(index);}); return {version:2,storage_key:STORAGE_KEY,global_reviewer:globalReviewer(),segmentsByReviewId:segments,sampledFrameIndexByReviewId:{...sampledFrameIndexByReviewId},saved_at:new Date().toISOString()};}\n"
        "function countSegmentRows(progress){return Object.values(progress.segmentsByReviewId || {}).reduce((total,segments)=>total + (Array.isArray(segments) ? segments.length : 0),0);}\n"
        "function formatTimestamp(date){const pad=value=>String(value).padStart(2,'0'); return `${date.getFullYear()}-${pad(date.getMonth()+1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;}\n"
        "function restoreProgress(value){restoreInProgress=true; try{segmentsByReviewId={}; sampledFrameIndexByReviewId={}; if(Array.isArray(value)){value.forEach(row=>{const key=String(row.review_id ?? ''); if(!segmentsByReviewId[key]) segmentsByReviewId[key]=[]; segmentsByReviewId[key].push(row);}); const firstReviewer=value.find(row=>row && row.reviewer)?.reviewer; if(firstReviewer !== undefined){document.getElementById('global-reviewer').value=firstReviewer || '';}} else if(value && value.segmentsByReviewId){segmentsByReviewId=value.segmentsByReviewId || {}; sampledFrameIndexByReviewId=value.sampledFrameIndexByReviewId || {}; if(value.global_reviewer !== undefined){document.getElementById('global-reviewer').value=value.global_reviewer || '';}} else {segmentsByReviewId=value || {};}} finally{restoreInProgress=false;}}\n"
        "function csvEscape(value){const text=String(value ?? ''); return /[\",\\n\\r]/.test(text) ? '\"' + text.replace(/\"/g,'\"\"') + '\"' : text;}\n"
        "function rowsToCsv(rows){return MANUAL_COLUMNS.join(',')+'\\n'+rows.map(row=>MANUAL_COLUMNS.map(col=>csvEscape(row[col])).join(',')).join('\\n')+'\\n';}\n"
        "function buildManualLabelsCsv(){return rowsToCsv(collectManualRows());}\n"
        "function buildProgressJson(){return captureProgress();}\n"
        "function runLabel(){const parts=String(STORAGE_KEY).split(':'); return parts.length > 1 ? parts.slice(1).join(':') : STORAGE_KEY;}\n"
        "function exportManualLabelsCsv(){if(!validateAllSegments()){setStatus('Cannot export manual_labels.csv: fix invalid affected frame ranges first'); return;} const csv=buildManualLabelsCsv(); const blob=new Blob([csv],{type:'text/csv;charset=utf-8'}); const url=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; a.download='manual_labels.csv'; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url); setStatus('Exported manual_labels.csv');}\n"
        "function saveToLocalStorage(options={}){const progress=buildProgressJson(); localStorage.setItem(STORAGE_KEY,JSON.stringify(progress)); if(!options.silent){setStatus(`Saved at ${formatTimestamp(new Date())}: saved ${REVIEW_ROWS.length} review items, ${countSegmentRows(progress)} segment rows`);} else if(!options.quietStatus){setStatus('Saved locally');} return progress;}\n"
        "function saveProgress(options={}){return saveToLocalStorage(options);}\n"
        "async function saveToServer(source='autosave'){try{if(!validateAllSegments()){setStatus('Server save skipped: fix invalid affected frame ranges first'); return false;} const progress=buildProgressJson(); const payload={run_label:runLabel(),reviewer:globalReviewer(),manual_labels_csv:buildManualLabelsCsv(),progress_json:progress,source:source}; const response=await fetch(SERVER_SAVE_ENDPOINT,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}); if(!response.ok){throw new Error(`HTTP ${response.status}`);} await response.json().catch(()=>({})); const now=formatTimestamp(new Date()).slice(11); setStatus(`Saved to server at ${now}`); return true;}catch(error){const message=String(error && error.message ? error.message : error).slice(0,120); setStatus(`Server save failed: ${message}`); return false;}}\n"
        "function debouncedServerAutosave(){clearTimeout(serverAutosaveTimer); serverAutosaveTimer=setTimeout(()=>saveToServer('autosave'),1000);}\n"
        "function autosaveProgress(){if(restoreInProgress) return; saveToLocalStorage({silent:true}); debouncedServerAutosave();}\n"
        "function refreshRestoredUi(){REVIEW_ROWS.forEach((_row,rowIndex)=>{renderSegments(rowIndex); updateSampledFrame(rowIndex);}); validateAllSegments();}\n"
        "function loadProgress(){const raw=localStorage.getItem(STORAGE_KEY); if(!raw){setStatus('No saved progress found'); return false;} const progress=JSON.parse(raw); restoreProgress(progress); render(); refreshRestoredUi(); setStatus(`Loaded saved progress from localStorage: loaded ${REVIEW_ROWS.length} review items, ${countSegmentRows(captureProgress())} segment rows`); return true;}\n"
        "function loadProgressOnStartup(){try{loadProgress();}catch(error){setStatus(`Could not load saved progress: ${error.message}`);}}\n"
        "function exportProgressJson(){const progress=saveToLocalStorage({silent:true,quietStatus:true}); const blob=new Blob([JSON.stringify(progress,null,2)],{type:'application/json;charset=utf-8'}); const url=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; a.download='manual_review_progress.json'; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url); setStatus(`Exported progress JSON, ${countSegmentRows(progress)} segment rows`);}\n"
        "function importProgressJson(event){const file=event.target.files && event.target.files[0]; if(!file) return; const reader=new FileReader(); reader.onload=()=>{try{const progress=JSON.parse(String(reader.result || '{}')); restoreProgress(progress); render(); refreshRestoredUi(); saveProgress({silent:true}); setStatus(`Imported progress JSON: loaded ${REVIEW_ROWS.length} review items, ${countSegmentRows(captureProgress())} segment rows`);}catch(error){setStatus(`Could not import progress JSON: ${error.message}`);} finally{event.target.value='';}}; reader.readAsText(file);}\n"
        "function clearProgress(){localStorage.removeItem(STORAGE_KEY); setStatus('Cleared local progress');}\n"
        "function parseSegmentTarget(target){const match=String(target && target.id || '').match(/^seg-(\\d+)-(\\d+)-(.+)$/); return match ? {rowIndex:Number(match[1]),segmentIndex:Number(match[2]),field:match[3]} : null;}\n"
        "function clearAcceptedFrameInputs(rowIndex,segmentIndex){const outcome=getSegmentField(rowIndex,segmentIndex,'manual_outcome'); const status=getSegmentField(rowIndex,segmentIndex,'acceptance_status'); if(outcome === 'false_positive' || outcome === 'acceptable_flagged' || status === 'accepted'){['affected_start_frame','affected_end_frame'].forEach(field=>{const el=document.getElementById(segmentFieldId(rowIndex,segmentIndex,field)); if(el) el.value='';});}}\n"
        "function applyAcceptedOutcomeDefaults(rowIndex,segmentIndex){const outcome=getSegmentField(rowIndex,segmentIndex,'manual_outcome'); if(outcome === 'false_positive' || outcome === 'acceptable_flagged'){const statusEl=document.getElementById(segmentFieldId(rowIndex,segmentIndex,'acceptance_status')); const severityEl=document.getElementById(segmentFieldId(rowIndex,segmentIndex,'severity')); if(statusEl) statusEl.value='accepted'; if(severityEl) severityEl.value='low'; clearAcceptedFrameInputs(rowIndex,segmentIndex);}}\n"
        "function handleManualFieldChange(event){if(event.target && event.target.id === 'progress-json-input') return; const parsed=parseSegmentTarget(event.target); if(parsed){applyAcceptedOutcomeDefaults(parsed.rowIndex,parsed.segmentIndex); clearAcceptedFrameInputs(parsed.rowIndex,parsed.segmentIndex); validateSegment(parsed.rowIndex,parsed.segmentIndex);} if(event.target && (event.target.closest('.human') || event.target.id === 'global-reviewer')) autosaveProgress();}\n"
        "function setStatus(text){document.getElementById('status').textContent=text;}\n"
        "function initializePage(){document.getElementById('storage-key').textContent=STORAGE_KEY; render(); loadProgressOnStartup(); document.addEventListener('input', handleManualFieldChange); document.addEventListener('change', handleManualFieldChange);}\n"
        "document.addEventListener('DOMContentLoaded', initializePage);\n"
        "</script></body></html>\n"
    ).replace("__CLIP_JS__", clip_js)


def build_review_ui_enums(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        "manual_outcome": list(OVERLAY_MANUAL_OUTCOME_ENUM),
        "failure_mode": list(OVERLAY_FAILURE_MODE_ENUM),
        "severity": SEVERITY_ENUM,
        "confidence": CONFIDENCE_ENUM,
        "acceptance_status": ACCEPTANCE_STATUS_ENUM,
        "accept_reason": ACCEPT_REASON_ENUM,
    }


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def build_storage_key(rows: list[dict[str, Any]]) -> str:
    run_label = "empty"
    if rows:
        first = rows[0]
        run_label = safe_filename(
            str(
                first.get("review_run_label")
                or first.get("run_label")
                or first.get("output_dir_name")
                or first.get("supplier_id")
                or first.get("asset_id")
                or "review"
            )
        )
    return f"manual_review_state_v3:{run_label}"


def resolve_fps(row: dict[str, Any]) -> float | None:
    fps = float_or_none(row.get("fps"))
    if fps is not None and fps > 0:
        return fps
    frame_count = float_or_none(row.get("frame_count"))
    duration = float_or_none(row.get("duration_sec"))
    if frame_count is not None and duration is not None and duration > 0:
        return frame_count / duration
    return None


def int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def parse_sampled_frames(row: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        frames = json.loads(str(row.get("sampled_frames_json") or "[]"))
    except json.JSONDecodeError:
        return []
    if not isinstance(frames, list):
        return []
    return [dict(frame) for frame in frames if isinstance(frame, dict)]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
