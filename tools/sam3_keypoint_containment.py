#!/usr/bin/env python3
"""Sample videos, run SAM3 hand masks, and score projected hand keypoints.

This is a cloud-side acceptance utility. It intentionally lives outside
precheck/checks because it loads SAM3. The precheck package should consume the
resulting masks or JSON metrics, not import or run this script.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from annotation.segmentation.sam3 import SAM3Segmenter
from qc_common.keypoints import acceptance_joint_names, project_points

LOGGER = logging.getLogger("sam3_keypoint_containment")


PROJECTION_MODES = ("direct", "camera_inverse", "camera_forward")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SAM3 on sampled video frames and compute how many 21-hand "
            "acceptance keypoints fall inside the generated hand mask."
        )
    )
    parser.add_argument("--hdf5-dir", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--sam3-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-fraction", type=float, default=0.10)
    parser.add_argument(
        "--start-clip",
        type=int,
        default=0,
        help=(
            "0-based inclusive start index after sorting hdf5 files. "
            "Use --start-clip 1 --end-clip 2 to run only the second clip."
        ),
    )
    parser.add_argument(
        "--end-clip",
        type=int,
        default=None,
        help="0-based exclusive end index after sorting hdf5 files.",
    )
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--max-sampled-frames-per-clip", type=int, default=None)
    parser.add_argument(
        "--projection-mode",
        choices=("auto", *PROJECTION_MODES),
        default="auto",
        help=(
            "direct assumes joint transforms are already camera coordinates; "
            "camera_inverse applies inv(transforms/camera); camera_forward "
            "applies transforms/camera; auto chooses the mode with the most "
            "in-frame projected points."
        ),
    )
    parser.add_argument(
        "--queries",
        default="hand,left hand,right hand,robot hand,gripper",
        help="Comma-separated SAM3 text prompts.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--max-instances-per-query", type=int, default=10)
    parser.add_argument(
        "--write-overlays",
        action="store_true",
        help="Write per-frame RGB + SAM3 mask + projected keypoint overlay PNGs.",
    )
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=None,
        help="Overlay output directory. Defaults to <output-dir>/overlays.",
    )
    parser.add_argument(
        "--abnormal-inside-ratio-threshold",
        type=float,
        default=1.0,
        help=(
            "Frame-level abnormal threshold. A sampled frame is abnormal when "
            "keypoint_inside_ratio is below this value. The default 1.0 means "
            "any expected acceptance keypoint outside the SAM3 mask, including "
            "unprojectable keypoints, marks the frame abnormal."
        ),
    )
    parser.add_argument(
        "--video-patterns",
        default="{episode_id}.mp4,{stem}.mp4,{stem_no_hdf5}.mp4",
        help=(
            "Comma-separated filename patterns searched under --video-dir. "
            "Available fields: stem, stem_no_hdf5, episode_id."
        ),
    )
    parser.add_argument("--recursive-videos", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if not 0.0 < args.sample_fraction <= 1.0:
        raise ValueError("--sample-fraction must be in (0, 1]")
    if not 0.0 <= args.abnormal_inside_ratio_threshold <= 1.0:
        raise ValueError("--abnormal-inside-ratio-threshold must be in [0, 1]")
    if args.start_clip < 0:
        raise ValueError("--start-clip must be >= 0")
    if args.end_clip is not None and args.end_clip < args.start_clip:
        raise ValueError("--end-clip must be >= --start-clip")
    if args.max_clips is not None and args.max_clips < 1:
        raise ValueError("--max-clips must be >= 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_hdf5_paths = sorted(args.hdf5_dir.glob("*.hdf5"))
    hdf5_paths = all_hdf5_paths[args.start_clip : args.end_clip]
    if args.max_clips is not None:
        hdf5_paths = hdf5_paths[: args.max_clips]
    if not hdf5_paths:
        raise FileNotFoundError(
            f"No .hdf5 files selected under {args.hdf5_dir}; "
            f"available={len(all_hdf5_paths)}, "
            f"start_clip={args.start_clip}, end_clip={args.end_clip}"
        )

    queries = [query.strip() for query in args.queries.split(",") if query.strip()]
    segmenter = SAM3Segmenter(
        args.sam3_model,
        {
            "confidence_threshold": args.confidence_threshold,
            "mask_threshold": args.mask_threshold,
            "max_instances_per_query": args.max_instances_per_query,
        },
    )

    frame_rows: list[dict[str, Any]] = []
    clip_rows: list[dict[str, Any]] = []
    for offset, hdf5_path in enumerate(hdf5_paths):
        LOGGER.info("Clip %d/%d: %s", offset + 1, len(hdf5_paths), hdf5_path.name)
        video_path = find_video_path(
            hdf5_path,
            args.video_dir,
            args.video_patterns.split(","),
            recursive=args.recursive_videos,
        )
        if video_path is None:
            LOGGER.warning("No matching video for %s", hdf5_path.name)
            clip_rows.append(error_clip_row(hdf5_path, "matching video not found"))
            continue

        try:
            clip_frame_rows, clip_summary = process_clip(
                hdf5_path=hdf5_path,
                video_path=video_path,
                segmenter=segmenter,
                queries=queries,
                sample_fraction=args.sample_fraction,
                max_sampled_frames=args.max_sampled_frames_per_clip,
                projection_mode=args.projection_mode,
                abnormal_inside_ratio_threshold=args.abnormal_inside_ratio_threshold,
                overlay_dir=(args.overlay_dir or args.output_dir / "overlays")
                if args.write_overlays
                else None,
                sam3_config={
                    "confidence_threshold": args.confidence_threshold,
                    "mask_threshold": args.mask_threshold,
                    "max_instances_per_query": args.max_instances_per_query,
                },
            )
        except Exception as exc:
            LOGGER.exception("Clip failed: %s", hdf5_path.name)
            clip_rows.append(error_clip_row(hdf5_path, str(exc), video_path))
            continue
        frame_rows.extend(clip_frame_rows)
        clip_rows.append(clip_summary)

    write_json(frame_rows, args.output_dir / "frame_keypoint_containment.json")
    write_json(clip_rows, args.output_dir / "clip_keypoint_containment.json")
    write_json(
        {
            "hdf5_dir": str(args.hdf5_dir),
            "video_dir": str(args.video_dir),
            "sam3_model": str(args.sam3_model),
            "sample_fraction": args.sample_fraction,
            "start_clip": args.start_clip,
            "end_clip": args.end_clip,
            "max_clips": args.max_clips,
            "available_hdf5_count": len(all_hdf5_paths),
            "selected_hdf5_paths": [str(path) for path in hdf5_paths],
            "queries": queries,
            "abnormal_inside_ratio_threshold": args.abnormal_inside_ratio_threshold,
            "write_overlays": args.write_overlays,
            "overlay_dir": str(args.overlay_dir or args.output_dir / "overlays")
            if args.write_overlays
            else None,
            "num_clips": len(clip_rows),
        },
        args.output_dir / "run_manifest.json",
    )
    LOGGER.info("Wrote results under %s", args.output_dir)


def process_clip(
    hdf5_path: Path,
    video_path: Path,
    segmenter: SAM3Segmenter,
    queries: list[str],
    sample_fraction: float,
    max_sampled_frames: int | None,
    projection_mode: str,
    abnormal_inside_ratio_threshold: float,
    overlay_dir: Path | None,
    sam3_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import h5py

    with h5py.File(hdf5_path, "r") as handle:
        intrinsics = np.asarray(handle["camera/intrinsic"], dtype=np.float64)
        camera_transforms = np.asarray(handle["transforms/camera"], dtype=np.float64)
        joint_names = [
            name for name in acceptance_joint_names() if f"transforms/{name}" in handle
        ]
        if not joint_names:
            raise ValueError("no acceptance hand joints found under transforms/")
        points = np.stack(
            [
                np.asarray(handle[f"transforms/{name}"][:, :3, 3], dtype=np.float64)
                for name in joint_names
            ],
            axis=1,
        )
        num_frames = int(points.shape[0])

    sampled_frames = sample_frame_indices(num_frames, sample_fraction)
    if max_sampled_frames is not None:
        sampled_frames = sampled_frames[:max_sampled_frames]
    if not sampled_frames:
        raise ValueError("no sampled frames selected")

    first_frame = extract_frame(video_path, sampled_frames[0])
    height, width = first_frame.shape[:2]
    resolved_projection_mode = resolve_projection_mode(
        points,
        camera_transforms,
        intrinsics,
        sampled_frames,
        width,
        height,
        projection_mode,
    )

    frame_rows: list[dict[str, Any]] = []
    totals = Counter()
    mode_counts = Counter()
    for frame_idx in sampled_frames:
        frame = first_frame if frame_idx == sampled_frames[0] else extract_frame(video_path, frame_idx)
        masks = segmenter.segment_frame(frame, queries, sam3_config)
        union_mask = union_instance_masks(masks, frame.shape[:2])
        projected = project_frame_points(
            points[frame_idx],
            camera_transforms[frame_idx],
            intrinsics,
            resolved_projection_mode,
        )
        valid = projected["valid"]
        pixels = projected["pixels"]

        total_expected = len(joint_names)
        valid_count = int(np.sum(valid))
        inside = np.zeros(total_expected, dtype=bool)
        if union_mask is not None and valid_count:
            rounded = np.rint(pixels).astype(np.int64)
            in_bounds = (
                valid
                & (rounded[:, 0] >= 0)
                & (rounded[:, 0] < frame.shape[1])
                & (rounded[:, 1] >= 0)
                & (rounded[:, 1] < frame.shape[0])
            )
            inside[in_bounds] = union_mask[rounded[in_bounds, 1], rounded[in_bounds, 0]]

        inside_indices = np.flatnonzero(inside).tolist()
        missing_indices = np.flatnonzero(~inside).tolist()
        valid_missing_indices = np.flatnonzero(valid & ~inside).tolist()
        invalid_indices = np.flatnonzero(~valid).tolist()
        inside_count = int(np.sum(inside))
        missing_from_mask_count = int(valid_count - inside_count)
        expected_missing_from_mask_count = int(total_expected - inside_count)
        valid_projected_inside_ratio = safe_ratio(inside_count, valid_count)
        valid_projected_missing_ratio = safe_ratio(missing_from_mask_count, valid_count)
        keypoint_inside_ratio = safe_ratio(inside_count, total_expected)
        keypoint_missing_ratio = safe_ratio(expected_missing_from_mask_count, total_expected)
        abnormal_frame = (
            keypoint_inside_ratio is not None
            and keypoint_inside_ratio < abnormal_inside_ratio_threshold
        )
        mask_area = int(np.sum(union_mask)) if union_mask is not None else 0
        row = {
            "clip_id": clip_id_from_path(hdf5_path),
            "hdf5_path": str(hdf5_path),
            "video_path": str(video_path),
            "frame_idx": int(frame_idx),
            "projection_mode": resolved_projection_mode,
            "image_width": int(frame.shape[1]),
            "image_height": int(frame.shape[0]),
            "sampled_keypoints": int(total_expected),
            "valid_projected_keypoints": valid_count,
            "inside_keypoints": inside_count,
            "missing_from_mask_keypoints": missing_from_mask_count,
            "expected_missing_from_mask_keypoints": expected_missing_from_mask_count,
            "inside_joint_names": [joint_names[index] for index in inside_indices],
            "missing_from_mask_joint_names": [
                joint_names[index] for index in valid_missing_indices
            ],
            "expected_missing_from_mask_joint_names": [
                joint_names[index] for index in missing_indices
            ],
            "invalid_projected_joint_names": [
                joint_names[index] for index in invalid_indices
            ],
            "keypoint_inside_ratio": keypoint_inside_ratio,
            "keypoint_missing_ratio": keypoint_missing_ratio,
            "valid_projected_inside_ratio": valid_projected_inside_ratio,
            "valid_projected_missing_ratio": valid_projected_missing_ratio,
            "abnormal_inside_ratio_threshold": abnormal_inside_ratio_threshold,
            "abnormal_frame": abnormal_frame,
            "mask_instance_count": len(masks),
            "mask_area": mask_area,
            "mask_area_ratio": safe_ratio(mask_area, frame.shape[0] * frame.shape[1]),
            "sam3_categories": sorted({mask.category for mask in masks}),
        }
        if overlay_dir is not None:
            overlay_path = write_overlay_image(
                frame=frame,
                mask=union_mask,
                pixels=pixels,
                valid=valid,
                inside=inside,
                joint_names=joint_names,
                clip_id=clip_id_from_path(hdf5_path),
                frame_idx=frame_idx,
                output_dir=overlay_dir,
            )
            row["overlay_path"] = str(overlay_path)
        frame_rows.append(row)
        totals["sampled_frames"] += 1
        totals["total_expected_keypoints"] += total_expected
        totals["valid_projected_keypoints"] += valid_count
        totals["inside_keypoints"] += inside_count
        totals["missing_from_mask_keypoints"] += missing_from_mask_count
        totals["expected_missing_from_mask_keypoints"] += expected_missing_from_mask_count
        totals["abnormal_frames"] += int(abnormal_frame)
        totals["frames_with_mask"] += int(union_mask is not None and mask_area > 0)
        totals["frames_without_mask"] += int(union_mask is None or mask_area == 0)
        mode_counts[resolved_projection_mode] += 1

    ratios = [row["keypoint_inside_ratio"] for row in frame_rows]
    valid_ratios = [
        row["valid_projected_inside_ratio"]
        for row in frame_rows
        if row["valid_projected_inside_ratio"] is not None
    ]
    clip_summary = {
        "clip_id": clip_id_from_path(hdf5_path),
        "hdf5_path": str(hdf5_path),
        "video_path": str(video_path),
        "sample_fraction": sample_fraction,
        "sampled_frame_indices": [row["frame_idx"] for row in frame_rows],
        "abnormal_frame_indices": [
            row["frame_idx"] for row in frame_rows if row["abnormal_frame"]
        ],
        "overlay_paths": [
            row["overlay_path"] for row in frame_rows if "overlay_path" in row
        ],
        "sampled_frames": int(totals["sampled_frames"]),
        "joint_count_per_frame": len(joint_names),
        "joint_names": joint_names,
        "projection_mode": resolved_projection_mode,
        "projection_mode_counts": dict(mode_counts),
        "total_expected_keypoints": int(totals["total_expected_keypoints"]),
        "valid_projected_keypoints": int(totals["valid_projected_keypoints"]),
        "inside_keypoints": int(totals["inside_keypoints"]),
        "missing_from_mask_keypoints": int(totals["missing_from_mask_keypoints"]),
        "expected_missing_from_mask_keypoints": int(
            totals["expected_missing_from_mask_keypoints"]
        ),
        "clip_keypoint_inside_ratio": safe_ratio(
            totals["inside_keypoints"], totals["total_expected_keypoints"]
        ),
        "clip_keypoint_missing_ratio": safe_ratio(
            totals["expected_missing_from_mask_keypoints"],
            totals["total_expected_keypoints"],
        ),
        "valid_projected_inside_ratio": safe_ratio(
            totals["inside_keypoints"], totals["valid_projected_keypoints"]
        ),
        "valid_projected_missing_ratio": safe_ratio(
            totals["missing_from_mask_keypoints"], totals["valid_projected_keypoints"]
        ),
        "mean_frame_inside_ratio": float(np.mean(ratios)) if ratios else None,
        "mean_valid_projected_inside_ratio": (
            float(np.mean(valid_ratios)) if valid_ratios else None
        ),
        "abnormal_inside_ratio_threshold": abnormal_inside_ratio_threshold,
        "abnormal_frames": int(totals["abnormal_frames"]),
        "abnormal_frame_ratio": safe_ratio(
            totals["abnormal_frames"], totals["sampled_frames"]
        ),
        "frames_with_mask": int(totals["frames_with_mask"]),
        "frames_without_mask": int(totals["frames_without_mask"]),
        "error": None,
    }
    return frame_rows, clip_summary


def sample_frame_indices(num_frames: int, fraction: float) -> list[int]:
    count = min(max(1, int(math.ceil(num_frames * fraction))), num_frames)
    return sorted(set(np.linspace(0, num_frames - 1, count, dtype=int).tolist()))


def find_video_path(
    hdf5_path: Path,
    video_dir: Path,
    patterns: list[str],
    recursive: bool,
) -> Path | None:
    stem = hdf5_path.stem
    stem_no_hdf5 = stem.removesuffix("_hdf5")
    episode_id = "".join(ch for ch in stem_no_hdf5 if ch.isdigit()) or stem_no_hdf5
    values = {
        "stem": stem,
        "stem_no_hdf5": stem_no_hdf5,
        "episode_id": episode_id,
    }
    for pattern in patterns:
        filename = pattern.strip().format(**values)
        candidate = video_dir / filename
        if candidate.exists():
            return candidate
        if recursive:
            matches = sorted(video_dir.rglob(filename))
            if matches:
                return matches[0]
    return None


def extract_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video_path),
                "-vf",
                f"select=eq(n\\,{int(frame_idx)})",
                "-vframes",
                "1",
                tmp.name,
            ],
            check=True,
        )
        from PIL import Image

        return np.asarray(Image.open(tmp.name).convert("RGB"))


def resolve_projection_mode(
    points: np.ndarray,
    camera_transforms: np.ndarray,
    intrinsics: np.ndarray,
    sampled_frames: list[int],
    width: int,
    height: int,
    projection_mode: str,
) -> str:
    if projection_mode != "auto":
        return projection_mode
    scores: dict[str, float] = {}
    for mode in PROJECTION_MODES:
        valid_counts = []
        for frame_idx in sampled_frames[: min(10, len(sampled_frames))]:
            projected = project_frame_points(
                points[frame_idx],
                camera_transforms[frame_idx],
                intrinsics,
                mode,
            )
            pixels = projected["pixels"]
            valid = projected["valid"]
            in_frame = (
                valid
                & (pixels[:, 0] >= 0)
                & (pixels[:, 0] < width)
                & (pixels[:, 1] >= 0)
                & (pixels[:, 1] < height)
            )
            valid_counts.append(float(np.mean(in_frame)))
        scores[mode] = float(np.mean(valid_counts)) if valid_counts else 0.0
    best_mode = max(scores, key=scores.get)
    LOGGER.info("Projection auto scores=%s; selected=%s", scores, best_mode)
    return best_mode


def project_frame_points(
    points_xyz: np.ndarray,
    camera_transform: np.ndarray,
    intrinsics: np.ndarray,
    mode: str,
) -> dict[str, np.ndarray]:
    points_camera = np.asarray(points_xyz, dtype=np.float64)
    if mode in {"camera_inverse", "camera_forward"}:
        points_h = np.concatenate(
            [points_camera, np.ones((points_camera.shape[0], 1), dtype=np.float64)],
            axis=1,
        )
        transform = (
            np.linalg.inv(camera_transform)
            if mode == "camera_inverse"
            else camera_transform
        )
        points_camera = (transform @ points_h.T).T[:, :3]
    pixels = project_points(points_camera, intrinsics)
    valid = (
        np.isfinite(points_camera).all(axis=1)
        & np.isfinite(pixels).all(axis=1)
        & (points_camera[:, 2] > 1e-8)
    )
    return {"points_camera": points_camera, "pixels": pixels, "valid": valid}


def union_instance_masks(masks: list[Any], image_shape: tuple[int, int]) -> np.ndarray | None:
    if not masks:
        return None
    union = np.zeros(image_shape, dtype=bool)
    for instance in masks:
        mask = np.asarray(instance.mask, dtype=bool)
        if mask.shape != image_shape:
            LOGGER.warning("Skipping mask with shape %s; expected %s", mask.shape, image_shape)
            continue
        union |= mask
    return union


def write_overlay_image(
    frame: np.ndarray,
    mask: np.ndarray | None,
    pixels: np.ndarray,
    valid: np.ndarray,
    inside: np.ndarray,
    joint_names: list[str],
    clip_id: str,
    frame_idx: int,
    output_dir: Path,
) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGBA")
    width, height = image.size

    if mask is not None:
        mask_bool = np.asarray(mask, dtype=bool)
        if mask_bool.shape == (height, width):
            overlay = np.zeros((height, width, 4), dtype=np.uint8)
            overlay[mask_bool] = np.array([0, 220, 160, 90], dtype=np.uint8)
            image = Image.alpha_composite(image, Image.fromarray(overlay, mode="RGBA"))

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    valid_count = int(np.sum(valid))
    inside_count = int(np.sum(inside))
    invalid_count = int(len(joint_names) - valid_count)
    outside_count = int(valid_count - inside_count)

    header = (
        f"{clip_id} frame={int(frame_idx)} "
        f"inside={inside_count}/{len(joint_names)} "
        f"outside={outside_count} invalid={invalid_count}"
    )
    draw.rectangle((0, 0, min(width, 760), 26), fill=(0, 0, 0, 175))
    draw.text((6, 6), header, fill=(255, 255, 255, 255), font=font)

    for index, (x_raw, y_raw) in enumerate(pixels):
        if not bool(valid[index]) or not np.isfinite([x_raw, y_raw]).all():
            continue
        x = int(round(float(x_raw)))
        y = int(round(float(y_raw)))
        if x < 0 or x >= width or y < 0 or y >= height:
            continue

        is_inside = bool(inside[index])
        fill = (0, 255, 90, 255) if is_inside else (255, 45, 45, 255)
        radius = 4 if is_inside else 5
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=fill,
            outline=(0, 0, 0, 230),
            width=1,
        )
        if not is_inside:
            draw.text(
                (x + 7, y - 7),
                compact_joint_label(joint_names[index]),
                fill=(255, 255, 255, 255),
                font=font,
                stroke_width=2,
                stroke_fill=(0, 0, 0, 220),
            )

    output_path = output_dir / f"{clip_id}_frame_{int(frame_idx):06d}.png"
    image.convert("RGB").save(output_path)
    return output_path


def compact_joint_label(joint_name: str) -> str:
    label = joint_name
    side = ""
    if label.startswith("left"):
        side = "L:"
        label = label.removeprefix("left")
    elif label.startswith("right"):
        side = "R:"
        label = label.removeprefix("right")

    replacements = {
        "Thumb": "Th",
        "Index": "Idx",
        "Middle": "Mid",
        "Ring": "Ring",
        "Little": "Lit",
        "Finger": "F",
        "Intermediate": "Int",
        "Knuckle": "Kn",
        "Base": "Base",
        "Tip": "Tip",
    }
    for old, new in replacements.items():
        label = label.replace(old, new)
    return f"{side}{label}"


def clip_id_from_path(path: Path) -> str:
    return path.stem.removesuffix("_hdf5")


def safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def error_clip_row(hdf5_path: Path, error: str, video_path: Path | None = None) -> dict[str, Any]:
    return {
        "clip_id": clip_id_from_path(hdf5_path),
        "hdf5_path": str(hdf5_path),
        "video_path": str(video_path) if video_path is not None else None,
        "sampled_frames": 0,
        "total_expected_keypoints": 0,
        "valid_projected_keypoints": 0,
        "inside_keypoints": 0,
        "missing_from_mask_keypoints": 0,
        "expected_missing_from_mask_keypoints": 0,
        "clip_keypoint_inside_ratio": None,
        "clip_keypoint_missing_ratio": None,
        "valid_projected_inside_ratio": None,
        "valid_projected_missing_ratio": None,
        "abnormal_frames": 0,
        "abnormal_frame_ratio": None,
        "error": error,
    }


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(value), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    main()
