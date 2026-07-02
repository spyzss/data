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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hdf5_paths = sorted(args.hdf5_dir.glob("*.hdf5"))
    if args.max_clips is not None:
        hdf5_paths = hdf5_paths[: args.max_clips]
    if not hdf5_paths:
        raise FileNotFoundError(f"No .hdf5 files under {args.hdf5_dir}")

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
            "queries": queries,
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

        inside_count = int(np.sum(inside))
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
            "keypoint_inside_ratio": safe_ratio(inside_count, total_expected),
            "valid_projected_inside_ratio": safe_ratio(inside_count, valid_count),
            "mask_instance_count": len(masks),
            "mask_area": mask_area,
            "mask_area_ratio": safe_ratio(mask_area, frame.shape[0] * frame.shape[1]),
            "sam3_categories": sorted({mask.category for mask in masks}),
        }
        frame_rows.append(row)
        totals["sampled_frames"] += 1
        totals["total_expected_keypoints"] += total_expected
        totals["valid_projected_keypoints"] += valid_count
        totals["inside_keypoints"] += inside_count
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
        "sampled_frames": int(totals["sampled_frames"]),
        "joint_count_per_frame": len(joint_names),
        "joint_names": joint_names,
        "projection_mode": resolved_projection_mode,
        "projection_mode_counts": dict(mode_counts),
        "total_expected_keypoints": int(totals["total_expected_keypoints"]),
        "valid_projected_keypoints": int(totals["valid_projected_keypoints"]),
        "inside_keypoints": int(totals["inside_keypoints"]),
        "clip_keypoint_inside_ratio": safe_ratio(
            totals["inside_keypoints"], totals["total_expected_keypoints"]
        ),
        "valid_projected_inside_ratio": safe_ratio(
            totals["inside_keypoints"], totals["valid_projected_keypoints"]
        ),
        "mean_frame_inside_ratio": float(np.mean(ratios)) if ratios else None,
        "mean_valid_projected_inside_ratio": (
            float(np.mean(valid_ratios)) if valid_ratios else None
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
        "clip_keypoint_inside_ratio": None,
        "valid_projected_inside_ratio": None,
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
