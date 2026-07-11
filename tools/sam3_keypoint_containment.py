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
from qc_common.keypoints import (
    acceptance_joint_names,
    derive_finger_bones,
    project_points,
)

LOGGER = logging.getLogger("sam3_keypoint_containment")


PROJECTION_MODES = ("direct", "camera_inverse", "camera_forward")


def create_sam3_segmenter(
    model_path: Path,
    sam3_config: dict[str, Any],
) -> SAM3Segmenter:
    """Construct the sidecar's canonical SAM3 segmenter."""
    return SAM3Segmenter(model_path, sam3_config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SAM3 on sampled video frames and compute how many 21-hand "
            "acceptance keypoints fall inside the generated hand mask."
        )
    )
    parser.add_argument("--hdf5-dir", type=Path, default=None)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--sam3-model", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--aggregate-frame-results",
        type=Path,
        default=None,
        help=(
            "Read an existing frame_keypoint_containment.json and write "
            "window_keypoint_containment_summary outputs without running SAM3."
        ),
    )
    parser.add_argument("--sample-fraction", type=float, default=0.10)
    parser.add_argument(
        "--candidate-windows",
        type=Path,
        default=None,
        help=(
            "Optional precheck candidate_windows.json. When provided, frame "
            "selection is window-based and --sample-fraction is ignored."
        ),
    )
    parser.add_argument(
        "--asset-ids",
        default=None,
        help="Optional comma-separated asset ids to keep, e.g. 100030,100044.",
    )
    parser.add_argument(
        "--frames-per-window",
        type=int,
        default=3,
        help="Number of evenly spaced interior frames sampled per candidate window.",
    )
    parser.add_argument(
        "--no-window-boundaries",
        dest="include_window_boundaries",
        action="store_false",
        help="Do not force include candidate window start_frame and end_frame.",
    )
    parser.set_defaults(include_window_boundaries=True)
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
    parser.add_argument("--projected-in-image-ratio-threshold", type=float, default=0.8)
    parser.add_argument(
        "--strong-containment-inside-ratio-threshold",
        type=float,
        default=0.2,
    )
    parser.add_argument("--acceptable-inside-ratio-threshold", type=float, default=0.6)
    parser.add_argument("--mask-tiny-area-ratio-threshold", type=float, default=0.0)
    parser.add_argument("--containment-fail-min-strong-frames", type=int, default=3)
    parser.add_argument("--containment-fail-strong-frame-ratio", type=float, default=0.6)
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
    if not 0.0 <= args.projected_in_image_ratio_threshold <= 1.0:
        raise ValueError("--projected-in-image-ratio-threshold must be in [0, 1]")
    if not 0.0 <= args.strong_containment_inside_ratio_threshold <= 1.0:
        raise ValueError("--strong-containment-inside-ratio-threshold must be in [0, 1]")
    if not 0.0 <= args.acceptable_inside_ratio_threshold <= 1.0:
        raise ValueError("--acceptable-inside-ratio-threshold must be in [0, 1]")
    if args.strong_containment_inside_ratio_threshold >= args.acceptable_inside_ratio_threshold:
        raise ValueError(
            "--strong-containment-inside-ratio-threshold must be less than "
            "--acceptable-inside-ratio-threshold"
        )
    if args.start_clip < 0:
        raise ValueError("--start-clip must be >= 0")
    if args.end_clip is not None and args.end_clip < args.start_clip:
        raise ValueError("--end-clip must be >= --start-clip")
    if args.max_clips is not None and args.max_clips < 1:
        raise ValueError("--max-clips must be >= 1")
    if args.frames_per_window < 1:
        raise ValueError("--frames-per-window must be >= 1")
    if args.containment_fail_min_strong_frames < 1:
        raise ValueError("--containment-fail-min-strong-frames must be >= 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.aggregate_frame_results is not None:
        frame_rows = json.loads(args.aggregate_frame_results.read_text(encoding="utf-8"))
        if not isinstance(frame_rows, list):
            raise ValueError("--aggregate-frame-results must point to a JSON list")
        frame_rows = classify_containment_rows(
            frame_rows,
            projected_in_image_ratio_threshold=args.projected_in_image_ratio_threshold,
            strong_inside_ratio_threshold=args.strong_containment_inside_ratio_threshold,
            acceptable_inside_ratio_threshold=args.acceptable_inside_ratio_threshold,
            mask_tiny_area_ratio_threshold=args.mask_tiny_area_ratio_threshold,
        )
        summaries = aggregate_window_containment_summaries(
            frame_rows,
            fail_min_strong_frames=args.containment_fail_min_strong_frames,
            fail_strong_frame_ratio=args.containment_fail_strong_frame_ratio,
        )
        write_window_summary_outputs(summaries, args.output_dir)
        write_json(
            {
                "aggregate_frame_results": str(args.aggregate_frame_results),
                "projected_in_image_ratio_threshold": args.projected_in_image_ratio_threshold,
                "strong_containment_inside_ratio_threshold": (
                    args.strong_containment_inside_ratio_threshold
                ),
                "acceptable_inside_ratio_threshold": args.acceptable_inside_ratio_threshold,
                "mask_tiny_area_ratio_threshold": args.mask_tiny_area_ratio_threshold,
                "num_windows": len(summaries),
            },
            args.output_dir / "run_manifest.json",
        )
        LOGGER.info("Wrote aggregate-only results under %s", args.output_dir)
        return

    if args.hdf5_dir is None or args.video_dir is None or args.sam3_model is None:
        raise ValueError(
            "--hdf5-dir, --video-dir, and --sam3-model are required unless "
            "--aggregate-frame-results is used"
        )

    all_hdf5_paths = sorted(
        path
        for suffix in ("*.hdf5", "*.h5")
        for path in args.hdf5_dir.glob(suffix)
    )
    if not all_hdf5_paths:
        raise FileNotFoundError(
            f"No .hdf5 files found under {args.hdf5_dir}"
        )

    queries = [query.strip() for query in args.queries.split(",") if query.strip()]
    segmenter = create_sam3_segmenter(
        args.sam3_model,
        {
            "confidence_threshold": args.confidence_threshold,
            "mask_threshold": args.mask_threshold,
            "max_instances_per_query": args.max_instances_per_query,
        },
    )

    frame_rows: list[dict[str, Any]] = []
    clip_rows: list[dict[str, Any]] = []
    selected_hdf5_paths: list[Path] = []
    candidate_windows: list[dict[str, Any]] | None = None
    asset_id_filter = parse_asset_ids(args.asset_ids)
    if args.candidate_windows is not None:
        candidate_windows = filter_candidate_windows(
            load_candidate_windows(args.candidate_windows),
            asset_id_filter,
        )
        LOGGER.info(
            "Candidate-window mode: %d windows selected from %s",
            len(candidate_windows),
            args.candidate_windows,
        )
        for offset, window in enumerate(candidate_windows):
            asset_label = window.get("asset_id") or f"episode:{window.get('episode_idx')}"
            LOGGER.info(
                "Window %d/%d: asset=%s frames=%s-%s",
                offset + 1,
                len(candidate_windows),
                asset_label,
                window.get("start_frame"),
                window.get("end_frame"),
            )
            resolved = resolve_candidate_asset_paths(
                window=window,
                hdf5_dir=args.hdf5_dir,
                video_dir=args.video_dir,
                all_hdf5_paths=all_hdf5_paths,
                video_patterns=args.video_patterns.split(","),
                recursive_videos=args.recursive_videos,
            )
            if resolved.get("error"):
                LOGGER.warning("Skipping candidate window: %s", resolved["error"])
                frame_rows.append(error_candidate_row(window, resolved["error"]))
                continue
            hdf5_path = resolved["hdf5_path"]
            video_path = resolved["video_path"]
            selected_hdf5_paths.append(hdf5_path)
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
                    projected_in_image_ratio_threshold=args.projected_in_image_ratio_threshold,
                    strong_inside_ratio_threshold=args.strong_containment_inside_ratio_threshold,
                    acceptable_inside_ratio_threshold=args.acceptable_inside_ratio_threshold,
                    mask_tiny_area_ratio_threshold=args.mask_tiny_area_ratio_threshold,
                    overlay_dir=(args.overlay_dir or args.output_dir / "overlays")
                    if args.write_overlays
                    else None,
                    sam3_config={
                        "confidence_threshold": args.confidence_threshold,
                        "mask_threshold": args.mask_threshold,
                        "max_instances_per_query": args.max_instances_per_query,
                    },
                    candidate_window=window,
                    frames_per_window=args.frames_per_window,
                    include_window_boundaries=args.include_window_boundaries,
                )
            except Exception as exc:
                LOGGER.exception("Candidate window failed: %s", asset_label)
                frame_rows.append(
                    error_candidate_row(window, str(exc), hdf5_path, video_path)
                )
                continue
            frame_rows.extend(clip_frame_rows)
            clip_rows.append(clip_summary)
    else:
        hdf5_paths = all_hdf5_paths[args.start_clip : args.end_clip]
        if args.max_clips is not None:
            hdf5_paths = hdf5_paths[: args.max_clips]
        if not hdf5_paths:
            raise FileNotFoundError(
                f"No .hdf5 files selected under {args.hdf5_dir}; "
                f"available={len(all_hdf5_paths)}, "
                f"start_clip={args.start_clip}, end_clip={args.end_clip}"
            )
        selected_hdf5_paths = hdf5_paths
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
                    projected_in_image_ratio_threshold=args.projected_in_image_ratio_threshold,
                    strong_inside_ratio_threshold=args.strong_containment_inside_ratio_threshold,
                    acceptable_inside_ratio_threshold=args.acceptable_inside_ratio_threshold,
                    mask_tiny_area_ratio_threshold=args.mask_tiny_area_ratio_threshold,
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
    window_summaries = aggregate_window_containment_summaries(
        frame_rows,
        fail_min_strong_frames=args.containment_fail_min_strong_frames,
        fail_strong_frame_ratio=args.containment_fail_strong_frame_ratio,
    )
    write_window_summary_outputs(window_summaries, args.output_dir)
    write_json(
        {
            "hdf5_dir": str(args.hdf5_dir),
            "video_dir": str(args.video_dir),
            "sam3_model": str(args.sam3_model),
            "sample_fraction": args.sample_fraction,
            "candidate_windows": str(args.candidate_windows)
            if args.candidate_windows is not None
            else None,
            "candidate_window_mode": args.candidate_windows is not None,
            "asset_ids": sorted(asset_id_filter) if asset_id_filter else None,
            "frames_per_window": args.frames_per_window,
            "include_window_boundaries": args.include_window_boundaries,
            "start_clip": args.start_clip,
            "end_clip": args.end_clip,
            "max_clips": args.max_clips,
            "available_hdf5_count": len(all_hdf5_paths),
            "selected_hdf5_paths": [str(path) for path in selected_hdf5_paths],
            "queries": queries,
            "abnormal_inside_ratio_threshold": args.abnormal_inside_ratio_threshold,
            "projected_in_image_ratio_threshold": args.projected_in_image_ratio_threshold,
            "strong_containment_inside_ratio_threshold": (
                args.strong_containment_inside_ratio_threshold
            ),
            "acceptable_inside_ratio_threshold": args.acceptable_inside_ratio_threshold,
            "mask_tiny_area_ratio_threshold": args.mask_tiny_area_ratio_threshold,
            "write_overlays": args.write_overlays,
            "overlay_dir": str(args.overlay_dir or args.output_dir / "overlays")
            if args.write_overlays
            else None,
            "num_clips": len(clip_rows),
            "num_window_summaries": len(window_summaries),
        },
        args.output_dir / "run_manifest.json",
    )
    LOGGER.info("Wrote results under %s", args.output_dir)


def score_keypoints_against_masks(
    *,
    frame: np.ndarray,
    pixels: np.ndarray,
    joint_names: list[str],
    masks: list[Any],
    valid: np.ndarray | None = None,
    abnormal_inside_ratio_threshold: float = 1.0,
    projected_in_image_ratio_threshold: float = 0.8,
    strong_inside_ratio_threshold: float = 0.2,
    acceptable_inside_ratio_threshold: float = 0.6,
    mask_tiny_area_ratio_threshold: float = 0.0,
) -> tuple[dict[str, Any], np.ndarray | None, np.ndarray, np.ndarray]:
    """Score already-projected 2D keypoints using the canonical mask rules."""
    pixels = np.asarray(pixels, dtype=np.float64)
    if pixels.shape != (len(joint_names), 2):
        raise ValueError(
            f"pixels must have shape ({len(joint_names)}, 2); got {pixels.shape}"
        )
    finite = np.isfinite(pixels).all(axis=1)
    valid_points = finite if valid is None else np.asarray(valid, dtype=bool) & finite
    if valid_points.shape != (len(joint_names),):
        raise ValueError(
            f"valid must have shape ({len(joint_names)},); got {valid_points.shape}"
        )

    union_mask = union_instance_masks(masks, frame.shape[:2])
    rounded = np.zeros_like(pixels, dtype=np.int64)
    rounded[valid_points] = np.rint(pixels[valid_points]).astype(np.int64)
    inside = np.zeros(len(joint_names), dtype=bool)
    if union_mask is not None:
        in_bounds = (
            valid_points
            & (rounded[:, 0] >= 0)
            & (rounded[:, 0] < frame.shape[1])
            & (rounded[:, 1] >= 0)
            & (rounded[:, 1] < frame.shape[0])
        )
        if bool(np.any(in_bounds)):
            inside[in_bounds] = union_mask[
                rounded[in_bounds, 1],
                rounded[in_bounds, 0],
            ]
    else:
        in_bounds = (
            valid_points
            & (pixels[:, 0] >= 0)
            & (pixels[:, 0] < frame.shape[1])
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < frame.shape[0])
        )

    total_expected = len(joint_names)
    valid_count = int(np.sum(valid_points))
    in_image_count = int(np.sum(in_bounds))
    inside_count = int(np.sum(inside))
    missing_from_mask_count = int(valid_count - inside_count)
    expected_missing_from_mask_count = int(total_expected - inside_count)
    valid_projected_inside_ratio = safe_ratio(inside_count, valid_count)
    valid_projected_missing_ratio = safe_ratio(missing_from_mask_count, valid_count)
    keypoint_inside_ratio = safe_ratio(inside_count, total_expected)
    keypoint_missing_ratio = safe_ratio(
        expected_missing_from_mask_count,
        total_expected,
    )
    projected_keypoints_in_image_ratio = safe_ratio(
        in_image_count,
        total_expected,
    )
    abnormal_frame = (
        keypoint_inside_ratio is not None
        and keypoint_inside_ratio < abnormal_inside_ratio_threshold
    )
    mask_area = int(np.sum(union_mask)) if union_mask is not None else 0
    mask_area_ratio = safe_ratio(mask_area, frame.shape[0] * frame.shape[1])
    hand_mask_present = bool(union_mask is not None and mask_area > 0)
    hand_mask_tiny = bool(
        hand_mask_present
        and mask_area_ratio is not None
        and mask_area_ratio <= mask_tiny_area_ratio_threshold
    )
    containment_verdict, reason = classify_containment_frame(
        projected_in_image_ratio=projected_keypoints_in_image_ratio,
        inside_ratio=keypoint_inside_ratio,
        hand_mask_present=hand_mask_present,
        hand_mask_tiny=hand_mask_tiny,
        projected_in_image_ratio_threshold=projected_in_image_ratio_threshold,
        strong_inside_ratio_threshold=strong_inside_ratio_threshold,
        acceptable_inside_ratio_threshold=acceptable_inside_ratio_threshold,
    )
    inside_indices = np.flatnonzero(inside).tolist()
    missing_indices = np.flatnonzero(~inside).tolist()
    valid_missing_indices = np.flatnonzero(valid_points & ~inside).tolist()
    invalid_indices = np.flatnonzero(~valid_points).tolist()
    metrics = {
        "sampled_keypoints": total_expected,
        "valid_projected_keypoints": valid_count,
        "projected_keypoints_in_image": in_image_count,
        "projected_keypoints_in_image_ratio": projected_keypoints_in_image_ratio,
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
        "projected_in_image_ratio_threshold": projected_in_image_ratio_threshold,
        "strong_containment_inside_ratio_threshold": strong_inside_ratio_threshold,
        "acceptable_inside_ratio_threshold": acceptable_inside_ratio_threshold,
        "mask_tiny_area_ratio_threshold": mask_tiny_area_ratio_threshold,
        "keypoints_inside_hand_mask_ratio": keypoint_inside_ratio,
        "containment_verdict": containment_verdict,
        "reason": reason,
        "mask_instance_count": len(masks),
        "mask_area": mask_area,
        "mask_area_ratio": mask_area_ratio,
        "hand_mask_present": hand_mask_present,
        "hand_mask_area_ratio": mask_area_ratio,
        "hand_mask_tiny": hand_mask_tiny,
        "hand_mask_touches_border": bool(mask_touches_border(union_mask)),
        "sam3_categories": sorted(
            {str(getattr(mask, "category", "")) for mask in masks}
        ),
    }
    return metrics, union_mask, valid_points, inside


def process_clip(
    hdf5_path: Path,
    video_path: Path,
    segmenter: SAM3Segmenter,
    queries: list[str],
    sample_fraction: float,
    max_sampled_frames: int | None,
    projection_mode: str,
    abnormal_inside_ratio_threshold: float,
    projected_in_image_ratio_threshold: float,
    strong_inside_ratio_threshold: float,
    acceptable_inside_ratio_threshold: float,
    mask_tiny_area_ratio_threshold: float,
    overlay_dir: Path | None,
    sam3_config: dict[str, Any],
    candidate_window: dict[str, Any] | None = None,
    frames_per_window: int = 3,
    include_window_boundaries: bool = True,
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

    asset_id = str(candidate_window.get("asset_id")) if candidate_window and candidate_window.get("asset_id") else clip_id_from_path(hdf5_path)
    hand_side = normalize_hand_side(candidate_window.get("hand_side") if candidate_window else None)
    if hand_side in {"left", "right"}:
        indices = [idx for idx, name in enumerate(joint_names) if name.startswith(hand_side)]
        joint_names = [joint_names[idx] for idx in indices]
        points = points[:, indices, :]
    if candidate_window is not None:
        sampled_frames = sample_candidate_window_frames(
            candidate_window,
            num_frames=num_frames,
            frames_per_window=frames_per_window,
            include_boundaries=include_window_boundaries,
        )
    else:
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
        projected = project_frame_points(
            points[frame_idx],
            camera_transforms[frame_idx],
            intrinsics,
            resolved_projection_mode,
        )
        pixels = projected["pixels"]
        containment, union_mask, valid, inside = score_keypoints_against_masks(
            frame=frame,
            pixels=pixels,
            joint_names=joint_names,
            masks=masks,
            valid=projected["valid"],
            abnormal_inside_ratio_threshold=abnormal_inside_ratio_threshold,
            projected_in_image_ratio_threshold=projected_in_image_ratio_threshold,
            strong_inside_ratio_threshold=strong_inside_ratio_threshold,
            acceptable_inside_ratio_threshold=acceptable_inside_ratio_threshold,
            mask_tiny_area_ratio_threshold=mask_tiny_area_ratio_threshold,
        )
        row = {
            "clip_id": clip_id_from_path(hdf5_path),
            "asset_id": asset_id,
            "episode_idx": candidate_window.get("episode_idx") if candidate_window else None,
            "hdf5_path": str(hdf5_path),
            "video_path": str(video_path),
            "frame_idx": int(frame_idx),
            "projection_mode": resolved_projection_mode,
            "projection_mode_used": resolved_projection_mode,
            "image_width": int(frame.shape[1]),
            "image_height": int(frame.shape[0]),
            "hand_side": hand_side,
            **containment,
        }
        if candidate_window is not None:
            row.update(candidate_window_metadata(candidate_window))
        if overlay_dir is not None:
            overlay_clip_id = (
                f"{asset_id}_window_"
                f"{candidate_window.get('start_frame')}_{candidate_window.get('end_frame')}"
                if candidate_window is not None
                else clip_id_from_path(hdf5_path)
            )
            overlay_path = write_overlay_image(
                frame=frame,
                mask=union_mask,
                pixels=pixels,
                valid=valid,
                inside=inside,
                joint_names=joint_names,
                clip_id=overlay_clip_id,
                frame_idx=frame_idx,
                output_dir=overlay_dir,
            )
            row["overlay_path"] = str(overlay_path)
        frame_rows.append(row)
        totals["sampled_frames"] += 1
        totals["total_expected_keypoints"] += containment["sampled_keypoints"]
        totals["valid_projected_keypoints"] += containment[
            "valid_projected_keypoints"
        ]
        totals["inside_keypoints"] += containment["inside_keypoints"]
        totals["missing_from_mask_keypoints"] += containment[
            "missing_from_mask_keypoints"
        ]
        totals["expected_missing_from_mask_keypoints"] += containment[
            "expected_missing_from_mask_keypoints"
        ]
        totals["abnormal_frames"] += int(containment["abnormal_frame"])
        totals["frames_with_mask"] += int(containment["hand_mask_present"])
        totals["frames_without_mask"] += int(
            not containment["hand_mask_present"]
        )
        mode_counts[resolved_projection_mode] += 1

    ratios = [row["keypoint_inside_ratio"] for row in frame_rows]
    valid_ratios = [
        row["valid_projected_inside_ratio"]
        for row in frame_rows
        if row["valid_projected_inside_ratio"] is not None
    ]
    clip_summary = {
        "clip_id": clip_id_from_path(hdf5_path),
        "asset_id": asset_id,
        "episode_idx": candidate_window.get("episode_idx") if candidate_window else None,
        "hdf5_path": str(hdf5_path),
        "video_path": str(video_path),
        "sample_fraction": sample_fraction,
        "candidate_window_mode": candidate_window is not None,
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
    if candidate_window is not None:
        clip_summary.update(candidate_window_metadata(candidate_window))
    return frame_rows, clip_summary


def parse_asset_ids(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def load_candidate_windows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"candidate windows must be a JSON list: {path}")
    windows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"candidate window {index} is not an object")
        windows.append(row)
    return windows


def filter_candidate_windows(
    windows: list[dict[str, Any]],
    asset_ids: set[str],
) -> list[dict[str, Any]]:
    if not asset_ids:
        return windows
    return [
        window
        for window in windows
        if window.get("asset_id") is not None
        and str(window.get("asset_id")) in asset_ids
    ]


def sample_candidate_window_frames(
    window: dict[str, Any],
    num_frames: int,
    frames_per_window: int,
    include_boundaries: bool,
) -> list[int]:
    start = int(window["start_frame"])
    end = int(window["end_frame"])
    if num_frames <= 0:
        return []
    start = max(0, min(start, num_frames - 1))
    end = max(0, min(end, num_frames - 1))
    if end < start:
        start, end = end, start
    if include_boundaries:
        frame_indices = {start, end}
        if end - start > 1 and frames_per_window > 0:
            interior = np.linspace(
                start + 1,
                end - 1,
                min(frames_per_window, end - start - 1),
                dtype=int,
            ).tolist()
            frame_indices.update(interior)
        return sorted(frame_indices)
    count = min(max(1, frames_per_window), end - start + 1)
    return sorted(set(np.linspace(start, end, count, dtype=int).tolist()))


def resolve_candidate_asset_paths(
    window: dict[str, Any],
    hdf5_dir: Path,
    video_dir: Path,
    all_hdf5_paths: list[Path],
    video_patterns: list[str],
    recursive_videos: bool,
) -> dict[str, Any]:
    asset_id = window.get("asset_id")
    hdf5_path: Path | None = None
    if asset_id is not None:
        hdf5_path = find_hdf5_path_for_asset(str(asset_id), hdf5_dir)
        if hdf5_path is None:
            return {"error": f"HDF5 not found for asset_id={asset_id}"}
    else:
        episode_idx = window.get("episode_idx")
        if episode_idx is None:
            return {"error": "candidate window has neither asset_id nor episode_idx"}
        try:
            hdf5_path = all_hdf5_paths[int(episode_idx)]
        except (IndexError, TypeError, ValueError):
            return {
                "error": (
                    "candidate window has missing asset_id and episode_idx "
                    f"cannot map safely: {episode_idx}"
                )
            }

    video_path = find_video_path_for_asset(
        asset_id=clip_id_from_path(hdf5_path),
        hdf5_path=hdf5_path,
        video_dir=video_dir,
        patterns=video_patterns,
        recursive=recursive_videos,
    )
    if video_path is None:
        return {"error": f"matching video not found for {hdf5_path.name}"}
    return {"hdf5_path": hdf5_path, "video_path": video_path, "error": None}


def find_hdf5_path_for_asset(asset_id: str, hdf5_dir: Path) -> Path | None:
    for filename in (
        f"{asset_id}_hdf5.hdf5",
        f"{asset_id}_hdf5.h5",
        f"{asset_id}.hdf5",
        f"{asset_id}.h5",
    ):
        candidate = hdf5_dir / filename
        if candidate.exists():
            return candidate
    return None


def find_video_path_for_asset(
    asset_id: str,
    hdf5_path: Path,
    video_dir: Path,
    patterns: list[str],
    recursive: bool,
) -> Path | None:
    for filename in (f"{asset_id}_video.mp4", f"{asset_id}.mp4"):
        candidate = video_dir / filename
        if candidate.exists():
            return candidate
        if recursive:
            matches = sorted(video_dir.rglob(filename))
            if matches:
                return matches[0]
    return find_video_path(hdf5_path, video_dir, patterns, recursive=recursive)


def normalize_hand_side(value: Any) -> str:
    side = str(value or "both").lower()
    return side if side in {"left", "right", "both"} else "both"


def candidate_window_metadata(window: dict[str, Any]) -> dict[str, Any]:
    return {
        "window_start_frame": window.get("start_frame"),
        "window_end_frame": window.get("end_frame"),
        "seed_run_start": window.get("seed_run_start"),
        "seed_run_end": window.get("seed_run_end"),
        "seed_run_frames": window.get("seed_run_frames"),
        "source_window_source": window.get("window_source"),
        "source_review_type": window.get("review_type"),
        "source_trigger_reason": window.get("trigger_reason"),
        "source_priority": window.get("priority"),
        "source_needs_manual_review": window.get("needs_manual_review"),
        "source_sam3_containment_eligible": window.get("sam3_containment_eligible"),
    }


def classify_containment_frame(
    projected_in_image_ratio: float | None,
    inside_ratio: float | None,
    hand_mask_present: bool,
    hand_mask_tiny: bool,
    projected_in_image_ratio_threshold: float = 0.8,
    strong_inside_ratio_threshold: float = 0.2,
    acceptable_inside_ratio_threshold: float = 0.6,
) -> tuple[str, str]:
    if (
        projected_in_image_ratio is None
        or projected_in_image_ratio < projected_in_image_ratio_threshold
    ):
        return "projection_review", "insufficient_projection_evidence"
    if not hand_mask_present or hand_mask_tiny:
        return "mask_missing_or_tiny_review", "hand mask missing or tiny"
    if inside_ratio is None:
        return "projection_review", "insufficient_projection_evidence"
    if inside_ratio <= strong_inside_ratio_threshold:
        return (
            "strong_containment_mismatch",
            "projected keypoints are in image but mostly outside hand mask",
        )
    if inside_ratio < acceptable_inside_ratio_threshold:
        return "containment_review", "partial keypoint-mask mismatch"
    return "likely_visible_ok", "keypoints mostly consistent with hand mask"


def classify_containment_rows(
    rows: list[dict[str, Any]],
    projected_in_image_ratio_threshold: float = 0.8,
    strong_inside_ratio_threshold: float = 0.2,
    acceptable_inside_ratio_threshold: float = 0.6,
    mask_tiny_area_ratio_threshold: float = 0.0,
) -> list[dict[str, Any]]:
    classified = []
    for row in rows:
        next_row = dict(row)
        hand_mask_present = bool(next_row.get("hand_mask_present", False))
        mask_area_ratio = next_row.get("hand_mask_area_ratio")
        if mask_area_ratio is None:
            mask_area_ratio = next_row.get("mask_area_ratio")
        hand_mask_tiny = bool(
            hand_mask_present
            and mask_area_ratio is not None
            and float(mask_area_ratio) <= mask_tiny_area_ratio_threshold
        )
        projected_ratio = ratio_value(
            next_row.get("projected_keypoints_in_image_ratio")
        )
        inside_ratio = ratio_value(
            next_row.get("keypoints_inside_hand_mask_ratio")
            if "keypoints_inside_hand_mask_ratio" in next_row
            else next_row.get("keypoint_inside_ratio")
        )
        verdict, reason = classify_containment_frame(
            projected_in_image_ratio=projected_ratio,
            inside_ratio=inside_ratio,
            hand_mask_present=hand_mask_present,
            hand_mask_tiny=hand_mask_tiny,
            projected_in_image_ratio_threshold=projected_in_image_ratio_threshold,
            strong_inside_ratio_threshold=strong_inside_ratio_threshold,
            acceptable_inside_ratio_threshold=acceptable_inside_ratio_threshold,
        )
        next_row["containment_verdict"] = verdict
        next_row["reason"] = reason
        next_row["hand_mask_tiny"] = hand_mask_tiny
        next_row["projected_in_image_ratio_threshold"] = (
            projected_in_image_ratio_threshold
        )
        next_row["strong_containment_inside_ratio_threshold"] = (
            strong_inside_ratio_threshold
        )
        next_row["acceptable_inside_ratio_threshold"] = (
            acceptable_inside_ratio_threshold
        )
        next_row["mask_tiny_area_ratio_threshold"] = mask_tiny_area_ratio_threshold
        classified.append(next_row)
    return classified


def aggregate_window_containment_summaries(
    rows: list[dict[str, Any]],
    fail_min_strong_frames: int = 3,
    fail_strong_frame_ratio: float = 0.6,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row.get("asset_id") or row.get("clip_id"),
            row.get("episode_idx"),
            row.get("window_start_frame"),
            row.get("window_end_frame"),
            row.get("hand_side"),
        )
        groups.setdefault(key, []).append(row)

    summaries = []
    for group_rows in groups.values():
        summaries.append(
            summarize_window_containment(
                group_rows,
                fail_min_strong_frames=fail_min_strong_frames,
                fail_strong_frame_ratio=fail_strong_frame_ratio,
            )
        )
    return sorted(
        summaries,
        key=lambda item: (
            str(item.get("asset_id") or ""),
            item.get("window_start_frame")
            if item.get("window_start_frame") is not None
            else -1,
            str(item.get("hand_side") or ""),
        ),
    )


def summarize_window_containment(
    rows: list[dict[str, Any]],
    fail_min_strong_frames: int,
    fail_strong_frame_ratio: float,
) -> dict[str, Any]:
    first = rows[0]
    verdict_counts = Counter(row.get("containment_verdict") for row in rows)
    sampled_count = len(rows)
    strong_count = int(verdict_counts["strong_containment_mismatch"])
    containment_review_count = int(verdict_counts["containment_review"])
    projection_review_count = int(verdict_counts["projection_review"])
    acceptable_count = int(verdict_counts["likely_visible_ok"])
    mask_missing_count = int(verdict_counts["mask_missing_or_tiny_review"])
    strong_ratio = safe_ratio(strong_count, sampled_count) or 0.0
    inside_values = [
        value
        for value in (
            ratio_value(
                row.get("keypoints_inside_hand_mask_ratio")
                if "keypoints_inside_hand_mask_ratio" in row
                else row.get("keypoint_inside_ratio")
            )
            for row in rows
        )
        if value is not None
    ]
    projected_values = [
        value
        for value in (
            ratio_value(row.get("projected_keypoints_in_image_ratio"))
            for row in rows
        )
        if value is not None
    ]
    inside_ratio_mean = float(np.mean(inside_values)) if inside_values else None
    projected_ratio_mean = float(np.mean(projected_values)) if projected_values else None
    window_verdict, reason = window_containment_verdict(
        sampled_count=sampled_count,
        strong_fail_frame_count=strong_count,
        strong_fail_frame_ratio=strong_ratio,
        review_frame_count=containment_review_count,
        projection_review_frame_count=projection_review_count,
        acceptable_frame_count=acceptable_count,
        mask_missing_or_tiny_frame_count=mask_missing_count,
        inside_ratio_mean=inside_ratio_mean,
        projected_in_image_ratio_mean=projected_ratio_mean,
        fail_min_strong_frames=fail_min_strong_frames,
        fail_strong_frame_ratio=fail_strong_frame_ratio,
        source_review_type=first.get("source_review_type"),
        source_trigger_reason=first.get("source_trigger_reason"),
        source_sam3_containment_eligible=first.get(
            "source_sam3_containment_eligible"
        ),
        source_needs_manual_review=first.get("source_needs_manual_review"),
    )
    start_frame = first.get("window_start_frame")
    end_frame = first.get("window_end_frame")
    if start_frame is None:
        start_frame = min(int(row.get("frame_idx", 0)) for row in rows)
    if end_frame is None:
        end_frame = max(int(row.get("frame_idx", 0)) for row in rows)
    return {
        "asset_id": first.get("asset_id") or first.get("clip_id"),
        "episode_idx": first.get("episode_idx"),
        "hand_side": first.get("hand_side"),
        "window_start_frame": start_frame,
        "window_end_frame": end_frame,
        "seed_run_start": first.get("seed_run_start"),
        "seed_run_end": first.get("seed_run_end"),
        "seed_run_frames": first.get("seed_run_frames"),
        "source_window_source": first.get("source_window_source"),
        "source_review_type": first.get("source_review_type"),
        "source_trigger_reason": first.get("source_trigger_reason"),
        "source_priority": first.get("source_priority"),
        "source_needs_manual_review": first.get("source_needs_manual_review"),
        "source_sam3_containment_eligible": first.get(
            "source_sam3_containment_eligible"
        ),
        "sampled_frame_count": sampled_count,
        "sampled_frame_indices": sorted(
            int(row["frame_idx"]) for row in rows if row.get("frame_idx") is not None
        ),
        "strong_fail_frame_count": strong_count,
        "strong_fail_frame_ratio": strong_ratio,
        "review_frame_count": containment_review_count,
        "projection_review_frame_count": projection_review_count,
        "acceptable_frame_count": acceptable_count,
        "mask_missing_or_tiny_frame_count": mask_missing_count,
        "inside_ratio_min": min(inside_values) if inside_values else None,
        "inside_ratio_mean": inside_ratio_mean,
        "inside_ratio_max": max(inside_values) if inside_values else None,
        "projected_in_image_ratio_min": min(projected_values)
        if projected_values
        else None,
        "projected_in_image_ratio_mean": projected_ratio_mean,
        "projected_in_image_ratio_max": max(projected_values)
        if projected_values
        else None,
        "window_containment_verdict": window_verdict,
        "reason": reason,
    }


def window_containment_verdict(
    sampled_count: int,
    strong_fail_frame_count: int,
    strong_fail_frame_ratio: float,
    review_frame_count: int,
    projection_review_frame_count: int,
    acceptable_frame_count: int,
    mask_missing_or_tiny_frame_count: int,
    inside_ratio_mean: float | None = None,
    projected_in_image_ratio_mean: float | None = None,
    fail_min_strong_frames: int = 3,
    fail_strong_frame_ratio: float = 0.6,
    source_review_type: Any = None,
    source_trigger_reason: Any = None,
    source_sam3_containment_eligible: Any = None,
    source_needs_manual_review: Any = None,
) -> tuple[str, str]:
    if contains_metadata_value(source_review_type, "side_view_manual_review"):
        if contains_any_metadata_value(
            source_trigger_reason,
            {
                "acceleration_seed",
                "displacement_seed",
                "multi_signal_seed",
                "extreme_rotation_delta",
            },
        ):
            return (
                "mixed_review",
                "side-view hand orientation makes SAM3 containment unreliable; requires manual review",
            )
        return (
            "side_view_manual_review",
            "side-view hand orientation makes SAM3 containment unreliable; requires manual review",
        )
    if (
        contains_metadata_value(source_review_type, "rotation_manual_review")
        or is_false_value(source_sam3_containment_eligible)
        or is_true_value(source_needs_manual_review)
    ):
        return (
            "rotation_manual_review",
            "extreme rotation makes SAM3 containment unreliable; requires manual review",
        )
    mean_clean_fail = (
        projected_in_image_ratio_mean is not None
        and inside_ratio_mean is not None
        and projected_in_image_ratio_mean >= 0.8
        and inside_ratio_mean <= 0.2
        and projection_review_frame_count == 0
    )
    if strong_fail_frame_count >= fail_min_strong_frames:
        return "containment_fail", "sustained strong keypoint-mask mismatch"
    if strong_fail_frame_ratio >= fail_strong_frame_ratio:
        return "containment_fail", "sustained strong keypoint-mask mismatch"
    if mean_clean_fail:
        return "containment_fail", "clean window-level containment mismatch"
    if projection_review_frame_count > 0 and strong_fail_frame_count == 0:
        return "projection_review", "only insufficient projection evidence"
    if acceptable_frame_count > sampled_count / 2 and strong_fail_frame_count == 0:
        return "acceptable_flagged", "majority frames are visually acceptable"
    if strong_fail_frame_count > 0:
        return "mixed_review", "mixed containment evidence with strong outliers"
    nonzero_classes = sum(
        count > 0
        for count in (
            strong_fail_frame_count,
            review_frame_count,
            projection_review_frame_count,
            acceptable_frame_count,
            mask_missing_or_tiny_frame_count,
        )
    )
    if nonzero_classes > 1:
        return "mixed_review", "mixed containment evidence"
    return "review", "uncertain containment evidence"


def contains_metadata_value(value: Any, expected: str) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value == expected
    if isinstance(value, (list, tuple, set)):
        return any(contains_metadata_value(item, expected) for item in value)
    return False


def contains_any_metadata_value(value: Any, expected: set[str]) -> bool:
    return any(contains_metadata_value(value, item) for item in expected)


def is_false_value(value: Any) -> bool:
    if isinstance(value, bool):
        return not value
    if isinstance(value, str):
        return value.strip().lower() in {"false", "0", "no"}
    return False


def is_true_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return False


def ratio_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return None
    return ratio if math.isfinite(ratio) else None


def mask_touches_border(mask: np.ndarray | None) -> bool:
    if mask is None:
        return False
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.size == 0 or not bool(np.any(mask_bool)):
        return False
    return bool(
        np.any(mask_bool[0, :])
        or np.any(mask_bool[-1, :])
        or np.any(mask_bool[:, 0])
        or np.any(mask_bool[:, -1])
    )


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


def write_combined_overlay_image(
    *,
    frame: np.ndarray,
    hands: dict[str, dict[str, Any]],
    clip_id: str,
    frame_idx: int,
    output_dir: Path,
) -> Path:
    """Write a review overlay with both hands and side-consistent colors."""
    import cv2

    side_colors_rgb = {
        "left": (40, 220, 90),
        "right": (40, 130, 255),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(
        np.asarray(frame, dtype=np.uint8),
        cv2.COLOR_RGB2BGR,
    )
    height, width = image_bgr.shape[:2]

    for hand_side in ("left", "right"):
        hand = hands.get(hand_side)
        if hand is None:
            continue
        pixels = np.asarray(hand["pixels"], dtype=np.float64)
        valid = np.asarray(hand["valid"], dtype=bool)
        inside = np.asarray(hand["inside"], dtype=bool)
        joint_names = [str(name) for name in hand["joint_names"]]
        color_rgb = side_colors_rgb[hand_side]
        color_bgr = color_rgb[::-1]
        index_by_name = {
            joint_name: index for index, joint_name in enumerate(joint_names)
        }

        for parent, child in derive_finger_bones(joint_names):
            parent_index = index_by_name[parent]
            child_index = index_by_name[child]
            if not valid[parent_index] or not valid[child_index]:
                continue
            endpoints = pixels[[parent_index, child_index]]
            if not np.isfinite(endpoints).all():
                continue
            parent_xy = tuple(np.rint(endpoints[0]).astype(int))
            child_xy = tuple(np.rint(endpoints[1]).astype(int))
            cv2.line(
                image_bgr,
                parent_xy,
                child_xy,
                color_bgr,
                2,
                cv2.LINE_AA,
            )

        for index, (x_raw, y_raw) in enumerate(pixels):
            if not valid[index] or not np.isfinite([x_raw, y_raw]).all():
                continue
            x = int(round(float(x_raw)))
            y = int(round(float(y_raw)))
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            cv2.circle(image_bgr, (x, y), 4, color_bgr, -1, cv2.LINE_AA)
            outline_bgr = (0, 0, 0) if inside[index] else (45, 45, 255)
            cv2.circle(image_bgr, (x, y), 5, outline_bgr, 1, cv2.LINE_AA)

    # The SAM3 mask is shared across queried hands, so the combined view avoids
    # assigning that union mask a misleading side-specific color.
    legend_height = min(48, height)
    legend_layer = image_bgr.copy()
    cv2.rectangle(legend_layer, (0, 0), (width, legend_height), (0, 0, 0), -1)
    image_bgr = cv2.addWeighted(legend_layer, 0.72, image_bgr, 0.28, 0.0)
    cv2.putText(
        image_bgr,
        f"{clip_id} source_frame={int(frame_idx)}",
        (7, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    legend_items = (
        ("LEFT=green", side_colors_rgb["left"], 7),
        ("RIGHT=blue", side_colors_rgb["right"], 105),
    )
    for label, color_rgb, x in legend_items:
        color_bgr = color_rgb[::-1]
        cv2.rectangle(image_bgr, (x, 27), (x + 12, 39), color_bgr, -1)
        cv2.putText(
            image_bgr,
            label,
            (x + 17, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        image_bgr,
        "red ring=outside mask",
        (220, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (80, 80, 255),
        1,
        cv2.LINE_AA,
    )

    output_path = output_dir / f"{clip_id}_frame_{int(frame_idx):06d}.png"
    if not cv2.imwrite(str(output_path), image_bgr):
        raise OSError(f"failed to write combined overlay: {output_path}")
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


def error_candidate_row(
    window: dict[str, Any],
    error: str,
    hdf5_path: Path | None = None,
    video_path: Path | None = None,
) -> dict[str, Any]:
    row = {
        "asset_id": window.get("asset_id"),
        "episode_idx": window.get("episode_idx"),
        "hdf5_path": str(hdf5_path) if hdf5_path is not None else None,
        "video_path": str(video_path) if video_path is not None else None,
        "frame_idx": None,
        "hand_side": normalize_hand_side(window.get("hand_side")),
        "hand_mask_present": False,
        "hand_mask_area_ratio": None,
        "hand_mask_touches_border": False,
        "projected_keypoints_in_image_ratio": None,
        "keypoints_inside_hand_mask_ratio": None,
        "containment_verdict": "projection_review",
        "reason": f"candidate window failed: {error}",
        "error": error,
    }
    row.update(candidate_window_metadata(window))
    return row


def write_window_summary_outputs(
    summaries: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    json_path = output_dir / "window_keypoint_containment_summary.json"
    write_json(summaries, json_path)
    parquet_path = output_dir / "window_keypoint_containment_summary.parquet"
    try:
        import pandas as pd

        pd.DataFrame(summaries).to_parquet(parquet_path, index=False)
    except Exception as exc:
        LOGGER.warning("Could not write %s: %s", parquet_path, exc)


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(value), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    main()
