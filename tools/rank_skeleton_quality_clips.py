"""Rank supplier HDF5 clips by skeleton QC quality.

This is an offline precheck helper. It consumes HDF5 only, does not load SAM3 or
any annotation model, and ranks clips by the existing skeleton_quality_score
signals plus supplier low-quality markers when present.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from precheck.adapters.supplier_hdf5 import load_supplier_hdf5_clip
from precheck.checks.skeleton_quality_score import SkeletonQualityScoreCheck


DEFAULT_SKELETON_CONFIG: dict[str, Any] = {
    "decision_mode": "temporal_triage",
    "joint_angle_change_deg_max_threshold": 10.0,
    "rotation_delta_max_threshold": 0.45,
    "joint_acceleration_m_s2_max_threshold": 15.0,
    "joint_displacement_m_max_threshold": 0.05,
    "hard_exceeded_metric_count": 3,
    "strong_acceleration_ratio": 2.5,
    "strong_displacement_ratio": 1.8,
    "rotation_mask_review_ratio": 1.0,
    "promote_sustained_review": True,
    "sustained_review_min_frames": 6,
    "reject_missing_keypoints": True,
    "reject_low_quality_hand": False,
    "allowed_missing_keypoints_per_hand": 0,
    "pass_threshold": 0.90,
}


def asset_id_from_path(path: Path) -> str:
    match = re.search(r"(\d+)", path.stem)
    return match.group(1) if match else path.stem


def hdf5_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    files = [
        path
        for suffix in ("*.hdf5", "*.h5")
        for path in input_path.rglob(suffix)
        if path.is_file()
    ]
    return sorted(files)


def run_clip(path: Path, fps: float | None) -> dict[str, Any]:
    asset_id = asset_id_from_path(path)
    clip = load_supplier_hdf5_clip(path, episode_idx=0, fps=fps)
    check = SkeletonQualityScoreCheck(DEFAULT_SKELETON_CONFIG)
    results = check.run(clip)
    frame_results = [result for result in results if result.frame_idx >= 0]
    summary = next((result.metrics for result in results if result.frame_idx == -1), {})

    num_frames = int(summary.get("num_frames", len(frame_results)))
    rotation_review = 0
    mask_review = 0
    sustained_promoted = 0
    low_quality = 0
    max_metrics = {
        "joint_angle_change_deg_max": 0.0,
        "rotation_delta_max": 0.0,
        "joint_acceleration_m_s2_max": 0.0,
        "joint_displacement_m_max": 0.0,
    }
    for result in frame_results:
        metrics = result.metrics
        rotation_review += int(metrics.get("needs_rotation_mask_review") == 1.0)
        mask_review += int(metrics.get("needs_mask_containment_review") == 1.0)
        sustained_promoted += int(metrics.get("sustained_review_promoted") == 1.0)
        low_quality += int(
            metrics.get("low_quality_hand_left") == 1.0
            or metrics.get("low_quality_hand_right") == 1.0
        )
        for name in max_metrics:
            value = metrics.get(name)
            if isinstance(value, int | float) and value > max_metrics[name]:
                max_metrics[name] = float(value)

    suspect_ratio = float(summary.get("suspect_ratio", 0.0))
    review_ratio = float(summary.get("review_ratio", 0.0))
    invalid_ratio = float(summary.get("invalid_ratio", 0.0))
    rotation_review_ratio = rotation_review / num_frames if num_frames else 0.0
    low_quality_ratio = low_quality / num_frames if num_frames else 0.0

    # Lower is better. Suspect/invalid are hard penalties; review/rotation and
    # supplier low-quality are softer risk signals that approximate out-of-view
    # or mask-containment risk until SAM3 is available.
    rank_score = (
        100.0 * suspect_ratio
        + 100.0 * invalid_ratio
        + 35.0 * review_ratio
        + 25.0 * rotation_review_ratio
        + 20.0 * low_quality_ratio
    )
    return {
        "asset_id": asset_id,
        "path": str(path),
        "num_frames": num_frames,
        "rank_score": rank_score,
        "pass_ratio": float(summary.get("pass_ratio", 0.0)),
        "mean_skeleton_score": float(summary.get("mean_skeleton_score", 0.0)),
        "suspect_ratio": suspect_ratio,
        "review_ratio": review_ratio,
        "invalid_ratio": invalid_ratio,
        "rotation_review_ratio": rotation_review_ratio,
        "mask_review_ratio": mask_review / num_frames if num_frames else 0.0,
        "supplier_low_quality_ratio": low_quality_ratio,
        "count_suspect": int(summary.get("count_suspect", 0)),
        "count_review": int(summary.get("count_review", 0)),
        "count_invalid": int(summary.get("count_invalid", 0)),
        "count_rotation_review": rotation_review,
        "count_mask_review": mask_review,
        "count_sustained_promoted": sustained_promoted,
        "count_supplier_low_quality": low_quality,
        **{f"max_{name}": value for name, value in max_metrics.items()},
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/xjgt_skeleton_rank"))
    parser.add_argument("--fps", type=float, default=29.97)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-clips", type=int)
    args = parser.parse_args()

    files = hdf5_files(args.hdf5_dir)
    if args.max_clips is not None:
        files = files[: args.max_clips]
    if not files:
        raise FileNotFoundError(f"No HDF5 files found under {args.hdf5_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {path.name}", flush=True)
        try:
            rows.append(run_clip(path, args.fps))
        except Exception as exc:  # Keep batch ranking robust.
            failures.append({"path": str(path), "error": str(exc)})

    rows.sort(
        key=lambda row: (
            row["rank_score"],
            row["suspect_ratio"],
            row["review_ratio"],
            row["rotation_review_ratio"],
            row["supplier_low_quality_ratio"],
            -row["pass_ratio"],
        )
    )
    top_rows = rows[: args.top_k]

    (args.output_dir / "skeleton_quality_ranking.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n"
    )
    (args.output_dir / "top_skeleton_quality_clips.json").write_text(
        json.dumps(top_rows, indent=2, ensure_ascii=False) + "\n"
    )
    (args.output_dir / "ranking_failures.json").write_text(
        json.dumps(failures, indent=2, ensure_ascii=False) + "\n"
    )
    write_csv(rows, args.output_dir / "skeleton_quality_ranking.csv")
    write_csv(top_rows, args.output_dir / "top_skeleton_quality_clips.csv")

    print("\nTop clips:")
    for rank, row in enumerate(top_rows, start=1):
        print(
            f"{rank:02d}. {row['asset_id']} "
            f"score={row['rank_score']:.3f} "
            f"suspect={row['suspect_ratio']:.3f} "
            f"review={row['review_ratio']:.3f} "
            f"rotation_review={row['rotation_review_ratio']:.3f} "
            f"low_quality={row['supplier_low_quality_ratio']:.3f}"
        )
    if failures:
        print(f"\nSkipped {len(failures)} files; see {args.output_dir / 'ranking_failures.json'}")


if __name__ == "__main__":
    main()
