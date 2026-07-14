#!/usr/bin/env python3
"""Probe supplier HDF5 keypoint temporal metrics without video decode."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from precheck.config import KeypointTemporalConfig, PrecheckConfig
from precheck.runner import PrecheckRunner
from precheck.adapters import load_supplier_hdf5_clip
from qc_common.io import write_dataframe


def _expand_metrics(results) -> pd.DataFrame:
    rows = []
    for result in results:
        base = {
            "episode_idx": result.episode_idx,
            "frame_idx": result.frame_idx,
            "check": result.check,
        }
        for metric, value in result.metrics.items():
            rows.append({**base, "metric": metric, "value": value})
    return pd.DataFrame(rows)


def _quality_overlap(metric_df: pd.DataFrame) -> list[dict]:
    if metric_df.empty:
        return []
    quality = metric_df[metric_df["metric"] == "quality_hand_low_fraction"][
        ["episode_idx", "frame_idx", "value"]
    ].rename(columns={"value": "quality_low"})
    rows = []
    core_metrics = [
        "joint_velocity_m_s_p95",
        "joint_acceleration_m_s2_p95",
        "bone_length_change_m_p95",
        "displacement_2d_px_p95",
    ]
    for metric in core_metrics:
        metric_rows = metric_df[metric_df["metric"] == metric]
        if metric_rows.empty:
            continue
        threshold = metric_rows["value"].quantile(0.9)
        merged = metric_rows.merge(quality, on=["episode_idx", "frame_idx"], how="left")
        spikes = merged[merged["value"] >= threshold]
        rows.append(
            {
                "metric": metric,
                "top_decile_threshold": float(threshold),
                "spike_frames": int(len(spikes)),
                "spike_frames_supplier_low_quality": int(
                    (spikes["quality_low"].fillna(0.0) > 0).sum()
                ),
                "supplier_low_quality_overlap_fraction": float(
                    (spikes["quality_low"].fillna(0.0) > 0).mean()
                )
                if len(spikes)
                else 0.0,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run keypoint_temporal on supplier HDF5 clips and dump metric distributions"
    )
    parser.add_argument("hdf5", nargs="+", type=Path, help="Supplier HDF5 episode files")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/keypoint_temporal_probe"))
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    clips = [
        load_supplier_hdf5_clip(path, episode_idx=idx, fps=args.fps)
        for idx, path in enumerate(args.hdf5)
    ]
    config = PrecheckConfig(
        output_dir=args.output_dir,
        enabled_checks=["keypoint_temporal"],
        overwrite=True,
        fps=args.fps,
        keypoint_temporal=KeypointTemporalConfig(fps=args.fps),
    )
    results = PrecheckRunner(config).run(clips)

    metric_df = _expand_metrics(results)
    metric_path = write_dataframe(metric_df, args.output_dir / "metric_long.parquet")
    distribution = (
        metric_df.groupby("metric")["value"]
        .describe(percentiles=[0.5, 0.9, 0.95, 0.99])
        .reset_index()
    )
    dist_path = write_dataframe(distribution, args.output_dir / "metric_distributions.parquet")
    overlap_rows = _quality_overlap(metric_df)
    overlap_path = args.output_dir / "quality_overlap.json"
    overlap_path.write_text(json.dumps(overlap_rows, indent=2), encoding="utf-8")

    print(distribution.to_string(index=False))
    print(f"\nMetric rows: {metric_path}")
    print(f"Distributions: {dist_path}")
    print(f"Quality overlap: {overlap_path}")


if __name__ == "__main__":
    main()
