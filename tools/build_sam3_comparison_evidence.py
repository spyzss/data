#!/usr/bin/env python3
"""Materialize overlays only for exact SAM3-comparison false-positive frames."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from qc_common.config import load_qc_acceptance_config
from qc_pipeline.artifacts import canonical_sha256
from qc_pipeline.sam3_runtime import Sam3RuntimeProvider
from tools.run_manifest_sam3_containment import (
    DEFAULT_QUERIES,
    SAM3_CONFIG,
    _joint_names,
    _manifest_index,
    configured_sam3_thresholds,
    read_records,
    reshape_jdt_keypoints,
)
from tools.run_manifest_sam3_exhaustive import (
    SequentialVideoSource,
    _atomic_write_json,
    _evidence_clip_id,
    _model_identity,
    aggregate_frame_verdict,
    classify_audit_hand,
)
from tools.sam3_keypoint_containment import (
    score_keypoints_against_masks,
    write_combined_overlay_image,
)


def _validate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = [dict(row) for row in rows]
    if any(str(row.get("comparison_class", "")) != "FP" for row in normalized):
        raise ValueError("false-positive evidence accepts only FP rows")
    keys = [
        (str(row.get("asset_id", "")), int(row.get("source_frame", -1)))
        for row in normalized
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate asset/source-frame rows in false-positive input")
    return sorted(
        normalized,
        key=lambda row: (str(row["asset_id"]), int(row["source_frame"])),
    )


def build_false_positive_evidence(
    *,
    manifest: Path,
    false_positive_rows: Sequence[Mapping[str, Any]],
    sam3_model: Path | None,
    output_dir: Path,
    config_path: Path | None = None,
    source_reader: Any | None = None,
    segmenter: Any | None = None,
    overlay_writer: Any = write_combined_overlay_image,
    asset_ids: Sequence[str] | None = None,
    max_frames: int | None = None,
    decode_workers: int = 4,
    prefetch_frames: int = 32,
) -> dict[str, Any]:
    rows = _validate_rows(false_positive_rows)
    if asset_ids is not None:
        selected_assets = {str(value) for value in asset_ids}
        rows = [row for row in rows if str(row["asset_id"]) in selected_assets]
    if max_frames is not None:
        if max_frames < 1:
            raise ValueError("max_frames must be >= 1")
        rows = rows[:max_frames]
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"evidence output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_records(Path(manifest))
    indexed, _, manifest_failures = _manifest_index(
        manifest_rows, manifest_dir=Path(manifest).parent.resolve()
    )
    if manifest_failures:
        raise ValueError(f"manifest contains invalid rows: {manifest_failures[:3]}")
    missing_assets = sorted({str(row["asset_id"]) for row in rows} - set(indexed))
    if missing_assets:
        raise ValueError("false-positive assets missing from manifest: " + ", ".join(missing_assets))
    config = load_qc_acceptance_config(config_path)
    thresholds, _ = configured_sam3_thresholds(config)
    model_identity = _model_identity(sam3_model)
    identity = {
        "producer": "sam3-comparison-fp-evidence-v1",
        "manifest": str(Path(manifest).resolve()),
        "model": model_identity,
        "config_hash": config.sha256,
        "thresholds": thresholds,
        "sam3_runtime_config": dict(SAM3_CONFIG),
    }
    identity_sha256 = canonical_sha256(identity)
    source_reader = source_reader or SequentialVideoSource(
        decode_workers=decode_workers, prefetch_frames=prefetch_frames
    )
    queries = [value.strip() for value in DEFAULT_QUERIES.split(",") if value.strip()]
    evidence: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    inference_seconds = 0.0
    decode_seconds = 0.0
    started = time.perf_counter()
    try:
        for asset_id in sorted({str(row["asset_id"]) for row in rows}):
            asset_rows = [row for row in rows if str(row["asset_id"]) == asset_id]
            manifest_row = indexed[asset_id]
            source_data = source_reader.read_parquet(Path(manifest_row["parquet_path"]))
            metadata = source_reader.video_metadata(Path(manifest_row["primary_video_path"]))
            frame_count = int(metadata["frame_count"])
            requested = [int(row["video_frame"]) for row in asset_rows]
            decoded = {
                int(frame_idx): (frame, float(elapsed))
                for frame_idx, frame, elapsed in source_reader.iter_frames(
                    Path(manifest_row["primary_video_path"]), requested
                )
            }
            for row in asset_rows:
                source_frame = int(row["source_frame"])
                video_frame = int(row["video_frame"])
                failure_reason: str | None = None
                if source_frame != video_frame:
                    failure_reason = "jd_source_video_mapping_mismatch"
                elif not 0 <= video_frame < frame_count:
                    failure_reason = "video_frame_out_of_range"
                elif video_frame not in decoded:
                    failure_reason = "video_early_eof"
                if failure_reason is not None:
                    failures.append(
                        {**row, "stage": "input", "reason": failure_reason}
                    )
                    continue
                frame, frame_decode_seconds = decoded[video_frame]
                decode_seconds += frame_decode_seconds
                hands: dict[str, dict[str, Any]] = {}
                hand_verdicts: dict[str, str] = {}
                try:
                    for hand in ("left", "right"):
                        field = str(manifest_row[f"{hand}_hand_2d_field"])
                        pixels = reshape_jdt_keypoints(
                            source_data.iloc[source_frame][field], field, source_frame
                        )
                        if not bool(np.isfinite(pixels).all()):
                            raise ValueError(f"{hand}:direct_2d_nonfinite")
                        hands[hand] = {
                            "pixels": pixels,
                            "joint_names": _joint_names(hand),
                        }
                    if segmenter is None:
                        if sam3_model is None:
                            raise ValueError(
                                "sam3_model is required when segmenter is not injected"
                            )
                        segmenter = Sam3RuntimeProvider().get_segmenter(
                            sam3_model, SAM3_CONFIG
                        )
                    inference_started = time.perf_counter()
                    masks = segmenter.segment_frame(frame, queries, dict(SAM3_CONFIG))
                    inference_seconds += time.perf_counter() - inference_started
                    metrics_by_hand: dict[str, Mapping[str, Any]] = {}
                    for hand, values in hands.items():
                        metrics, _mask, valid, inside = score_keypoints_against_masks(
                            frame=frame,
                            pixels=values["pixels"],
                            joint_names=values["joint_names"],
                            masks=masks,
                            **thresholds,
                        )
                        values["valid"] = valid
                        values["inside"] = inside
                        metrics_by_hand[hand] = metrics
                        hand_verdicts[hand] = classify_audit_hand(
                            str(metrics["containment_verdict"])
                        )
                    rerun = aggregate_frame_verdict(
                        hand_verdicts, required_hands=("left", "right")
                    )
                    overlay_path = overlay_writer(
                        frame=frame,
                        hands=hands,
                        clip_id=_evidence_clip_id(
                            asset_id, source_frame, video_frame
                        ),
                        frame_idx=source_frame,
                        output_dir=output_dir / "overlays" / asset_id,
                    )
                    evidence.append(
                        {
                            **row,
                            "source_video": str(manifest_row["primary_video_path"]),
                            "overlay_path": str(overlay_path),
                            "rerun_frame_verdict": rerun["frame_verdict"],
                            "left_inside_ratio": metrics_by_hand["left"].get(
                                "keypoint_inside_ratio"
                            ),
                            "right_inside_ratio": metrics_by_hand["right"].get(
                                "keypoint_inside_ratio"
                            ),
                            "model_config_identity": identity_sha256,
                        }
                    )
                except Exception as exc:
                    failures.append(
                        {
                            **row,
                            "stage": "materialize_evidence",
                            "reason": f"{type(exc).__name__}:{exc}",
                        }
                    )
    finally:
        source_reader.close()

    evidence_frame = pd.DataFrame(evidence)
    evidence_frame.to_csv(
        output_dir / "false_positive_evidence_manifest.csv", index=False
    )
    (output_dir / "failures.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in failures),
        encoding="utf-8",
    )
    summary = {
        "producer": "sam3-comparison-fp-evidence-v1",
        "requested_frame_count": len(rows),
        "materialized_evidence_count": len(evidence),
        "failure_count": len(failures),
        "timing": {
            "decode_seconds": decode_seconds,
            "inference_seconds": inference_seconds,
            "total_seconds": time.perf_counter() - started,
        },
        "identity": {**identity, "identity_sha256": identity_sha256},
    }
    _atomic_write_json(output_dir / "run_config.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize overlays for exact FP frames without rerunning exhaustive SAM3."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--false-positive-frames", type=Path, required=True)
    parser.add_argument("--sam3-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--asset-ids", nargs="*")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--decode-workers", type=int, default=4)
    parser.add_argument("--prefetch-frames", type=int, default=32)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = build_false_positive_evidence(
        manifest=args.manifest,
        false_positive_rows=read_records(args.false_positive_frames),
        sam3_model=args.sam3_model,
        output_dir=args.output_dir,
        config_path=args.config,
        asset_ids=args.asset_ids,
        max_frames=args.max_frames,
        decode_workers=args.decode_workers,
        prefetch_frames=args.prefetch_frames,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
