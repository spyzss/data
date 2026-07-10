#!/usr/bin/env python3
"""Run the existing video-quality producer on manifest-defined frame ranges."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acceptance_pull.video_quality import (  # noqa: E402
    Hdf5Alignment,
    VideoQualityResult,
    analyze_video_frame_range,
    asset_qc_result_to_json,
    evaluate_video_quality,
    load_video_quality_config,
)


LOGGER = logging.getLogger(__name__)


def read_manifest(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix in {".jsonl", ".ndjson"}:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return [dict(row) for row in rows if isinstance(row, dict)]
    else:
        raise ValueError(f"unsupported manifest extension: {path.suffix}")
    frame = frame.where(pd.notna(frame), None)
    return [dict(row) for row in frame.to_dict(orient="records")]


def _integer(value: Any, field: str) -> int:
    if value is None or value == "":
        raise ValueError(f"missing {field}")
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{field} must be an integer")
    return int(numeric)


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def probe_video(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open source video: {path}")
        return {
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        capture.release()


def _validated_row(
    row: dict[str, Any],
    *,
    asset_id_column: str,
    video_column: str,
    start_frame_column: str,
    end_frame_column: str,
    hdf5_column: str | None,
) -> dict[str, Any]:
    asset_id = _text(row.get(asset_id_column))
    if not asset_id:
        raise ValueError(f"missing {asset_id_column}")
    video_text = _text(row.get(video_column))
    if not video_text:
        raise ValueError(f"missing {video_column}")
    video_path = Path(video_text).expanduser()
    start_frame = _integer(row.get(start_frame_column), start_frame_column)
    end_frame = _integer(row.get(end_frame_column), end_frame_column)
    if start_frame < 0:
        raise ValueError("start_frame must be >= 0")
    if end_frame < start_frame:
        raise ValueError("end_frame must be >= start_frame")
    metadata = probe_video(video_path)
    if end_frame >= int(metadata["frame_count"]):
        raise ValueError(
            f"end_frame {end_frame} outside source frame count "
            f"{metadata['frame_count']}"
        )
    hdf5_text = _text(row.get(hdf5_column)) if hdf5_column else ""
    return {
        "asset_id": asset_id,
        "video_path": video_path,
        "hdf5_path": Path(hdf5_text).expanduser() if hdf5_text else None,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "metadata": metadata,
        "manifest_row": row,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _parquet_safe(value: Any) -> Any:
    if isinstance(value, dict):
        if not value:
            return None
        return {str(key): _parquet_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_parquet_safe(item) for item in value]
    return value


def _read_existing_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"existing results must be a JSON list: {path}")
    return [dict(row) for row in payload if isinstance(row, dict)]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _result_record(
    validated: dict[str, Any],
    config: Any,
) -> dict[str, Any]:
    start_frame = int(validated["start_frame"])
    end_frame = int(validated["end_frame"])
    analysis = analyze_video_frame_range(
        validated["video_path"],
        config,
        start_frame,
        end_frame,
        hdf5_path=validated["hdf5_path"],
    )
    metrics = replace(analysis.metrics, asset_id=validated["asset_id"])
    alignment = Hdf5Alignment(
        status="range_not_evaluated",
        hdf5_path=validated["hdf5_path"],
        hdf5_frame_count=None,
        frame_count_match=None,
        reason="logical range alignment validated by manifest bounds",
    )
    evaluation = evaluate_video_quality(metrics, config, alignment=None)
    report = asset_qc_result_to_json(
        VideoQualityResult(metrics, alignment, evaluation),
        config,
    )
    report.update(
        {
            "asset_id": validated["asset_id"],
            "source_video_path": str(validated["video_path"]),
            "hdf5_path": (
                str(validated["hdf5_path"])
                if validated["hdf5_path"] is not None
                else ""
            ),
            "clip_start_frame": start_frame,
            "clip_end_frame": end_frame,
            "clip_frame_count": end_frame - start_frame + 1,
            "source_video_frame_count": analysis.source_video_frame_count,
            "decoded_frame_count": analysis.decoded_frame_count,
            "sampled_local_frame_indices": list(
                analysis.sampled_local_frame_indices
            ),
            "sampled_source_frame_indices": [
                start_frame + frame_idx
                for frame_idx in analysis.sampled_local_frame_indices
            ],
            "sampled_frame_mappings": [
                {
                    "local_frame_idx": frame_idx,
                    "source_frame_idx": start_frame + frame_idx,
                }
                for frame_idx in analysis.sampled_local_frame_indices
            ],
            "frame_coordinate_system": "source_video_inclusive",
        }
    )
    report["source_files"]["video"]["path"] = str(validated["video_path"])
    freeze = report["video_quality"]["metrics"]["freeze_metrics"]
    for interval in freeze.get("frozen_intervals", []):
        interval["source_start_frame"] = int(interval["start_frame"])
        interval["source_end_frame"] = int(interval["end_frame"])
        interval["local_start_frame"] = int(interval["start_frame"]) - start_frame
        interval["local_end_frame"] = int(interval["end_frame"]) - start_frame
    return _json_safe(report)


def _summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        summary = record.get("qc_summary", {})
        rows.append(
            {
                "asset_id": record.get("asset_id"),
                "source_video_path": record.get("source_video_path"),
                "clip_start_frame": record.get("clip_start_frame"),
                "clip_end_frame": record.get("clip_end_frame"),
                "clip_frame_count": record.get("clip_frame_count"),
                "source_video_frame_count": record.get(
                    "source_video_frame_count"
                ),
                "decoded_frame_count": record.get("decoded_frame_count"),
                "status": summary.get("status"),
                "passed": summary.get("passed"),
                "reasons": json.dumps(summary.get("reasons", [])),
                "warn_reasons": json.dumps(summary.get("warn_reasons", [])),
                "frame_coordinate_system": "source_video_inclusive",
            }
        )
    return rows


def run_manifest_video_quality(
    manifest: Path,
    output_dir: Path,
    *,
    asset_id_column: str = "asset_id",
    video_column: str = "primary_video_path",
    start_frame_column: str = "start_frame",
    end_frame_column: str = "end_frame",
    hdf5_column: str | None = None,
    max_clips: int | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    log_level: str = "INFO",
) -> dict[str, Any]:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    rows = read_manifest(Path(manifest))
    if max_clips is not None:
        if max_clips < 0:
            raise ValueError("max_clips must be >= 0")
        rows = rows[:max_clips]

    output_dir = Path(output_dir)
    results_path = output_dir / "video_quality_results.json"
    existing = [] if dry_run else _read_existing_results(results_path)
    existing_by_asset = {
        str(row.get("asset_id")): row for row in existing if row.get("asset_id")
    }
    config = load_video_quality_config(None)
    new_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    skipped = 0
    validated_count = 0
    for row_index, row in enumerate(rows):
        asset_id = _text(row.get(asset_id_column)) or f"row-{row_index}"
        if not overwrite and asset_id in existing_by_asset:
            skipped += 1
            continue
        try:
            validated = _validated_row(
                row,
                asset_id_column=asset_id_column,
                video_column=video_column,
                start_frame_column=start_frame_column,
                end_frame_column=end_frame_column,
                hdf5_column=hdf5_column,
            )
            validated_count += 1
            if not dry_run:
                new_records.append(_result_record(validated, config))
        except Exception as exc:
            LOGGER.error("Manifest row %d (%s) failed: %s", row_index, asset_id, exc)
            failures.append(
                {
                    "row_index": row_index,
                    "asset_id": asset_id,
                    "source_video_path": _text(row.get(video_column)),
                    "clip_start_frame": row.get(start_frame_column),
                    "clip_end_frame": row.get(end_frame_column),
                    "frame_coordinate_system": "source_video_inclusive",
                    "error": str(exc),
                }
            )

    summary = {
        "manifest_row_count": len(rows),
        "validated_clip_count": validated_count,
        "completed_clip_count": len(new_records),
        "failed_clip_count": len(failures),
        "skipped_clip_count": skipped,
        "dry_run": dry_run,
    }
    if dry_run:
        return summary

    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        replaced_assets = {str(row["asset_id"]) for row in new_records}
        retained = [
            row
            for row in existing
            if str(row.get("asset_id")) not in replaced_assets
        ]
    else:
        retained = existing
    combined = [*retained, *new_records]
    _write_json(results_path, combined)
    pd.DataFrame([_parquet_safe(row) for row in combined]).to_parquet(
        output_dir / "video_quality_results.parquet",
        index=False,
    )
    pd.DataFrame(_summary_rows(combined)).to_csv(
        output_dir / "video_quality_decision_summary.csv",
        index=False,
    )
    _write_json(output_dir / "video_quality_failures.json", failures)
    run_config = {
        "manifest": str(manifest),
        "output_dir": str(output_dir),
        "asset_id_column": asset_id_column,
        "video_column": video_column,
        "start_frame_column": start_frame_column,
        "end_frame_column": end_frame_column,
        "hdf5_column": hdf5_column,
        "max_clips": max_clips,
        "overwrite": overwrite,
        "frame_range_semantics": "inclusive_source_frames",
        "video_quality_config": config.to_dict(),
    }
    _write_json(output_dir / "run_config.json", run_config)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run video quality on inclusive manifest frame ranges"
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--asset-id-column", default="asset_id")
    parser.add_argument("--video-column", default="primary_video_path")
    parser.add_argument("--start-frame-column", default="start_frame")
    parser.add_argument("--end-frame-column", default="end_frame")
    parser.add_argument("--hdf5-column")
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_manifest_video_quality(
        args.manifest,
        args.output_dir,
        asset_id_column=args.asset_id_column,
        video_column=args.video_column,
        start_frame_column=args.start_frame_column,
        end_frame_column=args.end_frame_column,
        hdf5_column=args.hdf5_column,
        max_clips=args.max_clips,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        log_level=args.log_level,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if not summary["failed_clip_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
