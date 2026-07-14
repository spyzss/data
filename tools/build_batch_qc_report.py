#!/usr/bin/env python3
"""Build one per-asset QC JSON from decoupled module outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_MAX_FLAGGED_ROWS = 20
SUMMARY_FRAME_IDX = -1


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def infer_source_files(asset_id: str, batch_dir: Path | None) -> dict[str, Any]:
    if batch_dir is None:
        return {}
    return {
        "hdf5": str(batch_dir / "hdf5" / f"{asset_id}_hdf5.hdf5"),
        "video": str(batch_dir / "video" / f"{asset_id}_video.mp4"),
    }


def load_video_quality_report(
    asset_id: str,
    batch_dir: Path | None,
    video_quality_json: Path | None,
) -> dict[str, Any] | None:
    candidates: list[Path] = []
    if video_quality_json is not None:
        candidates.append(video_quality_json)
    if batch_dir is not None:
        candidates.append(batch_dir / "quality_archive" / f"{asset_id}.json")

    for path in candidates:
        if path.is_file():
            return load_json(path)
    return None


def aggregate_by_check(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        check = str(row.get("check", ""))
        if check:
            result[check] = row
    return result


def summary_rows_by_check(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for row in rows:
        if int(row.get("frame_idx", 0)) == SUMMARY_FRAME_IDX:
            check = str(row.get("check", ""))
            if check:
                summaries[check] = row
    return summaries


def flagged_samples_by_check(
    rows: list[dict[str, Any]],
    max_rows: int,
) -> dict[str, list[dict[str, Any]]]:
    samples: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("flag") is not True:
            continue
        check = str(row.get("check", ""))
        if not check:
            continue
        bucket = samples.setdefault(check, [])
        if len(bucket) >= max_rows:
            continue
        bucket.append(
            {
                "frame_idx": row.get("frame_idx"),
                "metrics": row.get("metrics", {}),
                "reason": row.get("reason"),
            }
        )
    return samples


def check_has_no_flagged_frames(
    aggregates: dict[str, dict[str, Any]],
    check: str,
) -> bool | None:
    aggregate = aggregates.get(check)
    if aggregate is None:
        return None
    return int(aggregate.get("flagged_frames") or 0) == 0


def summary_flag(
    summaries: dict[str, dict[str, Any]],
    check: str,
) -> bool | None:
    summary = summaries.get(check)
    if summary is None:
        return None
    flag = summary.get("flag")
    return bool(flag) if flag is not None else None


def build_precheck_report(
    precheck_dir: Path | None,
    include_frame_results: bool,
    max_flagged_rows: int,
) -> dict[str, Any] | None:
    if precheck_dir is None:
        return None

    check_results_path = precheck_dir / "check_results.json"
    clip_aggregates_path = precheck_dir / "clip_aggregates.json"
    if not check_results_path.is_file() or not clip_aggregates_path.is_file():
        raise FileNotFoundError(
            "precheck report requires check_results.json and clip_aggregates.json "
            f"under {precheck_dir}"
        )

    check_results = load_json(check_results_path)
    clip_aggregates = load_json(clip_aggregates_path)
    aggregates = aggregate_by_check(clip_aggregates)
    summaries = summary_rows_by_check(check_results)
    flagged_samples = flagged_samples_by_check(check_results, max_flagged_rows)

    derived = {
        "text_integrity_pass": check_has_no_flagged_frames(aggregates, "text_integrity"),
        "quality_score_pass": summary_flag(summaries, "quality_score"),
        "keypoint_missing_pass": check_has_no_flagged_frames(
            aggregates, "keypoint_missing"
        ),
        "skeleton_quality_score_pass": summary_flag(
            summaries, "skeleton_quality_score"
        ),
        "composite_frame_verdict_pass": summary_flag(
            summaries, "composite_frame_verdict"
        ),
    }

    report: dict[str, Any] = {
        "paths": {
            "check_results": str(check_results_path),
            "clip_aggregates": str(clip_aggregates_path),
        },
        "aggregates": clip_aggregates,
        "summary_rows": summaries,
        "flagged_frame_samples": flagged_samples,
        "derived": derived,
    }
    if include_frame_results:
        report["check_results"] = check_results
    return report


def video_quality_pass(video_quality: dict[str, Any] | None) -> bool | None:
    if video_quality is None:
        return None
    summary = video_quality.get("qc_summary") or {}
    passed = summary.get("passed")
    return bool(passed) if passed is not None else None


def build_final_decision(
    precheck: dict[str, Any] | None,
    video_quality: dict[str, Any] | None,
) -> dict[str, Any]:
    hard_fail_reasons: list[str] = []
    risk_reasons: list[str] = []

    if precheck is not None:
        derived = precheck.get("derived", {})
        if derived.get("text_integrity_pass") is False:
            hard_fail_reasons.append("text_integrity_failed")
        if derived.get("quality_score_pass") is False:
            hard_fail_reasons.append("quality_score_failed")
        if derived.get("keypoint_missing_pass") is False:
            hard_fail_reasons.append("keypoint_missing_failed")
        if derived.get("skeleton_quality_score_pass") is False:
            risk_reasons.append("skeleton_quality_score_failed")
        if derived.get("composite_frame_verdict_pass") is False:
            risk_reasons.append("composite_frame_verdict_failed")

    vq_pass = video_quality_pass(video_quality)
    if vq_pass is False:
        hard_fail_reasons.append("video_quality_failed")

    if hard_fail_reasons:
        status = "failed"
        risk_level = "high"
    elif risk_reasons:
        status = "risk"
        risk_level = "medium"
    else:
        status = "passed"
        risk_level = "low"

    return {
        "status": status,
        "risk_level": risk_level,
        "hard_fail_reasons": hard_fail_reasons,
        "risk_reasons": risk_reasons,
        "run_sam3_containment_recommended": not hard_fail_reasons,
    }


def build_report(
    asset_id: str,
    batch_dir: Path | None,
    precheck_dir: Path | None,
    video_quality_json: Path | None,
    include_frame_results: bool = False,
    max_flagged_rows: int = DEFAULT_MAX_FLAGGED_ROWS,
) -> dict[str, Any]:
    precheck = build_precheck_report(
        precheck_dir=precheck_dir,
        include_frame_results=include_frame_results,
        max_flagged_rows=max_flagged_rows,
    )
    video_quality = load_video_quality_report(asset_id, batch_dir, video_quality_json)
    return {
        "schema_version": "batch_qc_asset_report.v1",
        "asset_id": asset_id,
        "source_files": infer_source_files(asset_id, batch_dir),
        "modules": {
            "precheck": precheck,
            "video_quality": video_quality,
        },
        "final": build_final_decision(precheck, video_quality),
    }


def flatten_report(report: dict[str, Any]) -> dict[str, Any]:
    precheck = (report.get("modules") or {}).get("precheck") or {}
    precheck_derived = precheck.get("derived") or {}
    video_quality = (report.get("modules") or {}).get("video_quality") or {}
    video_summary = video_quality.get("qc_summary") or {}
    final = report.get("final") or {}
    return {
        "asset_id": report.get("asset_id"),
        "final_status": final.get("status"),
        "risk_level": final.get("risk_level"),
        "run_sam3_containment_recommended": final.get(
            "run_sam3_containment_recommended"
        ),
        "hard_fail_reasons": "|".join(final.get("hard_fail_reasons") or []),
        "risk_reasons": "|".join(final.get("risk_reasons") or []),
        "text_integrity_pass": precheck_derived.get("text_integrity_pass"),
        "quality_score_pass": precheck_derived.get("quality_score_pass"),
        "keypoint_missing_pass": precheck_derived.get("keypoint_missing_pass"),
        "skeleton_quality_score_pass": precheck_derived.get(
            "skeleton_quality_score_pass"
        ),
        "composite_frame_verdict_pass": precheck_derived.get(
            "composite_frame_verdict_pass"
        ),
        "video_quality_pass": video_summary.get("passed"),
        "video_quality_reasons": "|".join(video_summary.get("reasons") or []),
    }


def write_csv_row(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = flatten_report(report)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one per-asset QC report JSON from module outputs."
    )
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--batch-dir", type=Path)
    parser.add_argument("--precheck-dir", type=Path)
    parser.add_argument("--video-quality-json", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument(
        "--include-frame-results",
        action="store_true",
        help="Embed full precheck check_results.json. Defaults to summaries only.",
    )
    parser.add_argument(
        "--max-flagged-rows",
        type=int,
        default=DEFAULT_MAX_FLAGGED_ROWS,
        help="Maximum flagged frame samples to embed per check.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        asset_id=args.asset_id,
        batch_dir=args.batch_dir,
        precheck_dir=args.precheck_dir,
        video_quality_json=args.video_quality_json,
        include_frame_results=args.include_frame_results,
        max_flagged_rows=args.max_flagged_rows,
    )
    write_json(args.output, report)
    if args.csv_output is not None:
        write_csv_row(args.csv_output, report)
    print(f"Wrote {args.output}")
    if args.csv_output is not None:
        print(f"Wrote {args.csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
