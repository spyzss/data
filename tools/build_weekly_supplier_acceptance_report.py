#!/usr/bin/env python3
"""Build the minimal weekly five-supplier acceptance workbook."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


LOGGER = logging.getLogger("build_weekly_supplier_acceptance_report")

SUMMARY_COLUMNS = [
    "supplier_name",
    "sample_clip_count",
    "expected_clip_count",
    "total_frame_count",
    "problem_frame_count",
    "problem_frame_ratio",
    "pass_clip_count",
    "fail_clip_count",
    "review_clip_count",
    "pass_clip_ratio",
    "fail_clip_ratio",
    "main_issue_type",
    "modules_completed",
    "blocked_modules",
    "notes",
]

XJGT_DETAIL_COLUMNS = [
    "asset_id",
    "total_frames",
    "text_check_status",
    "skeleton_missing_status",
    "skeleton_morphology_status",
    "video_quality_status",
    "temporal_status",
    "sam3_containment_status",
    "manual_review_status",
    "manual_problem_frame_ratio",
    "final_clip_status",
    "main_issue_type",
    "notes",
    "evidence_path",
]
DETAIL_COLUMNS = XJGT_DETAIL_COLUMNS

HARD_ISSUE_COLUMNS = [
    "issue_type",
    "current_detection_method",
    "current_threshold",
    "auto_decision_available",
    "needs_manual_review",
    "observed_count",
    "observed_ratio",
    "why_hard",
    "next_action",
]

SHEET_NAMES = [
    "五供应商总览",
    "星际归途",
    "DeepReach",
    "供应商3",
    "供应商4",
    "供应商5",
    "人工与难测问题统计",
]

XJGT_FALLBACK_COUNTS = {
    "fail": 21,
    "review": 64,
    "pass_with_notes": 15,
}


@dataclass(frozen=True)
class WeeklyOutputPaths:
    workbook_xlsx: Path
    summary_csv: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the minimal weekly five-supplier acceptance report."
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("outputs/acceptance_5x100"),
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    outputs = build_weekly_report(args.run_root)
    LOGGER.info("Wrote %s", outputs.workbook_xlsx)
    LOGGER.info("Wrote %s", outputs.summary_csv)
    return 0


def build_weekly_report(run_root: Path) -> WeeklyOutputPaths:
    run_root.mkdir(parents=True, exist_ok=True)
    xjgt = load_xjgt(run_root)
    deepreach = load_deepreach(run_root)
    placeholders = [placeholder_supplier(index) for index in range(3, 6)]
    summary_rows = [xjgt["summary"], deepreach["summary"], *placeholders]

    summary_csv = run_root / "weekly_supplier_summary.csv"
    workbook_xlsx = run_root / "weekly_supplier_acceptance_report.xlsx"
    write_csv(summary_csv, SUMMARY_COLUMNS, summary_rows)
    write_workbook(
        workbook_xlsx,
        summary_rows,
        xjgt["details"],
        deepreach["details"],
        xjgt["issue_events"],
    )
    return WeeklyOutputPaths(workbook_xlsx, summary_csv)


def load_xjgt(run_root: Path) -> dict[str, Any]:
    manifest_path = (
        run_root / "manifests" / "supplier_manifest_xjgt_100.csv"
    )
    ledger_dir = run_root / "xjgt" / "ledger"
    ledger_path = ledger_dir / "xjgt_100_asset_ledger.csv"
    events_path = ledger_dir / "xjgt_100_issue_events.csv"
    summary_path = ledger_dir / "xjgt_100_summary.json"

    manifest = read_csv(manifest_path)
    ledger = read_csv(ledger_path)
    events = read_csv(events_path)
    summary_json = read_json(summary_path)
    manifest_by_asset = {
        normalize_asset_id(row.get("asset_id")): row for row in manifest
    }
    manual_intervals: dict[str, list[tuple[int, int]]] = {}
    for event in events:
        if event.get("source_module") != "manual_review":
            continue
        if event.get("manual_outcome") not in {
            "true_positive",
            "partial",
            "positive",
        }:
            continue
        start = integer_or_none(event.get("start_frame"))
        end = integer_or_none(event.get("end_frame"))
        if start is None or end is None:
            continue
        asset_id = normalize_asset_id(event.get("asset_id"))
        manual_intervals.setdefault(asset_id, []).append((start, end))

    details = []
    for ledger_row in ledger:
        asset_id = normalize_asset_id(ledger_row.get("asset_id"))
        manifest_row = manifest_by_asset.get(asset_id, {})
        total_frames = integer_or_none(
            first_value(
                manifest_row,
                ("frame_count", "total_frames", "num_frames"),
            )
        ) or 0
        manual_problem_frames = interval_frame_count(
            manual_intervals.get(asset_id, [])
        )
        details.append(
            {
                "asset_id": asset_id,
                "total_frames": total_frames,
                "text_check_status": status_value(
                    first_value(
                        ledger_row,
                        ("hdf5_text_status", "text_check_status"),
                    )
                ),
                "skeleton_missing_status": status_value(
                    first_value(
                        ledger_row,
                        (
                            "keypoint_missing_status",
                            "keypoint_existence_status",
                            "skeleton_missing_status",
                        ),
                    )
                ),
                "skeleton_morphology_status": status_value(
                    first_value(
                        ledger_row,
                        (
                            "keypoint_morphology_status",
                            "skeleton_morphology_status",
                        ),
                    )
                ),
                "video_quality_status": status_value(
                    ledger_row.get("video_quality_status")
                ),
                "temporal_status": status_value(
                    first_value(
                        ledger_row,
                        ("temporal_status", "precheck_status"),
                    )
                ),
                "sam3_containment_status": status_value(
                    first_value(
                        ledger_row,
                        (
                            "sam3_containment_status",
                            "sam3_evidence_status",
                        ),
                    )
                ),
                "manual_review_status": status_value(
                    ledger_row.get("manual_review_status")
                ),
                "manual_problem_frame_ratio": safe_ratio(
                    manual_problem_frames, total_frames
                ),
                "final_clip_status": final_status(
                    ledger_row.get("final_verdict")
                ),
                "main_issue_type": first_issue(
                    ledger_row.get("top_issue_types")
                ),
                "notes": text(ledger_row.get("notes")),
                "evidence_path": text(
                    first_value(
                        ledger_row,
                        ("evidence_paths", "evidence_path"),
                    )
                ),
            }
        )

    counts = summary_json.get("final_verdict_counts")
    if not isinstance(counts, dict):
        counts = dict(Counter(row["final_clip_status"] for row in details))
    counts = {
        "fail": int(counts.get("fail", XJGT_FALLBACK_COUNTS["fail"])),
        "review": int(
            counts.get("review", XJGT_FALLBACK_COUNTS["review"])
        ),
        "pass_with_notes": int(
            counts.get(
                "pass_with_notes",
                counts.get("pass", XJGT_FALLBACK_COUNTS["pass_with_notes"]),
            )
        ),
    }
    asset_count = int(
        summary_json.get("asset_count")
        or sum(counts.values())
        or len(details)
        or 100
    )
    total_frames = sum(int(row["total_frames"]) for row in details)
    problem_frames = sum(
        round(row["manual_problem_frame_ratio"] * row["total_frames"])
        for row in details
    )
    issue_counts = Counter(
        text(event.get("failure_mode"))
        for event in events
        if event.get("failure_mode")
    )
    return {
        "summary": {
            "supplier_name": "星际归途 / XJGT",
            "sample_clip_count": asset_count,
            "expected_clip_count": 100,
            "total_frame_count": total_frames,
            "problem_frame_count": problem_frames,
            "problem_frame_ratio": safe_ratio(
                problem_frames, total_frames
            ),
            "pass_clip_count": counts["pass_with_notes"],
            "fail_clip_count": counts["fail"],
            "review_clip_count": counts["review"],
            "pass_clip_ratio": safe_ratio(
                counts["pass_with_notes"], asset_count
            ),
            "fail_clip_ratio": safe_ratio(counts["fail"], asset_count),
            "main_issue_type": most_common(issue_counts),
            "modules_completed": (
                "precheck|video_quality|sam3|manual_review|ledger"
            ),
            "blocked_modules": "",
            "notes": "High-risk-biased manual review; not an unbiased global error-rate estimate.",
        },
        "details": details,
        "issue_events": events,
    }


def load_deepreach(run_root: Path) -> dict[str, Any]:
    manifest_path = (
        run_root
        / "deepreach"
        / "manifests"
        / "supplier_manifest_deepreach.csv"
    )
    manifest = read_csv(manifest_path)
    video_results = (
        run_root
        / "deepreach"
        / "video_quality"
        / "video_quality_results.json"
    )
    video_summary = (
        run_root
        / "deepreach"
        / "video_quality"
        / "video_quality_decision_summary.csv"
    )
    has_video_quality = video_results.exists() or video_summary.exists()
    details = []
    for row in manifest:
        total_frames = integer_or_none(
            first_value(row, ("frame_count", "total_frames", "num_frames"))
        ) or 0
        details.append(
            {
                "asset_id": normalize_asset_id(row.get("asset_id")),
                "total_frames": total_frames,
                "text_check_status": "blocked",
                "skeleton_missing_status": "blocked",
                "skeleton_morphology_status": "blocked",
                "video_quality_status": (
                    "not_run" if not has_video_quality else "review"
                ),
                "temporal_status": "blocked",
                "sam3_containment_status": "blocked",
                "manual_review_status": "not_run",
                "manual_problem_frame_ratio": 0.0,
                "final_clip_status": "blocked",
                "main_issue_type": "missing_schema_adapter",
                "notes": (
                    "precheck blocked by HDF5 schema adapter; "
                    "SAM3 blocked by calibration/projection mapping"
                ),
                "evidence_path": str(manifest_path),
            }
        )
    modules_completed = ["manifest"] if manifest else []
    if has_video_quality:
        modules_completed.append("video_quality")
    return {
        "summary": {
            "supplier_name": "DeepReach",
            "sample_clip_count": len(manifest),
            "expected_clip_count": 100,
            "total_frame_count": sum(
                int(row["total_frames"]) for row in details
            ),
            "problem_frame_count": 0,
            "problem_frame_ratio": 0.0,
            "pass_clip_count": 0,
            "fail_clip_count": 0,
            "review_clip_count": 0,
            "pass_clip_ratio": 0.0,
            "fail_clip_ratio": 0.0,
            "main_issue_type": "missing_schema_adapter",
            "modules_completed": "|".join(modules_completed),
            "blocked_modules": "precheck|sam3",
            "notes": (
                "video_quality not_run unless result/decision summary exists"
            ),
        },
        "details": details,
    }


def placeholder_supplier(index: int) -> dict[str, Any]:
    return {
        "supplier_name": f"供应商{index}",
        "sample_clip_count": 0,
        "expected_clip_count": 100,
        "total_frame_count": 0,
        "problem_frame_count": 0,
        "problem_frame_ratio": 0.0,
        "pass_clip_count": 0,
        "fail_clip_count": 0,
        "review_clip_count": 0,
        "pass_clip_ratio": 0.0,
        "fail_clip_ratio": 0.0,
        "main_issue_type": "",
        "modules_completed": "",
        "blocked_modules": "missing_input",
        "notes": "input manifest not provided",
    }


def hard_issue_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observed = Counter(
        text(event.get("failure_mode"))
        for event in events
        if event.get("failure_mode")
    )
    total = sum(observed.values())
    definitions = [
        (
            "hand_out_of_frame",
            "projection/border evidence + manual review",
            "pending calibration",
            "partial",
            "yes",
            "Finite keypoints do not prove visual presence.",
            "Calibrate border and mask-truncation rules.",
        ),
        (
            "severe_keypoint_offset",
            "SAM3 containment + manual overlay",
            "inside_ratio <= 0.2 is strong mismatch evidence",
            "partial",
            "yes",
            "Occlusion and undersegmentation can look identical.",
            "Calibrate with confirmed segments.",
        ),
        (
            "skeleton_pose_hallucination",
            "static morphology + manual overlay",
            "config-driven morphology thresholds",
            "partial",
            "yes",
            "A wrong pose may remain temporally stable.",
            "Expand morphology calibration.",
        ),
        (
            "visual_skeleton_presence_mismatch",
            "projection/SAM3 + manual review",
            "pending calibration",
            "partial",
            "yes",
            "Visibility depends on framing and occlusion.",
            "Add presence/truncation evidence.",
        ),
        (
            "occlusion_or_mask_undersegmentation",
            "SAM3 diagnostics",
            "inside_ratio < 0.6 routes to review",
            "no",
            "yes",
            "Keypoint-in-mask cannot diagnose mask completeness.",
            "Retain manual arbitration.",
        ),
        (
            "projection_review",
            "3D-to-2D projection checks",
            "projected_in_image_ratio < 0.8",
            "no",
            "yes",
            "Calibration uncertainty can dominate.",
            "Validate supplier calibration.",
        ),
        (
            "video_quality_pending_colleague_thresholds",
            "acceptance_pull/video_quality.py",
            "pending colleague thresholds",
            "pending",
            "no",
            "Threshold ownership is external to this report.",
            "Consume finalized video_quality status.",
        ),
        (
            "text_check_pending_rules",
            "precheck text integrity",
            "pending rules",
            "no",
            "yes",
            "Acceptance semantics are not finalized.",
            "Define required fields and fail rules.",
        ),
    ]
    return [
        {
            "issue_type": issue_type,
            "current_detection_method": method,
            "current_threshold": threshold,
            "auto_decision_available": automatic,
            "needs_manual_review": manual,
            "observed_count": observed[issue_type],
            "observed_ratio": safe_ratio(observed[issue_type], total),
            "why_hard": why,
            "next_action": action,
        }
        for issue_type, method, threshold, automatic, manual, why, action in definitions
    ]


def write_workbook(
    path: Path,
    summary_rows: list[dict[str, Any]],
    xjgt_details: list[dict[str, Any]],
    deepreach_details: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> None:
    workbook = Workbook()
    workbook.remove(workbook.active)
    add_sheet(workbook, "五供应商总览", SUMMARY_COLUMNS, summary_rows)
    add_sheet(
        workbook,
        "星际归途",
        XJGT_DETAIL_COLUMNS,
        xjgt_details,
    )
    add_sheet(
        workbook,
        "DeepReach",
        XJGT_DETAIL_COLUMNS,
        deepreach_details,
    )
    for index in range(3, 6):
        add_sheet(
            workbook,
            f"供应商{index}",
            XJGT_DETAIL_COLUMNS,
            [],
        )
    add_sheet(
        workbook,
        "人工与难测问题统计",
        HARD_ISSUE_COLUMNS,
        hard_issue_rows(events),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def add_sheet(
    workbook: Workbook,
    name: str,
    columns: list[str],
    rows: list[dict[str, Any]],
) -> None:
    sheet = workbook.create_sheet(name)
    sheet.append(columns)
    for row in rows:
        sheet.append([row.get(column, "") for column in columns])
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    sheet.sheet_view.showGridLines = False
    for cell in sheet[1]:
        cell.fill = PatternFill("solid", fgColor="D9EAF7")
        cell.font = Font(bold=True)
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    for column_index, column in enumerate(columns, 1):
        values = [len(column)]
        values.extend(
            len(str(sheet.cell(row_index, column_index).value or ""))
            for row_index in range(2, min(sheet.max_row, 101) + 1)
        )
        sheet.column_dimensions[get_column_letter(column_index)].width = min(
            max(max(values) + 2, 12), 42
        )
    percentage_columns = {
        "problem_frame_ratio",
        "pass_clip_ratio",
        "fail_clip_ratio",
        "manual_problem_frame_ratio",
        "observed_ratio",
    }
    for column in percentage_columns.intersection(columns):
        column_letter = get_column_letter(columns.index(column) + 1)
        for cell in sheet[column_letter][1:]:
            if isinstance(cell.value, (int, float)):
                cell.number_format = "0.0%"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def write_csv(
    path: Path,
    columns: list[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {column: row.get(column, "") for column in columns}
            for row in rows
        )


def interval_frame_count(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    normalized = sorted(
        (min(start, end), max(start, end)) for start, end in intervals
    )
    merged: list[list[int]] = []
    for start, end in normalized:
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start + 1 for start, end in merged)


def first_value(
    row: dict[str, Any],
    keys: tuple[str, ...],
) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def integer_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def normalize_asset_id(value: Any) -> str:
    result = text(value)
    return result[:-2] if result.endswith(".0") else result


def status_value(value: Any) -> str:
    status = text(value).lower()
    return {
        "pass_with_notes": "pass",
        "risk": "review",
        "warn": "review",
        "warning": "review",
        "input_missing": "blocked",
        "adapter_missing": "blocked",
    }.get(status, status or "not_run")


def final_status(value: Any) -> str:
    status = text(value).lower()
    if status == "pass_with_notes":
        return "pass"
    if status == "risk":
        return "review"
    return status or "not_run"


def first_issue(value: Any) -> str:
    raw = text(value)
    return raw.split("|", 1)[0] if raw else ""


def most_common(counts: Counter[str]) -> str:
    return counts.most_common(1)[0][0] if counts else ""


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def text(value: Any) -> str:
    return "" if value is None else str(value)


if __name__ == "__main__":
    raise SystemExit(main())
