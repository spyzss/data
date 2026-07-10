#!/usr/bin/env python3
"""Build the minimal weekly five-supplier acceptance workbook."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
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
    "frame_count_status",
    "text_check_status",
    "skeleton_missing_status",
    "skeleton_missing_fail_frame_count",
    "skeleton_missing_fail_frame_ratio",
    "skeleton_morphology_status",
    "skeleton_morphology_fail_frame_count",
    "skeleton_morphology_fail_frame_ratio",
    "skeleton_static_fail_frame_count",
    "skeleton_static_fail_frame_ratio",
    "skeleton_static_status",
    "supplier_quality_signal",
    "video_quality_status",
    "video_quality_fail_frame_count",
    "video_quality_fail_frame_ratio",
    "temporal_status",
    "sam3_containment_status",
    "manual_review_status",
    "temporal_sam3_manual_status",
    "manual_problem_frame_count",
    "manual_problem_frame_ratio_of_clip",
    "manual_reviewed_frame_count",
    "manual_problem_ratio_of_reviewed",
    "abnormal_fail_frame_count",
    "abnormal_fail_frame_ratio",
    "abnormal_frame_status",
    "fail_indicator_count",
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
    print_sanity_checks(
        xjgt["summary"],
        xjgt["details"],
        deepreach["summary"],
    )
    return WeeklyOutputPaths(workbook_xlsx, summary_csv)


def load_xjgt(run_root: Path) -> dict[str, Any]:
    manifest_path = (
        run_root / "manifests" / "supplier_manifest_xjgt_100.csv"
    )
    ledger_dir = run_root / "xjgt" / "ledger"
    ledger_path = ledger_dir / "xjgt_100_asset_ledger.csv"
    events_path = ledger_dir / "xjgt_100_issue_events.csv"
    manual_path = (
        run_root
        / "xjgt"
        / "manual_review"
        / "manual_labels_autosave.normalized.csv"
    )

    manifest = read_csv(manifest_path)
    ledger = read_csv(ledger_path)
    events = read_csv(events_path)
    manual_labels = read_csv(manual_path)
    ledger_by_asset = {
        normalize_asset_id(row.get("asset_id")): row for row in ledger
    }
    manual_intervals: dict[str, list[tuple[int, int]]] = {}
    reviewed_intervals: dict[str, list[tuple[int, int]]] = {}
    missing_intervals: dict[str, list[tuple[int, int]]] = {}
    morphology_intervals: dict[str, list[tuple[int, int]]] = {}
    video_intervals: dict[str, list[tuple[int, int]]] = {}
    abnormal_auto_fail_intervals: dict[str, list[tuple[int, int]]] = {}
    for event in events:
        interval = event_frame_interval(event)
        if interval is None:
            continue
        asset_id = normalize_asset_id(event.get("asset_id"))
        if event_is_hard_fail(event):
            category = event_failure_category(event)
            target = {
                "skeleton_missing": missing_intervals,
                "skeleton_morphology": morphology_intervals,
                "video_quality": video_intervals,
            }.get(category)
            if target is not None:
                target.setdefault(asset_id, []).append(interval)
        if event_is_abnormal_auto_fail(event):
            abnormal_auto_fail_intervals.setdefault(asset_id, []).append(
                interval
            )
    manually_labeled_assets: set[str] = set()
    for label in manual_labels:
        asset_id = normalize_asset_id(label.get("asset_id"))
        if not asset_id:
            continue
        manually_labeled_assets.add(asset_id)
        reviewed_start = integer_or_none(label.get("window_start_frame"))
        reviewed_end = integer_or_none(label.get("window_end_frame"))
        if reviewed_start is None or reviewed_end is None:
            reviewed_start = integer_or_none(
                label.get("affected_start_frame")
            )
            reviewed_end = integer_or_none(label.get("affected_end_frame"))
        if reviewed_start is not None and reviewed_end is not None:
            reviewed_intervals.setdefault(asset_id, []).append(
                (reviewed_start, reviewed_end)
            )
        if label.get("manual_outcome") != "true_positive":
            continue
        start = integer_or_none(label.get("affected_start_frame"))
        end = integer_or_none(label.get("affected_end_frame"))
        if start is None or end is None:
            continue
        manual_intervals.setdefault(asset_id, []).append((start, end))

    details = []
    for manifest_row in manifest:
        asset_id = normalize_asset_id(manifest_row.get("asset_id"))
        if not asset_id:
            continue
        ledger_row = ledger_by_asset.get(asset_id, {})
        video_path = resolve_xjgt_video_path(manifest_row, asset_id)
        probed_frames = probe_video_frame_count(video_path)
        if probed_frames > 0:
            total_frames = probed_frames
            frame_count_status = "ok"
            frame_note = ""
        else:
            total_frames = integer_or_none(
                first_value(
                    manifest_row,
                    ("frame_count", "total_frames", "num_frames"),
                )
            ) or 0
            frame_count_status = (
                "manifest_fallback" if total_frames > 0 else "unreadable"
            )
            frame_note = f"frame_count_unreadable:{video_path}"
        manual_problem_frames = interval_frame_count(
            manual_intervals.get(asset_id, [])
        )
        manual_reviewed_frames = interval_frame_count(
            reviewed_intervals.get(asset_id, [])
        )
        manual_problem_ratio = safe_ratio(
            manual_problem_frames, total_frames
        )
        manual_problem_ratio_of_reviewed = safe_ratio(
            manual_problem_frames, manual_reviewed_frames
        )
        if asset_id not in manually_labeled_assets:
            manual_status = "not_reviewed"
        elif manual_problem_ratio >= 0.10:
            manual_status = "fail"
        else:
            manual_status = "pass"
        text_value = first_value(
            ledger_row,
            ("hdf5_text_status", "text_check_status"),
        )
        text_status = (
            status_value(text_value)
            if text_value not in (None, "")
            else "pending_rule"
        )
        missing_base_status = status_value(
            first_value(
                ledger_row,
                (
                    "keypoint_missing_status",
                    "keypoint_existence_status",
                    "skeleton_missing_status",
                ),
            )
        )
        morphology_base_status = status_value(
            first_value(
                ledger_row,
                (
                    "keypoint_morphology_status",
                    "skeleton_morphology_status",
                ),
            )
        )
        quality_signal = supplier_quality_signal(
            ledger_row.get("quality_hand_status")
        )
        video_status = status_value(
            ledger_row.get("video_quality_status")
        )
        if video_status == "not_run":
            video_status = "no_valid_output"
        temporal_status = status_value(
            first_value(
                ledger_row,
                ("temporal_status",),
            )
        )
        sam3_status = status_value(
            first_value(
                ledger_row,
                (
                    "sam3_containment_status",
                    "sam3_evidence_status",
                ),
            )
        )
        missing_fail_frames = interval_frame_count(
            missing_intervals.get(asset_id, [])
        )
        missing_fail_ratio = safe_ratio(
            missing_fail_frames, total_frames
        )
        morphology_fail_frames = interval_frame_count(
            morphology_intervals.get(asset_id, [])
        )
        morphology_fail_ratio = safe_ratio(
            morphology_fail_frames, total_frames
        )
        video_fail_frames = interval_frame_count(
            video_intervals.get(asset_id, [])
        )
        video_fail_ratio = safe_ratio(video_fail_frames, total_frames)
        skeleton_static_intervals = [
            *missing_intervals.get(asset_id, []),
            *morphology_intervals.get(asset_id, []),
        ]
        skeleton_static_fail_frames = interval_frame_count(
            skeleton_static_intervals
        )
        skeleton_static_fail_ratio = safe_ratio(
            skeleton_static_fail_frames, total_frames
        )
        skeleton_static_status = (
            "fail" if skeleton_static_fail_ratio >= 0.10 else "pass"
        )
        abnormal_intervals = [
            *subtract_intervals(
                abnormal_auto_fail_intervals.get(asset_id, []),
                reviewed_intervals.get(asset_id, []),
            ),
            *manual_intervals.get(asset_id, []),
        ]
        abnormal_fail_frames = interval_frame_count(abnormal_intervals)
        abnormal_fail_ratio = safe_ratio(abnormal_fail_frames, total_frames)
        missing_status = frame_ratio_status(
            missing_base_status,
            missing_fail_frames,
            missing_fail_ratio,
        )
        morphology_status = frame_ratio_status(
            morphology_base_status,
            morphology_fail_frames,
            morphology_fail_ratio,
        )
        temporal_sam3_manual_status = combined_temporal_status(
            manual_review_status=manual_status,
            temporal_status=temporal_status,
            sam3_containment_status=sam3_status,
        )
        abnormal_frame_status = abnormal_status(
            abnormal_fail_frame_ratio=abnormal_fail_ratio,
            manual_review_status=manual_status,
            temporal_status=temporal_status,
            sam3_containment_status=sam3_status,
        )
        expected_module_missing = any(
            status in {
                "not_run",
                "blocked",
                "no_valid_output",
                "pending_rule",
            }
            for status in (
                missing_base_status,
                morphology_base_status,
                video_status,
                temporal_status,
                sam3_status,
            )
        )
        fail_indicator_count = count_fail_indicators(
            text_check_status=text_status,
            video_quality_status=video_status,
            skeleton_static_status=skeleton_static_status,
            abnormal_frame_status=abnormal_frame_status,
        )
        final_clip_status = recompute_xjgt_final_status(
            text_check_status=text_status,
            video_quality_status=video_status,
            skeleton_static_status=skeleton_static_status,
            abnormal_frame_status=abnormal_frame_status,
            expected_module_missing=expected_module_missing,
        )
        problem_intervals = [
            *skeleton_static_intervals,
            *abnormal_intervals,
            *video_intervals.get(asset_id, []),
        ]
        details.append(
            {
                "asset_id": asset_id,
                "total_frames": total_frames,
                "frame_count_status": frame_count_status,
                "text_check_status": text_status,
                "skeleton_missing_status": missing_status,
                "skeleton_missing_fail_frame_count": (
                    missing_fail_frames
                ),
                "skeleton_missing_fail_frame_ratio": missing_fail_ratio,
                "skeleton_morphology_status": morphology_status,
                "skeleton_morphology_fail_frame_count": (
                    morphology_fail_frames
                ),
                "skeleton_morphology_fail_frame_ratio": (
                    morphology_fail_ratio
                ),
                "skeleton_static_fail_frame_count": (
                    skeleton_static_fail_frames
                ),
                "skeleton_static_fail_frame_ratio": (
                    skeleton_static_fail_ratio
                ),
                "skeleton_static_status": skeleton_static_status,
                "supplier_quality_signal": quality_signal,
                "video_quality_status": video_status,
                "video_quality_fail_frame_count": video_fail_frames,
                "video_quality_fail_frame_ratio": video_fail_ratio,
                "temporal_status": temporal_status,
                "sam3_containment_status": sam3_status,
                "manual_review_status": manual_status,
                "temporal_sam3_manual_status": (
                    temporal_sam3_manual_status
                ),
                "manual_problem_frame_count": manual_problem_frames,
                "manual_problem_frame_ratio_of_clip": manual_problem_ratio,
                "manual_reviewed_frame_count": manual_reviewed_frames,
                "manual_problem_ratio_of_reviewed": (
                    manual_problem_ratio_of_reviewed
                ),
                "abnormal_fail_frame_count": abnormal_fail_frames,
                "abnormal_fail_frame_ratio": abnormal_fail_ratio,
                "abnormal_frame_status": abnormal_frame_status,
                "fail_indicator_count": fail_indicator_count,
                "final_clip_status": final_clip_status,
                "main_issue_type": first_issue(
                    ledger_row.get("top_issue_types")
                ),
                "notes": "; ".join(
                    value
                    for value in (
                        text(ledger_row.get("notes")),
                        frame_note,
                        (
                            f"supplier_quality_signal={quality_signal}"
                            if quality_signal != "not_provided"
                            else ""
                        ),
                    )
                    if value
                ),
                "evidence_path": text(
                    first_value(
                        ledger_row,
                        ("evidence_paths", "evidence_path"),
                    )
                ),
                "_problem_intervals": problem_intervals,
            }
        )

    counts = Counter(row["final_clip_status"] for row in details)
    asset_count = len(details)
    total_frames = sum(int(row["total_frames"]) for row in details)
    problem_frames = sum(
        interval_frame_count(row["_problem_intervals"])
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
            "pass_clip_count": counts["pass"],
            "fail_clip_count": counts["fail"],
            "review_clip_count": counts["review"],
            "pass_clip_ratio": safe_ratio(
                counts["pass"], asset_count
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
                "frame_count_status": (
                    "manifest" if total_frames > 0 else "not_run"
                ),
                "text_check_status": "blocked",
                "skeleton_missing_status": "blocked",
                "skeleton_missing_fail_frame_count": 0,
                "skeleton_missing_fail_frame_ratio": 0.0,
                "skeleton_morphology_status": "blocked",
                "skeleton_morphology_fail_frame_count": 0,
                "skeleton_morphology_fail_frame_ratio": 0.0,
                "skeleton_static_fail_frame_count": 0,
                "skeleton_static_fail_frame_ratio": 0.0,
                "skeleton_static_status": "blocked",
                "supplier_quality_signal": "not_provided",
                "video_quality_status": (
                    "no_valid_output" if not has_video_quality else "review"
                ),
                "video_quality_fail_frame_count": 0,
                "video_quality_fail_frame_ratio": 0.0,
                "temporal_status": "blocked",
                "sam3_containment_status": "blocked",
                "manual_review_status": "not_run",
                "temporal_sam3_manual_status": "blocked",
                "manual_problem_frame_count": 0,
                "manual_problem_frame_ratio_of_clip": 0.0,
                "manual_reviewed_frame_count": 0,
                "manual_problem_ratio_of_reviewed": 0.0,
                "abnormal_fail_frame_count": 0,
                "abnormal_fail_frame_ratio": 0.0,
                "abnormal_frame_status": "blocked",
                "fail_indicator_count": 0,
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
            "video_quality_status": (
                "review" if has_video_quality else "no_valid_output"
            ),
            "modules_completed": "|".join(modules_completed),
            "blocked_modules": "precheck|sam3",
            "notes": (
                "video_quality not_run/no_valid_output unless result/decision summary exists"
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
        "skeleton_missing_fail_frame_ratio",
        "skeleton_morphology_fail_frame_ratio",
        "skeleton_static_fail_frame_ratio",
        "video_quality_fail_frame_ratio",
        "manual_problem_frame_ratio_of_clip",
        "manual_problem_ratio_of_reviewed",
        "abnormal_fail_frame_ratio",
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


def subtract_intervals(
    intervals: list[tuple[int, int]],
    excluded: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    if not intervals or not excluded:
        return intervals
    excluded_normalized = sorted(
        (min(start, end), max(start, end)) for start, end in excluded
    )
    remaining: list[tuple[int, int]] = []
    for interval_start, interval_end in (
        (min(start, end), max(start, end)) for start, end in intervals
    ):
        pieces = [(interval_start, interval_end)]
        for exclude_start, exclude_end in excluded_normalized:
            next_pieces: list[tuple[int, int]] = []
            for piece_start, piece_end in pieces:
                if exclude_end < piece_start or exclude_start > piece_end:
                    next_pieces.append((piece_start, piece_end))
                    continue
                if exclude_start > piece_start:
                    next_pieces.append((piece_start, exclude_start - 1))
                if exclude_end < piece_end:
                    next_pieces.append((exclude_end + 1, piece_end))
            pieces = next_pieces
            if not pieces:
                break
        remaining.extend(pieces)
    return remaining


def resolve_xjgt_video_path(
    manifest_row: dict[str, Any],
    asset_id: str,
) -> Path:
    video_path = text(manifest_row.get("video_path")).strip()
    if video_path:
        return Path(video_path)
    return Path(
        f"/mnt/oss/egodata/XJGT_20260629/video/{asset_id}_video.mp4"
    )


def probe_video_frame_count(path: Path) -> int:
    try:
        import cv2

        capture = cv2.VideoCapture(str(path))
        try:
            if capture.isOpened():
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                if frame_count > 0:
                    return frame_count
        finally:
            capture.release()
    except (ImportError, OSError, ValueError):
        pass

    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames,nb_frames",
                "-of",
                "json",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return 0
    if completed.returncode != 0:
        return 0
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return 0
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        return 0
    stream = streams[0] if isinstance(streams[0], dict) else {}
    return (
        integer_or_none(stream.get("nb_read_frames"))
        or integer_or_none(stream.get("nb_frames"))
        or 0
    )


def frame_ratio_status(
    base_status: str,
    fail_frame_count: int,
    fail_frame_ratio: float,
) -> str:
    if fail_frame_ratio >= 0.10:
        return "fail"
    if fail_frame_count > 0 or base_status == "fail":
        return "review"
    return base_status


def combined_temporal_status(
    *,
    manual_review_status: str,
    temporal_status: str,
    sam3_containment_status: str,
) -> str:
    if manual_review_status == "fail":
        return "fail"
    if manual_review_status == "pass":
        return "pass"
    if temporal_status in {"review", "fail"} or sam3_containment_status in {
        "review",
        "fail",
    }:
        return "review"
    if temporal_status in {"not_run", "blocked"} or sam3_containment_status in {
        "not_run",
        "blocked",
    }:
        return "review"
    return "pass"


def abnormal_status(
    *,
    abnormal_fail_frame_ratio: float,
    manual_review_status: str,
    temporal_status: str,
    sam3_containment_status: str,
) -> str:
    if abnormal_fail_frame_ratio >= 0.10:
        return "fail"
    if manual_review_status == "pass":
        return "pass"
    if temporal_status in {"review", "fail"} or sam3_containment_status in {
        "review",
        "fail",
    }:
        return "review"
    if temporal_status in {"not_run", "blocked"} or sam3_containment_status in {
        "not_run",
        "blocked",
    }:
        return "review"
    return "pass"


def count_fail_indicators(
    *,
    text_check_status: str,
    video_quality_status: str,
    skeleton_static_status: str,
    abnormal_frame_status: str,
) -> int:
    return sum(
        status == "fail"
        for status in (
            text_check_status,
            video_quality_status,
            skeleton_static_status,
            abnormal_frame_status,
        )
    )


def recompute_xjgt_final_status(
    *,
    text_check_status: str,
    video_quality_status: str,
    skeleton_static_status: str,
    abnormal_frame_status: str,
    expected_module_missing: bool,
) -> str:
    # quality_hand is intentionally absent: it is supplier evidence, not a
    # vendor-agnostic skeleton validity or final acceptance condition.
    fail_indicator_count = count_fail_indicators(
        text_check_status=text_check_status,
        video_quality_status=video_quality_status,
        skeleton_static_status=skeleton_static_status,
        abnormal_frame_status=abnormal_frame_status,
    )
    if fail_indicator_count >= 2:
        return "fail"
    if abnormal_frame_status == "review" or expected_module_missing:
        return "review"
    return "pass"


def event_frame_interval(
    event: dict[str, Any],
) -> tuple[int, int] | None:
    start = integer_or_none(
        first_value(event, ("start_frame", "window_start_frame", "frame_idx"))
    )
    end = integer_or_none(
        first_value(
            event,
            ("end_frame", "window_end_frame", "frame_idx"),
        )
    )
    if start is None or end is None:
        return None
    return (start, end)


def event_is_hard_fail(event: dict[str, Any]) -> bool:
    return text(
        first_value(event, ("source_verdict", "auto_verdict"))
    ).lower() in {"fail", "failed"}


def event_is_abnormal_auto_fail(event: dict[str, Any]) -> bool:
    source_module = text(event.get("source_module")).lower()
    failure_mode = text(
        first_value(event, ("failure_mode", "issue_type"))
    ).lower()
    if source_module == "manual_review":
        return False
    if failure_mode in {
        "projection_review",
        "projection_ambiguous",
        "acceptable_flagged",
    }:
        return False
    is_temporal_or_containment = (
        "sam3" in source_module
        or "containment" in source_module
        or "temporal" in source_module
        or "containment" in failure_mode
        or failure_mode in {"strong_containment_mismatch", "temporal_jump"}
    )
    if not is_temporal_or_containment:
        return False
    return event_is_hard_fail(event) or failure_mode in {
        "containment_fail",
        "strong_containment_mismatch",
    }


def event_failure_category(event: dict[str, Any]) -> str:
    source_module = text(event.get("source_module")).lower()
    failure_mode = text(
        first_value(event, ("failure_mode", "issue_type"))
    ).lower()
    if source_module == "video_quality" or failure_mode.startswith("video_"):
        return "video_quality"
    if any(
        token in failure_mode
        for token in (
            "morphology",
            "bone_length",
            "collapsed_finger",
            "duplicate_joint",
        )
    ):
        return "skeleton_morphology"
    if any(
        token in failure_mode
        for token in (
            "keypoint_raw_invalid",
            "keypoint_missing",
            "missing_keypoint",
            "valid_point",
            "nan",
            "inf",
        )
    ):
        return "skeleton_missing"
    return ""


def supplier_quality_signal(value: Any) -> str:
    status = text(value).strip().lower()
    if not status or status in {"not_run", "not_provided", "not_applicable"}:
        return "not_provided"
    if status in {"fail", "failed", "risk", "review", "low"}:
        return "low"
    if status in {"pass", "passed", "ok"}:
        return "provided_ok"
    return f"provided:{status}"


def print_sanity_checks(
    xjgt_summary: dict[str, Any],
    xjgt_details: list[dict[str, Any]],
    deepreach_summary: dict[str, Any],
) -> None:
    total_frames = int(xjgt_summary["total_frame_count"])
    problem_frames = int(xjgt_summary["problem_frame_count"])
    has_nonzero_manual_ratio = any(
        float(row["manual_problem_frame_ratio_of_clip"]) > 0
        for row in xjgt_details
    )
    final_counts = dict(
        sorted(Counter(row["final_clip_status"] for row in xjgt_details).items())
    )
    print(f"XJGT total_frame_count={total_frames} ok={total_frames > 0}")
    print(
        f"XJGT problem_frame_count={problem_frames} "
        f"ok={problem_frames > 0}"
    )
    print(
        "XJGT manual_problem_ratio_nonzero="
        f"{has_nonzero_manual_ratio}"
    )
    print(f"XJGT final_clip_status counts={final_counts}")
    print(
        "DeepReach sample_clip_count="
        f"{deepreach_summary['sample_clip_count']} "
        "video_quality="
        f"{deepreach_summary['video_quality_status']}"
    )


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
