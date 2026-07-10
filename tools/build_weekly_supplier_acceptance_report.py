#!/usr/bin/env python3
"""Build the minimal weekly five-supplier acceptance workbook."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.utils import get_column_letter

if __package__:
    from tools.build_batch_qc_ledger import normalize_video_quality_row
else:
    from build_batch_qc_ledger import normalize_video_quality_row


LOGGER = logging.getLogger("build_weekly_supplier_acceptance_report")

SUMMARY_COLUMNS = [
    "supplier_name",
    "sample_clip_count",
    "total_frame_count",
    "problem_frame_count",
    "problem_frame_ratio",
    "pass_clip_count",
    "pass_clip_ratio",
    "fail_clip_count",
    "fail_clip_ratio",
]

CORE_DETAIL_COLUMNS = [
    "asset_id",
    "total_frames",
    "text_check_status",
    "skeleton_static_status",
    "video_quality_status",
    "abnormal_frame_status",
    "fail_indicator_count",
    "acceptance_status",
    "abnormal_frame_status_v2",
    "acceptance_status_v2",
    "review_status_v2",
]

TEXT_DETAIL_COLUMNS = ["text_status_reason"]

VIDEO_DETAIL_COLUMNS = [
    "video_quality_fail_frame_count",
    "video_quality_fail_frame_ratio",
    "video_quality_status_reason",
]

SKELETON_DETAIL_COLUMNS = [
    "skeleton_missing_status",
    "skeleton_missing_fail_frame_count",
    "skeleton_missing_fail_frame_ratio",
    "skeleton_missing_fail_intervals",
    "skeleton_morphology_status",
    "skeleton_morphology_fail_frame_count",
    "skeleton_morphology_fail_frame_ratio",
    "skeleton_morphology_fail_intervals",
    "skeleton_static_fail_frame_count",
    "skeleton_static_fail_frame_ratio",
    "skeleton_static_fail_intervals",
    "skeleton_static_status_reason",
    "supplier_quality_signal",
]

ABNORMAL_DETAIL_COLUMNS = [
    "temporal_status",
    "sam3_containment_status",
    "manual_review_status",
    "abnormal_fail_frame_count",
    "abnormal_fail_frame_ratio",
    "auto_fail_frame_count",
    "reviewed_auto_fail_frame_count",
    "reviewed_auto_fail_true_positive_frame_count",
    "reviewed_auto_fail_false_positive_frame_count",
    "unreviewed_auto_fail_frame_count",
    "auto_fail_precision_on_reviewed",
    "manual_problem_frame_count",
    "manual_problem_frame_ratio_of_clip",
    "manual_reviewed_frame_count",
    "manual_problem_ratio_of_reviewed",
    "abnormal_status_reason",
    "abnormal_fail_frame_count_v2",
    "abnormal_fail_frame_ratio_v2",
    "abnormal_status_reason_v2",
    "submitted_review_interval_count",
    "submitted_review_frame_count",
    "reviewed_submitted_interval_count",
    "unreviewed_submitted_interval_count",
    "unreviewed_submitted_frame_count",
    "abnormal_v1_unreviewed_as_fail_frame_count",
    "abnormal_v2_unreviewed_as_review_frame_count",
]

DIAGNOSTIC_DETAIL_COLUMNS = [
    "frame_count_status",
    "mapped_precheck_checks",
    "missing_expected_checks",
    "precheck_mapping_status",
    "final_status_reason",
    "main_issue_type",
    "notes",
    "evidence_path",
]

XJGT_DETAIL_COLUMNS = [
    *CORE_DETAIL_COLUMNS,
    *TEXT_DETAIL_COLUMNS,
    *VIDEO_DETAIL_COLUMNS,
    *SKELETON_DETAIL_COLUMNS,
    *ABNORMAL_DETAIL_COLUMNS,
    *DIAGNOSTIC_DETAIL_COLUMNS,
]
DETAIL_COLUMNS = XJGT_DETAIL_COLUMNS

MANUAL_ISSUE_COLUMNS = [
    "supplier_name",
    "issue_type",
    "confirmed_issue_clip_count",
    "confirmed_issue_clip_ratio_of_reviewed",
    "confirmed_problem_frame_count",
    "confirmed_problem_frame_ratio_of_reviewed_frames",
    "manual_label_count",
    "true_positive_count",
    "false_positive_count",
    "acceptable_flagged_count",
    "review_count",
]

THRESHOLD_RULE_COLUMNS = [
    "parent_indicator",
    "module",
    "check_name",
    "metric_or_field",
    "rule_type",
    "threshold_level",
    "operator",
    "effective_value",
    "unit",
    "aggregation_scope",
    "output_status",
    "config_key",
    "value_source",
    "source_path",
    "notes",
]

SHEET_NAMES = [
    "五供应商总览",
    "星际归途",
    "DeepReach",
    "供应商3",
    "供应商4",
    "供应商5",
    "人工问题与阈值",
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


@dataclass
class SourceRead:
    input_name: str
    path: Path | None
    records: list[dict[str, Any]]
    status: str


@dataclass
class PrecheckEvidence:
    by_asset: dict[str, dict[str, Any]]
    source_status: str
    unmatched_assets: list[str]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the minimal weekly five-supplier acceptance report."
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("outputs/acceptance_5x100"),
    )
    parser.add_argument(
        "--audit-inputs",
        action="store_true",
        help="Inspect candidate input schemas and mapping coverage without writing outputs.",
    )
    parser.add_argument(
        "--require-xjgt-text",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--skip-xjgt-text",
        action="store_true",
        help=(
            "Ignore XJGT text_integrity evidence and mark the text dimension "
            "not_applicable."
        ),
    )
    parser.add_argument("--xjgt-precheck-config", type=Path)
    parser.add_argument("--xjgt-video-quality-config", type=Path)
    parser.add_argument("--xjgt-sam3-config", type=Path)
    parser.add_argument("--xjgt-weekly-policy-config", type=Path)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if args.audit_inputs:
        print(json.dumps(collect_input_audit(args.run_root), indent=2, ensure_ascii=False))
        return 0
    outputs = build_weekly_report(
        args.run_root,
        require_xjgt_text=(
            args.require_xjgt_text or not args.skip_xjgt_text
        ),
        xjgt_precheck_config=args.xjgt_precheck_config,
        xjgt_video_quality_config=args.xjgt_video_quality_config,
        xjgt_sam3_config=args.xjgt_sam3_config,
        xjgt_weekly_policy_config=args.xjgt_weekly_policy_config,
    )
    LOGGER.info("Wrote %s", outputs.workbook_xlsx)
    LOGGER.info("Wrote %s", outputs.summary_csv)
    return 0


def build_weekly_report(
    run_root: Path,
    *,
    require_xjgt_text: bool = True,
    xjgt_precheck_config: Path | None = None,
    xjgt_video_quality_config: Path | None = None,
    xjgt_sam3_config: Path | None = None,
    xjgt_weekly_policy_config: Path | None = None,
) -> WeeklyOutputPaths:
    run_root.mkdir(parents=True, exist_ok=True)
    xjgt = load_xjgt(
        run_root,
        require_xjgt_text=require_xjgt_text,
        precheck_config=xjgt_precheck_config,
        video_quality_config=xjgt_video_quality_config,
        sam3_config=xjgt_sam3_config,
        weekly_policy_config=xjgt_weekly_policy_config,
    )
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
        xjgt["manual_issue_rows"],
        xjgt["threshold_rule_rows"],
        xjgt["config_audit"],
    )
    print_sanity_checks(
        xjgt["summary"],
        xjgt["details"],
        deepreach["summary"],
    )
    return WeeklyOutputPaths(workbook_xlsx, summary_csv)


def load_xjgt_source_reads(run_root: Path) -> dict[str, SourceRead]:
    precheck_dir = run_root / "xjgt" / "precheck"
    video_dir = run_root / "xjgt" / "video_quality"
    sam3_dir = run_root / "xjgt" / "sam3_containment"
    ledger_dir = run_root / "xjgt" / "ledger"
    manual_dir = run_root / "xjgt" / "manual_review"
    return {
        "precheck_check_results": read_first_records(
            "precheck_check_results",
            [
                precheck_dir / "check_results.parquet",
                precheck_dir / "check_results.csv",
                precheck_dir / "check_results.json",
            ],
        ),
        "precheck_clip_aggregates": read_first_records(
            "precheck_clip_aggregates",
            [
                precheck_dir / "clip_aggregates.parquet",
                precheck_dir / "clip_aggregates.csv",
                precheck_dir / "clip_aggregates.json",
            ],
        ),
        "precheck_candidate_windows": read_first_records(
            "precheck_candidate_windows",
            [
                precheck_dir / "candidate_windows.parquet",
                precheck_dir / "candidate_windows.csv",
                precheck_dir / "candidate_windows.json",
            ],
        ),
        "video_quality_summary": read_first_records(
            "video_quality_summary",
            [
                video_dir / "video_quality_decision_summary.csv",
                video_dir / "video_quality_acceptance_summary.csv",
            ],
        ),
        "video_quality_results": read_first_records(
            "video_quality_results",
            [video_dir / "video_quality_results.json"],
        ),
        "sam3_window_summary": read_first_records(
            "sam3_window_summary",
            [
                sam3_dir / "window_keypoint_containment_summary.parquet",
                sam3_dir / "window_keypoint_containment_summary.csv",
                sam3_dir / "window_keypoint_containment_summary.json",
            ],
        ),
        "ledger_asset_ledger": read_first_records(
            "ledger_asset_ledger",
            [ledger_dir / "xjgt_100_asset_ledger.csv"],
        ),
        "ledger_issue_events": read_first_records(
            "ledger_issue_events",
            [ledger_dir / "xjgt_100_issue_events.csv"],
        ),
        "manual_normalized_labels": read_first_records(
            "manual_normalized_labels",
            [manual_dir / "manual_labels_autosave.normalized.csv"],
        ),
        "manual_patch_labels": read_first_records(
            "manual_patch_labels",
            [manual_dir / "manual_labels_patch.json"],
        ),
        "manual_review_queue": read_first_records(
            "manual_review_queue",
            [
                run_root
                / "xjgt"
                / "video_review_full"
                / "review_queue_with_clips.csv",
                run_root / "xjgt" / "review_full" / "review_queue.csv",
                run_root
                / "xjgt"
                / "video_review"
                / "review_queue_with_clips.csv",
                run_root / "xjgt" / "review" / "review_queue.csv",
                run_root
                / "xjgt"
                / "manual_review"
                / "review_queue_with_clips.csv",
                run_root / "xjgt" / "manual_review" / "review_queue.csv",
                run_root / "xjgt" / "review_queue.csv",
            ],
        ),
    }


def build_episode_asset_map(manifest: list[dict[str, Any]]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for fallback_idx, row in enumerate(manifest):
        asset_id = normalize_asset_id(row.get("asset_id"))
        if not asset_id:
            continue
        episode_idx = integer_or_none(row.get("episode_idx"))
        mapping[episode_idx if episode_idx is not None else fallback_idx] = asset_id
    return mapping


def source_asset_id(
    row: dict[str, Any],
    episode_asset: dict[int, str],
) -> str:
    for key in ("asset_id", "clip_id"):
        asset_id = normalize_asset_id(row.get(key))
        if asset_id:
            return asset_id
    for key in ("video_path", "hdf5_path", "path"):
        asset_id = normalize_asset_id(row.get(key))
        if asset_id:
            return asset_id
    episode_idx = integer_or_none(row.get("episode_idx"))
    if episode_idx is not None:
        return episode_asset.get(episode_idx, f"episode_idx:{episode_idx}")
    return ""


def parse_metrics(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("metrics")
    if isinstance(raw, dict):
        return raw
    if raw in (None, ""):
        return {}
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return text(value).strip().lower() in {"true", "1", "yes", "fail", "failed"}


def boolish_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = text(value).strip().lower()
    if normalized in {"true", "1", "yes", "pass", "passed"}:
        return True
    if normalized in {"false", "0", "no", "fail", "failed"}:
        return False
    return None


def load_precheck_evidence(
    manifest: list[dict[str, Any]],
    sources: dict[str, SourceRead],
) -> PrecheckEvidence:
    episode_asset = build_episode_asset_map(manifest)
    check_rows = sources["precheck_check_results"].records
    candidate_rows = sources["precheck_candidate_windows"].records
    by_asset: dict[str, dict[str, Any]] = {}
    unmatched: set[str] = set()
    for row in manifest:
        asset_id = normalize_asset_id(row.get("asset_id"))
        if not asset_id:
            continue
        by_asset[asset_id] = {
            "checks": set(),
            "missing_intervals": [],
            "morphology_intervals": [],
            "text_status": "not_applicable",
            "text_atomic_statuses": {},
            "morphology_status": "not_run",
            "missing_status": "not_run",
            "supplier_quality_signal": "not_provided",
            "temporal_status": "not_run",
            "temporal_seen": False,
            "quality_seen": False,
            "has_skeleton_quality": False,
            "has_morphology": False,
        }

    for row in check_rows:
        asset_id = source_asset_id(row, episode_asset)
        if not asset_id or asset_id not in by_asset:
            if asset_id:
                unmatched.add(asset_id)
            continue
        record = by_asset[asset_id]
        check = text(row.get("check"))
        metrics = parse_metrics(row)
        frame_idx = integer_or_none(row.get("frame_idx"))
        record["checks"].add(check)

        if check == "text_integrity":
            for metric_name, value in metrics.items():
                if not metric_name.startswith(
                    ("field_present_", "field_nonempty_")
                ):
                    continue
                record["text_atomic_statuses"][
                    f"text_{metric_name}_status"
                ] = "pass" if truthy(value) else "fail"
            if truthy(row.get("flag")) or float_or_zero(metrics.get("missing_field_count")) > 0:
                record["text_status"] = "fail"
            elif record["text_status"] != "fail":
                record["text_status"] = "pass"
            continue

        if check == "quality_score":
            record["quality_seen"] = True
            if frame_idx == -1:
                record["supplier_quality_signal"] = supplier_quality_from_quality_score(row, metrics)
            elif float_or_zero(metrics.get("frame_score")) == 0.0:
                record["supplier_quality_signal"] = "low"
            continue

        if check == "keypoint_missing":
            # Despite its historical name, this check consumes supplier
            # quality_hand windows; it is not vendor-agnostic point existence.
            record["quality_seen"] = True
            if (
                float_or_zero(metrics.get("quality_low_left")) > 0
                or float_or_zero(metrics.get("quality_low_right")) > 0
            ):
                record["supplier_quality_signal"] = "low"
            elif record["supplier_quality_signal"] == "not_provided":
                record["supplier_quality_signal"] = "provided_ok"
            continue

        if check == "skeleton_quality_score":
            record["has_skeleton_quality"] = True
            record["temporal_seen"] = True
            if frame_idx is not None and frame_idx >= 0:
                if skeleton_presence_invalid(metrics):
                    record["missing_intervals"].append((frame_idx, frame_idx))
                verdict = text(metrics.get("skeleton_verdict")).lower()
                if verdict in {"review", "suspect"} or truthy(metrics.get("needs_visual_review")):
                    record["temporal_status"] = "review"
            continue

        if check == "keypoint_temporal":
            record["temporal_seen"] = True
            continue

        if check == "keypoint_morphology":
            record["has_morphology"] = True
            verdict = status_value(metrics.get("morphology_verdict"))
            if verdict in {"fail", "review", "pass", "not_applicable"}:
                record["morphology_status"] = worst_status(record["morphology_status"], verdict)
            if frame_idx is not None and frame_idx >= 0 and (
                verdict == "fail" or truthy(row.get("flag"))
            ):
                record["morphology_intervals"].append((frame_idx, frame_idx))

    for row in candidate_rows:
        asset_id = source_asset_id(row, episode_asset)
        if not asset_id or asset_id not in by_asset:
            if asset_id:
                unmatched.add(asset_id)
            continue
        by_asset[asset_id]["temporal_status"] = "review"
        by_asset[asset_id]["temporal_seen"] = True

    for record in by_asset.values():
        if record["has_skeleton_quality"]:
            record["missing_status"] = "fail" if record["missing_intervals"] else "pass"
        if record["has_morphology"] and record["morphology_status"] == "not_run":
            record["morphology_status"] = "pass"
        record["mapped_precheck_checks"] = "|".join(sorted(record["checks"]))
        missing_expected = []
        if not record["has_skeleton_quality"]:
            missing_expected.append("skeleton_quality_score")
        if not record["has_morphology"]:
            missing_expected.append("keypoint_morphology")
        record["missing_expected_checks"] = "|".join(missing_expected)
        if not record["checks"]:
            record["precheck_mapping_status"] = (
                "unmatched" if check_rows or candidate_rows else "source_missing"
            )
        elif missing_expected:
            record["precheck_mapping_status"] = "partial"
        else:
            record["precheck_mapping_status"] = "mapped"

    source_status = sources["precheck_check_results"].status
    return PrecheckEvidence(
        by_asset=by_asset,
        source_status=source_status,
        unmatched_assets=sorted(unmatched),
    )


def skeleton_presence_invalid(metrics: dict[str, Any]) -> bool:
    saw_geometry_count = False
    geometry_invalid = False
    for side in ("left", "right"):
        valid = float_or_none(metrics.get(f"valid_keypoint_count_{side}"))
        missing = float_or_none(metrics.get(f"missing_keypoint_count_{side}"))
        if valid is not None:
            saw_geometry_count = True
            geometry_invalid = geometry_invalid or valid < 21
        if missing is not None:
            saw_geometry_count = True
            geometry_invalid = geometry_invalid or missing > 0
    if saw_geometry_count:
        return geometry_invalid
    return truthy(metrics.get("keypoint_presence_invalid")) and not truthy(
        metrics.get("low_quality_hand_invalid")
    )


def supplier_quality_from_quality_score(
    row: dict[str, Any],
    metrics: dict[str, Any],
) -> str:
    if truthy(row.get("flag")):
        return "provided_ok"
    pass_ratio = float_or_none(metrics.get("pass_ratio"))
    threshold = float_or_none(metrics.get("pass_threshold"))
    if pass_ratio is not None and threshold is not None:
        return "provided_ok" if pass_ratio >= threshold else "low"
    return "not_provided"


def worst_status(current: str, incoming: str) -> str:
    rank = {
        "fail": 5,
        "review": 4,
        "blocked": 3,
        "pass": 2,
        "not_applicable": 1,
        "not_run": 0,
    }
    return incoming if rank.get(incoming, 0) > rank.get(current, 0) else current


def load_xjgt(
    run_root: Path,
    *,
    require_xjgt_text: bool = True,
    precheck_config: Path | None = None,
    video_quality_config: Path | None = None,
    sam3_config: Path | None = None,
    weekly_policy_config: Path | None = None,
) -> dict[str, Any]:
    manifest_path = run_root / "manifests" / "supplier_manifest_xjgt_100.csv"
    manifest = read_csv(manifest_path)
    asset_rows, episode_asset = manifest_asset_maps(manifest)
    asset_ids = set(asset_rows)
    sources = load_xjgt_source_reads(run_root)
    config_paths = discover_xjgt_config_paths(
        run_root,
        precheck_config=precheck_config,
        video_quality_config=video_quality_config,
        sam3_config=sam3_config,
        weekly_policy_config=weekly_policy_config,
    )
    run_config_rows, config_audit = build_run_config_inventory(
        config_paths, sources
    )
    precheck = load_precheck_evidence(manifest, sources)
    candidate_by_asset, candidate_unmatched = map_window_rows(
        sources["precheck_candidate_windows"].records,
        asset_ids,
        episode_asset,
    )
    video_by_asset, video_unmatched = map_video_quality_evidence(
        [
            *sources["video_quality_results"].records,
            *sources["video_quality_summary"].records,
        ],
        asset_ids,
        episode_asset,
    )
    sam3_by_asset, sam3_unmatched = map_sam3_evidence(
        sources["sam3_window_summary"].records,
        asset_ids,
        episode_asset,
    )
    submitted_review_by_asset, submitted_review_unmatched = (
        map_submitted_review_intervals(
            sources["manual_review_queue"].records,
            asset_ids,
            episode_asset,
        )
    )
    manual_labels = dedupe_manual_labels(
        [
            *sources["manual_normalized_labels"].records,
            *sources["manual_patch_labels"].records,
        ]
    )
    ledger_by_asset = {
        normalize_asset_id(row.get("asset_id")): row
        for row in sources["ledger_asset_ledger"].records
    }
    frame_info: dict[str, dict[str, Any]] = {}
    for asset_id, manifest_row in asset_rows.items():
        video_path = resolve_xjgt_video_path(manifest_row, asset_id)
        probed_frames = probe_video_frame_count(video_path)
        fallback_frames = integer_or_none(
            first_value(
                manifest_row,
                ("frame_count", "total_frames", "num_frames"),
            )
        ) or 0
        frame_info[asset_id] = {
            "total_frames": probed_frames or fallback_frames,
            "frame_count_status": (
                "ok"
                if probed_frames > 0
                else "manifest_fallback"
                if fallback_frames > 0
                else "unreadable"
            ),
            "video_path": video_path,
        }
    manual_by_asset = aggregate_manual_review(
        manual_labels,
        {
            asset_id: int(info["total_frames"])
            for asset_id, info in frame_info.items()
        },
    )
    evidence_paths = [
        source.path
        for source in sources.values()
        if source.path is not None
    ]
    details = [
        build_xjgt_detail(
            asset_id=asset_id,
            total_frames=int(frame_info[asset_id]["total_frames"]),
            frame_count_status=str(frame_info[asset_id]["frame_count_status"]),
            video_path=Path(frame_info[asset_id]["video_path"]),
            precheck_row=precheck.by_asset.get(
                asset_id, empty_precheck_evidence()
            ),
            precheck_source_status=precheck.source_status,
            precheck_unmatched=precheck.unmatched_assets,
            candidate_row=candidate_by_asset.get(asset_id, {}),
            video_row=video_by_asset.get(asset_id, {}),
            sam3_row=sam3_by_asset.get(asset_id, {}),
            submitted_review_row=submitted_review_by_asset.get(asset_id, {}),
            submitted_review_source_status=sources["manual_review_queue"].status,
            manual_row=manual_by_asset.get(
                asset_id,
                empty_manual_evidence(
                    int(frame_info[asset_id]["total_frames"])
                ),
            ),
            ledger_row=ledger_by_asset.get(asset_id, {}),
            require_xjgt_text=require_xjgt_text,
            evidence_paths=evidence_paths,
        )
        for asset_id in asset_rows
    ]
    manual_main_by_asset = manual_main_issue_by_asset(manual_labels)
    for row in details:
        row["main_issue_type"] = manual_main_by_asset.get(
            row["asset_id"], ""
        )
    manual_issue_rows = build_manual_issue_rows(
        supplier_name="星际归途 / XJGT",
        labels=manual_labels,
    )
    modules_completed = []
    if details and all(
        row["precheck_mapping_status"] == "mapped" for row in details
    ):
        modules_completed.append("precheck")
    if details and all(
        row["video_quality_status"] in {"pass", "fail"}
        for row in details
    ):
        modules_completed.append("video_quality")
    if sources["sam3_window_summary"].status == "readable":
        modules_completed.append("sam3")
    if sources["manual_normalized_labels"].status == "readable":
        modules_completed.append("manual_review")
    blocked_modules = []
    if "precheck" not in modules_completed:
        blocked_modules.append("precheck_mapping_incomplete")
    if "video_quality" not in modules_completed:
        blocked_modules.append("video_quality_mapping_incomplete")
    summary = supplier_summary_from_details(
        supplier_name="星际归途 / XJGT",
        expected_clip_count=100,
        details=details,
        main_issue_type=(
            manual_issue_rows[0]["issue_type"]
            if manual_issue_rows
            else ""
        ),
        modules_completed="|".join(modules_completed),
        blocked_modules="|".join(blocked_modules),
        notes=(
            "High-risk-biased manual review; not an unbiased global "
            "error-rate estimate. "
            f"unmatched_precheck={len(precheck.unmatched_assets)}; "
            f"unmatched_candidate={len(candidate_unmatched)}; "
            f"unmatched_video={len(video_unmatched)}; "
            f"unmatched_sam3={len(sam3_unmatched)}"
            f"; unmatched_submitted_review={len(submitted_review_unmatched)}"
        ),
    )
    threshold_rule_rows = build_threshold_rule_rows(
        run_root,
        sources,
        config_paths=config_paths,
        run_config_rows=run_config_rows,
    )
    exported_run_config_rows = sum(
        row["value_source"] == "run_config" for row in threshold_rule_rows
    )
    config_audit["exported_run_config_row_count"] = exported_run_config_rows
    if exported_run_config_rows != config_audit["actual_config_leaf_count"]:
        config_audit["missing_config_keys"].append(
            "run_config_row_reconciliation"
        )
        config_audit["config_reconciliation_status"] = "incomplete"
    config_audit.update(
        {
            "submitted_review_source_path": str(
                sources["manual_review_queue"].path or ""
            ),
            "submitted_review_source_status": sources[
                "manual_review_queue"
            ].status,
            "code_default_row_count": sum(
                row["value_source"] == "code_default"
                for row in threshold_rule_rows
            ),
            "producer_metadata_row_count": sum(
                row["value_source"] == "producer_output_metadata"
                for row in threshold_rule_rows
            ),
            "unresolved_row_count": sum(
                row["value_source"] == "unavailable"
                or row["effective_value"] == "unresolved"
                for row in threshold_rule_rows
            ),
        }
    )
    return {
        "summary": summary,
        "details": details,
        "manual_issue_rows": manual_issue_rows,
        "threshold_rule_rows": threshold_rule_rows,
        "config_audit": config_audit,
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
        asset_id = normalize_asset_id(row.get("asset_id"))
        video_path = Path(text(row.get("video_path")))
        probed_frames = probe_video_frame_count(video_path) if video_path else 0
        fallback_frames = integer_or_none(
            first_value(row, ("frame_count", "total_frames", "num_frames"))
        ) or 0
        total_frames = probed_frames or fallback_frames
        details.append(
            {
                "asset_id": asset_id,
                "total_frames": total_frames,
                "frame_count_status": (
                    "ok"
                    if probed_frames > 0
                    else "manifest_fallback"
                    if fallback_frames > 0
                    else "not_run"
                ),
                "text_check_status": "not_ready",
                "text_status_reason": "supplier_text_schema_not_defined",
                "skeleton_missing_status": "blocked",
                "skeleton_missing_fail_frame_count": 0,
                "skeleton_missing_fail_frame_ratio": 0.0,
                "skeleton_morphology_status": "blocked",
                "skeleton_morphology_fail_frame_count": 0,
                "skeleton_morphology_fail_frame_ratio": 0.0,
                "skeleton_static_fail_frame_count": 0,
                "skeleton_static_fail_frame_ratio": 0.0,
                "skeleton_static_status": "not_ready",
                "skeleton_static_status_reason": (
                    "missing_status=blocked; morphology_status=blocked; "
                    "skeleton_static_status=blocked"
                ),
                "supplier_quality_signal": "not_provided",
                "video_quality_status": "not_ready",
                "video_quality_fail_frame_count": 0,
                "video_quality_fail_frame_ratio": 0.0,
                "video_quality_status_reason": (
                    "no_valid_video_quality_output"
                    if not has_video_quality
                    else "video_quality_output_requires_review"
                ),
                "temporal_status": "blocked",
                "sam3_containment_status": "blocked",
                "manual_review_status": "not_run",
                "manual_problem_frame_count": 0,
                "manual_problem_frame_ratio_of_clip": 0.0,
                "manual_reviewed_frame_count": 0,
                "manual_problem_ratio_of_reviewed": 0.0,
                "abnormal_fail_frame_count": 0,
                "abnormal_fail_frame_ratio": 0.0,
                "abnormal_frame_status": "not_ready",
                "abnormal_frame_status_v2": "not_ready",
                "auto_fail_frame_count": 0,
                "reviewed_auto_fail_frame_count": 0,
                "reviewed_auto_fail_true_positive_frame_count": 0,
                "reviewed_auto_fail_false_positive_frame_count": 0,
                "unreviewed_auto_fail_frame_count": 0,
                "auto_fail_precision_on_reviewed": 0.0,
                "abnormal_status_reason": "blocked:sam3_projection_mapping",
                "abnormal_fail_frame_count_v2": 0,
                "abnormal_fail_frame_ratio_v2": 0.0,
                "abnormal_status_reason_v2": "blocked:sam3_projection_mapping",
                "submitted_review_interval_count": 0,
                "submitted_review_frame_count": 0,
                "reviewed_submitted_interval_count": 0,
                "unreviewed_submitted_interval_count": 0,
                "unreviewed_submitted_frame_count": 0,
                "abnormal_v1_unreviewed_as_fail_frame_count": 0,
                "abnormal_v2_unreviewed_as_review_frame_count": 0,
                "fail_indicator_count": 0,
                "acceptance_status": "not_ready",
                "review_status": "blocked",
                "acceptance_status_v2": "not_ready",
                "review_status_v2": "blocked",
                "mapped_precheck_checks": "",
                "missing_expected_checks": "skeleton_quality_score|keypoint_morphology",
                "precheck_mapping_status": "blocked",
                "final_status_reason": "blocked:missing_deepreach_hdf5_adapter",
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
    review_counts = Counter(row["review_status"] for row in details)
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
            "pass_clip_count": "",
            "fail_clip_count": 0,
            "review_clip_count": 0,
            "blocked_clip_count": review_counts["blocked"],
            "not_run_clip_count": review_counts["not_run"],
            "coverage_ratio": 0.0,
            "sample_coverage_ratio": safe_ratio(len(manifest), 100),
            "pass_clip_ratio": 0.0,
            "fail_clip_ratio": 0.0,
            "manual_reviewed_clip_count": 0,
            "manual_review_coverage_ratio": 0.0,
            "manual_pass_clip_count": 0,
            "manual_fail_clip_count": 0,
            "manual_not_reviewed_clip_count": len(manifest),
            "manual_confirmed_problem_frame_count": 0,
            "abnormal_pass_clip_count": 0,
            "abnormal_fail_clip_count": 0,
            "abnormal_review_clip_count": 0,
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
        "pass_clip_count": "",
        "fail_clip_count": 0,
        "review_clip_count": 0,
        "blocked_clip_count": 0,
        "not_run_clip_count": 0,
        "coverage_ratio": 0.0,
        "sample_coverage_ratio": 0.0,
        "pass_clip_ratio": 0.0,
        "fail_clip_ratio": 0.0,
        "manual_reviewed_clip_count": 0,
        "manual_review_coverage_ratio": 0.0,
        "manual_pass_clip_count": 0,
        "manual_fail_clip_count": 0,
        "manual_not_reviewed_clip_count": 0,
        "manual_confirmed_problem_frame_count": 0,
        "abnormal_pass_clip_count": 0,
        "abnormal_fail_clip_count": 0,
        "abnormal_review_clip_count": 0,
        "main_issue_type": "",
        "modules_completed": "",
        "blocked_modules": "missing_input",
        "notes": "input manifest not provided",
    }


def threshold_rule_row(
    *,
    parent_indicator: str,
    module: str,
    check_name: str,
    metric_or_field: str,
    rule_type: str,
    threshold_level: str,
    operator: str,
    effective_value: Any,
    unit: str,
    aggregation_scope: str,
    output_status: str,
    config_key: str,
    value_source: str,
    source_path: str,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "parent_indicator": parent_indicator,
        "module": module,
        "check_name": check_name,
        "metric_or_field": metric_or_field,
        "rule_type": rule_type,
        "threshold_level": threshold_level,
        "operator": operator,
        "effective_value": effective_value,
        "unit": unit,
        "aggregation_scope": aggregation_scope,
        "output_status": output_status,
        "config_key": config_key,
        "value_source": value_source,
        "source_path": source_path,
        "notes": notes,
    }


def load_precheck_rule_config(
    run_root: Path,
    actual_config_path: Path | None = None,
) -> tuple[dict[str, Any], str, str]:
    if actual_config_path is not None:
        loaded = read_config_mapping(actual_config_path)
        return loaded, "run_config", str(actual_config_path)
    candidates = [
        run_root / "xjgt" / "precheck" / "precheck_config.yaml",
        run_root / "xjgt" / "precheck" / "config.yaml",
        run_root / "xjgt" / "precheck" / "run_config.yaml",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            loaded = read_config_mapping(path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if isinstance(loaded, dict):
            return loaded, "run_config", str(path)

    default_path = Path(__file__).parents[1] / "configs" / "precheck_example.yaml"
    try:
        import yaml

        loaded = yaml.safe_load(default_path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, TypeError):
        return {}, "unavailable", str(default_path)
    return (
        loaded if isinstance(loaded, dict) else {},
        "code_default",
        str(default_path),
    )


def read_config_mapping(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        loaded = json.loads(path.read_text(encoding="utf-8"))
    else:
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"config root must be an object: {path}")
    return loaded


def discover_single_config(
    *,
    explicit: Path | None,
    preferred: list[Path],
    glob_dir: Path,
    glob_patterns: tuple[str, ...],
    option_name: str,
) -> Path | None:
    if explicit is not None:
        if not explicit.is_file():
            raise ValueError(f"{option_name} does not exist: {explicit}")
        return explicit.resolve()
    for path in preferred:
        if path.is_file():
            return path.resolve()
    matches = sorted(
        {
            path.resolve()
            for pattern in glob_patterns
            for path in glob_dir.glob(pattern)
            if path.is_file()
        }
    )
    if len(matches) > 1:
        raise ValueError(
            f"ambiguous config discovery for {option_name}: "
            f"{', '.join(str(path) for path in matches)}"
        )
    return matches[0] if matches else None


def logged_config_candidates(
    run_root: Path,
    log_name_tokens: tuple[str, ...],
) -> list[Path]:
    log_roots = [run_root / "xjgt" / "logs", run_root / "logs"]
    pattern = re.compile(
        r"(?:--config(?:=|\s+)|config(?:_path)?\s*[=:]\s*)"
        r"[\"']?([^\s\"']+\.(?:yaml|yml|json))"
    )
    candidates: set[Path] = set()
    for log_root in log_roots:
        if not log_root.is_dir():
            continue
        for log_path in log_root.glob("*.log"):
            normalized_name = log_path.name.lower().replace("-", "_")
            if not any(token in normalized_name for token in log_name_tokens):
                continue
            try:
                content = log_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for raw_path in pattern.findall(content):
                path = Path(raw_path)
                search = (
                    [path]
                    if path.is_absolute()
                    else [
                        Path.cwd() / path,
                        Path(__file__).parents[1] / path,
                        run_root / path,
                        log_path.parent / path,
                    ]
                )
                candidates.update(
                    candidate.resolve()
                    for candidate in search
                    if candidate.is_file()
                )
    return sorted(candidates)


def discover_xjgt_config_paths(
    run_root: Path,
    *,
    precheck_config: Path | None,
    video_quality_config: Path | None,
    sam3_config: Path | None,
    weekly_policy_config: Path | None,
) -> dict[str, Path | None]:
    xjgt = run_root / "xjgt"
    logged_precheck = logged_config_candidates(run_root, ("precheck",))
    logged_video = logged_config_candidates(
        run_root, ("video_quality", "videoquality")
    )
    logged_weekly = logged_config_candidates(run_root, ("weekly", "report"))
    for option_name, candidates in (
        ("--xjgt-precheck-config", logged_precheck),
        ("--xjgt-video-quality-config", logged_video),
        ("--xjgt-weekly-policy-config", logged_weekly),
    ):
        if len(candidates) > 1:
            raise ValueError(
                f"ambiguous logged config paths for {option_name}: "
                f"{', '.join(str(path) for path in candidates)}"
            )
    return {
        "precheck": discover_single_config(
            explicit=precheck_config,
            preferred=[
                xjgt / "precheck" / "precheck_config.yaml",
                xjgt / "precheck" / "config.yaml",
                xjgt / "precheck" / "run_config.yaml",
                *logged_precheck,
            ],
            glob_dir=xjgt / "precheck",
            glob_patterns=("*.yaml", "*.yml"),
            option_name="--xjgt-precheck-config",
        ),
        "video_quality": discover_single_config(
            explicit=video_quality_config,
            preferred=[
                xjgt / "video_quality" / "video_quality_config.yaml",
                xjgt / "video_quality" / "config.yaml",
                *logged_video,
            ],
            glob_dir=xjgt / "video_quality",
            glob_patterns=("*.yaml", "*.yml"),
            option_name="--xjgt-video-quality-config",
        ),
        "sam3_containment": discover_single_config(
            explicit=sam3_config,
            preferred=[
                xjgt / "sam3_containment" / "sam3_config.yaml",
                xjgt / "sam3_containment" / "config.yaml",
                xjgt / "sam3_containment" / "run_manifest.json",
            ],
            glob_dir=xjgt / "sam3_containment",
            glob_patterns=("*.yaml", "*.yml", "*config*.json", "run_manifest.json"),
            option_name="--xjgt-sam3-config",
        ),
        "weekly_report": discover_single_config(
            explicit=weekly_policy_config,
            preferred=[
                xjgt / "weekly_policy.yaml",
                xjgt / "weekly_policy.json",
                run_root / "weekly_policy.yaml",
                *logged_weekly,
            ],
            glob_dir=xjgt,
            glob_patterns=("*weekly*policy*.yaml", "*weekly*policy*.yml", "*weekly*policy*.json"),
            option_name="--xjgt-weekly-policy-config",
        ),
    }


def config_leaf_items(
    value: Any,
    prefix: str = "",
) -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        output: list[tuple[str, Any]] = []
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            output.extend(config_leaf_items(nested, path))
        return output
    if isinstance(value, (list, tuple)):
        if not value:
            return [(prefix, [])]
        output = []
        for index, nested in enumerate(value):
            path = f"{prefix}.{index}" if prefix else str(index)
            output.extend(config_leaf_items(nested, path))
        return output
    return [(prefix, value)]


def config_parent_and_check(module: str, config_key: str) -> tuple[str, str]:
    parts = config_key.split(".")
    if module == "precheck":
        check = parts[1] if parts[0] == "checks" and len(parts) > 1 else parts[0]
        if check == "text_integrity":
            return "text", check
        if check == "skeleton_quality_score" and any(
            metric in config_key
            for metric in (
                "joint_angle_change",
                "rotation_delta",
                "joint_acceleration",
                "joint_displacement",
                "candidate_",
                "strong_",
                "hard_exceeded",
            )
        ):
            return "abnormal_frame", "keypoint_temporal"
        if check in {"keypoint_morphology", "keypoint_missing"} or any(
            token in config_key for token in ("missing_keypoint", "valid_keypoint")
        ):
            return "skeleton_static", check
        return "abnormal_frame", check
    if module == "video_quality":
        return "video_quality", parts[0]
    if module == "sam3_containment":
        return "abnormal_frame", "sam3_runtime_config"
    return "final_acceptance", "weekly_policy_config"


def config_metric_and_semantics(
    module: str,
    config_key: str,
    value: Any,
) -> tuple[str, str, str, str, str]:
    leaf = config_key.rsplit(".", 1)[-1]
    parent_leaf = config_key.rsplit(".", 2)[-2] if "." in config_key else leaf
    metric = str(value) if parent_leaf == "required_fields" else leaf
    threshold_tokens = (
        "threshold",
        "_min",
        "_max",
        "_pass",
        "_warn",
        "_fail",
        "_review",
        "ratio",
    )
    is_threshold = any(token in leaf for token in threshold_tokens)
    rule_type = "numeric_threshold" if is_threshold else "config_parameter"
    level = (
        "fail"
        if "fail" in leaf
        else "review"
        if "warn" in leaf or "review" in leaf
        else "pass"
        if "pass" in leaf
        else "configured"
    )
    if not is_threshold:
        return metric, rule_type, level, "==", "configured"
    if module == "video_quality":
        normalized_metric, level, operator, _unit, output = video_threshold_semantics(
            config_key
        )
        return normalized_metric, rule_type, level, operator, output
    operator = (
        "<="
        if "max" in leaf and "min" not in leaf
        else ">="
        if "min" in leaf
        else ">="
    )
    normalized_metric = metric.removesuffix("_threshold")
    if "keypoint_morphology" in config_key:
        normalized_metric = {
            "max_bone_length_ratio_spread_review": "bone_length_ratio_spread",
            "max_bone_length_ratio_spread_fail": "bone_length_ratio_spread",
            "max_normalized_bone_length_review": "normalized_bone_length_max",
            "max_normalized_bone_length_fail": "normalized_bone_length_max",
            "max_zero_length_bone_count_review": "zero_length_bone_count",
            "max_zero_length_bone_count_fail": "zero_length_bone_count",
            "max_duplicate_joint_pair_count_review": "duplicate_joint_pair_count",
            "max_duplicate_joint_pair_count_fail": "duplicate_joint_pair_count",
            "min_joint_angle_deg_review": "joint_angle_min_deg",
            "min_joint_angle_deg_fail": "joint_angle_min_deg",
            "max_joint_angle_violation_fraction_review": "joint_angle_violation_fraction",
            "max_joint_angle_violation_fraction_fail": "joint_angle_violation_fraction",
        }.get(leaf, normalized_metric)
    return normalized_metric, rule_type, level, operator, level


def config_unit(config_key: str, value: Any) -> str:
    leaf = config_key.rsplit(".", 1)[-1]
    if isinstance(value, bool):
        return "boolean"
    if "ratio" in leaf or "fraction" in leaf:
        return "ratio"
    if leaf.endswith("_deg") or "angle" in leaf and "degree" in leaf:
        return "deg"
    if leaf.endswith("_m_s2") or "acceleration_m_s2" in leaf:
        return "m/s^2"
    if leaf.endswith("_m") or "displacement_m" in leaf:
        return "m"
    if "frame" in leaf or "frames" in leaf:
        return "frame"
    if "fps" in leaf:
        return "fps"
    if leaf.endswith("_sec") or "seconds" in leaf:
        return "sec"
    return "text" if isinstance(value, str) else "scalar"


def build_run_config_inventory(
    config_paths: dict[str, Path | None],
    sources: dict[str, SourceRead],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    discovered: list[str] = []
    actual_leaf_count = 0
    for module, path in config_paths.items():
        source_path = str(path) if path is not None else ""
        mapping: dict[str, Any] | None = (
            read_config_mapping(path) if path is not None else None
        )
        if module == "video_quality" and mapping is None:
            metadata = first_video_threshold_metadata(sources)
            if metadata is not None:
                mapping, source_path = metadata
        if mapping is None:
            missing.append(module)
            continue
        discovered.append(source_path)
        leaves = config_leaf_items(mapping)
        actual_leaf_count += len(leaves)
        for config_key, value in leaves:
            parent, check_name = config_parent_and_check(module, config_key)
            metric, rule_type, level, operator, output = config_metric_and_semantics(
                module, config_key, value
            )
            rows.append(
                threshold_rule_row(
                    parent_indicator=parent,
                    module=module,
                    check_name=check_name,
                    metric_or_field=metric,
                    rule_type=rule_type,
                    threshold_level=level,
                    operator=operator,
                    effective_value=value,
                    unit=config_unit(config_key, value),
                    aggregation_scope="run_configuration",
                    output_status=output,
                    config_key=config_key,
                    value_source="run_config",
                    source_path=source_path,
                    notes="Exact scalar leaf from the executed run configuration snapshot.",
                )
            )
    exported = len(rows)
    if exported != actual_leaf_count:
        missing.append(
            f"config_leaf_reconciliation:{actual_leaf_count - exported}"
        )
    audit = {
        "actual_config_files_discovered": discovered,
        "actual_config_leaf_count": actual_leaf_count,
        "exported_run_config_row_count": exported,
        "missing_config_keys": missing,
        "config_reconciliation_status": (
            "complete" if not missing and exported == actual_leaf_count else "incomplete"
        ),
    }
    return rows, audit


def load_precheck_code_defaults() -> tuple[dict[str, Any], str]:
    path = Path(__file__).parents[1] / "configs" / "precheck_example.yaml"
    try:
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, TypeError):
        return {}, str(path)
    return loaded if isinstance(loaded, dict) else {}, str(path)


def precheck_effective_value(
    *,
    section: str,
    key: str,
    run_config: dict[str, Any],
    run_value_source: str,
    run_source_path: str,
    code_defaults: dict[str, Any],
    code_default_path: str,
) -> tuple[Any, str, str]:
    run_section = run_config.get(section)
    run_section = run_section if isinstance(run_section, dict) else {}
    if key in run_section:
        return run_section[key], run_value_source, run_source_path
    default_section = code_defaults.get(section)
    default_section = default_section if isinstance(default_section, dict) else {}
    if key in default_section:
        return default_section[key], "code_default", code_default_path
    return "unresolved", "unavailable", run_source_path


def first_morphology_threshold_metadata(
    sources: dict[str, SourceRead],
) -> tuple[dict[str, Any], str] | None:
    source = sources["precheck_check_results"]
    for row in source.records:
        if text(row.get("check")) != "keypoint_morphology":
            continue
        thresholds = parse_metrics(row).get("thresholds")
        if isinstance(thresholds, dict):
            return thresholds, (
                f"{source.path}::keypoint_morphology.summary.metrics.thresholds"
            )
    return None


def inferred_text_fields(sources: dict[str, SourceRead]) -> list[str]:
    fields: set[str] = set()
    for row in sources["precheck_check_results"].records:
        if text(row.get("check")) != "text_integrity":
            continue
        for key in parse_metrics(row):
            for prefix in ("field_present_", "field_nonempty_"):
                if key.startswith(prefix):
                    fields.add(key.removeprefix(prefix))
    return sorted(fields)


def first_video_threshold_metadata(
    sources: dict[str, SourceRead],
) -> tuple[dict[str, Any], str] | None:
    source = sources["video_quality_results"]
    for row in source.records:
        video_quality = row.get("video_quality")
        if not isinstance(video_quality, dict):
            continue
        thresholds = video_quality.get("thresholds")
        if isinstance(thresholds, dict):
            return thresholds, f"{source.path}::video_quality.thresholds"
    return None


def nested_scalar_items(
    value: Any,
    prefix: str = "",
) -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        output: list[tuple[str, Any]] = []
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            output.extend(nested_scalar_items(nested, path))
        return output
    if isinstance(value, (list, tuple)):
        return [(prefix, "|".join(str(item) for item in value))]
    return [(prefix, value)]


def video_threshold_semantics(config_key: str) -> tuple[str, str, str, str, str]:
    leaf = config_key.rsplit(".", 1)[-1]
    section = config_key.split(".")
    level = (
        "fail"
        if leaf.endswith("_fail")
        else "review"
        if leaf.endswith("_warn")
        else "pass"
        if leaf.endswith("_pass")
        else "configured"
    )
    metric = leaf
    for suffix in ("_fail", "_warn", "_pass"):
        if metric.endswith(suffix):
            metric = metric[: -len(suffix)]
            break
    if leaf in {"ratio_pass", "ratio_warn"} and len(section) >= 2:
        metric = {
            "black": "black_frame_ratio",
            "over_dark": "mean_over_dark_ratio",
            "over_exposed": "mean_over_exposed_ratio",
        }.get(section[-2], f"{section[-2]}_ratio")
    high_is_bad = any(
        token in metric
        for token in (
            "ratio",
            "delta",
            "gap",
            "duration",
            "count",
            "hamming",
            "under_100",
        )
    )
    low_is_bad = any(
        token in metric
        for token in (
            "fps",
            "decode_ratio",
            "laplacian",
            "tenengrad",
            "short_side",
            "long_side",
            "available_ratio",
        )
    )
    if config_key.endswith(("black.mean_y_max", "over_dark.mean_y_max")):
        operator = "<="
    elif config_key.endswith("over_exposed.mean_y_min"):
        operator = ">="
    elif isinstance(metric, str) and leaf.endswith("_min"):
        operator = ">="
    elif leaf.endswith("_max"):
        operator = "<="
    elif high_is_bad:
        operator = ">"
    elif low_is_bad:
        operator = "<"
    else:
        operator = "=="
    unit = (
        "ratio"
        if "ratio" in metric
        else "frame"
        if "frame" in metric or "count" in metric
        else "ms"
        if metric.endswith("_ms")
        else "sec"
        if metric.endswith("_sec")
        else "fps"
        if "fps" in metric
        else "pixel"
        if any(token in metric for token in ("side", "width", "height", "px"))
        else "boolean"
        if leaf in {"enabled", "pts_monotonic_required"}
        else "score"
    )
    output_status = "fail" if level == "fail" else "review" if level == "review" else "pass"
    return metric, level, operator, unit, output_status


def build_threshold_rule_rows(
    run_root: Path,
    sources: dict[str, SourceRead],
    *,
    config_paths: dict[str, Path | None],
    run_config_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    config, config_source, config_path = load_precheck_rule_config(
        run_root, config_paths.get("precheck")
    )
    code_defaults, code_default_path = load_precheck_code_defaults()

    text_config = config.get("text_integrity")
    text_config = text_config if isinstance(text_config, dict) else {}
    required_fields = text_config.get("required_fields")
    if isinstance(required_fields, list) and required_fields:
        text_fields = [str(field) for field in required_fields]
        text_source = "weekly_policy"
        text_path = "tools/build_weekly_supplier_acceptance_report.py"
    else:
        text_fields = inferred_text_fields(sources)
        text_source = "producer_output_metadata" if text_fields else "unavailable"
        text_path = str(sources["precheck_check_results"].path or "unavailable")
    for field in text_fields or ["required_text_fields"]:
        value = True if text_fields else "unresolved"
        for operator in ("present", "nonempty"):
            rows.append(
                threshold_rule_row(
                    parent_indicator="text",
                    module="precheck",
                    check_name="text_integrity",
                    metric_or_field=field,
                    rule_type="required_field",
                    threshold_level="required",
                    operator=operator,
                    effective_value=value,
                    unit="boolean",
                    aggregation_scope="per_clip",
                    output_status="fail",
                    config_key=f"text_integrity.required_fields.{field}.{operator}",
                    value_source=text_source,
                    source_path=text_path,
                    notes="Configured supplier text field must exist and be nonempty.",
                )
            )
    rows.append(
        threshold_rule_row(
            parent_indicator="text",
            module="weekly_report",
            check_name="text_status_aggregation",
            metric_or_field="text_check_status",
            rule_type="aggregation_rule",
            threshold_level="required",
            operator="any_fail",
            effective_value="fail",
            unit="status",
            aggregation_scope="per_clip",
            output_status="fail",
            config_key="weekly_policy.text_check_status",
            value_source="weekly_policy",
            source_path="tools/build_weekly_supplier_acceptance_report.py",
            notes="Missing or unknown supplier text schema is not_ready/pending, not a fabricated pass.",
        )
    )

    temporal_specs = {
        "joint_angle_change_deg_max_threshold": ("joint_angle_change_deg_max", "deg"),
        "rotation_delta_max_threshold": ("rotation_delta_max", "ratio"),
        "joint_acceleration_m_s2_max_threshold": ("joint_acceleration_m_s2_max", "m/s^2"),
        "joint_displacement_m_max_threshold": ("joint_displacement_m_max", "m"),
    }
    for key, (metric, unit) in temporal_specs.items():
        value, value_source, source_path = precheck_effective_value(
            section="skeleton_quality_score",
            key=key,
            run_config=config,
            run_value_source=config_source,
            run_source_path=config_path,
            code_defaults=code_defaults,
            code_default_path=code_default_path,
        )
        rows.append(
            threshold_rule_row(
                parent_indicator="abnormal_frame",
                module="precheck",
                check_name="keypoint_temporal",
                metric_or_field=metric,
                rule_type="numeric_threshold",
                threshold_level="review",
                operator=">",
                effective_value=value,
                unit=unit,
                aggregation_scope="max_over_21_points_and_both_hands",
                output_status="review",
                config_key=f"skeleton_quality_score.{key}",
                value_source=value_source,
                source_path=source_path,
                notes="Producer metric is a maximum, not an average or supplier quality score.",
            )
        )
    for key, metric, operator, level, unit in (
        ("hard_exceeded_metric_count", "exceeded_metric_count", ">=", "hard_fail", "metric_count"),
        ("strong_acceleration_ratio", "joint_acceleration_m_s2_ratio", ">=", "hard_fail", "ratio"),
        ("strong_displacement_ratio", "joint_displacement_m_ratio", ">=", "hard_fail", "ratio"),
        ("candidate_gap_close_frames", "candidate_gap", "<=", "configured", "frame"),
        ("candidate_min_seed_run_frames", "candidate_seed_run", ">=", "configured", "frame"),
        ("candidate_pre_context_frames", "candidate_pre_context", "==", "configured", "frame"),
        ("candidate_post_context_frames", "candidate_post_context", "==", "configured", "frame"),
    ):
        value, value_source, source_path = precheck_effective_value(
            section="skeleton_quality_score",
            key=key,
            run_config=config,
            run_value_source=config_source,
            run_source_path=config_path,
            code_defaults=code_defaults,
            code_default_path=code_default_path,
        )
        rows.append(
            threshold_rule_row(
                parent_indicator="abnormal_frame",
                module="precheck",
                check_name="keypoint_temporal",
                metric_or_field=metric,
                rule_type="numeric_threshold" if level == "hard_fail" else "aggregation_rule",
                threshold_level=level,
                operator=operator,
                effective_value=value,
                unit=unit,
                aggregation_scope="per_frame" if level == "hard_fail" else "per_window",
                output_status="fail" if level == "hard_fail" else "review_window",
                config_key=f"skeleton_quality_score.{key}",
                value_source=value_source,
                source_path=source_path,
            )
        )

    allowed_missing, allowed_missing_source, allowed_missing_path = (
        precheck_effective_value(
            section="skeleton_quality_score",
            key="allowed_missing_keypoints_per_hand",
            run_config=config,
            run_value_source=config_source,
            run_source_path=config_path,
            code_defaults=code_defaults,
            code_default_path=code_default_path,
        )
    )
    for metric, operator, value, config_key, rule_type in (
        ("valid_keypoint_count", "==", 21, "acceptance_joint_names", "required_field"),
        ("keypoint_coordinates", "all_finite", True, "finite_coordinate_rule", "finite_value_rule"),
        ("missing_keypoint_count", "<=", allowed_missing, "skeleton_quality_score.allowed_missing_keypoints_per_hand", "numeric_threshold"),
    ):
        rows.append(
            threshold_rule_row(
                parent_indicator="skeleton_static",
                module="precheck",
                check_name="keypoint_presence",
                metric_or_field=metric,
                rule_type=rule_type,
                threshold_level="required",
                operator=operator,
                effective_value=value,
                unit="keypoint_count" if "count" in metric else "boolean",
                aggregation_scope="per_frame_per_hand",
                output_status="fail",
                config_key=config_key,
                value_source=(
                    allowed_missing_source
                    if metric == "missing_keypoint_count"
                    else "code_default"
                ),
                source_path=(
                    allowed_missing_path
                    if metric == "missing_keypoint_count"
                    else "precheck/checks/skeleton_quality_score.py"
                ),
                notes="Existence invalidity is separate from static morphology.",
            )
        )

    morphology_metadata = first_morphology_threshold_metadata(sources)
    morphology_config = config.get("keypoint_morphology")
    morphology_config = morphology_config if isinstance(morphology_config, dict) else {}
    morphology_source = config_source
    morphology_path = config_path
    if morphology_metadata is not None:
        morphology_config = dict(morphology_metadata[0])
        morphology_config.pop("sides", None)
        morphology_source = "producer_output_metadata"
        morphology_path = morphology_metadata[1]
    morphology_specs = {
        "duplicate_joint_distance_m": ("duplicate_joint_distance_m", "configured", "<=", "m"),
        "min_palm_scale_m": ("palm_scale_m", "fail", "<", "m"),
        "max_bone_length_ratio_spread_review": ("bone_length_ratio_spread", "review", ">=", "ratio"),
        "max_bone_length_ratio_spread_fail": ("bone_length_ratio_spread", "fail", ">=", "ratio"),
        "max_normalized_bone_length_review": ("normalized_bone_length_max", "review", ">=", "ratio"),
        "max_normalized_bone_length_fail": ("normalized_bone_length_max", "fail", ">=", "ratio"),
        "max_zero_length_bone_count_review": ("zero_length_bone_count", "review", ">=", "bone_count"),
        "max_zero_length_bone_count_fail": ("zero_length_bone_count", "fail", ">=", "bone_count"),
        "max_duplicate_joint_pair_count_review": ("duplicate_joint_pair_count", "review", ">=", "pair_count"),
        "max_duplicate_joint_pair_count_fail": ("duplicate_joint_pair_count", "fail", ">=", "pair_count"),
        "min_joint_angle_deg_review": ("joint_angle_min_deg", "review", "<=", "deg"),
        "min_joint_angle_deg_fail": ("joint_angle_min_deg", "fail", "<=", "deg"),
        "max_joint_angle_violation_fraction_review": ("joint_angle_violation_fraction", "review", ">=", "ratio"),
        "max_joint_angle_violation_fraction_fail": ("joint_angle_violation_fraction", "fail", ">=", "ratio"),
    }
    for key, (metric, level, operator, unit) in morphology_specs.items():
        if key in morphology_config:
            value = morphology_config[key]
            value_source = morphology_source
            source_path = morphology_path
        else:
            value, value_source, source_path = precheck_effective_value(
                section="keypoint_morphology",
                key=key,
                run_config=config,
                run_value_source=config_source,
                run_source_path=config_path,
                code_defaults=code_defaults,
                code_default_path=code_default_path,
            )
        rows.append(
            threshold_rule_row(
                parent_indicator="skeleton_static",
                module="precheck",
                check_name="keypoint_morphology",
                metric_or_field=metric,
                rule_type="numeric_threshold",
                threshold_level=level,
                operator=operator,
                effective_value=value,
                unit=unit,
                aggregation_scope="per_frame_per_hand",
                output_status="fail" if level == "fail" else "review" if level == "review" else "metric",
                config_key=f"keypoint_morphology.{key}",
                value_source=value_source,
                source_path=source_path,
                notes="Fixed threshold drives verdict; per-clip statistics are calibration evidence only.",
            )
        )
    rows.extend(
        [
            threshold_rule_row(
                parent_indicator="skeleton_static",
                module="weekly_report",
                check_name="skeleton_static_aggregation",
                metric_or_field="missing_or_morphology_status",
                rule_type="aggregation_rule",
                threshold_level="fail",
                operator="any_fail",
                effective_value="missing fail OR morphology fail/review",
                unit="status",
                aggregation_scope="per_clip",
                output_status="fail",
                config_key="weekly_policy.skeleton_static_status",
                value_source="weekly_policy",
                source_path="tools/build_weekly_supplier_acceptance_report.py",
            ),
            threshold_rule_row(
                parent_indicator="skeleton_static",
                module="weekly_report",
                check_name="skeleton_static_intervals",
                metric_or_field="skeleton_static_fail_intervals",
                rule_type="interval_rule",
                threshold_level="official_v1",
                operator="union",
                effective_value="missing_intervals|morphology_fail_intervals",
                unit="frame",
                aggregation_scope="per_clip",
                output_status="problem_frame",
                config_key="weekly_policy.skeleton_static_interval_union",
                value_source="weekly_policy",
                source_path="tools/build_weekly_supplier_acceptance_report.py",
            ),
        ]
    )

    video_metadata = first_video_threshold_metadata(sources)
    if video_metadata is None:
        rows.append(
            threshold_rule_row(
                parent_indicator="video_quality",
                module="video_quality",
                check_name="video_quality_thresholds",
                metric_or_field="producer_threshold_metadata",
                rule_type="numeric_threshold",
                threshold_level="configured",
                operator="==",
                effective_value="unresolved",
                unit="unknown",
                aggregation_scope="per_clip",
                output_status="not_ready",
                config_key="video_quality.thresholds",
                value_source="unavailable",
                source_path=str(sources["video_quality_results"].path or "unavailable"),
                notes="No producer threshold metadata was available; no value was fabricated.",
            )
        )
    else:
        video_thresholds, video_path = video_metadata
        for config_key, value in nested_scalar_items(video_thresholds):
            metric, level, operator, unit, output_status = video_threshold_semantics(config_key)
            rows.append(
                threshold_rule_row(
                    parent_indicator="video_quality",
                    module="video_quality",
                    check_name=config_key.split(".", 1)[0],
                    metric_or_field=metric,
                    rule_type="numeric_threshold" if isinstance(value, (int, float)) and not isinstance(value, bool) else "aggregation_rule",
                    threshold_level=level,
                    operator=operator,
                    effective_value=value,
                    unit=unit,
                    aggregation_scope="per_frame" if config_key.startswith(("exposure.", "sharpness_global.", "freeze.")) else "per_clip",
                    output_status=output_status,
                    config_key=f"video_quality.{config_key}",
                    value_source="producer_output_metadata",
                    source_path=video_path,
                )
            )

    sam3_defaults = [
        ("keypoint_inside_ratio", "review", "<", 1.0, "ratio", "abnormal_inside_ratio_threshold"),
        ("projected_in_image_ratio", "review", "<", 0.8, "ratio", "projected_in_image_ratio_threshold"),
        ("keypoint_inside_ratio", "hard_fail", "<", 0.2, "ratio", "strong_containment_inside_ratio_threshold"),
        ("keypoint_inside_ratio", "acceptable", ">=", 0.6, "ratio", "acceptable_inside_ratio_threshold"),
        ("mask_area_ratio", "review", "<=", 0.0, "ratio", "mask_tiny_area_ratio_threshold"),
        ("strong_fail_frame_count", "hard_fail", ">=", 3, "frame", "containment_fail_min_strong_frames"),
        ("strong_fail_frame_ratio", "hard_fail", ">=", 0.6, "ratio", "containment_fail_strong_frame_ratio"),
    ]
    for metric, level, operator, value, unit, key in sam3_defaults:
        rows.append(
            threshold_rule_row(
                parent_indicator="abnormal_frame",
                module="sam3_containment",
                check_name="keypoint_mask_containment",
                metric_or_field=metric,
                rule_type="numeric_threshold",
                threshold_level=level,
                operator=operator,
                effective_value=value,
                unit=unit,
                aggregation_scope="per_frame" if "frame_count" not in metric and "frame_ratio" not in metric else "per_window",
                output_status="fail" if level == "hard_fail" else "review",
                config_key=f"sam3_keypoint_containment.{key}",
                value_source="code_default",
                source_path="tools/sam3_keypoint_containment.py::parse_args",
            )
        )
    rows.append(
        threshold_rule_row(
            parent_indicator="abnormal_frame",
            module="sam3_containment",
            check_name="hand_object_mask_containment",
            metric_or_field="minimum_valid_projected_keypoint_count",
            rule_type="numeric_threshold",
            threshold_level="required",
            operator=">=",
            effective_value="unresolved",
            unit="keypoint_count",
            aggregation_scope="per_frame",
            output_status="review",
            config_key="sam3_keypoint_containment.minimum_valid_projected_keypoint_count",
            value_source="unavailable",
            source_path="tools/sam3_keypoint_containment.py",
            notes="No separate configured minimum was found in the current CLI/output contract.",
        )
    )

    for check_name, level, metric, operator, value, output_status, notes in (
        ("abnormal_v1", "official_v1", "unreviewed_sam3_interval", "union", "fail_interval", "fail", "Every submitted SAM3 interval not covered by manual review is counted as failed."),
        ("abnormal_v1", "official_v1", "manual_true_positive_interval", "union", "affected_interval", "fail", "Only manually confirmed affected frames are added."),
        ("abnormal_v1", "official_v1", "manual_false_positive_or_acceptable", "subtract_overlap", "reviewed_interval", "pass", "Reviewed false-positive and acceptable overlap is removed."),
        ("abnormal_v2", "comparison_v2", "unreviewed_sam3_interval", "union", "review_interval", "review", "Unreviewed submitted SAM3 intervals remain review evidence."),
        ("abnormal_v2", "comparison_v2", "manual_true_positive_interval", "union", "affected_interval", "fail", "Only manually confirmed affected frames are added."),
        ("abnormal_v2", "comparison_v2", "manual_false_positive_or_acceptable", "subtract_overlap", "reviewed_interval", "pass", "Reviewed false-positive and acceptable overlap is removed."),
    ):
        rows.append(
            threshold_rule_row(
                parent_indicator="abnormal_frame",
                module="weekly_report" if metric.startswith("unreviewed") else "manual_review",
                check_name=check_name,
                metric_or_field=metric,
                rule_type="interval_rule",
                threshold_level=level,
                operator=operator,
                effective_value=value,
                unit="frame",
                aggregation_scope="per_clip",
                output_status=output_status,
                config_key=f"weekly_policy.{check_name}.{metric}",
                value_source="weekly_policy",
                source_path="tools/build_weekly_supplier_acceptance_report.py",
                notes=notes,
            )
        )

    rows.extend(
        [
            threshold_rule_row(
                parent_indicator="final_acceptance",
                module="weekly_report",
                check_name="acceptance_policy",
                metric_or_field="top_level_fail_count",
                rule_type="acceptance_rule",
                threshold_level="official_v1",
                operator=">=",
                effective_value=1,
                unit="indicator_count",
                aggregation_scope="per_clip",
                output_status="fail",
                config_key="weekly_policy.official_v1.fail_indicator_count",
                value_source="weekly_policy",
                source_path="tools/build_weekly_supplier_acceptance_report.py",
                notes="One fail among text, video_quality, skeleton_static, or abnormal_frame fails the clip.",
            ),
            threshold_rule_row(
                parent_indicator="final_acceptance",
                module="weekly_report",
                check_name="acceptance_policy",
                metric_or_field="required_module_readiness",
                rule_type="acceptance_rule",
                threshold_level="official_v1",
                operator="any_fail",
                effective_value="missing|unreadable|unmatched|not_run",
                unit="status",
                aggregation_scope="per_clip",
                output_status="not_ready",
                config_key="weekly_policy.official_v1.not_ready",
                value_source="weekly_policy",
                source_path="tools/build_weekly_supplier_acceptance_report.py",
            ),
            threshold_rule_row(
                parent_indicator="final_acceptance",
                module="weekly_report",
                check_name="acceptance_policy",
                metric_or_field="abnormal_frame_status_v2",
                rule_type="acceptance_rule",
                threshold_level="comparison_v2",
                operator="any_fail",
                effective_value="same four indicators with abnormal_v2",
                unit="status",
                aggregation_scope="per_clip",
                output_status="fail|pass|not_ready",
                config_key="weekly_policy.comparison_v2",
                value_source="weekly_policy",
                source_path="tools/build_weekly_supplier_acceptance_report.py",
            ),
            threshold_rule_row(
                parent_indicator="supplier_evidence",
                module="precheck",
                check_name="quality_score",
                metric_or_field="supplier_quality_signal",
                rule_type="informational_only",
                threshold_level="informational",
                operator="==",
                effective_value="low|provided_ok|not_provided",
                unit="status",
                aggregation_scope="per_clip",
                output_status="note_only",
                config_key="weekly_policy.supplier_quality_signal",
                value_source="weekly_policy",
                source_path="tools/build_weekly_supplier_acceptance_report.py",
                notes="Supplier quality_hand evidence only; it does not determine skeleton_static_status or acceptance.",
            ),
        ]
    )

    key_fields = (
        "module",
        "check_name",
        "metric_or_field",
        "threshold_level",
        "config_key",
    )
    seen: set[tuple[str, ...]] = set()
    deduped: list[dict[str, Any]] = []
    run_leaf_keys = {
        (text(row["module"]), text(row["config_key"]).rsplit(".", 1)[-1])
        for row in run_config_rows
    }
    candidates = [
        *run_config_rows,
        *[
            row
            for row in rows
            if row["value_source"] != "run_config"
            and not (
                row["value_source"] == "code_default"
                and (
                    text(row["module"]),
                    text(row["config_key"]).rsplit(".", 1)[-1],
                )
                in run_leaf_keys
            )
        ],
    ]
    for row in candidates:
        key = tuple(text(row[field]) for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def write_workbook(
    path: Path,
    summary_rows: list[dict[str, Any]],
    xjgt_details: list[dict[str, Any]],
    deepreach_details: list[dict[str, Any]],
    manual_issue_rows: list[dict[str, Any]],
    threshold_rule_rows: list[dict[str, Any]],
    config_audit: dict[str, Any],
) -> None:
    workbook = Workbook()
    workbook.remove(workbook.active)
    add_sheet(workbook, "五供应商总览", SUMMARY_COLUMNS, summary_rows)
    add_sheet(
        workbook,
        "星际归途",
        detail_columns_for_rows(xjgt_details),
        xjgt_details,
    )
    add_sheet(
        workbook,
        "DeepReach",
        detail_columns_for_rows(deepreach_details),
        deepreach_details,
    )
    for index in range(3, 6):
        add_sheet(
            workbook,
            f"供应商{index}",
            detail_columns_for_rows([]),
            [],
        )
    add_manual_threshold_sheet(
        workbook,
        manual_issue_rows,
        threshold_rule_rows,
        config_audit,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def add_manual_threshold_sheet(
    workbook: Workbook,
    manual_issue_rows: list[dict[str, Any]],
    threshold_rule_rows: list[dict[str, Any]],
    config_audit: dict[str, Any],
) -> None:
    sheet = workbook.create_sheet("人工问题与阈值")
    max_columns = max(len(MANUAL_ISSUE_COLUMNS), len(THRESHOLD_RULE_COLUMNS))
    section_fill = PatternFill("solid", fgColor="BDD7EE")
    header_fill = PatternFill("solid", fgColor="D9EAF7")

    manual_title_row = 1
    sheet.cell(manual_title_row, 1, "人工确认问题类型统计")
    sheet.merge_cells(
        start_row=manual_title_row,
        start_column=1,
        end_row=manual_title_row,
        end_column=max_columns,
    )
    sheet.cell(manual_title_row, 1).font = Font(bold=True, size=13)
    sheet.cell(manual_title_row, 1).fill = section_fill
    manual_header_row = manual_title_row + 1
    for column_index, column in enumerate(MANUAL_ISSUE_COLUMNS, 1):
        cell = sheet.cell(manual_header_row, column_index, column)
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    for row_index, row in enumerate(manual_issue_rows, manual_header_row + 1):
        for column_index, column in enumerate(MANUAL_ISSUE_COLUMNS, 1):
            sheet.cell(row_index, column_index, row.get(column, ""))

    manual_last_row = manual_header_row + len(manual_issue_rows)
    note_row = manual_last_row + 1
    note = (
        "人工复核采用高风险定向抽样，本表适合分析供应商问题构成，"
        "不代表供应商全量数据的无偏问题率。"
    )
    sheet.cell(note_row, 1, note)
    sheet.merge_cells(
        start_row=note_row,
        start_column=1,
        end_row=note_row,
        end_column=max_columns,
    )
    sheet.cell(note_row, 1).alignment = Alignment(wrap_text=True)
    sheet.cell(note_row, 1).font = Font(italic=True)

    config_title_row = note_row + 3
    sheet.cell(config_title_row, 1, "运行配置核对")
    sheet.merge_cells(
        start_row=config_title_row,
        start_column=1,
        end_row=config_title_row,
        end_column=max_columns,
    )
    sheet.cell(config_title_row, 1).font = Font(bold=True, size=13)
    sheet.cell(config_title_row, 1).fill = section_fill
    config_summary_rows = [
        (
            "actual config files discovered",
            "\n".join(config_audit.get("actual_config_files_discovered", [])),
        ),
        ("actual_config_leaf_count", config_audit.get("actual_config_leaf_count", 0)),
        (
            "exported_run_config_row_count",
            config_audit.get("exported_run_config_row_count", 0),
        ),
        ("code_default_row_count", config_audit.get("code_default_row_count", 0)),
        (
            "producer_metadata_row_count",
            config_audit.get("producer_metadata_row_count", 0),
        ),
        ("unresolved_row_count", config_audit.get("unresolved_row_count", 0)),
        (
            "missing_config_keys",
            "|".join(config_audit.get("missing_config_keys", [])),
        ),
        (
            "submitted_review_source_path",
            config_audit.get("submitted_review_source_path", ""),
        ),
        (
            "submitted_review_source_status",
            config_audit.get("submitted_review_source_status", "missing"),
        ),
        (
            "config_reconciliation_status",
            config_audit.get("config_reconciliation_status", "incomplete"),
        ),
    ]
    for row_offset, (label, value) in enumerate(config_summary_rows, 1):
        sheet.cell(config_title_row + row_offset, 1, label).font = Font(bold=True)
        value_cell = sheet.cell(config_title_row + row_offset, 2, value)
        value_cell.alignment = Alignment(vertical="top", wrap_text=True)

    threshold_title_row = config_title_row + len(config_summary_rows) + 3
    sheet.cell(
        threshold_title_row,
        1,
        "本次验收完整配置、阈值与规则",
    )
    sheet.merge_cells(
        start_row=threshold_title_row,
        start_column=1,
        end_row=threshold_title_row,
        end_column=max_columns,
    )
    sheet.cell(threshold_title_row, 1).font = Font(bold=True, size=13)
    sheet.cell(threshold_title_row, 1).fill = section_fill
    threshold_header_row = threshold_title_row + 1
    for column_index, column in enumerate(THRESHOLD_RULE_COLUMNS, 1):
        cell = sheet.cell(threshold_header_row, column_index, column)
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    for row_index, row in enumerate(
        threshold_rule_rows, threshold_header_row + 1
    ):
        for column_index, column in enumerate(THRESHOLD_RULE_COLUMNS, 1):
            cell = sheet.cell(row_index, column_index, row.get(column, ""))
            if column in {"notes", "source_path"}:
                cell.alignment = Alignment(vertical="top", wrap_text=True)

    sheet.freeze_panes = f"A{manual_header_row + 1}"
    table_style = TableStyleInfo(
        name="TableStyleMedium2",
        showFirstColumn=False,
        showLastColumn=False,
        showRowStripes=True,
        showColumnStripes=False,
    )
    if manual_issue_rows:
        manual_table = Table(
            displayName="ManualIssueStats",
            ref=(
                f"A{manual_header_row}:"
                f"{get_column_letter(len(MANUAL_ISSUE_COLUMNS))}{manual_last_row}"
            ),
        )
        manual_table.tableStyleInfo = table_style
        sheet.add_table(manual_table)
    if threshold_rule_rows:
        threshold_table = Table(
            displayName="EffectiveThresholdRules",
            ref=(
                f"A{threshold_header_row}:"
                f"{get_column_letter(len(THRESHOLD_RULE_COLUMNS))}"
                f"{threshold_header_row + len(threshold_rule_rows)}"
            ),
        )
        threshold_table.tableStyleInfo = table_style
        sheet.add_table(threshold_table)
    sheet.sheet_view.showGridLines = False
    percentage_columns = {
        "confirmed_issue_clip_ratio_of_reviewed",
        "confirmed_problem_frame_ratio_of_reviewed_frames",
    }
    for column_index, column in enumerate(MANUAL_ISSUE_COLUMNS, 1):
        if column in percentage_columns:
            for row_index in range(manual_header_row + 1, manual_last_row + 1):
                sheet.cell(row_index, column_index).number_format = "0.0%"

    for column_index in range(1, max_columns + 1):
        header_values = [
            sheet.cell(manual_header_row, column_index).value,
            sheet.cell(threshold_header_row, column_index).value,
        ]
        width = max(len(str(value or "")) for value in header_values) + 2
        if column_index in {
            THRESHOLD_RULE_COLUMNS.index("source_path") + 1,
            THRESHOLD_RULE_COLUMNS.index("notes") + 1,
        }:
            width = 42
        sheet.column_dimensions[get_column_letter(column_index)].width = min(
            max(width, 12), 42
        )


def detail_columns_for_rows(rows: list[dict[str, Any]]) -> list[str]:
    text_atomic: list[str] = []
    video_atomic: list[str] = []
    for row in rows:
        for key in row:
            if (
                key.startswith("text_field_")
                and key.endswith("_status")
                and key not in text_atomic
            ):
                text_atomic.append(key)
            elif (
                key.startswith("video_")
                and key.endswith("_status")
                and key != "video_quality_status"
                and key not in video_atomic
            ):
                video_atomic.append(key)
    return [
        *CORE_DETAIL_COLUMNS,
        *text_atomic,
        *TEXT_DETAIL_COLUMNS,
        *video_atomic,
        *VIDEO_DETAIL_COLUMNS,
        *SKELETON_DETAIL_COLUMNS,
        *ABNORMAL_DETAIL_COLUMNS,
        *DIAGNOSTIC_DETAIL_COLUMNS,
    ]


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
        "problem_frame_ratio_v2",
        "pass_clip_ratio_v2",
        "fail_clip_ratio_v2",
        "coverage_ratio",
        "sample_coverage_ratio",
        "manual_review_coverage_ratio",
        "skeleton_missing_fail_frame_ratio",
        "skeleton_morphology_fail_frame_ratio",
        "skeleton_static_fail_frame_ratio",
        "video_quality_fail_frame_ratio",
        "manual_problem_frame_ratio_of_clip",
        "manual_problem_ratio_of_reviewed",
        "abnormal_fail_frame_ratio",
        "abnormal_fail_frame_ratio_v2",
        "auto_fail_precision_on_reviewed",
        "observed_ratio",
        "confirmed_issue_clip_ratio_of_reviewed",
        "confirmed_problem_frame_ratio_of_reviewed_frames",
    }
    for column in percentage_columns.intersection(columns):
        column_letter = get_column_letter(columns.index(column) + 1)
        for cell in sheet[column_letter][1:]:
            if isinstance(cell.value, (int, float)):
                cell.number_format = "0.0%"


def read_first_records(input_name: str, paths: Sequence[Path]) -> SourceRead:
    for path in paths:
        if not path.exists():
            continue
        try:
            records = read_table_records(path)
        except Exception as exc:  # pragma: no cover - diagnostic path
            LOGGER.warning("Could not read %s: %s", path, exc)
            return SourceRead(input_name, path, [], f"unreadable:{type(exc).__name__}")
        return SourceRead(input_name, path, records, "readable")
    return SourceRead(input_name, paths[0] if paths else None, [], "missing")


def read_table_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        if isinstance(data, dict):
            for key in ("records", "rows", "items", "results", "segments"):
                value = data.get(key)
                if isinstance(value, list):
                    return [row for row in value if isinstance(row, dict)]
            if data and all(isinstance(value, dict) for value in data.values()):
                return [
                    {"asset_id": key, **dict(value)}
                    for key, value in data.items()
                ]
            return [data]
        return []
    if suffix == ".parquet":
        import pandas as pd

        frame = pd.read_parquet(path)
        frame = frame.where(frame.notna(), "")
        return frame.to_dict(orient="records")
    return read_csv(path)


def manifest_asset_maps(
    manifest: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[int, str]]:
    rows: dict[str, dict[str, Any]] = {}
    episode_asset: dict[int, str] = {}
    for fallback_idx, row in enumerate(manifest):
        asset_id = normalize_asset_id(row.get("asset_id"))
        if not asset_id:
            continue
        rows[asset_id] = row
        episode_idx = integer_or_none(row.get("episode_idx"))
        episode_asset[episode_idx if episode_idx is not None else fallback_idx] = asset_id
    return rows, episode_asset


def empty_precheck_evidence() -> dict[str, Any]:
    return {
        "checks": set(),
        "missing_intervals": [],
        "morphology_intervals": [],
        "text_status": "not_applicable",
        "text_atomic_statuses": {},
        "supplier_quality_signal": "not_provided",
        "temporal_status": "not_run",
        "has_skeleton_quality": False,
        "has_morphology": False,
    }


def map_window_rows(
    rows: list[dict[str, Any]],
    asset_ids: set[str],
    episode_asset: dict[int, str],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    mapped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"intervals": [], "auto_fail_intervals": [], "rows": []}
    )
    unmatched: set[str] = set()
    for row in rows:
        asset_id = source_asset_id(row, episode_asset)
        if not asset_id or asset_id not in asset_ids:
            if asset_id:
                unmatched.add(asset_id)
            continue
        interval = event_frame_interval(row)
        if interval is not None:
            mapped[asset_id]["intervals"].append(interval)
            verdict = text(
                first_value(
                    row,
                    ("auto_verdict", "source_verdict", "verdict", "status"),
                )
            ).lower()
            if verdict in {"fail", "failed", "hard_fail", "suspect"}:
                mapped[asset_id]["auto_fail_intervals"].append(interval)
        mapped[asset_id]["rows"].append(row)
    return dict(mapped), sorted(unmatched)


def map_submitted_review_intervals(
    rows: list[dict[str, Any]],
    asset_ids: set[str],
    episode_asset: dict[int, str],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    submitted = [row for row in rows if is_submitted_temporal_or_sam3_row(row)]
    return map_window_rows(submitted, asset_ids, episode_asset)


def is_submitted_temporal_or_sam3_row(row: dict[str, Any]) -> bool:
    if text(row.get("source_level")).lower() != "window":
        return False
    module = text(first_value(row, ("module", "source_module"))).lower()
    if module in {"precheck", "sam3_containment", "keypoint_temporal"}:
        return True
    issue = text(
        first_value(
            row,
            ("suggested_issue_type", "issue_type", "failure_mode"),
        )
    ).lower()
    if issue in {
        "temporal_jump",
        "strong_containment_mismatch",
        "containment_fail",
        "mixed_review",
        "side_view_mask_undersegmentation",
        "occlusion_or_mask_undersegmentation",
        "projection_review",
        "projection_ambiguous",
        "severe_keypoint_offset",
        "visual_skeleton_presence_mismatch",
        "skeleton_pose_hallucination",
        "hand_out_of_frame",
    }:
        return True
    evidence = text(row.get("evidence_path")).lower()
    return "candidate_windows" in evidence or "containment" in evidence


def map_video_quality_evidence(
    rows: list[dict[str, Any]],
    asset_ids: set[str],
    episode_asset: dict[int, str],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    mapped: dict[str, dict[str, Any]] = {}
    unmatched: set[str] = set()
    for raw_row in rows:
        row = normalize_video_quality_row(raw_row)
        asset_id = source_asset_id(row, episode_asset)
        if not asset_id or asset_id not in asset_ids:
            if asset_id:
                unmatched.add(asset_id)
            continue
        raw_status = text(
            first_value(
                row,
                (
                    "status",
                    "acceptance_status",
                    "video_quality_status",
                    "decision",
                    "final_status",
                ),
            )
        ).lower()
        passed = boolish_or_none(
            first_value(row, ("passed", "pass", "video_quality_pass"))
        )
        if passed is False or raw_status in {"fail", "failed"}:
            status = "fail"
        elif raw_status in {"review", "not_run", "no_valid_output"}:
            status = raw_status
        else:
            status = "pass"
        existing = mapped.get(asset_id, {})
        atomic_statuses = dict(existing.get("atomic_statuses", {}))
        atomic_statuses.update(video_quality_atomic_statuses(raw_row))
        reason = text(row.get("reason")) or text(existing.get("reason"))
        extracted_fail_intervals = extract_video_fail_intervals(raw_row)
        mapped[asset_id] = {
            "status": status,
            "raw_status": raw_status or "pass",
            "reason": reason or f"video_quality_status={status}",
            "atomic_statuses": atomic_statuses,
            "fail_intervals": (
                extracted_fail_intervals
                or existing.get("fail_intervals", [])
                if status == "fail"
                else existing.get("fail_intervals", [])
            ),
        }
    return mapped, sorted(unmatched)


def video_quality_atomic_statuses(row: dict[str, Any]) -> dict[str, str]:
    video_quality = row.get("video_quality")
    video_quality = video_quality if isinstance(video_quality, dict) else {}
    metrics = video_quality.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    qc_summary = row.get("qc_summary")
    qc_summary = qc_summary if isinstance(qc_summary, dict) else {}
    evaluation = video_quality.get("evaluation")
    evaluation = evaluation if isinstance(evaluation, dict) else {}
    fail_reasons = [
        text(value)
        for value in qc_summary.get("reasons", evaluation.get("reasons", []))
    ]
    warn_reasons = [
        text(value)
        for value in qc_summary.get(
            "warn_reasons", evaluation.get("warn_reasons", [])
        )
    ]

    exposure = metrics.get("exposure_metrics") or metrics.get("exposure")
    exposure = exposure if isinstance(exposure, dict) else {}
    sharpness = metrics.get("sharpness_global")
    sharpness = sharpness if isinstance(sharpness, dict) else {}
    freeze = metrics.get("freeze_metrics")
    freeze = freeze if isinstance(freeze, dict) else {}
    timeline = metrics.get("timeline_metrics")
    timeline = timeline if isinstance(timeline, dict) else {}
    alignment = metrics.get("hdf5_alignment")
    if not isinstance(alignment, dict):
        hdf5_text = row.get("hdf5_text_info")
        hdf5_text = hdf5_text if isinstance(hdf5_text, dict) else {}
        alignment = hdf5_text.get("alignment")
    alignment = alignment if isinstance(alignment, dict) else {}

    definitions = [
        (
            "video_black_screen_status",
            "black_frame_ratio" in exposure
            or "black_frame_count_estimate" in exposure,
            ("black_frame_",),
        ),
        (
            "video_underexposure_status",
            "mean_over_dark_ratio" in exposure,
            ("mean_over_dark_",),
        ),
        (
            "video_overexposure_status",
            "mean_over_exposed_ratio" in exposure,
            ("mean_over_exposed_",),
        ),
        (
            "video_blur_status",
            bool(sharpness) or metrics.get("hand_roi_metrics") is not None,
            ("laplacian_", "tenengrad_", "hand_roi_"),
        ),
        (
            "video_freeze_stutter_status",
            bool(freeze) or bool(timeline),
            (
                "frozen_",
                "freeze_",
                "max_consecutive_frozen_",
                "drop_frame_",
                "frame_interval_",
                "max_frame_gap_",
                "pts_monotonic_",
            ),
        ),
        (
            "video_frame_alignment_status",
            bool(alignment),
            ("hdf5_",),
        ),
    ]
    statuses: dict[str, str] = {}
    for column, available, prefixes in definitions:
        if not available:
            continue
        if any(reason.startswith(prefixes) for reason in fail_reasons):
            statuses[column] = "fail"
        elif any(reason.startswith(prefixes) for reason in warn_reasons):
            statuses[column] = "review"
        else:
            statuses[column] = "pass"
    for key, value in row.items():
        if (
            key.startswith("video_")
            and key.endswith("_status")
            and key != "video_quality_status"
        ):
            statuses[key] = status_value(value)
    return statuses


def extract_video_fail_intervals(row: dict[str, Any]) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    direct = event_frame_interval(row)
    if direct is not None:
        intervals.append(direct)
    video_quality = row.get("video_quality")
    if not isinstance(video_quality, dict):
        return intervals
    metrics = video_quality.get("metrics")
    if not isinstance(metrics, dict):
        return intervals
    freeze = metrics.get("freeze_metrics")
    if not isinstance(freeze, dict):
        return intervals
    for item in freeze.get("frozen_intervals") or []:
        if not isinstance(item, dict):
            continue
        interval = event_frame_interval(item)
        if interval is not None:
            intervals.append(interval)
    return intervals


def map_sam3_evidence(
    rows: list[dict[str, Any]],
    asset_ids: set[str],
    episode_asset: dict[int, str],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    mapped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "status": "pass",
            "auto_fail_intervals": [],
            "review_intervals": [],
        }
    )
    unmatched: set[str] = set()
    for row in rows:
        asset_id = source_asset_id(row, episode_asset)
        if not asset_id or asset_id not in asset_ids:
            if asset_id:
                unmatched.add(asset_id)
            continue
        verdict = text(row.get("window_containment_verdict")).lower()
        status = (
            "fail"
            if verdict == "containment_fail"
            else "review"
            if verdict
            in {
                "mixed_review",
                "projection_review",
                "review",
                "side_view_manual_review",
                "rotation_manual_review",
            }
            else "pass"
        )
        record = mapped[asset_id]
        record["status"] = worst_status(record["status"], status)
        interval = event_frame_interval(row)
        if interval is not None and status != "pass":
            record["review_intervals"].append(interval)
        if interval is not None and verdict == "containment_fail":
            record["auto_fail_intervals"].append(interval)
    return dict(mapped), sorted(unmatched)


def dedupe_manual_labels(
    labels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in labels:
        key = (
            normalize_asset_id(row.get("asset_id")),
            text(row.get("review_id")),
            text(row.get("segment_id")),
            text(row.get("manual_outcome")),
            integer_or_none(row.get("window_start_frame")),
            integer_or_none(row.get("window_end_frame")),
            integer_or_none(row.get("affected_start_frame")),
            integer_or_none(row.get("affected_end_frame")),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def aggregate_manual_review(
    labels: list[dict[str, Any]],
    total_frames_by_asset: dict[str, int],
) -> dict[str, dict[str, Any]]:
    problems: dict[str, list[tuple[int, int]]] = defaultdict(list)
    reviewed: dict[str, list[tuple[int, int]]] = defaultdict(list)
    labeled_assets: set[str] = set()
    for row in labels:
        asset_id = normalize_asset_id(row.get("asset_id"))
        if not asset_id or asset_id not in total_frames_by_asset:
            continue
        labeled_assets.add(asset_id)
        start = integer_or_none(row.get("window_start_frame"))
        end = integer_or_none(row.get("window_end_frame"))
        if start is None or end is None:
            start = integer_or_none(row.get("affected_start_frame"))
            end = integer_or_none(row.get("affected_end_frame"))
        if start is not None and end is not None:
            reviewed[asset_id].append((start, end))
        if not is_confirmed_manual_outcome(row):
            continue
        affected_start = integer_or_none(row.get("affected_start_frame"))
        affected_end = integer_or_none(row.get("affected_end_frame"))
        if affected_start is not None and affected_end is not None:
            problems[asset_id].append((affected_start, affected_end))

    output: dict[str, dict[str, Any]] = {}
    for asset_id in labeled_assets:
        total_frames = total_frames_by_asset[asset_id]
        problem_intervals = clip_intervals(problems[asset_id], total_frames)
        reviewed_intervals = clip_intervals(reviewed[asset_id], total_frames)
        problem_count = interval_frame_count(problem_intervals)
        reviewed_count = interval_frame_count(reviewed_intervals)
        problem_ratio = safe_ratio(problem_count, total_frames)
        output[asset_id] = {
            "manual_review_status": (
                "fail" if problem_ratio >= 0.10 else "pass"
            ),
            "manual_problem_frame_count": problem_count,
            "manual_problem_frame_ratio_of_clip": problem_ratio,
            "manual_reviewed_frame_count": reviewed_count,
            "manual_problem_ratio_of_reviewed": safe_ratio(
                problem_count, reviewed_count
            ),
            "problem_intervals": problem_intervals,
            "reviewed_intervals": reviewed_intervals,
        }
    return output


def is_confirmed_manual_outcome(row: dict[str, Any]) -> bool:
    outcome = text(
        first_value(row, ("manual_outcome", "algorithm_outcome", "label"))
    ).lower()
    return outcome in {"true_positive", "positive"}


def manual_main_issue_by_asset(
    labels: list[dict[str, Any]],
) -> dict[str, str]:
    grouped: dict[str, dict[str, list[tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    counts: Counter[tuple[str, str]] = Counter()
    for row in labels:
        if not is_confirmed_manual_outcome(row):
            continue
        asset_id = normalize_asset_id(row.get("asset_id"))
        issue_type = text(row.get("failure_mode")) or "unknown"
        if not asset_id:
            continue
        counts[(asset_id, issue_type)] += 1
        interval = manual_affected_interval(row)
        if interval is not None:
            grouped[asset_id][issue_type].append(interval)
    output: dict[str, str] = {}
    for asset_id, issues in grouped.items():
        output[asset_id] = max(
            issues,
            key=lambda issue: (
                interval_frame_count(issues[issue]),
                counts[(asset_id, issue)],
                issue,
            ),
        )
    for asset_id, issue_type in counts:
        output.setdefault(asset_id, issue_type)
    return output


def build_manual_issue_rows(
    *,
    supplier_name: str,
    labels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reviewed_assets = {
        normalize_asset_id(row.get("asset_id"))
        for row in labels
        if normalize_asset_id(row.get("asset_id"))
    }
    reviewed_intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in labels:
        asset_id = normalize_asset_id(row.get("asset_id"))
        interval = manual_reviewed_interval(row)
        if asset_id and interval is not None:
            reviewed_intervals[asset_id].append(interval)
    reviewed_frame_count = sum(
        interval_frame_count(intervals)
        for intervals in reviewed_intervals.values()
    )

    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    confirmed_assets: dict[str, set[str]] = defaultdict(set)
    confirmed_intervals: dict[
        str, dict[str, list[tuple[int, int]]]
    ] = defaultdict(lambda: defaultdict(list))
    for row in labels:
        issue_type = text(row.get("failure_mode")) or "unknown"
        grouped_rows[issue_type].append(row)
        if not is_confirmed_manual_outcome(row):
            continue
        asset_id = normalize_asset_id(row.get("asset_id"))
        if not asset_id:
            continue
        confirmed_assets[issue_type].add(asset_id)
        interval = manual_affected_interval(row)
        if interval is not None:
            confirmed_intervals[issue_type][asset_id].append(interval)

    output: list[dict[str, Any]] = []
    for issue_type, rows in grouped_rows.items():
        if not confirmed_assets[issue_type]:
            continue
        outcome_counts = Counter(
            text(
                first_value(
                    row,
                    ("manual_outcome", "algorithm_outcome", "label"),
                )
            ).lower()
            for row in rows
        )
        confirmed_frames = sum(
            interval_frame_count(intervals)
            for intervals in confirmed_intervals[issue_type].values()
        )
        output.append(
            {
                "supplier_name": supplier_name,
                "issue_type": issue_type,
                "confirmed_issue_clip_count": len(
                    confirmed_assets[issue_type]
                ),
                "confirmed_issue_clip_ratio_of_reviewed": safe_ratio(
                    len(confirmed_assets[issue_type]), len(reviewed_assets)
                ),
                "confirmed_problem_frame_count": confirmed_frames,
                "confirmed_problem_frame_ratio_of_reviewed_frames": safe_ratio(
                    confirmed_frames, reviewed_frame_count
                ),
                "manual_label_count": len(rows),
                "true_positive_count": outcome_counts["true_positive"]
                + outcome_counts["positive"],
                "false_positive_count": outcome_counts["false_positive"],
                "acceptable_flagged_count": outcome_counts[
                    "acceptable_flagged"
                ],
                "review_count": outcome_counts["review"],
            }
        )
    return sorted(
        output,
        key=lambda row: (
            -int(row["confirmed_problem_frame_count"]),
            -int(row["confirmed_issue_clip_count"]),
            text(row["issue_type"]),
        ),
    )


def manual_reviewed_interval(
    row: dict[str, Any],
) -> tuple[int, int] | None:
    start = integer_or_none(row.get("window_start_frame"))
    end = integer_or_none(row.get("window_end_frame"))
    if start is None or end is None:
        return manual_affected_interval(row)
    return (start, end)


def manual_affected_interval(
    row: dict[str, Any],
) -> tuple[int, int] | None:
    start = integer_or_none(
        first_value(row, ("affected_start_frame", "start"))
    )
    end = integer_or_none(
        first_value(row, ("affected_end_frame", "end"))
    )
    if start is None or end is None:
        return None
    return (start, end)


def empty_manual_evidence(total_frames: int) -> dict[str, Any]:
    return {
        "manual_review_status": "not_reviewed",
        "manual_problem_frame_count": 0,
        "manual_problem_frame_ratio_of_clip": safe_ratio(0, total_frames),
        "manual_reviewed_frame_count": 0,
        "manual_problem_ratio_of_reviewed": 0.0,
        "problem_intervals": [],
        "reviewed_intervals": [],
    }


def build_xjgt_detail(
    *,
    asset_id: str,
    total_frames: int,
    frame_count_status: str,
    video_path: Path,
    precheck_row: dict[str, Any],
    precheck_source_status: str,
    precheck_unmatched: list[str],
    candidate_row: dict[str, Any],
    video_row: dict[str, Any],
    sam3_row: dict[str, Any],
    submitted_review_row: dict[str, Any],
    submitted_review_source_status: str,
    manual_row: dict[str, Any],
    ledger_row: dict[str, Any],
    require_xjgt_text: bool,
    evidence_paths: list[Path],
) -> dict[str, Any]:
    missing_intervals = clip_intervals(
        precheck_row.get("missing_intervals", []), total_frames
    )
    morphology_intervals = clip_intervals(
        precheck_row.get("morphology_intervals", []), total_frames
    )
    missing_count = interval_frame_count(missing_intervals)
    morphology_count = interval_frame_count(morphology_intervals)
    missing_ratio = safe_ratio(missing_count, total_frames)
    morphology_ratio = safe_ratio(morphology_count, total_frames)
    missing_seen = bool(
        precheck_row.get("has_skeleton_quality")
        or precheck_row.get("presence_seen")
    )
    morphology_seen = bool(
        precheck_row.get("has_morphology")
        or precheck_row.get("morphology_seen")
    )
    missing_status = (
        frame_ratio_status("pass", missing_count, missing_ratio)
        if missing_seen
        else "not_run"
    )
    if not morphology_seen:
        morphology_status = "not_run"
    elif text(precheck_row.get("morphology_status")).lower() == "fail":
        morphology_status = "fail"
    elif text(precheck_row.get("morphology_status")).lower() == "review":
        morphology_status = "review"
    elif morphology_count:
        morphology_status = frame_ratio_status(
            "pass", morphology_count, morphology_ratio
        )
    else:
        morphology_status = "pass"
    static_intervals = [*missing_intervals, *morphology_intervals]
    static_count = interval_frame_count(static_intervals)
    static_ratio = safe_ratio(static_count, total_frames)
    static_status = skeleton_static_status_from_parts(
        missing_status=missing_status,
        morphology_status=morphology_status,
        fail_frame_ratio=static_ratio,
    )
    static_reason = skeleton_static_status_reason(
        missing_status=missing_status,
        morphology_status=morphology_status,
        fail_frame_ratio=static_ratio,
        static_status=static_status,
    )

    mapped_checks = sorted(precheck_row.get("checks", set()))
    missing_expected = [
        check
        for check, present in (
            ("skeleton_quality_score", missing_seen),
            ("keypoint_morphology", morphology_seen),
        )
        if not present
    ]
    if require_xjgt_text and "text_integrity" not in mapped_checks:
        missing_expected.append("text_integrity")
    mapping_status = source_mapping_status(
        precheck_source_status,
        mapped_checks,
        missing_expected,
        bool(precheck_unmatched),
    )
    text_status, text_reason = resolve_xjgt_text_status(
        source_status=precheck_source_status,
        mapped_checks=set(mapped_checks),
        observed_status=text(precheck_row.get("text_status")) or "not_run",
        has_unmatched_source_rows=bool(precheck_unmatched),
        skip_xjgt_text=not require_xjgt_text,
    )
    quality_signal = text(
        precheck_row.get("supplier_quality_signal")
    ) or "not_provided"
    candidate_intervals = clip_intervals(
        candidate_row.get("intervals", []), total_frames
    )
    temporal_seen = bool(
        precheck_row.get("has_skeleton_quality")
        or precheck_row.get("temporal_seen")
    )
    temporal_status = (
        "not_run"
        if not temporal_seen
        else "review"
        if text(precheck_row.get("temporal_status")) == "review"
        or candidate_intervals
        else "pass"
    )
    sam3_status = (
        text(sam3_row.get("status"))
        if sam3_row
        else "not_run"
        if candidate_intervals
        else "not_applicable"
    )
    auto_fail_intervals = clip_intervals(
        [
            *candidate_row.get("auto_fail_intervals", []),
            *sam3_row.get("auto_fail_intervals", []),
        ],
        total_frames,
    )
    submitted_review_intervals = clip_intervals(
        submitted_review_row.get("intervals", []), total_frames
    )
    reviewed_intervals = manual_row["reviewed_intervals"]
    manual_intervals = manual_row["problem_intervals"]
    reviewed_auto = intersect_intervals(
        auto_fail_intervals, reviewed_intervals
    )
    reviewed_auto_true_positive = intersect_intervals(
        reviewed_auto, manual_intervals
    )
    reviewed_auto_false_positive = subtract_intervals(
        reviewed_auto, manual_intervals
    )
    unreviewed_auto = subtract_intervals(
        auto_fail_intervals, reviewed_intervals
    )
    reviewed_submitted = intersect_intervals(
        submitted_review_intervals, reviewed_intervals
    )
    unreviewed_submitted = subtract_intervals(
        submitted_review_intervals, reviewed_intervals
    )
    abnormal_intervals = [
        *unreviewed_auto,
        *manual_intervals,
        *unreviewed_submitted,
    ]
    abnormal_intervals_v2 = [*unreviewed_auto, *manual_intervals]
    unresolved_review_v2 = unreviewed_submitted
    auto_count = interval_frame_count(auto_fail_intervals)
    reviewed_auto_count = interval_frame_count(reviewed_auto)
    reviewed_auto_true_positive_count = interval_frame_count(
        reviewed_auto_true_positive
    )
    reviewed_auto_false_positive_count = interval_frame_count(
        reviewed_auto_false_positive
    )
    unreviewed_auto_count = interval_frame_count(unreviewed_auto)
    submitted_review_count = len(submitted_review_intervals)
    submitted_review_frame_count = interval_frame_count(
        submitted_review_intervals
    )
    reviewed_submitted_count = len(reviewed_submitted)
    unreviewed_submitted_interval_count = len(unreviewed_submitted)
    unreviewed_submitted_count = interval_frame_count(unreviewed_submitted)
    unresolved_review_count_v2 = interval_frame_count(unresolved_review_v2)
    abnormal_count = interval_frame_count(abnormal_intervals)
    abnormal_ratio = safe_ratio(abnormal_count, total_frames)
    abnormal_count_v2 = interval_frame_count(abnormal_intervals_v2)
    abnormal_ratio_v2 = safe_ratio(abnormal_count_v2, total_frames)
    abnormal_inputs_ready = (
        submitted_review_source_status == "readable"
        and temporal_status != "not_run"
        and sam3_status != "not_run"
    )
    if not abnormal_inputs_ready:
        abnormal_frame_status = "not_ready"
    elif abnormal_count > 0:
        abnormal_frame_status = "fail"
    else:
        abnormal_frame_status = "pass"
    if not abnormal_inputs_ready:
        abnormal_frame_status_v2 = "not_ready"
    elif abnormal_count_v2 > 0:
        abnormal_frame_status_v2 = "fail"
    elif unresolved_review_count_v2:
        abnormal_frame_status_v2 = "review"
    else:
        abnormal_frame_status_v2 = "pass"
    abnormal_reason = (
        "policy=v1_conservative; "
        f"abnormal_ratio={abnormal_ratio:.6f}; "
        f"manual_true_positive_frames={manual_row['manual_problem_frame_count']}; "
        f"auto_fail_frames={auto_count}; "
        f"reviewed_auto_fail_frames={reviewed_auto_count}; "
        f"unreviewed_auto_fail_frames={unreviewed_auto_count}; "
        f"unreviewed_submitted_review_frames={unreviewed_submitted_count}"
    )
    abnormal_reason_v2 = (
        "policy=v2_review_first; "
        f"abnormal_ratio={abnormal_ratio_v2:.6f}; "
        f"manual_true_positive_frames={manual_row['manual_problem_frame_count']}; "
        f"unreviewed_auto_fail_frames={unreviewed_auto_count}; "
        f"unresolved_review_frames={unresolved_review_count_v2}"
    )

    raw_video_status = text(video_row.get("status")) or "no_valid_output"
    video_status = (
        "fail"
        if raw_video_status == "fail"
        else "pass"
        if raw_video_status in {"pass", "review", "warn", "warning"}
        else "not_ready"
    )
    video_reason = text(video_row.get("reason")) or (
        f"video_quality_status={raw_video_status}"
    )
    video_intervals = clip_intervals(
        video_row.get("fail_intervals", []), total_frames
    )
    video_count = interval_frame_count(video_intervals)
    video_ratio = safe_ratio(video_count, total_frames)
    unresolved_base = []
    if video_status == "not_ready":
        unresolved_base.append(f"video_quality_status={video_status}")
    if static_status in {"not_ready", "not_run", "blocked", "review"}:
        unresolved_base.append(f"skeleton_static_status={static_status}")
    if text_status == "not_ready":
        unresolved_base.append(f"text_check_status={text_status}")
    if frame_count_status == "unreadable":
        unresolved_base.append("frame_count_status=unreadable")
    unresolved = list(unresolved_base)
    if abnormal_frame_status == "review":
        unresolved.append("abnormal_frame_status=review")
    unresolved_v2 = list(unresolved_base)
    if unresolved_review_count_v2:
        unresolved_v2.append(
            f"abnormal_frame_unresolved_review_frames_v2={unresolved_review_count_v2}"
        )
    elif abnormal_frame_status_v2 == "review":
        unresolved_v2.append("abnormal_frame_status_v2=review")
    required_outputs_ready_base = (
        text_reason.startswith("mapped_text_integrity=")
        or text_status == "not_applicable"
    ) and frame_count_status != "unreadable" and all(
        status != "not_ready"
        for status in (
            video_status,
            static_status,
        )
    )
    required_outputs_ready = required_outputs_ready_base and (
        abnormal_frame_status
        not in {"not_ready", "not_run", "no_valid_output", "blocked"}
    )
    required_outputs_ready_v2 = required_outputs_ready_base and (
        abnormal_frame_status_v2
        not in {"not_ready", "not_run", "no_valid_output", "blocked"}
    )
    acceptance_status, review_status, final_reason = (
        acceptance_and_review_status(
        text_check_status=text_status,
        video_quality_status=video_status,
        skeleton_static_status=static_status,
        abnormal_frame_status=abnormal_frame_status,
        required_outputs_ready=required_outputs_ready,
        unresolved=unresolved,
        )
    )
    acceptance_status_v2, review_status_v2, final_reason_v2 = (
        acceptance_and_review_status(
            text_check_status=text_status,
            video_quality_status=video_status,
            skeleton_static_status=static_status,
            abnormal_frame_status=abnormal_frame_status_v2,
            required_outputs_ready=required_outputs_ready_v2,
            unresolved=unresolved_v2,
        )
    )
    notes = [
        text(ledger_row.get("notes")),
        f"supplier_quality_signal={quality_signal}",
    ]
    if frame_count_status == "unreadable":
        notes.append(f"frame_count_unreadable:{video_path}")
    if mapping_status != "mapped":
        notes.append(
            f"precheck_source={precheck_source_status}; "
            f"unmatched={','.join(precheck_unmatched[:3])}"
        )
    return {
        "asset_id": asset_id,
        "total_frames": total_frames,
        "frame_count_status": frame_count_status,
        "text_check_status": text_status,
        **precheck_row.get("text_atomic_statuses", {}),
        "text_status_reason": text_reason,
        "skeleton_missing_status": missing_status,
        "skeleton_missing_fail_intervals": format_intervals(missing_intervals),
        "skeleton_missing_fail_frame_count": missing_count,
        "skeleton_missing_fail_frame_ratio": missing_ratio,
        "skeleton_morphology_status": morphology_status,
        "skeleton_morphology_fail_intervals": format_intervals(
            morphology_intervals
        ),
        "skeleton_morphology_fail_frame_count": morphology_count,
        "skeleton_morphology_fail_frame_ratio": morphology_ratio,
        "skeleton_static_fail_intervals": format_intervals(
            merge_intervals(static_intervals)
        ),
        "skeleton_static_fail_frame_count": static_count,
        "skeleton_static_fail_frame_ratio": static_ratio,
        "skeleton_static_status": static_status,
        "skeleton_static_status_reason": static_reason,
        "supplier_quality_signal": quality_signal,
        "video_quality_status": video_status,
        **video_row.get("atomic_statuses", {}),
        "video_quality_fail_frame_count": video_count,
        "video_quality_fail_frame_ratio": video_ratio,
        "video_quality_status_reason": video_reason,
        "temporal_status": temporal_status,
        "sam3_containment_status": sam3_status,
        "manual_review_status": manual_row["manual_review_status"],
        "manual_problem_frame_count": manual_row[
            "manual_problem_frame_count"
        ],
        "manual_problem_frame_ratio_of_clip": manual_row[
            "manual_problem_frame_ratio_of_clip"
        ],
        "manual_reviewed_frame_count": manual_row[
            "manual_reviewed_frame_count"
        ],
        "manual_problem_ratio_of_reviewed": manual_row[
            "manual_problem_ratio_of_reviewed"
        ],
        "abnormal_fail_frame_count": abnormal_count,
        "abnormal_fail_frame_ratio": abnormal_ratio,
        "abnormal_frame_status": abnormal_frame_status,
        "auto_fail_frame_count": auto_count,
        "reviewed_auto_fail_frame_count": reviewed_auto_count,
        "reviewed_auto_fail_true_positive_frame_count": (
            reviewed_auto_true_positive_count
        ),
        "reviewed_auto_fail_false_positive_frame_count": (
            reviewed_auto_false_positive_count
        ),
        "unreviewed_auto_fail_frame_count": unreviewed_auto_count,
        "auto_fail_precision_on_reviewed": (
            safe_ratio(
                reviewed_auto_true_positive_count,
                reviewed_auto_count,
            )
            if reviewed_auto_count
            else ""
        ),
        "abnormal_status_reason": abnormal_reason,
        "abnormal_fail_frame_count_v2": abnormal_count_v2,
        "abnormal_fail_frame_ratio_v2": abnormal_ratio_v2,
        "abnormal_frame_status_v2": abnormal_frame_status_v2,
        "abnormal_status_reason_v2": abnormal_reason_v2,
        "submitted_review_interval_count": submitted_review_count,
        "submitted_review_frame_count": submitted_review_frame_count,
        "reviewed_submitted_interval_count": reviewed_submitted_count,
        "unreviewed_submitted_interval_count": (
            unreviewed_submitted_interval_count
        ),
        "unreviewed_submitted_frame_count": unreviewed_submitted_count,
        "abnormal_v1_unreviewed_as_fail_frame_count": (
            unreviewed_submitted_count
        ),
        "abnormal_v2_unreviewed_as_review_frame_count": (
            unreviewed_submitted_count
        ),
        "fail_indicator_count": count_fail_indicators(
            text_check_status=text_status,
            video_quality_status=video_status,
            skeleton_static_status=static_status,
            abnormal_frame_status=abnormal_frame_status,
        ),
        "acceptance_status": acceptance_status,
        "review_status": review_status,
        "acceptance_status_v2": acceptance_status_v2,
        "review_status_v2": review_status_v2,
        "mapped_precheck_checks": "|".join(mapped_checks),
        "missing_expected_checks": "|".join(missing_expected),
        "precheck_mapping_status": mapping_status,
        "final_status_reason": final_reason,
        "final_status_reason_v2": final_reason_v2,
        "main_issue_type": first_issue(ledger_row.get("top_issue_types")),
        "notes": "; ".join(note for note in notes if note),
        "evidence_path": "|".join(str(path) for path in evidence_paths),
        "_problem_intervals": [
            *static_intervals,
            *abnormal_intervals,
            *video_intervals,
        ],
        "_problem_intervals_v2": [
            *static_intervals,
            *abnormal_intervals_v2,
            *video_intervals,
        ],
    }


def source_mapping_status(
    source_status: str,
    mapped_checks: list[str],
    missing_expected: list[str],
    has_unmatched_source_rows: bool,
) -> str:
    if source_status == "missing":
        return "source_missing"
    if source_status.startswith("unreadable"):
        return "source_unreadable"
    if not mapped_checks:
        return "unmatched"
    if missing_expected:
        return "partial"
    return "mapped"


def resolve_xjgt_text_status(
    *,
    source_status: str,
    mapped_checks: set[str],
    observed_status: str,
    has_unmatched_source_rows: bool,
    skip_xjgt_text: bool,
) -> tuple[str, str]:
    if skip_xjgt_text:
        return "not_applicable", "text_integrity_skipped_by_cli"
    if "text_integrity" in mapped_checks:
        status = status_value(observed_status)
        if status not in {"pass", "fail", "review"}:
            status = "not_ready"
        elif status == "review":
            status = "not_ready"
        return status, f"mapped_text_integrity={status}"
    if source_status == "missing":
        return "not_ready", "text_integrity_source_missing"
    if source_status.startswith("unreadable"):
        detail = source_status.split(":", 1)[1] if ":" in source_status else "unknown"
        return "not_ready", f"text_integrity_source_unreadable:{detail}"
    if not mapped_checks:
        return "not_ready", "text_integrity_source_readable_asset_unmatched"
    return "not_ready", "text_integrity_expected_check_missing"


def acceptance_and_review_status(
    *,
    text_check_status: str,
    video_quality_status: str,
    skeleton_static_status: str,
    abnormal_frame_status: str,
    required_outputs_ready: bool,
    unresolved: list[str],
) -> tuple[str, str, str]:
    failed = [
        name
        for name, status in (
            ("text_check_status", text_check_status),
            ("video_quality_status", video_quality_status),
            ("skeleton_static_status", skeleton_static_status),
            ("abnormal_frame_status", abnormal_frame_status),
        )
        if status == "fail"
    ]
    acceptance_status = (
        "not_ready"
        if not required_outputs_ready
        else "fail"
        if failed
        else "pass"
    )
    statuses = (
        text_check_status,
        video_quality_status,
        skeleton_static_status,
        abnormal_frame_status,
    )
    if "blocked" in statuses:
        review_status = "blocked"
    elif not required_outputs_ready:
        review_status = "not_run"
    elif unresolved or "review" in statuses:
        review_status = "review"
    else:
        review_status = "completed"
    reason = (
        f"acceptance_status={acceptance_status}; "
        f"review_status={review_status}; "
        f"fail_indicators={','.join(failed) or 'none'}; "
        f"unresolved={','.join(unresolved) or 'none'}"
    )
    return acceptance_status, review_status, reason


def supplier_summary_from_details(
    *,
    supplier_name: str,
    expected_clip_count: int,
    details: list[dict[str, Any]],
    main_issue_type: str,
    modules_completed: str,
    blocked_modules: str,
    notes: str,
) -> dict[str, Any]:
    counts = Counter(text(row.get("acceptance_status")) for row in details)
    counts_v2 = Counter(
        text(row.get("acceptance_status_v2")) for row in details
    )
    review_counts = Counter(text(row.get("review_status")) for row in details)
    manual_counts = Counter(
        text(row.get("manual_review_status")) for row in details
    )
    abnormal_counts = Counter(
        text(row.get("abnormal_frame_status")) for row in details
    )
    sample_count = len(details)
    total_frames = sum(int(row.get("total_frames") or 0) for row in details)
    problem_frames = sum(
        interval_frame_count(row.get("_problem_intervals", []))
        for row in details
    )
    problem_frames_v2 = sum(
        interval_frame_count(row.get("_problem_intervals_v2", []))
        for row in details
    )
    covered = counts["pass"] + counts["fail"]
    covered_v2 = counts_v2["pass"] + counts_v2["fail"]
    return {
        "supplier_name": supplier_name,
        "sample_clip_count": sample_count,
        "expected_clip_count": expected_clip_count,
        "total_frame_count": total_frames,
        "problem_frame_count": problem_frames,
        "problem_frame_ratio": safe_ratio(problem_frames, total_frames),
        "pass_clip_count": counts["pass"] if covered else "",
        "fail_clip_count": counts["fail"] if covered else "",
        "review_clip_count": review_counts["review"],
        "blocked_clip_count": review_counts["blocked"],
        "not_run_clip_count": review_counts["not_run"],
        "coverage_ratio": safe_ratio(covered, sample_count),
        "sample_coverage_ratio": safe_ratio(sample_count, expected_clip_count),
        "pass_clip_ratio": safe_ratio(counts["pass"], covered) if covered else "",
        "fail_clip_ratio": safe_ratio(counts["fail"], covered) if covered else "",
        "problem_frame_count_v2": problem_frames_v2,
        "problem_frame_ratio_v2": safe_ratio(problem_frames_v2, total_frames),
        "pass_clip_count_v2": counts_v2["pass"] if covered_v2 else "",
        "pass_clip_ratio_v2": (
            safe_ratio(counts_v2["pass"], covered_v2) if covered_v2 else ""
        ),
        "fail_clip_count_v2": counts_v2["fail"] if covered_v2 else "",
        "fail_clip_ratio_v2": (
            safe_ratio(counts_v2["fail"], covered_v2) if covered_v2 else ""
        ),
        "manual_reviewed_clip_count": manual_counts["pass"] + manual_counts["fail"],
        "manual_review_coverage_ratio": safe_ratio(
            manual_counts["pass"] + manual_counts["fail"], sample_count
        ),
        "manual_pass_clip_count": manual_counts["pass"],
        "manual_fail_clip_count": manual_counts["fail"],
        "manual_not_reviewed_clip_count": sample_count
        - manual_counts["pass"]
        - manual_counts["fail"],
        "manual_confirmed_problem_frame_count": sum(
            int(row.get("manual_problem_frame_count") or 0) for row in details
        ),
        "abnormal_pass_clip_count": abnormal_counts["pass"],
        "abnormal_fail_clip_count": abnormal_counts["fail"],
        "abnormal_review_clip_count": abnormal_counts["review"],
        "main_issue_type": main_issue_type,
        "modules_completed": modules_completed,
        "blocked_modules": blocked_modules,
        "notes": notes,
    }


def collect_input_audit(run_root: Path) -> dict[str, Any]:
    manifest_path = run_root / "manifests" / "supplier_manifest_xjgt_100.csv"
    manifest_source = read_first_records("xjgt_manifest", [manifest_path])
    manifest = manifest_source.records
    asset_rows, episode_asset = manifest_asset_maps(manifest)
    asset_ids = set(asset_rows)
    sources = load_xjgt_source_reads(run_root)
    inputs = []
    mapped_fields: dict[str, set[str]] = defaultdict(set)
    for source in sources.values():
        normalized_ids: list[str] = []
        unmapped: list[str] = []
        checks: set[str] = set()
        statuses: set[str] = set()
        for row in source.records:
            asset_id = source_asset_id(row, episode_asset)
            if asset_id:
                normalized_ids.append(asset_id)
                if asset_id not in asset_ids:
                    unmapped.append(asset_id)
            check = text(row.get("check"))
            if check:
                checks.add(check)
                target = {
                    "text_integrity": "text_check_status",
                    "quality_score": "supplier_quality_signal",
                    "skeleton_quality_score": "skeleton_missing_status",
                    "keypoint_morphology": "skeleton_morphology_status",
                    "keypoint_temporal": "temporal_status",
                }.get(check)
                if target and asset_id in asset_ids:
                    mapped_fields[target].add(asset_id)
            for key in (
                "status",
                "decision",
                "final_status",
                "window_containment_verdict",
                "manual_outcome",
            ):
                value = text(row.get(key))
                if value:
                    statuses.add(value)
        overlap = len(set(normalized_ids) & asset_ids)
        read_status = source.status
        if source.status == "readable" and source.records and overlap == 0:
            read_status = "readable_unmatched"
        inputs.append(
            {
                "input_name": source.input_name,
                "path": str(source.path) if source.path else "",
                "read_status": read_status,
                "row_count": len(source.records),
                "columns_or_keys": sorted(
                    {key for row in source.records for key in row}
                ),
                "distinct_checks": sorted(checks),
                "distinct_status_values": sorted(statuses),
                "first_normalized_asset_ids": list(
                    dict.fromkeys(normalized_ids)
                )[:3],
                "asset_overlap_count": overlap,
                "unmapped_asset_examples": list(
                    dict.fromkeys(unmapped)
                )[:3],
            }
        )
    inputs.insert(
        0,
        {
            "input_name": "xjgt_manifest",
            "path": str(manifest_path),
            "read_status": manifest_source.status,
            "row_count": len(manifest),
            "columns_or_keys": sorted(
                {key for row in manifest for key in row}
            ),
            "distinct_checks": [],
            "distinct_status_values": [],
            "first_normalized_asset_ids": list(asset_rows)[:3],
            "asset_overlap_count": len(asset_rows),
            "unmapped_asset_examples": [],
        },
    )
    observed_checks = {
        check for item in inputs for check in item["distinct_checks"]
    }
    return {
        "run_root": str(run_root),
        "manifest_asset_count": len(asset_rows),
        "inputs": inputs,
        "mapped_weekly_fields": {
            field: len(ids) for field, ids in sorted(mapped_fields.items())
        },
        "missing_expected_check_names": sorted(
            {"skeleton_quality_score", "keypoint_morphology"}
            - observed_checks
        ),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))




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


def clip_intervals(
    intervals: list[tuple[int, int]],
    total_frames: int,
) -> list[tuple[int, int]]:
    if total_frames <= 0:
        return []
    clipped: list[tuple[int, int]] = []
    max_frame = total_frames - 1
    for start, end in intervals:
        normalized_start = max(0, min(start, end))
        normalized_end = min(max_frame, max(start, end))
        if normalized_start <= normalized_end:
            clipped.append((normalized_start, normalized_end))
    return merge_intervals(clipped)


def merge_intervals(
    intervals: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    if not intervals:
        return []
    normalized = sorted(
        (min(start, end), max(start, end)) for start, end in intervals
    )
    merged: list[list[int]] = []
    for start, end in normalized:
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def format_intervals(intervals: list[tuple[int, int]]) -> str:
    return "|".join(
        f"{start}-{end}" for start, end in merge_intervals(intervals)
    )


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


def intersect_intervals(
    left: list[tuple[int, int]],
    right: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    intersections: list[tuple[int, int]] = []
    for left_start, left_end in (
        (min(start, end), max(start, end)) for start, end in left
    ):
        for right_start, right_end in (
            (min(start, end), max(start, end)) for start, end in right
        ):
            start = max(left_start, right_start)
            end = min(left_end, right_end)
            if start <= end:
                intersections.append((start, end))
    return intersections


def skeleton_static_status_from_parts(
    *,
    missing_status: str,
    morphology_status: str,
    fail_frame_ratio: float,
) -> str:
    statuses = (missing_status, morphology_status)
    if "fail" in statuses:
        return "fail"
    if "review" in statuses:
        return "fail"
    if statuses == ("pass", "pass"):
        return "pass"
    if any(
        status in {"not_run", "blocked", "no_valid_output", "source_missing", "unmatched"}
        for status in statuses
    ):
        return "not_ready"
    return "not_ready"


def skeleton_static_status_reason(
    *,
    missing_status: str,
    morphology_status: str,
    fail_frame_ratio: float,
    static_status: str,
) -> str:
    return (
        f"missing_status={missing_status}; "
        f"morphology_status={morphology_status}; "
        f"union_fail_frame_ratio={fail_frame_ratio:.6f}; "
        f"skeleton_static_status={static_status}"
    )








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
    if expected_module_missing:
        return "not_ready"
    return "fail" if fail_indicator_count >= 1 else "pass"


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
    acceptance_counts = dict(
        sorted(Counter(row["acceptance_status"] for row in xjgt_details).items())
    )
    review_counts = dict(
        sorted(Counter(row["review_status"] for row in xjgt_details).items())
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
    print(f"XJGT acceptance_status counts={acceptance_counts}")
    print(f"XJGT review_status counts={review_counts}")
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


def float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def float_or_zero(value: Any) -> float:
    parsed = float_or_none(value)
    return parsed if parsed is not None else 0.0


def normalize_asset_id(value: Any) -> str:
    result = text(value).strip()
    if not result:
        return ""
    if "/" in result or "\\" in result:
        result = Path(result).name
    lower = result.lower()
    for suffix in (".hdf5", ".h5", ".mp4", ".mov", ".mkv", ".avi", ".json", ".parquet", ".csv"):
        if lower.endswith(suffix):
            result = result[: -len(suffix)]
            lower = result.lower()
            break
    for suffix in ("_video", "_hdf5"):
        if result.lower().endswith(suffix):
            result = result[: -len(suffix)]
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
