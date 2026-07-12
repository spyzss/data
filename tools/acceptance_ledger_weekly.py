"""Concise weekly-template mode for the reusable acceptance ledger builder."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

try:
    from tools import build_weekly_supplier_acceptance_report as weekly
    from tools.build_acceptance_ledger import normalize_manual_outcome
except ModuleNotFoundError as exc:  # Direct execution: python tools/<script>.py
    if exc.name != "tools":
        raise
    import build_weekly_supplier_acceptance_report as weekly
    from build_acceptance_ledger import normalize_manual_outcome


WEEKLY_SHEET_NAMES = (
    "五供应商总览",
    "星际归途",
    "DeepReach",
    "京东JDT",
    "供应商4",
    "供应商5",
    "人工问题与阈值",
)
DETAIL_COLUMNS = (
    "asset_id",
    "total_frames",
    "text_check_status",
    "skeleton_static_status",
    "video_quality_status",
    "abnormal_frame_status",
    "final_acceptance_status",
    "precheck_window_count",
    "precheck_fail_window_count",
    "precheck_to_sam3_window_count",
    "precheck_to_sam3_ratio",
    "sam3_processed_window_count",
    "sam3_fail_window_count",
    "sam3_to_manual_window_count",
    "sam3_to_manual_ratio",
    "manual_submitted_window_count",
    "manual_reviewed_window_count",
    "manual_fail_window_count",
    "manual_pass_window_count",
    "manual_pending_window_count",
    "manual_completion_ratio",
    "final_fail_window_count",
    "final_review_window_count",
    "main_reason",
    "evidence_path",
)
DETAIL_GROUPS = (
    ("基础与四大项", 1, 7, "D9EAF7"),
    ("Precheck", 8, 11, "E2F0D9"),
    ("SAM3", 12, 15, "FFF2CC"),
    ("人工复核", 16, 21, "FCE4D6"),
    ("最终结果", 22, 25, "E4DFEC"),
)
OVERVIEW_COLUMNS = (
    "supplier_name",
    "sample_clip_count",
    "total_frame_count",
    "fail_frame_count",
    "fail_frame_ratio",
    "review_frame_count",
    "review_frame_ratio",
    "problem_frame_count",
    "problem_frame_ratio",
    "pass_clip_count",
    "pass_clip_ratio",
    "fail_clip_count",
    "fail_clip_ratio",
    "review_clip_count",
    "review_clip_ratio",
)
SUPPLIER_NAMES = {
    "xjgt": "星际归途 / XJGT",
    "deepreach": "DeepReach",
    "jdt": "京东JDT / JDT",
}
STATUS_FILLS = {
    "pass": "C6EFCE",
    "not_applicable": "E7E6E6",
    "review": "FFF2CC",
    "not_run": "FCE4D6",
    "blocked": "FCE4D6",
    "fail": "FFC7CE",
}
LOGGER = logging.getLogger("acceptance_ledger_weekly")
REVIEW_STATUSES = {
    "review",
    "not_run",
    "blocked",
    "not_ready",
    "not_evaluated",
    "no_valid_output",
    "source_missing",
    "adapter_missing",
    "input_missing",
}


def build_weekly_template_workbook(
    *,
    config: dict[str, Any],
    config_path: Path,
    output_path: Path,
    existing_workbook: Path | None,
) -> Path:
    workbook = _load_template(existing_workbook)
    _ensure_weekly_sheets(workbook)
    details_by_supplier: dict[str, list[dict[str, Any]]] = {}
    labels_by_supplier: dict[str, list[dict[str, Any]]] = {}
    review_by_supplier: dict[str, list[dict[str, Any]]] = {}
    for raw_config in config.get("suppliers", []):
        supplier_config = dict(raw_config)
        supplier_id = str(supplier_config.get("supplier_id") or "").lower()
        details, labels, review_rows = _load_supplier_details(
            supplier_config,
            config_dir=config_path.parent,
        )
        details_by_supplier[supplier_id] = details
        labels_by_supplier[supplier_id] = labels
        review_by_supplier[supplier_id] = review_rows

    for supplier_id, sheet_name in (
        ("xjgt", "星际归途"),
        ("deepreach", "DeepReach"),
        ("jdt", "京东JDT"),
        ("supplier4", "供应商4"),
        ("supplier5", "供应商5"),
    ):
        _write_detail_sheet(
            workbook[sheet_name],
            details_by_supplier.get(supplier_id, []),
        )

    summaries = [
        _summary_from_details(
            SUPPLIER_NAMES[supplier_id], details_by_supplier.get(supplier_id, [])
        )
        for supplier_id in ("xjgt", "deepreach", "jdt")
    ]
    summaries.extend(
        [_summary_from_details("供应商4", []), _summary_from_details("供应商5", [])]
    )
    _rewrite_tabular_sheet(workbook["五供应商总览"], OVERVIEW_COLUMNS, summaries)
    _append_manual_issue_rows(
        workbook["人工问题与阈值"],
        labels_by_supplier=labels_by_supplier,
        review_by_supplier=review_by_supplier,
    )
    workbook._sheets = [workbook[name] for name in WEEKLY_SHEET_NAMES]
    workbook.properties.title = str(
        config.get("output_label") or "Weekly Supplier Acceptance"
    )
    _save_atomic(workbook, output_path)
    return output_path


def _load_supplier_details(
    config: dict[str, Any],
    *,
    config_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    supplier_id = str(config.get("supplier_id") or "").lower()
    manifest_source = _read_source(config, "manifest", config_dir, required=True)
    check_source = _read_source(config, "precheck_check_results", config_dir)
    aggregate_source = _read_source(config, "precheck_clip_aggregates", config_dir)
    ledger_source = _read_source(config, "ledger_asset_ledger", config_dir)
    candidate_source = _read_source(config, "candidate_windows", config_dir)
    video_source = _read_source(config, "video_quality", config_dir)
    sam3_source = _read_source(config, "sam3_summary", config_dir)
    review_source = _read_source(config, "review_queue", config_dir)
    manual_source = _read_source(config, "manual_labels", config_dir)
    blocker_source = _read_source(config, "blocker", config_dir)

    manifest = manifest_source.records
    asset_rows, episode_asset = weekly.manifest_asset_maps(manifest)
    asset_ids = set(asset_rows)
    precheck = weekly.load_precheck_evidence(
        manifest,
        {
            "precheck_check_results": check_source,
            "precheck_candidate_windows": candidate_source,
        },
    )
    video_by_asset, _ = weekly.map_video_quality_evidence(
        video_source.records, asset_ids, episode_asset
    )
    candidates = _group_rows(candidate_source.records, asset_ids, episode_asset)
    sam3_rows = _group_rows(sam3_source.records, asset_ids, episode_asset)
    review_rows = [_window_review_row(row) for row in review_source.records]
    reviews = _group_rows(review_rows, asset_ids, episode_asset)
    labels = weekly.dedupe_manual_labels(
        [_manual_fallback_bounds(row) for row in manual_source.records]
    )
    manual = _group_rows(labels, asset_ids, episode_asset)
    ledger_by_asset = {
        weekly.normalize_asset_id(row.get("asset_id")): row
        for row in ledger_source.records
        if weekly.normalize_asset_id(row.get("asset_id"))
    }
    blocked = _truthy(config.get("blocked"))
    blocker_reason = next(
        (str(row.get("reason")) for row in blocker_source.records if row.get("reason")),
        "",
    )
    evidence_paths = [
        source.path
        for source in (
            check_source,
            aggregate_source,
            ledger_source,
            candidate_source,
            video_source,
            sam3_source,
            review_source,
            manual_source,
            blocker_source,
        )
        if source.path is not None
    ]

    details: list[dict[str, Any]] = []
    for asset_id, manifest_row in asset_rows.items():
        frame_info = _manifest_frame_info(manifest_row)
        total_frames = int(frame_info["total_frames"])
        precheck_row = precheck.by_asset.get(
            asset_id, weekly.empty_precheck_evidence()
        )
        text_status, text_reason = _text_status(
            config=config,
            source_status=check_source.status,
            precheck_row=precheck_row,
        )
        skeleton_status, skeleton_reasons = _skeleton_status(
            source_status=check_source.status,
            precheck_row=precheck_row,
        )
        video_status, video_reason = _video_status(
            video_source.status, video_by_asset.get(asset_id)
        )
        window_result = _resolve_windows(
            asset_id=asset_id,
            total_frames=total_frames,
            precheck_row=precheck_row,
            precheck_source_status=check_source.status,
            candidate_rows=candidates.get(asset_id, []),
            sam3_rows=sam3_rows.get(asset_id, []),
            review_rows=reviews.get(asset_id, []),
            manual_rows=manual.get(asset_id, []),
            sam3_blocked=blocked,
            blocker_reason=blocker_reason,
        )
        final_status = _final_status(
            text_status,
            skeleton_status,
            video_status,
            window_result["abnormal_frame_status"],
        )
        reasons = [text_reason, *skeleton_reasons, video_reason]
        if window_result["main_reason"]:
            reasons.append(window_result["main_reason"])
        ledger_reason = str(
            ledger_by_asset.get(asset_id, {}).get("notes") or ""
        ).strip()
        if ledger_reason:
            reasons.append(ledger_reason)
        detail = {
            "asset_id": asset_id,
            "total_frames": total_frames,
            "text_check_status": text_status,
            "skeleton_static_status": skeleton_status,
            "video_quality_status": video_status,
            "abnormal_frame_status": window_result["abnormal_frame_status"],
            "final_acceptance_status": final_status,
            **{
                key: value
                for key, value in window_result.items()
                if key not in {"abnormal_frame_status", "main_reason"}
            },
            "main_reason": "; ".join(_dedupe(reasons)),
            "evidence_path": "|".join(str(path) for path in evidence_paths),
        }
        details.append(detail)
    return details, labels, review_rows


def _resolve_windows(
    *,
    asset_id: str,
    total_frames: int,
    precheck_row: dict[str, Any],
    precheck_source_status: str,
    candidate_rows: list[dict[str, Any]],
    sam3_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    manual_rows: list[dict[str, Any]],
    sam3_blocked: bool,
    blocker_reason: str,
) -> dict[str, Any]:
    routes: list[dict[str, Any]] = []
    for interval in weekly.merge_intervals(
        [
            *precheck_row.get("missing_intervals", []),
            *precheck_row.get("morphology_intervals", []),
        ]
    ):
        routes.append(_new_route(asset_id, interval, "fail"))
    for row in candidate_rows:
        interval = weekly.event_frame_interval(row)
        if interval is None:
            continue
        route = _find_route(routes, row, asset_id)
        if route is None:
            route = _new_route(asset_id, interval, _precheck_window_kind(row))
            routes.append(route)
        elif _precheck_window_kind(row) == "fail":
            route["precheck_kind"] = "fail"
        route["candidate_rows"].append(row)
        _add_review_id(route, row)
    for rows, key in (
        (sam3_rows, "sam3_rows"),
        (review_rows, "review_rows"),
        (manual_rows, "manual_rows"),
    ):
        for row in rows:
            interval = weekly.event_frame_interval(row)
            route = _find_route(routes, row, asset_id)
            if route is None and interval is not None:
                route = _new_route(asset_id, interval, None)
                routes.append(route)
            if route is None:
                continue
            route[key].append(row)
            _add_review_id(route, row)

    metrics = Counter()
    fail_intervals: list[tuple[int, int]] = []
    review_intervals: list[tuple[int, int]] = []
    reasons: list[str] = []
    for route in routes:
        precheck_kind = route["precheck_kind"]
        sam3_kind = _sam3_window_kind(route["sam3_rows"])
        manual_kind = _manual_window_kind(route["manual_rows"])
        submitted = bool(route["review_rows"])
        if precheck_kind:
            metrics["precheck_window_count"] += 1
        if precheck_kind == "fail":
            metrics["precheck_fail_window_count"] += 1
        if precheck_kind == "review" and route["sam3_rows"]:
            metrics["precheck_to_sam3_window_count"] += 1
        if route["sam3_rows"]:
            metrics["sam3_processed_window_count"] += 1
        if sam3_kind == "fail":
            metrics["sam3_fail_window_count"] += 1
        if sam3_kind in {"fail", "review"}:
            metrics["sam3_to_manual_window_count"] += 1
        if submitted:
            metrics["manual_submitted_window_count"] += 1
        if submitted and manual_kind in {"fail", "pass"}:
            metrics["manual_reviewed_window_count"] += 1
            metrics[f"manual_{manual_kind}_window_count"] += 1
        if sam3_kind in {"fail", "review"} and manual_kind in {"fail", "pass"}:
            metrics["completed_sam3_manual_window_count"] += 1

        outcome = _route_outcome(
            precheck_kind=precheck_kind,
            sam3_kind=sam3_kind,
            manual_kind=manual_kind,
            submitted=submitted,
            sam3_blocked=sam3_blocked,
        )
        if outcome == "fail":
            metrics["final_fail_window_count"] += 1
            manual_fail = _manual_fail_intervals(route["manual_rows"], total_frames)
            fail_intervals.extend(manual_fail or [route["interval"]])
        elif outcome == "review":
            metrics["final_review_window_count"] += 1
            review_intervals.append(route["interval"])
        if precheck_kind == "fail":
            reasons.append("precheck_hard_invalid")
        elif sam3_kind == "pass":
            reasons.append("sam3_pass_resolved")
        elif manual_kind:
            reasons.append(f"manual_{manual_kind}")
        elif sam3_blocked and precheck_kind == "review":
            reasons.append(blocker_reason or "sam3_blocked_candidate")
        elif outcome == "review":
            reasons.append("unresolved_review_window")
            diagnostic = {
                "asset_id": asset_id,
                "review_id": sorted(route["review_ids"]),
                "precheck_window_bounds": route["interval"],
                "sam3_verdict": sam3_kind or "not_run",
                "entered_review_queue": bool(route["review_rows"]),
                "matching_manual_label": bool(route["manual_rows"]),
                "resolved_manual_outcome": manual_kind or "incomplete",
            }
            LOGGER.warning("Unresolved weekly review window: %s", diagnostic)
            reasons.append(
                "unresolved_window="
                + json.dumps(diagnostic, ensure_ascii=False, sort_keys=True)
            )

    temporal_seen = bool(precheck_row.get("temporal_seen") or candidate_rows)
    metrics["manual_pending_window_count"] = max(
        0,
        metrics["sam3_to_manual_window_count"]
        - metrics["completed_sam3_manual_window_count"],
    )
    if metrics["final_fail_window_count"]:
        abnormal_status = "fail"
    elif metrics["final_review_window_count"]:
        abnormal_status = "review"
    elif not temporal_seen and precheck_source_status != "readable":
        abnormal_status = "not_run"
        reasons.append("precheck_temporal_not_run")
    elif not temporal_seen:
        abnormal_status = "not_run"
        reasons.append("keypoint_temporal_not_run")
    else:
        abnormal_status = "pass"

    fail_intervals = weekly.clip_intervals(fail_intervals, total_frames)
    review_intervals = weekly.clip_intervals(review_intervals, total_frames)
    review_intervals = weekly.subtract_intervals(review_intervals, fail_intervals)
    unmatched_queue = [
        row
        for row in review_rows
        if not any(_review_rows_match(row, label, asset_id) for label in manual_rows)
    ]
    unmatched_manual = [
        row
        for row in manual_rows
        if not any(_review_rows_match(row, queue, asset_id) for queue in review_rows)
    ]
    return {
        "abnormal_frame_status": abnormal_status,
        "precheck_window_count": metrics["precheck_window_count"],
        "precheck_fail_window_count": metrics["precheck_fail_window_count"],
        "precheck_to_sam3_window_count": metrics[
            "precheck_to_sam3_window_count"
        ],
        "precheck_to_sam3_ratio": _ratio(
            metrics["precheck_to_sam3_window_count"],
            metrics["precheck_window_count"],
        ),
        "sam3_processed_window_count": metrics["sam3_processed_window_count"],
        "sam3_fail_window_count": metrics["sam3_fail_window_count"],
        "sam3_to_manual_window_count": metrics["sam3_to_manual_window_count"],
        "sam3_to_manual_ratio": _ratio(
            metrics["sam3_to_manual_window_count"],
            metrics["sam3_processed_window_count"],
        ),
        "manual_submitted_window_count": metrics["manual_submitted_window_count"],
        "manual_reviewed_window_count": metrics["manual_reviewed_window_count"],
        "manual_fail_window_count": metrics["manual_fail_window_count"],
        "manual_pass_window_count": metrics["manual_pass_window_count"],
        "manual_pending_window_count": metrics["manual_pending_window_count"],
        "manual_completion_ratio": _ratio(
            metrics["manual_reviewed_window_count"],
            metrics["manual_submitted_window_count"],
        ),
        "final_fail_window_count": metrics["final_fail_window_count"],
        "final_review_window_count": metrics["final_review_window_count"],
        "main_reason": "|".join(_dedupe(reasons)),
        "_fail_intervals": fail_intervals,
        "_review_intervals": review_intervals,
        "_unmatched_queue_review_ids": [
            _normalized_review_id(row.get("review_id")) for row in unmatched_queue
        ],
        "_unmatched_manual_review_ids": [
            _normalized_review_id(row.get("review_id")) for row in unmatched_manual
        ],
    }


def _new_route(
    asset_id: str,
    interval: tuple[int, int],
    precheck_kind: str | None,
) -> dict[str, Any]:
    return {
        "asset_id": asset_id,
        "interval": interval,
        "review_ids": set(),
        "precheck_kind": precheck_kind,
        "candidate_rows": [],
        "sam3_rows": [],
        "review_rows": [],
        "manual_rows": [],
    }


def _find_route(
    routes: list[dict[str, Any]], row: dict[str, Any], asset_id: str
) -> dict[str, Any] | None:
    review_id = _normalized_review_id(row.get("review_id"))
    if review_id:
        for route in routes:
            if review_id in route["review_ids"]:
                return route
    interval = weekly.event_frame_interval(row)
    if interval is None:
        return None
    for route in routes:
        if route["asset_id"] == asset_id and route["interval"] == interval:
            return route
    return None


def _add_review_id(route: dict[str, Any], row: dict[str, Any]) -> None:
    review_id = _normalized_review_id(row.get("review_id"))
    if review_id:
        route["review_ids"].add(review_id)


def _normalized_review_id(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _review_rows_match(
    left: dict[str, Any], right: dict[str, Any], asset_id: str
) -> bool:
    left_id = _normalized_review_id(left.get("review_id"))
    right_id = _normalized_review_id(right.get("review_id"))
    if left_id and right_id and left_id == right_id:
        return True
    left_interval = weekly.event_frame_interval(left)
    right_interval = weekly.event_frame_interval(right)
    if left_interval is None or right_interval is None or left_interval != right_interval:
        return False
    left_asset = weekly.normalize_asset_id(left.get("asset_id")) or asset_id
    right_asset = weekly.normalize_asset_id(right.get("asset_id")) or asset_id
    return left_asset == right_asset == asset_id


def _precheck_window_kind(row: dict[str, Any]) -> str:
    verdict = str(
        weekly.first_value(
            row,
            ("auto_verdict", "source_verdict", "verdict", "status"),
        )
        or ""
    ).lower()
    if _truthy(row.get("hard_invalid")) or verdict in {
        "fail",
        "failed",
        "hard_fail",
        "invalid",
    }:
        return "fail"
    return "review"


def _sam3_window_kind(rows: list[dict[str, Any]]) -> str | None:
    kinds = []
    for row in rows:
        verdict = str(
            weekly.first_value(
                row,
                ("window_containment_verdict", "containment_verdict", "verdict", "status"),
            )
            or ""
        ).lower()
        if verdict in {"containment_fail", "fail", "failed", "hard_fail"}:
            kinds.append("fail")
        elif verdict in {
            "pass",
            "passed",
            "containment_pass",
            "acceptable_flagged",
        }:
            kinds.append("pass")
        else:
            kinds.append("review")
    return _worst(kinds)


def _manual_window_kind(rows: list[dict[str, Any]]) -> str | None:
    return _worst([normalize_manual_outcome(row) for row in rows])


def _route_outcome(
    *,
    precheck_kind: str | None,
    sam3_kind: str | None,
    manual_kind: str | None,
    submitted: bool,
    sam3_blocked: bool,
) -> str:
    if precheck_kind == "fail":
        return "fail"
    if sam3_kind == "pass":
        return "pass"
    if sam3_kind in {"fail", "review"}:
        return manual_kind if manual_kind in {"fail", "pass"} else "review"
    if manual_kind in {"fail", "pass"}:
        return manual_kind
    if precheck_kind == "review":
        return "review"
    if submitted or sam3_blocked:
        return "review"
    return "pass"


def _manual_fail_intervals(
    rows: list[dict[str, Any]], total_frames: int
) -> list[tuple[int, int]]:
    intervals = []
    for row in rows:
        if _manual_window_kind([row]) != "fail":
            continue
        start = weekly.integer_or_none(row.get("affected_start_frame"))
        end = weekly.integer_or_none(row.get("affected_end_frame"))
        if start is None or end is None:
            interval = weekly.event_frame_interval(row)
        else:
            interval = (start, end)
        if interval is not None:
            intervals.append(interval)
    return weekly.clip_intervals(intervals, total_frames)


def _text_status(
    *, config: dict[str, Any], source_status: str, precheck_row: dict[str, Any]
) -> tuple[str, str]:
    if not _truthy(config.get("text_required")):
        return "not_applicable", "supplier text schema not applicable"
    checks = set(precheck_row.get("checks", set()))
    if source_status != "readable" or "text_integrity" not in checks:
        return "not_run", "text_integrity applicable but not executed"
    status = str(precheck_row.get("text_status") or "not_run").lower()
    return (status if status in {"pass", "fail"} else "not_run", f"text_integrity={status}")


def _skeleton_status(
    *, source_status: str, precheck_row: dict[str, Any]
) -> tuple[str, list[str]]:
    checks = set(precheck_row.get("checks", set()))
    reasons: list[str] = []
    if source_status != "readable" or "skeleton_quality_score" not in checks:
        return "not_run", ["mandatory skeleton_quality_score not executed"]
    if precheck_row.get("missing_intervals"):
        reasons.append("keypoint presence hard invalid")
    morphology = str(precheck_row.get("morphology_status") or "not_run").lower()
    if "keypoint_morphology" not in checks:
        reasons.append("optional keypoint_morphology not executed")
        morphology = "not_run"
    elif morphology == "fail":
        reasons.append("keypoint morphology fail")
    elif morphology == "review":
        reasons.append("keypoint morphology review")
    if precheck_row.get("missing_intervals") or morphology == "fail":
        return "fail", reasons
    if morphology == "review":
        return "review", reasons
    return "pass", reasons


def _video_status(
    source_status: str, video_row: dict[str, Any] | None
) -> tuple[str, str]:
    if source_status != "readable" or not video_row:
        return "not_run", "video_quality not executed"
    status = str(video_row.get("status") or "not_run").lower()
    if status not in {"pass", "fail", "review", "not_run", "blocked"}:
        status = "review"
    return status, str(video_row.get("reason") or f"video_quality={status}")


def _final_status(*statuses: str) -> str:
    if "fail" in statuses:
        return "fail"
    applicable = [status for status in statuses if status != "not_applicable"]
    if any(status in REVIEW_STATUSES for status in applicable):
        return "review"
    return "pass" if applicable and all(status == "pass" for status in applicable) else "review"


def _summary_from_details(
    supplier_name: str, details: list[dict[str, Any]]
) -> dict[str, Any]:
    counts = Counter(
        str(row.get("final_acceptance_status") or "review") for row in details
    )
    total_frames = sum(int(row.get("total_frames") or 0) for row in details)
    fail_frames = 0
    review_frames = 0
    problem_frames = 0
    for row in details:
        fail_intervals = weekly.merge_intervals(row.get("_fail_intervals", []))
        review_intervals = weekly.subtract_intervals(
            row.get("_review_intervals", []), fail_intervals
        )
        fail_frames += weekly.interval_frame_count(fail_intervals)
        review_frames += weekly.interval_frame_count(review_intervals)
        problem_frames += weekly.interval_frame_count(
            [*fail_intervals, *review_intervals]
        )
    sample_count = len(details)
    return {
        "supplier_name": supplier_name,
        "sample_clip_count": sample_count,
        "total_frame_count": total_frames,
        "fail_frame_count": fail_frames,
        "fail_frame_ratio": _ratio(fail_frames, total_frames),
        "review_frame_count": review_frames,
        "review_frame_ratio": _ratio(review_frames, total_frames),
        "problem_frame_count": problem_frames,
        "problem_frame_ratio": _ratio(problem_frames, total_frames),
        "pass_clip_count": counts["pass"],
        "pass_clip_ratio": _ratio(counts["pass"], sample_count),
        "fail_clip_count": counts["fail"],
        "fail_clip_ratio": _ratio(counts["fail"], sample_count),
        "review_clip_count": counts["review"],
        "review_clip_ratio": _ratio(counts["review"], sample_count),
    }


def _append_manual_issue_rows(
    sheet,
    *,
    labels_by_supplier: dict[str, list[dict[str, Any]]],
    review_by_supplier: dict[str, list[dict[str, Any]]],
) -> None:
    marker = "新增供应商人工问题统计"
    for row_index in range(1, sheet.max_row + 1):
        if sheet.cell(row_index, 1).value == marker:
            sheet.delete_rows(row_index, sheet.max_row - row_index + 1)
            break
    issue_rows = []
    for supplier_id, labels in labels_by_supplier.items():
        issue_rows.extend(
            weekly.build_manual_issue_rows(
                supplier_name=SUPPLIER_NAMES.get(supplier_id, supplier_id),
                labels=labels,
            )
        )
        reviewed_ids = {
            str(row.get("review_id")) for row in labels if row.get("review_id")
        }
        unresolved = [
            row
            for row in review_by_supplier.get(supplier_id, [])
            if str(row.get("review_id") or "") not in reviewed_ids
        ]
        if unresolved:
            issue_rows.append(
                {
                    "supplier_name": SUPPLIER_NAMES.get(supplier_id, supplier_id),
                    "issue_type": "unresolved_review",
                    "manual_label_count": len(unresolved),
                    "review_count": len(unresolved),
                }
            )
    if not issue_rows:
        return
    start = sheet.max_row + 2
    sheet.cell(start, 1, marker).font = Font(bold=True)
    for column_index, column in enumerate(weekly.MANUAL_ISSUE_COLUMNS, 1):
        cell = sheet.cell(start + 1, column_index, column)
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="D9EAF7")
    for row_index, row in enumerate(issue_rows, start + 2):
        for column_index, column in enumerate(weekly.MANUAL_ISSUE_COLUMNS, 1):
            sheet.cell(row_index, column_index, row.get(column, ""))
    sheet.cell(
        start + 2 + len(issue_rows),
        1,
        "定向复核不是无偏的全局召回率估计；需要独立随机抽检评估召回率。",
    )


def _write_detail_sheet(sheet, rows: list[dict[str, Any]]) -> None:
    sheet.delete_rows(1, sheet.max_row)
    thin = Side(style="thin", color="B7B7B7")
    for label, start, end, color in DETAIL_GROUPS:
        sheet.merge_cells(start_row=1, start_column=start, end_row=1, end_column=end)
        cell = sheet.cell(1, start, label)
        cell.fill = PatternFill("solid", fgColor=color)
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center")
        for column in range(start, end + 1):
            group_cell = sheet.cell(1, column)
            group_cell.fill = PatternFill("solid", fgColor=color)
            group_cell.border = Border(bottom=thin)
    for column, name in enumerate(DETAIL_COLUMNS, 1):
        cell = sheet.cell(2, column, name)
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="D9EAF7")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = Border(bottom=thin)
    for row_index, row in enumerate(rows, 3):
        for column, name in enumerate(DETAIL_COLUMNS, 1):
            value = _excel_value(row.get(name, ""))
            cell = sheet.cell(row_index, column, value)
            if name.endswith("_status") or name in {
                "text_check_status",
                "skeleton_static_status",
                "final_acceptance_status",
            }:
                fill = STATUS_FILLS.get(str(value))
                if fill:
                    cell.fill = PatternFill("solid", fgColor=fill)
        for column in (11, 15, 21):
            sheet.cell(row_index, column).number_format = "0.0%"
    sheet.freeze_panes = "A3"
    data_end_row = 2 + len(rows)
    sheet.auto_filter.ref = f"A2:Y{max(data_end_row, 2)}"
    sheet.sheet_view.showGridLines = False
    sheet.row_dimensions[1].height = 24
    sheet.row_dimensions[2].height = 42
    for column, name in enumerate(DETAIL_COLUMNS, 1):
        width = 16
        if name == "asset_id":
            width = 24
        elif name in {"main_reason", "evidence_path"}:
            width = 42
        sheet.column_dimensions[get_column_letter(column)].width = width
    _append_pass_summary(sheet, rows, data_end_row)


def _append_pass_summary(
    sheet, rows: list[dict[str, Any]], data_end_row: int
) -> None:
    summary_header_row = data_end_row + 2
    headers = ("module", "pass_count", "total_clip_count", "pass_ratio")
    modules = (
        "text_check_status",
        "skeleton_static_status",
        "video_quality_status",
        "abnormal_frame_status",
    )
    for column, value in enumerate(headers, 1):
        cell = sheet.cell(summary_header_row, column, value)
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="D9EAF7")
        cell.alignment = Alignment(horizontal="center")
    total = len(rows)
    for offset, module in enumerate(modules, 1):
        row_index = summary_header_row + offset
        pass_count = sum(
            str(row.get(module) or "").strip().lower() == "pass" for row in rows
        )
        sheet.cell(row_index, 1, module)
        sheet.cell(row_index, 2, pass_count)
        sheet.cell(row_index, 3, total)
        ratio_cell = sheet.cell(row_index, 4, _ratio(pass_count, total))
        ratio_cell.number_format = "0.0%"


def _rewrite_tabular_sheet(
    sheet, columns: Iterable[str], rows: list[dict[str, Any]]
) -> None:
    columns = list(columns)
    sheet.delete_rows(1, sheet.max_row)
    sheet.append(columns)
    for row in rows:
        sheet.append([_excel_value(row.get(column, "")) for column in columns])
    for cell in sheet[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="D9EAF7")
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
    percentage_columns = {
        index
        for index, name in enumerate(columns, 1)
        if name.endswith("_ratio")
    }
    for row_index in range(2, sheet.max_row + 1):
        for column in percentage_columns:
            sheet.cell(row_index, column).number_format = "0.0%"
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    sheet.sheet_view.showGridLines = False


def _ensure_weekly_sheets(workbook) -> None:
    if "供应商3" in workbook.sheetnames and "京东JDT" not in workbook.sheetnames:
        workbook["供应商3"].title = "京东JDT"
    for name in list(workbook.sheetnames):
        if name not in WEEKLY_SHEET_NAMES:
            workbook.remove(workbook[name])
    for name in WEEKLY_SHEET_NAMES:
        if name not in workbook.sheetnames:
            workbook.create_sheet(name)


def _load_template(existing_workbook: Path | None):
    if existing_workbook is not None:
        return load_workbook(existing_workbook)
    workbook = Workbook()
    workbook.active.title = "五供应商总览"
    for name in WEEKLY_SHEET_NAMES[1:]:
        workbook.create_sheet(name)
    return workbook


def _group_rows(
    rows: list[dict[str, Any]],
    asset_ids: set[str],
    episode_asset: dict[int, str],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        asset_id = weekly.source_asset_id(row, episode_asset)
        if asset_id in asset_ids:
            grouped[asset_id].append(row)
    return dict(grouped)


def _read_source(
    config: dict[str, Any], key: str, config_dir: Path, required: bool = False
) -> weekly.SourceRead:
    value = config.get(key)
    if not value:
        if required:
            raise FileNotFoundError(f"required input {key} is not configured")
        return weekly.SourceRead(key, None, [], "missing")
    path = _resolve_path(value, config_dir)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"required input {key} not found: {path}")
        return weekly.SourceRead(key, path, [], "missing")
    try:
        return weekly.SourceRead(
            key, path, weekly.read_table_records(path), "readable"
        )
    except (OSError, ValueError) as exc:
        if required:
            raise
        return weekly.SourceRead(key, path, [], f"unreadable:{exc}")


def _resolve_path(value: Any, config_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    cwd = (Path.cwd() / path).resolve()
    local = (config_dir / path).resolve()
    return cwd if cwd.exists() or not local.exists() else local


def _manifest_frame_info(row: dict[str, Any]) -> dict[str, Any]:
    start = weekly.integer_or_none(
        weekly.first_value(row, ("start_frame", "source_start_frame", "clip_start_frame"))
    )
    end = weekly.integer_or_none(
        weekly.first_value(row, ("end_frame", "source_end_frame", "clip_end_frame"))
    )
    if start is not None and end is not None and end >= start:
        return {"total_frames": end - start + 1}
    count = weekly.integer_or_none(
        weekly.first_value(
            row,
            ("clip_frame_count", "total_frame_count", "total_frames", "num_frames", "frame_count"),
        )
    ) or 0
    return {"total_frames": count}


def _manual_fallback_bounds(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    if normalize_manual_outcome(output) == "fail" and (
        weekly.integer_or_none(output.get("affected_start_frame")) is None
        or weekly.integer_or_none(output.get("affected_end_frame")) is None
    ):
        output["affected_start_frame"] = output.get("window_start_frame", "")
        output["affected_end_frame"] = output.get("window_end_frame", "")
    return output


def _window_review_row(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    output.setdefault("source_level", "window")
    return output


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").lower() in {"1", "true", "yes", "on"}


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _worst(statuses: list[str]) -> str | None:
    if not statuses:
        return None
    rank = {"pass": 1, "review": 2, "fail": 3}
    return max(statuses, key=lambda status: rank[status])


def _dedupe(values: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        value = str(value or "").strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _excel_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _save_atomic(workbook, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".xlsx", dir=output_path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        workbook.save(temporary)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
