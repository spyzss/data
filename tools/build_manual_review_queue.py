#!/usr/bin/env python3
"""Build a static manual-review queue for supplier acceptance sampling."""

from __future__ import annotations

import argparse
import html
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_batch_qc_ledger import (  # noqa: E402
    asset_from_row,
    boolish_or_none,
    first_present,
    list_values,
    load_manifest,
    read_records,
    scalar,
)


LOGGER = logging.getLogger("build_manual_review_queue")

MANUAL_OUTCOME_ENUM = [
    "true_positive",
    "false_positive",
    "acceptable_flagged",
    "partial",
    "review",
    "false_negative",
]

FAILURE_MODE_ENUM = [
    "hdf5_text_invalid",
    "quality_hand_low",
    "keypoint_missing_or_nan",
    "temporal_jump",
    "severe_keypoint_offset",
    "strong_containment_mismatch",
    "side_view_mask_undersegmentation",
    "occlusion_or_mask_undersegmentation",
    "hand_out_of_frame",
    "projection_review",
    "skeleton_pose_hallucination",
    "video_blur",
    "video_exposure",
    "video_black_screen",
    "video_stutter",
    "semantic_mismatch",
    "acceptable_minor_misalignment",
    "unknown",
]

SEVERITY_ENUM = ["low", "medium", "high", "critical"]
CONFIDENCE_ENUM = ["low", "medium", "high"]

REVIEW_QUEUE_COLUMNS = [
    "review_id",
    "supplier_id",
    "asset_id",
    "window_start_frame",
    "window_end_frame",
    "representative_frame",
    "module",
    "auto_verdict",
    "suggested_issue_type",
    "severity_suggestion",
    "priority",
    "key_metrics_json",
    "reason",
    "evidence_path",
    "overlay_path",
    "needs_manual_review",
    "sam3_containment_eligible",
]

MANUAL_TEMPLATE_COLUMNS = [
    "review_id",
    "supplier_id",
    "asset_id",
    "window_start_frame",
    "window_end_frame",
    "representative_frame",
    "auto_verdict",
    "suggested_issue_type",
    "manual_outcome",
    "failure_mode",
    "severity",
    "confidence",
    "comment",
    "reviewer",
]

PRIORITY_RANK = {
    "critical": 100,
    "high": 80,
    "medium": 50,
    "low": 20,
    "pass_sample": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build review_queue.csv, manual_labels_template.csv, and a static "
            "review_index.html for supplier acceptance manual review."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--candidate-windows", required=True, type=Path)
    parser.add_argument("--sam3-window-summary", type=Path)
    parser.add_argument("--video-quality", type=Path)
    parser.add_argument("--overlay-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-items-per-supplier", type=int, default=60)
    parser.add_argument("--max-side-view-per-supplier", type=int, default=20)
    parser.add_argument("--max-pass-samples-per-supplier", type=int, default=10)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets, episode_to_asset = load_manifest(args.manifest)

    rows: list[dict[str, Any]] = []
    rows.extend(
        rows_from_candidate_windows(
            read_records(args.candidate_windows),
            args.candidate_windows,
            assets,
            episode_to_asset,
            args.overlay_dir,
        )
    )
    if args.sam3_window_summary:
        rows.extend(
            rows_from_sam3_summary(
                read_records(args.sam3_window_summary),
                args.sam3_window_summary,
                assets,
                episode_to_asset,
                args.overlay_dir,
            )
        )
    if args.video_quality:
        rows.extend(
            rows_from_video_quality(
                read_records(args.video_quality),
                args.video_quality,
                assets,
                episode_to_asset,
                args.overlay_dir,
            )
        )

    selected_rows = select_review_rows(
        rows,
        assets,
        max_items_per_supplier=args.max_items_per_supplier,
        max_side_view_per_supplier=args.max_side_view_per_supplier,
        max_pass_samples_per_supplier=args.max_pass_samples_per_supplier,
        overlay_dir=args.overlay_dir,
    )
    selected_rows = assign_review_ids(selected_rows)

    queue_df = pd.DataFrame(selected_rows, columns=REVIEW_QUEUE_COLUMNS)
    template_df = pd.DataFrame(
        [manual_template_row(row) for row in selected_rows],
        columns=MANUAL_TEMPLATE_COLUMNS,
    )

    review_queue_path = args.output_dir / "review_queue.csv"
    template_path = args.output_dir / "manual_labels_template.csv"
    html_path = args.output_dir / "review_index.html"
    queue_df.to_csv(review_queue_path, index=False)
    template_df.to_csv(template_path, index=False)
    html_path.write_text(build_review_index_html(selected_rows), encoding="utf-8")

    LOGGER.info("Wrote %s", review_queue_path)
    LOGGER.info("Wrote %s", template_path)
    LOGGER.info("Wrote %s", html_path)
    return 0


def rows_from_candidate_windows(
    rows: list[dict[str, Any]],
    evidence_path: Path,
    assets: dict[str, dict[str, Any]],
    episode_to_asset: dict[int, str],
    overlay_dir: Path | None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        asset_id = asset_from_row(row, episode_to_asset)
        if asset_id is None:
            continue
        supplier_id = supplier_for_asset(assets, asset_id)
        start = frame_value(first_present(row, ("start_frame", "window_start_frame", "seed_run_start")))
        end = frame_value(first_present(row, ("end_frame", "window_end_frame", "seed_run_end"), default=start))
        representative = frame_value(row.get("peak_frame"), default=midpoint(start, end))
        review_types = list_values(row.get("review_type"))
        reasons = list_values(row.get("trigger_reason"))
        issue_type = issue_from_candidate(row, review_types, reasons)
        severity = severity_from_priority(row.get("priority"))
        priority = priority_from_issue(issue_type, severity, row.get("priority"))
        output.append(
            queue_row(
                supplier_id=supplier_id,
                asset_id=asset_id,
                start=start,
                end=end,
                representative=representative,
                module="precheck",
                auto_verdict="review",
                suggested_issue_type=issue_type,
                severity_suggestion=severity,
                priority=priority,
                key_metrics=collect_candidate_metrics(row),
                reason=";".join(reasons) or ";".join(review_types) or "candidate window",
                evidence_path=evidence_path,
                overlay_path=find_overlay_path(overlay_dir, asset_id, start, end, representative),
                needs_manual_review=row.get("needs_manual_review"),
                sam3_containment_eligible=row.get("sam3_containment_eligible"),
            )
        )
    return output


def rows_from_sam3_summary(
    rows: list[dict[str, Any]],
    evidence_path: Path,
    assets: dict[str, dict[str, Any]],
    episode_to_asset: dict[int, str],
    overlay_dir: Path | None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        asset_id = asset_from_row(row, episode_to_asset)
        if asset_id is None:
            continue
        supplier_id = supplier_for_asset(assets, asset_id)
        start = frame_value(first_present(row, ("window_start_frame", "start_frame")))
        end = frame_value(first_present(row, ("window_end_frame", "end_frame"), default=start))
        representative = frame_value(row.get("representative_frame"), default=midpoint(start, end))
        verdict = str(first_present(row, ("window_containment_verdict", "verdict"), default="review"))
        issue_type = issue_from_sam3(row, verdict)
        severity = severity_from_sam3(row, verdict)
        output.append(
            queue_row(
                supplier_id=supplier_id,
                asset_id=asset_id,
                start=start,
                end=end,
                representative=representative,
                module="sam3_containment",
                auto_verdict=verdict,
                suggested_issue_type=issue_type,
                severity_suggestion=severity,
                priority=priority_from_issue(issue_type, severity, verdict),
                key_metrics=collect_sam3_metrics(row),
                reason=str(first_present(row, ("reason",), default=verdict)),
                evidence_path=evidence_path,
                overlay_path=find_overlay_path(overlay_dir, asset_id, start, end, representative),
                needs_manual_review=first_present(
                    row,
                    ("source_needs_manual_review", "needs_manual_review"),
                ),
                sam3_containment_eligible=first_present(
                    row,
                    ("source_sam3_containment_eligible", "sam3_containment_eligible"),
                ),
            )
        )
    return output


def rows_from_video_quality(
    rows: list[dict[str, Any]],
    evidence_path: Path,
    assets: dict[str, dict[str, Any]],
    episode_to_asset: dict[int, str],
    overlay_dir: Path | None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        asset_id = asset_from_row(row, episode_to_asset)
        if asset_id is None:
            continue
        status = str(first_present(row, ("status", "final_status"), default="")).lower()
        passed = boolish_or_none(first_present(row, ("passed", "pass", "video_quality_pass")))
        if passed is True and status not in {"fail", "failed", "review"}:
            continue
        supplier_id = supplier_for_asset(assets, asset_id)
        start = frame_value(first_present(row, ("window_start_frame", "start_frame"), default=None))
        end = frame_value(first_present(row, ("window_end_frame", "end_frame"), default=start))
        representative = frame_value(row.get("representative_frame"), default=midpoint(start, end))
        issue_type = issue_from_video_quality(row)
        severity = normalize_enum_value(row.get("severity"), SEVERITY_ENUM, "medium")
        auto_verdict = "fail" if status in {"fail", "failed"} else "review"
        output.append(
            queue_row(
                supplier_id=supplier_id,
                asset_id=asset_id,
                start=start,
                end=end,
                representative=representative,
                module="video_quality",
                auto_verdict=auto_verdict,
                suggested_issue_type=issue_type,
                severity_suggestion=severity,
                priority=priority_from_issue(issue_type, severity, auto_verdict),
                key_metrics=collect_video_metrics(row),
                reason=str(first_present(row, ("reason", "notes"), default="video quality review")),
                evidence_path=evidence_path,
                overlay_path=find_overlay_path(overlay_dir, asset_id, start, end, representative),
                needs_manual_review=True,
                sam3_containment_eligible=None,
            )
        )
    return output


def select_review_rows(
    rows: list[dict[str, Any]],
    assets: dict[str, dict[str, Any]],
    *,
    max_items_per_supplier: int,
    max_side_view_per_supplier: int,
    max_pass_samples_per_supplier: int,
    overlay_dir: Path | None,
) -> list[dict[str, Any]]:
    rows = sorted(
        rows,
        key=lambda row: (
            str(row["supplier_id"]),
            -priority_sort_value(row.get("priority")),
            str(row["asset_id"]),
            frame_sort_value(row.get("window_start_frame")),
        ),
    )
    selected: list[dict[str, Any]] = []
    per_supplier = Counter()
    side_view_per_supplier = Counter()
    assets_with_rows = {str(row["asset_id"]) for row in rows}

    for row in rows:
        supplier_id = str(row["supplier_id"])
        if per_supplier[supplier_id] >= max_items_per_supplier:
            continue
        if row["suggested_issue_type"] == "side_view_mask_undersegmentation":
            if side_view_per_supplier[supplier_id] >= max_side_view_per_supplier:
                continue
            side_view_per_supplier[supplier_id] += 1
        selected.append(row)
        per_supplier[supplier_id] += 1

    pass_counts = Counter()
    for asset_id in sorted(assets):
        if asset_id in assets_with_rows:
            continue
        supplier_id = str(assets[asset_id].get("supplier_id", "unknown"))
        if per_supplier[supplier_id] >= max_items_per_supplier:
            continue
        if pass_counts[supplier_id] >= max_pass_samples_per_supplier:
            continue
        selected.append(
            queue_row(
                supplier_id=supplier_id,
                asset_id=asset_id,
                start=None,
                end=None,
                representative=None,
                module="sampling",
                auto_verdict="pass_sample",
                suggested_issue_type="unknown",
                severity_suggestion="low",
                priority="pass_sample",
                key_metrics={},
                reason="pass sample for manual acceptance calibration",
                evidence_path=None,
                overlay_path=find_overlay_path(overlay_dir, asset_id, None, None, None),
                needs_manual_review=True,
                sam3_containment_eligible=None,
            )
        )
        per_supplier[supplier_id] += 1
        pass_counts[supplier_id] += 1
    return selected


def assign_review_ids(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counters: Counter[str] = Counter()
    output = []
    for row in rows:
        supplier_id = str(row["supplier_id"])
        counters[supplier_id] += 1
        start = empty_if_none(row.get("window_start_frame")) or "na"
        end = empty_if_none(row.get("window_end_frame")) or "na"
        new_row = dict(row)
        new_row["review_id"] = (
            f"{supplier_id}_{row['asset_id']}_{start}_{end}_{counters[supplier_id]:04d}"
        )
        output.append(new_row)
    return output


def manual_template_row(row: dict[str, Any]) -> dict[str, Any]:
    failure_mode = row.get("suggested_issue_type")
    if failure_mode not in FAILURE_MODE_ENUM:
        failure_mode = "unknown"
    return {
        "review_id": row["review_id"],
        "supplier_id": row["supplier_id"],
        "asset_id": row["asset_id"],
        "window_start_frame": row["window_start_frame"],
        "window_end_frame": row["window_end_frame"],
        "representative_frame": row["representative_frame"],
        "auto_verdict": row["auto_verdict"],
        "suggested_issue_type": row["suggested_issue_type"],
        "manual_outcome": "",
        "failure_mode": failure_mode,
        "severity": row["severity_suggestion"],
        "confidence": "",
        "comment": "",
        "reviewer": "",
    }


def queue_row(
    *,
    supplier_id: str,
    asset_id: str,
    start: int | None,
    end: int | None,
    representative: int | None,
    module: str,
    auto_verdict: str,
    suggested_issue_type: str,
    severity_suggestion: str,
    priority: str,
    key_metrics: dict[str, Any],
    reason: str,
    evidence_path: Path | None,
    overlay_path: str | None,
    needs_manual_review: Any,
    sam3_containment_eligible: Any,
) -> dict[str, Any]:
    return {
        "review_id": "",
        "supplier_id": supplier_id,
        "asset_id": asset_id,
        "window_start_frame": empty_if_none(start),
        "window_end_frame": empty_if_none(end),
        "representative_frame": empty_if_none(representative),
        "module": module,
        "auto_verdict": auto_verdict,
        "suggested_issue_type": suggested_issue_type,
        "severity_suggestion": normalize_enum_value(severity_suggestion, SEVERITY_ENUM, "medium"),
        "priority": priority,
        "key_metrics_json": json.dumps(json_safe(key_metrics), ensure_ascii=False, sort_keys=True),
        "reason": reason,
        "evidence_path": "" if evidence_path is None else str(evidence_path),
        "overlay_path": overlay_path or "",
        "needs_manual_review": boolish_or_none(needs_manual_review),
        "sam3_containment_eligible": boolish_or_none(sam3_containment_eligible),
    }


def issue_from_candidate(
    row: dict[str, Any],
    review_types: list[str],
    reasons: list[str],
) -> str:
    joined = " ".join(review_types + reasons).lower()
    if "side_view" in joined:
        return "side_view_mask_undersegmentation"
    if "projection" in joined or "out_of_frame" in joined:
        return "projection_review"
    if "keypoint_presence" in joined or "missing" in joined or "nan" in joined:
        return "keypoint_missing_or_nan"
    if "containment" in joined:
        return "strong_containment_mismatch"
    metrics = parse_jsonish(row.get("trigger_metrics"))
    if isinstance(metrics, dict):
        if metrics.get("joint_displacement_m_max") is not None:
            return "severe_keypoint_offset"
        if metrics.get("joint_acceleration_m_s2_max") is not None:
            return "temporal_jump"
    return "temporal_jump"


def issue_from_sam3(row: dict[str, Any], verdict: str) -> str:
    review_types = " ".join(list_values(row.get("source_review_type"))).lower()
    if verdict == "side_view_manual_review" or "side_view" in review_types:
        return "side_view_mask_undersegmentation"
    if verdict == "projection_review":
        return "projection_review"
    if verdict == "containment_fail":
        return "strong_containment_mismatch"
    if verdict == "mixed_review":
        strong_count = int(float(row.get("strong_fail_frame_count") or 0))
        if strong_count > 0:
            return "strong_containment_mismatch"
        return "occlusion_or_mask_undersegmentation"
    if verdict == "acceptable_flagged":
        return "acceptable_minor_misalignment"
    return "occlusion_or_mask_undersegmentation"


def issue_from_video_quality(row: dict[str, Any]) -> str:
    text = " ".join(
        str(first_present(row, ("issue_type", "reason", "notes", "metric_name"), default="")).lower().split()
    )
    if "blur" in text:
        return "video_blur"
    if "exposure" in text or "overexposure" in text:
        return "video_exposure"
    if "black" in text:
        return "video_black_screen"
    if "stutter" in text or "freeze" in text:
        return "video_stutter"
    return "video_blur"


def severity_from_priority(priority: Any) -> str:
    value = str(priority or "").lower()
    if value in {"critical", "high", "medium", "low"}:
        return value
    return "medium"


def severity_from_sam3(row: dict[str, Any], verdict: str) -> str:
    if verdict == "containment_fail":
        return "high"
    if verdict == "mixed_review":
        return "high" if int(float(row.get("strong_fail_frame_count") or 0)) > 0 else "medium"
    if verdict == "acceptable_flagged":
        return "low"
    return "medium"


def priority_from_issue(issue_type: str, severity: str, source_priority: Any) -> str:
    source = str(source_priority or "").lower()
    if source in PRIORITY_RANK:
        return source
    if severity in {"critical", "high", "medium", "low"}:
        return severity
    if issue_type in {"strong_containment_mismatch", "keypoint_missing_or_nan"}:
        return "high"
    if issue_type == "side_view_mask_undersegmentation":
        return "medium"
    return "medium"


def collect_candidate_metrics(row: dict[str, Any]) -> dict[str, Any]:
    metrics = parse_jsonish(row.get("trigger_metrics"))
    output: dict[str, Any] = {}
    if isinstance(metrics, dict):
        output.update(metrics)
    for key in (
        "peak_frame",
        "seed_run_start",
        "seed_run_end",
        "seed_run_frames",
        "palm_camera_angle_deg_max",
        "rotation_delta_max",
    ):
        if row.get(key) is not None:
            output[key] = row.get(key)
    return output


def collect_sam3_metrics(row: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for key in (
        "strong_fail_frame_count",
        "strong_fail_frame_ratio",
        "inside_ratio_mean",
        "projected_in_image_ratio_mean",
        "projection_review_frame_count",
        "acceptable_frame_count",
    ):
        if row.get(key) is not None:
            output[key] = row.get(key)
    return output


def collect_video_metrics(row: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for key in ("metric_name", "metric_value", "score", "status", "final_status"):
        if row.get(key) is not None:
            output[key] = row.get(key)
    return output


def build_review_index_html(rows: list[dict[str, Any]]) -> str:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["supplier_id"]), str(row["priority"]))].append(row)
    group_keys = sorted(
        grouped,
        key=lambda key: (key[0], -PRIORITY_RANK.get(key[1], 0), key[1]),
    )
    body = [
        "<!doctype html>",
        "<html><head><meta charset=\"utf-8\">",
        "<title>Manual Review Queue</title>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;line-height:1.35;color:#202124}",
        "table{border-collapse:collapse;width:100%;margin:16px 0 32px}",
        "th,td{border:1px solid #dadce0;padding:6px 8px;vertical-align:top;font-size:13px}",
        "th{background:#f1f3f4;text-align:left}",
        ".enum{background:#f8f9fa;border:1px solid #dadce0;padding:12px;margin:12px 0}",
        ".thumb{max-width:220px;max-height:140px;display:block}",
        "code{white-space:pre-wrap}",
        "</style></head><body>",
        "<h1>Manual Review Queue</h1>",
        "<p>Fill <code>manual_labels_template.csv</code>. Do not edit <code>review_queue.csv</code>.</p>",
        "<p>Only edit <code>manual_outcome</code>, <code>failure_mode</code>, "
        "<code>severity</code>, <code>confidence</code>, <code>comment</code>, "
        "and <code>reviewer</code>.</p>",
        "<div class=\"enum\">",
        f"<b>manual_outcome:</b> {html.escape(', '.join(MANUAL_OUTCOME_ENUM))}<br>",
        f"<b>failure_mode:</b> {html.escape(', '.join(FAILURE_MODE_ENUM))}<br>",
        f"<b>severity:</b> {html.escape(', '.join(SEVERITY_ENUM))}<br>",
        f"<b>confidence:</b> {html.escape(', '.join(CONFIDENCE_ENUM))}",
        "</div>",
    ]
    for supplier_id, priority in group_keys:
        rows_in_group = grouped[(supplier_id, priority)]
        body.append(f"<h2>Supplier {html.escape(supplier_id)} / {html.escape(priority)}</h2>")
        body.append("<table>")
        body.append(
            "<tr><th>review_id</th><th>asset/window</th><th>auto</th>"
            "<th>suggested issue</th><th>metrics</th><th>reason</th><th>overlay</th></tr>"
        )
        for row in rows_in_group:
            overlay = ""
            if row.get("overlay_path"):
                overlay_url = html.escape(str(row["overlay_path"]))
                overlay = f"<a href=\"{overlay_url}\"><img class=\"thumb\" src=\"{overlay_url}\"></a>"
            body.append(
                "<tr>"
                f"<td>{html.escape(str(row['review_id']))}</td>"
                f"<td>{html.escape(str(row['asset_id']))}<br>"
                f"{html.escape(str(row['window_start_frame']))}-"
                f"{html.escape(str(row['window_end_frame']))}<br>"
                f"rep {html.escape(str(row['representative_frame']))}</td>"
                f"<td>{html.escape(str(row['module']))}<br>{html.escape(str(row['auto_verdict']))}</td>"
                f"<td>{html.escape(str(row['suggested_issue_type']))}<br>"
                f"{html.escape(str(row['severity_suggestion']))}</td>"
                f"<td><code>{html.escape(str(row['key_metrics_json']))}</code></td>"
                f"<td>{html.escape(str(row['reason']))}</td>"
                f"<td>{overlay}</td>"
                "</tr>"
            )
        body.append("</table>")
    body.append("</body></html>")
    return "\n".join(body) + "\n"


def supplier_for_asset(assets: dict[str, dict[str, Any]], asset_id: str) -> str:
    return str(assets.get(asset_id, {}).get("supplier_id", "unknown"))


def frame_value(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def midpoint(start: int | None, end: int | None) -> int | None:
    if start is None and end is None:
        return None
    if start is None:
        return end
    if end is None:
        return start
    return int(round((start + end) / 2))


def empty_if_none(value: Any) -> Any:
    return "" if value is None else value


def priority_sort_value(value: Any) -> int:
    return PRIORITY_RANK.get(str(value or "").lower(), 0)


def frame_sort_value(value: Any) -> int:
    parsed = frame_value(value)
    return parsed if parsed is not None else -1


def parse_jsonish(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return value


def normalize_enum_value(value: Any, allowed: list[str], default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in allowed else default


def find_overlay_path(
    overlay_dir: Path | None,
    asset_id: str,
    start: int | None,
    end: int | None,
    representative: int | None,
) -> str | None:
    if overlay_dir is None or not overlay_dir.exists():
        return None
    suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    frame_tokens = [
        str(value)
        for value in (representative, start, end)
        if value is not None
    ]
    candidates: list[tuple[int, Path]] = []
    for path in overlay_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        name = path.name
        if asset_id not in name:
            continue
        score = 1
        if any(token in name for token in frame_tokens):
            score += 2
        if start is not None and end is not None and str(start) in name and str(end) in name:
            score += 2
        candidates.append((score, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], str(item[1])))
    return str(candidates[0][1])


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return scalar(value)


if __name__ == "__main__":
    raise SystemExit(main())
