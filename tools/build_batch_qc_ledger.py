#!/usr/bin/env python3
"""Build supplier acceptance v0 ledgers from decoupled QC module outputs."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qc_common.io import write_dataframe


LOGGER = logging.getLogger("build_batch_qc_ledger")

LEDGER_COLUMNS = [
    "supplier_id",
    "asset_id",
    "hdf5_text_status",
    "quality_hand_status",
    "keypoint_missing_status",
    "temporal_status",
    "side_view_status",
    "sam3_containment_status",
    "video_quality_status",
    "manual_review_status",
    "final_verdict",
    "risk_level",
    "top_issue_types",
    "evidence_count",
    "notes",
]

EVENT_COLUMNS = [
    "supplier_id",
    "asset_id",
    "module",
    "issue_type",
    "severity",
    "auto_verdict",
    "manual_outcome",
    "window_start_frame",
    "window_end_frame",
    "metric_name",
    "metric_value",
    "reason",
    "evidence_path",
    "needs_manual_review",
    "sam3_containment_eligible",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an asset-level supplier QC ledger and supplier issue "
            "frequency report from precheck, SAM3 containment, video quality, "
            "and optional manual-review outputs."
        )
    )
    parser.add_argument("--supplier-sample-manifest", required=True, type=Path)
    parser.add_argument("--precheck-clip-aggregates", type=Path)
    parser.add_argument("--precheck-candidate-windows", type=Path)
    parser.add_argument("--sam3-window-summary", type=Path)
    parser.add_argument("--video-quality-results", type=Path)
    parser.add_argument("--manual-review-labels", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    assets, episode_to_asset = load_manifest(args.supplier_sample_manifest)
    events: list[dict[str, Any]] = []
    module_status: dict[str, dict[str, str]] = {
        asset_id: default_module_statuses() for asset_id in assets
    }

    if args.precheck_clip_aggregates:
        add_precheck_aggregates(
            read_records(args.precheck_clip_aggregates),
            args.precheck_clip_aggregates,
            assets,
            episode_to_asset,
            module_status,
            events,
        )
    if args.precheck_candidate_windows:
        add_candidate_windows(
            read_records(args.precheck_candidate_windows),
            args.precheck_candidate_windows,
            assets,
            episode_to_asset,
            module_status,
            events,
        )
    if args.sam3_window_summary:
        add_sam3_window_summaries(
            read_records(args.sam3_window_summary),
            args.sam3_window_summary,
            assets,
            episode_to_asset,
            module_status,
            events,
        )
    if args.video_quality_results:
        add_video_quality(
            read_records(args.video_quality_results),
            args.video_quality_results,
            assets,
            episode_to_asset,
            module_status,
            events,
        )
    if args.manual_review_labels:
        add_manual_review(
            load_manual_review_records(args.manual_review_labels),
            args.manual_review_labels,
            assets,
            episode_to_asset,
            module_status,
            events,
        )

    ledger_rows = build_ledger_rows(assets, module_status, events)
    issue_frequency_rows = build_supplier_issue_frequency(assets, events)

    ledger_df = pd.DataFrame(ledger_rows, columns=LEDGER_COLUMNS)
    frequency_df = pd.DataFrame(issue_frequency_rows)
    events_df = pd.DataFrame(events, columns=EVENT_COLUMNS)

    ledger_csv = args.output_dir / "batch_qc_ledger.csv"
    frequency_csv = args.output_dir / "supplier_issue_frequency.csv"
    events_csv = args.output_dir / "issue_events.csv"
    ledger_df.to_csv(ledger_csv, index=False)
    frequency_df.to_csv(frequency_csv, index=False)
    events_df.to_csv(events_csv, index=False)
    ledger_parquet = write_dataframe(ledger_df, args.output_dir / "batch_qc_ledger.parquet")

    report_path = args.output_dir / "batch_report.md"
    report_path.write_text(
        build_markdown_report(ledger_rows, issue_frequency_rows, events),
        encoding="utf-8",
    )
    LOGGER.info("Wrote %s", ledger_csv)
    LOGGER.info("Wrote %s", ledger_parquet)
    LOGGER.info("Wrote %s", frequency_csv)
    LOGGER.info("Wrote %s", report_path)
    return 0


def default_module_statuses() -> dict[str, str]:
    return {
        "hdf5_text_status": "not_run",
        "quality_hand_status": "not_run",
        "keypoint_missing_status": "not_run",
        "temporal_status": "not_run",
        "side_view_status": "not_run",
        "sam3_containment_status": "not_run",
        "video_quality_status": "not_run",
        "manual_review_status": "not_run",
    }


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [dict(item) for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            return [data]
        raise ValueError(f"JSON input must be a list or object: {path}")
    if suffix == ".csv":
        return records_from_dataframe(pd.read_csv(path))
    if suffix == ".parquet":
        return records_from_dataframe(pd.read_parquet(path))
    raise ValueError(f"unsupported input extension: {path}")


def records_from_dataframe(df: pd.DataFrame) -> list[dict[str, Any]]:
    df = df.where(pd.notna(df), None)
    return [dict(row) for row in df.to_dict(orient="records")]


def load_manifest(path: Path) -> tuple[dict[str, dict[str, Any]], dict[int, str]]:
    rows = read_records(path)
    assets: dict[str, dict[str, Any]] = {}
    episode_to_asset: dict[int, str] = {}
    for index, row in enumerate(rows):
        asset_id = first_present(
            row,
            ("asset_id", "content_id", "clip_id", "id", "episode_id"),
        )
        if asset_id is None:
            asset_id = infer_asset_id_from_paths(row)
        if asset_id is None:
            asset_id = str(index)
        asset_id = normalize_asset_id(asset_id)
        supplier_id = first_present(
            row,
            ("supplier_id", "supplier", "vendor_id", "vendor"),
            default="unknown",
        )
        episode_idx = int(first_present(row, ("episode_idx",), default=index))
        assets.setdefault(
            asset_id,
            {
                "supplier_id": str(supplier_id or "unknown"),
                "asset_id": asset_id,
                "episode_idx": episode_idx,
            },
        )
        episode_to_asset[episode_idx] = asset_id
    return assets, episode_to_asset


def infer_asset_id_from_paths(row: dict[str, Any]) -> str | None:
    for key in ("hdf5_path", "hdf5", "video_path", "video", "path", "source_file"):
        value = row.get(key)
        if value:
            stem = Path(str(value)).stem
            for suffix in ("_hdf5", "-hdf5", "_video", "-video", "_overlay"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            return stem
    return None


def normalize_asset_id(value: Any) -> str:
    text = str(value)
    if text.endswith(".0"):
        return text[:-2]
    return text


def first_present(
    row: dict[str, Any],
    keys: tuple[str, ...],
    default: Any = None,
) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return default


def ensure_asset(
    assets: dict[str, dict[str, Any]],
    module_status: dict[str, dict[str, str]],
    asset_id: str,
    supplier_id: str = "unknown",
) -> None:
    assets.setdefault(
        asset_id,
        {"supplier_id": supplier_id, "asset_id": asset_id, "episode_idx": None},
    )
    module_status.setdefault(asset_id, default_module_statuses())


def asset_from_row(
    row: dict[str, Any],
    episode_to_asset: dict[int, str],
) -> str | None:
    asset_id = first_present(row, ("asset_id", "content_id", "clip_id"))
    if asset_id is not None:
        return normalize_asset_id(asset_id)
    episode_idx = row.get("episode_idx")
    if episode_idx is None:
        return infer_asset_id_from_paths(row)
    try:
        return episode_to_asset.get(int(float(episode_idx)))
    except (TypeError, ValueError):
        return None


def add_event(
    events: list[dict[str, Any]],
    assets: dict[str, dict[str, Any]],
    asset_id: str,
    *,
    module: str,
    issue_type: str,
    severity: str,
    auto_verdict: str,
    evidence_path: Path,
    manual_outcome: str | None = None,
    window_start_frame: Any = None,
    window_end_frame: Any = None,
    metric_name: str | None = None,
    metric_value: Any = None,
    reason: str | None = None,
    needs_manual_review: Any = None,
    sam3_containment_eligible: Any = None,
) -> None:
    supplier_id = str(assets.get(asset_id, {}).get("supplier_id", "unknown"))
    events.append(
        {
            "supplier_id": supplier_id,
            "asset_id": asset_id,
            "module": module,
            "issue_type": issue_type,
            "severity": severity,
            "auto_verdict": auto_verdict,
            "manual_outcome": manual_outcome,
            "window_start_frame": window_start_frame,
            "window_end_frame": window_end_frame,
            "metric_name": metric_name,
            "metric_value": scalar(metric_value),
            "reason": reason,
            "evidence_path": str(evidence_path),
            "needs_manual_review": boolish_or_none(needs_manual_review),
            "sam3_containment_eligible": boolish_or_none(sam3_containment_eligible),
        }
    )


def add_precheck_aggregates(
    rows: list[dict[str, Any]],
    path: Path,
    assets: dict[str, dict[str, Any]],
    episode_to_asset: dict[int, str],
    module_status: dict[str, dict[str, str]],
    events: list[dict[str, Any]],
) -> None:
    for row in rows:
        asset_id = asset_from_row(row, episode_to_asset)
        if asset_id is None:
            continue
        ensure_asset(assets, module_status, asset_id)
        check = str(row.get("check") or "")
        flagged_frames = int(float(row.get("flagged_frames") or 0))
        clip_flag = row.get("clip_flag")
        checked_frames = int(float(row.get("checked_frames") or 0))

        if check == "text_integrity":
            status = "fail" if flagged_frames > 0 or boolish_or_none(clip_flag) is True else "pass"
            module_status[asset_id]["hdf5_text_status"] = status
            if status == "fail":
                add_event(
                    events,
                    assets,
                    asset_id,
                    module="precheck",
                    issue_type="hdf5_text_integrity_failed",
                    severity="high",
                    auto_verdict="fail",
                    metric_name="flagged_frames",
                    metric_value=flagged_frames,
                    reason="supplier text integrity failed",
                    evidence_path=path,
                )
        elif check == "quality_score":
            status = pass_flag_status(clip_flag)
            module_status[asset_id]["quality_hand_status"] = status
            if status == "fail":
                add_event(
                    events,
                    assets,
                    asset_id,
                    module="precheck",
                    issue_type="quality_hand_failed",
                    severity="high",
                    auto_verdict="fail",
                    metric_name="checked_frames",
                    metric_value=checked_frames,
                    reason="quality_hand acceptance score failed",
                    evidence_path=path,
                )
        elif check == "keypoint_missing":
            has_issue = flagged_frames > 0 or boolish_or_none(clip_flag) is True
            raw_invalid = has_raw_keypoint_invalid_evidence(row)
            status = "fail" if raw_invalid else ("review" if has_issue else "pass")
            module_status[asset_id]["keypoint_missing_status"] = status
            if has_issue:
                issue_type = "keypoint_raw_invalid" if raw_invalid else "keypoint_low_quality_window"
                flagged_ratio = flagged_frames / checked_frames if checked_frames else 0.0
                severity = "high" if raw_invalid or flagged_ratio >= 0.2 else "medium"
                auto_verdict = "fail" if raw_invalid else "review"
                add_event(
                    events,
                    assets,
                    asset_id,
                    module="precheck",
                    issue_type=issue_type,
                    severity=severity,
                    auto_verdict=auto_verdict,
                    metric_name="flagged_frames",
                    metric_value=flagged_frames,
                    reason=(
                        raw_keypoint_invalid_reason(row)
                        if raw_invalid
                        else "keypoint low-quality window exceeded aggregate threshold"
                    ),
                    evidence_path=path,
                )
        elif check == "skeleton_quality_score":
            status = pass_flag_status(clip_flag)
            module_status[asset_id]["temporal_status"] = status
            if status == "fail":
                add_event(
                    events,
                    assets,
                    asset_id,
                    module="precheck",
                    issue_type="temporal_skeleton_quality_failed",
                    severity="medium",
                    auto_verdict="review",
                    metric_name="flagged_frames",
                    metric_value=flagged_frames,
                    reason="skeleton temporal quality score failed",
                    evidence_path=path,
                )


def pass_flag_status(value: Any) -> str:
    bool_value = boolish_or_none(value)
    if bool_value is True:
        return "pass"
    if bool_value is False:
        return "fail"
    return "not_run"


def has_raw_keypoint_invalid_evidence(row: dict[str, Any]) -> bool:
    keypoint_presence_invalid = numeric_or_none(row.get("keypoint_presence_invalid"))
    if keypoint_presence_invalid is not None and keypoint_presence_invalid >= 1:
        return True

    for key in ("nan_count", "inf_count", "missing_points", "raw_missing_points"):
        value = numeric_or_none(row.get(key))
        if value is not None and value > 0:
            return True

    valid_points = numeric_or_none(row.get("valid_points"))
    expected_points = first_numeric_present(
        row,
        (
            "expected_points",
            "expected_keypoints",
            "expected_valid_points",
            "expected_count",
            "valid_points_expected",
        ),
    )
    expected_valid_points = expected_points if expected_points is not None else 21.0
    if valid_points is not None and valid_points < expected_valid_points:
        return True

    diagnostic_text = raw_invalid_diagnostic_text(row)
    return has_raw_invalid_text_evidence(diagnostic_text)


def raw_keypoint_invalid_reason(row: dict[str, Any]) -> str:
    keypoint_presence_invalid = numeric_or_none(row.get("keypoint_presence_invalid"))
    if keypoint_presence_invalid is not None and keypoint_presence_invalid >= 1:
        return (
            "raw keypoint invalid evidence: "
            f"keypoint_presence_invalid={scalar(keypoint_presence_invalid)}"
        )
    for key in ("nan_count", "inf_count", "missing_points", "raw_missing_points"):
        value = numeric_or_none(row.get(key))
        if value is not None and value > 0:
            return f"raw keypoint invalid evidence: {key}={scalar(value)}"
    valid_points = numeric_or_none(row.get("valid_points"))
    expected_points = first_numeric_present(
        row,
        (
            "expected_points",
            "expected_keypoints",
            "expected_valid_points",
            "expected_count",
            "valid_points_expected",
        ),
    )
    expected_valid_points = expected_points if expected_points is not None else 21.0
    if valid_points is not None and valid_points < expected_valid_points:
        return (
            "raw keypoint invalid evidence: "
            f"valid_points={scalar(valid_points)} < expected={scalar(expected_valid_points)}"
        )
    return "raw keypoint invalid evidence in reason or metrics"


def raw_invalid_diagnostic_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("reason", "metrics", "metric_name", "issue_type", "notes"):
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            parts.append(value)
        else:
            parts.append(json.dumps(value, ensure_ascii=False, default=str))
    return " ".join(parts).lower()


def has_raw_invalid_text_evidence(text: str) -> bool:
    patterns = (
        r"\braw_invalid\b",
        r"\bnan\b",
        r"\binf\b",
        r"\bkeypoint_presence_invalid\b",
        r"\braw[_ ]missing[_ ]points?\b",
        r"\bvalid[_ ]points?\b",
        r"\bvalid point count\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def first_numeric_present(
    row: dict[str, Any],
    keys: tuple[str, ...],
) -> float | None:
    for key in keys:
        value = numeric_or_none(row.get(key))
        if value is not None:
            return value
    return None


def numeric_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def add_candidate_windows(
    rows: list[dict[str, Any]],
    path: Path,
    assets: dict[str, dict[str, Any]],
    episode_to_asset: dict[int, str],
    module_status: dict[str, dict[str, str]],
    events: list[dict[str, Any]],
) -> None:
    for row in rows:
        asset_id = asset_from_row(row, episode_to_asset)
        if asset_id is None:
            continue
        ensure_asset(assets, module_status, asset_id)
        review_types = list_values(row.get("review_type"))
        reasons = list_values(row.get("trigger_reason"))
        metric_name, metric_value = top_candidate_metric(row.get("trigger_metrics"))
        issue_type = "temporal_geometry_review"
        severity = "high" if row.get("priority") == "high" else "medium"
        if "side_view_manual_review" in review_types:
            issue_type = "side_view_manual_review"
            severity = "medium"
            module_status[asset_id]["side_view_status"] = "review"
        elif "rotation_manual_review" in review_types:
            issue_type = "rotation_manual_review"
            severity = "medium"
        elif "multi_signal_seed" in reasons:
            issue_type = "temporal_multi_signal_review"
        module_status[asset_id]["temporal_status"] = max_status(
            module_status[asset_id]["temporal_status"],
            "review",
        )
        add_event(
            events,
            assets,
            asset_id,
            module="precheck",
            issue_type=issue_type,
            severity=severity,
            auto_verdict="review",
            window_start_frame=row.get("start_frame"),
            window_end_frame=row.get("end_frame"),
            metric_name=metric_name,
            metric_value=metric_value,
            reason=";".join(reasons) or "candidate window",
            evidence_path=path,
            needs_manual_review=row.get("needs_manual_review"),
            sam3_containment_eligible=row.get("sam3_containment_eligible"),
        )


def top_candidate_metric(metrics: Any) -> tuple[str | None, Any]:
    if isinstance(metrics, str):
        try:
            metrics = json.loads(metrics)
        except json.JSONDecodeError:
            return None, None
    if not isinstance(metrics, dict):
        return None, None
    priority = [
        "palm_camera_angle_deg_max",
        "joint_acceleration_m_s2_max",
        "joint_displacement_m_max",
        "rotation_delta_max",
        "joint_angle_change_deg_max",
    ]
    for name in priority:
        if name in metrics and metrics[name] is not None:
            return name, metrics[name]
    if metrics:
        name = next(iter(metrics))
        return name, metrics[name]
    return None, None


def add_sam3_window_summaries(
    rows: list[dict[str, Any]],
    path: Path,
    assets: dict[str, dict[str, Any]],
    episode_to_asset: dict[int, str],
    module_status: dict[str, dict[str, str]],
    events: list[dict[str, Any]],
) -> None:
    for row in rows:
        asset_id = asset_from_row(row, episode_to_asset)
        if asset_id is None:
            continue
        ensure_asset(assets, module_status, asset_id)
        verdict = str(row.get("window_containment_verdict") or "review")
        if verdict in {"containment_fail"}:
            issue_type = "strong_containment_mismatch"
            severity = "high"
            auto_verdict = "fail"
            status = "fail"
        elif verdict == "mixed_review":
            issue_type = "mixed_containment_review"
            severity = "high" if int(float(row.get("strong_fail_frame_count") or 0)) > 0 else "medium"
            auto_verdict = "review"
            status = "review"
        elif verdict in {"side_view_manual_review", "rotation_manual_review"}:
            issue_type = verdict
            severity = "medium"
            auto_verdict = "review"
            status = "review"
        elif verdict == "projection_review":
            issue_type = "projection_review"
            severity = "medium"
            auto_verdict = "review"
            status = "review"
        elif verdict == "acceptable_flagged":
            issue_type = "acceptable_flagged"
            severity = "low"
            auto_verdict = "risk"
            status = "risk"
        else:
            module_status[asset_id]["sam3_containment_status"] = max_status(
                module_status[asset_id]["sam3_containment_status"],
                "pass",
            )
            continue
        module_status[asset_id]["sam3_containment_status"] = max_status(
            module_status[asset_id]["sam3_containment_status"],
            status,
        )
        add_event(
            events,
            assets,
            asset_id,
            module="sam3_containment",
            issue_type=issue_type,
            severity=severity,
            auto_verdict=auto_verdict,
            window_start_frame=row.get("window_start_frame"),
            window_end_frame=row.get("window_end_frame"),
            metric_name="inside_ratio_mean",
            metric_value=row.get("inside_ratio_mean"),
            reason=row.get("reason") or verdict,
            evidence_path=path,
            needs_manual_review=row.get("source_needs_manual_review"),
            sam3_containment_eligible=row.get("source_sam3_containment_eligible"),
        )


def add_video_quality(
    rows: list[dict[str, Any]],
    path: Path,
    assets: dict[str, dict[str, Any]],
    episode_to_asset: dict[int, str],
    module_status: dict[str, dict[str, str]],
    events: list[dict[str, Any]],
) -> None:
    for row in rows:
        asset_id = asset_from_row(row, episode_to_asset)
        if asset_id is None:
            continue
        ensure_asset(assets, module_status, asset_id)
        status_value = str(first_present(row, ("status", "final_status"), default="")).lower()
        passed = first_present(row, ("passed", "pass", "video_quality_pass"))
        failed = boolish_or_none(passed) is False or status_value in {"fail", "failed"}
        status = "fail" if failed else "pass"
        module_status[asset_id]["video_quality_status"] = status
        if failed:
            severity = str(row.get("severity") or "high").lower()
            add_event(
                events,
                assets,
                asset_id,
                module="video_quality",
                issue_type="video_quality_failed",
                severity="high" if severity == "high" else "medium",
                auto_verdict="fail" if severity == "high" else "risk",
                metric_name=first_present(row, ("metric_name",), default="video_quality"),
                metric_value=first_present(row, ("metric_value", "score")),
                reason=first_present(row, ("reason", "notes"), default="video quality failed"),
                evidence_path=path,
            )


def add_manual_review(
    rows: list[dict[str, Any]],
    path: Path,
    assets: dict[str, dict[str, Any]],
    episode_to_asset: dict[int, str],
    module_status: dict[str, dict[str, str]],
    events: list[dict[str, Any]],
) -> None:
    for row in rows:
        asset_id = asset_from_row(row, episode_to_asset)
        if asset_id is None:
            continue
        ensure_asset(assets, module_status, asset_id)
        label = str(first_present(row, ("label", "manual_outcome"), default="review"))
        if label == "positive":
            status, severity, auto_verdict = "fail", "high", "fail"
        elif label == "acceptable_flagged":
            status, severity, auto_verdict = "risk", "low", "risk"
        else:
            status, severity, auto_verdict = "review", "medium", "review"
        module_status[asset_id]["manual_review_status"] = max_status(
            module_status[asset_id]["manual_review_status"],
            status,
        )
        add_event(
            events,
            assets,
            asset_id,
            module="manual_review",
            issue_type=str(row.get("failure_mode") or label),
            severity=severity,
            auto_verdict=auto_verdict,
            manual_outcome=label,
            window_start_frame=first_present(row, ("start", "start_frame")),
            window_end_frame=first_present(row, ("end", "end_frame")),
            reason=row.get("note") or row.get("reason"),
            evidence_path=path,
            needs_manual_review=label == "review",
        )


def load_manual_review_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() != ".json":
        return read_records(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("segments"), list):
        return [dict(item) for item in data["segments"] if isinstance(item, dict)]
    if isinstance(data, list):
        return [dict(item) for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def list_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                loaded = json.loads(text)
                return list_values(loaded)
            except json.JSONDecodeError:
                pass
        return [item.strip() for item in text.split("|") if item.strip()] or [text]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


STATUS_RANK = {"not_run": 0, "pass": 1, "risk": 2, "review": 3, "fail": 4}


def max_status(current: str, candidate: str) -> str:
    return candidate if STATUS_RANK.get(candidate, 0) > STATUS_RANK.get(current, 0) else current


def build_ledger_rows(
    assets: dict[str, dict[str, Any]],
    module_status: dict[str, dict[str, str]],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    events_by_asset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        events_by_asset[str(event["asset_id"])].append(event)

    rows: list[dict[str, Any]] = []
    for asset_id in sorted(assets):
        asset_events = events_by_asset.get(asset_id, [])
        final_verdict, risk_level = final_decision(asset_events)
        issue_counts = Counter(event["issue_type"] for event in asset_events)
        top_issue_types = "|".join(
            issue for issue, _ in issue_counts.most_common(5)
        )
        statuses = module_status.get(asset_id, default_module_statuses())
        rows.append(
            {
                "supplier_id": assets[asset_id].get("supplier_id", "unknown"),
                "asset_id": asset_id,
                **statuses,
                "final_verdict": final_verdict,
                "risk_level": risk_level,
                "top_issue_types": top_issue_types,
                "evidence_count": len(asset_events),
                "notes": summarize_asset_notes(asset_events),
            }
        )
    return rows


def final_decision(events: list[dict[str, Any]]) -> tuple[str, str]:
    if not events:
        return "pass", "low"
    if any(event.get("auto_verdict") == "fail" for event in events):
        return "fail", "high"
    if any(event.get("auto_verdict") == "review" for event in events):
        risk = "high" if any(event.get("severity") == "high" for event in events) else "medium"
        return "review", risk
    if any(event.get("auto_verdict") == "risk" for event in events):
        risk = "medium" if any(event.get("severity") == "medium" for event in events) else "low"
        return "risk", risk
    return "pass", "low"


def summarize_asset_notes(events: list[dict[str, Any]]) -> str:
    if not events:
        return ""
    notes = []
    for event in events[:5]:
        window = ""
        if event.get("window_start_frame") is not None:
            window = f"@{event.get('window_start_frame')}-{event.get('window_end_frame')}"
        notes.append(f"{event.get('issue_type')}{window}")
    return "; ".join(notes)


def build_supplier_issue_frequency(
    assets: dict[str, dict[str, Any]],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    totals = Counter(str(asset.get("supplier_id", "unknown")) for asset in assets.values())
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[(str(event["supplier_id"]), str(event["issue_type"]))].append(event)

    rows: list[dict[str, Any]] = []
    for (supplier_id, issue_type), group in sorted(grouped.items()):
        affected_assets = sorted({str(event["asset_id"]) for event in group})
        total_assets = int(totals.get(supplier_id, 0))
        rows.append(
            {
                "supplier_id": supplier_id,
                "issue_type": issue_type,
                "affected_assets": len(affected_assets),
                "total_assets": total_assets,
                "asset_rate": len(affected_assets) / total_assets if total_assets else 0.0,
                "affected_windows": sum(event.get("window_start_frame") is not None for event in group),
                "example_asset_ids": "|".join(affected_assets[:5]),
            }
        )
    return rows


def build_markdown_report(
    ledger_rows: list[dict[str, Any]],
    frequency_rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> str:
    verdict_counts = Counter(row["final_verdict"] for row in ledger_rows)
    risk_counts = Counter(row["risk_level"] for row in ledger_rows)
    lines = [
        "# Batch QC Ledger Report",
        "",
        f"- Assets: {len(ledger_rows)}",
        f"- Evidence events: {len(events)}",
        f"- Final verdicts: {dict(verdict_counts)}",
        f"- Risk levels: {dict(risk_counts)}",
        "",
        "## Supplier Issue Frequency",
        "",
        "| supplier_id | issue_type | affected_assets | total_assets | asset_rate | examples |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in frequency_rows[:50]:
        lines.append(
            "| {supplier_id} | {issue_type} | {affected_assets} | {total_assets} | "
            "{asset_rate:.3f} | {example_asset_ids} |".format(**row)
        )
    lines.extend(["", "## High Risk Assets", ""])
    for row in ledger_rows:
        if row["risk_level"] == "high" or row["final_verdict"] in {"fail", "review"}:
            lines.append(
                f"- `{row['asset_id']}` supplier=`{row['supplier_id']}` "
                f"verdict={row['final_verdict']} risk={row['risk_level']} "
                f"issues={row['top_issue_types']}"
            )
    return "\n".join(lines) + "\n"


def boolish_or_none(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no"}:
            return False
    return None


def scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


if __name__ == "__main__":
    raise SystemExit(main())
