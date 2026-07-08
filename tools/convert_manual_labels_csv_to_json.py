#!/usr/bin/env python3
"""Convert completed manual review CSV rows into a reviewable JSON patch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_manual_review_queue import (  # noqa: E402
    CONFIDENCE_ENUM,
    FAILURE_MODE_ENUM,
    MANUAL_OUTCOME_ENUM,
    MANUAL_TEMPLATE_COLUMNS,
    SEVERITY_ENUM,
)


OPTIONAL_AUTO_EVIDENCE_COLUMNS = {"severity_suggestion", "key_metrics_json", "reason"}
OPTIONAL_SEGMENT_COLUMNS = {
    "segment_id",
    "affected_start_frame",
    "affected_end_frame",
    "acceptance_status",
}
ACCEPTANCE_STATUS_ENUM = ["accepted", "rejected", "review"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert completed manual_labels_template.csv rows into a JSON "
            "patch that can be reviewed before appending to manual_labels_xingjiguitu.json."
        )
    )
    parser.add_argument("--manual-labels-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reviewer", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = convert_csv_to_patch_records(
        args.manual_labels_csv,
        default_reviewer=args.reviewer,
    )
    patch = {
        "schema_version": "skeleton_qc_manual_patch.v1",
        "source": "manual_review_queue",
        "segments": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(patch, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


def convert_csv_to_patch_records(
    path: Path,
    *,
    default_reviewer: str = "",
) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    original_columns = set(df.columns)
    df = df.where(pd.notna(df), "")
    required_columns = [
        column for column in MANUAL_TEMPLATE_COLUMNS if column not in OPTIONAL_AUTO_EVIDENCE_COLUMNS
    ]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"manual labels CSV missing required columns: {missing}")
    for column in OPTIONAL_AUTO_EVIDENCE_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    for column in OPTIONAL_SEGMENT_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    has_segment_frame_columns = bool(
        {"affected_start_frame", "affected_end_frame"} & original_columns
    )

    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(df.to_dict(orient="records"), start=2):
        manual_outcome = clean(row.get("manual_outcome"))
        if not manual_outcome:
            continue
        failure_mode = clean(row.get("failure_mode"))
        severity = clean(row.get("severity"))
        confidence = clean(row.get("confidence"))
        acceptance_status = clean(row.get("acceptance_status"))
        validate_enum("manual_outcome", manual_outcome, MANUAL_OUTCOME_ENUM, row_index)
        validate_enum("failure_mode", failure_mode, FAILURE_MODE_ENUM, row_index)
        validate_enum("severity", severity, SEVERITY_ENUM, row_index)
        validate_enum("confidence", confidence, CONFIDENCE_ENUM, row_index)
        if acceptance_status:
            validate_enum(
                "acceptance_status",
                acceptance_status,
                ACCEPTANCE_STATUS_ENUM,
                row_index,
            )

        window_start = int_or_none(row.get("window_start_frame"))
        window_end = int_or_none(row.get("window_end_frame"))
        affected_start = int_or_none(row.get("affected_start_frame"))
        affected_end = int_or_none(row.get("affected_end_frame"))
        if has_segment_frame_columns:
            start = affected_start
            end = affected_end
        else:
            start = window_start
            end = window_end
        record = {
            "review_id": clean(row.get("review_id")),
            "segment_id": clean(row.get("segment_id")),
            "supplier_id": clean(row.get("supplier_id")),
            "asset_id": clean(row.get("asset_id")),
            "start": start,
            "end": end,
            "window_start_frame": window_start,
            "window_end_frame": window_end,
            "affected_start_frame": affected_start,
            "affected_end_frame": affected_end,
            "representative_frame": int_or_none(row.get("representative_frame")),
            "label": label_from_manual_outcome(manual_outcome),
            "algorithm_outcome": manual_outcome,
            "auto_verdict": clean(row.get("auto_verdict")),
            "suggested_issue_type": clean(row.get("suggested_issue_type")),
            "severity_suggestion": clean(row.get("severity_suggestion")),
            "key_metrics_json": clean(row.get("key_metrics_json")),
            "reason": clean(row.get("reason")),
            "manual_outcome": manual_outcome,
            "failure_mode": failure_mode,
            "severity": severity,
            "confidence": confidence,
            "acceptance_status": acceptance_status,
            "comment": clean(row.get("comment")),
            "reviewer": clean(row.get("reviewer")) or default_reviewer,
            "source": "manual_review_queue",
        }
        note = record["comment"]
        if note:
            record["note"] = note
        records.append(record)
    return records


def validate_enum(
    field: str,
    value: str,
    allowed: list[str],
    row_index: int,
) -> None:
    if value not in allowed:
        raise ValueError(
            f"row {row_index}: {field} must be one of {allowed}; got {value!r}"
        )


def label_from_manual_outcome(manual_outcome: str) -> str:
    if manual_outcome in {"true_positive", "partial", "false_negative"}:
        return "positive"
    if manual_outcome in {"false_positive", "acceptable_flagged"}:
        return "acceptable_flagged"
    return "review"


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def int_or_none(value: Any) -> int | None:
    text = clean(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        raise ValueError(f"expected integer-like frame value, got {value!r}") from None


if __name__ == "__main__":
    raise SystemExit(main())
