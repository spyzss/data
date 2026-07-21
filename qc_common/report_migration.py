from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any, Mapping


def migrate_v1_to_v2(
    report: Mapping[str, Any],
    *,
    config_reference: Mapping[str, str],
    profile: str = "acceptance",
) -> dict[str, Any]:
    if report.get("schema_version") == "asset_qc_report.v2":
        return copy.deepcopy(dict(report))
    if report.get("schema_version") != "asset_qc_report.v1":
        raise ValueError(f"unsupported asset QC schema: {report.get('schema_version')}")
    migrated = copy.deepcopy(dict(report))
    migrated["schema_version"] = "asset_qc_report.v2"
    migrated["qc_config"] = dict(config_reference)
    migrated.setdefault(
        "execution",
        {"profile": profile, "started_at": None, "updated_at": None},
    )
    migrated.setdefault("source_files", {})
    migrated.setdefault("issues", [])
    migrated.setdefault("runtime_errors", [])
    manual = migrated.setdefault("manual_review", {})
    if not isinstance(manual, dict):
        raise ValueError("manual_review must be an object")
    manual.setdefault("state", "not_evaluated")
    manual.setdefault("candidate_issue_ids", [])
    manual.setdefault("failures_for_batch_stats_issue_ids", [])
    selected_ids = manual.get("selected_issue_ids")
    issue_reviews = manual.get("issue_reviews")
    is_selected_sequence = isinstance(selected_ids, Sequence) and not isinstance(
        selected_ids, (str, bytes, bytearray)
    )
    has_all_selected_reviews = (
        manual.get("state") == "completed"
        and is_selected_sequence
        and isinstance(issue_reviews, Mapping)
        and set(selected_ids).issubset(issue_reviews)
    )
    if has_all_selected_reviews:
        manual.setdefault("selected_issue_id", None)
        manual.setdefault("completed_at", None)
        manual.setdefault("completion_mode", "all_reviewed")
        manual.setdefault("failure_reason", None)
    elif manual.get("state") != "completed":
        manual.setdefault("selected_issue_ids", [])
        manual.setdefault("selected_issue_id", None)
        manual.setdefault("issue_reviews", {})
        manual.setdefault("completed_at", None)
        manual.setdefault("completion_mode", None)
        manual.setdefault("failure_reason", None)
    semantic = migrated.setdefault("semantic_calibration", {})
    if not isinstance(semantic, dict):
        raise ValueError("semantic_calibration must be an object")
    semantic_defaults = {
        "state": "not_started",
        "source_dataset_path": None,
        "base_hdf5_sha256": None,
        "final_hdf5_sha256": None,
        "timeline_edit_count": 0,
        "subtask_text_edit_count": 0,
        "pending_edit": None,
        "audit": [],
    }
    for field, default in semantic_defaults.items():
        semantic.setdefault(field, copy.deepcopy(default))
    migrated["pipeline_state"].setdefault("stop_reason", None)
    return migrated
