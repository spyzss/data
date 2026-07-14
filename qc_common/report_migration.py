from __future__ import annotations

import copy
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
    manual.setdefault("state", "not_evaluated")
    manual.setdefault("candidate_issue_ids", [])
    manual.setdefault("failures_for_batch_stats_issue_ids", [])
    manual.setdefault("selected_issue_ids", [])
    manual.setdefault("selected_issue_id", None)
    manual.setdefault("issue_reviews", {})
    manual.setdefault("completed_at", None)
    migrated.setdefault(
        "semantic_calibration",
        {
            "state": "not_started",
            "source_dataset_path": None,
            "base_hdf5_sha256": None,
            "final_hdf5_sha256": None,
            "timeline_edit_count": 0,
            "subtask_text_edit_count": 0,
            "pending_edit": None,
            "audit": [],
        },
    )
    migrated["pipeline_state"].setdefault("stop_reason", None)
    return migrated
