"""Revision-aware persistence helpers for human-only report state.

Automatic QC modules own the machine observations in an asset report.  This
module deliberately routes every human write through the shared report loader
and atomic writer, and performs a copy-on-write ownership check before the
writer is called.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from qc_common.report import (
    StaleReportRevisionError,
    load_asset_qc_report,
    write_asset_qc_report,
)
from qc_common.schema import validate_asset_qc_report


_HUMAN_MUTABLE_TOP_LEVEL = frozenset(
    {
        "semantic_calibration",
        "manual_review",
        "pipeline_state",
        "overall_decision",
        "report_revision",
    }
)


def _require_non_empty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _validate_issue_ids(value: Sequence[str], field: str) -> list[str]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be a sequence of issue IDs")
    try:
        result = list(value)
    except TypeError as exc:
        raise ValueError(f"{field} must be a sequence of issue IDs") from exc
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{field} must contain non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must contain unique issue IDs")
    return result


def initialize_semantic_calibration(
    report: dict[str, Any],
    source_dataset_path: str,
    base_hdf5_sha256: str,
) -> None:
    """Create or complete the default semantic-calibration report block.

    Existing counters, audit entries, and extension keys are retained so a
    repeated initialization cannot erase work already captured by a caller.
    The source identity fields are intentionally refreshed from the caller's
    verified asset inputs.
    """

    if not isinstance(report, dict):
        raise TypeError("report must be a dictionary")
    source_path = _require_non_empty_string(source_dataset_path, "source_dataset_path")
    base_hash = _require_non_empty_string(base_hdf5_sha256, "base_hdf5_sha256")

    existing = report.get("semantic_calibration")
    if existing is None:
        block: dict[str, Any] = {}
    elif isinstance(existing, Mapping):
        block = copy.deepcopy(dict(existing))
    else:
        raise ValueError("semantic_calibration must be an object")

    defaults: dict[str, Any] = {
        "state": "not_started",
        "source_dataset_path": source_path,
        "base_hdf5_sha256": base_hash,
        "final_hdf5_sha256": None,
        "timeline_edit_count": 0,
        "subtask_text_edit_count": 0,
        "pending_edit": None,
        "audit": [],
    }
    for key, value in defaults.items():
        if key not in block:
            block[key] = copy.deepcopy(value)
    block["source_dataset_path"] = source_path
    block["base_hdf5_sha256"] = base_hash
    report["semantic_calibration"] = block


def initialize_manual_review(
    report: dict[str, Any], candidate_issue_ids: Sequence[str]
) -> None:
    """Initialize human review fields while preserving machine routing data."""

    if not isinstance(report, dict):
        raise TypeError("report must be a dictionary")
    candidates = _validate_issue_ids(candidate_issue_ids, "candidate_issue_ids")

    existing = report.get("manual_review")
    if existing is None:
        block: dict[str, Any] = {}
    elif isinstance(existing, Mapping):
        block = copy.deepcopy(dict(existing))
    else:
        raise ValueError("manual_review must be an object")

    prior_state = block.get("state")
    block["candidate_issue_ids"] = candidates
    block.setdefault("failures_for_batch_stats_issue_ids", [])
    if block.get("required") is None:
        block["required"] = bool(candidates)
    block.setdefault("selected_issue_ids", [])
    block.setdefault("selected_issue_id", None)
    block.setdefault("issue_reviews", {})
    block.setdefault("completed_at", None)
    if prior_state in (None, "not_evaluated"):
        block["state"] = "queued" if candidates else "not_required"
    report["manual_review"] = block


def _machine_failure(report: Mapping[str, Any]) -> bool:
    issues = report.get("issues")
    if isinstance(issues, Sequence) and not isinstance(issues, (str, bytes, bytearray)):
        if any(
            isinstance(issue, Mapping) and issue.get("severity") == "fail"
            for issue in issues
        ):
            return True

    # A module result gate is authoritative even when a malformed/legacy
    # report has no corresponding issue row.
    for key, block in report.items():
        if key in {
            "schema_version",
            "asset_id",
            "report_revision",
            "qc_config",
            "execution",
            "pipeline_state",
            "overall_decision",
            "source_files",
            "issues",
            "runtime_errors",
            "manual_review",
            "semantic_calibration",
        }:
            continue
        if not isinstance(block, Mapping):
            continue
        flow = block.get("flow")
        if not isinstance(flow, Mapping):
            continue
        result_gate = flow.get("result_gate")
        if isinstance(result_gate, Mapping) and result_gate.get("verdict") == "fail":
            return True
    return False


def _semantic_ready(report: Mapping[str, Any], machine_fail: bool) -> bool | None:
    semantic = report.get("semantic_calibration")
    if semantic is None:
        # Legacy v2 reports predate the semantic block.  Reducer callers can
        # still obtain the automatic decision; a newly created human block is
        # handled strictly below.
        return True
    if not isinstance(semantic, Mapping):
        return None
    state = semantic.get("state")
    if state == "completed":
        return True
    if state == "skipped_due_to_fail" and machine_fail:
        return True
    return False


def _manual_verdict(report: Mapping[str, Any]) -> str | None:
    manual = report.get("manual_review")
    if not isinstance(manual, Mapping):
        return None
    candidates = manual.get("candidate_issue_ids", [])
    selected = manual.get("selected_issue_ids", [])
    reviews = manual.get("issue_reviews", {})
    state = manual.get("state")

    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes, bytearray)):
        return None
    if not isinstance(selected, Sequence) or isinstance(selected, (str, bytes, bytearray)):
        return None
    if not isinstance(reviews, Mapping):
        return None
    candidate_set = set(candidates)
    selected_set = set(selected)
    if not selected_set.issubset(candidate_set):
        return None
    if not set(reviews).issubset(selected_set):
        return None
    if state == "not_required" and not candidate_set:
        return "pass"
    if state != "completed" or not selected_set.issubset(set(reviews)):
        return None
    if any(
        isinstance(review, Mapping) and review.get("verdict") == "fail"
        for review in reviews.values()
    ):
        return "fail"
    if all(
        isinstance(reviews.get(issue_id), Mapping)
        and reviews[issue_id].get("verdict") == "pass"
        for issue_id in selected_set
    ):
        return "pass"
    return None


def reduce_overall_decision(report: Mapping[str, Any]) -> str | None:
    """Reduce a validated report to the final binary business decision.

    The reducer never lets an artificial human pass erase a machine hard
    fail, and returns ``None`` whenever a required workflow stage is pending
    or has a runtime error.
    """

    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping")
    pipeline = report.get("pipeline_state")
    if not isinstance(pipeline, Mapping):
        return None
    status = pipeline.get("status")
    runtime_errors = report.get("runtime_errors")
    if isinstance(runtime_errors, Sequence) and not isinstance(
        runtime_errors, (str, bytes, bytearray)
    ) and runtime_errors:
        return None
    if status in {"pending", "running", "awaiting_external", "error"}:
        return None

    machine_fail = _machine_failure(report)
    if status == "stopped":
        return "fail"
    if status != "completed":
        return None

    semantic_ready = _semantic_ready(report, machine_fail)
    if semantic_ready is not True:
        return None

    manual_verdict = _manual_verdict(report)
    manual = report.get("manual_review")
    if isinstance(manual, Mapping):
        candidates = manual.get("candidate_issue_ids", [])
        state = manual.get("state")
        if candidates and manual_verdict is None:
            return None
        if not candidates and state not in {
            "not_required",
            "completed",
            "skipped_due_to_fail",
        }:
            return None
        if manual_verdict == "fail":
            return "fail"

    if machine_fail:
        return "fail"
    return "pass"


def _assert_human_owned_diff(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> None:
    keys = set(before) | set(after)
    for key in keys:
        if key in _HUMAN_MUTABLE_TOP_LEVEL:
            continue
        if key == "execution":
            before_execution = before.get(key)
            after_execution = after.get(key)
            if not isinstance(before_execution, Mapping) or not isinstance(
                after_execution, Mapping
            ):
                if before_execution != after_execution:
                    raise ValueError(
                        "human report mutation may only change allowed human fields; "
                        "execution identity is immutable"
                    )
                continue
            execution_keys = set(before_execution) | set(after_execution)
            for execution_key in execution_keys - {"updated_at"}:
                if before_execution.get(execution_key) != after_execution.get(
                    execution_key
                ) or (execution_key not in before_execution) != (
                    execution_key not in after_execution
                ):
                    raise ValueError(
                        "human report mutation may only change allowed human fields; "
                        "execution identity is immutable"
                    )
            continue
        if before.get(key) != after.get(key) or (key not in before) != (key not in after):
            raise ValueError(
                f"human report mutation may only change allowed human fields; "
                f"machine-owned or unknown field changed: {key}"
            )


def update_human_state(
    report_path: Path,
    expected_revision: int,
    mutate: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    """Apply one expected-revision human mutation through the shared writer."""

    if not callable(mutate):
        raise TypeError("mutate must be callable")
    path = Path(report_path)
    loaded = load_asset_qc_report(path)
    if loaded is None:
        raise FileNotFoundError(path)
    if loaded.get("schema_version") != "asset_qc_report.v2":
        raise ValueError("human report updates require an asset_qc_report.v2 report")
    current_revision = int(loaded.get("report_revision", 0))
    if current_revision != expected_revision:
        raise StaleReportRevisionError(
            f"expected revision {expected_revision}, found {current_revision}: {path}"
        )

    before = copy.deepcopy(loaded)
    candidate = copy.deepcopy(loaded)
    mutate(candidate)
    _assert_human_owned_diff(before, candidate)

    candidate["report_revision"] = expected_revision + 1
    validate_asset_qc_report(candidate)
    execution = candidate.get("execution")
    profile = execution.get("profile") if isinstance(execution, Mapping) else None
    write_asset_qc_report(
        path,
        candidate,
        expected_revision=expected_revision,
        profile=profile if isinstance(profile, str) else None,
    )
    return candidate


__all__ = [
    "initialize_manual_review",
    "initialize_semantic_calibration",
    "reduce_overall_decision",
    "update_human_state",
]
