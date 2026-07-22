"""Neutral final report decision reduction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _machine_failure(report: Mapping[str, Any]) -> bool:
    issues = report.get("issues")
    if isinstance(issues, Sequence) and not isinstance(issues, (str, bytes, bytearray)):
        if any(isinstance(issue, Mapping) and issue.get("severity") == "fail" for issue in issues):
            return True
    excluded = {
        "schema_version", "asset_id", "report_revision", "qc_config", "execution",
        "pipeline_state", "overall_decision", "source_files", "issues", "runtime_errors",
        "manual_review", "semantic_calibration",
    }
    for key, block in report.items():
        if key in excluded or not isinstance(block, Mapping):
            continue
        flow = block.get("flow")
        gate = flow.get("result_gate") if isinstance(flow, Mapping) else None
        if isinstance(gate, Mapping) and gate.get("verdict") == "fail":
            return True
    return False


def _semantic_ready(report: Mapping[str, Any], machine_fail: bool) -> bool | None:
    semantic = report.get("semantic_calibration")
    if semantic is None:
        return True
    if not isinstance(semantic, Mapping):
        return None
    state = semantic.get("state")
    return state == "completed" or (state == "skipped_due_to_fail" and machine_fail)


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
    candidate_set, selected_set = set(candidates), set(selected)
    if not selected_set.issubset(candidate_set) or not set(reviews).issubset(selected_set):
        return None
    if state == "not_required" and not selected_set:
        return "pass"
    if state != "completed" or not selected_set.issubset(set(reviews)):
        return None
    if any(isinstance(review, Mapping) and review.get("verdict") == "fail" for review in reviews.values()):
        return "fail"
    if all(isinstance(reviews.get(issue_id), Mapping) and reviews[issue_id].get("verdict") == "pass" for issue_id in selected_set):
        return "pass"
    return None


def reduce_overall_decision(report: Mapping[str, Any]) -> str | None:
    """Reduce a validated report to its final binary business decision."""

    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping")
    pipeline = report.get("pipeline_state")
    if not isinstance(pipeline, Mapping):
        return None
    status = pipeline.get("status")
    runtime_errors = report.get("runtime_errors")
    if isinstance(runtime_errors, Sequence) and not isinstance(runtime_errors, (str, bytes, bytearray)) and runtime_errors:
        return None
    if status == "error":
        return None
    machine_fail = _machine_failure(report)
    if machine_fail or status == "stopped":
        return "fail"
    if status in {"pending", "running", "awaiting_external"} or status != "completed":
        return None
    if _semantic_ready(report, machine_fail) is not True:
        return None
    manual_verdict = _manual_verdict(report)
    manual = report.get("manual_review")
    if isinstance(manual, Mapping):
        candidates = manual.get("candidate_issue_ids", [])
        selected = manual.get("selected_issue_ids", [])
        state = manual.get("state")
        if candidates:
            if not selected and state not in {"not_required", "skipped_due_to_fail"}:
                return None
            if selected and manual_verdict is None:
                return None
        if not candidates and state not in {"not_required", "completed", "skipped_due_to_fail"}:
            return None
        if manual_verdict == "fail":
            return "fail"
    return "pass"


__all__ = ["reduce_overall_decision"]
