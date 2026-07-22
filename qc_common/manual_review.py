"""Shared selection policy for creating a pending manual-review task."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Literal


ALL_CANDIDATES_SELECTION_POLICY = "all_candidates"
_PENDING_PIPELINE_STATUSES = frozenset({"pending", "running", "awaiting_external"})


def initialize_manual_review(
    report: dict[str, Any], candidate_issue_ids: Sequence[str]
) -> None:
    """Initialize manual-review state while preserving routing extensions."""

    if not isinstance(report, dict):
        raise TypeError("report must be a dictionary")
    candidates = _issue_ids(candidate_issue_ids, "candidate_issue_ids")
    existing = report.get("manual_review")
    if existing is None:
        block: dict[str, Any] = {}
    elif isinstance(existing, Mapping):
        block = deepcopy(dict(existing))
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


def semantic_eligibility(
    report: Mapping[str, Any],
) -> Literal["ready", "blocked", "skipped_due_to_fail"]:
    """Derive semantic-stage eligibility from the persisted manual result.

    A cursor, URL, or in-memory task projection cannot make an incomplete
    manual review eligible.  Historical or malformed completed records that
    lack a canonical completion mode fail closed.
    """

    manual = report.get("manual_review")
    if not isinstance(manual, Mapping):
        return "blocked"
    state = manual.get("state")
    if state == "not_required":
        return "ready"
    if state == "skipped_due_to_fail":
        return "skipped_due_to_fail"
    if state != "completed":
        return "blocked"
    mode = manual.get("completion_mode")
    if mode == "all_reviewed":
        return "ready"
    if mode == "early_fail":
        return "skipped_due_to_fail"
    return "blocked"


def mark_semantic_skipped_due_to_fail(report: dict[str, Any]) -> None:
    """Persist the canonical semantic skip block without erasing extensions."""

    existing = report.get("semantic_calibration")
    if existing is None:
        semantic: dict[str, Any] = {}
    elif isinstance(existing, Mapping):
        semantic = deepcopy(dict(existing))
    else:
        raise ValueError("semantic_calibration must be an object")
    defaults: dict[str, Any] = {
        "source_dataset_path": None,
        "base_hdf5_sha256": None,
        "final_hdf5_sha256": None,
        "timeline_edit_count": 0,
        "subtask_text_edit_count": 0,
        "pending_edit": None,
        "audit": [],
    }
    for key, value in defaults.items():
        semantic.setdefault(key, deepcopy(value))
    semantic["state"] = "skipped_due_to_fail"
    semantic["pending_edit"] = None
    semantic.pop("orchestrator_resume_required", None)
    report["semantic_calibration"] = semantic


def _issue_ids(value: object, field: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"manual_review.{field} must be an array")
    result = list(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"manual_review.{field} must contain non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"manual_review.{field} must contain unique issue IDs")
    return result


def select_pending_manual_review_candidates(
    report: dict[str, Any],
    *,
    policy: str = ALL_CANDIDATES_SELECTION_POLICY,
) -> bool:
    """Snapshot candidates into an empty pending review selection.

    ``candidate_issue_ids`` remains the complete machine-produced pool.  The
    current policy snapshots that pool into ``selected_issue_ids`` when the
    pipeline is about to enter ``manual_review``.  A non-empty existing
    selection is an explicit task snapshot and is never replaced.

    The named ``policy`` argument is the extension seam for future sampling,
    risk, or budget selectors.  Only the currently shipped policy is accepted
    so an unknown policy cannot silently change review coverage.
    """

    if policy != ALL_CANDIDATES_SELECTION_POLICY:
        raise ValueError(f"unsupported manual-review selection policy: {policy}")

    pipeline = report.get("pipeline_state")
    if not isinstance(pipeline, Mapping):
        raise ValueError("pipeline_state must be an object")
    if (
        pipeline.get("next_module") != "manual_review"
        or pipeline.get("status") not in _PENDING_PIPELINE_STATUSES
    ):
        return False

    manual = report.get("manual_review")
    if not isinstance(manual, dict):
        raise ValueError("manual_review must be an object")
    candidates = _issue_ids(manual.get("candidate_issue_ids", []), "candidate_issue_ids")
    selected = _issue_ids(manual.get("selected_issue_ids", []), "selected_issue_ids")
    if not set(selected).issubset(set(candidates)):
        raise ValueError(
            "manual_review.selected_issue_ids must be a subset of candidate_issue_ids"
        )
    if selected or not candidates:
        return False

    manual["selected_issue_ids"] = list(candidates)
    manual["selection_policy"] = policy
    manual["required"] = True
    manual["state"] = "queued"
    manual.setdefault("selected_issue_id", None)
    manual.setdefault("issue_reviews", {})
    manual.setdefault("completed_at", None)
    return True


__all__ = [
    "ALL_CANDIDATES_SELECTION_POLICY",
    "initialize_manual_review",
    "mark_semantic_skipped_due_to_fail",
    "select_pending_manual_review_candidates",
    "semantic_eligibility",
]
