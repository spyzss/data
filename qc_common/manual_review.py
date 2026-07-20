"""Shared selection policy for creating a pending manual-review task."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


ALL_CANDIDATES_SELECTION_POLICY = "all_candidates"
_PENDING_PIPELINE_STATUSES = frozenset({"pending", "running", "awaiting_external"})


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
    "select_pending_manual_review_candidates",
]
