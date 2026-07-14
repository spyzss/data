"""Revision-safe human review of selected warning issues.

Automatic QC owns the issue rows and their evidence.  This service owns only
the human ``manual_review`` block, the external pipeline completion fields,
and the derived overall decision.  Every mutation goes through
``update_human_state`` so machine observations remain copy-on-write protected.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from qc_common.report import StaleReportRevisionError, load_asset_qc_report

from .report_updates import reduce_overall_decision, update_human_state


class WarnServiceError(RuntimeError):
    """Base class for warning-review service errors."""


class WarnLeaseError(WarnServiceError):
    """The caller does not hold the asset's edit lease."""


class WarnRevisionError(StaleReportRevisionError, WarnServiceError):
    """The caller supplied an obsolete report revision."""


class WarnStateError(WarnServiceError):
    """The warning task cannot accept the requested operation."""


# Friendly aliases used by callers that share semantic-service terminology.
LeaseError = WarnLeaseError
StaleRevisionError = WarnRevisionError
PendingEditError = WarnStateError


@dataclass(frozen=True)
class WarnTaskView:
    """Immutable projection of one selected warning-review task."""

    asset_id: str
    report_revision: int
    state: str
    candidate_issue_ids: tuple[str, ...]
    selected_issue_ids: tuple[str, ...]
    issue_reviews: Mapping[str, Mapping[str, Any]]
    pipeline_state: str | None
    overall_decision: str | None
    lease_token: str | None = field(default=None, repr=False)

    @property
    def revision(self) -> int:
        return self.report_revision

    @property
    def manual_review_state(self) -> str:
        return self.state


def _non_empty(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _now(value: object | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    return _non_empty(value, "now")


def _string_ids(value: object, field_name: str) -> list[str]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise WarnStateError(f"manual_review.{field_name} must be a sequence")
    values = list(value)
    if any(not isinstance(item, str) or not item for item in values):
        raise WarnStateError(f"manual_review.{field_name} must contain non-empty strings")
    if len(set(values)) != len(values):
        raise WarnStateError(f"manual_review.{field_name} must contain unique IDs")
    return values


def _machine_verdict(issue: Mapping[str, Any]) -> str | None:
    """Normalize a machine issue to pass/warn/fail without mutating it."""

    raw = issue.get("verdict")
    if raw not in {"pass", "warn", "fail"}:
        raw = issue.get("severity")
    if raw not in {"pass", "warn", "fail"}:
        result_gate = issue.get("result_gate")
        if isinstance(result_gate, Mapping):
            raw = result_gate.get("verdict")
    return raw if raw in {"pass", "warn", "fail"} else None


def effective_issue_verdict(
    issue: Mapping[str, Any], review: Mapping[str, Any] | None
) -> str | None:
    """Return the effective issue result while preserving machine hard fails."""

    if not isinstance(issue, Mapping):
        raise TypeError("issue must be a mapping")
    machine = _machine_verdict(issue)
    if machine == "fail":
        return "fail"
    if review is None:
        return machine
    if not isinstance(review, Mapping):
        raise TypeError("review must be a mapping or None")
    verdict = review.get("verdict")
    if verdict not in {"pass", "fail"}:
        return machine
    return verdict


class WarnReviewService:
    """Coordinate selected warning verdicts with revision and lease checks."""

    def __init__(
        self,
        reports: Mapping[str, object] | None = None,
        leases: Mapping[str, str] | None = None,
        *,
        report_paths: Mapping[str, object] | None = None,
        report_path: str | Path | None = None,
        asset_id: str | None = None,
        lease_token: str | None = None,
        reviewer: str = "human",
        clock: Callable[[], str | datetime] | None = None,
        **legacy: object,
    ) -> None:
        if reports is None:
            reports = report_paths
        if reports is None and report_path is not None:
            inferred = asset_id or Path(report_path).stem
            reports = {inferred: report_path}
        if asset_id is not None and leases is None and lease_token is not None:
            leases = {asset_id: lease_token}
        if legacy:
            unknown = ", ".join(sorted(legacy))
            raise TypeError(f"unexpected WarnReviewService field(s): {unknown}")
        self._reports = {
            str(key): self._path(value) for key, value in (reports or {}).items()
        }
        self._leases = {str(key): str(value) for key, value in (leases or {}).items()}
        self._default_lease = lease_token
        self._reviewer = _non_empty(reviewer, "reviewer")
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    @staticmethod
    def _path(value: object) -> Path:
        if isinstance(value, (str, Path)):
            return Path(value)
        if isinstance(value, Mapping):
            for key in ("report_path", "report", "path"):
                if key in value:
                    return Path(value[key])  # type: ignore[arg-type]
        raise TypeError("report path value must be path-like")

    def report_path(self, asset_id: str) -> Path:
        try:
            return self._reports[asset_id]
        except KeyError as exc:
            raise KeyError(f"unknown warning asset: {asset_id}") from exc

    def _load(self, asset_id: str) -> dict[str, Any] | None:
        report = load_asset_qc_report(self.report_path(asset_id))
        if report is not None and str(report.get("asset_id", asset_id)) != asset_id:
            raise WarnStateError("asset identity does not match warning report")
        return report

    def _check_lease(self, asset_id: str, token: str) -> None:
        if not isinstance(token, str) or not token:
            raise WarnLeaseError("lease token is required")
        expected = self._leases.get(asset_id, self._default_lease)
        if expected is None:
            self._leases[asset_id] = token
            return
        if token != expected:
            raise WarnLeaseError(f"lease token is not held for asset {asset_id}")

    def _require_revision_lease(
        self, asset_id: str, report: Mapping[str, Any], expected_revision: int, lease_token: str
    ) -> None:
        self._check_lease(asset_id, lease_token)
        current = int(report.get("report_revision", 0))
        if current != expected_revision:
            raise WarnRevisionError(f"expected revision {expected_revision}, found {current}")

    @staticmethod
    def _manual(report: Mapping[str, Any]) -> dict[str, Any]:
        value = report.get("manual_review")
        if not isinstance(value, Mapping):
            raise WarnStateError("manual_review block is missing")
        manual = deepcopy(dict(value))
        manual.setdefault("candidate_issue_ids", [])
        manual.setdefault("selected_issue_ids", [])
        manual.setdefault("selected_issue_id", None)
        manual.setdefault("issue_reviews", {})
        manual.setdefault("completed_at", None)
        if not isinstance(manual.get("issue_reviews"), Mapping):
            raise WarnStateError("manual_review.issue_reviews must be an object")
        return manual

    @staticmethod
    def _issues(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
        raw = report.get("issues", [])
        if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
            raise WarnStateError("report issues must be a sequence")
        result: dict[str, Mapping[str, Any]] = {}
        for issue in raw:
            if not isinstance(issue, Mapping):
                raise WarnStateError("report issue must be an object")
            issue_id = issue.get("issue_id")
            if not isinstance(issue_id, str) or not issue_id:
                raise WarnStateError("report issue is missing issue_id")
            result[issue_id] = issue
        return result

    @classmethod
    def _ids(cls, report: Mapping[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
        manual = cls._manual(report)
        candidates = _string_ids(manual.get("candidate_issue_ids", []), "candidate_issue_ids")
        selected = _string_ids(manual.get("selected_issue_ids", []), "selected_issue_ids")
        if not set(selected).issubset(candidates):
            raise WarnStateError("selected issue IDs must be a subset of candidate issue IDs")
        return manual, candidates, selected

    def _view(self, asset_id: str, report: Mapping[str, Any]) -> WarnTaskView:
        manual, candidates, selected = self._ids(report)
        pipeline = report.get("pipeline_state")
        pipeline_status = pipeline.get("status") if isinstance(pipeline, Mapping) else None
        reviews = manual.get("issue_reviews", {})
        return WarnTaskView(
            asset_id=asset_id,
            report_revision=int(report.get("report_revision", 0)),
            state=str(manual.get("state", "not_evaluated")),
            candidate_issue_ids=tuple(candidates),
            selected_issue_ids=tuple(selected),
            issue_reviews=deepcopy(dict(reviews)),
            pipeline_state=pipeline_status if isinstance(pipeline_status, str) else None,
            overall_decision=report.get("overall_decision") if isinstance(report.get("overall_decision"), str) else None,
            lease_token=self._leases.get(asset_id, self._default_lease),
        )

    @staticmethod
    def _assert_semantic_ready(report: Mapping[str, Any]) -> None:
        semantic = report.get("semantic_calibration")
        if not isinstance(semantic, Mapping) or semantic.get("state") != "completed":
            raise WarnStateError("semantic calibration must be completed before warn review")
        if semantic.get("pending_edit") is not None:
            raise WarnStateError("semantic calibration has a pending edit")

    @classmethod
    def _assert_manual_cursor(cls, report: Mapping[str, Any]) -> None:
        pipeline = report.get("pipeline_state")
        if not isinstance(pipeline, Mapping):
            raise WarnStateError("pipeline_state block is missing")
        if pipeline.get("status") != "awaiting_external" or pipeline.get("next_module") != "manual_review":
            raise WarnStateError("warn review requires pipeline next_module=manual_review")

    def get_task(self, asset_id: str) -> WarnTaskView | None:
        asset_id = _non_empty(asset_id, "asset_id")
        report = self._load(asset_id)
        if report is None:
            return None
        view = self._view(asset_id, report)
        if not view.selected_issue_ids:
            return None
        return view

    def submit_verdict(
        self,
        asset_id: str,
        issue_id: str,
        verdict: Literal["pass", "fail"],
        reason: str | None,
        expected_revision: int,
        lease_token: str,
    ) -> WarnTaskView:
        report = self._load(_non_empty(asset_id, "asset_id"))
        if report is None:
            raise FileNotFoundError(self.report_path(asset_id))
        self._require_revision_lease(asset_id, report, expected_revision, lease_token)
        self._assert_semantic_ready(report)
        self._assert_manual_cursor(report)
        manual, candidates, selected = self._ids(report)
        issue_id = _non_empty(issue_id, "issue_id")
        if issue_id not in candidates:
            raise WarnStateError(f"issue {issue_id} is not a candidate")
        if issue_id not in selected:
            raise WarnStateError(f"issue {issue_id} is not selected for review")
        if verdict not in {"pass", "fail"}:
            raise ValueError("verdict must be exactly 'pass' or 'fail'")
        if reason is not None and not isinstance(reason, str):
            raise TypeError("reason must be a string or None")
        normalized_reason = reason.strip() if isinstance(reason, str) else None
        normalized_reason = normalized_reason or None
        issue = self._issues(report).get(issue_id)
        if issue is None:
            raise WarnStateError(f"selected issue {issue_id} is missing from report issues")
        machine = _machine_verdict(issue)
        if machine is None:
            raise WarnStateError(f"issue {issue_id} has no machine verdict")
        reviewed_at = _now(self._clock())
        review = {
            "verdict": verdict,
            "effective_verdict": effective_issue_verdict(
                issue, {"verdict": verdict}
            ),
            "machine_verdict": machine,
            "reason": normalized_reason,
            "reviewer": self._reviewer,
            "reviewed_at": reviewed_at,
        }

        def mutate(candidate: dict[str, Any]) -> None:
            block = candidate.get("manual_review")
            if not isinstance(block, dict):
                raise WarnStateError("manual_review block is missing")
            reviews = block.setdefault("issue_reviews", {})
            if not isinstance(reviews, dict):
                raise WarnStateError("manual_review.issue_reviews must be an object")
            prior = reviews.get(issue_id)
            if prior is not None:
                audit = block.setdefault("review_audit", [])
                if not isinstance(audit, list):
                    raise WarnStateError("manual_review.review_audit must be a list")
                audit.append(
                    {
                        "action": "resubmitted",
                        "issue_id": issue_id,
                        "previous": deepcopy(prior),
                        "reviewed_at": reviewed_at,
                    }
                )
            reviews[issue_id] = deepcopy(review)
            block["state"] = "in_progress"
            block["completed_at"] = None

        updated = update_human_state(self.report_path(asset_id), expected_revision, mutate)
        return self._view(asset_id, updated)

    def complete(
        self, asset_id: str, expected_revision: int, lease_token: str
    ) -> WarnTaskView:
        asset_id = _non_empty(asset_id, "asset_id")
        report = self._load(asset_id)
        if report is None:
            raise FileNotFoundError(self.report_path(asset_id))
        self._require_revision_lease(asset_id, report, expected_revision, lease_token)
        self._assert_semantic_ready(report)
        manual, candidates, selected = self._ids(report)
        state = str(manual.get("state", "not_evaluated"))
        pipeline = report.get("pipeline_state")
        if (
            isinstance(pipeline, Mapping)
            and pipeline.get("status") == "completed"
            and pipeline.get("next_module") is None
            and state in {"completed", "not_required"}
        ):
            return self._view(asset_id, report)
        self._assert_manual_cursor(report)
        reviews = manual.get("issue_reviews", {})
        if not isinstance(reviews, Mapping):
            raise WarnStateError("manual_review.issue_reviews must be an object")
        if selected and not set(selected).issubset(reviews):
            raise WarnStateError("every selected issue requires a verdict before completion")
        for issue_id in selected:
            review = reviews.get(issue_id)
            if not isinstance(review, Mapping) or review.get("verdict") not in {"pass", "fail"}:
                raise WarnStateError("every selected issue requires a verdict before completion")
        completed_at = _now(self._clock())

        def mutate(candidate: dict[str, Any]) -> None:
            block = candidate.get("manual_review")
            if not isinstance(block, dict):
                raise WarnStateError("manual_review block is missing")
            selected_now = _string_ids(block.get("selected_issue_ids", []), "selected_issue_ids")
            reviews_now = block.get("issue_reviews", {})
            if not isinstance(reviews_now, dict):
                raise WarnStateError("manual_review.issue_reviews must be an object")
            if selected_now:
                block["state"] = "completed"
                block["completed_at"] = completed_at
            else:
                block["state"] = "not_required"
                block["required"] = False
                block["selected_issue_id"] = None
                block["completed_at"] = None
            pipeline_now = candidate.get("pipeline_state")
            if not isinstance(pipeline_now, dict):
                raise WarnStateError("pipeline_state block is missing")
            pipeline_now["status"] = "completed"
            pipeline_now["last_completed_module"] = "manual_review"
            pipeline_now["next_module"] = None
            pipeline_now["stop_reason"] = None
            candidate["pipeline_state"] = pipeline_now
            candidate["overall_decision"] = reduce_overall_decision(candidate) or "pass"

        updated = update_human_state(self.report_path(asset_id), expected_revision, mutate)
        return self._view(asset_id, updated)


__all__ = [
    "LeaseError",
    "PendingEditError",
    "StaleRevisionError",
    "WarnLeaseError",
    "WarnRevisionError",
    "WarnReviewService",
    "WarnServiceError",
    "WarnStateError",
    "WarnTaskView",
    "effective_issue_verdict",
    "reduce_overall_decision",
]
