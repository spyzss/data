"""Warn-only application facade with an explicit browser-safe DTO."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Literal, Protocol
from urllib.parse import quote

from qc_common.report import load_asset_qc_report
from qc_common.reviewer_lease import (
    Lease,
    LeaseConflictError,
    LeaseStore,
    LeaseTokenError,
)
from qc_pipeline.context import AssetContext

from .media import MediaCatalog
from .warn_service import WarnReviewService, WarnRevisionError, WarnStateError


_STABLE_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}").fullmatch
_PUBLIC_ISSUE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}").fullmatch
_UNSAFE_PUBLIC_TEXT = re.compile(
    r"[/\\]|(?:command|cmd)\s*[:=]|\b(?:ffmpeg|traceback|stack\s+trace)\b",
    re.IGNORECASE,
).search
_UNSAFE_ISSUE_ID_TEXT = re.compile(
    r"(?:command|cmd|ffmpeg|traceback|stack[._-]?trace)", re.IGNORECASE
).search
_MANUAL_REVIEW_STATES = frozenset(
    {
        "not_evaluated",
        "queued",
        "in_progress",
        "completed",
        "not_required",
        "skipped_due_to_fail",
    }
)
_COMPLETION_MODES = frozenset({"all_reviewed", "early_fail"})
_THRESHOLD_OPERATORS = frozenset({"<", "<=", ">", ">=", "==", "!="})
_REVIEW_VERDICTS = frozenset({"pass", "fail", "warn"})
_AUDIT_ACTIONS = frozenset({"resubmitted", "failure_reason_changed"})
_OVERLAY_STATUSES = frozenset({"pending", "generating", "ready", "failed"})
_CONTINUOUS_SAM3_MODULES = frozenset({"sam3_containment"})
_CONTINUOUS_SAM3_EVIDENCE_TYPES = frozenset({"sam3_continuous"})
_CONTINUOUS_SAM3_FLAGS = ("continuous_sam3", "sam3_continuous")


class InvalidIssueRangeError(WarnStateError):
    """A selected issue cannot be represented as a non-empty half-open range."""


@dataclass(frozen=True)
class FrameRangeDto:
    start_frame: int
    end_frame_exclusive: int

    def to_dict(self) -> dict[str, int]:
        return {
            "start_frame": self.start_frame,
            "end_frame_exclusive": self.end_frame_exclusive,
        }


@dataclass(frozen=True)
class VideoDto:
    url: str
    fps: float
    total_frames: int

    def to_dict(self) -> dict[str, object]:
        return {
            "url": self.url,
            "fps": self.fps,
            "total_frames": self.total_frames,
        }


@dataclass(frozen=True)
class LeaseDto:
    read_only: bool
    token: str | None
    expires_at: str | None
    code: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "read_only": self.read_only,
            "token": self.token,
            "expires_at": self.expires_at,
            "code": self.code,
        }


@dataclass(frozen=True)
class ReasonOptionDto:
    code: str
    display_name: str
    requires_text: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "display_name": self.display_name,
            "requires_text": self.requires_text,
        }


@dataclass(frozen=True)
class OverlayDto:
    status: Literal["pending", "generating", "ready", "failed"]
    frame_range: FrameRangeDto
    url: str | None
    code: str | None
    segments: tuple["OverlaySegmentDto", ...] = ()
    retryable: bool = False

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "status": self.status,
            "frame_range": self.frame_range.to_dict(),
            "url": self.url,
            "code": self.code,
        }
        # Keep the original single-overlay wire shape stable for legacy callers.
        # Asset-level SAM3 jobs add segment detail only when it exists.
        if self.segments:
            result["segments"] = [segment.to_dict() for segment in self.segments]
        if self.retryable:
            result["retryable"] = True
        return result


@dataclass(frozen=True)
class OverlaySegmentDto:
    """Browser-safe status for one continuous source-frame overlay segment."""

    frame_range: FrameRangeDto
    status: Literal["pending", "generating", "ready", "failed"]
    overlay_id: str | None
    url: str | None
    code: str | None
    retryable: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "start_frame": self.frame_range.start_frame,
            "end_frame_exclusive": self.frame_range.end_frame_exclusive,
            "status": self.status,
            "overlay_id": self.overlay_id,
            "url": self.url,
            "code": self.code,
            "retryable": self.retryable,
        }


@dataclass(frozen=True)
class OverlayHandle:
    """Provider result whose path is registered server-side, never serialized."""

    status: Literal["pending", "generating", "ready", "failed"]
    overlay_id: str | None = None
    path: str | Path | None = None
    code: str | None = None
    segments: tuple["OverlaySegmentHandle", ...] = ()
    retryable: bool = False


@dataclass(frozen=True)
class OverlaySegmentHandle:
    """Server-side-only handle for one generated continuous overlay segment."""

    frame_range: FrameRangeDto
    status: Literal["pending", "generating", "ready", "failed"]
    overlay_id: str | None = None
    path: str | Path | None = None
    code: str | None = None
    retryable: bool = False


def _relevant_overlay_segments(
    frame_range: FrameRangeDto,
    segments: Sequence[OverlaySegmentHandle],
) -> tuple[OverlaySegmentHandle, ...]:
    return tuple(
        segment
        for segment in segments
        if segment.frame_range.start_frame < frame_range.end_frame_exclusive
        and frame_range.start_frame < segment.frame_range.end_frame_exclusive
    )


def _segments_cover_issue(
    frame_range: FrameRangeDto,
    segments: Sequence[OverlaySegmentHandle],
) -> bool:
    """Return whether the half-open union covers every frame of an issue."""

    cursor = frame_range.start_frame
    for start, end in sorted(
        (
            (
                max(frame_range.start_frame, segment.frame_range.start_frame),
                min(frame_range.end_frame_exclusive, segment.frame_range.end_frame_exclusive),
            )
            for segment in segments
        )
    ):
        if end <= cursor:
            continue
        if start > cursor:
            return False
        cursor = max(cursor, end)
        if cursor >= frame_range.end_frame_exclusive:
            return True
    return False


def _combined_overlay_status(
    segments: Sequence[OverlaySegmentHandle], *, fully_covered: bool
) -> Literal["pending", "generating", "ready", "failed"]:
    if any(segment.status == "failed" for segment in segments):
        return "failed"
    if any(segment.status == "generating" for segment in segments):
        return "generating"
    if segments and all(segment.status == "ready" for segment in segments):
        return "ready" if fully_covered else "failed"
    return "pending"


@dataclass(frozen=True)
class OverlayIssueInput:
    """The only selected SAM3 data passed from the facade to an asset provider.

    ``issue_id`` is a trusted server-side identifier.  It is deliberately not
    serialized and callers receive only the public projection built later.
    """

    issue_id: str
    frame_range: FrameRangeDto


class AssetOverlayProvider(Protocol):
    """Submit/read one asset-level SAM3 overlay job for selected issue windows."""

    def get_asset_overlays(
        self, asset_id: str, selected: tuple[OverlayIssueInput, ...]
    ) -> Mapping[str, object] | None:
        """Return internal per-issue handles keyed by trusted issue id."""


class WorkerOverlayProvider:
    """Adapt :class:`BoundedOverlayWorker` to the Warn facade without HTTP work.

    The request factory owns trusted cache/model/render configuration.  The
    facade passes only selected raw issue identifiers and normalized ranges;
    this adapter submits exactly one union job and maps its segments back to
    each issue.  ``submit`` is intentionally the only worker operation used
    here, so a task GET never waits for a renderer.
    """

    def __init__(
        self,
        *,
        worker: object,
        request_factory: Callable[[str, tuple[OverlayIssueInput, ...]], object],
    ) -> None:
        submit = getattr(worker, "submit", None)
        if not callable(submit):
            raise TypeError("worker must expose submit")
        if not callable(request_factory):
            raise TypeError("request_factory must be callable")
        self._worker = worker
        self._request_factory = request_factory

    @staticmethod
    def _worker_segment(value: object) -> OverlaySegmentHandle:
        start = getattr(value, "start_frame", None)
        end = getattr(value, "end_frame_exclusive", None)
        status = getattr(value, "status", None)
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 0
            or end <= start
            or status not in {"pending", "generating", "ready", "failed"}
        ):
            raise WarnStateError("invalid overlay segment")
        overlay_id = _safe_overlay_id(getattr(value, "overlay_id", None))
        path = getattr(value, "path", None)
        code = _safe_code(getattr(value, "code", None))
        retryable = getattr(value, "retryable", False) is True
        return OverlaySegmentHandle(
            FrameRangeDto(start, end),
            status,
            overlay_id,
            path if isinstance(path, (str, Path)) else None,
            code,
            retryable,
        )

    @staticmethod
    def _issue_handle(
        frame_range: FrameRangeDto, segments: tuple[OverlaySegmentHandle, ...]
    ) -> OverlayHandle:
        relevant = _relevant_overlay_segments(frame_range, segments)
        if not relevant:
            return OverlayHandle("failed", code="overlay_unavailable", retryable=True)
        fully_covered = _segments_cover_issue(frame_range, relevant)
        status = _combined_overlay_status(relevant, fully_covered=fully_covered)
        failed = next((segment for segment in relevant if segment.status == "failed"), None)
        return OverlayHandle(
            status,
            code=(
                failed.code
                if failed is not None
                else "overlay_incomplete" if not fully_covered else None
            ),
            segments=relevant,
            retryable=(
                any(segment.retryable for segment in relevant) or not fully_covered
            ),
        )

    def get_asset_overlays(
        self, asset_id: str, selected: tuple[OverlayIssueInput, ...]
    ) -> Mapping[str, object]:
        try:
            request = self._request_factory(asset_id, selected)
            if getattr(request, "asset_id", None) != asset_id:
                raise WarnStateError("overlay_request_asset_mismatch")
            view = self._worker.submit(request)
            raw_segments = getattr(view, "segments", None)
            if isinstance(raw_segments, (str, bytes, bytearray)) or not isinstance(
                raw_segments, Sequence
            ):
                raise WarnStateError("invalid overlay job view")
            segments = tuple(self._worker_segment(segment) for segment in raw_segments)
        except WarnStateError:
            raise
        except Exception:
            # Request factories/workers operate on private paths and renderer
            # details.  Keep their failures out of public task/error payloads.
            raise WarnStateError("overlay_provider_unavailable") from None
        return {
            item.issue_id: self._issue_handle(item.frame_range, segments)
            for item in selected
        }


@dataclass(frozen=True)
class WarnIssueDto:
    id: str
    display_name: str
    frame_range: FrameRangeDto
    default_reason: str | None
    threshold: dict[str, object] | None
    evidence_type: Literal["source_video"]
    review: dict[str, object] | None
    overlay: OverlayDto | None

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "frame_range": self.frame_range.to_dict(),
            "default_reason": self.default_reason,
            "threshold": self.threshold,
            "evidence_type": self.evidence_type,
            "review": self.review,
            "overlay": None if self.overlay is None else self.overlay.to_dict(),
        }


@dataclass(frozen=True)
class ReviewAuditDto:
    action: Literal["resubmitted", "failure_reason_changed"]
    issue_id: str | None
    reviewed_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "issue_id": self.issue_id,
            "reviewed_at": self.reviewed_at,
        }


@dataclass(frozen=True)
class WarnTaskDto:
    asset_id: str
    report_revision: int
    manual_review_state: str
    completion_mode: str | None
    failure_reason: dict[str, object] | None
    review_audit: tuple[ReviewAuditDto, ...]
    can_complete: bool
    video: VideoDto
    issues: tuple[WarnIssueDto, ...]
    reason_options: tuple[ReasonOptionDto, ...]
    lease: LeaseDto

    def to_dict(self) -> dict[str, object]:
        return {
            "asset_id": self.asset_id,
            "report_revision": self.report_revision,
            "manual_review_state": self.manual_review_state,
            "completion_mode": self.completion_mode,
            "failure_reason": self.failure_reason,
            "review_audit": [entry.to_dict() for entry in self.review_audit],
            "can_complete": self.can_complete,
            "video": self.video.to_dict(),
            "issues": [issue.to_dict() for issue in self.issues],
            "reason_options": [option.to_dict() for option in self.reason_options],
            "lease": self.lease.to_dict(),
        }


REASON_OPTIONS = (
    ReasonOptionDto("occlusion", "遮挡"),
    ReasonOptionDto("action_unrecognizable", "动作不可辨"),
    ReasonOptionDto("inaccurate_interval", "标注区间不准确"),
    ReasonOptionDto("other", "其他", requires_text=True),
)


def _non_empty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidIssueRangeError(f"invalid_issue_range: {field}")
    return value


def normalize_issue_range(
    issue: Mapping[str, Any], total_frames: int
) -> FrameRangeDto:
    """Normalize canonical inclusive and legacy half-open issue windows."""

    if isinstance(total_frames, bool) or not isinstance(total_frames, int) or total_frames <= 0:
        raise ValueError("total_frames must be a positive integer")
    window = issue.get("window")
    if not isinstance(window, Mapping):
        window = {}
    context = issue.get("context")
    if not isinstance(context, Mapping):
        context = {}

    start_value: object | None = None
    for source in (issue, window, context):
        if "start_frame" in source:
            start_value = source["start_frame"]
            break

    end_value: object | None = None
    inclusive = False
    for source in (issue, window, context):
        if "end_frame_exclusive" in source:
            end_value = source["end_frame_exclusive"]
            break
    if end_value is None and "end_frame" in context:
        end_value = context["end_frame"]
        inclusive = True
    if end_value is None:
        for source in (issue, window):
            if "end_frame" in source:
                end_value = source["end_frame"]
                break

    start = 0 if start_value is None else _integer(start_value, "start_frame")
    end = total_frames if end_value is None else _integer(end_value, "end_frame")
    if inclusive:
        end += 1
    start = max(0, min(start, total_frames))
    end = max(0, min(end, total_frames))
    if end <= start:
        raise InvalidIssueRangeError("invalid_issue_range")
    return FrameRangeDto(start, end)


def _safe_code(value: object) -> str | None:
    return value if isinstance(value, str) and _STABLE_CODE(value) else None


def _is_sam3_issue(issue: Mapping[str, Any]) -> bool:
    """Require the server-side continuous-SAM3 contract, never substrings."""

    for field in _CONTINUOUS_SAM3_FLAGS:
        explicit = issue.get(field)
        if isinstance(explicit, bool):
            return explicit

    evidence_types: list[str] = []
    modules: list[str] = []

    def add_token(value: object, target: list[str]) -> None:
        if isinstance(value, str):
            target.append(value.casefold())

    add_token(issue.get("evidence_type"), evidence_types)
    add_token(issue.get("kind"), evidence_types)
    add_token(issue.get("module"), modules)
    evidence = issue.get("evidence")
    if isinstance(evidence, Mapping):
        evidence = [evidence]
    if isinstance(evidence, Sequence) and not isinstance(
        evidence, (str, bytes, bytearray)
    ):
        for row in evidence:
            if isinstance(row, Mapping):
                add_token(row.get("evidence_type", row.get("kind")), evidence_types)
                add_token(row.get("module"), modules)

    # A declared evidence type is authoritative: sampled/still overlays do
    # not become continuous media merely because their issue module is SAM3.
    if evidence_types:
        return any(
            evidence_type in _CONTINUOUS_SAM3_EVIDENCE_TYPES
            for evidence_type in evidence_types
        )
    return any(module in _CONTINUOUS_SAM3_MODULES for module in modules)


def _public_issue_id(value: str) -> str:
    if _PUBLIC_ISSUE_ID(value) and _UNSAFE_ISSUE_ID_TEXT(value) is None:
        return value
    digest = hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()
    return f"issue-{digest}"


def _safe_overlay_id(value: object) -> str | None:
    if (
        isinstance(value, str)
        and value not in {".", ".."}
        and _PUBLIC_ISSUE_ID(value) is not None
    ):
        return value
    return None


def _selected_issue_maps(
    manual: Mapping[str, Any],
) -> tuple[tuple[str, ...], dict[str, str], dict[str, str]]:
    selected = manual.get("selected_issue_ids")
    if not isinstance(selected, (list, tuple)) or any(
        not isinstance(item, str) or not item for item in selected
    ):
        raise WarnStateError("manual_review.selected_issue_ids must contain strings")
    raw_to_public: dict[str, str] = {}
    public_to_raw: dict[str, str] = {}
    for raw_issue_id in selected:
        public_issue_id = _public_issue_id(raw_issue_id)
        existing = public_to_raw.get(public_issue_id)
        if existing is not None and existing != raw_issue_id:
            raise WarnStateError("selected issue identifiers conflict")
        raw_to_public[raw_issue_id] = public_issue_id
        public_to_raw[public_issue_id] = raw_issue_id
    return tuple(selected), raw_to_public, public_to_raw


def _safe_display_text(value: object, *, fallback: str | None = None) -> str | None:
    if not isinstance(value, str):
        return fallback
    normalized = " ".join(value.strip().split())
    if (
        not normalized
        or len(normalized) > 128
        or _UNSAFE_PUBLIC_TEXT(normalized) is not None
    ):
        return fallback
    return normalized


def _safe_timestamp(value: object) -> str | None:
    if not isinstance(value, str) or not value or len(value) > 64:
        return None
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return value


def _safe_scalar(value: object) -> object | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)) and len(value) <= 32:
        result: list[object] = []
        for item in value:
            safe = _safe_scalar(item)
            if isinstance(item, (Mapping, list, tuple)) or (
                item is not None and safe is None
            ):
                return None
            result.append(safe)
        return result
    return None


def _threshold(issue: Mapping[str, Any]) -> dict[str, object] | None:
    configured = issue.get("threshold")
    if isinstance(configured, Mapping):
        operator = configured.get("operator")
        value = _safe_scalar(configured.get("value"))
    else:
        operator = issue.get("operator")
        value = _safe_scalar(issue.get("boundary_value"))
    if operator not in _THRESHOLD_OPERATORS or value is None:
        return None
    return {"operator": operator, "value": value}


def _review(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    result: dict[str, object] = {}
    for field in ("verdict", "effective_verdict", "machine_verdict"):
        raw = value.get(field)
        if raw in _REVIEW_VERDICTS:
            result[field] = raw
    reason = _safe_code(value.get("reason"))
    if reason is not None:
        result["reason"] = reason
    reviewer = _safe_display_text(value.get("reviewer"))
    if reviewer is not None:
        result["reviewer"] = reviewer
    reviewed_at = _safe_timestamp(value.get("reviewed_at"))
    if reviewed_at is not None:
        result["reviewed_at"] = reviewed_at
    return result or None


def _failure_reason(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    if value.get("mode") != "manual":
        return None
    raw_codes = value.get("reason_codes")
    if isinstance(raw_codes, (str, bytes, bytearray)) or not isinstance(
        raw_codes, Sequence
    ):
        return None
    codes = [_safe_code(item) for item in raw_codes]
    allowed_codes = {option.code for option in REASON_OPTIONS}
    if not codes or any(code is None or code not in allowed_codes for code in codes):
        return None
    other_text = _safe_display_text(value.get("other_text"))
    if "other" in codes and other_text is None:
        return None
    return {
        "mode": "manual",
        "reason_codes": codes,
        "other_text": other_text,
    }


def _manual_state(value: object) -> str:
    if not isinstance(value, str) or value not in _MANUAL_REVIEW_STATES:
        raise WarnStateError("invalid manual review state")
    return value


def _completion_mode(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in _COMPLETION_MODES:
        raise WarnStateError("invalid manual review completion mode")
    return value


def _review_audit(
    value: object, selected_issue_ids: Mapping[str, str]
) -> tuple[ReviewAuditDto, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        return ()
    result: list[ReviewAuditDto] = []
    for entry in value:
        if not isinstance(entry, Mapping):
            continue
        action = entry.get("action")
        if not isinstance(action, str) or action not in _AUDIT_ACTIONS:
            continue
        issue_id = entry.get("issue_id")
        if issue_id is not None and (
            not isinstance(issue_id, str) or issue_id not in selected_issue_ids
        ):
            continue
        public_issue_id = (
            None if issue_id is None else selected_issue_ids[issue_id]
        )
        reviewed_at = _safe_timestamp(entry.get("reviewed_at"))
        if reviewed_at is None:
            continue
        result.append(
            ReviewAuditDto(action, public_issue_id, reviewed_at)  # type: ignore[arg-type]
        )
    return tuple(result)


class WarnWorkbenchService:
    """Own Warn DTO projection, reviewer lease and safe media handles."""

    def __init__(
        self,
        *,
        reviewer: str,
        asset_contexts: Mapping[str, AssetContext],
        warn_service: WarnReviewService | Any | None = None,
        lease_store: LeaseStore | None = None,
        media_catalog: MediaCatalog | None = None,
        overlay_provider: Callable[[str, str, FrameRangeDto], object] | Any | None = None,
        lease_ttl_seconds: int = 900,
        profile: str | None = None,
    ) -> None:
        self.reviewer = _non_empty(reviewer, "reviewer")
        if (
            isinstance(lease_ttl_seconds, bool)
            or not isinstance(lease_ttl_seconds, int)
            or lease_ttl_seconds <= 0
        ):
            raise ValueError("lease_ttl_seconds must be a positive integer")
        self.asset_contexts = dict(asset_contexts)
        self.warn_service = warn_service
        self.lease_store = lease_store if lease_store is not None else LeaseStore()
        self.media_catalog = (
            media_catalog
            if media_catalog is not None
            else MediaCatalog(self.asset_contexts)
        )
        self.overlay_provider = overlay_provider
        self.lease_ttl_seconds = lease_ttl_seconds
        self.profile = profile

    def _context(self, asset_id: str) -> AssetContext:
        try:
            return self.asset_contexts[asset_id]
        except KeyError as exc:
            raise KeyError("unknown_asset") from exc

    def _report(self, asset_id: str) -> dict[str, Any]:
        context = self._context(asset_id)
        report = load_asset_qc_report(context.report_path)
        if report is None:
            raise FileNotFoundError("report_not_found")
        if report.get("asset_id", asset_id) != asset_id:
            raise WarnStateError("asset identity does not match warning report")
        return report

    @staticmethod
    def _manual(report: Mapping[str, Any]) -> Mapping[str, Any]:
        manual = report.get("manual_review")
        if not isinstance(manual, Mapping):
            raise WarnStateError("manual_review block is missing")
        return manual

    @classmethod
    def _is_editable(cls, report: Mapping[str, Any]) -> bool:
        pipeline = report.get("pipeline_state")
        if not isinstance(pipeline, Mapping):
            return False
        manual = report.get("manual_review")
        if not isinstance(manual, Mapping):
            return False
        selected = manual.get("selected_issue_ids")
        return (
            pipeline.get("status") == "awaiting_external"
            and pipeline.get("next_module") == "manual_review"
            and manual.get("state") not in {"completed", "not_required", "skipped_due_to_fail"}
            and isinstance(selected, (list, tuple))
            and any(isinstance(item, str) and item for item in selected)
        )

    def list_assets(self) -> tuple[str, ...]:
        result: list[str] = []
        for asset_id in sorted(self.asset_contexts):
            try:
                report = self._report(asset_id)
            except (FileNotFoundError, KeyError, ValueError):
                continue
            execution = report.get("execution")
            report_profile = execution.get("profile") if isinstance(execution, Mapping) else None
            if self.profile is not None and report_profile not in {None, self.profile}:
                continue
            if self._is_editable(report):
                result.append(asset_id)
        return tuple(result)

    def current_revision(self, asset_id: str) -> int | None:
        try:
            value = self._report(asset_id).get("report_revision")
        except (FileNotFoundError, KeyError, ValueError):
            return None
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    def _lease_for_task(
        self, asset_id: str, report: Mapping[str, Any], lease_token: str | None
    ) -> LeaseDto:
        if not self._is_editable(report):
            return LeaseDto(True, None, None, None)
        try:
            if lease_token is None:
                lease = self.lease_store.acquire(
                    asset_id, self.reviewer, self.lease_ttl_seconds
                )
            else:
                lease = self.lease_store.renew(
                    asset_id, lease_token, self.lease_ttl_seconds
                )
        except LeaseConflictError:
            return LeaseDto(True, None, None, "lease_held")
        except LeaseTokenError:
            return LeaseDto(True, None, None, "lease_invalid")
        return LeaseDto(False, lease.token, lease.expires_at, None)

    @classmethod
    def _can_complete(cls, report: Mapping[str, Any]) -> bool:
        if not cls._is_editable(report):
            return False
        manual = cls._manual(report)
        selected = manual.get("selected_issue_ids")
        reviews = manual.get("issue_reviews")
        if not isinstance(selected, (list, tuple)) or not selected:
            return False
        if not isinstance(reviews, Mapping):
            return False
        selected_reviews = [reviews.get(issue_id) for issue_id in selected]
        if any(
            isinstance(review, Mapping) and review.get("verdict") == "fail"
            for review in selected_reviews
        ):
            return True
        return all(
            isinstance(review, Mapping) and review.get("verdict") == "pass"
            for review in selected_reviews
        )

    @staticmethod
    def _overlay_status(value: object) -> Literal["pending", "generating", "ready", "failed"]:
        if value not in _OVERLAY_STATUSES:
            raise WarnStateError("invalid overlay status")
        return value  # type: ignore[return-value]

    @staticmethod
    def _overlay_segment_handle(value: object) -> OverlaySegmentHandle:
        if isinstance(value, OverlaySegmentHandle):
            frame_range = value.frame_range
            if (
                not isinstance(frame_range, FrameRangeDto)
                or isinstance(frame_range.start_frame, bool)
                or not isinstance(frame_range.start_frame, int)
                or isinstance(frame_range.end_frame_exclusive, bool)
                or not isinstance(frame_range.end_frame_exclusive, int)
                or frame_range.start_frame < 0
                or frame_range.end_frame_exclusive <= frame_range.start_frame
            ):
                raise WarnStateError("invalid overlay segment")
            return OverlaySegmentHandle(
                frame_range=frame_range,
                status=WarnWorkbenchService._overlay_status(value.status),
                overlay_id=_safe_overlay_id(value.overlay_id),
                path=value.path if isinstance(value.path, (str, Path)) else None,
                code=_safe_code(value.code),
                retryable=value.retryable is True,
            )
        if not isinstance(value, Mapping):
            raise WarnStateError("invalid overlay segment")
        raw_range = value.get("frame_range")
        if isinstance(raw_range, FrameRangeDto):
            frame_range = raw_range
        else:
            range_mapping = raw_range if isinstance(raw_range, Mapping) else value
            start = range_mapping.get("start_frame")
            end = range_mapping.get("end_frame_exclusive")
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(end, bool)
                or not isinstance(end, int)
                or start < 0
                or end <= start
            ):
                raise WarnStateError("invalid overlay segment")
            frame_range = FrameRangeDto(start, end)
        return OverlaySegmentHandle(
            frame_range=frame_range,
            status=WarnWorkbenchService._overlay_status(value.get("status")),
            overlay_id=_safe_overlay_id(value.get("overlay_id")),
            path=(
                value.get("path")
                if isinstance(value.get("path"), (str, Path))
                else None
            ),
            code=_safe_code(value.get("code")),
            retryable=value.get("retryable") is True,
        )

    def _overlay_segment_dto(
        self,
        asset_id: str,
        value: OverlaySegmentHandle,
        *,
        expose_ready_url: bool,
    ) -> OverlaySegmentDto:
        url: str | None = None
        if value.status == "ready" and expose_ready_url:
            if not isinstance(value.overlay_id, str) or not isinstance(
                value.path, (str, Path)
            ):
                raise WarnStateError("ready overlay is missing its opaque handle")
            self.media_catalog.allow_overlay(asset_id, value.overlay_id, value.path)
            url = (
                f"/media/assets/{quote(asset_id, safe='')}/overlays/"
                f"{quote(value.overlay_id, safe='')}"
            )
        return OverlaySegmentDto(
            frame_range=value.frame_range,
            status=value.status,
            overlay_id=value.overlay_id,
            url=url,
            code=value.code,
            retryable=value.retryable,
        )

    def _overlay_from_value(
        self,
        asset_id: str,
        frame_range: FrameRangeDto,
        value: object,
    ) -> OverlayDto:
        if isinstance(value, OverlayHandle):
            status = self._overlay_status(value.status)
            overlay_id = value.overlay_id
            path = value.path
            code = _safe_code(value.code)
            raw_segments: object = value.segments
            retryable = value.retryable is True
        elif isinstance(value, Mapping):
            status = self._overlay_status(value.get("status"))
            overlay_id = value.get("overlay_id")
            path = value.get("path")
            code = _safe_code(value.get("code"))
            raw_segments = value.get("segments", ())
            retryable = value.get("retryable") is True
        else:
            raise TypeError("overlay provider result must be a mapping")

        if isinstance(raw_segments, (str, bytes, bytearray)) or not isinstance(
            raw_segments, Sequence
        ):
            raise WarnStateError("invalid overlay segments")
        if raw_segments:
            handles = tuple(
                self._overlay_segment_handle(segment) for segment in raw_segments
            )
            relevant = _relevant_overlay_segments(frame_range, handles)
            if not relevant:
                return OverlayDto(
                    "failed",
                    frame_range,
                    None,
                    "overlay_unavailable",
                    retryable=True,
                )
            fully_covered = _segments_cover_issue(frame_range, relevant)
            status = _combined_overlay_status(
                relevant, fully_covered=fully_covered
            )
            failed = next(
                (segment for segment in relevant if segment.status == "failed"), None
            )
            if failed is not None:
                code = failed.code or code
            elif not fully_covered:
                code = "overlay_incomplete"
            segment_dtos = tuple(
                self._overlay_segment_dto(
                    asset_id, segment, expose_ready_url=status == "ready"
                )
                for segment in relevant
            )
            url = (
                segment_dtos[0].url
                if status == "ready" and len(segment_dtos) == 1
                else None
            )
            return OverlayDto(
                status,
                frame_range,
                url,
                code,
                segment_dtos,
                (
                    retryable
                    or any(segment.retryable for segment in relevant)
                    or not fully_covered
                ),
            )

        url: str | None = None
        if status == "ready":
            if not isinstance(overlay_id, str) or not isinstance(path, (str, Path)):
                raise WarnStateError("ready overlay is missing its opaque handle")
            self.media_catalog.allow_overlay(asset_id, overlay_id, path)
            url = (
                f"/media/assets/{quote(asset_id, safe='')}/overlays/"
                f"{quote(overlay_id, safe='')}"
            )
        return OverlayDto(status, frame_range, url, code, retryable=retryable)

    def _asset_overlay_getter(self) -> Callable[..., object] | None:
        provider = self.overlay_provider
        getter = getattr(provider, "get_asset_overlays", None)
        return getter if callable(getter) else None

    def _asset_overlay_values(
        self,
        asset_id: str,
        selected: tuple[OverlayIssueInput, ...],
    ) -> Mapping[str, object]:
        getter = self._asset_overlay_getter()
        if getter is None:
            return {}
        try:
            value = getter(asset_id, selected)
        except Exception:
            raise WarnStateError("overlay_provider_unavailable") from None
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise WarnStateError("invalid overlay provider result")
        # Do not pass arbitrary provider keys farther into task construction.
        return {
            item.issue_id: value[item.issue_id]
            for item in selected
            if item.issue_id in value
        }

    def _legacy_overlay(
        self,
        asset_id: str,
        issue: Mapping[str, Any],
        issue_id: str,
        frame_range: FrameRangeDto,
    ) -> OverlayDto | None:
        provider = self.overlay_provider
        if provider is None:
            return (
                OverlayDto("pending", frame_range, None, None)
                if _is_sam3_issue(issue)
                else None
            )
        getter = provider if callable(provider) else getattr(provider, "get_overlay", None)
        if not callable(getter):
            raise TypeError("overlay_provider must be callable or expose get_overlay")
        try:
            value = getter(asset_id, issue_id, frame_range)
        except Exception:
            raise WarnStateError("overlay_provider_unavailable") from None
        return None if value is None else self._overlay_from_value(asset_id, frame_range, value)

    def _issues(
        self,
        asset_id: str,
        report: Mapping[str, Any],
        *,
        total_frames: int,
    ) -> tuple[WarnIssueDto, ...]:
        manual = self._manual(report)
        selected, raw_to_public, _ = _selected_issue_maps(manual)
        raw_issues = report.get("issues")
        if not isinstance(raw_issues, (list, tuple)):
            raise WarnStateError("report issues must be a sequence")
        by_id: dict[str, Mapping[str, Any]] = {}
        for issue in raw_issues:
            if isinstance(issue, Mapping) and isinstance(issue.get("issue_id"), str):
                by_id[issue["issue_id"]] = issue
        reviews = manual.get("issue_reviews")
        if not isinstance(reviews, Mapping):
            reviews = {}
        projected: list[tuple[str, str, Mapping[str, Any], FrameRangeDto]] = []
        for issue_id in selected:
            issue = by_id.get(issue_id)
            if issue is None:
                raise WarnStateError("selected issue is missing from report")
            frame_range = normalize_issue_range(issue, total_frames)
            public_issue_id = raw_to_public[issue_id]
            projected.append((issue_id, public_issue_id, issue, frame_range))

        sam3_selected = tuple(
            OverlayIssueInput(issue_id, frame_range)
            for issue_id, _, issue, frame_range in projected
            if _is_sam3_issue(issue)
        )
        asset_overlay_values: Mapping[str, object] | None = None
        if sam3_selected and self._asset_overlay_getter() is not None:
            asset_overlay_values = self._asset_overlay_values(asset_id, sam3_selected)

        result: list[WarnIssueDto] = []
        for issue_id, public_issue_id, issue, frame_range in projected:
            code = (
                _safe_code(issue.get("code"))
                or (_safe_code(issue_id) if public_issue_id == issue_id else None)
                or "issue"
            )
            if asset_overlay_values is not None and _is_sam3_issue(issue):
                overlay_value = asset_overlay_values.get(issue_id)
                overlay = (
                    self._overlay_from_value(asset_id, frame_range, overlay_value)
                    if overlay_value is not None
                    else OverlayDto("pending", frame_range, None, None)
                )
            elif asset_overlay_values is not None:
                overlay = None
            else:
                overlay = self._legacy_overlay(asset_id, issue, issue_id, frame_range)
            result.append(
                WarnIssueDto(
                    id=public_issue_id,
                    display_name=(
                        _safe_display_text(issue.get("display_name"))
                        or _safe_display_text(issue.get("title"))
                        or code
                    ),
                    frame_range=frame_range,
                    default_reason=(
                        _safe_display_text(issue.get("default_reason"))
                        or code
                    ),
                    threshold=_threshold(issue),
                    evidence_type="source_video",
                    review=_review(reviews.get(issue_id)),
                    overlay=overlay,
                )
            )
        return tuple(result)

    def get_asset_task(
        self, asset_id: str, *, lease_token: str | None = None
    ) -> dict[str, object]:
        asset_id = _non_empty(asset_id, "asset_id")
        if lease_token is not None and (
            not isinstance(lease_token, str) or not lease_token
        ):
            raise ValueError("lease_token must be a non-empty string")
        report = self._report(asset_id)
        source = self.media_catalog.source(asset_id)
        manual = self._manual(report)
        revision = report.get("report_revision", 0)
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise WarnStateError("report_revision must be a non-negative integer")
        manual_state = _manual_state(manual.get("state", "not_evaluated"))
        completion_mode = _completion_mode(manual.get("completion_mode"))
        _, raw_to_public, _ = _selected_issue_maps(manual)
        dto = WarnTaskDto(
            asset_id=asset_id,
            report_revision=revision,
            manual_review_state=manual_state,
            completion_mode=completion_mode,
            failure_reason=_failure_reason(manual.get("failure_reason")),
            review_audit=_review_audit(manual.get("review_audit"), raw_to_public),
            can_complete=self._can_complete(report),
            video=VideoDto(
                f"/media/assets/{quote(asset_id, safe='')}/source",
                source.fps,
                source.total_frames,
            ),
            issues=self._issues(asset_id, report, total_frames=source.total_frames),
            reason_options=REASON_OPTIONS,
            lease=self._lease_for_task(asset_id, report, lease_token),
        )
        return dto.to_dict()

    def acquire_lease(self, asset_id: str, ttl_seconds: int | None = None) -> Lease:
        report = self._report(asset_id)
        if not self._is_editable(report):
            raise KeyError("asset_not_actionable")
        return self.lease_store.acquire(
            asset_id,
            self.reviewer,
            self.lease_ttl_seconds if ttl_seconds is None else ttl_seconds,
        )

    def renew_lease(
        self, asset_id: str, token: str, ttl_seconds: int | None = None
    ) -> Lease:
        report = self._report(asset_id)
        if not self._is_editable(report):
            raise KeyError("asset_not_actionable")
        return self.lease_store.renew(
            asset_id,
            token,
            self.lease_ttl_seconds if ttl_seconds is None else ttl_seconds,
        )

    def release_lease(self, asset_id: str, token: str) -> Lease:
        self._context(asset_id)
        return self.lease_store.release(asset_id, token)

    def validate_lease(self, asset_id: str, token: str) -> Lease:
        return self.lease_store.validate(asset_id, token)

    def source_media(self, asset_id: str):
        return self.media_catalog.source(asset_id)

    def overlay_media(self, asset_id: str, overlay_id: str):
        return self.media_catalog.overlay(asset_id, overlay_id)

    def _prepare_mutation(
        self, asset_id: str, expected_revision: int, lease_token: str
    ) -> Lease:
        lease = self.validate_lease(asset_id, lease_token)
        current = self.current_revision(asset_id)
        if current is None:
            raise KeyError("unknown_asset")
        if current != expected_revision:
            raise WarnRevisionError("stale_revision")
        return lease

    def _raw_issue_id(self, asset_id: str, public_issue_id: str) -> str:
        report = self._report(asset_id)
        _, _, public_to_raw = _selected_issue_maps(self._manual(report))
        raw_issue_id = public_to_raw.get(public_issue_id)
        if raw_issue_id is None:
            raise WarnStateError("selected issue is not available")
        return raw_issue_id

    def _domain_warn(self, asset_id: str, lease: Lease):
        service = self.warn_service
        if service is None or isinstance(service, WarnReviewService):
            path = (
                service.report_path(asset_id)
                if isinstance(service, WarnReviewService)
                else self._context(asset_id).report_path
            )
            return WarnReviewService(
                reports={asset_id: path},
                leases={asset_id: lease.token},
                reviewer=lease.reviewer,
            )
        return service

    def warn_verdict(
        self,
        asset_id: str,
        *,
        issue_id: str,
        verdict: str,
        reason: str | None = None,
        failure_reason: Mapping[str, object] | None = None,
        expected_revision: int,
        lease_token: str,
    ) -> dict[str, object]:
        lease = self._prepare_mutation(asset_id, expected_revision, lease_token)
        raw_issue_id = self._raw_issue_id(asset_id, issue_id)
        service = self._domain_warn(asset_id, lease)
        try:
            service.submit_verdict(
                asset_id,
                raw_issue_id,
                verdict,
                reason,
                expected_revision,
                lease_token,
                reviewer=lease.reviewer,
                failure_reason=failure_reason,
            )
        except WarnStateError as exc:
            if str(exc) != f"issue {raw_issue_id} has no machine verdict":
                raise
            raise WarnStateError("selected issue has no machine verdict") from None
        return self.get_asset_task(asset_id, lease_token=lease_token)

    def warn_complete(
        self,
        asset_id: str,
        *,
        expected_revision: int,
        lease_token: str,
        completion_mode: Literal["all_reviewed", "early_fail"] | None = None,
        failure_reason: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        lease = self._prepare_mutation(asset_id, expected_revision, lease_token)
        service = self._domain_warn(asset_id, lease)
        service.complete(
            asset_id,
            expected_revision,
            lease_token,
            completion_mode=completion_mode,
            failure_reason=failure_reason,
        )
        return self.get_asset_task(asset_id, lease_token=lease_token)


__all__ = [
    "AssetOverlayProvider",
    "FrameRangeDto",
    "InvalidIssueRangeError",
    "LeaseDto",
    "OverlayHandle",
    "OverlayDto",
    "OverlayIssueInput",
    "OverlaySegmentDto",
    "OverlaySegmentHandle",
    "REASON_OPTIONS",
    "ReasonOptionDto",
    "ReviewAuditDto",
    "VideoDto",
    "WarnIssueDto",
    "WarnTaskDto",
    "WarnWorkbenchService",
    "WorkerOverlayProvider",
    "normalize_issue_range",
]
