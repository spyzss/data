"""Revision-safe semantic calibration transactions.

The service is intentionally a small application-layer coordinator.  Timeline
math stays in :mod:`human_qc.timeline`, report ownership and atomic JSON writes
stay in :mod:`human_qc.report_updates`, and HDF5 replacement/recovery stays in
:mod:`human_qc.hdf5_commit`.  A pending edit is one report transaction per
asset; it is never applied to the source file until :meth:`complete`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import uuid
from typing import Any

from qc_common.report import StaleReportRevisionError, load_asset_qc_report
from qc_common.manual_review import semantic_eligibility

from .contracts import BoundaryEdit, BoundaryError, SegmentSnapshot, SubtaskSegment
from .hdf5_commit import (
    FinalizingRecord,
    Hdf5CommitError,
    PreparedReplacement,
    RecoveryAction,
    commit_hdf5_replacement,
    finalizing_record_from_prepared,
    prepare_hdf5_replacement,
    prepared_replacement_from_record,
    recover_hdf5_replacement,
)
from .report_updates import (
    initialize_semantic_calibration,
    reduce_overall_decision,
    update_human_state,
)
from .source_adapters import (
    Hdf5ScalarJsonSubtaskAdapter,
    LoadedSubtasks,
    encode_canonical_payload,
)
from .timeline import SharedBoundaryTimeline


class SemanticServiceError(RuntimeError):
    """Base class for semantic-workbench transaction errors."""


class LeaseError(SemanticServiceError):
    """The caller does not hold the asset's current edit lease."""


class PendingEditError(SemanticServiceError):
    """An operation is blocked while a pending edit exists."""


class TaskStateError(SemanticServiceError):
    """The report is in a state that cannot accept this operation."""


class StaleSemanticRevisionError(StaleReportRevisionError, SemanticServiceError):
    """The caller's report revision is no longer current."""


def _non_empty(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _iso_now(value: object | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    return _non_empty(value, "now")


@dataclass(frozen=True, init=False)
class BoundaryEditRequest:
    """Request to stage a move of one internal shared boundary.

    ``actor_segment_id`` is optional because a workbench can derive the actor
    from the segment whose handle was dragged.  ``boundary`` and
    ``new_boundary`` are accepted as aliases for integrations that use the
    shorter UI vocabulary.
    """

    boundary_index: int
    new_frame_exclusive: int
    actor_segment_id: str | None
    expected_revision: int
    lease_token: str
    reviewer: str
    now: str | datetime | None

    def __init__(
        self,
        boundary_index: int | None = None,
        new_frame_exclusive: int | None = None,
        actor_segment_id: str | None = None,
        expected_revision: int | None = None,
        lease_token: str | None = None,
        reviewer: str = "human",
        now: str | datetime | None = None,
        **aliases: object,
    ) -> None:
        if boundary_index is None:
            boundary_index = aliases.pop("boundary", aliases.pop("boundary_id", None))
        if new_frame_exclusive is None:
            new_frame_exclusive = aliases.pop(
                "new_boundary",
                aliases.pop(
                    "new_frame",
                    aliases.pop("new_boundary_frame", aliases.pop("target_frame", aliases.pop("frame", None))),
                ),
            )
        if actor_segment_id is None:
            actor_value = aliases.pop("actor", aliases.pop("segment_id", None))
            if actor_value is not None:
                actor_segment_id = actor_value  # type: ignore[assignment]
        if expected_revision is None:
            expected_revision = aliases.pop("revision", None)
        if lease_token is None:
            lease_token = aliases.pop("lease", None)
        if aliases:
            unknown = ", ".join(sorted(aliases))
            raise TypeError(f"unexpected BoundaryEditRequest field(s): {unknown}")
        if boundary_index is None or new_frame_exclusive is None:
            raise TypeError("boundary_index and new_frame_exclusive are required")
        if expected_revision is None:
            raise TypeError("expected_revision is required")
        if lease_token is None:
            raise TypeError("lease_token is required")
        object.__setattr__(self, "boundary_index", boundary_index)
        object.__setattr__(self, "new_frame_exclusive", new_frame_exclusive)
        object.__setattr__(self, "actor_segment_id", actor_segment_id)
        object.__setattr__(self, "expected_revision", expected_revision)
        object.__setattr__(self, "lease_token", lease_token)
        object.__setattr__(self, "reviewer", reviewer)
        object.__setattr__(self, "now", now)


@dataclass(frozen=True, init=False)
class TextEditRequest:
    """Request to stage a text replacement for exactly one segment."""

    segment_id: str
    text_cn: str
    text_en: str
    expected_revision: int
    lease_token: str
    reviewer: str
    now: str | datetime | None

    def __init__(
        self,
        segment_id: str | None = None,
        text_cn: str | None = None,
        text_en: str | None = None,
        expected_revision: int | None = None,
        lease_token: str | None = None,
        reviewer: str = "human",
        now: str | datetime | None = None,
        **aliases: object,
    ) -> None:
        if segment_id is None:
            segment_id = aliases.pop("internal_id", aliases.pop("id", None))  # type: ignore[assignment]
        if text_cn is None:
            text_cn = aliases.pop(
                "new_text_cn", aliases.pop("new_cn", aliases.pop("cn", None))
            )  # type: ignore[assignment]
        if text_en is None:
            text_en = aliases.pop(
                "new_text_en", aliases.pop("new_en", aliases.pop("en", None))
            )  # type: ignore[assignment]
        if expected_revision is None:
            expected_revision = aliases.pop("revision", None)  # type: ignore[assignment]
        if lease_token is None:
            lease_token = aliases.pop("lease", None)  # type: ignore[assignment]
        if aliases:
            unknown = ", ".join(sorted(aliases))
            raise TypeError(f"unexpected TextEditRequest field(s): {unknown}")
        for value, name in (
            (segment_id, "segment_id"),
            (text_cn, "text_cn"),
            (text_en, "text_en"),
            (lease_token, "lease_token"),
        ):
            if value is None:
                raise TypeError(f"{name} is required")
        if expected_revision is None:
            raise TypeError("expected_revision is required")
        object.__setattr__(self, "segment_id", segment_id)
        object.__setattr__(self, "text_cn", text_cn)
        object.__setattr__(self, "text_en", text_en)
        object.__setattr__(self, "expected_revision", expected_revision)
        object.__setattr__(self, "lease_token", lease_token)
        object.__setattr__(self, "reviewer", reviewer)
        object.__setattr__(self, "now", now)


@dataclass(frozen=True)
class TextEdit:
    """A one-segment pending text transaction."""

    segment_id: str
    before: tuple[SegmentSnapshot]
    after: tuple[SegmentSnapshot]
    reviewer: str
    created_at: str

    @property
    def edit_type(self) -> str:
        return "text"

    @property
    def affected_segment_ids(self) -> tuple[str]:
        return (self.segment_id,)


@dataclass(frozen=True)
class SemanticTaskView:
    """Immutable state returned to the workbench after each transaction."""

    asset_id: str
    timeline: SharedBoundaryTimeline
    pending_edit: BoundaryEdit | TextEdit | None
    report_revision: int
    timeline_edit_count: int
    subtask_text_edit_count: int
    report_state: str
    pipeline_state: str | None
    semantic_consistency_state: str | None
    hdf5_sha256: str
    hdf5_path: str
    source_dataset_path: str
    lease_token: str | None = field(default=None, repr=False)

    @property
    def revision(self) -> int:
        return self.report_revision

    @property
    def state(self) -> str:
        return self.report_state

    @property
    def semantic_state(self) -> str:
        return self.report_state

    @property
    def source_path(self) -> str:
        """Backward-compatible alias for the source HDF5 path."""

        return self.hdf5_path


@dataclass
class _AssetState:
    asset_id: str
    source_path: Path
    report_path: Path | None
    loaded: LoadedSubtasks
    timeline: SharedBoundaryTimeline
    report: dict[str, Any]
    pending_edit: BoundaryEdit | TextEdit | None
    revision: int
    lease_token: str | None


def _snapshot_dict(snapshot: SegmentSnapshot) -> dict[str, Any]:
    return {
        "internal_id": snapshot.internal_id,
        "start_frame": snapshot.start_frame,
        "end_frame_exclusive": snapshot.end_frame_exclusive,
        "text_cn": snapshot.text_cn,
        "text_en": snapshot.text_en,
        "canonical_record": deepcopy(dict(snapshot.canonical_record or {})),
    }


def _snapshot_from_dict(value: Mapping[str, Any]) -> SegmentSnapshot:
    if not isinstance(value, Mapping):
        raise TaskStateError("pending snapshot must be an object")
    try:
        return SegmentSnapshot(
            internal_id=str(value["internal_id"]),
            start_frame=_strict_int(value["start_frame"], "pending snapshot start_frame"),
            end_frame_exclusive=_strict_int(
                value["end_frame_exclusive"], "pending snapshot end_frame_exclusive"
            ),
            text_cn=str(value.get("text_cn", "")),
            text_en=str(value.get("text_en", "")),
            canonical_record=deepcopy(value.get("canonical_record", {})),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TaskStateError("pending snapshot is malformed") from exc


def _strict_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TaskStateError(f"{field_name} must be an integer")
    return value


def _pending_dict(edit: BoundaryEdit | TextEdit, *, action: str | None = None) -> dict[str, Any]:
    if isinstance(edit, BoundaryEdit):
        value: dict[str, Any] = {
            "edit_type": "boundary",
            "boundary_id": edit.boundary_id,
            "boundary_index": edit.boundary_index,
            "actor_segment_id": edit.actor_segment_id,
            "affected_segment_ids": list(edit.affected_segment_ids),
            "before": [_snapshot_dict(item) for item in edit.before],
            "after": [_snapshot_dict(item) for item in edit.after],
            "reviewer": edit.reviewer,
            "created_at": edit.created_at,
        }
    else:
        value = {
            "edit_type": "text",
            "segment_id": edit.segment_id,
            "affected_segment_ids": [edit.segment_id],
            "before": [_snapshot_dict(item) for item in edit.before],
            "after": [_snapshot_dict(item) for item in edit.after],
            "reviewer": edit.reviewer,
            "created_at": edit.created_at,
        }
    if action is not None:
        value["action"] = action
    return value


def _pending_from_dict(value: object) -> BoundaryEdit | TextEdit | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TaskStateError("pending_edit must be an object or null")
    kind = value.get("edit_type", value.get("type", value.get("kind")))
    aliases = {"boundary_edit": "boundary", "text_edit": "text"}
    kind = aliases.get(kind, kind)
    before_raw = value.get("before")
    after_raw = value.get("after")
    if isinstance(before_raw, Mapping):
        before_raw = [before_raw]
    if isinstance(after_raw, Mapping):
        after_raw = [after_raw]
    if not isinstance(before_raw, Sequence) or isinstance(before_raw, (str, bytes)):
        raise TaskStateError("pending before snapshots are malformed")
    if not isinstance(after_raw, Sequence) or isinstance(after_raw, (str, bytes)):
        raise TaskStateError("pending after snapshots are malformed")
    before = tuple(_snapshot_from_dict(item) for item in before_raw)
    after = tuple(_snapshot_from_dict(item) for item in after_raw)
    reviewer = str(value.get("reviewer", "human"))
    created_at = str(value.get("created_at", ""))
    affected = value.get("affected_segment_ids")
    if not isinstance(affected, Sequence) or isinstance(affected, (str, bytes)):
        affected = []
    affected_ids = tuple(str(item) for item in affected)
    if kind == "boundary":
        try:
            boundary_index = _strict_int(
                value["boundary_index"], "pending boundary_index"
            )
            boundary_id = str(value.get("boundary_id", f"b{boundary_index}"))
            actor = str(value["actor_segment_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TaskStateError("boundary pending edit is malformed") from exc
        if len(before) != 2 or len(after) != 2 or len(affected_ids) != 2:
            raise TaskStateError("boundary pending edit must contain two snapshots")
        return BoundaryEdit(
            boundary_id=boundary_id,
            boundary_index=boundary_index,
            actor_segment_id=actor,
            affected_segment_ids=(affected_ids[0], affected_ids[1]),
            before=(before[0], before[1]),
            after=(after[0], after[1]),
            reviewer=reviewer,
            created_at=created_at,
        )
    if kind == "text":
        segment_id = str(value.get("segment_id", affected_ids[0] if affected_ids else ""))
        if len(before) != 1 or len(after) != 1 or not segment_id:
            raise TaskStateError("text pending edit must contain one snapshot")
        return TextEdit(
            segment_id=segment_id,
            before=(before[0],),
            after=(after[0],),
            reviewer=reviewer,
            created_at=created_at,
        )
    raise TaskStateError("pending edit type must be boundary or text")


def _timeline_dict(timeline: SharedBoundaryTimeline) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for ordinal, segment in enumerate(timeline.segments):
        result.append(
            {
                # ``SubtaskSourceAdapter`` derives a source ID from the
                # original frame interval.  A boundary edit necessarily
                # changes that interval, so ordinal is the durable fallback
                # identity used after an HDF5 replacement/restart.
                "ordinal": ordinal,
                "internal_id": segment.internal_id,
                "start_frame": segment.start_frame,
                "end_frame_exclusive": segment.end_frame_exclusive,
                "text_cn": segment.text_cn,
                "text_en": segment.text_en,
                "canonical_record": deepcopy(dict(segment.canonical_record)),
            }
        )
    return result


def _timeline_from_report(
    loaded: LoadedSubtasks, semantic: Mapping[str, Any]
) -> SharedBoundaryTimeline:
    raw = semantic.get("working_timeline")
    if raw is None:
        return loaded.timeline
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TaskStateError("semantic working_timeline must be a list")
    by_id = {segment.internal_id: segment for segment in loaded.timeline.segments}
    segments: list[SubtaskSegment] = []
    seen_ordinals: set[int] = set()
    for position, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise TaskStateError("semantic working_timeline segment is malformed")
        internal_id = str(value.get("internal_id", ""))
        source = by_id.get(internal_id)
        ordinal = position
        if source is None:
            ordinal = value.get("ordinal", position)
            if (
                isinstance(ordinal, bool)
                or not isinstance(ordinal, int)
                or ordinal < 0
                or ordinal >= len(loaded.timeline.segments)
                or ordinal != position
            ):
                raise TaskStateError(
                    "semantic working_timeline contains an invalid stable ordinal"
                )
            source = loaded.timeline.segments[ordinal]
            _validate_stable_segment_identity(
                asset_id=loaded.asset_id,
                ordinal=ordinal,
                report_value=value,
                source=source,
            )
        elif "ordinal" in value:
            ordinal = value["ordinal"]
            if (
                isinstance(ordinal, bool)
                or not isinstance(ordinal, int)
                or ordinal < 0
                or ordinal >= len(loaded.timeline.segments)
                or ordinal != position
                or loaded.timeline.segments[ordinal].internal_id != source.internal_id
            ):
                raise TaskStateError(
                    "semantic working_timeline ordinal does not match source identity"
                )
        # An unchanged frame-derived ID is not sufficient evidence: an
        # external writer can alter verb/object/evidence while keeping the
        # same interval.  Validate immutable source fields on both exact-ID
        # and ordinal-fallback paths before accepting the report timeline.
        _validate_stable_segment_identity(
            asset_id=loaded.asset_id,
            ordinal=ordinal,
            report_value=value,
            source=source,
        )
        seen_ordinals.add(ordinal)
        try:
            segment = replace(
                source,
                # Keep the report's stable ID even when the freshly loaded
                # HDF5 adapter generated a new frame-derived source ID.
                internal_id=internal_id,
                start_frame=_strict_int(
                    value["start_frame"], "working_timeline start_frame"
                ),
                end_frame_exclusive=_strict_int(
                    value["end_frame_exclusive"],
                    "working_timeline end_frame_exclusive",
                ),
                text_cn=str(value.get("text_cn", source.text_cn)),
                text_en=str(value.get("text_en", source.text_en)),
                canonical_record=deepcopy(
                    value.get("canonical_record", source.canonical_record)
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TaskStateError("semantic working_timeline segment is malformed") from exc
        segments.append(segment)
    if len(segments) != len(loaded.timeline.segments) or seen_ordinals != set(
        range(len(loaded.timeline.segments))
    ):
        raise TaskStateError("semantic working_timeline must cover every source segment")
    return SharedBoundaryTimeline(
        frame_count=loaded.timeline.frame_count,
        fps=loaded.timeline.fps,
        segments=tuple(segments),
        asset_id=loaded.asset_id,
    )


def _validate_stable_segment_identity(
    *,
    asset_id: str,
    ordinal: int,
    report_value: Mapping[str, Any],
    source: SubtaskSegment,
) -> None:
    """Validate ordinal fallback without weakening asset/source checks.

    The source adapter's ID formula is retained as a proof that the report
    entry belongs to this asset and original source row.  Only frame/time and
    editable text fields are allowed to differ after a successful semantic
    edit; immutable semantic evidence must still match the freshly loaded
    source row.
    """

    canonical = report_value.get("canonical_record")
    if not isinstance(canonical, Mapping):
        raise TaskStateError(
            "semantic working_timeline cannot recover an unknown segment without canonical identity"
        )
    start = canonical.get("start_frame")
    end = canonical.get("end_frame")
    expected_id = ""
    if (
        isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(end, int)
        and not isinstance(end, bool)
    ):
        expected_id = hashlib.sha256(
            f"{asset_id}:{ordinal}:{start}:{end}".encode("utf-8")
        ).hexdigest()[:16]
    if str(report_value.get("internal_id", "")) != expected_id:
        raise TaskStateError(
            "semantic working_timeline stable identity does not match source ordinal"
        )
    immutable_fields = (
        "verb",
        "object",
        "target",
        "hand",
        "phase",
        "evidence_frames",
        "confidence",
        "status",
    )
    for field_name in immutable_fields:
        if field_name in canonical and canonical[field_name] != source.canonical_record.get(field_name):
            raise TaskStateError(
                f"semantic working_timeline source identity differs at {field_name!r}"
            )


def _advance_pipeline_after_semantic(candidate: dict[str, Any]) -> None:
    """Complete a config-less semantic cursor after manual review."""

    pipeline = _require_semantic_pipeline_cursor(candidate)
    pipeline["last_completed_module"] = "semantic_consistency"
    pipeline["stop_reason"] = None
    pipeline["status"] = "completed"
    pipeline["next_module"] = None
    candidate["pipeline_state"] = pipeline
    candidate["overall_decision"] = reduce_overall_decision(candidate) or "pass"


def _require_semantic_pipeline_cursor(value: Mapping[str, Any]) -> dict[str, Any]:
    """Require a report paused at the semantic external module."""

    pipeline = value.get("pipeline_state")
    if not isinstance(pipeline, dict):
        raise TaskStateError("pipeline_state must be an object")
    if pipeline.get("status") != "awaiting_external":
        raise TaskStateError(
            "semantic completion requires pipeline status=awaiting_external"
        )
    if pipeline.get("next_module") != "semantic_consistency":
        raise TaskStateError(
            "semantic completion requires pipeline next_module=semantic_consistency"
        )
    if semantic_eligibility(value) != "ready":
        raise TaskStateError(
            "semantic task is blocked until manual review is completed or not_required"
        )
    return pipeline


def _require_semantic_task_access(value: Mapping[str, Any]) -> dict[str, Any]:
    """Require the persisted manual gate for live and historical task reads."""

    semantic = value.get("semantic_calibration")
    pipeline = value.get("pipeline_state")
    if isinstance(semantic, Mapping) and semantic.get("state") == "completed":
        if not isinstance(pipeline, dict):
            raise TaskStateError("pipeline_state must be an object")
        if semantic_eligibility(value) != "ready":
            raise TaskStateError(
                "semantic task is blocked until manual review has a legal terminal state"
            )
        return pipeline
    return _require_semantic_pipeline_cursor(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SemanticCalibrationService:
    """Coordinate one pending semantic transaction per asset.

    ``assets`` and ``reports`` are mappings from asset ID to local paths.  A
    report path may be omitted when callers only need read-only timeline
    inspection; every mutation and completion requires a report path.  Lease
    tokens are injected by the caller so this service remains independent of a
    particular HTTP or queue implementation.
    """

    def __init__(
        self,
        assets: Mapping[str, object] | None = None,
        reports: Mapping[str, object] | None = None,
        leases: Mapping[str, str] | None = None,
        *,
        asset_paths: Mapping[str, object] | None = None,
        report_paths: Mapping[str, object] | None = None,
        lease_token: str | None = None,
        adapter: Hdf5ScalarJsonSubtaskAdapter | None = None,
        source_adapter: Hdf5ScalarJsonSubtaskAdapter | None = None,
        dataset_path: str = "/label/subtask_label",
        clock: Callable[[], str | datetime] | None = None,
        asset_id: str | None = None,
        source_path: str | Path | None = None,
        report_path: str | Path | None = None,
        **legacy: object,
    ) -> None:
        if assets is None and source_path is None:
            source_path = legacy.pop(
                "hdf5_path", legacy.pop("source_hdf5_path", None)
            )  # type: ignore[assignment]
        if assets is None:
            assets = legacy.pop("hdf5_paths", legacy.pop("source_paths", None))  # type: ignore[assignment]
        if reports is None and report_path is None:
            report_path = legacy.pop("qc_report_path", None)  # type: ignore[assignment]
        if assets is None and source_path is not None:
            inferred_id = asset_id or Path(source_path).stem
            assets = {inferred_id: source_path}
        if reports is None and report_path is not None:
            inferred_id = asset_id or (next(iter(assets)) if assets else Path(report_path).stem)
            reports = {inferred_id: report_path}
        if asset_id is not None and leases is None and lease_token is not None:
            leases = {asset_id: lease_token}
        if legacy:
            unknown = ", ".join(sorted(legacy))
            raise TypeError(f"unexpected SemanticCalibrationService field(s): {unknown}")
        values = assets if assets is not None else asset_paths
        if values is None:
            values = {}
        self._assets = {
            str(asset_id): self._extract_path(value, "hdf5")
            for asset_id, value in values.items()
        }
        report_values = reports if reports is not None else report_paths
        self._reports = {
            str(asset_id): self._extract_path(value, "report")
            for asset_id, value in (report_values or {}).items()
        }
        self._leases = {str(key): str(value) for key, value in (leases or {}).items()}
        self._default_lease = lease_token
        self._adapter = source_adapter or adapter or Hdf5ScalarJsonSubtaskAdapter(dataset_path)
        self._dataset_path = dataset_path
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._states: dict[str, _AssetState] = {}
        self._active_pending_asset: str | None = None

    @staticmethod
    def _extract_path(value: object, kind: str) -> Path:
        if isinstance(value, (str, Path)):
            return Path(value)
        if isinstance(value, Mapping):
            keys = (
                ("hdf5_path", "source_path", "path", "hdf5")
                if kind == "hdf5"
                else ("report_path", "report", "path")
            )
            for key in keys:
                if key in value:
                    return Path(value[key])  # type: ignore[arg-type]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if value:
                return Path(value[0])
        raise TypeError(f"{kind} path value must be path-like")

    def report_path(self, asset_id: str) -> Path:
        state_path = self._reports.get(asset_id)
        if state_path is not None:
            return state_path
        if asset_id in self._assets:
            return self._assets[asset_id].parent / "quality_archive" / f"{asset_id}.json"
        raise KeyError(asset_id)

    def _asset_path(self, asset_id: str) -> Path:
        try:
            return self._assets[asset_id]
        except KeyError as exc:
            raise KeyError(f"unknown semantic asset: {asset_id}") from exc

    def _lease_for(self, asset_id: str) -> str | None:
        return self._leases.get(asset_id, self._default_lease)

    def _check_lease(self, state: _AssetState, token: str) -> None:
        expected = state.lease_token or self._lease_for(state.asset_id)
        if not isinstance(token, str) or not token:
            raise LeaseError("lease token is required")
        if expected is None:
            # Bind an unconfigured token to this in-process service instance;
            # deployments can inject a fixed token through ``leases``.
            state.lease_token = token
            self._leases[state.asset_id] = token
            return
        if token != expected:
            raise LeaseError(f"lease token is not held for asset {state.asset_id}")

    def _load_state(self, asset_id: str, *, recover: bool = True) -> _AssetState:
        source_path = self._asset_path(asset_id)
        report_path = self._reports.get(asset_id)
        if report_path is None:
            report_path = self.report_path(asset_id) if self._reports or source_path else None
        loaded = self._adapter.load(source_path)
        report = load_asset_qc_report(report_path) if report_path is not None else None
        if report is None:
            report = {
                "asset_id": asset_id,
                "report_revision": 0,
                "semantic_calibration": {},
                "pipeline_state": {"status": "awaiting_external"},
            }
        if str(report.get("asset_id", asset_id)) != asset_id or loaded.asset_id != asset_id:
            raise TaskStateError("asset identity does not match semantic task")
        semantic = report.get("semantic_calibration")
        if not isinstance(semantic, Mapping):
            semantic = {}
        if recover and semantic.get("state") == "finalizing":
            self._recover_finalizing(
                asset_id,
                source_path,
                report_path,
                loaded,
                dict(report),
            )
            report = load_asset_qc_report(report_path) if report_path is not None else report
            if report is None:
                raise TaskStateError("finalizing recovery removed the report")
            semantic = report.get("semantic_calibration")
            if not isinstance(semantic, Mapping):
                semantic = {}
        timeline = _timeline_from_report(loaded, semantic)
        pending = _pending_from_dict(semantic.get("pending_edit"))
        state = _AssetState(
            asset_id=asset_id,
            source_path=source_path,
            report_path=report_path,
            loaded=loaded,
            timeline=timeline,
            report=dict(report),
            pending_edit=pending,
            revision=int(report.get("report_revision", 0)),
            lease_token=self._lease_for(asset_id),
        )
        self._states[asset_id] = state
        if pending is not None:
            self._active_pending_asset = asset_id
        elif self._active_pending_asset == asset_id:
            self._active_pending_asset = None
        return state

    def _assert_navigation(self, asset_id: str) -> None:
        if self._active_pending_asset is None:
            # A new service instance must honor a pending lock already durable
            # in another process; inspect report headers without loading HDF5.
            report_candidates = dict(self._reports)
            for other_id in self._assets:
                report_candidates.setdefault(other_id, self.report_path(other_id))
            for other_id, report_path in report_candidates.items():
                if other_id == asset_id or not report_path.is_file():
                    continue
                try:
                    report = load_asset_qc_report(report_path)
                    semantic = report.get("semantic_calibration") if isinstance(report, Mapping) else None
                    if isinstance(semantic, Mapping) and semantic.get("pending_edit") is not None:
                        self._active_pending_asset = other_id
                        break
                except (OSError, ValueError, TypeError):
                    continue
        if self._active_pending_asset is not None and self._active_pending_asset != asset_id:
            raise PendingEditError(
                f"asset {self._active_pending_asset} has a pending edit; navigation is locked"
            )

    def get_task(self, asset_id: str) -> SemanticTaskView:
        asset_id = _non_empty(asset_id, "asset_id")
        self._assert_navigation(asset_id)
        report_path = self._reports.get(asset_id)
        if report_path is not None and not report_path.is_file():
            raise FileNotFoundError(report_path)
        state = self._load_state(asset_id)
        if state.report_path is not None and state.report_path.is_file():
            _require_semantic_task_access(state.report)
        return self._view(state)

    def _view(self, state: _AssetState) -> SemanticTaskView:
        semantic = state.report.get("semantic_calibration")
        if not isinstance(semantic, Mapping):
            semantic = {}
        pipeline = state.report.get("pipeline_state")
        pipeline_status = pipeline.get("status") if isinstance(pipeline, Mapping) else None
        consistency = state.report.get("semantic_consistency")
        consistency_state = consistency.get("state") if isinstance(consistency, Mapping) else None
        return SemanticTaskView(
            asset_id=state.asset_id,
            timeline=state.timeline,
            pending_edit=state.pending_edit,
            report_revision=state.revision,
            timeline_edit_count=int(semantic.get("timeline_edit_count", 0)),
            subtask_text_edit_count=int(semantic.get("subtask_text_edit_count", 0)),
            report_state=str(semantic.get("state", "not_started")),
            pipeline_state=pipeline_status,
            semantic_consistency_state=consistency_state,
            hdf5_sha256=_sha256(state.source_path),
            hdf5_path=str(state.source_path),
            source_dataset_path=str(
                semantic.get("source_dataset_path", self._dataset_path)
            ),
            lease_token=state.lease_token,
        )

    def _require_mutable(self, state: _AssetState, expected_revision: int, lease_token: str) -> None:
        _require_semantic_pipeline_cursor(state.report)
        self._check_lease(state, lease_token)
        if state.revision != expected_revision:
            raise StaleSemanticRevisionError(
                f"expected revision {expected_revision}, found {state.revision}"
            )
        semantic = state.report.get("semantic_calibration")
        current_state = semantic.get("state") if isinstance(semantic, Mapping) else "not_started"
        if current_state in {"completed", "finalizing", "error", "skipped_due_to_fail"}:
            raise TaskStateError(f"semantic task is not editable in state {current_state}")
        pipeline = state.report.get("pipeline_state")
        pipeline_status = pipeline.get("status") if isinstance(pipeline, Mapping) else None
        if pipeline_status in {"completed", "stopped", "error"}:
            raise TaskStateError(f"pipeline is not editable in state {pipeline_status}")
        if state.pending_edit is not None:
            raise PendingEditError("a pending edit must be confirmed or cancelled first")

    def _mutate_report(
        self,
        state: _AssetState,
        expected_revision: int,
        mutate: Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]:
        if state.report_path is None:
            # A report-less read-only task is useful for timeline previews but
            # cannot be durable.  Keep this branch deterministic for tests and
            # local tools that inject only a timeline.
            candidate = deepcopy(state.report)
            mutate(candidate)
            candidate["report_revision"] = expected_revision + 1
            state.report = candidate
            state.revision = expected_revision + 1
            return candidate
        candidate = update_human_state(state.report_path, expected_revision, mutate)
        state.report = candidate
        state.revision = int(candidate.get("report_revision", expected_revision + 1))
        return candidate

    def _prepare_semantic_block(
        self, candidate: dict[str, Any], state: _AssetState, source_hash: str
    ) -> dict[str, Any]:
        initialize_semantic_calibration(candidate, self._dataset_path, source_hash)
        semantic = candidate.get("semantic_calibration")
        if not isinstance(semantic, dict):
            raise TaskStateError("semantic_calibration must be an object")
        return semantic

    def begin_boundary_edit(self, asset_id: str, request: BoundaryEditRequest) -> SemanticTaskView:
        if not isinstance(request, BoundaryEditRequest):
            raise TypeError("request must be a BoundaryEditRequest")
        self._assert_navigation(asset_id)
        state = self._load_state(asset_id)
        self._require_mutable(state, request.expected_revision, request.lease_token)
        actor = request.actor_segment_id
        if actor is None and isinstance(request.boundary_index, int) and 1 <= request.boundary_index < len(state.timeline.segments):
            actor = state.timeline.segments[request.boundary_index - 1].internal_id
        actor = actor or ""
        now = _iso_now(request.now if request.now is not None else self._clock())
        try:
            if (
                isinstance(request.new_frame_exclusive, int)
                and not isinstance(request.new_frame_exclusive, bool)
                and 1 <= request.boundary_index < len(state.timeline.segments)
                and request.new_frame_exclusive
                == state.timeline.segments[request.boundary_index - 1].end_frame_exclusive
            ):
                raise BoundaryError("new boundary must change the current boundary")
            _, edit = state.timeline.move_boundary(
                request.boundary_index,
                request.new_frame_exclusive,
                actor,
                request.reviewer,
                datetime.fromisoformat(now.replace("Z", "+00:00")),
            )
        except (BoundaryError, IndexError) as exc:
            raise ValueError(str(exc)) from exc
        source_hash = _sha256(state.source_path)
        pending_value = _pending_dict(edit)

        def mutate(candidate: dict[str, Any]) -> None:
            semantic = self._prepare_semantic_block(candidate, state, source_hash)
            semantic["state"] = "in_progress"
            semantic["pending_edit"] = deepcopy(pending_value)
            semantic.setdefault("working_timeline", _timeline_dict(state.timeline))

        self._mutate_report(state, request.expected_revision, mutate)
        state.pending_edit = edit
        self._active_pending_asset = asset_id
        return self._view(state)

    def begin_text_edit(self, asset_id: str, request: TextEditRequest) -> SemanticTaskView:
        if not isinstance(request, TextEditRequest):
            raise TypeError("request must be a TextEditRequest")
        self._assert_navigation(asset_id)
        state = self._load_state(asset_id)
        self._require_mutable(state, request.expected_revision, request.lease_token)
        try:
            segment = next(
                segment
                for segment in state.timeline.segments
                if segment.internal_id == request.segment_id
            )
        except StopIteration as exc:
            raise ValueError(f"unknown segment ID: {request.segment_id}") from exc
        if segment.text_cn == request.text_cn and segment.text_en == request.text_en:
            raise ValueError("text edit does not change either text value")
        before = SegmentSnapshot.from_segment(segment)
        after = SegmentSnapshot(
            internal_id=segment.internal_id,
            start_frame=segment.start_frame,
            end_frame_exclusive=segment.end_frame_exclusive,
            text_cn=request.text_cn,
            text_en=request.text_en,
            canonical_record=deepcopy(segment.canonical_record),
        )
        edit = TextEdit(
            segment_id=segment.internal_id,
            before=(before,),
            after=(after,),
            reviewer=request.reviewer,
            created_at=_iso_now(request.now if request.now is not None else self._clock()),
        )
        pending_value = _pending_dict(edit)
        source_hash = _sha256(state.source_path)

        def mutate(candidate: dict[str, Any]) -> None:
            semantic = self._prepare_semantic_block(candidate, state, source_hash)
            semantic["state"] = "in_progress"
            semantic["pending_edit"] = deepcopy(pending_value)
            semantic.setdefault("working_timeline", _timeline_dict(state.timeline))

        self._mutate_report(state, request.expected_revision, mutate)
        state.pending_edit = edit
        self._active_pending_asset = asset_id
        return self._view(state)

    def _apply_pending(self, timeline: SharedBoundaryTimeline, pending: BoundaryEdit | TextEdit) -> SharedBoundaryTimeline:
        by_id = {segment.internal_id: segment for segment in timeline.segments}
        if isinstance(pending, BoundaryEdit):
            if len(pending.before) != 2 or len(pending.after) != 2:
                raise TaskStateError("boundary pending edit must contain two snapshots")
            left_id, right_id = pending.affected_segment_ids
            if left_id not in by_id or right_id not in by_id:
                raise TaskStateError("pending boundary references an unknown segment")
            current_left, current_right = by_id[left_id], by_id[right_id]
            if (
                current_left.start_frame != pending.before[0].start_frame
                or current_left.end_frame_exclusive != pending.before[0].end_frame_exclusive
                or current_right.start_frame != pending.before[1].start_frame
                or current_right.end_frame_exclusive != pending.before[1].end_frame_exclusive
            ):
                raise TaskStateError("pending boundary is stale relative to the working timeline")
            updated = list(timeline.segments)
            for segment_id, snapshot in zip(pending.affected_segment_ids, pending.after):
                index = next(index for index, item in enumerate(updated) if item.internal_id == segment_id)
                current = updated[index]
                updated[index] = replace(
                    current,
                    start_frame=snapshot.start_frame,
                    end_frame_exclusive=snapshot.end_frame_exclusive,
                )
            return SharedBoundaryTimeline(
                frame_count=timeline.frame_count,
                fps=timeline.fps,
                segments=tuple(updated),
                asset_id=timeline.asset_id,
            )
        if pending.segment_id not in by_id:
            raise TaskStateError("pending text edit references an unknown segment")
        current = by_id[pending.segment_id]
        snapshot = pending.before[0]
        if current.text_cn != snapshot.text_cn or current.text_en != snapshot.text_en:
            raise TaskStateError("pending text edit is stale relative to the working timeline")
        updated = [
            replace(
                segment,
                text_cn=pending.after[0].text_cn,
                text_en=pending.after[0].text_en,
            )
            if segment.internal_id == pending.segment_id
            else segment
            for segment in timeline.segments
        ]
        return SharedBoundaryTimeline(
            frame_count=timeline.frame_count,
            fps=timeline.fps,
            segments=tuple(updated),
            asset_id=timeline.asset_id,
        )

    def confirm_pending(self, asset_id: str, expected_revision: int, lease_token: str) -> SemanticTaskView:
        self._assert_navigation(asset_id)
        state = self._load_state(asset_id)
        _require_semantic_pipeline_cursor(state.report)
        self._check_lease(state, lease_token)
        if state.revision != expected_revision:
            raise StaleSemanticRevisionError(
                f"expected revision {expected_revision}, found {state.revision}"
            )
        pending = state.pending_edit
        if pending is None:
            raise PendingEditError("there is no pending edit to confirm")
        updated_timeline = self._apply_pending(state.timeline, pending)
        source_hash = _sha256(state.source_path)
        audit_entry = _pending_dict(pending, action="confirmed")
        semantic_before = state.report.get("semantic_calibration")
        old_count = int(semantic_before.get("timeline_edit_count", 0)) if isinstance(semantic_before, Mapping) else 0
        old_text_count = int(semantic_before.get("subtask_text_edit_count", 0)) if isinstance(semantic_before, Mapping) else 0

        def mutate(candidate: dict[str, Any]) -> None:
            semantic = self._prepare_semantic_block(candidate, state, source_hash)
            audit = semantic.setdefault("audit", [])
            if not isinstance(audit, list):
                raise TaskStateError("semantic audit must be a list")
            audit.append(audit_entry)
            semantic["pending_edit"] = None
            semantic["state"] = "in_progress"
            semantic["working_timeline"] = _timeline_dict(updated_timeline)
            if isinstance(pending, BoundaryEdit):
                semantic["timeline_edit_count"] = old_count + 1
            else:
                semantic["subtask_text_edit_count"] = old_text_count + 1

        self._mutate_report(state, expected_revision, mutate)
        state.timeline = updated_timeline
        state.pending_edit = None
        self._active_pending_asset = None
        return self._view(state)

    def cancel_pending(self, asset_id: str, expected_revision: int, lease_token: str) -> SemanticTaskView:
        self._assert_navigation(asset_id)
        state = self._load_state(asset_id)
        _require_semantic_pipeline_cursor(state.report)
        self._check_lease(state, lease_token)
        if state.revision != expected_revision:
            raise StaleSemanticRevisionError(
                f"expected revision {expected_revision}, found {state.revision}"
            )
        pending = state.pending_edit
        if pending is None:
            raise PendingEditError("there is no pending edit to cancel")
        source_hash = _sha256(state.source_path)
        audit_entry = _pending_dict(pending, action="cancelled")

        def mutate(candidate: dict[str, Any]) -> None:
            semantic = self._prepare_semantic_block(candidate, state, source_hash)
            audit = semantic.setdefault("audit", [])
            if not isinstance(audit, list):
                raise TaskStateError("semantic audit must be a list")
            audit.append(audit_entry)
            semantic["pending_edit"] = None
            semantic["state"] = "in_progress"
            semantic["working_timeline"] = _timeline_dict(state.timeline)

        self._mutate_report(state, expected_revision, mutate)
        state.pending_edit = None
        self._active_pending_asset = None
        return self._view(state)

    def complete(
        self,
        asset_id: str,
        expected_revision: int,
        lease_token: str,
        *,
        advance_pipeline: bool = True,
    ) -> SemanticTaskView:
        self._assert_navigation(asset_id)
        state = self._load_state(asset_id)
        _require_semantic_task_access(state.report)
        self._check_lease(state, lease_token)
        if state.revision != expected_revision:
            raise StaleSemanticRevisionError(
                f"expected revision {expected_revision}, found {state.revision}"
            )
        if state.pending_edit is not None:
            raise PendingEditError("a pending edit must be confirmed or cancelled first")
        semantic = state.report.get("semantic_calibration")
        current_state = semantic.get("state") if isinstance(semantic, Mapping) else "not_started"
        if current_state == "completed":
            return self._view(state)
        if current_state == "finalizing":
            self._recover_finalizing(
                asset_id,
                state.source_path,
                state.report_path,
                state.loaded,
                state.report,
            )
            return self._view(self._load_state(asset_id, recover=False))
        if current_state in {"error", "skipped_due_to_fail"}:
            raise TaskStateError(f"semantic task cannot complete from state {current_state}")
        # Reconstructing SharedBoundaryTimeline above validates frame coverage,
        # positivity, and all shared-boundary invariants before any bytes stage.
        state.loaded = self._adapter.load(state.source_path)
        source_hash = _sha256(state.source_path)
        if isinstance(semantic, Mapping) and semantic.get("base_hdf5_sha256") not in (None, source_hash):
            raise TaskStateError("source HDF5 changed since semantic task initialization")
        payload = encode_canonical_payload(state.loaded, state.timeline)
        transaction_id = f"semantic-{asset_id}-{expected_revision}-{uuid.uuid4().hex}"
        prepared: PreparedReplacement
        try:
            prepared = prepare_hdf5_replacement(
                state.source_path,
                self._dataset_path,
                payload,
                transaction_id,
            )
        except Exception:
            # Preparation is deliberately before any report write; the report
            # remains in-progress and source bytes remain untouched.
            raise
        record = finalizing_record_from_prepared(prepared)
        finalizing_payload = self._record_dict(record)

        def mark_finalizing(candidate: dict[str, Any]) -> None:
            block = self._prepare_semantic_block(candidate, state, record.old_sha256)
            block["state"] = "finalizing"
            block["pending_edit"] = None
            block["transaction_id"] = record.transaction_id
            block["old_hdf5_sha256"] = record.old_sha256
            block["new_hdf5_sha256"] = record.new_sha256
            block["staged_path"] = str(record.staged_path)
            block["staged_identity"] = list(record.staged_identity or ())
            block["finalizing_record"] = finalizing_payload
            block["working_timeline"] = _timeline_dict(state.timeline)
            block["orchestrator_resume_required"] = not advance_pipeline

        try:
            self._mutate_report(state, expected_revision, mark_finalizing)
        except Exception:
            self._unlink_prepared(prepared)
            raise

        # If replacement succeeds but this process dies (or the final report
        # write fails), ``get_task`` observes finalizing and deterministically
        # recovers from the durable record.
        commit_hdf5_replacement(prepared)

        def mark_completed(candidate: dict[str, Any]) -> None:
            block = self._prepare_semantic_block(candidate, state, record.old_sha256)
            block["state"] = "completed"
            block["pending_edit"] = None
            block["final_hdf5_sha256"] = record.new_sha256
            block["completed_at"] = _iso_now(self._clock())
            block["staged_path"] = None
            block["staged_identity"] = None
            block["finalizing_record"] = finalizing_payload
            module = candidate.get("semantic_consistency")
            if not isinstance(module, dict):
                module = {}
            module["state"] = "completed"
            module["execution_kind"] = "external"
            candidate["semantic_consistency"] = module
            if advance_pipeline:
                _advance_pipeline_after_semantic(candidate)

        self._mutate_report(state, state.revision, mark_completed)
        state.pending_edit = None
        state.report = load_asset_qc_report(state.report_path) if state.report_path is not None else state.report
        if state.report is None:
            raise TaskStateError("completed report disappeared")
        state.revision = int(state.report.get("report_revision", state.revision))
        state.timeline = state.timeline
        self._active_pending_asset = None
        return self._view(state)

    @staticmethod
    def _record_dict(record: FinalizingRecord) -> dict[str, Any]:
        return {
            "source_path": str(record.source_path),
            "staged_path": str(record.staged_path),
            "old_sha256": record.old_sha256,
            "new_sha256": record.new_sha256,
            "transaction_id": record.transaction_id,
            "staged_identity": list(record.staged_identity or ()),
            "dataset_path": record.dataset_path,
            "asset_id": record.asset_id,
        }

    @staticmethod
    def _record_from_dict(value: Mapping[str, Any]) -> FinalizingRecord:
        try:
            identity = value.get("staged_identity")
            staged_identity = tuple(identity) if isinstance(identity, (tuple, list)) else None
            return FinalizingRecord(
                source_path=Path(value["source_path"]),
                staged_path=Path(value["staged_path"]),
                old_sha256=str(value["old_sha256"]),
                new_sha256=str(value["new_sha256"]),
                transaction_id=str(value["transaction_id"]),
                staged_identity=staged_identity,
                dataset_path=str(value["dataset_path"]),
                asset_id=str(value["asset_id"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise Hdf5CommitError("finalizing report record is malformed") from exc

    @staticmethod
    def _unlink_prepared(prepared: PreparedReplacement) -> None:
        path = Path(prepared.staged_path)
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def _recover_finalizing(
        self,
        asset_id: str,
        source_path: Path,
        report_path: Path | None,
        loaded: LoadedSubtasks,
        report: dict[str, Any],
    ) -> None:
        semantic = report.get("semantic_calibration")
        if not isinstance(semantic, Mapping):
            raise Hdf5CommitError("finalizing semantic block is missing")
        raw_record = semantic.get("finalizing_record")
        if not isinstance(raw_record, Mapping):
            raw_record = {
                "source_path": str(source_path),
                "staged_path": semantic.get("staged_path"),
                "old_sha256": semantic.get("old_hdf5_sha256", semantic.get("base_hdf5_sha256")),
                "new_sha256": semantic.get("new_hdf5_sha256"),
                "transaction_id": semantic.get("transaction_id"),
                "staged_identity": semantic.get("staged_identity"),
                "dataset_path": semantic.get("source_dataset_path", self._dataset_path),
                "asset_id": asset_id,
            }
        record = self._record_from_dict(raw_record)
        self._validate_finalizing_record(
            record,
            source_path=source_path,
            asset_id=asset_id,
        )
        _require_semantic_pipeline_cursor(report)
        action = recover_hdf5_replacement(record)
        if action == RecoveryAction.MARK_REPORT_COMPLETED:
            self._mark_recovered_report(asset_id, report_path, report, record, _sha256(source_path))
            return
        if action == RecoveryAction.RETRY_REPLACE:
            prepared = prepared_replacement_from_record(record)
            commit_hdf5_replacement(prepared)
            self._mark_recovered_report(asset_id, report_path, report, record, _sha256(source_path))
            return
        if action == RecoveryAction.REBUILD_STAGING:
            current_loaded = self._adapter.load(source_path)
            current_semantic = report.get("semantic_calibration")
            timeline = _timeline_from_report(current_loaded, current_semantic if isinstance(current_semantic, Mapping) else {})
            payload = encode_canonical_payload(current_loaded, timeline)
            rebuilt = prepare_hdf5_replacement(source_path, self._dataset_path, payload, record.transaction_id)
            rebuilt_record = finalizing_record_from_prepared(rebuilt)
            if report_path is not None:
                revision = int(report.get("report_revision", 0))
                update_human_state(
                    report_path,
                    revision,
                    lambda candidate: candidate.setdefault("semantic_calibration", {}).update(
                        {"finalizing_record": self._record_dict(rebuilt_record), "staged_path": str(rebuilt_record.staged_path), "staged_identity": list(rebuilt_record.staged_identity or ())}
                    ),
                )
            commit_hdf5_replacement(rebuilt)
            refreshed = load_asset_qc_report(report_path) if report_path is not None else report
            self._mark_recovered_report(asset_id, report_path, refreshed or report, rebuilt_record, _sha256(source_path))
            return
        raise Hdf5CommitError(f"unable to recover finalizing semantic transaction for {asset_id}")

    def _validate_finalizing_record(
        self,
        record: FinalizingRecord,
        *,
        source_path: Path,
        asset_id: str,
    ) -> None:
        """Bind a durable recovery record to this service's task identity.

        Task 3 validates staging ownership relative to the record's source;
        this service must additionally ensure that the record itself was not
        redirected to another asset/source/dataset by a stale or tampered
        report before invoking that recovery API.
        """

        try:
            current_source = Path(source_path).resolve(strict=False)
            record_source = Path(record.source_path).resolve(strict=False)
        except OSError as exc:
            raise Hdf5CommitError("unable to canonicalize finalizing source_path") from exc
        if record_source != current_source:
            raise Hdf5CommitError(
                "finalizing record source_path does not match the current asset source_path"
            )
        if record.asset_id != asset_id:
            raise Hdf5CommitError(
                "finalizing record asset_id does not match the requested asset"
            )
        if record.dataset_path != self._dataset_path:
            raise Hdf5CommitError(
                "finalizing record dataset_path does not match the configured dataset"
            )

    def _mark_recovered_report(
        self,
        asset_id: str,
        report_path: Path | None,
        report: dict[str, Any],
        record: FinalizingRecord,
        final_hash: str,
    ) -> None:
        if report_path is None:
            return
        revision = int(report.get("report_revision", 0))

        def mutate(candidate: dict[str, Any]) -> None:
            semantic = candidate.setdefault("semantic_calibration", {})
            if not isinstance(semantic, dict):
                raise Hdf5CommitError("semantic_calibration is not an object")
            semantic["state"] = "completed"
            semantic["pending_edit"] = None
            semantic["final_hdf5_sha256"] = final_hash
            semantic["staged_path"] = None
            semantic["staged_identity"] = None
            semantic["finalizing_record"] = self._record_dict(record)
            module = candidate.setdefault("semantic_consistency", {})
            if not isinstance(module, dict):
                module = {}
                candidate["semantic_consistency"] = module
            module["state"] = "completed"
            module["execution_kind"] = "external"
            pipeline = candidate.setdefault("pipeline_state", {})
            if not isinstance(pipeline, dict):
                pipeline = {}
                candidate["pipeline_state"] = pipeline
            if not semantic.get("orchestrator_resume_required", False):
                _advance_pipeline_after_semantic(candidate)

        update_human_state(report_path, revision, mutate)


__all__ = [
    "BoundaryEditRequest",
    "LeaseError",
    "PendingEditError",
    "SemanticCalibrationService",
    "SemanticServiceError",
    "SemanticTaskView",
    "StaleSemanticRevisionError",
    "TaskStateError",
    "TextEdit",
    "TextEditRequest",
]
