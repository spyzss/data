"""Independent queue, lease, and pipeline facade for semantic calibration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from qc_common.config import LoadedQcConfig
from qc_common.manual_review import semantic_eligibility
from qc_common.module_registry import ModuleRegistry
from qc_common.report import load_asset_qc_report
from qc_common.reviewer_lease import Lease, LeaseStore
from qc_pipeline.context import AssetContext
from qc_pipeline.default_registry import build_default_registry
from qc_pipeline.orchestrator import resume_after_external, run_asset

from .service import (
    BoundaryEditRequest,
    SemanticCalibrationService,
    SemanticEligibilityError,
    TextEditRequest,
)


def jsonable(value: Any) -> Any:
    """Convert immutable semantic projections to JSON-compatible values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [jsonable(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return jsonable(to_dict())
    if hasattr(value, "__dict__"):
        return jsonable({key: item for key, item in vars(value).items() if not key.startswith("_")})
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _field(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _segment_dto(value: object) -> dict[str, Any]:
    """Project one semantic row without source-domain/private fields."""

    return {
        "internal_id": str(_field(value, "internal_id", "")),
        "start_frame": int(_field(value, "start_frame", 0)),
        "end_frame_exclusive": int(_field(value, "end_frame_exclusive", 0)),
        "text_cn": str(_field(value, "text_cn", "")),
        "text_en": str(_field(value, "text_en", "")),
    }


def _snapshot_dto(value: object) -> dict[str, Any]:
    return _segment_dto(value)


def _pending_dto(value: object | None) -> dict[str, Any] | None:
    if value is None:
        return None
    before = _field(value, "before", ())
    after = _field(value, "after", ())
    result: dict[str, Any] = {
        "edit_type": str(_field(value, "edit_type", "boundary")),
        "affected_segment_ids": [
            str(item) for item in (_field(value, "affected_segment_ids", ()) or ())
        ],
        "before": [_snapshot_dto(item) for item in (before or ())],
        "after": [_snapshot_dto(item) for item in (after or ())],
        "reviewer": str(_field(value, "reviewer", "")),
        "created_at": str(_field(value, "created_at", "")),
    }
    for name in ("boundary_id", "boundary_index", "actor_segment_id", "segment_id"):
        item = _field(value, name)
        if item is not None:
            result[name] = item
    return result


def semantic_task_dto(
    value: object,
    *,
    video_url: str | None = None,
    editable: bool = False,
) -> dict[str, Any]:
    """Return the explicit public semantic DTO allowlist.

    The domain view deliberately contains server paths, hashes, source
    records, and the currently bound lease token.  None of those values are
    browser capabilities and none cross this application boundary.
    """

    timeline = _field(value, "timeline", {})
    segments = _field(timeline, "segments", ()) or ()
    revision = int(_field(value, "report_revision", _field(value, "revision", 0)))
    semantic = {
        "report_revision": revision,
        "report_state": str(_field(value, "report_state", _field(value, "state", "not_started"))),
        "pipeline_state": _field(value, "pipeline_state"),
        "semantic_consistency_state": _field(value, "semantic_consistency_state"),
        "timeline_edit_count": int(_field(value, "timeline_edit_count", 0)),
        "subtask_text_edit_count": int(_field(value, "subtask_text_edit_count", 0)),
        "pending_edit": _pending_dto(_field(value, "pending_edit")),
        "timeline": {
            "frame_count": int(_field(timeline, "frame_count", 0)),
            "fps": float(_field(timeline, "fps", 0.0)),
            "segments": [_segment_dto(segment) for segment in segments],
        },
    }
    return {
        "asset_id": str(_field(value, "asset_id", "")),
        "revision": revision,
        "report_revision": revision,
        "task_type": "semantic_calibration",
        "editable": bool(editable),
        "video_url": video_url,
        "semantic": semantic,
    }


class SemanticCalibrationApplication:
    """Expose only eligible semantic tasks through one application boundary."""

    def __init__(
        self,
        domain_service: SemanticCalibrationService | Any,
        *,
        lease_store: LeaseStore | None = None,
        lease_ttl_seconds: int = 900,
        asset_contexts: Mapping[str, AssetContext] | None = None,
        profile: str | None = None,
        config: LoadedQcConfig | None = None,
        registry_factory: Callable[[AssetContext, LoadedQcConfig], ModuleRegistry] | None = None,
        video_paths: Mapping[str, str | Path] | None = None,
        video_roots: Mapping[str, str | Path] | None = None,
    ) -> None:
        if isinstance(lease_ttl_seconds, bool) or not isinstance(lease_ttl_seconds, int) or lease_ttl_seconds <= 0:
            raise ValueError("lease_ttl_seconds must be a positive integer")
        self.domain_service = domain_service
        self.lease_store = lease_store or LeaseStore()
        self.lease_ttl_seconds = lease_ttl_seconds
        self.asset_contexts = dict(asset_contexts or {})
        self.profile = profile
        self.config = config
        self.registry_factory = registry_factory or build_default_registry
        self._video_paths: dict[str, Path] = {}
        self._video_roots: dict[str, Path] = {}
        explicit_roots = {str(asset_id): Path(path) for asset_id, path in (video_roots or {}).items()}
        for raw_asset_id, raw_path in (video_paths or {}).items():
            asset_id = str(raw_asset_id)
            context = self.asset_contexts.get(asset_id)
            root_value = explicit_roots.get(asset_id)
            if root_value is None and context is not None:
                root_value = context.batch_root
            if root_value is None:
                raise ValueError(f"semantic video root is required for {asset_id}")
            root = root_value.resolve()
            candidate = Path(raw_path)
            unresolved = candidate if candidate.is_absolute() else root / candidate
            absolute = unresolved.absolute()
            try:
                absolute.resolve().relative_to(root)
            except ValueError as exc:
                raise ValueError("semantic video must stay inside configured video root") from exc
            self._video_paths[asset_id] = absolute
            self._video_roots[asset_id] = root
        for asset_id, context in self.asset_contexts.items():
            source = context.source_files.get("video")
            raw_path = source.get("path") if isinstance(source, Mapping) else None
            if isinstance(raw_path, str) and raw_path:
                root = context.batch_root.resolve()
                candidate = (context.batch_root / raw_path).absolute()
                try:
                    candidate.resolve().relative_to(root)
                except ValueError as exc:
                    raise ValueError("semantic video must stay inside batch root") from exc
                self._video_paths.setdefault(asset_id, candidate)
                self._video_roots.setdefault(asset_id, root)

    def _asset_ids(self) -> tuple[str, ...]:
        getter = getattr(self.domain_service, "asset_ids", None)
        if callable(getter):
            return tuple(sorted(str(item) for item in getter()))
        assets = getattr(self.domain_service, "_assets", None)
        if isinstance(assets, Mapping):
            return tuple(sorted(str(item) for item in assets))
        return tuple(sorted(self.asset_contexts))

    def _report_path(self, asset_id: str) -> Path:
        resolver = getattr(self.domain_service, "report_path", None)
        if not callable(resolver):
            raise KeyError(asset_id)
        return Path(resolver(asset_id))

    def _report(self, asset_id: str) -> dict[str, Any] | None:
        return load_asset_qc_report(self._report_path(asset_id))

    def _require_eligible(self, asset_id: str, *, persisted: bool) -> dict[str, Any] | None:
        report = self._report(asset_id)
        if report is None:
            if persisted:
                raise SemanticEligibilityError("semantic mutation requires a persisted report")
            return None
        result = semantic_eligibility(report)
        if result == "ready":
            if persisted:
                pipeline = report.get("pipeline_state")
                semantic = report.get("semantic_calibration")
                if (
                    not isinstance(pipeline, Mapping)
                    or pipeline.get("status") != "awaiting_external"
                    or pipeline.get("next_module") != "semantic_consistency"
                    or (isinstance(semantic, Mapping) and semantic.get("state") == "completed")
                ):
                    raise SemanticEligibilityError("semantic_not_ready")
            return report
        if result == "skipped_due_to_fail":
            raise SemanticEligibilityError("semantic task was skipped_due_to_fail")
        raise SemanticEligibilityError("semantic_not_ready")

    def _bind_lease(self, asset_id: str, token: str) -> None:
        binder = getattr(self.domain_service, "bind_lease", None)
        if callable(binder):
            binder(asset_id, token)

    def list_assets(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for asset_id in self._asset_ids():
            try:
                report = self._require_eligible(asset_id, persisted=False)
            except SemanticEligibilityError:
                continue
            if report is None or not self._is_editable_report(report):
                continue
            semantic = report.get("semantic_calibration") if isinstance(report, Mapping) else None
            rows.append(
                {
                    "asset_id": asset_id,
                    "state": semantic.get("state", "not_started") if isinstance(semantic, Mapping) else "not_started",
                    "report_revision": int(report.get("report_revision", 0)) if isinstance(report, Mapping) else 0,
                    "editable": True,
                }
            )
        return rows

    def get_task(self, asset_id: str) -> dict[str, Any]:
        report = self._require_eligible(asset_id, persisted=False)
        view = self.domain_service.get_task(asset_id)
        if self._recover_resume(asset_id):
            view = self.domain_service.get_task(asset_id)
            report = self._report(asset_id)
        editable = report is not None and self._is_editable_report(report)
        video_url = (
            f"/api/semantic/assets/{quote(asset_id, safe='')}/video"
            if report is not None and asset_id in self._video_paths
            else None
        )
        return semantic_task_dto(view, video_url=video_url, editable=editable)

    @staticmethod
    def _is_editable_report(report: Mapping[str, Any]) -> bool:
        pipeline = report.get("pipeline_state")
        semantic = report.get("semantic_calibration")
        return bool(
            isinstance(pipeline, Mapping)
            and pipeline.get("status") == "awaiting_external"
            and pipeline.get("next_module") == "semantic_consistency"
            and not (isinstance(semantic, Mapping) and semantic.get("state") == "completed")
        )

    def semantic_video_path(self, asset_id: str) -> Path:
        """Resolve the configured browser video without exposing its path."""

        if asset_id not in self._asset_ids():
            raise KeyError(asset_id)
        report = self._require_eligible(asset_id, persisted=False)
        if report is None:
            raise SemanticEligibilityError("semantic video requires a persisted report")
        try:
            configured = self._video_paths[asset_id]
            root = self._video_roots[asset_id]
        except KeyError as exc:
            raise KeyError(asset_id) from exc
        path = configured.resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise KeyError(asset_id) from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def current_revision(self, asset_id: str) -> int | None:
        try:
            report = self._report(asset_id)
        except (KeyError, FileNotFoundError):
            return None
        if report is None:
            return 0
        value = report.get("report_revision")
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else None

    def acquire_lease(self, asset_id: str, reviewer: str, ttl_seconds: int | None = None) -> Lease:
        self._require_eligible(asset_id, persisted=True)
        lease = self.lease_store.acquire(
            asset_id,
            reviewer,
            self.lease_ttl_seconds if ttl_seconds is None else ttl_seconds,
        )
        self._bind_lease(asset_id, lease.token)
        return lease

    def renew_lease(self, asset_id: str, lease_token: str, ttl_seconds: int | None = None) -> Lease:
        self._require_eligible(asset_id, persisted=True)
        lease = self.lease_store.renew(
            asset_id,
            lease_token,
            self.lease_ttl_seconds if ttl_seconds is None else ttl_seconds,
        )
        self._bind_lease(asset_id, lease.token)
        return lease

    def release_lease(self, asset_id: str, lease_token: str) -> dict[str, bool]:
        # Release is cleanup, not an edit.  Completion moves the cursor out of
        # the live eligibility gate before the browser's finally block runs.
        self._report_path(asset_id)
        self.lease_store.release(asset_id, lease_token)
        return {"released": True}

    def _lease(self, asset_id: str, lease_token: str) -> Lease:
        self._require_eligible(asset_id, persisted=True)
        lease = self.lease_store.validate(asset_id, lease_token)
        self._bind_lease(asset_id, lease.token)
        return lease

    def begin_boundary_edit(self, asset_id: str, **payload: Any) -> dict[str, Any]:
        lease = self._lease(asset_id, str(payload["lease_token"]))
        request = BoundaryEditRequest(
            boundary_index=payload.get("boundary_index"),
            new_frame_exclusive=payload.get("new_frame_exclusive"),
            actor_segment_id=payload.get("actor_segment_id"),
            expected_revision=payload.get("expected_revision"),
            lease_token=lease.token,
            reviewer=lease.reviewer,
            now=payload.get("now"),
        )
        self.domain_service.begin_boundary_edit(asset_id, request)
        return self.get_task(asset_id)

    def begin_text_edit(self, asset_id: str, **payload: Any) -> dict[str, Any]:
        lease = self._lease(asset_id, str(payload["lease_token"]))
        request = TextEditRequest(
            segment_id=payload.get("segment_id"),
            text_cn=payload.get("text_cn"),
            text_en=payload.get("text_en", ""),
            expected_revision=payload.get("expected_revision"),
            lease_token=lease.token,
            reviewer=lease.reviewer,
            now=payload.get("now"),
        )
        self.domain_service.begin_text_edit(asset_id, request)
        return self.get_task(asset_id)

    def confirm_pending(self, asset_id: str, *, expected_revision: int, lease_token: str) -> dict[str, Any]:
        lease = self._lease(asset_id, lease_token)
        self.domain_service.confirm_pending(asset_id, expected_revision, lease.token)
        return self.get_task(asset_id)

    def cancel_pending(self, asset_id: str, *, expected_revision: int, lease_token: str) -> dict[str, Any]:
        lease = self._lease(asset_id, lease_token)
        self.domain_service.cancel_pending(asset_id, expected_revision, lease.token)
        return self.get_task(asset_id)

    def complete(self, asset_id: str, *, expected_revision: int, lease_token: str) -> dict[str, Any]:
        lease = self._lease(asset_id, lease_token)
        self.domain_service.complete(
            asset_id,
            expected_revision,
            lease.token,
            advance_pipeline=self.config is None,
        )
        if self.config is not None:
            self._resume(asset_id)
        return self.get_task(asset_id)

    def mutate(self, operation: str, asset_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        methods = {
            "semantic_boundary_pending": self.begin_boundary_edit,
            "semantic_text_pending": self.begin_text_edit,
            "semantic_pending_confirm": self.confirm_pending,
            "semantic_pending_cancel": self.cancel_pending,
            "semantic_complete": self.complete,
        }
        try:
            method = methods[operation]
        except KeyError as exc:
            raise KeyError(operation) from exc
        return method(asset_id, **payload)

    def _resume(self, asset_id: str) -> None:
        if self.config is None:
            return
        context = self.asset_contexts.get(asset_id)
        if context is None:
            raise KeyError(f"asset context is not configured for {asset_id}")
        report = self._report(asset_id)
        if report is None:
            raise FileNotFoundError(context.report_path)
        execution = report.get("execution")
        report_profile = execution.get("profile") if isinstance(execution, Mapping) else None
        profile = report_profile if isinstance(report_profile, str) else self.profile or self.config.default_profile
        registry = self.registry_factory(context, self.config)
        resume_after_external(
            context,
            config=self.config,
            profile=profile,
            completed_module="semantic_consistency",
            expected_revision=int(report.get("report_revision", 0)),
            registry=registry,
        )

    def _recover_resume(self, asset_id: str) -> bool:
        if self.config is None:
            return False
        report = self._report(asset_id)
        semantic = report.get("semantic_calibration") if isinstance(report, Mapping) else None
        pipeline = report.get("pipeline_state") if isinstance(report, Mapping) else None
        if (
            isinstance(pipeline, Mapping)
            and pipeline.get("status") == "running"
            and self._recover_running_transition(asset_id, report, pipeline)
        ):
            return True
        if (
            isinstance(semantic, Mapping)
            and semantic.get("state") == "completed"
            and semantic.get("orchestrator_resume_required", False)
            and isinstance(pipeline, Mapping)
            and pipeline.get("status") == "awaiting_external"
            and pipeline.get("next_module") == "semantic_consistency"
        ):
            self._resume(asset_id)
            return True
        return False

    def _recover_running_transition(
        self,
        asset_id: str,
        report: Mapping[str, Any],
        pipeline: Mapping[str, Any],
    ) -> bool:
        """Continue only a durable semantic transition owned by this service."""

        if self.config is None:
            return False
        handoff = pipeline.get("external_resume")
        completed_module = (
            handoff.get("completed_module") if isinstance(handoff, Mapping) else None
        )
        transition_revision = (
            handoff.get("transition_revision") if isinstance(handoff, Mapping) else None
        )
        next_module = pipeline.get("next_module")
        semantic = report.get("semantic_calibration")
        modules = self.config.pipeline_modules
        if (
            completed_module != "semantic_consistency"
            or not isinstance(transition_revision, int)
            or isinstance(transition_revision, bool)
            or transition_revision > int(report.get("report_revision", 0))
            or not isinstance(next_module, str)
            or not isinstance(semantic, Mapping)
            or semantic.get("state") != "completed"
            or completed_module not in modules
            or next_module not in modules
            or modules.index(next_module) <= modules.index(completed_module)
        ):
            return False
        context = self.asset_contexts.get(asset_id)
        if context is None:
            raise KeyError(f"asset context is not configured for {asset_id}")
        execution = report.get("execution")
        report_profile = execution.get("profile") if isinstance(execution, Mapping) else None
        profile = report_profile if isinstance(report_profile, str) else self.profile or self.config.default_profile
        registry = self.registry_factory(context, self.config)
        run_asset(context, config=self.config, profile=profile, registry=registry)
        return True


__all__ = [
    "SemanticCalibrationApplication",
    "jsonable",
    "semantic_task_dto",
]
