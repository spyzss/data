"""Independent queue, lease, and pipeline facade for semantic calibration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

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
            semantic = report.get("semantic_calibration") if isinstance(report, Mapping) else None
            rows.append(
                {
                    "asset_id": asset_id,
                    "state": semantic.get("state", "not_started") if isinstance(semantic, Mapping) else "preview",
                    "report_revision": int(report.get("report_revision", 0)) if isinstance(report, Mapping) else 0,
                }
            )
        return rows

    def get_task(self, asset_id: str) -> dict[str, Any]:
        self._require_eligible(asset_id, persisted=False)
        self._recover_resume(asset_id)
        view = self.domain_service.get_task(asset_id)
        value = jsonable(view)
        revision = value.get("report_revision", value.get("revision", 0)) if isinstance(value, Mapping) else 0
        return {
            "asset_id": asset_id,
            "revision": revision,
            "report_revision": revision,
            "task_type": "semantic_calibration",
            "semantic": value,
        }

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
        self._require_eligible(asset_id, persisted=True)
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
            reviewer=str(payload.get("reviewer") or lease.reviewer),
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
            reviewer=str(payload.get("reviewer") or lease.reviewer),
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

    def _recover_resume(self, asset_id: str) -> None:
        if self.config is None:
            return
        report = self._report(asset_id)
        semantic = report.get("semantic_calibration") if isinstance(report, Mapping) else None
        pipeline = report.get("pipeline_state") if isinstance(report, Mapping) else None
        if (
            isinstance(pipeline, Mapping)
            and pipeline.get("status") == "running"
            and self._recover_running_transition(asset_id, report, pipeline)
        ):
            return
        if (
            isinstance(semantic, Mapping)
            and semantic.get("state") == "completed"
            and semantic.get("orchestrator_resume_required", False)
            and isinstance(pipeline, Mapping)
            and pipeline.get("status") == "awaiting_external"
            and pipeline.get("next_module") == "semantic_consistency"
        ):
            self._resume(asset_id)

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


__all__ = ["SemanticCalibrationApplication", "jsonable"]
