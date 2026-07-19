"""Application facade for the human semantic/warn review workbench.

The semantic and warning services deliberately own their respective state
machines.  This module only composes their read projections, coordinates a
short-lived reviewer lease, and translates workbench actions into the typed
service requests.  The HTTP layer can therefore stay transport-only and
non-authoritative.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from qc_common.config import LoadedQcConfig
from qc_common.module_registry import ModuleRegistry
from qc_common.report import load_asset_qc_report
from qc_pipeline.context import AssetContext
from qc_pipeline.default_registry import build_default_registry
from qc_pipeline.orchestrator import resume_after_external, run_asset

from .evidence import EvidenceService
from .lease import Lease, LeaseStore
from .semantic_service import (
    BoundaryEditRequest,
    SemanticCalibrationService,
    TextEditRequest,
)
from .warn_service import WarnReviewService


def jsonable(value: Any) -> Any:
    """Convert service projections to JSON-compatible values.

    ``dataclasses.asdict`` is used only for the immutable view objects.  The
    recursive conversion also handles tuple-valued timeline fields, Paths and
    datetime values without leaking implementation objects through the API.
    """

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
    # A few integrations pass lightweight value objects instead of the
    # built-in view dataclasses.  Prefer an explicit ``to_dict`` when present,
    # then expose public attributes only as a last resort.
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return jsonable(to_dict())
    if hasattr(value, "__dict__"):
        return jsonable(
            {
                key: item
                for key, item in vars(value).items()
                if not key.startswith("_")
            }
        )
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


class WorkbenchService:
    """Compose semantic calibration and warning review for one asset queue.

    ``semantic_service`` and ``warn_service`` are intentionally duck-typed in
    read paths so tests and downstream adapters can provide a compatible
    facade.  The concrete services from this package are used by default in
    the mutation helpers below.
    """

    def __init__(
        self,
        semantic_service: SemanticCalibrationService | Any | None = None,
        warn_service: WarnReviewService | Any | None = None,
        evidence_service: EvidenceService | Any | None = None,
        *,
        lease_store: LeaseStore | None = None,
        asset_contexts: Mapping[str, AssetContext] | None = None,
        context_provider: Callable[[str], AssetContext | None] | None = None,
        lease_ttl_seconds: int = 900,
        profile: str | None = None,
        config: LoadedQcConfig | None = None,
        registry_factory: Callable[
            [AssetContext, LoadedQcConfig], ModuleRegistry
        ] | None = None,
    ) -> None:
        self.semantic_service = semantic_service
        self.warn_service = warn_service
        self.evidence_service = evidence_service
        self.lease_store = lease_store or LeaseStore()
        self.asset_contexts = dict(asset_contexts or {})
        self.context_provider = context_provider
        if isinstance(lease_ttl_seconds, bool) or not isinstance(lease_ttl_seconds, int):
            raise ValueError("lease_ttl_seconds must be an integer")
        if lease_ttl_seconds <= 0:
            raise ValueError("lease_ttl_seconds must be positive")
        self.lease_ttl_seconds = lease_ttl_seconds
        self.profile = profile
        self.config = config
        self.registry_factory = registry_factory or build_default_registry

    def _context(self, asset_id: str) -> AssetContext | None:
        context = self.asset_contexts.get(asset_id)
        if context is not None:
            return context
        if self.context_provider is not None:
            return self.context_provider(asset_id)
        return None

    @staticmethod
    def _get_task(service: Any, asset_id: str) -> Any | None:
        if service is None:
            return None
        getter = getattr(service, "get_task", None)
        if not callable(getter):
            getter = getattr(service, "get_asset_task", None)
        if not callable(getter):
            return None
        try:
            return getter(asset_id)
        except (KeyError, FileNotFoundError):
            return None

    def _report(self, asset_id: str, context: AssetContext | None) -> Mapping[str, Any] | None:
        report_path: Path | None = context.report_path if context is not None else None
        if report_path is None:
            for service in (self.warn_service, self.semantic_service):
                resolver = getattr(service, "report_path", None) if service is not None else None
                if callable(resolver):
                    try:
                        report_path = Path(resolver(asset_id))
                        break
                    except (KeyError, FileNotFoundError):
                        continue
        if report_path is None or not report_path.is_file():
            return None
        report = load_asset_qc_report(report_path)
        if report is not None and report.get("asset_id", asset_id) != asset_id:
            raise ValueError("asset report identity does not match requested asset")
        return report

    @staticmethod
    def _is_actionable_report(report: Mapping[str, Any] | None) -> bool:
        if not isinstance(report, Mapping):
            return False
        pipeline = report.get("pipeline_state")
        if not isinstance(pipeline, Mapping):
            return False
        if pipeline.get("status") != "awaiting_external":
            return False
        next_module = pipeline.get("next_module")
        if next_module == "semantic_consistency":
            return True
        if next_module != "manual_review":
            return False
        manual = report.get("manual_review")
        selected = (
            manual.get("selected_issue_ids", [])
            if isinstance(manual, Mapping)
            else []
        )
        return isinstance(selected, (list, tuple)) and any(
            isinstance(item, str) and item for item in selected
        )

    def _evidence(self, asset_id: str, report: Mapping[str, Any] | None, context: AssetContext | None) -> list[dict[str, Any]]:
        service = self.evidence_service
        if service is None or report is None or context is None:
            return []
        issues = report.get("issues", [])
        if not isinstance(issues, (list, tuple)):
            return []
        manual = report.get("manual_review")
        selected = manual.get("selected_issue_ids", []) if isinstance(manual, Mapping) else []
        if not isinstance(selected, (list, tuple)):
            selected = []
        selected_set = {item for item in selected if isinstance(item, str)}
        result: list[dict[str, Any]] = []
        resolver = getattr(service, "resolve", None)
        if not callable(resolver):
            return result
        evidence_by_id: dict[str, Mapping[str, Any]] = {}
        for block in report.values():
            if not isinstance(block, Mapping):
                continue
            rows = block.get("evidence")
            if not isinstance(rows, (list, tuple)):
                continue
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                evidence_id = row.get("evidence_id")
                if isinstance(evidence_id, str) and evidence_id:
                    evidence_by_id[evidence_id] = row
        for issue in issues:
            if not isinstance(issue, Mapping) or issue.get("issue_id") not in selected_set:
                continue
            resolved_issue = deepcopy(dict(issue))
            evidence_ids = issue.get("evidence_ids", [])
            if isinstance(evidence_ids, (list, tuple)) and evidence_ids:
                resolved_issue["evidence"] = [
                    deepcopy(dict(evidence_by_id[evidence_id]))
                    for evidence_id in evidence_ids
                    if isinstance(evidence_id, str) and evidence_id in evidence_by_id
                ]
            try:
                view = resolver(resolved_issue, context)
            except Exception as exc:
                # Evidence is optional in the task DTO.  Preserve the issue
                # identifier and a stable error so the UI can show a retry.
                result.append({"issue_id": issue.get("issue_id"), "generation_error": str(exc)})
                continue
            result.append(jsonable(view))
        return result

    def get_asset_task(self, asset_id: str) -> dict[str, Any]:
        """Return the latest non-mutating task projection for ``asset_id``."""

        if not isinstance(asset_id, str) or not asset_id.strip():
            raise ValueError("asset_id must be a non-empty string")
        context = self._context(asset_id)
        report = self._report(asset_id, context)
        if self._recover_external_resume(asset_id, report):
            report = self._report(asset_id, context)
        pipeline = report.get("pipeline_state") if isinstance(report, Mapping) else None
        pipeline_status = pipeline.get("status") if isinstance(pipeline, Mapping) else None
        pipeline_next = pipeline.get("next_module") if isinstance(pipeline, Mapping) else None
        # A persisted report is authoritative over stale domain projections.
        # Only a live external stage with editable selected work may expose a
        # semantic/warn task; every other state is navigation-only.
        report_is_noneditable = isinstance(report, Mapping) and not self._is_actionable_report(
            report
        )
        if pipeline_status in {"stopped", "completed", "error"} or report_is_noneditable:
            semantic = None
            warn = None
        else:
            semantic = self._get_task(self.semantic_service, asset_id)
            warn = self._get_task(self.warn_service, asset_id)
        if semantic is None and warn is None and report is None and context is None:
            raise KeyError(f"unknown asset: {asset_id}")

        revisions: list[int] = []
        for value in (semantic, warn):
            if value is None:
                continue
            raw_revision = getattr(value, "report_revision", getattr(value, "revision", None))
            if isinstance(raw_revision, int):
                revisions.append(raw_revision)
        if report is not None and isinstance(report.get("report_revision"), int):
            revisions.append(int(report["report_revision"]))
        revision = max(revisions, default=0)
        semantic_data = jsonable(semantic) if semantic is not None else None
        warn_data = jsonable(warn) if warn is not None else None
        # Lease tokens are write credentials and should never be sent in the
        # read projection.  The browser receives one only from acquire/renew.
        for value in (semantic_data, warn_data):
            if isinstance(value, dict):
                value.pop("lease_token", None)
        if isinstance(semantic_data, dict) and context is not None:
            raw_hdf5 = semantic_data.get("hdf5_path")
            if isinstance(raw_hdf5, str):
                try:
                    semantic_data["hdf5_path"] = Path(raw_hdf5).resolve().relative_to(
                        context.batch_root.resolve()
                    ).as_posix()
                except ValueError:
                    # Keep a stable basename rather than exposing a host path
                    # if an adapter returns an unexpected external location.
                    semantic_data["hdf5_path"] = Path(raw_hdf5).name
        if isinstance(warn_data, dict) and report is not None:
            selected = warn_data.get("selected_issue_ids", [])
            selected_ids = set(selected) if isinstance(selected, list) else set()
            raw_issues = report.get("issues", [])
            if isinstance(raw_issues, (list, tuple)):
                # Machine issue rows are copied into the read DTO only; all
                # mutations continue to target manual_review.issue_reviews.
                selected_rows = [
                    jsonable(issue)
                    for issue in raw_issues
                    if isinstance(issue, Mapping) and issue.get("issue_id") in selected_ids
                ]
                warn_data["issues"] = selected_rows
                warn_data["selected_issues"] = {
                    str(issue["issue_id"]): issue
                    for issue in selected_rows
                    if isinstance(issue, Mapping) and isinstance(issue.get("issue_id"), str)
                }

        task_type = "semantic_calibration" if semantic is not None else None
        if semantic is not None:
            state = (
                semantic_data.get("report_state", semantic_data.get("state"))
                if isinstance(semantic_data, dict)
                else None
            )
            if state == "completed":
                selected = report.get("manual_review", {}) if isinstance(report, Mapping) else {}
                selected_ids = selected.get("selected_issue_ids", []) if isinstance(selected, Mapping) else []
                warn_state = warn_data.get("state") if isinstance(warn_data, dict) else None
                warn_terminal = warn_state in {"completed", "not_required", "skipped_due_to_fail"}
                task_type = "completed" if warn_terminal or (warn is None and not selected_ids) else "warn_review"
            elif state in {"skipped_due_to_fail", "error"}:
                task_type = "completed"
        elif warn is not None:
            task_type = "warn_review"
        elif report is not None:
            task_type = "error" if pipeline_status == "error" else "completed"

        execution = report.get("execution") if isinstance(report, Mapping) else None
        report_profile = execution.get("profile") if isinstance(execution, Mapping) else None

        return {
            "asset_id": asset_id,
            "revision": revision,
            "report_revision": revision,
            "profile": report_profile if isinstance(report_profile, str) else self.profile,
            "task_type": task_type,
            "pipeline_state": pipeline_status,
            "next_module": pipeline_next,
            "semantic": semantic_data,
            "warn": warn_data,
            "evidence": self._evidence(asset_id, report, context),
        }

    def current_revision(self, asset_id: str) -> int | None:
        try:
            task = self.get_asset_task(asset_id)
        except (KeyError, FileNotFoundError, ValueError):
            return None
        value = task.get("revision")
        return value if isinstance(value, int) else None

    def list_actionable_assets(self, profile: str) -> tuple[str, ...]:
        """Return assets paused at an editable human external stage."""

        if not isinstance(profile, str) or not profile:
            raise ValueError("profile must be a non-empty string")
        asset_ids = set(self.asset_contexts)
        for service in (self.semantic_service, self.warn_service):
            reports = getattr(service, "_reports", None) if service is not None else None
            if isinstance(reports, Mapping):
                asset_ids.update(str(asset_id) for asset_id in reports)
        actionable: list[str] = []
        for asset_id in sorted(asset_ids):
            context = self._context(asset_id)
            report = self._report(asset_id, context)
            if not isinstance(report, Mapping):
                continue
            execution = report.get("execution")
            if isinstance(execution, Mapping) and execution.get("profile") not in {None, profile}:
                continue
            pipeline = report.get("pipeline_state")
            if not isinstance(pipeline, Mapping):
                continue
            if self._is_actionable_report(report):
                actionable.append(asset_id)
        return tuple(actionable)

    def acquire_lease(
        self, asset_id: str, reviewer: str, ttl_seconds: int | None = None
    ) -> Lease:
        context = self._context(asset_id)
        report = self._report(asset_id, context)
        if not self._is_actionable_report(report):
            raise KeyError(f"asset is not actionable: {asset_id}")
        task = self.get_asset_task(asset_id)
        if task.get("task_type") in {"completed", "error"}:
            raise KeyError(f"asset is not actionable: {asset_id}")
        lease = self.lease_store.acquire(
            asset_id,
            reviewer,
            self.lease_ttl_seconds if ttl_seconds is None else ttl_seconds,
        )
        self._bind_domain_lease(asset_id, lease.token)
        return lease

    def renew_lease(
        self, asset_id: str, token: str, ttl_seconds: int | None = None
    ) -> Lease:
        context = self._context(asset_id)
        report = self._report(asset_id, context)
        if not self._is_actionable_report(report):
            raise KeyError(f"asset is not actionable: {asset_id}")
        lease = self.lease_store.renew(
            asset_id,
            token,
            self.lease_ttl_seconds if ttl_seconds is None else ttl_seconds,
        )
        self._bind_domain_lease(asset_id, lease.token)
        return lease

    def _bind_domain_lease(self, asset_id: str, token: str) -> None:
        """Keep concrete domain adapters aligned after lease rotation.

        The domain services intentionally support an in-process injected lease
        for direct use.  The workbench owns the authoritative expiring lease,
        so a replacement reviewer must replace that injected value as well;
        otherwise a token acquired after expiry would be rejected by a stale
        service-local binding.
        """

        for service in (self.semantic_service, self.warn_service):
            leases = getattr(service, "_leases", None) if service is not None else None
            if isinstance(leases, dict):
                leases[asset_id] = token

    def validate_lease(self, asset_id: str, lease_token: str) -> Lease:
        return self.lease_store.validate(asset_id, lease_token)

    def _prepare_mutation(self, asset_id: str, lease_token: str) -> Lease:
        return self.validate_lease(asset_id, lease_token)

    def _latest(self, asset_id: str) -> dict[str, Any]:
        return self.get_asset_task(asset_id)

    def _resume_external(self, asset_id: str, completed_module: str) -> None:
        """Continue the configured pipeline after a domain completion write."""

        if self.config is None:
            return
        context = self._context(asset_id)
        if context is None:
            raise KeyError(f"asset context is not configured for {asset_id}")
        report = self._report(asset_id, context)
        if not isinstance(report, Mapping):
            raise FileNotFoundError(context.report_path)
        execution = report.get("execution")
        report_profile = (
            execution.get("profile") if isinstance(execution, Mapping) else None
        )
        profile = (
            report_profile
            if isinstance(report_profile, str)
            else self.profile or self.config.default_profile
        )
        registry = self.registry_factory(context, self.config)
        resume_after_external(
            context,
            config=self.config,
            profile=profile,
            completed_module=completed_module,
            expected_revision=int(report.get("report_revision", 0)),
            registry=registry,
        )

    def _recover_external_resume(
        self,
        asset_id: str,
        report: Mapping[str, Any] | None,
    ) -> bool:
        """Consume a durable domain-completion marker at the exact cursor."""

        if self.config is None or not isinstance(report, Mapping):
            return False
        pipeline = report.get("pipeline_state")
        if not isinstance(pipeline, Mapping):
            return False
        if pipeline.get("status") == "running":
            handoff = pipeline.get("external_resume")
            completed_module = (
                handoff.get("completed_module") if isinstance(handoff, Mapping) else None
            )
            transition_revision = (
                handoff.get("transition_revision") if isinstance(handoff, Mapping) else None
            )
            next_module = pipeline.get("next_module")
            if completed_module not in {"semantic_consistency", "manual_review"}:
                return False
            if (
                not isinstance(transition_revision, int)
                or isinstance(transition_revision, bool)
                or transition_revision > int(report.get("report_revision", 0))
                or not isinstance(next_module, str)
            ):
                return False
            if completed_module == "semantic_consistency":
                domain = report.get("semantic_calibration")
                domain_completed = (
                    isinstance(domain, Mapping) and domain.get("state") == "completed"
                )
            else:
                domain = report.get("manual_review")
                domain_completed = isinstance(domain, Mapping) and domain.get("state") in {
                    "completed",
                    "not_required",
                }
            modules = self.config.pipeline_modules
            if (
                not domain_completed
                or completed_module not in modules
                or next_module not in modules
                or modules.index(next_module) <= modules.index(completed_module)
            ):
                return False
            context = self._context(asset_id)
            if context is None:
                raise KeyError(f"asset context is not configured for {asset_id}")
            execution = report.get("execution")
            report_profile = (
                execution.get("profile") if isinstance(execution, Mapping) else None
            )
            profile = (
                report_profile
                if isinstance(report_profile, str)
                else self.profile or self.config.default_profile
            )
            registry = self.registry_factory(context, self.config)
            run_asset(
                context,
                config=self.config,
                profile=profile,
                registry=registry,
            )
            return True
        if pipeline.get("status") != "awaiting_external":
            return False
        completed_module = pipeline.get("next_module")
        if completed_module == "semantic_consistency":
            block = report.get("semantic_calibration")
            completed = isinstance(block, Mapping) and block.get("state") == "completed"
        elif completed_module == "manual_review":
            block = report.get("manual_review")
            completed = isinstance(block, Mapping) and block.get("state") in {
                "completed",
                "not_required",
            }
        else:
            return False
        if not completed or not block.get("orchestrator_resume_required", False):
            return False
        try:
            self._resume_external(asset_id, completed_module)
        except Exception:
            # A competing fetch (or a failure after the transition write) may
            # already have consumed the exact marker.  Treat that as success;
            # retry only while the same durable crash window remains.
            latest = self._report(asset_id, self._context(asset_id))
            latest_pipeline = (
                latest.get("pipeline_state") if isinstance(latest, Mapping) else None
            )
            if completed_module == "semantic_consistency":
                latest_block = (
                    latest.get("semantic_calibration")
                    if isinstance(latest, Mapping)
                    else None
                )
            else:
                latest_block = (
                    latest.get("manual_review") if isinstance(latest, Mapping) else None
                )
            still_pending = (
                isinstance(latest_pipeline, Mapping)
                and latest_pipeline.get("status") == "awaiting_external"
                and latest_pipeline.get("next_module") == completed_module
                and isinstance(latest_block, Mapping)
                and latest_block.get("orchestrator_resume_required", False)
            )
            if still_pending:
                raise
        return True

    def semantic_boundary_pending(
        self,
        asset_id: str,
        *,
        boundary_index: int,
        new_frame_exclusive: int,
        expected_revision: int,
        lease_token: str,
        actor_segment_id: str | None = None,
        reviewer: str = "human",
        now: str | datetime | None = None,
    ) -> dict[str, Any]:
        self._prepare_mutation(asset_id, lease_token)
        if self.semantic_service is None:
            raise KeyError(f"semantic service is not configured for {asset_id}")
        request = BoundaryEditRequest(
            boundary_index=boundary_index,
            new_frame_exclusive=new_frame_exclusive,
            actor_segment_id=actor_segment_id,
            expected_revision=expected_revision,
            lease_token=lease_token,
            reviewer=reviewer,
            now=now,
        )
        self.semantic_service.begin_boundary_edit(asset_id, request)
        return self._latest(asset_id)

    def semantic_text_pending(
        self,
        asset_id: str,
        *,
        segment_id: str,
        text_cn: str,
        text_en: str,
        expected_revision: int,
        lease_token: str,
        reviewer: str = "human",
        now: str | datetime | None = None,
    ) -> dict[str, Any]:
        self._prepare_mutation(asset_id, lease_token)
        if self.semantic_service is None:
            raise KeyError(f"semantic service is not configured for {asset_id}")
        request = TextEditRequest(
            segment_id=segment_id,
            text_cn=text_cn,
            text_en=text_en,
            expected_revision=expected_revision,
            lease_token=lease_token,
            reviewer=reviewer,
            now=now,
        )
        self.semantic_service.begin_text_edit(asset_id, request)
        return self._latest(asset_id)

    def semantic_pending_confirm(
        self, asset_id: str, *, expected_revision: int, lease_token: str
    ) -> dict[str, Any]:
        self._prepare_mutation(asset_id, lease_token)
        if self.semantic_service is None:
            raise KeyError(f"semantic service is not configured for {asset_id}")
        self.semantic_service.confirm_pending(asset_id, expected_revision, lease_token)
        return self._latest(asset_id)

    def semantic_pending_cancel(
        self, asset_id: str, *, expected_revision: int, lease_token: str
    ) -> dict[str, Any]:
        self._prepare_mutation(asset_id, lease_token)
        if self.semantic_service is None:
            raise KeyError(f"semantic service is not configured for {asset_id}")
        self.semantic_service.cancel_pending(asset_id, expected_revision, lease_token)
        return self._latest(asset_id)

    def semantic_complete(
        self, asset_id: str, *, expected_revision: int, lease_token: str
    ) -> dict[str, Any]:
        self._prepare_mutation(asset_id, lease_token)
        if self.semantic_service is None:
            raise KeyError(f"semantic service is not configured for {asset_id}")
        if self.config is None:
            self.semantic_service.complete(asset_id, expected_revision, lease_token)
        else:
            self.semantic_service.complete(
                asset_id,
                expected_revision,
                lease_token,
                advance_pipeline=False,
            )
            self._resume_external(asset_id, "semantic_consistency")
        return self._latest(asset_id)

    def warn_verdict(
        self,
        asset_id: str,
        *,
        issue_id: str,
        verdict: str,
        reason: str | None = None,
        expected_revision: int,
        lease_token: str,
    ) -> dict[str, Any]:
        lease = self._prepare_mutation(asset_id, lease_token)
        if self.warn_service is None:
            raise KeyError(f"warn service is not configured for {asset_id}")
        self.warn_service.submit_verdict(
            asset_id,
            issue_id,
            verdict,
            reason,
            expected_revision,
            lease_token,
            reviewer=lease.reviewer,
        )
        return self._latest(asset_id)

    def warn_complete(
        self, asset_id: str, *, expected_revision: int, lease_token: str
    ) -> dict[str, Any]:
        self._prepare_mutation(asset_id, lease_token)
        if self.warn_service is None:
            raise KeyError(f"warn service is not configured for {asset_id}")
        if self.config is None:
            self.warn_service.complete(asset_id, expected_revision, lease_token)
        else:
            self.warn_service.complete(
                asset_id,
                expected_revision,
                lease_token,
                advance_pipeline=False,
            )
            self._resume_external(asset_id, "manual_review")
        return self._latest(asset_id)


__all__ = ["WorkbenchService", "jsonable"]
