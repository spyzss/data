"""CAS-backed Source Gate reports created before Canonical QC can run."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from qc_common.config import LoadedQcConfig
from qc_common.report import load_asset_qc_report, write_asset_qc_report
from qc_common.report_mutation import initialize_v2_report, validate_report_identity
from qc_common.schema import validate_asset_qc_report
from qc_pipeline.context import AssetContext, validate_asset_id

from .config import LoadedCanonicalQcConfig
from .errors import CanonicalInputError


@dataclass(frozen=True, slots=True)
class DeclaredEpisodeIdentity:
    asset_id: str
    batch_id: str
    supplier_id: str

    def __post_init__(self) -> None:
        validate_asset_id(self.asset_id)
        for field in ("batch_id", "supplier_id"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise CanonicalInputError(
                    "field_mapping_error", field, "must be a non-empty string"
                )


@dataclass(frozen=True, slots=True)
class SourceGateLocator:
    source_path: str
    source_root: str
    source_format: str
    episode_index: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "source_path": self.source_path,
            "source_root": self.source_root,
            "source_format": self.source_format,
            "episode_index": self.episode_index,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_config_reference(
    config: LoadedCanonicalQcConfig,
) -> dict[str, str]:
    return {
        "schema_version": str(config.raw["schema_version"]),
        "config_version": config.config_version,
        "config_path": str(config.path),
        "config_hash": config.sha256,
    }


def _provisional_context(
    *,
    identity: DeclaredEpisodeIdentity,
    locator: SourceGateLocator,
    batch_root: Path,
    report_path: Path,
) -> AssetContext:
    return AssetContext(
        asset_id=identity.asset_id,
        batch_root=batch_root,
        report_path=report_path,
        source_files={
            "source_gate_input": {
                "path": locator.source_path,
                "source_root": locator.source_root,
                "source_format": locator.source_format,
                "episode_index": locator.episode_index,
            }
        },
        metadata={
            "supplier_id": identity.supplier_id,
            "batch_id": identity.batch_id,
        },
    )


def _identity_payload(identity: DeclaredEpisodeIdentity) -> dict[str, str]:
    return {
        "asset_id": identity.asset_id,
        "batch_id": identity.batch_id,
        "supplier_id": identity.supplier_id,
    }


def _issue_id(
    identity: DeclaredEpisodeIdentity,
    locator: SourceGateLocator,
    diagnostic: CanonicalInputError,
) -> str:
    payload = {
        **_identity_payload(identity),
        **locator.to_dict(),
        "code": diagnostic.code,
        "field": diagnostic.field,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return f"source_gate:contract:{hashlib.sha256(encoded).hexdigest()[:20]}"


def _source_gate_block(
    *,
    verdict: str | None,
    diagnostic: CanonicalInputError | None,
    identity: DeclaredEpisodeIdentity,
    locator: SourceGateLocator,
    canonical_config: LoadedCanonicalQcConfig,
) -> dict[str, Any]:
    if verdict == "pass":
        exit_gate = {
            "state": "continue",
            "continue_to_next_module": True,
        }
    elif verdict == "fail":
        exit_gate = {
            "state": "stop_qc",
            "continue_to_next_module": False,
            "next_module": None,
        }
    else:
        exit_gate = {
            "state": "stop_incomplete",
            "continue_to_next_module": False,
            "next_module": None,
        }
    flow: dict[str, Any] = {
        "entry_gate": {"state": "entered"},
        "exit_gate": exit_gate,
    }
    if verdict is not None:
        flow["result_gate"] = {
            "verdict": verdict,
            "has_fail": verdict == "fail",
            "has_warn": False,
        }
    evaluation: dict[str, Any] = {
        "status": verdict or "runtime_error",
        "declared_identity": _identity_payload(identity),
        "locator": locator.to_dict(),
    }
    if diagnostic is not None:
        evaluation["diagnostic"] = {
            "code": diagnostic.code,
            "field": diagnostic.field,
            "message": diagnostic.detail,
            "retryable": diagnostic.retryable,
        }
    return {
        "flow": flow,
        "evaluation": evaluation,
        "metrics": {},
        "evidence": [],
        "runtime": {
            "canonical_config": _canonical_config_reference(canonical_config),
        },
    }


def _assert_existing_source_gate(
    report: Mapping[str, Any],
    *,
    identity: DeclaredEpisodeIdentity,
    locator: SourceGateLocator,
    canonical_config: LoadedCanonicalQcConfig,
    qc_config: LoadedQcConfig,
    profile: str,
) -> None:
    validate_asset_qc_report(dict(report))
    if report.get("asset_id") != identity.asset_id:
        raise CanonicalInputError(
            "report_identity_mismatch", "asset_id", "existing report belongs to another asset"
        )
    if report.get("supplier_id") != identity.supplier_id:
        raise CanonicalInputError(
            "report_identity_mismatch",
            "supplier_id",
            "existing report belongs to another supplier",
        )
    if report.get("batch_id") != identity.batch_id:
        raise CanonicalInputError(
            "report_identity_mismatch", "batch_id", "existing report belongs to another batch"
        )
    qc_config.assert_same_reference(report.get("qc_config", {}))
    source_gate = report.get("source_gate")
    evaluation = source_gate.get("evaluation") if isinstance(source_gate, Mapping) else None
    declared_identity = (
        evaluation.get("declared_identity") if isinstance(evaluation, Mapping) else None
    )
    expected_identity = _identity_payload(identity)
    if declared_identity != expected_identity:
        raise CanonicalInputError(
            "report_identity_mismatch",
            "source_gate.evaluation.declared_identity",
            "existing report declared identity drift",
        )
    if not isinstance(evaluation, Mapping) or evaluation.get("locator") != locator.to_dict():
        raise CanonicalInputError(
            "report_identity_mismatch",
            "source_gate.evaluation.locator",
            "existing report belongs to another source locator",
        )
    runtime = source_gate.get("runtime") if isinstance(source_gate, Mapping) else None
    recorded_canonical = (
        runtime.get("canonical_config") if isinstance(runtime, Mapping) else None
    )
    expected_canonical = _canonical_config_reference(canonical_config)
    for field in ("schema_version", "config_version", "config_hash"):
        observed = (
            recorded_canonical.get(field)
            if isinstance(recorded_canonical, Mapping)
            else None
        )
        if observed != expected_canonical[field]:
            raise CanonicalInputError(
                "report_identity_mismatch",
                f"source_gate.runtime.canonical_config.{field}",
                "existing report Canonical config drift",
            )
    execution = report.get("execution")
    if not isinstance(execution, Mapping) or execution.get("profile") != profile:
        raise CanonicalInputError(
            "report_identity_mismatch", "execution.profile", "existing report profile drift"
        )


def _is_source_gate_runtime_error(report: Mapping[str, Any]) -> bool:
    source_gate = report.get("source_gate")
    evaluation = source_gate.get("evaluation") if isinstance(source_gate, Mapping) else None
    execution = report.get("execution")
    states = execution.get("module_states") if isinstance(execution, Mapping) else None
    state = states.get("source_gate") if isinstance(states, Mapping) else None
    return (
        isinstance(evaluation, Mapping)
        and evaluation.get("status") == "runtime_error"
        and isinstance(state, Mapping)
        and state.get("state") == "runtime_error"
    )


def _downstream_runtime_module(
    report: Mapping[str, Any], qc_config: LoadedQcConfig
) -> str | None:
    source_gate = report.get("source_gate")
    evaluation = source_gate.get("evaluation") if isinstance(source_gate, Mapping) else None
    execution = report.get("execution")
    states = execution.get("module_states") if isinstance(execution, Mapping) else None
    source_gate_state = states.get("source_gate") if isinstance(states, Mapping) else None
    pipeline = report.get("pipeline_state")
    next_module = pipeline.get("next_module") if isinstance(pipeline, Mapping) else None
    next_state = (
        states.get(next_module)
        if isinstance(states, Mapping) and isinstance(next_module, str)
        else None
    )
    runtime_errors = report.get("runtime_errors")
    return (
        next_module
        if isinstance(evaluation, Mapping)
        and evaluation.get("status") == "pass"
        and isinstance(source_gate_state, Mapping)
        and source_gate_state.get("state") == "completed"
        and isinstance(pipeline, Mapping)
        and pipeline.get("status") == "error"
        and isinstance(next_module, str)
        and next_module in qc_config.pipeline_modules
        and isinstance(next_state, Mapping)
        and next_state.get("state") == "runtime_error"
        and isinstance(runtime_errors, list)
        and bool(runtime_errors)
        and isinstance(runtime_errors[-1], Mapping)
        and runtime_errors[-1].get("module") == next_module
        else None
    )


def _resume_downstream_runtime_error(
    path: Path,
    report: Mapping[str, Any],
    *,
    module: str,
    profile: str,
) -> dict[str, Any]:
    resumed = copy.deepcopy(dict(report))
    revision = int(resumed["report_revision"])
    execution = resumed["execution"]
    runtime_errors = list(resumed.get("runtime_errors", []))
    prior_history = execution.get("runtime_error_history", [])
    execution["runtime_error_history"] = [
        *(list(prior_history) if isinstance(prior_history, list) else []),
        *runtime_errors,
    ]
    states = execution["module_states"]
    states.pop(module, None)
    now = _now()
    execution["updated_at"] = now
    resumed["runtime_errors"] = []
    resumed["pipeline_state"].update(
        status="running",
        next_module=module,
        stop_reason=None,
    )
    resumed["overall_decision"] = None
    resumed["report_revision"] = revision + 1
    write_asset_qc_report(
        path,
        resumed,
        expected_revision=revision,
        profile=profile,
    )
    return resumed


def record_source_gate_pass(
    path: Path,
    *,
    context: AssetContext,
    identity: DeclaredEpisodeIdentity,
    locator: SourceGateLocator,
    canonical_config: LoadedCanonicalQcConfig,
    qc_config: LoadedQcConfig,
    profile: str,
) -> dict[str, Any]:
    existing = load_asset_qc_report(path)
    if existing is not None:
        _assert_existing_source_gate(
            existing,
            identity=identity,
            locator=locator,
            canonical_config=canonical_config,
            qc_config=qc_config,
            profile=profile,
        )
        if _is_source_gate_runtime_error(existing):
            current_revision = int(existing["report_revision"])
            now = _now()
            recovered = initialize_v2_report(context, qc_config, profile, now)
            prior_execution = existing.get("execution")
            if isinstance(prior_execution, Mapping):
                recovered["execution"]["started_at"] = prior_execution.get(
                    "started_at", now
                )
            recovered["execution"]["runtime_error_history"] = list(
                existing.get("runtime_errors", [])
            )
            recovered["batch_id"] = identity.batch_id
            recovered["source_gate"] = _source_gate_block(
                verdict="pass",
                diagnostic=None,
                identity=identity,
                locator=locator,
                canonical_config=canonical_config,
            )
            recovered["execution"]["module_states"] = {
                "source_gate": {"state": "completed"}
            }
            recovered["pipeline_state"] = {
                "status": "running",
                "last_completed_module": "source_gate",
                "next_module": qc_config.pipeline_modules[0],
                "stop_reason": None,
            }
            recovered["runtime_errors"] = []
            recovered["report_revision"] = current_revision + 1
            write_asset_qc_report(
                path,
                recovered,
                expected_revision=current_revision,
                profile=profile,
            )
            return recovered
        downstream_module = _downstream_runtime_module(existing, qc_config)
        if downstream_module is not None:
            validate_report_identity(
                existing,
                context=context,
                config=qc_config,
                profile=profile,
            )
            return _resume_downstream_runtime_error(
                path,
                existing,
                module=downstream_module,
                profile=profile,
            )
        validate_report_identity(
            existing,
            context=context,
            config=qc_config,
            profile=profile,
        )
        return existing

    now = _now()
    report = initialize_v2_report(context, qc_config, profile, now)
    report["batch_id"] = identity.batch_id
    report["source_gate"] = _source_gate_block(
        verdict="pass",
        diagnostic=None,
        identity=identity,
        locator=locator,
        canonical_config=canonical_config,
    )
    report["execution"]["module_states"] = {"source_gate": {"state": "completed"}}
    report["pipeline_state"] = {
        "status": "running",
        "last_completed_module": "source_gate",
        "next_module": qc_config.pipeline_modules[0],
        "stop_reason": None,
    }
    report["report_revision"] = 1
    write_asset_qc_report(path, report, expected_revision=0, profile=profile)
    return report


def record_source_gate_failure(
    path: Path,
    *,
    identity: DeclaredEpisodeIdentity,
    locator: SourceGateLocator,
    batch_root: Path,
    diagnostic: CanonicalInputError,
    canonical_config: LoadedCanonicalQcConfig,
    qc_config: LoadedQcConfig,
    profile: str,
) -> dict[str, Any]:
    existing = load_asset_qc_report(path)
    prior_runtime_report: Mapping[str, Any] | None = None
    if existing is not None:
        _assert_existing_source_gate(
            existing,
            identity=identity,
            locator=locator,
            canonical_config=canonical_config,
            qc_config=qc_config,
            profile=profile,
        )
        block = existing.get("source_gate")
        evaluation = block.get("evaluation") if isinstance(block, Mapping) else None
        recorded = evaluation.get("diagnostic") if isinstance(evaluation, Mapping) else None
        expected = {
            "code": diagnostic.code,
            "field": diagnostic.field,
            "message": diagnostic.detail,
            "retryable": diagnostic.retryable,
        }
        same_retryable_kind = (
            diagnostic.retryable
            and isinstance(recorded, Mapping)
            and recorded.get("retryable") is True
            and recorded.get("code") == diagnostic.code
            and recorded.get("field") == diagnostic.field
        )
        if recorded == expected or same_retryable_kind:
            return existing
        if _is_source_gate_runtime_error(existing) and not diagnostic.retryable:
            prior_runtime_report = existing
        else:
            raise CanonicalInputError(
                "report_outcome_conflict",
                "source_gate.evaluation.diagnostic",
                "existing report records a different Source Gate outcome",
            )

    context = _provisional_context(
        identity=identity,
        locator=locator,
        batch_root=batch_root,
        report_path=path,
    )
    now = _now()
    report = initialize_v2_report(context, qc_config, profile, now)
    expected_revision = 0
    if prior_runtime_report is not None:
        expected_revision = int(prior_runtime_report["report_revision"])
        prior_execution = prior_runtime_report.get("execution")
        if isinstance(prior_execution, Mapping):
            report["execution"]["started_at"] = prior_execution.get("started_at", now)
        history = list(prior_runtime_report.get("runtime_errors", []))
        existing_history = (
            prior_execution.get("runtime_error_history", [])
            if isinstance(prior_execution, Mapping)
            else []
        )
        report["execution"]["runtime_error_history"] = [
            *(list(existing_history) if isinstance(existing_history, list) else []),
            *history,
        ]
    report["batch_id"] = identity.batch_id
    report["source_gate"] = _source_gate_block(
        verdict=None if diagnostic.retryable else "fail",
        diagnostic=diagnostic,
        identity=identity,
        locator=locator,
        canonical_config=canonical_config,
    )
    if diagnostic.retryable:
        report["execution"]["module_states"] = {
            "source_gate": {"state": "runtime_error", "reason": diagnostic.code}
        }
        report["pipeline_state"] = {
            "status": "error",
            "last_completed_module": None,
            "next_module": "source_gate",
            "stop_reason": diagnostic.code,
        }
        report["runtime_errors"] = [
            {
                "module": "source_gate",
                "error_type": diagnostic.code,
                "message": diagnostic.detail,
                "occurred_at": now,
                "retryable": True,
            }
        ]
    else:
        issue_id = _issue_id(identity, locator, diagnostic)
        report["execution"]["module_states"] = {
            "source_gate": {"state": "completed"},
            **{
                module: {"state": "skipped_due_to_fail"}
                for module in qc_config.pipeline_modules
            },
        }
        report["pipeline_state"] = {
            "status": "stopped",
            "last_completed_module": "source_gate",
            "next_module": None,
            "stop_reason": "quality_fail:source_gate",
        }
        report["overall_decision"] = "fail"
        report["issues"] = [
            {
                "issue_id": issue_id,
                "code": diagnostic.code,
                "severity": "fail",
                "module": "source_gate",
                "issue_type": "source_contract_failure",
                "metric": "canonical_input_contract",
                "observed_value": {
                    "code": diagnostic.code,
                    "field": diagnostic.field,
                },
                "operator": "satisfies",
                "boundary_value": "canonical_qc_episode.v1",
                "rule_id": canonical_config.source_gate_rule_id,
                "needs_manual_review": False,
                "context": {
                    **locator.to_dict(),
                    "field": diagnostic.field,
                    "retryable": False,
                },
                "evidence_ids": [],
            }
        ]
        report["manual_review"] = {
            "required": False,
            "state": "skipped_due_to_fail",
            "candidate_issue_ids": [],
            "failures_for_batch_stats_issue_ids": [issue_id],
        }
    report["report_revision"] = expected_revision + 1
    write_asset_qc_report(
        path, report, expected_revision=expected_revision, profile=profile
    )
    return report


__all__ = [
    "DeclaredEpisodeIdentity",
    "SourceGateLocator",
    "record_source_gate_failure",
    "record_source_gate_pass",
]
