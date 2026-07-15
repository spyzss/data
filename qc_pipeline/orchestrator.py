"""Config-ordered, revision-aware orchestration for one QC asset."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from qc_common.config import LoadedQcConfig
from qc_common.module_registry import (
    ModulePrerequisiteError,
    ModuleRegistry,
    ModuleUnavailableError,
)
from qc_common.report import (
    StaleReportRevisionError,
    load_asset_qc_report,
    write_asset_qc_report,
)
from qc_common.report_mutation import (
    _has_machine_fail,
    apply_module_result,
    initialize_v2_report,
    record_awaiting_external,
    record_disabled_transition,
    record_runtime_error,
    validate_report_identity,
)
from qc_pipeline.context import AssetContext
from qc_pipeline.default_registry import build_default_registry


_TERMINAL_STATUSES = frozenset({"stopped", "completed", "error"})


@dataclass(frozen=True)
class RunOutcome:
    report: dict[str, Any]
    executed_modules: tuple[str, ...]
    status: str
    config: LoadedQcConfig


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _successor(modules: tuple[str, ...], module: str) -> str | None:
    index = modules.index(module)
    return modules[index + 1] if index + 1 < len(modules) else None


def _disabled_overall_decision(report: Mapping[str, Any], completed: bool) -> str | None:
    if not completed:
        return None
    issues = report.get("issues")
    has_fail = isinstance(issues, list) and any(
        isinstance(issue, Mapping) and issue.get("severity") == "fail"
        for issue in issues
    )
    return "fail" if has_fail else "pass"


def _manual_selected_issue_ids(report: Mapping[str, Any]) -> tuple[str, ...]:
    manual = report.get("manual_review")
    if not isinstance(manual, Mapping):
        return ()
    value = manual.get("selected_issue_ids", [])
    if not isinstance(value, list):
        raise ValueError("manual_review.selected_issue_ids must be an array")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError("manual_review.selected_issue_ids must contain strings")
    return tuple(value)


def _record_empty_manual_review(
    context: AssetContext,
    *,
    config: LoadedQcConfig,
    profile: str,
    report: Mapping[str, Any],
    expected_revision: int,
    now: str,
) -> dict[str, Any]:
    """Skip a candidate-free manual stage without exposing a human task."""

    candidate = copy.deepcopy(dict(report))
    manual = candidate.get("manual_review")
    if not isinstance(manual, dict):
        manual = {}
        candidate["manual_review"] = manual
    candidate_ids = manual.get("candidate_issue_ids", [])
    if not isinstance(candidate_ids, list):
        raise ValueError("manual_review.candidate_issue_ids must be an array")
    manual.update(
        {
            "required": False,
            "state": "not_required",
            "candidate_issue_ids": candidate_ids,
            "selected_issue_ids": [],
            "selected_issue_id": None,
            "issue_reviews": {},
            "completed_at": None,
        }
    )
    manual.setdefault("failures_for_batch_stats_issue_ids", [])

    execution = candidate.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("execution must be an object")
    module_states = execution.setdefault("module_states", {})
    if not isinstance(module_states, dict):
        raise ValueError("execution.module_states must be an object")
    module_states["manual_review"] = {
        "state": "skipped",
        "reason": "no_selected_issues",
    }
    execution["updated_at"] = now

    next_module = _successor(config.pipeline_modules, "manual_review")
    pipeline = candidate.get("pipeline_state")
    if not isinstance(pipeline, dict):
        raise ValueError("pipeline_state must be an object")
    pipeline.update(
        {
            "status": "running" if next_module is not None else "completed",
            "last_completed_module": "manual_review",
            "next_module": next_module,
            "stop_reason": None,
        }
    )
    candidate["overall_decision"] = (
        None
        if next_module is not None
        else (
            "fail"
            if _has_machine_fail(candidate, config.pipeline_modules)
            else "pass"
        )
    )
    candidate["report_revision"] = expected_revision + 1
    write_asset_qc_report(
        context.report_path,
        candidate,
        expected_revision=expected_revision,
        profile=profile,
    )
    return candidate


def _record_stale_revision_if_current(
    context: AssetContext,
    *,
    config: LoadedQcConfig,
    profile: str,
    module: str,
    message: str,
    now: str,
) -> dict[str, Any]:
    """Record a stale write only when no other worker advanced the cursor."""
    latest = load_asset_qc_report(context.report_path)
    if latest is None:
        raise StaleReportRevisionError(message)
    validate_report_identity(
        latest,
        context=context,
        config=config,
        profile=profile,
    )
    pipeline_state = latest.get("pipeline_state")
    if not isinstance(pipeline_state, Mapping):
        raise ValueError("pipeline_state must be an object")
    if pipeline_state.get("status") == "error":
        return latest
    if (
        pipeline_state.get("next_module") != module
        or pipeline_state.get("status") not in {"pending", "running"}
    ):
        return latest
    return record_runtime_error(
        context.report_path,
        module=module,
        error_type="stale_revision",
        message=message,
        expected_revision=int(latest.get("report_revision", 0)),
        context=context,
        config=config,
        profile=profile,
        now=now,
    )


def run_asset(
    context: AssetContext,
    *,
    config: LoadedQcConfig,
    profile: str,
    registry: ModuleRegistry,
    now: Callable[[], str] = utc_now,
) -> RunOutcome:
    """Run or resume one asset strictly from its persisted next_module."""
    config.execution_profile(profile)
    loaded = load_asset_qc_report(context.report_path)
    initial_now: str | None = None
    if loaded is None:
        initial_now = now()
        report = initialize_v2_report(context, config, profile, initial_now)
    else:
        validate_report_identity(
            loaded,
            context=context,
            config=config,
            profile=profile,
        )
        report = copy.deepcopy(loaded)

    status = str(report["pipeline_state"]["status"])
    if status in _TERMINAL_STATUSES:
        return RunOutcome(report, (), status, config)
    current = report["pipeline_state"].get("next_module")
    if not isinstance(current, str) or current not in config.pipeline_modules:
        raise ValueError(f"invalid persisted next_module: {current!r}")
    start = config.pipeline_modules.index(current)
    executed: list[str] = []

    def stop_incomplete(
        module: str,
        error_type: str,
        message: str,
        expected_revision: int,
        timestamp: str,
    ) -> RunOutcome:
        error_report = record_runtime_error(
            context.report_path,
            module=module,
            error_type=error_type,
            message=message,
            expected_revision=expected_revision,
            context=context,
            config=config,
            profile=profile,
            now=timestamp,
        )
        return RunOutcome(error_report, tuple(executed), "error", config)

    for module_name in config.pipeline_modules[start:]:
        module_config = config.module_config(module_name)
        expected_revision = int(report.get("report_revision", 0))
        timestamp = initial_now if expected_revision == 0 and initial_now else now()
        if not module_config.get("enabled"):
            next_module = _successor(config.pipeline_modules, module_name)
            completed = next_module is None
            report = record_disabled_transition(
                context.report_path,
                context=context,
                config=config,
                profile=profile,
                expected_revision=expected_revision,
                module=module_name,
                next_module=next_module,
                overall_decision=_disabled_overall_decision(report, completed),
                now=timestamp,
            )
            continue
        if module_config.get("execution_kind") == "external":
            if module_name == "manual_review" and not _manual_selected_issue_ids(report):
                report = _record_empty_manual_review(
                    context,
                    config=config,
                    profile=profile,
                    report=report,
                    expected_revision=expected_revision,
                    now=timestamp,
                )
                continue
            report = record_awaiting_external(
                context.report_path,
                context=context,
                config=config,
                profile=profile,
                expected_revision=expected_revision,
                module=module_name,
                now=timestamp,
            )
            return RunOutcome(report, tuple(executed), "awaiting_external", config)

        implementation = module_config.get("implementation")
        if not isinstance(implementation, str) or not implementation:
            raise ModulePrerequisiteError(module_name, "Config implementation")
        try:
            runner = registry.resolve(implementation)
        except ModuleUnavailableError as exc:
            return stop_incomplete(
                module_name, "module_unavailable", str(exc), expected_revision, timestamp
            )
        try:
            result = runner(context, config)
        except ModulePrerequisiteError as exc:
            return stop_incomplete(
                module_name, "input_missing", str(exc), expected_revision, timestamp
            )
        except ValueError as exc:
            return stop_incomplete(
                module_name, "detector_error", str(exc), expected_revision, timestamp
            )
        except StaleReportRevisionError as exc:
            report = _record_stale_revision_if_current(
                context,
                config=config,
                profile=profile,
                module=module_name,
                message=str(exc),
                now=timestamp,
            )
            return RunOutcome(
                report,
                tuple(executed),
                str(report["pipeline_state"]["status"]),
                config,
            )
        except Exception as exc:
            return stop_incomplete(
                module_name, "process_error", str(exc), expected_revision, timestamp
            )
        try:
            report = apply_module_result(
                context.report_path,
                context=context,
                config=config,
                profile=profile,
                result=result,
                expected_revision=expected_revision,
                next_module=_successor(config.pipeline_modules, module_name),
                now=timestamp,
                mark_remaining_skipped_on_stop=True,
            )
        except StaleReportRevisionError as exc:
            report = _record_stale_revision_if_current(
                context,
                config=config,
                profile=profile,
                module=module_name,
                message=str(exc),
                now=timestamp,
            )
            return RunOutcome(
                report,
                tuple(executed),
                str(report["pipeline_state"]["status"]),
                config,
            )
        except Exception as exc:
            return stop_incomplete(
                module_name,
                "evidence_integrity_error",
                str(exc),
                expected_revision,
                timestamp,
            )
        executed.append(module_name)
        if report["pipeline_state"]["status"] in _TERMINAL_STATUSES:
            break

    final_status = str(report["pipeline_state"]["status"])
    return RunOutcome(report, tuple(executed), final_status, config)


def resume_after_external(
    context: AssetContext,
    *,
    config: LoadedQcConfig,
    profile: str,
    completed_module: str,
    expected_revision: int,
    registry: ModuleRegistry | None = None,
    now: Callable[[], str] = utc_now,
) -> RunOutcome:
    """Advance a persisted external stage and optionally run its successors.

    Human services own their domain payloads, while this helper owns only the
    orchestrator cursor.  It is intentionally revision-aware and copy-on-write
    so a stale browser cannot resume a different stage.  Supplying a registry
    continues configured downstream modules immediately; without one the
    function returns at the next persisted cursor for a caller that schedules
    detector execution separately.
    """

    config.execution_profile(profile)
    if completed_module not in config.pipeline_modules:
        raise ValueError(f"unknown external module: {completed_module}")
    module_config = config.module_config(completed_module)
    if module_config.get("execution_kind") != "external":
        raise ValueError(f"module is not external: {completed_module}")
    report = load_asset_qc_report(context.report_path)
    if report is None:
        raise FileNotFoundError(context.report_path)
    validate_report_identity(report, context=context, config=config, profile=profile)
    current_revision = int(report.get("report_revision", 0))
    if current_revision != expected_revision:
        raise StaleReportRevisionError(
            f"expected revision {expected_revision}, found {current_revision}: {context.report_path}"
        )
    pipeline = report.get("pipeline_state")
    if not isinstance(pipeline, Mapping):
        raise ValueError("pipeline_state must be an object")
    awaiting_completion = (
        pipeline.get("status") == "awaiting_external"
        and pipeline.get("next_module") == completed_module
    )
    if not awaiting_completion:
        raise ValueError(
            "external completion requires pipeline status=awaiting_external "
            f"and next_module={completed_module}"
        )

    timestamp = now()
    candidate = copy.deepcopy(report)
    block = candidate.get(completed_module)
    if not isinstance(block, dict):
        block = {}
    block["state"] = "completed"
    block["execution_kind"] = "external"
    candidate[completed_module] = block
    if completed_module == "semantic_consistency":
        semantic = candidate.get("semantic_calibration")
        if isinstance(semantic, dict):
            semantic.pop("orchestrator_resume_required", None)
    elif completed_module == "manual_review":
        manual = candidate.get("manual_review")
        if isinstance(manual, dict):
            manual.pop("orchestrator_resume_required", None)
    execution = candidate.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("execution must be an object")
    module_states = execution.setdefault("module_states", {})
    if not isinstance(module_states, dict):
        raise ValueError("execution.module_states must be an object")
    module_states[completed_module] = {"state": "completed"}
    execution["updated_at"] = timestamp

    next_module = _successor(config.pipeline_modules, completed_module)
    pipeline_state = dict(pipeline)
    pipeline_state.update(
        {
            "status": "running" if next_module is not None else "completed",
            "last_completed_module": completed_module,
            "next_module": next_module,
            "stop_reason": None,
        }
    )
    candidate["pipeline_state"] = pipeline_state
    if next_module is None:
        manual = candidate.get("manual_review")
        manual_state = manual.get("state") if isinstance(manual, Mapping) else None
        reviews = manual.get("issue_reviews", {}) if isinstance(manual, Mapping) else {}
        human_fail = isinstance(reviews, Mapping) and any(
            isinstance(review, Mapping) and review.get("verdict") == "fail"
            for review in reviews.values()
        )
        candidate["overall_decision"] = (
            "fail" if _has_machine_fail(candidate, config.pipeline_modules) or human_fail else
            (
                "pass"
                if completed_module != "manual_review"
                or manual_state in {None, "completed", "not_required"}
                else None
            )
        )
    else:
        candidate["overall_decision"] = None
    candidate["report_revision"] = expected_revision + 1
    write_asset_qc_report(
        context.report_path,
        candidate,
        expected_revision=expected_revision,
        profile=profile,
    )
    transition = RunOutcome(
        candidate,
        (completed_module,),
        str(candidate["pipeline_state"]["status"]),
        config,
    )
    if registry is None or next_module is None:
        return transition
    resumed = run_asset(
        context,
        config=config,
        profile=profile,
        registry=registry,
        now=now,
    )
    return RunOutcome(
        resumed.report,
        transition.executed_modules + resumed.executed_modules,
        resumed.status,
        config,
    )


__all__ = [
    "ModulePrerequisiteError",
    "RunOutcome",
    "build_default_registry",
    "resume_after_external",
    "run_asset",
    "utc_now",
]
