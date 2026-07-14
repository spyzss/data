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
)
from qc_common.report import load_asset_qc_report
from qc_common.report_mutation import (
    apply_module_result,
    initialize_v2_report,
    record_awaiting_external,
    validate_report_identity,
    write_pipeline_transition,
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

    for module_name in config.pipeline_modules[start:]:
        module_config = config.module_config(module_name)
        expected_revision = int(report.get("report_revision", 0))
        timestamp = initial_now if expected_revision == 0 and initial_now else now()
        if not module_config.get("enabled"):
            next_module = _successor(config.pipeline_modules, module_name)
            if expected_revision == 0:
                raise ModulePrerequisiteError(
                    module_name,
                    "a preceding committed module before a disabled transition",
                )
            completed = next_module is None
            report = write_pipeline_transition(
                context.report_path,
                expected_revision=expected_revision,
                module=module_name,
                state="completed" if completed else "running",
                next_module=next_module,
                stop_reason=None,
                overall_decision=_disabled_overall_decision(report, completed),
                now=timestamp,
            )
            continue
        if module_config.get("execution_kind") == "external":
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
        result = registry.resolve(implementation)(context, config)
        report = apply_module_result(
            context.report_path,
            context=context,
            config=config,
            profile=profile,
            result=result,
            expected_revision=expected_revision,
            next_module=_successor(config.pipeline_modules, module_name),
            now=timestamp,
        )
        executed.append(module_name)
        if report["pipeline_state"]["status"] in _TERMINAL_STATUSES:
            break

    final_status = str(report["pipeline_state"]["status"])
    return RunOutcome(report, tuple(executed), final_status, config)


__all__ = [
    "ModulePrerequisiteError",
    "RunOutcome",
    "build_default_registry",
    "run_asset",
    "utc_now",
]
