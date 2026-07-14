from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from qc_common.config import LoadedQcConfig
from qc_common.contracts import EvidenceRef, ModuleResult
from qc_common.report import (
    StaleReportRevisionError,
    load_asset_qc_report,
    write_asset_qc_report,
)
from qc_common.report_migration import migrate_v1_to_v2
from qc_pipeline.context import AssetContext


class ModuleOrderError(RuntimeError):
    pass


class ConfigDriftError(RuntimeError):
    pass


def initialize_v2_report(
    context: AssetContext,
    config: LoadedQcConfig,
    profile: str,
    now: str,
) -> dict[str, Any]:
    config.execution_profile(profile)
    modules = config.pipeline_modules
    return {
        "schema_version": "asset_qc_report.v2",
        "asset_id": context.asset_id,
        "report_revision": 0,
        "qc_config": config.json_reference(),
        "execution": {
            "profile": profile,
            "started_at": now,
            "updated_at": now,
        },
        "pipeline_state": {
            "status": "pending",
            "last_completed_module": None,
            "next_module": modules[0] if modules else None,
            "stop_reason": None,
        },
        "overall_decision": None,
        "source_files": copy.deepcopy(dict(context.source_files)),
        "issues": [],
        "runtime_errors": [],
        "manual_review": {
            "required": None,
            "state": "not_evaluated",
            "candidate_issue_ids": [],
            "failures_for_batch_stats_issue_ids": [],
        },
    }


def _assert_same_report_path(path: Path, context: AssetContext) -> None:
    if path.resolve() != context.report_path.resolve():
        raise ValueError("path must match context.report_path")


def _assert_config_reference(
    report: Mapping[str, Any],
    config: LoadedQcConfig,
    profile: str,
) -> None:
    reference = report.get("qc_config")
    if not isinstance(reference, Mapping):
        raise ConfigDriftError("asset QC report has no qc_config reference")
    try:
        config.assert_same_reference(reference)
    except ValueError as exc:
        raise ConfigDriftError(str(exc)) from exc

    execution = report.get("execution")
    if isinstance(execution, Mapping) and execution.get("profile") != profile:
        raise ConfigDriftError(
            f"QC profile drift: {execution.get('profile')} != {profile}"
        )


def _expected_next_module(
    modules: tuple[str, ...],
    module: str,
) -> str | None:
    try:
        index = modules.index(module)
    except ValueError:
        raise ModuleOrderError(f"module is not in configured pipeline: {module}") from None
    if index + 1 == len(modules):
        return None
    return modules[index + 1]


def _assert_module_order(
    report: Mapping[str, Any],
    *,
    result_module: str,
    next_module: str | None,
    modules: tuple[str, ...],
    existing: bool,
) -> None:
    configured_next = _expected_next_module(modules, result_module)
    if next_module != configured_next:
        raise ModuleOrderError(
            f"next_module for {result_module} must be {configured_next}, got {next_module}"
        )
    if not existing:
        return

    pipeline_state = report.get("pipeline_state")
    if not isinstance(pipeline_state, Mapping):
        raise ModuleOrderError("pipeline_state must be an object")
    current_next = pipeline_state.get("next_module")
    last_completed = pipeline_state.get("last_completed_module")
    if current_next == result_module:
        return

    block = report.get(result_module)
    recorded_next: Any = None
    if isinstance(block, Mapping):
        flow = block.get("flow")
        if isinstance(flow, Mapping):
            exit_gate = flow.get("exit_gate")
            if isinstance(exit_gate, Mapping):
                recorded_next = exit_gate.get("next_module")
    if last_completed == result_module and recorded_next == next_module:
        return

    raise ModuleOrderError(
        f"expected current module {current_next}, got {result_module}"
    )


def _assert_evidence_path(context: AssetContext, evidence: EvidenceRef) -> None:
    relative_path = Path(evidence.path)
    if relative_path.is_absolute():
        raise ValueError(f"evidence path must be relative to batch_root: {evidence.path}")
    resolved = (context.batch_root.resolve() / relative_path).resolve()
    try:
        resolved.relative_to(context.batch_root.resolve())
    except ValueError:
        raise ValueError(
            f"evidence path is outside batch_root: {evidence.path}"
        ) from None


def _module_block(
    result: ModuleResult,
    *,
    exit_state: str,
    continue_to_next: bool,
    next_module: str | None,
) -> dict[str, Any]:
    return {
        "flow": {
            "entry_gate": {
                "state": "ready",
                "eligible": True,
                "blocked_by_module": None,
                "required_inputs": [],
                "missing_inputs": [],
                "upstream_continue": True,
            },
            "result_gate": {
                "verdict": result.verdict,
                "has_fail": result.verdict == "fail",
                "has_warn": result.verdict == "warn",
            },
            "exit_gate": {
                "state": exit_state,
                "continue_to_next_module": continue_to_next,
                "next_module": next_module,
            },
        },
        "evaluation": copy.deepcopy(dict(result.evaluation)),
        "metrics": copy.deepcopy(dict(result.metrics)),
        "evidence": [item.to_dict() for item in result.evidence],
        "runtime": copy.deepcopy(dict(result.runtime)),
    }


def _replace_owned_issues(
    report: dict[str, Any],
    result: ModuleResult,
) -> None:
    existing_issues = report.get("issues", [])
    if not isinstance(existing_issues, list):
        raise ValueError("issues must be an array")
    preserved = [
        copy.deepcopy(issue)
        for issue in existing_issues
        if not isinstance(issue, Mapping) or issue.get("module") != result.module
    ]
    owned: list[dict[str, Any]] = []
    seen_issue_ids: set[str] = set()
    for issue in result.issues:
        if issue.module != result.module:
            raise ValueError(
                f"issue {issue.issue_id} belongs to {issue.module}, not {result.module}"
            )
        if issue.issue_id in seen_issue_ids:
            continue
        seen_issue_ids.add(issue.issue_id)
        owned.append(issue.to_dict())
    report["issues"] = preserved + owned


def _rebuild_issue_collections(report: dict[str, Any]) -> None:
    manual_review = report.get("manual_review")
    if not isinstance(manual_review, dict):
        raise ValueError("manual_review must be an object")
    manual_review["candidate_issue_ids"] = sorted(
        {
            issue["issue_id"]
            for issue in report["issues"]
            if issue["severity"] == "warn" and issue["needs_manual_review"]
        }
    )
    manual_review["failures_for_batch_stats_issue_ids"] = sorted(
        {
            issue["issue_id"]
            for issue in report["issues"]
            if issue["severity"] == "fail"
        }
    )


def _has_machine_fail(
    report: Mapping[str, Any],
    modules: tuple[str, ...],
) -> bool:
    issues = report.get("issues", [])
    if isinstance(issues, list) and any(
        isinstance(issue, Mapping) and issue.get("severity") == "fail"
        for issue in issues
    ):
        return True
    for module in modules:
        block = report.get(module)
        if not isinstance(block, Mapping):
            continue
        flow = block.get("flow")
        if not isinstance(flow, Mapping):
            continue
        result_gate = flow.get("result_gate")
        if isinstance(result_gate, Mapping) and result_gate.get("verdict") == "fail":
            return True
    return False


def apply_module_result(
    path: Path,
    *,
    context: AssetContext,
    config: LoadedQcConfig,
    profile: str,
    result: ModuleResult,
    expected_revision: int,
    next_module: str | None,
    now: str,
) -> dict[str, Any]:
    _assert_same_report_path(path, context)
    profile_config = config.execution_profile(profile)
    loaded = load_asset_qc_report(path)
    existing = loaded is not None
    if loaded is None:
        report = initialize_v2_report(context, config, profile, now)
    elif loaded.get("schema_version") == "asset_qc_report.v1":
        reference = loaded.get("qc_config")
        if not isinstance(reference, Mapping):
            raise ConfigDriftError("asset QC report has no qc_config reference")
        report = migrate_v1_to_v2(
            loaded,
            config_reference=reference,
            profile=profile,
        )
    else:
        report = copy.deepcopy(loaded)

    current_revision = int(report.get("report_revision", 0))
    if current_revision != expected_revision:
        raise StaleReportRevisionError(
            f"expected revision {expected_revision}, found {current_revision}: {path}"
        )
    if report.get("asset_id") != context.asset_id:
        raise ValueError(
            f"asset_id mismatch: {report.get('asset_id')} != {context.asset_id}"
        )
    _assert_config_reference(report, config, profile)
    _assert_module_order(
        report,
        result_module=result.module,
        next_module=next_module,
        modules=config.pipeline_modules,
        existing=existing,
    )
    for evidence in result.evidence:
        _assert_evidence_path(context, evidence)

    report.pop(result.module, None)
    _replace_owned_issues(report, result)

    hard_stop = result.verdict == "fail" and profile_config["fail_action"] == "stop"
    continue_to_next = not hard_stop
    exit_state = "continue" if continue_to_next else "stop_qc"
    report[result.module] = _module_block(
        result,
        exit_state=exit_state,
        continue_to_next=continue_to_next,
        next_module=next_module,
    )

    _rebuild_issue_collections(report)

    if hard_stop:
        pipeline_status = "stopped"
        pipeline_next = None
        stop_reason = f"quality_fail:{result.module}"
    elif next_module is None:
        pipeline_status = "completed"
        pipeline_next = None
        stop_reason = None
    else:
        pipeline_status = "running"
        pipeline_next = next_module
        stop_reason = None
    report["pipeline_state"] = {
        **copy.deepcopy(dict(report.get("pipeline_state", {}))),
        "status": pipeline_status,
        "last_completed_module": result.module,
        "next_module": pipeline_next,
        "stop_reason": stop_reason,
    }
    execution = report.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("execution must be an object")
    execution["updated_at"] = now

    if pipeline_status == "stopped":
        report["overall_decision"] = "fail"
    elif pipeline_status == "completed":
        report["overall_decision"] = (
            "fail" if _has_machine_fail(report, config.pipeline_modules) else "pass"
        )
    else:
        report["overall_decision"] = None

    report["report_revision"] = expected_revision + 1
    write_asset_qc_report(
        path,
        report,
        expected_revision=expected_revision,
        profile=profile,
    )
    return report


def write_pipeline_transition(
    path: Path,
    *,
    expected_revision: int,
    module: str,
    state: str,
    next_module: str | None,
    stop_reason: str | None,
    overall_decision: str | None,
    now: str,
) -> dict[str, Any]:
    loaded = load_asset_qc_report(path)
    if loaded is None:
        raise FileNotFoundError(path)
    current_revision = int(loaded.get("report_revision", 0))
    if current_revision != expected_revision:
        raise StaleReportRevisionError(
            f"expected revision {expected_revision}, found {current_revision}: {path}"
        )
    if loaded.get("schema_version") != "asset_qc_report.v2":
        raise ValueError("pipeline transitions require an asset_qc_report.v2 report")

    report = copy.deepcopy(loaded)
    pipeline_state = report.get("pipeline_state")
    if not isinstance(pipeline_state, dict):
        raise ValueError("pipeline_state must be an object")
    pipeline_state.update(
        {
            "status": state,
            "last_completed_module": module,
            "next_module": next_module,
            "stop_reason": stop_reason,
        }
    )
    report["overall_decision"] = overall_decision
    execution = report.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("execution must be an object")
    execution["updated_at"] = now
    if state == "error" and stop_reason:
        runtime_errors = report.get("runtime_errors")
        if not isinstance(runtime_errors, list):
            raise ValueError("runtime_errors must be an array")
        runtime_errors.append({"module": module, "message": stop_reason})

    report["report_revision"] = expected_revision + 1
    profile = execution.get("profile")
    write_asset_qc_report(
        path,
        report,
        expected_revision=expected_revision,
        profile=profile if isinstance(profile, str) else None,
    )
    return report
