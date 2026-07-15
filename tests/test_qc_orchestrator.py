from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from qc_common.config import LoadedQcConfig
from qc_common.contracts import EvidenceRef, Issue, ModuleResult
from qc_common.module_registry import ModuleRegistry, ModuleUnavailableError
from qc_common.report import StaleReportRevisionError, write_asset_qc_report
from qc_common.report_mutation import (
    ConfigDriftError,
    apply_module_result,
    initialize_v2_report,
    mark_remaining_skipped_due_to_fail,
    record_awaiting_external,
    record_runtime_error,
)
from qc_pipeline.context import AssetContext
from qc_pipeline.orchestrator import (
    ModulePrerequisiteError,
    build_default_registry,
    run_asset,
)


_HASH = "sha256:" + "1" * 64


def _config(
    tmp_path: Path,
    modules: list[str],
    *,
    disabled: set[str] | None = None,
) -> LoadedQcConfig:
    disabled = disabled or set()
    module_configs: dict[str, dict[str, object]] = {}
    for name in modules:
        if name in disabled:
            module_configs[name] = {
                "enabled": False,
                "disabled_reason": "not_available",
                "parameters": {},
                "rules": {},
            }
        elif name in {"semantic_consistency", "manual_review"}:
            module_configs[name] = {
                "enabled": True,
                "execution_kind": "external",
                "parameters": {},
                "rules": {},
            }
        else:
            module_configs[name] = {
                "enabled": True,
                "implementation": f"test.{name}",
                "parameters": {},
                "rules": {},
            }
    return LoadedQcConfig(
        path=tmp_path / "qc.yaml",
        raw={
            "schema_version": "qc_acceptance_config_schema.v2",
            "config_version": "qc_acceptance_v2.0.0",
            "config_name": "test",
            "execution_profiles": {
                "acceptance": {
                    "fail_action": "stop",
                    "runtime_error_action": "stop_incomplete",
                },
                "supplier_evaluation": {
                    "fail_action": "record_and_continue",
                    "runtime_error_action": "stop_incomplete",
                },
            },
            "pipeline": {
                "default_profile": "acceptance",
                "terminal_module": modules[-1] if modules else "batch_statistics",
                "modules": modules,
            },
            "modules": module_configs,
        },
        sha256=_HASH,
    )


def _context(tmp_path: Path, asset_id: str = "asset-a") -> AssetContext:
    return AssetContext(
        asset_id=asset_id,
        batch_root=tmp_path,
        report_path=tmp_path / "quality_archive" / f"{asset_id}.json",
        source_files={"video": {"path": "video/clip.mp4"}},
    )


def _registry(
    calls: list[str],
    config: LoadedQcConfig,
    verdicts: dict[str, str],
    *,
    errors: set[str] | None = None,
) -> ModuleRegistry:
    registry = ModuleRegistry()
    errors = errors or set()
    for module_name, verdict in verdicts.items():
        implementation = str(config.module_config(module_name)["implementation"])

        def run(
            context: AssetContext,
            loaded: LoadedQcConfig,
            *,
            module_name: str = module_name,
            verdict: str = verdict,
        ) -> ModuleResult:
            assert loaded is config
            calls.append(module_name)
            if module_name in errors:
                raise RuntimeError(f"failed:{module_name}")
            return ModuleResult(module_name, verdict, {"decision": verdict}, {})

        registry.register(implementation, run)
    return registry


def _run_stub_pipeline(
    tmp_path: Path,
    *,
    profile: str,
    modules: list[str],
    verdicts: dict[str, str],
) -> tuple[dict[str, Any], list[str]]:
    config = _config(tmp_path, modules)
    calls: list[str] = []
    registry = ModuleRegistry()
    for module_name, verdict in verdicts.items():
        implementation = str(config.module_config(module_name)["implementation"])

        def run(
            context: AssetContext,
            loaded: LoadedQcConfig,
            *,
            module_name: str = module_name,
            verdict: str = verdict,
        ) -> ModuleResult:
            assert loaded is config
            calls.append(module_name)
            issues = ()
            if verdict == "fail":
                issues = (
                    Issue(
                        f"{module_name}:failure:11111111111111111111",
                        "failure",
                        "fail",
                        module_name,
                        "test_failure",
                        "failure_metric",
                        1,
                        ">",
                        0,
                        f"{module_name}.failure",
                        False,
                    ),
                )
            return ModuleResult(
                module_name,
                verdict,
                {"decision": verdict},
                {},
                issues,
            )

        registry.register(implementation, run)

    outcome = run_asset(
        _context(tmp_path),
        config=config,
        profile=profile,
        registry=registry,
        now=lambda: "2026-07-14T00:00:00Z",
    )
    return outcome.report, calls


def test_registry_resolves_config_implementation_names() -> None:
    registry = ModuleRegistry()
    runner = lambda context, config: ModuleResult("quality_hand", "pass", {}, {})

    registry.register("precheck.quality_hand", runner)

    assert registry.has("precheck.quality_hand")
    assert registry.resolve("precheck.quality_hand") is runner
    assert not registry.has("quality_hand")
    with pytest.raises(ModuleUnavailableError, match="unavailable") as caught:
        registry.resolve("quality_hand")
    assert caught.value.module == "quality_hand"


def test_generic_registry_import_does_not_load_qc_pipeline() -> None:
    script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith('qc_pipeline'):
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import qc_common.module_registry
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_orchestrator_runs_in_config_order_and_retains_external_pause(
    tmp_path: Path,
) -> None:
    modules = ["hdf5_text_info", "quality_hand", "semantic_consistency"]
    config = _config(tmp_path, modules)
    calls: list[str] = []
    registry = _registry(
        calls,
        config,
        {"hdf5_text_info": "pass", "quality_hand": "pass"},
    )

    first = run_asset(
        _context(tmp_path),
        config=config,
        profile="acceptance",
        registry=registry,
        now=lambda: "2026-07-14T00:00:00Z",
    )

    assert calls == ["hdf5_text_info", "quality_hand"]
    assert first.executed_modules == ("hdf5_text_info", "quality_hand")
    assert first.report["pipeline_state"] == {
        "status": "awaiting_external",
        "last_completed_module": "quality_hand",
        "next_module": "semantic_consistency",
        "stop_reason": None,
    }
    assert first.report["report_revision"] == 3
    assert "semantic_consistency" not in first.report
    assert first.report["manual_review"]["state"] == "not_evaluated"

    calls.clear()
    second = run_asset(
        _context(tmp_path),
        config=first.config,
        profile="acceptance",
        registry=registry,
        now=lambda: "2026-07-14T01:00:00Z",
    )

    assert calls == []
    assert second.status == "awaiting_external"
    assert second.report["report_revision"] == first.report["report_revision"]
    assert second.report["execution"]["updated_at"] == "2026-07-14T00:00:00Z"


def test_module_states_distinguish_completed_skipped_and_awaiting_external(
    tmp_path: Path,
) -> None:
    modules = ["hdf5_text_info", "quality_hand", "semantic_consistency"]
    config = _config(tmp_path, modules)

    outcome = run_asset(
        _context(tmp_path),
        config=config,
        profile="acceptance",
        registry=_registry(
            [],
            config,
            {"hdf5_text_info": "pass", "quality_hand": "skipped"},
        ),
        now=lambda: "2026-07-14T00:00:00Z",
    )

    assert outcome.status == "awaiting_external"
    assert outcome.report["execution"]["module_states"] == {
        "hdf5_text_info": {"state": "completed"},
        "quality_hand": {"state": "skipped"},
        "semantic_consistency": {"state": "awaiting_external"},
    }
    assert outcome.report["quality_hand"]["flow"]["result_gate"]["verdict"] == (
        "skipped"
    )
    assert "semantic_consistency" not in outcome.report


def test_profiles_keep_machine_fail_but_change_flow(tmp_path: Path) -> None:
    modules = [
        "hdf5_text_info",
        "video_quality",
        "sam3_containment",
        "semantic_consistency",
        "manual_review",
    ]
    verdicts = {
        "hdf5_text_info": "pass",
        "video_quality": "fail",
        "sam3_containment": "pass",
    }

    acceptance, acceptance_calls = _run_stub_pipeline(
        tmp_path / "acceptance",
        profile="acceptance",
        modules=modules,
        verdicts=verdicts,
    )
    supplier, supplier_calls = _run_stub_pipeline(
        tmp_path / "supplier",
        profile="supplier_evaluation",
        modules=modules,
        verdicts=verdicts,
    )

    assert acceptance_calls == ["hdf5_text_info", "video_quality"]
    assert supplier_calls == [
        "hdf5_text_info",
        "video_quality",
        "sam3_containment",
    ]
    assert acceptance["video_quality"]["flow"]["result_gate"]["verdict"] == "fail"
    assert supplier["video_quality"]["flow"]["result_gate"]["verdict"] == "fail"
    assert acceptance["issues"] == supplier["issues"]
    failure_ids = ["video_quality:failure:11111111111111111111"]
    assert (
        acceptance["manual_review"]["failures_for_batch_stats_issue_ids"]
        == failure_ids
    )
    assert (
        supplier["manual_review"]["failures_for_batch_stats_issue_ids"]
        == failure_ids
    )
    assert acceptance["pipeline_state"]["status"] == "stopped"
    assert acceptance["overall_decision"] == "fail"
    assert acceptance["manual_review"]["state"] == "skipped_due_to_fail"
    assert acceptance["semantic_calibration"]["state"] == "skipped_due_to_fail"
    assert acceptance["semantic_calibration"]["pending_edit"] is None
    assert "sam3_containment" not in acceptance
    assert "semantic_consistency" not in acceptance
    assert acceptance["execution"]["module_states"] == {
        "hdf5_text_info": {"state": "completed"},
        "video_quality": {"state": "completed"},
        "sam3_containment": {"state": "skipped_due_to_fail"},
        "semantic_consistency": {"state": "skipped_due_to_fail"},
        "manual_review": {"state": "skipped_due_to_fail"},
    }
    assert "continued_after_fail" not in acceptance["video_quality"]["runtime"]
    assert supplier["video_quality"]["flow"]["exit_gate"]["state"] == "continue"
    assert supplier["video_quality"]["runtime"]["continued_after_fail"] is True
    assert supplier["sam3_containment"]["flow"]["result_gate"]["verdict"] == "pass"
    assert "continued_after_fail" not in supplier["sam3_containment"]["runtime"]
    assert supplier["pipeline_state"]["status"] == "awaiting_external"
    assert "semantic_consistency" not in supplier
    assert supplier["overall_decision"] is None
    assert supplier["manual_review"]["state"] == "not_evaluated"
    assert acceptance["report_revision"] == 2
    assert supplier["report_revision"] == 4
    acceptance_path = _context(tmp_path / "acceptance").report_path
    supplier_path = _context(tmp_path / "supplier").report_path
    assert json.loads(acceptance_path.read_text(encoding="utf-8")) == acceptance
    assert json.loads(supplier_path.read_text(encoding="utf-8")) == supplier


def test_fail_skip_marking_is_copy_on_write_and_config_ordered() -> None:
    original = {
        "execution": {"profile": "acceptance"},
        "manual_review": {"state": "not_evaluated"},
    }
    before = copy.deepcopy(original)

    marked = mark_remaining_skipped_due_to_fail(
        original,
        ("hdf5_text_info", "video_quality", "semantic_consistency"),
        failed_module="hdf5_text_info",
    )

    assert original == before
    assert marked["execution"]["module_states"] == {
        "video_quality": {"state": "skipped_due_to_fail"},
        "semantic_consistency": {"state": "skipped_due_to_fail"},
    }
    assert marked["manual_review"]["state"] == "skipped_due_to_fail"
    assert marked["semantic_calibration"] == {
        "state": "skipped_due_to_fail",
        "source_dataset_path": None,
        "base_hdf5_sha256": None,
        "final_hdf5_sha256": None,
        "timeline_edit_count": 0,
        "subtask_text_edit_count": 0,
        "pending_edit": None,
        "audit": [],
    }


def test_supplier_profile_preserves_prior_fail_at_automatic_completion(
    tmp_path: Path,
) -> None:
    report, calls = _run_stub_pipeline(
        tmp_path,
        profile="supplier_evaluation",
        modules=["hdf5_text_info", "sam3_containment"],
        verdicts={"hdf5_text_info": "fail", "sam3_containment": "pass"},
    )

    assert calls == ["hdf5_text_info", "sam3_containment"]
    assert report["pipeline_state"]["status"] == "completed"
    assert report["overall_decision"] == "fail"


def test_unknown_profile_is_rejected_before_runner_work(tmp_path: Path) -> None:
    config = _config(tmp_path, ["hdf5_text_info"])
    context = _context(tmp_path)
    calls: list[str] = []

    with pytest.raises(ValueError, match="unknown execution profile"):
        run_asset(
            context,
            config=config,
            profile="unexpected",
            registry=_registry(calls, config, {"hdf5_text_info": "pass"}),
        )

    assert calls == []
    assert not context.report_path.exists()


def test_fresh_asset_starts_at_first_configured_module(tmp_path: Path) -> None:
    modules = ["quality_hand", "hdf5_text_info"]
    config = _config(tmp_path, modules)
    calls: list[str] = []

    outcome = run_asset(
        _context(tmp_path),
        config=config,
        profile="acceptance",
        registry=_registry(
            calls,
            config,
            {"quality_hand": "pass", "hdf5_text_info": "pass"},
        ),
        now=lambda: "2026-07-14T00:00:00Z",
    )

    assert calls == modules
    assert outcome.status == "completed"
    assert outcome.report["report_revision"] == 2


@pytest.mark.parametrize("profile", ["acceptance", "supplier_evaluation"])
def test_runner_error_stops_incomplete_and_terminal_rerun_is_idempotent(
    tmp_path: Path,
    profile: str,
) -> None:
    modules = ["hdf5_text_info", "quality_hand", "keypoint_presence"]
    config = _config(tmp_path, modules)
    first_calls: list[str] = []

    first = run_asset(
        _context(tmp_path),
        config=config,
        profile=profile,
        registry=_registry(
            first_calls,
            config,
            {name: "pass" for name in modules},
            errors={"quality_hand"},
        ),
        now=lambda: "2026-07-14T00:00:00Z",
    )

    assert first_calls == ["hdf5_text_info", "quality_hand"]
    assert first.executed_modules == ("hdf5_text_info",)
    assert first.status == "error"
    assert first.report["overall_decision"] is None
    assert first.report["pipeline_state"]["next_module"] == "quality_hand"
    assert first.report["runtime_errors"] == [
        {
            "module": "quality_hand",
            "error_type": "process_error",
            "message": "failed:quality_hand",
            "occurred_at": "2026-07-14T00:00:00Z",
            "retryable": True,
        }
    ]
    assert first.report["execution"]["module_states"]["quality_hand"] == {
        "state": "runtime_error",
        "reason": "process_error",
    }
    assert "quality_hand" not in first.report
    second_calls: list[str] = []
    second = run_asset(
        _context(tmp_path),
        config=config,
        profile=profile,
        registry=_registry(
            second_calls,
            config,
            {name: "pass" for name in modules},
        ),
        now=lambda: "2026-07-14T00:01:00Z",
    )

    assert second_calls == []
    assert second.report == first.report
    assert second.status == "error"


def test_missing_runner_input_is_structured_without_quality_verdict(
    tmp_path: Path,
) -> None:
    module = "hdf5_text_info"
    config = _config(tmp_path, [module])
    registry = ModuleRegistry()

    def run(context: AssetContext, loaded: LoadedQcConfig) -> ModuleResult:
        raise ModulePrerequisiteError(module, "source_files.hdf5.path")

    registry.register("test.hdf5_text_info", run)

    outcome = run_asset(
        _context(tmp_path),
        config=config,
        profile="acceptance",
        registry=registry,
        now=lambda: "2026-07-14T00:00:00Z",
    )

    assert outcome.status == "error"
    assert outcome.report["overall_decision"] is None
    assert outcome.report["runtime_errors"][0]["error_type"] == "input_missing"
    assert outcome.report["runtime_errors"][0]["retryable"] is False
    assert outcome.report["pipeline_state"]["stop_reason"] == "input_missing"
    assert "hdf5_text_info" not in outcome.report


def test_evidence_integrity_error_is_structured_without_partial_result(
    tmp_path: Path,
) -> None:
    module = "hdf5_text_info"
    config = _config(tmp_path, [module])
    registry = ModuleRegistry()

    def run(context: AssetContext, loaded: LoadedQcConfig) -> ModuleResult:
        return ModuleResult(
            module,
            "pass",
            {},
            {},
            evidence=(
                EvidenceRef(
                    "bad-evidence",
                    "test",
                    "/outside-batch.json",
                    "frame_index",
                ),
            ),
        )

    registry.register("test.hdf5_text_info", run)

    outcome = run_asset(
        _context(tmp_path),
        config=config,
        profile="acceptance",
        registry=registry,
        now=lambda: "2026-07-14T00:00:00Z",
    )

    assert outcome.status == "error"
    assert outcome.report["runtime_errors"][0]["error_type"] == (
        "evidence_integrity_error"
    )
    assert outcome.report["runtime_errors"][0]["retryable"] is True
    assert outcome.report["overall_decision"] is None
    assert outcome.report["issues"] == []
    assert module not in outcome.report


def test_detector_value_error_is_structured_as_nonretryable_runtime_error(
    tmp_path: Path,
) -> None:
    module = "hdf5_text_info"
    config = _config(tmp_path, [module])
    registry = ModuleRegistry()

    def run(context: AssetContext, loaded: LoadedQcConfig) -> ModuleResult:
        raise ValueError("detector payload is malformed")

    registry.register("test.hdf5_text_info", run)

    outcome = run_asset(
        _context(tmp_path),
        config=config,
        profile="acceptance",
        registry=registry,
        now=lambda: "2026-07-14T00:00:00Z",
    )

    assert outcome.status == "error"
    assert outcome.report["runtime_errors"][0] == {
        "module": module,
        "error_type": "detector_error",
        "message": "detector payload is malformed",
        "occurred_at": "2026-07-14T00:00:00Z",
        "retryable": False,
    }
    assert outcome.report["overall_decision"] is None


def test_disabled_module_advances_without_a_fake_result(tmp_path: Path) -> None:
    modules = ["hdf5_text_info", "quality_hand", "semantic_consistency"]
    config = _config(tmp_path, modules, disabled={"quality_hand"})
    calls: list[str] = []

    outcome = run_asset(
        _context(tmp_path),
        config=config,
        profile="acceptance",
        registry=_registry(calls, config, {"hdf5_text_info": "pass"}),
        now=lambda: "2026-07-14T00:00:00Z",
    )

    assert calls == ["hdf5_text_info"]
    assert "quality_hand" not in outcome.report
    assert outcome.report["execution"]["module_states"]["quality_hand"] == {
        "state": "disabled",
        "reason": "not_available",
    }
    assert outcome.report["pipeline_state"]["next_module"] == "semantic_consistency"


def test_leading_disabled_module_persists_before_first_enabled_module(
    tmp_path: Path,
) -> None:
    modules = ["effective_duration", "hdf5_text_info"]
    config = _config(tmp_path, modules, disabled={"effective_duration"})
    calls: list[str] = []

    outcome = run_asset(
        _context(tmp_path),
        config=config,
        profile="acceptance",
        registry=_registry(calls, config, {"hdf5_text_info": "pass"}),
        now=lambda: "2026-07-14T00:00:00Z",
    )

    assert calls == ["hdf5_text_info"]
    assert outcome.status == "completed"
    assert outcome.report["report_revision"] == 2
    assert outcome.report["execution"]["module_states"]["effective_duration"] == {
        "state": "disabled",
        "reason": "not_available",
    }
    assert "effective_duration" not in outcome.report


def test_unavailable_automatic_runner_is_persisted_as_incomplete_error(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, ["hdf5_text_info"])
    context = _context(tmp_path)

    outcome = run_asset(
        context,
        config=config,
        profile="acceptance",
        registry=ModuleRegistry(),
        now=lambda: "2026-07-14T00:00:00Z",
    )

    assert outcome.status == "error"
    assert outcome.report["pipeline_state"] == {
        "status": "error",
        "last_completed_module": None,
        "next_module": "hdf5_text_info",
        "stop_reason": "module_unavailable",
    }
    assert outcome.report["overall_decision"] is None
    assert outcome.report["runtime_errors"] == [
        {
            "module": "hdf5_text_info",
            "error_type": "module_unavailable",
            "message": (
                "automatic module implementation is unavailable: "
                "test.hdf5_text_info"
            ),
            "occurred_at": "2026-07-14T00:00:00Z",
            "retryable": False,
        }
    ]
    assert outcome.report["execution"]["module_states"]["hdf5_text_info"] == {
        "state": "not_implemented",
        "reason": "module_unavailable",
    }
    assert "hdf5_text_info" not in outcome.report
    assert outcome.report["issues"] == []


def test_config_drift_is_rejected_before_resuming(tmp_path: Path) -> None:
    modules = ["hdf5_text_info", "semantic_consistency"]
    config = _config(tmp_path, modules)
    context = _context(tmp_path)
    first = run_asset(
        context,
        config=config,
        profile="acceptance",
        registry=_registry([], config, {"hdf5_text_info": "pass"}),
        now=lambda: "2026-07-14T00:00:00Z",
    )
    before = context.report_path.read_bytes()
    drifted = LoadedQcConfig(config.path, copy.deepcopy(config.raw), "sha256:" + "f" * 64)

    with pytest.raises(ConfigDriftError, match="config drift"):
        run_asset(
            context,
            config=drifted,
            profile="acceptance",
            registry=ModuleRegistry(),
        )

    assert context.report_path.read_bytes() == before
    assert first.report["pipeline_state"]["status"] == "awaiting_external"


def test_external_pause_transaction_preserves_extensions_and_is_idempotent(
    tmp_path: Path,
) -> None:
    modules = ["hdf5_text_info", "semantic_consistency"]
    config = _config(tmp_path, modules)
    context = _context(tmp_path)
    report = apply_module_result(
        context.report_path,
        context=context,
        config=config,
        profile="acceptance",
        result=ModuleResult("hdf5_text_info", "pass", {}, {}),
        expected_revision=0,
        next_module="semantic_consistency",
        now="2026-07-14T00:00:00Z",
    )
    report["future_extension"] = {"keep": True}
    context.report_path.write_text(json.dumps(report), encoding="utf-8")

    paused = record_awaiting_external(
        context.report_path,
        context=context,
        config=config,
        profile="acceptance",
        expected_revision=1,
        module="semantic_consistency",
        now="2026-07-14T00:01:00Z",
    )
    same = record_awaiting_external(
        context.report_path,
        context=context,
        config=config,
        profile="acceptance",
        expected_revision=2,
        module="semantic_consistency",
        now="2026-07-14T00:02:00Z",
    )

    assert paused["future_extension"] == {"keep": True}
    assert paused["pipeline_state"]["last_completed_module"] == "hdf5_text_info"
    assert paused["pipeline_state"]["next_module"] == "semantic_consistency"
    assert paused["report_revision"] == 2
    assert same == paused


def test_runtime_error_transaction_preserves_report_and_is_idempotent(
    tmp_path: Path,
) -> None:
    modules = ["hdf5_text_info", "quality_hand"]
    config = _config(tmp_path, modules)
    context = _context(tmp_path)
    prior_issue = Issue(
        "hdf5_text_info:warning:11111111111111111111",
        "warning",
        "warn",
        "hdf5_text_info",
        "test_warning",
        "test_metric",
        1,
        ">",
        0,
        "hdf5_text_info.warning",
        True,
    )
    report = apply_module_result(
        context.report_path,
        context=context,
        config=config,
        profile="acceptance",
        result=ModuleResult(
            "hdf5_text_info",
            "warn",
            {},
            {},
            issues=(prior_issue,),
        ),
        expected_revision=0,
        next_module="quality_hand",
        now="2026-07-14T00:00:00Z",
    )
    report["future_extension"] = {"keep": True}
    report["manual_review"]["semantic_revision"] = {"revision_id": "semantic-r4"}
    context.report_path.write_text(json.dumps(report), encoding="utf-8")

    errored = record_runtime_error(
        context.report_path,
        module="quality_hand",
        error_type="process_error",
        message="worker exited 9",
        expected_revision=1,
        context=context,
        config=config,
        profile="acceptance",
        now="2026-07-14T00:02:00Z",
    )
    same = record_runtime_error(
        context.report_path,
        module="quality_hand",
        error_type="process_error",
        message="worker exited 9",
        expected_revision=2,
        context=context,
        config=config,
        profile="acceptance",
        now="2026-07-14T00:01:00Z",
    )

    assert errored["hdf5_text_info"] == report["hdf5_text_info"]
    assert errored["issues"] == report["issues"]
    assert errored["manual_review"] == report["manual_review"]
    assert errored["future_extension"] == {"keep": True}
    assert errored["report_revision"] == 2
    assert same == errored
    assert json.loads(context.report_path.read_text(encoding="utf-8")) == errored


def test_runtime_error_transaction_rejects_stale_revision_without_overwrite(
    tmp_path: Path,
) -> None:
    modules = ["hdf5_text_info", "semantic_consistency"]
    config = _config(tmp_path, modules)
    context = _context(tmp_path)
    apply_module_result(
        context.report_path,
        context=context,
        config=config,
        profile="acceptance",
        result=ModuleResult("hdf5_text_info", "pass", {}, {}),
        expected_revision=0,
        next_module="semantic_consistency",
        now="2026-07-14T00:00:00Z",
    )
    record_awaiting_external(
        context.report_path,
        context=context,
        config=config,
        profile="acceptance",
        expected_revision=1,
        module="semantic_consistency",
        now="2026-07-14T00:01:00Z",
    )
    before = context.report_path.read_bytes()

    with pytest.raises(StaleReportRevisionError, match="expected revision 1, found 2"):
        record_runtime_error(
            context.report_path,
            module="semantic_consistency",
            error_type="stale_revision",
            message="concurrent update",
            expected_revision=1,
            context=context,
            config=config,
            profile="acceptance",
            now="2026-07-14T00:02:00Z",
        )

    assert context.report_path.read_bytes() == before


def test_orchestrator_records_stale_revision_when_latest_cursor_is_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = "hdf5_text_info"
    config = _config(tmp_path, [module])
    context = _context(tmp_path)

    def stale_apply(*args: object, **kwargs: object) -> dict[str, Any]:
        concurrent = initialize_v2_report(
            context,
            config,
            "acceptance",
            "2026-07-14T00:00:00Z",
        )
        concurrent["future_extension"] = {"keep": True}
        concurrent["report_revision"] = 1
        write_asset_qc_report(
            context.report_path,
            concurrent,
            expected_revision=0,
            profile="acceptance",
        )
        raise StaleReportRevisionError("expected revision 0, found 1")

    monkeypatch.setattr("qc_pipeline.orchestrator.apply_module_result", stale_apply)

    outcome = run_asset(
        context,
        config=config,
        profile="acceptance",
        registry=_registry([], config, {module: "pass"}),
        now=lambda: "2026-07-14T00:01:00Z",
    )

    assert outcome.status == "error"
    assert outcome.report["report_revision"] == 2
    assert outcome.report["future_extension"] == {"keep": True}
    assert outcome.report["runtime_errors"][0]["error_type"] == "stale_revision"
    assert outcome.report["runtime_errors"][0]["retryable"] is True


def test_orchestrator_does_not_mark_stale_when_latest_cursor_advanced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modules = ["hdf5_text_info", "quality_hand"]
    config = _config(tmp_path, modules)
    context = _context(tmp_path)
    real_apply = apply_module_result
    latest_bytes: list[bytes] = []

    def concurrent_apply(*args: object, **kwargs: object) -> dict[str, Any]:
        real_apply(*args, **kwargs)
        latest_bytes.append(context.report_path.read_bytes())
        raise StaleReportRevisionError("expected revision 0, found 1")

    monkeypatch.setattr("qc_pipeline.orchestrator.apply_module_result", concurrent_apply)

    calls: list[str] = []
    outcome = run_asset(
        context,
        config=config,
        profile="acceptance",
        registry=_registry(calls, config, {name: "pass" for name in modules}),
        now=lambda: "2026-07-14T00:00:00Z",
    )

    assert calls == ["hdf5_text_info"]
    assert outcome.status == "running"
    assert outcome.report["pipeline_state"]["next_module"] == "quality_hand"
    assert outcome.report["runtime_errors"] == []
    assert context.report_path.read_bytes() == latest_bytes[0]


def test_asset_context_rejects_report_and_sources_outside_batch(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="report_path must be inside batch_root"):
        AssetContext("a", tmp_path / "batch", tmp_path / "outside.json", {})
    with pytest.raises(ValueError, match="source_files.video.path"):
        AssetContext(
            "a",
            tmp_path / "batch",
            tmp_path / "batch" / "a.json",
            {"video": {"path": "../outside.mp4"}},
        )


def test_default_registry_exposes_only_enabled_automatic_implementations(
    tmp_path: Path,
) -> None:
    from qc_common.config import load_qc_acceptance_config

    config = load_qc_acceptance_config()
    registry = build_default_registry(_context(tmp_path), config)

    expected = {
        str(config.module_config(name)["implementation"])
        for name in config.pipeline_modules
        if config.module_config(name).get("enabled")
        and config.module_config(name).get("execution_kind") != "external"
    }
    assert all(registry.has(name) for name in expected)
    assert not registry.has("semantic_consistency")
    assert not registry.has("manual_review")
    assert not registry.has("duplicate_check")


def test_default_runner_reports_missing_declared_source_as_prerequisite(
    tmp_path: Path,
) -> None:
    from qc_common.config import load_qc_acceptance_config

    config = load_qc_acceptance_config()
    context = AssetContext(
        "asset-a",
        tmp_path,
        tmp_path / "quality_archive" / "asset-a.json",
        {},
    )
    registry = build_default_registry(context, config)
    implementation = str(
        config.module_config("hdf5_text_info")["implementation"]
    )

    with pytest.raises(ModulePrerequisiteError) as raised:
        registry.resolve(implementation)(context, config)

    assert raised.value.module == "hdf5_text_info"
    assert raised.value.prerequisite == "source_files.hdf5.path"


def test_default_registry_is_bound_to_the_exact_asset_context(
    tmp_path: Path,
) -> None:
    from qc_common.config import load_qc_acceptance_config

    config = load_qc_acceptance_config()
    context = AssetContext(
        "asset-a",
        tmp_path,
        tmp_path / "quality_archive" / "asset-a.json",
        {},
    )
    other_context = AssetContext(
        "asset-a",
        tmp_path,
        tmp_path / "other" / "asset-a.json",
        {},
    )
    registry = build_default_registry(context, config)
    implementation = str(
        config.module_config("hdf5_text_info")["implementation"]
    )

    with pytest.raises(ValueError, match="cannot be shared"):
        registry.resolve(implementation)(other_context, config)


def test_resume_rejects_source_identity_drift_before_runner_work(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, ["hdf5_text_info", "semantic_consistency"])
    original = _context(tmp_path)
    run_asset(
        original,
        config=config,
        profile="acceptance",
        registry=_registry([], config, {"hdf5_text_info": "pass"}),
        now=lambda: "2026-07-14T00:00:00Z",
    )
    changed = AssetContext(
        original.asset_id,
        original.batch_root,
        original.report_path,
        {"video": {"path": "video/other.mp4"}},
    )

    with pytest.raises(ValueError, match="source_files mismatch"):
        run_asset(
            changed,
            config=config,
            profile="acceptance",
            registry=ModuleRegistry(),
        )


def test_batch_rejects_duplicate_asset_ids_before_building_registries(
    tmp_path: Path,
) -> None:
    from tools.run_qc_pipeline import run_batch

    config = _config(tmp_path, ["hdf5_text_info"])
    contexts = [_context(tmp_path, "same"), _context(tmp_path, "same")]
    built: list[str] = []

    with pytest.raises(ValueError, match="duplicate asset_id: same"):
        run_batch(
            contexts,
            config=config,
            profile="acceptance",
            registry_factory=lambda context: built.append(context.asset_id),
            max_workers=2,
        )

    assert built == []


def test_manifest_rejects_duplicate_ids_before_source_validation(
    tmp_path: Path,
) -> None:
    from tools.run_qc_pipeline import contexts_from_manifest

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "asset_id": "same",
                        "primary_video_path": "../outside.mp4",
                    }
                ),
                json.dumps({"asset_id": "same"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate asset_id: same"):
        contexts_from_manifest(manifest, batch_root=tmp_path)


def test_manifest_accepts_integral_float_frame_bounds(tmp_path: Path) -> None:
    from tools.run_qc_pipeline import contexts_from_manifest

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "asset_id": "a",
                "start_frame": 1.0,
                "end_frame": 3.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    contexts = contexts_from_manifest(manifest, batch_root=tmp_path)

    assert contexts[0].source_range == (1, 4)


def test_batch_builds_an_independent_registry_per_asset(tmp_path: Path) -> None:
    from tools.run_qc_pipeline import run_batch

    config = _config(tmp_path, ["hdf5_text_info"])
    contexts = [_context(tmp_path, "a"), _context(tmp_path, "b")]
    built: list[str] = []

    def registry_factory(context: AssetContext) -> ModuleRegistry:
        built.append(context.asset_id)
        return _registry([], config, {"hdf5_text_info": "pass"})

    outcomes = run_batch(
        contexts,
        config=config,
        profile="acceptance",
        registry_factory=registry_factory,
        max_workers=2,
    )

    assert sorted(built) == ["a", "b"]
    assert set(outcomes) == {"a", "b"}
    assert outcomes["a"].report is not outcomes["b"].report
    assert outcomes["a"].status == outcomes["b"].status == "completed"


def test_batch_keeps_other_assets_running_when_one_runner_errors(
    tmp_path: Path,
) -> None:
    from tools.run_qc_pipeline import run_batch

    module = "hdf5_text_info"
    config = _config(tmp_path, [module])
    contexts = [_context(tmp_path, "good"), _context(tmp_path, "bad")]

    def registry_factory(context: AssetContext) -> ModuleRegistry:
        return _registry(
            [],
            config,
            {module: "pass"},
            errors={module} if context.asset_id == "bad" else set(),
        )

    outcomes = run_batch(
        contexts,
        config=config,
        profile="supplier_evaluation",
        registry_factory=registry_factory,
        max_workers=2,
    )

    assert outcomes["good"].status == "completed"
    assert outcomes["good"].report["overall_decision"] == "pass"
    assert outcomes["bad"].status == "error"
    assert outcomes["bad"].report["overall_decision"] is None
    assert outcomes["bad"].report["runtime_errors"][0]["error_type"] == (
        "process_error"
    )
    assert outcomes["good"].report is not outcomes["bad"].report


def test_cli_accepts_required_batch_and_resume_options(tmp_path: Path) -> None:
    from tools.run_qc_pipeline import build_parser

    args = build_parser().parse_args(
        [
            "--batch-root",
            str(tmp_path),
            "--manifest",
            str(tmp_path / "manifest.jsonl"),
            "--profile",
            "supplier_evaluation",
            "--config",
            str(tmp_path / "qc.yaml"),
            "--max-workers",
            "3",
            "--no-resume",
        ]
    )

    assert args.batch_root == tmp_path
    assert args.manifest == tmp_path / "manifest.jsonl"
    assert args.profile == "supplier_evaluation"
    assert args.config == tmp_path / "qc.yaml"
    assert args.max_workers == 3
    assert args.resume is False
