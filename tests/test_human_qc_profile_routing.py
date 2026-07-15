from __future__ import annotations

import json
from pathlib import Path

import pytest

from human_qc.workbench_service import WorkbenchService
from qc_common.config import LoadedQcConfig
from qc_common.contracts import ModuleResult
from qc_common.module_registry import ModuleRegistry
from qc_common.report import StaleReportRevisionError
from qc_pipeline.context import AssetContext
from qc_pipeline.orchestrator import resume_after_external, run_asset


HASH = "sha256:" + "1" * 64


def _config(tmp_path: Path) -> LoadedQcConfig:
    modules = ["auto", "semantic_consistency", "tail"]
    return LoadedQcConfig(
        path=tmp_path / "qc.yaml",
        sha256=HASH,
        raw={
            "schema_version": "qc_acceptance_config_schema.v2",
            "config_version": "qc_acceptance_v2.0.0",
            "config_name": "test",
            "execution_profiles": {
                "acceptance": {"fail_action": "stop", "runtime_error_action": "stop_incomplete"},
                "supplier_evaluation": {"fail_action": "record_and_continue", "runtime_error_action": "stop_incomplete"},
            },
            "pipeline": {"default_profile": "acceptance", "terminal_module": "tail", "modules": modules},
            "modules": {
                "auto": {"enabled": True, "implementation": "test.auto", "parameters": {}, "rules": {}},
                "semantic_consistency": {"enabled": True, "execution_kind": "external", "parameters": {}, "rules": {}},
                "tail": {"enabled": True, "implementation": "test.tail", "parameters": {}, "rules": {}},
            },
        },
    )


def _context(tmp_path: Path, asset_id: str = "asset-a") -> AssetContext:
    return AssetContext(
        asset_id,
        tmp_path,
        tmp_path / "quality_archive" / f"{asset_id}.json",
        {"video": {"path": "video/clip.mp4"}},
    )


def _registry(auto_verdict: str = "pass") -> ModuleRegistry:
    registry = ModuleRegistry()
    registry.register("test.auto", lambda context, config: ModuleResult("auto", auto_verdict, {}, {}))
    registry.register("test.tail", lambda context, config: ModuleResult("tail", "pass", {}, {}))
    return registry


def test_resume_after_external_advances_once_and_runs_successors(tmp_path: Path) -> None:
    context = _context(tmp_path)
    config = _config(tmp_path)
    first = run_asset(
        context,
        config=config,
        profile="acceptance",
        registry=_registry(),
        now=lambda: "2026-07-15T00:00:00Z",
    )
    assert first.status == "awaiting_external"
    resumed = resume_after_external(
        context,
        config=config,
        profile="acceptance",
        completed_module="semantic_consistency",
        expected_revision=first.report["report_revision"],
        registry=_registry(),
        now=lambda: "2026-07-15T00:01:00Z",
    )
    assert resumed.status == "completed"
    assert resumed.executed_modules == ("semantic_consistency", "tail")
    assert resumed.report["execution"]["module_states"]["semantic_consistency"] == {"state": "completed"}
    assert resumed.report["pipeline_state"]["next_module"] is None


def test_resume_after_external_rejects_stale_revision(tmp_path: Path) -> None:
    context = _context(tmp_path)
    config = _config(tmp_path)
    first = run_asset(context, config=config, profile="acceptance", registry=_registry())
    with pytest.raises(StaleReportRevisionError):
        resume_after_external(
            context,
            config=config,
            profile="acceptance",
            completed_module="semantic_consistency",
            expected_revision=first.report["report_revision"] - 1,
        )


def test_supplier_machine_fail_continues_human_stage_but_remains_final_fail(tmp_path: Path) -> None:
    context = _context(tmp_path)
    config = _config(tmp_path)
    first = run_asset(
        context,
        config=config,
        profile="supplier_evaluation",
        registry=_registry("fail"),
    )
    assert first.status == "awaiting_external"
    resumed = resume_after_external(
        context,
        config=config,
        profile="supplier_evaluation",
        completed_module="semantic_consistency",
        expected_revision=first.report["report_revision"],
        registry=_registry("fail"),
    )
    assert resumed.status == "completed"
    assert resumed.report["overall_decision"] == "fail"


def test_workbench_lists_only_profile_matching_external_assets(tmp_path: Path) -> None:
    supplier = _context(tmp_path, "supplier")
    stopped = _context(tmp_path, "stopped")
    for context, profile, status, next_module in (
        (supplier, "supplier_evaluation", "awaiting_external", "semantic_consistency"),
        (stopped, "acceptance", "stopped", None),
    ):
        context.report_path.parent.mkdir(parents=True, exist_ok=True)
        context.report_path.write_text(
            json.dumps(
                {
                    "asset_id": context.asset_id,
                    "report_revision": 2,
                    "execution": {"profile": profile},
                    "pipeline_state": {"status": status, "next_module": next_module},
                    "manual_review": {"state": "not_evaluated", "candidate_issue_ids": []},
                    "issues": [],
                }
            ),
            encoding="utf-8",
        )
    service = WorkbenchService(asset_contexts={"supplier": supplier, "stopped": stopped})
    assert service.list_actionable_assets("supplier_evaluation") == ("supplier",)
    assert service.list_actionable_assets("acceptance") == ()
    assert service.get_asset_task("stopped")["task_type"] == "completed"
    with pytest.raises(KeyError):
        service.acquire_lease("stopped", "alice", 60)
