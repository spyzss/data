from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest

from human_qc.semantic_service import SemanticCalibrationService
from human_qc.warn_service import WarnReviewService
from human_qc.workbench_service import WorkbenchService
from qc_common.config import LoadedQcConfig
from qc_common.contracts import Issue, ModuleResult
from qc_common.module_registry import ModuleRegistry
from qc_common.report import (
    StaleReportRevisionError,
    load_asset_qc_report,
)
from qc_pipeline.context import AssetContext
from qc_pipeline.orchestrator import resume_after_external, run_asset


HASH = "sha256:" + "1" * 64


def _config(
    tmp_path: Path,
    modules: list[str] | None = None,
) -> LoadedQcConfig:
    modules = modules or ["auto", "semantic_consistency", "tail"]
    module_configs = {}
    for module in modules:
        if module in {"semantic_consistency", "manual_review"}:
            module_configs[module] = {
                "enabled": True,
                "execution_kind": "external",
                "parameters": {},
                "rules": {},
            }
        else:
            module_configs[module] = {
                "enabled": True,
                "implementation": f"test.{module}",
                "parameters": {},
                "rules": {},
            }
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
            "modules": module_configs,
        },
    )


def _context(tmp_path: Path, asset_id: str = "asset-a") -> AssetContext:
    return AssetContext(
        asset_id,
        tmp_path,
        tmp_path / "quality_archive" / f"{asset_id}.json",
        {"video": {"path": "video/clip.mp4"}},
    )


def _registry(
    auto_verdict: str = "pass",
    *,
    calls: list[str] | None = None,
    auto_issues: tuple[Issue, ...] = (),
) -> ModuleRegistry:
    registry = ModuleRegistry()

    def result(module: str, verdict: str, issues: tuple[Issue, ...] = ()) -> ModuleResult:
        if calls is not None:
            calls.append(module)
        return ModuleResult(module, verdict, {}, {}, issues=issues)

    registry.register(
        "test.auto",
        lambda context, config: result("auto", auto_verdict, auto_issues),
    )
    registry.register("test.tail", lambda context, config: result("tail", "pass"))
    return registry


def _warn_issue() -> Issue:
    return Issue(
        issue_id="warn-1",
        code="blur",
        severity="warn",
        module="auto",
        issue_type="metric_threshold",
        metric="blur_score",
        observed_value=0.2,
        operator="<",
        boundary_value=0.5,
        rule_id="auto.blur",
        needs_manual_review=True,
    )


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


def test_terminal_external_completion_preserves_machine_fail(tmp_path: Path) -> None:
    context = _context(tmp_path)
    config = _config(tmp_path, ["auto", "semantic_consistency"])
    first = run_asset(
        context,
        config=config,
        profile="supplier_evaluation",
        registry=_registry("fail"),
    )

    resumed = resume_after_external(
        context,
        config=config,
        profile="supplier_evaluation",
        completed_module="semantic_consistency",
        expected_revision=first.report["report_revision"],
    )

    assert resumed.status == "completed"
    assert resumed.report["overall_decision"] == "fail"


def test_empty_manual_review_stage_skips_and_runs_downstream(tmp_path: Path) -> None:
    context = _context(tmp_path)
    config = _config(
        tmp_path,
        ["auto", "semantic_consistency", "manual_review", "tail"],
    )
    calls: list[str] = []
    registry = _registry(calls=calls)
    first = run_asset(
        context,
        config=config,
        profile="acceptance",
        registry=registry,
    )

    resumed = resume_after_external(
        context,
        config=config,
        profile="acceptance",
        completed_module="semantic_consistency",
        expected_revision=first.report["report_revision"],
        registry=registry,
    )

    assert resumed.status == "completed"
    assert calls == ["auto", "tail"]
    assert resumed.report["manual_review"]["state"] == "not_required"
    assert resumed.report["pipeline_state"]["next_module"] is None


def test_machine_warn_with_empty_selection_skips_manual_and_runs_downstream(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    config = _config(
        tmp_path,
        ["auto", "semantic_consistency", "manual_review", "tail"],
    )
    calls: list[str] = []
    registry = _registry(calls=calls, auto_issues=(_warn_issue(),))
    first = run_asset(
        context,
        config=config,
        profile="supplier_evaluation",
        registry=registry,
    )
    assert first.report["manual_review"]["candidate_issue_ids"] == ["warn-1"]
    assert first.report["manual_review"].get("selected_issue_ids", []) == []

    resumed = resume_after_external(
        context,
        config=config,
        profile="supplier_evaluation",
        completed_module="semantic_consistency",
        expected_revision=first.report["report_revision"],
        registry=registry,
    )

    assert resumed.status == "completed"
    assert calls == ["auto", "tail"]
    assert resumed.report["manual_review"]["candidate_issue_ids"] == ["warn-1"]
    assert resumed.report["manual_review"]["selected_issue_ids"] == []
    assert resumed.report["manual_review"]["state"] == "not_required"


def test_terminal_external_completion_replay_is_rejected_without_write(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    config = _config(tmp_path, ["auto", "semantic_consistency"])
    first = run_asset(
        context,
        config=config,
        profile="acceptance",
        registry=_registry(),
    )
    completed = resume_after_external(
        context,
        config=config,
        profile="acceptance",
        completed_module="semantic_consistency",
        expected_revision=first.report["report_revision"],
    )
    before = context.report_path.read_bytes()

    with pytest.raises(ValueError, match="awaiting_external|completion"):
        resume_after_external(
            context,
            config=config,
            profile="acceptance",
            completed_module="semantic_consistency",
            expected_revision=completed.report["report_revision"],
        )

    assert context.report_path.read_bytes() == before


def test_successor_pending_external_replay_is_rejected_without_write(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    config = _config(tmp_path, ["auto", "semantic_consistency", "tail"])
    first = run_asset(
        context,
        config=config,
        profile="acceptance",
        registry=_registry(),
    )
    advanced = resume_after_external(
        context,
        config=config,
        profile="acceptance",
        completed_module="semantic_consistency",
        expected_revision=first.report["report_revision"],
    )
    assert advanced.report["pipeline_state"]["status"] == "running"
    assert advanced.report["pipeline_state"]["next_module"] == "tail"
    before = context.report_path.read_bytes()

    with pytest.raises(ValueError, match="awaiting_external|completion"):
        resume_after_external(
            context,
            config=config,
            profile="acceptance",
            completed_module="semantic_consistency",
            expected_revision=advanced.report["report_revision"],
        )

    assert context.report_path.read_bytes() == before


def test_workbench_warn_completion_resumes_configured_downstream_once(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    config = _config(
        tmp_path,
        ["auto", "semantic_consistency", "manual_review", "tail"],
    )
    calls: list[str] = []
    registry = _registry(calls=calls, auto_issues=(_warn_issue(),))
    first = run_asset(
        context,
        config=config,
        profile="supplier_evaluation",
        registry=registry,
    )
    selected_report = load_asset_qc_report(context.report_path)
    assert selected_report is not None
    selected_report["manual_review"].update(
        {
            "required": True,
            "state": "queued",
            "selected_issue_ids": ["warn-1"],
            "selected_issue_id": "warn-1",
            "issue_reviews": {},
            "completed_at": None,
        }
    )
    context.report_path.write_text(json.dumps(selected_report), encoding="utf-8")
    semantic_done = resume_after_external(
        context,
        config=config,
        profile="supplier_evaluation",
        completed_module="semantic_consistency",
        expected_revision=first.report["report_revision"],
        registry=registry,
    )
    assert semantic_done.status == "awaiting_external"
    report = load_asset_qc_report(context.report_path)
    assert report is not None
    report["semantic_calibration"] = {
        "state": "completed",
        "source_dataset_path": "/label/subtask_label",
        "base_hdf5_sha256": "sha256:" + "0" * 64,
        "final_hdf5_sha256": "sha256:" + "1" * 64,
        "timeline_edit_count": 0,
        "subtask_text_edit_count": 0,
        "pending_edit": None,
        "audit": [],
    }
    context.report_path.write_text(json.dumps(report), encoding="utf-8")
    warn = WarnReviewService(
        reports={context.asset_id: context.report_path},
        reviewer="alice",
    )

    def registry_after_domain_write(
        _context: AssetContext, _config: LoadedQcConfig
    ) -> ModuleRegistry:
        domain_report = load_asset_qc_report(context.report_path)
        assert domain_report is not None
        assert domain_report["manual_review"]["state"] == "completed"
        assert domain_report["pipeline_state"] == {
            "status": "awaiting_external",
            "last_completed_module": "semantic_consistency",
            "next_module": "manual_review",
            "stop_reason": None,
        }
        return registry

    service = WorkbenchService(
        warn_service=warn,
        asset_contexts={context.asset_id: context},
        profile="supplier_evaluation",
        config=config,
        registry_factory=registry_after_domain_write,
    )
    lease = service.acquire_lease(context.asset_id, "alice", 60)
    reviewed = service.warn_verdict(
        context.asset_id,
        issue_id="warn-1",
        verdict="pass",
        expected_revision=report["report_revision"],
        lease_token=lease.token,
    )

    completed = service.warn_complete(
        context.asset_id,
        expected_revision=reviewed["revision"],
        lease_token=lease.token,
    )

    assert completed["task_type"] == "completed"
    assert calls == ["auto", "tail"]
    persisted = load_asset_qc_report(context.report_path)
    assert persisted is not None
    assert persisted["execution"]["module_states"]["manual_review"] == {
        "state": "completed"
    }
    assert persisted["pipeline_state"]["last_completed_module"] == "tail"


def test_workbench_semantic_completion_keeps_cursor_for_orchestrator(
    tmp_path: Path,
) -> None:
    asset_id = "semantic-asset"
    hdf5_path = tmp_path / "source" / f"{asset_id}.hdf5"
    hdf5_path.parent.mkdir(parents=True)
    payload = {
        "id": asset_id,
        "scene": "kitchen",
        "task": "move block",
        "fps": 30.0,
        "frame_count": 30,
        "annotations": [
            {
                "start_frame": 0,
                "end_frame": 29,
                "start_time_sec": 0.0,
                "end_time_sec": 29 / 30,
                "subtask_cn": "移动方块",
                "subtask_en": "move block",
                "verb": "move",
                "object": "block",
                "target": "tray",
                "hand": "right",
                "phase": "move",
                "evidence_frames": [0, 29],
                "confidence": 0.9,
                "status": "confirmed",
            }
        ],
    }
    with h5py.File(hdf5_path, "w") as handle:
        dataset = handle.create_group("label").create_dataset(
            "subtask_label", shape=(), dtype="S65536"
        )
        dataset[()] = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    context = AssetContext(
        asset_id,
        tmp_path,
        tmp_path / "quality_archive" / f"{asset_id}.json",
        {"hdf5": {"path": hdf5_path.relative_to(tmp_path).as_posix()}},
    )
    config = _config(
        tmp_path,
        ["auto", "semantic_consistency", "manual_review", "tail"],
    )
    calls: list[str] = []
    registry = _registry(calls=calls)
    first = run_asset(
        context,
        config=config,
        profile="acceptance",
        registry=registry,
    )
    semantic = SemanticCalibrationService(
        assets={asset_id: hdf5_path},
        reports={asset_id: context.report_path},
    )

    def registry_after_domain_write(
        _context: AssetContext, _config: LoadedQcConfig
    ) -> ModuleRegistry:
        domain_report = load_asset_qc_report(context.report_path)
        assert domain_report is not None
        assert domain_report["semantic_calibration"]["state"] == "completed"
        assert domain_report["pipeline_state"]["status"] == "awaiting_external"
        assert domain_report["pipeline_state"]["next_module"] == "semantic_consistency"
        return registry

    service = WorkbenchService(
        semantic_service=semantic,
        asset_contexts={asset_id: context},
        profile="acceptance",
        config=config,
        registry_factory=registry_after_domain_write,
    )
    lease = service.acquire_lease(asset_id, "alice", 60)

    completed = service.semantic_complete(
        asset_id,
        expected_revision=first.report["report_revision"],
        lease_token=lease.token,
    )

    assert completed["task_type"] == "completed"
    assert calls == ["auto", "tail"]
    persisted = load_asset_qc_report(context.report_path)
    assert persisted is not None
    assert persisted["manual_review"]["state"] == "not_required"
    assert persisted["semantic_calibration"]["final_hdf5_sha256"]
    assert persisted["pipeline_state"]["last_completed_module"] == "tail"


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


class _StaleSemanticProjection:
    def __init__(self, report_path: Path) -> None:
        self._reports = {"stale": report_path}

    def get_task(self, asset_id: str) -> SimpleNamespace:
        report = load_asset_qc_report(self._reports[asset_id])
        assert report is not None
        return SimpleNamespace(
            report_revision=report["report_revision"],
            report_state="in_progress",
            state="in_progress",
        )


class _StaleWarnProjection:
    def __init__(self, report_path: Path, asset_id: str) -> None:
        self._reports = {asset_id: report_path}

    def get_task(self, asset_id: str) -> SimpleNamespace:
        report = load_asset_qc_report(self._reports[asset_id])
        assert report is not None
        return SimpleNamespace(
            report_revision=report["report_revision"],
            state="queued",
            selected_issue_ids=("warn-1",),
        )


@pytest.mark.parametrize("terminal_status", ["completed", "stopped", "error"])
def test_terminal_status_blocks_stale_projection_acquire_and_renew(
    tmp_path: Path,
    terminal_status: str,
) -> None:
    context = _context(tmp_path, "stale")
    report = {
        "asset_id": "stale",
        "report_revision": 1,
        "execution": {"profile": "acceptance"},
        "pipeline_state": {
            "status": "awaiting_external",
            "last_completed_module": "auto",
            "next_module": "semantic_consistency",
            "stop_reason": None,
        },
        "manual_review": {"state": "not_evaluated", "candidate_issue_ids": []},
        "issues": [],
    }
    context.report_path.parent.mkdir(parents=True, exist_ok=True)
    context.report_path.write_text(json.dumps(report), encoding="utf-8")
    service = WorkbenchService(
        semantic_service=_StaleSemanticProjection(context.report_path),
        asset_contexts={"stale": context},
    )
    lease = service.acquire_lease("stale", "alice", 60)
    report["report_revision"] = 2
    report["pipeline_state"].update(
        {"status": terminal_status, "next_module": None}
    )
    context.report_path.write_text(json.dumps(report), encoding="utf-8")

    expected_task_type = "error" if terminal_status == "error" else "completed"
    assert service.get_asset_task("stale")["task_type"] == expected_task_type
    with pytest.raises(KeyError, match="not actionable"):
        service.acquire_lease("stale", "bob", 60)
    with pytest.raises(KeyError, match="not actionable"):
        service.renew_lease("stale", lease.token, 60)


def test_empty_manual_projection_is_never_actionable(tmp_path: Path) -> None:
    context = _context(tmp_path, "empty-manual")
    context.report_path.parent.mkdir(parents=True, exist_ok=True)
    context.report_path.write_text(
        json.dumps(
            {
                "asset_id": context.asset_id,
                "report_revision": 3,
                "execution": {"profile": "acceptance"},
                "pipeline_state": {
                    "status": "awaiting_external",
                    "last_completed_module": "semantic_consistency",
                    "next_module": "manual_review",
                    "stop_reason": None,
                },
                "manual_review": {
                    "state": "queued",
                    "candidate_issue_ids": [],
                    "selected_issue_ids": [],
                },
                "issues": [],
            }
        ),
        encoding="utf-8",
    )
    service = WorkbenchService(
        warn_service=_StaleWarnProjection(context.report_path, context.asset_id),
        asset_contexts={context.asset_id: context},
    )

    assert service.get_asset_task(context.asset_id)["task_type"] == "completed"
    assert service.list_actionable_assets("acceptance") == ()
    with pytest.raises(KeyError, match="not actionable"):
        service.acquire_lease(context.asset_id, "alice", 60)


def test_unselected_manual_projection_is_never_actionable(tmp_path: Path) -> None:
    context = _context(tmp_path, "unselected-manual")
    context.report_path.parent.mkdir(parents=True, exist_ok=True)
    context.report_path.write_text(
        json.dumps(
            {
                "asset_id": context.asset_id,
                "report_revision": 3,
                "execution": {"profile": "acceptance"},
                "pipeline_state": {
                    "status": "awaiting_external",
                    "last_completed_module": "semantic_consistency",
                    "next_module": "manual_review",
                    "stop_reason": None,
                },
                "semantic_calibration": {"state": "completed"},
                "manual_review": {
                    "state": "queued",
                    "candidate_issue_ids": ["warn-1"],
                    "selected_issue_ids": [],
                },
                "issues": [],
            }
        ),
        encoding="utf-8",
    )
    service = WorkbenchService(
        warn_service=_StaleWarnProjection(context.report_path, context.asset_id),
        asset_contexts={context.asset_id: context},
    )

    assert service.get_asset_task(context.asset_id)["task_type"] == "completed"
    assert service.list_actionable_assets("acceptance") == ()
    with pytest.raises(KeyError, match="not actionable"):
        service.acquire_lease(context.asset_id, "alice", 60)
