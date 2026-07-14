from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pytest

import qc_common.report_mutation as report_mutation
from qc_common.config import LoadedQcConfig
from qc_common.contracts import Issue, ModuleResult
from qc_common.module_registry import ModulePrerequisiteError, ModuleRegistry
from qc_common.report import load_asset_qc_report
from qc_pipeline.context import AssetContext
from qc_pipeline.orchestrator import run_asset


_HASH = "sha256:" + "1" * 64
_MODULES = (
    "hdf5_text_info",
    "quality_hand",
    "semantic_consistency",
    "manual_review",
)
_WARN_ID = "quality_hand:fixture_warn:11111111111111111111"


def _config(tmp_path: Path) -> LoadedQcConfig:
    modules = {
        "hdf5_text_info": {
            "enabled": True,
            "implementation": "fixture.hdf5_text_info",
            "parameters": {},
            "rules": {},
        },
        "quality_hand": {
            "enabled": True,
            "implementation": "fixture.quality_hand",
            "parameters": {},
            "rules": {},
        },
        "semantic_consistency": {
            "enabled": True,
            "execution_kind": "external",
            "parameters": {},
            "rules": {},
        },
        "manual_review": {
            "enabled": True,
            "execution_kind": "external",
            "parameters": {},
            "rules": {},
        },
    }
    return LoadedQcConfig(
        path=tmp_path / "qc.yaml",
        raw={
            "schema_version": "qc_acceptance_config_schema.v2",
            "config_version": "qc_acceptance_v2.0.0",
            "config_name": "fixture_trace",
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
                "terminal_module": "manual_review",
                "modules": list(_MODULES),
            },
            "modules": modules,
        },
        sha256=_HASH,
    )


def _context(root: Path, asset_id: str) -> AssetContext:
    return AssetContext(
        asset_id=asset_id,
        batch_root=root,
        report_path=root / "quality_archive" / f"{asset_id}.json",
        source_files={"video": {"path": "video/clip.mp4"}},
    )


def _registry(
    config: LoadedQcConfig,
    *,
    verdicts: dict[str, str],
    errors: dict[str, str],
) -> ModuleRegistry:
    registry = ModuleRegistry()
    for module_name in ("hdf5_text_info", "quality_hand"):
        implementation = str(config.module_config(module_name)["implementation"])
        verdict = verdicts.get(module_name, "pass")

        def runner(
            context: AssetContext,
            loaded: LoadedQcConfig,
            *,
            module_name: str = module_name,
            verdict: str = verdict,
        ) -> ModuleResult:
            assert loaded is config
            error_type = errors.get(module_name)
            if error_type == "input_missing":
                raise ModulePrerequisiteError(module_name, "input_missing")
            if error_type:
                raise RuntimeError(f"fixture:{error_type}")
            issues: tuple[Issue, ...] = ()
            if verdict == "warn":
                issues = (
                    Issue(
                        _WARN_ID,
                        "fixture_warn",
                        "warn",
                        module_name,
                        "fixture_warning",
                        "fixture_metric",
                        1,
                        ">",
                        0,
                        "quality_hand.fixture_warn",
                        True,
                    ),
                )
            return ModuleResult(
                module_name,
                verdict,
                {"decision": verdict},
                {"fixture": True},
                issues,
            )

        registry.register(implementation, runner)
    return registry


def _trace_row(report: dict[str, Any]) -> dict[str, Any]:
    pipeline_state = report["pipeline_state"]
    status = pipeline_state["status"]
    module = pipeline_state["last_completed_module"]
    result_verdict: str | None = None
    exit_state: str | None = None
    block = report.get(module) if isinstance(module, str) else None
    if isinstance(block, dict):
        flow = block.get("flow")
        if isinstance(flow, dict):
            result_gate = flow.get("result_gate")
            exit_gate = flow.get("exit_gate")
            if isinstance(result_gate, dict):
                result_verdict = result_gate.get("verdict")
            if isinstance(exit_gate, dict):
                exit_state = exit_gate.get("state")
    if status == "awaiting_external":
        module = pipeline_state["next_module"]
        result_verdict = None
        exit_state = "awaiting_external"
    elif status == "error":
        module = pipeline_state["next_module"]
        result_verdict = None
        exit_state = "error"
    return {
        "asset_id": report["asset_id"],
        "module": module,
        "revision": report["report_revision"],
        "pipeline_status": status,
        "result_verdict": result_verdict,
        "exit_state": exit_state,
        "overall_decision": report["overall_decision"],
        "next_module": pipeline_state["next_module"],
    }


def run_trace_fixtures(tmp_path: Path, manifest_path: Path) -> list[dict[str, Any]]:
    """Run each fixture and capture rows from reports after real atomic writes."""
    traces: list[dict[str, Any]] = []
    reports: dict[str, list[dict[str, Any]]] = defaultdict(list)
    original_writer = report_mutation.write_asset_qc_report

    def capture_writer(
        path: Path,
        report: dict[str, Any],
        expected_revision: int,
        *,
        profile: str | None = None,
    ) -> None:
        original_writer(
            path,
            report,
            expected_revision,
            profile=profile,
        )
        persisted = load_asset_qc_report(path)
        assert persisted is not None
        reports[persisted["asset_id"]].append(copy.deepcopy(persisted))

    # Patch only the shared mutation writer; run_asset and all mutation paths
    # still perform their real validation/CAS/atomic replacement.
    report_mutation.write_asset_qc_report = capture_writer
    try:
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            fixture = json.loads(line)
            asset_id = str(fixture["asset_id"])
            root = tmp_path / asset_id
            config = _config(root)
            outcome = run_asset(
                _context(root, asset_id),
                config=config,
                profile="acceptance",
                registry=_registry(
                    config,
                    verdicts=dict(fixture.get("verdicts", {})),
                    errors=dict(fixture.get("errors", {})),
                ),
                now=lambda: "2026-07-15T00:00:00Z",
            )
            assert outcome.report["asset_id"] == asset_id
    finally:
        report_mutation.write_asset_qc_report = original_writer

    for asset_id in [
        json.loads(line)["asset_id"]
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]:
        asset_reports = reports[asset_id]
        assert asset_reports
        revisions = [int(report["report_revision"]) for report in asset_reports]
        assert revisions == list(range(1, len(revisions) + 1))
        traces.extend(_trace_row(report) for report in asset_reports)
    return traces


def group_by_asset(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["asset_id"]].append(row)
    return dict(grouped)


def test_four_fixture_revision_trace_matches_golden(tmp_path: Path) -> None:
    actual = run_trace_fixtures(
        tmp_path,
        Path("tests/fixtures/qc_pipeline/manifest.jsonl"),
    )
    expected = json.loads(
        Path("tests/fixtures/qc_pipeline/expected_revision_trace.json").read_text(
            encoding="utf-8"
        )
    )
    assert actual == expected
    by_asset = group_by_asset(actual)
    assert by_asset["hard-fail"][-1]["pipeline_status"] == "stopped"
    assert by_asset["hard-fail"][-1]["overall_decision"] == "fail"
    assert by_asset["runtime-error"][-1]["pipeline_status"] == "error"
    assert by_asset["runtime-error"][-1]["overall_decision"] is None


def test_trace_records_warn_candidates_and_skips_external_after_hard_fail(
    tmp_path: Path,
) -> None:
    actual = run_trace_fixtures(
        tmp_path,
        Path("tests/fixtures/qc_pipeline/manifest.jsonl"),
    )
    by_asset = group_by_asset(actual)
    assert by_asset["warn"][-1]["module"] == "semantic_consistency"
    assert by_asset["warn"][-1]["pipeline_status"] == "awaiting_external"
    warn_report = json.loads(
        (tmp_path / "warn" / "quality_archive" / "warn.json").read_text(
            encoding="utf-8"
        )
    )
    assert warn_report["manual_review"]["candidate_issue_ids"] == [_WARN_ID]
    assert [issue["issue_id"] for issue in warn_report["issues"]] == [_WARN_ID]
    assert by_asset["hard-fail"][-1]["next_module"] is None
    assert by_asset["hard-fail"][-1]["exit_state"] == "stop_qc"
    assert all(row["module"] not in {"semantic_consistency", "manual_review"} for row in by_asset["hard-fail"])
    hard_fail_report = json.loads(
        (tmp_path / "hard-fail" / "quality_archive" / "hard-fail.json").read_text(
            encoding="utf-8"
        )
    )
    assert hard_fail_report["manual_review"]["state"] == "skipped_due_to_fail"
    assert hard_fail_report["execution"]["module_states"]["semantic_consistency"] == {
        "state": "skipped_due_to_fail"
    }
    assert hard_fail_report["execution"]["module_states"]["manual_review"] == {
        "state": "skipped_due_to_fail"
    }


@pytest.mark.parametrize("asset_id", ["pass", "warn"])
def test_successful_automatic_revisions_are_monotonic(
    tmp_path: Path,
    asset_id: str,
) -> None:
    rows = group_by_asset(
        run_trace_fixtures(
            tmp_path,
            Path("tests/fixtures/qc_pipeline/manifest.jsonl"),
        )
    )[asset_id]
    assert [row["revision"] for row in rows] == sorted(row["revision"] for row in rows)
    assert rows[0]["module"] == "hdf5_text_info"
    assert rows[0]["result_verdict"] == "pass"
