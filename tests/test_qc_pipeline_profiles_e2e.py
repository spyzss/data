from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from qc_common.config import LoadedQcConfig
from qc_common.contracts import Issue, ModuleResult
from qc_common.module_registry import ModulePrerequisiteError, ModuleRegistry
from qc_pipeline.context import AssetContext
from qc_reporting.aggregate import aggregate_projection
from qc_reporting.projection import project_quality_archive
from tools.run_qc_pipeline import run_batch


_CONFIG_HASH = "sha256:" + "1" * 64
_MANIFEST = Path("tests/fixtures/qc_pipeline/manifest.jsonl")
_AUTOMATIC_MODULES = (
    "hdf5_text_info",
    "quality_hand",
    "video_quality",
    "sam3_containment",
)
_PIPELINE_MODULES = (
    *_AUTOMATIC_MODULES,
    "semantic_consistency",
    "manual_review",
)


def _fixture_config() -> LoadedQcConfig:
    modules: dict[str, dict[str, Any]] = {
        module: {
            "enabled": True,
            "implementation": f"fixture.{module}",
            "parameters": {},
            "rules": {},
        }
        for module in _AUTOMATIC_MODULES
    }
    modules.update(
        {
            module: {
                "enabled": True,
                "execution_kind": "external",
                "parameters": {},
                "rules": {},
            }
            for module in ("semantic_consistency", "manual_review")
        }
    )
    return LoadedQcConfig(
        path=Path("configs/qc_acceptance.yaml").resolve(),
        raw={
            "schema_version": "qc_acceptance_config_schema.v2",
            "config_version": "qc_acceptance_v2.0.0",
            "config_name": "fixture_profile_e2e",
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
                "modules": list(_PIPELINE_MODULES),
            },
            "modules": modules,
        },
        sha256=_CONFIG_HASH,
    )


def _manifest_rows() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in _MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _context(batch_root: Path, row: dict[str, Any]) -> AssetContext:
    asset_id = str(row["asset_id"])
    return AssetContext(
        asset_id=asset_id,
        batch_root=batch_root,
        report_path=batch_root / "quality_archive" / f"{asset_id}.json",
        source_files={
            "video": {"path": str(row.get("primary_video_path") or "video/clip.mp4")}
        },
        metadata={"supplier_id": "fixture-supplier", "manifest_row": row},
    )


def _issue(module: str, verdict: str) -> Issue:
    severity = "warn" if verdict == "warn" else "fail"
    suffix = "2" * 20 if severity == "warn" else "3" * 20
    return Issue(
        f"{module}:fixture_{severity}:{suffix}",
        f"fixture.{module}.{severity}",
        severity,
        module,
        "fixture_issue",
        "fixture_metric",
        1,
        ">",
        0,
        f"fixture.{module}.{severity}",
        severity == "warn",
        {"start_frame": 10, "end_frame": 20, "source_level": "asset"},
    )


def _registry(
    config: LoadedQcConfig,
    row: dict[str, Any],
) -> ModuleRegistry:
    verdicts = row.get("verdicts", {})
    errors = row.get("errors", {})
    registry = ModuleRegistry()

    for module in _AUTOMATIC_MODULES:
        implementation = str(config.module_config(module)["implementation"])
        verdict = str(verdicts.get(module, "pass"))

        def runner(
            context: AssetContext,
            loaded: LoadedQcConfig,
            *,
            module: str = module,
            verdict: str = verdict,
        ) -> ModuleResult:
            assert loaded is config
            error_type = errors.get(module)
            if error_type == "input_missing":
                raise ModulePrerequisiteError(module, "fixture input missing")
            if error_type:
                raise RuntimeError(f"fixture:{error_type}")
            issues = (_issue(module, verdict),) if verdict in {"warn", "fail"} else ()
            return ModuleResult(
                module,
                verdict,
                {"decision": verdict},
                {"fixture": True},
                issues,
                runtime={"fixture_asset_id": context.asset_id},
            )

        registry.register(implementation, runner)
    return registry


def run_fixture_batch(
    batch_root: Path,
    *,
    profile: str,
    config: LoadedQcConfig | None = None,
) -> Path:
    """Run the shared Task19 fixture manifest through the real batch runner."""
    batch_root.mkdir(parents=True, exist_ok=True)
    loaded_config = config or _fixture_config()
    rows = _manifest_rows()
    contexts = [_context(batch_root, row) for row in rows]
    rows_by_asset = {str(row["asset_id"]): row for row in rows}

    def registry_factory(context: AssetContext) -> ModuleRegistry:
        return _registry(loaded_config, rows_by_asset[context.asset_id])

    outcomes = run_batch(
        contexts,
        config=loaded_config,
        profile=profile,
        registry_factory=registry_factory,
        max_workers=4,
        resume=False,
    )
    assert set(outcomes) == set(rows_by_asset)
    return batch_root


def read_report(batch_root: Path, asset_id: str) -> dict[str, Any]:
    return json.loads(
        (batch_root / "quality_archive" / f"{asset_id}.json").read_text(
            encoding="utf-8"
        )
    )


def normalize_machine_issues(report: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        sorted(
            json.dumps(issue, ensure_ascii=False, sort_keys=True)
            for issue in report.get("issues", [])
        )
    )


def test_same_batch_has_profile_specific_flow_and_same_machine_findings(
    tmp_path: Path,
) -> None:
    config = _fixture_config()
    acceptance = run_fixture_batch(
        tmp_path / "acceptance",
        profile="acceptance",
        config=config,
    )
    supplier = run_fixture_batch(
        tmp_path / "supplier",
        profile="supplier_evaluation",
        config=config,
    )

    for row in _manifest_rows():
        asset_id = str(row["asset_id"])
        assert normalize_machine_issues(read_report(acceptance, asset_id)) == (
            normalize_machine_issues(read_report(supplier, asset_id))
        )

    a_report = read_report(acceptance, "hard-fail")
    s_report = read_report(supplier, "hard-fail")
    assert a_report["pipeline_state"]["status"] == "stopped"
    assert a_report["overall_decision"] == "fail"
    assert "sam3_containment" not in a_report
    assert s_report["sam3_containment"]["flow"]["result_gate"]["verdict"] == "pass"
    assert s_report["pipeline_state"]["status"] == "awaiting_external"
    assert s_report["overall_decision"] is None
    assert s_report["manual_review"]["failures_for_batch_stats_issue_ids"]

    a_stats = aggregate_projection(
        project_quality_archive(acceptance / "quality_archive")
    )
    s_stats = aggregate_projection(
        project_quality_archive(supplier / "quality_archive")
    )
    assert (
        a_stats["overall"]["module_coverage"]["sam3_containment"]
        < s_stats["overall"]["module_coverage"]["sam3_containment"]
    )
    assert a_stats["overall"]["automatic_hard_fail_issue_count"] == (
        s_stats["overall"]["automatic_hard_fail_issue_count"]
    )
    assert a_stats["overall"]["final_fail_asset_count"] == 1
    assert s_stats["overall"]["automatic_hard_fail_asset_count"] == 1


def test_batch_rejects_duplicate_fixture_asset_ids_before_workers(
    tmp_path: Path,
) -> None:
    config = _fixture_config()
    row = _manifest_rows()[0]
    contexts = [_context(tmp_path, row), _context(tmp_path, row)]

    with pytest.raises(ValueError, match="duplicate asset_id: pass"):
        run_batch(
            contexts,
            config=config,
            profile="acceptance",
            registry_factory=lambda context: _registry(config, row),
            max_workers=2,
            resume=False,
        )


def test_supplier_fail_is_retained_while_other_assets_complete(
    tmp_path: Path,
) -> None:
    config = _fixture_config()
    supplier = run_fixture_batch(
        tmp_path / "supplier",
        profile="supplier_evaluation",
        config=config,
    )
    hard_fail = read_report(supplier, "hard-fail")
    passed = read_report(supplier, "pass")

    assert hard_fail["issues"]
    assert hard_fail["issues"][0]["severity"] == "fail"
    assert hard_fail["pipeline_state"]["status"] == "awaiting_external"
    assert hard_fail["execution"]["module_states"]["hdf5_text_info"] == {
        "state": "completed"
    }
    assert hard_fail["hdf5_text_info"]["flow"]["exit_gate"]["state"] == "continue"
    assert passed["pipeline_state"]["status"] == "awaiting_external"


def test_batch_runtime_error_does_not_cancel_sibling_fixture(
    tmp_path: Path,
) -> None:
    config = _fixture_config()
    supplier = run_fixture_batch(
        tmp_path / "supplier",
        profile="supplier_evaluation",
        config=config,
    )
    errored = read_report(supplier, "runtime-error")
    healthy = read_report(supplier, "pass")

    assert errored["pipeline_state"]["status"] == "error"
    assert errored["overall_decision"] is None
    assert healthy["pipeline_state"]["status"] == "awaiting_external"


def test_fixture_batch_reports_have_independent_revision_cursors(tmp_path: Path) -> None:
    config = _fixture_config()
    supplier = run_fixture_batch(
        tmp_path / "supplier",
        profile="supplier_evaluation",
        config=config,
    )
    revisions = {
        asset_id: read_report(supplier, asset_id)["report_revision"]
        for asset_id in ("pass", "warn", "hard-fail", "runtime-error")
    }

    assert revisions["pass"] == revisions["warn"] == revisions["hard-fail"] == 5
    assert revisions["runtime-error"] == 2
