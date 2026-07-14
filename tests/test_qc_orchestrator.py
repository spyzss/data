from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from qc_common.config import LoadedQcConfig
from qc_common.contracts import ModuleResult
from qc_common.module_registry import ModuleRegistry, ModuleUnavailableError
from qc_common.report_mutation import (
    ConfigDriftError,
    apply_module_result,
    record_awaiting_external,
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


def test_registry_resolves_config_implementation_names() -> None:
    registry = ModuleRegistry()
    runner = lambda context, config: ModuleResult("quality_hand", "pass", {}, {})

    registry.register("precheck.quality_hand", runner)

    assert registry.has("precheck.quality_hand")
    assert registry.resolve("precheck.quality_hand") is runner
    assert not registry.has("quality_hand")
    with pytest.raises(ModuleUnavailableError, match="unavailable"):
        registry.resolve("quality_hand")


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


def test_orchestrator_resumes_exactly_from_persisted_next_module(
    tmp_path: Path,
) -> None:
    modules = ["hdf5_text_info", "quality_hand", "keypoint_presence"]
    config = _config(tmp_path, modules)
    first_calls: list[str] = []

    with pytest.raises(RuntimeError, match="failed:quality_hand"):
        run_asset(
            _context(tmp_path),
            config=config,
            profile="acceptance",
            registry=_registry(
                first_calls,
                config,
                {name: "pass" for name in modules},
                errors={"quality_hand"},
            ),
            now=lambda: "2026-07-14T00:00:00Z",
        )

    assert first_calls == ["hdf5_text_info", "quality_hand"]
    second_calls: list[str] = []
    outcome = run_asset(
        _context(tmp_path),
        config=config,
        profile="acceptance",
        registry=_registry(
            second_calls,
            config,
            {name: "pass" for name in modules},
        ),
        now=lambda: "2026-07-14T00:01:00Z",
    )

    assert second_calls == ["quality_hand", "keypoint_presence"]
    assert outcome.report["report_revision"] == 3
    assert outcome.status == "completed"


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
    assert outcome.report["pipeline_state"]["next_module"] == "semantic_consistency"


def test_unavailable_automatic_runner_never_becomes_a_pass(tmp_path: Path) -> None:
    config = _config(tmp_path, ["hdf5_text_info"])
    context = _context(tmp_path)

    with pytest.raises(ModuleUnavailableError, match="test.hdf5_text_info"):
        run_asset(
            context,
            config=config,
            profile="acceptance",
            registry=ModuleRegistry(),
            now=lambda: "2026-07-14T00:00:00Z",
        )

    assert not context.report_path.exists()


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
