from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from qc_common.config import LoadedQcConfig
from qc_common.contracts import ModuleResult
from qc_common.frame_survival import FrameExclusion
from qc_common.module_registry import ModuleRegistry
from qc_common.module_registry import ModulePrerequisiteError
from qc_common.types import CheckResult
from qc_pipeline.context import AssetContext
from qc_pipeline.orchestrator import run_asset
from qc_pipeline.runners import precheck


def _context(
    tmp_path: Path,
    asset_id: str = "asset-a",
    *,
    source_range: tuple[int, int] | None = None,
    profile: str | None = None,
) -> AssetContext:
    source = tmp_path / "source" / "clip.hdf5"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"synthetic-source-identity")
    return AssetContext(
        asset_id=asset_id,
        batch_root=tmp_path,
        report_path=tmp_path / "quality_archive" / f"{asset_id}.json",
        source_files={"hdf5": {"path": "source/clip.hdf5"}},
        source_range=source_range,
        metadata={
            "supplier": "xjgt",
            **({"profile": profile} if profile is not None else {}),
        },
    )


def _config(tmp_path: Path) -> LoadedQcConfig:
    modules = list(precheck.MODULES)
    return LoadedQcConfig(
        path=tmp_path / "qc.yaml",
        raw={
            "schema_version": "qc_acceptance_config_schema.v2",
            "config_version": "qc_acceptance_v2.1.0",
            "config_name": "test",
            "execution_profiles": {
                "acceptance": {
                    "fail_action": "stop",
                    "runtime_error_action": "stop_incomplete",
                },
                "supplier_evaluation": {
                    "fail_action": "record_and_continue",
                    "runtime_error_action": "record_and_continue",
                },
            },
            "pipeline": {"default_profile": "acceptance", "modules": modules},
            "modules": {
                name: {
                    "enabled": True,
                    "implementation": f"precheck.{name}",
                    "parameters": {},
                    "rules": {},
                }
                for name in modules
            },
        },
        sha256="sha256:" + "2" * 64,
    )


def _registry(session: object, config: LoadedQcConfig) -> ModuleRegistry:
    registry = ModuleRegistry()
    for module in precheck.MODULES:
        registry.register(
            str(config.module_config(module)["implementation"]),
            session.runner_for(module),
        )
    return registry


def _enable_frame_survival(config: LoadedQcConfig) -> None:
    config.raw["acceptance_policy"] = {
        "frame_survival": {
            "version": "frame_survival_v1",
            "enabled": True,
            "min_remaining_frame_ratio": 0.90,
            "stop_when_below": True,
            "exclude_severities": ["fail"],
            "skeleton_dependent_modules": [
                "keypoint_morphology",
                "keypoint_temporal",
                "sam3_containment",
                "manual_review",
            ],
            "terminal_asset_fail_modules": [],
        }
    }


def test_acceptance_session_applies_presence_then_morphology_exclusions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_eligibility: dict[str, tuple[tuple[int, int], ...]] = {}

    def execute(
        context: AssetContext,
        config: LoadedQcConfig,
        module: str,
        clip: object,
    ) -> precheck.PrecheckModuleExecution:
        seen_eligibility[module] = tuple(
            getattr(clip, "eligible_frame_ranges", ())
        )
        exclusions = {
            "keypoint_presence": (
                FrameExclusion(
                    start_frame=20,
                    end_frame=29,
                    module="keypoint_presence",
                    reason="nonfinite_keypoints",
                    raw_severity="fail",
                    hand_side="left",
                    first_introduced_stage="keypoint_presence",
                ),
            ),
            "keypoint_morphology": (
                FrameExclusion(
                    start_frame=100,
                    end_frame=299,
                    module="keypoint_morphology",
                    reason="keypoint_morphology.joint_angle_min_deg",
                    raw_severity="fail",
                    hand_side="right",
                    first_introduced_stage="keypoint_morphology",
                ),
            ),
        }.get(module, ())
        verdict = "fail" if module == "keypoint_morphology" else "pass"
        return precheck.PrecheckModuleExecution(
            result=ModuleResult(
                module,
                verdict,
                {"decision": verdict},
                {},
                frame_exclusions=exclusions,
            ),
            check_results=(CheckResult(module, 0, -1, {}, False, "ok"),),
        )

    monkeypatch.setattr(
        precheck,
        "_load_clip",
        lambda context, module: SimpleNamespace(num_frames=10_000),
    )
    monkeypatch.setattr(precheck, "_run_module_on_clip", execute)
    config = _config(tmp_path)
    _enable_frame_survival(config)
    context = _context(tmp_path, source_range=(0, 10_000))
    session = precheck.PrecheckSession(context, config)

    presence = session.run_module("keypoint_presence")
    morphology = session.run_module("keypoint_morphology")
    temporal = session.run_module("keypoint_temporal")

    assert seen_eligibility["keypoint_presence"] == ((0, 9_999),)
    assert seen_eligibility["keypoint_morphology"] == ((0, 19), (30, 9_999))
    assert seen_eligibility["keypoint_temporal"] == ((0, 19), (30, 99), (300, 9_999))
    assert morphology.verdict == "fail"
    assert presence.evaluation["frame_survival"] == {
        "original_frame_count": 10_000,
        "eligible_frame_count_before_module": 10_000,
        "newly_excluded_frame_count": 10,
        "cumulative_excluded_frame_count": 10,
        "remaining_frame_count": 9_990,
        "remaining_frame_ratio": 0.999,
        "raw_exclusions": [
            {
                "start_frame": 20,
                "end_frame": 29,
                "module": "keypoint_presence",
                "reason": "nonfinite_keypoints",
                "raw_severity": "fail",
                "hand_side": "left",
                "first_introduced_stage": "keypoint_presence",
            }
        ],
        "cumulative_excluded_frame_ranges": [[20, 29]],
        "eligible_frame_ranges": [[0, 19], [30, 9_999]],
        "stop_threshold": 0.90,
        "stop_triggered": False,
        "stop_trigger_module": None,
        "stop_reason": None,
    }
    assert morphology.evaluation["frame_survival"]["remaining_frame_count"] == 9_790
    assert morphology.evaluation["frame_survival"]["remaining_frame_ratio"] == pytest.approx(
        0.979
    )


def test_manifest_inclusive_range_counts_nonzero_start_and_last_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.run_qc_pipeline import contexts_from_manifest

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {"asset_id": "asset-a", "start_frame": 37, "end_frame": 41}
        )
        + "\n",
        encoding="utf-8",
    )
    context = contexts_from_manifest(manifest, batch_root=tmp_path)[0]
    config = _config(tmp_path)
    _enable_frame_survival(config)
    monkeypatch.setattr(
        precheck,
        "_load_clip",
        lambda context, module: SimpleNamespace(
            num_frames=5,
            frame_indices=[37, 38, 39, 40, 41],
            clip_start_frame=37,
        ),
    )
    monkeypatch.setattr(
        precheck,
        "_run_module_on_clip",
        lambda context, config, module, clip: ModuleResult(
            module,
            "fail" if module == "keypoint_presence" else "pass",
            {},
            {},
            frame_exclusions=(
                FrameExclusion(
                    start_frame=41,
                    end_frame=41,
                    module="keypoint_presence",
                    reason="nonfinite_keypoints",
                    raw_severity="fail",
                    hand_side="left",
                    first_introduced_stage="keypoint_presence",
                ),
            )
            if module == "keypoint_presence"
            else (),
        ),
    )

    result = precheck.PrecheckSession(context, config).run_module(
        "keypoint_presence"
    )

    assert context.source_range == (37, 42)
    assert result.evaluation["frame_survival"]["original_frame_count"] == 5
    assert result.evaluation["frame_survival"]["remaining_frame_count"] == 4
    assert result.evaluation["frame_survival"]["eligible_frame_ranges"] == [[37, 40]]


def test_acceptance_e2e_one_morphology_fail_keeps_independent_modules_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    eligibility: dict[str, tuple[tuple[int, int], ...]] = {}
    context = _context(tmp_path, source_range=(1_000, 11_000))
    config = _config(tmp_path)
    config.raw["pipeline"]["modules"].append("video_quality")
    config.raw["modules"]["video_quality"] = {
        "enabled": True,
        "implementation": "video_quality.test",
        "parameters": {},
        "rules": {},
    }
    _enable_frame_survival(config)

    monkeypatch.setattr(
        precheck,
        "_load_clip",
        lambda context, module: SimpleNamespace(
            num_frames=10_000,
            frame_indices=list(range(1_000, 11_000)),
            clip_start_frame=1_000,
        ),
    )

    def execute(
        context: AssetContext,
        config: LoadedQcConfig,
        module: str,
        clip: object,
    ) -> ModuleResult:
        calls.append(module)
        eligibility[module] = tuple(getattr(clip, "eligible_frame_ranges", ()))
        return ModuleResult(
            module,
            "fail" if module == "keypoint_morphology" else "pass",
            {"decision": "fail" if module == "keypoint_morphology" else "pass"},
            {},
            frame_exclusions=(
                FrameExclusion(
                    start_frame=1_042,
                    end_frame=1_042,
                    module="keypoint_morphology",
                    reason="keypoint_morphology.joint_angle_min_deg",
                    raw_severity="fail",
                    hand_side="left",
                    first_introduced_stage="keypoint_morphology",
                ),
            )
            if module == "keypoint_morphology"
            else (),
        )

    monkeypatch.setattr(precheck, "_run_module_on_clip", execute)
    session = precheck.PrecheckSession(context, config)
    registry = _registry(session, config)
    registry.register(
        "video_quality.test",
        lambda context, config: calls.append("video_quality")
        or ModuleResult("video_quality", "pass", {"decision": "pass"}, {}),
    )

    outcome = run_asset(
        context,
        config=config,
        profile="acceptance",
        registry=registry,
        now=lambda: "2026-07-16T00:00:00Z",
    )

    assert calls == [*precheck.MODULES, "video_quality"]
    assert outcome.report["keypoint_morphology"]["flow"]["result_gate"]["verdict"] == "fail"
    assert outcome.report["keypoint_morphology"]["flow"]["exit_gate"]["continue_to_next_module"] is True
    assert outcome.report["overall_decision"] == "pass"
    assert outcome.report["acceptance_frame_survival"]["remaining_frame_count"] == 9_999
    assert outcome.report["acceptance_frame_survival"]["remaining_frame_ratio"] == pytest.approx(0.9999)
    assert eligibility["keypoint_temporal"] == ((1_000, 1_041), (1_043, 10_999))


def test_supplier_evaluation_loads_source_once_for_five_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads: list[str] = []
    executions: list[str] = []
    clip = object()

    def load(context: AssetContext, module: str) -> object:
        loads.append(module)
        return clip

    def execute(
        context: AssetContext,
        config: LoadedQcConfig,
        module: str,
        loaded_clip: object,
    ) -> ModuleResult:
        assert loaded_clip is clip
        executions.append(module)
        return ModuleResult(module, "pass", {"decision": "pass"}, {})

    monkeypatch.setattr(precheck, "_load_clip", load)
    monkeypatch.setattr(precheck, "_run_module_on_clip", execute)
    context = _context(tmp_path)
    config = _config(tmp_path)
    session = precheck.PrecheckSession(context, config)

    outcome = run_asset(
        context,
        config=config,
        profile="supplier_evaluation",
        registry=_registry(session, config),
        now=lambda: "2026-07-15T00:00:00Z",
    )

    assert loads == ["hdf5_text_info"]
    assert executions == list(precheck.MODULES)
    assert outcome.executed_modules == precheck.MODULES
    assert outcome.status == "completed"
    for module in precheck.MODULES:
        assert outcome.report[module]["flow"]["result_gate"]["verdict"] == "pass"


def test_supplier_evaluation_keeps_all_frames_eligible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_eligibility: dict[str, object] = {}

    def execute(
        context: AssetContext,
        config: LoadedQcConfig,
        module: str,
        clip: object,
    ) -> precheck.PrecheckModuleExecution:
        seen_eligibility[module] = getattr(clip, "eligible_frame_ranges", None)
        return precheck.PrecheckModuleExecution(
            result=ModuleResult(
                module,
                "fail" if module == "keypoint_presence" else "pass",
                {},
                {},
                frame_exclusions=(
                    FrameExclusion(
                        start_frame=10,
                        end_frame=20,
                        module="keypoint_presence",
                        reason="nonfinite_keypoints",
                        raw_severity="fail",
                        hand_side="left",
                        first_introduced_stage="keypoint_presence",
                    ),
                )
                if module == "keypoint_presence"
                else (),
            ),
            check_results=(CheckResult(module, 0, -1, {}, False, "ok"),),
        )

    monkeypatch.setattr(
        precheck,
        "_load_clip",
        lambda context, module: SimpleNamespace(num_frames=100),
    )
    monkeypatch.setattr(precheck, "_run_module_on_clip", execute)
    config = _config(tmp_path)
    _enable_frame_survival(config)
    session = precheck.PrecheckSession(
        _context(
            tmp_path,
            source_range=(0, 100),
            profile="supplier_evaluation",
        ),
        config,
    )

    presence = session.run_module("keypoint_presence")
    session.run_module("keypoint_morphology")

    assert seen_eligibility == {
        "keypoint_presence": None,
        "keypoint_morphology": None,
    }
    assert "frame_survival" not in presence.evaluation


def test_supplier_evaluation_keeps_source_indices_for_temporal_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_source_indices: tuple[int, ...] | None = None

    def execute(
        context: AssetContext,
        config: LoadedQcConfig,
        module: str,
        clip: object,
    ) -> ModuleResult:
        nonlocal seen_source_indices
        if module == "keypoint_temporal":
            seen_source_indices = tuple(clip.source_frame_indices)
        return ModuleResult(module, "pass", {}, {})

    monkeypatch.setattr(
        precheck,
        "_load_clip",
        lambda context, module: SimpleNamespace(
            num_frames=3,
            frame_indices=[50, 51, 52],
            clip_start_frame=50,
        ),
    )
    monkeypatch.setattr(precheck, "_run_module_on_clip", execute)
    config = _config(tmp_path)
    _enable_frame_survival(config)
    session = precheck.PrecheckSession(
        _context(
            tmp_path,
            source_range=(50, 53),
            profile="supplier_evaluation",
        ),
        config,
    )

    session.run_module("keypoint_temporal")

    assert seen_source_indices == (50, 51, 52)


def test_acceptance_fail_does_not_preexecute_later_prechecks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads = 0
    executions: list[str] = []

    def load(context: AssetContext, module: str) -> object:
        nonlocal loads
        loads += 1
        return object()

    def execute(
        context: AssetContext,
        config: LoadedQcConfig,
        module: str,
        loaded_clip: object,
    ) -> ModuleResult:
        executions.append(module)
        verdict = "fail" if module == "quality_hand" else "pass"
        return ModuleResult(module, verdict, {"decision": verdict}, {})

    monkeypatch.setattr(precheck, "_load_clip", load)
    monkeypatch.setattr(precheck, "_run_module_on_clip", execute)
    context = _context(tmp_path)
    config = _config(tmp_path)
    session = precheck.PrecheckSession(context, config)

    outcome = run_asset(
        context,
        config=config,
        profile="acceptance",
        registry=_registry(session, config),
        now=lambda: "2026-07-15T00:00:00Z",
    )

    assert loads == 1
    assert executions == ["hdf5_text_info", "quality_hand"]
    assert outcome.status == "stopped"
    assert outcome.report["execution"]["module_states"]["keypoint_presence"] == {
        "state": "not_run",
        "reason": "blocked_by_quality_fail:quality_hand",
    }


def test_session_caches_same_module_result_without_rerunning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executions: list[str] = []
    monkeypatch.setattr(precheck, "_load_clip", lambda context, module: object())

    def execute(
        context: AssetContext,
        config: LoadedQcConfig,
        module: str,
        loaded_clip: object,
    ) -> ModuleResult:
        executions.append(module)
        return ModuleResult(module, "pass", {}, {})

    monkeypatch.setattr(precheck, "_run_module_on_clip", execute)
    context = _context(tmp_path)
    config = _config(tmp_path)
    session = precheck.PrecheckSession(context, config)
    runner = session.runner_for("hdf5_text_info")

    first = runner(context, config)
    second = runner(context, config)

    assert first is second
    assert executions == ["hdf5_text_info"]


def test_sessions_are_isolated_by_exact_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads: list[str] = []
    monkeypatch.setattr(
        precheck,
        "_load_clip",
        lambda context, module: loads.append(context.asset_id) or object(),
    )
    monkeypatch.setattr(
        precheck,
        "_run_module_on_clip",
        lambda context, config, module, clip: ModuleResult(module, "pass", {}, {}),
    )
    config = _config(tmp_path)
    first_context = _context(tmp_path, "first")
    second_context = _context(tmp_path, "second")
    first = precheck.PrecheckSession(first_context, config)
    second = precheck.PrecheckSession(second_context, config)

    first.runner_for("hdf5_text_info")(first_context, config)
    second.runner_for("hdf5_text_info")(second_context, config)

    assert loads == ["first", "second"]
    with pytest.raises(ValueError, match="cannot be shared across assets"):
        first.runner_for("quality_hand")(second_context, config)


def test_completed_session_publishes_canonical_precheck_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(precheck, "_load_clip", lambda context, module: object())

    def execute(
        context: AssetContext,
        config: LoadedQcConfig,
        module: str,
        clip: object,
    ) -> object:
        candidates = (
            [
                {
                    "asset_id": context.asset_id,
                    "supplier": "jdt",
                    "coordinate_space": "source",
                    "frame_coordinate_system": "source_inclusive",
                    "start_frame": 10,
                    "end_frame": 12,
                }
            ]
            if module == "keypoint_temporal"
            else []
        )
        return precheck.PrecheckModuleExecution(
            result=ModuleResult(module, "pass", {}, {}),
            check_results=(CheckResult(module, 0, -1, {}, False, "ok"),),
            candidate_windows=tuple(candidates),
        )

    monkeypatch.setattr(precheck, "_run_module_on_clip", execute)
    context = _context(tmp_path)
    config = _config(tmp_path)
    session = precheck.PrecheckSession(context, config)

    for module in precheck.MODULES:
        session.runner_for(module)(context, config)

    artifact = tmp_path / "module_outputs" / "asset-a" / "precheck"
    check_rows = json.loads((artifact / "check_results.json").read_text())
    candidates = json.loads((artifact / "candidate_windows.json").read_text())
    run_config = json.loads((artifact / "run_config.json").read_text())
    assert {row["pipeline_module"] for row in check_rows} == set(precheck.MODULES)
    assert candidates == [
        {
            "asset_id": "asset-a",
            "supplier": "jdt",
            "coordinate_space": "source",
            "frame_coordinate_system": "source_inclusive",
            "start_frame": 10,
            "end_frame": 12,
        }
    ]
    assert run_config["outcome"] == "completed"
    assert run_config["fingerprint"]["implementation_version"] == (
        "precheck-session-v4-frame-survival-lineage"
    )


def test_temporal_success_publishes_empty_candidate_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(precheck, "_load_clip", lambda context, module: object())
    monkeypatch.setattr(
        precheck,
        "_run_module_on_clip",
        lambda context, config, module, clip: precheck.PrecheckModuleExecution(
            result=ModuleResult(module, "pass", {}, {}),
            check_results=(CheckResult(module, 0, -1, {}, False, "ok"),),
            candidate_windows=(),
        ),
    )
    context = _context(tmp_path)
    config = _config(tmp_path)
    session = precheck.PrecheckSession(context, config)

    for module in precheck.MODULES:
        session.run_module(module)

    candidate_path = (
        tmp_path / "module_outputs" / "asset-a" / "precheck" / "candidate_windows.json"
    )
    assert json.loads(candidate_path.read_text()) == []


def test_matching_precheck_artifact_reuses_without_source_load_or_check_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    config = _config(tmp_path)
    monkeypatch.setattr(precheck, "_load_clip", lambda context, module: object())
    monkeypatch.setattr(
        precheck,
        "_run_module_on_clip",
        lambda context, config, module, clip: precheck.PrecheckModuleExecution(
            result=ModuleResult(module, "pass", {}, {}),
            check_results=(CheckResult(module, 0, -1, {}, False, "ok"),),
            candidate_windows=(),
        ),
    )
    first = precheck.PrecheckSession(context, config)
    for module in precheck.MODULES:
        first.run_module(module)

    monkeypatch.setattr(
        precheck,
        "_load_clip",
        lambda context, module: pytest.fail("cache hit must not load source"),
    )
    monkeypatch.setattr(
        precheck,
        "_run_module_on_clip",
        lambda context, config, module, clip: pytest.fail("cache hit must not run check"),
    )
    adapted: list[str] = []

    def adapt(
        context: AssetContext,
        config: LoadedQcConfig,
        module: str,
        results: object,
        candidates: object,
        *,
        artifact_state: str,
    ) -> ModuleResult:
        adapted.append(module)
        assert artifact_state == "reused"
        return ModuleResult(module, "pass", {}, {}, runtime={"artifact_state": "reused"})

    monkeypatch.setattr(precheck, "_adapt_module", adapt)
    second = precheck.PrecheckSession(context, config)

    results = [second.run_module(module) for module in precheck.MODULES]

    assert adapted == list(precheck.MODULES)
    assert all(result.runtime["artifact_state"] == "reused" for result in results)


def test_source_identity_change_invalidates_precheck_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    config = _config(tmp_path)
    loads: list[str] = []
    monkeypatch.setattr(
        precheck,
        "_load_clip",
        lambda context, module: loads.append(module) or object(),
    )
    monkeypatch.setattr(
        precheck,
        "_run_module_on_clip",
        lambda context, config, module, clip: precheck.PrecheckModuleExecution(
            result=ModuleResult(module, "pass", {}, {}),
            check_results=(CheckResult(module, 0, -1, {}, False, "ok"),),
            candidate_windows=(),
        ),
    )
    first = precheck.PrecheckSession(context, config)
    for module in precheck.MODULES:
        first.run_module(module)
    loads.clear()
    (tmp_path / "source" / "clip.hdf5").write_bytes(b"changed-source-identity")

    second = precheck.PrecheckSession(context, config)
    second.run_module("hdf5_text_info")

    assert loads == ["hdf5_text_info"]


def test_temporal_publishes_current_partial_artifact_after_prior_module_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    config = _config(tmp_path)
    monkeypatch.setattr(precheck, "_load_clip", lambda context, module: object())
    monkeypatch.setattr(
        precheck,
        "_run_module_on_clip",
        lambda context, config, module, clip: precheck.PrecheckModuleExecution(
            result=ModuleResult(module, "pass", {}, {}),
            check_results=(CheckResult(module, 0, -1, {}, False, "ok"),),
            candidate_windows=(
                (
                    {
                        "asset_id": context.asset_id,
                        "coordinate_space": "source",
                        "frame_coordinate_system": "source_inclusive",
                        "start_frame": 3,
                        "end_frame": 5,
                    },
                )
                if module == "keypoint_temporal"
                else ()
            ),
        ),
    )
    session = precheck.PrecheckSession(context, config)
    for module in precheck.MODULES:
        if module != "quality_hand":
            session.run_module(module)

    artifact = tmp_path / "module_outputs" / "asset-a" / "precheck"
    run_config = json.loads((artifact / "run_config.json").read_text())
    candidates = json.loads((artifact / "candidate_windows.json").read_text())

    assert run_config["outcome"] == "partial"
    assert "quality_hand" not in run_config["completed_modules"]
    assert "keypoint_temporal" in run_config["completed_modules"]
    assert candidates[0]["start_frame"] == 3


def test_missing_declared_source_remains_structured_input_missing(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    (tmp_path / "source" / "clip.hdf5").unlink()
    session = precheck.PrecheckSession(context, _config(tmp_path))

    with pytest.raises(ModulePrerequisiteError) as raised:
        session.run_module("hdf5_text_info")

    assert raised.value.module == "hdf5_text_info"
    assert "existing source_files.hdf5.path" in raised.value.prerequisite
