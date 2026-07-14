import json
from dataclasses import replace
from pathlib import Path

import pytest

import qc_common.report_mutation as report_mutation
from qc_common.config import LoadedQcConfig
from qc_common.contracts import EvidenceRef, Issue, ModuleResult
from qc_common.report import StaleReportRevisionError
from qc_common.report_mutation import (
    ConfigDriftError,
    ModuleOrderError,
    apply_module_result,
    initialize_v2_report,
    write_pipeline_transition,
)
from qc_common.schema import validate_asset_qc_report
from qc_pipeline.context import AssetContext
from tests.qc_report_fixtures import make_asset_context, loaded_test_config


def advance_to_module(
    context: AssetContext,
    *,
    target_module: str,
    profile: str = "acceptance",
) -> tuple[LoadedQcConfig, int]:
    config = loaded_test_config()
    revision = 0
    modules = config.pipeline_modules
    target_index = modules.index(target_module)
    for index, module in enumerate(modules[:target_index]):
        next_module = modules[index + 1]
        if module in {"semantic_consistency", "manual_review"}:
            report = write_pipeline_transition(
                context.report_path,
                expected_revision=revision,
                module=module,
                state="running",
                next_module=next_module,
                stop_reason=None,
                overall_decision=None,
                now=f"2026-07-14T00:00:{revision:02d}Z",
            )
        else:
            report = apply_module_result(
                context.report_path,
                context=context,
                config=config,
                profile=profile,
                result=ModuleResult(module, "pass", {}, {}),
                expected_revision=revision,
                next_module=next_module,
                now=f"2026-07-14T00:00:{revision:02d}Z",
            )
        revision = report["report_revision"]
    return config, revision


def test_module_rerun_replaces_only_owned_block_and_rebuilds_candidates(
    tmp_path: Path,
) -> None:
    context = make_asset_context(tmp_path, "a")
    config, revision = advance_to_module(
        context,
        target_module="keypoint_temporal",
    )
    issue = Issue(
        "keypoint_temporal:jump:11111111111111111111",
        "jump",
        "warn",
        "keypoint_temporal",
        "temporal_jump",
        "joint_displacement_m_max",
        0.2,
        ">",
        0.05,
        "keypoint_temporal.strong_temporal_failure",
        True,
        {"start_frame": 4, "end_frame": 8},
    )
    first = apply_module_result(
        context.report_path,
        context=context,
        config=config,
        profile="acceptance",
        result=ModuleResult(
            "keypoint_temporal", "warn", {}, {"run": 1}, (issue,)
        ),
        expected_revision=revision,
        next_module="video_quality",
        now="2026-07-14T00:00:00Z",
    )
    first["extension"] = {"keep": True}
    context.report_path.write_text(json.dumps(first), encoding="utf-8")
    second = apply_module_result(
        context.report_path,
        context=context,
        config=config,
        profile="acceptance",
        result=ModuleResult("keypoint_temporal", "pass", {}, {"run": 2}),
        expected_revision=revision + 1,
        next_module="video_quality",
        now="2026-07-14T00:01:00Z",
    )
    assert second["extension"] == {"keep": True}
    assert second["keypoint_temporal"]["metrics"] == {"run": 2}
    assert second["issues"] == []
    assert second["manual_review"]["candidate_issue_ids"] == []


def test_stale_revision_never_changes_file(tmp_path: Path) -> None:
    context = make_asset_context(tmp_path, "a")
    config = loaded_test_config()
    apply_module_result(
        context.report_path,
        context=context,
        config=config,
        profile="acceptance",
        result=ModuleResult("hdf5_text_info", "pass", {}, {}),
        expected_revision=0,
        next_module="quality_hand",
        now="2026-07-14T00:00:00Z",
    )
    before = context.report_path.read_bytes()
    with pytest.raises(StaleReportRevisionError):
        apply_module_result(
            context.report_path,
            context=context,
            config=config,
            profile="acceptance",
            result=ModuleResult("hdf5_text_info", "pass", {}, {}),
            expected_revision=0,
            next_module="quality_hand",
            now="2026-07-14T00:00:01Z",
        )
    assert context.report_path.read_bytes() == before


def test_initialize_v2_report_is_uncommitted_and_source_faithful(
    tmp_path: Path,
) -> None:
    context = make_asset_context(tmp_path, "a")
    report = initialize_v2_report(
        context,
        loaded_test_config(),
        "acceptance",
        "2026-07-14T00:00:00Z",
    )

    assert report["schema_version"] == "asset_qc_report.v2"
    assert report["asset_id"] == "a"
    assert report["report_revision"] == 0
    assert report["execution"] == {
        "profile": "acceptance",
        "started_at": "2026-07-14T00:00:00Z",
        "updated_at": "2026-07-14T00:00:00Z",
    }
    assert report["source_files"] == context.source_files
    assert not context.report_path.exists()


def test_asset_context_rejects_report_outside_batch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="report_path must be inside batch_root"):
        AssetContext("a", tmp_path / "batch", tmp_path / "outside.json", {})


def test_evidence_path_must_stay_inside_batch_root(tmp_path: Path) -> None:
    context = make_asset_context(tmp_path, "a")
    result = ModuleResult(
        "hdf5_text_info",
        "pass",
        {},
        {},
        evidence=(
            EvidenceRef(
                "evidence-1",
                "json",
                "../outside.json",
                "source_frame_half_open",
            ),
        ),
    )

    with pytest.raises(ValueError, match="evidence path"):
        apply_module_result(
            context.report_path,
            context=context,
            config=loaded_test_config(),
            profile="acceptance",
            result=result,
            expected_revision=0,
            next_module="quality_hand",
            now="2026-07-14T00:00:00Z",
        )
    assert not context.report_path.exists()


def test_config_drift_never_changes_file(tmp_path: Path) -> None:
    context = make_asset_context(tmp_path, "a")
    config = loaded_test_config()
    apply_module_result(
        context.report_path,
        context=context,
        config=config,
        profile="acceptance",
        result=ModuleResult("hdf5_text_info", "pass", {}, {}),
        expected_revision=0,
        next_module="quality_hand",
        now="2026-07-14T00:00:00Z",
    )
    before = context.report_path.read_bytes()
    drifted = replace(config, sha256="sha256:" + "f" * 64)

    with pytest.raises(ConfigDriftError):
        apply_module_result(
            context.report_path,
            context=context,
            config=drifted,
            profile="acceptance",
            result=ModuleResult("hdf5_text_info", "pass", {}, {}),
            expected_revision=1,
            next_module="quality_hand",
            now="2026-07-14T00:00:01Z",
        )
    assert context.report_path.read_bytes() == before


def test_module_order_error_never_changes_file(tmp_path: Path) -> None:
    context = make_asset_context(tmp_path, "a")
    config = loaded_test_config()
    apply_module_result(
        context.report_path,
        context=context,
        config=config,
        profile="acceptance",
        result=ModuleResult("hdf5_text_info", "pass", {}, {}),
        expected_revision=0,
        next_module="quality_hand",
        now="2026-07-14T00:00:00Z",
    )
    before = context.report_path.read_bytes()

    with pytest.raises(ModuleOrderError):
        apply_module_result(
            context.report_path,
            context=context,
            config=config,
            profile="acceptance",
            result=ModuleResult("keypoint_presence", "pass", {}, {}),
            expected_revision=1,
            next_module="keypoint_morphology",
            now="2026-07-14T00:00:01Z",
        )
    assert context.report_path.read_bytes() == before


def test_fresh_report_rejects_out_of_order_first_module(tmp_path: Path) -> None:
    context = make_asset_context(tmp_path, "a")

    with pytest.raises(ModuleOrderError, match="expected current module hdf5_text_info"):
        apply_module_result(
            context.report_path,
            context=context,
            config=loaded_test_config(),
            profile="acceptance",
            result=ModuleResult("keypoint_temporal", "pass", {}, {}),
            expected_revision=0,
            next_module="video_quality",
            now="2026-07-14T00:00:00Z",
        )

    assert not context.report_path.exists()


def test_invalid_issue_is_preflighted_before_owned_candidate_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = make_asset_context(tmp_path, "a")
    invalid_issue = Issue(
        "quality_hand:wrong-owner:11111111111111111111",
        "wrong_owner",
        "warn",
        "quality_hand",
        "wrong_owner",
        "score",
        0.5,
        "<",
        0.9,
        "quality_hand.single_hand_low_quality",
        True,
    )
    replacement_called = False
    original = report_mutation._replace_owned_issues

    def track_replacement(*args: object, **kwargs: object) -> None:
        nonlocal replacement_called
        replacement_called = True
        original(*args, **kwargs)

    monkeypatch.setattr(report_mutation, "_replace_owned_issues", track_replacement)

    with pytest.raises(ValueError, match="belongs to quality_hand, not hdf5_text_info"):
        apply_module_result(
            context.report_path,
            context=context,
            config=loaded_test_config(),
            profile="acceptance",
            result=ModuleResult(
                "hdf5_text_info",
                "warn",
                {},
                {},
                (invalid_issue,),
            ),
            expected_revision=0,
            next_module="quality_hand",
            now="2026-07-14T00:00:00Z",
        )

    assert replacement_called is False
    assert not context.report_path.exists()


def test_unserializable_issue_is_preflighted_before_owned_candidate_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = make_asset_context(tmp_path, "a")
    invalid_issue = Issue(
        "hdf5_text_info:bad-context:11111111111111111111",
        "bad_context",
        "warn",
        "hdf5_text_info",
        "bad_context",
        "field_count",
        1,
        ">",
        0,
        "hdf5_text.missing_required_field",
        True,
        {"not_json": tmp_path / "evidence.json"},
    )
    replacement_called = False
    original = report_mutation._replace_owned_issues

    def track_replacement(*args: object, **kwargs: object) -> None:
        nonlocal replacement_called
        replacement_called = True
        original(*args, **kwargs)

    monkeypatch.setattr(report_mutation, "_replace_owned_issues", track_replacement)

    with pytest.raises(ValueError, match="issue .* is not JSON serializable"):
        apply_module_result(
            context.report_path,
            context=context,
            config=loaded_test_config(),
            profile="acceptance",
            result=ModuleResult(
                "hdf5_text_info",
                "warn",
                {},
                {},
                (invalid_issue,),
            ),
            expected_revision=0,
            next_module="quality_hand",
            now="2026-07-14T00:00:00Z",
        )

    assert replacement_called is False
    assert not context.report_path.exists()


def test_schema_failure_never_changes_file(tmp_path: Path) -> None:
    context = make_asset_context(tmp_path, "a")
    config = loaded_test_config()
    report = apply_module_result(
        context.report_path,
        context=context,
        config=config,
        profile="acceptance",
        result=ModuleResult("hdf5_text_info", "pass", {}, {}),
        expected_revision=0,
        next_module="quality_hand",
        now="2026-07-14T00:00:00Z",
    )
    report["execution"]["started_at"] = 1
    context.report_path.write_text(json.dumps(report), encoding="utf-8")
    before = context.report_path.read_bytes()

    with pytest.raises(ValueError, match="execution.started_at"):
        apply_module_result(
            context.report_path,
            context=context,
            config=config,
            profile="acceptance",
            result=ModuleResult("hdf5_text_info", "pass", {}, {}),
            expected_revision=1,
            next_module="quality_hand",
            now="2026-07-14T00:00:01Z",
        )
    assert context.report_path.read_bytes() == before


def test_manual_semantic_extension_is_preserved_opaque_on_rerun(
    tmp_path: Path,
) -> None:
    context = make_asset_context(tmp_path, "a")
    config, revision = advance_to_module(
        context,
        target_module="keypoint_temporal",
    )
    report = apply_module_result(
        context.report_path,
        context=context,
        config=config,
        profile="acceptance",
        result=ModuleResult("keypoint_temporal", "pass", {}, {}),
        expected_revision=revision,
        next_module="video_quality",
        now="2026-07-14T00:00:00Z",
    )
    semantic_calibration = {
        "revision_id": "semantic-r4",
        "before": [
            {"start": 0, "end": 20, "label": "reach"},
            {"start": 20, "end": 40, "label": "grasp"},
        ],
        "after": [
            {"start": 0, "end": 24, "label": "reach"},
            {"start": 24, "end": 40, "label": "grasp"},
        ],
    }
    report["semantic_calibration"] = semantic_calibration
    context.report_path.write_text(json.dumps(report), encoding="utf-8")

    updated = apply_module_result(
        context.report_path,
        context=context,
        config=config,
        profile="acceptance",
        result=ModuleResult("keypoint_temporal", "pass", {}, {"run": 2}),
        expected_revision=revision + 1,
        next_module="video_quality",
        now="2026-07-14T00:01:00Z",
    )

    assert updated["semantic_calibration"] == semantic_calibration


def test_profile_changes_exit_action_without_rewriting_machine_fail(
    tmp_path: Path,
) -> None:
    config = loaded_test_config()
    acceptance = make_asset_context(tmp_path / "acceptance", "a")
    supplier = make_asset_context(tmp_path / "supplier", "a")

    stopped = apply_module_result(
        acceptance.report_path,
        context=acceptance,
        config=config,
        profile="acceptance",
        result=ModuleResult("hdf5_text_info", "fail", {}, {}),
        expected_revision=0,
        next_module="quality_hand",
        now="2026-07-14T00:00:00Z",
    )
    continued = apply_module_result(
        supplier.report_path,
        context=supplier,
        config=config,
        profile="supplier_evaluation",
        result=ModuleResult("hdf5_text_info", "fail", {}, {}),
        expected_revision=0,
        next_module="quality_hand",
        now="2026-07-14T00:00:00Z",
    )

    assert stopped["hdf5_text_info"]["flow"]["result_gate"]["verdict"] == "fail"
    assert stopped["hdf5_text_info"]["flow"]["exit_gate"] == {
        "state": "stop_qc",
        "continue_to_next_module": False,
        "next_module": None,
    }
    assert stopped["pipeline_state"]["status"] == "stopped"
    assert stopped["overall_decision"] == "fail"
    validate_asset_qc_report(stopped)
    assert continued["hdf5_text_info"]["flow"]["result_gate"]["verdict"] == "fail"
    assert continued["hdf5_text_info"]["flow"]["exit_gate"] == {
        "state": "continue",
        "continue_to_next_module": True,
        "next_module": "quality_hand",
    }
    assert continued["pipeline_state"]["status"] == "running"
    assert continued["overall_decision"] is None
    validate_asset_qc_report(continued)


def test_inconsistent_stopped_module_rerun_keeps_report_unchanged(
    tmp_path: Path,
) -> None:
    context = make_asset_context(tmp_path, "a")
    config = loaded_test_config()
    stopped = apply_module_result(
        context.report_path,
        context=context,
        config=config,
        profile="acceptance",
        result=ModuleResult("hdf5_text_info", "fail", {}, {}),
        expected_revision=0,
        next_module="quality_hand",
        now="2026-07-14T00:00:00Z",
    )
    stopped["pipeline_state"].update(
        {"status": "running", "next_module": "quality_hand"}
    )
    context.report_path.write_text(json.dumps(stopped), encoding="utf-8")
    before = context.report_path.read_bytes()

    with pytest.raises(ModuleOrderError):
        apply_module_result(
            context.report_path,
            context=context,
            config=config,
            profile="acceptance",
            result=ModuleResult("hdf5_text_info", "fail", {}, {}),
            expected_revision=1,
            next_module="quality_hand",
            now="2026-07-14T00:01:00Z",
        )

    assert context.report_path.read_bytes() == before


def test_terminal_supplier_fail_remains_machine_fail(tmp_path: Path) -> None:
    context = make_asset_context(tmp_path, "a")
    config, revision = advance_to_module(
        context,
        target_module="effective_duration",
        profile="supplier_evaluation",
    )
    report = apply_module_result(
        context.report_path,
        context=context,
        config=config,
        profile="supplier_evaluation",
        result=ModuleResult("effective_duration", "fail", {}, {}),
        expected_revision=revision,
        next_module=None,
        now="2026-07-14T00:00:00Z",
    )

    assert report["pipeline_state"]["status"] == "completed"
    assert report["overall_decision"] == "fail"
    assert report["effective_duration"]["flow"]["result_gate"]["verdict"] == "fail"


def test_error_transition_keeps_null_quality_decision(tmp_path: Path) -> None:
    context = make_asset_context(tmp_path, "a")
    apply_module_result(
        context.report_path,
        context=context,
        config=loaded_test_config(),
        profile="acceptance",
        result=ModuleResult("hdf5_text_info", "pass", {}, {}),
        expected_revision=0,
        next_module="quality_hand",
        now="2026-07-14T00:00:00Z",
    )

    report = write_pipeline_transition(
        context.report_path,
        expected_revision=1,
        module="quality_hand",
        state="error",
        next_module=None,
        stop_reason="decoder crashed",
        overall_decision=None,
        now="2026-07-14T00:00:01Z",
    )

    assert report["report_revision"] == 2
    assert report["pipeline_state"]["status"] == "error"
    assert report["overall_decision"] is None
    validate_asset_qc_report(report)
