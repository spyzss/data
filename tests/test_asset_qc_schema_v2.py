from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from qc_common.report import load_asset_qc_report, write_asset_qc_report
from qc_common.report_migration import migrate_v1_to_v2
from qc_common.schema import validate_asset_qc_report
from tests.qc_report_fixtures import make_v1_video_report, make_v2_report


@pytest.mark.parametrize(
    "status", ["pending", "running", "awaiting_external", "incomplete", "error"]
)
def test_unfinished_v2_report_requires_null_decision(status: str) -> None:
    report = make_v2_report(status=status, overall_decision=None)
    validate_asset_qc_report(report)
    report["overall_decision"] = "pass"
    with pytest.raises(ValueError, match="overall_decision"):
        validate_asset_qc_report(report)


def test_error_cannot_be_quality_fail() -> None:
    report = make_v2_report(status="error", overall_decision="fail")
    with pytest.raises(ValueError, match="overall_decision"):
        validate_asset_qc_report(report)


def test_stopped_v2_report_requires_fail_decision() -> None:
    validate_asset_qc_report(make_v2_report(status="stopped", overall_decision="fail"))


@pytest.mark.parametrize("decision", [None, "pass"])
def test_stopped_v2_report_rejects_non_fail_decision(decision: str | None) -> None:
    report = make_v2_report(status="stopped", overall_decision=decision)
    with pytest.raises(ValueError, match="overall_decision"):
        validate_asset_qc_report(report)


def test_runtime_error_v2_report_uses_error_state_and_null_decision() -> None:
    report = make_v2_report(status="error", overall_decision=None)
    report["runtime_errors"] = [{"module": "video_quality", "message": "decoder crashed"}]

    validate_asset_qc_report(report)


def test_supplier_runtime_error_can_finish_as_incomplete() -> None:
    report = make_v2_report(status="incomplete", overall_decision=None)
    report["execution"]["profile"] = "supplier_evaluation"
    report["runtime_errors"] = [
        {"module": "video_quality", "message": "decoder crashed"}
    ]
    report["execution"]["module_states"] = {
        "video_quality": {"state": "runtime_error", "reason": "process_error"},
        "sam3_containment": {"state": "completed"},
    }

    validate_asset_qc_report(report)


@pytest.mark.parametrize(
    ("status", "decision", "error_path"),
    [
        ("completed", "pass", "overall_decision"),
    ],
)
def test_runtime_errors_reject_non_error_pipeline_outcomes(
    status: str,
    decision: str | None,
    error_path: str,
) -> None:
    report = make_v2_report(status=status, overall_decision=decision)
    report["runtime_errors"] = [{"module": "video_quality", "message": "decoder crashed"}]

    with pytest.raises(ValueError, match=error_path):
        validate_asset_qc_report(report)


def test_supplier_runtime_error_can_remain_running_until_other_modules_finish() -> None:
    report = make_v2_report(status="running", overall_decision=None)
    report["execution"]["profile"] = "supplier_evaluation"
    report["runtime_errors"] = [
        {"module": "hdf5_text_info", "message": "source parse failed"}
    ]

    validate_asset_qc_report(report)


@pytest.mark.parametrize("decision", ["pass", "fail"])
def test_completed_v2_report_requires_quality_decision(decision: str) -> None:
    validate_asset_qc_report(make_v2_report(status="completed", overall_decision=decision))

    report = make_v2_report(status="completed", overall_decision=None)
    with pytest.raises(ValueError, match="overall_decision"):
        validate_asset_qc_report(report)


def test_migrate_v1_preserves_video_unknown_fields_and_revision() -> None:
    old = make_v1_video_report()
    old["extension_from_colleague"] = {"keep": True}
    frozen = copy.deepcopy(old)
    migrated = migrate_v1_to_v2(old, config_reference=old["qc_config"])
    assert old == frozen
    assert migrated["schema_version"] == "asset_qc_report.v2"
    assert migrated["report_revision"] == old["report_revision"]
    assert migrated["video_quality"] == old["video_quality"]
    assert migrated["extension_from_colleague"] == {"keep": True}


def test_migration_preserves_open_manual_semantic_revision_atomically() -> None:
    old = make_v1_video_report()
    semantic_revision = {
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
    old["manual_review"]["semantic_revision"] = semantic_revision

    migrated = migrate_v1_to_v2(old, config_reference=old["qc_config"])

    assert migrated["manual_review"]["semantic_revision"] == semantic_revision
    validate_asset_qc_report(migrated)


def test_migration_fills_v2_defaults_and_validates() -> None:
    old = make_v1_video_report()
    migrated = migrate_v1_to_v2(
        old,
        config_reference={
            "schema_version": "qc_acceptance_config_schema.v2",
            "config_version": "qc_acceptance_v2.0.0",
            "config_name": "acceptance_gate",
            "config_path": "configs/qc_acceptance.yaml",
            "config_hash": "sha256:" + "2" * 64,
        },
        profile="supplier_evaluation",
    )

    assert migrated["execution"] == {
        "profile": "supplier_evaluation",
        "started_at": None,
        "updated_at": None,
    }
    assert migrated["source_files"] == {}
    assert migrated["runtime_errors"] == []
    assert migrated["pipeline_state"]["stop_reason"] is None
    validate_asset_qc_report(migrated)


def test_migration_rejects_unknown_report_version() -> None:
    with pytest.raises(ValueError, match="unsupported asset QC schema: future"):
        migrate_v1_to_v2(
            {"schema_version": "future"},
            config_reference={},
        )


def test_migrating_v2_returns_defensive_copy() -> None:
    report = make_v2_report()
    migrated = migrate_v1_to_v2(report, config_reference={})

    assert migrated == report
    assert migrated is not report
    assert migrated["manual_review"] is not report["manual_review"]


def test_load_v1_does_not_project_or_modify_file(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    report = make_v1_video_report()
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    frozen_bytes = path.read_bytes()

    loaded = load_asset_qc_report(path)

    assert loaded == report
    assert loaded["schema_version"] == "asset_qc_report.v1"
    assert path.read_bytes() == frozen_bytes


def test_load_v1_projects_only_when_explicitly_requested(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    report = make_v1_video_report()
    path.write_text(json.dumps(report), encoding="utf-8")
    frozen_bytes = path.read_bytes()

    migrated = load_asset_qc_report(
        path,
        migrate_to_v2=True,
        config_reference=report["qc_config"],
    )

    assert migrated is not None
    assert migrated["schema_version"] == "asset_qc_report.v2"
    validate_asset_qc_report(migrated)
    assert path.read_bytes() == frozen_bytes


def test_load_v1_projection_requires_config_reference(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(make_v1_video_report()), encoding="utf-8")

    with pytest.raises(ValueError, match="config_reference"):
        load_asset_qc_report(path, migrate_to_v2=True)


def test_load_projection_rejects_unknown_report_version(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"schema_version": "future"}), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported asset QC schema: future"):
        load_asset_qc_report(path, migrate_to_v2=True)


def test_v2_schema_keeps_registered_and_unknown_module_blocks_open() -> None:
    report = make_v2_report()
    report["semantic_consistency"] = {"future": {"segments": [1, 2]}}
    report["unregistered_future_module"] = {"payload": {"keep": True}}

    validate_asset_qc_report(report)


def test_write_promotes_v1_report_with_explicit_acceptance_profile(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    report = make_v1_video_report()
    report["qc_config"] = {
        "schema_version": "qc_acceptance_config_schema.v2",
        "config_version": "qc_acceptance_v2.0.0",
        "config_name": "acceptance_gate",
        "config_path": "configs/qc_acceptance.yaml",
        "config_hash": "sha256:" + "3" * 64,
    }

    write_asset_qc_report(path, report, expected_revision=0, profile="acceptance")

    written = json.loads(path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "asset_qc_report.v1"
    assert written["schema_version"] == "asset_qc_report.v2"
    assert written["qc_config"] == report["qc_config"]
    assert written["execution"]["profile"] == "acceptance"
    validate_asset_qc_report(written)


def test_write_rejects_config_v2_promotion_without_profile(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    report = make_v1_video_report()
    report["qc_config"] = {
        "schema_version": "qc_acceptance_config_schema.v2",
        "config_version": "qc_acceptance_v2.0.0",
        "config_name": "acceptance_gate",
        "config_path": "configs/qc_acceptance.yaml",
        "config_hash": "sha256:" + "4" * 64,
    }

    with pytest.raises(ValueError, match="profile is required"):
        write_asset_qc_report(path, report, expected_revision=0)

    assert not path.exists()


def test_write_persists_explicit_supplier_evaluation_profile(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    report = make_v1_video_report()
    report["qc_config"] = {
        "schema_version": "qc_acceptance_config_schema.v2",
        "config_version": "qc_acceptance_v2.0.0",
        "config_name": "acceptance_gate",
        "config_path": "configs/qc_acceptance.yaml",
        "config_hash": "sha256:" + "5" * 64,
    }

    write_asset_qc_report(
        path,
        report,
        expected_revision=0,
        profile="supplier_evaluation",
    )

    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["execution"]["profile"] == "supplier_evaluation"
    validate_asset_qc_report(written)
