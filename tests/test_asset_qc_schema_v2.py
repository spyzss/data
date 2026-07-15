from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Barrier, BrokenBarrierError, Event

import pytest

import qc_common.report as report_module
from qc_common.report import (
    StaleReportRevisionError,
    load_asset_qc_report,
    write_asset_qc_report,
)
from qc_common.report_migration import migrate_v1_to_v2
from qc_common.schema import validate_asset_qc_report
from tests.qc_report_fixtures import make_v1_video_report, make_v2_report


def _runtime_error() -> dict[str, object]:
    return {
        "module": "video_quality",
        "error_type": "process_error",
        "message": "decoder crashed",
        "occurred_at": "2026-07-14T00:00:00Z",
        "retryable": True,
    }


@pytest.mark.parametrize("status", ["pending", "running", "awaiting_external", "error"])
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
    report["runtime_errors"] = [_runtime_error()]

    validate_asset_qc_report(report)


@pytest.mark.parametrize(
    ("status", "decision", "error_path"),
    [
        ("running", None, "pipeline_state.status"),
        ("completed", "pass", "overall_decision"),
    ],
)
def test_runtime_errors_reject_non_error_pipeline_outcomes(
    status: str,
    decision: str | None,
    error_path: str,
) -> None:
    report = make_v2_report(status=status, overall_decision=decision)
    report["runtime_errors"] = [_runtime_error()]

    with pytest.raises(ValueError, match=error_path):
        validate_asset_qc_report(report)


@pytest.mark.parametrize(
    "runtime_error",
    [
        {
            "module": "video_quality",
            "error_type": "process_error",
            "message": "decoder crashed",
            "occurred_at": "2026-07-14T00:00:00Z",
        },
        {**_runtime_error(), "retryable": "yes"},
        {**_runtime_error(), "future_extension": True},
    ],
)
def test_runtime_error_schema_rejects_missing_wrong_and_unknown_fields(
    runtime_error: dict[str, object],
) -> None:
    report = make_v2_report(status="error", overall_decision=None)
    report["runtime_errors"] = [runtime_error]

    with pytest.raises(ValueError, match="runtime_errors"):
        validate_asset_qc_report(report)


def test_module_state_schema_accepts_all_registered_states() -> None:
    report = make_v2_report()
    states = [
        "completed",
        "disabled",
        "skipped",
        "not_implemented",
        "runtime_error",
        "awaiting_external",
        "skipped_due_to_fail",
    ]
    report["execution"]["module_states"] = {
        f"module_{index}": {
            "state": state,
            **({"reason": "configured"} if state == "disabled" else {}),
        }
        for index, state in enumerate(states)
    }

    validate_asset_qc_report(report)


@pytest.mark.parametrize(
    "module_state",
    [
        {"state": "unknown"},
        {"reason": "missing state"},
        {"state": "completed", "unexpected": True},
        {"state": "disabled", "reason": 1},
    ],
)
def test_module_state_schema_rejects_invalid_shape(
    module_state: dict[str, object],
) -> None:
    report = make_v2_report()
    report["execution"]["module_states"] = {"video_quality": module_state}

    with pytest.raises(ValueError, match="execution.module_states"):
        validate_asset_qc_report(report)


@pytest.mark.parametrize("decision", ["pass", "fail"])
def test_completed_v2_report_requires_quality_decision(decision: str) -> None:
    validate_asset_qc_report(make_v2_report(status="completed", overall_decision=decision))

    report = make_v2_report(status="completed", overall_decision=None)
    with pytest.raises(ValueError, match="overall_decision"):
        validate_asset_qc_report(report)


def test_v2_schema_registers_canonical_publish_binding_and_full_range() -> None:
    report = make_v2_report(status="completed", overall_decision="pass")
    report["canonical_binding"] = {
        "schema_version": "canonical_publish_binding.v1",
        "canonical_revision": 3,
        "semantic_fingerprint": "a" * 64,
        "source_fingerprint": "b" * 64,
        "qc_report_revision": report["report_revision"],
    }
    report["canonical_qc_range"] = {
        "start_frame": 0,
        "end_frame_exclusive": 10,
        "interval_semantics": "half_open",
    }

    validate_asset_qc_report(report)

    invalid = copy.deepcopy(report)
    invalid["canonical_binding"]["semantic_fingerprint"] = "not-a-hash"
    with pytest.raises(ValueError, match="canonical_binding.semantic_fingerprint"):
        validate_asset_qc_report(invalid)

    invalid = copy.deepcopy(report)
    invalid["canonical_qc_range"]["start_frame"] = 1
    with pytest.raises(ValueError, match="canonical_qc_range.start_frame"):
        validate_asset_qc_report(invalid)


def test_v2_schema_registers_formal_manual_review_records() -> None:
    report = make_v2_report()
    report["manual_review"].update(
        {
            "selected_issue_ids": ["issue-1"],
            "reviews": [
                {
                    "review_id": "review-1",
                    "issue_id": "issue-1",
                    "reviewer": "reviewer-1",
                    "reviewed_at": "2026-07-16T00:00:00Z",
                    "verdict": "reject_issue",
                    "asset_action": "accept",
                    "comment": "false positive",
                    "evidence_paths": [],
                }
            ],
        }
    )

    validate_asset_qc_report(report)

    invalid = copy.deepcopy(report)
    invalid["manual_review"]["reviews"][0]["verdict"] = "pass"
    with pytest.raises(ValueError, match="manual_review.reviews.0.verdict"):
        validate_asset_qc_report(invalid)


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


def test_schema_failure_preserves_bytes_and_releases_report_lock(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    initial = make_v2_report()
    write_asset_qc_report(path, initial, expected_revision=0, profile="acceptance")
    before = path.read_bytes()
    invalid = copy.deepcopy(initial)
    invalid["report_revision"] = 2
    invalid["execution"]["module_states"] = {
        "video_quality": {"state": "unknown"}
    }

    with pytest.raises(ValueError, match="execution.module_states"):
        write_asset_qc_report(
            path,
            invalid,
            expected_revision=1,
            profile="acceptance",
        )

    assert path.read_bytes() == before
    valid = copy.deepcopy(initial)
    valid["report_revision"] = 2
    valid["future_extension"] = {"after_validation_failure": True}
    write_asset_qc_report(path, valid, expected_revision=1, profile="acceptance")
    assert json.loads(path.read_text(encoding="utf-8")) == valid


def test_report_revision_cas_allows_exactly_one_racing_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "report.json"
    initial = make_v2_report()
    write_asset_qc_report(path, initial, expected_revision=0, profile="acceptance")
    candidates: dict[str, dict[str, object]] = {}
    for writer in ("a", "b"):
        candidate = copy.deepcopy(initial)
        candidate["report_revision"] = 2
        candidate["future_extension"] = {"writer": writer}
        candidates[writer] = candidate

    replace_barrier = Barrier(2)
    real_replace = report_module.os.replace

    def synchronized_replace(source: Path, destination: Path) -> None:
        try:
            replace_barrier.wait(timeout=0.5)
        except BrokenBarrierError:
            pass
        real_replace(source, destination)

    monkeypatch.setattr(report_module.os, "replace", synchronized_replace)

    def commit(writer: str) -> str:
        try:
            write_asset_qc_report(
                path,
                candidates[writer],
                expected_revision=1,
                profile="acceptance",
            )
        except StaleReportRevisionError:
            return "stale"
        return "committed"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = sorted(pool.map(commit, ("a", "b")))

    assert outcomes == ["committed", "stale"]
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written in candidates.values()
    assert written["report_revision"] == 2


def test_report_locks_do_not_serialize_different_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = {name: tmp_path / f"{name}.json" for name in ("a", "b")}
    candidates: dict[str, dict[str, object]] = {}
    for name, path in paths.items():
        initial = make_v2_report()
        initial["asset_id"] = name
        write_asset_qc_report(path, initial, expected_revision=0, profile="acceptance")
        candidate = copy.deepcopy(initial)
        candidate["report_revision"] = 2
        candidates[name] = candidate

    first_replace_entered = Event()
    release_first_replace = Event()
    real_replace = report_module.os.replace

    def block_first_path(source: Path, destination: Path) -> None:
        if Path(destination) == paths["a"]:
            first_replace_entered.set()
            assert release_first_replace.wait(timeout=2)
        real_replace(source, destination)

    monkeypatch.setattr(report_module.os, "replace", block_first_path)

    def commit(name: str) -> None:
        write_asset_qc_report(
            paths[name],
            candidates[name],
            expected_revision=1,
            profile="acceptance",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(commit, "a")
        assert first_replace_entered.wait(timeout=2)
        second = pool.submit(commit, "b")
        try:
            second.result(timeout=1)
        finally:
            release_first_replace.set()
        first.result(timeout=2)

    assert json.loads(paths["a"].read_text(encoding="utf-8")) == candidates["a"]
    assert json.loads(paths["b"].read_text(encoding="utf-8")) == candidates["b"]
