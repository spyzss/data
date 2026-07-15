from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import os
from pathlib import Path

import pytest

from canonical_qc import CanonicalQcBridge, StandardHdf5Adapter
from canonical_qc.provenance import semantic_fingerprint
from lerobot_v3_publisher import (
    PUBLISHER_VERSION,
    ManifestFile,
    PublishPrerequisiteError,
    PublishRequest,
    ReleaseManifest,
    release_id_for,
    revalidate_publish_plan,
    validate_publish_request,
)
from lerobot_v3_publisher.layout import derive_release_id
import lerobot_v3_publisher.prerequisites as publisher_prerequisites
from qc_common.config import load_qc_acceptance_config
from tests.fixtures import write_standard_hdf5_episode


def _write_publish_fixture(
    tmp_path: Path,
    *,
    manual_state: str = "not_required",
) -> tuple[PublishRequest, dict[str, object]]:
    batch_root = tmp_path / "batch"
    source_root = batch_root / "asset-001"
    write_standard_hdf5_episode(source_root)
    episode = StandardHdf5Adapter().load(source_root)
    report_path = batch_root / "quality_archive" / "asset-001.json"
    context = CanonicalQcBridge(episode, source_root=source_root).asset_context(
        batch_root=batch_root,
        report_path=report_path,
    )
    revision = 9
    semantic_hash = semantic_fingerprint(episode)
    config = load_qc_acceptance_config()
    module_states = {
        module: {
            "state": "completed"
            if config.module_config(module)["enabled"]
            else "disabled"
        }
        for module in config.pipeline_modules
    }
    issue_id = "video_quality:warn:fixture" if manual_state == "completed" else None
    reviews = (
        []
        if issue_id is None
        else [
            {
                "review_id": "review-000001",
                "issue_id": issue_id,
                "reviewer": "reviewer-001",
                "reviewed_at": "2026-07-15T00:00:30Z",
                "verdict": "accept_issue",
                "asset_action": "accept_with_risk",
                "comment": "accepted after checking evidence",
                "evidence_paths": [],
            }
        ]
    )
    manual = {
        "required": manual_state == "completed",
        "state": manual_state,
        "candidate_issue_ids": [] if issue_id is None else [issue_id],
        "selected_issue_ids": [] if issue_id is None else [issue_id],
        "failures_for_batch_stats_issue_ids": [],
        "reviews": reviews,
    }
    module_blocks = {}
    for module in config.pipeline_modules:
        module_config = config.module_config(module)
        if not module_config["enabled"] or module_config.get("execution_kind") == "external":
            continue
        module_index = config.pipeline_modules.index(module)
        next_module = (
            config.pipeline_modules[module_index + 1]
            if module_index + 1 < len(config.pipeline_modules)
            else None
        )
        module_blocks[module] = {
            "flow": {
                "entry_gate": {
                    "state": "ready",
                    "eligible": True,
                    "blocked_by_module": None,
                    "required_inputs": [],
                    "missing_inputs": [],
                    "upstream_continue": True,
                },
                "result_gate": {
                    "verdict": "pass",
                    "has_fail": False,
                    "has_warn": False,
                },
                "exit_gate": {
                    "state": "continue" if next_module is not None else "complete_qc",
                    "continue_to_next_module": next_module is not None,
                    "next_module": next_module,
                },
            },
            "evaluation": {"decision": "pass"},
            "metrics": {},
        }
    report: dict[str, object] = {
        "schema_version": "asset_qc_report.v2",
        "asset_id": episode.identity.asset_id,
        "supplier_id": episode.identity.supplier_id,
        "report_revision": revision,
        "qc_config": config.json_reference(),
        "execution": {
            "profile": "acceptance",
            "started_at": "2026-07-15T00:00:00Z",
            "updated_at": "2026-07-15T00:01:00Z",
            "module_states": module_states,
        },
        "pipeline_state": {
            "status": "completed",
            "last_completed_module": config.pipeline_modules[-1],
            "next_module": None,
            "stop_reason": None,
        },
        "overall_decision": "pass",
        "source_files": copy.deepcopy(dict(context.source_files)),
        "issues": (
            []
            if issue_id is None
            else [
                {
                    "issue_id": issue_id,
                    "severity": "warn",
                    "needs_manual_review": True,
                }
            ]
        ),
        "runtime_errors": [],
        "semantic_consistency": {"state": "completed"},
        "semantic_calibration": {
            "state": "completed",
            "canonical_revision": 3,
        },
        "manual_review": manual,
        "canonical_binding": {
            "schema_version": "canonical_publish_binding.v1",
            "canonical_revision": 3,
            "semantic_fingerprint": semantic_hash,
            "source_fingerprint": episode.provenance.source_fingerprint,
            "qc_report_revision": revision,
        },
        "canonical_qc_range": {
            "start_frame": 0,
            "end_frame_exclusive": episode.time_axis.frame_count,
            "interval_semantics": "half_open",
        },
        **module_blocks,
    }
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    request = PublishRequest(
        episode=episode,
        canonical_revision=3,
        canonical_source_root=source_root,
        qc_report_path=report_path,
        expected_report_revision=revision,
        release_root=tmp_path / "curated",
    )
    return request, report


def _rewrite_report(request: PublishRequest, report: dict[str, object]) -> None:
    request.qc_report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _assert_rejected(
    request: PublishRequest,
    *,
    field: str,
    code: str = "publish_prerequisite_failed",
) -> PublishPrerequisiteError:
    with pytest.raises(PublishPrerequisiteError) as raised:
        validate_publish_request(request)
    assert raised.value.diagnostic.code == code
    assert raised.value.diagnostic.stage == "publish_prerequisite"
    assert raised.value.diagnostic.field == field
    return raised.value


@pytest.mark.parametrize("manual_state", ["completed", "not_required"])
def test_valid_request_builds_immutable_deterministic_safe_plan_without_writes(
    tmp_path: Path,
    manual_state: str,
) -> None:
    request, _report = _write_publish_fixture(
        tmp_path,
        manual_state=manual_state,
    )
    before_report = request.qc_report_path.read_bytes()
    before_sources = {
        path.relative_to(request.canonical_source_root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in request.canonical_source_root.rglob("*")
        if path.is_file()
    }

    first = validate_publish_request(request)
    second = validate_publish_request(request)

    assert first == second
    assert first.release_id == release_id_for(request)
    assert first.publisher_version == PUBLISHER_VERSION
    assert first.release_path == request.release_root / "releases" / first.release_id
    assert first.current_path == request.release_root / "CURRENT.json"
    assert first.release_id.startswith("lerobot-v3-")
    assert "/" not in first.release_id and ".." not in first.release_id
    assert request.qc_report_path.read_bytes() == before_report
    assert {
        path.relative_to(request.canonical_source_root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in request.canonical_source_root.rglob("*")
        if path.is_file()
    } == before_sources
    with pytest.raises(FrozenInstanceError):
        first.release_id = "changed"  # type: ignore[misc]


def test_manifest_contract_is_deeply_immutable_at_sequence_boundaries() -> None:
    files = [ManifestFile("meta/info.json", 12, "a" * 64)]
    manifest = ReleaseManifest(
        schema_version="curated_lerobot_v3_release_manifest.v1",
        release_id="lerobot-v3-" + "a" * 32,
        publisher_version=PUBLISHER_VERSION,
        asset_id="asset-001",
        canonical_revision=3,
        semantic_fingerprint="b" * 64,
        source_fingerprint="c" * 64,
        qc_report_revision=9,
        qc_report_sha256="d" * 64,
        files=files,
    )

    files.append(ManifestFile("data/file.parquet", 1, "e" * 64))

    assert isinstance(manifest.files, tuple)
    assert len(manifest.files) == 1
    with pytest.raises(FrozenInstanceError):
        manifest.asset_id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError, match="ManifestFile"):
        ReleaseManifest(
            schema_version="curated_lerobot_v3_release_manifest.v1",
            release_id="lerobot-v3-" + "a" * 32,
            publisher_version=PUBLISHER_VERSION,
            asset_id="asset-001",
            canonical_revision=3,
            semantic_fingerprint="b" * 64,
            source_fingerprint="c" * 64,
            qc_report_revision=9,
            qc_report_sha256="d" * 64,
            files=({"relative_path": "mutable"},),  # type: ignore[arg-type]
        )


def test_publish_plan_rejects_mutable_source_snapshot_entries(tmp_path: Path) -> None:
    request, _report = _write_publish_fixture(tmp_path)
    plan = validate_publish_request(request)

    with pytest.raises(TypeError, match="SourceSnapshot"):
        replace(plan, source_snapshot=({},))  # type: ignore[arg-type]


def test_release_id_four_tuple_is_deterministic_and_domain_separated() -> None:
    base = {
        "asset_id": "asset-001",
        "canonical_revision": 3,
        "semantic_fingerprint": "a" * 64,
        "publisher_version": PUBLISHER_VERSION,
    }
    first = derive_release_id(**base)

    assert first == derive_release_id(**base)
    assert len(
        {
            first,
            derive_release_id(**{**base, "asset_id": "asset-002"}),
            derive_release_id(**{**base, "canonical_revision": 4}),
            derive_release_id(**{**base, "semantic_fingerprint": "b" * 64}),
            derive_release_id(**{**base, "publisher_version": "next"}),
        }
    ) == 5


@pytest.mark.parametrize(
    ("status", "decision", "field"),
    [
        ("completed", "fail", "overall_decision"),
        ("completed", None, "overall_decision"),
        ("running", None, "pipeline_state.status"),
        ("error", None, "pipeline_state.status"),
    ],
)
def test_non_final_or_non_pass_report_is_rejected_explicitly(
    tmp_path: Path,
    status: str,
    decision: str | None,
    field: str,
) -> None:
    request, report = _write_publish_fixture(tmp_path)
    report["pipeline_state"]["status"] = status  # type: ignore[index]
    report["overall_decision"] = decision
    _rewrite_report(request, report)

    _assert_rejected(request, field=field)


@pytest.mark.parametrize(
    ("mutate", "field"),
    [
        (lambda report: report.update(schema_version="asset_qc_report.v1"), "schema_version"),
        (lambda report: report.update(asset_id="other"), "asset_id"),
        (lambda report: report.update(supplier_id="other"), "supplier_id"),
        (
            lambda report: report["execution"].update(profile="supplier_evaluation"),
            "execution.profile",
        ),
        (
            lambda report: report["pipeline_state"].update(next_module="manual_review"),
            "pipeline_state.next_module",
        ),
        (
            lambda report: report["pipeline_state"].update(stop_reason="hard_fail"),
            "pipeline_state.stop_reason",
        ),
        (
            lambda report: report.update(runtime_errors=[{"error_type": "process_error"}]),
            "runtime_errors",
        ),
        (
            lambda report: report["semantic_consistency"].update(state="awaiting_external"),
            "semantic_consistency.state",
        ),
        (
            lambda report: report["semantic_calibration"].update(state="in_progress"),
            "semantic_calibration.state",
        ),
        (
            lambda report: report["manual_review"].update(state="queued"),
            "manual_review.state",
        ),
    ],
)
def test_report_gate_combinations_fail_closed(
    tmp_path: Path,
    mutate,
    field: str,
) -> None:
    request, report = _write_publish_fixture(tmp_path)
    mutate(report)
    _rewrite_report(request, report)

    _assert_rejected(request, field=field)


def test_warn_candidates_require_completed_passing_human_reviews(tmp_path: Path) -> None:
    request, report = _write_publish_fixture(tmp_path, manual_state="completed")
    issue_id = "video_quality:warn:001"
    report["issues"] = [
        {
            "issue_id": issue_id,
            "severity": "warn",
            "needs_manual_review": True,
        }
    ]
    report["manual_review"].update(  # type: ignore[union-attr]
        candidate_issue_ids=[issue_id],
        selected_issue_ids=[issue_id],
        reviews=[
            {
                "review_id": "review-000002",
                "issue_id": issue_id,
                "reviewer": "reviewer-001",
                "reviewed_at": "2026-07-15T00:00:30Z",
                "verdict": "reject_issue",
                "asset_action": "accept",
                "comment": "machine warning rejected",
                "evidence_paths": [],
            }
        ],
    )
    _rewrite_report(request, report)
    assert validate_publish_request(request).release_id

    report["manual_review"]["reviews"] = []  # type: ignore[index]
    _rewrite_report(request, report)
    _assert_rejected(request, field="manual_review.reviews")


@pytest.mark.parametrize("asset_action", ["reject", "return_for_rework"])
def test_final_manual_asset_action_controls_publishability(
    tmp_path: Path,
    asset_action: str,
) -> None:
    request, report = _write_publish_fixture(tmp_path, manual_state="completed")
    report["manual_review"]["reviews"][0]["asset_action"] = asset_action  # type: ignore[index]
    _rewrite_report(request, report)

    _assert_rejected(request, field="manual_review.reviews[0].asset_action")


@pytest.mark.parametrize(
    ("mutation", "field"),
    [
        ("missing_block", "video_quality"),
        ("fail_gate", "video_quality.flow.result_gate.verdict"),
        ("state_mismatch", "execution.module_states.video_quality"),
        ("cursor", "pipeline_state.last_completed_module"),
    ],
)
def test_automatic_module_results_and_final_cursor_are_required(
    tmp_path: Path,
    mutation: str,
    field: str,
) -> None:
    request, report = _write_publish_fixture(tmp_path)
    if mutation == "missing_block":
        report.pop("video_quality")
    elif mutation == "fail_gate":
        report["video_quality"]["flow"]["result_gate"]["verdict"] = "fail"  # type: ignore[index]
    elif mutation == "state_mismatch":
        report["execution"]["module_states"]["video_quality"]["state"] = "skipped"  # type: ignore[index]
    else:
        report["pipeline_state"]["last_completed_module"] = "manual_review"  # type: ignore[index]
    _rewrite_report(request, report)

    _assert_rejected(request, field=field)


def test_sam3_zero_candidate_clean_skip_is_publishable(tmp_path: Path) -> None:
    request, report = _write_publish_fixture(tmp_path)
    report["execution"]["module_states"]["sam3_containment"]["state"] = "skipped"  # type: ignore[index]
    block = report["sam3_containment"]
    block["flow"]["result_gate"] = {  # type: ignore[index]
        "verdict": "skipped",
        "has_fail": False,
        "has_warn": False,
    }
    block["evaluation"] = {"decision": "skipped", "window_count": 0}  # type: ignore[index]
    block["metrics"] = {"window_count": 0}  # type: ignore[index]
    _rewrite_report(request, report)

    assert validate_publish_request(request).release_id


@pytest.mark.parametrize(
    ("mutation", "field"),
    [
        ("missing_entry_field", "video_quality.flow.entry_gate.eligible"),
        ("wrong_result_boolean", "video_quality.flow.result_gate.has_warn"),
        ("evaluation_mismatch", "video_quality.evaluation.decision"),
        ("evaluation_fail", "video_quality.evaluation.decision"),
        ("stop_qc", "video_quality.flow.exit_gate.state"),
        ("wrong_continue", "video_quality.flow.exit_gate.continue_to_next_module"),
        ("wrong_next", "video_quality.flow.exit_gate.next_module"),
    ],
)
def test_automatic_module_flow_is_deeply_consistent(
    tmp_path: Path,
    mutation: str,
    field: str,
) -> None:
    request, report = _write_publish_fixture(tmp_path)
    block = report["video_quality"]
    if mutation == "missing_entry_field":
        block["flow"]["entry_gate"].pop("eligible")  # type: ignore[index]
    elif mutation == "wrong_result_boolean":
        block["flow"]["result_gate"]["has_warn"] = True  # type: ignore[index]
    elif mutation == "evaluation_mismatch":
        block["evaluation"]["decision"] = "warn"  # type: ignore[index]
    elif mutation == "evaluation_fail":
        block["evaluation"]["decision"] = "fail"  # type: ignore[index]
    elif mutation == "stop_qc":
        block["flow"]["exit_gate"].update(  # type: ignore[index]
            state="stop_qc",
            continue_to_next_module=False,
            next_module=None,
        )
    elif mutation == "wrong_continue":
        block["flow"]["exit_gate"]["continue_to_next_module"] = False  # type: ignore[index]
    else:
        block["flow"]["exit_gate"]["next_module"] = "manual_review"  # type: ignore[index]
    _rewrite_report(request, report)

    _assert_rejected(request, field=field)


def test_fail_issue_is_rejected_even_if_top_level_claims_pass(tmp_path: Path) -> None:
    request, report = _write_publish_fixture(tmp_path)
    report["issues"] = [{"issue_id": "x", "severity": "fail"}]
    _rewrite_report(request, report)

    _assert_rejected(request, field="issues")


def test_report_revision_and_binding_are_cas_bound(tmp_path: Path) -> None:
    request, report = _write_publish_fixture(tmp_path)

    stale = PublishRequest(
        episode=request.episode,
        canonical_revision=request.canonical_revision,
        canonical_source_root=request.canonical_source_root,
        qc_report_path=request.qc_report_path,
        expected_report_revision=request.expected_report_revision - 1,
        release_root=request.release_root,
    )
    error = _assert_rejected(stale, field="report_revision")
    assert error.diagnostic.retryable is True

    report["canonical_binding"]["qc_report_revision"] = 8  # type: ignore[index]
    _rewrite_report(request, report)
    _assert_rejected(request, field="canonical_binding.qc_report_revision")

    different_canonical_revision = PublishRequest(
        episode=request.episode,
        canonical_revision=4,
        canonical_source_root=request.canonical_source_root,
        qc_report_path=request.qc_report_path,
        expected_report_revision=request.expected_report_revision,
        release_root=request.release_root,
    )
    _assert_rejected(
        different_canonical_revision,
        field="canonical_binding.canonical_revision",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("canonical_revision", 4),
        ("semantic_fingerprint", "0" * 64),
        ("source_fingerprint", "1" * 64),
    ],
)
def test_canonical_binding_must_match_request_and_episode(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    request, report = _write_publish_fixture(tmp_path)
    report["canonical_binding"][field] = value  # type: ignore[index]
    _rewrite_report(request, report)

    _assert_rejected(request, field=f"canonical_binding.{field}")


def test_only_full_canonical_half_open_range_is_publishable(tmp_path: Path) -> None:
    request, report = _write_publish_fixture(tmp_path)
    report["canonical_qc_range"]["end_frame_exclusive"] = 2  # type: ignore[index]
    _rewrite_report(request, report)

    _assert_rejected(request, field="canonical_qc_range")


def test_config_reference_must_match_verified_immutable_snapshot(tmp_path: Path) -> None:
    request, report = _write_publish_fixture(tmp_path)
    report["qc_config"]["config_hash"] = "sha256:" + "f" * 64  # type: ignore[index]
    _rewrite_report(request, report)

    _assert_rejected(request, field="qc_config.config_hash")


def test_current_source_hash_drift_is_rejected_without_touching_report(tmp_path: Path) -> None:
    request, _report = _write_publish_fixture(tmp_path)
    before = request.qc_report_path.read_bytes()
    source = request.canonical_source_root / request.episode.provenance.source_files[0].relative_path
    source.write_bytes(source.read_bytes() + b"drift")

    error = _assert_rejected(
        request,
        field="provenance.source_files[0]",
        code="source_integrity_error",
    )

    assert error.diagnostic.retryable is False
    assert request.qc_report_path.read_bytes() == before


def test_source_root_and_report_paths_reject_relative_symlink_and_escape(
    tmp_path: Path,
) -> None:
    request, _report = _write_publish_fixture(tmp_path)
    relative = PublishRequest(
        episode=request.episode,
        canonical_revision=request.canonical_revision,
        canonical_source_root=Path("relative-source"),
        qc_report_path=request.qc_report_path,
        expected_report_revision=request.expected_report_revision,
        release_root=request.release_root,
    )
    _assert_rejected(relative, field="canonical_source_root")

    linked_root = tmp_path / "linked-source"
    linked_root.symlink_to(request.canonical_source_root, target_is_directory=True)
    linked = PublishRequest(
        episode=request.episode,
        canonical_revision=request.canonical_revision,
        canonical_source_root=linked_root,
        qc_report_path=request.qc_report_path,
        expected_report_revision=request.expected_report_revision,
        release_root=request.release_root,
    )
    _assert_rejected(linked, field="canonical_source_root")

    linked_report = tmp_path / "linked-report.json"
    linked_report.symlink_to(request.qc_report_path)
    unsafe_report = PublishRequest(
        episode=request.episode,
        canonical_revision=request.canonical_revision,
        canonical_source_root=request.canonical_source_root,
        qc_report_path=linked_report,
        expected_report_revision=request.expected_report_revision,
        release_root=request.release_root,
    )
    _assert_rejected(unsafe_report, field="qc_report_path")

    unsafe_release = PublishRequest(
        episode=request.episode,
        canonical_revision=request.canonical_revision,
        canonical_source_root=request.canonical_source_root,
        qc_report_path=request.qc_report_path,
        expected_report_revision=request.expected_report_revision,
        release_root=Path("relative-release"),
    )
    _assert_rejected(unsafe_release, field="release_root")


@pytest.mark.parametrize("relation", ["equal", "inside", "contains"])
def test_release_root_must_not_overlap_canonical_source_root(
    tmp_path: Path,
    relation: str,
) -> None:
    request, _report = _write_publish_fixture(tmp_path)
    release_root = {
        "equal": request.canonical_source_root,
        "inside": request.canonical_source_root / "curated",
        "contains": request.canonical_source_root.parent,
    }[relation]
    unsafe = replace(request, release_root=release_root)

    _assert_rejected(unsafe, field="release_root")


def test_qc_report_must_not_alias_a_canonical_source_file(tmp_path: Path) -> None:
    request, _report = _write_publish_fixture(tmp_path)
    source = (
        request.canonical_source_root
        / request.episode.provenance.source_files[0].relative_path
    )
    request.qc_report_path.unlink()
    os.link(source, request.qc_report_path)

    _assert_rejected(request, field="qc_report_path")


@pytest.mark.parametrize("target", ["report", "source"])
def test_snapshot_detects_in_place_drift_during_hash(
    tmp_path: Path,
    monkeypatch,
    target: str,
) -> None:
    request, _report = _write_publish_fixture(tmp_path)
    path = (
        request.qc_report_path
        if target == "report"
        else request.canonical_source_root
        / request.episode.provenance.source_files[0].relative_path
    )
    target_inode = path.stat().st_ino
    original_read = publisher_prerequisites.os.read
    mutated = False

    def drifting_read(fd: int, size: int) -> bytes:
        nonlocal mutated
        data = original_read(fd, size)
        if data and not mutated and os.fstat(fd).st_ino == target_inode:
            mutated = True
            with path.open("ab") as stream:
                stream.write(b"drift-during-hash")
        return data

    monkeypatch.setattr(publisher_prerequisites.os, "read", drifting_read)

    error = _assert_rejected(
        request,
        field="qc_report_path" if target == "report" else "provenance.source_files[0]",
        code=(
            "publish_prerequisite_failed"
            if target == "report"
            else "source_integrity_error"
        ),
    )
    assert error.diagnostic.retryable is True


def test_large_source_snapshot_streams_hash_without_capturing_payload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "large-source.bin"
    payload = b"0123456789abcdef" * (1024 * 256)
    path.write_bytes(payload)

    snapshot = publisher_prerequisites._snapshot_file(
        path,
        field="source",
        code="source_integrity_error",
        retryable_io=True,
        capture_payload=False,
    )

    assert snapshot.payload is None
    assert snapshot.size_bytes == len(payload)
    assert snapshot.sha256 == hashlib.sha256(payload).hexdigest()


def test_qc_report_snapshot_rejects_oversized_payload(tmp_path: Path) -> None:
    request, _report = _write_publish_fixture(tmp_path)
    request.qc_report_path.write_bytes(
        b"{" + b" " * publisher_prerequisites.MAX_QC_REPORT_BYTES + b"}"
    )

    error = _assert_rejected(request, field="qc_report_path")
    assert "maximum" in error.diagnostic.message


def test_malformed_issue_is_rejected_before_release(tmp_path: Path) -> None:
    request, report = _write_publish_fixture(tmp_path)
    report["issues"] = [{"issue_id": "x", "severity": "warn"}]
    _rewrite_report(request, report)

    _assert_rejected(request, field="issues")


def test_revalidation_detects_same_revision_report_content_drift(tmp_path: Path) -> None:
    request, report = _write_publish_fixture(tmp_path)
    plan = validate_publish_request(request)
    report["audit_extension"] = {"changed": True}
    _rewrite_report(request, report)

    error = _assert_rejected_plan(plan, field="qc_report_sha256")
    assert error.diagnostic.retryable is True


def _assert_rejected_plan(plan, *, field: str) -> PublishPrerequisiteError:
    with pytest.raises(PublishPrerequisiteError) as raised:
        revalidate_publish_plan(plan)
    assert raised.value.diagnostic.code == "publish_prerequisite_failed"
    assert raised.value.diagnostic.field == field
    return raised.value
