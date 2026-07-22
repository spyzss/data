from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from canonical_qc import CanonicalQcBridge, StandardHdf5Adapter, StandardLeRobotAdapter
from canonical_qc.config import load_canonical_qc_config
from canonical_qc.errors import CanonicalInputError
from canonical_qc.provenance import semantic_fingerprint
from canonical_qc.source_gate import (
    DeclaredEpisodeIdentity,
    SourceGateLocator,
    record_source_gate_failure,
    record_source_gate_pass,
)
from canonical_qc.workflow import run_canonical_source_qc
from lerobot_v3_publisher import CanonicalDiagnostic, PublishPrerequisiteError
from qc_common.config import load_qc_acceptance_config
from qc_common.contracts import EvidenceRef, Issue, ModuleResult
from qc_common.module_registry import ModuleRegistry
from qc_common.report import write_asset_qc_report
from qc_common.schema import validate_asset_qc_report
from qc_reporting.aggregate import aggregate_projection
from qc_reporting.projection import project_quality_archive
from tests.fixtures import (
    write_standard_hdf5_episode,
    write_standard_lerobot_dataset,
)
from tests.test_lerobot_v3_writer import _plan_for_episode


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)


def _run(tool: str, *args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(PYTHON), str(ROOT / "tools" / tool), *(str(arg) for arg in args)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=240,
    )


def _json_line(value: str) -> dict[str, object]:
    assert value.endswith("\n")
    assert "Traceback" not in value
    return json.loads(value)


def _identity_args(asset_id: str = "asset-001") -> tuple[str, ...]:
    return (
        "--asset-id", asset_id,
        "--batch-id", "batch-001",
        "--supplier-id", "supplier-001",
    )


def _run_cli_fixture(
    tmp_path: Path,
    source_format: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> tuple[Path, dict[str, object]]:
    batch = tmp_path / source_format
    source = batch / "asset-001"
    if source_format == "hdf5":
        write_standard_hdf5_episode(source)
    else:
        write_standard_lerobot_dataset(source)
    archive = batch / "quality_archive"
    import canonical_qc.workflow as workflow
    import tools.run_canonical_qc as cli

    monkeypatch.setattr(
        cli,
        "run_canonical_source_qc",
        lambda **kwargs: workflow.run_canonical_source_qc(
            **kwargs, registry_factory=_pass_registry
        ),
    )
    args = [str(item) for item in (
        "--source", source,
        "--source-format", source_format,
        "--source-root", source,
        "--batch-root", batch,
        "--quality-archive", archive,
        "--profile", "acceptance",
        *_identity_args(),
    )]

    assert cli.main(args) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    first_json = _json_line(captured.out)
    assert first_json["ok"] is True
    assert first_json["result"]["status"] == "awaiting_external"
    assert first_json["result"]["resumed"] is False
    report_path = archive / "asset-001.json"
    first_report = report_path.read_bytes()

    assert cli.main(args) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    second_json = _json_line(captured.out)
    assert second_json["result"]["resumed"] is True
    assert second_json["result"]["report_revision"] == first_json["result"]["report_revision"]
    assert report_path.read_bytes() == first_report
    assert cli.main([
        *args,
        "--config",
        str(ROOT / "configs/canonical_qc/canonical_qc_v1.1.1.yaml"),
    ]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert _json_line(captured.err)["error"]["code"] == "input_contract_error"
    assert report_path.read_bytes() == first_report
    return source, first_json


@pytest.mark.parametrize("source_format", ["hdf5", "lerobot"])
def test_run_cli_ingests_and_resumes_both_registered_source_formats(
    tmp_path: Path,
    source_format: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, payload = _run_cli_fixture(
        tmp_path, source_format, monkeypatch, capsys
    )

    assert payload["command"] == "run_canonical_qc"
    assert payload["result"]["asset_id"] == "asset-001"
    assert payload["result"]["source_format"] == source_format
    assert payload["result"]["source"] == str(source.resolve())


def test_canonical_config_active_alias_matches_immutable_validated_snapshot() -> None:
    loaded = load_canonical_qc_config()

    alias = ROOT / "configs/canonical_qc.yaml"
    snapshot = ROOT / "configs/canonical_qc/canonical_qc_v1.3.0.yaml"
    assert alias.read_bytes() == snapshot.read_bytes()
    assert loaded.config_version == "canonical_qc_v1.3.0"
    assert loaded.source_gate_rule_id == "canonical.source_gate.contract_failure"
    qc = load_qc_acceptance_config(loaded.qc_config_path)
    registered_qc_rule_ids = {
        rule["rule_id"]
        for module in qc.raw["modules"].values()
        for rule in module["rules"].values()
    }
    assert loaded.source_gate_rule_id not in registered_qc_rule_ids
    assert loaded.path == alias.resolve()
    assert loaded.qc_config_path == (
        ROOT / "configs/qc_acceptance/qc_acceptance_v2.3.0.yaml"
    ).resolve()


def test_canonical_config_rejects_runtime_version_and_qc_hash_drift(tmp_path: Path) -> None:
    payload = yaml.safe_load((ROOT / "configs/canonical_qc.yaml").read_text())
    payload["runtime_contract"]["standard_lerobot_adapter_version"] = "9.9.9"
    drifted = tmp_path / "runtime-drift.yaml"
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(ValueError, match="Canonical QC config validation failed"):
        load_canonical_qc_config(drifted)

    payload = yaml.safe_load((ROOT / "configs/canonical_qc.yaml").read_text())
    payload["config_version"] = "canonical_qc_v1.0.1"
    payload["qc"]["config_path"] = str((ROOT / "configs/qc_acceptance.yaml").resolve())
    payload["qc"]["config_sha256"] = "sha256:" + "0" * 64
    drifted = tmp_path / "canonical_qc_v1.0.1.yaml"
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(ValueError, match="immutable snapshot"):
        load_canonical_qc_config(drifted)

    same_version = yaml.safe_load((ROOT / "configs/canonical_qc.yaml").read_text())
    same_version["source"]["timestamp_tolerance_ns"] = 0
    same_version_path = tmp_path / "canonical_qc_v1.0.0.yaml"
    same_version_path.write_text(yaml.safe_dump(same_version, sort_keys=False))
    with pytest.raises(ValueError, match="immutable snapshot"):
        load_canonical_qc_config(same_version_path)


def test_pre_source_gate_config_is_explicitly_non_executable_for_malformed_source(
    tmp_path: Path,
) -> None:
    legacy_config = ROOT / "configs/canonical_qc/canonical_qc_v1.0.0.yaml"
    with pytest.raises(ValueError, match="before v1.1.0 are not executable"):
        load_canonical_qc_config(legacy_config)

    batch = tmp_path / "batch"
    source = batch / "broken-hdf"
    write_standard_hdf5_episode(source, asset_id="broken-hdf")
    (source / "broken-hdf.h5").write_bytes(b"not hdf5")
    archive = batch / "quality_archive"
    result = _run(
        "run_canonical_qc.py",
        "--source", source,
        "--source-format", "hdf5",
        "--source-root", source,
        "--batch-root", batch,
        "--quality-archive", archive,
        "--profile", "acceptance",
        *_identity_args("broken-hdf"),
        "--config", legacy_config,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    error = _json_line(result.stderr)["error"]
    assert error["code"] == "input_contract_error"
    assert "Source Gate rule registry is required" in error["message"]
    assert not archive.exists()


def test_run_cli_requires_episode_selector_and_applies_configured_timestamp_tolerance(
    tmp_path: Path,
) -> None:
    batch = tmp_path / "batch"
    multi = write_standard_lerobot_dataset(
        batch / "multi", episodes=((3, "asset-three"), (7, "asset-seven"))
    )
    base = (
        "--source", multi,
        "--source-format", "lerobot",
        "--source-root", multi,
        "--batch-root", batch,
        "--quality-archive", batch / "quality_archive",
        "--profile", "acceptance",
        *_identity_args("asset-seven"),
        "--dry-run",
    )
    ambiguous = _run("run_canonical_qc.py", *base)
    assert ambiguous.returncode == 2
    assert _json_line(ambiguous.stderr)["error"]["field"] == "episode_index"
    selected = _run("run_canonical_qc.py", *base, "--episode-index", 7)
    assert selected.returncode == 0, selected.stderr
    assert _json_line(selected.stdout)["result"]["asset_id"] == "asset-seven"

    source = write_standard_lerobot_dataset(batch / "drift")
    data_path = next((source / "data").rglob("*.parquet"))
    table = pq.read_table(data_path)
    index = table.schema.get_field_index("timestamp")
    table = table.set_column(
        index,
        "timestamp",
        pa.array([0.0, 0.100001, 0.2], type=pa.float64()),
    )
    pq.write_table(table, data_path)
    strict_config = ROOT / "configs/canonical_qc/canonical_qc_v1.1.1.yaml"
    rejected = _run(
        "run_canonical_qc.py",
        "--source", source,
        "--source-format", "lerobot",
        "--source-root", source,
        "--batch-root", batch,
        "--quality-archive", batch / "quality_archive",
        "--profile", "acceptance",
        *_identity_args(),
        "--config", strict_config,
        "--dry-run",
    )
    assert rejected.returncode == 2
    assert _json_line(rejected.stderr)["error"]["code"] == "timebase_invalid"


def _pass_registry(context: object, config: object) -> ModuleRegistry:
    registry = ModuleRegistry()
    for module_name in config.pipeline_modules:
        module = config.module_config(module_name)
        implementation = module.get("implementation")
        if not module.get("enabled") or not isinstance(implementation, str):
            continue

        def run(
            _context: object,
            _config: object,
            *,
            module_name: str = module_name,
        ) -> ModuleResult:
            return ModuleResult(module_name, "pass", {"decision": "pass"}, {})

        registry.register(implementation, run)
    return registry


def _publish_cli_fixture(
    tmp_path: Path,
    source_format: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> tuple[dict[str, object], Path]:
    batch = tmp_path / source_format
    source = batch / "asset-001"
    if source_format == "hdf5":
        write_standard_hdf5_episode(source)
        episode = StandardHdf5Adapter().load(source)
    else:
        write_standard_lerobot_dataset(source)
        episode = StandardLeRobotAdapter().load(source)
    import canonical_qc.workflow as workflow
    import tools.run_canonical_qc as run_cli

    monkeypatch.setattr(
        run_cli,
        "run_canonical_source_qc",
        lambda **kwargs: workflow.run_canonical_source_qc(
            **kwargs, registry_factory=_pass_registry
        ),
    )
    run_args = [str(item) for item in (
        "--source", source,
        "--source-format", source_format,
        "--source-root", source,
        "--batch-root", batch,
        "--quality-archive", batch / "quality_archive",
        "--profile", "acceptance",
        *_identity_args(),
    )]
    qc_code = run_cli.main(run_args)
    captured = capsys.readouterr()
    assert qc_code == 0, captured.err
    qc_payload = _json_line(captured.out)
    assert qc_payload["result"]["state"] == "awaiting_external"
    report_path = Path(qc_payload["result"]["report_path"])
    report_before = report_path.read_bytes()
    assert run_cli.main(run_args) == 0
    capsys.readouterr()
    assert report_path.read_bytes() == report_before
    report = json.loads(report_path.read_text())
    prior_revision = int(report["report_revision"])
    final_revision = prior_revision + 1
    canonical_revision = 1
    qc_config = load_qc_acceptance_config()
    report["execution"]["module_states"] = {
        "source_gate": {"state": "completed"},
        **{
            module: {
                "state": "completed"
                if qc_config.module_config(module)["enabled"]
                else "disabled"
            }
            for module in qc_config.pipeline_modules
        },
    }
    report["pipeline_state"] = {
        "status": "completed",
        "last_completed_module": qc_config.pipeline_modules[-1],
        "next_module": None,
        "stop_reason": None,
    }
    report.update(
        report_revision=final_revision,
        overall_decision="pass",
        semantic_consistency={"state": "completed"},
        semantic_calibration={
            "state": "completed",
            "canonical_revision": canonical_revision,
            "timeline_edit_count": 0,
            "subtask_text_edit_count": 0,
        },
        manual_review={
            "required": False,
            "state": "not_required",
            "candidate_issue_ids": [],
            "selected_issue_ids": [],
            "failures_for_batch_stats_issue_ids": [],
            "reviews": [],
        },
        canonical_binding={
            "schema_version": "canonical_publish_binding.v1",
            "canonical_revision": canonical_revision,
            "semantic_fingerprint": semantic_fingerprint(episode),
            "source_fingerprint": episode.provenance.source_fingerprint,
            "qc_report_revision": final_revision,
        },
        canonical_qc_range={
            "start_frame": 0,
            "end_frame_exclusive": episode.time_axis.frame_count,
            "interval_semantics": "half_open",
        },
    )
    write_asset_qc_report(
        report_path,
        report,
        expected_revision=prior_revision,
        profile="acceptance",
    )
    release_root = batch / "curated"
    args = (
        "--source", source,
        "--source-format", source_format,
        "--canonical-source-root", source,
        "--qc-report", report_path,
        "--release-root", release_root,
    )

    dry = _run("publish_lerobot_v3.py", *args, "--dry-run")
    assert dry.returncode == 0, dry.stderr
    assert not release_root.exists()
    assert _json_line(dry.stdout)["result"]["state"] == "validated"

    published = _run("publish_lerobot_v3.py", *args)
    assert published.returncode == 0, published.stderr
    assert published.stderr == ""
    payload = _json_line(published.stdout)
    current = json.loads((release_root / "CURRENT.json").read_text())
    release = release_root / "releases" / current["release_id"]
    official = StandardLeRobotAdapter().load(release)
    assert semantic_fingerprint(official) == semantic_fingerprint(episode)
    return payload, release


def test_publish_cli_rewrites_both_inputs_to_equivalent_official_releases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    hdf, hdf_release = _publish_cli_fixture(
        tmp_path, "hdf5", monkeypatch, capsys
    )
    lerobot, lerobot_release = _publish_cli_fixture(
        tmp_path, "lerobot", monkeypatch, capsys
    )
    hdf_episode = StandardLeRobotAdapter().load(hdf_release)
    lerobot_episode = StandardLeRobotAdapter().load(lerobot_release)

    assert hdf["result"]["state"] == "published"
    assert lerobot["result"]["state"] == "published"
    assert semantic_fingerprint(hdf_episode) == semantic_fingerprint(lerobot_episode)
    assert np.array_equal(
        hdf_episode.time_axis.timestamps_ns,
        lerobot_episode.time_axis.timestamps_ns,
    )
    assert np.array_equal(
        hdf_episode.observation.hand_keypoints_3d,
        lerobot_episode.observation.hand_keypoints_3d,
        equal_nan=True,
    )


def test_run_cli_input_contract_error_is_json_without_traceback(tmp_path: Path) -> None:
    result = _run(
        "run_canonical_qc.py",
        "--source", tmp_path / "missing",
        "--source-format", "hdf5",
        "--source-root", tmp_path / "missing",
        "--batch-root", tmp_path,
        "--quality-archive", tmp_path / "quality_archive",
        "--profile", "acceptance",
        *_identity_args(),
    )

    assert result.returncode == 2
    assert result.stderr == ""
    payload = _json_line(result.stdout)
    assert payload["category"] == "quality_fail"
    error = payload["error"]
    assert error["category"] == "input_contract"
    assert error["retryable"] is False
    assert Path(payload["result"]["report_path"]).is_file()


def test_malformed_sources_persist_source_gate_failures_for_json_only_aggregation(
    tmp_path: Path,
) -> None:
    batch = tmp_path / "batch"
    archive = batch / "quality_archive"
    cases = (("hdf5", "broken-hdf"), ("lerobot", "broken-lerobot"))

    for source_format, asset_id in cases:
        source = batch / asset_id
        if source_format == "hdf5":
            write_standard_hdf5_episode(source, asset_id=asset_id)
            (source / f"{asset_id}.h5").write_bytes(b"not hdf5")
        else:
            write_standard_lerobot_dataset(source, episodes=((0, asset_id),))
            (source / "meta/info.json").write_text("{}", encoding="utf-8")
        result = _run(
            "run_canonical_qc.py",
            "--source", source,
            "--source-format", source_format,
            "--source-root", source,
            "--batch-root", batch,
            "--quality-archive", archive,
            "--profile", "acceptance",
            *_identity_args(asset_id),
        )

        assert result.returncode == 2
        assert result.stderr == ""
        payload = _json_line(result.stdout)
        assert payload["category"] == "quality_fail"
        assert payload["result"]["asset_id"] == asset_id
        report_path = archive / f"{asset_id}.json"
        assert payload["result"]["report_path"] == str(report_path.resolve())
        report = json.loads(report_path.read_text())
        assert report["overall_decision"] == "fail"
        assert report["pipeline_state"]["status"] == "stopped"
        assert report["execution"]["module_states"]["source_gate"] == {
            "state": "completed"
        }
        assert report["issues"][0]["module"] == "source_gate"
        assert report["issues"][0]["severity"] == "fail"
        if source_format == "hdf5":
            missing_batch = json.loads(json.dumps(report))
            missing_batch.pop("batch_id")
            with pytest.raises(ValueError, match="batch_id"):
                validate_asset_qc_report(missing_batch)
            empty_gate = json.loads(json.dumps(report))
            empty_gate["source_gate"] = {}
            with pytest.raises(ValueError, match="source_gate"):
                validate_asset_qc_report(empty_gate)
            contradictory_gate = json.loads(json.dumps(report))
            contradictory_gate["source_gate"]["flow"]["result_gate"] = {
                "verdict": "pass",
                "has_fail": False,
                "has_warn": False,
            }
            with pytest.raises(ValueError, match="source_gate"):
                validate_asset_qc_report(contradictory_gate)
            missing_result_gate = json.loads(json.dumps(report))
            missing_result_gate["source_gate"]["flow"].pop("result_gate")
            with pytest.raises(ValueError, match="source_gate"):
                validate_asset_qc_report(missing_result_gate)
            fake_terminal_pass = json.loads(json.dumps(report))
            fake_terminal_pass["pipeline_state"]["status"] = "completed"
            fake_terminal_pass["overall_decision"] = "pass"
            with pytest.raises(ValueError, match="pipeline_state|overall_decision"):
                validate_asset_qc_report(fake_terminal_pass)
        report_bytes = report_path.read_bytes()
        repeated = _run(
            "run_canonical_qc.py",
            "--source", source,
            "--source-format", source_format,
            "--source-root", source,
            "--batch-root", batch,
            "--quality-archive", archive,
            "--profile", "acceptance",
            *_identity_args(asset_id),
        )
        assert repeated.returncode == 2
        assert report_path.read_bytes() == report_bytes
        drifted = _run(
            "run_canonical_qc.py",
            "--source", source,
            "--source-format", source_format,
            "--source-root", source,
            "--batch-root", batch,
            "--quality-archive", archive,
            "--profile", "acceptance",
            *_identity_args(asset_id),
            "--config", ROOT / "configs/canonical_qc/canonical_qc_v1.1.1.yaml",
        )
        assert drifted.returncode == 2
        assert _json_line(drifted.stderr)["error"]["code"] == "input_contract_error"
        assert report_path.read_bytes() == report_bytes

    for _source_format, asset_id in cases:
        shutil.rmtree(batch / asset_id)
    projection = project_quality_archive(archive)
    assert {row["batch_id"] for row in projection.asset_rows} == {"batch-001"}
    assert {row["batch_id"] for row in projection.source_manifest} == {"batch-001"}
    summary = aggregate_projection(projection)["overall"]
    assert summary["asset_count"] == 2
    assert summary["terminal_asset_count"] == 2
    assert summary["final_fail_asset_count"] == 2
    assert summary["automatic_hard_fail_asset_count"] == 2
    assert summary["automatic_hard_fail_issue_count"] == 2
    assert summary["module_state_counts"]["source_gate"] == {"completed": 2}


def test_declared_identity_mismatch_persists_source_gate_failure(
    tmp_path: Path,
) -> None:
    batch = tmp_path / "batch"
    source = batch / "asset-001"
    write_standard_hdf5_episode(source)
    archive = batch / "quality_archive"

    result = _run(
        "run_canonical_qc.py",
        "--source", source,
        "--source-format", "hdf5",
        "--source-root", source,
        "--batch-root", batch,
        "--quality-archive", archive,
        "--profile", "acceptance",
        *_identity_args("declared-other"),
    )

    assert result.returncode == 2
    assert result.stderr == ""
    payload = _json_line(result.stdout)
    assert payload["error"]["code"] == "identity_mismatch"
    report = json.loads((archive / "declared-other.json").read_text())
    assert report["asset_id"] == "declared-other"
    assert report["issues"][0]["observed_value"] == {
        "code": "identity_mismatch",
        "field": "asset_id",
    }


def test_publish_cli_distinguishes_prerequisite_validation_and_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "asset-001"
    write_standard_hdf5_episode(source)
    episode = StandardHdf5Adapter().load(source)
    plan = _plan_for_episode(tmp_path / "fixture", source, episode)
    args = [
        "--source", str(source),
        "--source-format", "hdf5",
        "--canonical-source-root", str(source),
        "--qc-report", str(plan.request.qc_report_path),
        "--release-root", str(tmp_path / "curated"),
    ]
    import tools.publish_lerobot_v3 as cli

    for code, stage, expected_exit, category in (
        ("publish_prerequisite_failed", "publish_prerequisite", 4, "publish_prerequisite"),
        ("source_integrity_error", "publish_prerequisite", 4, "publish_prerequisite"),
        ("validation_failed", "publish_validation", 5, "publish_validation"),
        ("staging_failed", "publish_staging", 3, "publish_runtime"),
        ("commit_conflict", "publish_commit", 6, "commit_conflict"),
    ):
        monkeypatch.setattr(
            cli,
            "publish_from_paths",
            lambda **_kwargs: (_ for _ in ()).throw(
                PublishPrerequisiteError(
                    CanonicalDiagnostic(code, stage, "fixture", "boom", False)
                )
            ),
        )
        assert cli.main(args) == expected_exit
        captured = capsys.readouterr()
        assert captured.out == ""
        error = _json_line(captured.err)["error"]
        assert error["category"] == category
        assert error["code"] == code

    monkeypatch.setattr(
        cli,
        "publish_from_paths",
        lambda **_kwargs: (_ for _ in ()).throw(
            PublishPrerequisiteError(
                CanonicalDiagnostic(
                    "source_integrity_error",
                    "publish_prerequisite",
                    "source",
                    "temporary read failure",
                    True,
                )
            )
        ),
    )
    assert cli.main(args) == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    error = _json_line(captured.err)["error"]
    assert error["category"] == "publish_runtime"
    assert error["code"] == "source_integrity_error"
    assert error["retryable"] is True

    monkeypatch.setattr(
        cli,
        "publish_from_paths",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("disk unavailable")),
    )
    assert cli.main(args) == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    assert _json_line(captured.err)["error"]["category"] == "publish_runtime"

    monkeypatch.setattr(
        cli,
        "publish_from_paths",
        lambda **_kwargs: (_ for _ in ()).throw(
            CanonicalInputError(
                "source_integrity_error",
                "source",
                "temporary adapter read failure",
                retryable=True,
            )
        ),
    )
    assert cli.main(args) == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    error = _json_line(captured.err)["error"]
    assert error["category"] == "publish_runtime"
    assert error["code"] == "source_integrity_error"
    assert error["retryable"] is True


def test_cli_rejects_symlink_release_root_and_overlapping_quality_archive(
    tmp_path: Path,
) -> None:
    source = tmp_path / "batch/asset-001"
    write_standard_hdf5_episode(source)
    episode = StandardHdf5Adapter().load(source)
    plan = _plan_for_episode(tmp_path / "fixture", source, episode)
    real_release = tmp_path / "real-release"
    real_release.mkdir()
    linked_release = tmp_path / "linked-release"
    linked_release.symlink_to(real_release, target_is_directory=True)

    publish_result = _run(
        "publish_lerobot_v3.py",
        "--source", source,
        "--source-format", "hdf5",
        "--canonical-source-root", source,
        "--qc-report", plan.request.qc_report_path,
        "--release-root", linked_release,
        "--dry-run",
    )
    assert publish_result.returncode == 2
    assert _json_line(publish_result.stderr)["error"]["category"] == "input_contract"
    assert list(real_release.iterdir()) == []

    run_result = _run(
        "run_canonical_qc.py",
        "--source", source,
        "--source-format", "hdf5",
        "--source-root", source,
        "--batch-root", tmp_path / "batch",
        "--quality-archive", source / "quality_archive",
        "--profile", "acceptance",
        *_identity_args(),
        "--dry-run",
    )
    assert run_result.returncode == 2
    assert _json_line(run_result.stderr)["error"]["field"] == "quality_archive"


def test_run_cli_runtime_error_has_stable_exit_and_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    import tools.run_canonical_qc as cli

    monkeypatch.setattr(
        cli,
        "run_canonical_source_qc",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("runner exploded")),
    )
    code = cli.main([
        "--source", str(tmp_path),
        "--source-format", "hdf5",
        "--source-root", str(tmp_path),
        "--batch-root", str(tmp_path),
        "--quality-archive", str(tmp_path / "quality_archive"),
        "--profile", "acceptance",
        *_identity_args(),
    ])

    assert code == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    error = _json_line(captured.err)["error"]
    assert error == {
        "category": "qc_runtime",
        "code": "qc_runtime_error",
        "stage": "qc",
        "field": None,
        "message": "runner exploded",
        "retryable": True,
    }

    monkeypatch.setattr(
        cli,
        "run_canonical_source_qc",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("disk unavailable")),
    )
    assert cli.main([
        "--source", str(tmp_path),
        "--source-format", "hdf5",
        "--source-root", str(tmp_path),
        "--batch-root", str(tmp_path),
        "--quality-archive", str(tmp_path / "quality_archive"),
        "--profile", "acceptance",
        *_identity_args(),
    ]) == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    error = _json_line(captured.err)["error"]
    assert error["category"] == "qc_runtime"
    assert error["retryable"] is True


def test_hdf5_adapter_io_error_reaches_run_cli_as_retryable_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    batch = tmp_path / "batch"
    source = batch / "asset-001"
    write_standard_hdf5_episode(source)
    original_open = Path.open

    def fail_hdf5_open(path: Path, *args: object, **kwargs: object) -> object:
        if path.suffix == ".h5" and args and args[0] == "rb":
            raise OSError("temporary HDF5 read failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_hdf5_open)
    import tools.run_canonical_qc as cli

    code = cli.main([str(item) for item in (
        "--source", source,
        "--source-format", "hdf5",
        "--source-root", source,
        "--batch-root", batch,
        "--quality-archive", batch / "quality_archive",
        "--profile", "acceptance",
        *_identity_args(),
    )])
    captured = capsys.readouterr()

    assert code == 3
    assert captured.out == ""
    error = _json_line(captured.err)["error"]
    assert error["category"] == "qc_runtime"
    assert error["code"] == "source_integrity_error"
    assert error["field"] == "asset-001.h5"
    assert error["retryable"] is True


def test_source_gate_runtime_error_recovers_on_same_report_and_preserves_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    batch = tmp_path / "batch"
    source = batch / "asset-001"
    archive = batch / "quality_archive"
    write_standard_hdf5_episode(source)
    import canonical_qc.workflow as workflow
    import tools.run_canonical_qc as cli

    monkeypatch.setattr(
        cli,
        "run_canonical_source_qc",
        lambda **kwargs: workflow.run_canonical_source_qc(
            **kwargs, registry_factory=_pass_registry
        ),
    )
    args = [str(item) for item in (
        "--source", source,
        "--source-format", "hdf5",
        "--source-root", source,
        "--batch-root", batch,
        "--quality-archive", archive,
        "--profile", "acceptance",
        *_identity_args(),
    )]
    original_open = Path.open
    failure_attempt = 0

    def fail_hdf5_open(path: Path, *open_args: object, **kwargs: object) -> object:
        nonlocal failure_attempt
        if path.suffix == ".h5" and open_args and open_args[0] == "rb":
            failure_attempt += 1
            raise OSError(f"temporary HDF5 read failure attempt {failure_attempt}")
        return original_open(path, *open_args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_hdf5_open)
    assert cli.main(args) == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    first_error = _json_line(captured.err)
    assert first_error["error"]["retryable"] is True
    report_path = archive / "asset-001.json"
    failed = json.loads(report_path.read_text())
    assert failed["report_revision"] == 1
    assert failed["pipeline_state"]["status"] == "error"
    unexpected_result_gate = json.loads(json.dumps(failed))
    unexpected_result_gate["source_gate"]["flow"]["result_gate"] = {
        "verdict": "pass",
        "has_fail": False,
        "has_warn": False,
    }
    with pytest.raises(ValueError, match="source_gate"):
        validate_asset_qc_report(unexpected_result_gate)
    failed_bytes = report_path.read_bytes()

    assert cli.main(args) == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    assert _json_line(captured.err)["error"]["retryable"] is True
    assert report_path.read_bytes() == failed_bytes

    monkeypatch.setattr(Path, "open", original_open)
    assert cli.main([
        *args,
        "--config",
        str(ROOT / "configs/canonical_qc/canonical_qc_v1.1.1.yaml"),
    ]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert _json_line(captured.err)["error"]["code"] == "input_contract_error"
    assert report_path.read_bytes() == failed_bytes

    assert cli.main(args) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert _json_line(captured.out)["result"]["state"] == "awaiting_external"
    recovered = json.loads(report_path.read_text())
    assert recovered["report_revision"] > 2
    assert recovered["runtime_errors"] == []
    assert recovered["execution"]["runtime_error_history"] == failed["runtime_errors"]
    assert recovered["execution"]["module_states"]["source_gate"] == {
        "state": "completed"
    }


def _source_gate_fixture(tmp_path: Path) -> tuple[
    Path,
    object,
    DeclaredEpisodeIdentity,
    SourceGateLocator,
    object,
    object,
]:
    batch = tmp_path / "batch"
    source = batch / "asset-001"
    write_standard_hdf5_episode(source)
    episode = StandardHdf5Adapter().load(source)
    report_path = batch / "quality_archive" / "asset-001.json"
    context = CanonicalQcBridge(episode, source_root=source).asset_context(
        batch_root=batch, report_path=report_path
    )
    identity = DeclaredEpisodeIdentity(
        asset_id="asset-001", batch_id="batch-001", supplier_id="supplier-001"
    )
    locator = SourceGateLocator(
        source_path="asset-001",
        source_root="asset-001",
        source_format="hdf5",
        episode_index=None,
    )
    return (
        report_path,
        context,
        identity,
        locator,
        load_canonical_qc_config(),
        load_qc_acceptance_config(),
    )


def test_source_gate_pass_resumes_downstream_runtime_without_rebuilding_completed_data(
    tmp_path: Path,
) -> None:
    report_path, context, identity, locator, canonical, qc = _source_gate_fixture(
        tmp_path
    )
    report = record_source_gate_pass(
        report_path,
        context=context,
        identity=identity,
        locator=locator,
        canonical_config=canonical,
        qc_config=qc,
        profile="acceptance",
    )
    report["report_revision"] = 2
    report["hdf5_text_info"] = {
        "flow": {},
        "evaluation": {},
        "metrics": {},
        "evidence": [{"evidence_id": "kept-evidence"}],
    }
    report["issues"] = [{"issue_id": "kept-issue", "module": "hdf5_text_info"}]
    report["execution"]["module_states"]["hdf5_text_info"] = {"state": "completed"}
    report["execution"]["module_states"]["quality_hand"] = {
        "state": "runtime_error",
        "reason": "process_error",
    }
    report["pipeline_state"] = {
        "status": "error",
        "last_completed_module": "hdf5_text_info",
        "next_module": "quality_hand",
        "stop_reason": "process_error",
    }
    report["runtime_errors"] = [{
        "module": "quality_hand",
        "error_type": "process_error",
        "message": "temporary downstream failure",
        "occurred_at": "2026-07-16T00:00:00Z",
        "retryable": True,
    }]
    write_asset_qc_report(
        report_path, report, expected_revision=1, profile="acceptance"
    )
    before = report_path.read_bytes()

    resumed = record_source_gate_pass(
        report_path,
        context=context,
        identity=identity,
        locator=locator,
        canonical_config=canonical,
        qc_config=qc,
        profile="acceptance",
    )

    assert report_path.read_bytes() != before
    assert resumed["report_revision"] == 3
    assert resumed["hdf5_text_info"]["evidence"] == [
        {"evidence_id": "kept-evidence"}
    ]
    assert resumed["issues"] == [
        {"issue_id": "kept-issue", "module": "hdf5_text_info"}
    ]
    assert resumed["runtime_errors"] == []
    assert resumed["execution"]["runtime_error_history"] == report["runtime_errors"]
    assert "quality_hand" not in resumed["execution"]["module_states"]
    assert resumed["pipeline_state"] == {
        "status": "running",
        "last_completed_module": "hdf5_text_info",
        "next_module": "quality_hand",
        "stop_reason": None,
    }


def test_workflow_resume_retries_downstream_runtime_without_losing_completed_data(
    tmp_path: Path,
) -> None:
    batch = tmp_path / "batch"
    source = batch / "asset-001"
    archive = batch / "quality_archive"
    write_standard_hdf5_episode(source)
    attempt = 0
    calls: list[tuple[int, str]] = []

    def registry_factory(_context: object, config: object) -> ModuleRegistry:
        registry = ModuleRegistry()
        for module_name in config.pipeline_modules:
            module = config.module_config(module_name)
            implementation = module.get("implementation")
            if not module.get("enabled") or not isinstance(implementation, str):
                continue

            def run(
                _context: object,
                _config: object,
                *,
                module_name: str = module_name,
            ) -> ModuleResult:
                calls.append((attempt, module_name))
                if attempt == 0 and module_name == "quality_hand":
                    raise RuntimeError("temporary quality_hand dependency failure")
                if module_name == "hdf5_text_info":
                    evidence = EvidenceRef(
                        "resume-evidence",
                        "test",
                        "evidence/resume.json",
                        "asset",
                    )
                    issue = Issue(
                        "hdf5_text_info:warn:resume",
                        "resume_warn",
                        "warn",
                        module_name,
                        "resume_fixture",
                        "fixture_metric",
                        1,
                        ">",
                        0,
                        "hdf5_text_info.resume_fixture",
                        True,
                        evidence_ids=(evidence.evidence_id,),
                    )
                    return ModuleResult(
                        module_name,
                        "warn",
                        {"decision": "warn"},
                        {"fixture_metric": 1},
                        issues=(issue,),
                        evidence=(evidence,),
                    )
                return ModuleResult(
                    module_name, "pass", {"decision": "pass"}, {}
                )

            registry.register(implementation, run)
        return registry

    kwargs = dict(
        source=source,
        source_format="hdf5",
        source_root=source,
        batch_root=batch,
        quality_archive=archive,
        profile="acceptance",
        expected_asset_id="asset-001",
        expected_batch_id="batch-001",
        expected_supplier_id="supplier-001",
        registry_factory=registry_factory,
    )
    first = run_canonical_source_qc(**kwargs)
    assert first.status == "error"
    failed = json.loads((archive / "asset-001.json").read_text())
    preserved_block = failed["hdf5_text_info"]
    preserved_issues = failed["issues"]
    first_runtime_errors = failed["runtime_errors"]
    failed_revision = failed["report_revision"]

    attempt = 1
    resumed = run_canonical_source_qc(**kwargs)
    final_report = json.loads((archive / "asset-001.json").read_text())

    assert resumed.status == "awaiting_external"
    assert (1, "hdf5_text_info") not in calls
    assert (1, "quality_hand") in calls
    assert final_report["report_revision"] > failed_revision + 1
    assert final_report["hdf5_text_info"] == preserved_block
    assert final_report["issues"] == preserved_issues
    assert final_report["hdf5_text_info"]["evidence"][0]["evidence_id"] == (
        "resume-evidence"
    )
    assert final_report["runtime_errors"] == []
    assert final_report["execution"]["runtime_error_history"] == first_runtime_errors


@pytest.mark.parametrize("field", ["asset_id", "batch_id", "supplier_id"])
def test_source_gate_resume_rejects_tampered_declared_identity(
    tmp_path: Path,
    field: str,
) -> None:
    report_path, context, identity, locator, canonical, qc = _source_gate_fixture(
        tmp_path
    )
    report = record_source_gate_pass(
        report_path,
        context=context,
        identity=identity,
        locator=locator,
        canonical_config=canonical,
        qc_config=qc,
        profile="acceptance",
    )
    report["source_gate"]["evaluation"]["declared_identity"][field] = "tampered"
    report["report_revision"] = 2
    write_asset_qc_report(
        report_path, report, expected_revision=1, profile="acceptance"
    )
    before = report_path.read_bytes()

    with pytest.raises(CanonicalInputError, match="declared_identity"):
        record_source_gate_pass(
            report_path,
            context=context,
            identity=identity,
            locator=locator,
            canonical_config=canonical,
            qc_config=qc,
            profile="acceptance",
        )

    assert report_path.read_bytes() == before


def test_source_gate_retryable_runtime_can_transition_to_deterministic_fail(
    tmp_path: Path,
) -> None:
    report_path, _context, identity, locator, canonical, qc = _source_gate_fixture(
        tmp_path
    )
    transient = CanonicalInputError(
        "source_integrity_error",
        "source",
        "temporary read failure",
        retryable=True,
    )
    first = record_source_gate_failure(
        report_path,
        identity=identity,
        locator=locator,
        batch_root=tmp_path / "batch",
        diagnostic=transient,
        canonical_config=canonical,
        qc_config=qc,
        profile="acceptance",
    )
    deterministic = CanonicalInputError(
        "field_mapping_error",
        "episode_index",
        "must be non-negative",
    )

    failed = record_source_gate_failure(
        report_path,
        identity=identity,
        locator=locator,
        batch_root=tmp_path / "batch",
        diagnostic=deterministic,
        canonical_config=canonical,
        qc_config=qc,
        profile="acceptance",
    )

    assert failed["report_revision"] == first["report_revision"] + 1
    assert failed["source_gate"]["evaluation"]["status"] == "fail"
    assert failed["overall_decision"] == "fail"
    assert failed["runtime_errors"] == []
    assert failed["execution"]["runtime_error_history"] == first["runtime_errors"]


@pytest.mark.parametrize(
    ("source_format", "episode_index", "expected_field"),
    [
        ("bogus", None, "source_format"),
        ("lerobot", -1, "episode_index"),
    ],
)
def test_invalid_locator_values_still_persist_deterministic_source_gate_json(
    tmp_path: Path,
    source_format: str,
    episode_index: int | None,
    expected_field: str,
) -> None:
    batch = tmp_path / source_format
    source = batch / "asset-001"
    if source_format == "lerobot":
        write_standard_lerobot_dataset(source)
    else:
        source.mkdir(parents=True)
    archive = batch / "quality_archive"
    args: list[object] = [
        "--source", source,
        "--source-format", source_format,
        "--source-root", source,
        "--batch-root", batch,
        "--quality-archive", archive,
        "--profile", "acceptance",
        *_identity_args(),
    ]
    if episode_index is not None:
        args.extend(("--episode-index", episode_index))

    result = _run("run_canonical_qc.py", *args)

    assert result.returncode == 2
    payload = _json_line(result.stdout)
    assert payload["category"] == "quality_fail"
    assert payload["error"]["field"] == expected_field
    report = json.loads((archive / "asset-001.json").read_text())
    assert report["source_gate"]["evaluation"]["status"] == "fail"
    assert report["source_gate"]["evaluation"]["locator"][expected_field] == (
        source_format if expected_field == "source_format" else episode_index
    )
    validate_asset_qc_report(report)


def test_corrupt_hdf5_reaches_run_cli_as_deterministic_input_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    batch = tmp_path / "batch"
    source = batch / "asset-001"
    write_standard_hdf5_episode(source)
    (source / "asset-001.h5").write_bytes(b"not hdf5")
    import tools.run_canonical_qc as cli

    code = cli.main([str(item) for item in (
        "--source", source,
        "--source-format", "hdf5",
        "--source-root", source,
        "--batch-root", batch,
        "--quality-archive", batch / "quality_archive",
        "--profile", "acceptance",
        *_identity_args(),
    )])
    captured = capsys.readouterr()

    assert code == 2
    assert captured.err == ""
    payload = _json_line(captured.out)
    assert payload["category"] == "quality_fail"
    error = payload["error"]
    assert error["category"] == "input_contract"
    assert error["code"] == "source_integrity_error"
    assert error["field"] == "source"
    assert error["retryable"] is False


def test_run_cli_distinguishes_retryable_source_io_from_deterministic_input(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    import tools.run_canonical_qc as cli

    args = [
        "--source", str(tmp_path),
        "--source-format", "hdf5",
        "--source-root", str(tmp_path),
        "--batch-root", str(tmp_path),
        "--quality-archive", str(tmp_path / "quality_archive"),
        "--profile", "acceptance",
        *_identity_args(),
    ]
    for retryable, expected_exit, expected_category in (
        (False, 2, "input_contract"),
        (True, 3, "qc_runtime"),
    ):
        monkeypatch.setattr(
            cli,
            "run_canonical_source_qc",
            lambda **_kwargs: (_ for _ in ()).throw(
                CanonicalInputError(
                    "source_integrity_error",
                    "source",
                    "temporary read failure" if retryable else "hash mismatch",
                    retryable=retryable,
                )
            ),
        )
        assert cli.main(args) == expected_exit
        captured = capsys.readouterr()
        assert captured.out == ""
        error = _json_line(captured.err)["error"]
        assert error["category"] == expected_category
        assert error["code"] == "source_integrity_error"
        assert error["retryable"] is retryable


def test_run_cli_persisted_orchestrator_error_uses_same_stderr_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    batch = tmp_path / "batch"
    source = batch / "asset-001"
    write_standard_hdf5_episode(source)

    import canonical_qc.workflow as workflow
    import tools.run_canonical_qc as cli

    monkeypatch.setattr(
        cli,
        "run_canonical_source_qc",
        lambda **kwargs: workflow.run_canonical_source_qc(
            **kwargs, registry_factory=lambda _context, _config: ModuleRegistry()
        ),
    )
    args = [str(item) for item in (
        "--source", source,
        "--source-format", "hdf5",
        "--source-root", source,
        "--batch-root", batch,
        "--quality-archive", batch / "quality_archive",
        "--profile", "acceptance",
        *_identity_args(),
    )]
    code = cli.main(args)
    captured = capsys.readouterr()

    assert code == 3
    assert captured.out == ""
    error = _json_line(captured.err)["error"]
    assert error["category"] == "qc_runtime"
    assert error["code"] == "module_unavailable"
    assert error["field"] == "hdf5_text_info"
    report = json.loads((batch / "quality_archive/asset-001.json").read_text())
    assert report["pipeline_state"]["status"] == "error"
    assert report["overall_decision"] is None


def test_run_cli_awaiting_external_is_successful_and_resume_is_byte_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, payload = _run_cli_fixture(tmp_path, "hdf5", monkeypatch, capsys)
    assert payload["result"]["state"] == "awaiting_external"
    assert payload["result"]["overall_decision"] is None
    assert source.name == "asset-001"


def test_run_cli_quality_fail_and_argparse_have_frozen_machine_contract(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    import tools.run_canonical_qc as cli

    episode = SimpleNamespace(identity=SimpleNamespace(asset_id="asset-001"))
    result = SimpleNamespace(
        episode=episode,
        source=tmp_path,
        source_format="hdf5",
        status="stopped",
        overall_decision="fail",
        report_path=tmp_path / "quality_archive/asset-001.json",
        report_revision=4,
        executed_modules=("keypoint_presence",),
        resumed=False,
        dry_run=False,
        canonical_config_path=ROOT / "configs/canonical_qc.yaml",
        canonical_config_version="canonical_qc_v1.0.0",
        canonical_config_hash="sha256:" + "1" * 64,
        qc_config_path=ROOT / "configs/qc_acceptance.yaml",
        qc_config_version="qc_acceptance_v2.1.0",
        qc_config_hash="sha256:" + "2" * 64,
        runtime_error=None,
    )
    monkeypatch.setattr(cli, "run_canonical_source_qc", lambda **_kwargs: result)
    code = cli.main([
        "--source", str(tmp_path), "--source-format", "hdf5",
        "--source-root", str(tmp_path), "--batch-root", str(tmp_path),
        "--quality-archive", str(tmp_path / "quality_archive"),
        "--profile", "acceptance",
        *_identity_args(),
    ])
    captured = capsys.readouterr()
    assert code == 2 and captured.err == ""
    payload = _json_line(captured.out)
    assert payload["category"] == "quality_fail"
    assert payload["result"]["overall_decision"] == "fail"
    assert payload["result"]["report_path"].endswith("asset-001.json")

    assert cli.main([]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert _json_line(captured.err)["error"]["code"] == "cli_usage_error"


def test_publish_failure_keeps_current_and_selected_release_bytes_unchanged(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset-001"
    write_standard_hdf5_episode(source)
    episode = StandardHdf5Adapter().load(source)
    plan = _plan_for_episode(tmp_path / "fixture", source, episode)
    release_root = tmp_path / "curated"
    args = (
        "--source", source,
        "--source-format", "hdf5",
        "--canonical-source-root", source,
        "--qc-report", plan.request.qc_report_path,
        "--release-root", release_root,
    )
    first = _run("publish_lerobot_v3.py", *args)
    assert first.returncode == 0, first.stderr
    current_path = release_root / "CURRENT.json"
    current_before = current_path.read_bytes()
    current = json.loads(current_before)
    selected = release_root / "releases" / current["release_id"]
    release_before = {
        path.relative_to(selected).as_posix(): path.read_bytes()
        for path in selected.rglob("*")
        if path.is_file()
    }
    report = json.loads(plan.request.qc_report_path.read_text())
    report["overall_decision"] = "fail"
    plan.request.qc_report_path.write_text(json.dumps(report))

    rejected = _run("publish_lerobot_v3.py", *args)

    assert rejected.returncode == 4
    assert _json_line(rejected.stderr)["error"]["category"] == "publish_prerequisite"
    assert current_path.read_bytes() == current_before
    assert {
        path.relative_to(selected).as_posix(): path.read_bytes()
        for path in selected.rglob("*")
        if path.is_file()
    } == release_before
