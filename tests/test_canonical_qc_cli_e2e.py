from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from canonical_qc import StandardHdf5Adapter, StandardLeRobotAdapter
from canonical_qc.config import load_canonical_qc_config
from canonical_qc.errors import CanonicalInputError
from canonical_qc.provenance import semantic_fingerprint
from lerobot_v3_publisher import CanonicalDiagnostic, PublishPrerequisiteError
from qc_common.config import load_qc_acceptance_config
from qc_common.contracts import ModuleResult
from qc_common.module_registry import ModuleRegistry
from qc_common.report import write_asset_qc_report
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
    snapshot = ROOT / "configs/canonical_qc/canonical_qc_v1.0.0.yaml"
    assert alias.read_bytes() == snapshot.read_bytes()
    assert loaded.config_version == "canonical_qc_v1.0.0"
    assert loaded.path == alias.resolve()
    assert loaded.qc_config_path == (ROOT / "configs/qc_acceptance.yaml").resolve()


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
    strict_config = ROOT / "configs/canonical_qc/canonical_qc_v1.0.1.yaml"
    rejected = _run(
        "run_canonical_qc.py",
        "--source", source,
        "--source-format", "lerobot",
        "--source-root", source,
        "--batch-root", batch,
        "--quality-archive", batch / "quality_archive",
        "--profile", "acceptance",
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
        module: {
            "state": "completed" if qc_config.module_config(module)["enabled"] else "disabled"
        }
        for module in qc_config.pipeline_modules
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
    )

    assert result.returncode == 2
    assert result.stdout == ""
    error = _json_line(result.stderr)["error"]
    assert error["category"] == "input_contract"
    assert error["retryable"] is False


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
    )])
    captured = capsys.readouterr()

    assert code == 3
    assert captured.out == ""
    error = _json_line(captured.err)["error"]
    assert error["category"] == "qc_runtime"
    assert error["code"] == "source_integrity_error"
    assert error["field"] == "asset-001.h5"
    assert error["retryable"] is True


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
    )])
    captured = capsys.readouterr()

    assert code == 2
    assert captured.out == ""
    error = _json_line(captured.err)["error"]
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
