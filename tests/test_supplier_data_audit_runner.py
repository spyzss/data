from __future__ import annotations

import json
from pathlib import Path

import pytest

from qc_common.config import LoadedQcConfig
from qc_pipeline.context import AssetContext
from tests.fixtures import solid_frame, write_test_video


def config(tmp_path: Path) -> LoadedQcConfig:
    disabled_prechecks = {
        name: {
            "enabled": False,
            "disabled_reason": "not_in_test_pipeline",
            "parameters": {},
            "rules": {},
        }
        for name in (
            "hdf5_text_info",
            "quality_hand",
            "keypoint_presence",
            "keypoint_morphology",
            "keypoint_temporal",
            "video_quality",
            "sam3_containment",
        )
    }
    return LoadedQcConfig(
        path=tmp_path / "qc.yaml",
        raw={
            "schema_version": "qc_acceptance_config_schema.v2",
            "config_version": "qc_acceptance_v2.2.0",
            "config_name": "test",
            "execution_profiles": {
                "supplier_evaluation": {
                    "fail_action": "record_and_continue",
                    "runtime_error_action": "record_and_continue",
                }
            },
            "pipeline": {
                "default_profile": "supplier_evaluation",
                "modules": ["supplier_data_audit"],
            },
            "modules": {
                **disabled_prechecks,
                "supplier_data_audit": {
                    "enabled": True,
                    "implementation": "supplier_data_audit.v1",
                    "parameters": {
                        "suppliers": {
                            "potentia": {
                                "mapping_status": "unverified",
                                "mapping": {},
                            }
                        }
                    },
                    "rules": {
                        "required_source_missing": {
                            "rule_id": "supplier_data_audit.required_source_missing",
                            "verdict": "fail",
                        },
                        "mapping_unverified": {
                            "rule_id": "supplier_data_audit.mapping_unverified",
                            "verdict": "warn",
                        },
                    },
                }
            },
        },
        sha256="sha256:" + "7" * 64,
    )


def context(tmp_path: Path) -> AssetContext:
    source = tmp_path / "source"
    source.mkdir()
    names = {
        "video": "video.mp4",
        "meta": "meta.json",
        "frames": "frames.csv",
        "aligned": "aligned.csv",
        "imu": "imu.csv",
        "calibration": "calibration.json",
    }
    for source_name, filename in names.items():
        path = source / filename
        if path.suffix == ".mp4":
            write_test_video(
                path,
                [solid_frame(80, width=160, height=120)],
                fps=30.0,
            )
        elif path.suffix == ".json":
            path.write_text("{}", encoding="utf-8")
        elif path.suffix == ".csv":
            path.write_text("frame_index,timestamp\n1,0.0\n", encoding="utf-8")
        else:
            path.write_bytes(b"video")
    return AssetContext(
        asset_id="potentia__task-a",
        batch_root=tmp_path,
        report_path=tmp_path / "quality_archive" / "potentia__task-a.json",
        source_files={
            name: {"path": f"source/{filename}"}
            for name, filename in names.items()
        },
        metadata={
            "supplier": "potentia",
            "supplier_id": "potentia",
            "task_id": "task-a",
            "manifest_row": {"supplier": "potentia", "task_id": "task-a"},
            "reuse_artifacts": True,
        },
    )


def test_supplier_data_audit_writes_isolated_reusable_artifact(
    tmp_path: Path,
) -> None:
    from qc_pipeline.artifacts import artifact_for
    from qc_pipeline.runners.supplier_data_audit import run

    ctx = context(tmp_path)
    loaded = config(tmp_path)

    first = run(ctx, loaded)
    second = run(ctx, loaded)

    artifact = artifact_for(ctx, "supplier_data_audit")
    payload = json.loads(
        (artifact.directory / "supplier_data_audit_result.json").read_text(
            encoding="utf-8"
        )
    )
    run_config = json.loads(artifact.run_config_path.read_text(encoding="utf-8"))
    assert first.module == "supplier_data_audit"
    assert first.verdict == "warn"
    assert first.evaluation["mapping_status"] == "unverified"
    assert first.runtime["artifact_state"] == "computed"
    assert second.runtime["artifact_state"] == "reused"
    assert payload["raw_result"]["inventory"]["meta"]["status"] == "present"
    assert payload["raw_result"]["schema_version"] == "supplier_data_audit.raw.v2"
    assert payload["raw_result"]["mapping_config_identity"].startswith("sha256:")
    assert payload["module_result"]["module"] == "supplier_data_audit"
    assert run_config["fingerprint"]["implementation_version"] == (
        "supplier-data-audit-producer-v3"
    )
    assert run_config["fingerprint"]["output_schema_version"] == (
        "supplier_data_audit.raw.v2"
    )
    assert artifact.required_files == (
        "supplier_data_audit_result.json",
        "run_config.json",
    )
    assert not (tmp_path / "module_outputs" / ctx.asset_id / "precheck").exists()
    assert not (tmp_path / "module_outputs" / ctx.asset_id / "video_quality").exists()


def test_supplier_data_audit_missing_file_is_fail_not_pass(tmp_path: Path) -> None:
    from qc_pipeline.runners.supplier_data_audit import run

    ctx = context(tmp_path)
    (tmp_path / "source" / "imu.csv").unlink()

    result = run(ctx, config(tmp_path))

    assert result.verdict == "fail"
    assert result.metrics["missing_source_count"] == 1
    assert result.evaluation["decision"] == "fail"
    assert any(issue.code == "required_source_missing" for issue in result.issues)


@pytest.mark.parametrize(
    "legacy_version",
    ["supplier-data-audit-producer-v1", "supplier-data-audit-producer-v2"],
)
def test_supplier_data_audit_legacy_cache_is_stale_for_v3_producer(
    tmp_path: Path,
    monkeypatch,
    legacy_version: str,
) -> None:
    from qc_pipeline.runners import supplier_data_audit

    ctx = context(tmp_path)
    loaded = config(tmp_path)
    with monkeypatch.context() as legacy:
        legacy.setattr(
            supplier_data_audit,
            "_IMPLEMENTATION_VERSION",
            legacy_version,
        )
        old_result = supplier_data_audit.run(ctx, loaded)

    current_result = supplier_data_audit.run(ctx, loaded)

    assert old_result.runtime["artifact_state"] == "computed"
    assert current_result.runtime["artifact_state"] == "computed"
    assert old_result.runtime["fingerprint_sha256"] != current_result.runtime[
        "fingerprint_sha256"
    ]


def test_supplier_data_audit_output_schema_change_invalidates_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from acceptance_pull import supplier_audit
    from qc_pipeline.runners import supplier_data_audit

    ctx = context(tmp_path)
    loaded = config(tmp_path)
    current_audit = supplier_audit.audit_supplier_data
    with monkeypatch.context() as legacy:
        legacy.setattr(
            supplier_data_audit,
            "_RAW_OUTPUT_SCHEMA_VERSION",
            "supplier_data_audit.raw.v1",
        )
        legacy.setattr(
            supplier_audit,
            "audit_supplier_data",
            lambda context, parameters: {
                **current_audit(context, parameters),
                "schema_version": "supplier_data_audit.raw.v1",
            },
        )
        old_result = supplier_data_audit.run(ctx, loaded)

    current_result = supplier_data_audit.run(ctx, loaded)

    assert old_result.runtime["artifact_state"] == "computed"
    assert current_result.runtime["artifact_state"] == "computed"
    assert old_result.runtime["fingerprint_sha256"] != current_result.runtime[
        "fingerprint_sha256"
    ]


def test_supplier_data_audit_is_not_applicable_for_existing_jdt_supplier(
    tmp_path: Path,
) -> None:
    from acceptance_pull.supplier_audit import audit_supplier_data
    from qc_pipeline.adapters.supplier_data_audit import adapt_supplier_data_audit

    ctx = context(tmp_path)
    jdt_context = AssetContext(
        asset_id="jdt-a",
        batch_root=ctx.batch_root,
        report_path=ctx.batch_root / "quality_archive" / "jdt-a.json",
        source_files={},
        metadata={"supplier": "jdt", "task": "existing behavior"},
    )

    raw = audit_supplier_data(
        jdt_context,
        config(tmp_path).module_parameters("supplier_data_audit"),
    )
    result = adapt_supplier_data_audit(raw, config(tmp_path))

    assert raw["decision"] == "skipped"
    assert raw["mapping_status"] == "not_applicable"
    assert raw["issues"] == []
    assert result.verdict == "skipped"
    assert result.evaluation == {
        "decision": "skipped",
        "mapping_status": "not_applicable",
        "reason": "supplier_not_applicable",
    }


def test_default_registry_writes_supplier_audit_into_same_qc_report(
    tmp_path: Path,
) -> None:
    from qc_pipeline.default_registry import build_default_registry
    from qc_pipeline.orchestrator import run_asset

    ctx = context(tmp_path)
    loaded = config(tmp_path)
    outcome = run_asset(
        ctx,
        config=loaded,
        profile="supplier_evaluation",
        registry=build_default_registry(ctx, loaded),
    )

    assert outcome.status == "completed"
    assert outcome.report["supplier_data_audit"]["flow"]["result_gate"][
        "verdict"
    ] == "warn"
    assert outcome.report["execution"]["module_states"]["supplier_data_audit"] == {
        "state": "completed"
    }
