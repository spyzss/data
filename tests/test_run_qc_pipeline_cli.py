from __future__ import annotations

import json
from pathlib import Path

from qc_common.config import LoadedQcConfig
from qc_common.contracts import ModuleResult
from qc_common.module_registry import ModuleRegistry
from qc_pipeline.context import AssetContext


def _config(tmp_path: Path) -> LoadedQcConfig:
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
            "pipeline": {"default_profile": "acceptance", "modules": ["video_quality"]},
            "modules": {
                "video_quality": {
                    "enabled": True,
                    "implementation": "test.video_quality",
                    "parameters": {},
                    "rules": {},
                }
            },
        },
        sha256="sha256:" + "3" * 64,
    )


def test_manifest_legacy_candidate_path_is_metadata_not_required_source(
    tmp_path: Path,
) -> None:
    from tools.run_qc_pipeline import contexts_from_manifest

    legacy = tmp_path / "legacy.json"
    legacy.write_text("[]", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "asset_id": "asset-a",
                "candidate_windows_path": str(legacy),
                "start_frame": 0,
                "end_frame": 9,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    context = contexts_from_manifest(manifest, batch_root=tmp_path)[0]

    assert "candidate_windows" not in context.source_files
    assert context.metadata["candidate_windows_path"] == str(legacy)


def test_manifest_maps_dr_and_potentia_sidecar_paths_into_source_contract(
    tmp_path: Path,
) -> None:
    from tools.run_qc_pipeline import contexts_from_manifest

    source = tmp_path / "source"
    source.mkdir()
    files = {
        "primary_video_path": "head.mp4",
        "head_video_path": "head.mp4",
        "left_wrist_video_path": "left.mp4",
        "right_wrist_video_path": "right.mp4",
        "calib_path": "calib.json",
        "camera_trajectory_path": "trajectory.csv",
        "meta_path": "meta.json",
        "frames_path": "frames.csv",
        "aligned_path": "aligned.csv",
        "imu_path": "imu.csv",
    }
    for filename in set(files.values()):
        (source / filename).write_text("{}", encoding="utf-8")
    task_dir = source / "task"
    task_dir.mkdir()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "asset_id": "asset-a",
                "supplier": "potentia",
                **{key: f"source/{value}" for key, value in files.items()},
                "task_dir": "source/task",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    context = contexts_from_manifest(manifest, batch_root=tmp_path)[0]

    assert set(context.source_files) == {
        "video",
        "head_video",
        "left_wrist_video",
        "right_wrist_video",
        "calibration",
        "trajectory",
        "meta",
        "frames",
        "aligned",
        "imu",
        "task_dir",
    }
    assert context.source_files["trajectory"]["path"] == "source/trajectory.csv"
    assert context.source_files["task_dir"]["path"] == "source/task"


def test_run_batch_propagates_artifact_reuse_and_profile_to_runtime_context(
    tmp_path: Path,
) -> None:
    from tools.run_qc_pipeline import run_batch

    config = _config(tmp_path)
    context = AssetContext(
        "asset-a",
        tmp_path,
        tmp_path / "quality_archive" / "asset-a.json",
        {},
    )
    observed: list[tuple[bool, str]] = []

    def factory(runtime_context: AssetContext) -> ModuleRegistry:
        registry = ModuleRegistry()

        def run(
            runner_context: AssetContext,
            loaded: LoadedQcConfig,
        ) -> ModuleResult:
            observed.append(
                (
                    bool(runner_context.metadata["reuse_artifacts"]),
                    str(runner_context.metadata["profile"]),
                )
            )
            return ModuleResult(
                "video_quality",
                "pass",
                {"decision": "pass"},
                {},
                runtime={"artifact_state": "computed"},
            )

        registry.register("test.video_quality", run)
        return registry

    outcome = run_batch(
        [context],
        config=config,
        profile="supplier_evaluation",
        registry_factory=factory,
        resume=False,
    )["asset-a"]

    assert observed == [(False, "supplier_evaluation")]
    assert outcome.elapsed_seconds >= 0


def test_outcome_summary_groups_five_prechecks_as_one_producer(tmp_path: Path) -> None:
    from qc_pipeline.orchestrator import RunOutcome
    from tools.run_qc_pipeline import summarize_outcome

    report: dict[str, object] = {
        "report_revision": 8,
        "hdf5_text_info": {"runtime": {"artifact_state": "reused"}},
        "quality_hand": {"runtime": {"artifact_state": "reused"}},
        "keypoint_presence": {"runtime": {"artifact_state": "reused"}},
        "keypoint_morphology": {"runtime": {"artifact_state": "reused"}},
        "keypoint_temporal": {"runtime": {"artifact_state": "reused"}},
        "video_quality": {"runtime": {"artifact_state": "computed"}},
        "sam3_containment": {"runtime": {"artifact_state": "no_candidates"}},
    }
    outcome = RunOutcome(
        report=report,
        executed_modules=(
            "hdf5_text_info",
            "quality_hand",
            "keypoint_presence",
            "keypoint_morphology",
            "keypoint_temporal",
            "video_quality",
            "sam3_containment",
        ),
        status="completed",
        config=_config(tmp_path),
        elapsed_seconds=1.25,
    )

    summary = summarize_outcome(outcome)

    assert summary["producers"] == {
        "precheck": "reused",
        "video_quality": "computed",
        "sam3_containment": "skipped",
    }
    assert summary["producer_counts"] == {
        "computed": 1,
        "reused": 1,
        "skipped": 1,
    }
    assert summary["elapsed_seconds"] == 1.25
