from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
import time

import pytest

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


def test_csv_nan_optional_paths_stay_missing_in_canonical_manifest_identity(
    tmp_path: Path,
) -> None:
    from qc_common.manifest_metadata import manifest_metadata
    from tools.run_qc_pipeline import contexts_from_manifest

    source = tmp_path / "source" / "dr" / "hdf5"
    source.mkdir(parents=True)
    (source / "task-a.h5").write_bytes(b"synthetic hdf5")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "asset_id,supplier,hdf5_path,hdf5_reference_dataset,content_id,calib_path,camera_trajectory_path\n"
        "dr-task-a,dr,source/dr/hdf5/task-a.h5,timestamp,,,\n",
        encoding="utf-8",
    )

    context = contexts_from_manifest(manifest, batch_root=tmp_path)[0]
    identity = manifest_metadata(context.metadata)

    assert set(context.source_files) == {"hdf5"}
    assert context.source_files["hdf5"]["path"] == "source/dr/hdf5/task-a.h5"
    assert identity["content_id"] is None
    assert identity["calib_path"] is None
    assert identity["camera_trajectory_path"] is None
    assert all(
        "nan" not in str(source["path"]).lower()
        for source in context.source_files.values()
    )


def test_dr_heuristic_projection_mapping_is_explicit_runtime_metadata_only(
    tmp_path: Path,
) -> None:
    from qc_common.manifest_metadata import manifest_metadata
    from tools.run_qc_pipeline import apply_dr_heuristic_projection_mapping

    context = AssetContext(
        "dr__task-a",
        tmp_path,
        tmp_path / "quality_archive" / "dr__task-a.json",
        {},
        metadata={
            "supplier": "dr",
            "asset_id": "dr__task-a",
            "calibration_status": "mapping_missing",
            "manifest_row": {
                "supplier": "dr",
                "asset_id": "dr__task-a",
                "calibration_status": "mapping_missing",
            },
        },
    )
    mapping = tmp_path / "deepreach_heuristic_head_projection_mapping.csv"
    mapping.write_text(
        "asset_id,projection_mode,head_hfov_deg,fx,fy,cx,cy,image_width,image_height,"
        "calibration_status,projection_validation_status,distortion_applied,"
        "camera_trajectory_applied,selected_by\n"
        "dr__task-a,approx_pinhole_from_hfov,100,805.533,805.533,960,540,"
        "1920,1080,heuristic,pending_visual_validation,false,false,"
        "user_visual_confirmation\n",
        encoding="utf-8",
    )

    updated = apply_dr_heuristic_projection_mapping([context], mapping)[0]

    assert updated.metadata["projection_mode"] == "approx_pinhole_from_hfov"
    assert updated.metadata["head_hfov_deg"] == 100.0
    assert updated.metadata["calibration_status"] == "heuristic"
    assert updated.metadata["selected_by"] == "user_visual_confirmation"
    assert updated.metadata["dr_heuristic_projection_mapping_path"] == str(
        mapping.resolve()
    )
    assert manifest_metadata(updated.metadata) == manifest_metadata(context.metadata)
    assert updated.metadata["manifest_row"] == context.metadata["manifest_row"]


def test_run_qc_pipeline_cli_accepts_explicit_dr_heuristic_mapping(
    tmp_path: Path,
) -> None:
    from tools.run_qc_pipeline import build_parser

    mapping = tmp_path / "mapping.csv"
    args = build_parser().parse_args(
        [
            "--batch-root",
            str(tmp_path),
            "--manifest",
            str(tmp_path / "manifest.csv"),
            "--profile",
            "supplier_evaluation",
            "--dr-heuristic-projection-map",
            str(mapping),
        ]
    )

    assert args.dr_heuristic_projection_map == mapping


def test_manifest_identity_uses_logical_symlink_path_while_access_path_resolves(
    tmp_path: Path,
) -> None:
    from qc_common.manifest_metadata import manifest_metadata
    from qc_common.report_mutation import (
        apply_module_result,
        validate_report_identity,
    )
    from tools.run_qc_pipeline import contexts_from_manifest

    batch_root = tmp_path / "batch"
    logical_parent = batch_root / "source" / "dr"
    logical_parent.mkdir(parents=True)
    first_target = tmp_path / "mnt-a" / "hdf5"
    second_target = tmp_path / "mnt-b" / "hdf5"
    for target, payload in ((first_target, b"first"), (second_target, b"second")):
        target.mkdir(parents=True)
        (target / "task-a.h5").write_bytes(payload)
    link = logical_parent / "hdf5"
    link.symlink_to(first_target, target_is_directory=True)
    manifest = batch_root / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "asset_id": "dr-task-a",
                "supplier": "dr",
                "hdf5_path": str(link / "task-a.h5"),
                "hdf5_reference_dataset": "timestamp",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    first = contexts_from_manifest(
        manifest,
        batch_root=batch_root,
        allow_symlinked_sources=True,
    )[0]
    config = _config(batch_root)
    report = apply_module_result(
        first.report_path,
        context=first,
        config=config,
        profile="acceptance",
        result=ModuleResult("video_quality", "pass", {}, {}),
        expected_revision=0,
        next_module=None,
        now="2026-07-18T00:00:00Z",
    )

    link.unlink()
    link.symlink_to(second_target, target_is_directory=True)
    second = contexts_from_manifest(
        manifest,
        batch_root=batch_root,
        allow_symlinked_sources=True,
    )[0]

    assert first.source_files == second.source_files == {
        "hdf5": {"path": "source/dr/hdf5/task-a.h5"}
    }
    assert first.metadata["hdf5_path"] == str((first_target / "task-a.h5").resolve())
    assert second.metadata["hdf5_path"] == str((second_target / "task-a.h5").resolve())
    assert manifest_metadata(first.metadata) == manifest_metadata(second.metadata)
    assert manifest_metadata(first.metadata)["hdf5_path"] == (
        "source/dr/hdf5/task-a.h5"
    )
    validate_report_identity(
        report,
        context=second,
        config=config,
        profile="acceptance",
    )


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


@pytest.mark.parametrize(
    ("max_workers", "execution"),
    [(1, "sequential"), (3, "concurrent")],
)
def test_run_batch_shares_one_sam3_runtime_provider_across_three_jdt_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    max_workers: int,
    execution: str,
) -> None:
    from qc_pipeline import sam3_runtime
    from tools import run_qc_pipeline as cli

    model = tmp_path / "sam3-model"
    model.mkdir()
    raw_segmenter = object()
    factory_calls = 0
    observed_segmenters: list[object] = []
    state_lock = Lock()

    def segmenter_factory(path: Path, config: dict[str, object]) -> object:
        nonlocal factory_calls
        with state_lock:
            factory_calls += 1
        time.sleep(0.03)
        return raw_segmenter

    monkeypatch.setattr(sam3_runtime, "_default_factory", segmenter_factory)

    def fake_build_default_registry(
        context: AssetContext,
        loaded_config: LoadedQcConfig,
        *,
        segmenter_factory: object | None = None,
        segmenter_provider: object | None = None,
    ) -> ModuleRegistry:
        assert segmenter_factory is None
        assert callable(segmenter_provider)
        registry = ModuleRegistry()

        def run(
            runner_context: AssetContext,
            loaded: LoadedQcConfig,
        ) -> ModuleResult:
            segmenter = segmenter_provider(  # type: ignore[operator]
                model,
                {
                    "device": "cuda",
                    "dtype": "bfloat16",
                    "mask_threshold": 0.5,
                },
            )
            with state_lock:
                observed_segmenters.append(segmenter)
            return ModuleResult(
                "video_quality",
                "pass",
                {"decision": "pass"},
                {},
                runtime={"artifact_state": "computed"},
            )

        registry.register("test.video_quality", run)
        return registry

    monkeypatch.setattr(cli, "build_default_registry", fake_build_default_registry)
    contexts = [
        AssetContext(
            f"jdt-{index}",
            tmp_path,
            tmp_path / "quality_archive" / f"jdt-{index}.json",
            {},
            metadata={"supplier": "jdt"},
        )
        for index in range(3)
    ]

    outcomes = cli.run_batch(
        contexts,
        config=_config(tmp_path),
        profile="acceptance",
        max_workers=max_workers,
    )

    assert set(outcomes) == {"jdt-0", "jdt-1", "jdt-2"}, execution
    assert factory_calls == 1, execution
    assert len(observed_segmenters) == 3
    assert observed_segmenters[0] is observed_segmenters[1] is observed_segmenters[2]


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
