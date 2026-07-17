from __future__ import annotations

from pathlib import Path

import pytest

from qc_common.config import LoadedQcConfig, load_qc_acceptance_config
from qc_common.contracts import ModuleResult
from qc_common.module_registry import ModuleRegistry
from qc_pipeline.context import AssetContext
from qc_pipeline.orchestrator import run_asset
from qc_pipeline.runners import precheck
from tests.fixtures import solid_frame, write_test_video


def pipeline_config(tmp_path: Path) -> LoadedQcConfig:
    modules = [*precheck.MODULES, "video_quality", "supplier_data_audit"]
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
                "modules": modules,
            },
            "modules": {
                **{
                    name: {
                        "enabled": True,
                        "implementation": f"precheck.{name}",
                        "parameters": {},
                        "rules": {},
                    }
                    for name in precheck.MODULES
                },
                "supplier_data_audit": {
                    "enabled": True,
                    "implementation": "test.supplier_data_audit",
                    "parameters": {},
                    "rules": {},
                },
                "video_quality": {
                    "enabled": True,
                    "implementation": "test.video_quality",
                    "parameters": {},
                    "rules": {},
                },
            },
        },
        sha256="sha256:" + "9" * 64,
    )


def test_canonical_dr_uses_existing_deepreach_hdf5_loader(
    tmp_path: Path,
    monkeypatch,
) -> None:
    hdf5 = tmp_path / "clip.h5"
    hdf5.write_bytes(b"hdf5")
    context = AssetContext(
        "task-a",
        tmp_path,
        tmp_path / "quality_archive" / "task-a.json",
        {"hdf5": {"path": "clip.h5"}},
        source_range=(10, 13),
        metadata={
            "supplier": "dr",
            "task": "pick",
            "hdf5_reference_dataset": "timestamp",
        },
    )
    observed: list[dict[str, object]] = []

    def load(row, episode_idx):
        observed.append(row)
        return object()

    monkeypatch.setattr("tools.run_manifest_precheck.load_deepreach_clip", load)
    loaded = precheck._load_clip(context, "hdf5_text_info")

    assert loaded is not None
    assert observed[0]["start_frame"] == 10
    assert observed[0]["end_frame"] == 12
    assert observed[0]["hdf5_path"] == str(hdf5)
    assert observed[0]["hdf5_reference_dataset"] == "timestamp"


def test_potentia_missing_keypoints_do_not_stop_independent_modules(
    tmp_path: Path,
) -> None:
    context = AssetContext(
        "potentia__task-a",
        tmp_path,
        tmp_path / "quality_archive" / "potentia__task-a.json",
        {},
        metadata={"supplier": "potentia"},
    )
    config = pipeline_config(tmp_path)
    session = precheck.PrecheckSession(context, config)
    registry = ModuleRegistry()
    for module in precheck.MODULES:
        registry.register(f"precheck.{module}", session.runner_for(module))
    executed: list[str] = []
    for module in ("video_quality", "supplier_data_audit"):
        registry.register(
            f"test.{module}",
            lambda context, config, module=module: (
                executed.append(module)
                or ModuleResult(module, "pass", {"decision": "pass"}, {})
            ),
        )

    outcome = run_asset(
        context,
        config=config,
        profile="supplier_evaluation",
        registry=registry,
    )

    states = outcome.report["execution"]["module_states"]
    assert {states[module]["state"] for module in precheck.MODULES} == {
        "input_missing"
    }
    assert executed == ["video_quality", "supplier_data_audit"]
    assert outcome.status == "incomplete"


def test_dr_inconsistent_hdf5_marks_precheck_input_invalid_and_continues(
    tmp_path: Path,
) -> None:
    import h5py
    import numpy as np

    from tests.test_manifest_precheck_runner import _write_deepreach_hdf5

    hdf5 = _write_deepreach_hdf5(tmp_path / "clip.h5", frame_count=4)
    with h5py.File(hdf5, "a") as handle:
        del handle["hand/right/valid"]
        handle.create_dataset("hand/right/valid", data=np.ones(3, dtype=np.uint8))
    context = AssetContext(
        "task-a",
        tmp_path,
        tmp_path / "quality_archive" / "task-a.json",
        {"hdf5": {"path": "clip.h5"}},
        source_range=(0, 4),
        metadata={
            "supplier": "dr",
            "hdf5_reference_dataset": "timestamp",
        },
    )
    config = pipeline_config(tmp_path)
    session = precheck.PrecheckSession(context, config)
    registry = ModuleRegistry()
    for module in precheck.MODULES:
        registry.register(f"precheck.{module}", session.runner_for(module))
    executed: list[str] = []
    for module in ("video_quality", "supplier_data_audit"):
        registry.register(
            f"test.{module}",
            lambda context, config, module=module: (
                executed.append(module)
                or ModuleResult(module, "pass", {"decision": "pass"}, {})
            ),
        )

    outcome = run_asset(
        context,
        config=config,
        profile="supplier_evaluation",
        registry=registry,
    )

    states = outcome.report["execution"]["module_states"]
    assert {states[module]["state"] for module in precheck.MODULES} == {
        "input_invalid"
    }
    assert all(
        "inconsistent_frame_count" in error["message"]
        for error in outcome.report["runtime_errors"]
        if error["module"] in precheck.MODULES
    )
    assert executed == ["video_quality", "supplier_data_audit"]


def test_potentia_video_quality_disables_only_hdf5_alignment(tmp_path: Path) -> None:
    from qc_pipeline.runners.video_quality import _compute_video_quality

    video = tmp_path / "video.mp4"
    write_test_video(video, [solid_frame(90, width=1280, height=720) for _ in range(4)], fps=25.0)
    config = load_qc_acceptance_config()

    def context(supplier: str, asset_id: str) -> AssetContext:
        return AssetContext(
            asset_id,
            tmp_path,
            tmp_path / "quality_archive" / f"{asset_id}.json",
            {"video": {"path": "video.mp4"}},
            metadata={"supplier": supplier},
        )

    _, potentia_raw = _compute_video_quality(
        context("potentia", "potentia__task-a"),
        config,
    )
    _, dr_raw = _compute_video_quality(context("dr", "task-a"), config)

    assert potentia_raw["alignment"]["status"] == "disabled"
    assert "hdf5_missing" not in potentia_raw["evaluation"]["reasons"]
    assert dr_raw["alignment"]["status"] == "missing"
    assert "hdf5_missing" in dr_raw["evaluation"]["reasons"]


@pytest.mark.parametrize(
    ("supplier", "reason"),
    (("dr", "calibration_unverified"), ("potentia", "no_keypoint_input")),
)
def test_supplier_evaluation_report_preserves_explicit_sam3_block_reason(
    tmp_path: Path,
    supplier: str,
    reason: str,
) -> None:
    from qc_pipeline.runners.sam3_containment import runner

    config = LoadedQcConfig(
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
                "modules": ["sam3_containment"],
            },
            "modules": {
                "sam3_containment": {
                    "enabled": True,
                    "implementation": "sam3_containment.unified",
                    "parameters": {},
                    "rules": {},
                }
            },
        },
        sha256="sha256:" + "8" * 64,
    )
    context = AssetContext(
        f"{supplier}-asset",
        tmp_path,
        tmp_path / "quality_archive" / f"{supplier}-asset.json",
        {},
        metadata={"supplier": supplier},
    )
    registry = ModuleRegistry()
    registry.register("sam3_containment.unified", runner(None))

    outcome = run_asset(
        context,
        config=config,
        profile="supplier_evaluation",
        registry=registry,
    )

    state = outcome.report["execution"]["module_states"]["sam3_containment"]
    assert state["state"] == "blocked"
    assert state["reason"] == reason
    assert reason in outcome.report["runtime_errors"][0]["message"]
    assert outcome.report["overall_decision"] is None
