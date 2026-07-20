from __future__ import annotations

from pathlib import Path
import copy

import pandas as pd
import pytest

from acceptance_pull.build_supplier_manifest import parse_args, run
from acceptance_pull.supplier_adapters.qingyu import (
    build_qingyu_manifest,
    write_qingyu_manifest,
)
from qc_common.config import load_qc_acceptance_config
from qc_common.schema import validate_qc_config
from qc_common.suppliers import normalize_supplier
from qc_common.module_registry import ModulePrerequisiteError
from qc_pipeline.runners.precheck import PrecheckSession, precheck_fingerprint
from tests.fixtures import solid_frame, write_test_video
from tests.qingyu_fixtures import make_qy_episode, trajectory_rows
from tools.run_qc_pipeline import contexts_from_manifest


def _context_for_qy(tmp_path: Path, *, sparse: bool = False):
    root = tmp_path / "source" / "QY"
    trajectory = trajectory_rows(source_steps=(100, 102)) if sparse else None
    make_qy_episode(root, trajectory=trajectory)
    rows = build_qingyu_manifest(root, primary_camera="mid_cam_left")
    manifest = write_qingyu_manifest(rows, tmp_path)
    return contexts_from_manifest(manifest, batch_root=tmp_path)[0]


def test_qy_supplier_alias_normalizes_to_canonical_identity() -> None:
    assert normalize_supplier("qy") == ("qy", "QY", None)
    assert normalize_supplier("qingyu") == ("qy", "QY", "qingyu")


def test_qy_camera_selection_config_is_supported_by_unified_schema() -> None:
    from acceptance_pull.supplier_adapters.qingyu import (
        DEFAULT_CAMERA_SELECTION_CONFIG,
    )

    raw = copy.deepcopy(load_qc_acceptance_config().raw)
    raw["supplier_adapters"] = {
        "qy": {"camera_selection": DEFAULT_CAMERA_SELECTION_CONFIG}
    }

    validate_qc_config(raw)


def test_supplier_manifest_cli_accepts_qy_without_dr_only_arguments(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source" / "QY"
    make_qy_episode(root)
    sam3_model = tmp_path / "models" / "sam3"
    sam3_model.mkdir(parents=True)
    args = parse_args(
        [
            "--supplier",
            "qy",
            "--root",
            str(root),
            "--output-dir",
            str(tmp_path / "output"),
            "--primary-camera",
            "mid_cam_left",
            "--sam3-model",
            str(sam3_model),
        ]
    )

    assert args.supplier == "qy"
    assert args.sam3_model == sam3_model
    assert run(
        supplier="qy",
        root=root,
        output_dir=tmp_path / "output",
        primary_camera="mid_cam_left",
        sam3_model=sam3_model,
    ) == 0
    manifest = tmp_path / "output" / "manifests" / "supplier_manifest_qy.csv"
    assert manifest.is_file()
    row = pd.read_csv(manifest).iloc[0]
    assert row["sam3_model_path"] == str(sam3_model)


def test_qy_camera_selection_rejects_unknown_config_field() -> None:
    from acceptance_pull.supplier_adapters.qingyu import camera_selection_config

    with pytest.raises(ValueError, match="unknown"):
        camera_selection_config({"minimum_hand_coverge": 0.5})


def test_qy_context_separates_direct_2d_trajectory_and_timebase_sources(
    tmp_path: Path,
) -> None:
    context = _context_for_qy(tmp_path)

    assert context.metadata["supplier"] == "qy"
    assert context.source_range == (100, 103)
    assert set(context.source_files) >= {
        "video",
        "observations_2d",
        "trajectory_3d",
        "coordinate_system",
        "quality",
        "timebase",
        "semantic",
        "episode_manifest",
        "review_video",
    }
    assert context.source_files["observations_2d"]["path"].endswith(
        "hand_pose/observations_2d.parquet"
    )
    assert context.source_files["trajectory_3d"]["path"].endswith(
        "hand_pose/trajectory_3d.parquet"
    )


def test_qy_precheck_fingerprint_has_isolated_adapter_contract(
    tmp_path: Path,
) -> None:
    context = _context_for_qy(tmp_path)
    config = load_qc_acceptance_config()

    fingerprint = precheck_fingerprint(context, config)

    assert fingerprint["source_contract"]["supplier_adapter"] == (
        "qingyu-hand-pose-precheck-v3"
    )
    assert set(fingerprint["sources"]) >= {
        "observations_2d",
        "trajectory_3d",
        "coordinate_system",
        "timebase",
    }


def test_qy_supplier_data_audit_records_inventory_without_second_quality_pass(
    tmp_path: Path,
) -> None:
    from acceptance_pull.supplier_audit import audit_supplier_data

    context = _context_for_qy(tmp_path)
    config = load_qc_acceptance_config()

    raw = audit_supplier_data(
        context,
        config.module_parameters("supplier_data_audit"),
    )

    assert raw["supplier"] == "qy"
    assert raw["decision"] == "warn"
    assert raw["mapping_status"] == "verified"
    assert set(raw["inventory"]) >= {
        "video",
        "observations_2d",
        "trajectory_3d",
        "coordinate_system",
        "timebase",
    }
    assert raw["structured"]["skeleton_3d_status"] == "valid"
    assert raw["supplier_quality_signal"]["status"] == "auxiliary_only"
    assert raw["supplier_quality_signal"]["value"] is None
    assert {
        issue["code"] for issue in raw["issues"]
    } >= {"joint_topology_unverified", "coordinate_system_schema_unverified"}


def test_qy_derived_timebase_is_audited_as_missing_supplier_file_not_hard_failure(
    tmp_path: Path,
) -> None:
    from acceptance_pull.supplier_audit import audit_supplier_data

    root = tmp_path / "source" / "QY"
    episode = make_qy_episode(root)
    (episode / "timestamps" / "episode_timebase.json").unlink()
    write_test_video(
        episode / "videos" / "mid_cam_left.mp4",
        [solid_frame(value, width=64, height=48) for value in (10, 20, 30, 40)],
        fps=30.0,
    )
    manifest = write_qingyu_manifest(
        build_qingyu_manifest(root, primary_camera="mid_cam_left"), tmp_path
    )
    context = contexts_from_manifest(manifest, batch_root=tmp_path)[0]

    raw = audit_supplier_data(
        context,
        load_qc_acceptance_config().module_parameters("supplier_data_audit"),
    )

    assert raw["inventory"]["timebase"]["status"] == "missing"
    assert "timebase" not in raw["missing_sources"]
    assert raw["structured"]["timebase_source"] == (
        "derived_from_video_and_observations_2d"
    )
    assert any(
        issue["code"] == "supplier_timebase_missing_derived"
        and issue["severity"] == "warn"
        for issue in raw["issues"]
    )
    assert raw["decision"] == "warn"


def test_qy_supplier_data_audit_rejects_incomplete_required_skeleton(
    tmp_path: Path,
) -> None:
    from acceptance_pull.supplier_audit import audit_supplier_data

    context = _context_for_qy(tmp_path, sparse=True)
    config = load_qc_acceptance_config()

    raw = audit_supplier_data(
        context,
        config.module_parameters("supplier_data_audit"),
    )

    assert raw["decision"] == "fail"
    assert any(
        issue["code"] == "required_skeleton_input_missing"
        for issue in raw["issues"]
    )


def test_qy_precheck_session_reads_trajectory_once_across_five_modules(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _context_for_qy(tmp_path)
    config = load_qc_acceptance_config()
    calls: list[Path] = []
    original = pd.read_parquet

    def counted(path, *args, **kwargs):
        calls.append(Path(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", counted)
    session = PrecheckSession(context, config)

    results = [session.run_module(module) for module in session_modules()]

    trajectory = Path(str(context.metadata["trajectory_3d_path"]))
    assert calls.count(trajectory) == 1
    assert [result.module for result in results] == list(session_modules())
    assert results[0].module == "hdf5_text_info"
    assert results[0].verdict == "pass"


def test_qy_unconfirmed_topology_is_not_applicable_to_morphology_or_temporal(
    tmp_path: Path,
) -> None:
    context = _context_for_qy(tmp_path)
    session = PrecheckSession(context, load_qc_acceptance_config())

    presence = session.run_module("keypoint_presence")
    morphology = session.run_module("keypoint_morphology")
    temporal = session.run_module("keypoint_temporal")

    assert presence.verdict == "pass"
    for result in (morphology, temporal):
        assert result.verdict == "skipped"
        assert result.evaluation["decision"] == "not_applicable"
        assert result.evaluation["output_status"] == "not_applicable"
        assert result.evaluation["reason"] == "qy_hand_topology_not_validated"
        assert result.frame_exclusions == ()
    assert temporal.evaluation["valid_frame_count"] == 0
    assert temporal.evaluation["candidate_window_count"] == 1
    assert all(
        result.metrics.get("temporal_output_valid") is False
        for result in session._raw_results["keypoint_temporal"]
    )


def test_qy_supplier_inventory_rejects_directory_at_required_file_path(
    tmp_path: Path,
) -> None:
    from acceptance_pull.supplier_audit import audit_supplier_data

    context = _context_for_qy(tmp_path)
    observations = Path(str(context.metadata["observations_2d_path"]))
    observations.unlink()
    observations.mkdir()

    raw = audit_supplier_data(
        context,
        load_qc_acceptance_config().module_parameters("supplier_data_audit"),
    )

    assert raw["inventory"]["observations_2d"]["status"] == "wrong_type"
    assert raw["decision"] == "fail"
    assert any(
        issue["code"] == "required_source_wrong_type"
        and issue["source_name"] == "observations_2d"
        for issue in raw["issues"]
    )


def test_qy_missing_precheck_source_reports_prerequisite_not_cache_error(
    tmp_path: Path,
) -> None:
    context = _context_for_qy(tmp_path)
    trajectory = Path(str(context.metadata["trajectory_3d_path"]))
    trajectory.unlink()

    with pytest.raises(ModulePrerequisiteError, match="trajectory_3d"):
        PrecheckSession(context, load_qc_acceptance_config()).run_module(
            "keypoint_presence"
        )


def test_qy_sparse_required_skeleton_is_hard_presence_fail_without_interpolation(
    tmp_path: Path,
) -> None:
    context = _context_for_qy(tmp_path, sparse=True)
    config = load_qc_acceptance_config()
    # supplier_evaluation keeps later modules independent.
    context = type(context)(
        asset_id=context.asset_id,
        batch_root=context.batch_root,
        report_path=context.report_path,
        source_files=context.source_files,
        source_range=context.source_range,
        metadata={**dict(context.metadata), "profile": "supplier_evaluation"},
    )
    session = PrecheckSession(context, config)

    presence = session.run_module("keypoint_presence")
    morphology = session.run_module("keypoint_morphology")
    temporal = session.run_module("keypoint_temporal")

    assert presence.verdict == "fail"
    assert any(
        exclusion.start_frame == 101 and exclusion.end_frame == 101
        for exclusion in presence.frame_exclusions
    )
    assert morphology.module == "keypoint_morphology"
    assert temporal.module == "keypoint_temporal"


def session_modules() -> tuple[str, ...]:
    from qc_pipeline.runners.precheck import MODULES

    return MODULES
