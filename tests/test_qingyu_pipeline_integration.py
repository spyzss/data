from __future__ import annotations

from pathlib import Path
import copy
import json

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
        "qingyu-hand-pose-precheck-v4"
    )
    assert fingerprint["source_contract"]["observation_canonicalization"] == (
        "stable-first-structurally-valid-v1"
    )
    assert fingerprint["source_contract"]["supplier_quality_policy"] == (
        "ignored-for-acceptance-v1"
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


def test_qy_invalid_supplier_timebase_is_preserved_when_derived_contract_is_valid(
    tmp_path: Path,
) -> None:
    from acceptance_pull.supplier_audit import audit_supplier_data

    root = tmp_path / "source" / "QY"
    episode = make_qy_episode(root)
    timebase_path = episode / "timestamps" / "episode_timebase.json"
    payload = json.loads(timebase_path.read_text(encoding="utf-8"))
    payload["episode_id"] = "mismatched_supplier_episode"
    timebase_path.write_text(json.dumps(payload), encoding="utf-8")
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

    assert raw["mapping_status"] == "verified"
    assert raw["structured"]["timebase_status"] == "derived"
    assert raw["structured"]["timebase_source"] == (
        "derived_from_video_and_observations_2d"
    )
    issue = next(
        issue
        for issue in raw["issues"]
        if issue["code"] == "supplier_timebase_invalid_derived"
    )
    assert issue["severity"] == "warn"
    assert "does not match" in issue["observed_value"]


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


def test_qy_multiple_raw_candidates_do_not_block_supported_precheck(
    tmp_path: Path,
) -> None:
    from tests.qingyu_fixtures import keypoints_2d, observation_rows

    root = tmp_path / "source" / "QY"
    rows = observation_rows()
    rows.append(
        {
            **rows[0],
            "pred_keypoints_2d": keypoints_2d(10.0),
        }
    )
    make_qy_episode(root, observations=rows)
    manifest = write_qingyu_manifest(
        build_qingyu_manifest(root, primary_camera="mid_cam_left"), tmp_path
    )
    context = contexts_from_manifest(manifest, batch_root=tmp_path)[0]
    session = PrecheckSession(context, load_qc_acceptance_config())

    assert context.metadata["adapter_status"] == "ready"
    assert context.metadata["primary_camera"] == "mid_cam_left"
    assert session.run_module("hdf5_text_info").module == "hdf5_text_info"

    for module in ("keypoint_morphology", "keypoint_temporal"):
        result = session.run_module(module)
        assert result.verdict == "skipped"
        assert result.evaluation["decision"] == "not_applicable"
        assert result.evaluation["reason"] == "qy_hand_topology_not_validated"


def test_qy_quality_hand_is_policy_skipped_even_when_supplier_quality_exists(
    tmp_path: Path,
) -> None:
    context = _context_for_qy(tmp_path)
    session = PrecheckSession(context, load_qc_acceptance_config())

    result = session.run_module("quality_hand")

    assert result.verdict == "skipped"
    assert result.evaluation == {
        "decision": "not_applicable",
        "output_status": "not_applicable",
        "reason": "supplier_quality_signal_ignored_by_policy",
    }
    assert result.issues == ()


def test_qy_missing_supplier_quality_does_not_change_quality_module_semantics(
    tmp_path: Path,
) -> None:
    from acceptance_pull.supplier_audit import audit_supplier_data

    root = tmp_path / "source" / "QY"
    episode = make_qy_episode(root)
    (episode / "hand_pose" / "quality.json").unlink()
    manifest = write_qingyu_manifest(
        build_qingyu_manifest(root, primary_camera="mid_cam_left"), tmp_path
    )
    context = contexts_from_manifest(manifest, batch_root=tmp_path)[0]

    result = PrecheckSession(
        context, load_qc_acceptance_config()
    ).run_module("quality_hand")
    supplier_audit = audit_supplier_data(
        context,
        load_qc_acceptance_config().module_parameters("supplier_data_audit"),
    )

    assert context.metadata["adapter_status"] == "ready"
    assert context.source_files["quality"]["path"].endswith(
        "hand_pose/quality.json"
    )
    assert not (
        context.batch_root / context.source_files["quality"]["path"]
    ).exists()
    assert result.verdict == "skipped"
    assert result.evaluation["reason"] == (
        "supplier_quality_signal_ignored_by_policy"
    )
    assert supplier_audit["inventory"]["quality"]["status"] == "missing"
    assert "quality" not in supplier_audit["missing_sources"]
    assert not any(
        issue.get("source_name") == "quality"
        for issue in supplier_audit["issues"]
    )


def test_qy_supplier_audit_fingerprint_records_ignored_quality_policy(
    tmp_path: Path,
) -> None:
    from qc_pipeline.artifacts import artifact_for
    from qc_pipeline.runners.supplier_data_audit import run

    context = _context_for_qy(tmp_path)

    run(context, load_qc_acceptance_config())

    run_config = json.loads(
        (
            artifact_for(context, "supplier_data_audit").directory
            / "run_config.json"
        ).read_text(encoding="utf-8")
    )
    assert run_config["fingerprint"]["supplier_contract"] == {
        "quality_source_policy": "inventory_only_ignored_for_acceptance_v1"
    }


def test_qy_missing_primary_fps_or_video_blocks_precheck_precisely(
    tmp_path: Path,
) -> None:
    context = _context_for_qy(tmp_path)
    cases = (
        (
            {**dict(context.metadata), "primary_camera": ""},
            dict(context.source_files),
            "QY primary_camera",
        ),
        (
            {**dict(context.metadata), "fps": None},
            dict(context.source_files),
            "QY finite positive fps",
        ),
        (
            dict(context.metadata),
            {
                name: value
                for name, value in context.source_files.items()
                if name != "video"
            },
            "source_files.video.path",
        ),
    )

    for metadata, source_files, expected in cases:
        invalid_context = type(context)(
            asset_id=context.asset_id,
            batch_root=context.batch_root,
            report_path=context.report_path,
            source_files=source_files,
            source_range=context.source_range,
            metadata=metadata,
        )
        with pytest.raises(ModulePrerequisiteError, match=expected):
            PrecheckSession(
                invalid_context,
                load_qc_acceptance_config(),
            ).run_module("hdf5_text_info")


def test_qy_valid_explicit_camera_reaches_video_quality(
    tmp_path: Path,
) -> None:
    from qc_pipeline.runners.video_quality import run as run_video_quality
    from tests.qingyu_fixtures import observation_rows

    root = tmp_path / "source" / "QY"
    rows = observation_rows()
    for row in rows:
        row["source_frame_index"] = {100: 0, 101: 1, 102: 2}[
            row["source_frame_index"]
        ]
    episode = make_qy_episode(
        root,
        observations=rows,
        trajectory=trajectory_rows(source_steps=(0, 1, 2)),
    )
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

    result = run_video_quality(context, load_qc_acceptance_config())

    assert context.metadata["adapter_status"] == "ready"
    assert context.source_files["video"]["path"].endswith("mid_cam_left.mp4")
    assert result.module == "video_quality"


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
