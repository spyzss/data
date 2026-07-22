from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from precheck.checks.skeleton_quality_score import SkeletonQualityScoreCheck
from qc_common.types import CheckResult, ClipInputs
from tools.audit_temporal_timebase import audit_temporal_timebase, main


def _clip() -> ClipInputs:
    fps = 60.0
    timestamps = np.arange(12, dtype=np.float64) / fps
    x = timestamps + 0.5 * 2.0 * timestamps**2
    points = np.column_stack((x, np.ones_like(x), np.ones_like(x)))
    clip = ClipInputs(
        episode_idx=0,
        frame_indices=list(range(100, 112)),
        keypoints={"probe_joint": points},
        timestamps_ns=np.rint(timestamps * 1_000_000_000.0).astype(np.int64),
        fps=fps,
    )
    clip.asset_id = "jdt__episode_000001"
    return clip


def _extreme_clip() -> ClipInputs:
    frame_count = 8
    keypoints: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        for joint_index in range(21):
            values = np.zeros((frame_count, 3), dtype=np.float64)
            values[:, 0] = 0.01 * joint_index
            values[:, 1] = 0.02
            values[:, 2] = 0.5
            keypoints[f"{side}joint_{joint_index}"] = values
    keypoints["leftjoint_7"][3, 0] = 7_360.0
    clip = ClipInputs(
        episode_idx=0,
        frame_indices=list(range(100, 108)),
        keypoints=keypoints,
        fps=60.0,
    )
    clip.asset_id = "jdt__episode_extreme"
    clip.topology_agnostic_joint_names = list(keypoints)
    return clip


def _write_jdt_parquet(path: Path, frame_count: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for frame_index in range(frame_count):
        points_3d = np.asarray(
            [
                [0.01 * joint + 0.001 * frame_index, 0.02, 0.5]
                for joint in range(21)
            ],
            dtype=np.float32,
        )
        points_2d = np.asarray(
            [[10.0 + joint, 20.0 + joint] for joint in range(21)],
            dtype=np.float32,
        )
        rows.append(
            {
                "left_kp3d": points_3d.reshape(-1).tolist(),
                "right_kp3d": (points_3d + 0.01).reshape(-1).tolist(),
                "leftcam_left_kp2d": points_2d.reshape(-1).tolist(),
                "leftcam_right_kp2d": (points_2d + 1.0).reshape(-1).tolist(),
                "language_instruction": "move object",
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_manifest(path: Path, source_path: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "asset_id": "jdt__episode_000001",
                "start_frame": 0,
                "end_frame": 3,
                "fps": 30.0,
                "parquet_path": source_path,
            }
        ]
    ).to_csv(path, index=False)


def test_temporal_ab_audit_writes_per_asset_and_overall_outputs(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "asset_id": "jdt__episode_000001",
                "start_frame": 100,
                "end_frame": 111,
                "parquet_path": "unused.parquet",
            }
        ]
    ).to_csv(manifest, index=False)
    output_dir = tmp_path / "audit"

    summary = audit_temporal_timebase(
        manifest=manifest,
        supplier="jdt",
        output_dir=output_dir,
        asset_ids=None,
        max_clips=1,
        config_path=None,
        clip_loader=lambda _row, _index: _clip(),
        temporal_config_overrides={
            "joint_names": ["probe_joint"],
            "project_2d": False,
        },
    )

    assert summary["completed_asset_count"] == 1
    rows = json.loads(
        (output_dir / "temporal_timebase_ab_audit.json").read_text()
    )
    assert [row["asset_id"] for row in rows] == [
        "jdt__episode_000001",
        "__overall__",
    ]
    asset = rows[0]
    assert asset["source_fps"] == 60.0
    assert asset["temporal_target_hz"] == 30.0
    assert asset["sampling_method"] == "nearest_monotonic_no_reuse"
    assert asset["standardized_sample_count"] == 6
    assert json.loads(asset["source_frame_mapping_json"]) == [
        100,
        102,
        104,
        106,
        108,
        110,
    ]
    assert "native_acceleration_p95" in asset
    assert "standardized_acceleration_p95" in asset
    assert "native_candidate_seed_rate" in asset
    assert "standardized_candidate_seed_rate" in asset
    assert (output_dir / "temporal_timebase_ab_audit.csv").is_file()
    assert (output_dir / "temporal_timebase_ab_audit.parquet").is_file()
    run_config = json.loads((output_dir / "run_config.json").read_text())
    assert run_config["models_loaded"] == []
    assert run_config["mutates_precheck_outputs"] is False
    assert run_config["decision_metric_source"] == "standardized_30hz"


def test_cli_resolves_jdt_relative_path_from_manifest_parent_when_cwd_differs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "batch" / "manifest.csv"
    relative_source = Path("source/jd/data/episode.parquet")
    _write_jdt_parquet(manifest.parent / relative_source)
    _write_manifest(manifest, relative_source.as_posix())
    other_cwd = tmp_path / "unrelated-cwd"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    output_dir = tmp_path / "audit-relative"

    exit_code = main(
        [
            "--manifest",
            str(manifest),
            "--supplier",
            "jdt",
            "--max-clips",
            "1",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert Path.cwd() == other_cwd
    run_config = json.loads((output_dir / "run_config.json").read_text())
    assert run_config["failures"] == []


def test_failure_records_raw_and_attempted_manifest_relative_path(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "batch" / "manifest.csv"
    raw_path = "source/jd/data/missing.parquet"
    _write_manifest(manifest, raw_path)
    output_dir = tmp_path / "audit-missing-relative"

    summary = audit_temporal_timebase(
        manifest=manifest,
        supplier="jdt",
        output_dir=output_dir,
        asset_ids=None,
        max_clips=1,
        config_path=None,
    )

    assert summary["failed_asset_count"] == 1
    run_config = json.loads((output_dir / "run_config.json").read_text())
    failure = run_config["failures"][0]
    assert failure["raw_path"] == raw_path
    assert failure["manifest_parent"] == str(manifest.parent.resolve())
    assert failure["attempted_resolved_path"] == str(
        (manifest.parent / raw_path).resolve()
    )
    assert run_config["evaluated_counts"]["source_frame_universe_count"] == 4
    assert run_config["evaluated_counts"][
        "successfully_loaded_source_frame_count"
    ] == 0


def test_absolute_supplier_path_is_not_rebased(tmp_path: Path) -> None:
    manifest = tmp_path / "batch" / "manifest.csv"
    absolute_path = tmp_path / "outside" / "missing.parquet"
    _write_manifest(manifest, str(absolute_path))
    output_dir = tmp_path / "audit-missing-absolute"

    audit_temporal_timebase(
        manifest=manifest,
        supplier="jdt",
        output_dir=output_dir,
        asset_ids=None,
        max_clips=1,
        config_path=None,
    )

    failure = json.loads((output_dir / "run_config.json").read_text())["failures"][0]
    assert failure["raw_path"] == str(absolute_path)
    assert failure["manifest_parent"] == str(manifest.parent.resolve())
    assert failure["attempted_resolved_path"] == str(absolute_path)


def test_metric_breakdown_uses_runtime_thresholds_and_preserves_ab_fields(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, "unused.parquet")
    output_dir = tmp_path / "metric-breakdown"

    audit_temporal_timebase(
        manifest=manifest,
        supplier="jdt",
        output_dir=output_dir,
        asset_ids=None,
        max_clips=1,
        config_path=None,
        clip_loader=lambda _row, _index: _clip(),
        temporal_config_overrides={
            "joint_names": ["probe_joint"],
            "project_2d": False,
            "projection_enabled": False,
            "reject_missing_keypoints": False,
            "joint_acceleration_m_s2_max_threshold": 0.5,
            "joint_displacement_m_max_threshold": 0.01,
            "strong_acceleration_ratio": 3.0,
            "strong_displacement_ratio": 2.0,
        },
    )

    breakdown = json.loads(
        (output_dir / "temporal_metric_exceedance_breakdown.json").read_text()
    )
    native_acceleration = next(
        row
        for row in breakdown
        if row["asset_id"] == "jdt__episode_000001"
        and row["timebase"] == "native"
        and row["metric"] == "acceleration"
    )
    assert native_acceleration["threshold"] == 0.5
    assert native_acceleration["strong_threshold"] == 1.5
    assert native_acceleration["eligible_frame_count"] > 0
    assert native_acceleration["exceed_frame_count"] > 0
    assert native_acceleration["exceed_rate"] is not None
    rotation = next(
        row
        for row in breakdown
        if row["asset_id"] == "jdt__episode_000001"
        and row["timebase"] == "standardized"
        and row["metric"] == "rotation"
    )
    assert rotation["strong_threshold"] is None
    assert rotation["strong_exceed_rate"] is None
    ab_rows = json.loads(
        (output_dir / "temporal_timebase_ab_audit.json").read_text()
    )
    assert "native_acceleration_p95" in ab_rows[0]
    assert "standardized_candidate_seed_rate" in ab_rows[0]


def test_candidate_coverage_uses_inclusive_source_frames_and_source_fps(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, "unused.parquet")
    output_dir = tmp_path / "coverage"

    audit_temporal_timebase(
        manifest=manifest,
        supplier="jdt",
        output_dir=output_dir,
        asset_ids=None,
        max_clips=1,
        config_path=None,
        clip_loader=lambda _row, _index: _clip(),
        temporal_config_overrides={
            "joint_names": ["probe_joint"],
            "project_2d": False,
            "projection_enabled": False,
            "reject_missing_keypoints": False,
            "joint_acceleration_m_s2_max_threshold": 0.5,
            "joint_displacement_m_max_threshold": 0.01,
            "candidate_min_seed_run_frames": 2,
            "candidate_gap_close_frames": 2,
            "candidate_pre_context_frames": 0,
            "candidate_post_context_frames": 0,
        },
    )

    rows = json.loads(
        (output_dir / "temporal_candidate_window_coverage.json").read_text()
    )
    for row in rows:
        assert row["candidate_source_frame_union_count"] == sum(
            end - start + 1 for start, end in row["candidate_source_frame_union"]
        )
        assert row["candidate_duration_seconds"] == pytest.approx(
            row["candidate_source_frame_union_count"] / 60.0
        )
        assert row["pre_context_frames"] == 0
        assert row["post_context_frames"] == 0
        assert row["min_seed_run"] == 2
    standardized = next(row for row in rows if row["timebase"] == "standardized")
    assert all(
        100 <= start <= end <= 111
        for start, end in standardized["candidate_source_frame_union"]
    )


def test_extreme_records_preserve_raw_values_and_audit_only_semantics(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, "unused.parquet")
    output_dir = tmp_path / "extremes"

    audit_temporal_timebase(
        manifest=manifest,
        supplier="jdt",
        output_dir=output_dir,
        asset_ids=None,
        max_clips=1,
        config_path=None,
        clip_loader=lambda _row, _index: _extreme_clip(),
        temporal_config_overrides={
            "project_2d": False,
            "projection_enabled": False,
            "reject_missing_keypoints": False,
        },
        finite_extreme_displacement_m=100.0,
        finite_extreme_acceleration_m_s2=1_000.0,
        finite_extreme_position_abs_m=100.0,
        top_extreme_records=5,
    )

    records = json.loads(
        (output_dir / "temporal_extreme_coordinate_records.json").read_text()
    )
    extreme = next(
        row
        for row in records
        if row["joint_name"] == "leftjoint_7"
        and row["current_position"][0] == 7_360.0
    )
    assert extreme["current_position"][0] == 7_360.0
    assert extreme["anomaly_classification"] == (
        "finite_extreme_coordinate_anomaly_candidate"
    )
    assert extreme["supplier_acknowledged_issue_pattern"] is True
    assert extreme["frame_level_supplier_confirmed"] is False
    assert extreme["presence_status"] == "valid"
    assert extreme["presence_reason"] == "keypoint_presence_valid"
    run_config = json.loads((output_dir / "run_config.json").read_text())
    audit_parameters = run_config["finite_extreme_audit_parameters"]
    assert audit_parameters["finite_extreme_position_abs_m"] == {
        "value": 100.0,
        "source": "cli",
        "enabled": True,
    }
    assert run_config["models_loaded"] == []
    assert run_config["mutates_precheck_outputs"] is False


def test_seed_reason_breakdown_preserves_production_side_view_reason() -> None:
    from tools.audit_temporal_timebase import (
        _seed_reason_breakdown_rows,
        _seed_reason_category,
    )

    check = SkeletonQualityScoreCheck(
        {
            "palm_camera_angle_review_threshold_deg": 80.0,
            "palm_camera_angle_min_valid_hands": 1,
            "projection_enabled": False,
        }
    )
    result = CheckResult(
        check="skeleton_quality_score",
        episode_idx=0,
        frame_idx=12,
        metrics={
            "temporal_output_valid": True,
            "keypoint_presence_invalid": 0.0,
            "which_thresholds_exceeded": [],
            "decision_metric_values": {
                "joint_acceleration_m_s2_max": 0.0,
                "joint_displacement_m_max": 0.0,
                "joint_angle_change_deg_max": 0.0,
                "rotation_delta_max": 0.0,
            },
            "palm_camera_angle_deg_max": 90.0,
            "side_view_hand_count": 1.0,
        },
    )

    rows = _seed_reason_breakdown_rows(
        [result],
        check=check,
        asset_id="asset",
        timebase="native",
    )
    side_view = next(
        row
        for row in rows
        if row["breakdown_kind"] == "exclusive_audit_category"
        and row["reason_category"] == "side_view_hand_orientation"
    )
    production = next(
        row
        for row in rows
        if row["breakdown_kind"] == "production_trigger_reason_combination"
    )

    assert side_view["candidate_seed_count"] == 1
    assert production["production_trigger_reason"] == (
        "side_view_hand_orientation"
    )
    assert _seed_reason_category(
        result,
        seed={
            "trigger_reason": [
                "acceleration_seed",
                "displacement_seed",
                "multi_signal_seed",
            ]
        },
    ) == "acceleration_and_displacement"


def test_extreme_ranking_keeps_metric_scales_and_nonfinite_separate() -> None:
    from tools.audit_temporal_timebase import _extreme_records_for_clip

    clip = _extreme_clip()
    for joint_index in range(10):
        clip.keypoints[f"rightjoint_{joint_index}"][0, 0] = np.nan

    records = _extreme_records_for_clip(
        clip=clip,
        asset_id="jdt__episode_extreme",
        temporal_rows=[],
        top_n=6,
        finite_extreme_displacement_m=None,
        finite_extreme_acceleration_m_s2=None,
        finite_extreme_position_abs_m=None,
        supplier_acknowledged_issue_pattern=True,
    )

    position_record = next(
        row for row in records if row["ranking_metric"] == "position_abs_m"
    )
    nonfinite_record = next(
        row
        for row in records
        if row["ranking_metric"] == "nonfinite_coordinate"
    )
    assert position_record["position_abs_m"] == 7_360.0
    assert position_record["ranking_value"] == 7_360.0
    assert nonfinite_record["anomaly_classification"] == "nonfinite_or_missing"
    assert None in nonfinite_record["current_position"]


def test_jdt_single_entry_cache_updates_only_after_successful_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.audit_temporal_timebase as audit_module

    path_a = tmp_path / "a.parquet"
    path_b = tmp_path / "b.parquet"
    calls: list[Path] = []
    failed_b_once = False

    def fake_read_parquet(path: Path) -> pd.DataFrame:
        nonlocal failed_b_once
        resolved = Path(path)
        calls.append(resolved)
        if resolved == path_b and not failed_b_once:
            failed_b_once = True
            raise OSError("transient B read failure")
        return pd.DataFrame({"marker": [resolved.stem]})

    monkeypatch.setattr(audit_module.pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(
        audit_module,
        "load_jdt_clip",
        lambda _row, episode_idx, source_frame: (
            episode_idx,
            str(source_frame.loc[0, "marker"]),
        ),
    )
    loader = audit_module._default_clip_loader("jdt", tmp_path / "manifest.csv")

    assert loader({"parquet_path": str(path_a)}, 0) == (0, "a")
    with pytest.raises(OSError, match="transient B"):
        loader({"parquet_path": str(path_b)}, 1)
    assert loader({"parquet_path": str(path_b)}, 2) == (2, "b")
    assert calls == [path_a, path_b, path_b]
