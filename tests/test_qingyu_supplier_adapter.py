from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from acceptance_pull.supplier_adapters.qingyu import (
    DEFAULT_CAMERA_SELECTION_CONFIG,
    MANIFEST_COLUMNS,
    build_qingyu_manifest,
)
from acceptance_pull.supplier_adapters.qingyu_hand_pose import (
    QingyuHandPoseSession,
    load_qingyu_clip,
)
from qc_common.manifest_metadata import canonical_manifest_metadata
from tests.qingyu_fixtures import (
    QY_CAMERAS,
    keypoints_2d,
    keypoints_3d,
    make_qy_episode,
    observation_rows,
    trajectory_rows,
)


def test_qy_manifest_uses_authoritative_timebase_and_unique_recommendation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "QY"
    rows = observation_rows(cameras=("mid_cam_left",))
    # A second qualified camera has lower bilateral coverage.
    rows.extend(
        observation_rows(
            cameras=("left_cam_left",),
            source_frames=(100, 101),
        )
    )
    make_qy_episode(root, observations=rows)

    row = build_qingyu_manifest(root)[0]

    assert list(row) == MANIFEST_COLUMNS
    assert row["schema_version"] == "supplier_manifest.qy.v1"
    assert row["supplier"] == row["supplier_id"] == "qy"
    assert row["supplier_name"] == "QY"
    assert row["supplier_alias"] == "qingyu"
    assert row["asset_id"] == "qy__packing__task_pack_box__episode_000001"
    assert row["primary_camera"] == "mid_cam_left"
    assert row["primary_camera_source"] == "coverage_recommendation"
    assert row["video_frame_count"] == "4"
    assert row["source_frame_count"] == "3"
    assert row["start_frame"] == "100"
    assert row["end_frame"] == "102"
    assert row["frame_coordinate_system"] == "source_inclusive"
    assert row["timebase_status"] == "valid"
    assert row["frame_mapping_status"] == "verified"
    assert row["adapter_status"] == "ready"
    coverage = json.loads(row["camera_coverage"])
    assert coverage["mid_cam_left"]["video_frame_count"] == 4
    assert coverage["mid_cam_left"]["source_frame_count"] == 3
    assert coverage["mid_cam_left"]["explicit_mapping_valid_count"] == 6
    assert coverage["mid_cam_left"]["source_video_identity_assumed"] is False
    assert float(row["camera_recommendation_score"]) > float(
        row["camera_second_best_score"]
    )
    assert json.loads(row["camera_selection_config"]) == (
        DEFAULT_CAMERA_SELECTION_CONFIG
    )


def test_qy_default_camera_thresholds_are_versioned_config() -> None:
    payload = yaml.safe_load(
        Path("configs/qy_camera_selection.yaml").read_text(encoding="utf-8")
    )

    assert payload == DEFAULT_CAMERA_SELECTION_CONFIG


def test_qy_timebase_never_derives_source_range_from_physical_frames(
    tmp_path: Path,
) -> None:
    root = tmp_path / "QY"
    episode = make_qy_episode(root)
    timebase_path = episode / "timestamps" / "episode_timebase.json"
    payload = json.loads(timebase_path.read_text(encoding="utf-8"))
    payload["videos"][2]["frames"] = 999
    timebase_path.write_text(json.dumps(payload), encoding="utf-8")

    row = build_qingyu_manifest(root, primary_camera="mid_cam_left")[0]

    assert row["video_frame_count"] == "999"
    assert row["start_frame"] == "100"
    assert row["end_frame"] == "102"
    assert row["source_frame_count"] == "3"


def test_qy_timebase_episode_identity_mismatch_is_not_accepted(
    tmp_path: Path,
) -> None:
    root = tmp_path / "QY"
    episode = make_qy_episode(root)
    timebase_path = episode / "timestamps" / "episode_timebase.json"
    payload = json.loads(timebase_path.read_text(encoding="utf-8"))
    payload["episode_id"] = "different_episode"
    timebase_path.write_text(json.dumps(payload), encoding="utf-8")

    row = build_qingyu_manifest(root, primary_camera="mid_cam_left")[0]

    assert row["primary_camera"] == ""
    assert row["timebase_status"] == "input_invalid"
    assert row["adapter_status"] == "input_missing"
    assert row["reason"] == "primary_camera_missing"


def test_qy_duplicate_timebase_camera_is_invalid_even_if_first_is_not_ok(
    tmp_path: Path,
) -> None:
    root = tmp_path / "QY"
    episode = make_qy_episode(root)
    timebase_path = episode / "timestamps" / "episode_timebase.json"
    payload = json.loads(timebase_path.read_text(encoding="utf-8"))
    duplicate = dict(payload["videos"][2])
    duplicate["status"] = "error"
    payload["videos"].insert(2, duplicate)
    timebase_path.write_text(json.dumps(payload), encoding="utf-8")

    row = build_qingyu_manifest(root, primary_camera="mid_cam_left")[0]

    assert row["timebase_status"] == "input_invalid"
    assert row["frame_mapping_status"] == "mapping_unverified"
    assert row["primary_camera"] == ""
    assert row["adapter_status"] == "input_missing"
    assert row["reason"] == "primary_camera_missing"


def test_qy_tied_recommendation_does_not_use_camera_name_tiebreak(
    tmp_path: Path,
) -> None:
    root = tmp_path / "QY"
    make_qy_episode(
        root,
        observations=observation_rows(
            cameras=("left_cam_left", "right_cam_right")
        ),
    )

    row = build_qingyu_manifest(root)[0]

    assert row["primary_camera"] == ""
    assert row["primary_camera_source"] == ""
    assert row["adapter_status"] == "input_missing"
    assert row["reason"] == "primary_camera_missing"
    assert row["camera_recommendation_reason"] == "best_camera_tied"


def test_qy_explicit_missing_or_ineligible_camera_never_falls_back(
    tmp_path: Path,
) -> None:
    root = tmp_path / "QY"
    make_qy_episode(root, observations=observation_rows(cameras=("mid_cam_left",)))

    row = build_qingyu_manifest(root, primary_camera="left_cam_left")[0]

    assert row["primary_camera"] == ""
    assert row["requested_primary_camera"] == "left_cam_left"
    assert row["adapter_status"] == "input_missing"
    assert row["reason"] == "primary_camera_missing"
    assert row["camera_recommendation_reason"] == "explicit_camera_ineligible"


def test_qy_mapping_conflict_is_not_guessed_from_rows_or_video_count(
    tmp_path: Path,
) -> None:
    root = tmp_path / "QY"
    rows = observation_rows()
    rows[1]["video_frame"] = rows[0]["video_frame"]
    rows[1]["source_frame_index"] = rows[0]["source_frame_index"] + 1
    make_qy_episode(root, observations=rows)

    row = build_qingyu_manifest(root)[0]

    assert row["frame_mapping_status"] == "mapping_unverified"
    assert row["primary_camera"] == ""
    assert row["adapter_status"] == "input_missing"
    assert row["reason"] == "primary_camera_missing"


def test_qy_load_clip_indexes_source_step_and_preserves_sparse_missing_frames(
    tmp_path: Path,
) -> None:
    root = tmp_path / "QY"
    make_qy_episode(root, trajectory=trajectory_rows(source_steps=(100, 102)))
    row = build_qingyu_manifest(root, primary_camera="mid_cam_left")[0]

    clip = load_qingyu_clip(row, episode_idx=0)

    assert clip.frame_indices == [100, 101, 102]
    assert clip.hand_keypoints_3d.shape == (3, 2, 21, 3)
    assert clip.hand_joint_valid_3d.shape == (3, 2, 21)
    assert clip.hand_joint_valid_3d[0].all()
    assert not clip.hand_joint_valid_3d[1].any()
    assert np.isnan(clip.hand_keypoints_3d[1]).all()
    assert clip.hand_joint_valid_3d[2].all()
    assert clip.sparse_trajectory is True
    assert clip.interpolation_applied is False
    assert clip.supplier_quality_signal["policy"] == "auxiliary_only"
    # Different timestamp origins do not affect source-step alignment.
    assert clip.timestamps_ns.tolist() == [0, -1, 33_333_333]


def test_qy_empty_or_null_3d_is_no_valid_output_not_file_presence_pass(
    tmp_path: Path,
) -> None:
    root = tmp_path / "QY"
    episode = make_qy_episode(root)
    pd.DataFrame(
        {
            "source_step": [None],
            "timestamp_seconds": [None],
            "hand": [None],
            "keypoints_3d_ref": [None],
            "reference_camera": [None],
            "joint_cameras_used": [None],
            "quality_tier": [None],
            "trajectory_quality": [None],
            "joint_mean_reprojection_error_px": [None],
            "joint_max_reprojection_error_px": [None],
        }
    ).to_parquet(episode / "hand_pose" / "trajectory_3d.parquet", index=False)

    row = build_qingyu_manifest(root, primary_camera="mid_cam_left")[0]
    clip = load_qingyu_clip(row, episode_idx=0)

    assert row["skeleton_3d_status"] == "no_valid_output"
    assert row["skeleton_3d_valid_row_count"] == "0"
    assert row["skeleton_3d_coverage_status"] == "no_valid_output"
    assert row["adapter_status"] == "input_missing"
    assert not clip.hand_joint_valid_3d.any()
    assert np.isnan(clip.hand_keypoints_3d).all()


def test_qy_single_required_hand_step_missing_is_input_missing(
    tmp_path: Path,
) -> None:
    root = tmp_path / "QY"
    rows = [
        row
        for row in trajectory_rows()
        if not (row["source_step"] == 101 and row["hand"] == "right")
    ]
    make_qy_episode(root, trajectory=rows)

    row = build_qingyu_manifest(root, primary_camera="mid_cam_left")[0]
    clip = load_qingyu_clip(row, episode_idx=0)

    assert row["skeleton_3d_status"] == "input_missing"
    assert row["skeleton_3d_coverage_status"] == "sparse"
    assert row["adapter_status"] == "input_missing"
    assert row["reason"] == "incomplete_3d_skeleton"
    assert clip.hand_joint_valid_3d[1, 0].all()
    assert not clip.hand_joint_valid_3d[1, 1].any()


def test_qy_trajectory_outside_authoritative_range_is_mapping_invalid(
    tmp_path: Path,
) -> None:
    root = tmp_path / "QY"
    make_qy_episode(root, trajectory=trajectory_rows(source_steps=(1, 2, 3)))

    row = build_qingyu_manifest(root, primary_camera="mid_cam_left")[0]

    assert row["start_frame"] == "100"
    assert row["end_frame"] == "102"
    assert row["skeleton_3d_status"] == "input_invalid"
    assert row["skeleton_3d_valid_row_count"] == "0"
    assert row["skeleton_3d_coverage_status"] == "mapping_invalid"
    assert row["adapter_status"] == "input_invalid"


def test_qy_partial_out_of_range_trajectory_is_mapping_invalid(
    tmp_path: Path,
) -> None:
    root = tmp_path / "QY"
    rows = trajectory_rows()
    rows.extend(trajectory_rows(source_steps=(103,)))
    make_qy_episode(root, trajectory=rows)

    row = build_qingyu_manifest(root, primary_camera="mid_cam_left")[0]
    audit = json.loads(row["skeleton_3d_audit"])

    assert audit["out_of_range_valid_row_count"] == 2
    assert audit["status"] == "input_invalid"
    assert audit["coverage_status"] == "mapping_invalid"
    assert row["skeleton_3d_status"] == "input_invalid"
    assert row["adapter_status"] == "input_invalid"


@pytest.mark.parametrize("reference_camera", [None, "unknown_camera"])
def test_qy_trajectory_requires_valid_stable_reference_camera(
    tmp_path: Path,
    reference_camera: str | None,
) -> None:
    root = tmp_path / "QY"
    rows = trajectory_rows()
    rows[0]["reference_camera"] = reference_camera
    make_qy_episode(root, trajectory=rows)

    row = build_qingyu_manifest(root, primary_camera="mid_cam_left")[0]
    audit = json.loads(row["skeleton_3d_audit"])

    assert audit["status"] == "input_invalid"
    assert audit["reference_camera_status"] == "mapping_invalid"
    assert row["adapter_status"] == "input_invalid"


def test_qy_trajectory_rejects_reference_camera_drift(tmp_path: Path) -> None:
    root = tmp_path / "QY"
    rows = trajectory_rows()
    rows[-1]["reference_camera"] = "left_cam_left"
    make_qy_episode(root, trajectory=rows)

    row = build_qingyu_manifest(root, primary_camera="mid_cam_left")[0]
    audit = json.loads(row["skeleton_3d_audit"])

    assert audit["status"] == "input_invalid"
    assert audit["reference_camera_status"] == "mapping_invalid"


def test_qy_corrupt_parquet_isolated_to_episode_manifest_row(tmp_path: Path) -> None:
    root = tmp_path / "QY"
    bad = make_qy_episode(root, episode_id="episode_bad")
    make_qy_episode(root, episode_id="episode_good")
    (bad / "hand_pose" / "observations_2d.parquet").write_bytes(b"not parquet")

    rows = build_qingyu_manifest(root, primary_camera="mid_cam_left")
    by_episode = {row["episode_id"]: row for row in rows}

    assert set(by_episode) == {"episode_bad", "episode_good"}
    assert by_episode["episode_bad"]["adapter_status"] == "input_invalid"
    assert by_episode["episode_bad"]["reason"] == "observations_2d_invalid"
    assert by_episode["episode_good"]["adapter_status"] == "ready"


def test_qy_invalid_coordinate_system_json_is_input_invalid(tmp_path: Path) -> None:
    root = tmp_path / "QY"
    episode = make_qy_episode(root)
    (episode / "hand_pose" / "coordinate_system.json").write_text(
        "not json", encoding="utf-8"
    )

    row = build_qingyu_manifest(root, primary_camera="mid_cam_left")[0]

    assert row["coordinate_system_status"] == "input_invalid"
    assert row["adapter_status"] == "input_invalid"
    assert row["reason"] == "coordinate_system_invalid"


@pytest.mark.parametrize(
    "bad_points",
    [
        keypoints_3d()[:20],
        [[0.0, 0.0]] * 21,
        [[float("nan"), 0.0, 1.0]] + keypoints_3d()[1:],
        [[float("inf"), 0.0, 1.0]] + keypoints_3d()[1:],
    ],
)
def test_qy_invalid_3d_shape_or_nonfinite_is_hard_invalid(
    tmp_path: Path,
    bad_points: list[list[float]],
) -> None:
    root = tmp_path / "QY"
    rows = trajectory_rows()
    rows[0]["keypoints_3d_ref"] = bad_points
    make_qy_episode(root, trajectory=rows)

    row = build_qingyu_manifest(root, primary_camera="mid_cam_left")[0]
    clip = load_qingyu_clip(row, episode_idx=0)

    assert row["skeleton_3d_status"] == "input_invalid"
    assert not clip.hand_joint_valid_3d[0, 0].any()
    assert np.isnan(clip.hand_keypoints_3d[0, 0]).all()


def test_qy_duplicate_hand_step_is_input_invalid(tmp_path: Path) -> None:
    root = tmp_path / "QY"
    rows = trajectory_rows()
    rows.append(dict(rows[0]))
    make_qy_episode(root, trajectory=rows)

    row = build_qingyu_manifest(root, primary_camera="mid_cam_left")[0]

    assert row["skeleton_3d_status"] == "input_invalid"
    audit = json.loads(row["skeleton_3d_audit"])
    assert audit["duplicate_hand_step_count"] == 1


def test_qy_session_reads_each_parquet_only_once(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "QY"
    episode = make_qy_episode(root)
    calls: list[Path] = []
    original = pd.read_parquet

    def counted(path, *args, **kwargs):
        calls.append(Path(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", counted)
    session = QingyuHandPoseSession(
        observations_path=episode / "hand_pose" / "observations_2d.parquet",
        trajectory_path=episode / "hand_pose" / "trajectory_3d.parquet",
    )
    session.audit_2d()
    session.audit_3d()
    session.build_clip(
        asset_id="asset",
        start_frame=100,
        end_frame=102,
        primary_camera="mid_cam_left",
        fps=30.0,
        episode_idx=0,
    )

    assert calls.count(session.observations_path) == 1
    assert calls.count(session.trajectory_path) == 1


def test_qy_2d_requires_exact_finite_21_by_2(tmp_path: Path) -> None:
    root = tmp_path / "QY"
    rows = observation_rows()
    rows[0]["pred_keypoints_2d"] = keypoints_2d()[:20]
    make_qy_episode(root, observations=rows)

    row = build_qingyu_manifest(root)[0]

    coverage = json.loads(row["camera_coverage"])["mid_cam_left"]
    assert coverage["invalid_shape_row_count"] == 1
    assert coverage["eligible"] is False
    assert row["primary_camera"] == ""


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_qy_2d_nonfinite_joint_is_ineligible(
    tmp_path: Path,
    bad_value: float,
) -> None:
    root = tmp_path / "QY"
    rows = observation_rows()
    rows[0]["pred_keypoints_2d"][0][0] = bad_value
    make_qy_episode(root, observations=rows)

    row = build_qingyu_manifest(root)[0]

    coverage = json.loads(row["camera_coverage"])["mid_cam_left"]
    assert coverage["invalid_shape_row_count"] == 1
    assert coverage["eligible"] is False
    assert row["adapter_status"] == "input_missing"


def test_qy_manifest_identity_keeps_logical_symlink_paths(
    tmp_path: Path,
) -> None:
    first = tmp_path / "mount-a" / "QY"
    second = tmp_path / "mount-b" / "QY"
    make_qy_episode(first)
    make_qy_episode(second)
    logical = tmp_path / "source" / "QY"
    logical.parent.mkdir(parents=True)
    logical.symlink_to(first, target_is_directory=True)

    first_row = build_qingyu_manifest(
        logical, primary_camera="mid_cam_left"
    )[0]
    logical.unlink()
    logical.symlink_to(second, target_is_directory=True)
    second_row = build_qingyu_manifest(
        logical, primary_camera="mid_cam_left"
    )[0]

    assert first_row["episode_root"] == second_row["episode_root"]
    assert first_row["trajectory_3d_path"] == second_row["trajectory_3d_path"]
    assert str(logical) in first_row["trajectory_3d_path"]
    assert canonical_manifest_metadata({"manifest_row": first_row}) == (
        canonical_manifest_metadata({"manifest_row": second_row})
    )
