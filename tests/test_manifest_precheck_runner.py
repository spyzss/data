from __future__ import annotations

import ast
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from qc_common.types import ClipInputs


def _hand_points(frame_count: int, jump: bool = False) -> np.ndarray:
    base = np.stack(
        [
            np.array([joint_idx * 0.002, (joint_idx % 5) * 0.003, 0.5])
            for joint_idx in range(21)
        ],
        axis=0,
    )
    frames = []
    for frame_idx in range(frame_count):
        offset = 0.2 if jump and frame_idx % 2 else 0.0
        frames.append(base + np.array([offset, 0.0, 0.0]))
    return np.asarray(frames, dtype=np.float32)


def _write_deepreach_hdf5(
    path: Path,
    *,
    frame_count: int = 12,
    jump: bool = False,
    invalid_frame: int | None = None,
) -> Path:
    with h5py.File(path, "w") as handle:
        handle.attrs["fps"] = 29.97
        handle.attrs["coordinate_frame"] = "head_camera"
        handle.attrs["units"] = "meters"
        handle.attrs["task"] = "fold cloth"
        handle.create_dataset("timestamp", data=np.arange(frame_count) / 29.97)
        hand = handle.create_group("hand")
        for side in ("left", "right"):
            group = hand.create_group(side)
            valid = np.ones(frame_count, dtype=np.uint8)
            if invalid_frame is not None:
                valid[invalid_frame] = 0
            group.create_dataset("valid", data=valid)
            group.create_dataset(
                "joints3d",
                data=_hand_points(frame_count, jump=jump),
            )
    return path


def _jdt_frame(frame_idx: int) -> dict[str, object]:
    points = _hand_points(8, jump=True)[frame_idx]
    points_2d = np.arange(42, dtype=np.float32) + frame_idx
    return {
        "left_kp3d": points.reshape(-1).tolist(),
        "right_kp3d": (points + 0.01).reshape(-1).tolist(),
        "leftcam_left_kp2d": points_2d.tolist(),
        "leftcam_right_kp2d": (points_2d + 10).tolist(),
        "language_instruction": f"instruction-{frame_idx}",
        "first_scene_cn": "场景一",
        "first_scene_en": "scene one",
        "second_scene_cn": "场景二",
        "second_scene_en": "scene two",
        "third_scene_cn": "场景三",
        "third_scene_en": "scene three",
    }


def test_manifest_precheck_reads_jsonl_manifest(tmp_path: Path) -> None:
    from tools.run_manifest_precheck import read_manifest

    path = tmp_path / "manifest.jsonl"
    row = {"asset_id": "clip-a", "start_frame": 0, "end_frame": 2}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    assert read_manifest(path) == [row]


def test_deepreach_adapter_slices_inclusive_range_without_quality_hand(
    tmp_path: Path,
) -> None:
    from tools.run_manifest_precheck import load_deepreach_clip

    hdf5_path = _write_deepreach_hdf5(
        tmp_path / "deepreach.h5",
        invalid_frame=3,
    )
    clip = load_deepreach_clip(
        {
            "asset_id": "dr-clip",
            "hdf5_path": str(hdf5_path),
            "start_frame": 2,
            "end_frame": 5,
            "task": "fold cloth",
            "subtask_description": "fold the left edge",
        },
        episode_idx=7,
    )

    assert clip.episode_idx == 7
    assert clip.frame_indices == [2, 3, 4, 5]
    assert clip.num_frames == 4
    assert clip.quality_hand is None
    assert clip.text_label == {
        "task": "fold cloth",
        "subtask_description": "fold the left edge",
    }
    assert clip.keypoints is not None
    assert clip.keypoints["leftHand"].shape == (4, 3)
    assert np.isnan(clip.keypoints["leftHand"][1]).all()
    assert getattr(clip, "supplier_quality_signal") == "not_provided"
    assert getattr(clip, "morphology_status") == "not_ready_topology"


def test_jdt_adapter_reshapes_3d_and_preserves_cam_left_2d(tmp_path: Path) -> None:
    from tools.run_manifest_precheck import load_jdt_clip

    parquet_path = tmp_path / "jdt.parquet"
    pd.DataFrame([_jdt_frame(index) for index in range(8)]).to_parquet(
        parquet_path,
        index=False,
    )
    clip = load_jdt_clip(
        {
            "asset_id": "jdt-clip",
            "parquet_path": str(parquet_path),
            "start_frame": 3,
            "end_frame": 5,
        },
        episode_idx=9,
    )

    assert clip.frame_indices == [3, 4, 5]
    assert clip.num_frames == 3
    assert clip.keypoints is not None
    assert clip.keypoints["leftHand"].shape == (3, 3)
    source_points = np.asarray(_jdt_frame(3)["left_kp3d"]).reshape(21, 3)
    assert np.array_equal(clip.keypoints["leftHand"][0], source_points[0])
    assert np.array_equal(
        clip.keypoints["leftThumbKnuckle"][0],
        source_points[13],
    )
    assert np.array_equal(
        clip.keypoints["leftIndexFingerTip"][0],
        source_points[17],
    )
    assert getattr(clip, "leftcam_left_kp2d").shape == (3, 21, 2)
    assert getattr(clip, "leftcam_right_kp2d").shape == (3, 21, 2)
    assert clip.text_label is not None
    assert clip.text_label["language_instruction"] == "instruction-3"
    assert getattr(clip, "primary_camera") == "observation.images.cam_left"
    assert getattr(clip, "morphology_status") == "not_ready_topology"


def test_text_integrity_prefers_nonempty_manifest_scene_and_task() -> None:
    from precheck.checks.text_integrity import TextIntegrityCheck

    clip = ClipInputs(
        episode_idx=1,
        text_label={
            "scene": "canonical scene",
            "task": "canonical task",
            "language_instruction": "do not rewrite",
        },
        manifest_metadata={
            "scene": "manifest scene",
            "task": "manifest task",
            "task_name": "must not substitute",
            "text_en": "do not substitute",
            "text_label": "do not substitute",
        },
    )

    text_label, parse_error, absent = TextIntegrityCheck(
        {"required_fields": ["scene", "task"]}
    )._text_label(clip)

    assert parse_error is None
    assert absent is False
    assert text_label == {
        "scene": "manifest scene",
        "task": "manifest task",
        "language_instruction": "do not rewrite",
    }


def test_text_integrity_empty_manifest_values_preserve_canonical_scene_and_task() -> None:
    from precheck.checks.text_integrity import TextIntegrityCheck

    clip = ClipInputs(
        episode_idx=1,
        text_label={"scene": "canonical scene", "task": "canonical task"},
        manifest_metadata={"scene": "  ", "task": ""},
    )

    text_label, _parse_error, _absent = TextIntegrityCheck(
        {"required_fields": ["scene", "task"]}
    )._text_label(clip)

    assert text_label == {"scene": "canonical scene", "task": "canonical task"}


def test_text_integrity_does_not_promote_task_name_or_text_fields_to_task() -> None:
    from precheck.checks.text_integrity import TextIntegrityCheck

    clip = ClipInputs(
        episode_idx=1,
        text_label={"language_instruction": "existing canonical text"},
        manifest_metadata={
            "task_name": "not a task fallback",
            "text_en": "not a task fallback",
            "text_label": "not a task fallback",
        },
    )

    result = TextIntegrityCheck({"required_fields": ["task"]}).run(clip)[0]

    assert result.flag is True
    assert result.metrics["field_present_task"] == 0.0
    assert '"task"' in result.reason


def test_jdt_unified_presence_uses_3d_keypoints_without_quality_hand(
    tmp_path: Path,
) -> None:
    from qc_common.config import load_qc_acceptance_config
    from qc_pipeline.context import AssetContext
    from qc_pipeline.runners.precheck import PrecheckSession

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    parquet_path = source_dir / "jdt.parquet"
    pd.DataFrame([_jdt_frame(index) for index in range(8)]).to_parquet(
        parquet_path,
        index=False,
    )
    context = AssetContext(
        asset_id="jdt-clip",
        batch_root=tmp_path,
        report_path=tmp_path / "quality_archive" / "jdt-clip.json",
        source_files={"parquet": {"path": "source/jdt.parquet"}},
        source_range=(3, 6),
        metadata={"supplier": "jdt"},
    )

    result = PrecheckSession(context, load_qc_acceptance_config()).run_module(
        "keypoint_presence"
    )

    assert result.verdict == "pass"
    assert result.evaluation["decision"] == "pass"
    assert result.evaluation["checked_frame_count"] == 3
    assert result.evaluation["invalid_frame_count"] == 0
    assert result.evaluation.get("reason") != "source_signal_not_provided"
    assert result.metrics["min_valid_keypoint_count_left"] == 21.0
    assert result.metrics["min_valid_keypoint_count_right"] == 21.0


def test_unified_jdt_text_integrity_receives_manifest_metadata(
    tmp_path: Path,
) -> None:
    from qc_common.config import load_qc_acceptance_config
    from qc_pipeline.context import AssetContext
    from qc_pipeline.runners.precheck import PrecheckSession

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    pd.DataFrame([_jdt_frame(index) for index in range(8)]).to_parquet(
        source_dir / "jdt.parquet",
        index=False,
    )
    manifest_metadata = {
        "supplier": "jdt",
        "scene": "manifest kitchen",
        "task": "manifest pick",
        "task_name": "pick_task_name",
        "text_en": "Pick the cup.",
        "text_label": "display-only label",
    }
    context = AssetContext(
        asset_id="jdt-text-metadata",
        batch_root=tmp_path,
        report_path=tmp_path / "quality_archive" / "jdt-text-metadata.json",
        source_files={"parquet": {"path": "source/jdt.parquet"}},
        source_range=(3, 6),
        metadata=manifest_metadata,
    )
    session = PrecheckSession(context, load_qc_acceptance_config())

    result = session.run_module("hdf5_text_info")

    assert result.verdict == "pass"
    assert result.metrics["field_present_scene"] == 1.0
    assert result.metrics["field_present_task"] == 1.0
    assert session._clip is not None
    assert session._clip.manifest_metadata == manifest_metadata
    assert session._clip.manifest_metadata["task_name"] == "pick_task_name"
    assert session._clip.manifest_metadata["text_en"] == "Pick the cup."
    assert session._clip.manifest_metadata["text_label"] == "display-only label"


def test_jdt_short_3d_cell_becomes_presence_invalid_instead_of_load_failure(
    tmp_path: Path,
) -> None:
    from qc_common.config import load_qc_acceptance_config
    from qc_pipeline.context import AssetContext
    from qc_pipeline.runners.precheck import PrecheckSession

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    rows = [_jdt_frame(index) for index in range(8)]
    rows[4]["left_kp3d"] = rows[4]["left_kp3d"][:60]
    pd.DataFrame(rows).to_parquet(source_dir / "jdt.parquet", index=False)
    context = AssetContext(
        asset_id="jdt-short-cell",
        batch_root=tmp_path,
        report_path=tmp_path / "quality_archive" / "jdt-short-cell.json",
        source_files={"parquet": {"path": "source/jdt.parquet"}},
        source_range=(3, 6),
        metadata={"supplier": "jdt"},
    )

    result = PrecheckSession(context, load_qc_acceptance_config()).run_module(
        "keypoint_presence"
    )

    assert result.verdict == "fail"
    assert result.evaluation["affected_frame_count"] == 1
    assert result.evaluation["affected_frame_ranges"] == ((4, 4),)
    detail = result.evaluation["invalid_frame_details"][0]
    assert detail["side"] == "left"
    assert detail["frame_idx"] == 4
    assert detail["valid_point_count"] == 20
    assert "invalid_coordinate_shape" in detail["invalid_reasons"]
    assert "insufficient_valid_keypoint_count" in detail["invalid_reasons"]


def test_jdt_missing_hand_3d_column_isolated_as_presence_invalid(
    tmp_path: Path,
) -> None:
    from qc_common.config import load_qc_acceptance_config
    from qc_pipeline.context import AssetContext
    from qc_pipeline.runners.precheck import PrecheckSession

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    rows = [_jdt_frame(index) for index in range(8)]
    pd.DataFrame(rows).drop(columns=["left_kp3d"]).to_parquet(
        source_dir / "jdt.parquet",
        index=False,
    )
    context = AssetContext(
        asset_id="jdt-missing-left",
        batch_root=tmp_path,
        report_path=tmp_path / "quality_archive" / "jdt-missing-left.json",
        source_files={"parquet": {"path": "source/jdt.parquet"}},
        source_range=(3, 6),
        metadata={"supplier": "jdt"},
    )

    result = PrecheckSession(context, load_qc_acceptance_config()).run_module(
        "keypoint_presence"
    )

    assert result.verdict == "fail"
    assert result.evaluation["affected_frame_count"] == 3
    assert result.evaluation["affected_frame_ranges"] == ((3, 5),)
    assert {
        (detail["side"], detail["frame_idx"])
        for detail in result.evaluation["invalid_frame_details"]
    } == {("left", 3), ("left", 4), ("left", 5)}
    assert all(
        "invalid_coordinate_shape" in detail["invalid_reasons"]
        for detail in result.evaluation["invalid_frame_details"]
    )


def test_manifest_precheck_outputs_source_frame_mapping_and_candidate_windows(
    tmp_path: Path,
) -> None:
    from tools.run_manifest_precheck import run_manifest_precheck

    hdf5_path = _write_deepreach_hdf5(
        tmp_path / "deepreach.h5",
        frame_count=12,
        jump=True,
    )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "asset_id": "dr-range",
                "hdf5_path": str(hdf5_path),
                "start_frame": 3,
                "end_frame": 10,
                "task": "fold cloth",
                "subtask_description": "fold the left edge",
            }
        ]
    ).to_csv(manifest, index=False)
    output_dir = tmp_path / "precheck"

    run_manifest_precheck(
        manifest,
        supplier="deepreach",
        output_dir=output_dir,
        checks=["text_integrity", "keypoint_temporal", "skeleton_quality_score"],
    )

    rows = json.loads((output_dir / "check_results.json").read_text())
    temporal = [
        row
        for row in rows
        if row["check"] == "keypoint_temporal" and row["frame_idx"] >= 0
    ]
    assert [row["source_frame_idx"] for row in temporal] == list(range(3, 11))
    assert [row["local_frame_idx"] for row in temporal] == list(range(8))
    windows = json.loads((output_dir / "candidate_windows.json").read_text())
    assert windows
    assert all(window["asset_id"] == "dr-range" for window in windows)
    assert all(
        3 <= window["start_frame"] <= window["end_frame"] <= 10
        for window in windows
    )
    assert any(
        window["local_start_frame"] == 0
        and window["local_end_frame"] == 7
        and window["start_frame"] == 3
        and window["end_frame"] == 10
        for window in windows
    )
    assert all(window["coordinate_space"] == "source" for window in windows)
    assert all(window["source_start_frame"] == window["start_frame"] for window in windows)
    assert all(window["source_end_frame"] == window["end_frame"] for window in windows)
    assert all(window["local_start_frame"] == window["start_frame"] - 3 for window in windows)
    assert all(window["local_end_frame"] == window["end_frame"] - 3 for window in windows)
    from qc_pipeline.adapters.precheck import adapt_keypoint_temporal
    from tests.qc_report_fixtures import loaded_test_config

    adapted = adapt_keypoint_temporal(
        asset_id="dr-range",
        source_relative_path="deepreach.h5",
        results=[],
        candidate_windows=windows,
        config=loaded_test_config(),
    )
    assert adapted.verdict == "warn"
    assert len(adapted.issues) == len(windows)
    assert all(issue.module == "keypoint_temporal" for issue in adapted.issues)
    assert [
        (issue.context["start_frame"], issue.context["end_frame"])
        for issue in adapted.issues
    ] == [(window["start_frame"], window["end_frame"]) for window in windows]
    assert all(
        evidence.path == "candidate_windows.json"
        and evidence.coordinate_system == "source_inclusive"
        for evidence in adapted.evidence
    )
    aggregates = json.loads((output_dir / "clip_aggregates.json").read_text())
    assert all(row["morphology_status"] == "not_ready_topology" for row in aggregates)
    assert all(row["supplier_quality_signal"] == "not_provided" for row in aggregates)
    run_config = json.loads(
        (output_dir / "run_config.json").read_text(encoding="utf-8")
    )
    assert (
        run_config["presence_check_source"]
        == "skeleton_quality_score.keypoint_presence_invalid"
    )
    assert run_config["topology_status"] == "not_ready_topology"
    for filename in (
        "check_results.json",
        "check_results.parquet",
        "clip_aggregates.json",
        "clip_aggregates.parquet",
        "candidate_windows.json",
        "candidate_windows.parquet",
        "failures.json",
        "run_config.json",
    ):
        assert (output_dir / filename).exists()


def test_candidate_window_maps_local_boundaries_to_source_once() -> None:
    from tools.run_manifest_precheck import _map_candidate_window_to_source

    mapped = _map_candidate_window_to_source(
        {"start_frame": 0, "end_frame": 7},
        asset_id="dr-range",
        supplier="deepreach",
        source_path="deepreach.h5",
        clip_start_frame=3,
        clip_end_frame=10,
        clip_frame_count=8,
    )

    assert mapped["local_start_frame"] == 0
    assert mapped["local_end_frame"] == 7
    assert mapped["start_frame"] == 3
    assert mapped["end_frame"] == 10
    assert mapped["source_start_frame"] == 3
    assert mapped["source_end_frame"] == 10
    assert mapped["coordinate_space"] == "source"
    assert mapped["asset_id"] == "dr-range"
    assert mapped["clip_start_frame"] == 3
    assert mapped["clip_end_frame"] == 10


def test_candidate_window_explicit_source_coordinates_are_not_double_offset() -> None:
    from tools.run_manifest_precheck import _map_candidate_window_to_source

    mapped = _map_candidate_window_to_source(
        {
            "start_frame": 3,
            "end_frame": 10,
            "coordinate_space": "source",
        },
        asset_id="dr-range",
        supplier="deepreach",
        source_path="deepreach.h5",
        clip_start_frame=3,
        clip_end_frame=10,
        clip_frame_count=8,
    )

    assert mapped["local_start_frame"] == 0
    assert mapped["local_end_frame"] == 7
    assert mapped["start_frame"] == 3
    assert mapped["end_frame"] == 10
    assert mapped["coordinate_space"] == "source"


@pytest.mark.parametrize(
    ("candidate", "message"),
    [
        ({"start_frame": -1, "end_frame": 0}, "local candidate bounds"),
        ({"start_frame": 0, "end_frame": 8}, "local candidate bounds"),
        ({"start_frame": 4, "end_frame": 3}, "local candidate bounds"),
        (
            {
                "start_frame": 2,
                "end_frame": 10,
                "coordinate_space": "source",
            },
            "source candidate bounds",
        ),
    ],
)
def test_candidate_window_rejects_invalid_bounds(
    candidate: dict[str, object],
    message: str,
) -> None:
    from tools.run_manifest_precheck import _map_candidate_window_to_source

    with pytest.raises(ValueError, match=message):
        _map_candidate_window_to_source(
            candidate,
            asset_id="dr-range",
            supplier="deepreach",
            source_path="deepreach.h5",
            clip_start_frame=3,
            clip_end_frame=10,
            clip_frame_count=8,
        )


def test_manifest_precheck_isolates_invalid_candidate_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.run_manifest_precheck as runner

    hdf5_path = _write_deepreach_hdf5(tmp_path / "deepreach.h5", frame_count=8)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "asset_id": "invalid-window",
                "hdf5_path": str(hdf5_path),
                "start_frame": 0,
                "end_frame": 3,
                "task": "fold",
                "subtask_description": "fold edge",
            },
            {
                "asset_id": "valid-window",
                "hdf5_path": str(hdf5_path),
                "start_frame": 4,
                "end_frame": 7,
                "task": "fold",
                "subtask_description": "fold edge",
            },
        ]
    ).to_csv(manifest, index=False)
    original = runner.PrecheckRunner.run_clip

    def add_candidate(self, clip):
        results = original(self, clip)
        end_frame = clip.num_frames if clip.asset_id == "invalid-window" else 3
        self.candidate_window_records.append(
            {"start_frame": 0, "end_frame": end_frame}
        )
        return results

    monkeypatch.setattr(runner.PrecheckRunner, "run_clip", add_candidate)
    output_dir = tmp_path / "precheck"
    summary = runner.run_manifest_precheck(
        manifest,
        supplier="deepreach",
        output_dir=output_dir,
        checks=["text_integrity"],
    )

    assert summary["completed_clip_count"] == 2
    windows = json.loads((output_dir / "candidate_windows.json").read_text())
    assert [window["asset_id"] for window in windows] == ["valid-window"]
    assert windows[0]["local_start_frame"] == 0
    assert windows[0]["local_end_frame"] == 3
    assert windows[0]["start_frame"] == 4
    assert windows[0]["end_frame"] == 7
    failures = json.loads((output_dir / "failures.json").read_text())
    assert len(failures) == 1
    assert failures[0]["asset_id"] == "invalid-window"
    assert failures[0]["failure_stage"] == "candidate_window_mapping"
    assert "local candidate bounds" in failures[0]["error"]


def test_manifest_precheck_isolates_bad_manifest_rows(tmp_path: Path) -> None:
    from tools.run_manifest_precheck import run_manifest_precheck

    valid_hdf5 = _write_deepreach_hdf5(tmp_path / "valid.h5", frame_count=5)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "asset_id": "valid",
                "hdf5_path": str(valid_hdf5),
                "start_frame": 0,
                "end_frame": 4,
                "task": "fold",
                "subtask_description": "fold edge",
            },
            {
                "asset_id": "missing",
                "hdf5_path": str(tmp_path / "missing.h5"),
                "start_frame": 0,
                "end_frame": 4,
                "task": "fold",
                "subtask_description": "fold edge",
            },
        ]
    ).to_csv(manifest, index=False)

    summary = run_manifest_precheck(
        manifest,
        supplier="deepreach",
        output_dir=tmp_path / "precheck",
        checks=["text_integrity", "keypoint_temporal"],
    )

    assert summary["completed_clip_count"] == 1
    assert summary["failed_clip_count"] == 1
    failures = json.loads(
        (tmp_path / "precheck" / "failures.json").read_text()
    )
    assert failures[0]["asset_id"] == "missing"
    assert failures[0]["source_path"].endswith("missing.h5")
    assert failures[0]["clip_start_frame"] == 0
    assert failures[0]["clip_end_frame"] == 4


def test_manifest_precheck_runner_has_no_model_or_annotation_imports() -> None:
    path = Path("tools/run_manifest_precheck.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = ("annotation", "sam3", "da3", "qwen")
    assert not {
        name
        for name in imported
        if name.lower().startswith(forbidden)
    }


def test_manifest_precheck_dry_run_validates_without_outputs(tmp_path: Path) -> None:
    from tools.run_manifest_precheck import run_manifest_precheck

    hdf5_path = _write_deepreach_hdf5(tmp_path / "deepreach.h5", frame_count=6)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "asset_id": "valid",
                "hdf5_path": str(hdf5_path),
                "start_frame": 1,
                "end_frame": 4,
                "task": "fold",
                "subtask_description": "fold edge",
            },
            {
                "asset_id": "invalid",
                "hdf5_path": str(hdf5_path),
                "start_frame": 5,
                "end_frame": 9,
                "task": "fold",
                "subtask_description": "fold edge",
            },
        ]
    ).to_csv(manifest, index=False)
    output_dir = tmp_path / "precheck"

    summary = run_manifest_precheck(
        manifest,
        supplier="deepreach",
        output_dir=output_dir,
        dry_run=True,
    )

    assert summary == {
        "manifest_row_count": 2,
        "validated_clip_count": 1,
        "completed_clip_count": 0,
        "failed_clip_count": 1,
        "skipped_clip_count": 0,
        "dry_run": True,
    }
    assert not output_dir.exists()


def test_manifest_precheck_reuses_jdt_source_for_multiple_ranges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.run_manifest_precheck as runner

    parquet_path = tmp_path / "jdt.parquet"
    pd.DataFrame([_jdt_frame(index) for index in range(8)]).to_parquet(
        parquet_path,
        index=False,
    )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "asset_id": "jdt-left",
                "parquet_path": str(parquet_path),
                "start_frame": 0,
                "end_frame": 2,
            },
            {
                "asset_id": "jdt-right",
                "parquet_path": str(parquet_path),
                "start_frame": 5,
                "end_frame": 7,
            },
        ]
    ).to_csv(manifest, index=False)
    calls = 0
    original = runner.pd.read_parquet

    def counted(path: Path):
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(runner.pd, "read_parquet", counted)
    output_dir = tmp_path / "precheck"
    summary = runner.run_manifest_precheck(
        manifest,
        supplier="jdt",
        output_dir=output_dir,
        checks=["keypoint_temporal"],
    )

    assert summary["completed_clip_count"] == 2
    assert calls == 1
    rows = json.loads((output_dir / "check_results.json").read_text())
    by_asset = {
        asset_id: [row for row in rows if row["asset_id"] == asset_id]
        for asset_id in ("jdt-left", "jdt-right")
    }
    for asset_id, start_frame in (("jdt-left", 0), ("jdt-right", 5)):
        temporal = [
            row
            for row in by_asset[asset_id]
            if row["check"] == "keypoint_temporal" and row["frame_idx"] >= 0
        ]
        assert [row["source_frame_idx"] for row in temporal] == list(
            range(start_frame, start_frame + 3)
        )
        assert [row["local_frame_idx"] for row in temporal] == [0, 1, 2]


def test_manifest_precheck_skips_completed_assets_unless_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.run_manifest_precheck as runner

    hdf5_path = _write_deepreach_hdf5(tmp_path / "deepreach.h5", frame_count=5)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "asset_id": "dr-range",
                "hdf5_path": str(hdf5_path),
                "start_frame": 0,
                "end_frame": 4,
                "task": "fold",
                "subtask_description": "fold edge",
            }
        ]
    ).to_csv(manifest, index=False)
    output_dir = tmp_path / "precheck"
    runner.run_manifest_precheck(
        manifest,
        supplier="deepreach",
        output_dir=output_dir,
        checks=["keypoint_temporal"],
    )

    calls = 0
    original = runner.PrecheckRunner.run_clip

    def counted(self, clip):
        nonlocal calls
        calls += 1
        return original(self, clip)

    monkeypatch.setattr(runner.PrecheckRunner, "run_clip", counted)
    skipped = runner.run_manifest_precheck(
        manifest,
        supplier="deepreach",
        output_dir=output_dir,
        checks=["keypoint_temporal"],
    )
    assert skipped["skipped_clip_count"] == 1
    assert calls == 0

    runner.run_manifest_precheck(
        manifest,
        supplier="deepreach",
        output_dir=output_dir,
        checks=["keypoint_temporal"],
        overwrite=True,
    )
    assert calls == 1
