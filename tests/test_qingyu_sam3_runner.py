from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from acceptance_pull.supplier_adapters.qingyu import (
    build_qingyu_manifest,
    write_qingyu_manifest,
)
from qc_common.config import load_qc_acceptance_config
from qc_common.module_registry import ModuleBlockedError
from qc_pipeline.artifacts import artifact_for, write_run_config
from qc_pipeline.runners.precheck import MODULES, precheck_fingerprint
from qc_pipeline.runners.sam3_containment import runner
from tests.fixtures import solid_frame, write_test_video
from tests.qingyu_fixtures import make_qy_episode, observation_rows
from tools.run_qc_pipeline import contexts_from_manifest


class _FullMaskSegmenter:
    def __init__(self) -> None:
        self.calls = 0

    def segment_frame(self, frame, queries, config):
        self.calls += 1
        return [
            SimpleNamespace(
                mask=np.ones(frame.shape[:2], dtype=bool),
                category="hand",
            )
        ]


def _qy_context(tmp_path: Path, *, observations=None, camera_selection=None):
    root = tmp_path / "source" / "QY"
    episode = make_qy_episode(root, observations=observations)
    write_test_video(
        episode / "videos" / "mid_cam_left.mp4",
        [solid_frame(value, width=64, height=48) for value in (10, 20, 30, 40)],
        fps=30.0,
    )
    rows = build_qingyu_manifest(
        root,
        primary_camera="mid_cam_left",
        camera_selection=camera_selection,
    )
    manifest = write_qingyu_manifest(rows, tmp_path)
    context = contexts_from_manifest(manifest, batch_root=tmp_path)[0]
    return context


def _write_precheck_gate(context, *, candidate_hand: str = "left") -> None:
    config = load_qc_acceptance_config()
    artifact = artifact_for(context, "precheck")
    artifact.directory.mkdir(parents=True, exist_ok=True)
    (artifact.directory / "candidate_windows.json").write_text(
        json.dumps(
            [
                {
                    "asset_id": context.asset_id,
                    "coordinate_space": "source",
                    "frame_coordinate_system": "source_inclusive",
                    "start_frame": 100,
                    "end_frame": 102,
                    "hand_side": candidate_hand,
                    "sam3_eligible": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    write_run_config(
        artifact.directory,
        producer="precheck",
        outcome="completed",
        fingerprint=precheck_fingerprint(context, config),
        elapsed_seconds=0.0,
        metadata={
            "completed_modules": list(MODULES),
            "temporal_output": {
                "schema_version": "keypoint_temporal.output.v2",
                "status": "valid",
                "valid_frame_count": 2,
                "uncalibrated_frame_count": 1,
                "reason": "calibrated_temporal_output",
            },
        },
    )


def test_qy_sam3_uses_direct_primary_camera_2d_and_explicit_video_frames(
    tmp_path: Path,
) -> None:
    context = _qy_context(tmp_path)
    _write_precheck_gate(context)
    segmenter = _FullMaskSegmenter()

    result = runner(lambda: segmenter)(
        context,
        load_qc_acceptance_config(),
    )

    assert result.module == "sam3_containment"
    rows = json.loads(
        (
            artifact_for(context, "sam3_containment").directory
            / "frame_results.json"
        ).read_text(encoding="utf-8")
    )
    assert [row["source_frame_idx"] for row in rows] == [100, 101, 102]
    assert [row["video_frame_idx"] for row in rows] == [0, 2, 3]
    assert {row["camera_name"] for row in rows} == {"mid_cam_left"}
    assert {row["projection_mode"] for row in rows} == {"qy_direct_2d"}
    assert all(
        name.startswith("qy_left_joint_")
        for name in rows[0]["inside_joint_names"]
    )
    assert rows[0]["joint_topology_status"] == "unverified_points_only"
    assert segmenter.calls == 3


def test_qy_primary_camera_missing_requested_2d_blocks_before_model_load(
    tmp_path: Path,
) -> None:
    # The alternative camera is complete, but the configured primary lacks frame 101.
    rows = observation_rows(cameras=("mid_cam_left",))
    rows = [
        row for row in rows if not (
            row["source_frame_index"] == 101 and row["hand/current_hand"] == "left"
        )
    ]
    rows.extend(observation_rows(cameras=("left_cam_left",)))
    context = _qy_context(
        tmp_path,
        observations=rows,
        camera_selection={
            "minimum_hand_coverage": 0.5,
            "minimum_both_hand_coverage": 0.5,
        },
    )
    # Simulate an explicitly accepted manifest whose selected camera has sparse
    # coverage; SAM3 must still validate every requested sampled frame.
    context = replace(
        context,
        metadata={
            **dict(context.metadata),
            "primary_camera": "mid_cam_left",
            "frame_mapping_status": "verified",
        },
    )
    _write_precheck_gate(context)
    factory_calls = 0

    def factory():
        nonlocal factory_calls
        factory_calls += 1
        return _FullMaskSegmenter()

    with pytest.raises(ModuleBlockedError, match="primary_camera_2d_missing"):
        runner(factory)(context, load_qc_acceptance_config())

    assert factory_calls == 0


def test_qy_unverified_mapping_blocks_without_loading_model(tmp_path: Path) -> None:
    context = _qy_context(tmp_path)
    context = replace(
        context,
        metadata={
            **dict(context.metadata),
            "frame_mapping_status": "mapping_unverified",
        },
    )
    factory_calls = 0

    def factory():
        nonlocal factory_calls
        factory_calls += 1
        return _FullMaskSegmenter()

    with pytest.raises(ModuleBlockedError, match="frame_mapping_unverified"):
        runner(factory)(context, load_qc_acceptance_config())

    assert factory_calls == 0
