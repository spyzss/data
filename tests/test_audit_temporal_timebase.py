from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from qc_common.types import ClipInputs
from tools.audit_temporal_timebase import audit_temporal_timebase


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
