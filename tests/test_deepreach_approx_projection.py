from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from tests.fixtures import solid_frame, write_test_video


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    hdf5 = tmp_path / "task-a.h5"
    with h5py.File(hdf5, "w") as handle:
        handle.create_dataset("timestamp", data=np.arange(5, dtype=np.float64) / 30.0)
        handle.attrs["coordinate_frame"] = "head_camera"
        handle.attrs["units"] = "meters"
        for side, x in (("left", -0.12), ("right", 0.12)):
            group = handle.create_group(f"hand/{side}")
            group.create_dataset("valid", data=np.ones(5, dtype=np.uint8))
            points = np.zeros((5, 21, 3), dtype=np.float32)
            points[..., 0] = x
            points[..., 2] = 1.0
            group.create_dataset("joints3d", data=points)

    video = tmp_path / "head.mp4"
    write_test_video(
        video,
        [solid_frame(value, width=160, height=90) for value in (20, 40, 60, 80, 100)],
        fps=30.0,
    )
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "supplier",
                "asset_id",
                "task_name",
                "start_frame",
                "end_frame",
                "hdf5_path",
                "hdf5_reference_dataset",
                "head_video_path",
                "primary_video_path",
                "primary_camera",
                "calib_path",
                "camera_trajectory_path",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "supplier": "dr",
                "asset_id": "dr__task-a",
                "task_name": "task-a",
                "start_frame": 100,
                "end_frame": 104,
                "hdf5_path": hdf5,
                "hdf5_reference_dataset": "timestamp",
                "head_video_path": video,
                "primary_video_path": video,
                "primary_camera": "head",
                # These are intentionally unusable. Approximate head projection
                # must never read wrist calibration or camera trajectory.
                "calib_path": tmp_path / "wrist_calibration.json",
                "camera_trajectory_path": tmp_path / "trajectory.csv",
            }
        )
    candidates = tmp_path / "candidate_windows.json"
    candidates.write_text(
        json.dumps(
            [
                {
                    "asset_id": "dr__task-a",
                    "start_frame": 100,
                    "end_frame": 104,
                    "peak_frame": 103,
                    "sam3_eligible": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    return manifest, candidates


def test_hfov_pinhole_intrinsics_are_explicitly_heuristic() -> None:
    from acceptance_pull.supplier_adapters.deepreach_projection import (
        approximate_head_calibration,
    )

    calibration = approximate_head_calibration(
        width=1920,
        height=1080,
        horizontal_fov_deg=100.0,
    )

    expected_focal = 1920.0 / (2.0 * math.tan(math.radians(50.0)))
    assert calibration.status == "heuristic"
    assert calibration.resolution == (1920, 1080)
    assert calibration.intrinsics is not None
    assert calibration.intrinsics[0, 0] == pytest.approx(expected_focal)
    assert calibration.intrinsics[1, 1] == pytest.approx(expected_focal)
    assert calibration.intrinsics[0, 2] == pytest.approx(960.0)
    assert calibration.intrinsics[1, 2] == pytest.approx(540.0)
    assert calibration.reason == "pending_visual_validation"


def test_approximate_audit_writes_bounded_fov_comparison_without_mapping(
    tmp_path: Path,
) -> None:
    from tools.audit_deepreach_projection import (
        run_approximate_head_projection_audit,
    )

    manifest, candidates = _fixture(tmp_path)
    manifest_before = manifest.read_bytes()
    output = tmp_path / "audit"
    summary = run_approximate_head_projection_audit(
        manifest=manifest,
        output_dir=output,
        candidate_hfov_deg=(80.0, 120.0),
        candidate_windows=(candidates,),
        asset_ids=("dr__task-a",),
        max_assets=1,
    )

    payload = json.loads(
        (output / "dr_approximate_head_projection_audit.json").read_text()
    )
    rows = pd.read_csv(output / "dr_approximate_head_projection_audit.csv")
    assert summary == {"asset_count": 1, "candidate_count": 2, "contact_sheet_count": 1}
    assert payload["schema_version"] == "dr_approximate_head_projection_audit.v1"
    assert payload["projection_mode"] == "approx_pinhole_from_hfov"
    assert payload["calibration_status"] == "heuristic"
    assert payload["projection_validation_status"] == "pending_visual_validation"
    assert payload["distortion_applied"] is False
    assert payload["camera_trajectory_applied"] is False
    assert rows["head_hfov_deg"].tolist() == [80.0, 120.0]
    assert set(rows["sampled_source_frames"]) == {"[100, 102, 103, 104]"}
    assert set(rows["positive_z_ratio"]) == {1.0}
    assert set(rows["finite_projection_ratio"]) == {1.0}
    assert set(rows["projection_mode"]) == {"approx_pinhole_from_hfov"}
    assert set(rows["calibration_status"]) == {"heuristic"}
    assert set(rows["projection_validation_status"]) == {"pending_visual_validation"}
    assert set(rows["distortion_applied"]) == {False}
    contact_sheet = output / payload["assets"][0]["contact_sheet_path"]
    assert contact_sheet.is_file()
    assert manifest.read_bytes() == manifest_before


def test_heuristic_mapping_export_requires_explicit_user_selection(tmp_path: Path) -> None:
    from tools.audit_deepreach_projection import run_approximate_head_projection_audit
    from tools.export_deepreach_heuristic_head_projection_mapping import (
        export_heuristic_mapping,
    )

    manifest, candidates = _fixture(tmp_path)
    output = tmp_path / "audit"
    run_approximate_head_projection_audit(
        manifest=manifest,
        output_dir=output,
        candidate_hfov_deg=(80.0, 120.0),
        candidate_windows=(candidates,),
        max_assets=1,
    )
    audit = output / "dr_approximate_head_projection_audit.json"

    with pytest.raises(ValueError, match="explicit --select"):
        export_heuristic_mapping(audit=audit, output=tmp_path / "missing.csv", selections={})

    target = tmp_path / "deepreach_heuristic_head_projection_mapping.csv"
    export_heuristic_mapping(
        audit=audit,
        output=target,
        selections={"dr__task-a": 120.0},
    )
    row = next(csv.DictReader(target.open(newline="", encoding="utf-8")))
    assert row["asset_id"] == "dr__task-a"
    assert row["projection_mode"] == "approx_pinhole_from_hfov"
    assert row["head_hfov_deg"] == "120.0"
    assert row["calibration_status"] == "heuristic"
    assert row["projection_validation_status"] == "pending_visual_validation"
    assert row["distortion_applied"] == "false"
    assert row["camera_trajectory_applied"] == "false"
    assert row["selected_by"] == "user_visual_confirmation"
    assert "verified" not in row.values()


def test_projection_audit_cli_accepts_explicit_hfov_mode_without_calibration_mapping(
    tmp_path: Path,
) -> None:
    from tools.audit_deepreach_projection import build_parser

    args = build_parser().parse_args(
        [
            "--manifest",
            str(tmp_path / "manifest.csv"),
            "--output-dir",
            str(tmp_path / "audit"),
            "--projection-mode",
            "approx-pinhole-from-hfov",
            "--candidate-hfov-deg",
            "80,100,120,140",
            "--candidate-windows",
            str(tmp_path / "candidate_windows.json"),
            "--asset-id",
            "dr__task-a",
            "--max-assets",
            "1",
        ]
    )

    assert args.mapping_config is None
    assert args.projection_mode == "approx-pinhole-from-hfov"
    assert args.candidate_hfov_deg == "80,100,120,140"
    assert args.candidate_windows == [tmp_path / "candidate_windows.json"]
