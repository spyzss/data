from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tests.fixtures import solid_frame, write_test_video


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _result_rows(output_dir: Path) -> list[dict[str, object]]:
    return json.loads(
        (output_dir / "video_quality_results.json").read_text(encoding="utf-8")
    )


@pytest.mark.parametrize("extension", [".parquet", ".jsonl"])
def test_manifest_video_quality_reads_supported_manifest_formats(
    tmp_path: Path, extension: str
) -> None:
    from tools.run_manifest_video_quality import read_manifest

    path = tmp_path / f"manifest{extension}"
    rows = [{"asset_id": "clip-a", "start_frame": 1, "end_frame": 2}]
    if extension == ".parquet":
        pd.DataFrame(rows).to_parquet(path, index=False)
    else:
        path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")

    assert read_manifest(path) == rows


def test_manifest_video_quality_uses_inclusive_ranges_and_source_coordinates(
    tmp_path: Path,
) -> None:
    from tools.run_manifest_video_quality import run_manifest_video_quality

    video = tmp_path / "source.mp4"
    write_test_video(video, [solid_frame(90) for _ in range(16)], fps=10.0)
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [
            {
                "asset_id": "logical-a",
                "primary_video_path": str(video),
                "start_frame": 2,
                "end_frame": 13,
            }
        ],
    )

    run_manifest_video_quality(manifest, tmp_path / "quality")

    rows = _result_rows(tmp_path / "quality")
    assert len(rows) == 1
    row = rows[0]
    assert row["asset_id"] == "logical-a"
    assert row["source_video_path"] == str(video)
    assert row["clip_start_frame"] == 2
    assert row["clip_end_frame"] == 13
    assert row["clip_frame_count"] == 12
    assert row["decoded_frame_count"] == 12
    assert row["source_video_frame_count"] == 16
    assert row["sampled_frame_mappings"] == [
        {
            "local_frame_idx": local_frame_idx,
            "source_frame_idx": 2 + local_frame_idx,
        }
        for local_frame_idx in row["sampled_local_frame_indices"]
    ]
    intervals = row["video_quality"]["metrics"]["freeze_metrics"][
        "frozen_intervals"
    ]
    assert intervals[0]["local_start_frame"] == 0
    assert intervals[0]["local_end_frame"] == 11
    assert intervals[0]["start_frame"] == 2
    assert intervals[0]["end_frame"] == 13
    assert intervals[0]["source_start_frame"] == 2
    assert intervals[0]["source_end_frame"] == 13
    assert (tmp_path / "quality" / "video_quality_results.parquet").exists()
    assert (tmp_path / "quality" / "video_quality_decision_summary.csv").exists()
    assert (tmp_path / "quality" / "video_quality_failures.json").exists()
    run_config = json.loads(
        (tmp_path / "quality" / "run_config.json").read_text(encoding="utf-8")
    )
    assert run_config["frame_range_semantics"] == "inclusive_source_frames"
    assert not list((tmp_path / "quality").glob("*.mp4"))


def test_manifest_video_quality_handles_repeated_source_ranges_independently(
    tmp_path: Path,
) -> None:
    from tools.run_manifest_video_quality import run_manifest_video_quality

    video = tmp_path / "shared.mp4"
    write_test_video(
        video,
        [solid_frame(40 + frame_idx * 10) for frame_idx in range(10)],
        fps=10.0,
    )
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [
            {
                "asset_id": "left-range",
                "primary_video_path": str(video),
                "start_frame": 0,
                "end_frame": 2,
            },
            {
                "asset_id": "right-range",
                "primary_video_path": str(video),
                "start_frame": 6,
                "end_frame": 9,
            },
        ],
    )

    run_manifest_video_quality(manifest, tmp_path / "quality")

    rows = {row["asset_id"]: row for row in _result_rows(tmp_path / "quality")}
    assert rows["left-range"]["decoded_frame_count"] == 3
    assert rows["left-range"]["clip_frame_count"] == 3
    assert rows["right-range"]["decoded_frame_count"] == 4
    assert rows["right-range"]["clip_frame_count"] == 4


def test_manifest_video_quality_isolates_invalid_ranges(tmp_path: Path) -> None:
    from tools.run_manifest_video_quality import run_manifest_video_quality

    video = tmp_path / "source.mp4"
    write_test_video(video, [solid_frame(100) for _ in range(5)], fps=10.0)
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [
            {
                "asset_id": "valid",
                "primary_video_path": str(video),
                "start_frame": 1,
                "end_frame": 3,
            },
            {
                "asset_id": "invalid",
                "primary_video_path": str(video),
                "start_frame": 4,
                "end_frame": 9,
            },
        ],
    )

    summary = run_manifest_video_quality(manifest, tmp_path / "quality")

    assert summary["completed_clip_count"] == 1
    assert summary["failed_clip_count"] == 1
    assert [row["asset_id"] for row in _result_rows(tmp_path / "quality")] == [
        "valid"
    ]
    failures = json.loads(
        (tmp_path / "quality" / "video_quality_failures.json").read_text(
            encoding="utf-8"
        )
    )
    assert failures[0]["asset_id"] == "invalid"
    assert failures[0]["source_video_path"] == str(video)
    assert failures[0]["clip_start_frame"] == 4
    assert failures[0]["clip_end_frame"] == 9
    assert "outside source frame count" in failures[0]["error"]


def test_manifest_video_quality_dry_run_writes_no_producer_outputs(
    tmp_path: Path,
) -> None:
    from tools.run_manifest_video_quality import run_manifest_video_quality

    video = tmp_path / "source.mp4"
    write_test_video(video, [solid_frame(100) for _ in range(5)], fps=10.0)
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [
            {
                "asset_id": "valid",
                "primary_video_path": str(video),
                "start_frame": 0,
                "end_frame": 4,
            }
        ],
    )
    output_dir = tmp_path / "quality"

    summary = run_manifest_video_quality(manifest, output_dir, dry_run=True)

    assert summary["validated_clip_count"] == 1
    assert not (output_dir / "video_quality_results.json").exists()
    assert not (output_dir / "video_quality_results.parquet").exists()
    assert not (output_dir / "video_quality_decision_summary.csv").exists()
    assert not (output_dir / "run_config.json").exists()


def test_manifest_video_quality_skips_completed_assets_unless_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.run_manifest_video_quality as runner

    video = tmp_path / "source.mp4"
    write_test_video(video, [solid_frame(100) for _ in range(5)], fps=10.0)
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [
            {
                "asset_id": "clip-a",
                "primary_video_path": str(video),
                "start_frame": 0,
                "end_frame": 4,
            }
        ],
    )
    output_dir = tmp_path / "quality"
    runner.run_manifest_video_quality(manifest, output_dir)

    calls = 0
    original = runner.analyze_video_frame_range

    def counted(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "analyze_video_frame_range", counted)
    skipped = runner.run_manifest_video_quality(manifest, output_dir)
    assert calls == 0
    assert skipped["skipped_clip_count"] == 1

    runner.run_manifest_video_quality(manifest, output_dir, overwrite=True)
    assert calls == 1
