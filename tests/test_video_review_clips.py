from pathlib import Path

import pandas as pd
import pytest

from tools.build_video_review_clips import (
    build_clip_rows,
    build_review_index_video_html,
    compute_clip_timing,
)


def test_compute_clip_timing_for_frame_window_clamps_to_asset_duration() -> None:
    timing = compute_clip_timing(
        window_start_frame=120,
        window_end_frame=180,
        representative_frame=None,
        fps=30.0,
        asset_duration_sec=5.0,
        padding_sec=1.0,
    )

    assert timing["start_time_sec"] == pytest.approx(3.0)
    assert timing["duration_sec"] == pytest.approx(2.0)
    assert timing["error"] == ""


def test_compute_clip_timing_for_asset_level_uses_representative_frame() -> None:
    timing = compute_clip_timing(
        window_start_frame=None,
        window_end_frame=None,
        representative_frame=90,
        fps=30.0,
        asset_duration_sec=20.0,
        padding_sec=1.0,
    )

    assert timing["start_time_sec"] == pytest.approx(0.5)
    assert timing["duration_sec"] == pytest.approx(5.0)
    assert timing["error"] == ""


def test_build_clip_rows_joins_manifest_and_review_queue(tmp_path: Path) -> None:
    video_path = tmp_path / "100030_video.mp4"
    manifest = pd.DataFrame(
        [
            {
                "supplier_id": "supplier_a",
                "asset_id": "100030",
                "video_path": str(video_path),
                "fps": 30.0,
                "frame_count": 300,
                "duration_sec": 10.0,
            }
        ]
    )
    review_queue = pd.DataFrame(
        [
            {
                "review_id": "rq_001",
                "supplier_id": "supplier_a",
                "asset_id": "100030",
                "window_start_frame": 120,
                "window_end_frame": 150,
                "representative_frame": 135,
                "auto_verdict": "review",
                "suggested_issue_type": "keypoint_low_quality_window",
                "severity_suggestion": "medium",
                "key_metrics_json": "{\"flagged_frames\": 31}",
                "reason": "test row",
            }
        ]
    )

    rows = build_clip_rows(manifest, review_queue, output_dir=tmp_path, padding_sec=1.0)

    assert len(rows) == 1
    row = rows[0]
    assert row["video_path"] == str(video_path)
    assert row["clip_path"] == str(tmp_path / "clips" / "rq_001.mp4")
    assert row["display_clip_path"] == "clips/rq_001.mp4"
    assert row["clip_start_time_sec"] == pytest.approx(3.0)
    assert row["clip_duration_sec"] == pytest.approx((150 - 120 + 1) / 30.0 + 2.0)
    assert row["clip_error"] == ""


def test_video_review_html_contains_video_and_export_button() -> None:
    rows = [
        {
            "review_id": "rq_001",
            "supplier_id": "supplier_a",
            "asset_id": "100030",
            "window_start_frame": 120,
            "window_end_frame": 150,
            "representative_frame": 135,
            "auto_verdict": "review",
            "suggested_issue_type": "keypoint_low_quality_window",
            "severity_suggestion": "medium",
            "key_metrics_json": "{\"flagged_frames\": 31}",
            "reason": "test row",
            "display_clip_path": "clips/rq_001.mp4",
            "clip_start_time_sec": 3.0,
            "clip_duration_sec": 3.033333,
            "clip_error": "",
        }
    ]

    html = build_review_index_video_html(rows)

    assert "<video" in html
    assert "controls" in html
    assert "Export manual_labels.csv" in html
    assert "function exportManualLabelsCsv" in html
