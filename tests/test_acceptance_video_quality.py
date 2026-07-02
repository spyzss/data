import csv
import json
from pathlib import Path

import numpy as np
import pytest

from acceptance_pull.video_quality import (
    AlignmentMode,
    analyze_video,
    check_hdf5_alignment,
    discover_batch_videos,
    evaluate_video_quality,
    load_video_quality_config,
    main,
    run_video_quality_check,
)
from tests.fixtures import solid_frame, write_quality_hdf5, write_test_video


def textured_frame(offset: int, width: int = 32, height: int = 24) -> np.ndarray:
    grid = np.indices((height, width)).sum(axis=0)
    gray = ((grid % 2) * 120 + 60 + offset).clip(0, 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def test_default_video_quality_config() -> None:
    config = load_video_quality_config(None)

    assert config.sample_count == 10
    assert config.alignment_mode == AlignmentMode.WARN
    assert config.thresholds.min_fps == 1.0
    assert config.thresholds.min_width == 1
    assert config.thresholds.min_height == 1
    assert config.thresholds.min_sample_decode_ratio == 1.0
    assert config.thresholds.max_mean_over_dark_ratio == 0.10
    assert config.thresholds.max_mean_over_exposed_ratio == 0.05
    assert config.thresholds.max_black_frame_ratio == 0.05
    assert config.thresholds.max_frozen_frame_ratio == 0.8


def test_video_quality_config_yaml_override(tmp_path: Path) -> None:
    path = tmp_path / "quality.yaml"
    path.write_text(
        """
sample_count: 4
alignment_mode: fail
thresholds:
  min_fps: 20
  min_width: 640
  min_height: 480
  max_black_frame_ratio: 0.1
""",
        encoding="utf-8",
    )

    config = load_video_quality_config(path)

    assert config.sample_count == 4
    assert config.alignment_mode == AlignmentMode.FAIL
    assert config.thresholds.min_fps == 20
    assert config.thresholds.min_width == 640
    assert config.thresholds.min_height == 480
    assert config.thresholds.max_black_frame_ratio == 0.1


def test_discover_batch_videos_requires_video_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="video directory not found"):
        discover_batch_videos(tmp_path)


def test_discover_batch_videos_sorts_supported_files(tmp_path: Path) -> None:
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    (video_dir / "b_video.mp4").write_bytes(b"b")
    (video_dir / "a_video.mov").write_bytes(b"a")
    (video_dir / "ignore.txt").write_text("x", encoding="utf-8")

    assert [path.name for path in discover_batch_videos(tmp_path)] == ["a_video.mov", "b_video.mp4"]


def test_analyze_video_reports_metadata_and_sample_metrics(tmp_path: Path) -> None:
    video = tmp_path / "408817_video.mp4"
    frames = [
        solid_frame(80),
        solid_frame(120),
        solid_frame(160),
        solid_frame(200),
    ]
    write_test_video(video, frames, fps=12.0)

    metrics = analyze_video(video, load_video_quality_config(None))

    assert metrics.opened is True
    assert metrics.frame_count == 4
    assert metrics.fps == pytest.approx(12.0, rel=0.1)
    assert metrics.width == 32
    assert metrics.height == 24
    assert metrics.decoded_sample_count >= 1
    assert metrics.sample_decode_ratio > 0
    assert 70 <= metrics.mean_brightness <= 210
    assert metrics.mean_over_dark_ratio < 0.1
    assert metrics.mean_over_exposed_ratio < 0.1


def test_analyze_video_marks_invalid_video_unopened(tmp_path: Path) -> None:
    video = tmp_path / "bad_video.mp4"
    video.write_bytes(b"not a video")

    metrics = analyze_video(video, load_video_quality_config(None))

    assert metrics.opened is False
    assert metrics.frame_count == 0
    assert metrics.decoded_sample_count == 0
    assert "cannot_open_video" in metrics.errors


def test_analyze_video_detects_black_and_frozen_samples(tmp_path: Path) -> None:
    video = tmp_path / "408817_video.mp4"
    write_test_video(video, [solid_frame(0) for _ in range(6)], fps=10.0)

    metrics = analyze_video(video, load_video_quality_config(None))

    assert metrics.black_frame_ratio >= 0.9
    assert metrics.frozen_frame_ratio >= 0.8


def test_evaluate_video_quality_passes_good_metrics(tmp_path: Path) -> None:
    video = tmp_path / "408817_video.mp4"
    write_test_video(video, [textured_frame(0), textured_frame(10), textured_frame(20)], fps=10.0)
    metrics = analyze_video(video, load_video_quality_config(None))

    evaluation = evaluate_video_quality(metrics, load_video_quality_config(None))

    assert evaluation.passed is True
    assert evaluation.reasons == ()


def test_evaluate_video_quality_fails_decode_and_threshold_reasons(tmp_path: Path) -> None:
    video = tmp_path / "bad_video.mp4"
    video.write_bytes(b"not a video")
    metrics = analyze_video(video, load_video_quality_config(None))

    evaluation = evaluate_video_quality(metrics, load_video_quality_config(None))

    assert evaluation.passed is False
    assert "cannot_open_video" in evaluation.reasons
    assert "sample_decode_ratio_below_min" in evaluation.reasons


def test_evaluate_video_quality_fails_black_and_frozen_video(tmp_path: Path) -> None:
    video = tmp_path / "408817_video.mp4"
    write_test_video(video, [solid_frame(0) for _ in range(6)], fps=10.0)
    metrics = analyze_video(video, load_video_quality_config(None))

    evaluation = evaluate_video_quality(metrics, load_video_quality_config(None))

    assert evaluation.passed is False
    assert "black_frame_ratio_above_max" in evaluation.reasons
    assert "frozen_frame_ratio_above_max" in evaluation.reasons


def test_check_hdf5_alignment_passes_matching_frame_count(tmp_path: Path) -> None:
    batch = tmp_path
    video_dir = batch / "video"
    video_dir.mkdir()
    video = video_dir / "408817_video.mp4"
    write_test_video(video, [solid_frame(100) for _ in range(4)], fps=10.0)
    write_quality_hdf5(batch / "hdf5" / "408817_hdf5.hdf5", 4)
    metrics = analyze_video(video, load_video_quality_config(None))

    alignment = check_hdf5_alignment(video, batch, metrics, load_video_quality_config(None))

    assert alignment.status == "matched"
    assert alignment.hdf5_frame_count == 4
    assert alignment.frame_count_match is True


def test_hdf5_alignment_mismatch_fails_evaluation(tmp_path: Path) -> None:
    batch = tmp_path
    video_dir = batch / "video"
    video_dir.mkdir()
    video = video_dir / "408817_video.mp4"
    write_test_video(video, [solid_frame(100) for _ in range(4)], fps=10.0)
    write_quality_hdf5(batch / "hdf5" / "408817_hdf5.hdf5", 5)
    config = load_video_quality_config(None)
    metrics = analyze_video(video, config)
    alignment = check_hdf5_alignment(video, batch, metrics, config)

    evaluation = evaluate_video_quality(metrics, config, alignment)

    assert alignment.status == "mismatch"
    assert alignment.frame_count_match is False
    assert evaluation.passed is False
    assert "hdf5_frame_count_mismatch" in evaluation.reasons


def test_hdf5_alignment_missing_warn_does_not_fail(tmp_path: Path) -> None:
    batch = tmp_path
    video_dir = batch / "video"
    video_dir.mkdir()
    video = video_dir / "408817_video.mp4"
    write_test_video(video, [solid_frame(100) for _ in range(4)], fps=10.0)
    config = load_video_quality_config(None)
    metrics = analyze_video(video, config)

    alignment = check_hdf5_alignment(video, batch, metrics, config)
    evaluation = evaluate_video_quality(metrics, config, alignment)

    assert alignment.status == "missing"
    assert "hdf5_missing" not in evaluation.reasons


def test_run_video_quality_check_writes_reports_and_returns_zero(tmp_path: Path) -> None:
    batch = tmp_path
    video_dir = batch / "video"
    video_dir.mkdir()
    video = video_dir / "408817_video.mp4"
    write_test_video(video, [textured_frame(0), textured_frame(10), textured_frame(20)], fps=10.0)
    write_quality_hdf5(batch / "hdf5" / "408817_hdf5.hdf5", 3)

    exit_code = run_video_quality_check(batch)

    assert exit_code == 0
    csv_path = batch / "reports" / "video_quality.csv"
    summary_path = batch / "reports" / "video_quality_summary.json"
    assert csv_path.is_file()
    assert summary_path.is_file()

    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    assert rows[0]["asset_id"] == "408817"
    assert rows[0]["passed"] == "true"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["batch_status"] == "passed"
    assert summary["total_videos"] == 1
    assert summary["failed_videos"] == 0
    assert summary["effective_config"]["sample_count"] == 10


def test_run_video_quality_check_returns_nonzero_for_failed_video(tmp_path: Path) -> None:
    batch = tmp_path
    video_dir = batch / "video"
    video_dir.mkdir()
    (video_dir / "bad_video.mp4").write_bytes(b"not a video")

    exit_code = run_video_quality_check(batch)

    assert exit_code == 2
    summary = json.loads((batch / "reports" / "video_quality_summary.json").read_text(encoding="utf-8"))
    assert summary["batch_status"] == "failed"
    assert summary["failed_videos"] == 1
    assert summary["reason_counts"]["cannot_open_video"] == 1


def test_video_quality_main_accepts_config_and_reports_dir(tmp_path: Path) -> None:
    batch = tmp_path / "batch"
    video_dir = batch / "video"
    reports = tmp_path / "custom-reports"
    video_dir.mkdir(parents=True)
    write_test_video(video_dir / "408817_video.mp4", [textured_frame(0), textured_frame(10)], fps=10.0)
    config = tmp_path / "quality.yaml"
    config.write_text("sample_count: 2\nalignment_mode: ignore\n", encoding="utf-8")

    exit_code = main(["--batch", str(batch), "--config", str(config), "--reports-dir", str(reports)])

    assert exit_code == 0
    assert (reports / "video_quality.csv").is_file()
    assert (reports / "video_quality_summary.json").is_file()
