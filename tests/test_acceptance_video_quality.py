import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import h5py
import yaml

import acceptance_pull.video_quality as video_quality
from acceptance_pull.video_quality import (
    AlignmentMode,
    FrozenInterval,
    analyze_video,
    check_hdf5_alignment,
    discover_batch_videos,
    evaluate_video_quality,
    load_video_quality_config,
    main,
    run_video_quality_check,
)
from qc_common.config import load_qc_acceptance_config
from qc_common.contracts import ModuleResult
from qc_common.report_mutation import apply_module_result
from qc_pipeline.context import AssetContext
from tests.fixtures import solid_frame, write_hand_keypoint_hdf5, write_quality_hdf5, write_test_video
from tests.fixtures import write_quality_hdf5_with_text


def textured_frame(offset: int, width: int = 1280, height: int = 720) -> np.ndarray:
    y, x = np.indices((height, width))
    gray = (80 + ((x * 7 + y * 5 + offset) % 140)).astype(np.uint8)
    gray[:, width // 3 : width // 3 + 2] = 20
    gray[:, 2 * width // 3 : 2 * width // 3 + 2] = 235
    return np.repeat(gray[:, :, None], 3, axis=2)


def slow_motion_frame(index: int, width: int = 128, height: int = 128) -> np.ndarray:
    frame = np.full((height, width, 3), 120, dtype=np.uint8)
    left = min(width - 34, 4 + index)
    frame[44:84, left : left + 30] = 205
    return frame


def write_unified_video_config(tmp_path: Path, mutate: Callable[[dict[str, Any]], None]) -> Path:
    raw = yaml.safe_load(Path("configs/qc_acceptance.yaml").read_text(encoding="utf-8"))
    raw["config_version"] = "qc_acceptance_v1.1.1"
    mutate(raw["modules"]["video_quality"]["parameters"])
    path = tmp_path / "qc_acceptance.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def advance_batch_report_to_video(
    batch: Path,
    asset_id: str,
    *,
    config_path: Path | None = None,
    manifest_metadata: dict[str, object] | None = None,
) -> None:
    config = load_qc_acceptance_config(config_path)
    video_path = batch / "video" / f"{asset_id}_video.mp4"
    hdf5_path = batch / "hdf5" / f"{asset_id}_hdf5.hdf5"
    source_files: dict[str, Any] = {
        "video": {"path": video_path.relative_to(batch).as_posix()}
    }
    if hdf5_path.exists():
        source_files["hdf5"] = {
            "path": hdf5_path.relative_to(batch).as_posix()
        }
    context = AssetContext(
        asset_id=asset_id,
        batch_root=batch,
        report_path=batch / "quality_archive" / f"{asset_id}.json",
        source_files=source_files,
        metadata=manifest_metadata or {},
    )
    revision = 0
    modules = config.pipeline_modules
    for index, module in enumerate(modules[: modules.index("video_quality")]):
        report = apply_module_result(
            context.report_path,
            context=context,
            config=config,
            profile="acceptance",
            result=ModuleResult(module, "pass", {}, {}),
            expected_revision=revision,
            next_module=modules[index + 1],
            now=f"2026-07-14T00:00:{index:02d}Z",
        )
        revision = report["report_revision"]


def test_default_video_quality_config_comes_from_unified_config() -> None:
    config = load_video_quality_config(None)

    assert config.module_version == "video_prefilter_v0.3.2"
    assert config.qc_config_reference["config_version"] == "qc_acceptance_v2.3.0"
    assert not hasattr(config, "threshold_version")
    assert config.pipeline.stop_before_mask_if_fail is True
    assert config.pipeline.do_keypoint_quality_check is False
    assert config.fps.expected_fps is None
    assert config.fps.min_fps_pass == 24
    assert config.fps.min_fps_warn == 20
    assert config.resolution.min_short_side_fail == 720
    assert config.resolution.min_long_side_fail == 1280
    assert config.decode.sample_decode_ratio_pass == 0.995
    assert config.decode.sample_decode_ratio_warn == 0.98
    assert config.decode.max_sample_frames == 300
    assert config.exposure.black.ratio_pass == 0.01
    assert config.exposure.black.ratio_warn == 0.90
    assert config.exposure.black.max_frame_count_fail == 10
    assert config.exposure.over_dark.ratio_pass == 0.05
    assert config.exposure.over_dark.ratio_warn == 0.90
    assert config.exposure.over_exposed.ratio_pass == 0.05
    assert config.exposure.over_exposed.ratio_warn == 0.90
    assert config.sharpness_global.target_short_side == 720
    assert config.sharpness_global.laplacian_p10_pass == 15
    assert config.sharpness_global.laplacian_p10_warn == 0
    assert config.sharpness_global.laplacian_median_pass == 20
    assert config.sharpness_global.laplacian_median_warn == 0
    assert config.sharpness_global.laplacian_under_100_ratio_pass == 1.00
    assert config.sharpness_global.laplacian_under_100_ratio_warn == 1.00
    assert config.sharpness_global.tenengrad_p10_pass == 6
    assert config.sharpness_global.tenengrad_p10_warn == 4
    assert config.sharpness_global.tenengrad_median_pass == 7
    assert config.sharpness_global.tenengrad_median_warn == 4
    assert config.timeline.drop_frame_ratio_pass == 0.05
    assert config.timeline.drop_frame_ratio_warn == 0.10
    assert config.freeze.frozen_frame_ratio_pass == 0.05
    assert config.freeze.frozen_frame_ratio_warn == 0.10
    assert config.freeze.min_interval_frames == 6
    assert config.freeze.min_interval_duration_ms == 100
    assert config.freeze.freeze_candidate_window_sec == 0.5
    assert config.freeze.confirmed_freeze_window_sec == 1.0
    assert config.freeze.adjacent_near_duplicate_ratio_warn == 0.90
    assert config.freeze.ssim_min == 0.995
    assert config.freeze.phash_hamming_max == 4
    assert config.freeze.motion_conflict_enabled is True
    assert config.freeze.critical_window_enabled is True
    assert config.freeze.video_state_conflict_noncritical_duration_ms_fail == 1000
    assert config.freeze.video_state_conflict_critical_duration_ms_fail == 500
    assert config.freeze.max_consecutive_frozen_sec_fail == 1.0
    assert config.defects.max_duration_ratio_fail == 0.10
    assert config.defects.duration_ratio_warn == 0.05
    assert config.hdf5_alignment.mode == AlignmentMode.FAIL
    assert config.hdf5_alignment.max_delta_frames_pass == 2


def test_unified_video_threshold_override_changes_runtime_config(tmp_path: Path) -> None:
    def mutate(parameters: dict[str, Any]) -> None:
        parameters["fps"]["min_fps_pass"] = 26
        parameters["decode"]["max_sample_frames"] = 4
        parameters["hdf5_alignment"]["mode"] = "warn"

    path = write_unified_video_config(tmp_path, mutate)
    config = load_video_quality_config(path)

    assert config.fps.min_fps_pass == 26
    assert config.decode.max_sample_frames == 4
    assert config.hdf5_alignment.mode == AlignmentMode.WARN
    assert config.qc_config_reference["config_version"] == "qc_acceptance_v1.1.1"


def test_unified_config_missing_video_parameters_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/qc_acceptance.yaml").read_text(encoding="utf-8"))
    del raw["modules"]["video_quality"]["parameters"]["freeze"]
    path = tmp_path / "missing-freeze.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="freeze"):
        load_video_quality_config(path)


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
    assert metrics.short_side == 24
    assert metrics.long_side == 32
    assert metrics.decoded_sample_count >= 1
    assert metrics.sample_decode_ratio > 0
    assert 70 <= metrics.mean_brightness <= 210
    assert metrics.mean_over_dark_ratio < 0.1
    assert metrics.mean_over_exposed_ratio < 0.1


def test_analyze_video_reports_clear_screen_sharpness_distribution(tmp_path: Path) -> None:
    video = tmp_path / "408817_video.mp4"
    write_test_video(video, [textured_frame(0), textured_frame(10), textured_frame(20)], fps=12.0)

    metrics = analyze_video(video, load_video_quality_config(None))

    assert metrics.laplacian_p10 >= 300
    assert metrics.laplacian_median >= 450
    assert metrics.laplacian_under_100_ratio == 0
    assert metrics.tenengrad_p10 >= 30
    assert metrics.tenengrad_median >= 35
    assert metrics.sharpness_scale_short_side == 720


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
    write_test_video(video, [solid_frame(0) for _ in range(16)], fps=10.0)

    metrics = analyze_video(video, load_video_quality_config(None))

    assert metrics.black_frame_ratio >= 0.9
    assert metrics.frozen_frame_ratio >= 0.8


def test_analyze_video_records_frozen_intervals_over_five_frames(tmp_path: Path) -> None:
    video = tmp_path / "408817_video.mp4"
    frames = [solid_frame(80) for _ in range(16)]
    frames.extend(textured_frame(index, width=32, height=24) for index in range(6))
    frames.extend(solid_frame(200) for _ in range(16))
    write_test_video(video, frames, fps=10.0)

    metrics = analyze_video(video, load_video_quality_config(None))

    assert metrics.adjacent_near_duplicate_ratio > 0
    assert metrics.freeze_candidate_ratio > 0
    assert metrics.confirmed_freeze_ratio == metrics.frozen_frame_ratio
    assert [interval.frame_count for interval in metrics.frozen_intervals] == [16, 16]
    assert metrics.frozen_intervals[0].start_frame == 0
    assert metrics.frozen_intervals[0].end_frame == 15
    assert metrics.frozen_intervals[0].start_time_sec == 0.0
    assert metrics.frozen_intervals[0].end_time_sec == 1.6
    assert metrics.frozen_intervals[0].duration_sec == 1.6
    assert metrics.frozen_intervals[0].duration_ms == 1600.0
    assert metrics.frozen_intervals[0].mean_ssim >= 0.995
    assert metrics.frozen_intervals[0].max_phash_hamming <= 4
    assert metrics.frozen_intervals[0].motion_conflict is False
    assert metrics.frozen_intervals[0].critical_window is False
    assert metrics.frozen_intervals[1].start_frame == 22
    assert metrics.frozen_intervals[1].end_frame == 37


def test_adjacent_near_duplicates_are_low_motion_not_rejection(tmp_path: Path) -> None:
    video = tmp_path / "408817_video.mp4"
    write_test_video(
        video,
        [
            slow_motion_frame(index // 20, width=256, height=128)
            for index in range(120)
        ],
        fps=60.0,
    )
    metrics = analyze_video(video, load_video_quality_config(None))
    metrics = replace(
        metrics,
        short_side=720,
        long_side=1280,
        laplacian_p10=100.0,
        laplacian_median=120.0,
        laplacian_under_100_ratio=0.0,
        tenengrad_p10=20.0,
        tenengrad_median=22.0,
        sample_decode_ratio=1.0,
        black_frame_ratio=0.0,
        black_frame_count_estimate=0,
        mean_over_dark_ratio=0.0,
        mean_over_exposed_ratio=0.0,
        exposure_defect_frame_ratio=0.0,
        adjacent_near_duplicate_count=114,
        adjacent_near_duplicate_ratio=0.95,
        freeze_candidate_frame_count=0,
        freeze_candidate_duration_sec=0.0,
        freeze_candidate_ratio=0.0,
        confirmed_freeze_frame_count=0,
        confirmed_freeze_duration_sec=0.0,
        confirmed_freeze_ratio=0.0,
        frozen_frame_ratio=0.0,
        max_consecutive_frozen_sec=0.0,
        frozen_intervals=(),
        defect_duration_ratio=metrics.drop_frame_ratio,
    )

    config = load_video_quality_config(None)
    evaluation = evaluate_video_quality(metrics, config)

    assert metrics.adjacent_near_duplicate_ratio > 0.90
    assert metrics.freeze_candidate_ratio == 0.0
    assert metrics.confirmed_freeze_ratio == 0.0
    assert metrics.frozen_frame_ratio == 0.0
    assert evaluation.passed is True
    assert evaluation.decision == "warn"
    assert "adjacent_near_duplicate_ratio_warn" in evaluation.warn_reasons
    assert "frozen_frame_ratio_above_max" not in evaluation.reasons


def test_half_second_near_duplicate_is_candidate_not_confirmed_freeze(tmp_path: Path) -> None:
    video = tmp_path / "408817_video.mp4"
    frames = [solid_frame(120) for _ in range(8)]
    frames.extend(textured_frame(index, width=32, height=24) for index in range(12))
    write_test_video(video, frames, fps=10.0)

    metrics = analyze_video(video, load_video_quality_config(None))

    assert metrics.freeze_candidate_ratio > 0.0
    assert metrics.confirmed_freeze_ratio == 0.0
    assert metrics.frozen_intervals == ()


def test_timeline_metrics_prefers_ffprobe_pts_and_estimates_missing_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_video_quality_config(None)

    monkeypatch.setattr(
        video_quality,
        "_read_frame_timestamps_ffprobe",
        lambda _path: video_quality.FrameTimestamps(
            timestamps_ms=(0.0, 33.333, 66.666, 166.666),
            source="ffprobe",
            reliable=True,
        ),
    )

    timeline = video_quality._timeline_metrics(Path("missing.mp4"), frame_count=5, fps=30.0, config=config.timeline)

    assert timeline["drop_detection_source"] == "ffprobe"
    assert timeline["drop_detection_reliable"] is True
    assert timeline["estimated_missing_frames"] == 2
    assert timeline["drop_frame_ratio"] == pytest.approx(2 / 7)


def test_timeline_metrics_marks_opencv_fallback_unreliable(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_video_quality_config(None)

    monkeypatch.setattr(video_quality, "_read_frame_timestamps_ffprobe", lambda _path: None)
    monkeypatch.setattr(video_quality, "_read_frame_timestamps_pyav", lambda _path: None)
    monkeypatch.setattr(
        video_quality,
        "_read_frame_timestamps_opencv",
        lambda _path, _frame_count: video_quality.FrameTimestamps(
            timestamps_ms=(0.0, 33.333, 66.666),
            source="opencv",
            reliable=False,
        ),
    )

    timeline = video_quality._timeline_metrics(Path("missing.mp4"), frame_count=3, fps=30.0, config=config.timeline)

    assert timeline["drop_detection_source"] == "opencv"
    assert timeline["drop_detection_reliable"] is False
    assert timeline["estimated_missing_frames"] == 0
    assert timeline["drop_frame_ratio"] == 0.0


def test_freeze_with_hdf5_keypoint_motion_conflict_fails(tmp_path: Path) -> None:
    batch = tmp_path
    video_dir = batch / "video"
    video_dir.mkdir()
    video = video_dir / "408817_video.mp4"
    write_test_video(video, [solid_frame(120) for _ in range(16)], fps=10.0)
    hdf5_path = batch / "hdf5" / "408817_hdf5.hdf5"
    hdf5_path.parent.mkdir()
    base = np.full((2, 21, 2), 0.45, dtype=np.float32)
    keypoints = np.stack([base + np.array([frame_index * 0.02, 0.0], dtype=np.float32) for frame_index in range(16)])
    with h5py.File(hdf5_path, "w") as handle:
        label = handle.create_group("label")
        label.create_dataset("quality_hand", data=keypoints)

    config = load_video_quality_config(None)
    metrics = analyze_video(video, config, hdf5_path=hdf5_path)
    evaluation = evaluate_video_quality(metrics, config, check_hdf5_alignment(video, batch, metrics, config))

    assert metrics.frozen_intervals
    assert any(interval.motion_conflict for interval in metrics.frozen_intervals)
    assert "video_state_conflict" in evaluation.reasons


def test_freeze_in_hdf5_critical_window_fails(tmp_path: Path) -> None:
    batch = tmp_path
    video_dir = batch / "video"
    video_dir.mkdir()
    video = video_dir / "408817_video.mp4"
    write_test_video(video, [solid_frame(120) for _ in range(16)], fps=10.0)
    hdf5_path = batch / "hdf5" / "408817_hdf5.hdf5"
    hdf5_path.parent.mkdir()
    write_hand_keypoint_hdf5(hdf5_path, frame_count=16, normalized=True)
    with h5py.File(hdf5_path, "a") as handle:
        handle.attrs["task"] = "grasp the cube and place it on the tray"

    config = load_video_quality_config(None)
    metrics = analyze_video(video, config, hdf5_path=hdf5_path)
    evaluation = evaluate_video_quality(metrics, config, check_hdf5_alignment(video, batch, metrics, config))

    assert metrics.frozen_intervals
    assert any(interval.critical_window for interval in metrics.frozen_intervals)
    assert any(interval.motion_conflict for interval in metrics.frozen_intervals)
    assert "video_state_conflict" in evaluation.reasons


def test_video_state_conflict_uses_critical_window_duration_thresholds(tmp_path: Path) -> None:
    video = tmp_path / "408817_video.mp4"
    write_test_video(video, [textured_frame(0), textured_frame(10), textured_frame(20)], fps=30.0)
    metrics = analyze_video(video, load_video_quality_config(None))
    metrics = replace(
        metrics,
        short_side=720,
        long_side=1280,
        frozen_frame_ratio=0.0,
        max_consecutive_frozen_sec=0.0,
        defect_duration_ratio=0.0,
    )

    noncritical_short = replace(
        metrics,
        frozen_intervals=(
            FrozenInterval(0, 26, 27, 0.0, 0.9, 0.9, 900.0, motion_conflict=True),
        ),
    )
    noncritical_long = replace(
        metrics,
        frozen_intervals=(
            FrozenInterval(0, 29, 30, 0.0, 1.0, 1.0, 1000.0, motion_conflict=True),
        ),
    )
    critical_short = replace(
        metrics,
        frozen_intervals=(
            FrozenInterval(
                0,
                14,
                15,
                0.0,
                0.5,
                0.5,
                500.0,
                motion_conflict=True,
                critical_window=True,
                critical_keywords=("grasp",),
            ),
        ),
    )

    config = load_video_quality_config(None)
    assert "video_state_conflict" not in evaluate_video_quality(noncritical_short, config).reasons
    assert "video_state_conflict" in evaluate_video_quality(noncritical_long, config).reasons
    assert "video_state_conflict" in evaluate_video_quality(critical_short, config).reasons


def test_evaluate_video_quality_passes_good_metrics(tmp_path: Path) -> None:
    video = tmp_path / "408817_video.mp4"
    write_test_video(video, [textured_frame(0), textured_frame(10), textured_frame(20)], fps=30.0)
    metrics = analyze_video(video, load_video_quality_config(None))

    evaluation = evaluate_video_quality(metrics, load_video_quality_config(None))

    assert evaluation.passed is True
    assert evaluation.decision == "pass"
    assert evaluation.should_run_mask_qc is True
    assert evaluation.reasons == ()
    assert evaluation.warn_reasons == ()


def test_evaluate_video_quality_passes_calibrated_provider_quality_metrics(tmp_path: Path) -> None:
    video = tmp_path / "408817_video.mp4"
    write_test_video(video, [textured_frame(0), textured_frame(10), textured_frame(20)], fps=30.0)
    metrics = analyze_video(video, load_video_quality_config(None))
    metrics = replace(
        metrics,
        laplacian_p10=134.3,
        laplacian_median=204.8,
        laplacian_under_100_ratio=0.026,
        tenengrad_p10=35.0,
        tenengrad_median=37.1,
        frozen_frame_ratio=0.0,
        max_consecutive_frozen_sec=0.0,
    )

    evaluation = evaluate_video_quality(metrics, load_video_quality_config(None))

    assert evaluation.passed is True
    assert evaluation.decision == "pass"
    assert evaluation.should_run_mask_qc is True
    assert evaluation.reasons == ()
    assert evaluation.warn_reasons == ()


def test_evaluate_video_quality_passes_moderate_global_blur_for_pretraining(tmp_path: Path) -> None:
    video = tmp_path / "408817_video.mp4"
    write_test_video(video, [textured_frame(0), textured_frame(10), textured_frame(20)], fps=30.0)
    metrics = analyze_video(video, load_video_quality_config(None))
    metrics = replace(
        metrics,
        laplacian_p10=95.4,
        laplacian_median=184.8,
        laplacian_under_100_ratio=0.145,
        tenengrad_p10=32.9,
        tenengrad_median=41.9,
    )

    config = load_video_quality_config(None)
    evaluation = evaluate_video_quality(metrics, config)

    assert evaluation.passed is True
    assert evaluation.decision == "pass"
    assert evaluation.should_run_mask_qc is True
    assert evaluation.warn_reasons == ()
    assert "laplacian_p10_below_min" not in evaluation.reasons
    assert "laplacian_median_below_min" not in evaluation.reasons


def test_evaluate_video_quality_passes_borderline_but_usable_global_blur(
    tmp_path: Path,
) -> None:
    video = tmp_path / "408817_video.mp4"
    write_test_video(video, [textured_frame(0), textured_frame(10), textured_frame(20)], fps=30.0)
    metrics = analyze_video(video, load_video_quality_config(None))
    metrics = replace(
        metrics,
        laplacian_p10=44.9,
        laplacian_median=64.0,
        laplacian_under_100_ratio=1.0,
        tenengrad_p10=12.6,
        tenengrad_median=13.7,
        frozen_frame_ratio=0.01834862385321101,
        max_consecutive_frozen_sec=0.06669376218323587,
    )

    config = load_video_quality_config(None)
    evaluation = evaluate_video_quality(metrics, config)

    assert evaluation.passed is True
    assert evaluation.decision == "pass"
    assert evaluation.should_run_mask_qc is True
    assert evaluation.warn_reasons == ()
    assert evaluation.reasons == ()


def test_evaluate_video_quality_warns_low_edge_quality_after_pretraining_calibration(tmp_path: Path) -> None:
    video = tmp_path / "408817_video.mp4"
    write_test_video(video, [textured_frame(0), textured_frame(10), textured_frame(20)], fps=30.0)
    metrics = analyze_video(video, load_video_quality_config(None))
    metrics = replace(
        metrics,
        laplacian_p10=58.2,
        laplacian_median=65.6,
        laplacian_under_100_ratio=1.0,
        tenengrad_p10=11.6,
        tenengrad_median=12.3,
        frozen_frame_ratio=0.08256880733944955,
        max_consecutive_frozen_sec=0.13338898635477583,
    )

    evaluation = evaluate_video_quality(metrics, load_video_quality_config(None))

    assert evaluation.passed is True
    assert evaluation.decision == "warn"
    assert evaluation.should_run_mask_qc is True
    assert "laplacian_under_100_ratio_warn" not in evaluation.warn_reasons
    assert "laplacian_p10_warn" not in evaluation.warn_reasons
    assert "laplacian_median_warn" not in evaluation.warn_reasons
    assert "tenengrad_p10_warn" not in evaluation.warn_reasons
    assert "tenengrad_median_warn" not in evaluation.warn_reasons
    assert evaluation.reasons == ()
    assert "frozen_frame_ratio_warn" in evaluation.warn_reasons


def test_evaluate_video_quality_warns_cross_provider_low_detail_tail(tmp_path: Path) -> None:
    video = tmp_path / "408817_video.mp4"
    write_test_video(video, [textured_frame(0), textured_frame(10), textured_frame(20)], fps=30.0)
    metrics = analyze_video(video, load_video_quality_config(None))
    metrics = replace(
        metrics,
        laplacian_p10=26.2,
        laplacian_median=46.7,
        laplacian_under_100_ratio=0.991,
        tenengrad_p10=11.5,
        tenengrad_median=14.5,
    )

    evaluation = evaluate_video_quality(metrics, load_video_quality_config(None))

    assert evaluation.passed is True
    assert evaluation.decision == "pass"
    assert evaluation.should_run_mask_qc is True
    assert evaluation.warn_reasons == ()
    assert evaluation.reasons == ()


def test_evaluate_video_quality_accepts_human_calibrated_jd_clarity_samples(tmp_path: Path) -> None:
    video = tmp_path / "408817_video.mp4"
    write_test_video(video, [textured_frame(0), textured_frame(10), textured_frame(20)], fps=30.0)
    base_metrics = analyze_video(video, load_video_quality_config(None))
    config = load_video_quality_config(None)

    clear_file003 = replace(
        base_metrics,
        laplacian_p10=18.016677077060322,
        laplacian_median=23.905696234995084,
        laplacian_under_100_ratio=1.0,
        tenengrad_p10=6.7815338475290465,
        tenengrad_median=7.50551405606198,
        adjacent_near_duplicate_ratio=0.9474576271186441,
    )
    clear_file008 = replace(
        base_metrics,
        laplacian_p10=31.281383419232434,
        laplacian_median=41.00769471101951,
        laplacian_under_100_ratio=0.9898305084745763,
        tenengrad_p10=7.953767432246552,
        tenengrad_median=8.57515469377898,
        adjacent_near_duplicate_ratio=0.8739130434782608,
    )
    borderline_file006 = replace(
        base_metrics,
        laplacian_p10=59.8,
        laplacian_median=72.0,
        laplacian_under_100_ratio=0.9104477611940298,
        tenengrad_p10=13.0,
        tenengrad_median=14.5,
        adjacent_near_duplicate_ratio=0.9708333333333333,
    )

    clear_file003_evaluation = evaluate_video_quality(clear_file003, config)
    clear_file008_evaluation = evaluate_video_quality(clear_file008, config)
    borderline_evaluation = evaluate_video_quality(borderline_file006, config)

    sharpness_codes = (
        "laplacian_p10",
        "laplacian_median",
        "laplacian_under_100_ratio",
        "tenengrad_p10",
        "tenengrad_median",
    )
    assert clear_file003_evaluation.decision == "warn"
    assert clear_file003_evaluation.reasons == ()
    assert "adjacent_near_duplicate_ratio_warn" in clear_file003_evaluation.warn_reasons
    assert not any(any(code in reason for code in sharpness_codes) for reason in clear_file003_evaluation.warn_reasons)

    assert clear_file008_evaluation.decision == "pass"
    assert clear_file008_evaluation.reasons == ()
    assert clear_file008_evaluation.warn_reasons == ()

    assert "adjacent_near_duplicate_ratio_warn" in borderline_evaluation.warn_reasons
    assert not any(any(code in reason for code in sharpness_codes) for reason in borderline_evaluation.warn_reasons)


def test_evaluate_video_quality_still_fails_severe_global_blur(tmp_path: Path) -> None:
    video = tmp_path / "408817_video.mp4"
    write_test_video(video, [textured_frame(0), textured_frame(10), textured_frame(20)], fps=30.0)
    metrics = analyze_video(video, load_video_quality_config(None))
    metrics = replace(
        metrics,
        laplacian_p10=5.0,
        laplacian_median=8.0,
        laplacian_under_100_ratio=1.0,
        tenengrad_p10=3.0,
        tenengrad_median=3.5,
    )

    evaluation = evaluate_video_quality(metrics, load_video_quality_config(None))

    assert evaluation.passed is False
    assert evaluation.decision == "fail"
    assert evaluation.should_run_mask_qc is False
    assert "laplacian_p10_warn" in evaluation.warn_reasons
    assert "laplacian_median_warn" in evaluation.warn_reasons
    assert "laplacian_under_100_ratio_above_max" not in evaluation.reasons
    assert "tenengrad_p10_below_min" in evaluation.reasons
    assert "tenengrad_median_below_min" in evaluation.reasons


def test_evaluate_video_quality_fails_decode_and_threshold_reasons(tmp_path: Path) -> None:
    video = tmp_path / "bad_video.mp4"
    video.write_bytes(b"not a video")
    metrics = analyze_video(video, load_video_quality_config(None))

    evaluation = evaluate_video_quality(metrics, load_video_quality_config(None))

    assert evaluation.passed is False
    assert evaluation.decision == "fail"
    assert evaluation.should_run_mask_qc is False
    assert "cannot_open_video" in evaluation.reasons
    assert "sample_decode_ratio_below_min" in evaluation.reasons


def test_evaluate_video_quality_fails_black_and_frozen_video(tmp_path: Path) -> None:
    video = tmp_path / "408817_video.mp4"
    write_test_video(video, [solid_frame(0) for _ in range(16)], fps=10.0)
    metrics = analyze_video(video, load_video_quality_config(None))

    evaluation = evaluate_video_quality(metrics, load_video_quality_config(None))

    assert evaluation.passed is False
    assert "black_frame_ratio_above_max" in evaluation.reasons
    assert "mean_over_dark_ratio_above_max" in evaluation.reasons
    assert "frozen_frame_ratio_above_max" in evaluation.reasons
    assert "max_consecutive_frozen_sec_above_max" in evaluation.reasons


def test_evaluate_video_quality_fails_when_black_frame_count_exceeds_ten(tmp_path: Path) -> None:
    video = tmp_path / "408817_video.mp4"
    write_test_video(video, [textured_frame(0), textured_frame(10), textured_frame(20)], fps=30.0)
    metrics = analyze_video(video, load_video_quality_config(None))
    metrics = replace(
        metrics,
        frame_count=2000,
        black_frame_ratio=11 / 2000,
        black_frame_count_estimate=11,
        mean_over_dark_ratio=0.0,
        exposure_defect_frame_ratio=11 / 2000,
        defect_duration_ratio=11 / 2000,
        frozen_frame_ratio=0.0,
        max_consecutive_frozen_sec=0.0,
    )

    evaluation = evaluate_video_quality(metrics, load_video_quality_config(None))

    assert evaluation.passed is False
    assert evaluation.decision == "fail"
    assert "black_frame_count_above_max" in evaluation.reasons
    assert "black_frame_ratio_above_max" not in evaluation.reasons


def test_evaluate_video_quality_fails_when_total_defect_duration_exceeds_ten_percent(tmp_path: Path) -> None:
    video = tmp_path / "408817_video.mp4"
    write_test_video(video, [textured_frame(0), textured_frame(10), textured_frame(20)], fps=30.0)
    metrics = analyze_video(video, load_video_quality_config(None))

    borderline = replace(
        metrics,
        mean_over_exposed_ratio=0.04,
        exposure_defect_frame_ratio=0.04,
        frozen_frame_ratio=0.03,
        drop_frame_ratio=0.02,
        defect_duration_ratio=0.09,
    )
    borderline_evaluation = evaluate_video_quality(borderline, load_video_quality_config(None))

    assert borderline_evaluation.passed is True
    assert borderline_evaluation.decision == "warn"
    assert "defect_duration_ratio_warn" in borderline_evaluation.warn_reasons
    assert "defect_duration_ratio_above_max" not in borderline_evaluation.reasons

    too_many_defects = replace(
        metrics,
        mean_over_exposed_ratio=0.04,
        exposure_defect_frame_ratio=0.04,
        frozen_frame_ratio=0.04,
        drop_frame_ratio=0.03,
        defect_duration_ratio=0.11,
    )
    too_many_evaluation = evaluate_video_quality(too_many_defects, load_video_quality_config(None))

    assert too_many_evaluation.passed is False
    assert too_many_evaluation.decision == "fail"
    assert "defect_duration_ratio_above_max" in too_many_evaluation.reasons

    mostly_exposed = replace(metrics, mean_over_exposed_ratio=0.91, exposure_defect_frame_ratio=0.91, defect_duration_ratio=0.91)
    mostly_evaluation = evaluate_video_quality(mostly_exposed, load_video_quality_config(None))

    assert mostly_evaluation.passed is False
    assert "mean_over_exposed_ratio_above_max" in mostly_evaluation.reasons
    assert "defect_duration_ratio_above_max" in mostly_evaluation.reasons


def test_evaluate_video_quality_fails_clear_screen_thresholds(tmp_path: Path) -> None:
    video = tmp_path / "408817_video.mp4"
    write_test_video(video, [solid_frame(100), solid_frame(120), solid_frame(140)], fps=10.0)
    metrics = analyze_video(video, load_video_quality_config(None))

    evaluation = evaluate_video_quality(metrics, load_video_quality_config(None))

    assert evaluation.passed is False
    assert "laplacian_p10_warn" in evaluation.warn_reasons
    assert "laplacian_median_warn" in evaluation.warn_reasons
    assert "laplacian_under_100_ratio_above_max" not in evaluation.reasons
    assert "laplacian_under_100_ratio_warn" not in evaluation.warn_reasons
    assert "tenengrad_p10_below_min" in evaluation.reasons
    assert "tenengrad_median_below_min" in evaluation.reasons


def test_evaluate_video_quality_fails_low_resolution(tmp_path: Path) -> None:
    video = tmp_path / "408817_video.mp4"
    write_test_video(video, [textured_frame(0, width=640, height=360), textured_frame(10, width=640, height=360)])
    metrics = analyze_video(video, load_video_quality_config(None))

    evaluation = evaluate_video_quality(metrics, load_video_quality_config(None))

    assert evaluation.passed is False
    assert "short_side_below_min" in evaluation.reasons
    assert "long_side_below_min" in evaluation.reasons


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


def test_hdf5_alignment_small_delta_warns_without_stopping_mask_qc(tmp_path: Path) -> None:
    batch = tmp_path
    video_dir = batch / "video"
    video_dir.mkdir()
    video = video_dir / "408817_video.mp4"
    write_test_video(video, [textured_frame(0), textured_frame(10), textured_frame(20), textured_frame(30)], fps=30.0)
    write_quality_hdf5(batch / "hdf5" / "408817_hdf5.hdf5", 8)
    config_path = write_unified_video_config(
        tmp_path,
        lambda parameters: parameters["hdf5_alignment"].update(max_delta_ratio_warn=1.0),
    )
    config = load_video_quality_config(config_path)
    metrics = analyze_video(video, config)
    alignment = check_hdf5_alignment(video, batch, metrics, config)

    evaluation = evaluate_video_quality(metrics, config, alignment)

    assert alignment.status == "mismatch"
    assert alignment.frame_count_match is False
    assert alignment.frame_count_delta == 4
    assert alignment.frame_count_delta_ratio == pytest.approx(0.5)
    assert evaluation.passed is True
    assert evaluation.decision == "warn"
    assert evaluation.should_run_mask_qc is True
    assert "hdf5_frame_count_mismatch_warn" in evaluation.warn_reasons


def test_hdf5_alignment_missing_warn_mode_does_not_fail(tmp_path: Path) -> None:
    batch = tmp_path
    video_dir = batch / "video"
    video_dir.mkdir()
    video = video_dir / "408817_video.mp4"
    write_test_video(video, [solid_frame(100) for _ in range(4)], fps=10.0)
    config_path = write_unified_video_config(
        tmp_path,
        lambda parameters: parameters["hdf5_alignment"].update(mode="warn"),
    )
    config = load_video_quality_config(config_path)
    metrics = analyze_video(video, config)

    alignment = check_hdf5_alignment(video, batch, metrics, config)
    evaluation = evaluate_video_quality(metrics, config, alignment)

    assert alignment.status == "missing"
    assert "hdf5_missing" not in evaluation.reasons
    assert "hdf5_missing" in evaluation.warn_reasons


def test_hdf5_alignment_large_delta_fails_by_default(tmp_path: Path) -> None:
    batch = tmp_path
    video_dir = batch / "video"
    video_dir.mkdir()
    video = video_dir / "408817_video.mp4"
    write_test_video(video, [textured_frame(0), textured_frame(10), textured_frame(20)], fps=30.0)
    write_quality_hdf5(batch / "hdf5" / "408817_hdf5.hdf5", 20)
    config = load_video_quality_config(None)
    metrics = analyze_video(video, config)

    alignment = check_hdf5_alignment(video, batch, metrics, config)
    evaluation = evaluate_video_quality(metrics, config, alignment)

    assert alignment.frame_count_delta == 17
    assert evaluation.passed is False
    assert "hdf5_frame_count_mismatch" in evaluation.reasons


def test_video_qc_has_no_hand_roi_surface(tmp_path: Path) -> None:
    batch = tmp_path
    video_dir = batch / "video"
    video_dir.mkdir()
    video = video_dir / "408817_video.mp4"
    write_test_video(video, [textured_frame(0), textured_frame(10), textured_frame(20)], fps=30.0)
    write_quality_hdf5(batch / "hdf5" / "408817_hdf5.hdf5", 3)
    advance_batch_report_to_video(batch, "408817")

    assert run_video_quality_check(batch) == 0
    report = json.loads((batch / "quality_archive" / "408817.json").read_text(encoding="utf-8"))
    serialized = json.dumps(report)

    assert "hand_roi" not in serialized
    assert "hand_roi_metrics" not in report["video_quality"]["metrics"]
    assert "hand_roi" not in load_video_quality_config(None).to_dict()



def test_run_video_quality_check_does_not_bypass_pipeline_on_fresh_report(
    tmp_path: Path,
) -> None:
    batch = tmp_path
    video_dir = batch / "video"
    video_dir.mkdir()
    video = video_dir / "408817_video.mp4"
    write_test_video(
        video,
        [textured_frame(0), textured_frame(10), textured_frame(20)],
        fps=30.0,
    )

    assert run_video_quality_check(batch) == 3
    assert not (batch / "quality_archive" / "408817.json").exists()


def _completed_batch_video_report(tmp_path: Path) -> Path:
    video = tmp_path / "video" / "408817_video.mp4"
    write_test_video(
        video,
        [textured_frame(0), textured_frame(10), textured_frame(20)],
        fps=30.0,
    )
    write_quality_hdf5(tmp_path / "hdf5" / "408817_hdf5.hdf5", 3)
    advance_batch_report_to_video(tmp_path, "408817")
    assert run_video_quality_check(tmp_path) == 0
    return tmp_path / "quality_archive" / "408817.json"


def test_batch_video_rejects_completed_block_without_valid_exit_gate(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    report_path = _completed_batch_video_report(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    del report["video_quality"]["flow"]["exit_gate"]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    before = report_path.read_bytes()

    with caplog.at_level("WARNING"):
        exit_code = run_video_quality_check(tmp_path, overwrite=True)

    assert exit_code == 3
    assert report_path.read_bytes() == before
    assert '"condition": "invalid_report"' in caplog.text
    assert '"reason": "video_quality_exit_gate_invalid"' in caplog.text


def test_batch_video_rejects_completed_source_range_mismatch(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    report_path = _completed_batch_video_report(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["video_quality"]["evidence"][0]["end_frame"] = 99
    report_path.write_text(json.dumps(report), encoding="utf-8")
    before = report_path.read_bytes()

    with caplog.at_level("WARNING"):
        exit_code = run_video_quality_check(tmp_path, overwrite=True)

    assert exit_code == 3
    assert report_path.read_bytes() == before
    assert '"condition": "invalid_report"' in caplog.text
    assert '"reason": "video_quality_source_range_mismatch"' in caplog.text


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("null_successor", "video_quality_exit_gate_invalid"),
        ("wrong_successor", "video_quality_exit_gate_invalid"),
        ("exit_pipeline_mismatch", "video_quality_pipeline_state_invalid"),
        ("invalid_continue_state", "video_quality_pipeline_state_invalid"),
        ("invalid_stop_state", "video_quality_exit_gate_invalid"),
    ],
)
def test_batch_video_rejects_inconsistent_completed_flow(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    case: str,
    reason: str,
) -> None:
    report_path = _completed_batch_video_report(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    exit_gate = report["video_quality"]["flow"]["exit_gate"]
    if case == "null_successor":
        exit_gate["next_module"] = None
    elif case == "wrong_successor":
        exit_gate["next_module"] = "semantic_consistency"
    elif case == "exit_pipeline_mismatch":
        report["pipeline_state"]["next_module"] = "semantic_consistency"
    elif case == "invalid_continue_state":
        report["pipeline_state"].update(
            {
                "status": "stopped",
                "next_module": None,
                "stop_reason": "quality_fail:video_quality",
            }
        )
        report["overall_decision"] = "fail"
    else:
        exit_gate.update(
            {
                "state": "stop_qc",
                "continue_to_next_module": False,
                "next_module": None,
            }
        )
        report["pipeline_state"].update(
            {
                "status": "stopped",
                "next_module": None,
                "stop_reason": "quality_fail:video_quality",
            }
        )
        report["overall_decision"] = "fail"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    before = report_path.read_bytes()

    with caplog.at_level("WARNING"):
        exit_code = run_video_quality_check(tmp_path)

    assert exit_code == 3
    assert report_path.read_bytes() == before
    assert f'"reason": "{reason}"' in caplog.text


def test_batch_video_skips_valid_completed_block(tmp_path: Path) -> None:
    report_path = _completed_batch_video_report(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["video_quality"]["flow"]["exit_gate"] == {
        "state": "continue",
        "continue_to_next_module": True,
        "next_module": "supplier_data_audit",
    }
    assert report["pipeline_state"]["status"] == "running"
    before = report_path.read_bytes()

    assert run_video_quality_check(tmp_path) == 0
    assert report_path.read_bytes() == before


def test_batch_video_overwrite_reruns_only_valid_completed_block(
    tmp_path: Path,
) -> None:
    report_path = _completed_batch_video_report(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    revision = report["report_revision"]
    report["video_quality"]["review_marker"] = "replace"
    report["hdf5_text_info"]["review_marker"] = "keep"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    assert run_video_quality_check(tmp_path, overwrite=True) == 0

    overwritten = json.loads(report_path.read_text(encoding="utf-8"))
    assert overwritten["report_revision"] == revision + 1
    assert "review_marker" not in overwritten["video_quality"]
    assert overwritten["hdf5_text_info"]["review_marker"] == "keep"


def test_run_video_quality_check_writes_only_quality_archive_and_returns_zero(tmp_path: Path) -> None:
    batch = tmp_path
    video_dir = batch / "video"
    video_dir.mkdir()
    video = video_dir / "408817_video.mp4"
    write_test_video(video, [textured_frame(0), textured_frame(10), textured_frame(20)], fps=30.0)
    write_quality_hdf5(batch / "hdf5" / "408817_hdf5.hdf5", 3)
    advance_batch_report_to_video(batch, "408817")

    exit_code = run_video_quality_check(batch)

    assert exit_code == 0
    assert not (batch / "reports").exists()
    report = json.loads((batch / "quality_archive" / "408817.json").read_text(encoding="utf-8"))
    assert report["asset_id"] == "408817"
    assert report["schema_version"] == "asset_qc_report.v2"
    assert report["qc_config"]["schema_version"] == "qc_acceptance_config_schema.v2"
    assert report["qc_config"]["config_version"] == "qc_acceptance_v2.3.0"
    assert report["qc_config"]["config_name"] == "acceptance_gate"
    assert report["qc_config"]["config_path"] == "configs/qc_acceptance.yaml"
    assert report["qc_config"]["config_hash"].startswith("sha256:")
    assert report["pipeline_state"] == {
        "status": "running",
        "last_completed_module": "video_quality",
        "next_module": "supplier_data_audit",
        "stop_reason": None,
    }
    assert report["overall_decision"] is None
    assert report["manual_review"]["required"] is None
    assert "thresholds" not in report["video_quality"]
    assert "threshold_version" not in report["video_quality"]


def test_run_video_quality_check_writes_one_qc_json_report_per_asset_id(tmp_path: Path) -> None:
    batch = tmp_path
    video_dir = batch / "video"
    video_dir.mkdir()
    video = video_dir / "408817_video.mp4"
    write_test_video(video, [textured_frame(0), textured_frame(10), textured_frame(20)], fps=30.0)
    write_quality_hdf5_with_text(batch / "hdf5" / "408817_hdf5.hdf5", 3)
    advance_batch_report_to_video(batch, "408817")

    exit_code = run_video_quality_check(batch)

    assert exit_code == 0
    report_path = batch / "quality_archive" / "408817.json"
    assert report_path.is_file()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "asset_qc_report.v2"
    assert report["qc_config"]["config_version"] == "qc_acceptance_v2.3.0"
    assert report["asset_id"] == "408817"
    assert report["report_revision"] == 6
    assert report["issues"] == []
    assert report["overall_decision"] is None
    assert report["source_files"]["video"]["path"] == "video/408817_video.mp4"
    assert report["source_files"]["hdf5"]["path"] == "hdf5/408817_hdf5.hdf5"
    assert report["hdf5_text_info"]["flow"]["result_gate"]["verdict"] == "pass"
    assert report["video_quality"]["metrics"]["metadata"]["frame_count"] == 3
    assert report["video_quality"]["evaluation"] == {
        "decision": "pass",
        "reasons": [],
        "warn_reasons": [],
        "issue_ids": [],
        "should_run_mask_qc": True,
    }
    assert report["video_quality"]["flow"]["result_gate"]["verdict"] == "pass"
    assert report["video_quality"]["flow"]["exit_gate"] == {
        "state": "continue",
        "continue_to_next_module": True,
        "next_module": "supplier_data_audit",
    }
    assert "thresholds" not in report["video_quality"]
    assert "threshold_version" not in report["video_quality"]
    assert report["video_quality"]["metrics"]["sampling"]["decoded_sample_count"] >= 1
    assert report["video_quality"]["metrics"]["video_basic"]["short_side"] == 720
    assert report["video_quality"]["metrics"]["exposure"]["mean_over_dark_ratio"] < 0.1
    assert report["video_quality"]["metrics"]["defect_metrics"] == {
        "defect_duration_ratio": 0.0,
        "exposure_defect_frame_ratio": 0.0,
        "frozen_frame_ratio": 0.0,
        "drop_frame_ratio": 0.0,
    }
    assert report["video_quality"]["metrics"]["freeze_metrics"]["frozen_intervals"] == []
    assert report["video_quality"]["metrics"]["freeze_metrics"]["frozen_interval_count"] == 0
    assert report["video_quality"]["metrics"]["freeze_metrics"]["frozen_interval_frame_count"] == 0
    timeline_metrics = report["video_quality"]["metrics"]["timeline_metrics"]
    assert timeline_metrics["drop_detection_source"] in {"ffprobe", "pyav", "opencv", "synthetic"}
    assert isinstance(timeline_metrics["drop_detection_reliable"], bool)
    assert timeline_metrics["estimated_missing_frames"] == 0
    assert timeline_metrics["observed_frame_interval_count"] >= 0
    freeze_metrics = report["video_quality"]["metrics"]["freeze_metrics"]
    assert freeze_metrics["frozen_interval_min_duration_ms"] == 100.0
    assert freeze_metrics["frozen_interval_motion_conflict_count"] == 0
    assert freeze_metrics["frozen_interval_critical_window_count"] == 0
    assert freeze_metrics["ssim_min"] == 0.995
    assert freeze_metrics["phash_hamming_max"] == 4
    assert report["video_quality"]["metrics"]["reference_quality"]["mode"] == "none"


def test_run_video_quality_check_records_warn_issue_values(tmp_path: Path) -> None:
    batch = tmp_path
    video_dir = batch / "video"
    video_dir.mkdir()
    video = video_dir / "warn_video.mp4"
    write_test_video(video, [textured_frame(0), textured_frame(10), textured_frame(20)], fps=22.0)
    write_quality_hdf5(batch / "hdf5" / "warn_hdf5.hdf5", 3)
    advance_batch_report_to_video(batch, "warn")

    assert run_video_quality_check(batch) == 0
    report = json.loads((batch / "quality_archive" / "warn.json").read_text(encoding="utf-8"))

    assert report["pipeline_state"]["status"] == "running"
    assert report["overall_decision"] is None
    assert report["video_quality"]["flow"]["result_gate"]["verdict"] == "warn"
    issue = next(item for item in report["issues"] if item["code"] == "fps_below_pass")
    assert issue["severity"] == "warn"
    assert issue["module"] == "video_quality"
    assert issue["issue_type"] == "low_fps"
    assert issue["metric"] == "video_basic.fps"
    assert issue["observed_value"] == pytest.approx(22.0)
    assert issue["operator"] == "<"
    assert issue["boundary_value"] == 24.0
    assert issue["rule_id"] == "video_quality.fps_below_pass"
    assert issue["needs_manual_review"] is True
    assert "config_version" not in issue
    assert report["manual_review"]["candidate_issue_ids"] == [issue["issue_id"]]


def test_video_qc_preserves_existing_module_blocks_and_increments_revision(tmp_path: Path) -> None:
    batch = tmp_path
    video_dir = batch / "video"
    video_dir.mkdir()
    video = video_dir / "408817_video.mp4"
    write_test_video(video, [textured_frame(0), textured_frame(10)], fps=30.0)
    write_quality_hdf5(batch / "hdf5" / "408817_hdf5.hdf5", 2)
    archive = batch / "quality_archive"
    advance_batch_report_to_video(batch, "408817")
    report_path = archive / "408817.json"
    seeded = json.loads(report_path.read_text(encoding="utf-8"))
    existing = {
        "hdf5_text_info": {"owner": "hdf5_text_info", "text_fields": {"task": "pick cup"}},
        "quality_hand": {"owner": "quality_hand", "metrics": {"score": 1.0}},
        "future_extension": {"keep": True},
    }
    seeded.update(existing)
    report_path.write_text(json.dumps(seeded), encoding="utf-8")

    assert run_video_quality_check(batch) == 0
    updated = json.loads((archive / "408817.json").read_text(encoding="utf-8"))

    assert updated["report_revision"] == 6
    assert updated["hdf5_text_info"] == existing["hdf5_text_info"]
    assert updated["quality_hand"] == existing["quality_hand"]
    assert updated["future_extension"] == {"keep": True}
    assert updated["video_quality"]["flow"]["result_gate"]["verdict"] == "pass"


def test_run_video_quality_check_returns_nonzero_for_failed_video(tmp_path: Path) -> None:
    batch = tmp_path
    video_dir = batch / "video"
    video_dir.mkdir()
    (video_dir / "bad_video.mp4").write_bytes(b"not a video")
    manifest_metadata = {
        "scene": "manifest scene",
        "task": "manifest task",
        "task_name": "pick_cup",
        "text_en": "Pick the cup.",
        "text_label": "display label",
    }
    advance_batch_report_to_video(
        batch,
        "bad",
        manifest_metadata=manifest_metadata,
    )

    exit_code = run_video_quality_check(batch)

    assert exit_code == 2
    assert not (batch / "reports").exists()
    report = json.loads((batch / "quality_archive" / "bad.json").read_text(encoding="utf-8"))
    assert report["manifest_metadata"] == manifest_metadata
    assert report["pipeline_state"] == {
        "status": "running",
        "last_completed_module": "video_quality",
        "next_module": "supplier_data_audit",
        "stop_reason": None,
    }
    assert report["overall_decision"] is None
    assert report["manual_review"]["state"] == "not_evaluated"
    assert "cannot_open_video" in report["video_quality"]["evaluation"]["reasons"]
    detail = next(
        item
        for item in report["issues"]
        if item["code"] == "cannot_open_video"
    )
    assert detail["rule_id"] == "video_quality.cannot_open_video"
    assert detail["issue_type"] == "video_unreadable"
    assert detail["needs_manual_review"] is False
    assert "config_version" not in detail
    assert report["video_quality"]["flow"]["exit_gate"] == {
        "state": "continue",
        "continue_to_next_module": True,
        "next_module": "supplier_data_audit",
    }
    before = (batch / "quality_archive" / "bad.json").read_bytes()
    assert run_video_quality_check(batch) == 2
    assert (batch / "quality_archive" / "bad.json").read_bytes() == before


def test_video_quality_main_accepts_config_and_writes_quality_archive(tmp_path: Path) -> None:
    batch = tmp_path / "batch"
    video_dir = batch / "video"
    video_dir.mkdir(parents=True)
    write_test_video(video_dir / "408817_video.mp4", [textured_frame(0), textured_frame(10)], fps=30.0)
    def mutate(parameters: dict[str, Any]) -> None:
        parameters["decode"]["max_sample_frames"] = 2
        parameters["hdf5_alignment"]["mode"] = "warn"

    config = write_unified_video_config(tmp_path, mutate)
    advance_batch_report_to_video(batch, "408817", config_path=config)

    exit_code = main(["--batch", str(batch), "--config", str(config)])

    assert exit_code == 0
    report = json.loads((batch / "quality_archive" / "408817.json").read_text(encoding="utf-8"))
    assert report["video_quality"]["metrics"]["sampling"]["sample_count_configured"] == 2
