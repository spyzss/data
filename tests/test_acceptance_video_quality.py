import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import h5py

from acceptance_pull.video_quality import (
    AlignmentMode,
    HandRoiMetrics,
    analyze_video,
    check_hdf5_alignment,
    discover_batch_videos,
    evaluate_video_quality,
    load_video_quality_config,
    main,
    run_video_quality_check,
)
from tests.fixtures import solid_frame, write_hand_keypoint_hdf5, write_quality_hdf5, write_test_video
from tests.fixtures import write_quality_hdf5_with_text


def textured_frame(offset: int, width: int = 1280, height: int = 720) -> np.ndarray:
    y, x = np.indices((height, width))
    gray = (80 + ((x * 7 + y * 5 + offset) % 140)).astype(np.uint8)
    gray[:, width // 3 : width // 3 + 2] = 20
    gray[:, 2 * width // 3 : 2 * width // 3 + 2] = 235
    return np.repeat(gray[:, :, None], 3, axis=2)


def test_default_video_quality_config() -> None:
    config = load_video_quality_config(None)

    assert config.threshold_version == "video_prefilter_v0.2.6"
    assert config.pipeline.stop_before_mask_if_fail is True
    assert config.pipeline.run_hand_roi is True
    assert config.pipeline.do_keypoint_quality_check is False
    assert config.fps.expected_fps is None
    assert config.fps.min_fps_pass == 24
    assert config.fps.min_fps_warn == 20
    assert config.resolution.min_short_side_fail == 720
    assert config.resolution.min_long_side_fail == 1280
    assert config.decode.sample_decode_ratio_pass == 0.995
    assert config.decode.sample_decode_ratio_warn == 0.98
    assert config.decode.max_sample_frames == 300
    assert config.exposure.black.ratio_pass == 0.005
    assert config.exposure.over_dark.ratio_pass == 0.05
    assert config.exposure.over_exposed.ratio_pass == 0.05
    assert config.sharpness_global.target_short_side == 720
    assert config.sharpness_global.laplacian_p10_pass == 100
    assert config.sharpness_global.laplacian_p10_warn == 35
    assert config.sharpness_global.laplacian_median_pass == 120
    assert config.sharpness_global.laplacian_median_warn == 50
    assert config.sharpness_global.laplacian_under_100_ratio_pass == 0.15
    assert config.sharpness_global.laplacian_under_100_ratio_warn == 1.00
    assert config.sharpness_global.tenengrad_p10_pass == 18
    assert config.sharpness_global.tenengrad_p10_warn == 12
    assert config.sharpness_global.tenengrad_median_pass == 19
    assert config.sharpness_global.tenengrad_median_warn == 13
    assert config.freeze.frozen_frame_ratio_pass == 0.09
    assert config.freeze.frozen_frame_ratio_warn == 0.15
    assert config.freeze.max_consecutive_frozen_sec_fail == 1.0
    assert config.hdf5_alignment.mode == AlignmentMode.FAIL
    assert config.hdf5_alignment.max_delta_frames_pass == 2
    assert config.hand_roi.mode == "warn_except_severe_fail"
    assert config.hand_roi.use_keypoints_as_bbox_only is True


def test_video_quality_config_yaml_override(tmp_path: Path) -> None:
    path = tmp_path / "quality.yaml"
    path.write_text(
        """
threshold_version: video_prefilter_v0.2-custom
decode:
  max_sample_frames: 4
hdf5_alignment:
  mode: warn
fps:
  expected_fps: 30
resolution:
  min_short_side_fail: 480
hand_roi:
  enabled: false
""",
        encoding="utf-8",
    )

    config = load_video_quality_config(path)

    assert config.threshold_version == "video_prefilter_v0.2-custom"
    assert config.decode.max_sample_frames == 4
    assert config.hdf5_alignment.mode == AlignmentMode.WARN
    assert config.fps.expected_fps == 30
    assert config.resolution.min_short_side_fail == 480
    assert config.hand_roi.enabled is False


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
    write_test_video(video, [solid_frame(0) for _ in range(6)], fps=10.0)

    metrics = analyze_video(video, load_video_quality_config(None))

    assert metrics.black_frame_ratio >= 0.9
    assert metrics.frozen_frame_ratio >= 0.8


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


def test_evaluate_video_quality_warns_moderate_global_blur_without_stopping_mask_qc(tmp_path: Path) -> None:
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

    evaluation = evaluate_video_quality(metrics, load_video_quality_config(None))

    assert evaluation.passed is True
    assert evaluation.decision == "warn"
    assert evaluation.should_run_mask_qc is True
    assert "laplacian_p10_warn" in evaluation.warn_reasons
    assert "laplacian_median_warn" not in evaluation.warn_reasons
    assert "laplacian_under_100_ratio_warn" not in evaluation.warn_reasons
    assert "laplacian_p10_below_min" not in evaluation.reasons
    assert "laplacian_median_below_min" not in evaluation.reasons


def test_evaluate_video_quality_warns_borderline_but_usable_global_blur_without_stopping_mask_qc(
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

    evaluation = evaluate_video_quality(metrics, load_video_quality_config(None))

    assert evaluation.passed is True
    assert evaluation.decision == "warn"
    assert evaluation.should_run_mask_qc is True
    assert "laplacian_p10_warn" in evaluation.warn_reasons
    assert "laplacian_median_warn" in evaluation.warn_reasons
    assert "laplacian_under_100_ratio_warn" in evaluation.warn_reasons
    assert "tenengrad_p10_warn" in evaluation.warn_reasons
    assert "tenengrad_median_warn" in evaluation.warn_reasons
    assert evaluation.reasons == ()


def test_evaluate_video_quality_fails_low_edge_quality_after_provider_calibration(tmp_path: Path) -> None:
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

    assert evaluation.passed is False
    assert evaluation.decision == "fail"
    assert evaluation.should_run_mask_qc is False
    assert "laplacian_p10_warn" in evaluation.warn_reasons
    assert "laplacian_median_warn" in evaluation.warn_reasons
    assert "laplacian_under_100_ratio_warn" in evaluation.warn_reasons
    assert "tenengrad_p10_below_min" in evaluation.reasons
    assert "tenengrad_median_below_min" in evaluation.reasons
    assert "frozen_frame_ratio_warn" not in evaluation.warn_reasons


def test_evaluate_video_quality_fails_cross_provider_extreme_blur_tail(tmp_path: Path) -> None:
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

    assert evaluation.passed is False
    assert evaluation.decision == "fail"
    assert evaluation.should_run_mask_qc is False
    assert "laplacian_p10_below_min" in evaluation.reasons
    assert "laplacian_median_below_min" in evaluation.reasons
    assert "tenengrad_p10_below_min" in evaluation.reasons
    assert "tenengrad_median_below_min" not in evaluation.reasons


def test_evaluate_video_quality_still_fails_severe_global_blur(tmp_path: Path) -> None:
    video = tmp_path / "408817_video.mp4"
    write_test_video(video, [textured_frame(0), textured_frame(10), textured_frame(20)], fps=30.0)
    metrics = analyze_video(video, load_video_quality_config(None))
    metrics = replace(
        metrics,
        laplacian_p10=10.0,
        laplacian_median=20.0,
        laplacian_under_100_ratio=0.853,
        tenengrad_p10=5.0,
        tenengrad_median=8.0,
    )

    evaluation = evaluate_video_quality(metrics, load_video_quality_config(None))

    assert evaluation.passed is False
    assert evaluation.decision == "fail"
    assert evaluation.should_run_mask_qc is False
    assert "laplacian_p10_below_min" in evaluation.reasons
    assert "laplacian_median_below_min" in evaluation.reasons
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


def test_evaluate_video_quality_fails_clear_screen_thresholds(tmp_path: Path) -> None:
    video = tmp_path / "408817_video.mp4"
    write_test_video(video, [solid_frame(100), solid_frame(120), solid_frame(140)], fps=10.0)
    metrics = analyze_video(video, load_video_quality_config(None))

    evaluation = evaluate_video_quality(metrics, load_video_quality_config(None))

    assert evaluation.passed is False
    assert "laplacian_p10_below_min" in evaluation.reasons
    assert "laplacian_median_below_min" in evaluation.reasons
    assert "laplacian_under_100_ratio_above_max" not in evaluation.reasons
    assert "laplacian_under_100_ratio_warn" in evaluation.warn_reasons
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
    config_path = tmp_path / "quality.yaml"
    config_path.write_text("hdf5_alignment:\n  max_delta_ratio_warn: 1.0\n", encoding="utf-8")
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
    config_path = tmp_path / "quality.yaml"
    config_path.write_text("hdf5_alignment:\n  mode: warn\n", encoding="utf-8")
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


def test_hand_roi_bbox_metrics_warn_without_keypoint_quality_check(tmp_path: Path) -> None:
    batch = tmp_path
    video_dir = batch / "video"
    video_dir.mkdir()
    video = video_dir / "408817_video.mp4"
    write_test_video(video, [textured_frame(0), textured_frame(10), textured_frame(20)], fps=30.0)
    write_hand_keypoint_hdf5(batch / "hdf5" / "408817_hdf5.hdf5", frame_count=3, normalized=True)
    config = load_video_quality_config(None)

    metrics = analyze_video(video, config, hdf5_path=batch / "hdf5" / "408817_hdf5.hdf5")
    evaluation = evaluate_video_quality(metrics, config, check_hdf5_alignment(video, batch, metrics, config))

    assert metrics.hand_roi is not None
    assert metrics.hand_roi.source == "hdf5_keypoints_bbox"
    assert metrics.hand_roi.available_ratio == 1.0
    assert metrics.hand_roi.laplacian_p10 > 0
    assert "hand_roi_available_ratio_below_min" not in evaluation.reasons
    assert evaluation.should_run_mask_qc is True


def test_hand_roi_uses_transform_matrices_as_keypoints(tmp_path: Path) -> None:
    batch = tmp_path
    video_dir = batch / "video"
    video_dir.mkdir()
    video = video_dir / "408817_video.mp4"
    write_test_video(video, [textured_frame(0), textured_frame(10), textured_frame(20)], fps=30.0)
    hdf5_path = batch / "hdf5" / "408817_hdf5.hdf5"
    hdf5_path.parent.mkdir()
    joint_offsets = [
        ("leftHand", (-0.10, -0.05, 1.0)),
        ("leftIndexFingerKnuckle", (-0.08, -0.03, 1.0)),
        ("leftIndexFingerTip", (-0.06, -0.01, 1.0)),
        ("leftThumbTip", (-0.04, 0.03, 1.0)),
        ("rightHand", (0.05, -0.04, 1.0)),
        ("rightIndexFingerKnuckle", (0.07, -0.02, 1.0)),
        ("rightIndexFingerTip", (0.09, 0.00, 1.0)),
        ("rightThumbTip", (0.11, 0.04, 1.0)),
    ]
    with h5py.File(hdf5_path, "w") as handle:
        label = handle.create_group("label")
        label.create_dataset("quality_hand", data=np.ones((3, 2), dtype=np.float32))
        camera = handle.create_group("camera")
        camera.create_dataset(
            "intrinsic",
            data=np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        )
        transforms = handle.create_group("transforms")
        for name, offset in joint_offsets:
            matrices = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 3, axis=0)
            for frame_index in range(3):
                matrices[frame_index, :3, 3] = np.array(offset, dtype=np.float32) + np.array(
                    [frame_index * 0.002, 0.0, 0.0],
                    dtype=np.float32,
                )
            transforms.create_dataset(name, data=matrices)
    config = load_video_quality_config(None)

    metrics = analyze_video(video, config, hdf5_path=hdf5_path)
    evaluation = evaluate_video_quality(metrics, config, check_hdf5_alignment(video, batch, metrics, config))

    assert metrics.hand_roi is not None
    assert metrics.hand_roi.source == "hdf5_transform_keypoints_bbox"
    assert metrics.hand_roi.available_ratio == 1.0
    assert metrics.hand_roi.unavailable_reasons == ()
    assert "hand_roi_available_ratio_below_min" not in evaluation.warn_reasons


def test_hand_roi_soft_blur_warns_under_warn_except_severe_fail(tmp_path: Path) -> None:
    video = tmp_path / "408817_video.mp4"
    write_test_video(video, [textured_frame(0), textured_frame(10), textured_frame(20)], fps=30.0)
    metrics = analyze_video(video, load_video_quality_config(None))
    metrics = replace(
        metrics,
        hand_roi=HandRoiMetrics(
            enabled=True,
            source="hdf5_transform_keypoints_bbox",
            sampled_frame_count=10,
            available_frame_count=10,
            available_ratio=1.0,
            laplacian_p10=80.0,
            laplacian_median=150.0,
            tenengrad_p10=14.0,
            tenengrad_median=20.0,
            blur_bad_frame_ratio=0.20,
        ),
    )

    evaluation = evaluate_video_quality(metrics, load_video_quality_config(None))

    assert evaluation.passed is True
    assert evaluation.decision == "warn"
    assert evaluation.should_run_mask_qc is True
    assert "hand_roi_laplacian_p10_warn" in evaluation.warn_reasons
    assert "hand_roi_laplacian_median_warn" in evaluation.warn_reasons
    assert "hand_roi_laplacian_p10_below_min" not in evaluation.reasons
    assert "hand_roi_laplacian_median_below_min" not in evaluation.reasons
    assert "hand_roi_severe_blur" not in evaluation.reasons
    assert "hand_roi_blur_bad_frame_ratio_above_max" not in evaluation.reasons


def test_hand_roi_unavailable_warns_without_severe_blur_fail(tmp_path: Path) -> None:
    batch = tmp_path
    video_dir = batch / "video"
    video_dir.mkdir()
    video = video_dir / "408817_video.mp4"
    write_test_video(video, [textured_frame(0), textured_frame(10), textured_frame(20)], fps=30.0)
    hdf5_path = batch / "hdf5" / "408817_hdf5.hdf5"
    hdf5_path.parent.mkdir()
    with h5py.File(hdf5_path, "w") as handle:
        label = handle.create_group("label")
        label.create_dataset("quality_hand", data=np.ones((3, 2), dtype=np.float32))
        transforms = handle.create_group("transforms")
        transforms.create_dataset("leftHand", data=np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 3, axis=0))
    config = load_video_quality_config(None)

    metrics = analyze_video(video, config, hdf5_path=hdf5_path)
    evaluation = evaluate_video_quality(metrics, config, check_hdf5_alignment(video, batch, metrics, config))

    assert metrics.hand_roi is not None
    assert metrics.hand_roi.available_ratio == 0.0
    assert evaluation.passed is True
    assert evaluation.decision == "warn"
    assert evaluation.should_run_mask_qc is True
    assert "hand_roi_available_ratio_below_min" in evaluation.warn_reasons
    assert "hand_roi_severe_blur" not in evaluation.reasons
    assert "hand_roi_blur_bad_frame_ratio_above_max" not in evaluation.reasons


def test_run_video_quality_check_writes_only_quality_archive_and_returns_zero(tmp_path: Path) -> None:
    batch = tmp_path
    video_dir = batch / "video"
    video_dir.mkdir()
    video = video_dir / "408817_video.mp4"
    write_test_video(video, [textured_frame(0), textured_frame(10), textured_frame(20)], fps=30.0)
    write_quality_hdf5(batch / "hdf5" / "408817_hdf5.hdf5", 3)

    exit_code = run_video_quality_check(batch)

    assert exit_code == 0
    assert not (batch / "reports").exists()
    report = json.loads((batch / "quality_archive" / "408817.json").read_text(encoding="utf-8"))
    assert report["asset_id"] == "408817"
    assert report["qc_summary"]["status"] == "pass"
    assert report["qc_summary"]["should_run_mask_qc"] is True
    assert report["video_quality"]["thresholds"]["decode"]["sample_decode_ratio_pass"] == 0.995


def test_run_video_quality_check_writes_one_qc_json_report_per_asset_id(tmp_path: Path) -> None:
    batch = tmp_path
    video_dir = batch / "video"
    video_dir.mkdir()
    video = video_dir / "408817_video.mp4"
    write_test_video(video, [textured_frame(0), textured_frame(10), textured_frame(20)], fps=30.0)
    write_quality_hdf5_with_text(batch / "hdf5" / "408817_hdf5.hdf5", 3)

    exit_code = run_video_quality_check(batch)

    assert exit_code == 0
    report_path = batch / "quality_archive" / "408817.json"
    assert report_path.is_file()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "asset_qc_report.v1"
    assert report["asset_id"] == "408817"
    assert report["qc_summary"] == {
        "status": "pass",
        "passed": True,
        "completed_modules": ["video_quality"],
        "failed_modules": [],
        "reasons": [],
        "warn_reasons": [],
        "should_run_mask_qc": True,
    }
    assert report["source_files"]["video"]["path"] == "video/408817_video.mp4"
    assert report["source_files"]["hdf5"]["path"] == "hdf5/408817_hdf5.hdf5"
    assert report["hdf5_text_info"]["alignment"]["status"] == "matched"
    assert report["hdf5_text_info"]["text_fields"]["attributes"]["/"]["task"] == "pick up red cup"
    assert report["hdf5_text_info"]["text_fields"]["attributes"]["/meta"]["scene"] == "kitchen"
    assert report["hdf5_text_info"]["text_fields"]["datasets"]["/meta/instruction"] == "move the cup to the tray"
    assert report["hdf5_text_info"]["text_fields"]["datasets"]["/meta/structured_label"] == {
        "language": "zh",
        "task": "整理桌面",
    }
    assert report["video_quality"]["metadata"]["frame_count"] == 3
    assert report["video_quality"]["evaluation"] == {
        "decision": "pass",
        "passed": True,
        "reasons": [],
        "warn_reasons": [],
        "should_run_mask_qc": True,
    }
    assert report["video_quality"]["sampling"]["decoded_sample_count"] >= 1
    assert report["video_quality"]["metrics"]["video_basic"]["short_side"] == 720
    assert report["video_quality"]["metrics"]["exposure"]["mean_over_dark_ratio"] < 0.1
    assert "timeline_metrics" in report["video_quality"]["metrics"]
    assert "hand_roi_metrics" in report["video_quality"]["metrics"]
    assert report["reference_quality"]["mode"] == "none"


def test_run_video_quality_check_returns_nonzero_for_failed_video(tmp_path: Path) -> None:
    batch = tmp_path
    video_dir = batch / "video"
    video_dir.mkdir()
    (video_dir / "bad_video.mp4").write_bytes(b"not a video")

    exit_code = run_video_quality_check(batch)

    assert exit_code == 2
    assert not (batch / "reports").exists()
    report = json.loads((batch / "quality_archive" / "bad.json").read_text(encoding="utf-8"))
    assert report["qc_summary"]["status"] == "fail"
    assert report["qc_summary"]["failed_modules"] == ["video_quality"]
    assert "cannot_open_video" in report["video_quality"]["evaluation"]["reasons"]


def test_video_quality_main_accepts_config_and_writes_quality_archive(tmp_path: Path) -> None:
    batch = tmp_path / "batch"
    video_dir = batch / "video"
    video_dir.mkdir(parents=True)
    write_test_video(video_dir / "408817_video.mp4", [textured_frame(0), textured_frame(10)], fps=30.0)
    config = tmp_path / "quality.yaml"
    config.write_text("decode:\n  max_sample_frames: 2\nhdf5_alignment:\n  mode: warn\n", encoding="utf-8")

    exit_code = main(["--batch", str(batch), "--config", str(config)])

    assert exit_code == 0
    report = json.loads((batch / "quality_archive" / "408817.json").read_text(encoding="utf-8"))
    assert report["video_quality"]["sampling"]["sample_count_configured"] == 2
