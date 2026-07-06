from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import yaml


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi"}
LAPLACIAN_LOW_DETAIL_THRESHOLD = 100.0
DARK_PIXEL_Y_THRESHOLD = 20
OVER_EXPOSED_PIXEL_Y_THRESHOLD = 245


class AlignmentMode(StrEnum):
    IGNORE = "ignore"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class PipelineConfig:
    stop_before_mask_if_fail: bool = True
    run_hand_roi: bool = True
    hand_roi_source: str = "hdf5_keypoints"
    do_keypoint_quality_check: bool = False
    do_keypoint_mask_matching: bool = False
    do_keypoint_temporal_check: bool = False


@dataclass(frozen=True)
class FpsConfig:
    expected_fps: float | None = None
    min_fps_pass: float = 24.0
    min_fps_warn: float = 20.0
    min_fps_fail: float = 20.0


@dataclass(frozen=True)
class ResolutionConfig:
    exact_resolution_required: bool = False
    use_display_size_after_rotation: bool = True
    min_short_side_fail: int = 720
    min_long_side_fail: int = 1280
    min_short_side_warn: int = 720
    min_long_side_warn: int = 1280


@dataclass(frozen=True)
class TimelineConfig:
    use_actual_fps: bool = True
    drop_interval_factor: float = 1.35
    drop_interval_extra_ms: float = 10.0
    drop_frame_ratio_pass: float = 0.01
    drop_frame_ratio_warn: float = 0.02
    max_gap_factor: float = 3.0
    max_gap_floor_ms: float = 100.0
    pts_monotonic_required: bool = True


@dataclass(frozen=True)
class DecodeConfig:
    sample_decode_ratio_pass: float = 0.995
    sample_decode_ratio_warn: float = 0.98
    max_sample_frames: int = 300
    sample_interval_sec: float = 0.5
    include_head_tail_frames: int = 10


@dataclass(frozen=True)
class BlackFrameConfig:
    mean_y_max: float = 10.0
    dark_pixel_ratio_min: float = 0.98
    ratio_pass: float = 0.005
    ratio_warn: float = 0.01


@dataclass(frozen=True)
class OverDarkConfig:
    mean_y_max: float = 35.0
    dark_pixel_ratio_min: float = 0.75
    ratio_pass: float = 0.05
    ratio_warn: float = 0.10


@dataclass(frozen=True)
class OverExposedConfig:
    mean_y_min: float = 235.0
    over_exposed_pixel_ratio_min: float = 0.35
    ratio_pass: float = 0.05
    ratio_warn: float = 0.10


@dataclass(frozen=True)
class ExposureConfig:
    black: BlackFrameConfig = field(default_factory=BlackFrameConfig)
    over_dark: OverDarkConfig = field(default_factory=OverDarkConfig)
    over_exposed: OverExposedConfig = field(default_factory=OverExposedConfig)


@dataclass(frozen=True)
class SharpnessGlobalConfig:
    normalize_before_compute: bool = True
    target_short_side: int = 720
    no_upscale: bool = True
    laplacian_p10_pass: float = 150.0
    laplacian_p10_warn: float = 20.0
    laplacian_median_pass: float = 220.0
    laplacian_median_warn: float = 40.0
    laplacian_under_100_ratio_pass: float = 0.05
    laplacian_under_100_ratio_warn: float = 1.00
    tenengrad_p10_pass: float = 25.0
    tenengrad_p10_warn: float = 10.0
    tenengrad_median_pass: float = 30.0
    tenengrad_median_warn: float = 12.0


@dataclass(frozen=True)
class FreezeConfig:
    enabled: bool = True
    downscale_short_side: int = 360
    frame_diff_mean_abs_max: float = 1.0
    hist_diff_max: float = 0.01
    frozen_frame_ratio_pass: float = 0.03
    frozen_frame_ratio_warn: float = 0.15
    max_consecutive_frozen_sec_pass: float = 0.5
    max_consecutive_frozen_sec_fail: float = 1.0


@dataclass(frozen=True)
class Hdf5AlignmentConfig:
    enabled: bool = True
    mode: AlignmentMode = AlignmentMode.FAIL
    max_delta_frames_pass: int = 2
    max_delta_frames_warn: int = 5
    max_delta_ratio_pass: float = 0.001
    max_delta_ratio_warn: float = 0.005


@dataclass(frozen=True)
class HandRoiSevereFailConfig:
    enabled: bool = True
    require_both_lap_and_ten_fail: bool = True
    laplacian_p10_fail: float = 100.0
    tenengrad_p10_fail: float = 12.0
    blur_bad_frame_ratio_fail: float = 0.25


@dataclass(frozen=True)
class HandRoiConfig:
    enabled: bool = True
    mode: str = "warn_except_severe_fail"
    use_keypoints_as_bbox_only: bool = True
    min_valid_points_for_bbox: int = 8
    bbox_expand_scale: float = 1.8
    min_roi_width_px: int = 64
    min_roi_height_px: int = 64
    min_roi_area_ratio: float = 0.002
    roi_target_short_side: int = 256
    no_upscale_if_roi_too_small: bool = True
    available_ratio_pass: float = 0.70
    available_ratio_warn: float = 0.40
    laplacian_p10_pass: float = 160.0
    laplacian_p10_warn: float = 100.0
    laplacian_median_pass: float = 240.0
    laplacian_median_warn: float = 160.0
    tenengrad_p10_pass: float = 18.0
    tenengrad_p10_warn: float = 12.0
    tenengrad_median_pass: float = 24.0
    tenengrad_median_warn: float = 16.0
    blur_bad_frame_ratio_pass: float = 0.12
    blur_bad_frame_ratio_warn: float = 0.25
    severe_fail: HandRoiSevereFailConfig = field(default_factory=HandRoiSevereFailConfig)


@dataclass(frozen=True)
class VideoQualityConfig:
    threshold_version: str = "video_prefilter_v0.2.2"
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    fps: FpsConfig = field(default_factory=FpsConfig)
    resolution: ResolutionConfig = field(default_factory=ResolutionConfig)
    timeline: TimelineConfig = field(default_factory=TimelineConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    exposure: ExposureConfig = field(default_factory=ExposureConfig)
    sharpness_global: SharpnessGlobalConfig = field(default_factory=SharpnessGlobalConfig)
    freeze: FreezeConfig = field(default_factory=FreezeConfig)
    hdf5_alignment: Hdf5AlignmentConfig = field(default_factory=Hdf5AlignmentConfig)
    hand_roi: HandRoiConfig = field(default_factory=HandRoiConfig)

    @property
    def sample_count(self) -> int:
        return self.decode.max_sample_frames

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)


@dataclass(frozen=True)
class HandRoiMetrics:
    enabled: bool
    source: str
    sampled_frame_count: int
    available_frame_count: int
    available_ratio: float
    laplacian_p10: float
    laplacian_median: float
    tenengrad_p10: float
    tenengrad_median: float
    blur_bad_frame_ratio: float
    unavailable_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class VideoMetrics:
    path: Path
    asset_id: str
    opened: bool
    video_stream_present: bool
    codec_readable: bool
    metadata_read_ok: bool
    frame_count: int
    fps: float
    duration_seconds: float
    width: int
    height: int
    display_width: int
    display_height: int
    short_side: int
    long_side: int
    sampled_frame_count: int
    decoded_sample_count: int
    sample_decode_ratio: float
    mean_brightness: float
    black_frame_ratio: float
    mean_over_dark_ratio: float
    mean_over_exposed_ratio: float
    laplacian_min: float
    laplacian_p10: float
    laplacian_median: float
    mean_blur_laplacian_var: float
    laplacian_p90: float
    laplacian_under_100_ratio: float
    tenengrad_p10: float
    tenengrad_median: float
    tenengrad_mean: float
    sharpness_scale_short_side: int
    frozen_frame_ratio: float
    max_consecutive_frozen_sec: float
    pts_monotonic_valid: bool
    drop_frame_ratio: float
    frame_interval_p99_ms: float
    max_frame_gap_ms: float
    expected_interval_ms: float
    drop_interval_ms: float
    max_gap_fail_ms: float
    hand_roi: HandRoiMetrics | None = None
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class QualityEvaluation:
    decision: str
    passed: bool
    reasons: tuple[str, ...]
    warn_reasons: tuple[str, ...] = ()
    should_run_mask_qc: bool = True


@dataclass(frozen=True)
class Hdf5Alignment:
    status: str
    hdf5_path: Path | None
    hdf5_frame_count: int | None
    frame_count_match: bool | None
    frame_count_delta: int | None = None
    frame_count_delta_ratio: float | None = None
    reason: str | None = None


@dataclass(frozen=True)
class VideoQualityResult:
    metrics: VideoMetrics
    alignment: Hdf5Alignment
    evaluation: QualityEvaluation


def _to_plain(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if is_dataclass(value):
        return {key: _to_plain(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _merge_dataclass(default_obj: Any, raw: dict[str, Any], section: str) -> Any:
    if not isinstance(raw, dict):
        raise ValueError(f"{section} must be a mapping")

    known = {item.name for item in fields(default_obj)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"unknown config key in {section}: {unknown[0]}")

    values: dict[str, Any] = {}
    for item in fields(default_obj):
        current = getattr(default_obj, item.name)
        if item.name not in raw:
            values[item.name] = current
            continue

        incoming = raw[item.name]
        if is_dataclass(current):
            values[item.name] = _merge_dataclass(current, incoming or {}, f"{section}.{item.name}")
        elif isinstance(current, AlignmentMode):
            values[item.name] = AlignmentMode(str(incoming).lower())
        else:
            values[item.name] = incoming

    return type(default_obj)(**values)


def _apply_legacy_config(raw: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(raw)
    if "sample_count" in migrated:
        migrated.setdefault("decode", {})["max_sample_frames"] = int(migrated.pop("sample_count"))
    if "alignment_mode" in migrated:
        migrated.setdefault("hdf5_alignment", {})["mode"] = migrated.pop("alignment_mode")
    if "thresholds" in migrated:
        thresholds = migrated.pop("thresholds") or {}
        decode = migrated.setdefault("decode", {})
        exposure = migrated.setdefault("exposure", {})
        black = exposure.setdefault("black", {})
        over_dark = exposure.setdefault("over_dark", {})
        over_exposed = exposure.setdefault("over_exposed", {})
        sharpness = migrated.setdefault("sharpness_global", {})
        freeze = migrated.setdefault("freeze", {})
        hdf5_alignment = migrated.setdefault("hdf5_alignment", {})

        if "min_sample_decode_ratio" in thresholds:
            decode["sample_decode_ratio_pass"] = thresholds["min_sample_decode_ratio"]
        if "max_black_frame_ratio" in thresholds:
            black["ratio_warn"] = thresholds["max_black_frame_ratio"]
        if "max_mean_over_dark_ratio" in thresholds:
            over_dark["ratio_warn"] = thresholds["max_mean_over_dark_ratio"]
        if "max_mean_over_exposed_ratio" in thresholds:
            over_exposed["ratio_warn"] = thresholds["max_mean_over_exposed_ratio"]
        if "min_laplacian_p10" in thresholds:
            sharpness["laplacian_p10_pass"] = thresholds["min_laplacian_p10"]
        if "min_laplacian_median" in thresholds:
            sharpness["laplacian_median_pass"] = thresholds["min_laplacian_median"]
        if "max_laplacian_under_100_ratio" in thresholds:
            sharpness["laplacian_under_100_ratio_pass"] = thresholds["max_laplacian_under_100_ratio"]
        if "min_tenengrad_p10" in thresholds:
            sharpness["tenengrad_p10_pass"] = thresholds["min_tenengrad_p10"]
        if "min_tenengrad_median" in thresholds:
            sharpness["tenengrad_median_pass"] = thresholds["min_tenengrad_median"]
        if "max_frozen_frame_ratio" in thresholds:
            freeze["frozen_frame_ratio_warn"] = thresholds["max_frozen_frame_ratio"]
        if "fail_on_hdf5_frame_mismatch" in thresholds and not thresholds["fail_on_hdf5_frame_mismatch"]:
            hdf5_alignment["mode"] = AlignmentMode.WARN.value

    return migrated


def load_video_quality_config(path: Path | None) -> VideoQualityConfig:
    default = VideoQualityConfig()
    if path is None:
        return default

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw = _apply_legacy_config(raw)

    known = {item.name for item in fields(default)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"unknown video quality config key: {unknown[0]}")

    config = _merge_dataclass(default, raw, "video_quality")
    if config.decode.max_sample_frames < 1:
        raise ValueError("decode.max_sample_frames must be >= 1")
    return config


def discover_batch_videos(batch_dir: Path) -> list[Path]:
    video_dir = batch_dir / "video"
    if not video_dir.is_dir():
        raise FileNotFoundError(f"video directory not found: {video_dir}")
    return sorted(path for path in video_dir.iterdir() if path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS)


def asset_id_from_video(path: Path) -> str:
    stem = path.stem
    return stem.removesuffix("_video")


def _sample_indexes(frame_count: int, fps: float, duration_seconds: float, config: DecodeConfig) -> list[int]:
    if frame_count <= 0:
        return []

    indexes: set[int] = set()
    head_tail = max(0, config.include_head_tail_frames)
    indexes.update(range(min(head_tail, frame_count)))
    indexes.update(range(max(0, frame_count - head_tail), frame_count))

    if duration_seconds <= 60:
        interval = 10
    else:
        interval = max(1, round(max(fps, 1.0) * config.sample_interval_sec))
    indexes.update(range(0, frame_count, max(1, interval)))

    ordered = sorted(indexes)
    if len(ordered) <= config.max_sample_frames:
        return ordered

    count = config.max_sample_frames
    if count == 1:
        return [ordered[0]]
    return sorted({ordered[round(index * (len(ordered) - 1) / (count - 1))] for index in range(count)})


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values else 0.0


def _resize_keep_aspect(frame: np.ndarray, target_short_side: int, no_upscale: bool) -> tuple[np.ndarray, int]:
    height, width = frame.shape[:2]
    short_side = min(width, height)
    if short_side <= 0 or target_short_side <= 0:
        return frame, short_side
    if no_upscale and short_side < target_short_side:
        return frame, short_side
    scale = target_short_side / short_side
    if abs(scale - 1.0) < 1e-6:
        return frame, short_side
    resized = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    return resized, min(resized.shape[:2])


def _tenengrad(gray: np.ndarray) -> float:
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)))


def _sharpness_for_frame(frame: np.ndarray, target_short_side: int, no_upscale: bool) -> tuple[float, float, int]:
    normalized, scale_short_side = _resize_keep_aspect(frame, target_short_side, no_upscale)
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()), _tenengrad(gray), scale_short_side


def _chi_square_hist_diff(left: np.ndarray, right: np.ndarray) -> float:
    left_hist = cv2.calcHist([left], [0], None, [32], [0, 256]).astype(np.float64).ravel()
    right_hist = cv2.calcHist([right], [0], None, [32], [0, 256]).astype(np.float64).ravel()
    left_hist /= left_hist.sum() + 1e-12
    right_hist /= right_hist.sum() + 1e-12
    return float(0.5 * np.sum(((left_hist - right_hist) ** 2) / (left_hist + right_hist + 1e-12)))


def _timeline_metrics(path: Path, frame_count: int, fps: float, config: TimelineConfig) -> dict[str, float | bool]:
    expected_interval_ms = 1000.0 / fps if fps > 0 else 0.0
    drop_interval_ms = max(
        expected_interval_ms * config.drop_interval_factor,
        expected_interval_ms + config.drop_interval_extra_ms,
    )
    max_gap_fail_ms = max(expected_interval_ms * config.max_gap_factor, config.max_gap_floor_ms)
    if frame_count <= 1 or fps <= 0:
        return {
            "pts_monotonic_valid": True,
            "drop_frame_ratio": 0.0,
            "frame_interval_p99_ms": 0.0,
            "max_frame_gap_ms": 0.0,
            "expected_interval_ms": expected_interval_ms,
            "drop_interval_ms": drop_interval_ms,
            "max_gap_fail_ms": max_gap_fail_ms,
        }

    timestamps: list[float] = []
    capture = cv2.VideoCapture(str(path))
    try:
        while len(timestamps) < frame_count:
            ok, _frame = capture.read()
            if not ok:
                break
            timestamps.append(float(capture.get(cv2.CAP_PROP_POS_MSEC)))
    finally:
        capture.release()

    if len(timestamps) < 2 or len(set(round(item, 3) for item in timestamps)) <= 1:
        intervals = [expected_interval_ms] * max(0, frame_count - 1)
        pts_monotonic_valid = True
    else:
        intervals = [timestamps[index] - timestamps[index - 1] for index in range(1, len(timestamps))]
        pts_monotonic_valid = all(interval >= 0 for interval in intervals)

    positive_intervals = [interval for interval in intervals if interval >= 0]
    if not positive_intervals:
        positive_intervals = [0.0]
    drop_count = sum(1 for interval in positive_intervals if interval > drop_interval_ms)
    return {
        "pts_monotonic_valid": pts_monotonic_valid,
        "drop_frame_ratio": drop_count / len(positive_intervals) if positive_intervals else 0.0,
        "frame_interval_p99_ms": _percentile(positive_intervals, 99),
        "max_frame_gap_ms": float(max(positive_intervals)),
        "expected_interval_ms": expected_interval_ms,
        "drop_interval_ms": drop_interval_ms,
        "max_gap_fail_ms": max_gap_fail_ms,
    }


def _keypoint_dataset_has_points(dataset: h5py.Dataset) -> bool:
    if len(dataset.shape) < 3 or dataset.shape[-1] not in {2, 3}:
        return False
    point_count = int(np.prod(dataset.shape[1:-1]))
    return point_count >= 8


def _read_keypoint_data(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None

    try:
        with h5py.File(path, "r") as handle:
            for candidate in ("label/quality_hand", "label/hand_keypoints", "hand_keypoints", "keypoints"):
                if candidate in handle and isinstance(handle[candidate], h5py.Dataset):
                    dataset = handle[candidate]
                    if _keypoint_dataset_has_points(dataset):
                        return np.asarray(dataset[()])

            found: np.ndarray | None = None

            def visit(name: str, obj: h5py.Group | h5py.Dataset) -> None:
                nonlocal found
                lower_name = name.lower()
                keypoint_like_name = any(token in lower_name for token in ("keypoint", "landmark", "quality_hand"))
                if (
                    found is None
                    and keypoint_like_name
                    and isinstance(obj, h5py.Dataset)
                    and _keypoint_dataset_has_points(obj)
                ):
                    found = np.asarray(obj[()])

            handle.visititems(visit)
            return found
    except OSError:
        return None


def _points_for_frame(keypoints: np.ndarray, frame_index: int, width: int, height: int) -> np.ndarray | None:
    if frame_index >= keypoints.shape[0]:
        return None
    points = np.asarray(keypoints[frame_index], dtype=np.float64)
    if points.size == 0 or points.shape[-1] < 2:
        return None
    points = points.reshape(-1, points.shape[-1])[:, :2]
    finite = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])
    points = points[finite]
    if points.size == 0:
        return None
    if np.nanmax(np.abs(points)) <= 1.5:
        points = points * np.array([width, height], dtype=np.float64)
    return points


def _expanded_bbox(points: np.ndarray, scale: float, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1 = np.min(points, axis=0)
    x2, y2 = np.max(points, axis=0)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    half_width = max(1.0, (x2 - x1) * scale / 2.0)
    half_height = max(1.0, (y2 - y1) * scale / 2.0)
    left = max(0, int(np.floor(center_x - half_width)))
    top = max(0, int(np.floor(center_y - half_height)))
    right = min(width, int(np.ceil(center_x + half_width)))
    bottom = min(height, int(np.ceil(center_y + half_height)))
    return left, top, right, bottom


def _compute_hand_roi_metrics(
    path: Path,
    hdf5_path: Path | None,
    indexes: list[int],
    width: int,
    height: int,
    config: HandRoiConfig,
) -> HandRoiMetrics | None:
    if not config.enabled or hdf5_path is None or not hdf5_path.is_file() or not indexes:
        return None

    keypoints = _read_keypoint_data(hdf5_path)
    if keypoints is None:
        return HandRoiMetrics(
            enabled=True,
            source="hdf5_keypoints_bbox",
            sampled_frame_count=len(indexes),
            available_frame_count=0,
            available_ratio=0.0,
            laplacian_p10=0.0,
            laplacian_median=0.0,
            tenengrad_p10=0.0,
            tenengrad_median=0.0,
            blur_bad_frame_ratio=1.0,
            unavailable_reasons=("keypoint_dataset_missing",),
        )

    capture = cv2.VideoCapture(str(path))
    laplacian_values: list[float] = []
    tenengrad_values: list[float] = []
    blur_bad_values: list[float] = []
    unavailable_reasons: list[str] = []
    available = 0
    try:
        for index in indexes:
            points = _points_for_frame(keypoints, index, width, height)
            if points is None or len(points) < config.min_valid_points_for_bbox:
                unavailable_reasons.append("not_enough_keypoints")
                continue

            left, top, right, bottom = _expanded_bbox(points, config.bbox_expand_scale, width, height)
            roi_width = right - left
            roi_height = bottom - top
            roi_area_ratio = (roi_width * roi_height) / max(1, width * height)
            if (
                roi_width < config.min_roi_width_px
                or roi_height < config.min_roi_height_px
                or roi_area_ratio < config.min_roi_area_ratio
            ):
                unavailable_reasons.append("roi_too_small")
                continue

            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                unavailable_reasons.append("roi_frame_decode_failed")
                continue

            roi = frame[top:bottom, left:right]
            if roi.size == 0:
                unavailable_reasons.append("roi_empty")
                continue

            laplacian, tenengrad, _scale = _sharpness_for_frame(
                roi,
                config.roi_target_short_side,
                config.no_upscale_if_roi_too_small,
            )
            laplacian_values.append(laplacian)
            tenengrad_values.append(tenengrad)
            blur_bad_values.append(
                1.0 if laplacian < config.laplacian_p10_warn or tenengrad < config.tenengrad_p10_warn else 0.0
            )
            available += 1
    finally:
        capture.release()

    sampled = len(indexes)
    return HandRoiMetrics(
        enabled=True,
        source="hdf5_keypoints_bbox",
        sampled_frame_count=sampled,
        available_frame_count=available,
        available_ratio=available / sampled if sampled else 0.0,
        laplacian_p10=_percentile(laplacian_values, 10),
        laplacian_median=_percentile(laplacian_values, 50),
        tenengrad_p10=_percentile(tenengrad_values, 10),
        tenengrad_median=_percentile(tenengrad_values, 50),
        blur_bad_frame_ratio=float(np.mean(blur_bad_values)) if blur_bad_values else 1.0,
        unavailable_reasons=tuple(dict.fromkeys(unavailable_reasons)),
    )


def _empty_metrics(path: Path, errors: tuple[str, ...]) -> VideoMetrics:
    return VideoMetrics(
        path=path,
        asset_id=asset_id_from_video(path),
        opened=False,
        video_stream_present=False,
        codec_readable=False,
        metadata_read_ok=False,
        frame_count=0,
        fps=0.0,
        duration_seconds=0.0,
        width=0,
        height=0,
        display_width=0,
        display_height=0,
        short_side=0,
        long_side=0,
        sampled_frame_count=0,
        decoded_sample_count=0,
        sample_decode_ratio=0.0,
        mean_brightness=0.0,
        black_frame_ratio=1.0,
        mean_over_dark_ratio=1.0,
        mean_over_exposed_ratio=0.0,
        laplacian_min=0.0,
        laplacian_p10=0.0,
        laplacian_median=0.0,
        mean_blur_laplacian_var=0.0,
        laplacian_p90=0.0,
        laplacian_under_100_ratio=1.0,
        tenengrad_p10=0.0,
        tenengrad_median=0.0,
        tenengrad_mean=0.0,
        sharpness_scale_short_side=0,
        frozen_frame_ratio=0.0,
        max_consecutive_frozen_sec=0.0,
        pts_monotonic_valid=False,
        drop_frame_ratio=1.0,
        frame_interval_p99_ms=0.0,
        max_frame_gap_ms=0.0,
        expected_interval_ms=0.0,
        drop_interval_ms=0.0,
        max_gap_fail_ms=0.0,
        errors=errors,
    )


def analyze_video(path: Path, config: VideoQualityConfig, hdf5_path: Path | None = None) -> VideoMetrics:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return _empty_metrics(path, ("cannot_open_video",))

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if frame_count > 0 and fps > 0 else 0.0
        display_width = width
        display_height = height
        short_side = min(display_width, display_height) if display_width and display_height else 0
        long_side = max(display_width, display_height) if display_width and display_height else 0
        indexes = _sample_indexes(frame_count, fps, duration, config.decode)
        timeline = _timeline_metrics(path, frame_count, fps, config.timeline)

        brightness_values: list[float] = []
        black_values: list[float] = []
        dark_values: list[float] = []
        exposed_values: list[float] = []
        blur_values: list[float] = []
        tenengrad_values: list[float] = []
        scale_values: list[int] = []
        frozen_pairs = 0
        max_frozen_run = 0
        current_frozen_run = 0
        previous_gray: np.ndarray | None = None
        errors: list[str] = []

        for index in indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                errors.append(f"sample_decode_failed:{index}")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))
            dark_pixel_ratio = float(np.mean(gray < DARK_PIXEL_Y_THRESHOLD))
            over_exposed_pixel_ratio = float(np.mean(gray > OVER_EXPOSED_PIXEL_Y_THRESHOLD))
            brightness_values.append(brightness)
            black_values.append(
                1.0
                if brightness < config.exposure.black.mean_y_max
                or dark_pixel_ratio > config.exposure.black.dark_pixel_ratio_min
                else 0.0
            )
            dark_values.append(
                1.0
                if brightness < config.exposure.over_dark.mean_y_max
                or dark_pixel_ratio > config.exposure.over_dark.dark_pixel_ratio_min
                else 0.0
            )
            exposed_values.append(
                1.0
                if brightness > config.exposure.over_exposed.mean_y_min
                or over_exposed_pixel_ratio > config.exposure.over_exposed.over_exposed_pixel_ratio_min
                else 0.0
            )

            laplacian, tenengrad, scale_short_side = _sharpness_for_frame(
                frame,
                config.sharpness_global.target_short_side,
                config.sharpness_global.no_upscale,
            )
            blur_values.append(laplacian)
            tenengrad_values.append(tenengrad)
            scale_values.append(scale_short_side)

            if config.freeze.enabled:
                freeze_frame, _freeze_scale = _resize_keep_aspect(frame, config.freeze.downscale_short_side, no_upscale=True)
                freeze_gray = cv2.cvtColor(freeze_frame, cv2.COLOR_BGR2GRAY)
                if previous_gray is not None:
                    diff = float(np.mean(cv2.absdiff(previous_gray, freeze_gray)))
                    hist_diff = _chi_square_hist_diff(previous_gray, freeze_gray)
                    if diff < config.freeze.frame_diff_mean_abs_max and hist_diff < config.freeze.hist_diff_max:
                        frozen_pairs += 1
                        current_frozen_run += 1
                        max_frozen_run = max(max_frozen_run, current_frozen_run)
                    else:
                        current_frozen_run = 0
                previous_gray = freeze_gray

        decoded = len(brightness_values)
        sampled = len(indexes)
        if sampled and decoded == 0:
            errors.append("no_sample_frames_decoded")
        laplacian_under_100_ratio = (
            float(np.mean(np.array(blur_values) < LAPLACIAN_LOW_DETAIL_THRESHOLD)) if blur_values else 1.0
        )
        metadata_read_ok = frame_count > 0 and fps > 0 and width > 0 and height > 0
        hand_roi = _compute_hand_roi_metrics(path, hdf5_path, indexes, width, height, config.hand_roi)

        return VideoMetrics(
            path=path,
            asset_id=asset_id_from_video(path),
            opened=True,
            video_stream_present=frame_count > 0,
            codec_readable=metadata_read_ok,
            metadata_read_ok=metadata_read_ok,
            frame_count=frame_count,
            fps=fps,
            duration_seconds=duration,
            width=width,
            height=height,
            display_width=display_width,
            display_height=display_height,
            short_side=short_side,
            long_side=long_side,
            sampled_frame_count=sampled,
            decoded_sample_count=decoded,
            sample_decode_ratio=decoded / sampled if sampled else 0.0,
            mean_brightness=float(np.mean(brightness_values)) if brightness_values else 0.0,
            black_frame_ratio=float(np.mean(black_values)) if black_values else 1.0,
            mean_over_dark_ratio=float(np.mean(dark_values)) if dark_values else 1.0,
            mean_over_exposed_ratio=float(np.mean(exposed_values)) if exposed_values else 0.0,
            laplacian_min=float(min(blur_values)) if blur_values else 0.0,
            laplacian_p10=_percentile(blur_values, 10),
            laplacian_median=_percentile(blur_values, 50),
            mean_blur_laplacian_var=float(np.mean(blur_values)) if blur_values else 0.0,
            laplacian_p90=_percentile(blur_values, 90),
            laplacian_under_100_ratio=laplacian_under_100_ratio,
            tenengrad_p10=_percentile(tenengrad_values, 10),
            tenengrad_median=_percentile(tenengrad_values, 50),
            tenengrad_mean=float(np.mean(tenengrad_values)) if tenengrad_values else 0.0,
            sharpness_scale_short_side=max(scale_values) if scale_values else short_side,
            frozen_frame_ratio=frozen_pairs / (decoded - 1) if decoded > 1 and config.freeze.enabled else 0.0,
            max_consecutive_frozen_sec=max_frozen_run / fps if fps > 0 else 0.0,
            pts_monotonic_valid=bool(timeline["pts_monotonic_valid"]),
            drop_frame_ratio=float(timeline["drop_frame_ratio"]),
            frame_interval_p99_ms=float(timeline["frame_interval_p99_ms"]),
            max_frame_gap_ms=float(timeline["max_frame_gap_ms"]),
            expected_interval_ms=float(timeline["expected_interval_ms"]),
            drop_interval_ms=float(timeline["drop_interval_ms"]),
            max_gap_fail_ms=float(timeline["max_gap_fail_ms"]),
            hand_roi=hand_roi,
            errors=tuple(errors),
        )
    finally:
        capture.release()


def hdf5_path_for_video(video_path: Path, batch_dir: Path) -> Path:
    return batch_dir / "hdf5" / f"{asset_id_from_video(video_path)}_hdf5.hdf5"


def infer_hdf5_frame_count(path: Path) -> int:
    with h5py.File(path, "r") as handle:
        if "label/quality_hand" not in handle:
            raise ValueError("missing label/quality_hand")
        dataset = handle["label/quality_hand"]
        if not dataset.shape:
            raise ValueError("label/quality_hand is scalar")
        return int(dataset.shape[0])


def check_hdf5_alignment(
    video_path: Path,
    batch_dir: Path,
    metrics: VideoMetrics,
    config: VideoQualityConfig,
) -> Hdf5Alignment:
    hdf5_path = hdf5_path_for_video(video_path, batch_dir)
    if not config.hdf5_alignment.enabled:
        return Hdf5Alignment("disabled", hdf5_path, None, None, None, None, None)
    if not hdf5_path.is_file():
        return Hdf5Alignment("missing", hdf5_path, None, None, None, None, "hdf5_missing")

    try:
        hdf5_frame_count = infer_hdf5_frame_count(hdf5_path)
    except (OSError, ValueError) as exc:
        return Hdf5Alignment("unreadable", hdf5_path, None, None, None, None, f"hdf5_unreadable:{exc}")

    delta = abs(hdf5_frame_count - metrics.frame_count)
    denominator = max(metrics.frame_count, hdf5_frame_count, 1)
    delta_ratio = delta / denominator
    matches = delta == 0
    return Hdf5Alignment(
        "matched" if matches else "mismatch",
        hdf5_path,
        hdf5_frame_count,
        matches,
        delta,
        delta_ratio,
        None if matches else "hdf5_frame_count_mismatch",
    )


def _add_threshold_reason(
    fail: list[str],
    warn: list[str],
    value: float,
    pass_limit: float,
    warn_limit: float,
    fail_reason: str,
    warn_reason: str,
    higher_is_bad: bool,
) -> None:
    if higher_is_bad:
        if value > warn_limit:
            fail.append(fail_reason)
        elif value > pass_limit:
            warn.append(warn_reason)
    else:
        if value < warn_limit:
            fail.append(fail_reason)
        elif value < pass_limit:
            warn.append(warn_reason)


def _alignment_exceeds(alignment: Hdf5Alignment, frame_limit: int, ratio_limit: float) -> bool:
    return bool(
        alignment.frame_count_delta is not None
        and alignment.frame_count_delta_ratio is not None
        and (alignment.frame_count_delta > frame_limit or alignment.frame_count_delta_ratio > ratio_limit)
    )


def _make_evaluation(fail_reasons: list[str], warn_reasons: list[str]) -> QualityEvaluation:
    unique_fail = tuple(dict.fromkeys(fail_reasons))
    unique_warn = tuple(reason for reason in dict.fromkeys(warn_reasons) if reason not in unique_fail)
    decision = "fail" if unique_fail else "warn" if unique_warn else "pass"
    should_run_mask_qc = decision != "fail"
    return QualityEvaluation(
        decision=decision,
        passed=should_run_mask_qc,
        reasons=unique_fail,
        warn_reasons=unique_warn,
        should_run_mask_qc=should_run_mask_qc,
    )


def evaluate_video_quality(
    metrics: VideoMetrics,
    config: VideoQualityConfig,
    alignment: object | None = None,
) -> QualityEvaluation:
    fail: list[str] = list(metrics.errors)
    warn: list[str] = []

    if not metrics.opened:
        fail.append("video_not_opened")
    if not metrics.video_stream_present:
        fail.append("video_stream_missing")
    if not metrics.codec_readable:
        fail.append("codec_unreadable")
    if not metrics.metadata_read_ok:
        fail.append("metadata_unreadable")

    fps = config.fps
    if metrics.fps < fps.min_fps_fail:
        fail.append("fps_below_min")
    elif metrics.fps < fps.min_fps_pass:
        warn.append("fps_below_pass")
    if fps.expected_fps is not None and metrics.fps < fps.expected_fps * 0.95:
        warn.append("fps_below_expected")

    resolution = config.resolution
    if metrics.short_side < resolution.min_short_side_fail:
        fail.append("short_side_below_min")
    if metrics.long_side < resolution.min_long_side_fail:
        fail.append("long_side_below_min")

    if config.timeline.pts_monotonic_required and not metrics.pts_monotonic_valid:
        fail.append("pts_monotonic_invalid")
    _add_threshold_reason(
        fail,
        warn,
        metrics.drop_frame_ratio,
        config.timeline.drop_frame_ratio_pass,
        config.timeline.drop_frame_ratio_warn,
        "drop_frame_ratio_above_max",
        "drop_frame_ratio_warn",
        higher_is_bad=True,
    )
    if metrics.expected_interval_ms > 0 and metrics.frame_interval_p99_ms > metrics.expected_interval_ms * 2:
        fail.append("frame_interval_p99_ms_above_max")
    elif metrics.frame_interval_p99_ms > metrics.drop_interval_ms:
        warn.append("frame_interval_p99_ms_warn")
    if metrics.max_frame_gap_ms > metrics.max_gap_fail_ms:
        fail.append("max_frame_gap_ms_above_max")

    _add_threshold_reason(
        fail,
        warn,
        metrics.sample_decode_ratio,
        config.decode.sample_decode_ratio_pass,
        config.decode.sample_decode_ratio_warn,
        "sample_decode_ratio_below_min",
        "sample_decode_ratio_warn",
        higher_is_bad=False,
    )
    _add_threshold_reason(
        fail,
        warn,
        metrics.black_frame_ratio,
        config.exposure.black.ratio_pass,
        config.exposure.black.ratio_warn,
        "black_frame_ratio_above_max",
        "black_frame_ratio_warn",
        higher_is_bad=True,
    )
    _add_threshold_reason(
        fail,
        warn,
        metrics.mean_over_dark_ratio,
        config.exposure.over_dark.ratio_pass,
        config.exposure.over_dark.ratio_warn,
        "mean_over_dark_ratio_above_max",
        "mean_over_dark_ratio_warn",
        higher_is_bad=True,
    )
    _add_threshold_reason(
        fail,
        warn,
        metrics.mean_over_exposed_ratio,
        config.exposure.over_exposed.ratio_pass,
        config.exposure.over_exposed.ratio_warn,
        "mean_over_exposed_ratio_above_max",
        "mean_over_exposed_ratio_warn",
        higher_is_bad=True,
    )

    sharpness = config.sharpness_global
    _add_threshold_reason(
        fail,
        warn,
        metrics.laplacian_p10,
        sharpness.laplacian_p10_pass,
        sharpness.laplacian_p10_warn,
        "laplacian_p10_below_min",
        "laplacian_p10_warn",
        higher_is_bad=False,
    )
    _add_threshold_reason(
        fail,
        warn,
        metrics.laplacian_median,
        sharpness.laplacian_median_pass,
        sharpness.laplacian_median_warn,
        "laplacian_median_below_min",
        "laplacian_median_warn",
        higher_is_bad=False,
    )
    _add_threshold_reason(
        fail,
        warn,
        metrics.laplacian_under_100_ratio,
        sharpness.laplacian_under_100_ratio_pass,
        sharpness.laplacian_under_100_ratio_warn,
        "laplacian_under_100_ratio_above_max",
        "laplacian_under_100_ratio_warn",
        higher_is_bad=True,
    )
    _add_threshold_reason(
        fail,
        warn,
        metrics.tenengrad_p10,
        sharpness.tenengrad_p10_pass,
        sharpness.tenengrad_p10_warn,
        "tenengrad_p10_below_min",
        "tenengrad_p10_warn",
        higher_is_bad=False,
    )
    _add_threshold_reason(
        fail,
        warn,
        metrics.tenengrad_median,
        sharpness.tenengrad_median_pass,
        sharpness.tenengrad_median_warn,
        "tenengrad_median_below_min",
        "tenengrad_median_warn",
        higher_is_bad=False,
    )

    if config.freeze.enabled:
        _add_threshold_reason(
            fail,
            warn,
            metrics.frozen_frame_ratio,
            config.freeze.frozen_frame_ratio_pass,
            config.freeze.frozen_frame_ratio_warn,
            "frozen_frame_ratio_above_max",
            "frozen_frame_ratio_warn",
            higher_is_bad=True,
        )
        if metrics.max_consecutive_frozen_sec > config.freeze.max_consecutive_frozen_sec_fail:
            fail.append("max_consecutive_frozen_sec_above_max")
        elif metrics.max_consecutive_frozen_sec > config.freeze.max_consecutive_frozen_sec_pass:
            warn.append("max_consecutive_frozen_sec_warn")

    if isinstance(alignment, Hdf5Alignment) and config.hdf5_alignment.mode != AlignmentMode.IGNORE:
        if alignment.status == "missing":
            (fail if config.hdf5_alignment.mode == AlignmentMode.FAIL else warn).append("hdf5_missing")
        elif alignment.status == "unreadable":
            (fail if config.hdf5_alignment.mode == AlignmentMode.FAIL else warn).append("hdf5_unreadable")
        elif alignment.status == "mismatch":
            severe = _alignment_exceeds(
                alignment,
                config.hdf5_alignment.max_delta_frames_warn,
                config.hdf5_alignment.max_delta_ratio_warn,
            )
            warning = _alignment_exceeds(
                alignment,
                config.hdf5_alignment.max_delta_frames_pass,
                config.hdf5_alignment.max_delta_ratio_pass,
            )
            if severe and config.hdf5_alignment.mode == AlignmentMode.FAIL:
                fail.append("hdf5_frame_count_mismatch")
            elif severe or warning:
                warn.append("hdf5_frame_count_mismatch_warn")

    roi = metrics.hand_roi
    if roi is not None and config.hand_roi.enabled:
        if roi.available_ratio < config.hand_roi.available_ratio_warn:
            warn.append("hand_roi_available_ratio_below_min")
        elif roi.available_ratio < config.hand_roi.available_ratio_pass:
            warn.append("hand_roi_available_ratio_warn")

        if roi.available_frame_count > 0:
            severe = config.hand_roi.severe_fail
            if severe.enabled:
                lap_fail = roi.laplacian_p10 < severe.laplacian_p10_fail
                ten_fail = roi.tenengrad_p10 < severe.tenengrad_p10_fail
                if severe.require_both_lap_and_ten_fail and lap_fail and ten_fail:
                    fail.append("hand_roi_severe_blur")
                elif not severe.require_both_lap_and_ten_fail and (lap_fail or ten_fail):
                    fail.append("hand_roi_severe_blur")
                if roi.blur_bad_frame_ratio > severe.blur_bad_frame_ratio_fail:
                    fail.append("hand_roi_blur_bad_frame_ratio_above_max")

            _add_threshold_reason(
                fail,
                warn,
                roi.laplacian_p10,
                config.hand_roi.laplacian_p10_pass,
                config.hand_roi.laplacian_p10_warn,
                "hand_roi_laplacian_p10_below_min",
                "hand_roi_laplacian_p10_warn",
                higher_is_bad=False,
            )
            _add_threshold_reason(
                fail,
                warn,
                roi.laplacian_median,
                config.hand_roi.laplacian_median_pass,
                config.hand_roi.laplacian_median_warn,
                "hand_roi_laplacian_median_below_min",
                "hand_roi_laplacian_median_warn",
                higher_is_bad=False,
            )
            _add_threshold_reason(
                fail,
                warn,
                roi.tenengrad_p10,
                config.hand_roi.tenengrad_p10_pass,
                config.hand_roi.tenengrad_p10_warn,
                "hand_roi_tenengrad_p10_below_min",
                "hand_roi_tenengrad_p10_warn",
                higher_is_bad=False,
            )
            _add_threshold_reason(
                fail,
                warn,
                roi.tenengrad_median,
                config.hand_roi.tenengrad_median_pass,
                config.hand_roi.tenengrad_median_warn,
                "hand_roi_tenengrad_median_below_min",
                "hand_roi_tenengrad_median_warn",
                higher_is_bad=False,
            )
            _add_threshold_reason(
                fail,
                warn,
                roi.blur_bad_frame_ratio,
                config.hand_roi.blur_bad_frame_ratio_pass,
                config.hand_roi.blur_bad_frame_ratio_warn,
                "hand_roi_blur_bad_frame_ratio_above_max",
                "hand_roi_blur_bad_frame_ratio_warn",
                higher_is_bad=True,
            )

    return _make_evaluation(fail, warn)


def _hdf5_object_path(name: str) -> str:
    return "/" if not name else f"/{name}"


def _parse_json_text_if_possible(text: str) -> Any:
    stripped = text.strip()
    if not stripped.startswith(("{", "[")):
        return text
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return text


def _decode_hdf5_text(value: Any) -> Any | None:
    if isinstance(value, bytes | np.bytes_):
        return _parse_json_text_if_possible(bytes(value).decode("utf-8", errors="replace"))
    if isinstance(value, str | np.str_):
        return _parse_json_text_if_possible(str(value))
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in {"S", "U", "O"}:
            return None
        return _decode_hdf5_text(value.tolist())
    if isinstance(value, list | tuple):
        decoded = [_decode_hdf5_text(item) for item in value]
        if any(item is None for item in decoded):
            return None
        return decoded
    return None


def _collect_hdf5_attrs(obj: h5py.Group | h5py.Dataset, _path: str) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for key, value in obj.attrs.items():
        decoded = _decode_hdf5_text(value)
        if decoded is not None:
            attrs[str(key)] = decoded
    return attrs


def read_hdf5_text_fields(path: Path | None) -> dict[str, dict[str, Any]]:
    text_fields: dict[str, dict[str, Any]] = {"attributes": {}, "datasets": {}}
    if path is None or not path.is_file():
        return text_fields

    try:
        with h5py.File(path, "r") as handle:
            root_attrs = _collect_hdf5_attrs(handle, "/")
            if root_attrs:
                text_fields["attributes"]["/"] = root_attrs

            def visit(name: str, obj: h5py.Group | h5py.Dataset) -> None:
                object_path = _hdf5_object_path(name)
                attrs = _collect_hdf5_attrs(obj, object_path)
                if attrs:
                    text_fields["attributes"][object_path] = attrs
                if isinstance(obj, h5py.Dataset):
                    decoded = _decode_hdf5_text(obj[()])
                    if decoded is not None:
                        text_fields["datasets"][object_path] = decoded

            handle.visititems(visit)
    except OSError:
        return text_fields

    return text_fields


def _evaluation_json(evaluation: QualityEvaluation) -> dict[str, Any]:
    return {
        "decision": evaluation.decision,
        "passed": evaluation.passed,
        "reasons": list(evaluation.reasons),
        "warn_reasons": list(evaluation.warn_reasons),
        "should_run_mask_qc": evaluation.should_run_mask_qc,
    }


def _hand_roi_json(metrics: HandRoiMetrics | None) -> dict[str, Any] | None:
    if metrics is None:
        return None
    return {
        "enabled": metrics.enabled,
        "source": metrics.source,
        "hand_roi_available_ratio": metrics.available_ratio,
        "hand_roi_available_frame_count": metrics.available_frame_count,
        "hand_roi_sampled_frame_count": metrics.sampled_frame_count,
        "hand_roi_laplacian_p10": metrics.laplacian_p10,
        "hand_roi_laplacian_median": metrics.laplacian_median,
        "hand_roi_tenengrad_p10": metrics.tenengrad_p10,
        "hand_roi_tenengrad_median": metrics.tenengrad_median,
        "hand_roi_blur_bad_frame_ratio": metrics.blur_bad_frame_ratio,
        "unavailable_reasons": list(metrics.unavailable_reasons),
    }


def _infer_batch_dir(metrics: VideoMetrics, alignment: Hdf5Alignment) -> Path | None:
    if alignment.hdf5_path is not None and alignment.hdf5_path.parent.name == "hdf5":
        return alignment.hdf5_path.parent.parent
    if metrics.path.parent.name == "video":
        return metrics.path.parent.parent
    return None


def _archive_path(path: Path | None, batch_dir: Path | None) -> str | None:
    if path is None:
        return None
    if batch_dir is None:
        return str(path)
    try:
        return str(path.relative_to(batch_dir))
    except ValueError:
        return str(path)


def asset_qc_result_to_json(result: VideoQualityResult, config: VideoQualityConfig) -> dict[str, Any]:
    metrics = result.metrics
    alignment = result.alignment
    hdf5_exists = alignment.hdf5_path.is_file() if alignment.hdf5_path is not None else False
    failed_modules = [] if result.evaluation.passed else ["video_quality"]
    evaluation = _evaluation_json(result.evaluation)
    batch_dir = _infer_batch_dir(metrics, alignment)

    video_basic = {
        "video_open_ok": metrics.opened,
        "video_stream_present": metrics.video_stream_present,
        "codec_readable": metrics.codec_readable,
        "metadata_read_ok": metrics.metadata_read_ok,
        "fps": metrics.fps,
        "width": metrics.width,
        "height": metrics.height,
        "display_width": metrics.display_width,
        "display_height": metrics.display_height,
        "short_side": metrics.short_side,
        "long_side": metrics.long_side,
        "frame_count": metrics.frame_count,
        "duration_seconds": metrics.duration_seconds,
    }
    timeline_metrics = {
        "pts_monotonic_valid": metrics.pts_monotonic_valid,
        "drop_frame_ratio": metrics.drop_frame_ratio,
        "frame_interval_p99_ms": metrics.frame_interval_p99_ms,
        "max_frame_gap_ms": metrics.max_frame_gap_ms,
        "expected_interval_ms": metrics.expected_interval_ms,
        "drop_interval_ms": metrics.drop_interval_ms,
        "max_gap_fail_ms": metrics.max_gap_fail_ms,
    }
    decode_metrics = {
        "sample_decode_ratio": metrics.sample_decode_ratio,
        "sampled_frame_count": metrics.sampled_frame_count,
        "decoded_sample_count": metrics.decoded_sample_count,
    }
    exposure_metrics = {
        "mean_brightness": metrics.mean_brightness,
        "black_frame_ratio": metrics.black_frame_ratio,
        "mean_over_dark_ratio": metrics.mean_over_dark_ratio,
        "mean_over_exposed_ratio": metrics.mean_over_exposed_ratio,
    }
    sharpness_global = {
        "sharpness_scale_short_side": metrics.sharpness_scale_short_side,
        "laplacian_min": metrics.laplacian_min,
        "laplacian_p10": metrics.laplacian_p10,
        "laplacian_median": metrics.laplacian_median,
        "mean_blur_laplacian_var": metrics.mean_blur_laplacian_var,
        "laplacian_p90": metrics.laplacian_p90,
        "laplacian_under_100_ratio": metrics.laplacian_under_100_ratio,
        "tenengrad_p10": metrics.tenengrad_p10,
        "tenengrad_median": metrics.tenengrad_median,
        "tenengrad_mean": metrics.tenengrad_mean,
    }
    freeze_metrics = {
        "frozen_frame_ratio": metrics.frozen_frame_ratio,
        "max_consecutive_frozen_sec": metrics.max_consecutive_frozen_sec,
    }
    hdf5_alignment = {
        "enabled": config.hdf5_alignment.enabled,
        "mode": config.hdf5_alignment.mode.value,
        "status": alignment.status,
        "video_frame_count": metrics.frame_count,
        "hdf5_frame_count": alignment.hdf5_frame_count,
        "frame_count_delta": alignment.frame_count_delta,
        "frame_count_delta_ratio": alignment.frame_count_delta_ratio,
        "frame_count_match": alignment.frame_count_match,
        "reason": alignment.reason,
    }
    hand_roi_metrics = _hand_roi_json(metrics.hand_roi)

    return {
        "schema_version": "asset_qc_report.v1",
        "asset_id": metrics.asset_id,
        "source_files": {
            "video": {
                "path": _archive_path(metrics.path, batch_dir),
                "filename": metrics.path.name,
                "extension": metrics.path.suffix.lower(),
            },
            "hdf5": {
                "path": _archive_path(alignment.hdf5_path, batch_dir),
                "exists": hdf5_exists,
            },
        },
        "qc_summary": {
            "status": result.evaluation.decision,
            "passed": result.evaluation.passed,
            "completed_modules": ["video_quality"],
            "failed_modules": failed_modules,
            "reasons": list(result.evaluation.reasons),
            "warn_reasons": list(result.evaluation.warn_reasons),
            "should_run_mask_qc": result.evaluation.should_run_mask_qc,
        },
        "hdf5_text_info": {
            "alignment": hdf5_alignment,
            "text_fields": read_hdf5_text_fields(alignment.hdf5_path),
        },
        "video_quality": {
            "stage": "video_prefilter",
            "threshold_version": config.threshold_version,
            "evaluation": evaluation,
            "metadata": {
                "opened": metrics.opened,
                "frame_count": metrics.frame_count,
                "fps": metrics.fps,
                "duration_seconds": metrics.duration_seconds,
                "width": metrics.width,
                "height": metrics.height,
                "short_side": metrics.short_side,
                "long_side": metrics.long_side,
            },
            "sampling": {
                "sample_count_configured": config.decode.max_sample_frames,
                "sampled_frame_count": metrics.sampled_frame_count,
                "decoded_sample_count": metrics.decoded_sample_count,
                "sample_decode_ratio": metrics.sample_decode_ratio,
            },
            "metrics": {
                "video_basic": video_basic,
                "timeline_metrics": timeline_metrics,
                "decode_metrics": decode_metrics,
                "exposure_metrics": exposure_metrics,
                "exposure": exposure_metrics,
                "sharpness_global": sharpness_global,
                "freeze_metrics": freeze_metrics,
                "hdf5_alignment": hdf5_alignment,
                "hand_roi_metrics": hand_roi_metrics,
            },
            "thresholds": config.to_dict(),
            "errors": list(metrics.errors),
        },
        "reference_quality": {
            "mode": "none",
            "reference_video_path": None,
            "vmaf": None,
            "note": "当前无标准对照视频，未计算 VMAF。",
        },
    }


def write_per_asset_qc_json_reports(
    quality_archive_dir: Path,
    results: list[VideoQualityResult],
    config: VideoQualityConfig,
) -> None:
    quality_archive_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        path = quality_archive_dir / f"{result.metrics.asset_id}.json"
        path.write_text(
            json.dumps(asset_qc_result_to_json(result, config), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def run_video_quality_check(
    batch_dir: Path,
    config_path: Path | None = None,
) -> int:
    config = load_video_quality_config(config_path)
    video_paths = discover_batch_videos(batch_dir)
    results: list[VideoQualityResult] = []

    for video_path in video_paths:
        hdf5_path = hdf5_path_for_video(video_path, batch_dir)
        metrics = analyze_video(video_path, config, hdf5_path=hdf5_path)
        alignment = check_hdf5_alignment(video_path, batch_dir, metrics, config)
        evaluation = evaluate_video_quality(metrics, config, alignment)
        results.append(VideoQualityResult(metrics, alignment, evaluation))

    write_per_asset_qc_json_reports(batch_dir / "quality_archive", results, config)
    return 0 if all(result.evaluation.passed for result in results) else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    return run_video_quality_check(args.batch, args.config)
