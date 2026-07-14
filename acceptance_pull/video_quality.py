from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
from qc_common.config import LoadedQcConfig, load_qc_acceptance_config
from qc_common.report import load_asset_qc_report
from qc_pipeline.context import AssetContext


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi"}
LAPLACIAN_LOW_DETAIL_THRESHOLD = 100.0
DARK_PIXEL_Y_THRESHOLD = 20
OVER_EXPOSED_PIXEL_Y_THRESHOLD = 245
LOGGER = logging.getLogger(__name__)


class AlignmentMode(StrEnum):
    IGNORE = "ignore"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class PipelineConfig:
    stop_before_mask_if_fail: bool = True
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
    drop_frame_ratio_pass: float = 0.05
    drop_frame_ratio_warn: float = 0.10
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
    ratio_pass: float = 0.01
    ratio_warn: float = 0.90
    max_frame_count_fail: int = 10


@dataclass(frozen=True)
class OverDarkConfig:
    mean_y_max: float = 35.0
    dark_pixel_ratio_min: float = 0.75
    ratio_pass: float = 0.05
    ratio_warn: float = 0.90


@dataclass(frozen=True)
class OverExposedConfig:
    mean_y_min: float = 235.0
    over_exposed_pixel_ratio_min: float = 0.35
    ratio_pass: float = 0.05
    ratio_warn: float = 0.90


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
    laplacian_p10_pass: float = 15.0
    laplacian_p10_warn: float = 0.0
    laplacian_median_pass: float = 20.0
    laplacian_median_warn: float = 0.0
    laplacian_under_100_ratio_pass: float = 1.00
    laplacian_under_100_ratio_warn: float = 1.00
    tenengrad_p10_pass: float = 6.0
    tenengrad_p10_warn: float = 4.0
    tenengrad_median_pass: float = 7.0
    tenengrad_median_warn: float = 4.0


@dataclass(frozen=True)
class FreezeConfig:
    enabled: bool = True
    downscale_short_side: int = 360
    frame_diff_mean_abs_max: float = 1.0
    hist_diff_max: float = 0.01
    ssim_min: float = 0.995
    phash_hamming_max: int = 4
    freeze_candidate_window_sec: float = 0.5
    confirmed_freeze_window_sec: float = 1.0
    adjacent_near_duplicate_ratio_warn: float = 0.90
    frozen_frame_ratio_pass: float = 0.05
    frozen_frame_ratio_warn: float = 0.10
    min_interval_frames: int = 6
    min_interval_duration_ms: float = 100.0
    max_consecutive_frozen_sec_pass: float = 0.5
    max_consecutive_frozen_sec_fail: float = 1.0
    motion_conflict_enabled: bool = True
    motion_keypoint_delta_normalized_min: float = 0.02
    motion_keypoint_delta_px_min: float = 12.0
    motion_numeric_delta_min: float = 0.01
    critical_window_enabled: bool = True
    critical_window_keywords: tuple[str, ...] = (
        "grasp",
        "place",
        "contact",
        "hand-object",
        "hand object",
        "interaction",
        "抓取",
        "放置",
        "接触",
        "交互",
    )
    critical_window_interval_duration_ms_fail: float = 100.0
    video_state_conflict_noncritical_duration_ms_fail: float = 1000.0
    video_state_conflict_critical_duration_ms_fail: float = 500.0


@dataclass(frozen=True)
class DefectDurationConfig:
    max_duration_ratio_fail: float = 0.10
    duration_ratio_warn: float = 0.05


@dataclass(frozen=True)
class Hdf5AlignmentConfig:
    enabled: bool = True
    mode: AlignmentMode = AlignmentMode.FAIL
    max_delta_frames_pass: int = 2
    max_delta_frames_warn: int = 5
    max_delta_ratio_pass: float = 0.001
    max_delta_ratio_warn: float = 0.005


@dataclass(frozen=True)
class VideoQualityConfig:
    module_version: str = ""
    qc_config_reference: dict[str, str] = field(default_factory=dict)
    next_module: str = ""
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    fps: FpsConfig = field(default_factory=FpsConfig)
    resolution: ResolutionConfig = field(default_factory=ResolutionConfig)
    timeline: TimelineConfig = field(default_factory=TimelineConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    exposure: ExposureConfig = field(default_factory=ExposureConfig)
    sharpness_global: SharpnessGlobalConfig = field(default_factory=SharpnessGlobalConfig)
    freeze: FreezeConfig = field(default_factory=FreezeConfig)
    defects: DefectDurationConfig = field(default_factory=DefectDurationConfig)
    hdf5_alignment: Hdf5AlignmentConfig = field(default_factory=Hdf5AlignmentConfig)

    @property
    def sample_count(self) -> int:
        return self.decode.max_sample_frames

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)


@dataclass(frozen=True)
class FrozenInterval:
    start_frame: int
    end_frame: int
    frame_count: int
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    duration_ms: float
    mean_frame_diff: float = 0.0
    max_frame_diff: float = 0.0
    mean_hist_diff: float = 0.0
    max_hist_diff: float = 0.0
    mean_ssim: float = 0.0
    min_ssim: float = 0.0
    mean_phash_hamming: float = 0.0
    max_phash_hamming: int = 0
    motion_conflict: bool = False
    motion_conflict_signals: tuple[str, ...] = ()
    critical_window: bool = False
    critical_keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class FrameTimestamps:
    timestamps_ms: tuple[float, ...]
    source: str
    reliable: bool


@dataclass(frozen=True)
class FreezeScan:
    frozen_intervals: tuple[FrozenInterval, ...]
    adjacent_near_duplicate_count: int
    adjacent_near_duplicate_ratio: float
    freeze_candidate_intervals: tuple[FrozenInterval, ...]
    freeze_candidate_frame_count: int
    freeze_candidate_duration_sec: float
    freeze_candidate_ratio: float
    confirmed_freeze_frame_count: int
    confirmed_freeze_duration_sec: float
    confirmed_freeze_ratio: float
    max_confirmed_freeze_sec: float


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
    black_frame_count_estimate: int
    exposure_defect_frame_ratio: float
    defect_duration_ratio: float
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
    adjacent_near_duplicate_count: int
    adjacent_near_duplicate_ratio: float
    freeze_candidate_ratio: float
    freeze_candidate_frame_count: int
    freeze_candidate_duration_sec: float
    confirmed_freeze_ratio: float
    confirmed_freeze_frame_count: int
    confirmed_freeze_duration_sec: float
    frozen_frame_ratio: float
    max_consecutive_frozen_sec: float
    frozen_intervals: tuple[FrozenInterval, ...]
    pts_monotonic_valid: bool
    drop_frame_ratio: float
    drop_detection_source: str
    drop_detection_reliable: bool
    estimated_missing_frames: int
    observed_frame_interval_count: int
    frame_interval_p99_ms: float
    max_frame_gap_ms: float
    expected_interval_ms: float
    drop_interval_ms: float
    max_gap_fail_ms: float
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class IssueDetail:
    issue_id: str
    code: str
    severity: str
    module: str
    issue_type: str
    metric: str | None = None
    observed_value: Any | None = None
    operator: str | None = None
    boundary_value: Any | None = None
    rule_id: str = ""
    needs_manual_review: bool = False
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QualityEvaluation:
    decision: str
    passed: bool
    reasons: tuple[str, ...]
    warn_reasons: tuple[str, ...] = ()
    reason_details: tuple[IssueDetail, ...] = ()
    warn_reason_details: tuple[IssueDetail, ...] = ()
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
class Hdf5KeypointData:
    points: np.ndarray
    source: str


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
        elif isinstance(current, tuple):
            values[item.name] = tuple(incoming)
        else:
            values[item.name] = incoming

    return type(default_obj)(**values)


def load_video_quality_config(path: Path | None) -> VideoQualityConfig:
    loaded = load_qc_acceptance_config(path)
    module = loaded.raw["modules"]["video_quality"]
    parameters = loaded.module_parameters("video_quality")
    pipeline_modules = list(loaded.raw["pipeline"]["modules"])
    module_index = pipeline_modules.index("video_quality")
    next_module = (
        pipeline_modules[module_index + 1]
        if module_index + 1 < len(pipeline_modules)
        else str(loaded.raw["pipeline"]["terminal_module"])
    )
    default = VideoQualityConfig()
    known = {item.name for item in fields(default)}
    unknown = sorted(set(parameters) - known)
    if unknown:
        raise ValueError(f"unknown video quality config key: {unknown[0]}")

    config = _merge_dataclass(default, parameters, "video_quality")
    config = replace(
        config,
        module_version=str(module["module_version"]),
        qc_config_reference=loaded.json_reference(),
        next_module=next_module,
    )
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


def _timestamps_are_usable(timestamps_ms: tuple[float, ...]) -> bool:
    if len(timestamps_ms) < 2:
        return False
    if not all(np.isfinite(item) for item in timestamps_ms):
        return False
    return len({round(item, 3) for item in timestamps_ms}) > 1


def _read_frame_timestamps_ffprobe(path: Path) -> FrameTimestamps | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None

    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=best_effort_timestamp_time,pts_time,pkt_pts_time",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None

    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        return None

    timestamps: list[float] = []
    for frame in payload.get("frames", []):
        if not isinstance(frame, dict):
            continue
        for key in ("best_effort_timestamp_time", "pts_time", "pkt_pts_time"):
            raw = frame.get(key)
            if raw in (None, "N/A"):
                continue
            try:
                timestamp_ms = float(raw) * 1000.0
            except (TypeError, ValueError):
                continue
            if np.isfinite(timestamp_ms):
                timestamps.append(timestamp_ms)
                break

    timestamps_tuple = tuple(timestamps)
    if not _timestamps_are_usable(timestamps_tuple):
        return None
    return FrameTimestamps(timestamps_ms=timestamps_tuple, source="ffprobe", reliable=True)


def _read_frame_timestamps_pyav(path: Path) -> FrameTimestamps | None:
    try:
        import av  # type: ignore[import-not-found]
    except ImportError:
        return None

    timestamps: list[float] = []
    try:
        with av.open(str(path)) as container:
            stream = next((item for item in container.streams if item.type == "video"), None)
            if stream is None:
                return None
            for frame in container.decode(stream):
                timestamp_sec = frame.time
                if timestamp_sec is None and frame.pts is not None and stream.time_base is not None:
                    timestamp_sec = float(frame.pts * stream.time_base)
                if timestamp_sec is None:
                    continue
                timestamp_ms = float(timestamp_sec) * 1000.0
                if np.isfinite(timestamp_ms):
                    timestamps.append(timestamp_ms)
    except Exception:
        return None

    timestamps_tuple = tuple(timestamps)
    if not _timestamps_are_usable(timestamps_tuple):
        return None
    return FrameTimestamps(timestamps_ms=timestamps_tuple, source="pyav", reliable=True)


def _read_frame_timestamps_opencv(path: Path, frame_count: int) -> FrameTimestamps | None:
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

    timestamps_tuple = tuple(timestamps)
    if not _timestamps_are_usable(timestamps_tuple):
        return None
    return FrameTimestamps(timestamps_ms=timestamps_tuple, source="opencv", reliable=False)


def _select_frame_timestamps(path: Path, frame_count: int) -> FrameTimestamps | None:
    return (
        _read_frame_timestamps_ffprobe(path)
        or _read_frame_timestamps_pyav(path)
        or _read_frame_timestamps_opencv(path, frame_count)
    )


def _timeline_metrics(path: Path, frame_count: int, fps: float, config: TimelineConfig) -> dict[str, float | int | bool | str]:
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
            "drop_detection_source": "not_applicable",
            "drop_detection_reliable": True,
            "estimated_missing_frames": 0,
            "observed_frame_interval_count": 0,
            "frame_interval_p99_ms": 0.0,
            "max_frame_gap_ms": 0.0,
            "expected_interval_ms": expected_interval_ms,
            "drop_interval_ms": drop_interval_ms,
            "max_gap_fail_ms": max_gap_fail_ms,
        }

    timestamp_result = _select_frame_timestamps(path, frame_count)
    if timestamp_result is None:
        intervals = [expected_interval_ms] * max(0, frame_count - 1)
        pts_monotonic_valid = True
        drop_detection_source = "synthetic"
        drop_detection_reliable = False
    else:
        timestamps = list(timestamp_result.timestamps_ms)
        intervals = [timestamps[index] - timestamps[index - 1] for index in range(1, len(timestamps))]
        pts_monotonic_valid = all(interval >= 0 for interval in intervals)
        drop_detection_source = timestamp_result.source
        drop_detection_reliable = timestamp_result.reliable

    positive_intervals = [interval for interval in intervals if interval >= 0]
    if not positive_intervals:
        positive_intervals = [0.0]
    estimated_missing_frames = 0
    if expected_interval_ms > 0:
        for interval in positive_intervals:
            if interval > drop_interval_ms:
                estimated_missing_frames += max(1, int(round(interval / expected_interval_ms)) - 1)

    denominator = max(1, frame_count + estimated_missing_frames)
    return {
        "pts_monotonic_valid": pts_monotonic_valid,
        "drop_frame_ratio": estimated_missing_frames / denominator,
        "drop_detection_source": drop_detection_source,
        "drop_detection_reliable": drop_detection_reliable,
        "estimated_missing_frames": estimated_missing_frames,
        "observed_frame_interval_count": len(positive_intervals),
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


def _is_hand_transform_name(name: str) -> bool:
    lower_name = name.lower()
    return any(token in lower_name for token in ("hand", "finger", "thumb"))


def _read_camera_intrinsic(handle: h5py.File) -> np.ndarray | None:
    for candidate in ("camera/intrinsic", "camera_intrinsic", "intrinsic"):
        if candidate in handle and isinstance(handle[candidate], h5py.Dataset):
            intrinsic = np.asarray(handle[candidate][()], dtype=np.float64)
            if intrinsic.shape == (3, 3) and np.all(np.isfinite(intrinsic)):
                return intrinsic
    return None


def _project_camera_points(points_3d: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    projected = np.full(points_3d.shape[:2] + (2,), np.nan, dtype=np.float64)
    finite = np.all(np.isfinite(points_3d), axis=-1)
    homogeneous = points_3d @ intrinsic.T
    depth = homogeneous[..., 2]
    valid = finite & np.isfinite(depth) & (np.abs(depth) > 1e-9) & (points_3d[..., 2] > 1e-9)
    projected[..., 0][valid] = homogeneous[..., 0][valid] / depth[valid]
    projected[..., 1][valid] = homogeneous[..., 1][valid] / depth[valid]
    return projected


def _read_transform_keypoint_data(handle: h5py.File) -> Hdf5KeypointData | None:
    if "transforms" not in handle or not isinstance(handle["transforms"], h5py.Group):
        return None
    intrinsic = _read_camera_intrinsic(handle)
    if intrinsic is None:
        return None

    datasets: list[tuple[str, h5py.Dataset]] = []

    def visit(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if isinstance(obj, h5py.Dataset) and len(obj.shape) == 3 and obj.shape[1:] == (4, 4):
            if _is_hand_transform_name(name):
                datasets.append((name, obj))

    handle["transforms"].visititems(visit)
    if not datasets:
        return None

    ordered = sorted(datasets, key=lambda item: item[0])
    frame_count = min(int(dataset.shape[0]) for _name, dataset in ordered)
    if frame_count <= 0:
        return None

    translations = np.stack(
        [np.asarray(dataset[:frame_count, :3, 3], dtype=np.float64) for _name, dataset in ordered],
        axis=1,
    )
    if translations.shape[1] < 8:
        return None
    return Hdf5KeypointData(
        points=_project_camera_points(translations, intrinsic),
        source="hdf5_transform_keypoints",
    )


def _read_keypoint_data(path: Path) -> Hdf5KeypointData | None:
    if not path.is_file():
        return None

    try:
        with h5py.File(path, "r") as handle:
            for candidate in ("label/quality_hand", "label/hand_keypoints", "hand_keypoints", "keypoints"):
                if candidate in handle and isinstance(handle[candidate], h5py.Dataset):
                    dataset = handle[candidate]
                    if _keypoint_dataset_has_points(dataset):
                        return Hdf5KeypointData(np.asarray(dataset[()]), "hdf5_keypoints")

            found: Hdf5KeypointData | None = None

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
                    found = Hdf5KeypointData(np.asarray(obj[()]), "hdf5_keypoints")

            handle.visititems(visit)
            return found or _read_transform_keypoint_data(handle)
    except OSError:
        return None


def _ssim_score(left: np.ndarray, right: np.ndarray) -> float:
    left_float = left.astype(np.float64)
    right_float = right.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    left_mean = float(np.mean(left_float))
    right_mean = float(np.mean(right_float))
    left_var = float(np.var(left_float))
    right_var = float(np.var(right_float))
    covariance = float(np.mean((left_float - left_mean) * (right_float - right_mean)))
    denominator = (left_mean**2 + right_mean**2 + c1) * (left_var + right_var + c2)
    if denominator == 0:
        return 1.0
    return float(((2 * left_mean * right_mean + c1) * (2 * covariance + c2)) / denominator)


def _phash_bits(gray: np.ndarray) -> np.ndarray:
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(resized.astype(np.float32))
    low_freq = dct[:8, :8]
    comparable = low_freq.flatten()[1:]
    threshold = float(np.median(comparable)) if comparable.size else float(np.median(low_freq))
    return low_freq > threshold


def _phash_hamming(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.count_nonzero(_phash_bits(left) != _phash_bits(right)))


def _near_duplicate_pair_metrics(left: np.ndarray, right: np.ndarray, config: FreezeConfig) -> dict[str, float | bool]:
    diff = float(np.mean(cv2.absdiff(left, right)))
    hist_diff = _chi_square_hist_diff(left, right)
    ssim = _ssim_score(left, right)
    phash_hamming = _phash_hamming(left, right)
    auxiliary_match = ssim >= config.ssim_min or phash_hamming <= config.phash_hamming_max
    near_duplicate = diff < config.frame_diff_mean_abs_max and hist_diff < config.hist_diff_max and auxiliary_match
    return {
        "frame_diff": diff,
        "hist_diff": hist_diff,
        "ssim": ssim,
        "phash_hamming": float(phash_hamming),
        "near_duplicate": near_duplicate,
    }


def _make_frozen_interval(
    start_frame: int,
    end_frame: int,
    fps: float,
    pair_metrics: list[dict[str, float]] | None = None,
) -> FrozenInterval:
    frame_count = end_frame - start_frame + 1
    if fps > 0:
        start_time_sec = start_frame / fps
        end_time_sec = (end_frame + 1) / fps
        duration_sec = frame_count / fps
    else:
        start_time_sec = 0.0
        end_time_sec = 0.0
        duration_sec = 0.0
    pair_metrics = pair_metrics or []
    frame_diffs = [item["frame_diff"] for item in pair_metrics]
    hist_diffs = [item["hist_diff"] for item in pair_metrics]
    ssim_values = [item["ssim"] for item in pair_metrics]
    phash_values = [item["phash_hamming"] for item in pair_metrics]
    return FrozenInterval(
        start_frame=start_frame,
        end_frame=end_frame,
        frame_count=frame_count,
        start_time_sec=start_time_sec,
        end_time_sec=end_time_sec,
        duration_sec=duration_sec,
        duration_ms=duration_sec * 1000.0,
        mean_frame_diff=float(np.mean(frame_diffs)) if frame_diffs else 0.0,
        max_frame_diff=float(max(frame_diffs)) if frame_diffs else 0.0,
        mean_hist_diff=float(np.mean(hist_diffs)) if hist_diffs else 0.0,
        max_hist_diff=float(max(hist_diffs)) if hist_diffs else 0.0,
        mean_ssim=float(np.mean(ssim_values)) if ssim_values else 0.0,
        min_ssim=float(min(ssim_values)) if ssim_values else 0.0,
        mean_phash_hamming=float(np.mean(phash_values)) if phash_values else 0.0,
        max_phash_hamming=int(max(phash_values)) if phash_values else 0,
    )


def _merge_duplicate_ranges(
    ranges: list[tuple[int, int, dict[str, float]]],
    fps: float,
) -> tuple[FrozenInterval, ...]:
    if not ranges:
        return ()

    merged: list[FrozenInterval] = []
    current_start, current_end, first_metrics = ranges[0]
    current_metrics = [first_metrics]
    for start, end, metrics in ranges[1:]:
        if start <= current_end + 1:
            current_end = max(current_end, end)
            current_metrics.append(metrics)
            continue
        merged.append(_make_frozen_interval(current_start, current_end, fps, current_metrics))
        current_start = start
        current_end = end
        current_metrics = [metrics]
    merged.append(_make_frozen_interval(current_start, current_end, fps, current_metrics))
    return tuple(merged)


def _scan_frozen_intervals(
    path: Path,
    fps: float,
    config: FreezeConfig,
) -> FreezeScan:
    empty = FreezeScan(
        frozen_intervals=(),
        adjacent_near_duplicate_count=0,
        adjacent_near_duplicate_ratio=0.0,
        freeze_candidate_intervals=(),
        freeze_candidate_frame_count=0,
        freeze_candidate_duration_sec=0.0,
        freeze_candidate_ratio=0.0,
        confirmed_freeze_frame_count=0,
        confirmed_freeze_duration_sec=0.0,
        confirmed_freeze_ratio=0.0,
        max_confirmed_freeze_sec=0.0,
    )
    if not config.enabled:
        return empty

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return empty

    candidate_step = max(1, round(fps * config.freeze_candidate_window_sec)) if fps > 0 else 1
    confirmed_step = max(1, round(fps * config.confirmed_freeze_window_sec)) if fps > 0 else 1
    max_step = max(1, candidate_step, confirmed_step)
    gray_buffer: list[tuple[int, np.ndarray]] = []
    frame_index = 0
    adjacent_near_duplicate_count = 0
    candidate_ranges: list[tuple[int, int, dict[str, float]]] = []
    confirmed_ranges: list[tuple[int, int, dict[str, float]]] = []

    def previous_gray_for_step(step: int) -> tuple[int, np.ndarray] | None:
        target_index = frame_index - step
        for buffered_index, buffered_gray in reversed(gray_buffer):
            if buffered_index == target_index:
                return buffered_index, buffered_gray
            if buffered_index < target_index:
                break
        return None

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            freeze_frame, _freeze_scale = _resize_keep_aspect(frame, config.downscale_short_side, no_upscale=True)
            freeze_gray = cv2.cvtColor(freeze_frame, cv2.COLOR_BGR2GRAY)
            previous_adjacent = previous_gray_for_step(1)
            if previous_adjacent is not None:
                _previous_index, previous_gray = previous_adjacent
                adjacent_metrics = _near_duplicate_pair_metrics(previous_gray, freeze_gray, config)
                if adjacent_metrics["near_duplicate"]:
                    adjacent_near_duplicate_count += 1

            previous_candidate = previous_gray_for_step(candidate_step)
            if previous_candidate is not None:
                previous_index, previous_gray = previous_candidate
                candidate_metrics = _near_duplicate_pair_metrics(previous_gray, freeze_gray, config)
                if candidate_metrics["near_duplicate"]:
                    candidate_ranges.append((previous_index, frame_index, candidate_metrics))

            previous_confirmed = previous_gray_for_step(confirmed_step)
            if previous_confirmed is not None:
                previous_index, previous_gray = previous_confirmed
                confirmed_metrics = _near_duplicate_pair_metrics(previous_gray, freeze_gray, config)
                if confirmed_metrics["near_duplicate"]:
                    confirmed_ranges.append((previous_index, frame_index, confirmed_metrics))

            gray_buffer.append((frame_index, freeze_gray))
            if len(gray_buffer) > max_step + 1:
                gray_buffer.pop(0)
            frame_index += 1
    finally:
        capture.release()

    if frame_index <= 0:
        return empty

    candidate_intervals = _merge_duplicate_ranges(candidate_ranges, fps)
    confirmed_intervals = _merge_duplicate_ranges(confirmed_ranges, fps)
    candidate_frame_count = sum(interval.frame_count for interval in candidate_intervals)
    confirmed_frame_count = sum(interval.frame_count for interval in confirmed_intervals)
    return FreezeScan(
        frozen_intervals=confirmed_intervals,
        adjacent_near_duplicate_count=adjacent_near_duplicate_count,
        adjacent_near_duplicate_ratio=(
            adjacent_near_duplicate_count / (frame_index - 1) if frame_index > 1 else 0.0
        ),
        freeze_candidate_intervals=candidate_intervals,
        freeze_candidate_frame_count=candidate_frame_count,
        freeze_candidate_duration_sec=sum(interval.duration_sec for interval in candidate_intervals),
        freeze_candidate_ratio=candidate_frame_count / frame_index,
        confirmed_freeze_frame_count=confirmed_frame_count,
        confirmed_freeze_duration_sec=sum(interval.duration_sec for interval in confirmed_intervals),
        confirmed_freeze_ratio=confirmed_frame_count / frame_index,
        max_confirmed_freeze_sec=max((interval.duration_sec for interval in confirmed_intervals), default=0.0),
    )


def _keypoint_motion_signal(
    keypoint_data: Hdf5KeypointData | None,
    interval: FrozenInterval,
    config: FreezeConfig,
) -> str | None:
    if keypoint_data is None:
        return None
    points = keypoint_data.points
    if interval.start_frame >= points.shape[0] or interval.end_frame >= points.shape[0]:
        return None

    start_points = np.asarray(points[interval.start_frame], dtype=np.float64)
    end_points = np.asarray(points[interval.end_frame], dtype=np.float64)
    if start_points.size == 0 or end_points.size == 0 or start_points.shape[-1] < 2 or end_points.shape[-1] < 2:
        return None
    start_points = start_points.reshape(-1, start_points.shape[-1])[:, :2]
    end_points = end_points.reshape(-1, end_points.shape[-1])[:, :2]
    point_count = min(len(start_points), len(end_points))
    if point_count == 0:
        return None
    start_points = start_points[:point_count]
    end_points = end_points[:point_count]
    finite = np.all(np.isfinite(start_points), axis=1) & np.all(np.isfinite(end_points), axis=1)
    if not np.any(finite):
        return None

    start_points = start_points[finite]
    end_points = end_points[finite]
    delta = float(np.mean(np.linalg.norm(end_points - start_points, axis=1)))
    normalized = max(float(np.nanmax(np.abs(start_points))), float(np.nanmax(np.abs(end_points)))) <= 1.5
    limit = config.motion_keypoint_delta_normalized_min if normalized else config.motion_keypoint_delta_px_min
    if delta <= limit:
        return None
    unit = "norm" if normalized else "px"
    return f"{keypoint_data.source}:{delta:.4f}{unit}"


def _numeric_motion_signals(handle: h5py.File, interval: FrozenInterval, config: FreezeConfig) -> tuple[str, ...]:
    signals: list[str] = []
    tokens = ("action", "cam_pose", "camera_pose", "pose", "transform", "hand", "finger", "thumb")
    excluded_tokens = ("quality_hand", "keypoint", "landmark")

    def visit(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if not isinstance(obj, h5py.Dataset) or len(obj.shape) < 1:
            return
        lower_name = name.lower()
        if any(token in lower_name for token in excluded_tokens):
            return
        if not any(token in lower_name for token in tokens):
            return
        if obj.dtype.kind not in {"f", "i", "u"}:
            return
        if interval.start_frame >= obj.shape[0] or interval.end_frame >= obj.shape[0]:
            return
        try:
            start_value = np.asarray(obj[interval.start_frame], dtype=np.float64)
            end_value = np.asarray(obj[interval.end_frame], dtype=np.float64)
        except (OSError, TypeError, ValueError):
            return
        if start_value.size == 0 or end_value.size == 0:
            return
        finite = np.isfinite(start_value) & np.isfinite(end_value)
        if not np.any(finite):
            return
        delta = float(np.linalg.norm(end_value[finite] - start_value[finite]) / np.sqrt(np.count_nonzero(finite)))
        if delta > config.motion_numeric_delta_min:
            signals.append(f"{name}:{delta:.4f}")

    handle.visititems(visit)
    return tuple(signals)


def _critical_keywords_in_hdf5(handle: h5py.File, config: FreezeConfig) -> tuple[str, ...]:
    if not config.critical_window_enabled:
        return ()
    text_parts: list[str] = []

    def append_decoded(value: Any) -> None:
        decoded = _decode_hdf5_text(value)
        if decoded is None:
            return
        if isinstance(decoded, str):
            text_parts.append(decoded)
        else:
            text_parts.append(json.dumps(decoded, ensure_ascii=False))

    for value in handle.attrs.values():
        append_decoded(value)

    def visit(_name: str, obj: h5py.Group | h5py.Dataset) -> None:
        for value in obj.attrs.values():
            append_decoded(value)
        if isinstance(obj, h5py.Dataset) and obj.dtype.kind in {"S", "U", "O"}:
            try:
                append_decoded(obj[()])
            except OSError:
                return

    handle.visititems(visit)
    text = "\n".join(text_parts).lower()
    return tuple(keyword for keyword in config.critical_window_keywords if keyword.lower() in text)


def _annotate_frozen_intervals_with_hdf5(
    intervals: tuple[FrozenInterval, ...],
    hdf5_path: Path | None,
    config: FreezeConfig,
) -> tuple[FrozenInterval, ...]:
    if not intervals or hdf5_path is None or not hdf5_path.is_file():
        return intervals

    try:
        keypoint_data = _read_keypoint_data(hdf5_path) if config.motion_conflict_enabled else None
        with h5py.File(hdf5_path, "r") as handle:
            critical_keywords = _critical_keywords_in_hdf5(handle, config)
            annotated: list[FrozenInterval] = []
            for interval in intervals:
                motion_signals: list[str] = []
                keypoint_signal = _keypoint_motion_signal(keypoint_data, interval, config)
                if keypoint_signal is not None:
                    motion_signals.append(keypoint_signal)
                if config.motion_conflict_enabled:
                    motion_signals.extend(_numeric_motion_signals(handle, interval, config))
                annotated.append(
                    replace(
                        interval,
                        motion_conflict=bool(motion_signals),
                        motion_conflict_signals=tuple(dict.fromkeys(motion_signals)),
                        critical_window=bool(critical_keywords),
                        critical_keywords=critical_keywords,
                    )
                )
            return tuple(annotated)
    except OSError:
        return intervals


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
        black_frame_count_estimate=0,
        exposure_defect_frame_ratio=1.0,
        defect_duration_ratio=1.0,
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
        adjacent_near_duplicate_count=0,
        adjacent_near_duplicate_ratio=0.0,
        freeze_candidate_ratio=0.0,
        freeze_candidate_frame_count=0,
        freeze_candidate_duration_sec=0.0,
        confirmed_freeze_ratio=0.0,
        confirmed_freeze_frame_count=0,
        confirmed_freeze_duration_sec=0.0,
        frozen_frame_ratio=0.0,
        max_consecutive_frozen_sec=0.0,
        frozen_intervals=(),
        pts_monotonic_valid=False,
        drop_frame_ratio=1.0,
        drop_detection_source="unavailable",
        drop_detection_reliable=False,
        estimated_missing_frames=0,
        observed_frame_interval_count=0,
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
        exposure_defect_values: list[float] = []
        blur_values: list[float] = []
        tenengrad_values: list[float] = []
        scale_values: list[int] = []
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
            black_frame = (
                brightness < config.exposure.black.mean_y_max
                or dark_pixel_ratio > config.exposure.black.dark_pixel_ratio_min
            )
            over_dark_frame = (
                brightness < config.exposure.over_dark.mean_y_max
                or dark_pixel_ratio > config.exposure.over_dark.dark_pixel_ratio_min
            )
            over_exposed_frame = (
                brightness > config.exposure.over_exposed.mean_y_min
                or over_exposed_pixel_ratio > config.exposure.over_exposed.over_exposed_pixel_ratio_min
            )
            black_values.append(1.0 if black_frame else 0.0)
            dark_values.append(1.0 if over_dark_frame else 0.0)
            exposed_values.append(1.0 if over_exposed_frame else 0.0)
            exposure_defect_values.append(1.0 if black_frame or over_dark_frame or over_exposed_frame else 0.0)

            laplacian, tenengrad, scale_short_side = _sharpness_for_frame(
                frame,
                config.sharpness_global.target_short_side,
                config.sharpness_global.no_upscale,
            )
            blur_values.append(laplacian)
            tenengrad_values.append(tenengrad)
            scale_values.append(scale_short_side)

        decoded = len(brightness_values)
        sampled = len(indexes)
        if sampled and decoded == 0:
            errors.append("no_sample_frames_decoded")
        black_frame_ratio = float(np.mean(black_values)) if black_values else 1.0
        black_frame_count_estimate = int(round(black_frame_ratio * frame_count)) if frame_count > 0 else 0
        exposure_defect_frame_ratio = float(np.mean(exposure_defect_values)) if exposure_defect_values else 1.0
        freeze_scan = _scan_frozen_intervals(path, fps, config.freeze)
        frozen_intervals = _annotate_frozen_intervals_with_hdf5(
            freeze_scan.frozen_intervals,
            hdf5_path,
            config.freeze,
        )
        frozen_frame_ratio = freeze_scan.confirmed_freeze_ratio
        defect_duration_ratio = min(
            1.0,
            exposure_defect_frame_ratio + frozen_frame_ratio + float(timeline["drop_frame_ratio"]),
        )
        laplacian_under_100_ratio = (
            float(np.mean(np.array(blur_values) < LAPLACIAN_LOW_DETAIL_THRESHOLD)) if blur_values else 1.0
        )
        metadata_read_ok = frame_count > 0 and fps > 0 and width > 0 and height > 0
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
            black_frame_ratio=black_frame_ratio,
            black_frame_count_estimate=black_frame_count_estimate,
            exposure_defect_frame_ratio=exposure_defect_frame_ratio,
            defect_duration_ratio=defect_duration_ratio,
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
            adjacent_near_duplicate_count=freeze_scan.adjacent_near_duplicate_count,
            adjacent_near_duplicate_ratio=freeze_scan.adjacent_near_duplicate_ratio,
            freeze_candidate_ratio=freeze_scan.freeze_candidate_ratio,
            freeze_candidate_frame_count=freeze_scan.freeze_candidate_frame_count,
            freeze_candidate_duration_sec=freeze_scan.freeze_candidate_duration_sec,
            confirmed_freeze_ratio=frozen_frame_ratio,
            confirmed_freeze_frame_count=freeze_scan.confirmed_freeze_frame_count,
            confirmed_freeze_duration_sec=freeze_scan.confirmed_freeze_duration_sec,
            frozen_frame_ratio=frozen_frame_ratio,
            max_consecutive_frozen_sec=freeze_scan.max_confirmed_freeze_sec,
            frozen_intervals=frozen_intervals,
            pts_monotonic_valid=bool(timeline["pts_monotonic_valid"]),
            drop_frame_ratio=float(timeline["drop_frame_ratio"]),
            drop_detection_source=str(timeline["drop_detection_source"]),
            drop_detection_reliable=bool(timeline["drop_detection_reliable"]),
            estimated_missing_frames=int(timeline["estimated_missing_frames"]),
            observed_frame_interval_count=int(timeline["observed_frame_interval_count"]),
            frame_interval_p99_ms=float(timeline["frame_interval_p99_ms"]),
            max_frame_gap_ms=float(timeline["max_frame_gap_ms"]),
            expected_interval_ms=float(timeline["expected_interval_ms"]),
            drop_interval_ms=float(timeline["drop_interval_ms"]),
            max_gap_fail_ms=float(timeline["max_gap_fail_ms"]),
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


def _issue_type_for_code(code: str) -> str:
    if code in {
        "video_not_opened",
        "cannot_open_video",
        "video_stream_missing",
        "codec_unreadable",
        "metadata_unreadable",
        "no_sample_frames_decoded",
    }:
        return "video_unreadable"
    if code.startswith("fps_"):
        return "low_fps"
    if code in {"short_side_below_min", "long_side_below_min"}:
        return "low_resolution"
    if code.startswith(("drop_frame_", "frame_interval_", "max_frame_gap_", "pts_monotonic_")):
        return "timeline_discontinuity"
    if code.startswith("sample_decode_"):
        return "decode_incomplete"
    if code.startswith("black_frame_"):
        return "black_frame"
    if code.startswith("defect_duration_"):
        return "visual_defect_duration"
    if code.startswith("mean_over_dark_"):
        return "over_dark"
    if code.startswith("mean_over_exposed_"):
        return "over_exposed"
    if code.startswith(("laplacian_", "tenengrad_")):
        return "low_sharpness"
    if code == "adjacent_near_duplicate_ratio_warn":
        return "low_motion"
    if code.startswith(("frozen_frame_", "max_consecutive_frozen_")):
        return "freeze"
    if code.startswith("video_state_conflict"):
        return "video_state_conflict"
    if code.startswith("hdf5_"):
        return "hdf5_alignment"
    return "video_quality"


def _detail(
    code: str,
    severity: str,
    *,
    metric: str | None = None,
    value: Any | None = None,
    pass_threshold: Any | None = None,
    fail_threshold: Any | None = None,
    comparison: str | None = None,
    rule_id: str | None = None,
    context: dict[str, Any] | None = None,
) -> IssueDetail:
    boundary_value = fail_threshold if severity == "fail" and fail_threshold is not None else pass_threshold
    return IssueDetail(
        issue_id=f"video_quality:{code}:001",
        code=code,
        severity=severity,
        module="video_quality",
        issue_type=_issue_type_for_code(code),
        metric=metric,
        observed_value=value,
        operator=comparison,
        boundary_value=boundary_value,
        rule_id=rule_id or f"video_quality.{code}",
        needs_manual_review=severity == "warn",
        context=context or {},
    )


def _reason_details_for_codes(
    codes: tuple[str, ...],
    severity: str,
    metrics: VideoMetrics,
    config: VideoQualityConfig,
    alignment: object | None,
) -> tuple[IssueDetail, ...]:
    details: list[IssueDetail] = []
    def add(
        code: str,
        *,
        metric: str | None = None,
        value: Any | None = None,
        pass_threshold: Any | None = None,
        fail_threshold: Any | None = None,
        comparison: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        details.append(
            _detail(
                code,
                severity,
                metric=metric,
                value=value,
                pass_threshold=pass_threshold,
                fail_threshold=fail_threshold,
                comparison=comparison,
                context=context,
            )
        )

    conflict_intervals = [interval for interval in metrics.frozen_intervals if interval.motion_conflict]
    critical_conflicts = [interval for interval in conflict_intervals if interval.critical_window]
    noncritical_conflicts = [interval for interval in conflict_intervals if not interval.critical_window]
    conflict_context = {
        "motion_conflict_interval_count": len(conflict_intervals),
        "critical_motion_conflict_count": len(critical_conflicts),
        "noncritical_motion_conflict_count": len(noncritical_conflicts),
        "max_motion_conflict_duration_ms": max((interval.duration_ms for interval in conflict_intervals), default=0.0),
    }

    for code in codes:
        if code == "video_not_opened":
            add(code, metric="video_basic.video_open_ok", value=metrics.opened, pass_threshold=True, comparison="==")
        elif code == "video_stream_missing":
            add(
                code,
                metric="video_basic.video_stream_present",
                value=metrics.video_stream_present,
                pass_threshold=True,
                comparison="==",
            )
        elif code == "codec_unreadable":
            add(code, metric="video_basic.codec_readable", value=metrics.codec_readable, pass_threshold=True, comparison="==")
        elif code == "metadata_unreadable":
            add(
                code,
                metric="video_basic.metadata_read_ok",
                value=metrics.metadata_read_ok,
                pass_threshold=True,
                comparison="==",
            )
        elif code in {"fps_below_min", "fps_below_pass"}:
            add(
                code,
                metric="video_basic.fps",
                value=metrics.fps,
                pass_threshold=config.fps.min_fps_pass,
                fail_threshold=config.fps.min_fps_fail,
                comparison="<",
            )
        elif code == "fps_below_expected":
            expected_threshold = None if config.fps.expected_fps is None else config.fps.expected_fps * 0.95
            add(
                code,
                metric="video_basic.fps",
                value=metrics.fps,
                pass_threshold=expected_threshold,
                comparison="<",
                context={"expected_fps": config.fps.expected_fps},
            )
        elif code == "short_side_below_min":
            add(
                code,
                metric="video_basic.short_side",
                value=metrics.short_side,
                pass_threshold=config.resolution.min_short_side_fail,
                comparison="<",
            )
        elif code == "long_side_below_min":
            add(
                code,
                metric="video_basic.long_side",
                value=metrics.long_side,
                pass_threshold=config.resolution.min_long_side_fail,
                comparison="<",
            )
        elif code == "pts_monotonic_invalid":
            add(
                code,
                metric="timeline_metrics.pts_monotonic_valid",
                value=metrics.pts_monotonic_valid,
                pass_threshold=True,
                comparison="==",
            )
        elif code in {"drop_frame_ratio_above_max", "drop_frame_ratio_warn"}:
            add(
                code,
                metric="timeline_metrics.drop_frame_ratio",
                value=metrics.drop_frame_ratio,
                pass_threshold=config.timeline.drop_frame_ratio_pass,
                fail_threshold=config.timeline.drop_frame_ratio_warn,
                comparison=">",
                context={"estimated_missing_frames": metrics.estimated_missing_frames},
            )
        elif code in {"frame_interval_p99_ms_above_max", "frame_interval_p99_ms_warn"}:
            add(
                code,
                metric="timeline_metrics.frame_interval_p99_ms",
                value=metrics.frame_interval_p99_ms,
                pass_threshold=metrics.drop_interval_ms,
                fail_threshold=metrics.expected_interval_ms * 2 if metrics.expected_interval_ms > 0 else None,
                comparison=">",
                context={"expected_interval_ms": metrics.expected_interval_ms},
            )
        elif code == "max_frame_gap_ms_above_max":
            add(
                code,
                metric="timeline_metrics.max_frame_gap_ms",
                value=metrics.max_frame_gap_ms,
                fail_threshold=metrics.max_gap_fail_ms,
                comparison=">",
            )
        elif code in {"sample_decode_ratio_below_min", "sample_decode_ratio_warn"}:
            add(
                code,
                metric="decode_metrics.sample_decode_ratio",
                value=metrics.sample_decode_ratio,
                pass_threshold=config.decode.sample_decode_ratio_pass,
                fail_threshold=config.decode.sample_decode_ratio_warn,
                comparison="<",
            )
        elif code in {"black_frame_ratio_above_max", "black_frame_ratio_warn"}:
            add(
                code,
                metric="exposure_metrics.black_frame_ratio",
                value=metrics.black_frame_ratio,
                pass_threshold=config.exposure.black.ratio_pass,
                fail_threshold=config.exposure.black.ratio_warn,
                comparison=">",
                context={"black_frame_count_estimate": metrics.black_frame_count_estimate},
            )
        elif code == "black_frame_count_above_max":
            add(
                code,
                metric="exposure_metrics.black_frame_count_estimate",
                value=metrics.black_frame_count_estimate,
                fail_threshold=config.exposure.black.max_frame_count_fail,
                comparison=">",
            )
        elif code in {"defect_duration_ratio_above_max", "defect_duration_ratio_warn"}:
            add(
                code,
                metric="defect_metrics.defect_duration_ratio",
                value=metrics.defect_duration_ratio,
                pass_threshold=config.defects.duration_ratio_warn,
                fail_threshold=config.defects.max_duration_ratio_fail,
                comparison=">",
                context={
                    "exposure_defect_frame_ratio": metrics.exposure_defect_frame_ratio,
                    "frozen_frame_ratio": metrics.frozen_frame_ratio,
                    "drop_frame_ratio": metrics.drop_frame_ratio,
                },
            )
        elif code in {"mean_over_dark_ratio_above_max", "mean_over_dark_ratio_warn"}:
            add(
                code,
                metric="exposure_metrics.mean_over_dark_ratio",
                value=metrics.mean_over_dark_ratio,
                pass_threshold=config.exposure.over_dark.ratio_pass,
                fail_threshold=config.exposure.over_dark.ratio_warn,
                comparison=">",
            )
        elif code in {"mean_over_exposed_ratio_above_max", "mean_over_exposed_ratio_warn"}:
            add(
                code,
                metric="exposure_metrics.mean_over_exposed_ratio",
                value=metrics.mean_over_exposed_ratio,
                pass_threshold=config.exposure.over_exposed.ratio_pass,
                fail_threshold=config.exposure.over_exposed.ratio_warn,
                comparison=">",
            )
        elif code in {"laplacian_p10_below_min", "laplacian_p10_warn"}:
            add(
                code,
                metric="sharpness_global.laplacian_p10",
                value=metrics.laplacian_p10,
                pass_threshold=config.sharpness_global.laplacian_p10_pass,
                fail_threshold=config.sharpness_global.laplacian_p10_warn,
                comparison="<",
            )
        elif code in {"laplacian_median_below_min", "laplacian_median_warn"}:
            add(
                code,
                metric="sharpness_global.laplacian_median",
                value=metrics.laplacian_median,
                pass_threshold=config.sharpness_global.laplacian_median_pass,
                fail_threshold=config.sharpness_global.laplacian_median_warn,
                comparison="<",
            )
        elif code in {"laplacian_under_100_ratio_above_max", "laplacian_under_100_ratio_warn"}:
            add(
                code,
                metric="sharpness_global.laplacian_under_100_ratio",
                value=metrics.laplacian_under_100_ratio,
                pass_threshold=config.sharpness_global.laplacian_under_100_ratio_pass,
                fail_threshold=config.sharpness_global.laplacian_under_100_ratio_warn,
                comparison=">",
            )
        elif code in {"tenengrad_p10_below_min", "tenengrad_p10_warn"}:
            add(
                code,
                metric="sharpness_global.tenengrad_p10",
                value=metrics.tenengrad_p10,
                pass_threshold=config.sharpness_global.tenengrad_p10_pass,
                fail_threshold=config.sharpness_global.tenengrad_p10_warn,
                comparison="<",
            )
        elif code in {"tenengrad_median_below_min", "tenengrad_median_warn"}:
            add(
                code,
                metric="sharpness_global.tenengrad_median",
                value=metrics.tenengrad_median,
                pass_threshold=config.sharpness_global.tenengrad_median_pass,
                fail_threshold=config.sharpness_global.tenengrad_median_warn,
                comparison="<",
            )
        elif code == "adjacent_near_duplicate_ratio_warn":
            add(
                code,
                metric="freeze_metrics.adjacent_near_duplicate_ratio",
                value=metrics.adjacent_near_duplicate_ratio,
                pass_threshold=config.freeze.adjacent_near_duplicate_ratio_warn,
                comparison=">",
            )
        elif code in {"frozen_frame_ratio_above_max", "frozen_frame_ratio_warn"}:
            add(
                code,
                metric="freeze_metrics.frozen_frame_ratio",
                value=metrics.frozen_frame_ratio,
                pass_threshold=config.freeze.frozen_frame_ratio_pass,
                fail_threshold=config.freeze.frozen_frame_ratio_warn,
                comparison=">",
                context={"confirmed_freeze_frame_count": metrics.confirmed_freeze_frame_count},
            )
        elif code in {"max_consecutive_frozen_sec_above_max", "max_consecutive_frozen_sec_warn"}:
            add(
                code,
                metric="freeze_metrics.max_consecutive_frozen_sec",
                value=metrics.max_consecutive_frozen_sec,
                pass_threshold=config.freeze.max_consecutive_frozen_sec_pass,
                fail_threshold=config.freeze.max_consecutive_frozen_sec_fail,
                comparison=">",
            )
        elif code in {"video_state_conflict", "video_state_conflict_warn"}:
            add(code, metric="freeze_metrics.frozen_intervals.motion_conflict", value=True, comparison="==", context=conflict_context)
        elif code in {"hdf5_missing", "hdf5_unreadable"} and isinstance(alignment, Hdf5Alignment):
            add(code, metric="hdf5_alignment.status", value=alignment.status, pass_threshold="matched", comparison="==")
        elif code in {"hdf5_frame_count_mismatch", "hdf5_frame_count_mismatch_warn"} and isinstance(
            alignment, Hdf5Alignment
        ):
            add(
                code,
                metric="hdf5_alignment.frame_count_delta",
                value=alignment.frame_count_delta,
                pass_threshold=config.hdf5_alignment.max_delta_frames_pass,
                fail_threshold=config.hdf5_alignment.max_delta_frames_warn,
                comparison=">",
                context={
                    "video_frame_count": metrics.frame_count,
                    "hdf5_frame_count": alignment.hdf5_frame_count,
                    "frame_count_delta_ratio": alignment.frame_count_delta_ratio,
                },
            )
        else:
            add(code)

    return tuple(details)


def _make_evaluation(
    fail_reasons: list[str],
    warn_reasons: list[str],
    *,
    metrics: VideoMetrics,
    config: VideoQualityConfig,
    alignment: object | None,
) -> QualityEvaluation:
    unique_fail = tuple(dict.fromkeys(fail_reasons))
    unique_warn = tuple(reason for reason in dict.fromkeys(warn_reasons) if reason not in unique_fail)
    decision = "fail" if unique_fail else "warn" if unique_warn else "pass"
    should_run_mask_qc = decision != "fail"
    return QualityEvaluation(
        decision=decision,
        passed=should_run_mask_qc,
        reasons=unique_fail,
        warn_reasons=unique_warn,
        reason_details=_reason_details_for_codes(unique_fail, "fail", metrics, config, alignment),
        warn_reason_details=_reason_details_for_codes(unique_warn, "warn", metrics, config, alignment),
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
    if metrics.black_frame_count_estimate > config.exposure.black.max_frame_count_fail:
        fail.append("black_frame_count_above_max")
    _add_threshold_reason(
        fail,
        warn,
        metrics.defect_duration_ratio,
        config.defects.duration_ratio_warn,
        config.defects.max_duration_ratio_fail,
        "defect_duration_ratio_above_max",
        "defect_duration_ratio_warn",
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
        if metrics.adjacent_near_duplicate_ratio > config.freeze.adjacent_near_duplicate_ratio_warn:
            warn.append("adjacent_near_duplicate_ratio_warn")
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
        for interval in metrics.frozen_intervals:
            if not interval.motion_conflict:
                continue
            duration_limit = (
                config.freeze.video_state_conflict_critical_duration_ms_fail
                if interval.critical_window
                else config.freeze.video_state_conflict_noncritical_duration_ms_fail
            )
            if interval.duration_ms >= duration_limit:
                fail.append("video_state_conflict")
            else:
                warn.append("video_state_conflict_warn")

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

    return _make_evaluation(fail, warn, metrics=metrics, config=config, alignment=alignment)


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
    issues = (*evaluation.reason_details, *evaluation.warn_reason_details)
    return {
        "decision": evaluation.decision,
        "reasons": list(evaluation.reasons),
        "warn_reasons": list(evaluation.warn_reasons),
        "issue_ids": [issue.issue_id for issue in issues],
        "should_run_mask_qc": evaluation.should_run_mask_qc,
    }


def _frozen_interval_json(interval: FrozenInterval) -> dict[str, Any]:
    return {
        "start_frame": interval.start_frame,
        "end_frame": interval.end_frame,
        "frame_count": interval.frame_count,
        "start_time_sec": interval.start_time_sec,
        "end_time_sec": interval.end_time_sec,
        "duration_sec": interval.duration_sec,
        "duration_ms": interval.duration_ms,
        "mean_frame_diff": interval.mean_frame_diff,
        "max_frame_diff": interval.max_frame_diff,
        "mean_hist_diff": interval.mean_hist_diff,
        "max_hist_diff": interval.max_hist_diff,
        "mean_ssim": interval.mean_ssim,
        "min_ssim": interval.min_ssim,
        "mean_phash_hamming": interval.mean_phash_hamming,
        "max_phash_hamming": interval.max_phash_hamming,
        "motion_conflict": interval.motion_conflict,
        "motion_conflict_signals": list(interval.motion_conflict_signals),
        "critical_window": interval.critical_window,
        "critical_keywords": list(interval.critical_keywords),
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
    evaluation = _evaluation_json(result.evaluation)
    batch_dir = _infer_batch_dir(metrics, alignment)
    issues = (*result.evaluation.reason_details, *result.evaluation.warn_reason_details)
    fail_issue_ids = [issue.issue_id for issue in result.evaluation.reason_details]
    warn_issue_ids = [issue.issue_id for issue in result.evaluation.warn_reason_details]
    failed = result.evaluation.decision == "fail"
    next_module = "batch_statistics" if failed else config.next_module

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
        "drop_detection_source": metrics.drop_detection_source,
        "drop_detection_reliable": metrics.drop_detection_reliable,
        "estimated_missing_frames": metrics.estimated_missing_frames,
        "observed_frame_interval_count": metrics.observed_frame_interval_count,
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
        "black_frame_count_estimate": metrics.black_frame_count_estimate,
        "exposure_defect_frame_ratio": metrics.exposure_defect_frame_ratio,
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
        "adjacent_near_duplicate_count": metrics.adjacent_near_duplicate_count,
        "adjacent_near_duplicate_ratio": metrics.adjacent_near_duplicate_ratio,
        "freeze_candidate_window_sec": config.freeze.freeze_candidate_window_sec,
        "freeze_candidate_frame_count": metrics.freeze_candidate_frame_count,
        "freeze_candidate_duration_sec": metrics.freeze_candidate_duration_sec,
        "freeze_candidate_ratio": metrics.freeze_candidate_ratio,
        "confirmed_freeze_window_sec": config.freeze.confirmed_freeze_window_sec,
        "confirmed_freeze_frame_count": metrics.confirmed_freeze_frame_count,
        "confirmed_freeze_duration_sec": metrics.confirmed_freeze_duration_sec,
        "confirmed_freeze_ratio": metrics.confirmed_freeze_ratio,
        "frozen_frame_ratio": metrics.frozen_frame_ratio,
        "max_consecutive_frozen_sec": metrics.max_consecutive_frozen_sec,
        "frozen_interval_min_frames": config.freeze.min_interval_frames,
        "frozen_interval_min_duration_ms": config.freeze.min_interval_duration_ms,
        "frozen_interval_count": len(metrics.frozen_intervals),
        "frozen_interval_frame_count": sum(interval.frame_count for interval in metrics.frozen_intervals),
        "frozen_interval_duration_sec": sum(interval.duration_sec for interval in metrics.frozen_intervals),
        "frozen_interval_motion_conflict_count": sum(1 for interval in metrics.frozen_intervals if interval.motion_conflict),
        "frozen_interval_critical_window_count": sum(1 for interval in metrics.frozen_intervals if interval.critical_window),
        "ssim_min": config.freeze.ssim_min,
        "phash_hamming_max": config.freeze.phash_hamming_max,
        "frozen_intervals": [_frozen_interval_json(interval) for interval in metrics.frozen_intervals],
    }
    defect_metrics = {
        "defect_duration_ratio": metrics.defect_duration_ratio,
        "exposure_defect_frame_ratio": metrics.exposure_defect_frame_ratio,
        "frozen_frame_ratio": metrics.frozen_frame_ratio,
        "drop_frame_ratio": metrics.drop_frame_ratio,
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
    return {
        "schema_version": "asset_qc_report.v1",
        "qc_config": dict(config.qc_config_reference),
        "asset_id": metrics.asset_id,
        "report_revision": 1,
        "pipeline_state": {
            "status": "stopped" if failed else "running",
            "last_completed_module": "video_quality",
            "next_module": next_module,
        },
        "overall_decision": "fail" if failed else None,
        "issues": _to_plain(issues),
        "manual_review": {
            "required": False if failed else None,
            "state": "skipped_due_to_fail" if failed else "not_evaluated",
            "candidate_issue_ids": [] if failed else warn_issue_ids,
            "failures_for_batch_stats_issue_ids": fail_issue_ids,
        },
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
        "video_quality": {
            "stage": "video_prefilter",
            "module_version": config.module_version,
            "flow": {
                "entry_gate": {
                    "state": "ready",
                    "eligible": True,
                    "blocked_by_module": None,
                    "required_inputs": ["source_files.video.path"],
                    "missing_inputs": [],
                    "upstream_continue": True,
                },
                "result_gate": {
                    "verdict": result.evaluation.decision,
                    "has_fail": failed,
                    "has_warn": bool(result.evaluation.warn_reasons),
                },
                "exit_gate": {
                    "state": "stop_qc" if failed else "continue",
                    "continue_to_next_module": not failed,
                    "next_module": next_module,
                },
            },
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
                "defect_metrics": defect_metrics,
                "hdf5_alignment": hdf5_alignment,
            },
            "errors": list(metrics.errors),
        },
        "reference_quality": {
            "mode": "none",
            "reference_video_path": None,
            "vmaf": None,
            "note": "当前无标准对照视频，未计算 VMAF。",
        },
    }


def _video_report_ready(report: dict[str, Any] | None) -> bool:
    if report is None:
        return False
    pipeline_state = report.get("pipeline_state")
    if not isinstance(pipeline_state, dict):
        return False
    if pipeline_state.get("next_module") == "video_quality":
        return True
    return (
        pipeline_state.get("last_completed_module") == "video_quality"
        and isinstance(report.get("video_quality"), dict)
    )


def _relative_batch_path(path: Path, batch_dir: Path) -> str:
    try:
        return path.resolve().relative_to(batch_dir.resolve()).as_posix()
    except ValueError:
        raise ValueError(f"source path is outside batch root: {path}") from None


def _assert_report_source_path(
    report: dict[str, Any],
    *,
    source_name: str,
    expected_path: str,
) -> None:
    source_files = report.get("source_files")
    recorded = source_files.get(source_name) if isinstance(source_files, dict) else None
    recorded_path = recorded.get("path") if isinstance(recorded, dict) else None
    if recorded_path != expected_path:
        raise ValueError(
            f"source_files.{source_name}.path mismatch: "
            f"{recorded_path!r} != {expected_path!r}"
        )


def write_per_asset_qc_json_reports(
    batch_dir: Path,
    results: list[VideoQualityResult],
    config: LoadedQcConfig,
    *,
    profile: str = "acceptance",
) -> int:
    from qc_pipeline.adapters.video_quality import write_video_quality_result

    modules = config.pipeline_modules
    video_index = modules.index("video_quality")
    next_module = modules[video_index + 1]
    awaiting_pipeline = 0
    for result in results:
        path = batch_dir / "quality_archive" / f"{result.metrics.asset_id}.json"
        existing = load_asset_qc_report(path)
        if not _video_report_ready(existing):
            awaiting_pipeline += 1
            current_next = None
            if existing is not None and isinstance(existing.get("pipeline_state"), dict):
                current_next = existing["pipeline_state"].get("next_module")
            LOGGER.warning(
                "Skipping QC report write for %s: video_quality awaits pipeline "
                "state (current next_module=%r)",
                result.metrics.asset_id,
                current_next,
            )
            continue
        assert existing is not None
        relative_video_path = _relative_batch_path(result.metrics.path, batch_dir)
        _assert_report_source_path(
            existing,
            source_name="video",
            expected_path=relative_video_path,
        )
        source_files: dict[str, Any] = {
            "video": {"path": relative_video_path}
        }
        if (
            result.alignment.hdf5_path is not None
            and result.alignment.hdf5_path.is_file()
        ):
            relative_hdf5_path = _relative_batch_path(
                result.alignment.hdf5_path,
                batch_dir,
            )
            _assert_report_source_path(
                existing,
                source_name="hdf5",
                expected_path=relative_hdf5_path,
            )
            source_files["hdf5"] = {
                "path": relative_hdf5_path
            }
        context = AssetContext(
            asset_id=result.metrics.asset_id,
            batch_root=batch_dir,
            report_path=path,
            source_files=source_files,
        )
        write_video_quality_result(
            context=context,
            result=result,
            config=config,
            profile=profile,
            expected_revision=int(existing.get("report_revision", 0)),
            next_module=next_module,
        )
    return awaiting_pipeline


def run_video_quality_check(
    batch_dir: Path,
    config_path: Path | None = None,
) -> int:
    loaded_config = load_qc_acceptance_config(config_path)
    config = load_video_quality_config(config_path)
    video_paths = discover_batch_videos(batch_dir)
    results: list[VideoQualityResult] = []

    for video_path in video_paths:
        hdf5_path = hdf5_path_for_video(video_path, batch_dir)
        metrics = analyze_video(video_path, config, hdf5_path=hdf5_path)
        alignment = check_hdf5_alignment(video_path, batch_dir, metrics, config)
        evaluation = evaluate_video_quality(metrics, config, alignment)
        results.append(VideoQualityResult(metrics, alignment, evaluation))

    awaiting_pipeline = write_per_asset_qc_json_reports(
        batch_dir,
        results,
        loaded_config,
        profile="acceptance",
    )
    if awaiting_pipeline:
        return 3
    return 0 if all(result.evaluation.passed for result in results) else 2


@dataclass(frozen=True)
class VideoFrameRangeAnalysis:
    """Video metrics plus decode accounting for one inclusive source-frame range."""

    metrics: VideoMetrics
    source_video_frame_count: int
    decoded_frame_count: int
    sampled_local_frame_indices: tuple[int, ...]


def _timeline_metrics_for_frame_range(
    timestamps_ms: list[float],
    frame_count: int,
    fps: float,
    config: TimelineConfig,
) -> dict[str, float | int | bool | str]:
    """Compute timeline metrics using timestamps collected inside one range."""

    expected_interval_ms = 1000.0 / fps if fps > 0 else 0.0
    drop_interval_ms = max(
        expected_interval_ms * config.drop_interval_factor,
        expected_interval_ms + config.drop_interval_extra_ms,
    )
    max_gap_fail_ms = max(
        expected_interval_ms * config.max_gap_factor,
        config.max_gap_floor_ms,
    )

    if frame_count <= 1 or fps <= 0:
        return {
            "pts_monotonic_valid": True,
            "drop_frame_ratio": 0.0,
            "drop_detection_source": "not_applicable",
            "drop_detection_reliable": True,
            "estimated_missing_frames": 0,
            "observed_frame_interval_count": 0,
            "frame_interval_p99_ms": 0.0,
            "max_frame_gap_ms": 0.0,
            "expected_interval_ms": expected_interval_ms,
            "drop_interval_ms": drop_interval_ms,
            "max_gap_fail_ms": max_gap_fail_ms,
        }

    timestamps = tuple(timestamps_ms)
    if (
        len(timestamps) == frame_count
        and _timestamps_are_usable(timestamps)
    ):
        intervals = [
            timestamps[index] - timestamps[index - 1]
            for index in range(1, len(timestamps))
        ]
        pts_monotonic_valid = all(interval >= 0 for interval in intervals)
        drop_detection_source = "opencv_range"
        drop_detection_reliable = False
    else:
        intervals = [expected_interval_ms] * max(0, frame_count - 1)
        pts_monotonic_valid = True
        drop_detection_source = "synthetic_range"
        drop_detection_reliable = False

    positive_intervals = [
        interval for interval in intervals
        if interval >= 0
    ]
    if not positive_intervals:
        positive_intervals = [0.0]

    estimated_missing_frames = 0
    if expected_interval_ms > 0:
        for interval in positive_intervals:
            if interval > drop_interval_ms:
                estimated_missing_frames += max(
                    1,
                    int(round(interval / expected_interval_ms)) - 1,
                )

    denominator = max(1, frame_count + estimated_missing_frames)
    return {
        "pts_monotonic_valid": pts_monotonic_valid,
        "drop_frame_ratio": estimated_missing_frames / denominator,
        "drop_detection_source": drop_detection_source,
        "drop_detection_reliable": drop_detection_reliable,
        "estimated_missing_frames": estimated_missing_frames,
        "observed_frame_interval_count": len(positive_intervals),
        "frame_interval_p99_ms": _percentile(positive_intervals, 99),
        "max_frame_gap_ms": float(max(positive_intervals)),
        "expected_interval_ms": expected_interval_ms,
        "drop_interval_ms": drop_interval_ms,
        "max_gap_fail_ms": max_gap_fail_ms,
    }


def analyze_video_frame_range(
    path: Path,
    config: VideoQualityConfig,
    start_frame: int,
    end_frame: int,
    hdf5_path: Path | None = None,
) -> VideoFrameRangeAnalysis:
    """Analyze one inclusive logical clip without creating a split video."""

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"cannot open source video: {path}")

    try:
        source_frame_count = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if start_frame < 0:
            raise ValueError("start_frame must be >= 0")
        if end_frame < start_frame:
            raise ValueError("end_frame must be >= start_frame")
        if end_frame >= source_frame_count:
            raise ValueError(
                f"end_frame {end_frame} outside source frame count "
                f"{source_frame_count}"
            )

        clip_frame_count = end_frame - start_frame + 1
        duration = clip_frame_count / fps if fps > 0 else 0.0
        short_side = min(width, height) if width and height else 0
        long_side = max(width, height) if width and height else 0

        sample_local_indexes = _sample_indexes(
            clip_frame_count,
            fps,
            duration,
            config.decode,
        )
        sample_index_set = set(sample_local_indexes)

        brightness_values: list[float] = []
        black_values: list[float] = []
        dark_values: list[float] = []
        exposed_values: list[float] = []
        exposure_defect_values: list[float] = []
        blur_values: list[float] = []
        tenengrad_values: list[float] = []
        scale_values: list[int] = []
        timestamps_ms: list[float] = []
        errors: list[str] = []

        adjacent_near_duplicate_count = 0
        candidate_ranges: list[
            tuple[int, int, dict[str, float]]
        ] = []
        confirmed_ranges: list[
            tuple[int, int, dict[str, float]]
        ] = []

        candidate_step = (
            max(
                1,
                round(
                    fps
                    * config.freeze.freeze_candidate_window_sec
                ),
            )
            if fps > 0
            else 1
        )
        confirmed_step = (
            max(
                1,
                round(
                    fps
                    * config.freeze.confirmed_freeze_window_sec
                ),
            )
            if fps > 0
            else 1
        )
        max_step = max(1, candidate_step, confirmed_step)
        gray_buffer: list[tuple[int, np.ndarray]] = []

        def previous_gray_for_step(
            source_frame_idx: int,
            step: int,
        ) -> tuple[int, np.ndarray] | None:
            target_index = source_frame_idx - step
            for buffered_index, buffered_gray in reversed(
                gray_buffer
            ):
                if buffered_index == target_index:
                    return buffered_index, buffered_gray
                if buffered_index < target_index:
                    break
            return None

        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        decoded_frame_count = 0

        for source_frame_idx in range(
            start_frame,
            end_frame + 1,
        ):
            ok, frame = capture.read()
            if not ok or frame is None:
                errors.append(
                    f"range_decode_failed:{source_frame_idx}"
                )
                break

            local_frame_idx = source_frame_idx - start_frame
            decoded_frame_count += 1
            timestamps_ms.append(
                float(capture.get(cv2.CAP_PROP_POS_MSEC))
            )

            if local_frame_idx in sample_index_set:
                gray = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY,
                )
                brightness = float(np.mean(gray))
                dark_pixel_ratio = float(
                    np.mean(gray < DARK_PIXEL_Y_THRESHOLD)
                )
                over_exposed_pixel_ratio = float(
                    np.mean(
                        gray > OVER_EXPOSED_PIXEL_Y_THRESHOLD
                    )
                )

                brightness_values.append(brightness)

                black_frame = (
                    brightness
                    < config.exposure.black.mean_y_max
                    or dark_pixel_ratio
                    > config.exposure.black.dark_pixel_ratio_min
                )
                over_dark_frame = (
                    brightness
                    < config.exposure.over_dark.mean_y_max
                    or dark_pixel_ratio
                    > config.exposure.over_dark.dark_pixel_ratio_min
                )
                over_exposed_frame = (
                    brightness
                    > config.exposure.over_exposed.mean_y_min
                    or over_exposed_pixel_ratio
                    > config.exposure.over_exposed.over_exposed_pixel_ratio_min
                )

                black_values.append(
                    1.0 if black_frame else 0.0
                )
                dark_values.append(
                    1.0 if over_dark_frame else 0.0
                )
                exposed_values.append(
                    1.0 if over_exposed_frame else 0.0
                )
                exposure_defect_values.append(
                    1.0
                    if (
                        black_frame
                        or over_dark_frame
                        or over_exposed_frame
                    )
                    else 0.0
                )

                laplacian, tenengrad, scale_short_side = (
                    _sharpness_for_frame(
                        frame,
                        config.sharpness_global.target_short_side,
                        config.sharpness_global.no_upscale,
                    )
                )
                blur_values.append(laplacian)
                tenengrad_values.append(tenengrad)
                scale_values.append(scale_short_side)

            if config.freeze.enabled:
                freeze_frame, _freeze_scale = (
                    _resize_keep_aspect(
                        frame,
                        config.freeze.downscale_short_side,
                        no_upscale=True,
                    )
                )
                freeze_gray = cv2.cvtColor(
                    freeze_frame,
                    cv2.COLOR_BGR2GRAY,
                )

                previous_adjacent = previous_gray_for_step(
                    source_frame_idx,
                    1,
                )
                if previous_adjacent is not None:
                    _previous_index, previous_gray = (
                        previous_adjacent
                    )
                    adjacent_metrics = (
                        _near_duplicate_pair_metrics(
                            previous_gray,
                            freeze_gray,
                            config.freeze,
                        )
                    )
                    if adjacent_metrics["near_duplicate"]:
                        adjacent_near_duplicate_count += 1

                previous_candidate = previous_gray_for_step(
                    source_frame_idx,
                    candidate_step,
                )
                if previous_candidate is not None:
                    previous_index, previous_gray = (
                        previous_candidate
                    )
                    candidate_metrics = (
                        _near_duplicate_pair_metrics(
                            previous_gray,
                            freeze_gray,
                            config.freeze,
                        )
                    )
                    if candidate_metrics["near_duplicate"]:
                        candidate_ranges.append(
                            (
                                previous_index,
                                source_frame_idx,
                                candidate_metrics,
                            )
                        )

                previous_confirmed = previous_gray_for_step(
                    source_frame_idx,
                    confirmed_step,
                )
                if previous_confirmed is not None:
                    previous_index, previous_gray = (
                        previous_confirmed
                    )
                    confirmed_metrics = (
                        _near_duplicate_pair_metrics(
                            previous_gray,
                            freeze_gray,
                            config.freeze,
                        )
                    )
                    if confirmed_metrics["near_duplicate"]:
                        confirmed_ranges.append(
                            (
                                previous_index,
                                source_frame_idx,
                                confirmed_metrics,
                            )
                        )

                gray_buffer.append(
                    (source_frame_idx, freeze_gray)
                )
                if len(gray_buffer) > max_step + 1:
                    gray_buffer.pop(0)

        sampled_frame_count = len(sample_local_indexes)
        decoded_sample_count = len(brightness_values)
        if sampled_frame_count and decoded_sample_count == 0:
            errors.append("no_sample_frames_decoded")

        timeline = _timeline_metrics_for_frame_range(
            timestamps_ms,
            decoded_frame_count,
            fps,
            config.timeline,
        )

        candidate_intervals = (
            _merge_duplicate_ranges(candidate_ranges, fps)
            if config.freeze.enabled
            else ()
        )
        confirmed_intervals = (
            _merge_duplicate_ranges(confirmed_ranges, fps)
            if config.freeze.enabled
            else ()
        )
        confirmed_intervals = (
            _annotate_frozen_intervals_with_hdf5(
                confirmed_intervals,
                hdf5_path,
                config.freeze,
            )
        )

        candidate_frame_count = sum(
            interval.frame_count
            for interval in candidate_intervals
        )
        confirmed_frame_count = sum(
            interval.frame_count
            for interval in confirmed_intervals
        )

        denominator = max(1, decoded_frame_count)
        candidate_ratio = (
            candidate_frame_count / denominator
        )
        confirmed_ratio = (
            confirmed_frame_count / denominator
        )
        adjacent_ratio = (
            adjacent_near_duplicate_count
            / (decoded_frame_count - 1)
            if decoded_frame_count > 1
            else 0.0
        )

        black_frame_ratio = (
            float(np.mean(black_values))
            if black_values
            else 1.0
        )
        exposure_defect_frame_ratio = (
            float(np.mean(exposure_defect_values))
            if exposure_defect_values
            else 1.0
        )
        black_frame_count_estimate = (
            int(
                round(
                    black_frame_ratio
                    * clip_frame_count
                )
            )
            if clip_frame_count > 0
            else 0
        )
        laplacian_under_100_ratio = (
            float(
                np.mean(
                    np.asarray(blur_values)
                    < LAPLACIAN_LOW_DETAIL_THRESHOLD
                )
            )
            if blur_values
            else 1.0
        )
        defect_duration_ratio = min(
            1.0,
            exposure_defect_frame_ratio
            + confirmed_ratio
            + float(timeline["drop_frame_ratio"]),
        )

        metadata_read_ok = (
            source_frame_count > 0
            and fps > 0
            and width > 0
            and height > 0
        )

        metrics = VideoMetrics(
            path=path,
            asset_id=asset_id_from_video(path),
            opened=True,
            video_stream_present=source_frame_count > 0,
            codec_readable=metadata_read_ok,
            metadata_read_ok=metadata_read_ok,
            frame_count=clip_frame_count,
            fps=fps,
            duration_seconds=duration,
            width=width,
            height=height,
            display_width=width,
            display_height=height,
            short_side=short_side,
            long_side=long_side,
            sampled_frame_count=sampled_frame_count,
            decoded_sample_count=decoded_sample_count,
            sample_decode_ratio=(
                decoded_sample_count / sampled_frame_count
                if sampled_frame_count
                else 0.0
            ),
            mean_brightness=(
                float(np.mean(brightness_values))
                if brightness_values
                else 0.0
            ),
            black_frame_ratio=black_frame_ratio,
            black_frame_count_estimate=(
                black_frame_count_estimate
            ),
            exposure_defect_frame_ratio=(
                exposure_defect_frame_ratio
            ),
            defect_duration_ratio=defect_duration_ratio,
            mean_over_dark_ratio=(
                float(np.mean(dark_values))
                if dark_values
                else 1.0
            ),
            mean_over_exposed_ratio=(
                float(np.mean(exposed_values))
                if exposed_values
                else 0.0
            ),
            laplacian_min=(
                float(min(blur_values))
                if blur_values
                else 0.0
            ),
            laplacian_p10=_percentile(
                blur_values,
                10,
            ),
            laplacian_median=_percentile(
                blur_values,
                50,
            ),
            mean_blur_laplacian_var=(
                float(np.mean(blur_values))
                if blur_values
                else 0.0
            ),
            laplacian_p90=_percentile(
                blur_values,
                90,
            ),
            laplacian_under_100_ratio=(
                laplacian_under_100_ratio
            ),
            tenengrad_p10=_percentile(
                tenengrad_values,
                10,
            ),
            tenengrad_median=_percentile(
                tenengrad_values,
                50,
            ),
            tenengrad_mean=(
                float(np.mean(tenengrad_values))
                if tenengrad_values
                else 0.0
            ),
            sharpness_scale_short_side=(
                max(scale_values)
                if scale_values
                else short_side
            ),
            adjacent_near_duplicate_count=(
                adjacent_near_duplicate_count
            ),
            adjacent_near_duplicate_ratio=adjacent_ratio,
            freeze_candidate_ratio=candidate_ratio,
            freeze_candidate_frame_count=(
                candidate_frame_count
            ),
            freeze_candidate_duration_sec=sum(
                interval.duration_sec
                for interval in candidate_intervals
            ),
            confirmed_freeze_ratio=confirmed_ratio,
            confirmed_freeze_frame_count=(
                confirmed_frame_count
            ),
            confirmed_freeze_duration_sec=sum(
                interval.duration_sec
                for interval in confirmed_intervals
            ),
            frozen_frame_ratio=confirmed_ratio,
            max_consecutive_frozen_sec=max(
                (
                    interval.duration_sec
                    for interval in confirmed_intervals
                ),
                default=0.0,
            ),
            frozen_intervals=confirmed_intervals,
            pts_monotonic_valid=bool(
                timeline["pts_monotonic_valid"]
            ),
            drop_frame_ratio=float(
                timeline["drop_frame_ratio"]
            ),
            drop_detection_source=str(
                timeline["drop_detection_source"]
            ),
            drop_detection_reliable=bool(
                timeline["drop_detection_reliable"]
            ),
            estimated_missing_frames=int(
                timeline["estimated_missing_frames"]
            ),
            observed_frame_interval_count=int(
                timeline[
                    "observed_frame_interval_count"
                ]
            ),
            frame_interval_p99_ms=float(
                timeline["frame_interval_p99_ms"]
            ),
            max_frame_gap_ms=float(
                timeline["max_frame_gap_ms"]
            ),
            expected_interval_ms=float(
                timeline["expected_interval_ms"]
            ),
            drop_interval_ms=float(
                timeline["drop_interval_ms"]
            ),
            max_gap_fail_ms=float(
                timeline["max_gap_fail_ms"]
            ),
            errors=tuple(errors),
        )

        return VideoFrameRangeAnalysis(
            metrics=metrics,
            source_video_frame_count=source_frame_count,
            decoded_frame_count=decoded_frame_count,
            sampled_local_frame_indices=tuple(
                sample_local_indexes
            ),
        )
    finally:
        capture.release()

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    return run_video_quality_check(args.batch, args.config)
