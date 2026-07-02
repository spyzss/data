from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import yaml


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi"}
DEFAULT_SAMPLE_COUNT = 10


class AlignmentMode(StrEnum):
    IGNORE = "ignore"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class ThresholdConfig:
    min_fps: float = 1.0
    min_width: int = 1
    min_height: int = 1
    min_sample_decode_ratio: float = 1.0
    max_mean_over_dark_ratio: float = 0.10
    max_mean_over_exposed_ratio: float = 0.05
    min_mean_blur_laplacian_var: float = 1.0
    max_black_frame_ratio: float = 0.05
    max_frozen_frame_ratio: float = 0.8
    fail_on_hdf5_frame_mismatch: bool = True


@dataclass(frozen=True)
class VideoQualityConfig:
    sample_count: int = DEFAULT_SAMPLE_COUNT
    alignment_mode: AlignmentMode = AlignmentMode.WARN
    thresholds: ThresholdConfig = ThresholdConfig()

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "alignment_mode": self.alignment_mode.value,
            "thresholds": asdict(self.thresholds),
        }


def _merge_thresholds(raw: dict[str, Any]) -> ThresholdConfig:
    defaults = asdict(ThresholdConfig())
    for key, value in raw.items():
        if key not in defaults:
            raise ValueError(f"unknown threshold: {key}")
        defaults[key] = value
    return ThresholdConfig(**defaults)


def load_video_quality_config(path: Path | None) -> VideoQualityConfig:
    if path is None:
        return VideoQualityConfig()

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    sample_count = int(raw.get("sample_count", DEFAULT_SAMPLE_COUNT))
    if sample_count < 1:
        raise ValueError("sample_count must be >= 1")

    alignment_mode = AlignmentMode(str(raw.get("alignment_mode", AlignmentMode.WARN.value)).lower())
    thresholds = _merge_thresholds(raw.get("thresholds") or {})
    return VideoQualityConfig(sample_count=sample_count, alignment_mode=alignment_mode, thresholds=thresholds)


def discover_batch_videos(batch_dir: Path) -> list[Path]:
    video_dir = batch_dir / "video"
    if not video_dir.is_dir():
        raise FileNotFoundError(f"video directory not found: {video_dir}")
    return sorted(path for path in video_dir.iterdir() if path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS)


@dataclass(frozen=True)
class VideoMetrics:
    path: Path
    asset_id: str
    opened: bool
    frame_count: int
    fps: float
    duration_seconds: float
    width: int
    height: int
    sampled_frame_count: int
    decoded_sample_count: int
    sample_decode_ratio: float
    mean_brightness: float
    mean_over_dark_ratio: float
    mean_over_exposed_ratio: float
    mean_blur_laplacian_var: float
    black_frame_ratio: float
    frozen_frame_ratio: float
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class QualityEvaluation:
    passed: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Hdf5Alignment:
    status: str
    hdf5_path: Path | None
    hdf5_frame_count: int | None
    frame_count_match: bool | None
    reason: str | None = None


@dataclass(frozen=True)
class VideoQualityResult:
    metrics: VideoMetrics
    alignment: Hdf5Alignment
    evaluation: QualityEvaluation


def asset_id_from_video(path: Path) -> str:
    stem = path.stem
    return stem.removesuffix("_video")


def _sample_indexes(frame_count: int, sample_count: int) -> list[int]:
    if frame_count <= 0:
        return []
    count = min(frame_count, sample_count)
    if count == 1:
        return [0]
    return sorted({round(index * (frame_count - 1) / (count - 1)) for index in range(count)})


def _empty_metrics(path: Path, errors: tuple[str, ...]) -> VideoMetrics:
    return VideoMetrics(
        path=path,
        asset_id=asset_id_from_video(path),
        opened=False,
        frame_count=0,
        fps=0.0,
        duration_seconds=0.0,
        width=0,
        height=0,
        sampled_frame_count=0,
        decoded_sample_count=0,
        sample_decode_ratio=0.0,
        mean_brightness=0.0,
        mean_over_dark_ratio=1.0,
        mean_over_exposed_ratio=0.0,
        mean_blur_laplacian_var=0.0,
        black_frame_ratio=1.0,
        frozen_frame_ratio=0.0,
        errors=errors,
    )


def analyze_video(path: Path, config: VideoQualityConfig) -> VideoMetrics:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return _empty_metrics(path, ("cannot_open_video",))

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if frame_count > 0 and fps > 0 else 0.0
        indexes = _sample_indexes(frame_count, config.sample_count)

        brightness_values: list[float] = []
        dark_values: list[float] = []
        exposed_values: list[float] = []
        blur_values: list[float] = []
        black_values: list[float] = []
        frozen_pairs = 0
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
            brightness_values.append(brightness)
            dark_values.append(float(np.mean(gray < 16)))
            exposed_values.append(float(np.mean(gray > 245)))
            blur_values.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            black_values.append(1.0 if brightness < 16 else 0.0)

            if previous_gray is not None:
                diff = float(np.mean(cv2.absdiff(previous_gray, gray)))
                if diff < 1.0:
                    frozen_pairs += 1
            previous_gray = gray

        decoded = len(brightness_values)
        sampled = len(indexes)
        if sampled and decoded == 0:
            errors.append("no_sample_frames_decoded")

        return VideoMetrics(
            path=path,
            asset_id=asset_id_from_video(path),
            opened=True,
            frame_count=frame_count,
            fps=fps,
            duration_seconds=duration,
            width=width,
            height=height,
            sampled_frame_count=sampled,
            decoded_sample_count=decoded,
            sample_decode_ratio=decoded / sampled if sampled else 0.0,
            mean_brightness=float(np.mean(brightness_values)) if brightness_values else 0.0,
            mean_over_dark_ratio=float(np.mean(dark_values)) if dark_values else 1.0,
            mean_over_exposed_ratio=float(np.mean(exposed_values)) if exposed_values else 0.0,
            mean_blur_laplacian_var=float(np.mean(blur_values)) if blur_values else 0.0,
            black_frame_ratio=float(np.mean(black_values)) if black_values else 1.0,
            frozen_frame_ratio=frozen_pairs / (decoded - 1) if decoded > 1 else 0.0,
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
    if not hdf5_path.is_file():
        return Hdf5Alignment("missing", hdf5_path, None, None, "hdf5_missing")

    try:
        hdf5_frame_count = infer_hdf5_frame_count(hdf5_path)
    except (OSError, ValueError) as exc:
        return Hdf5Alignment("unreadable", hdf5_path, None, None, f"hdf5_unreadable:{exc}")

    matches = hdf5_frame_count == metrics.frame_count
    return Hdf5Alignment(
        "matched" if matches else "mismatch",
        hdf5_path,
        hdf5_frame_count,
        matches,
        None if matches else "hdf5_frame_count_mismatch",
    )


def evaluate_video_quality(
    metrics: VideoMetrics,
    config: VideoQualityConfig,
    alignment: object | None = None,
) -> QualityEvaluation:
    thresholds = config.thresholds
    reasons: list[str] = list(metrics.errors)

    if not metrics.opened:
        reasons.append("video_not_opened")
    if metrics.fps < thresholds.min_fps:
        reasons.append("fps_below_min")
    if metrics.width < thresholds.min_width:
        reasons.append("width_below_min")
    if metrics.height < thresholds.min_height:
        reasons.append("height_below_min")
    if metrics.sample_decode_ratio < thresholds.min_sample_decode_ratio:
        reasons.append("sample_decode_ratio_below_min")
    if metrics.mean_over_dark_ratio > thresholds.max_mean_over_dark_ratio:
        reasons.append("mean_over_dark_ratio_above_max")
    if metrics.mean_over_exposed_ratio > thresholds.max_mean_over_exposed_ratio:
        reasons.append("mean_over_exposed_ratio_above_max")
    if metrics.mean_blur_laplacian_var < thresholds.min_mean_blur_laplacian_var:
        reasons.append("mean_blur_laplacian_var_below_min")
    if metrics.black_frame_ratio > thresholds.max_black_frame_ratio:
        reasons.append("black_frame_ratio_above_max")
    if metrics.frozen_frame_ratio > thresholds.max_frozen_frame_ratio:
        reasons.append("frozen_frame_ratio_above_max")
    if isinstance(alignment, Hdf5Alignment):
        if alignment.status == "mismatch" and thresholds.fail_on_hdf5_frame_mismatch:
            reasons.append("hdf5_frame_count_mismatch")
        elif alignment.status == "missing" and config.alignment_mode == AlignmentMode.FAIL:
            reasons.append("hdf5_missing")
        elif alignment.status == "unreadable" and config.alignment_mode == AlignmentMode.FAIL:
            reasons.append("hdf5_unreadable")

    unique_reasons = tuple(dict.fromkeys(reasons))
    return QualityEvaluation(passed=not unique_reasons, reasons=unique_reasons)


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def write_video_quality_reports(
    reports_dir: Path,
    results: list[VideoQualityResult],
    config: VideoQualityConfig,
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "video_quality.csv"
    fields = [
        "asset_id",
        "video_path",
        "passed",
        "reasons",
        "frame_count",
        "fps",
        "duration_seconds",
        "width",
        "height",
        "sampled_frame_count",
        "decoded_sample_count",
        "sample_decode_ratio",
        "mean_brightness",
        "mean_over_dark_ratio",
        "mean_over_exposed_ratio",
        "mean_blur_laplacian_var",
        "black_frame_ratio",
        "frozen_frame_ratio",
        "hdf5_frame_count",
        "frame_count_match",
        "hdf5_alignment_status",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            metrics = result.metrics
            alignment = result.alignment
            writer.writerow(
                {
                    "asset_id": metrics.asset_id,
                    "video_path": str(metrics.path),
                    "passed": _format_bool(result.evaluation.passed),
                    "reasons": ";".join(result.evaluation.reasons),
                    "frame_count": metrics.frame_count,
                    "fps": metrics.fps,
                    "duration_seconds": metrics.duration_seconds,
                    "width": metrics.width,
                    "height": metrics.height,
                    "sampled_frame_count": metrics.sampled_frame_count,
                    "decoded_sample_count": metrics.decoded_sample_count,
                    "sample_decode_ratio": metrics.sample_decode_ratio,
                    "mean_brightness": metrics.mean_brightness,
                    "mean_over_dark_ratio": metrics.mean_over_dark_ratio,
                    "mean_over_exposed_ratio": metrics.mean_over_exposed_ratio,
                    "mean_blur_laplacian_var": metrics.mean_blur_laplacian_var,
                    "black_frame_ratio": metrics.black_frame_ratio,
                    "frozen_frame_ratio": metrics.frozen_frame_ratio,
                    "hdf5_frame_count": alignment.hdf5_frame_count if alignment.hdf5_frame_count is not None else "",
                    "frame_count_match": "" if alignment.frame_count_match is None else _format_bool(alignment.frame_count_match),
                    "hdf5_alignment_status": alignment.status,
                }
            )

    reason_counts: Counter[str] = Counter()
    for result in results:
        reason_counts.update(result.evaluation.reasons)

    failed = sum(1 for result in results if not result.evaluation.passed)
    summary = {
        "batch_status": "passed" if failed == 0 else "failed",
        "total_videos": len(results),
        "passed_videos": len(results) - failed,
        "failed_videos": failed,
        "reason_counts": dict(sorted(reason_counts.items())),
        "effective_config": config.to_dict(),
    }
    (reports_dir / "video_quality_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_video_quality_check(
    batch_dir: Path,
    config_path: Path | None = None,
    reports_dir: Path | None = None,
) -> int:
    config = load_video_quality_config(config_path)
    video_paths = discover_batch_videos(batch_dir)
    output_dir = reports_dir or (batch_dir / "reports")
    results: list[VideoQualityResult] = []

    for video_path in video_paths:
        metrics = analyze_video(video_path, config)
        alignment = check_hdf5_alignment(video_path, batch_dir, metrics, config)
        evaluation = evaluate_video_quality(metrics, config, alignment)
        results.append(VideoQualityResult(metrics, alignment, evaluation))

    write_video_quality_reports(output_dir, results, config)
    return 0 if all(result.evaluation.passed for result in results) else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--reports-dir", type=Path)
    args = parser.parse_args(argv)
    return run_video_quality_check(args.batch, args.config, args.reports_dir)
