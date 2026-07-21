#!/usr/bin/env python3
"""Benchmark true-batch SAM3 inference on one canonical JD asset.

This utility deliberately has its own runtime adapter.  It does not change the
formal exhaustive producer, its artifacts, or its resume/checkpoint contract.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from annotation.types import InstanceMask
from qc_common.config import load_qc_acceptance_config
from tools.run_manifest_sam3_containment import (
    DEFAULT_QUERIES,
    SAM3_CONFIG,
    _joint_names,
    _manifest_index,
    configured_sam3_thresholds,
    read_records,
    reshape_jdt_keypoints,
)
from tools.run_manifest_sam3_exhaustive import (
    SequentialVideoSource,
    aggregate_frame_verdict,
    classify_audit_hand,
    enumerate_asset_frames,
)
from tools.sam3_keypoint_containment import json_safe, score_keypoints_against_masks


BENCHMARK_SCHEMA = "jdt_sam3_throughput_benchmark.v1"


@dataclass(frozen=True)
class BatchSegmentationOutput:
    """Masks and measured phases for one true model batch."""

    masks_by_frame: list[list[InstanceMask]]
    preprocess_seconds: float
    inference_seconds: float
    postprocess_seconds: float


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    converted = value
    if hasattr(converted, "detach"):
        converted = converted.detach()
    if hasattr(converted, "cpu"):
        converted = converted.cpu()
    if hasattr(converted, "numpy"):
        converted = converted.numpy()
    return np.asarray(converted)


def _batch_size(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) < 1:
        raise ValueError("SAM3 batch tensor has no leading batch dimension")
    return int(shape[0])


def _repeat_frame_batch(value: Any, repeats: int, expected_batch: int) -> Any:
    """Repeat each frame entry Q times, preserving model-output containers."""
    if isinstance(value, Mapping):
        repeated = {
            key: _repeat_frame_batch(item, repeats, expected_batch)
            for key, item in value.items()
        }
        try:
            return type(value)(**repeated)
        except (TypeError, ValueError):
            return repeated
    if isinstance(value, tuple):
        return tuple(
            _repeat_frame_batch(item, repeats, expected_batch) for item in value
        )
    if isinstance(value, list):
        return [
            _repeat_frame_batch(item, repeats, expected_batch) for item in value
        ]
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) == 0 or int(shape[0]) != expected_batch:
        return value
    if isinstance(value, np.ndarray):
        return np.repeat(value, repeats, axis=0)
    if hasattr(value, "repeat_interleave"):
        return value.repeat_interleave(repeats, dim=0)
    raise TypeError(f"cannot repeat SAM3 vision batch value {type(value).__name__}")


def _tile_query_batch(value: Any, frame_count: int, query_count: int) -> Any:
    """Tile the Q prompt batch once for each frame (frame-major ordering)."""
    if value is None:
        return None
    if _batch_size(value) != query_count:
        raise ValueError("SAM3 text feature batch does not match query count")
    if isinstance(value, np.ndarray):
        multiples = (frame_count,) + (1,) * (value.ndim - 1)
        return np.tile(value, multiples)
    if hasattr(value, "repeat"):
        multiples = (frame_count,) + (1,) * (len(value.shape) - 1)
        return value.repeat(*multiples)
    raise TypeError(f"cannot tile SAM3 text batch value {type(value).__name__}")


def _move_to_device(value: Any, device: str) -> Any:
    return value.to(device) if hasattr(value, "to") else value


def _positive_resolution(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        integer = int(value)
        return integer if integer > 0 and float(value) == integer else None
    return None


def _runtime_resolution(processor: Any, model: Any) -> tuple[int, int]:
    """Read effective H/W from the actual loaded processor/config."""
    image_processor = getattr(processor, "image_processor", None)
    size = getattr(image_processor, "size", None)
    if isinstance(size, Mapping):
        height = _positive_resolution(size.get("height"))
        width = _positive_resolution(size.get("width"))
        if height is not None and width is not None:
            return height, width
        shortest = _positive_resolution(size.get("shortest_edge"))
        if shortest is not None:
            return shortest, shortest
    scalar_size = _positive_resolution(size)
    if scalar_size is not None:
        return scalar_size, scalar_size
    target = _positive_resolution(getattr(processor, "target_size", None))
    if target is not None:
        return target, target
    config = getattr(model, "config", None)
    for candidate in (
        getattr(config, "image_size", None),
        getattr(getattr(config, "vision_config", None), "image_size", None),
    ):
        if isinstance(candidate, Sequence) and not isinstance(candidate, str):
            values = [_positive_resolution(item) for item in candidate]
            if len(values) == 2 and all(item is not None for item in values):
                return int(values[0]), int(values[1])
        scalar = _positive_resolution(candidate)
        if scalar is not None:
            return scalar, scalar
    raise ValueError("could not determine effective SAM3 input resolution")


class HuggingFaceSam3BatchBackend:
    """True-batch SAM3 adapter used only by this benchmark."""

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        torch_module: Any,
        device: str,
        requested_resolution: int | None,
        effective_resolution: tuple[int, int] | None = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.torch = torch_module
        self.device = str(device)
        self.requested_resolution = requested_resolution
        self.effective_resolution = effective_resolution or _runtime_resolution(
            processor, model
        )
        self._text_cache: dict[tuple[str, ...], tuple[Any, Any]] = {}
        self.vision_forward_call_count = 0
        self.model_forward_call_count = 0

    @classmethod
    def from_pretrained(
        cls, model_path: Path, requested_resolution: int | None
    ) -> "HuggingFaceSam3BatchBackend":
        try:
            import torch
            from transformers import Sam3Config, Sam3Model, Sam3Processor
        except ImportError as exc:  # pragma: no cover - cloud dependency boundary
            raise RuntimeError(
                "SAM3 benchmark requires torch and transformers with SAM3 support"
            ) from exc

        model_id = str(Path(model_path).expanduser())
        config = Sam3Config.from_pretrained(model_id)
        processor_kwargs: dict[str, Any] = {}
        if requested_resolution is not None:
            config.image_size = int(requested_resolution)
            processor_kwargs["size"] = {
                "height": int(requested_resolution),
                "width": int(requested_resolution),
            }
        processor = Sam3Processor.from_pretrained(model_id, **processor_kwargs)
        model = Sam3Model.from_pretrained(model_id, config=config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        return cls(
            model=model,
            processor=processor,
            torch_module=torch,
            device=device,
            requested_resolution=requested_resolution,
        )

    def _synchronize(self) -> None:
        cuda = getattr(self.torch, "cuda", None)
        if self.device.startswith("cuda") and cuda is not None:
            cuda.synchronize()

    def reset_peak_memory(self) -> None:
        cuda = getattr(self.torch, "cuda", None)
        if self.device.startswith("cuda") and cuda is not None:
            cuda.reset_peak_memory_stats()

    def peak_memory_bytes(self) -> int:
        cuda = getattr(self.torch, "cuda", None)
        if self.device.startswith("cuda") and cuda is not None:
            return int(cuda.max_memory_allocated())
        return 0

    def _text_features(self, queries: Sequence[str]) -> tuple[Any, Any]:
        key = tuple(str(query) for query in queries)
        cached = self._text_cache.get(key)
        if cached is not None:
            return cached
        inputs = _move_to_device(
            self.processor(text=list(key), return_tensors="pt"), self.device
        )
        with self.torch.no_grad():
            output = self.model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
        embeddings = getattr(output, "pooler_output", output)
        cached = (embeddings, inputs.get("attention_mask"))
        self._text_cache[key] = cached
        return cached

    def prime_queries(self, queries: Sequence[str]) -> None:
        """Build the fixed prompt embedding once, outside measured frame batches."""
        self._text_features(queries)

    def segment_batch(
        self,
        frames: Sequence[np.ndarray],
        queries: Sequence[str],
        config: Mapping[str, Any],
    ) -> BatchSegmentationOutput:
        if not frames:
            return BatchSegmentationOutput([], 0.0, 0.0, 0.0)
        if not queries:
            return BatchSegmentationOutput(
                [[] for _ in frames], 0.0, 0.0, 0.0
            )

        expected_height, expected_width = self.effective_resolution
        for frame in frames:
            if tuple(np.asarray(frame).shape[:2]) != (
                expected_height,
                expected_width,
            ):
                raise ValueError(
                    "benchmark frame does not match effective SAM3 input resolution"
                )

        preprocess_started = time.perf_counter()
        image_inputs = _move_to_device(
            self.processor(images=list(frames), return_tensors="pt"), self.device
        )
        pixel_values = image_inputs["pixel_values"]
        frame_count = len(frames)
        if _batch_size(pixel_values) != frame_count:
            raise ValueError("SAM3 processor did not preserve image batch size")
        original_sizes = image_inputs.get(
            "original_sizes",
            np.asarray([frame.shape[:2] for frame in frames], dtype=np.int64),
        )
        preprocess_seconds = time.perf_counter() - preprocess_started

        query_count = len(queries)
        self._synchronize()
        inference_started = time.perf_counter()
        with self.torch.no_grad():
            text_embeds, attention_mask = self._text_features(queries)
            vision_embeds = self.model.get_vision_features(pixel_values=pixel_values)
            self.vision_forward_call_count += 1
            paired_vision = _repeat_frame_batch(
                vision_embeds, query_count, frame_count
            )
            paired_text = _tile_query_batch(
                text_embeds, frame_count, query_count
            )
            paired_attention = _tile_query_batch(
                attention_mask, frame_count, query_count
            )
            outputs = self.model(
                vision_embeds=paired_vision,
                text_embeds=paired_text,
                attention_mask=paired_attention,
            )
            self.model_forward_call_count += 1
        self._synchronize()
        inference_seconds = time.perf_counter() - inference_started

        target_sizes_array = _to_numpy(original_sizes).reshape(frame_count, 2)
        target_sizes = np.repeat(target_sizes_array, query_count, axis=0).tolist()
        postprocess_started = time.perf_counter()
        processed = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=float(config.get("confidence_threshold", 0.5)),
            mask_threshold=float(config.get("mask_threshold", 0.5)),
            target_sizes=target_sizes,
        )
        expected_pairs = frame_count * query_count
        if len(processed) != expected_pairs:
            raise ValueError(
                f"SAM3 postprocess returned {len(processed)} pairs; "
                f"expected {expected_pairs}"
            )
        max_instances = int(config.get("max_instances_per_query", 10))
        masks_by_frame: list[list[InstanceMask]] = [
            [] for _ in range(frame_count)
        ]
        for pair_index, result in enumerate(processed):
            frame_index = pair_index // query_count
            query_index = pair_index % query_count
            masks = _to_numpy(result.get("masks", []))
            boxes = _to_numpy(result.get("boxes", []))
            scores = _to_numpy(result.get("scores", []))
            for index in range(min(len(masks), max_instances)):
                x1, y1, x2, y2 = np.asarray(boxes[index], dtype=float)
                masks_by_frame[frame_index].append(
                    InstanceMask(
                        category=str(queries[query_index]),
                        mask=np.asarray(masks[index], dtype=bool),
                        score=float(scores[index]),
                        bbox=(
                            int(x1),
                            int(y1),
                            int(x2 - x1),
                            int(y2 - y1),
                        ),
                    )
                )
        postprocess_seconds = time.perf_counter() - postprocess_started
        return BatchSegmentationOutput(
            masks_by_frame=masks_by_frame,
            preprocess_seconds=preprocess_seconds,
            inference_seconds=inference_seconds,
            postprocess_seconds=postprocess_seconds,
        )


def _resize_frame(frame: np.ndarray, resolution: tuple[int, int]) -> np.ndarray:
    height, width = resolution
    array = np.asarray(frame)
    if tuple(array.shape[:2]) == (height, width):
        return array
    import cv2

    return cv2.resize(array, (width, height), interpolation=cv2.INTER_LINEAR)


def _scale_points(
    points: np.ndarray,
    *,
    original_shape: tuple[int, int],
    output_shape: tuple[int, int],
) -> np.ndarray:
    original_height, original_width = original_shape
    output_height, output_width = output_shape
    if original_height < 1 or original_width < 1:
        raise ValueError("source frame has invalid dimensions")
    scaled = np.asarray(points, dtype=np.float64).copy()
    scaled[:, 0] *= output_width / original_width
    scaled[:, 1] *= output_height / original_height
    return scaled


def _chunks(values: Sequence[Any], size: int) -> list[Sequence[Any]]:
    return [values[offset : offset + size] for offset in range(0, len(values), size)]


def _safe_rate(numerator: int, seconds: float) -> float | None:
    return float(numerator / seconds) if seconds > 0 else None


def _mean_or_none(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _max_or_none(values: Sequence[float]) -> float | None:
    return float(np.max(values)) if values else None


def _normalize_resolutions(values: Sequence[str]) -> list[tuple[str, int | None]]:
    normalized: list[tuple[str, int | None]] = []
    seen: set[str] = set()
    for raw in values:
        text = str(raw).strip().lower()
        if not text:
            raise ValueError("input resolution cannot be empty")
        if text == "baseline":
            resolution = None
            canonical = "baseline"
        else:
            try:
                resolution = int(text)
            except ValueError as exc:
                raise ValueError(f"invalid input resolution: {raw}") from exc
            if resolution < 1:
                raise ValueError("input resolutions must be positive")
            canonical = str(resolution)
        if canonical not in seen:
            seen.add(canonical)
            normalized.append((canonical, resolution))
    if "baseline" not in seen:
        raise ValueError("input resolutions must include baseline")
    return sorted(normalized, key=lambda item: item[0] != "baseline")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _score_configuration(
    *,
    requests: Sequence[Mapping[str, Any]],
    frames: Sequence[np.ndarray],
    points_by_frame: Sequence[Mapping[str, np.ndarray]],
    masks_by_frame: Sequence[Sequence[InstanceMask]],
    thresholds: Mapping[str, float],
    requested_resolution: str,
    effective_resolution: tuple[int, int],
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hand_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for request, frame, hand_points, masks in zip(
        requests, frames, points_by_frame, masks_by_frame, strict=True
    ):
        current_hands: list[dict[str, Any]] = []
        for hand in ("left", "right"):
            metrics, _union, _valid, _inside = score_keypoints_against_masks(
                frame=frame,
                pixels=hand_points[hand],
                joint_names=_joint_names(hand),
                masks=list(masks),
                **dict(thresholds),
            )
            raw_verdict = str(metrics["containment_verdict"])
            row = {
                **dict(request),
                "hand": hand,
                "requested_input_resolution": requested_resolution,
                "effective_input_height": effective_resolution[0],
                "effective_input_width": effective_resolution[1],
                "batch_size": int(batch_size),
                "inside_ratio": metrics["keypoint_inside_ratio"],
                "raw_containment_verdict": raw_verdict,
                "hand_verdict": classify_audit_hand(raw_verdict),
                "reason": metrics["reason"],
            }
            current_hands.append(row)
            hand_rows.append(row)
        aggregate = aggregate_frame_verdict(
            {str(row["hand"]): str(row["hand_verdict"]) for row in current_hands},
            required_hands=("left", "right"),
        )
        frame_rows.append(
            {
                **dict(request),
                "requested_input_resolution": requested_resolution,
                "effective_input_height": effective_resolution[0],
                "effective_input_width": effective_resolution[1],
                "batch_size": int(batch_size),
                "left_hand_verdict": current_hands[0]["hand_verdict"],
                "right_hand_verdict": current_hands[1]["hand_verdict"],
                "left_inside_ratio": current_hands[0]["inside_ratio"],
                "right_inside_ratio": current_hands[1]["inside_ratio"],
                **aggregate,
            }
        )
    return hand_rows, frame_rows


def _comparison_rows(
    current_hands: Sequence[Mapping[str, Any]],
    baseline_hands: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float | None]]:
    baseline_by_key = {
        (int(row["source_frame"]), str(row["hand"])): row
        for row in baseline_hands
    }
    details: list[dict[str, Any]] = []
    agreement_count = 0
    ratio_differences: list[float] = []
    for current in current_hands:
        key = (int(current["source_frame"]), str(current["hand"]))
        baseline = baseline_by_key[key]
        verdict_agrees = (
            str(current["hand_verdict"]) == str(baseline["hand_verdict"])
        )
        agreement_count += int(verdict_agrees)
        current_ratio = current.get("inside_ratio")
        baseline_ratio = baseline.get("inside_ratio")
        difference: float | None = None
        if current_ratio is not None and baseline_ratio is not None:
            difference = abs(float(current_ratio) - float(baseline_ratio))
            ratio_differences.append(difference)
        details.append(
            {
                **dict(current),
                "current_hand_verdict": current["hand_verdict"],
                "current_inside_ratio": current_ratio,
                "baseline_hand_verdict": baseline["hand_verdict"],
                "baseline_inside_ratio": baseline_ratio,
                "hand_verdict_agrees": verdict_agrees,
                "inside_ratio_absolute_difference": difference,
            }
        )
    return details, {
        "per_hand_verdict_agreement": (
            agreement_count / len(current_hands) if current_hands else None
        ),
        "inside_ratio_mean_absolute_difference": _mean_or_none(ratio_differences),
        "inside_ratio_max_absolute_difference": _max_or_none(ratio_differences),
    }


def benchmark_jdt_sam3_throughput(
    *,
    manifest: Path,
    asset_id: str,
    sam3_model: Path,
    config_path: Path | None,
    output_dir: Path,
    max_frames: int,
    warmup_frames: int,
    batch_sizes: Sequence[int],
    input_resolutions: Sequence[str],
    decode_workers: int = 4,
    prefetch_frames: int = 32,
    source_reader: Any | None = None,
    backend_factory: Callable[[Path, int | None], HuggingFaceSam3BatchBackend]
    | None = None,
) -> dict[str, Any]:
    """Benchmark one JD asset without publishing formal producer artifacts."""
    manifest = Path(manifest).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output_dir must be empty: {output_dir}")
    if max_frames < 1:
        raise ValueError("max_frames must be >= 1")
    if warmup_frames < 0:
        raise ValueError("warmup_frames must be >= 0")
    batches = sorted({int(value) for value in batch_sizes})
    if not batches or batches[0] < 1:
        raise ValueError("batch sizes must be positive")
    if 1 not in batches:
        raise ValueError("batch sizes must include 1 for baseline comparison")
    resolutions = _normalize_resolutions(input_resolutions)

    loaded_config = load_qc_acceptance_config(config_path)
    frame_thresholds, _window_thresholds = configured_sam3_thresholds(loaded_config)
    manifest_rows = read_records(manifest)
    manifest_index, _order, failures = _manifest_index(
        manifest_rows, manifest_dir=manifest.parent
    )
    if failures:
        raise ValueError(f"manifest contains invalid rows: {failures[0]}")
    if asset_id not in manifest_index:
        raise ValueError(f"asset_id missing from manifest: {asset_id}")
    manifest_row = manifest_index[asset_id]
    requests = enumerate_asset_frames(manifest_row)[: int(max_frames)]
    if not requests:
        raise ValueError("asset contains no frames to benchmark")

    reader = source_reader or SequentialVideoSource(
        decode_workers=decode_workers, prefetch_frames=prefetch_frames
    )
    requested_video_frames = [int(row["video_frame"]) for row in requests]
    decoded: list[tuple[int, np.ndarray, float]] = []
    decode_started = time.perf_counter()
    try:
        source_data = reader.read_parquet(Path(manifest_row["parquet_path"]))
        decoded = [
            (int(index), np.asarray(frame), float(seconds))
            for index, frame, seconds in reader.iter_frames(
                Path(manifest_row["primary_video_path"]), requested_video_frames
            )
        ]
    finally:
        reader.close()
    decode_seconds = time.perf_counter() - decode_started
    actual_video_frames = [item[0] for item in decoded]
    source_frame_order_valid = actual_video_frames == requested_video_frames
    source_frame_coverage_valid = (
        len(actual_video_frames) == len(requested_video_frames)
        and set(actual_video_frames) == set(requested_video_frames)
    )
    if not source_frame_order_valid or not source_frame_coverage_valid:
        raise ValueError(
            "sequential decoder did not return the exact requested frame order/coverage"
        )
    original_frames = [item[1] for item in decoded]
    decoder_reported_seconds = float(sum(item[2] for item in decoded))

    original_points: list[dict[str, np.ndarray]] = []
    for request in requests:
        source_frame = int(request["source_frame"])
        if not 0 <= source_frame < len(source_data):
            raise ValueError(f"source frame outside JD parquet: {source_frame}")
        row = source_data.iloc[source_frame]
        hand_points: dict[str, np.ndarray] = {}
        for hand in ("left", "right"):
            field = str(manifest_row[f"{hand}_hand_2d_field"])
            points = reshape_jdt_keypoints(row[field], field, source_frame)
            if not bool(np.isfinite(points).all()):
                raise ValueError(f"nonfinite JD direct-2D input: {hand}:{source_frame}")
            hand_points[hand] = points
        original_points.append(hand_points)

    queries = [query.strip() for query in DEFAULT_QUERIES.split(",") if query.strip()]
    sam3_config = dict(SAM3_CONFIG)
    factory = backend_factory or HuggingFaceSam3BatchBackend.from_pretrained
    output_dir.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    baseline_hands: list[dict[str, Any]] | None = None
    baseline_frames: list[dict[str, Any]] | None = None
    baseline_resolution: tuple[int, int] | None = None

    for requested_label, requested_resolution in resolutions:
        model_load_started = time.perf_counter()
        backend = factory(Path(sam3_model), requested_resolution)
        model_load_seconds = time.perf_counter() - model_load_started
        effective_resolution = tuple(int(v) for v in backend.effective_resolution)
        if requested_label == "baseline":
            baseline_resolution = effective_resolution

        query_embedding_started = time.perf_counter()
        backend.prime_queries(queries)
        query_embedding_seconds = time.perf_counter() - query_embedding_started

        resolution_started = time.perf_counter()
        resized_frames = [
            _resize_frame(frame, effective_resolution) for frame in original_frames
        ]
        scaled_points: list[dict[str, np.ndarray]] = []
        for frame, hand_points in zip(original_frames, original_points, strict=True):
            scaled_points.append(
                {
                    hand: _scale_points(
                        points,
                        original_shape=tuple(frame.shape[:2]),
                        output_shape=effective_resolution,
                    )
                    for hand, points in hand_points.items()
                }
            )
        resize_seconds = time.perf_counter() - resolution_started

        for batch_size in batches:
            warmup_count = min(int(warmup_frames), len(resized_frames))
            for chunk in _chunks(resized_frames[:warmup_count], batch_size):
                backend.segment_batch(chunk, queries, sam3_config)

            backend.reset_peak_memory()
            start_model_calls = backend.model_forward_call_count
            start_vision_calls = backend.vision_forward_call_count
            masks_by_frame: list[list[InstanceMask]] = []
            preprocess_seconds = 0.0
            inference_seconds = 0.0
            postprocess_seconds = 0.0
            for chunk in _chunks(resized_frames, batch_size):
                output = backend.segment_batch(chunk, queries, sam3_config)
                masks_by_frame.extend(output.masks_by_frame)
                preprocess_seconds += output.preprocess_seconds
                inference_seconds += output.inference_seconds
                postprocess_seconds += output.postprocess_seconds
            if len(masks_by_frame) != len(requests):
                raise ValueError("SAM3 batch output did not preserve frame coverage")

            scoring_started = time.perf_counter()
            hand_rows, frame_rows = _score_configuration(
                requests=requests,
                frames=resized_frames,
                points_by_frame=scaled_points,
                masks_by_frame=masks_by_frame,
                thresholds=frame_thresholds,
                requested_resolution=requested_label,
                effective_resolution=effective_resolution,
                batch_size=batch_size,
            )
            scoring_seconds = time.perf_counter() - scoring_started
            postprocess_seconds += scoring_seconds
            if requested_label == "baseline" and batch_size == 1:
                baseline_hands = [dict(row) for row in hand_rows]
                baseline_frames = [dict(row) for row in frame_rows]
            if baseline_hands is None or baseline_frames is None:
                raise RuntimeError("baseline batch=1 configuration was not evaluated first")
            details, hand_comparison = _comparison_rows(hand_rows, baseline_hands)
            comparison_rows.extend(details)
            baseline_frame_by_source = {
                int(row["source_frame"]): row for row in baseline_frames
            }
            frame_agreement = sum(
                str(row["frame_verdict"])
                == str(baseline_frame_by_source[int(row["source_frame"])]["frame_verdict"])
                for row in frame_rows
            ) / len(frame_rows)
            verdict_counts = {
                name: sum(row["frame_verdict"] == name for row in frame_rows)
                for name in ("fail", "review", "pass", "blocked")
            }
            total_seconds = (
                decode_seconds
                + resize_seconds
                + preprocess_seconds
                + inference_seconds
                + postprocess_seconds
            )
            result_rows.append(
                {
                    "batch_size": batch_size,
                    "requested_input_resolution": requested_label,
                    "effective_input_height": effective_resolution[0],
                    "effective_input_width": effective_resolution[1],
                    "processed_frame_count": len(requests),
                    "warmup_frame_count": warmup_count,
                    "decode_seconds": decode_seconds,
                    "decoder_reported_seconds": decoder_reported_seconds,
                    "resize_seconds": resize_seconds,
                    "preprocess_seconds": preprocess_seconds,
                    "inference_seconds": inference_seconds,
                    "postprocess_seconds": postprocess_seconds,
                    "total_seconds": total_seconds,
                    "inference_frames_per_second": _safe_rate(
                        len(requests), inference_seconds
                    ),
                    "end_to_end_frames_per_second": _safe_rate(
                        len(requests), total_seconds
                    ),
                    "peak_accelerator_memory_bytes": backend.peak_memory_bytes(),
                    "fail_count": verdict_counts["fail"],
                    "review_count": verdict_counts["review"],
                    "pass_count": verdict_counts["pass"],
                    "blocked_count": verdict_counts["blocked"],
                    "frame_verdict_agreement": frame_agreement,
                    **hand_comparison,
                    "source_frame_order_valid": source_frame_order_valid,
                    "source_frame_coverage_valid": source_frame_coverage_valid,
                    "vision_forward_call_count": (
                        backend.vision_forward_call_count - start_vision_calls
                    ),
                    "model_forward_call_count": (
                        backend.model_forward_call_count - start_model_calls
                    ),
                    "model_load_seconds": model_load_seconds,
                    "query_embedding_seconds": query_embedding_seconds,
                }
            )

        cuda = getattr(backend.torch, "cuda", None)
        uses_cuda = backend.device.startswith("cuda")
        del backend
        gc.collect()
        if uses_cuda and cuda is not None:
            cuda.empty_cache()

    assert baseline_hands is not None
    assert baseline_frames is not None
    assert baseline_resolution is not None
    pd.DataFrame(result_rows).to_csv(output_dir / "benchmark_results.csv", index=False)
    _write_json(
        output_dir / "benchmark_results.json",
        {"schema_version": BENCHMARK_SCHEMA, "configurations": result_rows},
    )
    pd.DataFrame(baseline_frames).to_parquet(
        output_dir / "baseline_frame_results.parquet", index=False
    )
    pd.DataFrame(comparison_rows).to_parquet(
        output_dir / "comparison_details.parquet", index=False
    )
    run_config = {
        "schema_version": BENCHMARK_SCHEMA,
        "manifest": str(manifest),
        "asset_id": asset_id,
        "sam3_model": str(Path(sam3_model).expanduser().resolve()),
        "qc_config": loaded_config.json_reference(),
        "max_frames": int(max_frames),
        "warmup_frames": int(warmup_frames),
        "batch_sizes": batches,
        "input_resolutions": [label for label, _value in resolutions],
        "decode_workers": int(decode_workers),
        "prefetch_frames": int(prefetch_frames),
        "queries": queries,
        "sam3_runtime_config": sam3_config,
        "containment_thresholds": frame_thresholds,
        "baseline": {
            "batch_size": 1,
            "requested_input_resolution": "baseline",
            "effective_input_resolution": list(baseline_resolution),
        },
        "model_lifecycle": {
            "precision": "unchanged",
            "torch_compile": False,
            "video_tracking": False,
        },
        "decode_contract": {
            "continuous_session_count": 1,
            "source_frame_order_valid": source_frame_order_valid,
            "source_frame_coverage_valid": source_frame_coverage_valid,
        },
        "artifacts": {"raw_masks_saved": False, "overlays_saved": False},
    }
    _write_json(output_dir / "run_config.json", run_config)
    return {
        "schema_version": BENCHMARK_SCHEMA,
        "configuration_count": len(result_rows),
        "processed_frame_count": len(requests),
        "output_dir": str(output_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark true-batch SAM3 throughput on one JD asset."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--sam3-model", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument(
        "--input-resolutions", nargs="+", default=["baseline", "768", "640"]
    )
    parser.add_argument("--decode-workers", type=int, default=4)
    parser.add_argument("--prefetch-frames", type=int, default=32)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = benchmark_jdt_sam3_throughput(
        manifest=args.manifest,
        asset_id=args.asset_id,
        sam3_model=args.sam3_model,
        config_path=args.config,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        warmup_frames=args.warmup_frames,
        batch_sizes=args.batch_sizes,
        input_resolutions=args.input_resolutions,
        decode_workers=args.decode_workers,
        prefetch_frames=args.prefetch_frames,
    )
    print(json.dumps(json_safe(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
