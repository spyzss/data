#!/usr/bin/env python3
"""Run a candidate-independent, frame-by-frame JD SAM3 audit."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from qc_common.config import load_qc_acceptance_config
from qc_pipeline.artifacts import canonical_sha256, file_sha256
from qc_pipeline.sam3_runtime import Sam3RuntimeProvider
from tools.run_manifest_sam3_containment import (
    DEFAULT_QUERIES,
    SAM3_CONFIG,
    _joint_names,
    _manifest_index,
    configured_sam3_thresholds,
    read_records,
    reshape_jdt_keypoints,
)
from tools.sam3_keypoint_containment import (
    json_safe,
    score_keypoints_against_masks,
    write_combined_overlay_image,
)


_RAW_HAND_VERDICT_MAP = {
    "strong_containment_mismatch": "fail",
    "containment_review": "review",
    "projection_review": "review",
    "mask_missing_or_tiny_review": "review",
    "likely_visible_ok": "pass",
}

EVIDENCE_COLUMNS = (
    "asset_id",
    "source_frame",
    "video_frame",
    "sam3_frame_verdict",
    "source_video",
    "overlay_path",
    "left_inside_ratio",
    "right_inside_ratio",
    "model_config_identity",
)

PRODUCER_VERSION = "jdt-sam3-exhaustive-producer-v1"


class StaleResumeError(RuntimeError):
    """Raised when an output root cannot be resumed with current inputs."""


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    return json_safe(value)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(json_safe(dict(value)), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _path_identity(path: Path, *, content_hash: bool = False) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    identity: dict[str, Any] = {
        "resolved_path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if content_hash:
        identity["sha256"] = file_sha256(resolved)
    return identity


def _model_identity(model_path: Path | None) -> dict[str, Any]:
    if model_path is None:
        return {"kind": "injected_segmenter"}
    resolved = Path(model_path).expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"sam3_model must be an existing directory: {model_path}")
    files: dict[str, Any] = {}
    for name in ("config.json", "model.safetensors", "sam3.pt"):
        child = resolved / name
        if not child.is_file():
            files[name] = {"missing": True}
            continue
        files[name] = _path_identity(child, content_hash=name == "config.json")
    return {"kind": "directory", "resolved_path": str(resolved), "files": files}


def _asset_component(asset_id: str) -> str:
    digest = hashlib.sha256(asset_id.encode("utf-8")).hexdigest()[:12]
    readable = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in asset_id
    )[:80]
    return f"{readable}--{digest}"


def _evidence_clip_id(asset_id: str, source_frame: int, video_frame: int) -> str:
    return (
        f"{_asset_component(asset_id)}"
        f"__source_{int(source_frame):012d}"
        f"__video_{int(video_frame):012d}"
    )


def _chunk_slices(
    requests: Sequence[Mapping[str, Any]], size: int
) -> list[list[dict[str, Any]]]:
    return [
        [dict(row) for row in requests[offset : offset + size]]
        for offset in range(0, len(requests), size)
    ]


def _chunk_directory(asset_root: Path, requests: Sequence[Mapping[str, Any]]) -> Path:
    start = int(requests[0]["source_frame"])
    end = int(requests[-1]["source_frame"])
    return asset_root / f"chunk_{start:012d}_{end:012d}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_chunk(
    chunk_dir: Path, expected_fingerprint: Mapping[str, Any]
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
] | None:
    try:
        completion = json.loads(
            (chunk_dir / "completion.json").read_text(encoding="utf-8")
        )
        if completion.get("fingerprint") != dict(expected_fingerprint):
            return None
        hands = _read_jsonl(chunk_dir / "hand_results.jsonl")
        frames = _read_jsonl(chunk_dir / "frame_results.jsonl")
        failures = _read_jsonl(chunk_dir / "failures.jsonl")
        evidence = _read_jsonl(chunk_dir / "evidence.jsonl")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if len(frames) != int(completion.get("frame_count", -1)):
        return None
    if len(hands) != int(completion.get("hand_row_count", -1)):
        return None
    if len(evidence) != int(completion.get("evidence_count", -1)):
        return None
    return hands, frames, failures, evidence


def _chunk_contract_valid(
    loaded: tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ],
    requests: Sequence[Mapping[str, Any]],
) -> bool:
    hands, frames, _failures, evidence = loaded
    expected_frames = [
        (str(row["asset_id"]), int(row["source_frame"]), int(row["video_frame"]))
        for row in requests
    ]
    actual_frames = [
        (str(row.get("asset_id", "")), int(row.get("source_frame", -1)), int(row.get("video_frame", -1)))
        for row in frames
    ]
    if actual_frames != expected_frames:
        return False
    expected_hands = {
        (*key, hand) for key in expected_frames for hand in ("left", "right")
    }
    actual_hands = {
        (
            str(row.get("asset_id", "")),
            int(row.get("source_frame", -1)),
            int(row.get("video_frame", -1)),
            str(row.get("hand", "")),
        )
        for row in hands
    }
    if len(hands) != len(expected_hands) or actual_hands != expected_hands:
        return False
    evidence_keys = [
        (str(row.get("asset_id", "")), int(row.get("source_frame", -1)))
        for row in evidence
    ]
    expected_keys = {(asset_id, source) for asset_id, source, _video in expected_frames}
    return len(evidence_keys) == len(set(evidence_keys)) and set(evidence_keys) <= expected_keys


def _write_chunk(
    chunk_dir: Path,
    *,
    fingerprint: Mapping[str, Any],
    hand_rows: Sequence[Mapping[str, Any]],
    frame_rows: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
) -> None:
    chunk_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{chunk_dir.name}.staging-", dir=chunk_dir.parent)
    )
    backup = chunk_dir.parent / f".{chunk_dir.name}.backup"
    try:
        _write_jsonl(staging / "hand_results.jsonl", hand_rows)
        _write_jsonl(staging / "frame_results.jsonl", frame_rows)
        _write_jsonl(staging / "failures.jsonl", failures)
        _write_jsonl(staging / "evidence.jsonl", evidence_rows)
        _atomic_write_json(
            staging / "completion.json",
            {
                "fingerprint": dict(fingerprint),
                "frame_count": len(frame_rows),
                "hand_row_count": len(hand_rows),
                "evidence_count": len(evidence_rows),
            },
        )
        if backup.exists():
            shutil.rmtree(backup)
        if chunk_dir.exists():
            os.replace(chunk_dir, backup)
        os.replace(staging, chunk_dir)
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run candidate-independent stride-1 JD SAM3 audit."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--supplier", choices=("jdt",), required=True)
    parser.add_argument("--sam3-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--asset-ids", nargs="*")
    parser.add_argument("--max-assets", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--decode-workers", type=int, default=4)
    parser.add_argument("--prefetch-frames", type=int, default=32)
    parser.add_argument("--writer-workers", type=int, default=1)
    parser.add_argument("--gpu-inference-workers", type=int, default=1)
    parser.add_argument("--asset-inference-concurrency", type=int, default=1)
    parser.add_argument("--checkpoint-every-frames", type=int, default=1000)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--save-positive-overlays",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-overlays", type=int)
    return parser


def cuda_runtime_status(torch_module: Any | None = None) -> dict[str, Any]:
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except ImportError:
            return {"cuda_available": False, "cuda_device_count": 0}
    cuda = torch_module.cuda
    available = bool(cuda.is_available())
    return {
        "cuda_available": available,
        "cuda_device_count": int(cuda.device_count()) if available else 0,
    }


@lru_cache(maxsize=1)
def _detected_cuda_runtime() -> dict[str, Any]:
    return cuda_runtime_status()


class SequentialVideoSource:
    """Sequential OpenCV decode with bounded parallel RGB conversion."""

    def __init__(
        self,
        *,
        decode_workers: int = 4,
        prefetch_frames: int = 32,
        capture_factory: Any | None = None,
    ) -> None:
        if decode_workers < 1:
            raise ValueError("decode_workers must be >= 1")
        if prefetch_frames < 1:
            raise ValueError("prefetch_frames must be >= 1")
        self.decode_workers = int(decode_workers)
        self.prefetch_frames = int(prefetch_frames)
        self._capture_factory = capture_factory or self._open_capture

    @staticmethod
    def _open_capture(path: Path) -> Any:
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"could not open source video: {path}")
        return capture

    @staticmethod
    def _rgb_frame(frame_bgr: Any, decode_seconds: float) -> tuple[Any, float]:
        return np.asarray(frame_bgr)[..., ::-1].copy(), float(decode_seconds)

    def read_parquet(self, path: Path) -> pd.DataFrame:
        return pd.read_parquet(Path(path))

    def video_metadata(self, path: Path) -> dict[str, float | int]:
        capture = self._capture_factory(Path(path))
        try:
            return {
                "frame_count": int(round(float(capture.get(7)))),
                "width": int(round(float(capture.get(3)))),
                "height": int(round(float(capture.get(4)))),
                "fps": float(capture.get(5)),
            }
        finally:
            capture.release()

    def iter_frames(self, path: Path, video_frames: list[int]):
        if not video_frames:
            return
        requested = [int(value) for value in video_frames]
        if requested != sorted(set(requested)):
            raise ValueError("requested video frames must be unique and increasing")
        requested_set = set(requested)
        first, last = requested[0], requested[-1]
        capture = self._capture_factory(Path(path))
        pending: deque[tuple[int, Any]] = deque()
        try:
            if not capture.set(1, first):
                raise ValueError(f"could not seek source video to frame {first}: {path}")
            with ThreadPoolExecutor(max_workers=self.decode_workers) as pool:
                current = first
                while current <= last:
                    started = time.perf_counter()
                    ok, frame_bgr = capture.read()
                    decode_seconds = time.perf_counter() - started
                    if not ok or frame_bgr is None:
                        break
                    if current in requested_set:
                        pending.append(
                            (
                                current,
                                pool.submit(
                                    self._rgb_frame, frame_bgr, decode_seconds
                                ),
                            )
                        )
                    current += 1
                    if len(pending) >= self.prefetch_frames:
                        frame_idx, future = pending.popleft()
                        frame_rgb, elapsed = future.result()
                        yield frame_idx, frame_rgb, elapsed
                while pending:
                    frame_idx, future = pending.popleft()
                    frame_rgb, elapsed = future.result()
                    yield frame_idx, frame_rgb, elapsed
        finally:
            capture.release()

    def close(self) -> None:
        return None


def enumerate_asset_frames(manifest_row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Enumerate the current JD identity mapping over an inclusive source range."""
    asset_id = str(manifest_row["asset_id"])
    start_frame = int(manifest_row["start_frame"])
    end_frame = int(manifest_row["end_frame"])
    if end_frame < start_frame:
        raise ValueError("manifest source-frame range is reversed")
    return [
        {
            "asset_id": asset_id,
            "source_frame": source_frame,
            "video_frame": source_frame,
            "local_frame": source_frame - start_frame,
        }
        for source_frame in range(start_frame, end_frame + 1)
    ]


def select_asset_ids(
    manifest_asset_ids: Sequence[str],
    *,
    requested: Sequence[str] | None,
    max_assets: int | None,
    num_shards: int,
    shard_index: int,
) -> list[str]:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= index < num_shards")
    if max_assets is not None and max_assets < 1:
        raise ValueError("max_assets must be >= 1")
    available = {str(asset_id) for asset_id in manifest_asset_ids}
    if requested is None:
        selected = sorted(available)
    else:
        requested_set = {str(asset_id) for asset_id in requested}
        missing = sorted(requested_set - available)
        if missing:
            raise ValueError("requested asset_ids missing from manifest: " + ", ".join(missing))
        selected = sorted(requested_set)
    if max_assets is not None:
        selected = selected[:max_assets]
    return [
        asset_id
        for position, asset_id in enumerate(selected)
        if position % num_shards == shard_index
    ]


def classify_audit_hand(raw_verdict: str) -> str:
    """Map the established raw containment class to an audit-only verdict."""
    return _RAW_HAND_VERDICT_MAP.get(str(raw_verdict), "blocked")


def aggregate_frame_verdict(
    hand_verdicts: Mapping[str, str],
    *,
    required_hands: Sequence[str],
) -> dict[str, Any]:
    """Combine required hands without converting an unevaluable hand to pass."""
    normalized = {
        hand: str(hand_verdicts.get(hand, "blocked")) for hand in required_hands
    }
    values = tuple(normalized.values())
    evaluable = sum(value in {"pass", "review", "fail"} for value in values)
    if "fail" in values:
        frame_verdict = "fail"
    elif "review" in values:
        frame_verdict = "review"
    elif values and all(value == "pass" for value in values):
        frame_verdict = "pass"
    else:
        frame_verdict = "blocked"
    return {
        "frame_verdict": frame_verdict,
        "required_hand_count": len(required_hands),
        "evaluable_hand_count": evaluable,
        "blocked_hand_count": len(required_hands) - evaluable,
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(json_safe(dict(row)), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _blocked_hand_row(
    request: Mapping[str, Any],
    hand: str,
    *,
    reason: str,
    manifest_row: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **dict(request),
        "supplier": "jdt",
        "hand": hand,
        "coordinate_space": "source",
        "source_video_path": str(manifest_row["primary_video_path"]),
        "direct_2d_path": str(manifest_row["parquet_path"]),
        "direct_2d_field": str(manifest_row[f"{hand}_hand_2d_field"]),
        "valid_joint_count": 0,
        "projected_in_image_count": 0,
        "projected_in_image_ratio": None,
        "mask_status": "not_run",
        "inside_count": 0,
        "inside_ratio": None,
        "raw_containment_verdict": "blocked",
        "hand_verdict": "blocked",
        "reason": reason,
        "elapsed_seconds": 0.0,
    }


def _frame_row(
    request: Mapping[str, Any],
    hand_rows: Sequence[Mapping[str, Any]],
    *,
    required_hands: Sequence[str],
    forced_reason: str | None = None,
) -> dict[str, Any]:
    by_hand = {str(row["hand"]): row for row in hand_rows}
    aggregate = aggregate_frame_verdict(
        {hand: str(row["hand_verdict"]) for hand, row in by_hand.items()},
        required_hands=required_hands,
    )
    reasons = sorted(
        {
            str(row.get("reason"))
            for row in hand_rows
            if row.get("reason") is not None
        }
        | ({forced_reason} if forced_reason else set())
    )
    first_hand = hand_rows[0] if hand_rows else {}
    decode_seconds = max(
        (float(row.get("decode_seconds", 0.0) or 0.0) for row in hand_rows),
        default=0.0,
    )
    inference_seconds = max(
        (float(row.get("inference_seconds", 0.0) or 0.0) for row in hand_rows),
        default=0.0,
    )
    scoring_seconds = sum(
        float(row.get("elapsed_seconds", 0.0) or 0.0) for row in hand_rows
    )
    return {
        **dict(request),
        "supplier": "jdt",
        "coordinate_space": first_hand.get("coordinate_space", "source"),
        "source_video_path": first_hand.get("source_video_path"),
        "direct_2d_path": first_hand.get("direct_2d_path"),
        "direct_2d_provenance": "jd_parquet_direct_2d",
        "left_hand_verdict": by_hand.get("left", {}).get(
            "hand_verdict", "blocked"
        ),
        "right_hand_verdict": by_hand.get("right", {}).get(
            "hand_verdict", "blocked"
        ),
        "left_inside_ratio": by_hand.get("left", {}).get("inside_ratio"),
        "right_inside_ratio": by_hand.get("right", {}).get("inside_ratio"),
        "left_projected_in_image_ratio": by_hand.get("left", {}).get(
            "projected_in_image_ratio"
        ),
        "right_projected_in_image_ratio": by_hand.get("right", {}).get(
            "projected_in_image_ratio"
        ),
        "left_mask_status": by_hand.get("left", {}).get("mask_status", "not_run"),
        "right_mask_status": by_hand.get("right", {}).get("mask_status", "not_run"),
        "frame_reason_codes": json.dumps(reasons, ensure_ascii=False),
        "decode_seconds": decode_seconds,
        "inference_seconds": inference_seconds,
        "processing_seconds": decode_seconds + inference_seconds + scoring_seconds,
        **aggregate,
    }


def _compute_chunk(
    *,
    requests: Sequence[Mapping[str, Any]],
    manifest_row: Mapping[str, Any],
    source_data: pd.DataFrame,
    source_reader: Any,
    segmenter: Any,
    frame_thresholds: Mapping[str, float],
    queries: Sequence[str],
    overlay_writer: Any,
    overlay_dir: Path,
    save_positive_overlays: bool,
    max_overlays: int | None,
    model_config_identity: str,
    video_metadata: Mapping[str, Any],
    decoded_frames: Mapping[int, tuple[Any, float]] | None = None,
    duplicate_decoded_frames: set[int] | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, float],
]:
    required_hands = ("left", "right")
    decoded_by_frame: dict[int, tuple[Any, float]] = dict(decoded_frames or {})
    duplicate_decoded_frames = set(duplicate_decoded_frames or ())
    video_frame_count = int(video_metadata["frame_count"])
    requested_video_frames = [
        int(row["video_frame"])
        for row in requests
        if 0 <= int(row["video_frame"]) < video_frame_count
    ]
    if decoded_frames is None:
        for video_frame, frame, decode_seconds in source_reader.iter_frames(
            Path(manifest_row["primary_video_path"]), requested_video_frames
        ):
            index = int(video_frame)
            if index in decoded_by_frame:
                duplicate_decoded_frames.add(index)
            else:
                decoded_by_frame[index] = (frame, float(decode_seconds))

    hand_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    decode_seconds_total = 0.0
    inference_seconds_total = 0.0
    evidence_write_seconds_total = 0.0
    for request in requests:
        source_frame = int(request["source_frame"])
        video_frame = int(request["video_frame"])
        if not 0 <= video_frame < video_frame_count:
            missing_reason = "video_frame_out_of_range"
        elif video_frame in duplicate_decoded_frames:
            missing_reason = "duplicate_decoded_video_frame"
        elif video_frame not in decoded_by_frame:
            missing_reason = "video_early_eof"
        else:
            missing_reason = None
        if missing_reason is not None:
            blocked = [
                _blocked_hand_row(
                    request,
                    hand,
                    reason=missing_reason,
                    manifest_row=manifest_row,
                )
                for hand in required_hands
            ]
            hand_rows.extend(blocked)
            frame_rows.append(
                _frame_row(
                    request,
                    blocked,
                    required_hands=required_hands,
                    forced_reason=missing_reason,
                )
            )
            failures.append(
                {
                    "asset_id": request["asset_id"],
                    "source_frame": source_frame,
                    "video_frame": video_frame,
                    "stage": "video_decode",
                    "reason": missing_reason,
                }
            )
            continue

        frame, decode_seconds = decoded_by_frame[video_frame]
        decode_seconds_total += decode_seconds
        prepared: dict[str, tuple[str, np.ndarray]] = {}
        blocked_by_hand: dict[str, dict[str, Any]] = {}
        for hand in required_hands:
            field = str(manifest_row[f"{hand}_hand_2d_field"])
            try:
                if not 0 <= source_frame < len(source_data):
                    raise ValueError("source_frame_outside_direct_2d")
                if field not in source_data.columns:
                    raise ValueError(f"direct_2d_field_missing:{field}")
                pixels = reshape_jdt_keypoints(
                    source_data.iloc[source_frame][field], field, source_frame
                )
                if not bool(np.isfinite(pixels).all()):
                    raise ValueError("direct_2d_nonfinite")
                prepared[hand] = (field, pixels)
            except (IndexError, KeyError, TypeError, ValueError) as exc:
                blocked_by_hand[hand] = _blocked_hand_row(
                    request,
                    hand,
                    reason=str(exc),
                    manifest_row=manifest_row,
                )

        masks: list[Any] = []
        inference_seconds = 0.0
        inference_error: str | None = None
        if prepared:
            inference_started = time.perf_counter()
            try:
                masks = segmenter.segment_frame(
                    frame,
                    list(queries),
                    dict(SAM3_CONFIG),
                )
            except Exception as exc:  # isolated runtime boundary
                inference_error = f"sam3_runtime_error:{type(exc).__name__}:{exc}"
            inference_seconds = time.perf_counter() - inference_started
            inference_seconds_total += inference_seconds

        current_hands: list[dict[str, Any]] = []
        overlay_hands: dict[str, dict[str, Any]] = {}
        for hand in required_hands:
            if hand in blocked_by_hand:
                current_hands.append(blocked_by_hand[hand])
                continue
            field, pixels = prepared[hand]
            if inference_error is not None:
                current_hands.append(
                    _blocked_hand_row(
                        request,
                        hand,
                        reason=inference_error,
                        manifest_row=manifest_row,
                    )
                )
                continue
            hand_started = time.perf_counter()
            metrics, _mask, valid, inside = score_keypoints_against_masks(
                frame=frame,
                pixels=pixels,
                joint_names=_joint_names(hand),
                masks=masks,
                **frame_thresholds,
            )
            raw_verdict = str(metrics["containment_verdict"])
            current_hands.append(
                {
                    **dict(request),
                    "supplier": "jdt",
                    "hand": hand,
                    "coordinate_space": "source",
                    "source_video_path": str(manifest_row["primary_video_path"]),
                    "direct_2d_path": str(manifest_row["parquet_path"]),
                    "direct_2d_field": field,
                    "valid_joint_count": metrics["valid_projected_keypoints"],
                    "projected_in_image_count": metrics[
                        "projected_keypoints_in_image"
                    ],
                    "projected_in_image_ratio": metrics[
                        "projected_keypoints_in_image_ratio"
                    ],
                    "mask_status": (
                        "present" if metrics["hand_mask_present"] else "missing"
                    ),
                    "inside_count": metrics["inside_keypoints"],
                    "inside_ratio": metrics["keypoint_inside_ratio"],
                    "raw_containment_verdict": raw_verdict,
                    "hand_verdict": classify_audit_hand(raw_verdict),
                    "reason": metrics["reason"],
                    "decode_seconds": decode_seconds,
                    "inference_seconds": inference_seconds,
                    "elapsed_seconds": time.perf_counter() - hand_started,
                }
            )
            overlay_hands[hand] = {
                "pixels": pixels,
                "valid": valid,
                "inside": inside,
                "joint_names": _joint_names(hand),
            }
        if inference_error is not None:
            failures.append(
                {
                    "asset_id": request["asset_id"],
                    "source_frame": source_frame,
                    "video_frame": video_frame,
                    "stage": "sam3_inference",
                    "reason": inference_error,
                }
            )
        hand_rows.extend(current_hands)
        combined = _frame_row(
            request, current_hands, required_hands=required_hands
        )
        if (
            save_positive_overlays
            and combined["frame_verdict"] in {"fail", "review"}
            and (max_overlays is None or len(evidence_rows) < max_overlays)
        ):
            evidence_write_started = time.perf_counter()
            try:
                overlay_path = overlay_writer(
                    frame=frame,
                    hands=overlay_hands,
                    clip_id=_evidence_clip_id(
                        str(request["asset_id"]), source_frame, video_frame
                    ),
                    frame_idx=source_frame,
                    output_dir=overlay_dir,
                )
                combined["evidence_path"] = str(overlay_path)
                evidence_rows.append(
                    {
                        "asset_id": request["asset_id"],
                        "source_frame": source_frame,
                        "video_frame": video_frame,
                        "sam3_frame_verdict": combined["frame_verdict"],
                        "source_video": str(manifest_row["primary_video_path"]),
                        "overlay_path": str(overlay_path),
                        "left_inside_ratio": combined["left_inside_ratio"],
                        "right_inside_ratio": combined["right_inside_ratio"],
                        "model_config_identity": model_config_identity,
                    }
                )
            except Exception as exc:  # evidence failure must not corrupt raw results
                failures.append(
                    {
                        "asset_id": request["asset_id"],
                        "source_frame": source_frame,
                        "video_frame": video_frame,
                        "stage": "evidence_write",
                        "reason": f"evidence_write_error:{type(exc).__name__}:{exc}",
                    }
                )
                combined["evidence_status"] = "write_failed"
            finally:
                evidence_write_seconds_total += (
                    time.perf_counter() - evidence_write_started
                )
        frame_rows.append(combined)
    return (
        hand_rows,
        frame_rows,
        failures,
        evidence_rows,
        {
            "decode_seconds": decode_seconds_total,
            "inference_seconds": inference_seconds_total,
            "evidence_write_seconds": evidence_write_seconds_total,
        },
    )


def _write_producer_outputs(
    output_dir: Path,
    *,
    hand_rows: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    summary: Mapping[str, Any],
    evidence_rows: Sequence[Mapping[str, Any]] = (),
    writer_workers: int = 1,
) -> tuple[float, int]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    hand_frame = pd.DataFrame([json_safe(row) for row in hand_rows])
    frame = pd.DataFrame([json_safe(row) for row in frame_rows])
    evidence_frame = pd.DataFrame(
        [json_safe(dict(row)) for row in evidence_rows],
        columns=EVIDENCE_COLUMNS,
    )

    def write_hand_jsonl() -> None:
        _write_jsonl(output_dir / "sam3_exhaustive_hand_results.jsonl", hand_rows)

    def write_hand_parquet() -> None:
        hand_frame.to_parquet(
            output_dir / "sam3_exhaustive_hand_results.parquet", index=False
        )

    def write_frame_parquet() -> None:
        frame.to_parquet(
            output_dir / "sam3_exhaustive_frame_results.parquet", index=False
        )

    def write_frame_csv() -> None:
        frame.to_csv(output_dir / "sam3_exhaustive_frame_results.csv", index=False)

    def write_failures() -> None:
        _write_jsonl(output_dir / "sam3_exhaustive_failures.jsonl", failures)

    def write_evidence() -> None:
        evidence_frame.to_csv(
            output_dir / "review_evidence_manifest.csv", index=False
        )

    jobs = (
        write_hand_jsonl,
        write_hand_parquet,
        write_frame_parquet,
        write_frame_csv,
        write_failures,
        write_evidence,
    )
    with ThreadPoolExecutor(max_workers=writer_workers) as pool:
        futures = [pool.submit(job) for job in jobs]
        for future in futures:
            future.result()
    result_paths = (
        output_dir / "sam3_exhaustive_hand_results.jsonl",
        output_dir / "sam3_exhaustive_hand_results.parquet",
        output_dir / "sam3_exhaustive_frame_results.parquet",
        output_dir / "sam3_exhaustive_frame_results.csv",
        output_dir / "sam3_exhaustive_failures.jsonl",
        output_dir / "review_evidence_manifest.csv",
    )
    result_bytes = sum(path.stat().st_size for path in result_paths)
    return time.perf_counter() - started, result_bytes


def _peak_cuda_memory_bytes() -> int | None:
    try:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            return None
        return int(torch.cuda.max_memory_allocated())
    except (ImportError, RuntimeError):
        return None


def _decoded_chunk(
    iterator: Any,
    lookahead: tuple[int, Any, float] | None,
    *,
    requested_video_frames: Sequence[int],
) -> tuple[
    dict[int, tuple[Any, float]],
    set[int],
    tuple[int, Any, float] | None,
]:
    """Consume one ordered checkpoint from a single per-asset decode stream."""

    expected = set(int(value) for value in requested_video_frames)
    if not expected:
        return {}, set(), lookahead
    upper = max(expected)
    decoded: dict[int, tuple[Any, float]] = {}
    duplicates: set[int] = set()
    current = lookahead
    while current is not None and int(current[0]) <= upper:
        frame_idx, frame, decode_seconds = current
        index = int(frame_idx)
        if index in expected:
            if index in decoded:
                duplicates.add(index)
            else:
                decoded[index] = (frame, float(decode_seconds))
        try:
            current = next(iterator)
        except StopIteration:
            current = None
    return decoded, duplicates, current


def _build_run_fingerprint(
    *,
    manifest: Path,
    manifest_by_asset: Mapping[str, Mapping[str, Any]],
    selected_asset_ids: Sequence[str],
    model_identity: Mapping[str, Any],
    config_hash: str,
    frame_thresholds: Mapping[str, float],
    checkpoint_every_frames: int,
    queries: Sequence[str],
    save_positive_overlays: bool,
    max_overlays: int | None,
) -> dict[str, Any]:
    assets: dict[str, Any] = {}
    for asset_id in selected_asset_ids:
        row = manifest_by_asset[asset_id]
        assets[asset_id] = {
            "manifest_row": _canonical_value(dict(row)),
            "direct_2d": _path_identity(Path(row["parquet_path"])),
            "source_video": _path_identity(Path(row["primary_video_path"])),
        }
    fingerprint = {
        "producer": PRODUCER_VERSION,
        "manifest": _path_identity(manifest, content_hash=True),
        "selected_asset_ids": list(selected_asset_ids),
        "assets": assets,
        "model": dict(model_identity),
        "config_hash": config_hash,
        "frame_thresholds": dict(frame_thresholds),
        "sam3_runtime_config": dict(SAM3_CONFIG),
        "queries": list(queries),
        "required_hands": ["left", "right"],
        "source_video_mapping": "jd_identity_source_frame_equals_video_frame_v1",
        "checkpoint_every_frames": checkpoint_every_frames,
        "evidence": {
            "save_positive_overlays": save_positive_overlays,
            "max_overlays": max_overlays,
        },
    }
    return {
        **fingerprint,
        "fingerprint_sha256": canonical_sha256(fingerprint),
    }


def _model_config_identity(
    *,
    model_identity: Mapping[str, Any],
    config_hash: str,
    frame_thresholds: Mapping[str, float],
    queries: Sequence[str],
) -> str:
    return canonical_sha256(
        {
            "model": dict(model_identity),
            "config_hash": config_hash,
            "frame_thresholds": dict(frame_thresholds),
            "sam3_runtime_config": dict(SAM3_CONFIG),
            "queries": list(queries),
        }
    )


def run_manifest_sam3_exhaustive(
    *,
    manifest: Path,
    supplier: str,
    sam3_model: Path | None,
    output_dir: Path,
    source_reader: Any | None = None,
    segmenter: Any | None = None,
    checkpoint_every_frames: int = 1000,
    config_path: Path | None = None,
    resume: bool = False,
    queries: Sequence[str] | None = None,
    overlay_writer: Any = write_combined_overlay_image,
    save_positive_overlays: bool = True,
    max_overlays: int | None = None,
    asset_ids: Sequence[str] | None = None,
    max_assets: int | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
    decode_workers: int = 4,
    prefetch_frames: int = 32,
    writer_workers: int = 1,
    gpu_inference_workers: int = 1,
    asset_inference_concurrency: int = 1,
) -> dict[str, Any]:
    """Run the explicit-file JD exhaustive producer with chunked resume."""
    if supplier != "jdt":
        raise ValueError("exhaustive SAM3 currently supports supplier=jdt only")
    if checkpoint_every_frames < 1:
        raise ValueError("checkpoint_every_frames must be >= 1")
    for name, value in (
        ("decode_workers", decode_workers),
        ("prefetch_frames", prefetch_frames),
        ("writer_workers", writer_workers),
        ("gpu_inference_workers", gpu_inference_workers),
        ("asset_inference_concurrency", asset_inference_concurrency),
    ):
        if value < 1:
            raise ValueError(f"{name} must be >= 1")
    manifest = Path(manifest)
    output_dir = Path(output_dir)
    if num_shards > 1:
        output_dir = output_dir / f"shard-{shard_index:05d}-of-{num_shards:05d}"
    config = load_qc_acceptance_config(config_path)
    frame_thresholds, _ = configured_sam3_thresholds(config)
    query_list = list(queries) if queries is not None else [
        query.strip() for query in DEFAULT_QUERIES.split(",") if query.strip()
    ]
    manifest_rows = read_records(manifest)
    indexed, order, manifest_failures = _manifest_index(
        manifest_rows, manifest_dir=manifest.parent.resolve()
    )
    order = select_asset_ids(
        order,
        requested=asset_ids,
        max_assets=max_assets,
        num_shards=num_shards,
        shard_index=shard_index,
    )
    if source_reader is None:
        source_reader = SequentialVideoSource(
            decode_workers=decode_workers,
            prefetch_frames=prefetch_frames,
        )
    model_identity = _model_identity(sam3_model)
    fingerprint = _build_run_fingerprint(
        manifest=manifest,
        manifest_by_asset=indexed,
        selected_asset_ids=order,
        model_identity=model_identity,
        config_hash=config.sha256,
        frame_thresholds=frame_thresholds,
        checkpoint_every_frames=checkpoint_every_frames,
        queries=query_list,
        save_positive_overlays=save_positive_overlays,
        max_overlays=max_overlays,
    )
    model_config_identity = _model_config_identity(
        model_identity=model_identity,
        config_hash=config.sha256,
        frame_thresholds=frame_thresholds,
        queries=query_list,
    )
    run_config_path = output_dir / "run_config.json"
    if run_config_path.exists():
        if not resume:
            raise FileExistsError(
                f"output already contains run_config.json; pass --resume: {output_dir}"
            )
        try:
            previous = json.loads(run_config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StaleResumeError("resume run_config is unreadable") from exc
        if previous.get("fingerprint") != fingerprint:
            raise StaleResumeError(
                "resume fingerprint does not match current manifest/model/config/sources"
            )
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_config: dict[str, Any] = {
        "producer": PRODUCER_VERSION,
        "status": "running",
        "candidate_independent": True,
        "source_frame_stride": 1,
        "frame_range_semantics": "inclusive_source_frames",
        "source_video_mapping": "jd_identity_source_frame_equals_video_frame_v1",
        "required_hands": ["left", "right"],
        "sam3_runtime_config": dict(SAM3_CONFIG),
        "thresholds": frame_thresholds,
        "fingerprint": fingerprint,
        "model_config_identity": model_config_identity,
        "runtime": {
            "requested": {
                "decode_workers": decode_workers,
                "prefetch_frames": prefetch_frames,
                "writer_workers": writer_workers,
                "gpu_inference_workers": gpu_inference_workers,
                "asset_inference_concurrency": asset_inference_concurrency,
            },
            "effective": {
                "decode_workers": decode_workers,
                "prefetch_frames": prefetch_frames,
                "writer_workers": writer_workers,
                "gpu_inference_workers": 1,
                "asset_inference_concurrency": 1,
            },
            "single_model_instance": True,
            "cuda": _detected_cuda_runtime(),
            "recommended_environment": {
                "CUDA_VISIBLE_DEVICES": "0",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        },
    }
    _atomic_write_json(run_config_path, runtime_config)

    hand_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    failures = [dict(row) for row in manifest_failures]
    completed_assets = 0
    reused_frame_count = 0
    computed_frame_count = 0
    decode_seconds_total = 0.0
    inference_seconds_total = 0.0
    checkpoint_write_seconds_total = 0.0
    evidence_write_seconds_total = 0.0
    per_asset_timing: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for asset_id in order:
            asset_started = time.perf_counter()
            asset_decode_seconds = 0.0
            asset_inference_seconds = 0.0
            asset_write_seconds = 0.0
            asset_computed_frames = 0
            asset_reused_frames = 0
            row = indexed[asset_id]
            requests = enumerate_asset_frames(row)
            asset_root = output_dir / "asset_chunks" / _asset_component(asset_id)
            plans: list[dict[str, Any]] = []
            for chunk_requests in _chunk_slices(requests, checkpoint_every_frames):
                chunk_dir = _chunk_directory(asset_root, chunk_requests)
                chunk_fingerprint = {
                    "run_fingerprint_sha256": fingerprint["fingerprint_sha256"],
                    "asset_id": asset_id,
                    "source_start": int(chunk_requests[0]["source_frame"]),
                    "source_end": int(chunk_requests[-1]["source_frame"]),
                }
                reused = _load_chunk(chunk_dir, chunk_fingerprint) if resume else None
                if reused is not None and not _chunk_contract_valid(
                    reused, chunk_requests
                ):
                    reused = None
                plans.append(
                    {
                        "requests": chunk_requests,
                        "directory": chunk_dir,
                        "fingerprint": chunk_fingerprint,
                        "reused": reused,
                    }
                )

            missing_requests = [
                request
                for plan in plans
                if plan["reused"] is None
                for request in plan["requests"]
            ]
            source_data: pd.DataFrame | None = None
            video_metadata: Mapping[str, Any] | None = None
            decode_iterator: Any = iter(())
            decode_lookahead: tuple[int, Any, float] | None = None
            if missing_requests:
                source_data = source_reader.read_parquet(Path(row["parquet_path"]))
                video_metadata = source_reader.video_metadata(
                    Path(row["primary_video_path"])
                )
                video_frame_count = int(video_metadata["frame_count"])
                requested_video_frames = [
                    int(request["video_frame"])
                    for request in missing_requests
                    if 0 <= int(request["video_frame"]) < video_frame_count
                ]
                decode_iterator = iter(
                    source_reader.iter_frames(
                        Path(row["primary_video_path"]), requested_video_frames
                    )
                )
                try:
                    decode_lookahead = next(decode_iterator)
                except StopIteration:
                    decode_lookahead = None

            for plan in plans:
                chunk_requests = plan["requests"]
                chunk_dir = plan["directory"]
                chunk_fingerprint = plan["fingerprint"]
                reused = plan["reused"]
                if reused is not None:
                    (
                        chunk_hands,
                        chunk_frames,
                        chunk_failures,
                        chunk_evidence,
                    ) = reused
                    reused_frame_count += len(chunk_frames)
                    asset_reused_frames += len(chunk_frames)
                else:
                    assert source_data is not None
                    assert video_metadata is not None
                    if segmenter is None:
                        if sam3_model is None:
                            raise ValueError(
                                "sam3_model is required when segmenter is not injected"
                            )
                        segmenter = Sam3RuntimeProvider().get_segmenter(
                            sam3_model, SAM3_CONFIG
                        )
                    valid_video_frames = [
                        int(request["video_frame"])
                        for request in chunk_requests
                        if 0
                        <= int(request["video_frame"])
                        < int(video_metadata["frame_count"])
                    ]
                    (
                        decoded_frames,
                        duplicate_decoded_frames,
                        decode_lookahead,
                    ) = _decoded_chunk(
                        decode_iterator,
                        decode_lookahead,
                        requested_video_frames=valid_video_frames,
                    )
                    (
                        chunk_hands,
                        chunk_frames,
                        chunk_failures,
                        chunk_evidence,
                        chunk_timing,
                    ) = _compute_chunk(
                        requests=chunk_requests,
                        manifest_row=row,
                        source_data=source_data,
                        source_reader=source_reader,
                        segmenter=segmenter,
                        frame_thresholds=frame_thresholds,
                        queries=query_list,
                        overlay_writer=overlay_writer,
                        overlay_dir=(
                            output_dir
                            / "overlays"
                            / _asset_component(asset_id)
                        ),
                        save_positive_overlays=save_positive_overlays,
                        max_overlays=(
                            None
                            if max_overlays is None
                            else max(0, max_overlays - len(evidence_rows))
                        ),
                        model_config_identity=model_config_identity,
                        video_metadata=video_metadata,
                        decoded_frames=decoded_frames,
                        duplicate_decoded_frames=duplicate_decoded_frames,
                    )
                    if not _chunk_contract_valid(
                        (
                            chunk_hands,
                            chunk_frames,
                            chunk_failures,
                            chunk_evidence,
                        ),
                        chunk_requests,
                    ):
                        raise RuntimeError(
                            f"internal exhaustive output contract violation: {asset_id}"
                        )
                    decode_seconds_total += chunk_timing["decode_seconds"]
                    inference_seconds_total += chunk_timing["inference_seconds"]
                    evidence_write_seconds_total += chunk_timing[
                        "evidence_write_seconds"
                    ]
                    asset_decode_seconds += chunk_timing["decode_seconds"]
                    asset_inference_seconds += chunk_timing["inference_seconds"]
                    asset_write_seconds += chunk_timing["evidence_write_seconds"]
                    computed_frame_count += len(chunk_frames)
                    asset_computed_frames += len(chunk_frames)
                    checkpoint_write_started = time.perf_counter()
                    _write_chunk(
                        chunk_dir,
                        fingerprint=chunk_fingerprint,
                        hand_rows=chunk_hands,
                        frame_rows=chunk_frames,
                        failures=chunk_failures,
                        evidence_rows=chunk_evidence,
                    )
                    checkpoint_write_seconds = (
                        time.perf_counter() - checkpoint_write_started
                    )
                    checkpoint_write_seconds_total += checkpoint_write_seconds
                    asset_write_seconds += checkpoint_write_seconds
                for result_row in (*chunk_hands, *chunk_frames):
                    result_row.setdefault("producer_version", PRODUCER_VERSION)
                    result_row.setdefault(
                        "model_config_identity", model_config_identity
                    )
                    result_row.setdefault(
                        "source_mapping_identity",
                        "jd_identity_source_frame_equals_video_frame_v1",
                    )
                hand_rows.extend(chunk_hands)
                frame_rows.extend(chunk_frames)
                failures.extend(chunk_failures)
                evidence_rows.extend(chunk_evidence)
                _atomic_write_json(
                    output_dir / "progress.json",
                    {
                        "status": "running",
                        "completed_assets": completed_assets,
                        "current_asset_id": asset_id,
                        "computed_frame_count": computed_frame_count,
                        "reused_frame_count": reused_frame_count,
                        "last_completed_chunk": str(chunk_dir.relative_to(output_dir)),
                    },
                )
            _atomic_write_json(
                asset_root / "asset_completion.json",
                {
                    "asset_id": asset_id,
                    "fingerprint_sha256": fingerprint["fingerprint_sha256"],
                    "source_frame_count": len(requests),
                },
            )
            completed_assets += 1
            asset_elapsed = time.perf_counter() - asset_started
            per_asset_timing.append(
                {
                    "asset_id": asset_id,
                    "source_frame_count": len(requests),
                    "computed_frame_count": asset_computed_frames,
                    "reused_frame_count": asset_reused_frames,
                    "decode_seconds": asset_decode_seconds,
                    "inference_seconds": asset_inference_seconds,
                    "write_seconds": asset_write_seconds,
                    "elapsed_seconds": asset_elapsed,
                    "frames_per_second": (
                        len(requests) / asset_elapsed if asset_elapsed > 0 else None
                    ),
                }
            )
    finally:
        source_reader.close()

    hand_rows.sort(
        key=lambda row: (
            order.index(str(row["asset_id"])),
            int(row["source_frame"]),
            str(row["hand"]),
        )
    )
    frame_rows.sort(
        key=lambda row: (
            order.index(str(row["asset_id"])),
            int(row["source_frame"]),
        )
    )
    elapsed = time.perf_counter() - started
    summary = {
        "producer": PRODUCER_VERSION,
        "total_manifest_assets": len(manifest_rows),
        "selected_asset_ids": list(order),
        "num_shards": num_shards,
        "shard_index": shard_index,
        "output_dir": str(output_dir),
        "completed_assets": completed_assets,
        "failed_assets": len(manifest_failures),
        "total_source_frames": len(frame_rows),
        "computed_frame_count": computed_frame_count,
        "reused_frame_count": reused_frame_count,
        "elapsed_seconds": elapsed,
        "frames_per_second": len(frame_rows) / elapsed if elapsed > 0 else None,
        "per_asset_timing": per_asset_timing,
        "peak_gpu_memory_bytes": _peak_cuda_memory_bytes(),
    }
    final_write_seconds, result_bytes = _write_producer_outputs(
        output_dir,
        hand_rows=hand_rows,
        frame_rows=frame_rows,
        failures=failures,
        summary=summary,
        evidence_rows=evidence_rows,
        writer_workers=writer_workers,
    )
    overlay_bytes = sum(
        Path(str(row["overlay_path"])).stat().st_size
        for row in evidence_rows
        if row.get("overlay_path") and Path(str(row["overlay_path"])).is_file()
    )
    summary.update(
        {
            "overlay_bytes": overlay_bytes,
            "result_bytes": result_bytes,
        }
    )
    _atomic_write_json(output_dir / "progress.json", {"status": "completed", **summary})
    run_config = {
        **runtime_config,
        "status": "completed",
        "timing": {
            "decode_seconds": decode_seconds_total,
            "inference_seconds": inference_seconds_total,
            "checkpoint_write_seconds": checkpoint_write_seconds_total,
            "evidence_write_seconds": evidence_write_seconds_total,
            "final_materialization_write_seconds": final_write_seconds,
            "write_seconds": (
                checkpoint_write_seconds_total
                + evidence_write_seconds_total
                + final_write_seconds
            ),
            "total_seconds": elapsed + final_write_seconds,
            "frames_per_second": (
                len(frame_rows) / (elapsed + final_write_seconds)
                if elapsed + final_write_seconds > 0
                else None
            ),
        },
        "summary": summary,
    }
    _atomic_write_json(run_config_path, run_config)
    return summary


def main() -> int:
    args = build_parser().parse_args()
    summary = run_manifest_sam3_exhaustive(
        manifest=args.manifest,
        supplier=args.supplier,
        sam3_model=args.sam3_model,
        output_dir=args.output_dir,
        checkpoint_every_frames=args.checkpoint_every_frames,
        config_path=args.config,
        resume=args.resume,
        save_positive_overlays=args.save_positive_overlays,
        max_overlays=args.max_overlays,
        asset_ids=args.asset_ids,
        max_assets=args.max_assets,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        decode_workers=args.decode_workers,
        prefetch_frames=args.prefetch_frames,
        writer_workers=args.writer_workers,
        gpu_inference_workers=args.gpu_inference_workers,
        asset_inference_concurrency=args.asset_inference_concurrency,
    )
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
