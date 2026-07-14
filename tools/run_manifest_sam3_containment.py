#!/usr/bin/env python3
"""Run SAM3 containment for manifest-defined JD source-frame windows."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qc_common.config import LoadedQcConfig, load_qc_acceptance_config  # noqa: E402
from qc_common.keypoints import acceptance_joint_names  # noqa: E402
from qc_common.report import load_asset_qc_report  # noqa: E402
from qc_pipeline.adapters.sam3_containment import (  # noqa: E402
    write_sam3_asset_result,
)
from qc_pipeline.context import AssetContext  # noqa: E402
from tools.sam3_keypoint_containment import (  # noqa: E402
    aggregate_window_containment_summaries,
    candidate_window_metadata,
    create_sam3_segmenter,
    json_safe,
    sample_candidate_window_frames,
    score_keypoints_against_masks,
    write_combined_overlay_image,
    write_json,
    write_overlay_image,
)


LOGGER = logging.getLogger("run_manifest_sam3_containment")
DEFAULT_QUERIES = "hand,left hand,right hand,robot hand,gripper"
SAM3_CONFIG = {
    "confidence_threshold": 0.5,
    "mask_threshold": 0.5,
    "max_instances_per_query": 10,
}
FRAME_THRESHOLDS = {
    "abnormal_inside_ratio_threshold": "abnormal_inside_ratio_threshold",
    "projected_in_image_ratio_threshold": "projected_in_image_ratio_threshold",
    "strong_inside_ratio_threshold": "strong_inside_ratio_threshold",
    "acceptable_inside_ratio_threshold": "acceptable_inside_ratio_threshold",
    "mask_tiny_area_ratio_threshold": "mask_tiny_area_ratio_threshold",
}
WINDOW_THRESHOLDS = {
    "fail_min_strong_frames": "fail_min_strong_frames",
    "fail_strong_frame_ratio": "fail_strong_frame_ratio",
}
OUTPUT_FILENAMES = (
    "frame_keypoint_containment.json",
    "frame_keypoint_containment.parquet",
    "window_keypoint_containment_summary.json",
    "window_keypoint_containment_summary.parquet",
    "failures.json",
    "run_config.json",
    "review_evidence_manifest.csv",
    "review_evidence_manifest.parquet",
    "qc_report_prerequisites.json",
)
EVIDENCE_MANIFEST_COLUMNS = (
    "review_id",
    "supplier_id",
    "asset_id",
    "window_start_frame",
    "window_end_frame",
    "frame_idx",
    "source_module",
    "evidence_type",
    "hand_side",
    "source_path",
    "metadata_json",
)
OVERLAY_MODES = ("combined", "per-hand", "both", "none")


def configured_sam3_thresholds(
    config: LoadedQcConfig,
) -> tuple[dict[str, float], dict[str, float | int]]:
    """Resolve every legacy algorithm threshold from the unified Config."""
    parameters = config.module_parameters("sam3_containment")

    def numeric(name: str, *, integer: bool = False) -> float | int:
        value = parameters.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"sam3_containment parameter {name} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"sam3_containment parameter {name} must be finite")
        if integer:
            if int(value) != value:
                raise ValueError(
                    f"sam3_containment parameter {name} must be an integer"
                )
            return int(value)
        return float(value)

    frame = {
        argument: float(numeric(parameter))
        for argument, parameter in FRAME_THRESHOLDS.items()
    }
    window: dict[str, float | int] = {
        argument: numeric(
            parameter,
            integer=argument == "fail_min_strong_frames",
        )
        for argument, parameter in WINDOW_THRESHOLDS.items()
    }
    return frame, window


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run manifest-aware SAM3 keypoint containment on JD cam_left "
            "source-frame candidate windows."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-windows", type=Path, required=True)
    parser.add_argument("--supplier", choices=("jdt",), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--profile", default="acceptance")
    parser.add_argument("--frames-per-window", type=int, default=3)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--sam3-model", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--queries", default=DEFAULT_QUERIES)
    parser.add_argument(
        "--overlay-mode",
        choices=OVERLAY_MODES,
        default=None,
        help="Overlay output policy (default: combined).",
    )
    parser.add_argument(
        "--write-overlays",
        action="store_const",
        const=True,
        default=None,
        help="Backward-compatible alias for --overlay-mode both.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return not math.isfinite(float(value))
    return False


def _clean_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): None if _missing_scalar(value) else json_safe(value)
        for key, value in row.items()
    }


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read a CSV/Parquet/JSON/JSONL file as plain record dictionaries."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        rows = pd.read_csv(path).to_dict(orient="records")
    elif suffix == ".parquet":
        rows = pd.read_parquet(path).to_dict(orient="records")
    elif suffix in {".jsonl", ".ndjson"}:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            rows = next(
                (value for value in payload.values() if isinstance(value, list)),
                None,
            )
            if rows is None:
                raise ValueError(f"JSON does not contain a record list: {path}")
        else:
            raise ValueError(f"JSON records must be a list: {path}")
    else:
        raise ValueError(f"unsupported record file extension: {path.suffix}")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"records must be objects: {path}")
    return [_clean_record(dict(row)) for row in rows]


def _asset_id(value: Any) -> str:
    if _missing_scalar(value):
        raise ValueError("missing asset_id")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    asset_id = str(value).strip()
    if not asset_id:
        raise ValueError("missing asset_id")
    return asset_id


def _integer(value: Any, name: str) -> int:
    if _missing_scalar(value):
        raise ValueError(f"missing {name}")
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(numeric)


def _input_path(value: Any, *, name: str, base_dir: Path) -> Path:
    if _missing_scalar(value) or not str(value).strip():
        raise ValueError(f"missing {name}")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def reshape_jdt_keypoints(
    value: Any,
    field_name: str,
    source_frame_idx: int,
) -> np.ndarray:
    """Decode one JD flat 42-value hand field into direct image pixels."""
    if value is None:
        return np.full((21, 2), np.nan, dtype=np.float64)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return np.full((21, 2), np.nan, dtype=np.float64)
        value = json.loads(text)
    points = np.asarray(value, dtype=np.float64).reshape(-1)
    if points.size != 42:
        raise ValueError(
            f"{field_name} source frame {source_frame_idx} has "
            f"{points.size} values; expected 42"
        )
    return points.reshape(21, 2)


class ManifestSourceCache:
    """Cache source parquet frames, open videos, and decoded source frames."""

    def __init__(
        self,
        *,
        parquet_reader: Callable[[Path], pd.DataFrame] | None = None,
        video_capture_factory: Callable[[Path], Any] | None = None,
    ) -> None:
        self._parquet_reader = parquet_reader or pd.read_parquet
        self._video_capture_factory = (
            video_capture_factory or self._open_video_capture
        )
        self._parquet_cache: dict[Path, pd.DataFrame] = {}
        self._video_cache: dict[Path, Any] = {}
        self._frame_cache: dict[tuple[Path, int], np.ndarray] = {}

    @staticmethod
    def _open_video_capture(path: Path) -> Any:
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"could not open source video: {path}")
        return capture

    def read_parquet(self, path: Path) -> pd.DataFrame:
        path = Path(path).resolve()
        if path not in self._parquet_cache:
            self._parquet_cache[path] = self._parquet_reader(path)
        return self._parquet_cache[path]

    def read_frame(self, path: Path, frame_idx: int) -> np.ndarray:
        path = Path(path).resolve()
        key = (path, int(frame_idx))
        if key in self._frame_cache:
            return self._frame_cache[key]
        if path not in self._video_cache:
            self._video_cache[path] = self._video_capture_factory(path)
        capture = self._video_cache[path]
        capture.set(1, int(frame_idx))
        ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            raise ValueError(f"could not decode source frame {frame_idx}: {path}")
        frame_rgb = np.asarray(frame_bgr)[..., ::-1].copy()
        self._frame_cache[key] = frame_rgb
        return frame_rgb

    def close(self) -> None:
        for capture in self._video_cache.values():
            capture.release()
        self._video_cache.clear()


def _hand_sides(value: Any) -> list[str]:
    side = str(value or "both").strip().lower()
    if side == "both":
        return ["left", "right"]
    if side in {"left", "right"}:
        return [side]
    raise ValueError(f"invalid candidate hand_side: {value}")


def sample_manifest_window_frames(
    window: dict[str, Any],
    *,
    clip_start_frame: int,
    clip_end_frame: int,
    frames_per_window: int,
) -> list[int]:
    """Validate source coordinates, then use the established sampling rule."""
    start_frame = _integer(window.get("start_frame"), "candidate start_frame")
    end_frame = _integer(window.get("end_frame"), "candidate end_frame")
    coordinate_space = str(window.get("coordinate_space") or "source").lower()
    if coordinate_space != "source":
        raise ValueError(
            f"candidate coordinate_space must be source; got {coordinate_space}"
        )
    if not (
        clip_start_frame
        <= start_frame
        <= end_frame
        <= clip_end_frame
    ):
        raise ValueError(
            f"candidate source bounds {start_frame}..{end_frame} outside "
            f"clip {clip_start_frame}..{clip_end_frame}"
        )
    return sample_candidate_window_frames(
        {"start_frame": start_frame, "end_frame": end_frame},
        num_frames=clip_end_frame + 1,
        frames_per_window=frames_per_window,
        include_boundaries=True,
    )


def _manifest_index(
    rows: list[dict[str, Any]],
    *,
    manifest_dir: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    list[str],
    list[dict[str, Any]],
]:
    indexed: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    failures: list[dict[str, Any]] = []
    required = (
        "episode_index",
        "start_frame",
        "end_frame",
        "primary_video_path",
        "parquet_path",
        "left_hand_2d_field",
        "right_hand_2d_field",
    )
    for row_index, row in enumerate(rows):
        asset_id: str | None = None
        try:
            asset_id = _asset_id(row.get("asset_id"))
            if asset_id in indexed:
                raise ValueError(f"duplicate manifest asset_id: {asset_id}")
            missing = [
                name for name in required if _missing_scalar(row.get(name))
            ]
            if missing:
                raise ValueError(
                    f"manifest row {row_index} ({asset_id}) missing: "
                    + ", ".join(missing)
                )
            normalized = dict(row)
            normalized.update(
                {
                    "asset_id": asset_id,
                    "episode_index": _integer(
                        row.get("episode_index"),
                        "episode_index",
                    ),
                    "start_frame": _integer(
                        row.get("start_frame"),
                        "start_frame",
                    ),
                    "end_frame": _integer(
                        row.get("end_frame"),
                        "end_frame",
                    ),
                    "primary_video_path": _input_path(
                        row.get("primary_video_path"),
                        name="primary_video_path",
                        base_dir=manifest_dir,
                    ),
                    "parquet_path": _input_path(
                        row.get("parquet_path"),
                        name="parquet_path",
                        base_dir=manifest_dir,
                    ),
                }
            )
            if normalized["end_frame"] < normalized["start_frame"]:
                raise ValueError(
                    f"manifest asset {asset_id} has reversed frame range"
                )
            indexed[asset_id] = normalized
            order.append(asset_id)
        except (TypeError, ValueError) as exc:
            failures.append(
                {
                    "failure_stage": "manifest_mapping",
                    "manifest_row_index": row_index,
                    "asset_id": asset_id,
                    "error": str(exc),
                }
            )
    return indexed, order, failures


def _failure(
    *,
    window_index: int,
    window: dict[str, Any],
    error: Exception | str,
) -> dict[str, Any]:
    return {
        "window_index": window_index,
        "asset_id": None
        if _missing_scalar(window.get("asset_id"))
        else str(window.get("asset_id")),
        "candidate_start_frame": window.get("start_frame"),
        "candidate_end_frame": window.get("end_frame"),
        "hand_side": window.get("hand_side"),
        "error": str(error),
    }


def _prepare_windows(
    windows: list[dict[str, Any]],
    manifest_by_asset: dict[str, dict[str, Any]],
    manifest_order: list[str],
    *,
    frames_per_window: int,
    max_windows: int | None,
    max_clips: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    allowed_assets = set(
        manifest_order if max_clips is None else manifest_order[:max_clips]
    )
    prepared: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for window_index, raw_window in enumerate(windows):
        try:
            asset_id = _asset_id(raw_window.get("asset_id"))
            if asset_id not in manifest_by_asset:
                raise ValueError(f"candidate asset_id not found in manifest: {asset_id}")
            if asset_id not in allowed_assets:
                continue
            manifest_row = manifest_by_asset[asset_id]
            sampled_frames = sample_manifest_window_frames(
                raw_window,
                clip_start_frame=manifest_row["start_frame"],
                clip_end_frame=manifest_row["end_frame"],
                frames_per_window=frames_per_window,
            )
            prepared.append(
                {
                    "window_index": window_index,
                    "asset_id": asset_id,
                    "window": {
                        **raw_window,
                        "asset_id": asset_id,
                        "start_frame": int(raw_window["start_frame"]),
                        "end_frame": int(raw_window["end_frame"]),
                        "coordinate_space": "source",
                    },
                    "manifest": manifest_row,
                    "sampled_frames": sampled_frames,
                    "hand_sides": _hand_sides(raw_window.get("hand_side")),
                }
            )
            if max_windows is not None and len(prepared) >= max_windows:
                break
        except (TypeError, ValueError) as exc:
            failures.append(
                _failure(
                    window_index=window_index,
                    window=raw_window,
                    error=exc,
                )
            )
    return prepared, failures


def _joint_names(side: str) -> list[str]:
    names = [name for name in acceptance_joint_names() if name.startswith(side)]
    if len(names) != 21:
        raise ValueError(f"acceptance topology has {len(names)} {side} joints")
    return names


def _write_record_outputs(
    rows: list[dict[str, Any]],
    *,
    json_path: Path,
    parquet_path: Path,
) -> None:
    write_json(rows, json_path)
    pd.DataFrame([json_safe(row) for row in rows]).to_parquet(
        parquet_path,
        index=False,
    )


def _write_evidence_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    frame = pd.DataFrame(
        [json_safe(row) for row in rows],
        columns=EVIDENCE_MANIFEST_COLUMNS,
    )
    frame.to_csv(output_dir / "review_evidence_manifest.csv", index=False)
    frame.to_parquet(output_dir / "review_evidence_manifest.parquet", index=False)


def _sam3_write_is_ready(report: dict[str, Any] | None) -> bool:
    if report is None:
        return False
    pipeline_state = report.get("pipeline_state")
    if not isinstance(pipeline_state, dict):
        return False
    if pipeline_state.get("next_module") == "sam3_containment":
        return True
    if pipeline_state.get("last_completed_module") != "sam3_containment":
        return False
    module = report.get("sam3_containment")
    flow = module.get("flow") if isinstance(module, dict) else None
    exit_gate = flow.get("exit_gate") if isinstance(flow, dict) else None
    return isinstance(exit_gate, dict)


def _sam3_prerequisite(
    *,
    asset_id: str,
    report: dict[str, Any] | None,
    report_path: Path,
    batch_root: Path,
) -> dict[str, Any]:
    pipeline_state = report.get("pipeline_state") if report is not None else None
    current_next = (
        pipeline_state.get("next_module")
        if isinstance(pipeline_state, dict)
        else None
    )
    return {
        "asset_id": asset_id,
        "condition": "awaiting_pipeline",
        "current_next_module": current_next,
        "required_module": "sam3_containment",
        "report_path": report_path.relative_to(batch_root).as_posix(),
    }


def resolve_overlay_mode(
    overlay_mode: str | None,
    write_overlays: bool | None,
) -> str:
    if overlay_mode is not None:
        if overlay_mode not in OVERLAY_MODES:
            raise ValueError(
                f"invalid overlay_mode {overlay_mode!r}; expected one of {OVERLAY_MODES}"
            )
        return overlay_mode
    if write_overlays is True:
        return "both"
    if write_overlays is False:
        return "none"
    return "combined"


def _augment_window_summaries(
    summaries: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence = {
        (
            row.get("asset_id"),
            row.get("window_start_frame"),
            row.get("window_end_frame"),
            row.get("hand_side"),
        ): row
        for row in frame_rows
    }
    for summary in summaries:
        row = evidence.get(
            (
                summary.get("asset_id"),
                summary.get("window_start_frame"),
                summary.get("window_end_frame"),
                summary.get("hand_side"),
            )
        )
        if row is not None:
            for key in (
                "clip_start_frame",
                "clip_end_frame",
                "candidate_start_frame",
                "candidate_end_frame",
                "video_path",
                "parquet_path",
                "coordinate_space",
            ):
                summary[key] = row.get(key)
    return summaries


def run_manifest_sam3_containment(
    *,
    manifest: Path,
    candidate_windows: Path,
    supplier: str,
    output_dir: Path,
    frames_per_window: int = 3,
    max_windows: int | None = None,
    max_clips: int | None = None,
    sam3_model: Path | None = None,
    queries: list[str] | None = None,
    overlay_mode: str | None = None,
    write_overlays: bool | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    source_cache: Any | None = None,
    segmenter: Any | None = None,
    config_path: Path | None = None,
    batch_root: Path | None = None,
    profile: str = "acceptance",
) -> dict[str, Any]:
    if supplier != "jdt":
        raise ValueError(f"unsupported supplier: {supplier}")
    if frames_per_window < 1:
        raise ValueError("--frames-per-window must be >= 1")
    for name, value in (("max_windows", max_windows), ("max_clips", max_clips)):
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    effective_overlay_mode = resolve_overlay_mode(overlay_mode, write_overlays)
    write_per_hand_overlays = effective_overlay_mode in {"per-hand", "both"}
    write_combined_overlays = effective_overlay_mode in {"combined", "both"}

    manifest = Path(manifest)
    candidate_windows = Path(candidate_windows)
    output_dir = Path(output_dir)
    batch_root = Path(batch_root) if batch_root is not None else output_dir.parent
    loaded_config = load_qc_acceptance_config(config_path)
    loaded_config.execution_profile(profile)
    frame_thresholds, window_thresholds = configured_sam3_thresholds(loaded_config)
    modules = loaded_config.pipeline_modules
    sam3_index = modules.index("sam3_containment")
    next_module = modules[sam3_index + 1] if sam3_index + 1 < len(modules) else None
    manifest_rows = read_records(manifest)
    windows = read_records(candidate_windows)
    manifest_by_asset, manifest_order, manifest_failures = _manifest_index(
        manifest_rows,
        manifest_dir=manifest.parent.resolve(),
    )
    prepared, window_failures = _prepare_windows(
        windows,
        manifest_by_asset,
        manifest_order,
        frames_per_window=frames_per_window,
        max_windows=max_windows,
        max_clips=max_clips,
    )
    failures = [*manifest_failures, *window_failures]
    sampled_source_frames = {
        (item["manifest"]["primary_video_path"], frame_idx)
        for item in prepared
        for frame_idx in item["sampled_frames"]
    }
    hand_frame_evaluations = sum(
        len(item["sampled_frames"]) * len(item["hand_sides"])
        for item in prepared
    )
    summary: dict[str, Any] = {
        "manifest_row_count": len(manifest_rows),
        "candidate_window_count": len(windows),
        "selected_window_count": len(prepared),
        "sampled_source_frame_count": len(sampled_source_frames),
        "hand_frame_evaluation_count": hand_frame_evaluations,
        "completed_window_count": 0,
        "failed_window_count": len(window_failures),
        "failed_manifest_row_count": len(manifest_failures),
        "failed_asset_count": 0,
        "qc_report_write_count": 0,
        "awaiting_pipeline_asset_count": 0,
        "dry_run": dry_run,
    }
    if dry_run:
        return summary
    if segmenter is None and sam3_model is None:
        raise ValueError("--sam3-model is required unless --dry-run is used")
    existing = [output_dir / name for name in OUTPUT_FILENAMES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "output files already exist; pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    query_list = queries or [
        query.strip() for query in DEFAULT_QUERIES.split(",") if query.strip()
    ]
    if segmenter is None:
        assert sam3_model is not None
        segmenter = create_sam3_segmenter(sam3_model, dict(SAM3_CONFIG))
    owned_cache = source_cache is None
    cache = source_cache or ManifestSourceCache()
    mask_cache: dict[tuple[Path, int], tuple[np.ndarray, list[Any]]] = {}
    frame_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    overlay_dir = output_dir / "overlays"
    try:
        for item in prepared:
            manifest_row = item["manifest"]
            window = item["window"]
            window_rows: list[dict[str, Any]] = []
            window_evidence_rows: list[dict[str, Any]] = []
            try:
                video_path = manifest_row["primary_video_path"]
                parquet_path = manifest_row["parquet_path"]
                if not video_path.exists():
                    raise FileNotFoundError(f"source video not found: {video_path}")
                if not parquet_path.exists():
                    raise FileNotFoundError(f"source parquet not found: {parquet_path}")
                source_data = cache.read_parquet(parquet_path)
                for source_frame_idx in item["sampled_frames"]:
                    if not 0 <= source_frame_idx < len(source_data):
                        raise ValueError(
                            f"source frame {source_frame_idx} outside parquet "
                            f"row count {len(source_data)}"
                        )
                    mask_key = (video_path, source_frame_idx)
                    if mask_key not in mask_cache:
                        frame = cache.read_frame(video_path, source_frame_idx)
                        masks = segmenter.segment_frame(
                            frame,
                            query_list,
                            dict(SAM3_CONFIG),
                        )
                        mask_cache[mask_key] = (frame, masks)
                    frame, masks = mask_cache[mask_key]
                    source_row = source_data.iloc[source_frame_idx]
                    combined_hands: dict[str, dict[str, Any]] = {}
                    for hand_side in item["hand_sides"]:
                        field_name = str(
                            manifest_row[f"{hand_side}_hand_2d_field"]
                        )
                        if field_name not in source_data.columns:
                            raise ValueError(
                                f"parquet missing JD 2D field: {field_name}"
                            )
                        pixels = reshape_jdt_keypoints(
                            source_row[field_name],
                            field_name,
                            source_frame_idx,
                        )
                        joint_names = _joint_names(hand_side)
                        containment, union_mask, valid, inside = (
                            score_keypoints_against_masks(
                                frame=frame,
                                pixels=pixels,
                                joint_names=joint_names,
                                masks=masks,
                                **frame_thresholds,
                            )
                        )
                        combined_hands[hand_side] = {
                            "pixels": pixels,
                            "valid": valid,
                            "inside": inside,
                            "joint_names": joint_names,
                        }
                        row = {
                            "clip_id": item["asset_id"],
                            "asset_id": item["asset_id"],
                            "episode_idx": manifest_row["episode_index"],
                            "video_path": str(video_path),
                            "parquet_path": str(parquet_path),
                            "frame_idx": source_frame_idx,
                            "source_frame_idx": source_frame_idx,
                            "clip_start_frame": manifest_row["start_frame"],
                            "clip_end_frame": manifest_row["end_frame"],
                            "candidate_start_frame": window["start_frame"],
                            "candidate_end_frame": window["end_frame"],
                            "coordinate_space": "source",
                            "projection_mode": "direct_2d",
                            "projection_mode_used": "direct_2d",
                            "image_width": int(frame.shape[1]),
                            "image_height": int(frame.shape[0]),
                            "hand_side": hand_side,
                            "candidate_hand_side": window.get("hand_side", "both"),
                            **containment,
                            **candidate_window_metadata(window),
                        }
                        if write_per_hand_overlays:
                            overlay_path = write_overlay_image(
                                frame=frame,
                                mask=union_mask,
                                pixels=pixels,
                                valid=valid,
                                inside=inside,
                                joint_names=joint_names,
                                clip_id=(
                                    f"{item['asset_id']}_window_"
                                    f"{window['start_frame']}_{window['end_frame']}_"
                                    f"{hand_side}"
                                ),
                                frame_idx=source_frame_idx,
                                output_dir=overlay_dir,
                            )
                            row["overlay_path"] = str(overlay_path)
                        window_rows.append(row)
                    if write_combined_overlays:
                        try:
                            for hand_side in ("left", "right"):
                                if hand_side in combined_hands:
                                    continue
                                field_name = str(
                                    manifest_row[f"{hand_side}_hand_2d_field"]
                                )
                                if field_name not in source_data.columns:
                                    raise ValueError(
                                        "parquet missing JD 2D field for combined "
                                        f"overlay: {field_name}"
                                    )
                                pixels = reshape_jdt_keypoints(
                                    source_row[field_name],
                                    field_name,
                                    source_frame_idx,
                                )
                                joint_names = _joint_names(hand_side)
                                _, _, valid, inside = score_keypoints_against_masks(
                                    frame=frame,
                                    pixels=pixels,
                                    joint_names=joint_names,
                                    masks=masks,
                                    **frame_thresholds,
                                )
                                combined_hands[hand_side] = {
                                    "pixels": pixels,
                                    "valid": valid,
                                    "inside": inside,
                                    "joint_names": joint_names,
                                }
                            combined_overlay_path = write_combined_overlay_image(
                                frame=frame,
                                hands=combined_hands,
                                clip_id=(
                                    f"{item['asset_id']}_window_"
                                    f"{window['start_frame']}_{window['end_frame']}_"
                                    "combined"
                                ),
                                frame_idx=source_frame_idx,
                                output_dir=output_dir / "combined_overlays",
                            )
                            window_evidence_rows.append(
                                {
                                    "review_id": "",
                                    "supplier_id": str(
                                        manifest_row.get("supplier_id") or supplier
                                    ),
                                    "asset_id": item["asset_id"],
                                    "window_start_frame": window["start_frame"],
                                    "window_end_frame": window["end_frame"],
                                    "frame_idx": source_frame_idx,
                                    "source_module": "sam3_containment",
                                    "evidence_type": "combined_overlay",
                                    "hand_side": "both",
                                    "source_path": str(
                                        Path(combined_overlay_path).resolve()
                                    ),
                                    "metadata_json": json.dumps(
                                        json_safe(
                                            {
                                                "candidate_hand_side": window.get(
                                                    "hand_side", "both"
                                                ),
                                                "clip_start_frame": manifest_row[
                                                    "start_frame"
                                                ],
                                                "clip_end_frame": manifest_row[
                                                    "end_frame"
                                                ],
                                                "source_frame_idx": source_frame_idx,
                                                "coordinate_space": "source",
                                            }
                                        ),
                                        sort_keys=True,
                                    ),
                                }
                            )
                        except Exception as exc:
                            raise RuntimeError(
                                "combined overlay failed for "
                                f"{item['asset_id']} frame {source_frame_idx}: {exc}"
                            ) from exc
                frame_rows.extend(window_rows)
                evidence_rows.extend(window_evidence_rows)
                summary["completed_window_count"] += 1
            except Exception as exc:
                LOGGER.exception(
                    "Window %d (%s) failed",
                    item["window_index"],
                    item["asset_id"],
                )
                failures.append(
                    _failure(
                        window_index=item["window_index"],
                        window=window,
                        error=exc,
                    )
                )
                summary["failed_window_count"] += 1
    finally:
        if owned_cache:
            cache.close()

    window_summaries = aggregate_window_containment_summaries(
        frame_rows,
        **window_thresholds,
    )
    _augment_window_summaries(window_summaries, frame_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_record_outputs(
        frame_rows,
        json_path=output_dir / "frame_keypoint_containment.json",
        parquet_path=output_dir / "frame_keypoint_containment.parquet",
    )
    _write_record_outputs(
        window_summaries,
        json_path=output_dir / "window_keypoint_containment_summary.json",
        parquet_path=output_dir / "window_keypoint_containment_summary.parquet",
    )
    _write_evidence_outputs(evidence_rows, output_dir)
    failed_assets = {
        str(row["asset_id"])
        for row in failures
        if row.get("asset_id") is not None
    }
    selected_assets = list(
        dict.fromkeys(str(item["asset_id"]) for item in prepared)
    )
    prerequisites: list[dict[str, Any]] = []
    for asset_id in selected_assets:
        if asset_id in failed_assets:
            continue
        report_path = batch_root / "quality_archive" / f"{asset_id}.json"
        report = load_asset_qc_report(report_path)
        if not _sam3_write_is_ready(report):
            prerequisites.append(
                _sam3_prerequisite(
                    asset_id=asset_id,
                    report=report,
                    report_path=report_path,
                    batch_root=batch_root,
                )
            )
            continue
        assert report is not None
        source_files = report.get("source_files")
        if not isinstance(source_files, dict):
            failures.append(
                {
                    "failure_stage": "qc_report_write",
                    "asset_id": asset_id,
                    "error": "asset QC report source_files must be an object",
                }
            )
            failed_assets.add(asset_id)
            continue
        try:
            context = AssetContext(
                asset_id=asset_id,
                batch_root=batch_root,
                report_path=report_path,
                source_files=copy.deepcopy(source_files),
            )
            write_sam3_asset_result(
                context=context,
                window_summaries=[
                    row for row in window_summaries if row.get("asset_id") == asset_id
                ],
                evidence_rows=[
                    row for row in evidence_rows if row.get("asset_id") == asset_id
                ],
                config=loaded_config,
                profile=profile,
                expected_revision=int(report.get("report_revision", 0)),
                next_module=next_module,
            )
            summary["qc_report_write_count"] += 1
        except Exception as exc:
            LOGGER.exception("SAM3 QC report write failed for %s", asset_id)
            failures.append(
                {
                    "failure_stage": "qc_report_write",
                    "asset_id": asset_id,
                    "error": str(exc),
                }
            )
            failed_assets.add(asset_id)
    summary["failed_asset_count"] = len(failed_assets)
    summary["awaiting_pipeline_asset_count"] = len(prerequisites)
    write_json(prerequisites, output_dir / "qc_report_prerequisites.json")
    write_json(failures, output_dir / "failures.json")
    run_config = {
        "manifest": str(manifest),
        "candidate_windows": str(candidate_windows),
        "supplier": supplier,
        "output_dir": str(output_dir),
        "batch_root": str(batch_root),
        "profile": profile,
        "qc_config": loaded_config.json_reference(),
        "frames_per_window": frames_per_window,
        "include_window_boundaries": True,
        "max_windows": max_windows,
        "max_clips": max_clips,
        "sam3_model": str(sam3_model) if sam3_model is not None else None,
        "queries": query_list,
        "overlay_mode": effective_overlay_mode,
        "write_overlays": effective_overlay_mode != "none",
        "write_per_hand_overlays": write_per_hand_overlays,
        "write_combined_overlays": write_combined_overlays,
        "frame_thresholds": frame_thresholds,
        "window_thresholds": window_thresholds,
        "abnormal_inside_ratio_threshold": frame_thresholds[
            "abnormal_inside_ratio_threshold"
        ],
        "projected_in_image_ratio_threshold": frame_thresholds[
            "projected_in_image_ratio_threshold"
        ],
        "strong_containment_inside_ratio_threshold": frame_thresholds[
            "strong_inside_ratio_threshold"
        ],
        "acceptable_inside_ratio_threshold": frame_thresholds[
            "acceptable_inside_ratio_threshold"
        ],
        "mask_tiny_area_ratio_threshold": frame_thresholds[
            "mask_tiny_area_ratio_threshold"
        ],
        "containment_fail_min_strong_frames": window_thresholds[
            "fail_min_strong_frames"
        ],
        "containment_fail_strong_frame_ratio": window_thresholds[
            "fail_strong_frame_ratio"
        ],
        "primary_camera": "observation.images.cam_left",
        "keypoint_source": "JD parquet direct 2D fields",
        **summary,
    }
    write_json(run_config, output_dir / "run_config.json")
    return summary


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    summary = run_manifest_sam3_containment(
        manifest=args.manifest,
        candidate_windows=args.candidate_windows,
        supplier=args.supplier,
        output_dir=args.output_dir,
        batch_root=args.batch_root,
        profile=args.profile,
        frames_per_window=args.frames_per_window,
        max_windows=args.max_windows,
        max_clips=args.max_clips,
        sam3_model=args.sam3_model,
        queries=[query.strip() for query in args.queries.split(",") if query.strip()],
        overlay_mode=args.overlay_mode,
        write_overlays=args.write_overlays,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        config_path=args.config,
    )
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
