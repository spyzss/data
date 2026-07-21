#!/usr/bin/env python3
"""Run registered prechecks on manifest-defined supplier frame ranges."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precheck.config import PrecheckConfig, load_precheck_config  # noqa: E402
from precheck.runner import PrecheckRunner  # noqa: E402
from precheck.adapters.supplier_hdf5 import (  # noqa: E402
    MANO_JOINT_INDEX_TO_ACCEPTANCE_BASE,
)
from qc_common.config import load_qc_acceptance_config  # noqa: E402
from qc_common.io import aggregate_results  # noqa: E402
from qc_common.types import CheckResult, ClipInputs  # noqa: E402
from qc_pipeline.adapters.precheck import precheck_config_from_unified  # noqa: E402


LOGGER = logging.getLogger(__name__)
DEFAULT_CHECKS = ("text_integrity", "keypoint_temporal", "skeleton_quality_score")
JDT_TEXT_FIELDS = (
    "language_instruction",
    "first_scene_cn",
    "first_scene_en",
    "second_scene_cn",
    "second_scene_en",
    "third_scene_cn",
    "third_scene_en",
)


def read_manifest(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix in {".jsonl", ".ndjson"}:
        return [
            dict(row)
            for row in (
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if isinstance(row, dict)
        ]
    else:
        raise ValueError(f"unsupported manifest extension: {path.suffix}")
    frame = frame.astype(object).where(pd.notna(frame), None)
    return [dict(row) for row in frame.to_dict(orient="records")]


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if math.isnan(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _integer(value: Any, name: str) -> int:
    if value is None or value == "":
        raise ValueError(f"missing {name}")
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(numeric)


def _optional_positive_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) and numeric > 0.0 else None


def _range(row: dict[str, Any], start_column: str, end_column: str) -> tuple[int, int]:
    start = _integer(row.get(start_column), start_column)
    end = _integer(row.get(end_column), end_column)
    if start < 0:
        raise ValueError("start_frame must be >= 0")
    if end < start:
        raise ValueError("end_frame must be >= start_frame")
    return start, end


def _asset_id(row: dict[str, Any]) -> str:
    value = _text(row.get("asset_id"))
    if not value:
        raise ValueError("missing asset_id")
    return value


def _canonical_keypoints(
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, np.ndarray]:
    keypoints: dict[str, np.ndarray] = {}
    for side, joints in (("left", left), ("right", right)):
        for index, base_name in MANO_JOINT_INDEX_TO_ACCEPTANCE_BASE.items():
            keypoints[f"{side}{base_name}"] = np.asarray(
                joints[:, index, :3],
                dtype=np.float32,
            )
    return keypoints


def _attach_common_metadata(
    clip: ClipInputs,
    *,
    asset_id: str,
    supplier: str,
    source_path: Path,
    start_frame: int,
    end_frame: int,
) -> ClipInputs:
    setattr(clip, "asset_id", asset_id)
    setattr(clip, "supplier_id", supplier)
    setattr(clip, "source_path", str(source_path))
    setattr(clip, "clip_start_frame", start_frame)
    setattr(clip, "clip_end_frame", end_frame)
    setattr(clip, "supplier_quality_signal", "not_provided")
    setattr(clip, "morphology_status", "not_ready_topology")
    return clip


def load_deepreach_clip(
    row: dict[str, Any],
    *,
    episode_idx: int,
    start_frame_column: str = "start_frame",
    end_frame_column: str = "end_frame",
    hdf5_column: str = "hdf5_path",
) -> ClipInputs:
    asset_id = _asset_id(row)
    start_frame, end_frame = _range(
        row,
        start_frame_column,
        end_frame_column,
    )
    path_text = _text(row.get(hdf5_column))
    if not path_text:
        raise ValueError(f"missing {hdf5_column}")
    path = Path(path_text).expanduser()
    from acceptance_pull.supplier_adapters.deepreach_hdf5 import (
        DeepReachFrameContractError,
        require_deepreach_frame_contract,
    )

    reference_dataset = _text(row.get("hdf5_reference_dataset"))
    if not reference_dataset:
        raise DeepReachFrameContractError(
            {
                "status": "mapping_invalid",
                "reason": "hdf5_reference_dataset_missing",
                "source_path": str(path),
                "reference_dataset": None,
                "expected_frame_count": None,
                "dataset_lengths": {},
                "missing_datasets": [],
                "invalid_shape_datasets": [],
                "mismatch_ranges": {},
                "mismatch_count": 0,
            }
        )
    frame_contract = require_deepreach_frame_contract(
        path,
        reference_dataset=reference_dataset,
    )
    source_frame_count = int(frame_contract["expected_frame_count"])
    with h5py.File(path, "r") as handle:
        required = [
            "timestamp",
            "hand/left/joints3d",
            "hand/right/joints3d",
        ]
        missing = [name for name in required if name not in handle]
        if missing:
            raise ValueError(f"missing DeepReach datasets: {', '.join(missing)}")
        if end_frame >= source_frame_count:
            raise ValueError(
                f"end_frame {end_frame} outside source frame count "
                f"{source_frame_count}"
            )
        source_slice = slice(start_frame, end_frame + 1)
        side_joints: dict[str, np.ndarray] = {}
        side_raw_joints: dict[str, np.ndarray] = {}
        side_joint_valid: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            joints = np.asarray(
                handle[f"hand/{side}/joints3d"][source_slice],
                dtype=np.float32,
            )
            if joints.ndim != 3 or joints.shape[1:] != (21, 3):
                raise ValueError(
                    f"hand/{side}/joints3d must have shape (N, 21, 3)"
                )
            valid_path = f"hand/{side}/valid"
            if valid_path in handle:
                valid = np.asarray(handle[valid_path][source_slice]).reshape(-1)
                if valid.shape[0] != joints.shape[0]:
                    raise ValueError(f"{valid_path} length mismatch")
                hand_valid = valid.astype(bool)
            else:
                hand_valid = np.ones(joints.shape[0], dtype=np.bool_)
            raw_joints = joints.copy()
            joint_valid = (
                hand_valid[:, np.newaxis]
                & np.isfinite(raw_joints).all(axis=-1)
            )
            checked_joints = raw_joints.copy()
            checked_joints[~joint_valid] = np.nan
            side_raw_joints[side] = raw_joints
            side_joint_valid[side] = joint_valid
            side_joints[side] = checked_joints
        fps = float(row.get("fps") or handle.attrs.get("fps") or 29.97)
        hdf5_task = _text(handle.attrs.get("task"))
        coordinate_frame = _text(handle.attrs.get("coordinate_frame"))
        units = _text(handle.attrs.get("units"))

    text_label = {
        "task": _text(row.get("task")) or hdf5_task,
        "subtask_description": _text(row.get("subtask_description")),
    }
    clip = ClipInputs(
        episode_idx=episode_idx,
        frame_indices=list(range(start_frame, end_frame + 1)),
        keypoints=_canonical_keypoints(
            side_joints["left"],
            side_joints["right"],
        ),
        quality_hand=None,
        instruction=text_label["subtask_description"] or text_label["task"],
        text_label=text_label,
        hand_keypoints_3d=np.stack(
            (side_raw_joints["left"], side_raw_joints["right"]),
            axis=1,
        ).astype(np.float32, copy=False),
        hand_joint_valid_3d=np.stack(
            (side_joint_valid["left"], side_joint_valid["right"]),
            axis=1,
        ).astype(np.bool_, copy=False),
        fps=fps,
    )
    setattr(clip, "source_frame_count", source_frame_count)
    setattr(clip, "coordinate_frame", coordinate_frame)
    setattr(clip, "units", units)
    return _attach_common_metadata(
        clip,
        asset_id=asset_id,
        supplier="deepreach",
        source_path=path,
        start_frame=start_frame,
        end_frame=end_frame,
    )


def _reshape_cells(
    frame: pd.DataFrame,
    column: str,
    shape: tuple[int, int],
) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"missing JDT parquet column: {column}")
    expected = shape[0] * shape[1]
    values: list[np.ndarray] = []
    for row_offset, value in enumerate(frame[column].tolist()):
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size != expected:
            raise ValueError(
                f"{column} row {row_offset} has {array.size} values; expected {expected}"
            )
        values.append(array.reshape(shape))
    return np.stack(values, axis=0)


def _reshape_keypoint_cells(
    frame: pd.DataFrame,
    column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Canonicalize up to 21x3 values while preserving short-cell invalidity."""
    if column not in frame.columns:
        return (
            np.full((len(frame), 21, 3), np.nan, dtype=np.float32),
            np.zeros(len(frame), dtype=np.int64),
        )
    expected = 21 * 3
    values: list[np.ndarray] = []
    source_value_counts: list[int] = []
    for row_offset, value in enumerate(frame[column].tolist()):
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size > expected:
            raise ValueError(
                f"{column} row {row_offset} has {array.size} values; "
                f"expected at most {expected}"
            )
        padded = np.full(expected, np.nan, dtype=np.float32)
        padded[: array.size] = array
        values.append(padded.reshape(21, 3))
        source_value_counts.append(int(array.size))
    return np.stack(values, axis=0), np.asarray(source_value_counts, dtype=np.int64)


def load_jdt_clip(
    row: dict[str, Any],
    *,
    episode_idx: int,
    start_frame_column: str = "start_frame",
    end_frame_column: str = "end_frame",
    parquet_column: str = "parquet_path",
    source_frame: pd.DataFrame | None = None,
) -> ClipInputs:
    asset_id = _asset_id(row)
    start_frame, end_frame = _range(
        row,
        start_frame_column,
        end_frame_column,
    )
    path_text = _text(row.get(parquet_column))
    if not path_text:
        raise ValueError(f"missing {parquet_column}")
    path = Path(path_text).expanduser()
    full_frame = source_frame if source_frame is not None else pd.read_parquet(path)
    if end_frame >= len(full_frame):
        raise ValueError(
            f"end_frame {end_frame} outside source frame count {len(full_frame)}"
        )
    sliced = full_frame.iloc[start_frame : end_frame + 1]
    left_3d, left_3d_counts = _reshape_keypoint_cells(sliced, "left_kp3d")
    right_3d, right_3d_counts = _reshape_keypoint_cells(sliced, "right_kp3d")
    left_2d = _reshape_cells(sliced, "leftcam_left_kp2d", (21, 2))
    right_2d = _reshape_cells(sliced, "leftcam_right_kp2d", (21, 2))
    first = sliced.iloc[0]
    text_label = {
        field: _text(first.get(field))
        for field in JDT_TEXT_FIELDS
        if field in sliced.columns
    }
    clip = ClipInputs(
        episode_idx=episode_idx,
        frame_indices=list(range(start_frame, end_frame + 1)),
        keypoints=_canonical_keypoints(left_3d, right_3d),
        quality_hand=None,
        instruction=text_label.get("language_instruction", ""),
        text_label=text_label,
        fps=_optional_positive_float(row.get("fps")),
    )
    setattr(clip, "leftcam_left_kp2d", left_2d)
    setattr(clip, "leftcam_right_kp2d", right_2d)
    setattr(
        clip,
        "keypoint_source_value_counts",
        {"left": left_3d_counts, "right": right_3d_counts},
    )
    setattr(clip, "primary_camera", "observation.images.cam_left")
    setattr(clip, "source_frame_count", len(full_frame))
    return _attach_common_metadata(
        clip,
        asset_id=asset_id,
        supplier="jdt",
        source_path=path,
        start_frame=start_frame,
        end_frame=end_frame,
    )


def _configured_precheck(
    output_dir: Path,
    supplier: str,
    checks: list[str] | None,
    config_path: Path | None,
    overwrite: bool,
    qc_config_path: Path | None = None,
) -> PrecheckConfig:
    if config_path is not None and qc_config_path is not None:
        raise ValueError("config_path and qc_config_path are mutually exclusive")
    if config_path is not None:
        config = load_precheck_config(config_path)
        requested = list(checks or config.enabled_checks)
        enabled_checks = requested
        config.text_integrity.required_fields = (
            ["task", "subtask_description"]
            if supplier == "deepreach"
            else ["language_instruction"]
        )
        config.skeleton_quality_score.reject_low_quality_hand = False
    else:
        requested = list(checks or DEFAULT_CHECKS)
        config = precheck_config_from_unified(
            load_qc_acceptance_config(qc_config_path),
            module_names=requested,
            output_dir=output_dir,
        )
        enabled_checks = config.enabled_checks
    config.output_dir = output_dir
    config.overwrite = overwrite
    if "keypoint_morphology" in enabled_checks:
        LOGGER.warning(
            "Disabling keypoint_morphology: %s 21-joint topology is not confirmed",
            supplier,
        )
    config.enabled_checks = [
        check for check in enabled_checks if check != "keypoint_morphology"
    ]
    return config


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected JSON list: {path}")
    return [dict(row) for row in payload if isinstance(row, dict)]


def _write_json(path: Path, rows: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(rows), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _run_clip_in_local_coordinates(
    producer: PrecheckRunner,
    clip: ClipInputs,
) -> list[CheckResult]:
    """Run slice-local checks without changing the adapter's source indices."""
    original_frame_indices = clip.frame_indices
    clip.frame_indices = list(range(clip.num_frames))
    try:
        return producer.run_clip(clip)
    finally:
        clip.frame_indices = original_frame_indices


def _result_in_source_coordinates(
    result: CheckResult,
    *,
    clip_start_frame: int,
) -> CheckResult:
    """Map a local result and its temporal pair lineage to source frames."""
    if result.frame_idx < 0:
        return result
    if result.metrics.get("frame_coordinate_system") == "source_inclusive":
        return result
    local_target = int(result.frame_idx)
    source_target = clip_start_frame + local_target
    metrics = dict(result.metrics)
    pair_end = metrics.get("temporal_pair_end_frame")
    if (
        isinstance(pair_end, (int, np.integer))
        and not isinstance(pair_end, bool)
        and int(pair_end) == local_target
    ):
        for key in (
            "temporal_pair_start_frame",
            "temporal_pair_end_frame",
        ):
            value = metrics.get(key)
            if isinstance(value, (int, np.integer)) and not isinstance(
                value,
                bool,
            ):
                metrics[key] = clip_start_frame + int(value)
    standardized_end = metrics.get("standardized_temporal_pair_end_frame")
    standardized_anchor = metrics.get("anchor_source_frame")
    standardized_is_local = (
        isinstance(standardized_end, (int, np.integer))
        and not isinstance(standardized_end, bool)
        and int(standardized_end) == local_target
    ) or (
        isinstance(standardized_anchor, (int, np.integer))
        and not isinstance(standardized_anchor, bool)
        and int(standardized_anchor) == local_target
    )
    native_end = metrics.get("native_temporal_pair_end_frame")
    native_is_local = (
        isinstance(native_end, (int, np.integer))
        and not isinstance(native_end, bool)
        and int(native_end) == local_target
    )
    source_value = metrics.get("source_frame_idx")
    if (
        isinstance(source_value, (int, np.integer))
        and not isinstance(source_value, bool)
        and int(source_value) == local_target
    ):
        metrics["source_frame_idx"] = clip_start_frame + int(source_value)
    for key in ("anchor_source_frame",):
        value = metrics.get(key)
        if (
            standardized_is_local
            and isinstance(value, (int, np.integer))
            and not isinstance(value, bool)
        ):
            metrics[key] = clip_start_frame + int(value)
    if standardized_is_local:
        for key in (
            "standardized_temporal_pair_start_frame",
            "standardized_temporal_pair_end_frame",
        ):
            value = metrics.get(key)
            if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
                metrics[key] = clip_start_frame + int(value)
    if native_is_local:
        for key in (
            "native_temporal_pair_start_frame",
            "native_temporal_pair_end_frame",
        ):
            value = metrics.get(key)
            if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
                metrics[key] = clip_start_frame + int(value)
    for key in ("evidence_source_frames",):
        values = metrics.get(key)
        if standardized_is_local and isinstance(values, list) and all(
            isinstance(value, (int, np.integer)) and not isinstance(value, bool)
            for value in values
        ):
            metrics[key] = [clip_start_frame + int(value) for value in values]
    sampling_audit = metrics.get("temporal_sampling_audit")
    if isinstance(sampling_audit, dict):
        mapping = sampling_audit.get("source_frame_mapping")
        if isinstance(mapping, list) and all(
            isinstance(value, (int, np.integer)) and not isinstance(value, bool)
            for value in mapping
        ):
            metrics["temporal_sampling_audit"] = {
                **sampling_audit,
                "source_frame_mapping": [
                    clip_start_frame + int(value) for value in mapping
                ],
            }
    metrics["frame_coordinate_system"] = "source_inclusive"
    return replace(result, frame_idx=source_target, metrics=metrics)


def _map_candidate_window_to_source(
    candidate: dict[str, Any],
    *,
    asset_id: str,
    supplier: str,
    source_path: str,
    clip_start_frame: int,
    clip_end_frame: int,
    clip_frame_count: int,
) -> dict[str, Any]:
    """Validate a candidate window and export source plus local coordinates."""
    if clip_frame_count <= 0 or clip_end_frame - clip_start_frame + 1 != clip_frame_count:
        raise ValueError(
            "clip frame bounds do not match clip_frame_count: "
            f"{clip_start_frame}..{clip_end_frame}, count={clip_frame_count}"
        )

    coordinate_space = _text(candidate.get("coordinate_space")).lower() or "local"
    if coordinate_space not in {"local", "source"}:
        raise ValueError(f"unsupported candidate coordinate_space: {coordinate_space}")

    start_frame = _integer(candidate.get("start_frame"), "candidate start_frame")
    end_frame = _integer(candidate.get("end_frame"), "candidate end_frame")
    if coordinate_space == "source":
        source_start_frame = start_frame
        source_end_frame = end_frame
        local_start_frame = source_start_frame - clip_start_frame
        local_end_frame = source_end_frame - clip_start_frame
    else:
        local_start_frame = start_frame
        local_end_frame = end_frame
        source_start_frame = clip_start_frame + local_start_frame
        source_end_frame = clip_start_frame + local_end_frame

    source_bounds_valid = (
        clip_start_frame
        <= source_start_frame
        <= source_end_frame
        <= clip_end_frame
    )
    local_bounds_valid = (
        0 <= local_start_frame <= local_end_frame < clip_frame_count
    )
    if coordinate_space == "source" and not source_bounds_valid:
        raise ValueError(
            "invalid source candidate bounds: "
            f"{source_start_frame}..{source_end_frame}; "
            f"clip={clip_start_frame}..{clip_end_frame}"
        )
    if not local_bounds_valid:
        raise ValueError(
            "invalid local candidate bounds: "
            f"{local_start_frame}..{local_end_frame}; "
            f"clip_frame_count={clip_frame_count}"
        )
    if not source_bounds_valid:
        raise ValueError(
            "invalid source candidate bounds: "
            f"{source_start_frame}..{source_end_frame}; "
            f"clip={clip_start_frame}..{clip_end_frame}"
        )

    return {
        **candidate,
        "asset_id": asset_id,
        "supplier_id": supplier,
        "source_path": source_path,
        "clip_start_frame": clip_start_frame,
        "clip_end_frame": clip_end_frame,
        "local_start_frame": local_start_frame,
        "local_end_frame": local_end_frame,
        "start_frame": source_start_frame,
        "end_frame": source_end_frame,
        "source_start_frame": source_start_frame,
        "source_end_frame": source_end_frame,
        "coordinate_space": "source",
        "frame_coordinate_system": "source_inclusive",
    }


def _result_records(
    results: list[CheckResult],
    clip: ClipInputs,
) -> list[dict[str, Any]]:
    start_frame = int(getattr(clip, "clip_start_frame"))
    end_frame = int(getattr(clip, "clip_end_frame"))
    rows: list[dict[str, Any]] = []
    for result in results:
        local_frame_idx = result.frame_idx if result.frame_idx >= 0 else None
        if local_frame_idx is not None and not (
            0 <= local_frame_idx < clip.num_frames
        ):
            raise ValueError(
                f"result local frame {local_frame_idx} outside clip frame count "
                f"{clip.num_frames}"
            )
        mapped_result = _result_in_source_coordinates(
            result,
            clip_start_frame=start_frame,
        )
        source_frame_idx = mapped_result.frame_idx if local_frame_idx is not None else None
        rows.append(
            {
                **mapped_result.to_record(),
                "asset_id": getattr(clip, "asset_id"),
                "supplier_id": getattr(clip, "supplier_id"),
                "source_path": getattr(clip, "source_path"),
                "clip_start_frame": start_frame,
                "clip_end_frame": end_frame,
                "clip_frame_count": end_frame - start_frame + 1,
                "local_frame_idx": local_frame_idx,
                "source_frame_idx": source_frame_idx,
                "frame_coordinate_system": "source_inclusive",
                "supplier_quality_signal": "not_provided",
                "morphology_status": "not_ready_topology",
            }
        )
    return rows


def _aggregate_records(
    results: list[CheckResult],
    clip: ClipInputs,
) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "asset_id": getattr(clip, "asset_id"),
            "supplier_id": getattr(clip, "supplier_id"),
            "source_path": getattr(clip, "source_path"),
            "clip_start_frame": getattr(clip, "clip_start_frame"),
            "clip_end_frame": getattr(clip, "clip_end_frame"),
            "clip_frame_count": clip.num_frames,
            "supplier_quality_signal": "not_provided",
            "morphology_status": "not_ready_topology",
            "frame_coordinate_system": "source_inclusive",
        }
        for row in aggregate_results(results)
    ]


def _combine_by_asset(
    existing: list[dict[str, Any]],
    new: list[dict[str, Any]],
    overwrite: bool,
) -> list[dict[str, Any]]:
    if not overwrite:
        return [*existing, *new]
    replaced = {str(row.get("asset_id")) for row in new}
    return [
        *[row for row in existing if str(row.get("asset_id")) not in replaced],
        *new,
    ]


def _write_parquet_records(
    path: Path,
    records: list[dict[str, Any]],
    *,
    json_columns: tuple[str, ...] = (),
) -> None:
    frame = pd.DataFrame(records)
    for column in json_columns:
        if column in frame.columns:
            frame[column] = frame[column].apply(
                lambda value: json.dumps(_json_safe(value), sort_keys=True)
            )
    frame.to_parquet(path, index=False)


def run_manifest_precheck(
    manifest: Path,
    *,
    supplier: str,
    output_dir: Path,
    start_frame_column: str = "start_frame",
    end_frame_column: str = "end_frame",
    hdf5_column: str = "hdf5_path",
    parquet_column: str = "parquet_path",
    max_clips: int | None = None,
    checks: list[str] | None = None,
    config_path: Path | None = None,
    qc_config_path: Path | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    log_level: str = "INFO",
) -> dict[str, Any]:
    if supplier not in {"deepreach", "jdt"}:
        raise ValueError(f"unsupported supplier: {supplier}")
    if config_path is not None and qc_config_path is not None:
        raise ValueError("config_path and qc_config_path are mutually exclusive")
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    rows = read_manifest(Path(manifest))
    if max_clips is not None:
        if max_clips < 0:
            raise ValueError("max_clips must be >= 0")
        rows = rows[:max_clips]
    output_dir = Path(output_dir)
    config = None
    producer = None
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        config = _configured_precheck(
            output_dir,
            supplier,
            checks,
            config_path,
            overwrite,
            qc_config_path,
        )
        producer = PrecheckRunner(config)

    existing_results = (
        [] if dry_run else _read_json_list(output_dir / "check_results.json")
    )
    existing_aggregates = (
        [] if dry_run else _read_json_list(output_dir / "clip_aggregates.json")
    )
    existing_windows = (
        [] if dry_run else _read_json_list(output_dir / "candidate_windows.json")
    )
    completed_assets = {
        str(row.get("asset_id"))
        for row in existing_aggregates
        if row.get("asset_id")
    }
    new_results: list[dict[str, Any]] = []
    new_aggregates: list[dict[str, Any]] = []
    new_windows: list[dict[str, Any]] = []
    temporal_sampling_by_asset: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    jdt_cache: dict[Path, pd.DataFrame] = {}
    completed = 0
    validated = 0
    skipped = 0
    failed_clips = 0
    for row_index, row in enumerate(rows):
        asset_id = _text(row.get("asset_id")) or f"row-{row_index}"
        if not overwrite and asset_id in completed_assets:
            skipped += 1
            continue
        try:
            if supplier == "deepreach":
                clip = load_deepreach_clip(
                    row,
                    episode_idx=row_index,
                    start_frame_column=start_frame_column,
                    end_frame_column=end_frame_column,
                    hdf5_column=hdf5_column,
                )
            else:
                parquet_text = _text(row.get(parquet_column))
                parquet_path = Path(parquet_text).expanduser()
                if parquet_path not in jdt_cache:
                    jdt_cache[parquet_path] = pd.read_parquet(parquet_path)
                clip = load_jdt_clip(
                    row,
                    episode_idx=row_index,
                    start_frame_column=start_frame_column,
                    end_frame_column=end_frame_column,
                    parquet_column=parquet_column,
                    source_frame=jdt_cache[parquet_path],
                )
            validated += 1
            if dry_run:
                continue
            assert producer is not None
            candidate_start = len(producer.candidate_window_records)
            results = _run_clip_in_local_coordinates(producer, clip)
            candidates = producer.candidate_window_records[candidate_start:]
            mapped_candidates: list[dict[str, Any]] = []
            for candidate_index, candidate in enumerate(candidates):
                try:
                    mapped_candidates.append(
                        _map_candidate_window_to_source(
                            candidate,
                            asset_id=asset_id,
                            supplier=supplier,
                            source_path=str(getattr(clip, "source_path")),
                            clip_start_frame=int(
                                getattr(clip, "clip_start_frame")
                            ),
                            clip_end_frame=int(getattr(clip, "clip_end_frame")),
                            clip_frame_count=clip.num_frames,
                        )
                    )
                except (TypeError, ValueError) as exc:
                    LOGGER.error(
                        "Manifest row %d (%s) candidate %d failed: %s",
                        row_index,
                        asset_id,
                        candidate_index,
                        exc,
                    )
                    failures.append(
                        {
                            "row_index": row_index,
                            "asset_id": asset_id,
                            "source_path": str(getattr(clip, "source_path")),
                            "clip_start_frame": getattr(
                                clip, "clip_start_frame"
                            ),
                            "clip_end_frame": getattr(clip, "clip_end_frame"),
                            "frame_coordinate_system": "source_inclusive",
                            "failure_stage": "candidate_window_mapping",
                            "candidate_index": candidate_index,
                            "candidate_window": _json_safe(candidate),
                            "error": str(exc),
                        }
                    )
            result_records = _result_records(results, clip)
            new_results.extend(result_records)
            temporal_sampling = next(
                (
                    record["metrics"]["temporal_sampling_audit"]
                    for record in result_records
                    if isinstance(record.get("metrics"), dict)
                    and isinstance(
                        record["metrics"].get("temporal_sampling_audit"),
                        dict,
                    )
                ),
                None,
            )
            if isinstance(temporal_sampling, dict):
                temporal_sampling_by_asset[asset_id] = dict(temporal_sampling)
            new_aggregates.extend(_aggregate_records(results, clip))
            new_windows.extend(mapped_candidates)
            completed += 1
        except Exception as exc:
            LOGGER.error("Manifest row %d (%s) failed: %s", row_index, asset_id, exc)
            failed_clips += 1
            failures.append(
                {
                    "row_index": row_index,
                    "asset_id": asset_id,
                    "source_path": _text(
                        row.get(hdf5_column if supplier == "deepreach" else parquet_column)
                    ),
                    "clip_start_frame": row.get(start_frame_column),
                    "clip_end_frame": row.get(end_frame_column),
                    "frame_coordinate_system": "source_inclusive",
                    "failure_stage": "manifest_row",
                    "error": str(exc),
                }
            )

    summary = {
        "manifest_row_count": len(rows),
        "validated_clip_count": validated,
        "completed_clip_count": completed,
        "failed_clip_count": failed_clips,
        "skipped_clip_count": skipped,
        "dry_run": dry_run,
    }
    if dry_run:
        return summary

    assert config is not None
    all_results = _combine_by_asset(existing_results, new_results, overwrite)
    all_aggregates = _combine_by_asset(
        existing_aggregates,
        new_aggregates,
        overwrite,
    )
    all_windows = _combine_by_asset(existing_windows, new_windows, overwrite)
    _write_json(output_dir / "check_results.json", all_results)
    _write_json(output_dir / "clip_aggregates.json", all_aggregates)
    _write_json(output_dir / "candidate_windows.json", all_windows)
    _write_json(output_dir / "failures.json", failures)
    _write_parquet_records(
        output_dir / "check_results.parquet",
        all_results,
        json_columns=("metrics",),
    )
    _write_parquet_records(output_dir / "clip_aggregates.parquet", all_aggregates)
    _write_parquet_records(
        output_dir / "candidate_windows.parquet",
        all_windows,
        json_columns=("trigger_reason", "review_type", "trigger_metrics", "seeds"),
    )
    snapshot = {
        "manifest": str(manifest),
        "supplier": supplier,
        "start_frame_column": start_frame_column,
        "end_frame_column": end_frame_column,
        "hdf5_column": hdf5_column,
        "parquet_column": parquet_column,
        "max_clips": max_clips,
        "overwrite": overwrite,
        "frame_range_semantics": "inclusive_source_frames",
        "topology_status": "not_ready_topology",
        "presence_check_source": (
            "skeleton_quality_score.keypoint_presence_invalid"
        ),
        "supplier_quality_signal": "not_provided",
        "temporal_output_schema_version": "keypoint_temporal.output.v3",
        "decision_metric_source": (
            "standardized_30hz"
            if config.skeleton_quality_score.temporal_decision_timebase
            == "standardized"
            else "native_source_fps"
        ),
        "temporal_sampling_by_asset": temporal_sampling_by_asset,
        "precheck_config": asdict(config),
    }
    _write_json(output_dir / "run_config.json", snapshot)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run prechecks on inclusive manifest frame ranges"
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--supplier", required=True, choices=("deepreach", "jdt"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start-frame-column", default="start_frame")
    parser.add_argument("--end-frame-column", default="end_frame")
    parser.add_argument("--hdf5-column", default="hdf5_path")
    parser.add_argument("--parquet-column", default="parquet_path")
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--checks", nargs="+")
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument(
        "--config-path",
        "--config",
        dest="config_path",
        type=Path,
        help="legacy precheck algorithm config (regression mode only)",
    )
    config_group.add_argument(
        "--qc-config",
        dest="qc_config_path",
        type=Path,
        help="unified QC v2 config",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_manifest_precheck(
        args.manifest,
        supplier=args.supplier,
        output_dir=args.output_dir,
        start_frame_column=args.start_frame_column,
        end_frame_column=args.end_frame_column,
        hdf5_column=args.hdf5_column,
        parquet_column=args.parquet_column,
        max_clips=args.max_clips,
        checks=args.checks,
        config_path=args.config_path,
        qc_config_path=args.qc_config_path,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        log_level=args.log_level,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if not summary["failed_clip_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
