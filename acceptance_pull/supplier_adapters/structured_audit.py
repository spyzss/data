"""Explicitly mapped JSON/CSV audit helpers for supplier sidecars."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import cv2


_MISSING = object()
_TIMESTAMP_UNIT_SCALES = {
    "s": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "ns": 1e-9,
}


def json_path_value(payload: Any, path: str) -> Any:
    current = payload
    for part in path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return _MISSING
        else:
            return _MISSING
    return current


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _finite_integer(value: Any) -> int | None:
    numeric = _finite_float(value)
    if numeric is None or not numeric.is_integer():
        return None
    return int(numeric)


def audit_csv_timeline(
    path: Path,
    mapping: Mapping[str, Any],
    *,
    max_gap_factor: float = 3.0,
    max_frame_step: int = 1,
    require_frame_index: bool = True,
) -> dict[str, Any]:
    if max_frame_step < 1:
        raise ValueError("max_frame_step must be >= 1")
    structure = audit_csv_structure(path)
    if structure["status"] != "pass":
        return structure
    frame_column = mapping.get("frame_index_column")
    timestamp_column = mapping.get("timestamp_column")
    required_columns = list(mapping.get("required_columns") or ())
    numeric_columns = list(mapping.get("numeric_columns") or ())
    if not isinstance(timestamp_column, str) or (
        require_frame_index and not isinstance(frame_column, str)
    ):
        return {
            **structure,
            "status": "unverified",
            "reason": "mapping_missing",
        }
    timestamp_unit = mapping.get("timestamp_unit")
    configured_scale = mapping.get("timestamp_scale_to_seconds")
    if timestamp_unit is None and configured_scale is None:
        return {
            **structure,
            "status": "unverified",
            "reason": "timestamp_unit_unverified",
            "timestamp_derived_fps": None,
            "timestamp_median_interval_sec": None,
        }
    if timestamp_unit is not None and configured_scale is not None:
        return {
            **structure,
            "status": "mapping_invalid",
            "reason": "timestamp_mapping_invalid",
            "timestamp_derived_fps": None,
            "timestamp_median_interval_sec": None,
        }
    if timestamp_unit is not None:
        if not isinstance(timestamp_unit, str) or timestamp_unit not in _TIMESTAMP_UNIT_SCALES:
            return {
                **structure,
                "status": "mapping_invalid",
                "reason": "timestamp_mapping_invalid",
                "timestamp_derived_fps": None,
                "timestamp_median_interval_sec": None,
            }
        timestamp_scale_to_seconds = _TIMESTAMP_UNIT_SCALES[timestamp_unit]
        normalized_unit = timestamp_unit
    else:
        timestamp_scale_to_seconds = _finite_float(configured_scale)
        if timestamp_scale_to_seconds is None or timestamp_scale_to_seconds <= 0:
            return {
                **structure,
                "status": "mapping_invalid",
                "reason": "timestamp_mapping_invalid",
                "timestamp_derived_fps": None,
                "timestamp_median_interval_sec": None,
            }
        normalized_unit = "explicit_scale"
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = tuple(reader.fieldnames or ())
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        return {"status": "invalid", "reason": f"csv_unreadable:{exc}"}
    expected = [timestamp_column, *required_columns, *numeric_columns]
    if require_frame_index:
        expected.append(frame_column)
    missing_columns = sorted(set(expected) - set(fieldnames))
    if missing_columns:
        return {
            "status": "invalid",
            "reason": "mapped_column_missing",
            "missing_columns": missing_columns,
            "row_count": len(rows),
        }

    timestamps: list[float] = []
    frame_indices: list[int] = []
    nonfinite_value_count = 0
    for row in rows:
        timestamp = _finite_float(row.get(timestamp_column))
        if timestamp is None:
            nonfinite_value_count += 1
        else:
            timestamps.append(timestamp)
        if require_frame_index:
            frame_index = _finite_integer(row.get(frame_column))
            if frame_index is None:
                nonfinite_value_count += 1
            else:
                frame_indices.append(frame_index)
        for column in numeric_columns:
            if _finite_float(row.get(column)) is None:
                nonfinite_value_count += 1

    normalized_timestamps = [
        value * timestamp_scale_to_seconds for value in timestamps
    ]
    timestamp_diffs = [
        current - previous
        for previous, current in zip(
            normalized_timestamps,
            normalized_timestamps[1:],
        )
    ]
    positive_timestamp_diffs = [value for value in timestamp_diffs if value > 0]
    median_interval = (
        float(median(positive_timestamp_diffs))
        if positive_timestamp_diffs
        else None
    )
    timestamp_gap_count = (
        sum(
            value > median_interval * max_gap_factor
            for value in positive_timestamp_diffs
        )
        if median_interval is not None
        else 0
    )
    result: dict[str, Any] = {
        "status": "pass",
        "row_count": len(rows),
        "timestamp_field": timestamp_column,
        "timestamp_unit": normalized_unit,
        "timestamp_scale_to_seconds": timestamp_scale_to_seconds,
        "raw_timestamp_min": min(timestamps) if timestamps else None,
        "raw_timestamp_max": max(timestamps) if timestamps else None,
        "normalized_timestamp_min_sec": (
            min(normalized_timestamps) if normalized_timestamps else None
        ),
        "normalized_timestamp_max_sec": (
            max(normalized_timestamps) if normalized_timestamps else None
        ),
        "timestamp_min": min(normalized_timestamps) if normalized_timestamps else None,
        "timestamp_max": max(normalized_timestamps) if normalized_timestamps else None,
        "timestamp_monotonic": all(value > 0 for value in timestamp_diffs),
        "timestamp_duplicate_count": sum(value == 0 for value in timestamp_diffs),
        "timestamp_gap_count": timestamp_gap_count,
        "timestamp_median_interval_sec": median_interval,
        "timestamp_derived_fps": (
            1.0 / median_interval
            if median_interval is not None and median_interval > 0
            else None
        ),
        "sampling_rate_hz": (
            1.0 / median_interval
            if median_interval is not None and median_interval > 0
            else None
        ),
        "nonfinite_value_count": nonfinite_value_count,
    }
    if require_frame_index:
        frame_diffs = [
            current - previous
            for previous, current in zip(frame_indices, frame_indices[1:])
        ]
        result.update(
            {
                "frame_index_start": frame_indices[0] if frame_indices else None,
                "frame_index_min": min(frame_indices) if frame_indices else None,
                "frame_index_max": max(frame_indices) if frame_indices else None,
                "frame_index_monotonic": all(value > 0 for value in frame_diffs),
                "frame_index_duplicate_count": sum(value == 0 for value in frame_diffs),
                "frame_index_gap_count": sum(
                    value > max_frame_step for value in frame_diffs
                ),
                "frame_index_max_step": max_frame_step,
            }
        )
    invalid = (
        nonfinite_value_count > 0
        or not result["timestamp_monotonic"]
        or timestamp_gap_count > 0
        or (
            require_frame_index
            and (
                not result["frame_index_monotonic"]
                or result["frame_index_gap_count"] > 0
            )
        )
    )
    if invalid:
        result["status"] = "invalid"
    return result


def audit_video_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "status": "input_missing",
            "reason": "required_source_missing",
        }
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return {"status": "invalid", "reason": "video_unreadable"}
        frame_count = _finite_integer(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = _finite_integer(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = _finite_integer(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = _finite_float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if (
        frame_count is None
        or frame_count < 1
        or width is None
        or width < 1
        or height is None
        or height < 1
        or fps is None
        or fps <= 0
    ):
        return {
            "status": "invalid",
            "reason": "video_metadata_missing_or_invalid",
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "fps": fps,
        }
    return {
        "status": "pass",
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "fps": fps,
        "duration_sec": frame_count / fps,
    }


def audit_csv_structure(path: Path) -> dict[str, Any]:
    """Record CSV inventory without interpreting supplier-specific columns."""
    if not path.is_file():
        return {
            "status": "input_missing",
            "reason": "required_source_missing",
        }
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = tuple(reader.fieldnames or ())
            row_count = sum(1 for _ in reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        return {"status": "invalid", "reason": f"csv_unreadable:{exc}"}
    if not fieldnames:
        return {
            "status": "invalid",
            "reason": "csv_header_missing",
            "row_count": row_count,
            "columns": [],
        }
    return {
        "status": "pass",
        "row_count": row_count,
        "columns": list(fieldnames),
    }


def audit_json_structure(path: Path) -> dict[str, Any]:
    """Validate JSON syntax and root shape without guessing semantic fields."""
    if not path.is_file():
        return {
            "status": "input_missing",
            "reason": "required_source_missing",
        }
    payload, error = _read_json(path)
    if error is not None:
        return {"status": "invalid", "reason": f"json_unreadable:{error}"}
    if not isinstance(payload, Mapping):
        return {"status": "invalid", "reason": "json_root_not_object"}
    return {
        "status": "pass",
        "root_type": "object",
        "top_level_keys": sorted(str(key) for key in payload),
    }


def _read_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, str(exc)


def _resolution(payload: Any, mapping: Mapping[str, Any], prefix: str) -> list[int] | None:
    width_path = mapping.get(f"{prefix}width_path")
    height_path = mapping.get(f"{prefix}height_path")
    if not isinstance(width_path, str) or not isinstance(height_path, str):
        return None
    width = _finite_integer(json_path_value(payload, width_path))
    height = _finite_integer(json_path_value(payload, height_path))
    if width is None or height is None or width <= 0 or height <= 0:
        return None
    return [width, height]


def _matrix(payload: Any, path: Any) -> list[list[float]] | None:
    if not isinstance(path, str):
        return None
    value = json_path_value(payload, path)
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(not isinstance(row, list) or len(row) != 3 for row in value)
    ):
        return None
    converted: list[list[float]] = []
    for row in value:
        numbers = [_finite_float(item) for item in row]
        if any(item is None for item in numbers):
            return None
        converted.append([float(item) for item in numbers if item is not None])
    return converted


def _intrinsics_valid(matrix: list[list[float]], resolution: list[int]) -> bool:
    width, height = resolution
    fx, fy = matrix[0][0], matrix[1][1]
    cx, cy = matrix[0][2], matrix[1][2]
    return fx > 0 and fy > 0 and 0 <= cx <= width and 0 <= cy <= height


def audit_potentia_calibration(
    path: Path,
    mapping: Mapping[str, Any],
    *,
    video_resolution: list[int] | None,
    max_scaled_intrinsics_relative_error: float = 0.001,
    scaling_mismatch_action: str = "review",
) -> dict[str, Any]:
    if not path.is_file():
        return {
            "status": "input_missing",
            "reason": "required_source_missing",
        }
    payload, error = _read_json(path)
    if error is not None:
        return {"status": "invalid", "reason": f"json_unreadable:{error}"}
    raw_mapping_fields = (
        "raw_intrinsics_matrix_path",
        "raw_width_path",
        "raw_height_path",
    )
    if any(not isinstance(mapping.get(name), str) for name in raw_mapping_fields):
        return {
            "status": "unverified",
            "reason": "raw_calibration_mapping_missing",
            "raw_resolution": None,
            "scaled_resolution": None,
            "video_resolution": video_resolution,
        }
    raw_resolution = _resolution(payload, mapping, "raw_")
    if raw_resolution is None:
        return {
            "status": "invalid",
            "reason": "calibration_resolution_invalid",
            "raw_resolution": None,
            "scaled_resolution": None,
            "video_resolution": video_resolution,
        }
    raw_matrix = _matrix(payload, mapping.get("raw_intrinsics_matrix_path"))
    if raw_matrix is None:
        return {
            "status": "invalid",
            "reason": "intrinsics_matrix_invalid",
            "raw_resolution": raw_resolution,
            "scaled_resolution": None,
            "video_resolution": video_resolution,
        }
    if not _intrinsics_valid(raw_matrix, raw_resolution):
        return {
            "status": "invalid",
            "reason": "intrinsics_parameters_invalid",
            "raw_resolution": raw_resolution,
            "scaled_resolution": None,
            "video_resolution": video_resolution,
        }

    scaled_mapping_fields = (
        "scaled_intrinsics_matrix_path",
        "scaled_width_path",
        "scaled_height_path",
    )
    has_scaled_mapping = any(
        isinstance(mapping.get(name), str) for name in scaled_mapping_fields
    )
    scaled_resolution = _resolution(payload, mapping, "scaled_")
    scaled_matrix = _matrix(payload, mapping.get("scaled_intrinsics_matrix_path"))
    if has_scaled_mapping and scaled_resolution is None:
        return {
            "status": "invalid",
            "reason": "calibration_resolution_invalid",
            "raw_resolution": raw_resolution,
            "scaled_resolution": None,
            "video_resolution": video_resolution,
        }
    if has_scaled_mapping and scaled_matrix is None:
        return {
            "status": "invalid",
            "reason": "intrinsics_matrix_invalid",
            "raw_resolution": raw_resolution,
            "scaled_resolution": scaled_resolution,
            "video_resolution": video_resolution,
        }
    if (
        scaled_resolution is not None
        and scaled_matrix is not None
        and not _intrinsics_valid(scaled_matrix, scaled_resolution)
    ):
        return {
            "status": "invalid",
            "reason": "intrinsics_parameters_invalid",
            "raw_resolution": raw_resolution,
            "scaled_resolution": scaled_resolution,
            "video_resolution": video_resolution,
        }
    resolution_scale: list[float] | None = None
    scaled_intrinsics_max_abs_error: float | None = None
    scaled_intrinsics_relative_error: float | None = None
    scaling_interpretable: bool | None = None
    if scaled_resolution is not None and scaled_matrix is not None:
        scale_x = scaled_resolution[0] / raw_resolution[0]
        scale_y = scaled_resolution[1] / raw_resolution[1]
        resolution_scale = [scale_x, scale_y]
        expected = [row[:] for row in raw_matrix]
        for column in range(3):
            expected[0][column] *= scale_x
            expected[1][column] *= scale_y
        scaled_intrinsics_max_abs_error = max(
            abs(scaled_matrix[row][column] - expected[row][column])
            for row in range(3)
            for column in range(3)
        )
        magnitude = max(
            1.0,
            *(abs(value) for row in expected for value in row),
        )
        scaled_intrinsics_relative_error = scaled_intrinsics_max_abs_error / magnitude
        scaling_interpretable = (
            scaled_intrinsics_relative_error
            <= max_scaled_intrinsics_relative_error
        )
    selected = "unverified"
    selected_matrix = raw_matrix
    selected_resolution = raw_resolution
    if video_resolution == raw_resolution:
        selected = "raw"
    elif (
        video_resolution is not None
        and scaled_resolution == video_resolution
        and scaled_matrix is not None
    ):
        selected = "scaled"
        selected_matrix = scaled_matrix
        selected_resolution = scaled_resolution
    valid = _intrinsics_valid(selected_matrix, selected_resolution)
    status = "pass" if valid and selected != "unverified" else "unverified"
    reason = None
    if video_resolution is None:
        status = "unverified"
        reason = "video_metadata_missing"
    elif selected == "unverified":
        reason = "resolution_mismatch_review"
    if scaling_interpretable is False:
        if scaling_mismatch_action == "fail":
            status = "invalid"
            reason = "scaling_mismatch_configured_fail"
        else:
            status = "unverified"
            reason = "scaling_unverified"
    return {
        "status": status,
        "reason": reason,
        "raw_resolution": raw_resolution,
        "scaled_resolution": scaled_resolution,
        "video_resolution": video_resolution,
        "selected_intrinsics": selected,
        "intrinsics_valid": valid,
        "resolution_scale": resolution_scale,
        "scaling_interpretable": scaling_interpretable,
        "scaled_intrinsics_max_abs_error": scaled_intrinsics_max_abs_error,
        "scaled_intrinsics_relative_error": scaled_intrinsics_relative_error,
        "max_scaled_intrinsics_relative_error": max_scaled_intrinsics_relative_error,
        "scaling_mismatch_action": scaling_mismatch_action,
    }


def audit_dr_calibration(
    path: Path,
    mapping: Mapping[str, Any],
) -> dict[str, Any]:
    if not path.is_file():
        return {
            "status": "input_missing",
            "reason": "required_source_missing",
        }
    payload, error = _read_json(path)
    if error is not None:
        return {"status": "invalid", "reason": f"json_unreadable:{error}"}
    resolution = _resolution(payload, mapping, "")
    matrix = _matrix(payload, mapping.get("intrinsics_matrix_path"))
    if resolution is None or matrix is None:
        return {"status": "unverified", "reason": "mapping_missing_or_invalid"}
    valid = _intrinsics_valid(matrix, resolution)
    return {
        "status": "pass" if valid else "invalid",
        "resolution": resolution,
        "intrinsics_shape": [3, 3],
        "intrinsics_valid": valid,
    }


def read_mapped_json_values(
    path: Path,
    mapping: Mapping[str, Any],
) -> dict[str, Any]:
    if not path.is_file():
        return {
            "status": "input_missing",
            "reason": "required_source_missing",
            "task": None,
            "qc": None,
        }
    payload, error = _read_json(path)
    if error is not None:
        return {"status": "invalid", "reason": f"json_unreadable:{error}"}
    result: dict[str, Any] = {"status": "pass"}
    for output_name, path_key in (("task", "task_path"), ("qc", "qc_path")):
        configured = mapping.get(path_key)
        if not isinstance(configured, str):
            result[output_name] = None
            continue
        value = json_path_value(payload, configured)
        result[output_name] = None if value is _MISSING else value
    return result


__all__ = [
    "audit_csv_structure",
    "audit_csv_timeline",
    "audit_dr_calibration",
    "audit_json_structure",
    "audit_potentia_calibration",
    "audit_video_metadata",
    "json_path_value",
    "read_mapped_json_values",
]
