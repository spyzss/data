"""Pure strict field readers for the fixed Standard HDF5 v1 contract."""

from __future__ import annotations

import json
from numbers import Integral
from typing import Any

import h5py
import numpy as np

from ..contracts import (
    EpisodeSemantics,
    Subtask,
    SupplierEvidence,
    SupplierHandQuality,
)
from ..errors import CanonicalInputError


STATUS_NAMES = np.asarray(["unknown", "bad", "warning", "good"], dtype="<U7")


def fail(code: str, field: str, detail: str) -> None:
    raise CanonicalInputError(code, field, detail)


def required_attr(container: Any, name: str, *, prefix: str = "/") -> object:
    if name not in container.attrs:
        fail("schema_missing", f"{prefix}@{name}", "required attribute is missing")
    return container.attrs[name]


def text(value: object, *, field: str) -> str:
    if isinstance(value, str):
        result = value
    elif isinstance(value, (bytes, np.bytes_)):
        try:
            result = bytes(value).decode("utf-8")
        except UnicodeDecodeError as exc:
            fail("field_mapping_error", field, f"must be valid UTF-8: {exc}")
    else:
        fail(
            "field_mapping_error",
            field,
            f"must be a UTF-8 string, got {type(value).__name__}",
        )
    if not result.strip():
        fail("field_mapping_error", field, "must be a non-empty string")
    return result


def integer(value: object, *, field: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        fail(
            "field_mapping_error",
            field,
            f"must be an integer, got {type(value).__name__}",
        )
    return int(value)


def typed_integer(
    value: object,
    *,
    field: str,
    dtype: np.dtype[object] | type[object],
) -> int:
    expected = np.dtype(dtype)
    actual = np.asarray(value).dtype
    if actual != expected or np.asarray(value).shape != ():
        fail(
            "field_mapping_error",
            field,
            f"expected scalar dtype {expected.name}, got {actual.name}",
        )
    return integer(value, field=field)


def boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        fail(
            "field_mapping_error",
            field,
            f"must be bool, got {type(value).__name__}",
        )
    return bool(value)


def _direct_link(handle: h5py.File, path: str) -> h5py.HardLink | None:
    parts = [part for part in path.split("/") if part]
    link: h5py.HardLink | None = None
    for index in range(1, len(parts) + 1):
        prefix = "/" + "/".join(parts[:index])
        candidate = handle.get(prefix, getlink=True)
        if candidate is None:
            return None
        if not isinstance(candidate, h5py.HardLink):
            fail(
                "field_mapping_error",
                path,
                f"fixed path prefix {prefix!r} must not use an HDF5 soft or external link",
            )
        link = candidate
    return link


def dataset(handle: h5py.File, path: str) -> h5py.Dataset:
    if _direct_link(handle, path) is None:
        fail("schema_missing", path, "required dataset is missing")
    value = handle[path]
    if not isinstance(value, h5py.Dataset):
        fail("field_mapping_error", path, "must be a dataset")
    return value


def group(
    handle: h5py.File,
    path: str,
    *,
    required: bool = True,
) -> h5py.Group | None:
    if _direct_link(handle, path) is None:
        if required:
            fail("schema_missing", path, "required group is missing")
        return None
    value = handle[path]
    if not isinstance(value, h5py.Group):
        fail("field_mapping_error", path, "must be a group")
    return value


def array(
    handle: h5py.File,
    path: str,
    *,
    dtype: np.dtype[object] | type[object],
    shape: tuple[int, ...],
) -> np.ndarray:
    value = dataset(handle, path)
    expected_dtype = np.dtype(dtype)
    if value.dtype != expected_dtype:
        fail(
            "field_mapping_error",
            path,
            f"expected dtype {expected_dtype.name}, got {value.dtype.name}",
        )
    if value.shape != shape:
        fail(
            "field_mapping_error",
            path,
            f"expected shape {shape}, got {value.shape}",
        )
    return value[()]


def distortion_coefficients(handle: h5py.File) -> np.ndarray:
    path = "/camera/main/distortion_coefficients"
    value = dataset(handle, path)
    if value.dtype != np.dtype(np.float64):
        fail(
            "field_mapping_error",
            path,
            f"expected dtype float64, got {value.dtype.name}",
        )
    if value.ndim != 1:
        fail(
            "field_mapping_error",
            path,
            f"expected one-dimensional shape, got {value.shape}",
        )
    return value[()]


def _json_string(value: h5py.Dataset) -> str:
    field = "/semantics/annotation_json"
    if value.shape != ():
        fail("field_mapping_error", field, "must be a scalar dataset")
    string_info = h5py.check_string_dtype(value.dtype)
    if string_info is None or string_info.encoding.lower().replace("-", "") != "utf8":
        fail("field_mapping_error", field, "must use an explicit UTF-8 string dtype")
    try:
        decoded = value.asstr()[()]
    except (UnicodeDecodeError, OSError, TypeError) as exc:
        fail("field_mapping_error", field, f"cannot decode UTF-8 scalar: {exc}")
    if not isinstance(decoded, str):
        fail("field_mapping_error", field, "must decode to a string")
    return decoded


def _json_required(mapping: dict[str, object], name: str, *, prefix: str) -> object:
    if name not in mapping:
        fail("field_mapping_error", f"{prefix}.{name}", "required field is missing")
    return mapping[name]


def _json_text(mapping: dict[str, object], name: str, *, prefix: str) -> str:
    return text(_json_required(mapping, name, prefix=prefix), field=f"{prefix}.{name}")


def _json_int(mapping: dict[str, object], name: str, *, prefix: str) -> int:
    return integer(
        _json_required(mapping, name, prefix=prefix), field=f"{prefix}.{name}"
    )


def semantics(handle: h5py.File) -> EpisodeSemantics:
    value = dataset(handle, "/semantics/annotation_json")
    try:
        payload = json.loads(_json_string(value))
    except json.JSONDecodeError as exc:
        fail(
            "field_mapping_error",
            "/semantics/annotation_json",
            f"must contain valid JSON: {exc}",
        )
    if not isinstance(payload, dict):
        fail(
            "field_mapping_error",
            "/semantics/annotation_json",
            "JSON root must be an object",
        )
    raw_subtasks = _json_required(payload, "subtask_sequence", prefix="semantics")
    if not isinstance(raw_subtasks, list):
        fail("field_mapping_error", "semantics.subtask_sequence", "must be an array")
    subtasks: list[Subtask] = []
    for index, raw_subtask in enumerate(raw_subtasks):
        prefix = f"semantics.subtask_sequence[{index}]"
        if not isinstance(raw_subtask, dict):
            fail("field_mapping_error", prefix, "must be an object")
        subtasks.append(
            Subtask(
                subtask_id=_json_text(raw_subtask, "subtask_id", prefix=prefix),
                start_frame=_json_int(raw_subtask, "start_frame", prefix=prefix),
                end_frame_exclusive=_json_int(
                    raw_subtask, "end_frame_exclusive", prefix=prefix
                ),
                description_cn=_json_text(
                    raw_subtask, "description_cn", prefix=prefix
                ),
                description_en=_json_text(
                    raw_subtask, "description_en", prefix=prefix
                ),
            )
        )
    return EpisodeSemantics(
        scene_id=_json_text(payload, "scene_id", prefix="semantics"),
        task_id=_json_text(payload, "task_id", prefix="semantics"),
        task_category=_json_text(payload, "task_category", prefix="semantics"),
        task_cn=_json_text(payload, "task_cn", prefix="semantics"),
        task_en=_json_text(payload, "task_en", prefix="semantics"),
        description_cn=_json_text(payload, "description_cn", prefix="semantics"),
        description_en=_json_text(payload, "description_en", prefix="semantics"),
        subtask_sequence=tuple(subtasks),
    )


def supplier_evidence(handle: h5py.File, frame_count: int) -> SupplierEvidence:
    path = "/supplier/hand_quality"
    quality_group = group(handle, path, required=False)
    if quality_group is None:
        return SupplierEvidence()
    provided = boolean(
        required_attr(quality_group, "provided", prefix=path),
        field=f"{path}@provided",
    )
    present_payload = {"raw_value", "normalized_score", "status"}.intersection(
        quality_group.keys()
    )
    if not provided:
        if present_payload or "mapping_version" in quality_group.attrs:
            fail(
                "field_mapping_error",
                path,
                "provided=false forbids payload datasets and mapping_version",
            )
        return SupplierEvidence(
            hand_quality=SupplierHandQuality(
                provided=False,
                status=np.full((frame_count, 2), "unknown", dtype="<U7"),
            )
        )
    if "mapping_version" not in quality_group.attrs:
        fail(
            "field_mapping_error",
            f"{path}@mapping_version",
            "provided=true requires a mapping_version",
        )
    mapping_version = text(
        quality_group.attrs["mapping_version"], field=f"{path}@mapping_version"
    )
    status_path = f"{path}/status"
    status_codes = array(
        handle, status_path, dtype=np.uint8, shape=(frame_count, 2)
    )
    if np.any(status_codes > 3):
        invalid = sorted(int(item) for item in np.unique(status_codes[status_codes > 3]))
        fail(
            "field_mapping_error",
            status_path,
            f"unknown status codes {invalid!r}",
        )
    raw_value: np.ndarray | None = None
    raw_path = f"{path}/raw_value"
    if raw_path in handle:
        raw_dataset = dataset(handle, raw_path)
        if raw_dataset.shape != (frame_count, 2):
            fail(
                "field_mapping_error",
                raw_path,
                f"expected shape {(frame_count, 2)}, got {raw_dataset.shape}",
            )
        if raw_dataset.dtype.hasobject:
            fail("field_mapping_error", raw_path, "object dtype is not supported")
        raw_value = raw_dataset[()]
    normalized_score: np.ndarray | None = None
    score_path = f"{path}/normalized_score"
    if score_path in handle:
        normalized_score = array(
            handle, score_path, dtype=np.float32, shape=(frame_count, 2)
        )
    return SupplierEvidence(
        hand_quality=SupplierHandQuality(
            provided=True,
            raw_value=raw_value,
            normalized_score=normalized_score,
            status=STATUS_NAMES[status_codes],
            mapping_version=mapping_version,
        )
    )
