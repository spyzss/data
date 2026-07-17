"""Read-only inventory helpers for non-Core supplier fields."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pyarrow as pa

from .contracts import SupplierExtensionField, SupplierExtensions
from .errors import CanonicalInputError


def _fail(code: str, field: str, detail: str) -> None:
    raise CanonicalInputError(code, field, detail)


def _hdf5_values(dataset: h5py.Dataset, *, field: str) -> np.ndarray:
    try:
        if h5py.check_string_dtype(dataset.dtype) is not None:
            values = np.asarray(dataset.asstr()[()])
        else:
            values = np.asarray(dataset[()])
    except (OSError, ValueError, TypeError, UnicodeError) as exc:
        _fail("field_mapping_error", field, f"cannot read extension losslessly: {exc}")
    if values.dtype.hasobject:
        _fail(
            "field_mapping_error",
            field,
            "object/vlen extension has no registered lossless publication policy",
        )
    return values


def _hdf5_published_name(path: str) -> str:
    parts = path.strip("/").split("/")
    if not parts or any(not part for part in parts):
        _fail("field_mapping_error", path, "extension path is not normalized")
    return "supplier.hdf5." + ".".join(parts)


def hdf5_extension_inventory(
    handle: h5py.File,
    *,
    frame_count: int,
    excluded_paths: Iterable[str] = (),
) -> SupplierExtensions:
    """Load every non-excluded HDF5 dataset without mutating the file."""

    excluded = {path.strip("/") for path in excluded_paths}
    fields: list[SupplierExtensionField] = []

    def visit(name: str, value: h5py.Group | h5py.Dataset) -> None:
        if not isinstance(value, h5py.Dataset) or name in excluded:
            return
        source_path = f"/{name}"
        values = _hdf5_values(value, field=source_path)
        alignment = (
            "frame"
            if values.ndim >= 1 and values.shape[0] == frame_count
            else "episode"
        )
        fields.append(
            SupplierExtensionField(
                published_name=_hdf5_published_name(source_path),
                source_path=source_path,
                values=values,
                time_alignment=alignment,
                metadata={
                    "source_dtype": str(value.dtype),
                    "source_shape": list(value.shape),
                    "compression": value.compression,
                },
            )
        )

    handle.visititems(visit)
    return SupplierExtensions(fields=tuple(fields))


def inventory_hdf5_file(
    path: Path,
    *,
    frame_count: int,
    excluded_paths: Iterable[str] = (),
) -> SupplierExtensions:
    """Public read-only helper used by supplier-specific adapters and smoke tests."""

    try:
        with h5py.File(Path(path), "r") as handle:
            return hdf5_extension_inventory(
                handle,
                frame_count=frame_count,
                excluded_paths=excluded_paths,
            )
    except CanonicalInputError:
        raise
    except OSError as exc:
        raise CanonicalInputError(
            "source_integrity_error",
            "source",
            f"cannot open HDF5 read-only: {exc}",
            retryable=True,
        ) from exc


def _arrow_dtype_name(value_type: pa.DataType) -> str | None:
    if pa.types.is_string(value_type) or pa.types.is_large_string(value_type):
        return "string"
    if pa.types.is_boolean(value_type):
        return "bool"
    if pa.types.is_integer(value_type) or pa.types.is_floating(value_type):
        return np.dtype(value_type.to_pandas_dtype()).name
    return None


def _arrow_leaf_and_shape(
    value_type: pa.DataType,
) -> tuple[pa.DataType, list[int], bool]:
    shape: list[int] = []
    current = value_type
    variable_encoded = False
    while pa.types.is_fixed_size_list(current) or pa.types.is_list(
        current
    ) or pa.types.is_large_list(current):
        if pa.types.is_fixed_size_list(current):
            shape.append(current.list_size)
        else:
            variable_encoded = True
        current = current.value_type
    return current, shape, variable_encoded


def lerobot_extension_inventory(
    table: pa.Table,
    *,
    features: Mapping[str, object],
    excluded_names: Iterable[str] = (),
) -> SupplierExtensions:
    """Load registered non-Core frame columns from one selected episode table."""

    excluded = set(excluded_names)
    fields: list[SupplierExtensionField] = []
    for name in table.column_names:
        if name in excluded:
            continue
        declaration = features.get(name)
        if not isinstance(declaration, dict):
            _fail(
                "schema_missing",
                f"info.features.{name}",
                "extra Parquet column requires an explicit feature declaration",
            )
        leaf, encoded_shape, variable_encoded = _arrow_leaf_and_shape(
            table.schema.field(name).type
        )
        dtype = _arrow_dtype_name(leaf)
        if dtype is None:
            _fail(
                "field_mapping_error",
                name,
                f"unsupported extension Arrow type {table.schema.field(name).type}",
            )
        try:
            values = np.asarray(
                table[name].to_pylist(),
                dtype=np.str_ if dtype == "string" else np.dtype(dtype),
            )
        except (TypeError, ValueError) as exc:
            _fail(
                "field_mapping_error",
                name,
                f"null or irregular extension values have no lossless publication policy: {exc}",
            )
        if values.dtype.hasobject or values.shape[0] != table.num_rows:
            _fail(
                "field_mapping_error",
                name,
                "null or irregular extension values have no lossless publication policy",
            )
        observed_shape = list(values.shape[1:])
        if not variable_encoded and encoded_shape != observed_shape:
            _fail(
                "field_mapping_error",
                name,
                f"Arrow fixed-size shape {encoded_shape!r} differs from values {observed_shape!r}",
            )
        declared_dtype = declaration.get("dtype")
        declared_shape = declaration.get("shape")
        normalized_shape = [1] if not observed_shape else observed_shape
        if declared_dtype != dtype or declared_shape not in (
            observed_shape,
            normalized_shape,
        ):
            _fail(
                "field_mapping_error",
                f"info.features.{name}",
                f"must declare dtype={dtype!r}, shape={normalized_shape!r}",
            )
        fields.append(
            SupplierExtensionField(
                published_name=name,
                source_path=f"parquet:{name}",
                values=values,
                time_alignment="frame",
                metadata=dict(declaration),
            )
        )
    return SupplierExtensions(fields=tuple(fields))


__all__ = [
    "hdf5_extension_inventory",
    "inventory_hdf5_file",
    "lerobot_extension_inventory",
]
