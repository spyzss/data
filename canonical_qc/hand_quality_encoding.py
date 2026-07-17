"""Stable wire encoding for canonical supplier hand-quality status."""

from __future__ import annotations

import base64
from typing import Final

import numpy as np


STATUS_ENCODING_SCHEMA: Final = "supplier_hand_quality_status.v1"
STATUS_NAMES: Final = np.asarray(
    ["unknown", "bad", "warning", "good"], dtype="<U7"
)
STATUS_NAME_TO_CODE: Final = {
    name: code for code, name in enumerate(STATUS_NAMES.tolist())
}
STATUS_ENCODING_SIDECAR: Final = {
    "schema_version": STATUS_ENCODING_SCHEMA,
    "codes": {str(code): name for code, name in enumerate(STATUS_NAMES.tolist())},
}
RAW_VALUE_SIDECAR_SCHEMA: Final = "supplier_hand_quality_raw_value.v1"


def encode_status(values: np.ndarray) -> np.ndarray:
    """Encode canonical status strings as the official-reader-safe uint8 wire type."""

    decoded = np.asarray(values).astype(str)
    unknown = sorted(set(decoded.flat).difference(STATUS_NAME_TO_CODE))
    if unknown:
        raise ValueError(f"unknown canonical hand-quality statuses {unknown!r}")
    return np.asarray(
        np.vectorize(STATUS_NAME_TO_CODE.__getitem__, otypes=[np.uint8])(decoded),
        dtype=np.uint8,
    )


def decode_status(values: np.ndarray) -> np.ndarray:
    """Decode the versioned uint8 wire representation into canonical strings."""

    codes = np.asarray(values)
    if codes.dtype != np.uint8 or np.any(codes >= len(STATUS_NAMES)):
        raise ValueError("status codes must be uint8 values in [0, 3]")
    return STATUS_NAMES[codes]


def encode_raw_value_sidecar(values: np.ndarray) -> dict[str, object]:
    """Encode string/bytes raw Evidence without exposing it to the official reader."""

    array = np.asarray(values)
    if array.dtype.kind == "U":
        encoding = "utf8"
        payload: object = array.tolist()
    elif array.dtype.kind == "S":
        encoding = "base64"
        payload = [
            [base64.b64encode(bytes(item)).decode("ascii") for item in row]
            for row in array
        ]
    else:
        raise ValueError("raw-value sidecar supports only numpy string or bytes arrays")
    return {
        "schema_version": RAW_VALUE_SIDECAR_SCHEMA,
        "numpy_dtype": str(array.dtype),
        "encoding": encoding,
        "shape": list(array.shape),
        "values": payload,
    }


def decode_raw_value_sidecar(payload: object) -> np.ndarray:
    """Strictly restore the exact numpy string/bytes dtype from a v1 sidecar."""

    if not isinstance(payload, dict) or payload.get("schema_version") != RAW_VALUE_SIDECAR_SCHEMA:
        raise ValueError("unsupported supplier raw-value sidecar schema")
    try:
        dtype = np.dtype(payload["numpy_dtype"])
        shape = tuple(payload["shape"])
        encoding = payload["encoding"]
        values = payload["values"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid supplier raw-value sidecar metadata") from exc
    if len(shape) != 2 or any(not isinstance(item, int) or item < 0 for item in shape):
        raise ValueError("supplier raw-value sidecar shape must be [T,2]")
    if dtype.kind == "U" and encoding == "utf8":
        decoded = values
    elif dtype.kind == "S" and encoding == "base64":
        try:
            decoded = [
                [base64.b64decode(item, validate=True) for item in row]
                for row in values
            ]
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid base64 supplier raw-value payload") from exc
    else:
        raise ValueError("supplier raw-value dtype and encoding are inconsistent")
    array = np.asarray(decoded, dtype=dtype)
    if array.shape != shape:
        raise ValueError(f"supplier raw-value sidecar expected shape {shape}, got {array.shape}")
    return array
