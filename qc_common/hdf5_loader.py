"""Small HDF5 scalar parsing helpers shared by QC loaders."""

from __future__ import annotations

import json
from typing import Any

import numpy as np


def read_scalar_text(value: Any) -> str | None:
    """Decode an HDF5 scalar/string-like value without interpreting it."""
    if hasattr(value, "shape") and value.shape == ():
        value = value[()]
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    return None


def read_scalar_json(value: Any) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """
    Decode a JSON scalar into an object.

    Returns (parsed_dict, raw_text, error). Invalid JSON is reported instead of
    raised so integrity checks can flag the clip without crashing the loader.
    """
    raw_text = read_scalar_text(value)
    if raw_text is None:
        return None, None, "text_label scalar is not string-like"
    try:
        parsed = json.loads(raw_text)
    except Exception as exc:
        return None, raw_text, str(exc)
    if not isinstance(parsed, dict):
        return None, raw_text, "text_label JSON is not an object"
    return parsed, raw_text, None
