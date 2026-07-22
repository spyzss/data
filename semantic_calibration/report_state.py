"""Semantic-calibration-owned report initialization."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any


def _require_non_empty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def initialize_semantic_calibration(
    report: dict[str, Any],
    source_dataset_path: str,
    base_hdf5_sha256: str,
) -> None:
    """Create or complete the semantic-calibration report block."""

    if not isinstance(report, dict):
        raise TypeError("report must be a dictionary")
    source_path = _require_non_empty_string(source_dataset_path, "source_dataset_path")
    base_hash = _require_non_empty_string(base_hdf5_sha256, "base_hdf5_sha256")
    existing = report.get("semantic_calibration")
    if existing is None:
        block: dict[str, Any] = {}
    elif isinstance(existing, Mapping):
        block = copy.deepcopy(dict(existing))
    else:
        raise ValueError("semantic_calibration must be an object")
    defaults: dict[str, Any] = {
        "state": "not_started",
        "source_dataset_path": source_path,
        "base_hdf5_sha256": base_hash,
        "final_hdf5_sha256": None,
        "timeline_edit_count": 0,
        "subtask_text_edit_count": 0,
        "pending_edit": None,
        "audit": [],
    }
    for key, value in defaults.items():
        block.setdefault(key, copy.deepcopy(value))
    block["source_dataset_path"] = source_path
    block["base_hdf5_sha256"] = base_hash
    report["semantic_calibration"] = block


__all__ = ["initialize_semantic_calibration"]
