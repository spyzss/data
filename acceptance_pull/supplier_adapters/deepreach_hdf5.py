"""Shape-only DeepReach HDF5 frame-contract inspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py


REQUIRED_FRAME_DATASETS = (
    "timestamp",
    "hand/left/joints3d",
    "hand/right/joints3d",
)
OPTIONAL_FRAME_DATASETS = (
    "hand/left/valid",
    "hand/right/valid",
)
FRAME_DATASETS = (*REQUIRED_FRAME_DATASETS, *OPTIONAL_FRAME_DATASETS)


class DeepReachFrameContractError(ValueError):
    def __init__(self, details: dict[str, Any]) -> None:
        self.details = details
        reason = str(details.get("status") or "invalid_frame_contract")
        super().__init__(
            f"{reason}: "
            + json.dumps(
                details,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )


def inspect_deepreach_frame_contract(
    path: Path,
    *,
    reference_dataset: str,
) -> dict[str, Any]:
    source_path = str(path)
    if reference_dataset not in FRAME_DATASETS:
        return {
            "status": "mapping_invalid",
            "source_path": source_path,
            "reference_dataset": reference_dataset,
            "expected_frame_count": None,
            "dataset_lengths": {},
            "missing_datasets": [],
            "invalid_shape_datasets": [],
            "mismatch_ranges": {},
            "mismatch_count": 0,
        }
    try:
        with h5py.File(path, "r") as handle:
            missing = [
                name for name in REQUIRED_FRAME_DATASETS if name not in handle
            ]
            optional_missing = [
                name for name in OPTIONAL_FRAME_DATASETS if name not in handle
            ]
            lengths: dict[str, int | None] = {}
            invalid_shape_datasets: list[str] = []
            for name in FRAME_DATASETS:
                if name not in handle:
                    continue
                shape = handle[name].shape
                if not shape:
                    lengths[name] = None
                    invalid_shape_datasets.append(name)
                else:
                    lengths[name] = int(shape[0])
    except (OSError, ValueError) as exc:
        return {
            "status": "unreadable",
            "reason": str(exc),
            "source_path": source_path,
            "reference_dataset": reference_dataset,
            "expected_frame_count": None,
            "dataset_lengths": {},
            "missing_datasets": [],
            "invalid_shape_datasets": [],
            "mismatch_ranges": {},
            "mismatch_count": 0,
        }
    if missing or reference_dataset not in lengths:
        return {
            "status": "missing_dataset",
            "source_path": source_path,
            "reference_dataset": reference_dataset,
            "expected_frame_count": lengths.get(reference_dataset),
            "dataset_lengths": lengths,
            "missing_datasets": missing,
            "optional_missing_datasets": optional_missing,
            "invalid_shape_datasets": invalid_shape_datasets,
            "mismatch_ranges": {},
            "mismatch_count": 0,
        }
    if invalid_shape_datasets:
        return {
            "status": "invalid_dataset_shape",
            "source_path": source_path,
            "reference_dataset": reference_dataset,
            "expected_frame_count": lengths.get(reference_dataset),
            "dataset_lengths": lengths,
            "missing_datasets": [],
            "optional_missing_datasets": optional_missing,
            "invalid_shape_datasets": invalid_shape_datasets,
            "mismatch_ranges": {},
            "mismatch_count": 0,
        }
    expected = lengths[reference_dataset]
    assert isinstance(expected, int)
    mismatch_ranges: dict[str, list[int]] = {}
    for name, length in lengths.items():
        assert isinstance(length, int)
        if length < expected:
            mismatch_ranges[name] = [length, expected - 1]
        elif length > expected:
            mismatch_ranges[name] = [expected, length - 1]
    if expected < 1:
        status = "invalid_frame_count"
    elif mismatch_ranges:
        status = "inconsistent_frame_count"
    else:
        status = "consistent"
    return {
        "status": status,
        "source_path": source_path,
        "reference_dataset": reference_dataset,
        "expected_frame_count": expected,
        "dataset_lengths": lengths,
        "missing_datasets": [],
        "optional_missing_datasets": optional_missing,
        "invalid_shape_datasets": [],
        "mismatch_ranges": mismatch_ranges,
        "mismatch_count": len(mismatch_ranges),
    }


def require_deepreach_frame_contract(
    path: Path,
    *,
    reference_dataset: str,
) -> dict[str, Any]:
    details = inspect_deepreach_frame_contract(
        path,
        reference_dataset=reference_dataset,
    )
    if details["status"] != "consistent":
        raise DeepReachFrameContractError(details)
    return details


__all__ = [
    "DeepReachFrameContractError",
    "FRAME_DATASETS",
    "OPTIONAL_FRAME_DATASETS",
    "REQUIRED_FRAME_DATASETS",
    "inspect_deepreach_frame_contract",
    "require_deepreach_frame_contract",
]
