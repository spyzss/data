"""Adapters for loading and encoding canonical subtask timelines.

Supplier files use an inclusive ``start_frame``/``end_frame`` pair while the
human-review workbench uses the shared half-open timeline from
``human_qc.timeline``.  This module is the boundary between those formats:
source payloads are validated and copied on load, and review-only fields are
never written back by the canonical encoder.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Protocol

import h5py
import numpy as np

from .contracts import BoundaryError, SubtaskSegment
from .timeline import (
    SharedBoundaryTimeline,
    closed_to_half_open,
    half_open_to_closed,
)


CANONICAL_FIELDS = frozenset(
    {
        "start_frame",
        "end_frame",
        "start_time_sec",
        "end_time_sec",
        "subtask_cn",
        "subtask_en",
        "verb",
        "object",
        "target",
        "hand",
        "phase",
        "evidence_frames",
        "confidence",
        "status",
    }
)

# A tuple gives encoded records stable ordering while the public set above is
# convenient for callers that need to validate the canonical field boundary.
_CANONICAL_FIELD_ORDER = (
    "start_frame",
    "end_frame",
    "start_time_sec",
    "end_time_sec",
    "subtask_cn",
    "subtask_en",
    "verb",
    "object",
    "target",
    "hand",
    "phase",
    "evidence_frames",
    "confidence",
    "status",
)
_ROOT_FIELDS = ("id", "scene", "task", "fps", "frame_count")

__all__ = [
    "CANONICAL_FIELDS",
    "Hdf5ScalarJsonSubtaskAdapter",
    "LoadedSubtasks",
    "SubtaskSourceAdapter",
    "SubtaskSourceError",
    "encode_canonical_payload",
]


class SubtaskSourceError(ValueError):
    """Raised when a subtask source cannot satisfy the canonical contract."""


class SubtaskSourceAdapter(Protocol):
    """Load a source file into a normalized subtask timeline."""

    def load(
        self, path: Path, *, sidecar_path: Path | None = None
    ) -> "LoadedSubtasks":
        ...


@dataclass(frozen=True)
class LoadedSubtasks:
    """Validated source metadata and its immutable shared-boundary timeline."""

    asset_id: str
    source_kind: str
    dataset_path: str
    timeline: SharedBoundaryTimeline
    root_payload: Mapping[str, Any]


@dataclass(frozen=True)
class Hdf5ScalarJsonSubtaskAdapter:
    """Read scalar JSON subtasks from HDF5, with explicit sidecar fallback."""

    dataset_path: str = "/label/subtask_label"

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_path, str) or not self.dataset_path:
            raise ValueError("dataset_path must be a non-empty string")

    def load(
        self, path: Path, *, sidecar_path: Path | None = None
    ) -> LoadedSubtasks:
        source_path = Path(path)
        source_kind = "hdf5"

        with h5py.File(source_path, "r") as handle:
            if self.dataset_path in handle:
                dataset = handle[self.dataset_path]
                if not isinstance(dataset, h5py.Dataset):
                    raise SubtaskSourceError(
                        f"subtask dataset {self.dataset_path!r} is not a dataset"
                    )
                if dataset.shape != ():
                    raise SubtaskSourceError(
                        f"subtask dataset {self.dataset_path!r} must be scalar"
                    )
                payload = _parse_hdf5_scalar(dataset[()])
            else:
                if sidecar_path is None:
                    raise SubtaskSourceError(
                        f"subtask dataset {self.dataset_path!r} is missing"
                    )
                source_kind = "sidecar"
                payload = _read_sidecar(Path(sidecar_path))

        return _build_loaded_subtasks(
            payload,
            source_kind=source_kind,
            dataset_path=self.dataset_path,
        )


def encode_canonical_payload(
    loaded: LoadedSubtasks, timeline: SharedBoundaryTimeline
) -> dict[str, Any]:
    """Encode a working timeline using the source-compatible canonical schema.

    ``SubtaskSegment.canonical_record`` is intentionally treated as untrusted
    working data.  Every field is copied through the explicit whitelist, text
    values come from the current segment value, and frame/time values are
    recomputed from the supplied half-open timeline.  Internal IDs and review
    helper fields therefore cannot leak into the persisted JSON.
    """

    if not isinstance(loaded, LoadedSubtasks):
        raise TypeError("loaded must be a LoadedSubtasks value")
    if not isinstance(timeline, SharedBoundaryTimeline):
        raise TypeError("timeline must be a SharedBoundaryTimeline value")

    root: dict[str, Any] = {}
    for field in _ROOT_FIELDS:
        if field not in loaded.root_payload:
            raise SubtaskSourceError(f"root field {field!r} is missing")
        root[field] = deepcopy(loaded.root_payload[field])

    # The timeline is the working source of truth for metadata after a review
    # edit.  The source adapter has already validated both values, so this
    # preserves the exact numeric type where practical while preventing stale
    # source metadata from overriding the working timeline.
    root["fps"] = timeline.fps
    root["frame_count"] = timeline.frame_count

    annotations: list[dict[str, Any]] = []
    for segment in timeline.segments:
        if not isinstance(segment, SubtaskSegment):
            raise SubtaskSourceError("timeline contains a non-subtask segment")
        missing = [
            field
            for field in _CANONICAL_FIELD_ORDER
            if field not in segment.canonical_record
        ]
        if missing:
            raise SubtaskSourceError(
                "segment canonical record is missing field(s): "
                + ", ".join(missing)
            )

        start_frame, end_frame = half_open_to_closed(
            segment.start_frame, segment.end_frame_exclusive
        )
        row = {
            field: deepcopy(segment.canonical_record[field])
            for field in _CANONICAL_FIELD_ORDER
        }
        # Text edits are represented directly on the workbench contract and
        # must supersede the original source values during encoding.
        row["subtask_cn"] = deepcopy(segment.text_cn)
        row["subtask_en"] = deepcopy(segment.text_en)
        row["start_frame"] = start_frame
        row["end_frame"] = end_frame
        row["start_time_sec"] = segment.start_frame / timeline.fps
        row["end_time_sec"] = (
            segment.end_frame_exclusive - 1
        ) / timeline.fps
        annotations.append(row)

    root["annotations"] = annotations
    return root


def _parse_hdf5_scalar(value: object) -> Mapping[str, Any]:
    if isinstance(value, np.ndarray):
        # A scalar HDF5 dataset returns a NumPy scalar, not an ndarray.  Keep a
        # defensive check here for unusual h5py adapters and clearer errors.
        raise SubtaskSourceError("subtask dataset must be scalar")
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        try:
            text = bytes(value).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SubtaskSourceError("subtask dataset is not valid UTF-8 JSON") from exc
    elif isinstance(value, (str, np.str_)):
        text = str(value)
    else:
        raise SubtaskSourceError(
            "subtask scalar dataset must contain a UTF-8 JSON string"
        )
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise SubtaskSourceError("subtask dataset does not contain valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise SubtaskSourceError("JSON root must be an object")
    return payload


def _read_sidecar(path: Path) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise SubtaskSourceError(f"sidecar cannot be read: {path}") from exc
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise SubtaskSourceError("sidecar does not contain valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise SubtaskSourceError("sidecar JSON root must be an object")
    return payload


def _build_loaded_subtasks(
    payload: Mapping[str, Any], *, source_kind: str, dataset_path: str
) -> LoadedSubtasks:
    for field in _ROOT_FIELDS:
        if field not in payload:
            raise SubtaskSourceError(f"root field {field!r} is missing")

    source_id = payload["id"]
    if isinstance(source_id, bool) or source_id is None or str(source_id) == "":
        raise SubtaskSourceError("root field 'id' must not be empty")
    asset_id = str(source_id)

    fps = payload["fps"]
    if isinstance(fps, bool) or not isinstance(fps, (int, float)):
        raise SubtaskSourceError("root field 'fps' must be a positive finite number")
    if not math.isfinite(float(fps)) or float(fps) <= 0:
        raise SubtaskSourceError("root field 'fps' must be a positive finite number")

    frame_count = payload["frame_count"]
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count <= 0
    ):
        raise SubtaskSourceError("root field 'frame_count' must be a positive integer")

    annotations = payload["annotations"]
    if not isinstance(annotations, (list, tuple)):
        raise SubtaskSourceError("root field 'annotations' must be a list")
    if not annotations:
        raise SubtaskSourceError("root field 'annotations' must not be empty")

    segments: list[SubtaskSegment] = []
    for ordinal, annotation in enumerate(annotations):
        if not isinstance(annotation, Mapping):
            raise SubtaskSourceError(
                f"annotation {ordinal} must be an object"
            )
        missing = [
            field for field in _CANONICAL_FIELD_ORDER if field not in annotation
        ]
        if missing:
            raise SubtaskSourceError(
                f"annotation {ordinal} is missing field(s): "
                + ", ".join(missing)
            )

        start = annotation["start_frame"]
        end = annotation["end_frame"]
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
        ):
            raise SubtaskSourceError(
                f"annotation {ordinal} frame boundaries must be integers"
            )
        try:
            start_frame, end_frame_exclusive = closed_to_half_open(start, end)
        except BoundaryError as exc:
            raise SubtaskSourceError(
                f"annotation {ordinal} has invalid closed frame interval"
            ) from exc

        text_cn = annotation["subtask_cn"]
        text_en = annotation["subtask_en"]
        if not isinstance(text_cn, str) or not isinstance(text_en, str):
            raise SubtaskSourceError(
                f"annotation {ordinal} subtask text fields must be strings"
            )

        canonical_record = {
            field: deepcopy(annotation[field]) for field in _CANONICAL_FIELD_ORDER
        }
        internal_id = hashlib.sha256(
            f"{asset_id}:{ordinal}:{start}:{end}".encode("utf-8")
        ).hexdigest()[:16]
        segments.append(
            SubtaskSegment(
                internal_id=internal_id,
                start_frame=start_frame,
                end_frame_exclusive=end_frame_exclusive,
                text_cn=deepcopy(text_cn),
                text_en=deepcopy(text_en),
                canonical_record=canonical_record,
            )
        )

    timeline = SharedBoundaryTimeline(
        frame_count=frame_count,
        fps=fps,
        segments=tuple(segments),
    )

    # Keep only source root metadata.  Annotation rows are normalized into
    # independent segment records above, so review-time mutations cannot alias
    # the source mapping or accidentally become persistence fields.
    root_payload = {field: deepcopy(payload[field]) for field in _ROOT_FIELDS}
    return LoadedSubtasks(
        asset_id=asset_id,
        source_kind=source_kind,
        dataset_path=dataset_path,
        timeline=timeline,
        root_payload=root_payload,
    )
