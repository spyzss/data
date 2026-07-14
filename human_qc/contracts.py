"""Immutable contracts for human semantic review.

The human review workbench operates on a normalized, half-open timeline.  The
contracts in this module deliberately contain no persistence or UI concerns;
they are small value objects shared by the timeline and later review services.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class BoundaryError(ValueError):
    """Raised when a timeline or boundary edit would violate its contract."""


@dataclass(frozen=True)
class SubtaskSegment:
    """A normalized subtask interval represented as ``[start, end)``."""

    internal_id: str
    start_frame: int
    end_frame_exclusive: int
    text_cn: str
    text_en: str
    canonical_record: Mapping[str, Any]


@dataclass(frozen=True)
class SegmentSnapshot:
    """The immutable before/after value captured for an edited segment.

    Snapshots retain the same fields as :class:`SubtaskSegment` so a confirmed
    edit can be audited without consulting the mutable source payload.  Text
    and canonical fields default to empty values to make the compact interval
    representation convenient for callers that only need boundary evidence.
    """

    internal_id: str
    start_frame: int
    end_frame_exclusive: int
    text_cn: str = ""
    text_en: str = ""
    canonical_record: Mapping[str, Any] | None = None

    @classmethod
    def from_segment(cls, segment: SubtaskSegment) -> "SegmentSnapshot":
        return cls(
            internal_id=segment.internal_id,
            start_frame=segment.start_frame,
            end_frame_exclusive=segment.end_frame_exclusive,
            text_cn=segment.text_cn,
            text_en=segment.text_en,
            canonical_record=segment.canonical_record,
        )


@dataclass(frozen=True)
class BoundaryEdit:
    """Audit record for one shared-boundary transaction."""

    boundary_id: str
    boundary_index: int
    actor_segment_id: str
    affected_segment_ids: tuple[str, str]
    before: tuple[SegmentSnapshot, SegmentSnapshot]
    after: tuple[SegmentSnapshot, SegmentSnapshot]
    reviewer: str
    created_at: str
