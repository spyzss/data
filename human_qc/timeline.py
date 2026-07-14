"""Shared-boundary timeline model for human semantic calibration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import math

from .contracts import (
    BoundaryEdit,
    BoundaryError,
    SegmentSnapshot,
    SubtaskSegment,
)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def closed_to_half_open(start_frame: int, end_frame: int) -> tuple[int, int]:
    """Convert an inclusive ``[start, end]`` interval to ``[start, end)``."""

    if not _is_int(start_frame) or not _is_int(end_frame):
        raise BoundaryError("closed interval frames must be integers")
    if start_frame < 0 or end_frame < start_frame:
        raise BoundaryError("closed interval must have non-negative positive length")
    return start_frame, end_frame + 1


def half_open_to_closed(
    start_frame: int, end_frame_exclusive: int
) -> tuple[int, int]:
    """Convert a non-empty half-open ``[start, end)`` interval to inclusive."""

    if not _is_int(start_frame) or not _is_int(end_frame_exclusive):
        raise BoundaryError("half-open interval frames must be integers")
    if start_frame < 0 or end_frame_exclusive <= start_frame:
        raise BoundaryError("half-open interval must have positive length")
    return start_frame, end_frame_exclusive - 1


@dataclass(frozen=True)
class SharedBoundaryTimeline:
    """An immutable sequence of contiguous, non-empty half-open segments."""

    frame_count: int
    fps: float
    segments: tuple[SubtaskSegment, ...]

    def __post_init__(self) -> None:
        if not _is_int(self.frame_count) or self.frame_count <= 0:
            raise BoundaryError("frame_count must be a positive integer")
        if not isinstance(self.fps, (int, float)) or isinstance(self.fps, bool):
            raise BoundaryError("fps must be a positive finite number")
        if not math.isfinite(float(self.fps)) or self.fps <= 0:
            raise BoundaryError("fps must be a positive finite number")

        # Normalize a sequence supplied by an adapter to a tuple so the frozen
        # timeline cannot be mutated through its top-level segment collection.
        try:
            segments = tuple(self.segments)
        except TypeError as exc:
            raise BoundaryError("segments must be a sequence of subtasks") from exc
        object.__setattr__(self, "segments", segments)

        if not segments:
            raise BoundaryError("timeline must contain at least one segment")

        seen_ids: set[str] = set()
        for index, segment in enumerate(segments):
            if not isinstance(segment, SubtaskSegment):
                raise BoundaryError("segments must contain SubtaskSegment values")
            if not isinstance(segment.internal_id, str) or not segment.internal_id:
                raise BoundaryError("segment internal_id must not be empty")
            if segment.internal_id in seen_ids:
                raise BoundaryError("segment internal_id values must be unique")
            seen_ids.add(segment.internal_id)
            if not _is_int(segment.start_frame) or not _is_int(
                segment.end_frame_exclusive
            ):
                raise BoundaryError("segment frame boundaries must be integers")
            if segment.start_frame < 0 or segment.end_frame_exclusive > self.frame_count:
                raise BoundaryError("segment frame boundaries must be within frame_count")
            if segment.end_frame_exclusive <= segment.start_frame:
                raise BoundaryError("each segment must have positive length")
            if index == 0 and segment.start_frame != 0:
                raise BoundaryError("first segment must start at frame 0")
            if index > 0:
                previous = segments[index - 1]
                if segment.start_frame != previous.end_frame_exclusive:
                    raise BoundaryError(
                        "segments must be strictly continuous without gaps or overlaps"
                    )

        if segments[-1].end_frame_exclusive != self.frame_count:
            raise BoundaryError("last segment must end at frame_count")

    def move_boundary(
        self,
        boundary_index: int,
        new_frame_exclusive: int,
        actor_segment_id: str,
        reviewer: str,
        now: datetime,
    ) -> tuple["SharedBoundaryTimeline", BoundaryEdit]:
        """Return a new timeline after moving one internal shared boundary.

        ``boundary_index`` identifies the segment on the right side of the
        boundary (the first draggable boundary is therefore ``1``).  Exactly
        the two adjacent segments are copied with updated endpoints; all other
        segments remain the same values.
        """

        if (
            not _is_int(boundary_index)
            or boundary_index < 1
            or boundary_index >= len(self.segments)
        ):
            raise BoundaryError(
                "boundary_index must identify an internal boundary "
                "(1 <= boundary_index < segment count)"
            )

        previous = self.segments[boundary_index - 1]
        following = self.segments[boundary_index]
        if actor_segment_id not in (previous.internal_id, following.internal_id):
            raise BoundaryError(
                "actor_segment_id must identify one of the two adjacent segments"
            )
        if not _is_int(new_frame_exclusive):
            raise BoundaryError("new boundary must leave adjacent segments with positive length")
        if not (
            previous.start_frame < new_frame_exclusive < following.end_frame_exclusive
        ):
            raise BoundaryError(
                "new boundary must leave adjacent segments with positive length"
            )

        updated_previous = replace(
            previous, end_frame_exclusive=new_frame_exclusive
        )
        updated_following = replace(
            following, start_frame=new_frame_exclusive
        )
        updated_segments = list(self.segments)
        updated_segments[boundary_index - 1] = updated_previous
        updated_segments[boundary_index] = updated_following
        updated_timeline = SharedBoundaryTimeline(
            frame_count=self.frame_count,
            fps=self.fps,
            segments=tuple(updated_segments),
        )

        before = (
            SegmentSnapshot.from_segment(previous),
            SegmentSnapshot.from_segment(following),
        )
        after = (
            SegmentSnapshot.from_segment(updated_previous),
            SegmentSnapshot.from_segment(updated_following),
        )
        edit = BoundaryEdit(
            boundary_id=f"b{boundary_index}",
            boundary_index=boundary_index,
            actor_segment_id=actor_segment_id,
            affected_segment_ids=(previous.internal_id, following.internal_id),
            before=before,
            after=after,
            reviewer=reviewer,
            created_at=now.isoformat(),
        )
        return updated_timeline, edit
