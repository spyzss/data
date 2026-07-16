"""Pure source-frame interval contracts for acceptance frame survival."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Literal


SourceFrameRange = tuple[int, int]


@dataclass(frozen=True)
class TemporalTransitionLineage:
    """Source-frame lineage for a temporal transition attributed to its target."""

    pair_start_frame: int
    pair_end_frame: int
    attribution: Literal["target_frame"] = "target_frame"

    def __post_init__(self) -> None:
        if self.pair_start_frame > self.pair_end_frame:
            raise ValueError("temporal pair start_frame must not exceed end_frame")

    def to_metrics(self) -> dict[str, int | str]:
        return {
            "temporal_pair_start_frame": self.pair_start_frame,
            "temporal_pair_end_frame": self.pair_end_frame,
            "temporal_transition_attribution": self.attribution,
        }


def temporal_transition_lineage(
    *,
    target_frame: int,
    pair_start_frame: object | None = None,
    pair_end_frame: object | None = None,
    attribution: object | None = None,
) -> TemporalTransitionLineage:
    """Resolve temporal hard-fail lineage under the target-frame policy.

    Legacy rows without explicit pair fields use the source target frame and
    its immediate predecessor. New producer rows always carry the explicit
    pair fields, so adapters never infer a different attribution policy.
    """
    target = int(target_frame)
    if pair_start_frame is None and pair_end_frame is None:
        start, end = max(0, target - 1), target
    elif pair_start_frame is not None and pair_end_frame is not None:
        start, end = int(pair_start_frame), int(pair_end_frame)
    else:
        raise ValueError("temporal pair lineage requires both start and end")
    if attribution not in {None, "target_frame"}:
        raise ValueError("temporal transition attribution must be target_frame")
    if end != target:
        raise ValueError("temporal target-frame attribution requires pair end == target")
    return TemporalTransitionLineage(start, end)


@dataclass(frozen=True)
class FrameExclusion:
    """One module's exact hard-fail source-frame interval and lineage."""

    start_frame: int
    end_frame: int
    module: str
    reason: str
    raw_severity: Literal["fail"]
    hand_side: str | None
    first_introduced_stage: str
    temporal_pair_start_frame: int | None = None
    temporal_pair_end_frame: int | None = None
    temporal_transition_attribution: Literal["target_frame"] | None = None

    def __post_init__(self) -> None:
        if self.start_frame > self.end_frame:
            raise ValueError("frame exclusion start_frame must not exceed end_frame")
        for field_name in ("module", "reason", "first_introduced_stage"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"frame exclusion {field_name} must not be empty")
        pair_start = self.temporal_pair_start_frame
        pair_end = self.temporal_pair_end_frame
        if (pair_start is None) != (pair_end is None):
            raise ValueError("temporal frame exclusion requires both pair endpoints")
        if pair_start is not None and pair_start > pair_end:
            raise ValueError("temporal pair start_frame must not exceed end_frame")
        if pair_start is not None and self.temporal_transition_attribution is None:
            raise ValueError("temporal frame exclusion requires an attribution policy")
        if pair_start is None and self.temporal_transition_attribution is not None:
            raise ValueError("temporal attribution requires pair endpoints")

    @property
    def source_range(self) -> SourceFrameRange:
        return (self.start_frame, self.end_frame)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "module": self.module,
            "reason": self.reason,
            "raw_severity": self.raw_severity,
            "hand_side": self.hand_side,
            "first_introduced_stage": self.first_introduced_stage,
        }
        if self.temporal_pair_start_frame is not None:
            payload.update(
                {
                    "temporal_pair_start_frame": self.temporal_pair_start_frame,
                    "temporal_pair_end_frame": self.temporal_pair_end_frame,
                    "temporal_transition_attribution": self.temporal_transition_attribution,
                }
            )
        return payload


@dataclass(frozen=True)
class FrameSurvivalUpdate:
    """The acceptance frame-budget state after one producing module."""

    original_frame_count: int
    eligible_frame_count_before_module: int
    newly_excluded_frame_count: int
    cumulative_excluded_frame_count: int
    remaining_frame_count: int
    remaining_frame_ratio: float
    raw_exclusions: tuple[FrameExclusion, ...]
    cumulative_excluded_frame_ranges: tuple[SourceFrameRange, ...]
    eligible_frame_ranges: tuple[SourceFrameRange, ...]
    stop_threshold: float
    stop_triggered: bool
    stop_trigger_module: str | None
    stop_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "original_frame_count": self.original_frame_count,
            "eligible_frame_count_before_module": self.eligible_frame_count_before_module,
            "newly_excluded_frame_count": self.newly_excluded_frame_count,
            "cumulative_excluded_frame_count": self.cumulative_excluded_frame_count,
            "remaining_frame_count": self.remaining_frame_count,
            "remaining_frame_ratio": self.remaining_frame_ratio,
            "raw_exclusions": [item.to_dict() for item in self.raw_exclusions],
            "cumulative_excluded_frame_ranges": [
                list(item) for item in self.cumulative_excluded_frame_ranges
            ],
            "eligible_frame_ranges": [list(item) for item in self.eligible_frame_ranges],
            "stop_threshold": self.stop_threshold,
            "stop_triggered": self.stop_triggered,
            "stop_trigger_module": self.stop_trigger_module,
            "stop_reason": self.stop_reason,
        }


@dataclass(frozen=True)
class FrameSurvivalState:
    """Immutable cumulative frame budget over one manifest-inclusive range."""

    start_frame: int
    end_frame: int
    min_remaining_frame_ratio: float
    raw_exclusions: tuple[FrameExclusion, ...] = ()
    stop_when_below: bool = True

    def __post_init__(self) -> None:
        if self.start_frame > self.end_frame:
            raise ValueError("manifest start_frame must not exceed end_frame")
        if not 0.0 <= self.min_remaining_frame_ratio <= 1.0:
            raise ValueError("min_remaining_frame_ratio must be within [0.0, 1.0]")
        for exclusion in self.raw_exclusions:
            if (
                exclusion.start_frame < self.start_frame
                or exclusion.end_frame > self.end_frame
            ):
                raise ValueError("frame exclusion must fall within manifest bounds")

    @classmethod
    def from_manifest_range(
        cls,
        *,
        start_frame: int,
        end_frame: int,
        min_remaining_frame_ratio: float,
        stop_when_below: bool = True,
    ) -> "FrameSurvivalState":
        return cls(
            start_frame=start_frame,
            end_frame=end_frame,
            min_remaining_frame_ratio=min_remaining_frame_ratio,
            stop_when_below=stop_when_below,
        )

    @property
    def original_frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1

    @property
    def cumulative_excluded_frame_ranges(self) -> tuple[SourceFrameRange, ...]:
        return merge_source_ranges(
            exclusion.source_range for exclusion in self.raw_exclusions
        )

    @property
    def cumulative_excluded_frame_count(self) -> int:
        return source_range_frame_count(self.cumulative_excluded_frame_ranges)

    @property
    def eligible_ranges(self) -> tuple[SourceFrameRange, ...]:
        return subtract_source_ranges(
            ((self.start_frame, self.end_frame),),
            self.cumulative_excluded_frame_ranges,
        )

    @property
    def remaining_frame_count(self) -> int:
        return source_range_frame_count(self.eligible_ranges)

    @property
    def remaining_frame_ratio(self) -> float:
        return self.remaining_frame_count / self.original_frame_count

    def apply(
        self,
        *,
        module: str,
        exclusions: tuple[FrameExclusion, ...],
    ) -> tuple["FrameSurvivalState", FrameSurvivalUpdate]:
        if not module.strip():
            raise ValueError("module must not be empty")
        next_state = replace(
            self,
            raw_exclusions=(*self.raw_exclusions, *exclusions),
        )
        before = self.cumulative_excluded_frame_count
        after = next_state.cumulative_excluded_frame_count
        remaining = next_state.remaining_frame_count
        ratio = remaining / next_state.original_frame_count
        stop = next_state.stop_when_below and ratio < next_state.min_remaining_frame_ratio
        update = FrameSurvivalUpdate(
            original_frame_count=next_state.original_frame_count,
            eligible_frame_count_before_module=self.remaining_frame_count,
            newly_excluded_frame_count=after - before,
            cumulative_excluded_frame_count=after,
            remaining_frame_count=remaining,
            remaining_frame_ratio=ratio,
            raw_exclusions=next_state.raw_exclusions,
            cumulative_excluded_frame_ranges=next_state.cumulative_excluded_frame_ranges,
            eligible_frame_ranges=next_state.eligible_ranges,
            stop_threshold=next_state.min_remaining_frame_ratio,
            stop_triggered=stop,
            stop_trigger_module=module if stop else None,
            stop_reason="insufficient_remaining_frames" if stop else None,
        )
        return next_state, update


def merge_source_ranges(
    ranges: Iterable[SourceFrameRange],
) -> tuple[SourceFrameRange, ...]:
    normalized = sorted(
        (min(int(start), int(end)), max(int(start), int(end)))
        for start, end in ranges
    )
    merged: list[list[int]] = []
    for start, end in normalized:
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return tuple((start, end) for start, end in merged)


def source_range_frame_count(ranges: tuple[SourceFrameRange, ...]) -> int:
    return sum(end - start + 1 for start, end in ranges)


def subtract_source_ranges(
    ranges: tuple[SourceFrameRange, ...],
    excluded: tuple[SourceFrameRange, ...],
) -> tuple[SourceFrameRange, ...]:
    remaining: list[SourceFrameRange] = []
    for start, end in merge_source_ranges(ranges):
        pieces = [(start, end)]
        for excluded_start, excluded_end in merge_source_ranges(excluded):
            next_pieces: list[SourceFrameRange] = []
            for piece_start, piece_end in pieces:
                if excluded_end < piece_start or excluded_start > piece_end:
                    next_pieces.append((piece_start, piece_end))
                    continue
                if piece_start < excluded_start:
                    next_pieces.append((piece_start, excluded_start - 1))
                if excluded_end < piece_end:
                    next_pieces.append((excluded_end + 1, piece_end))
            pieces = next_pieces
        remaining.extend(pieces)
    return merge_source_ranges(remaining)


def source_frame_at(
    frame_indices: Sequence[int] | None,
    frame_offset: int,
    *,
    fallback_start_frame: int = 0,
) -> int:
    if frame_indices is None:
        return fallback_start_frame + frame_offset
    return int(frame_indices[frame_offset])


def is_source_frame_eligible(
    frame_idx: int,
    eligible_ranges: Sequence[SourceFrameRange] | None,
) -> bool:
    if eligible_ranges is None:
        return True
    return any(start <= frame_idx <= end for start, end in eligible_ranges)
