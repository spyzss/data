from __future__ import annotations

from datetime import datetime, timezone

import pytest

from human_qc.contracts import BoundaryError, SubtaskSegment
from human_qc.timeline import (
    SharedBoundaryTimeline,
    closed_to_half_open,
    half_open_to_closed,
)


NOW = datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc)


def make_segment(
    internal_id: str, start_frame: int, end_frame_exclusive: int
) -> SubtaskSegment:
    return SubtaskSegment(
        internal_id=internal_id,
        start_frame=start_frame,
        end_frame_exclusive=end_frame_exclusive,
        text_cn=f"中文 {internal_id}",
        text_en=f"English {internal_id}",
        canonical_record={"source_id": internal_id},
    )


def make_timeline(
    *intervals: tuple[int, int], frame_count: int | None = None
) -> SharedBoundaryTimeline:
    if frame_count is None:
        frame_count = intervals[-1][1]
    return SharedBoundaryTimeline(
        frame_count=frame_count,
        fps=29.97,
        segments=tuple(
            make_segment(f"segment-{index}", start, end)
            for index, (start, end) in enumerate(intervals)
        ),
    )


def test_dragging_internal_boundary_changes_exactly_two_segments() -> None:
    timeline = make_timeline((0, 51), (51, 123), (123, 195))
    updated, edit = timeline.move_boundary(
        1, 60, actor_segment_id="segment-1", reviewer="alice", now=NOW,
    )

    assert edit.boundary_id == "b1"
    assert edit.affected_segment_ids == ("segment-0", "segment-1")
    assert [(s.start_frame, s.end_frame_exclusive) for s in edit.after] == [
        (0, 60),
        (60, 123),
    ]
    assert updated.segments[2] == make_segment("segment-2", 123, 195)


def test_move_preserves_asset_identity_and_detaches_nested_snapshots() -> None:
    source_record = {"details": {"status": "before"}}
    first = SubtaskSegment(
        internal_id="segment-0",
        start_frame=0,
        end_frame_exclusive=5,
        text_cn="中文 segment-0",
        text_en="English segment-0",
        canonical_record=source_record,
    )
    second = make_segment("segment-1", 5, 10)
    timeline = SharedBoundaryTimeline(
        frame_count=10,
        fps=30.0,
        segments=(first, second),
        asset_id="asset-1",
    )

    updated, edit = timeline.move_boundary(
        1, 6, actor_segment_id="segment-0", reviewer="alice", now=NOW,
    )
    source_record["details"]["status"] = "source-mutated"
    updated.segments[0].canonical_record["details"]["status"] = "segment-mutated"

    assert timeline.asset_id == updated.asset_id == "asset-1"
    assert edit.before[0].canonical_record["details"]["status"] == "before"
    assert edit.after[0].canonical_record["details"]["status"] == "before"


def test_outer_or_zero_length_boundary_is_rejected() -> None:
    timeline = make_timeline((0, 51), (51, 123))
    with pytest.raises(BoundaryError, match="internal boundary"):
        timeline.move_boundary(
            0, 10, actor_segment_id="segment-0", reviewer="alice", now=NOW,
        )
    with pytest.raises(BoundaryError, match="positive length"):
        timeline.move_boundary(
            1, 0, actor_segment_id="segment-1", reviewer="alice", now=NOW,
        )


def test_closed_interval_conversion_has_no_off_by_one_gap() -> None:
    assert closed_to_half_open(241, 410) == (241, 411)
    assert half_open_to_closed(241, 429) == (241, 428)


def test_first_and_last_internal_boundaries_can_move_to_adjacent_edges() -> None:
    timeline = make_timeline((0, 1), (1, 3), (3, 5))

    moved_first, first_edit = timeline.move_boundary(
        1, 2, actor_segment_id="segment-0", reviewer="alice", now=NOW,
    )
    moved_last, last_edit = moved_first.move_boundary(
        2, 4, actor_segment_id="segment-2", reviewer="alice", now=NOW,
    )

    assert [(s.start_frame, s.end_frame_exclusive) for s in moved_last.segments] == [
        (0, 2),
        (2, 4),
        (4, 5),
    ]
    assert first_edit.boundary_index == 1
    assert last_edit.boundary_index == 2


def test_single_frame_segments_remain_positive_length() -> None:
    timeline = make_timeline((0, 1), (1, 3), (3, 4))

    with pytest.raises(BoundaryError, match="positive length"):
        timeline.move_boundary(
            1, 0, actor_segment_id="segment-1", reviewer="alice", now=NOW,
        )

    updated, _ = timeline.move_boundary(
        1, 2, actor_segment_id="segment-0", reviewer="alice", now=NOW,
    )
    assert [(s.start_frame, s.end_frame_exclusive) for s in updated.segments] == [
        (0, 2),
        (2, 3),
        (3, 4),
    ]


@pytest.mark.parametrize(
    "intervals",
    [
        ((1, 3), (3, 5)),
        ((0, 3), (4, 5)),
        ((0, 4), (3, 5)),
        ((0, 3), (3, 3)),
    ],
)
def test_constructor_rejects_invalid_coverage(intervals: tuple[tuple[int, int], ...]) -> None:
    with pytest.raises(BoundaryError):
        make_timeline(*intervals, frame_count=5)


def test_constructor_rejects_invalid_frame_count_fps_and_empty_segments() -> None:
    with pytest.raises(BoundaryError, match="frame_count"):
        SharedBoundaryTimeline(frame_count=0, fps=30.0, segments=())
    with pytest.raises(BoundaryError, match="fps"):
        SharedBoundaryTimeline(
            frame_count=1, fps=0.0, segments=(make_segment("s", 0, 1),)
        )
    with pytest.raises(BoundaryError, match="at least one"):
        SharedBoundaryTimeline(frame_count=1, fps=30.0, segments=())


def test_move_is_immutable_and_actor_must_be_adjacent_segment() -> None:
    timeline = make_timeline((0, 5), (5, 10), (10, 15))

    with pytest.raises(BoundaryError, match="actor_segment_id"):
        timeline.move_boundary(
            1, 6, actor_segment_id="segment-2", reviewer="alice", now=NOW,
        )

    updated, edit = timeline.move_boundary(
        1, 6, actor_segment_id="segment-0", reviewer="alice", now=NOW,
    )
    assert timeline.segments[0].end_frame_exclusive == 5
    assert updated.segments[0].end_frame_exclusive == 6
    assert edit.before[0].end_frame_exclusive == 5
    assert edit.after[0].end_frame_exclusive == 6


def test_move_rejects_out_of_range_boundary_and_new_frame() -> None:
    timeline = make_timeline((0, 5), (5, 10))

    for index in (-1, 0, 2):
        with pytest.raises(BoundaryError, match="internal boundary"):
            timeline.move_boundary(
                index, 6, actor_segment_id="segment-0", reviewer="alice", now=NOW,
            )
    for frame in (0, 10, -1, 11):
        with pytest.raises(BoundaryError, match="positive length"):
            timeline.move_boundary(
                1, frame, actor_segment_id="segment-0", reviewer="alice", now=NOW,
            )


def test_closed_half_open_round_trip_for_valid_closed_interval() -> None:
    start, end_exclusive = closed_to_half_open(0, 0)
    assert half_open_to_closed(start, end_exclusive) == (0, 0)
    start, end_exclusive = closed_to_half_open(241, 410)
    assert half_open_to_closed(start, end_exclusive) == (241, 410)
