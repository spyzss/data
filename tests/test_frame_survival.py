from __future__ import annotations

import pytest

from qc_common.frame_survival import FrameExclusion, FrameSurvivalState


def exclusion(
    start_frame: int,
    end_frame: int,
    *,
    module: str = "keypoint_morphology",
    reason: str = "joint_angle_min_deg_fail",
    hand_side: str = "left",
) -> FrameExclusion:
    return FrameExclusion(
        start_frame=start_frame,
        end_frame=end_frame,
        module=module,
        reason=reason,
        raw_severity="fail",
        hand_side=hand_side,
        first_introduced_stage=module,
    )


def test_frame_survival_keeps_running_at_exactly_ninety_percent() -> None:
    state = FrameSurvivalState.from_manifest_range(
        start_frame=0,
        end_frame=9_999,
        min_remaining_frame_ratio=0.90,
    )

    next_state, update = state.apply(
        module="keypoint_morphology",
        exclusions=(exclusion(0, 999),),
    )

    assert update.original_frame_count == 10_000
    assert update.eligible_frame_count_before_module == 10_000
    assert update.newly_excluded_frame_count == 1_000
    assert update.cumulative_excluded_frame_count == 1_000
    assert update.remaining_frame_count == 9_000
    assert update.remaining_frame_ratio == pytest.approx(0.90)
    assert update.stop_triggered is False
    assert update.stop_trigger_module is None
    assert next_state.eligible_ranges == ((1_000, 9_999),)


def test_frame_survival_stops_strictly_below_ninety_percent() -> None:
    state = FrameSurvivalState.from_manifest_range(
        start_frame=0,
        end_frame=9_999,
        min_remaining_frame_ratio=0.90,
    )

    _next_state, update = state.apply(
        module="keypoint_temporal",
        exclusions=(exclusion(0, 1_000, module="keypoint_temporal"),),
    )

    assert update.remaining_frame_count == 8_999
    assert update.remaining_frame_ratio == pytest.approx(0.8999)
    assert update.stop_triggered is True
    assert update.stop_trigger_module == "keypoint_temporal"
    assert update.to_dict()["stop_reason"] == "insufficient_remaining_frames"


def test_frame_survival_retains_raw_lineage_but_counts_union_once() -> None:
    state = FrameSurvivalState.from_manifest_range(
        start_frame=100,
        end_frame=199,
        min_remaining_frame_ratio=0.0,
    )
    after_presence, _presence_update = state.apply(
        module="keypoint_presence",
        exclusions=(exclusion(100, 109, module="keypoint_presence"),),
    )

    after_morphology, update = after_presence.apply(
        module="keypoint_morphology",
        exclusions=(
            exclusion(105, 114),
            exclusion(115, 119, hand_side="right"),
        ),
    )

    assert len(after_morphology.raw_exclusions) == 3
    assert update.newly_excluded_frame_count == 10
    assert update.cumulative_excluded_frame_count == 20
    assert update.cumulative_excluded_frame_ranges == ((100, 119),)
    assert update.eligible_frame_ranges == ((120, 199),)


def test_stop_when_below_false_records_exclusions_without_stopping() -> None:
    state = FrameSurvivalState.from_manifest_range(
        start_frame=0,
        end_frame=99,
        min_remaining_frame_ratio=0.90,
        stop_when_below=False,
    )

    _state, update = state.apply(
        module="keypoint_presence",
        exclusions=(exclusion(0, 10),),
    )

    assert update.remaining_frame_ratio == pytest.approx(0.89)
    assert update.stop_triggered is False
