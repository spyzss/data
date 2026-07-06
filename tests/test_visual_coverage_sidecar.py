from acceptance_pull.visual_coverage_sidecar import (
    FRAME_STATUS_CONTAINED_BUT_TRUNCATED_REVIEW,
    FRAME_STATUS_INSUFFICIENT_VISUAL_EVIDENCE,
    FRAME_STATUS_VISIBLE_OK,
    VisualCoverageConfig,
    VisualCoverageFrameInput,
    aggregate_visual_coverage_windows,
    classify_visual_coverage_frame,
    classify_visual_coverage_frames,
)


def _frame(
    frame_idx: int,
    hand_side: str = "left",
    projected_ratio: float = 0.95,
    mask_present: bool = True,
    mask_area_ratio: float = 0.01,
    touches_border: bool = False,
    keypoints_inside_mask_ratio: float | None = None,
    asset_id: str = "asset-1",
) -> VisualCoverageFrameInput:
    return VisualCoverageFrameInput(
        asset_id=asset_id,
        episode_idx=0,
        frame_idx=frame_idx,
        hand_side=hand_side,
        projected_keypoints_in_image_ratio=projected_ratio,
        hand_mask_present=mask_present,
        hand_mask_area_ratio=mask_area_ratio,
        hand_mask_area_px=1000.0 * mask_area_ratio,
        hand_mask_touches_border=touches_border,
        keypoints_inside_hand_mask_ratio=keypoints_inside_mask_ratio,
    )


def test_short_slight_truncation_is_acceptable_flagged() -> None:
    config = VisualCoverageConfig(fps=10.0, short_accept_sec=0.3, fail_min_sec=1.0)
    frames = classify_visual_coverage_frames(
        [
            _frame(0, mask_present=False, mask_area_ratio=0.0),
            _frame(1, mask_present=False, mask_area_ratio=0.0),
            _frame(2),
            _frame(3),
        ],
        config,
    )

    windows = aggregate_visual_coverage_windows(frames, config)

    assert len(windows) == 1
    assert windows[0].bad_frame_count == 2
    assert windows[0].bad_duration_sec == 0.2
    assert windows[0].max_bad_run_frames == 2
    assert windows[0].max_bad_run_duration_sec == 0.2
    assert windows[0].verdict == "acceptable_flagged"


def test_scattered_bad_frames_do_not_fail_without_long_run() -> None:
    config = VisualCoverageConfig(
        fps=10.0,
        fail_min_sec=1.0,
        short_accept_sec=0.3,
        seed_pre_context_sec=0.0,
        seed_post_context_sec=0.0,
        seed_merge_gap_sec=10.0,
    )
    bad_indices = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18}
    frames = classify_visual_coverage_frames(
        [
            _frame(index, mask_present=index not in bad_indices, mask_area_ratio=0.0 if index in bad_indices else 0.01)
            for index in range(20)
        ],
        config,
    )

    windows = aggregate_visual_coverage_windows(frames, config)

    assert len(windows) == 1
    assert windows[0].bad_frame_count == 10
    assert windows[0].bad_duration_sec == 1.0
    assert windows[0].max_bad_run_frames == 1
    assert windows[0].max_bad_run_duration_sec == 0.1
    assert windows[0].verdict == "acceptable_flagged"


def test_consecutive_coverage_mismatch_fails() -> None:
    config = VisualCoverageConfig(fps=10.0, short_accept_sec=0.3, fail_min_sec=1.0)
    frames = classify_visual_coverage_frames(
        [
            _frame(index, mask_present=True, mask_area_ratio=0.0005)
            for index in range(12)
        ],
        config,
    )

    windows = aggregate_visual_coverage_windows(frames, config)

    assert len(windows) == 1
    assert windows[0].bad_frame_count == 12
    assert windows[0].bad_duration_sec == 1.2
    assert windows[0].bad_frame_ratio == 1.0
    assert windows[0].max_bad_run_frames == 12
    assert windows[0].max_bad_run_duration_sec == 1.2
    assert windows[0].verdict == "coverage_fail"


def test_sustained_35_frame_mask_missing_smoke_fails() -> None:
    config = VisualCoverageConfig(fps=29.97, short_accept_sec=0.3, fail_min_sec=1.0)
    frames = classify_visual_coverage_frames(
        [
            _frame(index, mask_present=False, mask_area_ratio=0.0)
            for index in range(35)
        ],
        config,
    )

    windows = aggregate_visual_coverage_windows(frames, config)

    assert len(windows) == 1
    assert windows[0].bad_frame_count == 35
    assert windows[0].max_bad_run_frames == 35
    assert windows[0].max_bad_run_duration_sec >= 1.0
    assert windows[0].verdict == "coverage_fail"


def test_visual_missing_but_keypoints_not_inside_is_not_fake_inside_failure() -> None:
    config = VisualCoverageConfig(fps=10.0)
    result = classify_visual_coverage_frame(
        _frame(
            5,
            projected_ratio=0.3,
            mask_present=False,
            mask_area_ratio=0.0,
        ),
        config,
    )

    assert result.coverage_frame_status == FRAME_STATUS_INSUFFICIENT_VISUAL_EVIDENCE
    assert result.coverage_mismatch_frame is False
    windows = aggregate_visual_coverage_windows([result], config)
    assert windows[0].verdict != "coverage_fail"


def test_mask_touching_border_is_truncated_review_not_visible_ok() -> None:
    result = classify_visual_coverage_frame(
        _frame(
            7,
            mask_present=True,
            mask_area_ratio=0.01,
            touches_border=True,
            keypoints_inside_mask_ratio=0.95,
        )
    )

    assert result.coverage_frame_status == FRAME_STATUS_CONTAINED_BUT_TRUNCATED_REVIEW
    assert result.coverage_frame_status != FRAME_STATUS_VISIBLE_OK
    assert result.contained_but_mask_truncated is True
    assert result.coverage_mismatch_frame is True


def test_normal_visible_hand_is_good() -> None:
    config = VisualCoverageConfig(fps=10.0)
    frames = classify_visual_coverage_frames(
        [_frame(index) for index in range(10)],
        config,
    )

    assert all(frame.coverage_frame_status == FRAME_STATUS_VISIBLE_OK for frame in frames)
    windows = aggregate_visual_coverage_windows(frames, config)
    assert windows[0].verdict == "good"
    assert windows[0].bad_frame_count == 0


def test_side_separation_left_bad_right_good() -> None:
    config = VisualCoverageConfig(fps=10.0, fail_min_sec=1.0)
    left_frames = [
        _frame(index, "left", mask_present=True, mask_area_ratio=0.0005)
        for index in range(12)
    ]
    right_frames = [_frame(index, "right") for index in range(12)]
    frames = classify_visual_coverage_frames(left_frames + right_frames, config)

    windows = aggregate_visual_coverage_windows(frames, config)
    by_side = {window.hand_side: window for window in windows}

    assert by_side["left"].verdict == "coverage_fail"
    assert by_side["right"].verdict == "good"


def test_candidate_both_window_keeps_left_and_right_separate() -> None:
    config = VisualCoverageConfig(fps=10.0, fail_min_sec=1.0)
    left_frames = [
        _frame(index, "left", mask_present=True, mask_area_ratio=0.0005)
        for index in range(12)
    ]
    right_frames = [_frame(index, "right") for index in range(12)]
    frames = classify_visual_coverage_frames(left_frames + right_frames, config)
    candidate_windows = [
        {
            "asset_id": "asset-1",
            "episode_idx": 0,
            "hand_side": "both",
            "start_frame": 0,
            "end_frame": 11,
        }
    ]

    windows = aggregate_visual_coverage_windows(
        frames,
        config,
        candidate_windows=candidate_windows,
    )
    by_side = {window.hand_side: window for window in windows}

    assert set(by_side) == {"left", "right"}
    assert by_side["left"].window_source == "candidate_window"
    assert by_side["left"].verdict == "coverage_fail"
    assert by_side["left"].max_bad_run_frames == 12
    assert by_side["right"].window_source == "candidate_window"
    assert by_side["right"].verdict == "good"
    assert by_side["right"].max_bad_run_frames == 0
