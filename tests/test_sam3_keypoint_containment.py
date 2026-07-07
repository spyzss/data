import json
from pathlib import Path

from tools.sam3_keypoint_containment import (
    aggregate_window_containment_summaries,
    classify_containment_frame,
    classify_containment_rows,
    filter_candidate_windows,
    find_video_path_for_asset,
    load_candidate_windows,
    parse_asset_ids,
    resolve_candidate_asset_paths,
    sample_candidate_window_frames,
    sample_frame_indices,
)


def _frame_row(
    verdict_input_inside: float | None,
    projected_ratio: float | None = 1.0,
    mask_present: bool = True,
    frame_idx: int = 0,
    asset_id: str = "100030",
    window_start: int = 10,
    window_end: int = 20,
) -> dict:
    return {
        "asset_id": asset_id,
        "episode_idx": 0,
        "window_start_frame": window_start,
        "window_end_frame": window_end,
        "seed_run_start": 12,
        "seed_run_end": 18,
        "seed_run_frames": 4.0,
        "frame_idx": frame_idx,
        "hand_side": "both",
        "hand_mask_present": mask_present,
        "hand_mask_area_ratio": 0.1 if mask_present else 0.0,
        "projected_keypoints_in_image_ratio": projected_ratio,
        "keypoints_inside_hand_mask_ratio": verdict_input_inside,
    }


def test_candidate_windows_json_is_parsed(tmp_path: Path) -> None:
    path = tmp_path / "candidate_windows.json"
    path.write_text(
        json.dumps(
            [
                {
                    "asset_id": "100030",
                    "start_frame": 10,
                    "end_frame": 20,
                }
            ]
        ),
        encoding="utf-8",
    )

    windows = load_candidate_windows(path)

    assert windows == [{"asset_id": "100030", "start_frame": 10, "end_frame": 20}]


def test_asset_ids_filter_candidate_windows() -> None:
    windows = [
        {"asset_id": "100030", "start_frame": 0, "end_frame": 10},
        {"asset_id": "100044", "start_frame": 20, "end_frame": 30},
        {"asset_id": None, "episode_idx": 2, "start_frame": 40, "end_frame": 50},
    ]

    filtered = filter_candidate_windows(windows, parse_asset_ids("100044,100560"))

    assert filtered == [{"asset_id": "100044", "start_frame": 20, "end_frame": 30}]


def test_candidate_window_frame_sampling_includes_boundaries() -> None:
    window = {"start_frame": 10, "end_frame": 20}

    frames = sample_candidate_window_frames(
        window,
        num_frames=100,
        frames_per_window=3,
        include_boundaries=True,
    )

    assert frames == [10, 11, 15, 19, 20]


def test_candidate_window_frame_sampling_without_boundaries() -> None:
    window = {"start_frame": 10, "end_frame": 20}

    frames = sample_candidate_window_frames(
        window,
        num_frames=100,
        frames_per_window=3,
        include_boundaries=False,
    )

    assert frames == [10, 15, 20]


def test_candidate_window_frame_sampling_clamps_and_deduplicates() -> None:
    window = {"start_frame": -4, "end_frame": 2}

    frames = sample_candidate_window_frames(
        window,
        num_frames=3,
        frames_per_window=3,
        include_boundaries=True,
    )

    assert frames == [0, 1, 2]


def test_asset_id_resolves_hdf5_and_video_paths(tmp_path: Path) -> None:
    hdf5_dir = tmp_path / "hdf5"
    video_dir = tmp_path / "video"
    hdf5_dir.mkdir()
    video_dir.mkdir()
    hdf5_path = hdf5_dir / "100030_hdf5.hdf5"
    video_path = video_dir / "100030_video.mp4"
    hdf5_path.write_bytes(b"hdf5")
    video_path.write_bytes(b"mp4")

    resolved = resolve_candidate_asset_paths(
        window={"asset_id": "100030", "start_frame": 1, "end_frame": 5},
        hdf5_dir=hdf5_dir,
        video_dir=video_dir,
        all_hdf5_paths=[hdf5_path],
        video_patterns=["{episode_id}.mp4"],
        recursive_videos=False,
    )

    assert resolved["error"] is None
    assert resolved["hdf5_path"] == hdf5_path
    assert resolved["video_path"] == video_path


def test_missing_asset_id_can_fall_back_to_episode_idx(tmp_path: Path) -> None:
    hdf5_dir = tmp_path / "hdf5"
    video_dir = tmp_path / "video"
    hdf5_dir.mkdir()
    video_dir.mkdir()
    hdf5_path = hdf5_dir / "100044_hdf5.hdf5"
    video_path = video_dir / "100044_video.mp4"
    hdf5_path.write_bytes(b"hdf5")
    video_path.write_bytes(b"mp4")

    resolved = resolve_candidate_asset_paths(
        window={"episode_idx": 0, "start_frame": 1, "end_frame": 5},
        hdf5_dir=hdf5_dir,
        video_dir=video_dir,
        all_hdf5_paths=[hdf5_path],
        video_patterns=["{episode_id}.mp4"],
        recursive_videos=False,
    )

    assert resolved["error"] is None
    assert resolved["hdf5_path"] == hdf5_path
    assert resolved["video_path"] == video_path


def test_missing_asset_id_without_safe_episode_idx_returns_error(tmp_path: Path) -> None:
    resolved = resolve_candidate_asset_paths(
        window={"start_frame": 1, "end_frame": 5},
        hdf5_dir=tmp_path,
        video_dir=tmp_path,
        all_hdf5_paths=[],
        video_patterns=["{episode_id}.mp4"],
        recursive_videos=False,
    )

    assert "neither asset_id nor episode_idx" in resolved["error"]


def test_existing_sample_fraction_helper_still_works() -> None:
    assert sample_frame_indices(10, 0.3) == [0, 4, 9]


def test_asset_video_resolution_preserves_existing_patterns(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "100030_hdf5.hdf5"
    video_path = tmp_path / "100030.mp4"
    hdf5_path.write_bytes(b"hdf5")
    video_path.write_bytes(b"mp4")

    resolved = find_video_path_for_asset(
        asset_id="100030",
        hdf5_path=hdf5_path,
        video_dir=tmp_path,
        patterns=["{episode_id}.mp4"],
        recursive=False,
    )

    assert resolved == video_path


def test_low_projected_ratio_is_projection_review_not_strong_fail() -> None:
    verdict, reason = classify_containment_frame(
        projected_in_image_ratio=0.0,
        inside_ratio=0.0,
        hand_mask_present=True,
        hand_mask_tiny=False,
    )

    assert verdict == "projection_review"
    assert reason == "insufficient_projection_evidence"


def test_inside_zero_with_valid_projection_is_strong_mismatch() -> None:
    verdict, reason = classify_containment_frame(
        projected_in_image_ratio=1.0,
        inside_ratio=0.0,
        hand_mask_present=True,
        hand_mask_tiny=False,
    )

    assert verdict == "strong_containment_mismatch"
    assert "mostly outside hand mask" in reason


def test_inside_ratio_high_is_likely_visible_ok() -> None:
    verdict, reason = classify_containment_frame(
        projected_in_image_ratio=1.0,
        inside_ratio=0.75,
        hand_mask_present=True,
        hand_mask_tiny=False,
    )

    assert verdict == "likely_visible_ok"
    assert reason == "keypoints mostly consistent with hand mask"


def test_mask_missing_with_valid_projection_is_mask_review() -> None:
    verdict, reason = classify_containment_frame(
        projected_in_image_ratio=1.0,
        inside_ratio=0.0,
        hand_mask_present=False,
        hand_mask_tiny=False,
    )

    assert verdict == "mask_missing_or_tiny_review"
    assert reason == "hand mask missing or tiny"


def test_classify_containment_rows_updates_verdicts() -> None:
    rows = classify_containment_rows(
        [
            _frame_row(0.0, projected_ratio=0.0, frame_idx=1),
            _frame_row(0.0, projected_ratio=1.0, frame_idx=2),
            _frame_row(0.75, projected_ratio=1.0, frame_idx=3),
            _frame_row(0.0, projected_ratio=1.0, mask_present=False, frame_idx=4),
        ]
    )

    assert [row["containment_verdict"] for row in rows] == [
        "projection_review",
        "strong_containment_mismatch",
        "likely_visible_ok",
        "mask_missing_or_tiny_review",
    ]


def test_window_aggregation_two_strong_frames_fail() -> None:
    rows = classify_containment_rows(
        [
            _frame_row(0.0, frame_idx=1),
            _frame_row(0.1, frame_idx=2),
            _frame_row(0.8, frame_idx=3),
        ]
    )

    summary = aggregate_window_containment_summaries(rows)[0]

    assert summary["strong_fail_frame_count"] == 2
    assert summary["window_containment_verdict"] == "containment_fail"


def test_window_aggregation_only_projection_review() -> None:
    rows = classify_containment_rows(
        [
            _frame_row(0.0, projected_ratio=0.0, frame_idx=1),
            _frame_row(0.1, projected_ratio=0.2, frame_idx=2),
        ]
    )

    summary = aggregate_window_containment_summaries(rows)[0]

    assert summary["projection_review_frame_count"] == 2
    assert summary["strong_fail_frame_count"] == 0
    assert summary["window_containment_verdict"] == "projection_review"


def test_window_aggregation_majority_acceptable() -> None:
    rows = classify_containment_rows(
        [
            _frame_row(0.75, frame_idx=1),
            _frame_row(0.80, frame_idx=2),
            _frame_row(0.45, frame_idx=3),
        ]
    )

    summary = aggregate_window_containment_summaries(rows)[0]

    assert summary["acceptable_frame_count"] == 2
    assert summary["window_containment_verdict"] == "acceptable_flagged"


def test_window_aggregation_mixed_review() -> None:
    rows = classify_containment_rows(
        [
            _frame_row(0.35, frame_idx=1),
            _frame_row(0.0, projected_ratio=0.0, frame_idx=2),
            _frame_row(0.0, projected_ratio=1.0, mask_present=False, frame_idx=3),
        ]
    )

    summary = aggregate_window_containment_summaries(rows)[0]

    assert summary["review_frame_count"] == 1
    assert summary["projection_review_frame_count"] == 1
    assert summary["mask_missing_or_tiny_frame_count"] == 1
    assert summary["window_containment_verdict"] == "mixed_review"
