import json
from pathlib import Path

from tools.sam3_keypoint_containment import (
    filter_candidate_windows,
    find_video_path_for_asset,
    load_candidate_windows,
    parse_asset_ids,
    resolve_candidate_asset_paths,
    sample_candidate_window_frames,
    sample_frame_indices,
)


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
