from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _flat_hand(x_offset: float = 0.0) -> list[float]:
    points = np.asarray(
        [[x_offset + 2 + index % 5, 2 + index // 5] for index in range(21)],
        dtype=np.float64,
    )
    return points.reshape(-1).tolist()


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    video_path = tmp_path / "episode.mp4"
    video_path.write_bytes(b"source video placeholder")
    parquet_path = tmp_path / "episode.parquet"
    pd.DataFrame(
        [
            {
                "leftcam_left_kp2d": _flat_hand(0.0),
                "leftcam_right_kp2d": _flat_hand(8.0),
            }
            for _ in range(12)
        ]
    ).to_parquet(parquet_path, index=False)
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "asset_id": "jd-range",
                "episode_index": 0,
                "start_frame": 3,
                "end_frame": 10,
                "primary_video_path": str(video_path),
                "parquet_path": str(parquet_path),
                "left_hand_2d_field": "leftcam_left_kp2d",
                "right_hand_2d_field": "leftcam_right_kp2d",
            }
        ]
    ).to_csv(manifest_path, index=False)
    windows_path = tmp_path / "candidate_windows.parquet"
    pd.DataFrame(
        [
            {
                "asset_id": "jd-range",
                "start_frame": 3,
                "end_frame": 10,
                "hand_side": "both",
                "review_type": ["temporal_geometry_review"],
                "trigger_reason": ["multi_signal_seed"],
            }
        ]
    ).to_parquet(windows_path, index=False)
    return manifest_path, windows_path, video_path, parquet_path


class FakeSourceCache:
    def __init__(self) -> None:
        self.parquet_reads: dict[Path, int] = {}
        self.video_reads: list[tuple[Path, int]] = []
        self.closed = False

    def read_parquet(self, path: Path) -> pd.DataFrame:
        resolved = path.resolve()
        self.parquet_reads[resolved] = self.parquet_reads.get(resolved, 0) + 1
        return pd.read_parquet(resolved)

    def read_frame(self, path: Path, frame_idx: int) -> np.ndarray:
        self.video_reads.append((path.resolve(), frame_idx))
        return np.zeros((96, 160, 3), dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


class FakeSegmenter:
    def __init__(self) -> None:
        self.calls: list[tuple[int, tuple[str, ...]]] = []

    def segment_frame(self, frame, queries, config):
        self.calls.append((int(frame.shape[0]), tuple(queries)))
        return [
            SimpleNamespace(
                mask=np.ones(frame.shape[:2], dtype=bool),
                category="hand",
            )
        ]


def test_jdt_flat_keypoints_reshape_to_21_by_2() -> None:
    from tools.run_manifest_sam3_containment import reshape_jdt_keypoints

    points = reshape_jdt_keypoints(_flat_hand(), "leftcam_left_kp2d", 7)

    assert points.shape == (21, 2)
    assert points.dtype == np.float64


def test_manifest_and_candidate_readers_support_required_formats(
    tmp_path: Path,
) -> None:
    from tools.run_manifest_sam3_containment import read_records

    jsonl_path = tmp_path / "manifest.jsonl"
    jsonl_path.write_text(
        json.dumps({"asset_id": "one"}) + "\n",
        encoding="utf-8",
    )
    parquet_path = tmp_path / "windows.parquet"
    pd.DataFrame([{"asset_id": "one", "start_frame": 4, "end_frame": 8}]).to_parquet(
        parquet_path,
        index=False,
    )

    assert read_records(jsonl_path) == [{"asset_id": "one"}]
    assert read_records(parquet_path)[0]["start_frame"] == 4


def test_source_cache_reuses_parquet_and_video_handles(tmp_path: Path) -> None:
    from tools.run_manifest_sam3_containment import ManifestSourceCache

    parquet_path = tmp_path / "episode.parquet"
    parquet_path.write_bytes(b"unused")
    video_path = tmp_path / "episode.mp4"
    video_path.write_bytes(b"unused")
    parquet_calls = 0
    capture_calls = 0

    def read_parquet(path: Path) -> pd.DataFrame:
        nonlocal parquet_calls
        parquet_calls += 1
        return pd.DataFrame([{"value": 1}])

    class Capture:
        def set(self, _property, _value):
            return True

        def read(self):
            return True, np.zeros((4, 5, 3), dtype=np.uint8)

        def release(self):
            return None

    def open_video(path: Path):
        nonlocal capture_calls
        capture_calls += 1
        return Capture()

    cache = ManifestSourceCache(
        parquet_reader=read_parquet,
        video_capture_factory=open_video,
    )
    assert cache.read_parquet(parquet_path) is cache.read_parquet(parquet_path)
    first = cache.read_frame(video_path, 3)
    second = cache.read_frame(video_path, 3)
    cache.read_frame(video_path, 4)
    cache.close()

    assert first is second
    assert parquet_calls == 1
    assert capture_calls == 1


def test_nonzero_window_sampling_matches_existing_boundary_behavior() -> None:
    from tools.run_manifest_sam3_containment import sample_manifest_window_frames

    assert sample_manifest_window_frames(
        {"start_frame": 3, "end_frame": 10},
        clip_start_frame=3,
        clip_end_frame=10,
        frames_per_window=3,
    ) == [3, 4, 6, 9, 10]


def test_sam3_thresholds_are_injected_from_unified_config() -> None:
    from qc_common.config import load_qc_acceptance_config
    from tools.run_manifest_sam3_containment import configured_sam3_thresholds

    frame_thresholds, window_thresholds = configured_sam3_thresholds(
        load_qc_acceptance_config()
    )

    assert frame_thresholds == {
        "abnormal_inside_ratio_threshold": 1.0,
        "projected_in_image_ratio_threshold": 0.8,
        "strong_inside_ratio_threshold": 0.2,
        "acceptable_inside_ratio_threshold": 0.6,
        "mask_tiny_area_ratio_threshold": 0.0,
    }
    assert window_thresholds == {
        "fail_min_strong_frames": 3,
        "fail_strong_frame_ratio": 0.6,
    }


def test_runner_evaluates_both_hands_with_source_coordinates(tmp_path: Path) -> None:
    from tools.run_manifest_sam3_containment import run_manifest_sam3_containment

    manifest, windows, video_path, parquet_path = _write_inputs(tmp_path)
    cache = FakeSourceCache()
    segmenter = FakeSegmenter()
    output_dir = tmp_path / "sam3"

    summary = run_manifest_sam3_containment(
        manifest=manifest,
        candidate_windows=windows,
        supplier="jdt",
        output_dir=output_dir,
        frames_per_window=3,
        sam3_model=None,
        queries=["hand"],
        overwrite=False,
        source_cache=cache,
        segmenter=segmenter,
    )

    frame_rows = json.loads(
        (output_dir / "frame_keypoint_containment.json").read_text()
    )
    assert summary["sampled_source_frame_count"] == 5
    assert summary["hand_frame_evaluation_count"] == 10
    assert {row["hand_side"] for row in frame_rows} == {"left", "right"}
    assert {row["source_frame_idx"] for row in frame_rows} == {3, 4, 6, 9, 10}
    assert {row["frame_idx"] for row in frame_rows} == {3, 4, 6, 9, 10}
    assert all(row["clip_start_frame"] == 3 for row in frame_rows)
    assert all(row["clip_end_frame"] == 10 for row in frame_rows)
    assert all(row["candidate_start_frame"] == 3 for row in frame_rows)
    assert all(row["candidate_end_frame"] == 10 for row in frame_rows)
    assert all(row["asset_id"] == "jd-range" for row in frame_rows)
    assert cache.parquet_reads == {parquet_path.resolve(): 1}
    assert {path for path, _ in cache.video_reads} == {video_path.resolve()}
    assert len(cache.video_reads) == 5
    assert len(segmenter.calls) == 5
    window_rows = json.loads(
        (output_dir / "window_keypoint_containment_summary.json").read_text()
    )
    assert {row["hand_side"] for row in window_rows} == {"left", "right"}
    assert all(row["window_start_frame"] == 3 for row in window_rows)
    assert all(row["window_end_frame"] == 10 for row in window_rows)
    for filename in (
        "frame_keypoint_containment.json",
        "frame_keypoint_containment.parquet",
        "window_keypoint_containment_summary.json",
        "window_keypoint_containment_summary.parquet",
        "failures.json",
        "run_config.json",
    ):
        assert (output_dir / filename).exists()
    assert not list(output_dir.rglob("*.mp4"))


def _advance_report_to_sam3(tmp_path: Path, asset_id: str) -> int:
    from qc_common.contracts import ModuleResult
    from qc_common.report_mutation import apply_module_result
    from qc_pipeline.context import AssetContext
    from tests.qc_report_fixtures import loaded_test_config

    config = loaded_test_config()
    modules = config.pipeline_modules
    target_index = modules.index("sam3_containment")
    context = AssetContext(
        asset_id=asset_id,
        batch_root=tmp_path,
        report_path=tmp_path / "quality_archive" / f"{asset_id}.json",
        source_files={},
    )
    revision = 0
    for index, module in enumerate(modules[:target_index]):
        report = apply_module_result(
            context.report_path,
            context=context,
            config=config,
            profile="acceptance",
            result=ModuleResult(module, "pass", {}, {}),
            expected_revision=revision,
            next_module=modules[index + 1],
            now=f"2026-07-14T00:00:{index:02d}Z",
        )
        revision = int(report["report_revision"])
    return revision


def test_runner_groups_asset_windows_into_one_shared_report_write(
    tmp_path: Path,
) -> None:
    from tools.run_manifest_sam3_containment import run_manifest_sam3_containment

    manifest, windows, _, _ = _write_inputs(tmp_path)
    pd.DataFrame(
        [
            {
                "asset_id": "jd-range",
                "start_frame": 3,
                "end_frame": 4,
                "hand_side": "left",
            },
            {
                "asset_id": "jd-range",
                "start_frame": 9,
                "end_frame": 10,
                "hand_side": "right",
            },
        ]
    ).to_parquet(windows, index=False)
    revision = _advance_report_to_sam3(tmp_path, "jd-range")
    output_dir = tmp_path / "sam3"

    summary = run_manifest_sam3_containment(
        manifest=manifest,
        candidate_windows=windows,
        supplier="jdt",
        output_dir=output_dir,
        batch_root=tmp_path,
        profile="acceptance",
        sam3_model=None,
        source_cache=FakeSourceCache(),
        segmenter=FakeSegmenter(),
    )

    report = json.loads(
        (tmp_path / "quality_archive" / "jd-range.json").read_text()
    )
    assert summary["qc_report_write_count"] == 1
    assert report["report_revision"] == revision + 1
    assert report["sam3_containment"]["metrics"]["window_count"] == 2
    assert report["sam3_containment"]["flow"]["result_gate"]["verdict"] == "pass"
    assert all(
        not Path(item["path"]).is_absolute()
        for item in report["sam3_containment"]["evidence"]
    )
    assert json.loads((output_dir / "run_config.json").read_text())[
        "frame_thresholds"
    ]["strong_inside_ratio_threshold"] == 0.2


def test_runner_does_not_commit_asset_when_any_window_fails(
    tmp_path: Path,
) -> None:
    from tools.run_manifest_sam3_containment import run_manifest_sam3_containment

    manifest, windows, _, _ = _write_inputs(tmp_path)
    pd.DataFrame(
        [
            {
                "asset_id": "jd-range",
                "start_frame": 3,
                "end_frame": 4,
                "hand_side": "left",
            },
            {
                "asset_id": "jd-range",
                "start_frame": 10,
                "end_frame": 20,
                "hand_side": "right",
            },
        ]
    ).to_parquet(windows, index=False)
    revision = _advance_report_to_sam3(tmp_path, "jd-range")
    report_path = tmp_path / "quality_archive" / "jd-range.json"
    before = report_path.read_bytes()

    summary = run_manifest_sam3_containment(
        manifest=manifest,
        candidate_windows=windows,
        supplier="jdt",
        output_dir=tmp_path / "sam3",
        batch_root=tmp_path,
        profile="acceptance",
        sam3_model=None,
        source_cache=FakeSourceCache(),
        segmenter=FakeSegmenter(),
    )

    assert summary["failed_asset_count"] == 1
    assert summary["qc_report_write_count"] == 0
    assert report_path.read_bytes() == before
    assert json.loads(before)["report_revision"] == revision


def test_runner_does_not_commit_when_combined_overlay_sidecar_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.run_manifest_sam3_containment as runner

    manifest, windows, _, _ = _write_inputs(tmp_path)
    _advance_report_to_sam3(tmp_path, "jd-range")
    report_path = tmp_path / "quality_archive" / "jd-range.json"
    before = report_path.read_bytes()

    def fail_overlay(**_kwargs):
        raise OSError("overlay encoder failed")

    monkeypatch.setattr(runner, "write_combined_overlay_image", fail_overlay)
    summary = runner.run_manifest_sam3_containment(
        manifest=manifest,
        candidate_windows=windows,
        supplier="jdt",
        output_dir=tmp_path / "sam3",
        batch_root=tmp_path,
        profile="acceptance",
        sam3_model=None,
        source_cache=FakeSourceCache(),
        segmenter=FakeSegmenter(),
    )

    assert summary["failed_window_count"] == 1
    assert summary["failed_asset_count"] == 1
    assert summary["qc_report_write_count"] == 0
    assert report_path.read_bytes() == before


def test_write_overlays_keeps_per_hand_and_adds_combined_review_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cv2
    import tools.run_manifest_sam3_containment as runner

    manifest, windows, _, _ = _write_inputs(tmp_path)
    output_dir = tmp_path / "sam3"

    def write_per_hand(**kwargs):
        target_dir = kwargs["output_dir"]
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / (
            f"{kwargs['clip_id']}_frame_{kwargs['frame_idx']:06d}.png"
        )
        path.write_bytes(b"per-hand-overlay")
        return path

    monkeypatch.setattr(runner, "write_overlay_image", write_per_hand)

    runner.run_manifest_sam3_containment(
        manifest=manifest,
        candidate_windows=windows,
        supplier="jdt",
        output_dir=output_dir,
        frames_per_window=3,
        sam3_model=None,
        write_overlays=True,
        source_cache=FakeSourceCache(),
        segmenter=FakeSegmenter(),
    )

    frame_rows = json.loads(
        (output_dir / "frame_keypoint_containment.json").read_text()
    )
    per_hand_paths = [Path(row["overlay_path"]) for row in frame_rows]
    assert len(per_hand_paths) == 10
    assert all(path.parent == output_dir / "overlays" for path in per_hand_paths)
    assert all(path.exists() for path in per_hand_paths)
    combined_paths = sorted((output_dir / "combined_overlays").glob("*.png"))
    assert len(combined_paths) == 5
    combined_bgr = cv2.imread(str(combined_paths[0]), cv2.IMREAD_COLOR)
    assert combined_bgr is not None
    colors = {
        tuple(pixel)
        for pixel in combined_bgr[..., ::-1].reshape(-1, 3)
    }
    assert (40, 220, 90) in colors
    assert (40, 130, 255) in colors
    assert all("combined_overlay_path" not in row for row in frame_rows)


def test_default_overlay_mode_writes_combined_only_and_canonical_evidence_manifest(
    tmp_path: Path,
) -> None:
    from tools.run_manifest_sam3_containment import run_manifest_sam3_containment

    manifest, windows, _, _ = _write_inputs(tmp_path)
    output_dir = tmp_path / "sam3"

    run_manifest_sam3_containment(
        manifest=manifest,
        candidate_windows=windows,
        supplier="jdt",
        output_dir=output_dir,
        frames_per_window=3,
        sam3_model=None,
        source_cache=FakeSourceCache(),
        segmenter=FakeSegmenter(),
    )

    frame_rows = json.loads(
        (output_dir / "frame_keypoint_containment.json").read_text()
    )
    assert {row["hand_side"] for row in frame_rows} == {"left", "right"}
    assert all("overlay_path" not in row for row in frame_rows)
    assert not (output_dir / "overlays").exists()
    combined_paths = sorted((output_dir / "combined_overlays").glob("*.png"))
    assert len(combined_paths) == 5

    csv_rows = pd.read_csv(output_dir / "review_evidence_manifest.csv")
    parquet_rows = pd.read_parquet(output_dir / "review_evidence_manifest.parquet")
    expected_columns = {
        "review_id",
        "supplier_id",
        "asset_id",
        "window_start_frame",
        "window_end_frame",
        "frame_idx",
        "source_module",
        "evidence_type",
        "hand_side",
        "source_path",
        "metadata_json",
    }
    assert set(csv_rows.columns) == expected_columns
    assert list(parquet_rows.columns) == list(csv_rows.columns)
    assert len(csv_rows) == 5
    assert csv_rows["review_id"].fillna("").eq("").all()
    assert csv_rows["supplier_id"].eq("jdt").all()
    assert csv_rows["source_module"].eq("sam3_containment").all()
    assert csv_rows["evidence_type"].eq("combined_overlay").all()
    assert csv_rows["hand_side"].eq("both").all()
    assert all(Path(path).exists() for path in csv_rows["source_path"])
    run_config = json.loads((output_dir / "run_config.json").read_text())
    assert run_config["overlay_mode"] == "combined"


def test_overlay_mode_per_hand_keeps_metrics_without_combined_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.run_manifest_sam3_containment as runner

    manifest, windows, _, _ = _write_inputs(tmp_path)
    output_dir = tmp_path / "sam3"

    def write_per_hand(**kwargs):
        target = kwargs["output_dir"] / (
            f"{kwargs['clip_id']}_{kwargs['frame_idx']:06d}.png"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"png")
        return target

    monkeypatch.setattr(runner, "write_overlay_image", write_per_hand)
    runner.run_manifest_sam3_containment(
        manifest=manifest,
        candidate_windows=windows,
        supplier="jdt",
        output_dir=output_dir,
        sam3_model=None,
        overlay_mode="per-hand",
        source_cache=FakeSourceCache(),
        segmenter=FakeSegmenter(),
    )

    rows = json.loads((output_dir / "frame_keypoint_containment.json").read_text())
    assert {row["hand_side"] for row in rows} == {"left", "right"}
    assert all(Path(row["overlay_path"]).exists() for row in rows)
    assert not (output_dir / "combined_overlays").exists()
    evidence = pd.read_csv(output_dir / "review_evidence_manifest.csv")
    assert evidence.empty


def test_left_only_window_combined_overlay_receives_both_hands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.run_manifest_sam3_containment as runner

    manifest, windows, _, _ = _write_inputs(tmp_path)
    pd.DataFrame(
        [
            {
                "asset_id": "jd-range",
                "start_frame": 3,
                "end_frame": 3,
                "hand_side": "left",
            }
        ]
    ).to_parquet(windows, index=False)
    combined_calls: list[set[str]] = []

    def capture_combined_overlay(**kwargs):
        combined_calls.append(set(kwargs["hands"]))
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "combined.png"
        output_path.write_bytes(b"png")
        return output_path

    def write_per_hand(**kwargs):
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "per-hand.png"
        output_path.write_bytes(b"png")
        return output_path

    monkeypatch.setattr(runner, "write_overlay_image", write_per_hand)
    monkeypatch.setattr(
        runner,
        "write_combined_overlay_image",
        capture_combined_overlay,
    )
    output_dir = tmp_path / "sam3"
    runner.run_manifest_sam3_containment(
        manifest=manifest,
        candidate_windows=windows,
        supplier="jdt",
        output_dir=output_dir,
        sam3_model=None,
        write_overlays=True,
        source_cache=FakeSourceCache(),
        segmenter=FakeSegmenter(),
    )

    rows = json.loads(
        (output_dir / "frame_keypoint_containment.json").read_text()
    )
    assert {row["hand_side"] for row in rows} == {"left"}
    assert combined_calls == [{"left", "right"}]


@pytest.mark.parametrize("hand_side", ["left", "right", "both"])
def test_hand_side_selection(hand_side: str, tmp_path: Path) -> None:
    from tools.run_manifest_sam3_containment import run_manifest_sam3_containment

    manifest, windows, _, _ = _write_inputs(tmp_path)
    pd.DataFrame(
        [
            {
                "asset_id": "jd-range",
                "start_frame": 3,
                "end_frame": 3,
                "hand_side": hand_side,
            }
        ]
    ).to_parquet(windows, index=False)
    output_dir = tmp_path / "sam3"
    run_manifest_sam3_containment(
        manifest=manifest,
        candidate_windows=windows,
        supplier="jdt",
        output_dir=output_dir,
        frames_per_window=3,
        sam3_model=None,
        source_cache=FakeSourceCache(),
        segmenter=FakeSegmenter(),
    )
    rows = json.loads(
        (output_dir / "frame_keypoint_containment.json").read_text()
    )

    expected = {"left", "right"} if hand_side == "both" else {hand_side}
    assert {row["hand_side"] for row in rows} == expected


@pytest.mark.parametrize(
    "invalid_points",
    [None, [float("nan")] * 42],
    ids=("missing", "nan"),
)
def test_missing_or_nan_keypoints_produce_projection_review(
    invalid_points: list[float] | None,
    tmp_path: Path,
) -> None:
    from tools.run_manifest_sam3_containment import run_manifest_sam3_containment

    manifest, windows, _, parquet_path = _write_inputs(tmp_path)
    frame = pd.read_parquet(parquet_path)
    frame.at[3, "leftcam_left_kp2d"] = invalid_points
    frame.to_parquet(parquet_path, index=False)
    pd.DataFrame(
        [
            {
                "asset_id": "jd-range",
                "start_frame": 3,
                "end_frame": 3,
                "hand_side": "left",
            }
        ]
    ).to_parquet(windows, index=False)
    output_dir = tmp_path / "sam3"

    run_manifest_sam3_containment(
        manifest=manifest,
        candidate_windows=windows,
        supplier="jdt",
        output_dir=output_dir,
        sam3_model=None,
        source_cache=FakeSourceCache(),
        segmenter=FakeSegmenter(),
    )

    row = json.loads(
        (output_dir / "frame_keypoint_containment.json").read_text()
    )[0]
    assert row["valid_projected_keypoints"] == 0
    assert row["containment_verdict"] == "projection_review"


def test_invalid_window_is_isolated_from_valid_window(tmp_path: Path) -> None:
    from tools.run_manifest_sam3_containment import run_manifest_sam3_containment

    manifest, windows, _, _ = _write_inputs(tmp_path)
    pd.DataFrame(
        [
            {
                "asset_id": "missing-asset",
                "start_frame": 3,
                "end_frame": 4,
                "hand_side": "left",
            },
            {
                "asset_id": "jd-range",
                "start_frame": 3,
                "end_frame": 3,
                "hand_side": "right",
            },
        ]
    ).to_parquet(windows, index=False)
    output_dir = tmp_path / "sam3"

    summary = run_manifest_sam3_containment(
        manifest=manifest,
        candidate_windows=windows,
        supplier="jdt",
        output_dir=output_dir,
        sam3_model=None,
        source_cache=FakeSourceCache(),
        segmenter=FakeSegmenter(),
    )

    assert summary["completed_window_count"] == 1
    assert summary["failed_window_count"] == 1
    failures = json.loads((output_dir / "failures.json").read_text())
    assert failures[0]["asset_id"] == "missing-asset"
    rows = json.loads(
        (output_dir / "frame_keypoint_containment.json").read_text()
    )
    assert {row["asset_id"] for row in rows} == {"jd-range"}


def test_invalid_manifest_row_is_isolated_from_valid_asset(tmp_path: Path) -> None:
    from tools.run_manifest_sam3_containment import run_manifest_sam3_containment

    manifest, windows, _, _ = _write_inputs(tmp_path)
    manifest_rows = pd.read_csv(manifest)
    invalid = manifest_rows.iloc[0].copy()
    invalid["asset_id"] = "invalid-manifest"
    invalid["primary_video_path"] = None
    pd.concat([manifest_rows, invalid.to_frame().T], ignore_index=True).to_csv(
        manifest,
        index=False,
    )
    output_dir = tmp_path / "sam3"

    summary = run_manifest_sam3_containment(
        manifest=manifest,
        candidate_windows=windows,
        supplier="jdt",
        output_dir=output_dir,
        sam3_model=None,
        source_cache=FakeSourceCache(),
        segmenter=FakeSegmenter(),
    )

    assert summary["completed_window_count"] == 1
    assert summary["failed_manifest_row_count"] == 1
    failures = json.loads((output_dir / "failures.json").read_text())
    assert any(
        row["asset_id"] == "invalid-manifest"
        and row["failure_stage"] == "manifest_mapping"
        for row in failures
    )


def test_dry_run_estimates_frames_without_outputs_or_model(tmp_path: Path) -> None:
    from tools.run_manifest_sam3_containment import run_manifest_sam3_containment

    manifest, windows, _, _ = _write_inputs(tmp_path)
    output_dir = tmp_path / "sam3"

    summary = run_manifest_sam3_containment(
        manifest=manifest,
        candidate_windows=windows,
        supplier="jdt",
        output_dir=output_dir,
        frames_per_window=3,
        sam3_model=None,
        dry_run=True,
    )

    assert summary["selected_window_count"] == 1
    assert summary["sampled_source_frame_count"] == 5
    assert summary["hand_frame_evaluation_count"] == 10
    assert not output_dir.exists()


def test_jdt_runner_does_not_require_calibration_or_split_video() -> None:
    source = Path("tools/run_manifest_sam3_containment.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "calibration" not in source.lower()
    assert "camera_forward" not in source
    assert "camera_inverse" not in source
    assert "write_video" not in calls
    assert "VideoWriter" not in source


def test_precheck_still_has_no_sam3_da3_or_vlm_imports() -> None:
    imported: set[str] = set()
    for path in Path("precheck").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.lower() for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.lower())

    assert not {
        name
        for name in imported
        if any(token in name for token in ("sam3", "da3", "qwen", "vlm"))
    }
