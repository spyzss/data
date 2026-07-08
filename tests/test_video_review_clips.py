import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.build_video_review_clips import (
    DEFAULT_FRAME_STRIDE,
    HAND_JOINT_NAMES,
    VIDEO_MANUAL_LABEL_COLUMNS,
    build_clip_rows,
    build_review_index_video_html,
    compute_clip_timing,
    estimate_sampled_frames,
    project_world_keypoints_to_image,
    render_sampled_frames,
    sampled_frame_indices,
)


def test_compute_clip_timing_for_frame_window_clamps_to_asset_duration() -> None:
    timing = compute_clip_timing(
        window_start_frame=120,
        window_end_frame=180,
        representative_frame=None,
        fps=30.0,
        asset_duration_sec=5.0,
        padding_sec=1.0,
    )

    assert timing["start_time_sec"] == pytest.approx(3.0)
    assert timing["duration_sec"] == pytest.approx(2.0)
    assert timing["error"] == ""


def test_compute_clip_timing_for_asset_level_uses_representative_frame() -> None:
    timing = compute_clip_timing(
        window_start_frame=None,
        window_end_frame=None,
        representative_frame=90,
        fps=30.0,
        asset_duration_sec=20.0,
        padding_sec=1.0,
    )

    assert timing["start_time_sec"] == pytest.approx(0.5)
    assert timing["duration_sec"] == pytest.approx(5.0)
    assert timing["error"] == ""


def test_build_clip_rows_joins_manifest_and_review_queue(tmp_path: Path) -> None:
    video_path = tmp_path / "100030_video.mp4"
    manifest = pd.DataFrame(
        [
            {
                "supplier_id": "supplier_a",
                "asset_id": "100030",
                "video_path": str(video_path),
                "fps": 30.0,
                "frame_count": 300,
                "duration_sec": 10.0,
            }
        ]
    )
    review_queue = pd.DataFrame(
        [
            {
                "review_id": "rq_001",
                "supplier_id": "supplier_a",
                "asset_id": "100030",
                "window_start_frame": 120,
                "window_end_frame": 150,
                "representative_frame": 135,
                "auto_verdict": "review",
                "suggested_issue_type": "keypoint_low_quality_window",
                "severity_suggestion": "medium",
                "key_metrics_json": "{\"flagged_frames\": 31}",
                "reason": "test row",
            }
        ]
    )

    rows = build_clip_rows(manifest, review_queue, output_dir=tmp_path, padding_sec=1.0)

    assert len(rows) == 1
    row = rows[0]
    assert row["video_path"] == str(video_path)
    assert row["clip_path"] == str(tmp_path / "clips" / "rq_001.mp4")
    assert row["display_clip_path"] == "clips/rq_001.mp4"
    assert row["clip_start_time_sec"] == pytest.approx(3.0)
    assert row["clip_duration_sec"] == pytest.approx((150 - 120 + 1) / 30.0 + 2.0)
    assert row["clip_error"] == ""
    assert row["frame_stride"] == DEFAULT_FRAME_STRIDE
    assert row["sampled_frame_count"] == 11


def test_video_review_html_contains_video_and_export_button() -> None:
    rows = [
        {
            "review_id": "rq_001",
            "supplier_id": "supplier_a",
            "asset_id": "100030",
            "window_start_frame": 120,
            "window_end_frame": 150,
            "representative_frame": 135,
            "auto_verdict": "review",
            "suggested_issue_type": "keypoint_low_quality_window",
            "severity_suggestion": "medium",
            "key_metrics_json": "{\"flagged_frames\": 31}",
            "reason": "test row",
            "display_clip_path": "clips/rq_001.mp4",
            "clip_start_time_sec": 3.0,
            "clip_duration_sec": 3.033333,
            "clip_error": "",
        }
    ]

    html = build_review_index_video_html(rows)

    assert "<video" in html
    assert "controls" in html
    assert "Export manual_labels.csv" in html
    assert "function exportManualLabelsCsv" in html


def test_sampled_frame_indices_use_stride_and_hard_cap() -> None:
    assert DEFAULT_FRAME_STRIDE == 3
    assert sampled_frame_indices(1, 10, frame_stride=3, max_frames_per_window=100) == [
        1,
        4,
        7,
        10,
    ]
    assert sampled_frame_indices(1, 100, frame_stride=3, max_frames_per_window=5) == [
        1,
        4,
        7,
        10,
        13,
    ]


def test_only_review_ids_filter_limits_rows_before_sampling(tmp_path: Path) -> None:
    manifest = pd.DataFrame(
        [
            {
                "supplier_id": "supplier_a",
                "asset_id": "100030",
                "video_path": str(tmp_path / "100030_video.mp4"),
                "fps": 30.0,
                "frame_count": 300,
                "duration_sec": 10.0,
            }
        ]
    )
    review_queue = pd.DataFrame(
        [
            {
                "review_id": "rq_keep",
                "supplier_id": "supplier_a",
                "asset_id": "100030",
                "window_start_frame": 1,
                "window_end_frame": 9,
            },
            {
                "review_id": "rq_skip",
                "supplier_id": "supplier_a",
                "asset_id": "100030",
                "window_start_frame": 1,
                "window_end_frame": 99,
            },
        ]
    )

    rows = build_clip_rows(
        manifest,
        review_queue,
        output_dir=tmp_path,
        padding_sec=1.0,
        only_review_ids={"rq_keep"},
    )

    assert [row["review_id"] for row in rows] == ["rq_keep"]
    assert rows[0]["sampled_frame_count"] == 4


def test_dry_run_estimate_counts_items_frames_and_storage(tmp_path: Path) -> None:
    rows = [
        {"sampled_frame_count": 2},
        {"sampled_frame_count": 3},
    ]

    estimate = estimate_sampled_frames(rows)

    assert estimate["review_items"] == 2
    assert estimate["sampled_frames"] == 5
    assert estimate["rough_storage_bytes"] > 0


def test_video_review_manual_export_schema_supports_affected_segments() -> None:
    assert VIDEO_MANUAL_LABEL_COLUMNS == [
        "review_id",
        "segment_id",
        "supplier_id",
        "asset_id",
        "window_start_frame",
        "window_end_frame",
        "representative_frame",
        "affected_start_frame",
        "affected_end_frame",
        "auto_verdict",
        "suggested_issue_type",
        "severity_suggestion",
        "key_metrics_json",
        "reason",
        "manual_outcome",
        "failure_mode",
        "severity",
        "confidence",
        "acceptance_status",
        "reviewer",
        "comment",
    ]
    assert "reviewer" in VIDEO_MANUAL_LABEL_COLUMNS


def test_video_review_html_supports_multi_segment_labeling() -> None:
    rows = [
        {
            "review_id": "rq_001",
            "supplier_id": "supplier_a",
            "asset_id": "100030",
            "window_start_frame": 1,
            "window_end_frame": 100,
            "representative_frame": 50,
            "fps": 30.0,
            "auto_verdict": "review",
            "suggested_issue_type": "keypoint_low_quality_window",
            "severity_suggestion": "medium",
            "key_metrics_json": "{\"flagged_frames\": 50}",
            "reason": "large candidate window",
            "display_clip_path": "clips/rq_001.mp4",
            "clip_start_time_sec": 0.0,
            "clip_duration_sec": 4.333333,
            "clip_error": "",
        }
    ]

    html = build_review_index_video_html(rows)

    assert "Add affected segment" in html
    assert "Confirm whole candidate window affected" in html
    assert "Mark whole candidate window false positive" in html
    assert "Mark whole candidate window acceptable" in html
    assert "Defer / needs SAM3 or second review" in html
    assert "Delete segment" in html
    assert "affected_start_frame" in html
    assert "affected_end_frame" in html
    assert "acceptance_status" in html
    assert "Set start = current overlay frame" in html
    assert "Set end = current overlay frame" in html
    assert "currentOverlayFrame" in html
    assert "currentVideoFrame" in html
    assert "segmentsByReviewId" in html
    assert "flatMap" in html
    assert "This page samples every N frames" in html
    assert "Sampled overlay frame carousel" in html
    assert "sampling stride" in html
    assert "current sampled frame index" in html
    assert "original frame_idx" in html
    assert "candidate window start/end" in html
    assert "ArrowLeft" in html
    assert "event.shiftKey ? 10 : 1" in html


def test_video_review_affected_frame_inputs_are_editable() -> None:
    html = build_review_index_video_html([])

    assert "affected_start_frame" in html
    assert "affected_end_frame" in html
    assert "readonly" not in html
    assert "disabled" not in html
    assert "handleManualFieldChange" in html


def test_video_review_html_validates_segment_ranges_and_blocks_invalid_export() -> None:
    html = build_review_index_video_html(
        [
            {
                "review_id": "rq_001",
                "supplier_id": "supplier_a",
                "asset_id": "100030",
                "window_start_frame": 10,
                "window_end_frame": 20,
                "frame_count": 50,
                "display_clip_path": "",
                "clip_error": "missing",
            }
        ]
    )

    assert "function validateSegment" in html
    assert "startFrame < windowStart" in html
    assert "endFrame > windowEnd" in html
    assert "startFrame > endFrame" in html
    assert "segment-invalid" in html
    assert "validateAllSegments()" in html
    assert "Cannot export manual_labels.csv" in html


def test_video_review_html_wraps_carousel_and_supports_frame_jump() -> None:
    html = build_review_index_video_html([])

    assert "((current+delta)%frames.length+frames.length)%frames.length" in html
    assert "Go to original frame" in html
    assert "function jumpToOriginalFrame" in html
    assert "nearestSampledFrameIndex" in html
    assert "clamped to nearest sampled frame" in html
    assert "sampled-frame-jump" in html


def test_video_review_html_action_semantics_clear_or_fill_affected_frames() -> None:
    html = build_review_index_video_html([])

    assert "markWholeWindowFalsePositive" in html
    assert "manual_outcome:'false_positive'" in html
    assert "affected_start_frame:'',affected_end_frame:''" in html
    assert "markWholeWindowAcceptable" in html
    assert "manual_outcome:'acceptable_flagged'" in html
    assert "failure_mode:'acceptable_minor_misalignment'" in html
    assert "confirmWholeWindowAffected" in html
    assert "affected_start_frame:row.window_start_frame,affected_end_frame:row.window_end_frame" in html
    assert "manual_outcome:'true_positive'" in html
    assert "acceptance_status:'rejected'" in html
    assert "markWholeWindowNeedsReview" in html
    assert "manual_outcome:'review'" in html
    assert "acceptance_status:'review'" in html


def test_video_review_html_contains_labeling_help_and_sam3_clarification() -> None:
    html = build_review_index_video_html([])

    assert "False positive = script flagged this window but human confirms there is no real issue." in html
    assert "Acceptable = flagged phenomenon exists but should not reduce usable duration." in html
    assert "Defer = cannot decide from current overlay; needs SAM3, stride=1, video_quality, or second reviewer." in html
    assert "This page shows HDF5 skeleton projection overlay only." in html
    assert "SAM3 mask containment has not been run unless a SAM3 containment input was provided upstream." in html


def test_video_review_html_autosaves_and_loads_saved_progress() -> None:
    html = build_review_index_video_html(
        [
            {
                "review_id": "rq_001",
                "supplier_id": "supplier_a",
                "asset_id": "100030",
                "window_start_frame": 1,
                "window_end_frame": 10,
                "representative_frame": 5,
                "fps": 30.0,
                "auto_verdict": "review",
                "suggested_issue_type": "keypoint_low_quality_window",
                "severity_suggestion": "medium",
                "key_metrics_json": "{}",
                "reason": "test persistence",
                "display_clip_path": "clips/rq_001.mp4",
                "clip_start_time_sec": 0.0,
                "clip_duration_sec": 1.0,
                "clip_error": "",
            }
        ]
    )

    assert "manual_review_state_v3:" in html
    assert "localStorage key" in html
    assert "function autosaveProgress" in html
    assert "document.addEventListener('input', handleManualFieldChange)" in html
    assert "document.addEventListener('change', handleManualFieldChange)" in html
    assert "DOMContentLoaded" in html
    assert "Loaded saved progress from localStorage" in html
    assert "No saved progress found" in html
    assert "Saved at" in html
    assert "renderSegments(rowIndex)" in html
    assert "validateAllSegments()" in html
    assert "updateSampledFrame(rowIndex)" in html


def test_video_review_storage_key_uses_v3_run_label_not_content_hash() -> None:
    html = build_review_index_video_html(
        [
            {
                "review_id": "rq_001",
                "supplier_id": "supplier_a",
                "asset_id": "100030",
                "window_start_frame": 1,
                "window_end_frame": 10,
                "review_run_label": "video_review",
            }
        ]
    )

    assert "manual_review_state_v3:video_review" in html
    assert "manual_review_state_v2" not in html
    assert "sha1" not in html


def test_video_review_html_supports_progress_json_backup() -> None:
    html = build_review_index_video_html([])

    assert "Export progress JSON" in html
    assert "Import progress JSON" in html
    assert "function exportProgressJson" in html
    assert "function importProgressJson" in html
    assert "progress-json-input" in html


def test_video_review_html_uses_global_reviewer_without_per_segment_reviewer_input() -> None:
    html = build_review_index_video_html(
        [
            {
                "review_id": "rq_001",
                "supplier_id": "supplier_a",
                "asset_id": "100030",
                "window_start_frame": 1,
                "window_end_frame": 10,
                "display_clip_path": "",
                "clip_error": "missing",
            }
        ]
    )

    assert "id=\"global-reviewer\"" in html
    assert "value=\"nathan\"" in html
    assert "function globalReviewer" in html
    assert "reviewer:globalReviewer()" in html
    assert "segmentFieldId(rowIndex,segmentIndex,'reviewer')" not in html


def test_hand_joint_names_cover_left_and_right_acceptance_topology() -> None:
    assert len(HAND_JOINT_NAMES) == 42
    assert HAND_JOINT_NAMES[0] == "leftHand"
    assert "rightThumbKnuckle" in HAND_JOINT_NAMES
    assert HAND_JOINT_NAMES[-1] == "rightLittleFingerTip"


def test_project_world_keypoints_to_image_uses_camera_intrinsic_pixels() -> None:
    positions = np.array([[[0.0, 0.0, 1.0], [0.5, 0.25, 2.0]]], dtype=float)
    camera_to_world = np.eye(4, dtype=float)[None, :, :]
    intrinsic = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]])

    uv, z, valid = project_world_keypoints_to_image(
        positions,
        camera_to_world,
        intrinsic,
        image_shape=(120, 160, 3),
    )

    assert uv.shape == (1, 2, 2)
    assert z.tolist() == [[1.0, 2.0]]
    assert uv[0, 0].tolist() == pytest.approx([50.0, 60.0])
    assert uv[0, 1].tolist() == pytest.approx([75.0, 72.5])
    assert valid.tolist() == [[True, True]]


def test_render_sampled_frames_writes_hdf5_skeleton_overlay(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    h5py = pytest.importorskip("h5py")

    video_path = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (160, 120),
    )
    assert writer.isOpened()
    for frame_idx in range(3):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[:, :, 0] = 20 * frame_idx
        writer.write(frame)
    writer.release()

    hdf5_path = tmp_path / "tiny.hdf5"
    with h5py.File(hdf5_path, "w") as h5:
        transforms = h5.create_group("transforms")
        transforms.create_dataset("camera", data=np.repeat(np.eye(4)[None, :, :], 3, axis=0))
        camera = h5.create_group("camera")
        camera.create_dataset(
            "intrinsic",
            data=np.array([[100.0, 0.0, 80.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]]),
        )
        label = h5.create_group("label")
        label.create_dataset("quality_hand", data=np.array([[1.0, 0.9]] * 3))
        for joint_index, joint_name in enumerate(HAND_JOINT_NAMES):
            values = np.repeat(np.eye(4)[None, :, :], 3, axis=0)
            values[:, 0, 3] = ((joint_index % 21) - 10) / 100.0
            values[:, 1, 3] = ((joint_index % 5) - 2) / 100.0
            values[:, 2, 3] = 1.0
            transforms.create_dataset(joint_name, data=values)

    manifest = pd.DataFrame(
        [
            {
                "supplier_id": "supplier_a",
                "asset_id": "asset_a",
                "video_path": str(video_path),
                "hdf5_path": str(hdf5_path),
                "fps": 5.0,
                "frame_count": 3,
                "duration_sec": 0.6,
            }
        ]
    )
    review_queue = pd.DataFrame(
        [
            {
                "review_id": "rq_overlay",
                "supplier_id": "supplier_a",
                "asset_id": "asset_a",
                "window_start_frame": 0,
                "window_end_frame": 0,
                "representative_frame": 0,
                "auto_verdict": "review",
                "suggested_issue_type": "keypoint_low_quality_window",
                "severity_suggestion": "medium",
                "key_metrics_json": "{}",
                "reason": "synthetic overlay",
            }
        ]
    )
    rows = build_clip_rows(
        manifest,
        review_queue,
        output_dir=tmp_path,
        padding_sec=0.0,
        frame_stride=1,
        max_frames_per_window=1,
    )

    render_sampled_frames(rows, overwrite=True, render_overlay=True, jpeg_quality=90)

    frame_path = Path(json.loads(rows[0]["sampled_frames_json"])[0]["frame_path"])
    image = cv2.imread(str(frame_path))
    assert frame_path.exists()
    assert image is not None
    assert int(image.sum()) > 0
    assert rows[0]["sampled_frame_error"] == ""
