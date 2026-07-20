from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from human_qc.evidence import EvidenceError, EvidenceService
from human_qc.warn_service import WarnReviewService
from human_qc.workbench_service import WorkbenchService
from qc_common.contracts import EvidenceRef, Issue
from qc_pipeline.context import AssetContext
from tests.fixtures import solid_frame, write_test_video
from tests.qc_report_fixtures import make_v2_report


ASSET_ID = "617856"


def _context(tmp_path: Path) -> AssetContext:
    video = tmp_path / "video" / "clip.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video-bytes")
    return AssetContext(
        asset_id=ASSET_ID,
        batch_root=tmp_path,
        report_path=tmp_path / "quality_archive" / f"{ASSET_ID}.json",
        source_files={"video": {"path": "video/clip.mp4"}},
        metadata={"fps": 30},
    )


def _issue(**overrides):
    value = {
        "issue_id": "warn-1",
        "start_frame": 10,
        "end_frame": 20,
        "context": {},
    }
    value.update(overrides)
    return value


def _context_with_video_and_hdf5(
    tmp_path: Path, *, frame_count: int, fps: float
) -> AssetContext:
    video = tmp_path / "video" / "clip.mp4"
    write_test_video(
        video,
        [solid_frame(30 + frame_index, width=64, height=48) for frame_index in range(frame_count)],
        fps=fps,
    )
    hdf5_path = tmp_path / "source" / "clip.hdf5"
    hdf5_path.parent.mkdir(parents=True)
    base_names = (
        "Hand",
        "ThumbKnuckle",
        "ThumbIntermediateBase",
        "ThumbIntermediateTip",
        "ThumbTip",
        "IndexFingerKnuckle",
        "IndexFingerIntermediateBase",
        "IndexFingerIntermediateTip",
        "IndexFingerTip",
        "MiddleFingerKnuckle",
        "MiddleFingerIntermediateBase",
        "MiddleFingerIntermediateTip",
        "MiddleFingerTip",
        "RingFingerKnuckle",
        "RingFingerIntermediateBase",
        "RingFingerIntermediateTip",
        "RingFingerTip",
        "LittleFingerKnuckle",
        "LittleFingerIntermediateBase",
        "LittleFingerIntermediateTip",
        "LittleFingerTip",
    )
    with h5py.File(hdf5_path, "w") as handle:
        camera = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], frame_count, axis=0)
        handle.create_dataset("transforms/camera", data=camera)
        handle.create_dataset(
            "camera/intrinsic",
            data=np.array([[40.0, 0.0, 32.0], [0.0, 40.0, 24.0], [0.0, 0.0, 1.0]]),
        )
        for hand_index, side in enumerate(("left", "right")):
            for joint_index, name in enumerate(base_names):
                transforms = camera.copy()
                transforms[:, 0, 3] = (-0.45 if hand_index == 0 else 0.15) + 0.025 * joint_index
                transforms[:, 1, 3] = -0.2 + 0.02 * (joint_index % 5)
                transforms[:, 2, 3] = 2.0
                handle.create_dataset(f"transforms/{side}{name}", data=transforms)
    return AssetContext(
        asset_id=ASSET_ID,
        batch_root=tmp_path,
        report_path=tmp_path / "quality_archive" / f"{ASSET_ID}.json",
        source_files={
            "video": {"path": video.relative_to(tmp_path).as_posix()},
            "hdf5": {"path": hdf5_path.relative_to(tmp_path).as_posix()},
        },
        source_range=(0, frame_count),
        metadata={"fps": fps},
    )


def test_evidence_module_is_missing_before_implementation() -> None:
    assert EvidenceService is not None


def test_existing_clip_and_overlay_are_reused(tmp_path: Path) -> None:
    context = _context(tmp_path)
    clip = tmp_path / "evidence" / "warn-1.mp4"
    overlay = tmp_path / "evidence" / "warn-1.png"
    clip.parent.mkdir()
    clip.write_bytes(b"clip")
    overlay.write_bytes(b"overlay")
    issue = _issue(
        evidence=[
            {"kind": "clip", "path": "evidence/warn-1.mp4"},
            {"kind": "skeleton_overlay", "path": "evidence/warn-1.png"},
        ]
    )
    calls = []
    service = EvidenceService(
        cache_root=tmp_path / "cache",
        ffmpeg_runner=lambda command, output: calls.append(command),
    )
    view = service.resolve(issue, context)
    assert view.clip_url.endswith("evidence/warn-1.mp4")
    assert view.overlay_url is not None and view.overlay_url.endswith("evidence/warn-1.png")
    assert calls == []


def test_missing_clip_uses_half_open_window_and_cache_key(tmp_path: Path) -> None:
    context = _context(tmp_path)
    calls: list[list[str]] = []

    def run(command, output: Path) -> None:
        calls.append(list(command))
        output.write_bytes(b"generated")

    service = EvidenceService(cache_root=tmp_path / "cache", ffmpeg_runner=run)
    issue = _issue()
    first = service.resolve(issue, context)
    second = service.resolve(issue, context)
    assert first.clip_url == second.clip_url
    assert len(calls) == 1
    command = " ".join(calls[0])
    assert "start_frame=10" in command
    assert "end_frame=20" in command
    assert "warn-1" in first.clip_url
    assert "10-20" in first.clip_url


def test_generated_clip_temporary_path_keeps_mp4_suffix(tmp_path: Path) -> None:
    context = _context(tmp_path)
    issue = _issue()
    observed: dict[str, Path] = {}

    def generate(command, output: Path) -> None:
        observed["command_output"] = Path(command[-1])
        observed["callback_output"] = output
        output.write_bytes(b"mp4")

    service = EvidenceService(tmp_path / "cache", ffmpeg_runner=generate)
    view = service.resolve(issue, context)

    assert observed["command_output"].suffix == ".mp4"
    assert observed["callback_output"].suffix == ".mp4"
    assert view.clip_url.endswith(".mp4")


def test_overlay_reads_only_window_and_limits_skeleton_points(tmp_path: Path) -> None:
    context = _context(tmp_path)
    captured = {}

    def run(command, output: Path) -> None:
        output.write_bytes(b"clip")

    def render(issue, output: Path, start: int, end: int) -> None:
        captured.update({"issue": issue, "start": start, "end": end})
        output.write_bytes(b"overlay")

    skeleton = {
        "frames": [
            {"points": [[point, point + 1] for point in range(25)]}
            for _ in range(30)
        ]
    }
    service = EvidenceService(
        cache_root=tmp_path / "cache",
        ffmpeg_runner=run,
        overlay_renderer=render,
    )
    view = service.resolve(_issue(skeleton=skeleton), context)
    assert view.overlay_url is not None
    assert captured["start"] == 10 and captured["end"] == 20
    assert len(captured["issue"]["skeleton"]["frames"]) == 10
    assert all(len(frame["points"]) == 21 for frame in captured["issue"]["skeleton"]["frames"])


def test_overlay_failure_keeps_clip_and_reports_generation_error(tmp_path: Path) -> None:
    context = _context(tmp_path)

    def run(command, output: Path) -> None:
        output.write_bytes(b"clip")

    def fail(*args, **kwargs):
        raise RuntimeError("renderer unavailable")

    service = EvidenceService(
        cache_root=tmp_path / "cache",
        ffmpeg_runner=run,
        overlay_renderer=fail,
    )
    view = service.resolve(_issue(skeleton={"frames": []}), context)
    assert view.clip_url
    assert view.overlay_url is None
    assert view.generation_error == "overlay_unavailable"
    assert "renderer unavailable" not in json.dumps(view.__dict__)


def test_sam3_review_clip_overlays_only_the_issue_interval(tmp_path: Path) -> None:
    from human_qc.skeleton_overlay_video import render_skeleton_overlay_video

    context = _context_with_video_and_hdf5(tmp_path, frame_count=12, fps=6)
    visited: list[int] = []

    render_skeleton_overlay_video(
        {"module": "sam3_containment", "issue_id": "sam3-1"},
        context,
        tmp_path / "overlay.mp4",
        3,
        8,
        on_frame=visited.append,
    )

    assert visited == [3, 4, 5, 6, 7]
    assert (tmp_path / "overlay.mp4").is_file()


def test_sam3_review_clip_prefers_generated_overlay_video(tmp_path: Path) -> None:
    context = _context(tmp_path)
    observed: dict[str, object] = {}

    def generate_clip(_command, output: Path) -> None:
        output.write_bytes(b"raw-mp4")

    def render_overlay_video(issue, renderer_context, output: Path, start: int, end: int) -> None:
        observed.update(issue=issue, context=renderer_context, start=start, end=end, output=output)
        output.write_bytes(b"overlay-mp4")

    service = EvidenceService(
        tmp_path / "cache",
        ffmpeg_runner=generate_clip,
        overlay_video_renderer=render_overlay_video,
    )
    view = service.resolve(_issue(module="sam3_containment"), context)

    assert view.clip_url.endswith(".mp4")
    assert "overlay" in view.clip_url
    assert view.generation_error is None
    assert observed["context"] is context
    assert (observed["start"], observed["end"]) == (10, 20)


def test_sam3_review_clip_falls_back_to_raw_clip_when_overlay_fails(tmp_path: Path) -> None:
    context = _context(tmp_path)

    def generate_clip(_command, output: Path) -> None:
        output.write_bytes(b"raw-mp4")

    def fail_overlay_video(*_args: object) -> None:
        raise RuntimeError("private renderer command and path")

    service = EvidenceService(
        tmp_path / "cache",
        ffmpeg_runner=generate_clip,
        overlay_video_renderer=fail_overlay_video,
    )
    view = service.resolve(_issue(module="sam3_containment"), context)

    assert "overlay" not in view.clip_url
    assert view.clip_url.endswith(".mp4")
    assert view.generation_error == "overlay_unavailable"


def test_evidence_path_escape_is_rejected(tmp_path: Path) -> None:
    context = _context(tmp_path)
    outside = tmp_path.parent / "outside.mp4"
    outside.write_bytes(b"outside")
    service = EvidenceService(cache_root=tmp_path / "cache")
    with pytest.raises(EvidenceError, match="inside|contain"):
        service.resolve(
            _issue(evidence=[{"kind": "clip", "path": "../outside.mp4"}]), context
        )


def test_workbench_joins_canonical_issue_evidence_and_converts_inclusive_context(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    clip = tmp_path / "evidence" / "warn-1.mp4"
    clip.parent.mkdir()
    clip.write_bytes(b"canonical-clip")
    issue = Issue(
        issue_id="warn-1",
        code="blur",
        severity="warn",
        module="video_quality",
        issue_type="metric_threshold",
        metric="blur_score",
        observed_value=0.2,
        operator="<",
        boundary_value=0.5,
        rule_id="video_quality.blur",
        needs_manual_review=True,
        context={"start_frame": 10, "end_frame": 19},
        evidence_ids=("evidence-1",),
    ).to_dict()
    evidence = EvidenceRef(
        evidence_id="evidence-1",
        kind="clip",
        path="evidence/warn-1.mp4",
        coordinate_system="source_frame_inclusive",
        start_frame=10,
        end_frame=19,
    ).to_dict()
    report = make_v2_report(status="awaiting_external")
    report["asset_id"] = ASSET_ID
    report["pipeline_state"].update(
        {
            "last_completed_module": "semantic_consistency",
            "next_module": "manual_review",
            "stop_reason": None,
        }
    )
    report["semantic_calibration"] = {
        "state": "completed",
        "source_dataset_path": "/label/subtask_label",
        "base_hdf5_sha256": "sha256:" + "a" * 64,
        "final_hdf5_sha256": "sha256:" + "b" * 64,
        "timeline_edit_count": 0,
        "subtask_text_edit_count": 0,
        "pending_edit": None,
        "audit": [],
    }
    report["issues"] = [issue]
    report["manual_review"].update(
        {
            "required": True,
            "state": "queued",
            "candidate_issue_ids": ["warn-1"],
            "selected_issue_ids": ["warn-1"],
            "selected_issue_id": "warn-1",
            "issue_reviews": {},
            "completed_at": None,
        }
    )
    report["video_quality"] = {"evidence": [evidence]}
    context.report_path.parent.mkdir(parents=True)
    context.report_path.write_text(json.dumps(report), encoding="utf-8")
    service = WorkbenchService(
        warn_service=WarnReviewService(reports={ASSET_ID: context.report_path}),
        evidence_service=EvidenceService(
            tmp_path / "cache",
            ffmpeg_runner=lambda *_: pytest.fail("canonical clip should be reused"),
        ),
        asset_contexts={ASSET_ID: context},
    )

    task = service.get_asset_task(ASSET_ID)

    assert task["evidence"] == [
        {
            "issue_id": "warn-1",
            "start_frame": 10,
            "end_frame_exclusive": 20,
            "clip_url": "/evidence/evidence/warn-1.mp4",
            "overlay_url": None,
            "generation_error": None,
        }
    ]
