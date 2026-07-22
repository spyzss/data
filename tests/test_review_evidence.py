from __future__ import annotations

import json
from pathlib import Path

import pytest

from human_qc.evidence import EvidenceError, EvidenceService
from human_qc.warn_service import WarnReviewService
from human_qc.workbench_service import WorkbenchService
from qc_common.contracts import EvidenceRef, Issue
from qc_pipeline.context import AssetContext
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
            {
                "kind": "skeleton_overlay",
                "path": "evidence/warn-1.png",
                "start_frame": 10,
            },
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
    assert view.overlay_images == (
        {"frame": 10, "url": "/evidence/evidence/warn-1.png"},
    )
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


def test_sampled_overlay_images_are_preserved_in_frame_order(tmp_path: Path) -> None:
    context = _context(tmp_path)
    overlay_dir = tmp_path / "sam3" / "combined_overlays"
    overlay_dir.mkdir(parents=True)
    for frame in (188, 120, 144):
        (overlay_dir / f"frame-{frame}.png").write_bytes(b"png")

    def generate_clip(_command, output: Path) -> None:
        output.write_bytes(b"raw-mp4")

    service = EvidenceService(
        tmp_path / "cache",
        ffmpeg_runner=generate_clip,
    )
    view = service.resolve(
        _issue(
            module="sam3_containment",
            evidence=[
                {
                    "kind": "combined_overlay",
                    "path": "sam3/combined_overlays/frame-188.png",
                    "start_frame": 188,
                },
                {
                    "kind": "overlay",
                    "path": "sam3/combined_overlays/frame-120.png",
                    "start_frame": 120,
                },
                {
                    "kind": "skeleton_overlay",
                    "path": "sam3/combined_overlays/frame-144.png",
                    "start_frame": 144,
                },
            ],
        ),
        context,
    )

    assert [row["frame"] for row in view.overlay_images] == [120, 144, 188]
    assert all(row["url"].endswith(".png") for row in view.overlay_images)
    assert view.clip_url.endswith(".mp4")
    assert "overlay" not in view.clip_url
    assert view.overlay_url == view.overlay_images[0]["url"]


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
            "overlay_images": [],
            "generation_error": None,
        }
    ]


def test_warn_dto_uses_opaque_overlay_handle_without_legacy_evidence_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from human_qc.warn_workbench_service import OverlayHandle
    from tests.test_human_qc_workbench import _service

    service, _, _, _ = _service(tmp_path)
    overlay = tmp_path / "overlays" / "continuous.mp4"
    overlay.parent.mkdir()
    overlay.write_bytes(b"continuous-overlay")
    service.overlay_provider = lambda asset_id, issue_id, frame_range: OverlayHandle(
        status="ready",
        overlay_id="opaque-continuous-1",
        path=overlay,
    ) if issue_id == "warn-2" else None
    monkeypatch.setattr(
        EvidenceService,
        "resolve",
        lambda *_args, **_kwargs: pytest.fail("legacy evidence must not run on task GET"),
    )

    task = service.get_asset_task("asset-1")

    ready = task["issues"][0]["overlay"]
    assert ready == {
        "status": "ready",
        "frame_range": {"start_frame": 0, "end_frame_exclusive": 20},
        "url": "/media/assets/asset-1/overlays/opaque-continuous-1",
        "code": None,
    }
    assert str(overlay) not in json.dumps(task)
    assert service.overlay_media("asset-1", "opaque-continuous-1").path == overlay


def test_asset_overlay_provider_projects_one_union_job_without_sync_rendering(
    tmp_path: Path,
) -> None:
    """A Warn task projects one server-side SAM3 union, never one job per issue."""

    from human_qc.overlay_worker import (
        OverlayJobView,
        OverlayRequest,
        OverlaySegmentView,
    )
    from human_qc.warn_workbench_service import WorkerOverlayProvider
    from tests.test_human_qc_workbench import _service

    class ExplodingRenderer:
        calls = 0

        def render_interval(self, *_args, **_kwargs):
            self.calls += 1
            raise AssertionError("task GET must not render an overlay synchronously")

    class RecordingWorker:
        def __init__(self, segments):
            self.requests = []
            self._segments = segments

        def submit(self, request):
            self.requests.append(request)
            return OverlayJobView(
                cache_key=request.cache_key,
                status="ready",
                segments=self._segments,
            )

    service, _, _, context = _service(tmp_path)
    report = json.loads(context.report_path.read_text(encoding="utf-8"))
    report["issues"] = [
        {
            "issue_id": "sam3-first",
            "code": "first_sam3",
            "module": "sam3_containment",
            "context": {"start_frame": 120, "end_frame": 168},
            "source_path": "/private/first.mp4",
        },
        {
            "issue_id": "sam3-overlap",
            "code": "overlap_sam3",
            "module": "sam3_containment",
            "context": {"start_frame": 142, "end_frame": 181},
            "command": "ffmpeg /private/overlap.mp4",
        },
        {
            "issue_id": "sam3-later",
            "code": "later_sam3",
            "module": "sam3_containment",
            "context": {"start_frame": 390, "end_frame": 426},
        },
        {
            "issue_id": "plain-warn",
            "code": "plain",
            "module": "video_quality",
            "context": {"start_frame": 10, "end_frame": 11},
        },
    ]
    report["manual_review"]["selected_issue_ids"] = [
        "sam3-first",
        "sam3-overlap",
        "sam3-later",
        "plain-warn",
    ]
    report["manual_review"]["candidate_issue_ids"] = list(
        report["manual_review"]["selected_issue_ids"]
    )
    report["manual_review"]["issue_reviews"] = {}
    context.report_path.write_text(json.dumps(report), encoding="utf-8")

    overlay_dir = tmp_path / "overlays"
    overlay_dir.mkdir()
    first_output = overlay_dir / "union-120-182.mp4"
    later_output = overlay_dir / "union-390-427.mp4"
    first_output.write_bytes(b"first")
    later_output.write_bytes(b"later")
    renderer = ExplodingRenderer()
    worker = RecordingWorker(
        (
            OverlaySegmentView(
                start_frame=120,
                end_frame_exclusive=182,
                status="ready",
                overlay_id="overlay-union-120-182",
                path=first_output,
            ),
            OverlaySegmentView(
                start_frame=390,
                end_frame_exclusive=427,
                status="ready",
                overlay_id="overlay-union-390-427",
                path=later_output,
            ),
        )
    )

    def request_factory(asset_id, selected):
        assert [set(item.__dict__) for item in selected] == [
            {"issue_id", "frame_range"},
            {"issue_id", "frame_range"},
            {"issue_id", "frame_range"},
        ]
        assert all("/private/" not in repr(item) for item in selected)
        return OverlayRequest(
            asset_id=asset_id,
            cache_root=tmp_path / ".overlay-cache",
            source_sha256="sha256:" + "a" * 64,
            intervals=tuple(
                (item.frame_range.start_frame, item.frame_range.end_frame_exclusive)
                for item in selected
            ),
            fps=30.0,
            total_frames=1800,
            model_hash="model-v1",
            config_hash="config-v1",
            input_fingerprint_hash="inputs-v1",
            renderer_version="renderer-v1",
            renderer=renderer,
        )

    service.overlay_provider = WorkerOverlayProvider(
        worker=worker,
        request_factory=request_factory,
    )

    task = service.get_asset_task("asset-1")

    assert len(worker.requests) == 1
    request = worker.requests[0]
    assert request.intervals == ((120, 182), (390, 427))
    assert renderer.calls == 0
    shared = {
        "start_frame": 120,
        "end_frame_exclusive": 182,
        "status": "ready",
        "overlay_id": "overlay-union-120-182",
        "url": "/media/assets/asset-1/overlays/overlay-union-120-182",
        "code": None,
        "retryable": False,
    }
    assert task["issues"][0]["overlay"]["segments"] == [shared]
    assert task["issues"][1]["overlay"]["segments"] == [shared]
    assert task["issues"][2]["overlay"]["segments"] == [
        {
            "start_frame": 390,
            "end_frame_exclusive": 427,
            "status": "ready",
            "overlay_id": "overlay-union-390-427",
            "url": "/media/assets/asset-1/overlays/overlay-union-390-427",
            "code": None,
            "retryable": False,
        }
    ]
    assert task["issues"][3]["overlay"] is None
    serialized = json.dumps(task)
    for forbidden in ("/private/", "ffmpeg", "cache_key", "model-v1", "inputs-v1"):
        assert forbidden not in serialized


def test_asset_overlay_projection_waits_for_every_relevant_segment(
    tmp_path: Path,
) -> None:
    from human_qc.warn_workbench_service import (
        FrameRangeDto,
        OverlayHandle,
        OverlaySegmentHandle,
    )
    from tests.test_human_qc_workbench import _service

    class Provider:
        def get_asset_overlays(self, _asset_id, selected):
            assert len(selected) == 1
            return {
                selected[0].issue_id: OverlayHandle(
                    status="generating",
                    segments=(
                        OverlaySegmentHandle(
                            frame_range=FrameRangeDto(120, 182),
                            status="generating",
                            overlay_id="current-part",
                        ),
                        OverlaySegmentHandle(
                            frame_range=FrameRangeDto(390, 427),
                            status="ready",
                            overlay_id="later-part",
                            path=tmp_path / "later.mp4",
                        ),
                    ),
                )
            }

    service, _, _, context = _service(tmp_path)
    (tmp_path / "later.mp4").write_bytes(b"later")
    report = json.loads(context.report_path.read_text(encoding="utf-8"))
    report["issues"] = [
        {
            "issue_id": "sam3-spanning",
            "code": "spanning",
            "module": "sam3_containment",
            "context": {"start_frame": 120, "end_frame": 168},
        }
    ]
    report["manual_review"]["selected_issue_ids"] = ["sam3-spanning"]
    report["manual_review"]["candidate_issue_ids"] = ["sam3-spanning"]
    report["manual_review"]["issue_reviews"] = {}
    context.report_path.write_text(json.dumps(report), encoding="utf-8")
    service.overlay_provider = Provider()

    overlay = service.get_asset_task("asset-1")["issues"][0]["overlay"]

    assert overlay["status"] == "generating"
    assert overlay["url"] is None
    assert overlay["segments"] == [{
        "start_frame": 120,
        "end_frame_exclusive": 182,
        "status": "generating",
        "overlay_id": "current-part",
        "url": None,
        "code": None,
        "retryable": False,
    }]


def test_asset_overlay_segments_must_stay_under_the_asset_batch_root(
    tmp_path: Path,
) -> None:
    from human_qc.media import MediaNotFoundError
    from human_qc.warn_workbench_service import OverlayHandle, OverlaySegmentHandle
    from tests.test_human_qc_workbench import _service

    class Provider:
        def get_asset_overlays(self, _asset_id, selected):
            return {
                selected[0].issue_id: OverlayHandle(
                    status="ready",
                    segments=(
                        OverlaySegmentHandle(
                            frame_range=selected[0].frame_range,
                            status="ready",
                            overlay_id="escaped-overlay",
                            path=tmp_path.parent / "outside.mp4",
                        ),
                    ),
                )
            }

    service, _, _, _ = _service(tmp_path)
    service.overlay_provider = Provider()
    (tmp_path.parent / "outside.mp4").write_bytes(b"outside")

    with pytest.raises(MediaNotFoundError, match="media_not_found"):
        service.get_asset_task("asset-1")


def test_asset_overlay_projection_drops_unsafe_pending_segment_identifier(
    tmp_path: Path,
) -> None:
    from human_qc.warn_workbench_service import OverlayHandle, OverlaySegmentHandle
    from tests.test_human_qc_workbench import _service

    class Provider:
        def get_asset_overlays(self, _asset_id, selected):
            return {
                selected[0].issue_id: OverlayHandle(
                    status="pending",
                    segments=(
                        OverlaySegmentHandle(
                            frame_range=selected[0].frame_range,
                            status="pending",
                            overlay_id="/private/cache-key",
                        ),
                    ),
                )
            }

    service, _, _, _ = _service(tmp_path)
    service.overlay_provider = Provider()

    task = service.get_asset_task("asset-1")

    segment = task["issues"][0]["overlay"]["segments"][0]
    assert segment["overlay_id"] is None
    assert "/private/" not in json.dumps(task)


def test_asset_overlay_requires_complete_segment_coverage_before_ready_or_url(
    tmp_path: Path,
) -> None:
    from human_qc.media import MediaNotFoundError
    from human_qc.warn_workbench_service import (
        FrameRangeDto,
        OverlayHandle,
        OverlaySegmentHandle,
    )
    from tests.test_human_qc_workbench import _service

    partial = tmp_path / "overlays" / "partial.mp4"
    partial.parent.mkdir()
    partial.write_bytes(b"partial")

    class Provider:
        def get_asset_overlays(self, _asset_id, selected):
            return {
                selected[0].issue_id: OverlayHandle(
                    status="ready",
                    segments=(
                        OverlaySegmentHandle(
                            frame_range=FrameRangeDto(120, 150),
                            status="ready",
                            overlay_id="partial-120-150",
                            path=partial,
                        ),
                    ),
                )
            }

    service, _, _, context = _service(tmp_path)
    report = json.loads(context.report_path.read_text(encoding="utf-8"))
    report["issues"] = [
        {
            "issue_id": "sam3-full-window",
            "code": "containment",
            "module": "sam3_containment",
            "context": {"start_frame": 120, "end_frame": 168},
        }
    ]
    report["manual_review"]["selected_issue_ids"] = ["sam3-full-window"]
    report["manual_review"]["candidate_issue_ids"] = ["sam3-full-window"]
    report["manual_review"]["issue_reviews"] = {}
    context.report_path.write_text(json.dumps(report), encoding="utf-8")
    service.overlay_provider = Provider()

    overlay = service.get_asset_task("asset-1")["issues"][0]["overlay"]

    assert overlay["status"] != "ready"
    assert overlay["url"] is None
    assert overlay["segments"][0]["url"] is None
    with pytest.raises(MediaNotFoundError, match="media_not_found"):
        service.overlay_media("asset-1", "partial-120-150")


def test_asset_overlay_accepts_adjacent_ready_segments_that_fully_cover_issue(
    tmp_path: Path,
) -> None:
    from human_qc.warn_workbench_service import (
        FrameRangeDto,
        OverlayHandle,
        OverlaySegmentHandle,
    )
    from tests.test_human_qc_workbench import _service

    first = tmp_path / "overlays" / "first.mp4"
    second = tmp_path / "overlays" / "second.mp4"
    first.parent.mkdir()
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    class Provider:
        def get_asset_overlays(self, _asset_id, selected):
            return {
                selected[0].issue_id: OverlayHandle(
                    status="ready",
                    segments=(
                        OverlaySegmentHandle(
                            frame_range=FrameRangeDto(120, 150),
                            status="ready",
                            overlay_id="first-120-150",
                            path=first,
                        ),
                        OverlaySegmentHandle(
                            frame_range=FrameRangeDto(150, 169),
                            status="ready",
                            overlay_id="second-150-169",
                            path=second,
                        ),
                    ),
                )
            }

    service, _, _, context = _service(tmp_path)
    report = json.loads(context.report_path.read_text(encoding="utf-8"))
    report["issues"] = [
        {
            "issue_id": "sam3-full-window",
            "code": "containment",
            "module": "sam3_containment",
            "context": {"start_frame": 120, "end_frame": 168},
        }
    ]
    report["manual_review"]["selected_issue_ids"] = ["sam3-full-window"]
    report["manual_review"]["candidate_issue_ids"] = ["sam3-full-window"]
    report["manual_review"]["issue_reviews"] = {}
    context.report_path.write_text(json.dumps(report), encoding="utf-8")
    service.overlay_provider = Provider()

    overlay = service.get_asset_task("asset-1")["issues"][0]["overlay"]

    assert overlay["status"] == "ready"
    assert overlay["url"] is None
    assert [segment["url"] for segment in overlay["segments"]] == [
        "/media/assets/asset-1/overlays/first-120-150",
        "/media/assets/asset-1/overlays/second-150-169",
    ]


def test_asset_overlay_provider_accepts_only_explicit_continuous_sam3_contract(
    tmp_path: Path,
) -> None:
    from tests.test_human_qc_workbench import _service

    class Provider:
        def __init__(self) -> None:
            self.selected = []

        def get_asset_overlays(self, _asset_id, selected):
            self.selected.append(selected)
            return {}

    service, _, _, context = _service(tmp_path)
    report = json.loads(context.report_path.read_text(encoding="utf-8"))
    report["issues"] = [
        {
            "issue_id": "continuous-sam3",
            "code": "continuous",
            "module": "sam3_containment",
            "context": {"start_frame": 120, "end_frame": 168},
        },
        {
            "issue_id": "not-continuous",
            "code": "not_continuous",
            "module": "not_sam3_continuous",
            "context": {"start_frame": 200, "end_frame": 220},
        },
        {
            "issue_id": "still-image",
            "code": "still",
            "module": "sam3_containment",
            "evidence_type": "combined_overlay",
            "context": {"start_frame": 300, "end_frame": 320},
        },
    ]
    report["manual_review"]["selected_issue_ids"] = [
        "continuous-sam3",
        "not-continuous",
        "still-image",
    ]
    report["manual_review"]["candidate_issue_ids"] = list(
        report["manual_review"]["selected_issue_ids"]
    )
    report["manual_review"]["issue_reviews"] = {}
    context.report_path.write_text(json.dumps(report), encoding="utf-8")
    provider = Provider()
    service.overlay_provider = provider

    task = service.get_asset_task("asset-1")

    assert [[entry.issue_id for entry in selected] for selected in provider.selected] == [
        ["continuous-sam3"]
    ]
    assert task["issues"][0]["overlay"]["status"] == "pending"
    assert task["issues"][1]["overlay"] is None
    assert task["issues"][2]["overlay"] is None
