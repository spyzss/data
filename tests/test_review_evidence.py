from __future__ import annotations

from pathlib import Path

import pytest

from human_qc.evidence import EvidenceError, EvidenceService
from qc_pipeline.context import AssetContext


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
    assert "renderer unavailable" in (view.generation_error or "")


def test_evidence_path_escape_is_rejected(tmp_path: Path) -> None:
    context = _context(tmp_path)
    outside = tmp_path.parent / "outside.mp4"
    outside.write_bytes(b"outside")
    service = EvidenceService(cache_root=tmp_path / "cache")
    with pytest.raises(EvidenceError, match="inside|contain"):
        service.resolve(
            _issue(evidence=[{"kind": "clip", "path": "../outside.mp4"}]), context
        )
