from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from threading import Event, Lock
import time

import pytest

def _request(
    tmp_path: Path,
    *,
    intervals: tuple[tuple[int, int], ...] = ((120, 169), (142, 182), (390, 427)),
    renderer: object | None = None,
    source_sha256: str = "sha256:" + "a" * 64,
    model_hash: str = "model-v1",
    config_hash: str = "config-v1",
    input_fingerprint_hash: str = "inputs-v1",
    renderer_version: str = "renderer-v1",
):
    from human_qc.overlay_worker import OverlayRequest

    return OverlayRequest(
        asset_id="asset-1",
        cache_root=tmp_path / ".human_qc_overlay_cache",
        source_sha256=source_sha256,
        intervals=intervals,
        fps=30.0,
        total_frames=500,
        model_hash=model_hash,
        config_hash=config_hash,
        input_fingerprint_hash=input_fingerprint_hash,
        renderer_version=renderer_version,
        renderer=renderer,
    )


def _wait_until_terminal(worker: object, request: object, *, timeout: float = 3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        view = worker.get(request)
        if view.status in {"ready", "failed"}:
            return view
        time.sleep(0.01)
    pytest.fail("overlay job did not become terminal")


class RecordingRenderer:
    def __init__(self) -> None:
        self.frames: list[int] = []
        self.intervals: list[tuple[int, int]] = []
        self.output_paths: list[Path] = []
        self.calls = 0
        self._lock = Lock()

    def render_interval(self, request: object, start_frame: int, end_frame_exclusive: int, output_path: Path) -> None:
        del request
        with self._lock:
            self.calls += 1
            self.intervals.append((start_frame, end_frame_exclusive))
            self.output_paths.append(output_path)
            self.frames.extend(range(start_frame, end_frame_exclusive))
        output_path.write_bytes(
            f"{start_frame}:{end_frame_exclusive}".encode("ascii")
        )


class BlockingRenderer(RecordingRenderer):
    def __init__(self) -> None:
        super().__init__()
        self.started = Event()
        self.release = Event()

    def render_interval(self, request: object, start_frame: int, end_frame_exclusive: int, output_path: Path) -> None:
        self.started.set()
        assert self.release.wait(2.0)
        super().render_interval(request, start_frame, end_frame_exclusive, output_path)


class FailingRenderer(RecordingRenderer):
    def __init__(self) -> None:
        super().__init__()
        self.fail = True

    def render_interval(self, request: object, start_frame: int, end_frame_exclusive: int, output_path: Path) -> None:
        if self.fail:
            raise RuntimeError("/private/ffmpeg traceback command=render")
        super().render_interval(request, start_frame, end_frame_exclusive, output_path)


def test_merge_frame_intervals_merges_overlap_and_adjacency_but_rejects_invalid_ranges() -> None:
    from human_qc.overlay_worker import merge_frame_intervals

    assert merge_frame_intervals(((120, 169), (142, 182), (390, 427))) == (
        (120, 182),
        (390, 427),
    )
    assert merge_frame_intervals(((7, 9), (0, 3), (3, 7))) == ((0, 9),)

    for invalid in (((1, 1),), ((-1, 2),), ((True, 2),), ((1, False),)):
        with pytest.raises((TypeError, ValueError)):
            merge_frame_intervals(invalid)


def test_worker_renders_each_union_frame_once_and_hits_only_the_full_identity_cache(
    tmp_path: Path,
) -> None:
    from human_qc.overlay_worker import BoundedOverlayWorker

    renderer = RecordingRenderer()
    request = _request(tmp_path, renderer=renderer)
    worker = BoundedOverlayWorker(max_workers=1, max_pending=1)
    try:
        first = worker.submit(request)
        assert first.status in {"pending", "generating"}
        ready = _wait_until_terminal(worker, request)
        assert ready.status == "ready"
        assert ready.cache_hit is False
        assert renderer.frames == list(range(120, 182)) + list(range(390, 427))
        assert all(path.suffix == ".mp4" for path in renderer.output_paths)
        assert [
            (segment.start_frame, segment.end_frame_exclusive)
            for segment in ready.segments
        ] == [(120, 182), (390, 427)]
        assert all(segment.path is not None and segment.path.is_file() for segment in ready.segments)

        cached = worker.submit(request)
        assert cached.status == "ready"
        assert cached.cache_hit is True
        assert renderer.calls == 2

        for changed in (
            replace(request, model_hash="model-v2"),
            replace(request, config_hash="config-v2"),
            replace(request, input_fingerprint_hash="inputs-v2"),
            replace(request, renderer_version="renderer-v2"),
            replace(request, source_sha256="sha256:" + "b" * 64),
        ):
            assert changed.cache_key != request.cache_key
            worker.submit(changed)
            assert _wait_until_terminal(worker, changed).status == "ready"
        assert renderer.calls == 12
    finally:
        worker.shutdown()


def test_worker_singleflights_identical_requests_and_returns_queue_full_without_blocking(
    tmp_path: Path,
) -> None:
    from human_qc.overlay_worker import BoundedOverlayWorker

    renderer = BlockingRenderer()
    request = _request(tmp_path, renderer=renderer)
    other = _request(
        tmp_path,
        renderer=RecordingRenderer(),
        source_sha256="sha256:" + "c" * 64,
    )
    worker = BoundedOverlayWorker(max_workers=1, max_pending=0)
    try:
        first = worker.submit(request)
        assert renderer.started.wait(1.0)
        duplicate = worker.submit(request)
        queue_full = worker.submit(other)

        assert first.status in {"pending", "generating"}
        assert duplicate.deduplicated is True
        assert queue_full.status == "failed"
        assert queue_full.code == "overlay_queue_full"
        assert queue_full.retryable is True
        assert renderer.calls == 0

        renderer.release.set()
        assert _wait_until_terminal(worker, request).status == "ready"
        assert renderer.calls == 2
    finally:
        renderer.release.set()
        worker.shutdown()


def test_worker_publishes_stable_failure_then_retries_without_leaking_renderer_details(
    tmp_path: Path,
) -> None:
    from human_qc.overlay_worker import BoundedOverlayWorker

    renderer = FailingRenderer()
    request = _request(tmp_path, renderer=renderer)
    worker = BoundedOverlayWorker(max_workers=1, max_pending=1)
    try:
        worker.submit(request)
        failed = _wait_until_terminal(worker, request)
        assert failed.status == "failed"
        assert failed.code == "overlay_render_failed"
        assert failed.retryable is True
        assert "/private" not in json.dumps(failed.to_safe_dict())
        assert "ffmpeg" not in json.dumps(failed.to_safe_dict())
        assert "traceback" not in json.dumps(failed.to_safe_dict())
        assert not any(segment.status == "ready" for segment in failed.segments)

        renderer.fail = False
        retried = worker.retry(request)
        assert retried.status in {"pending", "generating"}
        assert _wait_until_terminal(worker, request).status == "ready"
    finally:
        worker.shutdown()


def test_worker_recovers_interrupted_manifest_as_retryable_failure(tmp_path: Path) -> None:
    from human_qc.overlay_worker import BoundedOverlayWorker

    request = _request(tmp_path, renderer=RecordingRenderer())
    job_dir = request.cache_root / request.cache_key.digest
    job_dir.mkdir(parents=True)
    (job_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "key_digest": request.cache_key.digest,
                "status": "generating",
                "segments": [],
            }
        ),
        encoding="utf-8",
    )
    worker = BoundedOverlayWorker(max_workers=1, max_pending=1)
    try:
        recovered = worker.get(request)
        assert recovered.status == "failed"
        assert recovered.code == "overlay_interrupted"
        assert recovered.retryable is True
    finally:
        worker.shutdown()


def test_worker_reports_cache_full_without_publishing_a_ready_manifest(tmp_path: Path) -> None:
    from human_qc.overlay_worker import BoundedOverlayWorker

    request = _request(tmp_path, renderer=RecordingRenderer())
    worker = BoundedOverlayWorker(max_workers=1, max_pending=1, max_cache_bytes=1)
    try:
        worker.submit(request)
        failed = _wait_until_terminal(worker, request)

        assert failed.status == "failed"
        assert failed.code == "overlay_cache_full"
        manifest = request.cache_root / request.cache_key.digest / "manifest.json"
        assert json.loads(manifest.read_text(encoding="utf-8"))["status"] == "failed"
        assert not any(segment.status == "ready" for segment in failed.segments)
    finally:
        worker.shutdown()


def test_worker_validates_a_durable_ready_manifest_before_a_fresh_cache_hit(
    tmp_path: Path,
) -> None:
    from human_qc.overlay_worker import BoundedOverlayWorker

    first_renderer = RecordingRenderer()
    request = _request(tmp_path, renderer=first_renderer)
    first = BoundedOverlayWorker(max_workers=1, max_pending=1)
    try:
        first.submit(request)
        ready = _wait_until_terminal(first, request)
        assert ready.status == "ready"
    finally:
        first.shutdown()

    second_renderer = FailingRenderer()
    cached_request = replace(request, renderer=second_renderer)
    second = BoundedOverlayWorker(max_workers=1, max_pending=1)
    try:
        cached = second.submit(cached_request)
        assert cached.status == "ready"
        assert cached.cache_hit is True
        assert second_renderer.calls == 0

        assert ready.segments[0].path is not None
        ready.segments[0].path.write_bytes(b"tampered")
        invalid = BoundedOverlayWorker(max_workers=1, max_pending=1)
        try:
            repaired = invalid.get(cached_request)
            assert repaired.status == "failed"
            assert repaired.code == "overlay_cache_invalid"
            assert repaired.retryable is True
        finally:
            invalid.shutdown()
    finally:
        second.shutdown()
