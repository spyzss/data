from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
from threading import Barrier, BrokenBarrierError, Event, Lock
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


def _hold_render_fence_while_cpu_bound(
    fence_path: str,
    started: object,
    release: object,
) -> None:
    """Separate-process stand-in for a CPU-bound renderer holding its fence."""

    import fcntl

    descriptor = os.open(fence_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        started.set()
        accumulator = 0
        deadline = time.monotonic() + 3.0
        while not release.is_set() and time.monotonic() < deadline:
            accumulator = (accumulator + 1) % 104729
        assert accumulator >= 0
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


class RecordingRenderer:
    def __init__(self) -> None:
        self.frames: list[int] = []
        self.intervals: list[tuple[int, int]] = []
        self.output_paths: list[Path] = []
        self.calls = 0
        self._lock = Lock()

    def render_interval(
        self,
        request: object,
        start_frame: int,
        end_frame_exclusive: int,
        output_path: Path,
    ) -> dict[str, object]:
        with self._lock:
            self.calls += 1
            self.intervals.append((start_frame, end_frame_exclusive))
            self.output_paths.append(output_path)
            self.frames.extend(range(start_frame, end_frame_exclusive))
        output_path.write_bytes(
            f"{start_frame}:{end_frame_exclusive}".encode("ascii")
        )
        return {
            "frame_count": end_frame_exclusive - start_frame,
            "fps": request.fps,
            "renderer": {"kind": "recording"},
        }


class BlockingRenderer(RecordingRenderer):
    def __init__(self) -> None:
        super().__init__()
        self.started = Event()
        self.release = Event()

    def render_interval(
        self,
        request: object,
        start_frame: int,
        end_frame_exclusive: int,
        output_path: Path,
    ) -> dict[str, object]:
        self.started.set()
        assert self.release.wait(2.0)
        return super().render_interval(request, start_frame, end_frame_exclusive, output_path)


class FailingRenderer(RecordingRenderer):
    def __init__(self) -> None:
        super().__init__()
        self.fail = True

    def render_interval(
        self,
        request: object,
        start_frame: int,
        end_frame_exclusive: int,
        output_path: Path,
    ) -> dict[str, object]:
        if self.fail:
            raise RuntimeError("/private/ffmpeg traceback command=render")
        return super().render_interval(request, start_frame, end_frame_exclusive, output_path)


class LateFailingLargeRenderer(RecordingRenderer):
    """Produce one material file, then fail the later interval."""

    def render_interval(
        self,
        request: object,
        start_frame: int,
        end_frame_exclusive: int,
        output_path: Path,
    ) -> dict[str, object]:
        if self.calls:
            raise RuntimeError("private late render failure")
        with self._lock:
            self.calls += 1
            self.intervals.append((start_frame, end_frame_exclusive))
            self.output_paths.append(output_path)
        output_path.write_bytes(b"x" * 4096)
        return {
            "frame_count": end_frame_exclusive - start_frame,
            "fps": request.fps,
        }


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
        original = ready.segments[0].path.read_bytes()
        ready.segments[0].path.write_bytes(b"x" * len(original))

        # A cache entry must be rejected even if the mutation preserves its byte
        # length.  The fresh submit is deliberately expected to regenerate it
        # rather than returning the corrupt entry as a cache hit.
        repaired_renderer = RecordingRenderer()
        repaired_request = replace(request, renderer=repaired_renderer)
        repaired = second.get(repaired_request)
        assert repaired.status == "failed"
        assert repaired.code == "overlay_cache_invalid"
        regenerated = second.submit(repaired_request)
        assert regenerated.status in {"pending", "generating"}
        assert _wait_until_terminal(second, repaired_request).status == "ready"
        assert repaired_renderer.calls == 2

        manifest_path = request.cache_root / request.cache_key.digest / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        first_segment = manifest["segments"][0]
        assert first_segment["sha256"] == hashlib.sha256(
            (request.cache_root / request.cache_key.digest / first_segment["relative_path"]).read_bytes()
        ).hexdigest()
        assert first_segment["metadata"] == {
            "frame_count": 62,
            "fps": 30.0,
            "renderer": {"kind": "recording"},
        }

        first_segment["metadata"]["fps"] = 15.0
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        invalid_metadata = second.get(repaired_request)
        assert invalid_metadata.status == "failed"
        assert invalid_metadata.code == "overlay_cache_invalid"
        metadata_regenerated = second.submit(repaired_request)
        assert metadata_regenerated.status in {"pending", "generating"}
        assert _wait_until_terminal(second, repaired_request).status == "ready"
        assert repaired_renderer.calls == 4
    finally:
        second.shutdown()


def test_observer_never_recovers_a_live_cross_instance_generation(tmp_path: Path) -> None:
    from human_qc.overlay_worker import BoundedOverlayWorker

    renderer = BlockingRenderer()
    request = _request(tmp_path, renderer=renderer)
    owner = BoundedOverlayWorker(max_workers=1, max_pending=1)
    observer = BoundedOverlayWorker(max_workers=1, max_pending=1)
    try:
        owner.submit(request)
        assert renderer.started.wait(1.0)

        observed = observer.get(request)
        duplicate = observer.submit(request)
        assert observed.status in {"pending", "generating"}
        assert duplicate.status in {"pending", "generating"}
        assert duplicate.deduplicated is True

        manifest_path = request.cache_root / request.cache_key.digest / "manifest.json"
        assert json.loads(manifest_path.read_text(encoding="utf-8"))["status"] in {
            "pending",
            "generating",
        }

        renderer.release.set()
        assert _wait_until_terminal(owner, request).status == "ready"
        assert _wait_until_terminal(observer, request).status == "ready"
        assert renderer.calls == 2
    finally:
        renderer.release.set()
        owner.shutdown()
        observer.shutdown()


def test_owner_lease_heartbeats_keep_a_live_cross_instance_generation_active(
    tmp_path: Path,
) -> None:
    from human_qc.overlay_worker import BoundedOverlayWorker

    renderer = BlockingRenderer()
    request = _request(tmp_path, renderer=renderer)
    owner = BoundedOverlayWorker(
        max_workers=1,
        max_pending=1,
        owner_lease_seconds=0.08,
    )
    observer = BoundedOverlayWorker(
        max_workers=1,
        max_pending=1,
        owner_lease_seconds=0.08,
    )
    try:
        owner.submit(request)
        assert renderer.started.wait(1.0)
        time.sleep(0.20)

        observed = observer.get(request)
        assert observed.status in {"pending", "generating"}

        renderer.release.set()
        assert _wait_until_terminal(owner, request).status == "ready"
    finally:
        renderer.release.set()
        owner.shutdown()
        observer.shutdown()


def test_expired_remote_owner_lease_recovers_as_retryable_interrupted(
    tmp_path: Path,
) -> None:
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
    owner_path = job_dir / ".generation.owner.json"
    owner_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "process_incarnation": "reused-or-remote-process",
                "token": "stale-owner-token",
                "pid": 99999,
                "host": "unreachable-remote-host",
                "heartbeat_at": time.time() - 10.0,
                "lease_expires_at": time.time() - 1.0,
            }
        ),
        encoding="utf-8",
    )
    worker = BoundedOverlayWorker(max_workers=1, max_pending=1, owner_lease_seconds=0.08)
    try:
        recovered = worker.get(request)
        assert recovered.status == "failed"
        assert recovered.code == "overlay_interrupted"
        assert recovered.retryable is True
        assert not owner_path.exists()
    finally:
        worker.shutdown()


def test_owner_marker_with_an_unbounded_future_lease_is_not_live(tmp_path: Path) -> None:
    from human_qc.overlay_worker import BoundedOverlayWorker

    request = _request(tmp_path, renderer=RecordingRenderer())
    job_dir = request.cache_root / request.cache_key.digest
    job_dir.mkdir(parents=True)
    (job_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "key_digest": request.cache_key.digest,
                "status": "pending",
                "segments": [],
            }
        ),
        encoding="utf-8",
    )
    now = time.time()
    (job_dir / ".generation.owner.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "process_incarnation": "unbounded-stale-process",
                "token": "unbounded-owner-token",
                "pid": 99999,
                "host": "unreachable-remote-host",
                "heartbeat_at": now,
                "lease_expires_at": now + 3600.0,
            }
        ),
        encoding="utf-8",
    )
    worker = BoundedOverlayWorker(max_workers=1, max_pending=1)
    try:
        recovered = worker.get(request)
        assert recovered.status == "failed"
        assert recovered.code == "overlay_interrupted"
    finally:
        worker.shutdown()


def test_expired_owner_marker_is_not_recovered_while_a_separate_process_holds_the_render_fence(
    tmp_path: Path,
) -> None:
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
    now = time.time()
    (job_dir / ".generation.owner.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "process_incarnation": "cpu-bound-owner",
                "token": "expired-heartbeat",
                "pid": 99999,
                "host": "unreachable-remote-host",
                "heartbeat_at": now - 10.0,
                "lease_expires_at": now - 1.0,
            }
        ),
        encoding="utf-8",
    )
    process_context = multiprocessing.get_context("spawn")
    started = process_context.Event()
    release = process_context.Event()
    process = process_context.Process(
        target=_hold_render_fence_while_cpu_bound,
        args=(str(job_dir / ".render.fence"), started, release),
    )
    worker = BoundedOverlayWorker(max_workers=1, max_pending=1)
    process.start()
    try:
        assert started.wait(3.0)
        observed = worker.get(request)
        assert observed.status in {"pending", "generating"}
        assert json.loads((job_dir / "manifest.json").read_text(encoding="utf-8"))["status"] == "generating"

        release.set()
        process.join(timeout=3.0)
        assert process.exitcode == 0

        recovered = worker.get(request)
        assert recovered.status == "failed"
        assert recovered.code == "overlay_interrupted"
    finally:
        release.set()
        process.join(timeout=3.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=3.0)
        worker.shutdown()


def test_owner_cleanup_error_still_releases_capacity_and_returns_safe_failure(
    tmp_path: Path,
) -> None:
    from human_qc.overlay_worker import BoundedOverlayWorker

    request = _request(tmp_path, intervals=((20, 30),), renderer=RecordingRenderer())
    next_request = _request(
        tmp_path,
        intervals=((40, 50),),
        renderer=RecordingRenderer(),
        source_sha256="sha256:" + "f" * 64,
    )
    worker = BoundedOverlayWorker(max_workers=1, max_pending=0)
    original_release = worker._release_owner

    def broken_release(request: object, token: str) -> None:
        del request, token
        raise OSError("/private/overlay-owner-cleanup")

    worker._release_owner = broken_release
    try:
        worker.submit(request)
        failed = _wait_until_terminal(worker, request)
        assert failed.status == "failed"
        assert failed.code == "overlay_cleanup_failed"
        assert "/private" not in json.dumps(failed.to_safe_dict())
        assert worker._job_id(request) not in worker._scheduled

        worker._release_owner = original_release
        accepted = worker.submit(next_request)
        assert accepted.status in {"pending", "generating"}
        assert _wait_until_terminal(worker, next_request).status == "ready"
    finally:
        worker._release_owner = original_release
        worker.shutdown()


def test_durable_pin_blocks_other_worker_eviction_until_the_lease_expires(
    tmp_path: Path,
) -> None:
    from human_qc.overlay_worker import BoundedOverlayWorker

    request_a = _request(
        tmp_path,
        intervals=((120, 169),),
        renderer=RecordingRenderer(),
        source_sha256="sha256:" + "1" * 64,
    )
    request_b = _request(
        tmp_path,
        intervals=((220, 269),),
        renderer=RecordingRenderer(),
        source_sha256="sha256:" + "2" * 64,
    )
    worker_a = BoundedOverlayWorker(max_workers=1, max_pending=1, max_ready_jobs=1)
    worker_b = BoundedOverlayWorker(max_workers=1, max_pending=1, max_ready_jobs=1)
    try:
        worker_a.submit(request_a)
        assert _wait_until_terminal(worker_a, request_a).status == "ready"
        worker_a.pin(request_a, lease_seconds=0.10)

        worker_b.submit(request_b)
        blocked = _wait_until_terminal(worker_b, request_b)
        assert blocked.status == "failed"
        assert blocked.code == "overlay_cache_full"
        assert (request_a.cache_root / request_a.cache_key.digest / "manifest.json").is_file()

        time.sleep(0.16)
        retried = worker_b.retry(request_b)
        assert retried.status in {"pending", "generating"}
        assert _wait_until_terminal(worker_b, request_b).status == "ready"
        assert not (request_a.cache_root / request_a.cache_key.digest).exists()
        assert not (request_a.cache_root / ".overlay-pins" / request_a.cache_key.digest).exists()
    finally:
        worker_a.shutdown()
        worker_b.shutdown()


def test_corrupt_durable_pins_do_not_block_cross_worker_eviction(tmp_path: Path) -> None:
    from human_qc.overlay_worker import BoundedOverlayWorker

    request_a = _request(
        tmp_path,
        intervals=((120, 169),),
        renderer=RecordingRenderer(),
        source_sha256="sha256:" + "3" * 64,
    )
    request_b = _request(
        tmp_path,
        intervals=((220, 269),),
        renderer=RecordingRenderer(),
        source_sha256="sha256:" + "4" * 64,
    )
    worker_a = BoundedOverlayWorker(max_workers=1, max_pending=1, max_ready_jobs=1)
    worker_b = BoundedOverlayWorker(max_workers=1, max_pending=1, max_ready_jobs=1)
    try:
        worker_a.submit(request_a)
        assert _wait_until_terminal(worker_a, request_a).status == "ready"
        pin_directory = request_a.cache_root / ".overlay-pins" / request_a.cache_key.digest
        pin_directory.mkdir(parents=True)
        now = time.time()
        (pin_directory / "far-future.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "key_digest": request_a.cache_key.digest,
                    "process_incarnation": "otherwise-valid-owner",
                    "token": "far-future-token",
                    "expires_at": now + 365 * 24 * 60 * 60,
                }
            ),
            encoding="utf-8",
        )
        (pin_directory / "missing-owner.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "key_digest": request_a.cache_key.digest,
                    "token": "missing-owner-token",
                    "expires_at": now + 60.0,
                }
            ),
            encoding="utf-8",
        )

        worker_b.submit(request_b)
        assert _wait_until_terminal(worker_b, request_b).status == "ready"
        assert not (request_a.cache_root / request_a.cache_key.digest).exists()
        assert not pin_directory.exists()
    finally:
        worker_a.shutdown()
        worker_b.shutdown()


def test_failed_multi_segment_jobs_delete_media_and_remain_inside_batch_quota(
    tmp_path: Path,
) -> None:
    from human_qc.overlay_worker import BoundedOverlayWorker

    worker = BoundedOverlayWorker(
        max_workers=1,
        max_pending=1,
        max_cache_bytes=2048,
    )
    requests = [
        _request(
            tmp_path,
            intervals=((0, 1), (2, 3)),
            renderer=LateFailingLargeRenderer(),
            source_sha256="sha256:" + f"{index:064x}",
        )
        for index in range(1, 8)
    ]
    try:
        for request in requests:
            worker.submit(request)
            failed = _wait_until_terminal(worker, request)
            assert failed.status == "failed"
            assert all(segment.path is None for segment in failed.segments)
            assert not list(request.cache_root.rglob("*.mp4"))
        persisted_bytes = sum(
            path.stat().st_size
            for path in requests[0].cache_root.rglob("*")
            if path.is_file()
        )
        assert persisted_bytes <= 2048
    finally:
        worker.shutdown()


def test_distinct_concurrent_jobs_publish_within_the_shared_ready_cache_limit(
    tmp_path: Path,
) -> None:
    from human_qc.overlay_worker import BoundedOverlayWorker

    renderer_a = RecordingRenderer()
    renderer_b = RecordingRenderer()
    request_a = _request(
        tmp_path,
        intervals=((120, 169),),
        renderer=renderer_a,
        source_sha256="sha256:" + "d" * 64,
    )
    request_b = _request(
        tmp_path,
        intervals=((220, 269),),
        renderer=renderer_b,
        source_sha256="sha256:" + "e" * 64,
    )
    worker_a = BoundedOverlayWorker(max_workers=1, max_pending=1, max_ready_jobs=1)
    worker_b = BoundedOverlayWorker(max_workers=1, max_pending=1, max_ready_jobs=1)
    rendezvous = Barrier(2)

    def gate_eviction(worker: object) -> None:
        original = worker._evict_for

        def coordinated(request: object, size: int) -> bool:
            result = original(request, size)
            try:
                rendezvous.wait(timeout=0.25)
            except BrokenBarrierError:
                # With the root-wide publication lock, only one publisher enters
                # at a time; the barrier timing out is the expected safe path.
                pass
            return result

        worker._evict_for = coordinated

    gate_eviction(worker_a)
    gate_eviction(worker_b)
    try:
        worker_a.submit(request_a)
        worker_b.submit(request_b)
        deadline = time.monotonic() + 3.0
        while worker_a._scheduled or worker_b._scheduled:
            assert time.monotonic() < deadline, "concurrent jobs did not finish"
            time.sleep(0.01)
        assert renderer_a.calls == 1
        assert renderer_b.calls == 1

        manifests = list(request_a.cache_root.glob("*/manifest.json"))
        ready_count = sum(
            json.loads(path.read_text(encoding="utf-8")).get("status") == "ready"
            for path in manifests
        )
        assert ready_count == 1
    finally:
        worker_a.shutdown()
        worker_b.shutdown()
