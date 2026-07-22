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


def _regular_file_bytes(path: Path) -> int:
    return sum(
        candidate.stat().st_size
        for candidate in path.rglob("*")
        if candidate.is_file()
    )


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


def test_public_interrupted_recovery_respects_tiny_quota_without_accumulating_job_dirs(
    tmp_path: Path,
) -> None:
    from human_qc.overlay_worker import BoundedOverlayWorker

    requests = [
        (
            _request(
                tmp_path,
                intervals=((index, index + 1),),
                renderer=RecordingRenderer(),
                source_sha256="sha256:" + f"{index:064x}",
            ),
            status,
        )
        for index, status in enumerate(("pending", "generating"), start=1)
    ]
    for request, status in requests:
        job_dir = request.cache_root / request.cache_key.digest
        job_dir.mkdir(parents=True)
        (job_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "key_digest": request.cache_key.digest,
                    "status": status,
                    "segments": [],
                }
            ),
            encoding="utf-8",
        )

    worker = BoundedOverlayWorker(max_workers=1, max_pending=1, max_cache_bytes=1)
    try:
        recovered = [worker.get(request) for request, _status in requests]

        assert [view.status for view in recovered] == ["failed", "failed"]
        assert [view.code for view in recovered] == [
            "overlay_interrupted",
            "overlay_interrupted",
        ]
        assert all(view.retryable for view in recovered)
        assert all(worker.get(request).code == "overlay_interrupted" for request, _ in requests)
        assert not [
            path
            for path in requests[0][0].cache_root.glob("*")
            if path.is_dir() and len(path.name) == 64
        ]
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
        assert not any(segment.status == "ready" for segment in failed.segments)
        assert worker.get(request).code == "overlay_cache_full"
        assert not (request.cache_root / request.cache_key.digest).exists()
    finally:
        worker.shutdown()


def test_ready_publication_accounts_for_the_larger_final_manifest_before_quota(
    tmp_path: Path,
) -> None:
    from human_qc.overlay_worker import BoundedOverlayWorker

    max_cache_bytes = 1600
    request = _request(
        tmp_path,
        intervals=((0, 1), (2, 3), (4, 5), (6, 7), (8, 9)),
        renderer=RecordingRenderer(),
    )
    worker = BoundedOverlayWorker(
        max_workers=1,
        max_pending=0,
        max_cache_bytes=max_cache_bytes,
    )
    try:
        worker.submit(request)
        terminal = _wait_until_terminal(worker, request)

        assert terminal.status == "failed"
        assert terminal.code == "overlay_cache_full"
        assert _regular_file_bytes(request.cache_root) <= max_cache_bytes
    finally:
        worker.shutdown()


def test_job_footprint_counts_every_regular_file_and_can_replace_the_manifest(
    tmp_path: Path,
) -> None:
    from human_qc.overlay_worker import BoundedOverlayWorker

    job_dir = tmp_path / ("a" * 64)
    job_dir.mkdir()
    files = {
        "manifest.json": b"old-manifest",
        ".generation.lock": b"lock",
        ".render.fence": b"fence",
        ".generation-owner.json": b"owner",
        "segment-000.mp4": b"media",
    }
    for name, payload in files.items():
        (job_dir / name).write_bytes(payload)
    (job_dir / "ignored-directory").mkdir()

    assert BoundedOverlayWorker._job_footprint(job_dir) == sum(
        len(payload) for payload in files.values()
    )
    assert BoundedOverlayWorker._job_footprint(
        job_dir,
        manifest_size=101,
    ) == sum(
        len(payload)
        for name, payload in files.items()
        if name != "manifest.json"
    ) + 101


@pytest.mark.parametrize("failure_site", ("iterdir", "is_file", "stat"))
def test_job_footprint_is_unknown_after_any_filesystem_scan_error(
    failure_site: str,
) -> None:
    from human_qc.overlay_worker import BoundedOverlayWorker

    class BrokenEntry:
        name = ".generation-owner.json"

        def is_file(self) -> bool:
            if failure_site == "is_file":
                raise OSError("private is_file failure")
            return True

        def stat(self) -> object:
            if failure_site == "stat":
                raise OSError("private stat failure")
            return type("Stat", (), {"st_size": 17})()

    class BrokenDirectory:
        def iterdir(self) -> tuple[BrokenEntry, ...]:
            if failure_site == "iterdir":
                raise OSError("private iterdir failure")
            return (BrokenEntry(),)

    assert BoundedOverlayWorker._job_footprint(BrokenDirectory()) is None


def test_ready_publication_fails_closed_when_the_current_footprint_is_unknown(
    tmp_path: Path,
) -> None:
    from human_qc.overlay_worker import BoundedOverlayWorker

    request = _request(
        tmp_path,
        intervals=((0, 1),),
        renderer=RecordingRenderer(),
    )
    worker = BoundedOverlayWorker(max_workers=1, max_pending=0, max_cache_bytes=4096)
    original_footprint = worker._job_footprint
    job_dir = request.cache_root / request.cache_key.digest

    def unknown_current(
        path: Path,
        *,
        manifest_size: int | None = None,
    ) -> int | None:
        if path == job_dir:
            return None
        return original_footprint(path, manifest_size=manifest_size)

    worker._job_footprint = unknown_current
    try:
        worker.submit(request)
        failed = _wait_until_terminal(worker, request)

        assert failed.status == "failed"
        assert failed.code == "overlay_cache_full"
        assert not job_dir.exists()
    finally:
        worker._job_footprint = original_footprint
        worker.shutdown()


def test_ready_eviction_fails_closed_when_an_existing_footprint_is_unknown(
    tmp_path: Path,
) -> None:
    from human_qc.overlay_worker import BoundedOverlayWorker

    first = _request(
        tmp_path,
        intervals=((0, 1),),
        renderer=RecordingRenderer(),
        source_sha256="sha256:" + "5" * 64,
    )
    second = _request(
        tmp_path,
        intervals=((2, 3),),
        renderer=RecordingRenderer(),
        source_sha256="sha256:" + "6" * 64,
    )
    worker = BoundedOverlayWorker(max_workers=1, max_pending=0, max_ready_jobs=1)
    original_footprint = worker._job_footprint
    first_job_dir = first.cache_root / first.cache_key.digest

    def unknown_existing(
        path: Path,
        *,
        manifest_size: int | None = None,
    ) -> int | None:
        if path == first_job_dir:
            return None
        return original_footprint(path, manifest_size=manifest_size)

    try:
        worker.submit(first)
        assert _wait_until_terminal(worker, first).status == "ready"
        worker._job_footprint = unknown_existing

        worker.submit(second)
        second_terminal = _wait_until_terminal(worker, second)
        statuses = [
            json.loads(path.read_text(encoding="utf-8"))["status"]
            for path in first.cache_root.glob("*/manifest.json")
        ]

        assert second_terminal.status == "failed"
        assert second_terminal.code == "overlay_cache_full"
        assert first_job_dir.is_dir()
        assert statuses.count("ready") == 1
    finally:
        worker._job_footprint = original_footprint
        worker.shutdown()


def test_failed_job_eviction_does_not_release_a_ready_job_slot(tmp_path: Path) -> None:
    from human_qc.overlay_worker import BoundedOverlayWorker

    failed_request = _request(
        tmp_path,
        intervals=((0, 1),),
        renderer=FailingRenderer(),
        source_sha256="sha256:" + "1" * 64,
    )
    first_ready_request = _request(
        tmp_path,
        intervals=((1, 2),),
        renderer=RecordingRenderer(),
        source_sha256="sha256:" + "2" * 64,
    )
    second_ready_request = _request(
        tmp_path,
        intervals=((2, 3),),
        renderer=RecordingRenderer(),
        source_sha256="sha256:" + "3" * 64,
    )
    worker = BoundedOverlayWorker(max_workers=1, max_pending=1, max_ready_jobs=1)
    try:
        worker.submit(failed_request)
        assert _wait_until_terminal(worker, failed_request).status == "failed"
        worker.submit(first_ready_request)
        assert _wait_until_terminal(worker, first_ready_request).status == "ready"
        worker.submit(second_ready_request)
        assert _wait_until_terminal(worker, second_ready_request).status == "ready"

        statuses = [
            json.loads(path.read_text(encoding="utf-8"))["status"]
            for path in failed_request.cache_root.glob("*/manifest.json")
        ]
        assert statuses.count("ready") == 1
    finally:
        worker.shutdown()


def test_distinct_failed_keys_do_not_accumulate_empty_job_directories_under_tiny_quota(
    tmp_path: Path,
) -> None:
    from human_qc.overlay_worker import BoundedOverlayWorker

    worker = BoundedOverlayWorker(max_workers=1, max_pending=1, max_cache_bytes=1)
    requests = [
        _request(
            tmp_path,
            intervals=((0, 1),),
            renderer=FailingRenderer(),
            source_sha256="sha256:" + f"{index:064x}",
        )
        for index in range(1, 4)
    ]
    try:
        for request in requests:
            worker.submit(request)
            failed = _wait_until_terminal(worker, request)
            assert failed.status == "failed"
            assert worker.get(request).status == "failed"
            assert not [
                path
                for path in request.cache_root.glob("*")
                if path.is_dir() and len(path.name) == 64
            ]
    finally:
        worker.shutdown()


def test_tiny_quota_cleanup_reports_that_a_current_failed_job_was_not_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from human_qc import overlay_worker

    request = _request(
        tmp_path,
        intervals=((0, 1),),
        renderer=FailingRenderer(),
    )
    worker = overlay_worker.BoundedOverlayWorker(
        max_workers=1,
        max_pending=0,
        max_cache_bytes=1,
    )

    def broken_rmtree(_path: object) -> None:
        raise OSError("private current-job deletion failure")

    monkeypatch.setattr(overlay_worker.shutil, "rmtree", broken_rmtree)
    try:
        worker.submit(request)
        failed = _wait_until_terminal(worker, request)
        job_dir = request.cache_root / request.cache_key.digest

        assert failed.status == "failed"
        assert job_dir.exists()
        assert _regular_file_bytes(job_dir) > 1
        assert worker._safe_remove_job(request.cache_root, job_dir) is False
        assert job_dir.exists()
    finally:
        worker.shutdown()


def test_failed_eviction_does_not_deduct_bytes_when_victim_deletion_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from human_qc import overlay_worker

    max_cache_bytes = 700
    first = _request(
        tmp_path,
        intervals=((0, 1),),
        renderer=FailingRenderer(),
        source_sha256="sha256:" + "1" * 64,
    )
    second = _request(
        tmp_path,
        intervals=((2, 3),),
        renderer=FailingRenderer(),
        source_sha256="sha256:" + "2" * 64,
    )
    worker = overlay_worker.BoundedOverlayWorker(
        max_workers=1,
        max_pending=0,
        max_cache_bytes=max_cache_bytes,
    )
    original_rmtree = overlay_worker.shutil.rmtree
    first_job_dir = first.cache_root / first.cache_key.digest
    failed_deletions = 0

    def fail_first_victim(path: object) -> None:
        nonlocal failed_deletions
        if Path(path) == first_job_dir and failed_deletions == 0:
            failed_deletions += 1
            raise OSError("private failed-victim deletion failure")
        original_rmtree(path)

    try:
        worker.submit(first)
        assert _wait_until_terminal(worker, first).status == "failed"
        assert first_job_dir.is_dir()
        monkeypatch.setattr(overlay_worker.shutil, "rmtree", fail_first_victim)

        worker.submit(second)
        assert _wait_until_terminal(worker, second).status == "failed"

        assert failed_deletions == 1
        assert first_job_dir.is_dir()
        assert _regular_file_bytes(first.cache_root) <= max_cache_bytes
    finally:
        worker.shutdown()


def test_ready_eviction_does_not_release_a_slot_when_victim_deletion_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from human_qc import overlay_worker

    first = _request(
        tmp_path,
        intervals=((0, 1),),
        renderer=RecordingRenderer(),
        source_sha256="sha256:" + "3" * 64,
    )
    second = _request(
        tmp_path,
        intervals=((2, 3),),
        renderer=RecordingRenderer(),
        source_sha256="sha256:" + "4" * 64,
    )
    worker = overlay_worker.BoundedOverlayWorker(
        max_workers=1,
        max_pending=0,
        max_ready_jobs=1,
    )
    original_rmtree = overlay_worker.shutil.rmtree
    first_job_dir = first.cache_root / first.cache_key.digest
    failed_deletions = 0

    def fail_first_victim(path: object) -> None:
        nonlocal failed_deletions
        if Path(path) == first_job_dir and failed_deletions == 0:
            failed_deletions += 1
            raise OSError("private ready-victim deletion failure")
        original_rmtree(path)

    try:
        worker.submit(first)
        assert _wait_until_terminal(worker, first).status == "ready"
        monkeypatch.setattr(overlay_worker.shutil, "rmtree", fail_first_victim)

        worker.submit(second)
        second_terminal = _wait_until_terminal(worker, second)
        statuses = [
            json.loads(path.read_text(encoding="utf-8"))["status"]
            for path in first.cache_root.glob("*/manifest.json")
        ]

        assert failed_deletions == 1
        assert second_terminal.status == "failed"
        assert second_terminal.code == "overlay_cache_full"
        assert statuses.count("ready") == 1
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


def test_run_exception_fallback_failed_view_respects_tiny_quota(tmp_path: Path) -> None:
    from human_qc.overlay_worker import BoundedOverlayWorker

    request = _request(
        tmp_path,
        intervals=((20, 21),),
        renderer=RecordingRenderer(),
    )
    worker = BoundedOverlayWorker(max_workers=1, max_pending=0, max_cache_bytes=1)
    original_publish = worker._publish_rendered

    def broken_publish(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("private publish failure")

    worker._publish_rendered = broken_publish
    try:
        worker.submit(request)
        failed = _wait_until_terminal(worker, request)

        assert failed.status == "failed"
        assert failed.code == "overlay_render_failed"
        assert failed.retryable is True
        assert worker.get(request).code == "overlay_render_failed"
        assert not (request.cache_root / request.cache_key.digest).exists()
    finally:
        worker._publish_rendered = original_publish
        worker.shutdown()


def test_cleanup_failure_failed_view_respects_tiny_quota(tmp_path: Path) -> None:
    from human_qc.overlay_worker import BoundedOverlayWorker

    request = _request(
        tmp_path,
        intervals=((20, 21),),
        renderer=RecordingRenderer(),
    )
    worker = BoundedOverlayWorker(max_workers=1, max_pending=0, max_cache_bytes=1)
    original_release = worker._release_owner

    def broken_release(_request: object, _token: str) -> None:
        raise OSError("private owner cleanup failure")

    worker._release_owner = broken_release
    try:
        worker.submit(request)
        failed = _wait_until_terminal(worker, request)

        assert failed.status == "failed"
        assert failed.code == "overlay_cleanup_failed"
        assert failed.retryable is True
        assert worker.get(request).code == "overlay_cleanup_failed"
        assert not (request.cache_root / request.cache_key.digest).exists()
    finally:
        worker._release_owner = original_release
        worker.shutdown()


def test_cleanup_failure_quota_counts_the_remaining_owner_file_in_final_footprint(
    tmp_path: Path,
) -> None:
    from human_qc.overlay_worker import BoundedOverlayWorker

    request = _request(
        tmp_path,
        intervals=((20, 21),),
        renderer=RecordingRenderer(),
    )
    worker = BoundedOverlayWorker(
        max_workers=1,
        max_pending=0,
    )
    original_release = worker._release_owner
    observed: dict[str, int] = {}

    def broken_release(_request: object, _token: str) -> None:
        cleanup = worker._failed_view(
            request,
            "overlay_cleanup_failed",
            retryable=True,
        )
        manifest_size = len(
            json.dumps(
                worker._manifest_payload(request, cleanup),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        job_dir = request.cache_root / request.cache_key.digest
        retained_file_bytes = sum(
            path.stat().st_size
            for path in job_dir.iterdir()
            if path.is_file()
            and path.name != "manifest.json"
            and path.suffix != ".mp4"
            and ".partial-" not in path.name
        )
        projected_bytes = manifest_size + retained_file_bytes
        observed["projected_bytes"] = projected_bytes
        observed["max_cache_bytes"] = projected_bytes - 1
        worker._max_cache_bytes = projected_bytes - 1
        raise OSError("private owner cleanup failure")

    worker._release_owner = broken_release
    try:
        worker.submit(request)
        failed = _wait_until_terminal(worker, request)
        job_dir = request.cache_root / request.cache_key.digest

        assert failed.status == "failed"
        assert failed.code == "overlay_cleanup_failed"
        assert worker.get(request).code == "overlay_cleanup_failed"
        assert observed["projected_bytes"] > observed["max_cache_bytes"]
        assert not job_dir.exists() or sum(
            path.stat().st_size for path in job_dir.iterdir() if path.is_file()
        ) <= observed["max_cache_bytes"]
    finally:
        worker._release_owner = original_release
        worker.shutdown()


def test_stale_cleanup_failure_cannot_delete_a_successor_workers_owned_job(
    tmp_path: Path,
) -> None:
    from human_qc.overlay_worker import BoundedOverlayWorker

    old_request = _request(
        tmp_path,
        intervals=((20, 21),),
        renderer=FailingRenderer(),
    )
    successor_renderer = BlockingRenderer()
    successor_request = replace(old_request, renderer=successor_renderer)
    old_worker = BoundedOverlayWorker(
        max_workers=1,
        max_pending=0,
        owner_lease_seconds=0.03,
    )
    successor_worker = BoundedOverlayWorker(
        max_workers=1,
        max_pending=0,
        owner_lease_seconds=0.03,
    )
    original_release = old_worker._release_owner
    original_publish_failed = old_worker._publish_failed
    cleanup_waiting = Event()
    ownership_handoff = Barrier(2)
    observed_cleanup_tokens: list[str | None] = []

    def broken_release(_request: object, _token: str) -> None:
        # Apply the reviewer's cleanup boundary only after the first failed
        # manifest exists, so this test isolates stale-owner handoff semantics.
        old_worker._max_cache_bytes = 409
        raise OSError("private old-owner release failure")

    def pause_cleanup_publication(
        cleanup_request: object,
        candidate: object,
        *,
        owner_token: str | None = None,
    ):
        if getattr(candidate, "code", None) == "overlay_cleanup_failed":
            observed_cleanup_tokens.append(owner_token)
            cleanup_waiting.set()
            ownership_handoff.wait(timeout=2.0)
        return original_publish_failed(
            cleanup_request,
            candidate,
            owner_token=owner_token,
        )

    old_worker._release_owner = broken_release
    old_worker._publish_failed = pause_cleanup_publication
    try:
        old_worker.submit(old_request)
        assert cleanup_waiting.wait(2.0)

        accepted = successor_worker.retry(successor_request)
        assert accepted.status in {"pending", "generating"}
        assert successor_renderer.started.wait(2.0)
        ownership_handoff.wait(timeout=2.0)

        old_failed = _wait_until_terminal(old_worker, old_request)
        successor_renderer.release.set()
        successor_ready = _wait_until_terminal(successor_worker, successor_request)
        manifest_path = (
            successor_request.cache_root
            / successor_request.cache_key.digest
            / "manifest.json"
        )

        assert successor_ready.status == "ready"
        assert json.loads(manifest_path.read_text(encoding="utf-8"))["status"] == "ready"
        assert observed_cleanup_tokens and observed_cleanup_tokens[0] is not None
        assert old_failed.status == "failed"
        assert old_failed.code == "overlay_cleanup_failed"
    finally:
        successor_renderer.release.set()
        ownership_handoff.abort()
        old_worker._release_owner = original_release
        old_worker._publish_failed = original_publish_failed
        old_worker.shutdown()
        successor_worker.shutdown()


def test_heartbeat_after_quota_cleanup_does_not_recreate_an_empty_job_directory(
    tmp_path: Path,
) -> None:
    from human_qc.overlay_worker import BoundedOverlayWorker

    request = _request(
        tmp_path,
        intervals=((20, 21),),
        renderer=RecordingRenderer(),
    )
    worker = BoundedOverlayWorker(
        max_workers=1,
        max_pending=0,
        max_cache_bytes=1,
        owner_lease_seconds=0.03,
    )
    job_dir = request.cache_root / request.cache_key.digest
    quota_cleanup_done = Event()
    heartbeat_after_cleanup = Event()
    original_remove = worker._safe_remove_job
    original_heartbeat = worker._heartbeat_owner
    original_stop = worker._stop_owner_heartbeat

    def observed_remove(root: Path, candidate: Path) -> bool:
        removed = original_remove(root, candidate)
        if candidate == job_dir and not candidate.exists():
            quota_cleanup_done.set()
        return removed

    def observed_heartbeat(heartbeat_request: object, token: str) -> None:
        original_heartbeat(heartbeat_request, token)
        if quota_cleanup_done.is_set():
            heartbeat_after_cleanup.set()

    def stop_after_interleaving(stop: Event, thread: object) -> None:
        assert quota_cleanup_done.wait(1.0)
        assert heartbeat_after_cleanup.wait(1.0)
        original_stop(stop, thread)

    worker._safe_remove_job = observed_remove
    worker._heartbeat_owner = observed_heartbeat
    worker._stop_owner_heartbeat = stop_after_interleaving
    try:
        worker.submit(request)
        failed = _wait_until_terminal(worker, request)

        assert failed.status == "failed"
        assert failed.code == "overlay_cache_full"
        assert heartbeat_after_cleanup.is_set()
        assert not job_dir.exists()
    finally:
        worker._safe_remove_job = original_remove
        worker._heartbeat_owner = original_heartbeat
        worker._stop_owner_heartbeat = original_stop
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
