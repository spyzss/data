from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from threading import Barrier, Event, Lock

import pytest


class RecordingReader(BytesIO):
    def __init__(self, value: bytes) -> None:
        super().__init__(value)
        self.read_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return super().read(size)


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (None, None),
        ("bytes=0-99", (0, 99)),
        ("bytes=900-", (900, 999)),
        ("bytes=-100", (900, 999)),
        ("bytes=0-9999", (0, 999)),
    ],
)
def test_parse_single_byte_range(header: str | None, expected: tuple[int, int] | None) -> None:
    from human_qc.media import parse_byte_range

    value = parse_byte_range(header, 1000)
    actual = None if value is None else (value.start, value.end_inclusive)
    assert actual == expected


@pytest.mark.parametrize(
    "header",
    [
        "items=0-99",
        "bytes=",
        "bytes=0-99,200-299",
        "bytes=1000-",
        "bytes=10-9",
        "bytes=-0",
        "bytes=1 - 2",
    ],
)
def test_invalid_or_unsatisfiable_ranges_fail_closed(header: str) -> None:
    from human_qc.media import RangeNotSatisfiable

    from human_qc.media import parse_byte_range

    with pytest.raises(RangeNotSatisfiable):
        parse_byte_range(header, 1000)


def test_range_stream_reads_only_bounded_chunks() -> None:
    from human_qc.media import iter_file_chunks

    reader = RecordingReader(b"x" * 200_000)
    chunks = list(iter_file_chunks(reader, start=10, length=150_000, chunk_size=65_536))

    assert sum(map(len, chunks)) == 150_000
    assert max(reader.read_sizes) <= 65_536
    assert reader.tell() == 150_010


def test_overlay_catalog_is_scoped_by_asset_and_opaque_id(tmp_path: Path) -> None:
    from human_qc.media import MediaCatalog, MediaNotFoundError
    from qc_pipeline.context import AssetContext

    report_dir = tmp_path / "quality_archive"
    report_dir.mkdir()
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    context = AssetContext(
        asset_id="asset-1",
        batch_root=tmp_path,
        report_path=report_dir / "asset-1.json",
        source_files={"video": {"path": "video.mp4"}},
    )
    overlay = tmp_path / "overlays" / "ready.mp4"
    overlay.parent.mkdir()
    overlay.write_bytes(b"overlay")
    unlisted = overlay.parent / "unlisted.mp4"
    unlisted.write_bytes(b"secret")
    catalog = MediaCatalog(
        {"asset-1": context},
        probe=lambda _path: {"fps": 30.0, "total_frames": 1},
    )

    catalog.allow_overlay("asset-1", "opaque-1", overlay)

    assert catalog.overlay("asset-1", "opaque-1").path == overlay.resolve()
    with pytest.raises(MediaNotFoundError):
        catalog.overlay("other-asset", "opaque-1")
    with pytest.raises(MediaNotFoundError):
        catalog.overlay("asset-1", "unlisted.mp4")
    with pytest.raises(ValueError):
        catalog.allow_overlay("asset-1", "../ready", overlay)


def test_source_probe_failure_is_single_flight_and_later_request_retries(
    tmp_path: Path,
) -> None:
    from human_qc.media import MediaCatalog, MediaUnavailableError
    from qc_pipeline.context import AssetContext

    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    context = AssetContext(
        asset_id="asset-1",
        batch_root=tmp_path,
        report_path=tmp_path / "quality_archive" / "asset-1.json",
        source_files={"video": {"path": "video.mp4"}},
    )
    release_probe = Event()
    waiter_started = Event()
    retry_enabled = Event()
    calls_lock = Lock()
    probe_calls = 0

    def probe(_path: Path):
        nonlocal probe_calls
        with calls_lock:
            probe_calls += 1
        if retry_enabled.is_set():
            return {"fps": 30.0, "total_frames": 1}
        release_probe.wait(timeout=2)
        raise RuntimeError("shared probe failure")

    class ObservedCatalog(MediaCatalog):
        def _await_probe(self, flight):
            waiter_started.set()
            release_probe.set()
            return super()._await_probe(flight)

    catalog = ObservedCatalog({"asset-1": context}, probe=probe)
    original_source_hash = catalog._source_hash
    hash_barrier = Barrier(2)
    hash_calls_lock = Lock()
    hash_calls = 0

    def synchronized_source_hash(path: Path) -> str:
        nonlocal hash_calls
        value = original_source_hash(path)
        with hash_calls_lock:
            hash_calls += 1
            current = hash_calls
        if current <= 2:
            hash_barrier.wait(timeout=2)
        return value

    catalog._source_hash = synchronized_source_hash  # type: ignore[method-assign]

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(catalog.source, "asset-1") for _ in range(2)]
        for future in futures:
            with pytest.raises(MediaUnavailableError, match="source_video_unavailable"):
                future.result(timeout=2)

    assert waiter_started.is_set()
    assert probe_calls == 1

    retry_enabled.set()
    retried = catalog.source("asset-1")
    assert retried.fps == 30.0
    assert probe_calls == 2


def test_source_hashing_is_single_flight_for_one_file_identity(tmp_path: Path) -> None:
    from human_qc.media import MediaCatalog
    from qc_pipeline.context import AssetContext

    video = tmp_path / "video.mp4"
    video.write_bytes(b"video" * 100)
    context = AssetContext(
        asset_id="asset-1",
        batch_root=tmp_path,
        report_path=tmp_path / "quality_archive" / "asset-1.json",
        source_files={"video": {"path": "video.mp4"}},
    )
    release_hash = Event()
    waiter_started = Event()
    calls_lock = Lock()
    hash_calls = 0

    class ObservedCatalog(MediaCatalog):
        def _hash_file(self, path, expected_identity):
            nonlocal hash_calls
            with calls_lock:
                hash_calls += 1
                current = hash_calls
            if current == 1:
                release_hash.wait(timeout=2)
            else:
                release_hash.set()
            return super()._hash_file(path, expected_identity)

        def _await_hash(self, flight):
            waiter_started.set()
            release_hash.set()
            return super()._await_hash(flight)

    catalog = ObservedCatalog(
        {"asset-1": context},
        probe=lambda _path: {"fps": 30.0, "total_frames": 1},
    )
    start = Barrier(3)

    def load_source():
        start.wait(timeout=2)
        return catalog.source("asset-1")

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(load_source) for _ in range(2)]
        start.wait(timeout=2)
        resources = [future.result(timeout=2) for future in futures]

    assert waiter_started.is_set()
    assert hash_calls == 1
    assert resources[0].etag == resources[1].etag


def test_probe_result_is_not_cached_when_source_identity_changes(tmp_path: Path) -> None:
    from human_qc.media import MediaCatalog, MediaUnavailableError
    from qc_pipeline.context import AssetContext

    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"same-content")
    second.write_bytes(b"same-content")
    contexts = {
        asset_id: AssetContext(
            asset_id=asset_id,
            batch_root=tmp_path,
            report_path=tmp_path / "quality_archive" / f"{asset_id}.json",
            source_files={"video": {"path": path.name}},
        )
        for asset_id, path in (("first", first), ("second", second))
    }
    probe_calls: list[Path] = []

    def probe(path: Path):
        probe_calls.append(path)
        if path == first:
            replacement = tmp_path / "replacement.mp4"
            replacement.write_bytes(b"changed-content")
            replacement.replace(first)
            return {"fps": 99.0, "total_frames": 99}
        return {"fps": 30.0, "total_frames": 1}

    catalog = MediaCatalog(contexts, probe=probe)

    with pytest.raises(MediaUnavailableError, match="source_video_unavailable"):
        catalog.source("first")
    source = catalog.source("second")

    assert source.fps == 30.0
    assert probe_calls == [first.resolve(), second.resolve()]
