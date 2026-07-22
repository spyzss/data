from __future__ import annotations

from io import BytesIO
from pathlib import Path

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
