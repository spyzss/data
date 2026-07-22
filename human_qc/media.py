"""Containment-safe media catalog and strict single-range helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
import hashlib
import math
import mimetypes
from pathlib import Path
import re
from threading import RLock
from typing import Any, BinaryIO

from canonical_qc.video_probe import probe_video
from qc_pipeline.context import AssetContext


_OPAQUE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}").fullmatch
_RANGE = re.compile(r"bytes=(\d*)-(\d*)").fullmatch


class MediaError(RuntimeError):
    """Base class for stable workbench media failures."""


class MediaNotFoundError(MediaError):
    """The requested asset media is absent from the server-side catalog."""


class MediaUnavailableError(MediaError):
    """An allowlisted source exists but cannot be safely probed or opened."""


class RangeNotSatisfiable(MediaError):
    """A Range header is malformed, multiple, or outside the resource."""


@dataclass(frozen=True)
class ByteRange:
    start: int
    end_inclusive: int

    @property
    def length(self) -> int:
        return self.end_inclusive - self.start + 1


@dataclass(frozen=True)
class MediaResource:
    path: Path
    size: int
    mime_type: str
    etag: str | None = None


@dataclass(frozen=True)
class SourceMedia(MediaResource):
    fps: float = 0.0
    total_frames: int = 0


def parse_byte_range(header: str | None, size: int) -> ByteRange | None:
    """Parse one RFC 9110 byte range and clamp its end to ``size``."""

    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("size must be a non-negative integer")
    if header is None:
        return None
    match = _RANGE(header)
    if match is None or size == 0:
        raise RangeNotSatisfiable("range_not_satisfiable")
    raw_start, raw_end = match.groups()
    if not raw_start and not raw_end:
        raise RangeNotSatisfiable("range_not_satisfiable")
    if not raw_start:
        suffix = int(raw_end)
        if suffix <= 0:
            raise RangeNotSatisfiable("range_not_satisfiable")
        length = min(suffix, size)
        return ByteRange(size - length, size - 1)
    start = int(raw_start)
    if start >= size:
        raise RangeNotSatisfiable("range_not_satisfiable")
    if raw_end:
        end = int(raw_end)
        if end < start:
            raise RangeNotSatisfiable("range_not_satisfiable")
        end = min(end, size - 1)
    else:
        end = size - 1
    return ByteRange(start, end)


def iter_file_chunks(
    reader: BinaryIO,
    *,
    start: int,
    length: int,
    chunk_size: int = 64 * 1024,
) -> Iterator[bytes]:
    """Yield exactly one bounded file interval without an unbounded read."""

    for name, value in (("start", start), ("length", length), ("chunk_size", chunk_size)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if start < 0 or length < 0 or chunk_size <= 0:
        raise ValueError("start/length must be non-negative and chunk_size positive")
    reader.seek(start)
    remaining = length
    while remaining:
        chunk = reader.read(min(chunk_size, remaining))
        if not chunk:
            raise MediaUnavailableError("source_video_unavailable")
        if len(chunk) > remaining:
            chunk = chunk[:remaining]
        remaining -= len(chunk)
        yield chunk


class MediaCatalog:
    """Resolve source video and opaque overlays without accepting client paths."""

    def __init__(
        self,
        asset_contexts: Mapping[str, AssetContext],
        *,
        probe: Callable[[Path], Any] = probe_video,
    ) -> None:
        self._contexts = dict(asset_contexts)
        self._probe = probe
        self._hash_by_stat: dict[Path, tuple[tuple[int, int, int, int], str]] = {}
        self._probe_by_hash: dict[str, tuple[float, int]] = {}
        self._overlays: dict[tuple[str, str], Path] = {}
        self._lock = RLock()

    @staticmethod
    def _opaque_id(value: str) -> str:
        if not isinstance(value, str) or value in {".", ".."} or _OPAQUE_ID(value) is None:
            raise ValueError("overlay_id must be an opaque path component")
        return value

    def _context(self, asset_id: str) -> AssetContext:
        try:
            return self._contexts[asset_id]
        except KeyError as exc:
            raise MediaNotFoundError("media_not_found") from exc

    @staticmethod
    def _inside(context: AssetContext, value: str | Path) -> Path:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = context.batch_root / candidate
        resolved = candidate.resolve()
        try:
            resolved.relative_to(context.batch_root.resolve())
        except ValueError as exc:
            raise MediaNotFoundError("media_not_found") from exc
        if not resolved.is_file():
            raise MediaNotFoundError("media_not_found")
        return resolved

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()

    def _source_hash(self, path: Path) -> str:
        try:
            stat = path.stat()
        except OSError as exc:
            raise MediaUnavailableError("source_video_unavailable") from exc
        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        with self._lock:
            cached = self._hash_by_stat.get(path)
            if cached is not None and cached[0] == identity:
                return cached[1]
        try:
            digest = self._hash_file(path)
        except OSError as exc:
            raise MediaUnavailableError("source_video_unavailable") from exc
        with self._lock:
            self._hash_by_stat[path] = (identity, digest)
        return digest

    @staticmethod
    def _probe_values(value: Any) -> tuple[float, int]:
        if isinstance(value, Mapping):
            raw_fps = value.get("fps")
            raw_frames = value.get("total_frames", value.get("frame_count"))
        else:
            numerator = getattr(value, "fps_num", None)
            denominator = getattr(value, "fps_den", None)
            raw_fps = (
                numerator / denominator
                if isinstance(numerator, int)
                and isinstance(denominator, int)
                and denominator != 0
                else getattr(value, "fps", None)
            )
            raw_frames = getattr(value, "frame_count", None)
        if isinstance(raw_fps, bool) or not isinstance(raw_fps, (int, float)):
            raise MediaUnavailableError("source_video_unavailable")
        fps = float(raw_fps)
        if not math.isfinite(fps) or fps <= 0:
            raise MediaUnavailableError("source_video_unavailable")
        if isinstance(raw_frames, bool) or not isinstance(raw_frames, int) or raw_frames <= 0:
            raise MediaUnavailableError("source_video_unavailable")
        return fps, raw_frames

    def source(self, asset_id: str) -> SourceMedia:
        context = self._context(asset_id)
        raw = context.source_files.get("video")
        if not isinstance(raw, Mapping):
            raise MediaNotFoundError("media_not_found")
        source_path = raw.get("path")
        if not isinstance(source_path, str) or not source_path:
            raise MediaNotFoundError("media_not_found")
        path = self._inside(context, source_path)
        source_hash = self._source_hash(path)
        with self._lock:
            values = self._probe_by_hash.get(source_hash)
        if values is None:
            try:
                values = self._probe_values(self._probe(path))
            except MediaUnavailableError:
                raise
            except Exception as exc:
                raise MediaUnavailableError("source_video_unavailable") from exc
            with self._lock:
                self._probe_by_hash[source_hash] = values
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise MediaUnavailableError("source_video_unavailable") from exc
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return SourceMedia(
            path=path,
            size=size,
            mime_type=mime_type,
            etag=source_hash,
            fps=values[0],
            total_frames=values[1],
        )

    def allow_overlay(self, asset_id: str, overlay_id: str, path: str | Path) -> None:
        context = self._context(asset_id)
        opaque = self._opaque_id(overlay_id)
        resolved = self._inside(context, path)
        with self._lock:
            self._overlays[(asset_id, opaque)] = resolved

    def overlay(self, asset_id: str, overlay_id: str) -> MediaResource:
        try:
            opaque = self._opaque_id(overlay_id)
        except ValueError as exc:
            raise MediaNotFoundError("media_not_found") from exc
        context = self._context(asset_id)
        with self._lock:
            path = self._overlays.get((asset_id, opaque))
        if path is None:
            raise MediaNotFoundError("media_not_found")
        resolved = self._inside(context, path)
        try:
            size = resolved.stat().st_size
        except OSError as exc:
            raise MediaNotFoundError("media_not_found") from exc
        return MediaResource(
            path=resolved,
            size=size,
            mime_type=mimetypes.guess_type(resolved.name)[0] or "application/octet-stream",
        )


__all__ = [
    "ByteRange",
    "MediaCatalog",
    "MediaError",
    "MediaNotFoundError",
    "MediaResource",
    "MediaUnavailableError",
    "RangeNotSatisfiable",
    "SourceMedia",
    "iter_file_chunks",
    "parse_byte_range",
]
