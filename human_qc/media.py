"""Containment-safe media catalog and strict single-range helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import Future
from dataclasses import dataclass
import hashlib
import math
import mimetypes
import os
from pathlib import Path
import re
import tempfile
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
    identity: tuple[int, int, int, int] | None = None


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


def open_verified_media(resource: MediaResource) -> BinaryIO:
    """Open exactly the file identity whose metadata was cataloged."""

    try:
        source = resource.path.open("rb")
    except OSError as exc:
        error = (
            MediaUnavailableError
            if isinstance(resource, SourceMedia)
            else MediaNotFoundError
        )
        code = (
            "source_video_unavailable"
            if isinstance(resource, SourceMedia)
            else "media_not_found"
        )
        raise error(code) from exc
    try:
        stat = os.fstat(source.fileno())
        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if resource.identity is not None and identity != resource.identity:
            error = (
                MediaUnavailableError
                if isinstance(resource, SourceMedia)
                else MediaNotFoundError
            )
            raise error(
                "source_video_unavailable"
                if isinstance(resource, SourceMedia)
                else "media_not_found"
            )
        if isinstance(resource, SourceMedia):
            snapshot = tempfile.TemporaryFile(mode="w+b")
            try:
                digest = hashlib.sha256()
                copied = 0
                while True:
                    chunk = source.read(64 * 1024)
                    if not chunk:
                        break
                    snapshot.write(chunk)
                    digest.update(chunk)
                    copied += len(chunk)
                final = os.fstat(source.fileno())
                final_identity = (
                    final.st_dev,
                    final.st_ino,
                    final.st_size,
                    final.st_mtime_ns,
                )
                actual_etag = "sha256:" + digest.hexdigest()
                if (
                    final_identity != identity
                    or copied != resource.size
                    or resource.etag is None
                    or actual_etag != resource.etag
                ):
                    raise MediaUnavailableError("source_video_unavailable")
                snapshot.seek(0)
            except Exception:
                snapshot.close()
                raise
            source.close()
            return snapshot
        return source
    except Exception:
        source.close()
        raise


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
        self._hash_inflight: dict[
            tuple[Path, tuple[int, int, int, int]], Future[str]
        ] = {}
        self._probe_by_hash: dict[str, tuple[float, int]] = {}
        self._probe_inflight: dict[str, Future[tuple[float, int]]] = {}
        self._overlays: dict[tuple[str, str], Path] = {}
        self._lock = RLock()

    @staticmethod
    def _opaque_id(value: str) -> str:
        if (
            not isinstance(value, str)
            or value in {".", ".."}
            or _OPAQUE_ID(value) is None
        ):
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
    def _identity(path: Path) -> tuple[int, int, int, int]:
        try:
            stat = path.stat()
        except OSError as exc:
            raise MediaUnavailableError("source_video_unavailable") from exc
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)

    @staticmethod
    def _hash_file(
        path: Path, expected_identity: tuple[int, int, int, int]
    ) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            before = os.fstat(source.fileno())
            opened_identity = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            )
            if opened_identity != expected_identity:
                raise MediaUnavailableError("source_video_unavailable")
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(source.fileno())
            final_identity = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            )
            if final_identity != expected_identity:
                raise MediaUnavailableError("source_video_unavailable")
        return "sha256:" + digest.hexdigest()

    def _source_hash(self, path: Path) -> str:
        identity = self._identity(path)
        key = (path, identity)
        with self._lock:
            cached = self._hash_by_stat.get(path)
            if cached is not None and cached[0] == identity:
                return cached[1]
            flight = self._hash_inflight.get(key)
            if flight is None:
                flight = Future()
                self._hash_inflight[key] = flight
                leader = True
            else:
                leader = False
        if not leader:
            return self._await_hash(flight)
        try:
            digest = self._hash_file(path, identity)
            if self._identity(path) != identity:
                raise MediaUnavailableError("source_video_unavailable")
        except MediaUnavailableError as exc:
            failure = exc
        except OSError as exc:
            failure = MediaUnavailableError("source_video_unavailable")
            failure.__cause__ = exc
        else:
            with self._lock:
                self._hash_by_stat[path] = (identity, digest)
                self._hash_inflight.pop(key, None)
            flight.set_result(digest)
            return digest
        with self._lock:
            self._hash_inflight.pop(key, None)
        flight.set_exception(failure)
        raise failure

    def _await_hash(self, flight: Future[str]) -> str:
        return flight.result()

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
        if (
            isinstance(raw_frames, bool)
            or not isinstance(raw_frames, int)
            or raw_frames <= 0
        ):
            raise MediaUnavailableError("source_video_unavailable")
        return fps, raw_frames

    def _await_probe(
        self, flight: Future[tuple[float, int]]
    ) -> tuple[float, int]:
        return flight.result()

    def _probe_source(
        self,
        path: Path,
        source_hash: str,
        expected_identity: tuple[int, int, int, int],
    ) -> tuple[float, int]:
        with self._lock:
            cached = self._probe_by_hash.get(source_hash)
            if cached is not None:
                return cached
            flight = self._probe_inflight.get(source_hash)
            if flight is None:
                flight = Future()
                self._probe_inflight[source_hash] = flight
                leader = True
            else:
                leader = False
        if not leader:
            return self._await_probe(flight)
        try:
            if self._identity(path) != expected_identity:
                raise MediaUnavailableError("source_video_unavailable")
            values = self._probe_values(self._probe(path))
            if self._identity(path) != expected_identity:
                raise MediaUnavailableError("source_video_unavailable")
        except MediaUnavailableError as exc:
            failure = exc
        except Exception as exc:
            failure = MediaUnavailableError("source_video_unavailable")
            failure.__cause__ = exc
        else:
            with self._lock:
                self._probe_by_hash[source_hash] = values
                self._probe_inflight.pop(source_hash, None)
            flight.set_result(values)
            return values
        with self._lock:
            self._probe_inflight.pop(source_hash, None)
        flight.set_exception(failure)
        raise failure

    def source(self, asset_id: str) -> SourceMedia:
        context = self._context(asset_id)
        raw = context.source_files.get("video")
        if not isinstance(raw, Mapping):
            raise MediaNotFoundError("media_not_found")
        source_path = raw.get("path")
        if not isinstance(source_path, str) or not source_path:
            raise MediaNotFoundError("media_not_found")
        path = self._inside(context, source_path)
        identity = self._identity(path)
        source_hash = self._source_hash(path)
        if self._identity(path) != identity:
            raise MediaUnavailableError("source_video_unavailable")
        values = self._probe_source(path, source_hash, identity)
        if self._identity(path) != identity:
            raise MediaUnavailableError("source_video_unavailable")
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return SourceMedia(
            path=path,
            size=identity[2],
            mime_type=mime_type,
            etag=source_hash,
            identity=identity,
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
            stat = resolved.stat()
        except OSError as exc:
            raise MediaNotFoundError("media_not_found") from exc
        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        return MediaResource(
            path=resolved,
            size=identity[2],
            mime_type=mimetypes.guess_type(resolved.name)[0] or "application/octet-stream",
            identity=identity,
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
    "open_verified_media",
    "parse_byte_range",
]
