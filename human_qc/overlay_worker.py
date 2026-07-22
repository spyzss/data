"""Bounded, durable generation for continuous SAM3 overlay segments.

This module intentionally has no HTTP or browser dependencies.  A caller may
submit a complete asset-level set of source-frame intervals, then poll its
in-memory/durable view.  Rendering always happens in the bounded executor;
callers never wait for a renderer or receive filesystem details.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import tempfile
from threading import BoundedSemaphore, Event, RLock, Thread
import time
from typing import Literal, Protocol
import uuid


_STATUS = frozenset({"pending", "generating", "ready", "failed"})
_STABLE_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}").fullmatch
_DIGEST = re.compile(r"[0-9a-f]{64}").fullmatch
_MANIFEST_NAME = "manifest.json"
_OWNER_NAME = ".generation.owner.json"
_RENDER_FENCE_NAME = ".render.fence"
_PUBLISH_LOCK_NAME = ".overlay-publish.lock"
_PIN_ROOT_NAME = ".overlay-pins"
_SCHEMA_VERSION = 1
_PROCESS_INCARNATION = uuid.uuid4().hex
_LOCAL_OWNER_LOCK = RLock()
_LOCAL_OWNER_TOKENS: set[tuple[str, str]] = set()
_MIN_LEASE_SECONDS = 0.01
_MAX_OWNER_LEASE_SECONDS = 300.0
_MAX_PIN_LEASE_SECONDS = 3600.0

FrameInterval = tuple[int, int]
OverlayStatus = Literal["pending", "generating", "ready", "failed"]


class OverlayWorkerError(RuntimeError):
    """A stable, public-safe overlay worker failure."""


class OverlayRenderError(OverlayWorkerError):
    """A renderer may use this to select one stable public error code."""


class OverlayCacheFullError(OverlayWorkerError):
    """The configured cache cannot make room for another completed result."""


class OverlayRenderer(Protocol):
    """Render one complete, half-open source-frame interval to ``output_path``.

    Implementations must iterate every source frame in ``[start, end)`` exactly
    once.  Keeping the interval boundary at the worker means a caller cannot
    accidentally feed a renderer duplicate overlap frames.
    """

    def render_interval(
        self,
        request: "OverlayRequest",
        start_frame: int,
        end_frame_exclusive: int,
        output_path: Path,
    ) -> Mapping[str, object]:
        """Write one segment and return verified output metadata.

        ``frame_count`` and ``fps`` must describe the produced media exactly.
        Concrete MP4 renderers normally obtain those values from ffprobe after
        closing the encoder; the core worker deliberately stays codec agnostic.
        """


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _safe_code(value: object, fallback: str) -> str:
    return value if isinstance(value, str) and _STABLE_CODE(value) else fallback


def _bounded_lease_seconds(value: object, name: str, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    seconds = float(value)
    if not math.isfinite(seconds) or not _MIN_LEASE_SECONDS <= seconds <= maximum:
        raise ValueError(f"{name} must be between {_MIN_LEASE_SECONDS} and {maximum}")
    return seconds


def merge_frame_intervals(intervals: Iterable[FrameInterval]) -> tuple[FrameInterval, ...]:
    """Normalize non-empty half-open source-frame intervals.

    Overlap *and adjacency* are joined.  Adjacent ranges have no source-frame
    gap and therefore belong in one encoded video segment.
    """

    if isinstance(intervals, (str, bytes, Mapping)):
        raise TypeError("intervals must be an iterable of frame pairs")
    parsed: list[FrameInterval] = []
    try:
        iterator = iter(intervals)
    except TypeError as exc:
        raise TypeError("intervals must be iterable") from exc
    for interval in iterator:
        if isinstance(interval, (str, bytes, Mapping)):
            raise TypeError("each interval must be a two-item sequence")
        try:
            start, end = interval
        except (TypeError, ValueError) as exc:
            raise TypeError("each interval must be a two-item sequence") from exc
        if not _is_int(start) or not _is_int(end):
            raise TypeError("frame bounds must be integers")
        if start < 0 or end <= start:
            raise ValueError("frame interval must be non-negative and non-empty")
        parsed.append((start, end))
    if not parsed:
        return ()
    parsed.sort()
    result: list[FrameInterval] = []
    current_start, current_end = parsed[0]
    for start, end in parsed[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        result.append((current_start, current_end))
        current_start, current_end = start, end
    result.append((current_start, current_end))
    return tuple(result)


@dataclass(frozen=True)
class OverlayCacheKey:
    """All inputs that can change rendered pixels participate in cache identity."""

    source_sha256: str
    intervals: tuple[FrameInterval, ...]
    model_hash: str
    config_hash: str
    input_fingerprint_hash: str
    renderer_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_sha256", _required_text(self.source_sha256, "source_sha256"))
        object.__setattr__(self, "intervals", merge_frame_intervals(self.intervals))
        if not self.intervals:
            raise ValueError("intervals must not be empty")
        for name in (
            "model_hash",
            "config_hash",
            "input_fingerprint_hash",
            "renderer_version",
        ):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))

    @property
    def digest(self) -> str:
        payload = {
            "source_sha256": self.source_sha256,
            "intervals": self.intervals,
            "model_hash": self.model_hash,
            "config_hash": self.config_hash,
            "input_fingerprint_hash": self.input_fingerprint_hash,
            "renderer_version": self.renderer_version,
        }
        encoded = json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class OverlayRequest:
    """Server-side-only request used by :class:`BoundedOverlayWorker`."""

    asset_id: str
    cache_root: Path
    source_sha256: str
    intervals: tuple[FrameInterval, ...]
    fps: float
    total_frames: int
    model_hash: str
    config_hash: str
    input_fingerprint_hash: str
    renderer_version: str
    renderer: OverlayRenderer | object | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "asset_id", _required_text(self.asset_id, "asset_id"))
        object.__setattr__(self, "cache_root", Path(self.cache_root))
        if isinstance(self.fps, bool) or not isinstance(self.fps, (int, float)):
            raise TypeError("fps must be a positive finite number")
        if not math.isfinite(float(self.fps)) or float(self.fps) <= 0:
            raise ValueError("fps must be a positive finite number")
        object.__setattr__(self, "fps", float(self.fps))
        if not _is_int(self.total_frames) or self.total_frames <= 0:
            raise ValueError("total_frames must be a positive integer")
        normalized = merge_frame_intervals(self.intervals)
        if not normalized:
            raise ValueError("intervals must not be empty")
        if any(end > self.total_frames for _, end in normalized):
            raise ValueError("frame interval is outside total_frames")
        object.__setattr__(self, "intervals", normalized)
        if not callable(getattr(self.renderer, "render_interval", None)):
            raise TypeError("renderer must expose render_interval")

    @property
    def cache_key(self) -> OverlayCacheKey:
        return OverlayCacheKey(
            source_sha256=self.source_sha256,
            intervals=self.intervals,
            model_hash=self.model_hash,
            config_hash=self.config_hash,
            input_fingerprint_hash=self.input_fingerprint_hash,
            renderer_version=self.renderer_version,
        )


@dataclass(frozen=True)
class OverlaySegmentView:
    """One opaque continuous MP4 segment; ``path`` remains server-side only."""

    start_frame: int
    end_frame_exclusive: int
    status: OverlayStatus
    overlay_id: str
    path: Path | None = None
    code: str | None = None
    retryable: bool = False
    content_sha256: str | None = None
    metadata: Mapping[str, object] | None = None

    def to_safe_dict(self) -> dict[str, object]:
        return {
            "start_frame": self.start_frame,
            "end_frame_exclusive": self.end_frame_exclusive,
            "status": self.status,
            "overlay_id": self.overlay_id,
            "code": self.code,
            "retryable": self.retryable,
        }


@dataclass(frozen=True)
class OverlayJobView:
    """Internal job state.  Only ``to_safe_dict`` is browser-safe."""

    cache_key: OverlayCacheKey
    status: OverlayStatus
    segments: tuple[OverlaySegmentView, ...]
    cache_hit: bool = False
    deduplicated: bool = False
    code: str | None = None
    retryable: bool = False

    def to_safe_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "cache_hit": self.cache_hit,
            "deduplicated": self.deduplicated,
            "code": self.code,
            "retryable": self.retryable,
            "segments": [segment.to_safe_dict() for segment in self.segments],
        }


class BoundedOverlayWorker:
    """A process-local single-flight worker with a durable, validated cache.

    The semaphore covers both running workers and the executor's waiting work,
    avoiding ``ThreadPoolExecutor``'s otherwise unbounded submission queue.
    A file lock around generation prevents two server processes from publishing
    the same cache key concurrently; this class itself still owns in-memory
    single-flight state only for its process.
    """

    def __init__(
        self,
        *,
        max_workers: int = 1,
        max_pending: int = 1,
        max_cache_bytes: int | None = None,
        max_ready_jobs: int | None = None,
        owner_lease_seconds: float = 30.0,
        pin_lease_seconds: float = 60.0,
    ) -> None:
        for name, value in (("max_workers", max_workers), ("max_pending", max_pending)):
            if not _is_int(value) or value < 0 or (name == "max_workers" and value == 0):
                raise ValueError(f"{name} must be a valid non-negative integer")
        for name, value in (("max_cache_bytes", max_cache_bytes), ("max_ready_jobs", max_ready_jobs)):
            if value is not None and (not _is_int(value) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="sam3-overlay")
        self._capacity = BoundedSemaphore(max_workers + max_pending)
        self._max_cache_bytes = max_cache_bytes
        self._max_ready_jobs = max_ready_jobs
        self._owner_lease_seconds = _bounded_lease_seconds(
            owner_lease_seconds,
            "owner_lease_seconds",
            _MAX_OWNER_LEASE_SECONDS,
        )
        self._pin_lease_seconds = _bounded_lease_seconds(
            pin_lease_seconds,
            "pin_lease_seconds",
            _MAX_PIN_LEASE_SECONDS,
        )
        self._lock = RLock()
        self._inflight: dict[tuple[str, str], Future[None]] = {}
        self._scheduled: set[tuple[str, str]] = set()
        self._views: dict[tuple[str, str], OverlayJobView] = {}
        self._pin_leases: dict[tuple[str, str], tuple[str, Path]] = {}
        self._closed = False

    @staticmethod
    def _root(request: OverlayRequest) -> Path:
        return request.cache_root.resolve()

    def _job_id(self, request: OverlayRequest) -> tuple[str, str]:
        return (str(self._root(request)), request.cache_key.digest)

    def _job_dir(self, request: OverlayRequest) -> Path:
        return self._root(request) / request.cache_key.digest

    def _manifest_path(self, request: OverlayRequest) -> Path:
        return self._job_dir(request) / _MANIFEST_NAME

    def _owner_path(self, request: OverlayRequest) -> Path:
        return self._job_dir(request) / _OWNER_NAME

    def _render_fence_path(self, request: OverlayRequest) -> Path:
        return self._job_dir(request) / _RENDER_FENCE_NAME

    @staticmethod
    def _overlay_id(key: OverlayCacheKey, index: int) -> str:
        return f"overlay-{key.digest[:24]}-{index}"

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _json_metadata(value: object) -> object:
        """Return a finite, JSON-safe metadata value or reject it.

        The renderer owns codec-specific metadata, but a manifest must never
        contain objects whose later JSON encoding silently changes semantics.
        """

        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("metadata numbers must be finite")
            return value
        if isinstance(value, list):
            return [BoundedOverlayWorker._json_metadata(item) for item in value]
        if isinstance(value, tuple):
            return [BoundedOverlayWorker._json_metadata(item) for item in value]
        if isinstance(value, Mapping):
            normalized: dict[str, object] = {}
            for key, item in value.items():
                if not isinstance(key, str) or not key:
                    raise ValueError("metadata keys must be non-empty strings")
                normalized[key] = BoundedOverlayWorker._json_metadata(item)
            return normalized
        raise ValueError("metadata must be JSON-safe")

    @classmethod
    def _validated_segment_metadata(
        cls,
        value: object,
        request: OverlayRequest,
        start_frame: int,
        end_frame_exclusive: int,
    ) -> Mapping[str, object]:
        if not isinstance(value, Mapping):
            raise OverlayRenderError("overlay_metadata_invalid")
        try:
            normalized = cls._json_metadata(value)
        except ValueError as exc:
            raise OverlayRenderError("overlay_metadata_invalid") from exc
        assert isinstance(normalized, Mapping)
        frame_count = normalized.get("frame_count")
        fps = normalized.get("fps")
        if (
            not _is_int(frame_count)
            or frame_count != end_frame_exclusive - start_frame
            or isinstance(fps, bool)
            or not isinstance(fps, (int, float))
            or not math.isfinite(float(fps))
            or not math.isclose(float(fps), request.fps, rel_tol=0.0, abs_tol=1e-9)
        ):
            raise OverlayRenderError("overlay_metadata_invalid")
        # Persist canonical expected values rather than renderer numeric aliases
        # such as ``30`` vs ``30.0``.
        result = dict(normalized)
        result["frame_count"] = end_frame_exclusive - start_frame
        result["fps"] = request.fps
        return result

    def _empty_segments(self, request: OverlayRequest, status: OverlayStatus) -> tuple[OverlaySegmentView, ...]:
        return tuple(
            OverlaySegmentView(
                start_frame=start,
                end_frame_exclusive=end,
                status=status,
                overlay_id=self._overlay_id(request.cache_key, index),
            )
            for index, (start, end) in enumerate(request.intervals)
        )

    def _failed_view(
        self,
        request: OverlayRequest,
        code: str,
        *,
        retryable: bool,
        segments: tuple[OverlaySegmentView, ...] | None = None,
    ) -> OverlayJobView:
        normalized_code = _safe_code(code, "overlay_render_failed")
        if segments is None:
            segments = tuple(
                replace(
                    segment,
                    status="failed",
                    code=normalized_code,
                    retryable=retryable,
                )
                for segment in self._empty_segments(request, "failed")
            )
        return OverlayJobView(
            cache_key=request.cache_key,
            status="failed",
            segments=segments,
            code=normalized_code,
            retryable=retryable,
        )

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)

    @classmethod
    def _atomic_json(cls, path: Path, payload: Mapping[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, raw_temp = tempfile.mkstemp(
            prefix=".manifest-", suffix=".tmp", dir=path.parent
        )
        temporary = Path(raw_temp)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            cls._fsync_directory(path.parent)
        except Exception:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    @staticmethod
    def _safe_relative(job_dir: Path, path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            relative = path.resolve().relative_to(job_dir.resolve())
        except (OSError, ValueError) as exc:
            raise OverlayWorkerError("overlay_cache_invalid") from exc
        if len(relative.parts) != 1 or relative.name in {"", ".", ".."}:
            raise OverlayWorkerError("overlay_cache_invalid")
        return relative.name

    def _manifest_payload(self, request: OverlayRequest, view: OverlayJobView) -> dict[str, object]:
        job_dir = self._job_dir(request)
        segments: list[dict[str, object]] = []
        for segment in view.segments:
            relative = self._safe_relative(job_dir, segment.path)
            size_bytes: int | None = None
            content_sha256: str | None = None
            metadata: Mapping[str, object] | None = None
            if segment.path is not None:
                try:
                    size_bytes = segment.path.stat().st_size
                except OSError:
                    relative = None
                else:
                    if segment.status == "ready":
                        content_sha256 = segment.content_sha256 or self._sha256(segment.path)
                        if segment.metadata is not None:
                            metadata = self._validated_segment_metadata(
                                segment.metadata,
                                request,
                                segment.start_frame,
                                segment.end_frame_exclusive,
                            )
            segments.append(
                {
                    "start_frame": segment.start_frame,
                    "end_frame_exclusive": segment.end_frame_exclusive,
                    "status": segment.status,
                    "overlay_id": segment.overlay_id,
                    "relative_path": relative,
                    "size_bytes": size_bytes,
                    "sha256": content_sha256,
                    "metadata": metadata,
                    "code": segment.code,
                    "retryable": segment.retryable,
                }
            )
        return {
            "schema_version": _SCHEMA_VERSION,
            "key_digest": request.cache_key.digest,
            "status": view.status,
            "code": view.code,
            "retryable": view.retryable,
            "segments": segments,
        }

    def _persist(self, request: OverlayRequest, view: OverlayJobView) -> None:
        self._atomic_json(self._manifest_path(request), self._manifest_payload(request, view))

    def _parse_manifest(self, request: OverlayRequest) -> OverlayJobView | None:
        path = self._manifest_path(request)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError):
            return self._failed_view(request, "overlay_cache_invalid", retryable=True)
        if not isinstance(raw, Mapping):
            return self._failed_view(request, "overlay_cache_invalid", retryable=True)
        if raw.get("schema_version") != _SCHEMA_VERSION or raw.get("key_digest") != request.cache_key.digest:
            return self._failed_view(request, "overlay_cache_invalid", retryable=True)
        status = raw.get("status")
        if status not in _STATUS:
            return self._failed_view(request, "overlay_cache_invalid", retryable=True)
        raw_segments = raw.get("segments")
        if not isinstance(raw_segments, list):
            return self._failed_view(request, "overlay_cache_invalid", retryable=True)
        by_interval: dict[FrameInterval, Mapping[str, object]] = {}
        for value in raw_segments:
            if not isinstance(value, Mapping):
                return self._failed_view(request, "overlay_cache_invalid", retryable=True)
            start = value.get("start_frame")
            end = value.get("end_frame_exclusive")
            if not _is_int(start) or not _is_int(end) or (start, end) not in request.intervals:
                return self._failed_view(request, "overlay_cache_invalid", retryable=True)
            by_interval[(start, end)] = value
        if status == "ready" and set(by_interval) != set(request.intervals):
            return self._failed_view(request, "overlay_cache_invalid", retryable=True)
        job_dir = self._job_dir(request)
        segments: list[OverlaySegmentView] = []
        for index, interval in enumerate(request.intervals):
            value = by_interval.get(interval)
            if value is None:
                segment_status: OverlayStatus = "pending" if status != "failed" else "failed"
                segments.append(
                    OverlaySegmentView(
                        *interval,
                        status=segment_status,
                        overlay_id=self._overlay_id(request.cache_key, index),
                        code=("overlay_interrupted" if segment_status == "failed" else None),
                        retryable=segment_status == "failed",
                    )
                )
                continue
            segment_status = value.get("status")
            if segment_status not in _STATUS:
                return self._failed_view(request, "overlay_cache_invalid", retryable=True)
            overlay_id = value.get("overlay_id")
            if not isinstance(overlay_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", overlay_id):
                return self._failed_view(request, "overlay_cache_invalid", retryable=True)
            code = value.get("code")
            stable_code = code if isinstance(code, str) and _STABLE_CODE(code) else None
            retryable = value.get("retryable") is True
            candidate: Path | None = None
            content_sha256: str | None = None
            metadata: Mapping[str, object] | None = None
            if segment_status == "ready":
                relative = value.get("relative_path")
                size = value.get("size_bytes")
                digest = value.get("sha256")
                raw_metadata = value.get("metadata")
                if (
                    not isinstance(relative, str)
                    or not _is_int(size)
                    or size <= 0
                    or not isinstance(digest, str)
                    or _DIGEST(digest) is None
                ):
                    return self._failed_view(request, "overlay_cache_invalid", retryable=True)
                candidate = (job_dir / relative).resolve()
                try:
                    candidate.relative_to(job_dir.resolve())
                    stat = candidate.stat()
                except (OSError, ValueError):
                    return self._failed_view(request, "overlay_cache_invalid", retryable=True)
                if not candidate.is_file() or stat.st_size != size:
                    return self._failed_view(request, "overlay_cache_invalid", retryable=True)
                try:
                    content_sha256 = self._sha256(candidate)
                    metadata = self._validated_segment_metadata(
                        raw_metadata,
                        request,
                        interval[0],
                        interval[1],
                    )
                except (OSError, OverlayRenderError):
                    return self._failed_view(request, "overlay_cache_invalid", retryable=True)
                if content_sha256 != digest:
                    return self._failed_view(request, "overlay_cache_invalid", retryable=True)
            segments.append(
                OverlaySegmentView(
                    *interval,
                    status=segment_status,
                    overlay_id=overlay_id,
                    path=candidate,
                    code=stable_code,
                    retryable=retryable,
                    content_sha256=content_sha256,
                    metadata=metadata,
                )
            )
        all_ready = all(segment.status == "ready" for segment in segments)
        if status == "ready" and not all_ready:
            return self._failed_view(request, "overlay_cache_invalid", retryable=True)
        stable_code = raw.get("code")
        return OverlayJobView(
            cache_key=request.cache_key,
            status=status,
            segments=tuple(segments),
            code=(stable_code if isinstance(stable_code, str) and _STABLE_CODE(stable_code) else None),
            retryable=raw.get("retryable") is True,
        )

    def _recover_interrupted_locked(
        self, request: OverlayRequest, view: OverlayJobView
    ) -> OverlayJobView:
        """Persist a restart recovery while the short per-key state lock is held."""

        interrupted_segments = tuple(
            segment
            if segment.status == "ready"
            else replace(
                segment,
                status="failed",
                code="overlay_interrupted",
                retryable=True,
            )
            for segment in view.segments
        )
        recovered = self._failed_view(
            request,
            "overlay_interrupted",
            retryable=True,
            segments=interrupted_segments,
        )
        self._persist(request, recovered)
        return recovered

    def _read_owner_locked(self, request: OverlayRequest) -> Mapping[str, object] | None:
        try:
            raw = json.loads(self._owner_path(request).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError):
            return {}
        return raw if isinstance(raw, Mapping) else {}

    def _owner_payload(self, token: str) -> dict[str, object]:
        heartbeat_at = time.time()
        return {
            "schema_version": _SCHEMA_VERSION,
            "process_incarnation": _PROCESS_INCARNATION,
            "token": token,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "heartbeat_at": heartbeat_at,
            "lease_expires_at": heartbeat_at + self._owner_lease_seconds,
        }

    def _owner_is_live_locked(self, request: OverlayRequest) -> bool:
        """Conservatively decide whether a durable owner marker is still live.

        A marker created by this interpreter is tracked by an instance token,
        avoiding false liveness from a reused PID.  For another local process,
        ``kill(pid, 0)`` is the portable POSIX liveness probe.  An owner on an
        unknown host is intentionally treated as live: that is safer than
        clobbering a renderer on a shared cache mount.
        """

        owner = self._read_owner_locked(request)
        if owner is None:
            return False
        process_incarnation = owner.get("process_incarnation")
        token = owner.get("token")
        pid = owner.get("pid")
        host = owner.get("host")
        heartbeat_at = owner.get("heartbeat_at")
        lease_expires_at = owner.get("lease_expires_at")
        now = time.time()
        if (
            not isinstance(process_incarnation, str)
            or not isinstance(token, str)
            or not _is_int(pid)
            or pid <= 0
            or not isinstance(host, str)
            or isinstance(heartbeat_at, bool)
            or not isinstance(heartbeat_at, (int, float))
            or not math.isfinite(float(heartbeat_at))
            or isinstance(lease_expires_at, bool)
            or not isinstance(lease_expires_at, (int, float))
            or not math.isfinite(float(lease_expires_at))
            or not _MIN_LEASE_SECONDS
            <= float(lease_expires_at) - float(heartbeat_at)
            <= _MAX_OWNER_LEASE_SECONDS
            or float(lease_expires_at) <= now
            or float(lease_expires_at) > now + _MAX_OWNER_LEASE_SECONDS
        ):
            return False
        marker = (str(self._owner_path(request)), token)
        if process_incarnation == _PROCESS_INCARNATION:
            with _LOCAL_OWNER_LOCK:
                return marker in _LOCAL_OWNER_TOKENS
        if host != socket.gethostname():
            return True
        if pid == os.getpid():
            # The same PID with another incarnation is an old/reused marker,
            # not a currently executing owner in this process.
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _claim_owner_locked(self, request: OverlayRequest) -> str | None:
        """Atomically claim durable generation ownership or observe a live peer."""

        if self._owner_is_live_locked(request):
            return None
        owner_path = self._owner_path(request)
        try:
            owner_path.unlink(missing_ok=True)
        except OSError as exc:
            raise OverlayWorkerError("overlay_cache_invalid") from exc
        token = uuid.uuid4().hex
        self._atomic_json(owner_path, self._owner_payload(token))
        with _LOCAL_OWNER_LOCK:
            _LOCAL_OWNER_TOKENS.add((str(owner_path), token))
        return token

    def _owns_owner_locked(self, request: OverlayRequest, token: str) -> bool:
        owner = self._read_owner_locked(request)
        return bool(
            owner
            and owner.get("process_incarnation") == _PROCESS_INCARNATION
            and owner.get("token") == token
        )

    def _heartbeat_owner(self, request: OverlayRequest, token: str) -> None:
        """Renew a live owner's bounded lease without blocking a renderer."""

        try:
            with self._key_lock(request, blocking=False) as acquired:
                if acquired and self._owns_owner_locked(request, token):
                    self._atomic_json(self._owner_path(request), self._owner_payload(token))
        except OSError:
            # A later observer can safely recover once the last valid lease
            # expires; no filesystem details are exposed through the job view.
            return

    def _start_owner_heartbeat(self, request: OverlayRequest, token: str) -> tuple[Event, Thread]:
        stop = Event()
        interval = max(_MIN_LEASE_SECONDS, self._owner_lease_seconds / 3.0)

        def renew() -> None:
            while not stop.wait(interval):
                self._heartbeat_owner(request, token)

        thread = Thread(target=renew, name="sam3-overlay-owner-heartbeat", daemon=True)
        thread.start()
        return stop, thread

    @staticmethod
    def _stop_owner_heartbeat(stop: Event, thread: Thread) -> None:
        stop.set()
        thread.join()

    @staticmethod
    def _forget_local_owner(request: OverlayRequest, token: str) -> None:
        with _LOCAL_OWNER_LOCK:
            _LOCAL_OWNER_TOKENS.discard((str(request.cache_root.resolve() / request.cache_key.digest / _OWNER_NAME), token))

    def _release_owner(self, request: OverlayRequest, token: str) -> None:
        owner_path = self._owner_path(request)
        try:
            with self._key_lock(request) as acquired:
                if acquired and self._owns_owner_locked(request, token):
                    owner_path.unlink(missing_ok=True)
                    self._fsync_directory(owner_path.parent)
        finally:
            self._forget_local_owner(request, token)

    def _recover_if_interrupted(self, request: OverlayRequest, view: OverlayJobView) -> OverlayJobView:
        if view.status not in {"pending", "generating"}:
            return view
        # Do not hold a lock while rendering.  The owner marker is created
        # before pending state is published, so an observer can safely retain a
        # live view without acquiring renderer ownership.
        with self._key_lock(request, blocking=False) as acquired:
            if not acquired:
                return view
            latest = self._parse_manifest(request)
            if latest is None or latest.status not in {"pending", "generating"}:
                return latest or view
            if self._owner_is_live_locked(request):
                return latest
            # A delayed heartbeat can expire while a separate process is still
            # CPU-bound in the renderer.  Only recover an expired marker after
            # the OS-owned renderer fence is free; process death releases it.
            with self._render_fence(request, blocking=False) as fence_acquired:
                if not fence_acquired:
                    return latest
                try:
                    self._owner_path(request).unlink(missing_ok=True)
                except OSError:
                    return self._failed_view(request, "overlay_cache_invalid", retryable=True)
                return self._recover_interrupted_locked(request, latest)

    def _load_durable(self, request: OverlayRequest, *, recover: bool) -> OverlayJobView | None:
        view = self._parse_manifest(request)
        if view is not None and recover:
            view = self._recover_if_interrupted(request, view)
        return view

    @contextmanager
    def _key_lock(self, request: OverlayRequest, *, blocking: bool = True):
        """Acquire only a short per-key state lock, never a renderer lock."""

        job_dir = self._job_dir(request)
        job_dir.mkdir(parents=True, exist_ok=True)
        lock_path = job_dir / ".generation.lock"
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        acquired = True
        try:
            try:
                import fcntl

                flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
                try:
                    fcntl.flock(descriptor, flags)
                except BlockingIOError:
                    acquired = False
            except ImportError:  # pragma: no cover - Unix worker deployment uses fcntl.
                pass
            yield acquired
        finally:
            try:
                try:
                    import fcntl

                    if acquired:
                        fcntl.flock(descriptor, fcntl.LOCK_UN)
                except ImportError:  # pragma: no cover
                    pass
            finally:
                os.close(descriptor)

    @contextmanager
    def _render_fence(self, request: OverlayRequest, *, blocking: bool = True):
        """Fence one key's renderer for its complete CPU-bound lifetime.

        Unlike ``_key_lock``, this file lock remains held while rendering and
        is released by the OS when an owning process dies.  It never guards a
        root-wide publication or another key.
        """

        job_dir = self._job_dir(request)
        job_dir.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self._render_fence_path(request), os.O_CREAT | os.O_RDWR, 0o600)
        acquired = True
        try:
            try:
                import fcntl

                flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
                try:
                    fcntl.flock(descriptor, flags)
                except BlockingIOError:
                    acquired = False
            except ImportError:  # pragma: no cover - Unix worker deployment uses fcntl.
                pass
            yield acquired
        finally:
            try:
                try:
                    import fcntl

                    if acquired:
                        fcntl.flock(descriptor, fcntl.LOCK_UN)
                except ImportError:  # pragma: no cover
                    pass
            finally:
                os.close(descriptor)

    @contextmanager
    def _root_publish_lock(self, request: OverlayRequest):
        """Serialize only ready-cache recheck, eviction, and publication."""

        root = self._root(request)
        root.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(root / _PUBLISH_LOCK_NAME, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            try:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX)
            except ImportError:  # pragma: no cover - Unix worker deployment uses fcntl.
                pass
            yield
        finally:
            try:
                try:
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except ImportError:  # pragma: no cover
                    pass
            finally:
                os.close(descriptor)

    def _publish_view(self, request: OverlayRequest, view: OverlayJobView) -> None:
        with self._lock:
            self._views[self._job_id(request)] = view

    def _release_slot(self, request: OverlayRequest) -> None:
        with self._lock:
            job_id = self._job_id(request)
            self._inflight.pop(job_id, None)
            was_scheduled = job_id in self._scheduled
            self._scheduled.discard(job_id)
        if was_scheduled:
            self._capacity.release()

    def _segment_filename(self, index: int, interval: FrameInterval) -> str:
        start, end = interval
        return f"segment-{index:03d}-{start}-{end}.mp4"

    @staticmethod
    def _fsync_file(path: Path) -> None:
        with path.open("rb") as handle:
            os.fsync(handle.fileno())

    def _temporary_segment(self, job_dir: Path, filename: str) -> Path:
        descriptor, raw_temp = tempfile.mkstemp(
            # Keep the final media suffix: FFmpeg selects a muxer from it when
            # a concrete renderer does not pass an explicit ``-f`` argument.
            prefix=f".{filename}.", suffix=".partial.mp4", dir=job_dir
        )
        os.close(descriptor)
        return Path(raw_temp)

    @staticmethod
    def _exception_code(exc: BaseException) -> str:
        if isinstance(exc, OverlayCacheFullError):
            return "overlay_cache_full"
        if isinstance(exc, OverlayRenderError):
            return _safe_code(str(exc), "overlay_render_failed")
        return "overlay_render_failed"

    @staticmethod
    def _job_size(path: Path) -> int:
        total = 0
        try:
            for entry in path.iterdir():
                if entry.is_file() and not entry.name.startswith(".generation.lock"):
                    total += entry.stat().st_size
        except OSError:
            return total
        return total

    def _safe_remove_job(self, root: Path, job_dir: Path) -> None:
        try:
            if job_dir.parent.resolve() != root.resolve() or _DIGEST(job_dir.name) is None:
                return
            shutil.rmtree(job_dir)
        except OSError:
            return

    @staticmethod
    def _pin_directory(root: Path, digest: str) -> Path:
        return root / _PIN_ROOT_NAME / digest

    def _pin_payload(self, request: OverlayRequest, token: str, lease_seconds: float) -> dict[str, object]:
        now = time.time()
        return {
            "schema_version": _SCHEMA_VERSION,
            "key_digest": request.cache_key.digest,
            "process_incarnation": _PROCESS_INCARNATION,
            "token": token,
            "expires_at": now + lease_seconds,
        }

    def _has_live_pin(self, root: Path, digest: str) -> bool:
        """Return whether any unexpired root-visible lease protects a job.

        Callers hold the root publish lock, so removing expired leases cannot
        race cache eviction or publication by another worker.
        """

        pin_directory = self._pin_directory(root, digest)
        try:
            entries = list(pin_directory.iterdir())
        except FileNotFoundError:
            return False
        except OSError:
            # An unreadable pin is treated as live: evicting visible media is
            # riskier than declining one cache publication.
            return True
        now = time.time()
        live = False
        for path in entries:
            if not path.is_file() or path.suffix != ".json":
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None
            expires_at = payload.get("expires_at") if isinstance(payload, Mapping) else None
            valid = (
                isinstance(payload, Mapping)
                and payload.get("schema_version") == _SCHEMA_VERSION
                and payload.get("key_digest") == digest
                and isinstance(payload.get("process_incarnation"), str)
                and bool(payload.get("process_incarnation"))
                and isinstance(payload.get("token"), str)
                and bool(payload.get("token"))
                and isinstance(expires_at, (int, float))
                and not isinstance(expires_at, bool)
                and math.isfinite(float(expires_at))
                and float(expires_at) > now
                and float(expires_at) <= now + _MAX_PIN_LEASE_SECONDS
            )
            if valid:
                live = True
                continue
            try:
                path.unlink(missing_ok=True)
            except OSError:
                return True
        if not live:
            try:
                pin_directory.rmdir()
                (root / _PIN_ROOT_NAME).rmdir()
            except OSError:
                pass
        return live

    def _evict_for(self, request: OverlayRequest, current_job_size: int) -> bool:
        if self._max_cache_bytes is None and self._max_ready_jobs is None:
            return True
        root = self._root(request)
        try:
            candidates = [path for path in root.iterdir() if path.is_dir() and _DIGEST(path.name)]
        except OSError:
            return False
        ready: list[tuple[float, int, Path, bool]] = []
        total_size = 0
        ready_count = 0
        for path in candidates:
            if path == self._job_dir(request):
                continue
            key = (str(root), path.name)
            if key in self._scheduled:
                continue
            manifest = path / _MANIFEST_NAME
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, Mapping) or data.get("status") != "ready":
                continue
            size = self._job_size(path)
            total_size += size
            ready_count += 1
            try:
                modified = manifest.stat().st_mtime
            except OSError:
                modified = 0.0
            ready.append((modified, size, path, not self._has_live_pin(root, path.name)))
        ready.sort(key=lambda item: item[0])
        target_count = ready_count + 1
        while (
            (self._max_cache_bytes is not None and total_size + current_job_size > self._max_cache_bytes)
            or (self._max_ready_jobs is not None and target_count > self._max_ready_jobs)
        ):
            victim_index = next(
                (index for index, (_, _, _, evictable) in enumerate(ready) if evictable),
                None,
            )
            if victim_index is None:
                break
            _, size, victim, _ = ready.pop(victim_index)
            self._safe_remove_job(root, victim)
            total_size -= size
            target_count -= 1
        if self._max_cache_bytes is not None and total_size + current_job_size > self._max_cache_bytes:
            return False
        return self._max_ready_jobs is None or target_count <= self._max_ready_jobs

    def _render(self, request: OverlayRequest, initial: OverlayJobView) -> OverlayJobView:
        job_dir = self._job_dir(request)
        job_dir.mkdir(parents=True, exist_ok=True)
        reusable = {
            (segment.start_frame, segment.end_frame_exclusive): segment
            for segment in initial.segments
            if segment.status == "ready" and segment.path is not None and segment.path.is_file()
        }
        produced: list[OverlaySegmentView] = []
        for index, interval in enumerate(request.intervals):
            cached = reusable.get(interval)
            if cached is not None:
                produced.append(cached)
                continue
            start, end = interval
            final = job_dir / self._segment_filename(index, interval)
            temporary = self._temporary_segment(job_dir, final.name)
            try:
                renderer = request.renderer
                assert renderer is not None
                metadata = self._validated_segment_metadata(
                    renderer.render_interval(request, start, end, temporary),
                    request,
                    start,
                    end,
                )
                if not temporary.is_file() or temporary.stat().st_size <= 0:
                    raise OverlayRenderError("overlay_render_failed")
                self._fsync_file(temporary)
                os.replace(temporary, final)
                self._fsync_directory(job_dir)
                produced.append(
                    OverlaySegmentView(
                        start,
                        end,
                        "ready",
                        self._overlay_id(request.cache_key, index),
                        path=final,
                        content_sha256=self._sha256(final),
                        metadata=metadata,
                    )
                )
            except BaseException as exc:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
                code = self._exception_code(exc)
                produced.append(
                    OverlaySegmentView(
                        start,
                        end,
                        "failed",
                        self._overlay_id(request.cache_key, index),
                        code=code,
                        retryable=True,
                    )
                )
        if all(segment.status == "ready" for segment in produced):
            return OverlayJobView(
                cache_key=request.cache_key,
                status="ready",
                segments=tuple(produced),
            )
        code = next(
            (segment.code for segment in produced if segment.status == "failed" and segment.code),
            "overlay_render_failed",
        )
        return self._failed_view(
            request,
            code,
            retryable=True,
            segments=tuple(produced),
        )

    def _generating_view(
        self, request: OverlayRequest, previous: OverlayJobView | None
    ) -> OverlayJobView:
        """Build an active view without exposing completed media early."""

        reusable = () if previous is None else tuple(
            segment
            if segment.status == "ready"
            else replace(segment, status="generating", code=None, retryable=False, path=None)
            for segment in previous.segments
        )
        return OverlayJobView(
            cache_key=request.cache_key,
            status="generating",
            segments=reusable or self._empty_segments(request, "generating"),
        )

    def _discard_segments(
        self,
        segments: Iterable[OverlaySegmentView],
        *,
        keep_paths: Iterable[Path] = (),
    ) -> None:
        preserved = {path.resolve() for path in keep_paths}
        for segment in segments:
            if segment.path is not None and segment.path.resolve() not in preserved:
                try:
                    segment.path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _publish_rendered(
        self,
        request: OverlayRequest,
        candidate: OverlayJobView,
        owner_token: str,
    ) -> OverlayJobView:
        """Publish one terminal view under the root-wide cache quota lock.

        Rendering has already completed before this method runs.  The lock only
        spans durable ready recheck, eviction, and the ready/failed manifest
        write, so unrelated encoders never serialize behind it.
        """

        with self._root_publish_lock(request):
            with self._key_lock(request) as acquired:
                if not acquired or not self._owns_owner_locked(request, owner_token):
                    durable = self._parse_manifest(request)
                    return durable or self._failed_view(
                        request, "overlay_interrupted", retryable=True
                    )
                durable = self._parse_manifest(request)
                if durable is not None and durable.status == "ready":
                    self._discard_segments(
                        candidate.segments,
                        keep_paths=(
                            segment.path
                            for segment in durable.segments
                            if segment.path is not None
                        ),
                    )
                    return replace(durable, cache_hit=True)
                final = candidate
                if candidate.status == "ready" and not self._evict_for(
                    request, self._job_size(self._job_dir(request))
                ):
                    self._discard_segments(candidate.segments)
                    failed_segments = tuple(
                        replace(
                            segment,
                            status="failed",
                            path=None,
                            code="overlay_cache_full",
                            retryable=True,
                            content_sha256=None,
                            metadata=None,
                        )
                        for segment in candidate.segments
                    )
                    final = self._failed_view(
                        request,
                        "overlay_cache_full",
                        retryable=True,
                        segments=failed_segments,
                    )
                self._persist(request, final)
                return final

    def _persist_cleanup_failure(self, request: OverlayRequest, view: OverlayJobView) -> None:
        try:
            with self._root_publish_lock(request):
                with self._key_lock(request) as acquired:
                    if acquired:
                        self._persist(request, view)
        except BaseException:
            return

    def _run(
        self,
        request: OverlayRequest,
        initial: OverlayJobView,
        owner_token: str,
        heartbeat_stop: Event,
        heartbeat_thread: Thread,
    ) -> None:
        final = self._failed_view(request, "overlay_render_failed", retryable=True)
        try:
            generating: OverlayJobView | None = None
            with self._key_lock(request) as acquired:
                if not acquired or not self._owns_owner_locked(request, owner_token):
                    durable = self._parse_manifest(request)
                    final = durable or self._failed_view(
                        request, "overlay_interrupted", retryable=True
                    )
                else:
                    durable = self._parse_manifest(request)
                    if durable is not None and durable.status == "ready":
                        final = replace(durable, cache_hit=True)
                    else:
                        generating = self._generating_view(request, initial)
                        self._persist(request, generating)
                        final = generating
            if generating is not None:
                self._publish_view(request, generating)
                with self._render_fence(request) as fence_acquired:
                    if not fence_acquired:
                        rendered = self._failed_view(
                            request,
                            "overlay_interrupted",
                            retryable=True,
                        )
                    else:
                        rendered = self._render(request, generating)
                final = self._publish_rendered(request, rendered, owner_token)
        except BaseException:
            final = self._failed_view(request, "overlay_render_failed", retryable=True)
            try:
                with self._root_publish_lock(request):
                    with self._key_lock(request) as acquired:
                        if acquired and self._owns_owner_locked(request, owner_token):
                            self._persist(request, final)
            except BaseException:
                pass
        finally:
            cleanup_failed = False
            try:
                self._stop_owner_heartbeat(heartbeat_stop, heartbeat_thread)
            except BaseException:
                cleanup_failed = True
            try:
                self._release_owner(request, owner_token)
            except BaseException:
                cleanup_failed = True
            finally:
                self._forget_local_owner(request, owner_token)
            if cleanup_failed:
                final = self._failed_view(
                    request,
                    "overlay_cleanup_failed",
                    retryable=True,
                )
                self._persist_cleanup_failure(request, final)
            try:
                self._publish_view(request, final)
            finally:
                self._release_slot(request)

    def _schedule(self, request: OverlayRequest, previous: OverlayJobView | None) -> OverlayJobView:
        reusable = () if previous is None else tuple(
            segment
            if segment.status == "ready"
            else replace(segment, status="pending", code=None, retryable=False, path=None)
            for segment in previous.segments
        )
        pending = OverlayJobView(
            cache_key=request.cache_key,
            status="pending",
            segments=reusable or self._empty_segments(request, "pending"),
        )
        owner_token: str | None = None
        heartbeat_stop: Event | None = None
        heartbeat_thread: Thread | None = None
        try:
            with self._key_lock(request) as acquired:
                if not acquired:
                    self._capacity.release()
                    return self._failed_view(request, "overlay_queue_full", retryable=True)
                durable = self._parse_manifest(request)
                if durable is not None and durable.status == "ready":
                    self._capacity.release()
                    return replace(durable, cache_hit=True)
                if durable is not None and durable.status in {"pending", "generating"}:
                    if self._owner_is_live_locked(request):
                        self._capacity.release()
                        return replace(durable, deduplicated=True)
                    durable = self._recover_interrupted_locked(request, durable)
                owner_token = self._claim_owner_locked(request)
                if owner_token is None:
                    active = self._parse_manifest(request)
                    self._capacity.release()
                    return replace(active or pending, deduplicated=True)
                self._persist(request, pending)
            job_id = self._job_id(request)
            # Register capacity ownership before the executor can run a very
            # fast task and release the slot on another thread.
            self._scheduled.add(job_id)
            assert owner_token is not None
            heartbeat_stop, heartbeat_thread = self._start_owner_heartbeat(request, owner_token)
            future = self._executor.submit(
                self._run,
                request,
                pending,
                owner_token,
                heartbeat_stop,
                heartbeat_thread,
            )
        except BaseException:
            try:
                if heartbeat_stop is not None and heartbeat_thread is not None:
                    self._stop_owner_heartbeat(heartbeat_stop, heartbeat_thread)
            except BaseException:
                pass
            try:
                if owner_token is not None:
                    self._release_owner(request, owner_token)
            except BaseException:
                pass
            finally:
                if owner_token is not None:
                    self._forget_local_owner(request, owner_token)
                self._scheduled.discard(self._job_id(request))
                self._capacity.release()
            return self._failed_view(request, "overlay_render_failed", retryable=True)
        self._views[self._job_id(request)] = pending
        self._inflight[self._job_id(request)] = future
        return pending

    def submit(self, request: OverlayRequest) -> OverlayJobView:
        """Enqueue a job without waiting for SAM3, ffmpeg, or a future."""

        if not isinstance(request, OverlayRequest):
            raise TypeError("request must be an OverlayRequest")
        job_id = self._job_id(request)
        with self._lock:
            if self._closed:
                raise OverlayWorkerError("overlay_worker_closed")
            current = self._views.get(job_id)
            if current is not None and job_id in self._scheduled:
                return replace(current, deduplicated=True)
            if current is None or current.status == "ready" or current.status in {"pending", "generating"}:
                durable = self._load_durable(request, recover=True)
                if durable is not None:
                    current = durable
                    self._views[job_id] = current
                elif current is not None and current.status == "ready":
                    current = None
                    self._views.pop(job_id, None)
            if current is not None:
                if current.status in {"pending", "generating"}:
                    return replace(current, deduplicated=True)
                if current.status == "ready":
                    return replace(current, cache_hit=True)
                if current.status == "failed" and current.code != "overlay_cache_invalid":
                    return current
            if not self._capacity.acquire(blocking=False):
                return self._failed_view(request, "overlay_queue_full", retryable=True)
            return self._schedule(request, current)

    def get(self, request: OverlayRequest) -> OverlayJobView:
        """Return the current durable view; an old generating manifest is retryable."""

        if not isinstance(request, OverlayRequest):
            raise TypeError("request must be an OverlayRequest")
        job_id = self._job_id(request)
        with self._lock:
            current = self._views.get(job_id)
            if current is not None and job_id in self._scheduled:
                return current
            if current is not None and current.status == "failed":
                return current
            if current is not None and current.status not in {"pending", "generating", "ready"}:
                return current
            durable = self._load_durable(request, recover=True)
            if durable is None:
                return self._failed_view(request, "overlay_not_found", retryable=True)
            self._views[job_id] = durable
            return durable

    def retry(self, request: OverlayRequest) -> OverlayJobView:
        """Retry only a durable, retryable failed job; never rerender ready media."""

        if not isinstance(request, OverlayRequest):
            raise TypeError("request must be an OverlayRequest")
        job_id = self._job_id(request)
        with self._lock:
            if self._closed:
                raise OverlayWorkerError("overlay_worker_closed")
            current = self._views.get(job_id)
            if current is None or current.status in {"pending", "generating", "ready"}:
                durable = self._load_durable(request, recover=True)
                if durable is not None:
                    current = durable
            if current is None:
                return self._failed_view(request, "overlay_not_found", retryable=True)
            self._views[job_id] = current
            if job_id in self._scheduled:
                return replace(current, deduplicated=True)
            if current.status == "ready":
                return replace(current, cache_hit=True)
            if current.status != "failed" or not current.retryable:
                return current
            if not self._capacity.acquire(blocking=False):
                return self._failed_view(request, "overlay_queue_full", retryable=True)
            return self._schedule(request, current)

    def pin(self, request: OverlayRequest, *, lease_seconds: float | None = None) -> None:
        """Create or renew this worker's root-visible cache pin lease.

        Callers that keep media allowlisted for longer than the configured
        lease renew it by calling ``pin`` again.  Expired leases are removed by
        the next root-locked eviction pass, including in other worker
        instances/processes.
        """

        if not isinstance(request, OverlayRequest):
            raise TypeError("request must be an OverlayRequest")
        seconds = self._pin_lease_seconds if lease_seconds is None else _bounded_lease_seconds(
            lease_seconds,
            "lease_seconds",
            _MAX_PIN_LEASE_SECONDS,
        )
        job_id = self._job_id(request)
        with self._lock:
            lease = self._pin_leases.get(job_id)
            if lease is None:
                token = uuid.uuid4().hex
                lease_path = self._pin_directory(
                    self._root(request), request.cache_key.digest
                ) / f"{_PROCESS_INCARNATION}-{token}.json"
            else:
                token, lease_path = lease
            with self._root_publish_lock(request):
                lease_path.parent.mkdir(parents=True, exist_ok=True)
                self._atomic_json(lease_path, self._pin_payload(request, token, seconds))
            self._pin_leases[job_id] = (token, lease_path)

    def unpin(self, request: OverlayRequest) -> None:
        if not isinstance(request, OverlayRequest):
            raise TypeError("request must be an OverlayRequest")
        job_id = self._job_id(request)
        with self._lock:
            lease = self._pin_leases.pop(job_id, None)
            if lease is None:
                return
            _, lease_path = lease
            with self._root_publish_lock(request):
                try:
                    lease_path.unlink(missing_ok=True)
                    lease_path.parent.rmdir()
                    lease_path.parent.parent.rmdir()
                except OSError:
                    pass

    def shutdown(self, *, wait: bool = True) -> None:
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=False)


__all__ = [
    "BoundedOverlayWorker",
    "FrameInterval",
    "OverlayCacheFullError",
    "OverlayCacheKey",
    "OverlayJobView",
    "OverlayRenderError",
    "OverlayRequest",
    "OverlaySegmentView",
    "OverlayWorkerError",
    "merge_frame_intervals",
]
