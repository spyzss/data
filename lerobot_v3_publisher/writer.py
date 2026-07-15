"""Deterministic Curated LeRobot v3 staging materialization."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import stat
import subprocess
from typing import Any, Iterator, NoReturn

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from canonical_qc.contracts import CanonicalQcEpisode
from canonical_qc.errors import CanonicalInputError
from canonical_qc.video_probe import probe_video

from .contracts import (
    CanonicalDiagnostic,
    ManifestFile,
    PUBLISHER_VERSION,
    PublishPlan,
    PublishPrerequisiteError,
    ReleaseManifest,
    StagedRelease,
    VideoMaterialization,
)
from .prerequisites import revalidate_publish_plan
from .toolchain import current_toolchain


_STAGE = "publish_staging"
_VIDEO_KEY = "observation.images.main"
_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
_PAYLOAD_PATHS = (
    "data/chunk-000/file-000.parquet",
    "meta/episode_semantics.jsonl",
    "meta/episodes/chunk-000/file-000.parquet",
    "meta/info.json",
    "meta/stats.json",
    "meta/subtask.parquet",
    "meta/tasks.parquet",
    "videos/observation.images.main/chunk-000/file-000.mp4",
)
_MAX_TIMESTAMP_DELTA_NS = 1_000_000


def _directory_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


def _remove_tree_at(parent_fd: int, name: str) -> None:
    try:
        child_fd = os.open(name, _directory_flags(), dir_fd=parent_fd)
    except FileNotFoundError:
        return
    try:
        with os.scandir(child_fd) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    _remove_tree_at(child_fd, entry.name)
                else:
                    os.unlink(entry.name, dir_fd=child_fd)
    finally:
        os.close(child_fd)
    os.rmdir(name, dir_fd=parent_fd)


@dataclass(slots=True)
class _StagingTransaction:
    release_root: Path
    staging_root: Path
    actual_root: Path
    transaction_id: str
    release_fd: int
    staging_fd: int
    root_fd: int
    release_identity: tuple[int, int, int, int, int]
    staging_identity: tuple[int, int, int, int, int]
    root_identity: tuple[int, int, int, int, int]

    @property
    def safe_root(self) -> "_SafeRoot":
        return _SafeRoot(self.actual_root, self.root_fd)

    def verify_paths(self) -> None:
        checks = (
            (self.release_root, self.release_identity),
            (self.staging_root, self.staging_identity),
            (self.actual_root, self.root_identity),
        )
        for path, expected in checks:
            try:
                observed = os.stat(path, follow_symlinks=False)
            except OSError as exc:
                _reject("staging_root", f"staging path identity changed: {exc}", retryable=True)
            if _identity(observed)[:2] != expected[:2]:
                _reject("staging_root", "staging path identity changed", retryable=True)

    def cleanup(self) -> None:
        _remove_tree_at(self.staging_fd, self.transaction_id)

    def close(self) -> None:
        for descriptor in (self.root_fd, self.staging_fd, self.release_fd):
            try:
                os.close(descriptor)
            except OSError:
                pass


@dataclass(frozen=True, slots=True)
class _SafeRoot:
    path: Path
    descriptor: int

    def __truediv__(self, relative: str) -> Path:
        return self.path / relative

    def ensure_directory(self, relative: str) -> None:
        descriptor = os.dup(self.descriptor)
        try:
            for part in PurePosixPath(relative).parts:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(part, _directory_flags(), dir_fd=descriptor)
                os.fchmod(child, 0o700)
                os.close(descriptor)
                descriptor = child
        finally:
            os.close(descriptor)

    @contextmanager
    def parent(self, relative: str) -> Iterator[tuple[int, str]]:
        pure = PurePosixPath(relative)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
            _reject("staging", "artifact path must be normalized relative POSIX")
        descriptor = os.dup(self.descriptor)
        try:
            for part in pure.parts[:-1]:
                child = os.open(part, _directory_flags(), dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            yield descriptor, pure.name
        finally:
            os.close(descriptor)

    def open_exclusive(self, relative: str, *, read_write: bool = False) -> int:
        flags = (
            (os.O_RDWR if read_write else os.O_WRONLY)
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
        )
        with self.parent(relative) as (parent_fd, name):
            return os.open(name, flags, 0o600, dir_fd=parent_fd)

    def open_readonly(self, relative: str) -> int:
        try:
            with self.parent(relative) as (parent_fd, name):
                descriptor = os.open(
                    name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent_fd,
                )
        except OSError as exc:
            _reject(
                relative,
                f"cannot securely open staging artifact: {exc}",
                retryable=True,
            )
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            _reject(relative, "staging artifact must be a regular file")
        return descriptor

    def file_stat(self, relative: str) -> os.stat_result:
        descriptor = self.open_readonly(relative)
        try:
            return os.fstat(descriptor)
        finally:
            os.close(descriptor)

    def sha256(self, relative: str) -> str:
        descriptor = self.open_readonly(relative)
        try:
            digest = hashlib.sha256()
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            return digest.hexdigest()
        finally:
            os.close(descriptor)

    def file_paths(self) -> set[str]:
        result: set[str] = set()

        def visit(directory_fd: int, prefix: PurePosixPath) -> None:
            with os.scandir(directory_fd) as entries:
                for entry in entries:
                    relative = prefix / entry.name
                    if entry.is_dir(follow_symlinks=False):
                        child = os.open(entry.name, _directory_flags(), dir_fd=directory_fd)
                        try:
                            visit(child, relative)
                        finally:
                            os.close(child)
                    else:
                        result.add(relative.as_posix())

        descriptor = os.dup(self.descriptor)
        try:
            visit(descriptor, PurePosixPath())
        finally:
            os.close(descriptor)
        return result

    def replace(self, source: str, target: str) -> None:
        with self.parent(source) as (source_parent, source_name):
            with self.parent(target) as (target_parent, target_name):
                try:
                    os.stat(target_name, dir_fd=target_parent, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    _reject(target, "target path already exists")
                os.rename(
                    source_name,
                    target_name,
                    src_dir_fd=source_parent,
                    dst_dir_fd=target_parent,
                )


def _reject(field: str | None, message: str, *, retryable: bool = False) -> NoReturn:
    raise PublishPrerequisiteError(
        CanonicalDiagnostic("staging_failed", _STAGE, field, message, retryable)
    )


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_bytes(root: _SafeRoot, relative: str, payload: bytes) -> None:
    try:
        descriptor = root.open_exclusive(relative)
    except OSError as exc:
        _reject(relative, f"cannot create staging file: {exc}")
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
    finally:
        os.close(descriptor)


def _prepare_private_directories(root: _SafeRoot) -> None:
    for relative in (
        "data/chunk-000",
        "meta/episodes/chunk-000",
        "videos/observation.images.main/chunk-000",
    ):
        root.ensure_directory(relative)


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


@contextmanager
def _verified_video_source(plan: PublishPlan) -> Iterator[tuple[int, Path]]:
    episode = plan.request.episode
    relative = PurePosixPath(episode.main_video.path)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        _reject("main_video.path", "must be a normalized relative path")
    path = plan.request.canonical_source_root.joinpath(*relative.parts)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        _reject("main_video.path", f"cannot open source video without symlinks: {exc}", retryable=True)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            _reject("main_video.path", "must be a regular file")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        if digest.hexdigest() != episode.main_video.sha256:
            _reject("main_video.sha256", "source video changed after plan validation", retryable=True)
        os.lseek(descriptor, 0, os.SEEK_SET)
        yield descriptor, path
        os.lseek(descriptor, 0, os.SEEK_SET)
        after_digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            after_digest.update(chunk)
        after = os.fstat(descriptor)
        try:
            current = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            _reject("main_video.path", f"source video path changed: {exc}", retryable=True)
        if (
            _identity(before) != _identity(after)
            or _identity(before) != _identity(current)
            or after_digest.hexdigest() != episode.main_video.sha256
        ):
            _reject("main_video.path", "source video changed while staging", retryable=True)
    finally:
        os.close(descriptor)


def _copy_from_fd(source_fd: int, root: _SafeRoot, relative: str) -> str:
    try:
        target_fd = root.open_exclusive(relative)
    except OSError as exc:
        _reject("video", f"cannot create staged video: {exc}")
    digest = hashlib.sha256()
    try:
        os.lseek(source_fd, 0, os.SEEK_SET)
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(target_fd, view)
                view = view[written:]
    finally:
        os.close(target_fd)
    return digest.hexdigest()


def _ffmpeg_timeout_seconds(frame_count: int, fps_num: int, fps_den: int) -> int:
    duration = frame_count * fps_den / fps_num
    return min(3_600, max(60, round(duration * 10 + 30)))


def _transcode_range(
    source_fd: int,
    root: _SafeRoot,
    relative: str,
    start: int,
    end: int,
    fps_num: int,
    fps_den: int,
) -> None:
    temp_relative = str(
        PurePosixPath(relative).with_name(f".video-{secrets.token_hex(16)}.tmp")
    )
    output_fd = root.open_exclusive(temp_relative, read_write=True)
    output_identity = _identity(os.fstat(output_fd))[:2]
    argv = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-i",
        f"/dev/fd/{source_fd}",
        "-vf",
        f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS",
        "-fps_mode",
        "passthrough",
        "-an",
        "-map_metadata",
        "-1",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-fflags",
        "+bitexact",
        "-flags:v",
        "+bitexact",
        "-movflags",
        "+frag_keyframe+empty_moov+default_base_moof",
        "-f",
        "mp4",
        "pipe:1",
    ]
    try:
        completed = subprocess.run(
            argv,
            shell=False,
            stdout=output_fd,
            stderr=subprocess.PIPE,
            text=True,
            timeout=_ffmpeg_timeout_seconds(end - start, fps_num, fps_den),
            check=False,
            pass_fds=(source_fd, output_fd),
        )
        os.fsync(output_fd)
    except subprocess.TimeoutExpired as exc:
        _reject(
            "video",
            f"ffmpeg range materialization timed out: {exc}",
            retryable=True,
        )
    except OSError as exc:
        _reject("video", f"ffmpeg range materialization failed: {exc}")
    finally:
        os.close(output_fd)
    if completed.returncode != 0:
        _reject("video", f"ffmpeg range materialization failed: {completed.stderr.strip()}")
    try:
        observed_identity = _identity(root.file_stat(temp_relative))[:2]
    except OSError as exc:
        _reject("video", f"ffmpeg output temp path changed: {exc}", retryable=True)
    if observed_identity != output_identity:
        _reject("video", "ffmpeg output temp identity changed", retryable=True)
    root.replace(temp_relative, relative)


def _video_is_directly_reusable(episode: CanonicalQcEpisode, source: Path) -> bool:
    start, end = episode.main_video.source_frame_range
    if (start, end) != (0, episode.time_axis.frame_count):
        return False
    try:
        probed = probe_video(source)
    except CanonicalInputError as exc:
        _reject("main_video.path", f"source video cannot be probed: {exc}")
    if probed.frame_count != episode.time_axis.frame_count:
        return False
    if len(probed.timestamps_ns) != episode.time_axis.frame_count:
        return False
    expected = episode.time_axis.timestamps_ns - episode.time_axis.timestamps_ns[0]
    return bool(
        np.max(np.abs(np.asarray(probed.timestamps_ns, dtype=np.int64) - expected))
        <= _MAX_TIMESTAMP_DELTA_NS
    )


def _materialize_video(plan: PublishPlan, root: _SafeRoot) -> tuple[VideoMaterialization, Any]:
    episode = plan.request.episode
    relative = "videos/observation.images.main/chunk-000/file-000.mp4"
    with _verified_video_source(plan) as (source_fd, source_path):
        direct = _video_is_directly_reusable(episode, source_path)
        if direct:
            target_sha = _copy_from_fd(source_fd, root, relative)
            method = "verified_copy"
        else:
            start, end = episode.main_video.source_frame_range
            os.lseek(source_fd, 0, os.SEEK_SET)
            _transcode_range(
                source_fd,
                root,
                relative,
                start,
                end,
                episode.time_axis.fps_num,
                episode.time_axis.fps_den,
            )
            target_sha = root.sha256(relative)
            method = "transcoded_frame_range"
    if target_sha != root.sha256(relative):
        _reject("video", "staged video hash changed after materialization")
    target_fd = root.open_readonly(relative)
    try:
        probed = probe_video(Path(relative), file_descriptor=target_fd)
    except CanonicalInputError as exc:
        _reject("video", f"staged video cannot be probed: {exc}")
    finally:
        os.close(target_fd)
    expected = episode.time_axis.timestamps_ns - episode.time_axis.timestamps_ns[0]
    if (
        probed.frame_count != episode.time_axis.frame_count
        or probed.width_px != episode.main_video.width_px
        or probed.height_px != episode.main_video.height_px
        or (probed.fps_num, probed.fps_den)
        != (episode.time_axis.fps_num, episode.time_axis.fps_den)
        or len(probed.timestamps_ns) != episode.time_axis.frame_count
        or np.max(np.abs(np.asarray(probed.timestamps_ns, dtype=np.int64) - expected))
        > _MAX_TIMESTAMP_DELTA_NS
    ):
        _reject("video", "staged video frame count or PTS differs from Canonical")
    return (
        VideoMaterialization(
            relative_path=relative,
            source_relative_path=episode.main_video.path,
            source_frame_range=episode.main_video.source_frame_range,
            source_sha256=episode.main_video.sha256,
            target_sha256=target_sha,
            method=method,
        ),
        probed,
    )


def _fixed_array(array: np.ndarray) -> pa.Array:
    value = np.asarray(array)
    if value.dtype.kind in {"U", "S"}:
        arrow_type: pa.DataType = pa.string()
    else:
        arrow_type = pa.from_numpy_dtype(value.dtype)
    for size in reversed(value.shape[1:]):
        arrow_type = pa.list_(arrow_type, int(size))
    return pa.array(value.tolist(), type=arrow_type)


def _subtask_indices(episode: CanonicalQcEpisode) -> np.ndarray:
    indices = np.full(episode.time_axis.frame_count, -1, dtype=np.int64)
    for index, subtask in enumerate(episode.semantics.subtask_sequence):
        indices[subtask.start_frame : subtask.end_frame_exclusive] = index
    if np.any(indices < 0):
        _reject("semantics.subtask_sequence", "half-open boundaries do not cover [0,T)")
    return indices


def _data_table(episode: CanonicalQcEpisode) -> pa.Table:
    timestamps_ns = np.asarray(episode.time_axis.timestamps_ns, dtype=np.int64)
    relative_seconds = (timestamps_ns - timestamps_ns[0]) / 1_000_000_000
    frame_count = episode.time_axis.frame_count
    columns: dict[str, pa.Array] = {
        "index": pa.array(np.arange(frame_count, dtype=np.int64)),
        "episode_index": pa.array(np.zeros(frame_count, dtype=np.int64)),
        "frame_index": pa.array(np.arange(frame_count, dtype=np.int64)),
        "timestamp": pa.array(relative_seconds, type=pa.float64()),
        "timestamp_ns": pa.array(timestamps_ns, type=pa.int64()),
        "observation.hand_keypoints_3d": _fixed_array(episode.observation.hand_keypoints_3d),
        "observation.hand_joint_valid_3d": _fixed_array(episode.observation.hand_joint_valid_3d),
        "observation.hand_keypoints_2d": _fixed_array(episode.observation.hand_keypoints_2d),
        "observation.hand_joint_valid_2d": _fixed_array(episode.observation.hand_joint_valid_2d),
        "task_index": pa.array(np.zeros(frame_count, dtype=np.int64)),
        "subtask_index": pa.array(_subtask_indices(episode), type=pa.int64()),
    }
    quality = episode.supplier_evidence.hand_quality
    if quality is not None and quality.provided:
        if quality.raw_value is not None:
            columns["supplier.hand_quality.raw_value"] = _fixed_array(quality.raw_value)
        if quality.normalized_score is not None:
            columns["supplier.hand_quality.normalized_score"] = _fixed_array(
                quality.normalized_score
            )
        if quality.status is not None:
            columns["supplier.hand_quality.status"] = _fixed_array(quality.status)
    return pa.table(columns)


def _feature(dtype: str, shape: list[int], **extra: object) -> dict[str, object]:
    return {"dtype": dtype, "shape": shape, **extra}


def _features(episode: CanonicalQcEpisode, probed: Any, table: pa.Table) -> dict[str, object]:
    default = {"shape": [1], "names": None}
    features: dict[str, object] = {
        "timestamp": _feature("float64", **default),
        "frame_index": _feature("int64", **default),
        "episode_index": _feature("int64", **default),
        "index": _feature("int64", **default),
        "task_index": _feature("int64", **default),
        "timestamp_ns": _feature("int64", **default, unit="nanosecond"),
        "subtask_index": _feature("int64", **default),
        "observation.hand_keypoints_3d": _feature(
            "float32", [2, 21, 3], names=None, hand_order=["left", "right"],
            joint_topology="egodata_hand21.v1", coordinate_frame="camera:main", unit="meter"
        ),
        "observation.hand_joint_valid_3d": _feature("bool", [2, 21], names=None),
        "observation.hand_keypoints_2d": _feature(
            "float32", [2, 21, 2], names=None, hand_order=["left", "right"],
            joint_topology="egodata_hand21.v1", coordinate_space="pixel", unit="pixel"
        ),
        "observation.hand_joint_valid_2d": _feature("bool", [2, 21], names=None),
        _VIDEO_KEY: _feature(
            "video",
            [episode.main_video.height_px, episode.main_video.width_px, 3],
            names=["height", "width", "channels"],
            camera_id="main",
            camera_role="ego",
            info={
                "video.height": probed.height_px,
                "video.width": probed.width_px,
                "video.codec": probed.codec,
                "video.pix_fmt": probed.pixel_format,
                "video.fps": probed.fps_num / probed.fps_den,
                "video.channels": 3,
                "audio.has_audio": False,
                "is_depth_map": False,
            },
        ),
    }
    for name in (
        "supplier.hand_quality.raw_value",
        "supplier.hand_quality.normalized_score",
        "supplier.hand_quality.status",
    ):
        if name not in table.column_names:
            continue
        field = table.schema.field(name).type
        value_type = field.value_type
        if pa.types.is_string(value_type):
            dtype = "string"
        else:
            dtype = str(value_type)
        features[name] = _feature(dtype, [2], names=None)
    return features


def _safe_stats(
    array: np.ndarray, *, validity: np.ndarray | None = None
) -> dict[str, object]:
    values = np.asarray(array, dtype=np.float64)
    finite = np.isfinite(values)
    if validity is not None:
        valid = np.asarray(validity, dtype=bool)
        while valid.ndim < values.ndim:
            valid = valid[..., None]
        finite &= np.broadcast_to(valid, values.shape)
    count = finite.sum(axis=0)
    safe_count = np.maximum(count, 1)
    total = np.where(finite, values, 0).sum(axis=0)
    mean = total / safe_count
    variance = np.where(finite, (values - mean) ** 2, 0).sum(axis=0) / safe_count
    minimum = np.where(finite, values, np.inf).min(axis=0)
    maximum = np.where(finite, values, -np.inf).max(axis=0)
    minimum = np.where(np.isfinite(minimum), minimum, 0)
    maximum = np.where(np.isfinite(maximum), maximum, 0)
    return {
        "min": minimum.tolist(),
        "max": maximum.tolist(),
        "mean": mean.tolist(),
        "std": np.sqrt(variance).tolist(),
        "count": [int(values.shape[0])],
    }


def _range_stats(array: np.ndarray) -> dict[str, object]:
    values = np.asarray(array)
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "count": [int(values.shape[0])],
    }


def _boolean_stats(array: np.ndarray) -> dict[str, object]:
    values = np.asarray(array, dtype=bool)
    return {
        **_range_stats(values.astype(np.int8)),
        "true_count": np.count_nonzero(values, axis=0).tolist(),
    }


def _stats(episode: CanonicalQcEpisode) -> dict[str, object]:
    frame_count = episode.time_axis.frame_count
    timestamps_ns = np.asarray(episode.time_axis.timestamps_ns, dtype=np.int64)
    timestamps = (timestamps_ns - timestamps_ns[0]) / 1_000_000_000
    frame_indices = np.arange(frame_count, dtype=np.int64)
    valid_3d = episode.observation.hand_joint_valid_3d
    valid_2d = episode.observation.hand_joint_valid_2d
    stats: dict[str, object] = {
        "index": _range_stats(frame_indices),
        "episode_index": _range_stats(np.zeros(frame_count, dtype=np.int64)),
        "frame_index": _range_stats(frame_indices),
        "timestamp": _range_stats(timestamps),
        "timestamp_ns": _range_stats(timestamps_ns),
        "observation.hand_keypoints_3d": _safe_stats(
            episode.observation.hand_keypoints_3d, validity=valid_3d
        ),
        "observation.hand_joint_valid_3d": _boolean_stats(valid_3d),
        "observation.hand_keypoints_2d": _safe_stats(
            episode.observation.hand_keypoints_2d, validity=valid_2d
        ),
        "observation.hand_joint_valid_2d": _boolean_stats(valid_2d),
        "task_index": _range_stats(np.zeros(frame_count, dtype=np.int64)),
        "subtask_index": _range_stats(_subtask_indices(episode)),
    }
    quality = episode.supplier_evidence.hand_quality
    if quality is not None and quality.provided:
        if quality.raw_value is not None and np.issubdtype(
            quality.raw_value.dtype, np.number
        ):
            stats["supplier.hand_quality.raw_value"] = _safe_stats(quality.raw_value)
        if quality.normalized_score is not None:
            stats["supplier.hand_quality.normalized_score"] = _safe_stats(
                quality.normalized_score
            )
    return stats


def _semantics(episode: CanonicalQcEpisode) -> dict[str, object]:
    calibration = episode.calibration
    payload: dict[str, object] = {
        "episode_index": 0,
        "asset_id": episode.identity.asset_id,
        "batch_id": episode.identity.batch_id,
        "supplier_id": episode.identity.supplier_id,
        "task_index": 0,
        "fps_num": episode.time_axis.fps_num,
        "fps_den": episode.time_axis.fps_den,
        "joint_topology": episode.observation.joint_topology,
        "coordinate_frame_3d": episode.observation.coordinate_frame_3d,
        "length_unit": episode.observation.length_unit,
        "coordinate_space_2d": episode.observation.coordinate_space_2d,
        "calibration": {
            "intrinsic_matrix": calibration.intrinsic_matrix.tolist(),
            "distortion_model": calibration.distortion_model,
            "distortion_coefficients": calibration.distortion_coefficients.tolist(),
            "image_width_px": calibration.image_width_px,
            "image_height_px": calibration.image_height_px,
            "camera_axes": calibration.camera_axes,
            "pixel_origin": calibration.pixel_origin,
        },
        "scene_id": episode.semantics.scene_id,
        "task_id": episode.semantics.task_id,
        "task_category": episode.semantics.task_category,
        "task_cn": episode.semantics.task_cn,
        "task_en": episode.semantics.task_en,
        "description_cn": episode.semantics.description_cn,
        "description_en": episode.semantics.description_en,
        "subtask_sequence": [
            {
                "subtask_index": index,
                "subtask_id": item.subtask_id,
                "start_frame": item.start_frame,
                "end_frame_exclusive": item.end_frame_exclusive,
                "description_cn": item.description_cn,
                "description_en": item.description_en,
            }
            for index, item in enumerate(episode.semantics.subtask_sequence)
        ],
    }
    quality = episode.supplier_evidence.hand_quality
    if quality is not None:
        state: dict[str, object] = {"provided": quality.provided}
        if quality.provided:
            state["mapping_version"] = quality.mapping_version
        payload["supplier_hand_quality"] = state
    return payload


def _subtask_table(episode: CanonicalQcEpisode) -> pa.Table:
    rows = [
        {
            "episode_index": 0,
            "subtask_index": index,
            "subtask_id": item.subtask_id,
            "start_frame": item.start_frame,
            "end_frame_exclusive": item.end_frame_exclusive,
            "description_cn": item.description_cn,
            "description_en": item.description_en,
        }
        for index, item in enumerate(episode.semantics.subtask_sequence)
    ]
    schema = pa.schema(
        [
            pa.field("episode_index", pa.int64()),
            pa.field("subtask_index", pa.int64()),
            pa.field("subtask_id", pa.string()),
            pa.field("start_frame", pa.int64()),
            pa.field("end_frame_exclusive", pa.int64()),
            pa.field("description_cn", pa.string()),
            pa.field("description_en", pa.string()),
        ]
    )
    return pa.Table.from_pylist(rows, schema=schema)


def _write_parquet(root: _SafeRoot, relative: str, table: pa.Table) -> None:
    temp_relative = str(
        PurePosixPath(relative).with_name(f".parquet-{secrets.token_hex(16)}.tmp")
    )
    descriptor = root.open_exclusive(temp_relative, read_write=True)
    try:
        with os.fdopen(os.dup(descriptor), "wb") as stream:
            sink = pa.PythonFile(stream, mode="w")
            pq.write_table(table, sink, compression="snappy", use_dictionary=True)
            sink.close()
    finally:
        os.close(descriptor)
    root.replace(temp_relative, relative)


def _write_dataset_files(root: _SafeRoot, episode: CanonicalQcEpisode, probed: Any) -> None:
    table = _data_table(episode)
    _write_parquet(root, "data/chunk-000/file-000.parquet", table)
    task_table = pa.Table.from_pandas(
        pd.DataFrame(
            {"task_index": np.asarray([0], dtype=np.int64)},
            index=pd.Index([episode.semantics.task_en], name="task"),
        ),
        preserve_index=True,
    )
    _write_parquet(root, "meta/tasks.parquet", task_table)
    _write_parquet(root, "meta/subtask.parquet", _subtask_table(episode))
    duration = float(
        Fraction(
            episode.time_axis.frame_count * episode.time_axis.fps_den,
            episode.time_axis.fps_num,
        )
    )
    episode_row = {
        "episode_index": 0,
        "tasks": [episode.semantics.task_en],
        "length": episode.time_axis.frame_count,
        "data/chunk_index": 0,
        "data/file_index": 0,
        "dataset_from_index": 0,
        "dataset_to_index": episode.time_axis.frame_count,
        "videos/observation.images.main/chunk_index": 0,
        "videos/observation.images.main/file_index": 0,
        "videos/observation.images.main/from_timestamp": 0.0,
        "videos/observation.images.main/to_timestamp": duration,
        "meta/episodes/chunk_index": 0,
        "meta/episodes/file_index": 0,
    }
    _write_parquet(
        root,
        "meta/episodes/chunk-000/file-000.parquet",
        pa.Table.from_pylist([episode_row]),
    )
    fps = episode.time_axis.fps_num / episode.time_axis.fps_den
    info = {
        "codebase_version": "v3.0",
        "fps": int(fps) if fps.is_integer() else fps,
        "fps_num": episode.time_axis.fps_num,
        "fps_den": episode.time_axis.fps_den,
        "features": _features(episode, probed, table),
        "total_episodes": 1,
        "total_frames": episode.time_axis.frame_count,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "data_path": _DATA_PATH,
        "video_path": _VIDEO_PATH,
        "robot_type": "human_ego_hand_pose",
        "splits": {"train": "0:1"},
    }
    _write_bytes(root, "meta/info.json", _json_bytes(info))
    _write_bytes(root, "meta/stats.json", _json_bytes(_stats(episode)))
    _write_bytes(root, "meta/episode_semantics.jsonl", _json_bytes(_semantics(episode)))


def _manifest(
    plan: PublishPlan, root: _SafeRoot, video: VideoMaterialization
) -> ReleaseManifest:
    toolchain = current_toolchain()
    expected_version = (
        f"lerobot_v3_curated.v1+toolchain.{toolchain.fingerprint[:16]}"
    )
    if (
        plan.publisher_version != PUBLISHER_VERSION
        or plan.publisher_version != expected_version
    ):
        _reject(
            "publisher_version",
            "publish plan was created for a different writer toolchain",
            retryable=True,
        )
    files = tuple(
        ManifestFile(relative, root.file_stat(relative).st_size, root.sha256(relative))
        for relative in _PAYLOAD_PATHS
    )
    return ReleaseManifest(
        schema_version="curated_lerobot_v3_release_manifest.v1",
        release_id=plan.release_id,
        publisher_version=plan.publisher_version,
        asset_id=plan.request.episode.identity.asset_id,
        canonical_revision=plan.request.canonical_revision,
        semantic_fingerprint=plan.semantic_fingerprint,
        source_fingerprint=plan.source_fingerprint,
        qc_report_revision=plan.qc_report_revision,
        qc_report_sha256=plan.qc_report_sha256,
        files=files,
        video_materialization=video,
        toolchain=toolchain,
    )


def _manifest_payload(manifest: ReleaseManifest) -> dict[str, object]:
    payload = asdict(manifest)
    payload["files"] = [asdict(item) for item in manifest.files]
    if manifest.video_materialization is not None:
        payload["video_materialization"] = asdict(manifest.video_materialization)
    return payload


def _write_inventory(
    root: _SafeRoot, manifest: ReleaseManifest
) -> tuple[str, str]:
    _write_bytes(root, "release_manifest.json", _json_bytes(_manifest_payload(manifest)))
    manifest_sha = root.sha256("release_manifest.json")
    rows = [
        f"{item.sha256}  {item.relative_path}\n" for item in manifest.files
    ]
    rows.append(f"{manifest_sha}  release_manifest.json\n")
    _write_bytes(root, "checksums.sha256", "".join(sorted(rows)).encode("utf-8"))
    expected = set(_PAYLOAD_PATHS) | {"release_manifest.json", "checksums.sha256"}
    observed = root.file_paths()
    if observed != expected:
        _reject("staging", "staging contains missing or unregistered files")
    return manifest_sha, root.sha256("checksums.sha256")


def _open_staging_transaction(
    plan: PublishPlan, staging_root: Path
) -> _StagingTransaction:
    expected = plan.request.release_root / ".staging"
    candidate = Path(staging_root)
    if not candidate.is_absolute() or candidate != expected or ".." in candidate.parts:
        _reject("staging_root", "must be exactly <release_root>/.staging")
    if candidate.is_symlink():
        _reject("staging_root", "symlink staging roots are forbidden")
    release_root = plan.request.release_root
    try:
        release_root.mkdir(parents=True, exist_ok=True)
        release_fd = os.open(release_root, _directory_flags())
    except OSError as exc:
        _reject("staging_root", f"cannot securely open release_root: {exc}")
    staging_fd = -1
    root_fd = -1
    transaction_id = f"tx-{secrets.token_hex(16)}"
    try:
        release_stat = os.fstat(release_fd)
        if _identity(release_stat) != _identity(
            os.stat(release_root, follow_symlinks=False)
        ):
            _reject("staging_root", "release_root identity changed", retryable=True)
        try:
            os.mkdir(".staging", mode=0o700, dir_fd=release_fd)
        except FileExistsError:
            pass
        staging_fd = os.open(".staging", _directory_flags(), dir_fd=release_fd)
        os.fchmod(staging_fd, 0o700)
        staging_stat = os.fstat(staging_fd)
        if staging_stat.st_dev != release_stat.st_dev:
            _reject("staging_root", "staging must use the release filesystem")
        if _identity(staging_stat) != _identity(
            os.stat(candidate, follow_symlinks=False)
        ):
            _reject("staging_root", "staging root identity changed", retryable=True)
        os.mkdir(transaction_id, mode=0o700, dir_fd=staging_fd)
        root_fd = os.open(transaction_id, _directory_flags(), dir_fd=staging_fd)
        os.fchmod(root_fd, 0o700)
        root_stat = os.fstat(root_fd)
        actual_root = candidate / transaction_id
        if _identity(root_stat) != _identity(
            os.stat(actual_root, follow_symlinks=False)
        ):
            _reject("staging_root", "transaction identity changed", retryable=True)
        return _StagingTransaction(
            release_root=release_root,
            staging_root=candidate,
            actual_root=actual_root,
            transaction_id=transaction_id,
            release_fd=release_fd,
            staging_fd=staging_fd,
            root_fd=root_fd,
            release_identity=_identity(release_stat),
            staging_identity=_identity(staging_stat),
            root_identity=_identity(root_stat),
        )
    except BaseException:
        if staging_fd >= 0:
            _remove_tree_at(staging_fd, transaction_id)
        for descriptor in (root_fd, staging_fd, release_fd):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        raise


def write_staging(plan: PublishPlan, staging_root: Path) -> StagedRelease:
    """Revalidate and materialize one deterministic release in a private staging tx."""

    if type(plan) is not PublishPlan:
        _reject("plan", "must be an exact PublishPlan")
    revalidate_publish_plan(plan)
    transaction = _open_staging_transaction(plan, Path(staging_root))
    root = transaction.safe_root
    try:
        _prepare_private_directories(root)
        video, probed = _materialize_video(plan, root)
        transaction.verify_paths()
        _write_dataset_files(root, plan.request.episode, probed)
        transaction.verify_paths()
        manifest = _manifest(plan, root, video)
        manifest_sha, checksums_sha = _write_inventory(root, manifest)
        revalidate_publish_plan(plan)
        transaction.verify_paths()
        return StagedRelease(
            plan=plan,
            root=transaction.actual_root,
            transaction_id=transaction.transaction_id,
            manifest=manifest,
            manifest_sha256=manifest_sha,
            checksums_sha256=checksums_sha,
        )
    except BaseException:
        transaction.cleanup()
        raise
    finally:
        transaction.close()


__all__ = ["write_staging"]
