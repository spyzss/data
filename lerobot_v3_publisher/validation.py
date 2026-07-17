"""Independent nofollow validation for one staged Curated LeRobot v3 release."""

from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
from typing import Any, Iterator, NoReturn

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from annotation.lerobot_v3_dataset import LeRobotV3Dataset
from canonical_qc.contracts import CanonicalQcEpisode
from canonical_qc.errors import CanonicalInputError
from canonical_qc.hand_quality_encoding import (
    STATUS_ENCODING_SIDECAR,
    decode_status,
    encode_raw_value_sidecar,
)
from canonical_qc.video_probe import probe_video

from .contracts import (
    CanonicalDiagnostic,
    ManifestFile,
    PublishPrerequisiteError,
    ReleaseManifest,
    StagedRelease,
    ValidationReport,
    VideoMaterialization,
)
from .toolchain import WriterToolchain, current_toolchain


_STAGE = "publish_validation"
_MAX_TIMESTAMP_DELTA_NS = 1
_MAX_VIDEO_TIMESTAMP_DELTA_NS = 1_000_000
_INVENTORY_FILES = frozenset({"release_manifest.json", "checksums.sha256"})
_OFFICIAL_READER_PACKAGES = {
    "lerobot": "0.6.0",
    "datasets": "4.8.5",
    "torch": "2.11.0",
    "torchvision": "0.26.0",
    "torchcodec": "0.11.1",
    "av": "15.1.0",
    "numpy": "2.2.6",
    "pyarrow": "24.0.0",
    "pandas": "2.3.3",
    "huggingface-hub": "1.23.0",
}
_OFFICIAL_READER_VERSIONS = {
    **_OFFICIAL_READER_PACKAGES,
    "python": "3.12.13",
    "platform": "macOS-26.5.2-arm64-arm-64bit",
    "video_backend": "pyav",
}
_OFFICIAL_READER_TIMEOUT_SECONDS = 180
OFFICIAL_READER_VERSION = _OFFICIAL_READER_PACKAGES["lerobot"]


def _reject(field: str | None, message: str, *, retryable: bool = False) -> NoReturn:
    raise PublishPrerequisiteError(
        CanonicalDiagnostic("validation_failed", _STAGE, field, message, retryable)
    )


def _directory_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


def _normalized(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        _reject("manifest.files", f"unsafe artifact path {value!r}")
    return path


class _ReleaseReader:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.descriptor = -1
        try:
            self.descriptor = os.open(self.root, _directory_flags())
            self.identity = self._identity(os.fstat(self.descriptor))
            self.verify_identity()
        except PublishPrerequisiteError:
            if self.descriptor >= 0:
                os.close(self.descriptor)
            raise
        except OSError as exc:
            if self.descriptor >= 0:
                os.close(self.descriptor)
            _reject("staged.root", f"cannot securely open release root: {exc}", retryable=True)

    @staticmethod
    def _identity(value: os.stat_result) -> tuple[int, int]:
        return value.st_dev, value.st_ino

    def close(self) -> None:
        os.close(self.descriptor)

    def verify_identity(self) -> None:
        try:
            current = os.stat(self.root, follow_symlinks=False)
        except OSError as exc:
            _reject("staged.root", f"release root path changed: {exc}", retryable=True)
        if self._identity(current) != self.identity or not stat.S_ISDIR(current.st_mode):
            _reject("staged.root", "release root identity changed", retryable=True)

    @contextmanager
    def open_file(self, relative: str) -> Iterator[int]:
        path = _normalized(relative)
        descriptor = os.dup(self.descriptor)
        file_descriptor = -1
        try:
            for part in path.parts[:-1]:
                child = os.open(part, _directory_flags(), dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            file_descriptor = os.open(
                path.name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            metadata = os.fstat(file_descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                _reject(relative, "artifact must be a regular file")
            yield file_descriptor
        except PublishPrerequisiteError:
            raise
        except OSError as exc:
            _reject(relative, f"cannot securely open artifact: {exc}", retryable=True)
        finally:
            if file_descriptor >= 0:
                os.close(file_descriptor)
            os.close(descriptor)

    def read_bytes(self, relative: str) -> bytes:
        with self.open_file(relative) as descriptor:
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks)

    def sha256(self, relative: str) -> str:
        digest = hashlib.sha256()
        with self.open_file(relative) as descriptor:
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def size(self, relative: str) -> int:
        with self.open_file(relative) as descriptor:
            return os.fstat(descriptor).st_size

    def parquet(self, relative: str) -> pa.Table:
        with self.open_file(relative) as descriptor:
            with os.fdopen(os.dup(descriptor), "rb") as stream:
                return pq.read_table(pa.PythonFile(stream, mode="r"))

    def files(self) -> set[str]:
        result: set[str] = set()

        def visit(directory_fd: int, prefix: PurePosixPath) -> None:
            with os.scandir(directory_fd) as entries:
                for entry in entries:
                    relative = prefix / entry.name
                    if entry.is_symlink():
                        _reject(relative.as_posix(), "symlink artifacts are forbidden")
                    if entry.is_dir(follow_symlinks=False):
                        child = os.open(entry.name, _directory_flags(), dir_fd=directory_fd)
                        try:
                            visit(child, relative)
                        finally:
                            os.close(child)
                    elif entry.is_file(follow_symlinks=False):
                        result.add(relative.as_posix())
                    else:
                        _reject(relative.as_posix(), "non-regular artifacts are forbidden")

        descriptor = os.dup(self.descriptor)
        try:
            visit(descriptor, PurePosixPath())
        except OSError as exc:
            _reject("inventory", f"cannot enumerate release: {exc}", retryable=True)
        finally:
            os.close(descriptor)
        return result


def _manifest(payload: bytes) -> ReleaseManifest:
    try:
        value = json.loads(payload)
        files = tuple(ManifestFile(**row) for row in value.pop("files"))
        video_value = value.pop("video_materialization")
        toolchain_value = value.pop("toolchain")
        video = VideoMaterialization(**video_value)
        toolchain = WriterToolchain(**toolchain_value)
        return ReleaseManifest(
            **value,
            files=files,
            video_materialization=video,
            toolchain=toolchain,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        _reject("release_manifest.json", f"invalid manifest: {exc}")


def _checksum_rows(payload: bytes) -> dict[str, str]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        _reject("checksums.sha256", f"must be UTF-8: {exc}")
    rows: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            _reject("checksums.sha256", "invalid checksum row")
        digest, relative = parts
        if relative in rows:
            _reject("checksums.sha256", "duplicate checksum path")
        _normalized(relative)
        rows[relative] = digest
    return rows


def _expected_subtask_indices(episode: CanonicalQcEpisode) -> np.ndarray:
    indices = np.full(episode.time_axis.frame_count, -1, dtype=np.int64)
    for index, item in enumerate(episode.semantics.subtask_sequence):
        indices[item.start_frame : item.end_frame_exclusive] = index
    return indices


def _fixed_arrow_type(values: np.ndarray) -> pa.DataType:
    array = np.asarray(values)
    arrow_type: pa.DataType = (
        pa.string()
        if array.dtype.kind in {"U", "S"}
        else pa.from_numpy_dtype(array.dtype)
    )
    for size in reversed(array.shape[1:]):
        arrow_type = pa.list_(arrow_type, int(size))
    return arrow_type


def _validate_frame_data(reader: _ReleaseReader, expected: CanonicalQcEpisode) -> None:
    table = reader.parquet("data/chunk-000/file-000.parquet")
    frame_count = expected.time_axis.frame_count
    if table.num_rows != frame_count:
        _reject("data", "frame row count differs from Canonical")
    required = {
        "index",
        "episode_index",
        "frame_index",
        "timestamp",
        "timestamp_ns",
        "observation.hand_keypoints_3d",
        "observation.hand_joint_valid_3d",
        "observation.hand_keypoints_2d",
        "observation.hand_joint_valid_2d",
        "task_index",
        "subtask_index",
    }
    quality = expected.supplier_evidence.hand_quality
    optional: set[str] = set()
    if quality is not None and quality.provided:
        if quality.raw_value is not None and quality.raw_value.dtype.kind not in {"U", "S"}:
            optional.add("supplier.hand_quality.raw_value")
        if quality.normalized_score is not None:
            optional.add("supplier.hand_quality.normalized_score")
        if quality.status is not None:
            optional.add("supplier.hand_quality.status")
    frame_extensions = {
        item.published_name: item
        for item in expected.supplier_extensions.fields
        if item.time_alignment == "frame"
    }
    optional.update(frame_extensions)
    if set(table.column_names) != required | optional:
        _reject("data", "required frame columns are missing")
    exact_types = {
        "index": pa.int64(),
        "episode_index": pa.int64(),
        "frame_index": pa.int64(),
        "timestamp": pa.float64(),
        "timestamp_ns": pa.int64(),
        "observation.hand_keypoints_3d": pa.list_(
            pa.list_(pa.list_(pa.float32(), 3), 21), 2
        ),
        "observation.hand_joint_valid_3d": pa.list_(pa.list_(pa.bool_(), 21), 2),
        "observation.hand_keypoints_2d": pa.list_(
            pa.list_(pa.list_(pa.float32(), 2), 21), 2
        ),
        "observation.hand_joint_valid_2d": pa.list_(pa.list_(pa.bool_(), 21), 2),
        "task_index": pa.int64(),
        "subtask_index": pa.int64(),
    }
    for name, expected_type in exact_types.items():
        if table.schema.field(name).type != expected_type:
            _reject(name, f"expected Arrow type {expected_type}")
    sequence = np.arange(frame_count, dtype=np.int64)
    for name in ("index", "frame_index"):
        if not np.array_equal(np.asarray(table[name].to_pylist()), sequence):
            _reject(name, "must equal range(T)")
    if np.any(np.asarray(table["episode_index"].to_pylist()) != 0):
        _reject("episode_index", "must be zero for the single release episode")
    if np.any(np.asarray(table["task_index"].to_pylist()) != 0):
        _reject("task_index", "must be zero for the single registered task")
    timestamps_ns = np.asarray(table["timestamp_ns"].to_pylist(), dtype=np.int64)
    if not np.array_equal(timestamps_ns, expected.time_axis.timestamps_ns):
        _reject("timestamp_ns", "differs from Canonical")
    float_timestamp = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
    expected_float = (timestamps_ns - timestamps_ns[0]) / 1_000_000_000
    if np.max(np.abs((float_timestamp - expected_float) * 1_000_000_000)) > _MAX_TIMESTAMP_DELTA_NS:
        _reject("timestamp", "differs from authoritative timestamp_ns by more than 1 ns")
    arrays = (
        ("observation.hand_keypoints_3d", expected.observation.hand_keypoints_3d, True),
        ("observation.hand_joint_valid_3d", expected.observation.hand_joint_valid_3d, False),
        ("observation.hand_keypoints_2d", expected.observation.hand_keypoints_2d, True),
        ("observation.hand_joint_valid_2d", expected.observation.hand_joint_valid_2d, False),
    )
    for name, expected_array, allow_nan in arrays:
        observed = np.asarray(table[name].to_pylist(), dtype=expected_array.dtype)
        if observed.shape != expected_array.shape or not np.array_equal(
            observed, expected_array, equal_nan=allow_nan
        ):
            _reject(name, "shape, dtype, or values differ from Canonical")
    observed_subtasks = np.asarray(table["subtask_index"].to_pylist(), dtype=np.int64)
    if not np.array_equal(observed_subtasks, _expected_subtask_indices(expected)):
        _reject("subtask_index", "does not match Canonical half-open boundaries")
    if quality is not None and quality.provided:
        for suffix, expected_array in (
            ("raw_value", quality.raw_value),
            ("normalized_score", quality.normalized_score),
            ("status", quality.status),
        ):
            if expected_array is None:
                continue
            if suffix == "raw_value" and expected_array.dtype.kind in {"U", "S"}:
                continue
            name = f"supplier.hand_quality.{suffix}"
            value_type = (
                pa.uint8()
                if suffix == "status"
                else pa.from_numpy_dtype(expected_array.dtype)
            )
            expected_type = pa.list_(value_type, 2)
            if table.schema.field(name).type != expected_type:
                _reject(name, f"expected Arrow type {expected_type}")
            observed_wire = np.asarray(
                table[name].to_pylist(), dtype=np.uint8 if suffix == "status" else expected_array.dtype
            )
            observed = decode_status(observed_wire) if suffix == "status" else observed_wire
            if observed.shape != expected_array.shape or not np.array_equal(
                observed, expected_array, equal_nan=expected_array.dtype.kind == "f"
            ):
                _reject(name, "differs from Canonical supplier Evidence")
    for name, extension in frame_extensions.items():
        expected_type = _fixed_arrow_type(extension.values)
        if table.schema.field(name).type != expected_type:
            _reject(name, f"expected Arrow type {expected_type}")
        observed = np.asarray(table[name].to_pylist(), dtype=extension.values.dtype)
        if observed.shape != extension.values.shape or not np.array_equal(
            observed,
            extension.values,
            equal_nan=extension.values.dtype.kind == "f",
        ):
            _reject(name, "differs from Canonical supplier extension")


def _validate_subtasks(reader: _ReleaseReader, expected: CanonicalQcEpisode) -> None:
    rows = reader.parquet("meta/subtask.parquet").to_pylist()
    wanted = [
        {
            "episode_index": 0,
            "subtask_index": index,
            "subtask_id": item.subtask_id,
            "start_frame": item.start_frame,
            "end_frame_exclusive": item.end_frame_exclusive,
            "description_cn": item.description_cn,
            "description_en": item.description_en,
        }
        for index, item in enumerate(expected.semantics.subtask_sequence)
    ]
    if rows != wanted:
        _reject("meta/subtask.parquet", "differs from Canonical semantics")


def _validate_episode_metadata(
    reader: _ReleaseReader, expected: CanonicalQcEpisode
) -> None:
    rows = reader.parquet("meta/episodes/chunk-000/file-000.parquet").to_pylist()
    if len(rows) != 1:
        _reject("meta/episodes", "must contain exactly one episode row")
    row = rows[0]
    duration = (
        expected.time_axis.frame_count
        * expected.time_axis.fps_den
        / expected.time_axis.fps_num
    )
    wanted = {
        "episode_index": 0,
        "length": expected.time_axis.frame_count,
        "dataset_from_index": 0,
        "dataset_to_index": expected.time_axis.frame_count,
        "tasks": [expected.semantics.task_en],
        "data/chunk_index": 0,
        "data/file_index": 0,
        "videos/observation.images.main/chunk_index": 0,
        "videos/observation.images.main/file_index": 0,
        "videos/observation.images.main/from_timestamp": 0.0,
        "videos/observation.images.main/to_timestamp": duration,
        "meta/episodes/chunk_index": 0,
        "meta/episodes/file_index": 0,
    }
    if set(row) != set(wanted):
        _reject("meta/episodes", "columns differ from the single-episode contract")
    for name, value in wanted.items():
        observed = row[name]
        if isinstance(value, float):
            matches = abs(observed - value) <= 1e-12
        else:
            matches = observed == value
        if not matches:
            _reject(f"meta/episodes.{name}", "differs from Canonical")


def _stats_values(
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
    return {
        "min": np.where(np.isfinite(minimum), minimum, 0).tolist(),
        "max": np.where(np.isfinite(maximum), maximum, 0).tolist(),
        "mean": mean.tolist(),
        "std": np.sqrt(variance).tolist(),
        "count": [int(values.shape[0])],
    }


def _range_values(array: np.ndarray) -> dict[str, object]:
    values = np.asarray(array)
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "count": [int(values.shape[0])],
    }


def _expected_stats(expected: CanonicalQcEpisode) -> dict[str, object]:
    frame_count = expected.time_axis.frame_count
    indices = np.arange(frame_count, dtype=np.int64)
    timestamps_ns = np.asarray(expected.time_axis.timestamps_ns, dtype=np.int64)
    timestamps = (timestamps_ns - timestamps_ns[0]) / 1_000_000_000
    valid_3d = expected.observation.hand_joint_valid_3d
    valid_2d = expected.observation.hand_joint_valid_2d
    stats: dict[str, object] = {
        "index": _range_values(indices),
        "episode_index": _range_values(np.zeros(frame_count, dtype=np.int64)),
        "frame_index": _range_values(indices),
        "timestamp": _range_values(timestamps),
        "timestamp_ns": _range_values(timestamps_ns),
        "observation.hand_keypoints_3d": _stats_values(
            expected.observation.hand_keypoints_3d, validity=valid_3d
        ),
        "observation.hand_joint_valid_3d": {
            **_range_values(valid_3d.astype(np.int8)),
            "true_count": np.count_nonzero(valid_3d, axis=0).tolist(),
        },
        "observation.hand_keypoints_2d": _stats_values(
            expected.observation.hand_keypoints_2d, validity=valid_2d
        ),
        "observation.hand_joint_valid_2d": {
            **_range_values(valid_2d.astype(np.int8)),
            "true_count": np.count_nonzero(valid_2d, axis=0).tolist(),
        },
        "task_index": _range_values(np.zeros(frame_count, dtype=np.int64)),
        "subtask_index": _range_values(_expected_subtask_indices(expected)),
    }
    quality = expected.supplier_evidence.hand_quality
    if quality is not None and quality.provided:
        if quality.raw_value is not None and np.issubdtype(
            quality.raw_value.dtype, np.number
        ):
            stats["supplier.hand_quality.raw_value"] = _stats_values(quality.raw_value)
        if quality.normalized_score is not None:
            stats["supplier.hand_quality.normalized_score"] = _stats_values(
                quality.normalized_score
            )
    for extension in expected.supplier_extensions.fields:
        if extension.time_alignment != "frame":
            continue
        values = extension.values
        if np.issubdtype(values.dtype, np.bool_):
            stats[extension.published_name] = {
                **_range_values(values.astype(np.int8)),
                "true_count": np.count_nonzero(values, axis=0).tolist(),
            }
        elif np.issubdtype(values.dtype, np.number):
            stats[extension.published_name] = _stats_values(values)
    return stats


def _validate_registered_metadata(
    reader: _ReleaseReader, expected: CanonicalQcEpisode, probed: Any
) -> None:
    try:
        info = json.loads(reader.read_bytes("meta/info.json"))
        stats = json.loads(reader.read_bytes("meta/stats.json"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _reject("meta", f"invalid JSON metadata: {exc}")
    frame_count = expected.time_axis.frame_count
    fps = expected.time_axis.fps_num / expected.time_axis.fps_den
    default = {"shape": [1], "names": None}
    features: dict[str, object] = {
        "timestamp": {"dtype": "float64", **default},
        "frame_index": {"dtype": "int64", **default},
        "episode_index": {"dtype": "int64", **default},
        "index": {"dtype": "int64", **default},
        "task_index": {"dtype": "int64", **default},
        "timestamp_ns": {"dtype": "int64", **default, "unit": "nanosecond"},
        "subtask_index": {"dtype": "int64", **default},
        "observation.hand_keypoints_3d": {
            "dtype": "float32",
            "shape": [2, 21, 3],
            "names": None,
            "hand_order": ["left", "right"],
            "joint_topology": "egodata_hand21.v1",
            "coordinate_frame": "camera:main",
            "unit": "meter",
        },
        "observation.hand_joint_valid_3d": {
            "dtype": "bool",
            "shape": [2, 21],
            "names": None,
        },
        "observation.hand_keypoints_2d": {
            "dtype": "float32",
            "shape": [2, 21, 2],
            "names": None,
            "hand_order": ["left", "right"],
            "joint_topology": "egodata_hand21.v1",
            "coordinate_space": "pixel",
            "unit": "pixel",
        },
        "observation.hand_joint_valid_2d": {
            "dtype": "bool",
            "shape": [2, 21],
            "names": None,
        },
        "observation.images.main": {
            "dtype": "video",
            "shape": [expected.main_video.height_px, expected.main_video.width_px, 3],
            "names": ["height", "width", "channels"],
            "camera_id": "main",
            "camera_role": "ego",
            "info": {
                "video.height": probed.height_px,
                "video.width": probed.width_px,
                "video.codec": probed.codec,
                "video.pix_fmt": probed.pixel_format,
                "video.fps": probed.fps_num / probed.fps_den,
                "video.channels": 3,
                "audio.has_audio": False,
                "is_depth_map": False,
            },
        },
    }
    quality = expected.supplier_evidence.hand_quality
    if quality is not None and quality.provided:
        for suffix, value in (
            ("raw_value", quality.raw_value),
            ("normalized_score", quality.normalized_score),
            ("status", quality.status),
        ):
            if value is not None:
                if suffix == "raw_value" and value.dtype.kind in {"U", "S"}:
                    continue
                dtype = (
                    "uint8"
                    if suffix == "status"
                    else str(value.dtype)
                )
                features[f"supplier.hand_quality.{suffix}"] = {
                    "dtype": dtype,
                    "shape": [2],
                    "names": None,
                }
    for extension in expected.supplier_extensions.fields:
        if extension.time_alignment != "frame":
            continue
        values = extension.values
        features[extension.published_name] = {
            "dtype": "string" if values.dtype.kind in {"U", "S"} else values.dtype.name,
            "shape": list(values.shape[1:]) or [1],
            "names": None,
        }
    wanted_info = {
        "codebase_version": "v3.0",
        "fps": int(fps) if fps.is_integer() else fps,
        "fps_num": expected.time_axis.fps_num,
        "fps_den": expected.time_axis.fps_den,
        "features": features,
        "total_episodes": 1,
        "total_frames": frame_count,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "robot_type": "human_ego_hand_pose",
        "splits": {"train": "0:1"},
    }
    if info != wanted_info:
        _reject("meta/info.json", "differs from complete publisher metadata contract")
    if stats != _expected_stats(expected):
        _reject("meta/stats.json", "statistics differ from Canonical values")
    tasks = reader.parquet("meta/tasks.parquet").to_pylist()
    if tasks != [{"task_index": 0, "task": expected.semantics.task_en}]:
        _reject("meta/tasks.parquet", "task registry differs from Canonical")


def _validate_semantics(reader: _ReleaseReader, expected: CanonicalQcEpisode) -> None:
    try:
        lines = reader.read_bytes("meta/episode_semantics.jsonl").decode("utf-8").splitlines()
        payload = json.loads(lines[0]) if len(lines) == 1 else None
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _reject("meta/episode_semantics.jsonl", f"invalid semantics JSON: {exc}")
    if not isinstance(payload, dict):
        _reject("meta/episode_semantics.jsonl", "must contain exactly one object")
    calibration = expected.calibration
    wanted: dict[str, object] = {
        "episode_index": 0,
        "task_index": 0,
        "asset_id": expected.identity.asset_id,
        "batch_id": expected.identity.batch_id,
        "supplier_id": expected.identity.supplier_id,
        "scene_id": expected.semantics.scene_id,
        "task_id": expected.semantics.task_id,
        "task_category": expected.semantics.task_category,
        "task_cn": expected.semantics.task_cn,
        "task_en": expected.semantics.task_en,
        "description_cn": expected.semantics.description_cn,
        "description_en": expected.semantics.description_en,
        "fps_num": expected.time_axis.fps_num,
        "fps_den": expected.time_axis.fps_den,
        "joint_topology": expected.observation.joint_topology,
        "coordinate_frame_3d": expected.observation.coordinate_frame_3d,
        "length_unit": expected.observation.length_unit,
        "coordinate_space_2d": expected.observation.coordinate_space_2d,
        "calibration": {
            "intrinsic_matrix": calibration.intrinsic_matrix.tolist(),
            "distortion_model": calibration.distortion_model,
            "distortion_coefficients": calibration.distortion_coefficients.tolist(),
            "image_width_px": calibration.image_width_px,
            "image_height_px": calibration.image_height_px,
            "camera_axes": calibration.camera_axes,
            "pixel_origin": calibration.pixel_origin,
        },
        "subtask_sequence": [
            {
                "subtask_index": index,
                "subtask_id": item.subtask_id,
                "start_frame": item.start_frame,
                "end_frame_exclusive": item.end_frame_exclusive,
                "description_cn": item.description_cn,
                "description_en": item.description_en,
            }
            for index, item in enumerate(expected.semantics.subtask_sequence)
        ],
    }
    quality = expected.supplier_evidence.hand_quality
    if quality is not None:
        state: dict[str, object] = {"provided": quality.provided}
        if quality.provided:
            state["mapping_version"] = quality.mapping_version
            state["status_encoding"] = STATUS_ENCODING_SIDECAR
            if quality.raw_value is not None and quality.raw_value.dtype.kind in {"U", "S"}:
                state["raw_value_sidecar"] = encode_raw_value_sidecar(
                    quality.raw_value
                )
        wanted["supplier_hand_quality"] = state
    if expected.batch_metadata is not None:
        batch = expected.batch_metadata
        wanted["batch_metadata"] = {
            "schema_version": batch.schema_version,
            "batch_id": batch.batch_id,
            "supplier_id": batch.supplier_id,
            "dataset_attributes": batch.dataset_attributes,
            "content_sha256": batch.content_sha256,
        }
    extension_rows: list[dict[str, object]] = []
    for extension in expected.supplier_extensions.fields:
        row: dict[str, object] = {
            "published_name": extension.published_name,
            "source_path": extension.source_path,
            "time_alignment": extension.time_alignment,
            "dtype": extension.values.dtype.str,
            "shape": list(extension.values.shape),
            "metadata": extension.metadata,
            "storage": (
                "lerobot_feature"
                if extension.time_alignment == "frame"
                else "semantic_sidecar"
            ),
        }
        if extension.time_alignment != "frame":
            contiguous = np.ascontiguousarray(extension.values)
            row["payload"] = {
                "encoding": "ndarray_base64.v1",
                "dtype": contiguous.dtype.str,
                "shape": list(contiguous.shape),
                "data": base64.b64encode(contiguous.tobytes(order="C")).decode("ascii"),
            }
        extension_rows.append(row)
    wanted["supplier_extensions"] = extension_rows
    if payload != wanted:
        _reject("meta/episode_semantics.jsonl", "differs from complete Canonical semantics")


def _validate_video(reader: _ReleaseReader, expected: CanonicalQcEpisode) -> Any:
    relative = "videos/observation.images.main/chunk-000/file-000.mp4"
    try:
        with reader.open_file(relative) as descriptor:
            probed = probe_video(Path(relative), file_descriptor=descriptor)
    except CanonicalInputError as exc:
        _reject("video", f"cannot probe staged video: {exc}")
    expected_pts = expected.time_axis.timestamps_ns - expected.time_axis.timestamps_ns[0]
    observed_pts = np.asarray(probed.timestamps_ns, dtype=np.int64)
    if (
        probed.frame_count != expected.time_axis.frame_count
        or probed.width_px != expected.main_video.width_px
        or probed.height_px != expected.main_video.height_px
        or (probed.fps_num, probed.fps_den)
        != (expected.time_axis.fps_num, expected.time_axis.fps_den)
        or len(observed_pts) != len(expected_pts)
        or np.max(np.abs(observed_pts - expected_pts)) > _MAX_VIDEO_TIMESTAMP_DELTA_NS
    ):
        _reject("video", "frame count, size, FPS, or PTS differs from Canonical")
    return probed


def _validate_with_annotation_reader(root: Path, expected: CanonicalQcEpisode) -> None:
    selected = sorted(
        {
            boundary
            for item in expected.semantics.subtask_sequence
            for boundary in (item.start_frame, item.end_frame_exclusive - 1)
        }
    )
    try:
        dataset = LeRobotV3Dataset(
            root,
            camera_names=["observation.images.main"],
            instruction_config={"instruction_source": "episode_field"},
            frame_indices=selected,
            load_frames=True,
        )
        episode = dataset.get_episode(0)
    except Exception as exc:
        _reject("annotation_reader", f"local annotation reader failed: {exc}")
    if episode["num_frames"] != expected.time_axis.frame_count:
        _reject("annotation_reader", "local reader frame count differs")
    if episode["instruction"] != expected.semantics.task_en:
        _reject("annotation_reader", "local reader task instruction differs")
    if episode["frame_indices"] != selected:
        _reject("annotation_reader", "local reader did not return subtask boundaries")
    for frame_index in selected:
        expected_subtask = next(
            item
            for item in expected.semantics.subtask_sequence
            if item.start_frame <= frame_index < item.end_frame_exclusive
        )
        metadata = episode["frame_metadata"][frame_index]
        if (
            metadata.get("subtask_id") != expected_subtask.subtask_id
            or metadata.get("description_cn") != expected_subtask.description_cn
            or metadata.get("description_en") != expected_subtask.description_en
        ):
            _reject("annotation_reader", "local reader subtask metadata differs")
        frame = episode["frames"]["observation.images.main"][frame_index]
        if frame.shape != (
            expected.main_video.height_px,
            expected.main_video.width_px,
            3,
        ):
            _reject("annotation_reader", "local reader video frame shape differs")


def _official_fingerprint(versions: dict[str, str]) -> str:
    payload = json.dumps(
        versions, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_with_official_reader(
    root: Path, release_id: str, expected_episode: CanonicalQcEpisode
) -> tuple[str, tuple[tuple[str, str], ...], str]:
    script = r'''
import importlib.metadata
import json
import platform
from pathlib import Path
import sys

expected = json.loads(sys.argv[1])
contract = json.loads(sys.argv[2])
versions = {
    **{name: importlib.metadata.version(name) for name in expected["packages"]},
    "python": platform.python_version(),
    "platform": platform.platform(),
    "video_backend": expected["video_backend"],
}
expected_versions = {
    **expected["packages"],
    "python": expected["python"],
    "platform": expected["platform"],
    "video_backend": expected["video_backend"],
}
if versions != expected_versions:
    raise RuntimeError(f"official reader environment mismatch: {versions!r}")
from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset(
    repo_id=f"curated/{sys.argv[4]}",
    root=Path(sys.argv[3]),
    episodes=[0],
    download_videos=False,
    video_backend=expected["video_backend"],
)
if len(dataset) != contract["frame_count"]:
    raise RuntimeError(f"official reader length mismatch: {len(dataset)}")
first = dataset[0]
last = dataset[len(dataset) - 1]
observed = []
for row in (first, last):
    values = {}
    for key in ("index", "frame_index", "timestamp_ns"):
        if key not in row:
            raise RuntimeError(f"official reader row missing {key}")
        raw = row[key]
        values[key] = int(raw.item() if hasattr(raw, "item") else raw)
    video = row.get("observation.images.main")
    if video is None or not hasattr(video, "shape"):
        raise RuntimeError("official reader row missing decoded main video")
    values["video_shape"] = list(video.shape)
    observed.append(values)
if observed != contract["boundary_rows"]:
    raise RuntimeError(f"official reader boundary mismatch: {observed!r}")
print(json.dumps({"versions": versions, "length": len(dataset), "boundary_rows": observed}, sort_keys=True))
'''
    environment = os.environ.copy()
    environment.update(
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    frame_count = expected_episode.time_axis.frame_count
    contract = {
        "frame_count": frame_count,
        "boundary_rows": [
            {
                "index": index,
                "frame_index": index,
                "timestamp_ns": int(expected_episode.time_axis.timestamps_ns[index]),
                "video_shape": [
                    3,
                    expected_episode.main_video.height_px,
                    expected_episode.main_video.width_px,
                ],
            }
            for index in (0, frame_count - 1)
        ],
    }
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                json.dumps(
                    {
                        "packages": _OFFICIAL_READER_PACKAGES,
                        "python": _OFFICIAL_READER_VERSIONS["python"],
                        "platform": _OFFICIAL_READER_VERSIONS["platform"],
                        "video_backend": _OFFICIAL_READER_VERSIONS["video_backend"],
                    },
                    sort_keys=True,
                ),
                json.dumps(contract, sort_keys=True),
                str(root),
                release_id,
            ],
            shell=False,
            capture_output=True,
            text=True,
            timeout=_OFFICIAL_READER_TIMEOUT_SECONDS,
            check=False,
            env=environment,
        )
    except subprocess.TimeoutExpired as exc:
        _reject(
            "official_reader",
            f"official LeRobotDataset timed out after {_OFFICIAL_READER_TIMEOUT_SECONDS}s: {exc}",
            retryable=True,
        )
    except OSError as exc:
        _reject("official_reader", f"cannot start official reader subprocess: {exc}")
    if completed.returncode != 0:
        signal = -completed.returncode if completed.returncode < 0 else None
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostics"
        if signal is not None:
            detail = f"terminated by signal {signal}: {detail}"
        _reject("official_reader", f"official LeRobotDataset failed: {detail}")
    try:
        payload = json.loads(completed.stdout.splitlines()[-1])
        versions = payload["versions"]
    except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
        _reject("official_reader", f"invalid official reader response: {exc}")
    if versions != _OFFICIAL_READER_VERSIONS:
        _reject("official_reader", f"official reader environment mismatch: {versions!r}")
    if payload.get("length") != frame_count or payload.get("boundary_rows") != contract[
        "boundary_rows"
    ]:
        _reject("official_reader", "official reader response does not prove boundary rows")
    ordered = tuple(sorted(versions.items()))
    return versions["lerobot"], ordered, _official_fingerprint(versions)


def _inventory_snapshot(reader: _ReleaseReader) -> dict[str, tuple[int, str]]:
    return {
        relative: (reader.size(relative), reader.sha256(relative))
        for relative in sorted(reader.files())
    }


def validate_staged_release(
    staged: StagedRelease, expected: CanonicalQcEpisode
) -> ValidationReport:
    """Reopen and verify one release without trusting writer in-memory inventory."""

    if type(staged) is not StagedRelease:
        _reject("staged", "must be an exact StagedRelease")
    if type(expected) is not CanonicalQcEpisode:
        _reject("expected", "must be an exact CanonicalQcEpisode")
    reader = _ReleaseReader(staged.root)
    try:
        if staged.root_device and staged.root_inode and reader.identity != (
            staged.root_device,
            staged.root_inode,
        ):
            _reject("staged.root", "does not match the writer-held directory identity")
        actual_files = reader.files()
        manifest_payload = reader.read_bytes("release_manifest.json")
        manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
        manifest = _manifest(manifest_payload)
        registered = {item.relative_path for item in manifest.files}
        if len(registered) != len(manifest.files):
            _reject("manifest.files", "duplicate registered paths")
        expected_files = registered | _INVENTORY_FILES
        if actual_files != expected_files:
            _reject("inventory", "missing, extra, or non-regular artifacts")
        checksums = _checksum_rows(reader.read_bytes("checksums.sha256"))
        wanted_checksums = registered | {"release_manifest.json"}
        if set(checksums) != wanted_checksums:
            _reject("checksums.sha256", "checksum inventory differs from manifest")
        for item in manifest.files:
            if reader.size(item.relative_path) != item.size_bytes:
                _reject(item.relative_path, "size differs from manifest")
            if reader.sha256(item.relative_path) != item.sha256:
                _reject(item.relative_path, "hash differs from manifest")
            if checksums[item.relative_path] != item.sha256:
                _reject(item.relative_path, "checksum row differs from manifest")
        if checksums["release_manifest.json"] != manifest_sha:
            _reject("release_manifest.json", "manifest checksum differs")
        plan = staged.plan
        binding = {
            "release_id": plan.release_id,
            "publisher_version": plan.publisher_version,
            "asset_id": expected.identity.asset_id,
            "canonical_revision": plan.request.canonical_revision,
            "semantic_fingerprint": plan.semantic_fingerprint,
            "source_fingerprint": plan.source_fingerprint,
            "qc_report_revision": plan.qc_report_revision,
            "qc_report_sha256": plan.qc_report_sha256,
            "data_fingerprint": plan.data_fingerprint,
            "revision_artifact_sha256": plan.revision_artifact_sha256,
        }
        for name, value in binding.items():
            if getattr(manifest, name) != value:
                _reject(f"manifest.{name}", "does not match validated publish plan")
        if manifest.toolchain != current_toolchain():
            _reject("manifest.toolchain", "does not match current writer toolchain")
        video = manifest.video_materialization
        if video is None:
            _reject("manifest.video_materialization", "is required")
        if (
            video.source_relative_path != expected.main_video.path
            or video.source_frame_range != expected.main_video.source_frame_range
            or video.source_sha256 != expected.main_video.sha256
            or video.target_sha256
            != reader.sha256("videos/observation.images.main/chunk-000/file-000.mp4")
        ):
            _reject("manifest.video_materialization", "does not match source or target")
        _validate_frame_data(reader, expected)
        probed = _validate_video(reader, expected)
        _validate_registered_metadata(reader, expected, probed)
        _validate_episode_metadata(reader, expected)
        _validate_subtasks(reader, expected)
        _validate_semantics(reader, expected)
        reader.verify_identity()
        inventory_before_readers = _inventory_snapshot(reader)
        _validate_with_annotation_reader(staged.root, expected)
        official_version, official_versions, official_fingerprint = (
            _validate_with_official_reader(staged.root, plan.release_id, expected)
        )
        reader.verify_identity()
        if _inventory_snapshot(reader) != inventory_before_readers:
            _reject("inventory", "reader gate changed or observed a changed release")
        return ValidationReport(
            schema_version="curated_lerobot_v3_validation.v1",
            release_id=plan.release_id,
            file_count=len(manifest.files),
            frame_count=expected.time_axis.frame_count,
            manifest_sha256=manifest_sha,
            official_reader_version=official_version,
            official_reader_versions=official_versions,
            official_reader_fingerprint=official_fingerprint,
            manifest=manifest,
        )
    finally:
        reader.close()


__all__ = ["validate_staged_release"]
