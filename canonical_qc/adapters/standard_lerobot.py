"""Strict reader for registered file-based LeRobot v3 and episode-based v2.1."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from numbers import Integral
from pathlib import Path, PurePosixPath
import stat
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..contracts import (
    CameraCalibration,
    CanonicalQcEpisode,
    EpisodeIdentity,
    EpisodeSemantics,
    HandObservation,
    ProbedVideo,
    SourceFile,
    SourceProvenance,
    Subtask,
    SupplierEvidence,
    SupplierHandQuality,
    TimeAxis,
    VideoStream,
)
from ..errors import CanonicalInputError
from ..provenance import source_fingerprint
from ..validation import validate_episode, validate_video_alignment
from ..video_probe import probe_video
from .base import SourceInspection


_SCHEMA_VERSION = "egodata_lerobot_qc_input.v1"
_VIDEO_KEY = "observation.images.main"
_V3_DATA = "data/chunk-{episode_chunk:03d}/file-{episode_file:03d}.parquet"
_V3_VIDEO = "videos/{video_key}/chunk-{episode_chunk:03d}/file-{episode_file:03d}.mp4"
_V21_DATA = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
_V21_VIDEO = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
_CORE_FEATURES: dict[str, tuple[str, list[int]]] = {
    "episode_index": ("int64", []),
    "frame_index": ("int64", []),
    "timestamp": ("float64", []),
    "timestamp_ns": ("int64", []),
    "observation.hand_keypoints_3d": ("float32", [2, 21, 3]),
    "observation.hand_joint_valid_3d": ("bool", [2, 21]),
    "observation.hand_keypoints_2d": ("float32", [2, 21, 2]),
    "observation.hand_joint_valid_2d": ("bool", [2, 21]),
    "task_index": ("int64", []),
    "subtask_index": ("int64", []),
}
_RAW_PRIMITIVE_CONTRACTS: tuple[tuple[pa.DataType, str, Any], ...] = (
    (pa.int8(), "int8", np.int8),
    (pa.int16(), "int16", np.int16),
    (pa.int32(), "int32", np.int32),
    (pa.int64(), "int64", np.int64),
    (pa.uint8(), "uint8", np.uint8),
    (pa.uint16(), "uint16", np.uint16),
    (pa.uint32(), "uint32", np.uint32),
    (pa.uint64(), "uint64", np.uint64),
    (pa.float16(), "float16", np.float16),
    (pa.float32(), "float32", np.float32),
    (pa.float64(), "float64", np.float64),
    (pa.bool_(), "bool", np.bool_),
    (pa.string(), "string", str),
)


def _fail(code: str, field: str, detail: str) -> None:
    raise CanonicalInputError(code, field, detail)


def _mapped(error: CanonicalInputError) -> CanonicalInputError:
    if error.code in {"schema_missing", "field_mapping_error", "timebase_invalid", "source_integrity_error"}:
        return error
    return CanonicalInputError("field_mapping_error", error.field, error.detail)


def _reject_symlink_chain(path: Path, *, field: str) -> None:
    absolute = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except OSError:
            return
        if stat.S_ISLNK(metadata.st_mode):
            _fail("source_integrity_error", field, f"path component {current} must not be a symlink")


def _safe_file(root: Path, relative: str, *, field: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or "\\" in relative or any(part in {"", ".", ".."} for part in relative.split("/")):
        _fail("source_integrity_error", field, "must be a normalized relative path inside source root")
    lexical = root.joinpath(*pure.parts)
    _reject_symlink_chain(lexical, field=field)
    resolved = lexical.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        _fail("source_integrity_error", field, "path escapes source root")
    if not resolved.is_file():
        _fail("source_integrity_error", field, "required regular file is missing")
    return resolved


def _metadata(path: Path, root: Path, *, role: str) -> SourceFile:
    relative = path.relative_to(root).as_posix()
    try:
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            _fail("source_integrity_error", relative, "must be a regular file")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
    except CanonicalInputError:
        raise
    except OSError as exc:
        _fail("source_integrity_error", relative, f"cannot stat/hash file: {exc}")
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        _fail("source_integrity_error", relative, "file changed while hashing")
    return SourceFile(relative, role, after.st_size, digest.hexdigest())


def _json_file(path: Path, *, field: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _fail("field_mapping_error", field, f"must contain valid UTF-8 JSON: {exc}")


def _jsonl(path: Path, *, field: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        _fail("source_integrity_error", field, f"cannot read JSONL: {exc}")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            _fail("field_mapping_error", f"{field}[{index}]", f"invalid JSON: {exc}")
        if not isinstance(row, dict):
            _fail("field_mapping_error", f"{field}[{index}]", "must be an object")
        rows.append(row)
    return rows


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _fail("field_mapping_error", field, "must be an integer")
    return int(value)


def _positive_integer(value: object, *, field: str) -> int:
    result = _integer(value, field=field)
    if result <= 0:
        _fail("field_mapping_error", field, "must be greater than zero")
    return result


def _nonnegative_integer(value: object, *, field: str) -> int:
    result = _integer(value, field=field)
    if result < 0:
        _fail("field_mapping_error", field, "must be non-negative")
    return result


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("field_mapping_error", field, "must be a non-empty string")
    return value


def _required(mapping: dict[str, Any], key: str, *, prefix: str) -> Any:
    if key not in mapping:
        _fail("schema_missing", f"{prefix}.{key}", "required field is missing")
    return mapping[key]


def _feature_type(dtype: str, shape: list[int]) -> pa.DataType:
    result: pa.DataType = {"int64": pa.int64(), "float64": pa.float64(), "float32": pa.float32(), "bool": pa.bool_()}[dtype]
    for size in reversed(shape):
        result = pa.list_(result, size)
    return result


def _raw_primitive_contract(dtype: pa.DataType) -> tuple[str, Any] | None:
    for arrow_type, contract_name, numpy_dtype in _RAW_PRIMITIVE_CONTRACTS:
        if dtype == arrow_type:
            return contract_name, numpy_dtype
    return None


def _validate_info(info: Any) -> tuple[str, str, str, int, int]:
    if not isinstance(info, dict):
        _fail("field_mapping_error", "info", "JSON root must be an object")
    schema = _text(_required(info, "schema_version", prefix="info"), field="info.schema_version")
    if schema != _SCHEMA_VERSION:
        _fail("field_mapping_error", "info.schema_version", f"expected {_SCHEMA_VERSION!r}")
    version = _text(_required(info, "codebase_version", prefix="info"), field="info.codebase_version")
    if version.startswith("v3"):
        layout, data_expected, video_expected = "v3", _V3_DATA, _V3_VIDEO
    elif version == "v2.1":
        layout, data_expected, video_expected = "v2.1", _V21_DATA, _V21_VIDEO
    else:
        _fail("field_mapping_error", "info.codebase_version", "only registered v3 and v2.1 are supported")
    data_template = _text(_required(info, "data_path", prefix="info"), field="info.data_path")
    video_template = _text(_required(info, "video_path", prefix="info"), field="info.video_path")
    if data_template != data_expected:
        _fail("source_integrity_error", "info.data_path", f"expected registered template {data_expected!r}")
    if video_template != video_expected:
        _fail("source_integrity_error", "info.video_path", f"expected registered template {video_expected!r}")
    fps_num = _positive_integer(_required(info, "fps_num", prefix="info"), field="info.fps_num")
    fps_den = _positive_integer(_required(info, "fps_den", prefix="info"), field="info.fps_den")
    fps = _required(info, "fps", prefix="info")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or Fraction(str(fps)) != Fraction(fps_num, fps_den):
        _fail("field_mapping_error", "info.fps", "must exactly match fps_num/fps_den")
    features = _required(info, "features", prefix="info")
    if not isinstance(features, dict):
        _fail("field_mapping_error", "info.features", "must be an object")
    for name, (dtype, shape) in _CORE_FEATURES.items():
        feature = _required(features, name, prefix="info.features")
        if not isinstance(feature, dict) or feature.get("dtype") != dtype or feature.get("shape") != shape:
            _fail("field_mapping_error", f"info.features.{name}", f"must declare dtype={dtype!r}, shape={shape!r}")
    video_feature = _required(features, _VIDEO_KEY, prefix="info.features")
    if not isinstance(video_feature, dict) or video_feature.get("dtype") != "video":
        _fail("field_mapping_error", f"info.features.{_VIDEO_KEY}", "must declare dtype='video'")
    video_shape = video_feature.get("shape")
    if (
        not isinstance(video_shape, list)
        or len(video_shape) != 3
        or any(isinstance(value, bool) or not isinstance(value, Integral) or value <= 0 for value in video_shape)
        or video_shape[2] != 3
    ):
        _fail(
            "field_mapping_error",
            f"info.features.{_VIDEO_KEY}.shape",
            "must be [positive_height, positive_width, 3]",
        )
    constants = {
        "observation.hand_keypoints_3d": {"hand_order": ["left", "right"], "joint_topology": "egodata_hand21.v1", "coordinate_frame": "camera:main", "unit": "meter"},
        "observation.hand_keypoints_2d": {"hand_order": ["left", "right"], "joint_topology": "egodata_hand21.v1", "coordinate_space": "pixel", "unit": "pixel"},
        _VIDEO_KEY: {"camera_id": "main", "camera_role": "ego"},
        "timestamp": {"unit": "second"},
        "timestamp_ns": {"unit": "nanosecond"},
    }
    for name, declared in constants.items():
        feature = features[name]
        for key, expected in declared.items():
            if feature.get(key) != expected:
                _fail("field_mapping_error", f"info.features.{name}.{key}", f"expected {expected!r}")
    return layout, data_template, video_template, int(video_shape[1]), int(video_shape[0])


def _episode_paths(root: Path, layout: str) -> tuple[Path, ...]:
    if layout == "v3":
        paths = tuple(sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet")))
        if not paths:
            _fail("schema_missing", "meta/episodes", "no episode metadata parquet found")
        for path in paths:
            _reject_symlink_chain(path, field="meta/episodes")
        return paths
    path = _safe_file(root, "meta/episodes.jsonl", field="meta/episodes.jsonl")
    return (path,)


def _episode_rows(paths: tuple[Path, ...], layout: str) -> list[dict[str, Any]]:
    if layout == "v3":
        rows: list[dict[str, Any]] = []
        for path in paths:
            try:
                rows.extend(pq.read_table(path).to_pylist())
            except (OSError, pa.ArrowException) as exc:
                _fail("field_mapping_error", "meta/episodes", f"cannot read Parquet: {exc}")
        return rows
    return _jsonl(paths[0], field="meta/episodes.jsonl")


def _select(rows: list[dict[str, Any]], episode_index: int | None) -> dict[str, Any]:
    seen: set[int] = set()
    for row in rows:
        index = _integer(_required(row, "episode_index", prefix="episode"), field="episode.episode_index")
        if index in seen:
            _fail("field_mapping_error", "episode_index", f"duplicate episode_index {index}")
        seen.add(index)
    if episode_index is None:
        if len(rows) != 1:
            _fail("field_mapping_error", "episode_index", f"selector required for {len(rows)} episodes")
        return rows[0]
    if isinstance(episode_index, bool) or not isinstance(episode_index, Integral):
        _fail("field_mapping_error", "episode_index", "selector must be an integer")
    matches = [row for row in rows if int(row["episode_index"]) == int(episode_index)]
    if len(matches) != 1:
        _fail("field_mapping_error", "episode_index", f"selector matched {len(matches)} episodes")
    return matches[0]


def _revalidate_episode_paths(inspection: SourceInspection) -> tuple[Path, ...]:
    if inspection.layout_version not in ("v3", "v2.1"):
        _fail("source_integrity_error", "source", "inspection has no registered layout")
    try:
        current = _episode_paths(inspection.source_root, inspection.layout_version)
    except CanonicalInputError as exc:
        _fail(
            "source_integrity_error",
            "source",
            f"cannot revalidate episode metadata shard set: {exc.detail}",
        )
    if current != inspection.episode_metadata_paths:
        _fail(
            "source_integrity_error",
            "source",
            "episode metadata shard set changed after inspection",
        )
    return current


class StandardLeRobotAdapter:
    adapter_id = "standard_lerobot"
    adapter_version = "1.0.0"

    def __init__(self, *, max_timestamp_delta_ns: int = 1_000_000) -> None:
        if isinstance(max_timestamp_delta_ns, bool) or not isinstance(max_timestamp_delta_ns, Integral) or max_timestamp_delta_ns < 0:
            _fail("field_mapping_error", "max_timestamp_delta_ns", "must be a non-negative integer")
        self.max_timestamp_delta_ns = int(max_timestamp_delta_ns)

    def inspect(self, source: Path, *, episode_index: int | None = None) -> SourceInspection:
        try:
            return self._inspect(source, episode_index=episode_index)
        except CanonicalInputError:
            raise
        except (OSError, ValueError, TypeError, ZeroDivisionError, pa.ArrowException) as exc:
            _fail("source_integrity_error", "source", f"cannot inspect LeRobot dataset: {exc}")

    def _inspect(self, source: Path, *, episode_index: int | None = None) -> SourceInspection:
        source_path = Path(source).expanduser()
        if not source_path.is_absolute():
            source_path = Path.cwd() / source_path
        _reject_symlink_chain(source_path, field="source")
        root = source_path.resolve()
        if not root.is_dir():
            _fail("source_integrity_error", "source", "must be a dataset directory")
        info_path = _safe_file(root, "meta/info.json", field="meta/info.json")
        info_source = _metadata(info_path, root, role="dataset_info")
        info = _json_file(info_path, field="meta/info.json")
        (
            layout,
            data_template,
            video_template,
            video_width_px,
            video_height_px,
        ) = _validate_info(info)
        episode_paths = _episode_paths(root, layout)
        episode_sources = tuple(
            _metadata(path, root, role="episode_index") for path in episode_paths
        )
        inspection_sources = (info_source, *episode_sources)
        rows = _episode_rows(episode_paths, layout)
        row = _select(rows, episode_index)
        selected_index = _integer(row["episode_index"], field="episode.episode_index")
        asset_id = _text(_required(row, "asset_id", prefix="episode"), field="episode.asset_id")
        frame_count = _positive_integer(
            _required(row, "length", prefix="episode"), field="episode.length"
        )
        if layout == "v3":
            data_chunk = _nonnegative_integer(
                _required(row, "data/chunk_index", prefix="episode"),
                field="episode.data/chunk_index",
            )
            data_file = _nonnegative_integer(
                _required(row, "data/file_index", prefix="episode"),
                field="episode.data/file_index",
            )
            video_chunk = _nonnegative_integer(
                _required(row, f"videos/{_VIDEO_KEY}/chunk_index", prefix="episode"),
                field=f"episode.videos/{_VIDEO_KEY}/chunk_index",
            )
            video_file = _nonnegative_integer(
                _required(row, f"videos/{_VIDEO_KEY}/file_index", prefix="episode"),
                field=f"episode.videos/{_VIDEO_KEY}/file_index",
            )
        else:
            data_chunk = video_chunk = _nonnegative_integer(
                _required(row, "episode_chunk", prefix="episode"),
                field="episode.episode_chunk",
            )
            data_file = video_file = 0
        data_values = {
            "episode_chunk": data_chunk,
            "episode_file": data_file,
            "episode_index": selected_index,
            "video_key": _VIDEO_KEY,
        }
        video_values = {
            "episode_chunk": video_chunk,
            "episode_file": video_file,
            "episode_index": selected_index,
            "video_key": _VIDEO_KEY,
        }
        try:
            data_relative = data_template.format(**data_values)
            video_relative = video_template.format(**video_values)
        except (KeyError, ValueError) as exc:
            _fail("field_mapping_error", "info", f"cannot expand registered path template: {exc}")
        data_path = _safe_file(root, data_relative, field="episode.data_path")
        video_path = _safe_file(root, video_relative, field="main_video.path")
        semantics_path = _safe_file(root, "meta/episode_semantics.jsonl", field="meta/episode_semantics.jsonl")
        if layout == "v3":
            start = _nonnegative_integer(
                _required(row, "dataset_from_index", prefix="episode"),
                field="episode.dataset_from_index",
            )
            stop = _nonnegative_integer(
                _required(row, "dataset_to_index", prefix="episode"),
                field="episode.dataset_to_index",
            )
            video_from_key = f"videos/{_VIDEO_KEY}/from_index"
            video_to_key = f"videos/{_VIDEO_KEY}/to_index"
            if video_from_key not in row:
                _fail("schema_missing", "episode.video_from_index", "required field is missing")
            if video_to_key not in row:
                _fail("schema_missing", "episode.video_to_index", "required field is missing")
            video_offset = _nonnegative_integer(
                row[video_from_key], field="episode.video_from_index"
            )
            video_stop = _nonnegative_integer(
                row[video_to_key], field="episode.video_to_index"
            )
        else:
            start, stop = 0, frame_count
            video_offset, video_stop = 0, frame_count
        if stop - start != frame_count:
            _fail(
                "field_mapping_error",
                "episode.dataset_to_index",
                "[from, to) must contain exactly length rows",
            )
        if video_stop - video_offset != frame_count:
            _fail(
                "field_mapping_error",
                "episode.video_to_index",
                "[from, to) must contain exactly length frames",
            )
        current_episode_paths = _episode_paths(root, layout)
        final_inspection_sources = (
            _metadata(info_path, root, role="dataset_info"),
            *(
                _metadata(path, root, role="episode_index")
                for path in current_episode_paths
            ),
        )
        if current_episode_paths != episode_paths or final_inspection_sources != inspection_sources:
            _fail(
                "source_integrity_error",
                "source",
                "info or episode metadata changed while inspecting",
            )
        return SourceInspection(
            adapter_id=self.adapter_id, adapter_version=self.adapter_version,
            source_format="lerobot", source_schema_version=_SCHEMA_VERSION,
            asset_id=asset_id, source_root=root, main_video_path=video_path,
            info_path=info_path, episode_metadata_paths=episode_paths, data_path=data_path,
            semantics_path=semantics_path, episode_index=selected_index,
            frame_count=frame_count, data_row_offset=start, data_row_stop=stop,
            video_frame_offset=video_offset, video_frame_stop=video_stop,
            layout_version=layout, inspection_source_files=inspection_sources,
            video_width_px=video_width_px, video_height_px=video_height_px,
        )

    def load(self, source: Path, *, episode_index: int | None = None) -> CanonicalQcEpisode:
        inspection = self.inspect(source, episode_index=episode_index)
        if not inspection.info_path or not inspection.data_path or not inspection.semantics_path:
            _fail("source_integrity_error", "source", "inspection did not resolve required files")
        load_paths_roles = [
            (inspection.data_path, "episode_data"),
            (inspection.semantics_path, "episode_semantics"),
            (inspection.main_video_path, "main_video"),
        ]
        current_episode_paths = _revalidate_episode_paths(inspection)
        current_inspection = (
            _metadata(inspection.info_path, inspection.source_root, role="dataset_info"),
            *(
                _metadata(path, inspection.source_root, role="episode_index")
                for path in current_episode_paths
            ),
        )
        if current_inspection != inspection.inspection_source_files:
            _fail(
                "source_integrity_error",
                "source",
                "info or episode metadata changed after inspection",
            )
        initial = inspection.inspection_source_files + tuple(
            _metadata(path, inspection.source_root, role=role)
            for path, role in load_paths_roles
        )
        try:
            episode = self._read_episode(inspection, initial)
            full_video = probe_video(inspection.main_video_path)
            start = inspection.video_frame_offset
            stop = inspection.video_frame_stop
            if stop > full_video.frame_count:
                _fail("timebase_invalid", "main_video.frame_count", "video slice exceeds shared video")
            if inspection.layout_version == "v2.1" and full_video.frame_count != episode.time_axis.frame_count:
                _fail("timebase_invalid", "main_video.frame_count", "v2.1 episode video must contain exactly length frames")
            if (
                full_video.width_px != inspection.video_width_px
                or full_video.height_px != inspection.video_height_px
            ):
                _fail(
                    "field_mapping_error",
                    f"info.features.{_VIDEO_KEY}.shape",
                    "declared video dimensions disagree with ffprobe",
                )
            absolute = full_video.timestamps_ns[start:stop]
            relative = tuple(value - absolute[0] for value in absolute) if absolute else ()
            sliced = ProbedVideo(
                frame_count=episode.time_axis.frame_count, width_px=full_video.width_px,
                height_px=full_video.height_px, fps_num=full_video.fps_num,
                fps_den=full_video.fps_den, codec=full_video.codec,
                pixel_format=full_video.pixel_format, timestamps_ns=relative,
            )
            validate_video_alignment(episode.time_axis, sliced, max_delta_ns=self.max_timestamp_delta_ns)
            video_source = next(item for item in initial if item.role == "main_video")
            episode = CanonicalQcEpisode(
                schema_version=episode.schema_version, profile=episode.profile,
                identity=episode.identity, provenance=episode.provenance,
                time_axis=episode.time_axis,
                main_video=VideoStream(
                    path=video_source.relative_path, sha256=video_source.sha256,
                    frame_count=sliced.frame_count, width_px=sliced.width_px,
                    height_px=sliced.height_px, fps_num=sliced.fps_num, fps_den=sliced.fps_den,
                    codec=sliced.codec, pixel_format=sliced.pixel_format,
                    source_frame_range=(start, stop),
                ), observation=episode.observation, calibration=episode.calibration,
                semantics=episode.semantics, supplier_evidence=episode.supplier_evidence,
            )
            validate_episode(episode)
        except CanonicalInputError as exc:
            raise _mapped(exc) from exc
        except (OSError, ValueError, TypeError, pa.ArrowException) as exc:
            _fail("field_mapping_error", "source", f"cannot decode LeRobot dataset: {exc}")
        final_episode_paths = _revalidate_episode_paths(inspection)
        final_paths_roles = [(inspection.info_path, "dataset_info")]
        final_paths_roles.extend(
            (path, "episode_index") for path in final_episode_paths
        )
        final_paths_roles.extend(load_paths_roles)
        final = tuple(
            _metadata(path, inspection.source_root, role=role)
            for path, role in final_paths_roles
        )
        if tuple(item.sha256 for item in initial) != tuple(item.sha256 for item in final):
            _fail("source_integrity_error", "source", "source files changed while loading")
        return episode

    def _read_episode(self, inspection: SourceInspection, files: tuple[SourceFile, ...]) -> CanonicalQcEpisode:
        if (
            not inspection.data_path
            or not inspection.semantics_path
            or inspection.frame_count is None
            or inspection.episode_index is None
        ):
            _fail(
                "source_integrity_error",
                "source",
                "inspection is missing required episode fields",
            )
        table = pq.read_table(inspection.data_path)
        if inspection.layout_version == "v2.1" and table.num_rows != inspection.frame_count:
            _fail(
                "field_mapping_error",
                "episode.length",
                "v2.1 episode parquet must contain exactly length rows",
            )
        required = [name for name in _CORE_FEATURES if name != _VIDEO_KEY]
        for name in required:
            if name not in table.column_names:
                _fail("schema_missing", name, "required Parquet field is missing")
            dtype, shape = _CORE_FEATURES[name]
            if table.schema.field(name).type != _feature_type(dtype, shape):
                _fail("field_mapping_error", name, f"expected Arrow type {_feature_type(dtype, shape)}, got {table.schema.field(name).type}")
        start, stop = inspection.data_row_offset, inspection.data_row_stop
        if stop > table.num_rows:
            _fail("field_mapping_error", "episode.dataset_from_index", "row slice exceeds data shard")
        selected = table.slice(start, inspection.frame_count)
        episode_values = np.asarray(selected["episode_index"].to_numpy(), dtype=np.int64)
        frame_values = np.asarray(selected["frame_index"].to_numpy(), dtype=np.int64)
        if not np.all(episode_values == inspection.episode_index):
            _fail("field_mapping_error", "episode_index", "selected rows belong to another episode")
        if not np.array_equal(frame_values, np.arange(inspection.frame_count, dtype=np.int64)):
            _fail("field_mapping_error", "frame_index", "must equal 0..length-1")
        timestamps = np.asarray(selected["timestamp_ns"].to_numpy(), dtype=np.int64)
        floats = np.asarray(selected["timestamp"].to_numpy(), dtype=np.float64)
        expected_float = (timestamps - timestamps[0]) / 1_000_000_000
        deltas = np.abs((floats - expected_float) * 1_000_000_000)
        bad = np.flatnonzero(~np.isfinite(floats) | (deltas > self.max_timestamp_delta_ns))
        if bad.size:
            _fail("timebase_invalid", f"timestamp[{int(bad[0])}]", "float timestamp disagrees with authoritative timestamp_ns")
        semantics_rows = _jsonl(inspection.semantics_path, field="meta/episode_semantics.jsonl")
        semantic_keys: set[tuple[int, str]] = set()
        normalized_semantics: list[tuple[int, str, dict[str, Any]]] = []
        for row_index, row in enumerate(semantics_rows):
            prefix = f"episode_semantics[{row_index}]"
            semantic_episode_index = _integer(
                _required(row, "episode_index", prefix=prefix),
                field=f"{prefix}.episode_index",
            )
            semantic_asset_id = _text(
                _required(row, "asset_id", prefix=prefix),
                field=f"{prefix}.asset_id",
            )
            key = (semantic_episode_index, semantic_asset_id)
            if key in semantic_keys:
                _fail(
                    "field_mapping_error",
                    "episode_semantics",
                    f"duplicate episode_index + asset_id key {key!r}",
                )
            semantic_keys.add(key)
            normalized_semantics.append(
                (semantic_episode_index, semantic_asset_id, row)
            )
        matches = [
            row
            for semantic_episode_index, semantic_asset_id, row in normalized_semantics
            if semantic_episode_index == inspection.episode_index
            and semantic_asset_id == inspection.asset_id
        ]
        if len(matches) != 1:
            _fail("field_mapping_error", "episode_semantics", f"expected one episode_index + asset_id match, found {len(matches)}")
        semantic = matches[0]
        task_index = _integer(_required(semantic, "task_index", prefix="semantics"), field="semantics.task_index")
        task_values = np.asarray(selected["task_index"].to_numpy(), dtype=np.int64)
        subtask_values = np.asarray(selected["subtask_index"].to_numpy(), dtype=np.int64)
        if not np.all(task_values == task_index):
            _fail("field_mapping_error", "task_index", "per-frame values disagree with episode semantics")
        raw_subtasks = _required(semantic, "subtask_sequence", prefix="semantics")
        if not isinstance(raw_subtasks, list):
            _fail("field_mapping_error", "semantics.subtask_sequence", "must be an array")
        subtasks: list[Subtask] = []
        expected_subtask = np.full(inspection.frame_count, -1, dtype=np.int64)
        for index, raw in enumerate(raw_subtasks):
            if not isinstance(raw, dict):
                _fail("field_mapping_error", f"semantics.subtask_sequence[{index}]", "must be an object")
            begin = _integer(_required(raw, "start_frame", prefix="subtask"), field=f"subtask[{index}].start_frame")
            end = _integer(_required(raw, "end_frame_exclusive", prefix="subtask"), field=f"subtask[{index}].end_frame_exclusive")
            sub_index = _integer(_required(raw, "subtask_index", prefix="subtask"), field=f"subtask[{index}].subtask_index")
            if sub_index != index:
                _fail(
                    "field_mapping_error",
                    f"semantics.subtask_sequence[{index}].subtask_index",
                    f"must equal sequential index {index}",
                )
            if begin < 0 or end > inspection.frame_count or begin >= end:
                _fail("field_mapping_error", "semantics.subtask_sequence", "invalid half-open boundary")
            expected_subtask[begin:end] = sub_index
            subtasks.append(Subtask(
                subtask_id=_text(_required(raw, "subtask_id", prefix="subtask"), field=f"subtask[{index}].subtask_id"),
                start_frame=begin, end_frame_exclusive=end,
                description_cn=_text(_required(raw, "description_cn", prefix="subtask"), field=f"subtask[{index}].description_cn"),
                description_en=_text(_required(raw, "description_en", prefix="subtask"), field=f"subtask[{index}].description_en"),
            ))
        if not np.array_equal(subtask_values, expected_subtask):
            _fail("field_mapping_error", "subtask_index", "per-frame values disagree with semantic boundaries")
        constants = {"joint_topology": "egodata_hand21.v1", "coordinate_frame_3d": "camera:main", "length_unit": "meter", "coordinate_space_2d": "pixel"}
        for key, expected in constants.items():
            if semantic.get(key) != expected:
                _fail("field_mapping_error", f"semantics.{key}", f"expected {expected!r}")
        calibration_raw = _required(semantic, "calibration", prefix="semantics")
        if not isinstance(calibration_raw, dict):
            _fail("field_mapping_error", "semantics.calibration", "must be an object")
        calibration = CameraCalibration(
            intrinsic_matrix=np.asarray(_required(calibration_raw, "intrinsic_matrix", prefix="calibration"), dtype=np.float64),
            distortion_model=_text(_required(calibration_raw, "distortion_model", prefix="calibration"), field="calibration.distortion_model"),
            distortion_coefficients=np.asarray(_required(calibration_raw, "distortion_coefficients", prefix="calibration"), dtype=np.float64),
            image_width_px=_integer(_required(calibration_raw, "image_width_px", prefix="calibration"), field="calibration.image_width_px"),
            image_height_px=_integer(_required(calibration_raw, "image_height_px", prefix="calibration"), field="calibration.image_height_px"),
            camera_axes=_text(_required(calibration_raw, "camera_axes", prefix="calibration"), field="calibration.camera_axes"),
            pixel_origin=_text(_required(calibration_raw, "pixel_origin", prefix="calibration"), field="calibration.pixel_origin"),
        )
        if (
            calibration.image_width_px != inspection.video_width_px
            or calibration.image_height_px != inspection.video_height_px
        ):
            _fail(
                "field_mapping_error",
                f"info.features.{_VIDEO_KEY}.shape",
                "declared video dimensions disagree with calibration",
            )
        def array(name: str, dtype: Any) -> np.ndarray:
            return np.asarray(selected[name].to_pylist(), dtype=dtype)
        observation = HandObservation(
            hand_keypoints_3d=array("observation.hand_keypoints_3d", np.float32),
            hand_joint_valid_3d=array("observation.hand_joint_valid_3d", np.bool_),
            hand_keypoints_2d=array("observation.hand_keypoints_2d", np.float32),
            hand_joint_valid_2d=array("observation.hand_joint_valid_2d", np.bool_),
        )
        info = _json_file(inspection.info_path, field="meta/info.json")
        _validate_info(info)
        fps_num = _positive_integer(info["fps_num"], field="info.fps_num")
        fps_den = _positive_integer(info["fps_den"], field="info.fps_den")
        quality = self._quality(selected, semantic, inspection.frame_count, info)
        identity = EpisodeIdentity(
            asset_id=inspection.asset_id,
            batch_id=_text(_required(semantic, "batch_id", prefix="semantics"), field="semantics.batch_id"),
            supplier_id=_text(_required(semantic, "supplier_id", prefix="semantics"), field="semantics.supplier_id"),
            source_format="lerobot", source_schema_version=inspection.source_schema_version,
        )
        provenance = SourceProvenance(
            source_files=files,
            source_fingerprint=source_fingerprint(files, source_schema_version=identity.source_schema_version, adapter_id=self.adapter_id, adapter_version=self.adapter_version),
            adapter_id=self.adapter_id, adapter_version=self.adapter_version,
        )
        video_source = next(item for item in files if item.role == "main_video")
        return CanonicalQcEpisode(
            schema_version="canonical_qc_episode.v1", profile="human_ego_hand_pose.v1",
            identity=identity, provenance=provenance,
            time_axis=TimeAxis(frame_count=inspection.frame_count, timestamps_ns=timestamps, fps_num=fps_num, fps_den=fps_den),
            main_video=VideoStream(path=video_source.relative_path, sha256=video_source.sha256, frame_count=inspection.frame_count, width_px=calibration.image_width_px, height_px=calibration.image_height_px, fps_num=fps_num, fps_den=fps_den, codec="pending_probe", pixel_format="pending_probe", source_frame_range=(inspection.video_frame_offset, inspection.video_frame_stop)),
            observation=observation, calibration=calibration,
            semantics=EpisodeSemantics(
                scene_id=_text(_required(semantic, "scene_id", prefix="semantics"), field="semantics.scene_id"),
                task_id=_text(_required(semantic, "task_id", prefix="semantics"), field="semantics.task_id"),
                task_category=_text(_required(semantic, "task_category", prefix="semantics"), field="semantics.task_category"),
                task_cn=_text(_required(semantic, "task_cn", prefix="semantics"), field="semantics.task_cn"),
                task_en=_text(_required(semantic, "task_en", prefix="semantics"), field="semantics.task_en"),
                description_cn=_text(_required(semantic, "description_cn", prefix="semantics"), field="semantics.description_cn"),
                description_en=_text(_required(semantic, "description_en", prefix="semantics"), field="semantics.description_en"),
                subtask_sequence=tuple(subtasks),
            ), supplier_evidence=quality,
        )

    def _quality(
        self,
        table: pa.Table,
        semantic: dict[str, Any],
        frame_count: int,
        info: dict[str, Any],
    ) -> SupplierEvidence:
        state = semantic.get("supplier_hand_quality")
        names = {"supplier.hand_quality.raw_value", "supplier.hand_quality.normalized_score", "supplier.hand_quality.status"}
        present = names.intersection(table.column_names)
        if state is None:
            if present:
                _fail("field_mapping_error", "supplier.hand_quality", "payload requires authoritative semantic state")
            return SupplierEvidence()
        if not isinstance(state, dict) or not isinstance(state.get("provided"), bool):
            _fail("field_mapping_error", "supplier_hand_quality.provided", "must be bool")
        if not state["provided"]:
            if present or "mapping_version" in state:
                _fail("field_mapping_error", "supplier.hand_quality", "provided=false forbids payload and mapping_version")
            return SupplierEvidence(hand_quality=SupplierHandQuality(provided=False, status=np.full((frame_count, 2), "unknown", dtype="<U7")))
        status_name = "supplier.hand_quality.status"
        raw_name = "supplier.hand_quality.raw_value"
        score_name = "supplier.hand_quality.normalized_score"
        if status_name not in present:
            _fail(
                "schema_missing",
                status_name,
                "provided=true requires canonical status",
            )
        if not isinstance(state.get("mapping_version"), str) or not state["mapping_version"].strip():
            _fail(
                "field_mapping_error",
                "supplier_hand_quality.mapping_version",
                "provided=true requires a non-empty mapping_version",
            )
        features = info["features"]

        def declared(name: str, dtype: str, arrow_type: pa.DataType) -> None:
            if name not in features:
                _fail("schema_missing", f"info.features.{name}", "provided quality feature declaration is missing")
            declaration = features[name]
            if not isinstance(declaration, dict) or declaration.get("dtype") != dtype or declaration.get("shape") != [2]:
                _fail("field_mapping_error", f"info.features.{name}", f"must declare dtype={dtype!r}, shape=[2]")
            if table.schema.field(name).type != arrow_type:
                _fail("field_mapping_error", name, f"expected Arrow type {arrow_type}, got {table.schema.field(name).type}")

        declared(status_name, "string", pa.list_(pa.string(), 2))
        status_values = table[status_name].to_pylist()
        if len(status_values) != frame_count or any(
            not isinstance(row, list) or len(row) != 2 for row in status_values
        ):
            _fail("field_mapping_error", status_name, "must have shape (T, 2)")
        allowed_status = {"unknown", "bad", "warning", "good"}
        invalid_status = sorted(
            {
                value
                for row in status_values
                for value in row
                if not isinstance(value, str) or value not in allowed_status
            },
            key=repr,
        )
        if invalid_status:
            _fail(
                "field_mapping_error",
                status_name,
                f"unknown canonical status values {invalid_status!r}",
            )
        status = np.asarray(status_values, dtype="<U7")

        raw: np.ndarray | None = None
        if raw_name in present:
            declaration = features.get(raw_name)
            if not isinstance(declaration, dict):
                _fail("schema_missing", f"info.features.{raw_name}", "provided raw feature declaration is missing")
            raw_dtype = declaration.get("dtype")
            raw_arrow_type = table.schema.field(raw_name).type
            if not pa.types.is_fixed_size_list(raw_arrow_type) or raw_arrow_type.list_size != 2:
                _fail(
                    "field_mapping_error",
                    raw_name,
                    "must use a fixed-size list of two stable Arrow primitive values",
                )
            arrow_leaf = raw_arrow_type.value_type
            primitive_contract = _raw_primitive_contract(arrow_leaf)
            if primitive_contract is None:
                _fail(
                    "field_mapping_error",
                    raw_name,
                    f"unsupported supplier raw Arrow primitive {arrow_leaf}",
                )
            expected_dtype, numpy_dtype = primitive_contract
            if raw_dtype != expected_dtype:
                _fail(
                    "field_mapping_error",
                    f"info.features.{raw_name}.dtype",
                    f"expected {expected_dtype!r} for Arrow type {arrow_leaf}",
                )
            declared(raw_name, expected_dtype, raw_arrow_type)
            raw_values = table[raw_name].to_pylist()
            if len(raw_values) != frame_count or any(
                not isinstance(row, list)
                or len(row) != 2
                or any(value is None for value in row)
                for row in raw_values
            ):
                _fail("field_mapping_error", raw_name, "must have shape (T, 2) without nulls")
            raw = np.asarray(raw_values, dtype=numpy_dtype)

        score: np.ndarray | None = None
        if score_name in present:
            declared(score_name, "float32", pa.list_(pa.float32(), 2))
            score_values = table[score_name].to_pylist()
            if len(score_values) != frame_count or any(
                not isinstance(row, list)
                or len(row) != 2
                or any(value is None for value in row)
                for row in score_values
            ):
                _fail("field_mapping_error", score_name, "must have shape (T, 2) without nulls")
            score = np.asarray(score_values, dtype=np.float32)
        return SupplierEvidence(hand_quality=SupplierHandQuality(provided=True, raw_value=raw, normalized_score=score, status=status, mapping_version=state["mapping_version"]))
