"""Strict reader for the fixed ``egodata_hdf5_qc_input.v1`` layout."""

from __future__ import annotations

import hashlib
from numbers import Integral
from pathlib import Path
import stat

import h5py
import numpy as np

from ..contracts import (
    CameraCalibration,
    CanonicalQcEpisode,
    EpisodeIdentity,
    HandObservation,
    SourceFile,
    SourceProvenance,
    TimeAxis,
    VideoStream,
)
from ..errors import CanonicalInputError
from ..provenance import source_fingerprint
from ..validation import validate_episode, validate_video_alignment
from ..video_probe import probe_video
from ._standard_hdf5_readers import (
    array as _array,
    distortion_coefficients as _distortion_coefficients,
    fail as _fail,
    group as _group,
    required_attr as _required_attr,
    semantics as _semantics,
    supplier_evidence as _supplier_evidence,
    text as _text,
    typed_integer as _typed_integer,
)
from .base import SourceInspection


_SCHEMA_VERSION = "egodata_hdf5_qc_input.v1"
_CANONICAL_ERROR_CODES = frozenset(
    {
        "schema_missing",
        "field_mapping_error",
        "timebase_invalid",
        "source_integrity_error",
    }
)


def _mapped_error(error: CanonicalInputError) -> CanonicalInputError:
    if error.code in _CANONICAL_ERROR_CODES:
        return error
    return CanonicalInputError("field_mapping_error", error.field, error.detail)


def _file_metadata(path: Path, *, role: str, relative_path: str) -> SourceFile:
    try:
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            _fail("source_integrity_error", relative_path, "must be a regular file")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
    except CanonicalInputError:
        raise
    except OSError as exc:
        _fail("source_integrity_error", relative_path, f"cannot stat/hash file: {exc}")
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        _fail("source_integrity_error", relative_path, "file changed while hashing")
    return SourceFile(
        relative_path=relative_path,
        role=role,
        size_bytes=after.st_size,
        sha256=digest.hexdigest(),
    )


class StandardHdf5Adapter:
    adapter_id = "standard_hdf5"
    adapter_version = "1.0.0"

    def __init__(self, *, max_timestamp_delta_ns: int = 1_000_000) -> None:
        if (
            isinstance(max_timestamp_delta_ns, bool)
            or not isinstance(max_timestamp_delta_ns, Integral)
            or max_timestamp_delta_ns < 0
        ):
            raise CanonicalInputError(
                "field_mapping_error",
                "max_timestamp_delta_ns",
                "must be a non-negative integer",
            )
        self.max_timestamp_delta_ns = int(max_timestamp_delta_ns)

    def inspect(self, source: Path) -> SourceInspection:
        source_path = Path(source).expanduser()
        if not source_path.is_absolute():
            source_path = Path.cwd() / source_path
        if source_path.is_dir():
            source_root = source_path.resolve()
            candidates = sorted(source_root.glob("*.h5"))
            if len(candidates) != 1:
                _fail(
                    "source_integrity_error",
                    "source",
                    f"episode directory must contain exactly one .h5 file, found {len(candidates)}",
                )
            if candidates[0].is_symlink():
                _fail(
                    "source_integrity_error",
                    "source",
                    "HDF5 path must not be a symlink",
                )
            hdf5_path = candidates[0].resolve()
        elif source_path.suffix == ".h5" and source_path.is_file():
            if source_path.is_symlink():
                _fail(
                    "source_integrity_error",
                    "source",
                    "HDF5 path must not be a symlink",
                )
            source_root = source_path.parent.resolve()
            hdf5_path = source_path.resolve()
        else:
            _fail(
                "source_integrity_error",
                "source",
                "must be an episode directory or an existing .h5 file",
            )
        main_video_source = source_root / "main.mp4"
        if main_video_source.is_symlink():
            _fail(
                "source_integrity_error",
                "main_video.path",
                "main.mp4 must not be a symlink alias",
            )
        main_video_path = main_video_source.resolve()
        if hdf5_path.parent != source_root:
            _fail("source_integrity_error", "source", "HDF5 path escapes source root")
        if main_video_path.parent != source_root:
            _fail(
                "source_integrity_error",
                "main_video.path",
                "main video path escapes source root",
            )
        if not hdf5_path.is_file():
            _fail("source_integrity_error", "source", "HDF5 path is not a regular file")
        if not main_video_path.is_file():
            _fail(
                "source_integrity_error",
                "main_video.path",
                "required main.mp4 is missing or not a regular file",
            )
        try:
            with h5py.File(hdf5_path, "r") as handle:
                schema_version = _text(
                    _required_attr(handle, "schema_version"),
                    field="/@schema_version",
                )
                asset_id = _text(
                    _required_attr(handle, "asset_id"), field="/@asset_id"
                )
                _text(_required_attr(handle, "batch_id"), field="/@batch_id")
                _text(_required_attr(handle, "supplier_id"), field="/@supplier_id")
        except CanonicalInputError:
            raise
        except (OSError, ValueError, TypeError) as exc:
            _fail(
                "source_integrity_error",
                "source",
                f"cannot open HDF5 read-only: {exc}",
            )
        if schema_version != _SCHEMA_VERSION:
            _fail(
                "field_mapping_error",
                "/@schema_version",
                f"expected {_SCHEMA_VERSION!r}, got {schema_version!r}",
            )
        if hdf5_path.name != f"{asset_id}.h5" or source_root.name != asset_id:
            _fail(
                "field_mapping_error",
                "/@asset_id",
                "asset_id must match both episode directory and HDF5 filename",
            )
        return SourceInspection(
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            source_format="hdf5",
            source_schema_version=schema_version,
            asset_id=asset_id,
            source_root=source_root,
            hdf5_path=hdf5_path,
            main_video_path=main_video_path,
        )

    def load(self, source: Path) -> CanonicalQcEpisode:
        inspection = self.inspect(source)
        hdf5_relative = inspection.hdf5_path.relative_to(
            inspection.source_root
        ).as_posix()
        video_relative = inspection.main_video_path.relative_to(
            inspection.source_root
        ).as_posix()
        hdf5_source = _file_metadata(
            inspection.hdf5_path,
            role="episode_data",
            relative_path=hdf5_relative,
        )
        video_source = _file_metadata(
            inspection.main_video_path,
            role="main_video",
            relative_path=video_relative,
        )
        try:
            with h5py.File(inspection.hdf5_path, "r") as handle:
                episode = self._read_episode(
                    handle,
                    inspection=inspection,
                    source_files=(hdf5_source, video_source),
                )
        except CanonicalInputError as exc:
            raise _mapped_error(exc) from exc
        except (OSError, ValueError, TypeError) as exc:
            _fail(
                "source_integrity_error",
                "source",
                f"cannot read HDF5: {exc}",
            )
        try:
            probed_video = probe_video(inspection.main_video_path)
            validate_video_alignment(
                episode.time_axis,
                probed_video,
                max_delta_ns=self.max_timestamp_delta_ns,
            )
            episode = CanonicalQcEpisode(
                schema_version=episode.schema_version,
                profile=episode.profile,
                identity=episode.identity,
                provenance=episode.provenance,
                time_axis=episode.time_axis,
                main_video=VideoStream(
                    path=video_relative,
                    sha256=video_source.sha256,
                    frame_count=probed_video.frame_count,
                    width_px=probed_video.width_px,
                    height_px=probed_video.height_px,
                    fps_num=probed_video.fps_num,
                    fps_den=probed_video.fps_den,
                    codec=probed_video.codec,
                    pixel_format=probed_video.pixel_format,
                ),
                observation=episode.observation,
                calibration=episode.calibration,
                semantics=episode.semantics,
                supplier_evidence=episode.supplier_evidence,
            )
            validate_episode(episode)
        except CanonicalInputError as exc:
            raise _mapped_error(exc) from exc
        final_hdf5 = _file_metadata(
            inspection.hdf5_path,
            role="episode_data",
            relative_path=hdf5_relative,
        )
        final_video = _file_metadata(
            inspection.main_video_path,
            role="main_video",
            relative_path=video_relative,
        )
        if (final_hdf5.sha256, final_video.sha256) != (
            hdf5_source.sha256,
            video_source.sha256,
        ):
            _fail(
                "source_integrity_error",
                "source",
                "source files changed while loading",
            )
        return episode

    def _read_episode(
        self,
        handle: h5py.File,
        *,
        inspection: SourceInspection,
        source_files: tuple[SourceFile, SourceFile],
    ) -> CanonicalQcEpisode:
        frame_count = _typed_integer(
            _required_attr(handle, "frame_count"),
            field="/@frame_count",
            dtype=np.int64,
        )
        fps_num = _typed_integer(
            _required_attr(handle, "fps_num"), field="/@fps_num", dtype=np.int64
        )
        fps_den = _typed_integer(
            _required_attr(handle, "fps_den"), field="/@fps_den", dtype=np.int64
        )
        batch_id = _text(_required_attr(handle, "batch_id"), field="/@batch_id")
        supplier_id = _text(
            _required_attr(handle, "supplier_id"), field="/@supplier_id"
        )
        constants = {
            "joint_topology": "egodata_hand21.v1",
            "coordinate_frame_3d": "camera:main",
            "length_unit": "meter",
            "coordinate_space_2d": "pixel",
        }
        constant_values: dict[str, str] = {}
        for name, expected in constants.items():
            value = _text(_required_attr(handle, name), field=f"/@{name}")
            if value != expected:
                _fail(
                    "field_mapping_error",
                    f"/@{name}",
                    f"expected {expected!r}, got {value!r}",
                )
            constant_values[name] = value
        timestamps = _array(
            handle,
            "/time/timestamps_ns",
            dtype=np.int64,
            shape=(frame_count,),
        )
        observation = HandObservation(
            hand_keypoints_3d=_array(
                handle,
                "/observation/hand_keypoints_3d",
                dtype=np.float32,
                shape=(frame_count, 2, 21, 3),
            ),
            hand_joint_valid_3d=_array(
                handle,
                "/observation/hand_joint_valid_3d",
                dtype=np.bool_,
                shape=(frame_count, 2, 21),
            ),
            hand_keypoints_2d=_array(
                handle,
                "/observation/hand_keypoints_2d",
                dtype=np.float32,
                shape=(frame_count, 2, 21, 2),
            ),
            hand_joint_valid_2d=_array(
                handle,
                "/observation/hand_joint_valid_2d",
                dtype=np.bool_,
                shape=(frame_count, 2, 21),
            ),
            joint_topology=constant_values["joint_topology"],
            coordinate_frame_3d=constant_values["coordinate_frame_3d"],
            length_unit=constant_values["length_unit"],
            coordinate_space_2d=constant_values["coordinate_space_2d"],
        )
        camera_path = "/camera/main"
        camera = _group(handle, camera_path)
        assert camera is not None
        camera_constants = {
            "camera_axes": "x_right_y_down_z_forward",
            "pixel_origin": "top_left",
        }
        camera_values: dict[str, str] = {}
        for name, expected in camera_constants.items():
            value = _text(
                _required_attr(camera, name, prefix=camera_path),
                field=f"{camera_path}@{name}",
            )
            if value != expected:
                _fail(
                    "field_mapping_error",
                    f"{camera_path}@{name}",
                    f"expected {expected!r}, got {value!r}",
                )
            camera_values[name] = value
        calibration = CameraCalibration(
            intrinsic_matrix=_array(
                handle,
                "/camera/main/intrinsic_matrix",
                dtype=np.float64,
                shape=(3, 3),
            ),
            distortion_model=_text(
                _required_attr(camera, "distortion_model", prefix=camera_path),
                field=f"{camera_path}@distortion_model",
            ),
            distortion_coefficients=_distortion_coefficients(handle),
            image_width_px=_typed_integer(
                _required_attr(camera, "image_width_px", prefix=camera_path),
                field=f"{camera_path}@image_width_px",
                dtype=np.int32,
            ),
            image_height_px=_typed_integer(
                _required_attr(camera, "image_height_px", prefix=camera_path),
                field=f"{camera_path}@image_height_px",
                dtype=np.int32,
            ),
            camera_axes=camera_values["camera_axes"],
            pixel_origin=camera_values["pixel_origin"],
        )
        identity = EpisodeIdentity(
            asset_id=inspection.asset_id,
            batch_id=batch_id,
            supplier_id=supplier_id,
            source_format="hdf5",
            source_schema_version=inspection.source_schema_version,
        )
        provenance = SourceProvenance(
            source_files=source_files,
            source_fingerprint=source_fingerprint(
                source_files,
                source_schema_version=identity.source_schema_version,
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
            ),
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
        )
        # Video metadata is replaced after the real ffprobe call. These values are
        # placeholders only inside this private, not-yet-validated construction.
        placeholder_video = VideoStream(
            path=source_files[1].relative_path,
            sha256=source_files[1].sha256,
            frame_count=frame_count,
            width_px=calibration.image_width_px,
            height_px=calibration.image_height_px,
            fps_num=fps_num,
            fps_den=fps_den,
            codec="pending_probe",
            pixel_format="pending_probe",
        )
        return CanonicalQcEpisode(
            schema_version="canonical_qc_episode.v1",
            profile="human_ego_hand_pose.v1",
            identity=identity,
            provenance=provenance,
            time_axis=TimeAxis(
                frame_count=frame_count,
                timestamps_ns=timestamps,
                fps_num=fps_num,
                fps_den=fps_den,
            ),
            main_video=placeholder_video,
            observation=observation,
            calibration=calibration,
            semantics=_semantics(handle),
            supplier_evidence=_supplier_evidence(handle, frame_count),
        )
