"""Strict, non-repairing validation for ``CanonicalQcEpisode.v1``."""

from __future__ import annotations

from numbers import Integral
from pathlib import PurePosixPath
import re

import numpy as np

from .contracts import (
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
from .errors import CanonicalInputError
from .provenance import source_fingerprint


_SHA256 = re.compile(r"[0-9a-f]{64}").fullmatch
_HAND_QUALITY_STATUSES = frozenset({"bad", "warning", "good", "unknown"})


def _fail(code: str, field: str, detail: str) -> None:
    raise CanonicalInputError(code, field, detail)


def _contract_type(value: object, expected: type[object], field: str) -> None:
    if type(value) is not expected:
        _fail(
            "invalid_contract_type",
            field,
            f"must be {expected.__name__}, got {type(value).__name__}",
        )


def _validate_contract_types(episode: CanonicalQcEpisode) -> None:
    contracts = (
        (episode.identity, EpisodeIdentity, "identity"),
        (episode.provenance, SourceProvenance, "provenance"),
        (episode.time_axis, TimeAxis, "time_axis"),
        (episode.main_video, VideoStream, "main_video"),
        (episode.observation, HandObservation, "observation"),
        (episode.calibration, CameraCalibration, "calibration"),
        (episode.semantics, EpisodeSemantics, "semantics"),
        (episode.supplier_evidence, SupplierEvidence, "supplier_evidence"),
    )
    for value, expected, field in contracts:
        _contract_type(value, expected, field)
    for index, source_file in enumerate(episode.provenance.source_files):
        _contract_type(
            source_file,
            SourceFile,
            f"provenance.source_files[{index}]",
        )
    for index, subtask in enumerate(episode.semantics.subtask_sequence):
        _contract_type(
            subtask,
            Subtask,
            f"semantics.subtask_sequence[{index}]",
        )
    hand_quality = episode.supplier_evidence.hand_quality
    if hand_quality is not None:
        _contract_type(
            hand_quality,
            SupplierHandQuality,
            "supplier_evidence.hand_quality",
        )


def _nonempty_string(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        _fail("invalid_string", field, "must be a non-empty string")


def _positive_int(value: object, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        _fail("invalid_integer", field, "must be an integer greater than zero")


def _nonnegative_int(value: object, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        _fail("invalid_integer", field, "must be a non-negative integer")


def _constant(value: object, expected: object, field: str) -> None:
    if value != expected:
        _fail("invalid_constant", field, f"must equal {expected!r}")


def _sha256(value: object, field: str) -> None:
    if not isinstance(value, str) or _SHA256(value) is None:
        _fail("invalid_sha256", field, "must be 64 lowercase hexadecimal characters")


def _relative_path(value: object, field: str) -> None:
    _nonempty_string(value, field)
    assert isinstance(value, str)
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or "\\" in value
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        _fail("invalid_relative_path", field, "must be a normalized relative POSIX path")


def _array(
    value: np.ndarray,
    *,
    field: str,
    dtype: np.dtype[object] | type[object],
    shape: tuple[int, ...],
) -> None:
    expected_dtype = np.dtype(dtype)
    if value.dtype != expected_dtype:
        _fail(
            "invalid_dtype",
            field,
            f"expected {expected_dtype.name}, got {value.dtype.name}",
        )
    if value.shape != shape:
        _fail("invalid_shape", field, f"expected {shape}, got {value.shape}")
    if value.flags.writeable:
        _fail("mutable_array", field, "array must be read-only")


def _validate_identity(episode: CanonicalQcEpisode) -> None:
    _constant(episode.schema_version, "canonical_qc_episode.v1", "schema_version")
    _constant(episode.profile, "human_ego_hand_pose.v1", "profile")
    identity = episode.identity
    for name in ("asset_id", "batch_id", "supplier_id", "source_schema_version"):
        _nonempty_string(getattr(identity, name), f"identity.{name}")
    if "/" in identity.asset_id or "\\" in identity.asset_id:
        _fail(
            "invalid_asset_id",
            "identity.asset_id",
            "must not contain path separators",
        )
    if identity.source_format not in ("hdf5", "lerobot"):
        _fail(
            "invalid_source_format",
            "identity.source_format",
            "must be 'hdf5' or 'lerobot'",
        )


def _validate_provenance(episode: CanonicalQcEpisode) -> None:
    provenance = episode.provenance
    _nonempty_string(provenance.adapter_id, "provenance.adapter_id")
    _nonempty_string(provenance.adapter_version, "provenance.adapter_version")
    if not provenance.source_files:
        _fail("missing_source_files", "provenance.source_files", "must not be empty")
    seen: set[str] = set()
    for index, item in enumerate(provenance.source_files):
        prefix = f"provenance.source_files[{index}]"
        _relative_path(item.relative_path, f"{prefix}.relative_path")
        _nonempty_string(item.role, f"{prefix}.role")
        _nonnegative_int(item.size_bytes, f"{prefix}.size_bytes")
        _sha256(item.sha256, f"{prefix}.sha256")
        if item.relative_path in seen:
            _fail(
                "duplicate_source_path",
                "provenance.source_files",
                f"duplicate relative path {item.relative_path!r}",
            )
        seen.add(item.relative_path)
    video_sources = [
        item for item in provenance.source_files if item.role == "main_video"
    ]
    if not video_sources:
        _fail(
            "missing_main_video_source",
            "provenance.source_files",
            "must contain the main video source",
        )
    if len(video_sources) != 1:
        _fail(
            "main_video_source_mismatch",
            "provenance.source_files",
            "must contain exactly one main_video role",
        )
    video_source = video_sources[0]
    if (
        video_source.relative_path != episode.main_video.path
        or video_source.sha256 != episode.main_video.sha256
    ):
        _fail(
            "main_video_source_mismatch",
            "provenance.source_files",
            "main_video source path and sha256 must match main_video metadata",
        )
    _sha256(provenance.source_fingerprint, "provenance.source_fingerprint")
    expected = source_fingerprint(
        provenance.source_files,
        source_schema_version=episode.identity.source_schema_version,
        adapter_id=provenance.adapter_id,
        adapter_version=provenance.adapter_version,
    )
    if provenance.source_fingerprint != expected:
        _fail(
            "source_fingerprint_mismatch",
            "provenance.source_fingerprint",
            f"expected {expected}, got {provenance.source_fingerprint}",
        )


def _validate_time_axis(episode: CanonicalQcEpisode) -> None:
    time_axis = episode.time_axis
    _positive_int(time_axis.frame_count, "time_axis.frame_count")
    _positive_int(time_axis.fps_num, "time_axis.fps_num")
    _positive_int(time_axis.fps_den, "time_axis.fps_den")
    _nonnegative_int(time_axis.frame_index_base, "time_axis.frame_index_base")
    _constant(time_axis.frame_index_base, 0, "time_axis.frame_index_base")
    _constant(time_axis.interval_semantics, "half_open", "time_axis.interval_semantics")
    _array(
        time_axis.timestamps_ns,
        field="time_axis.timestamps_ns",
        dtype=np.int64,
        shape=(time_axis.frame_count,),
    )
    if time_axis.frame_count > 1 and np.any(
        time_axis.timestamps_ns[1:] <= time_axis.timestamps_ns[:-1]
    ):
        _fail(
            "timestamps_not_strictly_increasing",
            "time_axis.timestamps_ns",
            "adjacent timestamps must increase",
        )


def _validate_observation(episode: CanonicalQcEpisode) -> None:
    observation = episode.observation
    frame_count = episode.time_axis.frame_count
    specs = (
        ("hand_keypoints_3d", np.float32, (frame_count, 2, 21, 3)),
        ("hand_joint_valid_3d", np.bool_, (frame_count, 2, 21)),
        ("hand_keypoints_2d", np.float32, (frame_count, 2, 21, 2)),
        ("hand_joint_valid_2d", np.bool_, (frame_count, 2, 21)),
    )
    for name, dtype, shape in specs:
        _array(
            getattr(observation, name),
            field=f"observation.{name}",
            dtype=dtype,
            shape=shape,
        )
    constants = {
        "hand_order": ("left", "right"),
        "joint_topology": "egodata_hand21.v1",
        "coordinate_frame_3d": "camera:main",
        "length_unit": "meter",
        "coordinate_space_2d": "pixel",
    }
    for name, expected in constants.items():
        _constant(getattr(observation, name), expected, f"observation.{name}")
    for dimension in ("3d", "2d"):
        points = getattr(observation, f"hand_keypoints_{dimension}")
        valid = getattr(observation, f"hand_joint_valid_{dimension}")
        finite = np.isfinite(points).all(axis=-1)
        all_nan = np.isnan(points).all(axis=-1)
        if np.any(valid & ~finite) or np.any(~valid & ~all_nan):
            _fail(
                "invalid_coordinate_validity",
                f"observation.hand_keypoints_{dimension}",
                "valid joints must be finite and invalid joints must be all-NaN",
            )


def _validate_video(episode: CanonicalQcEpisode) -> None:
    video = episode.main_video
    _constant(video.camera_id, "main", "main_video.camera_id")
    _constant(video.camera_role, "ego", "main_video.camera_role")
    _relative_path(video.path, "main_video.path")
    _sha256(video.sha256, "main_video.sha256")
    for name in ("frame_count", "width_px", "height_px", "fps_num", "fps_den"):
        _positive_int(getattr(video, name), f"main_video.{name}")
    if video.frame_count != episode.time_axis.frame_count:
        _fail(
            "video_frame_count_mismatch",
            "main_video.frame_count",
            f"expected {episode.time_axis.frame_count}, got {video.frame_count}",
        )
    _nonempty_string(video.codec, "main_video.codec")
    _nonempty_string(video.pixel_format, "main_video.pixel_format")


def _validate_calibration(episode: CanonicalQcEpisode) -> None:
    calibration = episode.calibration
    _array(
        calibration.intrinsic_matrix,
        field="calibration.intrinsic_matrix",
        dtype=np.float64,
        shape=(3, 3),
    )
    if not np.isfinite(calibration.intrinsic_matrix).all():
        _fail(
            "nonfinite_calibration",
            "calibration.intrinsic_matrix",
            "must contain only finite values",
        )
    if calibration.distortion_model != "none":
        _fail(
            "unsupported_distortion_model",
            "calibration.distortion_model",
            "only the registered v1 model 'none' is supported",
        )
    coefficients = calibration.distortion_coefficients
    coefficient_field = "calibration.distortion_coefficients"
    if coefficients.dtype != np.dtype(np.float64):
        _fail(
            "invalid_dtype",
            coefficient_field,
            f"expected float64, got {coefficients.dtype.name}",
        )
    if coefficients.ndim != 1:
        _fail(
            "invalid_shape",
            coefficient_field,
            f"expected a one-dimensional array, got {coefficients.shape}",
        )
    if coefficients.flags.writeable:
        _fail("mutable_array", coefficient_field, "array must be read-only")
    if calibration.distortion_coefficients.size != 0:
        _fail(
            "invalid_distortion_coefficients",
            "calibration.distortion_coefficients",
            "model 'none' requires an empty array",
        )
    for name in ("image_width_px", "image_height_px"):
        _positive_int(getattr(calibration, name), f"calibration.{name}")
    if calibration.image_width_px != episode.main_video.width_px:
        _fail(
            "calibration_video_mismatch",
            "calibration.image_width_px",
            f"expected {episode.main_video.width_px}, got {calibration.image_width_px}",
        )
    if calibration.image_height_px != episode.main_video.height_px:
        _fail(
            "calibration_video_mismatch",
            "calibration.image_height_px",
            f"expected {episode.main_video.height_px}, got {calibration.image_height_px}",
        )
    _constant(
        calibration.camera_axes,
        "x_right_y_down_z_forward",
        "calibration.camera_axes",
    )
    _constant(calibration.pixel_origin, "top_left", "calibration.pixel_origin")


def _validate_semantics(episode: CanonicalQcEpisode) -> None:
    semantics = episode.semantics
    for name in (
        "scene_id",
        "task_id",
        "task_category",
        "task_cn",
        "task_en",
        "description_cn",
        "description_en",
    ):
        _nonempty_string(getattr(semantics, name), f"semantics.{name}")
    subtasks = semantics.subtask_sequence
    frame_count = episode.time_axis.frame_count
    if not subtasks:
        _fail(
            "invalid_subtask_sequence",
            "semantics.subtask_sequence",
            "must not be empty",
        )
    expected_start = 0
    seen_ids: set[str] = set()
    for index, item in enumerate(subtasks):
        for name in ("subtask_id", "description_cn", "description_en"):
            _nonempty_string(
                getattr(item, name),
                f"semantics.subtask_sequence[{index}].{name}",
            )
        if item.subtask_id in seen_ids:
            _fail(
                "invalid_subtask_sequence",
                "semantics.subtask_sequence",
                f"duplicate subtask_id {item.subtask_id!r}",
            )
        seen_ids.add(item.subtask_id)
        if (
            isinstance(item.start_frame, bool)
            or not isinstance(item.start_frame, Integral)
            or isinstance(item.end_frame_exclusive, bool)
            or not isinstance(item.end_frame_exclusive, Integral)
            or item.start_frame != expected_start
            or item.start_frame >= item.end_frame_exclusive
        ):
            _fail(
                "invalid_subtask_sequence",
                "semantics.subtask_sequence",
                f"segment {index} must start at {expected_start} and have positive length",
            )
        expected_start = item.end_frame_exclusive
    if expected_start != frame_count:
        _fail(
            "invalid_subtask_sequence",
            "semantics.subtask_sequence",
            f"final boundary must equal frame_count {frame_count}, got {expected_start}",
        )


def _validate_supplier_evidence(episode: CanonicalQcEpisode) -> None:
    hand_quality = episode.supplier_evidence.hand_quality
    if hand_quality is None:
        return
    if not isinstance(hand_quality.provided, bool):
        _fail(
            "invalid_boolean",
            "supplier_evidence.hand_quality.provided",
            "must be bool",
        )
    if hand_quality.provided:
        if hand_quality.status is None:
            _fail(
                "invalid_hand_quality_state",
                "supplier_evidence.hand_quality.status",
                "provided=true requires canonical status",
            )
        if (
            not isinstance(hand_quality.mapping_version, str)
            or not hand_quality.mapping_version.strip()
        ):
            _fail(
                "invalid_hand_quality_state",
                "supplier_evidence.hand_quality.mapping_version",
                "provided=true requires a non-empty mapping version",
            )
    elif any(
        value is not None
        for value in (
            hand_quality.raw_value,
            hand_quality.normalized_score,
            hand_quality.mapping_version,
        )
    ):
        _fail(
            "invalid_hand_quality_state",
            "supplier_evidence.hand_quality",
            "provided=false forbids raw, normalized, and mapping evidence",
        )
    shape = (episode.time_axis.frame_count, 2)
    if hand_quality.raw_value is not None:
        if hand_quality.raw_value.dtype.hasobject:
            _fail(
                "invalid_dtype",
                "supplier_evidence.hand_quality.raw_value",
                "object dtype is not a stable supplier-defined representation",
            )
        if hand_quality.raw_value.shape != shape:
            _fail(
                "invalid_shape",
                "supplier_evidence.hand_quality.raw_value",
                f"expected {shape}, got {hand_quality.raw_value.shape}",
            )
        if hand_quality.raw_value.flags.writeable:
            _fail(
                "mutable_array",
                "supplier_evidence.hand_quality.raw_value",
                "array must be read-only",
            )
    if hand_quality.normalized_score is not None:
        field = "supplier_evidence.hand_quality.normalized_score"
        _array(
            hand_quality.normalized_score,
            field=field,
            dtype=np.float32,
            shape=shape,
        )
        scores = hand_quality.normalized_score
        if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
            _fail(
                "invalid_hand_quality_score",
                field,
                "must contain finite values in [0, 1]",
            )
    if hand_quality.status is not None:
        field = "supplier_evidence.hand_quality.status"
        statuses = hand_quality.status
        if statuses.shape != shape:
            _fail("invalid_shape", field, f"expected {shape}, got {statuses.shape}")
        if statuses.dtype.kind not in ("U", "S"):
            _fail("invalid_dtype", field, "must contain canonical status strings")
        try:
            decoded = {
                item.decode("utf-8") if isinstance(item, bytes) else str(item)
                for item in statuses.flat
            }
        except UnicodeDecodeError:
            _fail(
                "invalid_hand_quality_status",
                field,
                "byte statuses must be valid UTF-8",
            )
        unknown = decoded - _HAND_QUALITY_STATUSES
        if unknown:
            _fail(
                "invalid_hand_quality_status",
                field,
                f"unknown statuses {sorted(unknown)!r}",
            )
        if not hand_quality.provided and decoded != {"unknown"}:
            _fail(
                "invalid_hand_quality_status",
                field,
                "provided=false permits only 'unknown'",
            )
    if hand_quality.mapping_version is not None:
        _nonempty_string(
            hand_quality.mapping_version,
            "supplier_evidence.hand_quality.mapping_version",
        )


def validate_episode(episode: CanonicalQcEpisode) -> None:
    """Validate without coercing, truncating, regenerating, or repairing data."""

    _contract_type(episode, CanonicalQcEpisode, "episode")
    _validate_contract_types(episode)
    _validate_identity(episode)
    _validate_provenance(episode)
    _validate_time_axis(episode)
    _validate_observation(episode)
    _validate_video(episode)
    _validate_calibration(episode)
    _validate_semantics(episode)
    _validate_supplier_evidence(episode)


def validate_video_alignment(
    time_axis: TimeAxis,
    video: ProbedVideo,
    *,
    max_delta_ns: int,
) -> None:
    """Compare authoritative Canonical timestamps with normalized video PTS."""

    _nonnegative_int(max_delta_ns, "max_delta_ns")
    expected_count = time_axis.frame_count
    _positive_int(expected_count, "time_axis.frame_count")
    if len(time_axis.timestamps_ns) != expected_count:
        _fail(
            "timebase_invalid",
            "time_axis.timestamps_ns",
            (
                f"expected {expected_count} canonical timestamps, "
                f"got {len(time_axis.timestamps_ns)}"
            ),
        )
    if video.frame_count != expected_count or len(video.timestamps_ns) != expected_count:
        _fail(
            "timebase_invalid",
            "main_video.timestamps_ns",
            (
                f"expected {expected_count} frame PTS, got {len(video.timestamps_ns)} "
                f"for {video.frame_count} probed frames"
            ),
        )

    video_timestamps: list[int] = []
    for index, value in enumerate(video.timestamps_ns):
        if isinstance(value, bool) or not isinstance(value, Integral):
            _fail(
                "timebase_invalid",
                f"main_video.timestamps_ns[{index}]",
                "must be an exact integer nanosecond timestamp",
            )
        video_timestamps.append(int(value))
    if any(
        current <= previous
        for previous, current in zip(video_timestamps, video_timestamps[1:])
    ):
        _fail(
            "timebase_invalid",
            "main_video.timestamps_ns",
            "video PTS must be strictly increasing",
        )

    canonical_timestamps: list[int] = []
    for index, value in enumerate(time_axis.timestamps_ns):
        if isinstance(value, bool) or not isinstance(value, Integral):
            _fail(
                "timebase_invalid",
                f"time_axis.timestamps_ns[{index}]",
                "must be an exact integer nanosecond timestamp",
            )
        canonical_timestamps.append(int(value))
    canonical_zero = canonical_timestamps[0]
    video_zero = video_timestamps[0]
    for index, (canonical_timestamp, video_timestamp) in enumerate(
        zip(canonical_timestamps, video_timestamps)
    ):
        delta_ns = abs(
            (canonical_timestamp - canonical_zero) - (video_timestamp - video_zero)
        )
        if delta_ns > max_delta_ns:
            _fail(
                "timebase_invalid",
                f"main_video.timestamps_ns[{index}]",
                f"timestamp delta {delta_ns} ns exceeds tolerance {max_delta_ns} ns",
            )
