"""Immutable in-memory contracts for ``CanonicalQcEpisode.v1``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .errors import CanonicalInputError


def _immutable_array(value: NDArray[object], *, field_name: str) -> NDArray[object]:
    """Copy an array onto immutable storage without changing dtype or values."""

    array = np.asarray(value)
    if array.dtype.hasobject:
        raise CanonicalInputError(
            "invalid_dtype",
            field_name,
            "object dtype cannot be backed by immutable byte storage",
        )
    immutable = np.frombuffer(
        array.tobytes(order="C"),
        dtype=array.dtype,
        count=array.size,
    ).reshape(array.shape)
    immutable.flags.writeable = False
    return immutable


@dataclass(frozen=True, slots=True)
class SourceFile:
    relative_path: str
    role: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class EpisodeIdentity:
    asset_id: str
    batch_id: str
    supplier_id: str
    source_format: Literal["hdf5", "lerobot"]
    source_schema_version: str


@dataclass(frozen=True, slots=True)
class SourceProvenance:
    source_files: tuple[SourceFile, ...]
    source_fingerprint: str
    adapter_id: str
    adapter_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_files", tuple(self.source_files))


@dataclass(frozen=True, slots=True)
class TimeAxis:
    frame_count: int
    timestamps_ns: NDArray[np.int64]
    fps_num: int
    fps_den: int
    frame_index_base: Literal[0] = 0
    interval_semantics: Literal["half_open"] = "half_open"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "timestamps_ns",
            _immutable_array(self.timestamps_ns, field_name="time_axis.timestamps_ns"),
        )


@dataclass(frozen=True, slots=True)
class VideoStream:
    path: str
    sha256: str
    frame_count: int
    width_px: int
    height_px: int
    fps_num: int
    fps_den: int
    codec: str
    pixel_format: str
    camera_id: Literal["main"] = "main"
    camera_role: Literal["ego"] = "ego"


@dataclass(frozen=True, slots=True)
class HandObservation:
    hand_keypoints_3d: NDArray[np.float32]
    hand_joint_valid_3d: NDArray[np.bool_]
    hand_keypoints_2d: NDArray[np.float32]
    hand_joint_valid_2d: NDArray[np.bool_]
    hand_order: tuple[Literal["left"], Literal["right"]] = ("left", "right")
    joint_topology: Literal["egodata_hand21.v1"] = "egodata_hand21.v1"
    coordinate_frame_3d: Literal["camera:main"] = "camera:main"
    length_unit: Literal["meter"] = "meter"
    coordinate_space_2d: Literal["pixel"] = "pixel"

    def __post_init__(self) -> None:
        for field_name in (
            "hand_keypoints_3d",
            "hand_joint_valid_3d",
            "hand_keypoints_2d",
            "hand_joint_valid_2d",
        ):
            object.__setattr__(
                self,
                field_name,
                _immutable_array(
                    getattr(self, field_name),
                    field_name=f"observation.{field_name}",
                ),
            )
        object.__setattr__(self, "hand_order", tuple(self.hand_order))


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    intrinsic_matrix: NDArray[np.float64]
    distortion_model: str
    distortion_coefficients: NDArray[np.float64]
    image_width_px: int
    image_height_px: int
    camera_axes: Literal["x_right_y_down_z_forward"] = "x_right_y_down_z_forward"
    pixel_origin: Literal["top_left"] = "top_left"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "intrinsic_matrix",
            _immutable_array(
                self.intrinsic_matrix,
                field_name="calibration.intrinsic_matrix",
            ),
        )
        object.__setattr__(
            self,
            "distortion_coefficients",
            _immutable_array(
                self.distortion_coefficients,
                field_name="calibration.distortion_coefficients",
            ),
        )


@dataclass(frozen=True, slots=True)
class Subtask:
    subtask_id: str
    start_frame: int
    end_frame_exclusive: int
    description_cn: str
    description_en: str


@dataclass(frozen=True, slots=True)
class EpisodeSemantics:
    scene_id: str
    task_id: str
    task_category: str
    task_cn: str
    task_en: str
    description_cn: str
    description_en: str
    subtask_sequence: tuple[Subtask, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "subtask_sequence", tuple(self.subtask_sequence))


@dataclass(frozen=True, slots=True)
class SupplierHandQuality:
    provided: bool
    raw_value: NDArray[object] | None = None
    normalized_score: NDArray[np.float32] | None = None
    status: NDArray[np.str_] | None = None
    mapping_version: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("raw_value", "normalized_score", "status"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _immutable_array(
                        value,
                        field_name=f"supplier_evidence.hand_quality.{field_name}",
                    ),
                )


@dataclass(frozen=True, slots=True)
class SupplierEvidence:
    hand_quality: SupplierHandQuality | None = None


@dataclass(frozen=True, slots=True)
class CanonicalQcEpisode:
    schema_version: Literal["canonical_qc_episode.v1"]
    profile: Literal["human_ego_hand_pose.v1"]
    identity: EpisodeIdentity
    provenance: SourceProvenance
    time_axis: TimeAxis
    main_video: VideoStream
    observation: HandObservation
    calibration: CameraCalibration
    semantics: EpisodeSemantics
    supplier_evidence: SupplierEvidence = field(default_factory=SupplierEvidence)
