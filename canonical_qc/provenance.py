"""Stable source and semantic fingerprints for Canonical QC."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from numbers import Integral
from typing import Any, Iterable

import numpy as np

from .contracts import CanonicalQcEpisode, SourceFile


def _json_native(value: object) -> object:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Mapping):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    return value


def _stable_json(value: object) -> bytes:
    return json.dumps(
        _json_native(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def source_fingerprint(
    source_files: Iterable[SourceFile],
    *,
    source_schema_version: str,
    adapter_id: str,
    adapter_version: str,
) -> str:
    """Hash source identity independent of input enumeration order."""

    files = sorted(
        (
            {
                "relative_path": item.relative_path,
                "role": item.role,
                "size_bytes": item.size_bytes,
                "sha256": item.sha256,
            }
            for item in source_files
        ),
        key=lambda item: (
            item["relative_path"],
            item["role"],
            item["size_bytes"],
            item["sha256"],
        ),
    )
    payload = {
        "source_schema_version": source_schema_version,
        "adapter_id": adapter_id,
        "adapter_version": adapter_version,
        "source_files": files,
    }
    return hashlib.sha256(_stable_json(payload)).hexdigest()

def _array_payload(
    array: np.ndarray,
    *,
    validity: np.ndarray | None = None,
) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    if np.issubdtype(contiguous.dtype, np.floating):
        contiguous = np.array(contiguous, copy=True, order="C")
        contiguous[contiguous == 0] = contiguous.dtype.type(0)
        if validity is not None:
            contiguous[~validity] = contiguous.dtype.type(np.nan)
    return {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "c_order_sha256": hashlib.sha256(
            contiguous.tobytes(order="C")
        ).hexdigest(),
    }


def semantic_fingerprint(episode: CanonicalQcEpisode) -> str:
    """Hash normalized Core semantics, excluding source format and paths."""

    from .validation import validate_episode

    validate_episode(episode)
    identity = episode.identity
    time_axis = episode.time_axis
    video = episode.main_video
    observation = episode.observation
    calibration = episode.calibration
    semantics = episode.semantics
    payload = {
        "schema_version": episode.schema_version,
        "profile": episode.profile,
        "identity": {
            "asset_id": identity.asset_id,
            "batch_id": identity.batch_id,
            "supplier_id": identity.supplier_id,
        },
        "time_axis": {
            "frame_count": time_axis.frame_count,
            "timestamps_ns": _array_payload(time_axis.timestamps_ns),
            "fps_num": time_axis.fps_num,
            "fps_den": time_axis.fps_den,
            "frame_index_base": time_axis.frame_index_base,
            "interval_semantics": time_axis.interval_semantics,
        },
        "main_video": {
            "sha256": video.sha256,
            "frame_count": video.frame_count,
            "width_px": video.width_px,
            "height_px": video.height_px,
            "fps_num": video.fps_num,
            "fps_den": video.fps_den,
            "codec": video.codec,
            "pixel_format": video.pixel_format,
            "camera_id": video.camera_id,
            "camera_role": video.camera_role,
        },
        "observation": {
            "hand_keypoints_3d": _array_payload(
                observation.hand_keypoints_3d,
                validity=observation.hand_joint_valid_3d,
            ),
            "hand_joint_valid_3d": _array_payload(
                observation.hand_joint_valid_3d
            ),
            "hand_keypoints_2d": _array_payload(
                observation.hand_keypoints_2d,
                validity=observation.hand_joint_valid_2d,
            ),
            "hand_joint_valid_2d": _array_payload(
                observation.hand_joint_valid_2d
            ),
            "hand_order": list(observation.hand_order),
            "joint_topology": observation.joint_topology,
            "coordinate_frame_3d": observation.coordinate_frame_3d,
            "length_unit": observation.length_unit,
            "coordinate_space_2d": observation.coordinate_space_2d,
        },
        "calibration": {
            "intrinsic_matrix": _array_payload(calibration.intrinsic_matrix),
            "distortion_model": calibration.distortion_model,
            "distortion_coefficients": _array_payload(
                calibration.distortion_coefficients
            ),
            "image_width_px": calibration.image_width_px,
            "image_height_px": calibration.image_height_px,
            "camera_axes": calibration.camera_axes,
            "pixel_origin": calibration.pixel_origin,
        },
        "semantics": {
            "scene_id": semantics.scene_id,
            "task_id": semantics.task_id,
            "task_category": semantics.task_category,
            "task_cn": semantics.task_cn,
            "task_en": semantics.task_en,
            "description_cn": semantics.description_cn,
            "description_en": semantics.description_en,
            "subtask_sequence": [
                {
                    "subtask_id": item.subtask_id,
                    "start_frame": item.start_frame,
                    "end_frame_exclusive": item.end_frame_exclusive,
                    "description_cn": item.description_cn,
                    "description_en": item.description_en,
                }
                for item in semantics.subtask_sequence
            ],
        },
    }
    return hashlib.sha256(_stable_json(payload)).hexdigest()
