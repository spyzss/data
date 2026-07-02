"""Default supplier HDF5 adapter for precheck ClipInputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from qc_common.hdf5_loader import read_scalar_json
from qc_common.types import ClipInputs


def load_supplier_hdf5_clip(
    path: str | Path,
    episode_idx: int | None = None,
    fps: float | None = None,
) -> ClipInputs:
    """
    Load one supplier HDF5 episode using the confirmed schema.

    The adapter converts transforms/<joint> SE(3) matrices into 3D joint
    positions from matrix[:3, 3] and rotations from matrix[:3, :3]. Other
    suppliers can inject a different callable that returns ClipInputs.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required to load supplier HDF5 files") from exc

    hdf5_path = Path(path)
    with h5py.File(hdf5_path, "r") as handle:
        transforms = handle["transforms"]
        keypoints: dict[str, np.ndarray] = {}
        rotations: dict[str, np.ndarray] = {}
        for joint in transforms.keys():
            matrix = np.asarray(transforms[joint], dtype=np.float32)
            keypoints[joint] = matrix[:, :3, 3]
            rotations[joint] = matrix[:, :3, :3]

        confidences: dict[str, np.ndarray] = {}
        if "confidences" in handle:
            confidences = {
                joint: np.asarray(handle["confidences"][joint], dtype=np.float32)
                for joint in handle["confidences"].keys()
            }

        quality_hand = None
        if "label" in handle and "quality_hand" in handle["label"]:
            quality_hand = np.asarray(handle["label"]["quality_hand"], dtype=np.float32)

        instruction = ""
        text_label = None
        text_label_raw = None
        text_label_parse_error = None
        if "label" in handle and "text_label" in handle["label"]:
            (
                text_label,
                text_label_raw,
                text_label_parse_error,
            ) = read_scalar_json(handle["label"]["text_label"])
            if text_label is not None:
                instruction = str(text_label.get("text_en", ""))

        intrinsics = None
        if "camera" in handle and "intrinsic" in handle["camera"]:
            intrinsics = np.asarray(handle["camera"]["intrinsic"], dtype=np.float32)

        discovered_fps = fps
        if discovered_fps is None:
            for key in ("fps", "frame_rate"):
                if key in handle.attrs:
                    discovered_fps = float(handle.attrs[key])
                    break
        if discovered_fps is None:
            discovered_fps = 30.0

    num_frames = int(next(iter(keypoints.values())).shape[0]) if keypoints else 0
    if episode_idx is None:
        digits = "".join(ch for ch in hdf5_path.stem if ch.isdigit())
        episode_idx = int(digits) if digits else 0

    return ClipInputs(
        episode_idx=episode_idx,
        frame_indices=list(range(num_frames)),
        keypoints=keypoints,
        rotations=rotations,
        confidences=confidences,
        quality_hand=quality_hand,
        instruction=instruction,
        text_label=text_label,
        text_label_raw=text_label_raw,
        text_label_parse_error=text_label_parse_error,
        intrinsics=intrinsics,
        fps=discovered_fps,
    )
