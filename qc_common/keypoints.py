"""Utilities for 3D skeleton/keypoint checks."""

from __future__ import annotations

import numpy as np

FINGER_NAMES = ("Thumb", "Index", "Middle", "Ring", "Little")
ACCEPTANCE_FINGER_CHAINS = {
    "Thumb": (
        "ThumbKnuckle",
        "ThumbIntermediateBase",
        "ThumbIntermediateTip",
        "ThumbTip",
    ),
    "Index": (
        "IndexFingerKnuckle",
        "IndexFingerIntermediateBase",
        "IndexFingerIntermediateTip",
        "IndexFingerTip",
    ),
    "Middle": (
        "MiddleFingerKnuckle",
        "MiddleFingerIntermediateBase",
        "MiddleFingerIntermediateTip",
        "MiddleFingerTip",
    ),
    "Ring": (
        "RingFingerKnuckle",
        "RingFingerIntermediateBase",
        "RingFingerIntermediateTip",
        "RingFingerTip",
    ),
    "Little": (
        "LittleFingerKnuckle",
        "LittleFingerIntermediateBase",
        "LittleFingerIntermediateTip",
        "LittleFingerTip",
    ),
}
ACCEPTANCE_HAND_BASE_NAMES = (
    "Hand",
    *ACCEPTANCE_FINGER_CHAINS["Thumb"],
    *ACCEPTANCE_FINGER_CHAINS["Index"],
    *ACCEPTANCE_FINGER_CHAINS["Middle"],
    *ACCEPTANCE_FINGER_CHAINS["Ring"],
    *ACCEPTANCE_FINGER_CHAINS["Little"],
)


def select_hand_joints(
    joint_names: list[str],
    sides: list[str] | tuple[str, ...] = ("left", "right"),
) -> list[str]:
    """Select only the supplier's official 21-per-hand acceptance keypoints."""
    selected: list[str] = []
    available = set(joint_names)
    for side in sides:
        for base_name in ACCEPTANCE_HAND_BASE_NAMES:
            joint_name = f"{side}{base_name}"
            if joint_name in available:
                selected.append(joint_name)
    return sorted(selected)


def acceptance_joint_names(
    sides: list[str] | tuple[str, ...] = ("left", "right"),
) -> list[str]:
    """Return the full canonical acceptance set for the requested sides."""
    return [
        f"{side}{base_name}"
        for side in sides
        for base_name in ACCEPTANCE_HAND_BASE_NAMES
    ]


def _finger_chain(side: str, finger: str) -> list[str]:
    return [f"{side}{base_name}" for base_name in ACCEPTANCE_FINGER_CHAINS[finger]]


def derive_finger_bones(joint_names: list[str]) -> list[tuple[str, str]]:
    """Derive per-finger bone connectivity without hand-listing every bone."""
    available = set(joint_names)
    bones: list[tuple[str, str]] = []
    for side in ("left", "right"):
        hand_root = f"{side}Hand"
        for finger in FINGER_NAMES:
            chain = _finger_chain(side, finger)
            if hand_root in available and chain[0] in available:
                bones.append((hand_root, chain[0]))
            bones.extend(
                (parent, child)
                for parent, child in zip(chain[:-1], chain[1:])
                if parent in available and child in available
            )
    return bones


def derive_angle_triples(joint_names: list[str]) -> list[tuple[str, str, str]]:
    """Derive consecutive triples for anatomical joint-angle metrics."""
    available = set(joint_names)
    triples: list[tuple[str, str, str]] = []
    for side in ("left", "right"):
        hand_root = f"{side}Hand"
        for finger in FINGER_NAMES:
            chain = _finger_chain(side, finger)
            if (
                hand_root in available
                and chain[0] in available
                and chain[1] in available
            ):
                triples.append((hand_root, chain[0], chain[1]))
            triples.extend(
                (a, b, c)
                for a, b, c in zip(chain[:-2], chain[1:-1], chain[2:])
                if a in available and b in available and c in available
            )
    return triples


def project_points(points_xyz: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Project 3D points to pixel coordinates using a 3x3 intrinsic matrix."""
    z = points_xyz[:, 2]
    valid_z = np.where(np.abs(z) < 1e-8, np.nan, z)
    u = intrinsics[0, 0] * points_xyz[:, 0] / valid_z + intrinsics[0, 2]
    v = intrinsics[1, 1] * points_xyz[:, 1] / valid_z + intrinsics[1, 2]
    return np.stack([u, v], axis=1)


def finite_stats(values: list[float] | np.ndarray, prefix: str) -> dict[str, float]:
    """Return compact distribution stats for finite values."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {}
    return {
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_p95": float(np.percentile(array, 95)),
        f"{prefix}_max": float(np.max(array)),
    }
