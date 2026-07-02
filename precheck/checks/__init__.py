"""Import check modules so decorators populate the registry."""

from . import (
    composite_frame_verdict,
    keypoint_missing,
    keypoint_temporal,
    mask_containment,
    overexposure,
    quality_score,
    skeleton_quality_score,
    text_integrity,
)

__all__ = [
    "composite_frame_verdict",
    "keypoint_missing",
    "keypoint_temporal",
    "mask_containment",
    "overexposure",
    "quality_score",
    "skeleton_quality_score",
    "text_integrity",
]
