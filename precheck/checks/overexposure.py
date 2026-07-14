"""Frame-level overexposure check."""

from __future__ import annotations

import numpy as np

from precheck.base import BaseCheck
from precheck.registry import register
from qc_common.types import CheckResult, ClipInputs


@register
class OverexposureCheck(BaseCheck):
    """
    Minimal frame-level check template.

    Future frame checks such as blur or decode-integrity should follow this
    pattern: consume frames only, emit raw metrics, and leave flags unset until
    a threshold is deliberately calibrated.
    """

    name = "overexposure"
    granularity = "frame"

    def __init__(self, config: dict) -> None:
        self.near_saturation = int(config.get("near_saturation", 250))
        self.fraction_threshold = config.get("fraction_threshold")

    def run(self, clip: ClipInputs) -> list[CheckResult]:
        frames = clip.frames
        if frames is None:
            return []

        results: list[CheckResult] = []
        for offset, frame in enumerate(frames):
            array = np.asarray(frame)
            saturated = array >= self.near_saturation
            fraction = float(np.mean(saturated))
            flag = (
                bool(fraction > float(self.fraction_threshold))
                if self.fraction_threshold is not None
                else None
            )
            results.append(
                CheckResult(
                    check=self.name,
                    episode_idx=clip.episode_idx,
                    frame_idx=clip.frame_idx_at(offset),
                    metrics={"near_saturated_fraction": fraction},
                    flag=flag,
                    reason="measured near-saturated pixel fraction",
                )
            )
        return results
