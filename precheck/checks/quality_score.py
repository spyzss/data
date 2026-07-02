"""Clip-level supplier quality_hand acceptance scoring."""

from __future__ import annotations

import numpy as np

from precheck.base import BaseCheck
from precheck.registry import register
from qc_common.types import CheckResult, ClipInputs

# Sentinel frame_idx for the single clip-level verdict row emitted alongside
# the per-frame score rows. Never collides with a real frame index.
SUMMARY_FRAME_IDX = -1


@register
class QualityScoreCheck(BaseCheck):
    """Apply the supplier quality_hand acceptance scoring rule."""

    name = "quality_score"
    granularity = "clip"

    def __init__(self, config: dict) -> None:
        self.pass_threshold = float(config.get("pass_threshold", 0.90))

    def run(self, clip: ClipInputs) -> list[CheckResult]:
        quality_hand = clip.quality_hand
        if quality_hand is None:
            return []

        quality = np.asarray(quality_hand, dtype=np.float64)
        if quality.ndim != 2 or quality.shape[1] < 2:
            return []

        num_frames = min(clip.num_frames, quality.shape[0])
        if num_frames == 0:
            return [
                CheckResult(
                    check=self.name,
                    episode_idx=clip.episode_idx,
                    frame_idx=SUMMARY_FRAME_IDX,
                    metrics={
                        "total_score": 0.0,
                        "num_frames": 0.0,
                        "pass_ratio": 0.0,
                        "pass_threshold": self.pass_threshold,
                    },
                    flag=False,
                    reason="no frames to score",
                )
            ]

        left = quality[:num_frames, 0]
        right = quality[:num_frames, 1]
        zero_either = (left == 0.0) | (right == 0.0)
        frame_scores = np.where(zero_either, 0.0, 1.0)

        results: list[CheckResult] = [
            CheckResult(
                check=self.name,
                episode_idx=clip.episode_idx,
                frame_idx=clip.frame_idx_at(offset),
                metrics={
                    "frame_score": float(frame_scores[offset]),
                    "quality_left": float(left[offset]),
                    "quality_right": float(right[offset]),
                },
                flag=None,
                reason="per-frame quality_hand acceptance score",
            )
            for offset in range(num_frames)
        ]

        total_score = float(np.sum(frame_scores))
        pass_ratio = total_score / num_frames
        passes = bool(pass_ratio >= self.pass_threshold)
        results.append(
            CheckResult(
                check=self.name,
                episode_idx=clip.episode_idx,
                frame_idx=SUMMARY_FRAME_IDX,
                metrics={
                    "total_score": total_score,
                    "num_frames": float(num_frames),
                    "pass_ratio": pass_ratio,
                    "pass_threshold": self.pass_threshold,
                },
                flag=passes,
                reason="clip-level quality_hand acceptance verdict",
            )
        )
        return results
