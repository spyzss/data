"""Clip-level supplier quality_hand acceptance scoring."""

from __future__ import annotations

import numpy as np

from precheck.base import BaseCheck
from precheck.registry import register
from qc_common.types import CheckResult, ClipInputs

# Sentinel frame_idx for the single clip-level verdict row emitted alongside
# the per-frame score rows. Never collides with a real frame index.
SUMMARY_FRAME_IDX = -1
_SUPPLIER_STATUSES = ("bad", "warning", "good", "unknown")


@register
class QualityScoreCheck(BaseCheck):
    """Apply the supplier quality_hand acceptance scoring rule."""

    name = "quality_score"
    granularity = "clip"

    def __init__(self, config: dict) -> None:
        self.pass_threshold = float(config.get("pass_threshold", 0.90))

    def run(self, clip: ClipInputs) -> list[CheckResult]:
        supplier_status = clip.supplier_hand_quality_status
        if supplier_status is not None:
            return self._run_supplier_evidence(clip, supplier_status)

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

    def _run_supplier_evidence(
        self,
        clip: ClipInputs,
        supplier_status: np.ndarray,
    ) -> list[CheckResult]:
        status = np.asarray(supplier_status)
        if status.ndim != 2 or status.shape[1] != 2:
            raise ValueError("supplier hand quality status must have shape [T,2]")
        if any(
            not isinstance(value, str) or value not in _SUPPLIER_STATUSES
            for value in status.reshape(-1).tolist()
        ):
            raise ValueError("supplier hand quality status must use registered enums")
        num_frames = min(clip.num_frames, status.shape[0])
        rows = [
            CheckResult(
                check=self.name,
                episode_idx=clip.episode_idx,
                frame_idx=clip.frame_idx_at(offset),
                metrics={
                    "supplier_status_left": str(status[offset, 0]),
                    "supplier_status_right": str(status[offset, 1]),
                },
                flag=None,
                reason="optional supplier enum Evidence; not a machine Gate",
            )
            for offset in range(num_frames)
        ]
        counts = {
            name: int(np.sum(status[:num_frames] == name))
            for name in _SUPPLIER_STATUSES
        }
        rows.append(
            CheckResult(
                check=self.name,
                episode_idx=clip.episode_idx,
                frame_idx=SUMMARY_FRAME_IDX,
                metrics={
                    "provided": True,
                    "observation_count": num_frames * 2,
                    "status_counts": counts,
                },
                flag=True,
                reason="supplier hand quality Evidence validated",
            )
        )
        return rows
