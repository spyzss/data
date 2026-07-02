"""Vendor-agnostic skeleton quality score from temporal geometry metrics."""

from __future__ import annotations

import math

import numpy as np

from precheck.base import BaseCheck
from precheck.registry import register
from qc_common.types import CheckResult, ClipInputs

from .keypoint_temporal import KeypointTemporalCheck
from .quality_score import SUMMARY_FRAME_IDX


GEOMETRY_METRIC_NAMES = (
    "joint_angle_change_deg_max",
    "rotation_delta_max",
    "joint_acceleration_m_s2_max",
    "joint_displacement_m_max",
)

SKELETON_VERDICT_CODES = {
    "good": 2.0,
    "suspect": 3.0,
}


@register
class SkeletonQualityScoreCheck(BaseCheck):
    """Score skeleton quality without relying on supplier self-labels."""

    name = "skeleton_quality_score"
    granularity = "clip"

    def __init__(self, config: dict) -> None:
        self.joint_angle_change_deg_max_threshold = float(
            config.get("joint_angle_change_deg_max_threshold", 10.0)
        )
        self.rotation_delta_max_threshold = float(
            config.get("rotation_delta_max_threshold", 0.45)
        )
        self.joint_acceleration_m_s2_max_threshold = float(
            config.get("joint_acceleration_m_s2_max_threshold", 15.0)
        )
        self.joint_displacement_m_max_threshold = float(
            config.get("joint_displacement_m_max_threshold", 0.05)
        )
        self.pass_threshold = float(config.get("pass_threshold", 0.90))
        self.temporal_check = KeypointTemporalCheck({})

    def run(self, clip: ClipInputs) -> list[CheckResult]:
        temporal_results = [
            result
            for result in self.temporal_check.run(clip)
            if result.frame_idx != SUMMARY_FRAME_IDX
        ]
        if not temporal_results:
            return []

        results: list[CheckResult] = []
        scores: list[float] = []
        counts = {"good": 0, "suspect": 0}
        for temporal_result in temporal_results:
            metric_values = self.geometry_metric_values(temporal_result.metrics)
            missing_metrics = [
                name for name, value in metric_values.items() if not math.isfinite(value)
            ]
            exceeded = self.exceeded_thresholds(metric_values)
            penalties = self.penalties(exceeded)
            skeleton_score = max(0.0, 1.0 - float(sum(penalties.values())))
            skeleton_verdict = "suspect" if exceeded else "good"
            counts[skeleton_verdict] += 1
            scores.append(skeleton_score)
            results.append(
                CheckResult(
                    check=self.name,
                    episode_idx=clip.episode_idx,
                    frame_idx=temporal_result.frame_idx,
                    metrics={
                        "joint_angle_change_deg_max": metric_values[
                            "joint_angle_change_deg_max"
                        ],
                        "rotation_delta_max": metric_values["rotation_delta_max"],
                        "joint_acceleration_m_s2_max": metric_values[
                            "joint_acceleration_m_s2_max"
                        ],
                        "joint_displacement_m_max": metric_values[
                            "joint_displacement_m_max"
                        ],
                        "joint_angle_change_deg_penalty": penalties[
                            "joint_angle_change_deg_max"
                        ],
                        "rotation_delta_penalty": penalties["rotation_delta_max"],
                        "joint_acceleration_m_s2_penalty": penalties[
                            "joint_acceleration_m_s2_max"
                        ],
                        "joint_displacement_m_penalty": penalties[
                            "joint_displacement_m_max"
                        ],
                        "which_thresholds_exceeded": exceeded,
                        "exceeded_threshold_count": float(len(exceeded)),
                        "missing_geometry_metric_count": float(len(missing_metrics)),
                        "skeleton_score": skeleton_score,
                        "skeleton_verdict_code": SKELETON_VERDICT_CODES[
                            skeleton_verdict
                        ],
                    },
                    flag=True if skeleton_verdict == "suspect" else None,
                    reason=self.reason(skeleton_verdict, missing_metrics),
                )
            )

        results.append(self.summary_row(clip, len(temporal_results), counts, scores))
        return results

    def geometry_metric_values(self, metrics: dict[str, float]) -> dict[str, float]:
        return {
            name: float(metrics.get(name, math.nan))
            for name in GEOMETRY_METRIC_NAMES
        }

    def exceeded_thresholds(self, metric_values: dict[str, float]) -> list[str]:
        thresholds = {
            "joint_angle_change_deg_max": self.joint_angle_change_deg_max_threshold,
            "rotation_delta_max": self.rotation_delta_max_threshold,
            "joint_acceleration_m_s2_max": self.joint_acceleration_m_s2_max_threshold,
            "joint_displacement_m_max": self.joint_displacement_m_max_threshold,
        }
        return [
            name
            for name, threshold in thresholds.items()
            if math.isfinite(metric_values[name]) and metric_values[name] > threshold
        ]

    def penalties(self, exceeded: list[str]) -> dict[str, float]:
        penalty = 1.0 / len(GEOMETRY_METRIC_NAMES)
        return {
            name: penalty if name in exceeded else 0.0
            for name in GEOMETRY_METRIC_NAMES
        }

    def reason(self, skeleton_verdict: str, missing_metrics: list[str]) -> str:
        if skeleton_verdict == "suspect":
            return "temporal skeleton geometry threshold exceeded"
        if len(missing_metrics) == len(GEOMETRY_METRIC_NAMES):
            return "temporal skeleton geometry metrics absent and treated as passed"
        if missing_metrics:
            return "temporal skeleton geometry reviewed with partial metrics"
        return "temporal skeleton geometry within thresholds"

    def summary_row(
        self,
        clip: ClipInputs,
        num_frames: int,
        counts: dict[str, int],
        scores: list[float],
    ) -> CheckResult:
        count_good = float(counts.get("good", 0))
        count_suspect = float(counts.get("suspect", 0))
        score_array = np.asarray(scores, dtype=np.float64)
        mean_score = float(np.mean(score_array)) if score_array.size else 0.0
        min_score = float(np.min(score_array)) if score_array.size else 0.0
        good_ratio = count_good / num_frames if num_frames > 0 else 0.0
        return CheckResult(
            check=self.name,
            episode_idx=clip.episode_idx,
            frame_idx=SUMMARY_FRAME_IDX,
            metrics={
                "count_good": count_good,
                "count_suspect": count_suspect,
                "num_frames": float(num_frames),
                "mean_skeleton_score": mean_score,
                "min_skeleton_score": min_score,
                "good_ratio": good_ratio,
                "pass_ratio": good_ratio,
                "pass_threshold": self.pass_threshold,
            },
            flag=bool(good_ratio >= self.pass_threshold),
            reason="clip-level skeleton quality score summary",
        )
