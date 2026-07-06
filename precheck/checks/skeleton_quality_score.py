"""Vendor-agnostic skeleton quality score from temporal geometry metrics."""

from __future__ import annotations

import math

import numpy as np

from precheck.base import BaseCheck
from precheck.registry import register
from qc_common.keypoints import acceptance_joint_names
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
    "invalid": 0.0,
    "review": 1.0,
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
        self.decision_mode = str(config.get("decision_mode", "any_threshold"))
        self.hard_exceeded_metric_count = int(
            config.get("hard_exceeded_metric_count", 3)
        )
        self.strong_acceleration_ratio = float(
            config.get("strong_acceleration_ratio", 2.5)
        )
        self.strong_displacement_ratio = float(
            config.get("strong_displacement_ratio", 1.8)
        )
        self.rotation_mask_review_ratio = float(
            config.get("rotation_mask_review_ratio", 1.0)
        )
        self.promote_sustained_review = bool(
            config.get("promote_sustained_review", False)
        )
        self.sustained_review_min_frames = int(
            config.get("sustained_review_min_frames", 6)
        )
        self.reject_missing_keypoints = bool(
            config.get("reject_missing_keypoints", True)
        )
        self.reject_low_quality_hand = bool(
            config.get("reject_low_quality_hand", False)
        )
        self.allowed_missing_keypoints_per_hand = int(
            config.get("allowed_missing_keypoints_per_hand", 0)
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
        for frame_offset, temporal_result in enumerate(temporal_results):
            metric_values = self.geometry_metric_values(temporal_result.metrics)
            missing_metrics = [
                name for name, value in metric_values.items() if not math.isfinite(value)
            ]
            exceeded = self.exceeded_thresholds(metric_values)
            presence_metrics = self.presence_metrics(clip, frame_offset)
            ratios = self.metric_ratios(metric_values)
            verdict, needs_mask_review, needs_rotation_review, source = self.classify_frame(
                metric_values,
                exceeded,
                presence_metrics,
            )
            penalties = self.penalties(exceeded)
            skeleton_score = (
                0.0
                if verdict == "invalid"
                else max(0.0, 1.0 - float(sum(penalties.values())))
            )
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
                        "joint_angle_change_deg_ratio": ratios[
                            "joint_angle_change_deg_max"
                        ],
                        "rotation_delta_ratio": ratios["rotation_delta_max"],
                        "joint_acceleration_m_s2_ratio": ratios[
                            "joint_acceleration_m_s2_max"
                        ],
                        "joint_displacement_m_ratio": ratios[
                            "joint_displacement_m_max"
                        ],
                        "which_thresholds_exceeded": exceeded,
                        "exceeded_threshold_count": float(len(exceeded)),
                        "missing_geometry_metric_count": float(len(missing_metrics)),
                        "skeleton_score": skeleton_score,
                        "skeleton_verdict": verdict,
                        "skeleton_verdict_code": SKELETON_VERDICT_CODES[verdict],
                        "needs_mask_containment_review": float(needs_mask_review),
                        "needs_rotation_mask_review": float(needs_rotation_review),
                        "skeleton_decision_source": source,
                        "sustained_review_promoted": 0.0,
                        "skeleton_decision_mode": self.decision_mode,
                        **presence_metrics,
                    },
                    flag=True if verdict == "suspect" else None,
                    reason=self.reason(verdict, missing_metrics),
                )
            )

        self.promote_sustained_review_runs(results)
        counts = self.count_verdicts(results)
        scores = [
            float(result.metrics.get("skeleton_score", 0.0))
            for result in results
            if result.frame_idx != SUMMARY_FRAME_IDX
        ]
        results.append(self.summary_row(clip, len(temporal_results), counts, scores))
        return results

    def geometry_metric_values(self, metrics: dict[str, float]) -> dict[str, float]:
        return {
            name: float(metrics.get(name, math.nan))
            for name in GEOMETRY_METRIC_NAMES
        }

    def exceeded_thresholds(self, metric_values: dict[str, float]) -> list[str]:
        thresholds = self.thresholds()
        return [
            name
            for name, threshold in thresholds.items()
            if math.isfinite(metric_values[name]) and metric_values[name] > threshold
        ]

    def thresholds(self) -> dict[str, float]:
        return {
            "joint_angle_change_deg_max": self.joint_angle_change_deg_max_threshold,
            "rotation_delta_max": self.rotation_delta_max_threshold,
            "joint_acceleration_m_s2_max": self.joint_acceleration_m_s2_max_threshold,
            "joint_displacement_m_max": self.joint_displacement_m_max_threshold,
        }

    def metric_ratios(self, metric_values: dict[str, float]) -> dict[str, float]:
        return {
            name: metric_values[name] / threshold
            if threshold > 0.0 and math.isfinite(metric_values[name])
            else math.nan
            for name, threshold in self.thresholds().items()
        }

    def classify_frame(
        self,
        metric_values: dict[str, float],
        exceeded: list[str],
        presence_metrics: dict[str, float],
    ) -> tuple[str, bool, bool, str]:
        if bool(presence_metrics.get("keypoint_presence_invalid", 0.0)):
            return "invalid", False, False, "keypoint_presence"
        if self.decision_mode == "any_threshold":
            return (
                "suspect",
                False,
                False,
                "any_threshold",
            ) if exceeded else ("good", False, False, "within_thresholds")
        if self.decision_mode != "temporal_triage":
            raise ValueError(f"unknown skeleton_quality_score decision_mode: {self.decision_mode}")
        if not exceeded:
            return "good", False, False, "within_thresholds"

        ratios = self.metric_ratios(metric_values)
        rotation_review = (
            ratios["rotation_delta_max"] >= self.rotation_mask_review_ratio
            if math.isfinite(ratios["rotation_delta_max"])
            else False
        )
        strong_motion = (
            ratios["joint_acceleration_m_s2_max"] >= self.strong_acceleration_ratio
            or ratios["joint_displacement_m_max"] >= self.strong_displacement_ratio
        )
        multi_signal = len(exceeded) >= self.hard_exceeded_metric_count
        if strong_motion or multi_signal:
            return "suspect", False, bool(rotation_review), "strong_temporal_geometry"
        return "review", True, bool(rotation_review), "moderate_temporal_geometry"

    def presence_metrics(
        self,
        clip: ClipInputs,
        frame_offset: int,
    ) -> dict[str, float]:
        keypoints = clip.keypoints or {}
        quality_hand = clip.quality_hand
        metrics: dict[str, float] = {
            "keypoint_presence_invalid": 0.0,
            "low_quality_hand_invalid": 0.0,
        }
        invalid = False
        for side_index, side in enumerate(("left", "right")):
            expected = acceptance_joint_names([side])
            valid_count = 0
            missing_count = 0
            for joint in expected:
                values = keypoints.get(joint)
                if values is None or values.shape[0] <= frame_offset:
                    missing_count += 1
                    continue
                point = np.asarray(values[frame_offset], dtype=np.float64)
                if point.shape[0] < 3 or not np.all(np.isfinite(point[:3])):
                    missing_count += 1
                    continue
                valid_count += 1
            if missing_count > self.allowed_missing_keypoints_per_hand:
                invalid = invalid or self.reject_missing_keypoints
            metrics[f"valid_keypoint_count_{side}"] = float(valid_count)
            metrics[f"missing_keypoint_count_{side}"] = float(missing_count)

            low_quality = False
            if (
                quality_hand is not None
                and quality_hand.shape[0] > frame_offset
                and quality_hand.shape[1] > side_index
            ):
                low_quality = float(quality_hand[frame_offset, side_index]) < 0.5
            metrics[f"low_quality_hand_{side}"] = float(low_quality)
            if low_quality:
                metrics["low_quality_hand_invalid"] = 1.0
                invalid = invalid or self.reject_low_quality_hand

        metrics["keypoint_presence_invalid"] = float(invalid)
        return metrics

    def promote_sustained_review_runs(self, results: list[CheckResult]) -> None:
        if (
            self.decision_mode != "temporal_triage"
            or not self.promote_sustained_review
            or self.sustained_review_min_frames <= 1
        ):
            return
        run: list[CheckResult] = []
        for result in results:
            if result.metrics.get("skeleton_verdict") == "review":
                run.append(result)
                continue
            self._promote_review_run(run)
            run = []
        self._promote_review_run(run)

    def _promote_review_run(self, run: list[CheckResult]) -> None:
        if len(run) < self.sustained_review_min_frames:
            return
        for result in run:
            result.metrics["skeleton_verdict"] = "suspect"
            result.metrics["skeleton_verdict_code"] = SKELETON_VERDICT_CODES["suspect"]
            result.metrics["sustained_review_promoted"] = 1.0
            result.metrics["needs_mask_containment_review"] = 1.0
            result.metrics["skeleton_decision_source"] = "sustained_review_run"
            result.flag = True
            result.reason = "sustained temporal review run needs mask containment"

    def count_verdicts(self, results: list[CheckResult]) -> dict[str, int]:
        counts = {"invalid": 0, "good": 0, "review": 0, "suspect": 0}
        for result in results:
            verdict = str(result.metrics.get("skeleton_verdict", "good"))
            if verdict in counts:
                counts[verdict] += 1
        return counts

    def penalties(self, exceeded: list[str]) -> dict[str, float]:
        penalty = 1.0 / len(GEOMETRY_METRIC_NAMES)
        return {
            name: penalty if name in exceeded else 0.0
            for name in GEOMETRY_METRIC_NAMES
        }

    def reason(self, skeleton_verdict: str, missing_metrics: list[str]) -> str:
        if skeleton_verdict == "invalid":
            return "keypoint presence or hand quality invalid"
        if skeleton_verdict == "suspect":
            return "temporal skeleton geometry threshold exceeded"
        if skeleton_verdict == "review":
            return "temporal skeleton geometry needs mask containment review"
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
        count_invalid = float(counts.get("invalid", 0))
        count_good = float(counts.get("good", 0))
        count_review = float(counts.get("review", 0))
        count_suspect = float(counts.get("suspect", 0))
        score_array = np.asarray(scores, dtype=np.float64)
        mean_score = float(np.mean(score_array)) if score_array.size else 0.0
        min_score = float(np.min(score_array)) if score_array.size else 0.0
        good_ratio = count_good / num_frames if num_frames > 0 else 0.0
        review_ratio = count_review / num_frames if num_frames > 0 else 0.0
        suspect_ratio = count_suspect / num_frames if num_frames > 0 else 0.0
        invalid_ratio = count_invalid / num_frames if num_frames > 0 else 0.0
        if self.decision_mode == "temporal_triage":
            pass_ratio = 1.0 - suspect_ratio - invalid_ratio
        else:
            pass_ratio = good_ratio
        return CheckResult(
            check=self.name,
            episode_idx=clip.episode_idx,
            frame_idx=SUMMARY_FRAME_IDX,
            metrics={
                "count_invalid": count_invalid,
                "count_good": count_good,
                "count_review": count_review,
                "count_suspect": count_suspect,
                "num_frames": float(num_frames),
                "mean_skeleton_score": mean_score,
                "min_skeleton_score": min_score,
                "good_ratio": good_ratio,
                "review_ratio": review_ratio,
                "suspect_ratio": suspect_ratio,
                "invalid_ratio": invalid_ratio,
                "pass_ratio": pass_ratio,
                "pass_threshold": self.pass_threshold,
                "decision_mode": self.decision_mode,
            },
            flag=bool(pass_ratio >= self.pass_threshold),
            reason="clip-level skeleton quality score summary",
        )
