"""Static per-frame hand keypoint morphology checks."""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import numpy as np

from precheck.base import BaseCheck
from precheck.registry import register
from qc_common.keypoints import (
    ACCEPTANCE_FINGER_CHAINS,
    acceptance_joint_names,
    derive_angle_triples,
    derive_finger_bones,
)
from qc_common.types import CheckResult, ClipInputs

from .quality_score import SUMMARY_FRAME_IDX


VERDICT_RANK = {
    "not_applicable": 0,
    "pass": 1,
    "review": 2,
    "fail": 3,
}

SUMMARY_METRICS = (
    "bone_length_ratio_spread",
    "normalized_bone_length_max",
    "zero_length_bone_count",
    "duplicate_joint_pair_count",
    "collapsed_finger_count",
    "joint_angle_violation_fraction",
)


@register
class KeypointMorphologyCheck(BaseCheck):
    """Evaluate static 21-point hand shape without temporal or supplier signals."""

    name = "keypoint_morphology"
    granularity = "frame"

    def __init__(self, config: dict[str, Any]) -> None:
        self.sides = list(config["sides"])
        self.thresholds = {
            key: value for key, value in config.items() if key != "sides"
        }
        self.duplicate_joint_distance_m = float(
            config["duplicate_joint_distance_m"]
        )
        self.min_palm_scale_m = float(config["min_palm_scale_m"])
        self.max_bone_length_ratio_spread_review = float(
            config["max_bone_length_ratio_spread_review"]
        )
        self.max_bone_length_ratio_spread_fail = float(
            config["max_bone_length_ratio_spread_fail"]
        )
        self.max_normalized_bone_length_review = float(
            config["max_normalized_bone_length_review"]
        )
        self.max_normalized_bone_length_fail = float(
            config["max_normalized_bone_length_fail"]
        )
        self.max_zero_length_bone_count_review = int(
            config["max_zero_length_bone_count_review"]
        )
        self.max_zero_length_bone_count_fail = int(
            config["max_zero_length_bone_count_fail"]
        )
        self.max_duplicate_joint_pair_count_review = int(
            config["max_duplicate_joint_pair_count_review"]
        )
        self.max_duplicate_joint_pair_count_fail = int(
            config["max_duplicate_joint_pair_count_fail"]
        )
        self.min_joint_angle_deg_review = float(
            config["min_joint_angle_deg_review"]
        )
        self.min_joint_angle_deg_fail = float(
            config["min_joint_angle_deg_fail"]
        )
        self.max_joint_angle_violation_fraction_review = float(
            config["max_joint_angle_violation_fraction_review"]
        )
        self.max_joint_angle_violation_fraction_fail = float(
            config["max_joint_angle_violation_fraction_fail"]
        )

    def run(self, clip: ClipInputs) -> list[CheckResult]:
        if not clip.keypoints or clip.num_frames <= 0:
            return []

        frame_results: list[CheckResult] = []
        for frame_offset in range(clip.num_frames):
            metrics: dict[str, Any] = {}
            reasons: list[str] = []
            exceeded: list[str] = []
            side_verdicts: list[str] = []
            for side in self.sides:
                side_metrics, side_verdict, side_reasons = self._evaluate_hand(
                    clip,
                    frame_offset,
                    side,
                )
                metrics.update(
                    {
                        f"{side}_{name}": value
                        for name, value in side_metrics.items()
                    }
                )
                metrics[f"{side}_morphology_verdict"] = side_verdict
                side_verdicts.append(side_verdict)
                reasons.extend(f"{side}:{reason}" for reason in side_reasons)
                exceeded.extend(
                    f"{side}:{reason}"
                    for reason in side_reasons
                    if reason != "skipped_due_to_existence_invalid"
                )

            verdict = worst_verdict(side_verdicts)
            metrics["morphology_verdict"] = verdict
            metrics["which_thresholds_exceeded"] = exceeded
            reason = "; ".join(reasons) if reasons else "static hand morphology within thresholds"
            frame_results.append(
                CheckResult(
                    check=self.name,
                    episode_idx=clip.episode_idx,
                    frame_idx=clip.frame_idx_at(frame_offset),
                    metrics=metrics,
                    flag=flag_for_verdict(verdict),
                    reason=reason,
                )
            )

        return [*frame_results, self._summary_row(clip, frame_results)]

    def _evaluate_hand(
        self,
        clip: ClipInputs,
        frame_offset: int,
        side: str,
    ) -> tuple[dict[str, Any], str, list[str]]:
        expected = acceptance_joint_names([side])
        points_by_name: dict[str, np.ndarray] = {}
        for name in expected:
            values = clip.keypoints.get(name) if clip.keypoints else None
            if values is None or values.shape[0] <= frame_offset:
                continue
            point = np.asarray(values[frame_offset], dtype=np.float64)
            if point.shape[0] < 3 or not np.all(np.isfinite(point[:3])):
                continue
            points_by_name[name] = point[:3]

        metrics = empty_hand_metrics()
        metrics["valid_keypoint_count"] = len(points_by_name)
        if len(points_by_name) != len(expected):
            return metrics, "not_applicable", ["skipped_due_to_existence_invalid"]

        points = np.stack([points_by_name[name] for name in expected])
        palm_scale = self._palm_scale(points_by_name, side)
        metrics["palm_scale_m"] = palm_scale

        bones = derive_finger_bones(expected)
        bone_lengths = np.asarray(
            [
                np.linalg.norm(points_by_name[parent] - points_by_name[child])
                for parent, child in bones
            ],
            dtype=np.float64,
        )
        metrics["bone_length_m_min"] = float(np.min(bone_lengths))
        metrics["bone_length_m_max"] = float(np.max(bone_lengths))
        metrics["bone_length_m_median"] = float(np.median(bone_lengths))
        metrics["zero_length_bone_count"] = int(
            np.sum(bone_lengths <= self.duplicate_joint_distance_m)
        )

        normalized = (
            bone_lengths / palm_scale
            if palm_scale > 0
            else np.full_like(bone_lengths, math.inf)
        )
        metrics["normalized_bone_length_min"] = float(np.min(normalized))
        metrics["normalized_bone_length_max"] = float(np.max(normalized))
        metrics["bone_length_ratio_spread"] = float(
            metrics["normalized_bone_length_max"]
            - metrics["normalized_bone_length_min"]
        )
        metrics["duplicate_joint_pair_count"] = self._duplicate_pair_count(points)
        metrics["collapsed_finger_count"] = self._collapsed_finger_count(
            points_by_name,
            side,
        )

        angles = self._joint_angles(points_by_name, expected)
        metrics["joint_angle_min_deg"] = (
            float(np.min(angles)) if angles.size else None
        )
        metrics["joint_angle_max_deg"] = (
            float(np.max(angles)) if angles.size else None
        )
        metrics["joint_angle_violation_fraction"] = (
            float(np.mean(angles < self.min_joint_angle_deg_review))
            if angles.size
            else 0.0
        )

        reasons = self._threshold_reasons(metrics)
        verdict = verdict_from_reasons(reasons)
        return metrics, verdict, reasons

    def _palm_scale(
        self,
        points: dict[str, np.ndarray],
        side: str,
    ) -> float:
        root = points[f"{side}Hand"]
        distances = [
            np.linalg.norm(
                points[f"{side}{chain[0]}"] - root
            )
            for chain in ACCEPTANCE_FINGER_CHAINS.values()
        ]
        return float(np.median(distances))

    def _duplicate_pair_count(self, points: np.ndarray) -> int:
        return sum(
            np.linalg.norm(points[first] - points[second])
            <= self.duplicate_joint_distance_m
            for first, second in combinations(range(len(points)), 2)
        )

    def _collapsed_finger_count(
        self,
        points: dict[str, np.ndarray],
        side: str,
    ) -> int:
        root_name = f"{side}Hand"
        collapsed = 0
        for chain in ACCEPTANCE_FINGER_CHAINS.values():
            names = [root_name, *(f"{side}{base_name}" for base_name in chain)]
            lengths = [
                np.linalg.norm(points[parent] - points[child])
                for parent, child in zip(names[:-1], names[1:])
            ]
            if sum(
                length <= self.duplicate_joint_distance_m
                for length in lengths
            ) >= 2:
                collapsed += 1
        return collapsed

    def _joint_angles(
        self,
        points: dict[str, np.ndarray],
        joint_names: list[str],
    ) -> np.ndarray:
        angles: list[float] = []
        for first, middle, last in derive_angle_triples(joint_names):
            first_vector = points[first] - points[middle]
            last_vector = points[last] - points[middle]
            denominator = np.linalg.norm(first_vector) * np.linalg.norm(last_vector)
            if denominator <= self.duplicate_joint_distance_m**2:
                continue
            cosine = float(
                np.clip(
                    np.dot(first_vector, last_vector) / denominator,
                    -1.0,
                    1.0,
                )
            )
            angles.append(float(np.degrees(np.arccos(cosine))))
        return np.asarray(angles, dtype=np.float64)

    def _threshold_reasons(self, metrics: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        if metrics["palm_scale_m"] < self.min_palm_scale_m:
            reasons.append("palm_scale_too_small")
        add_max_reason(
            reasons,
            "bone_length_ratio_spread",
            metrics["bone_length_ratio_spread"],
            self.max_bone_length_ratio_spread_review,
            self.max_bone_length_ratio_spread_fail,
        )
        add_max_reason(
            reasons,
            "max_normalized_bone_length",
            metrics["normalized_bone_length_max"],
            self.max_normalized_bone_length_review,
            self.max_normalized_bone_length_fail,
        )
        add_max_reason(
            reasons,
            "zero_length_bone_count",
            metrics["zero_length_bone_count"],
            self.max_zero_length_bone_count_review,
            self.max_zero_length_bone_count_fail,
        )
        add_max_reason(
            reasons,
            "duplicate_joint_pair_count",
            metrics["duplicate_joint_pair_count"],
            self.max_duplicate_joint_pair_count_review,
            self.max_duplicate_joint_pair_count_fail,
        )

        angle_min = metrics["joint_angle_min_deg"]
        if angle_min is not None:
            if angle_min <= self.min_joint_angle_deg_fail:
                reasons.append("joint_angle_min_deg_fail")
            elif angle_min <= self.min_joint_angle_deg_review:
                reasons.append("joint_angle_min_deg_review")
        add_max_reason(
            reasons,
            "joint_angle_violation_fraction",
            metrics["joint_angle_violation_fraction"],
            self.max_joint_angle_violation_fraction_review,
            self.max_joint_angle_violation_fraction_fail,
        )

        collapsed = int(metrics["collapsed_finger_count"])
        if collapsed:
            severe = (
                metrics["zero_length_bone_count"]
                >= self.max_zero_length_bone_count_fail
                or metrics["duplicate_joint_pair_count"]
                >= self.max_duplicate_joint_pair_count_fail
            )
            reasons.append(
                "collapsed_finger_count_fail"
                if severe
                else "collapsed_finger_count_review"
            )
        return reasons

    def _summary_row(
        self,
        clip: ClipInputs,
        frame_results: list[CheckResult],
    ) -> CheckResult:
        metrics: dict[str, Any] = {
            "method": "static_per_frame_21_point_hand_geometry",
            "decision_basis": "fixed_config_thresholds",
            "calibration_statistics_only": True,
            "threshold_version": "keypoint_morphology_v0",
            "thresholds": {"sides": self.sides, **self.thresholds},
            "num_frames": len(frame_results),
        }
        for side in self.sides:
            for metric_name in SUMMARY_METRICS:
                values = [
                    float(result.metrics[f"{side}_{metric_name}"])
                    for result in frame_results
                    if result.metrics.get(f"{side}_morphology_verdict")
                    != "not_applicable"
                    and finite_number(result.metrics.get(f"{side}_{metric_name}"))
                ]
                metrics.update(distribution_stats(values, f"{side}_{metric_name}"))

        verdict = worst_verdict(
            [
                str(result.metrics["morphology_verdict"])
                for result in frame_results
            ]
        )
        exceeded = sorted(
            {
                reason
                for result in frame_results
                for reason in result.metrics["which_thresholds_exceeded"]
            }
        )
        metrics["morphology_verdict"] = verdict
        metrics["which_thresholds_exceeded"] = exceeded
        return CheckResult(
            check=self.name,
            episode_idx=clip.episode_idx,
            frame_idx=SUMMARY_FRAME_IDX,
            metrics=metrics,
            flag=flag_for_verdict(verdict),
            reason=(
                "; ".join(exceeded)
                if exceeded
                else "clip static hand morphology within thresholds"
            ),
        )


def empty_hand_metrics() -> dict[str, Any]:
    return {
        "valid_keypoint_count": 0,
        "palm_scale_m": None,
        "bone_length_m_min": None,
        "bone_length_m_max": None,
        "bone_length_m_median": None,
        "normalized_bone_length_min": None,
        "normalized_bone_length_max": None,
        "bone_length_ratio_spread": None,
        "zero_length_bone_count": None,
        "duplicate_joint_pair_count": None,
        "collapsed_finger_count": None,
        "joint_angle_min_deg": None,
        "joint_angle_max_deg": None,
        "joint_angle_violation_fraction": None,
    }


def add_max_reason(
    reasons: list[str],
    name: str,
    value: float | int,
    review_threshold: float | int,
    fail_threshold: float | int,
) -> None:
    if value >= fail_threshold:
        reasons.append(f"{name}_fail")
    elif value >= review_threshold:
        reasons.append(f"{name}_review")


def verdict_from_reasons(reasons: list[str]) -> str:
    if "palm_scale_too_small" in reasons or any(
        reason.endswith("_fail") for reason in reasons
    ):
        return "fail"
    if reasons:
        return "review"
    return "pass"


def worst_verdict(verdicts: list[str]) -> str:
    if not verdicts:
        return "not_applicable"
    return max(verdicts, key=lambda verdict: VERDICT_RANK[verdict])


def flag_for_verdict(verdict: str) -> bool | None:
    if verdict == "fail":
        return True
    if verdict == "pass":
        return False
    return None


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(
        float(value)
    )


def distribution_stats(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_p95": float(np.percentile(array, 95)),
        f"{prefix}_max": float(np.max(array)),
    }
