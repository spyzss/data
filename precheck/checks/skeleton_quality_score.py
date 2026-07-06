"""Vendor-agnostic skeleton quality score from temporal geometry metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from precheck.base import BaseCheck
from precheck.registry import register
from qc_common.keypoints import acceptance_joint_names
from qc_common.projection import (
    ProjectionConfig,
    hand_projection_metrics,
    intrinsics_from_values,
)
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
        self.projection_enabled = bool(config.get("projection_enabled", True))
        self.projection_image_width = config.get("projection_image_width")
        self.projection_image_height = config.get("projection_image_height")
        self.projection_intrinsics_override = intrinsics_from_values(
            config.get("projection_fx"),
            config.get("projection_fy"),
            config.get("projection_cx"),
            config.get("projection_cy"),
        )
        self.projection_border_margin_px = float(
            config.get("projection_border_margin_px", 20.0)
        )
        self.projection_near_border_count_threshold = int(
            config.get("projection_near_border_count_threshold", 6)
        )
        self.projection_outside_count_threshold = int(
            config.get("projection_outside_count_threshold", 1)
        )
        self.projection_center_jump_px_threshold = float(
            config.get("projection_center_jump_px_threshold", 120.0)
        )
        self.projection_bbox_area_change_ratio_threshold = float(
            config.get("projection_bbox_area_change_ratio_threshold", 3.0)
        )
        self.candidate_gap_close_frames = int(
            config.get("candidate_gap_close_frames", 2)
        )
        self.candidate_min_seed_run_frames = int(
            config.get("candidate_min_seed_run_frames", 3)
        )
        self.candidate_pre_context_frames = int(config.get("candidate_pre_context_frames", 10))
        self.candidate_post_context_frames = int(
            config.get("candidate_post_context_frames", 10)
        )
        self.candidate_merge_overlapping_only = bool(
            config.get("candidate_merge_overlapping_only", True)
        )
        self.pass_threshold = float(config.get("pass_threshold", 0.90))
        self.temporal_check = KeypointTemporalCheck({})
        self.candidate_windows: list[dict[str, Any]] = []

    def run(self, clip: ClipInputs) -> list[CheckResult]:
        temporal_results = [
            result
            for result in self.temporal_check.run(clip)
            if result.frame_idx != SUMMARY_FRAME_IDX
        ]
        if not temporal_results:
            return []

        results: list[CheckResult] = []
        projection_state = self.projection_state(clip)
        for frame_offset, temporal_result in enumerate(temporal_results):
            metric_values = self.geometry_metric_values(temporal_result.metrics)
            missing_metrics = [
                name for name, value in metric_values.items() if not math.isfinite(value)
            ]
            exceeded = self.exceeded_thresholds(metric_values)
            presence_metrics = self.presence_metrics(clip, frame_offset)
            projection_metrics = self.projection_metrics(
                clip,
                frame_offset,
                projection_state,
            )
            combined_review_metrics = self.combined_review_metrics(
                metric_values,
                exceeded,
                projection_metrics,
            )
            ratios = self.metric_ratios(metric_values)
            verdict, needs_mask_review, needs_rotation_review, source = self.classify_frame(
                metric_values,
                exceeded,
                presence_metrics,
                combined_review_metrics,
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
                        "needs_projection_review": combined_review_metrics[
                            "needs_projection_review"
                        ],
                        "needs_out_of_frame_review": combined_review_metrics[
                            "needs_out_of_frame_review"
                        ],
                        "needs_visual_review": combined_review_metrics[
                            "needs_visual_review"
                        ],
                        "skeleton_decision_source": source,
                        "sustained_review_promoted": 0.0,
                        "skeleton_decision_mode": self.decision_mode,
                        **presence_metrics,
                        **projection_metrics,
                    },
                    flag=True if verdict in {"invalid", "suspect"} else None,
                    reason=self.reason(verdict, missing_metrics),
                )
            )

        self.promote_sustained_review_runs(results)
        self.refresh_visual_review_after_promotion(results)
        self.candidate_windows = self.build_candidate_windows(
            clip,
            results,
        )
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
        review_metrics: dict[str, float],
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
        rotation_review = bool(review_metrics["needs_rotation_mask_review"])
        strong_motion = (
            ratios["joint_acceleration_m_s2_max"] >= self.strong_acceleration_ratio
            or ratios["joint_displacement_m_max"] >= self.strong_displacement_ratio
        )
        multi_signal = len(exceeded) >= self.hard_exceeded_metric_count
        if strong_motion or multi_signal:
            return "suspect", False, bool(rotation_review), "strong_temporal_geometry"
        projection_review = bool(review_metrics["needs_projection_review"])
        out_of_frame_review = bool(review_metrics["needs_out_of_frame_review"])
        return (
            "review",
            projection_review or out_of_frame_review or bool(rotation_review),
            bool(rotation_review),
            "projection_visual_review" if projection_review else "moderate_temporal_geometry",
        )

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

    def projection_state(self, clip: ClipInputs) -> dict[str, Any]:
        image_size = self.projection_image_size(clip)
        intrinsics = self.projection_intrinsics_override
        if intrinsics is None:
            intrinsics = clip.intrinsics
        enabled = (
            self.projection_enabled
            and intrinsics is not None
            and image_size is not None
        )
        return {
            "enabled": bool(enabled),
            "intrinsics": intrinsics,
            "image_size": image_size,
            "previous_center": {},
            "previous_area": {},
        }

    def projection_image_size(self, clip: ClipInputs) -> tuple[int, int] | None:
        if self.projection_image_width is not None and self.projection_image_height is not None:
            return int(self.projection_image_width), int(self.projection_image_height)
        frames = clip.frames
        if frames is None or len(frames) == 0:
            return None
        first = np.asarray(frames[0])
        if first.ndim < 2:
            return None
        height, width = first.shape[:2]
        return int(width), int(height)

    def projection_metrics(
        self,
        clip: ClipInputs,
        frame_offset: int,
        state: dict[str, Any],
    ) -> dict[str, float]:
        width_height = state.get("image_size")
        metrics: dict[str, float] = {
            "projection_enabled": float(bool(state.get("enabled"))),
        }
        if width_height is not None:
            metrics["projection_image_width"] = float(width_height[0])
            metrics["projection_image_height"] = float(width_height[1])
        if not state.get("enabled"):
            return metrics

        config = ProjectionConfig(
            image_width=int(width_height[0]),
            image_height=int(width_height[1]),
            border_margin_px=self.projection_border_margin_px,
        )
        keypoints = clip.keypoints or {}
        intrinsics = state["intrinsics"]
        for side in ("left", "right"):
            expected = acceptance_joint_names([side])
            if not all(name in keypoints and keypoints[name].shape[0] > frame_offset for name in expected):
                continue
            points = np.asarray([keypoints[name][frame_offset] for name in expected], dtype=np.float64)
            side_metrics = hand_projection_metrics(
                points,
                intrinsics,
                config,
                previous_center=state["previous_center"].get(side),
            )
            area = float(side_metrics.get("hand_bbox_area_2d", math.nan))
            previous_area = state["previous_area"].get(side)
            if previous_area is None or previous_area <= 1e-8 or not math.isfinite(area):
                area_change_ratio = math.nan
            else:
                area_change_ratio = max(area / previous_area, previous_area / max(area, 1e-8))
            center_u = side_metrics.get("hand_bbox_center_u")
            center_v = side_metrics.get("hand_bbox_center_v")
            if isinstance(center_u, float) and isinstance(center_v, float) and math.isfinite(center_u) and math.isfinite(center_v):
                state["previous_center"][side] = (center_u, center_v)
            if math.isfinite(area):
                state["previous_area"][side] = area
            side_metrics["hand_bbox_area_change_ratio"] = area_change_ratio
            for name, value in side_metrics.items():
                metrics[f"{side}_{name}"] = float(value) if isinstance(value, (int, float, np.floating)) else value
        return metrics

    def combined_review_metrics(
        self,
        metric_values: dict[str, float],
        exceeded: list[str],
        projection_metrics: dict[str, float],
    ) -> dict[str, float]:
        ratios = self.metric_ratios(metric_values)
        any_projection_review = False
        any_out_of_frame_review = False
        any_rotation_projection_review = False
        for side in ("left", "right"):
            outside = projection_metrics.get(f"{side}_num_points_outside_image", 0.0)
            invalid = projection_metrics.get(f"{side}_num_projection_invalid", 0.0)
            near_border = projection_metrics.get(f"{side}_num_points_near_border", 0.0)
            touches_border = projection_metrics.get(f"{side}_keypoint_bbox_touches_border", 0.0)
            center_jump = projection_metrics.get(f"{side}_hand_bbox_center_jump_px", math.nan)
            area_ratio = projection_metrics.get(f"{side}_hand_bbox_area_change_ratio", math.nan)
            out_of_frame = (
                outside >= self.projection_outside_count_threshold
                or near_border >= self.projection_near_border_count_threshold
                or bool(touches_border)
            )
            projection_review = (
                out_of_frame
                or invalid >= self.projection_outside_count_threshold
                or (
                    math.isfinite(center_jump)
                    and center_jump >= self.projection_center_jump_px_threshold
                )
                or (
                    math.isfinite(area_ratio)
                    and area_ratio >= self.projection_bbox_area_change_ratio_threshold
                )
            )
            rotation_projection = (
                ratios["rotation_delta_max"] >= self.rotation_mask_review_ratio
                and out_of_frame
            ) if math.isfinite(ratios["rotation_delta_max"]) else False
            projection_metrics[f"{side}_needs_out_of_frame_review"] = float(out_of_frame)
            projection_metrics[f"{side}_needs_projection_review"] = float(projection_review)
            projection_metrics[f"{side}_needs_rotation_mask_review"] = float(rotation_projection)
            any_out_of_frame_review = any_out_of_frame_review or out_of_frame
            any_projection_review = any_projection_review or projection_review
            any_rotation_projection_review = (
                any_rotation_projection_review or rotation_projection
            )

        strong_temporal = (
            ratios["joint_acceleration_m_s2_max"] >= self.strong_acceleration_ratio
            or ratios["joint_displacement_m_max"] >= self.strong_displacement_ratio
            or len(exceeded) >= self.hard_exceeded_metric_count
        )
        return {
            "needs_projection_review": float(any_projection_review),
            "needs_out_of_frame_review": float(any_out_of_frame_review),
            "needs_rotation_mask_review": float(any_rotation_projection_review),
            "needs_visual_review": float(
                any_projection_review
                or any_out_of_frame_review
                or any_rotation_projection_review
                or strong_temporal
            ),
        }

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
            result.metrics["needs_visual_review"] = 1.0
            result.flag = True
            result.reason = "sustained temporal review run needs mask containment"

    def refresh_visual_review_after_promotion(self, results: list[CheckResult]) -> None:
        for result in results:
            if result.flag is True:
                result.metrics["needs_visual_review"] = 1.0

    def count_verdicts(self, results: list[CheckResult]) -> dict[str, int]:
        counts = {"invalid": 0, "good": 0, "review": 0, "suspect": 0}
        for result in results:
            verdict = str(result.metrics.get("skeleton_verdict", "good"))
            if verdict in counts:
                counts[verdict] += 1
        return counts

    def build_candidate_windows(
        self,
        clip: ClipInputs,
        results: list[CheckResult],
    ) -> list[dict[str, Any]]:
        asset_id = getattr(clip, "asset_id", None)
        seeds: list[dict[str, Any]] = []
        for result in results:
            if result.frame_idx < 0:
                continue
            seed = self.temporal_seed_record(result, asset_id)
            if seed is not None:
                seeds.append(seed)
        seed_runs = self.build_seed_runs(seeds)
        kept_runs = [
            run
            for run in seed_runs
            if len(run["seeds"]) >= self.candidate_min_seed_run_frames
        ]
        expanded = [
            self.expand_seed_run_window(clip, run)
            for run in kept_runs
        ]
        return self.merge_expanded_seed_windows(expanded)

    def temporal_seed_record(
        self,
        result: CheckResult,
        asset_id: str | None = None,
    ) -> dict[str, Any] | None:
        metrics = result.metrics
        if bool(metrics.get("keypoint_presence_invalid", 0.0)):
            return None
        exceeded = set(metrics.get("which_thresholds_exceeded", []))
        acceleration_seed = (
            "joint_acceleration_m_s2_max" in exceeded
            and float(metrics.get("joint_acceleration_m_s2_max", 0.0))
            > self.joint_acceleration_m_s2_max_threshold
        )
        displacement_seed = (
            "joint_displacement_m_max" in exceeded
            and float(metrics.get("joint_displacement_m_max", 0.0))
            > self.joint_displacement_m_max_threshold
        )
        multi_signal_seed = len(exceeded.intersection(GEOMETRY_METRIC_NAMES)) >= 2
        if not (acceleration_seed or displacement_seed or multi_signal_seed):
            return None

        reasons: list[str] = []
        if acceleration_seed:
            reasons.append("acceleration_seed")
        if displacement_seed:
            reasons.append("displacement_seed")
        if multi_signal_seed:
            reasons.append("multi_signal_seed")
        return {
            "episode_idx": result.episode_idx,
            "asset_id": asset_id,
            "hand_side": "both",
            "frame_idx": result.frame_idx,
            "trigger_reason": reasons,
            "trigger_metrics": self.window_trigger_metrics(metrics, "both"),
            "priority_score": self.temporal_seed_priority_score(metrics, reasons),
        }

    def temporal_seed_priority_score(
        self,
        metrics: dict[str, Any],
        reasons: list[str],
    ) -> float:
        score = 0.0
        if "multi_signal_seed" in reasons:
            score += 35.0
        acceleration_ratio = float(
            metrics.get("joint_acceleration_m_s2_ratio", 0.0) or 0.0
        )
        displacement_ratio = float(
            metrics.get("joint_displacement_m_ratio", 0.0) or 0.0
        )
        if "acceleration_seed" in reasons:
            score += 25.0 + min(30.0, acceleration_ratio * 10.0)
        if "displacement_seed" in reasons:
            score += 25.0 + min(30.0, displacement_ratio * 10.0)
        return score

    def build_seed_runs(self, seeds: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not seeds:
            return []
        sorted_seeds = sorted(seeds, key=lambda item: item["frame_idx"])
        runs: list[dict[str, Any]] = []
        active: dict[str, Any] | None = None
        for seed in sorted_seeds:
            frame_idx = int(seed["frame_idx"])
            if (
                active is None
                or frame_idx > int(active["seed_run_end"]) + self.candidate_gap_close_frames + 1
            ):
                if active is not None:
                    runs.append(active)
                active = {
                    "episode_idx": seed["episode_idx"],
                    "asset_id": seed["asset_id"],
                    "hand_side": seed["hand_side"],
                    "seed_run_start": frame_idx,
                    "seed_run_end": frame_idx,
                    "seeds": [seed],
                }
            else:
                active["seed_run_end"] = frame_idx
                active["seeds"].append(seed)
        if active is not None:
            runs.append(active)
        return runs

    def expand_seed_run_window(
        self,
        clip: ClipInputs,
        run: dict[str, Any],
    ) -> dict[str, Any]:
        max_frame = clip.frame_idx_at(clip.num_frames - 1) if clip.num_frames else 0
        seed_run_start = int(run["seed_run_start"])
        seed_run_end = int(run["seed_run_end"])
        return {
            **run,
            "start_frame": max(0, seed_run_start - self.candidate_pre_context_frames),
            "end_frame": min(max_frame, seed_run_end + self.candidate_post_context_frames),
        }

    def merge_expanded_seed_windows(
        self,
        windows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not windows:
            return []
        sorted_windows = sorted(windows, key=lambda item: item["start_frame"])
        merged: list[dict[str, Any]] = []
        active: dict[str, Any] | None = None
        for window in sorted_windows:
            if active is None or int(window["start_frame"]) > int(active["end_frame"]):
                if active is not None:
                    merged.append(self.finalize_window(active))
                active = dict(window)
            else:
                active["end_frame"] = max(int(active["end_frame"]), int(window["end_frame"]))
                active["seed_run_end"] = max(
                    int(active["seed_run_end"]),
                    int(window["seed_run_end"]),
                )
                active["seeds"].extend(window["seeds"])
        if active is not None:
            merged.append(self.finalize_window(active))
        return merged

    def trigger_record(self, result: CheckResult, side: str) -> dict[str, Any]:
        metrics = result.metrics
        reasons: list[str] = []
        review_types: list[str] = []
        side_prefix = "" if side == "both" else f"{side}_"
        if result.flag is True:
            reasons.append("temporal_jump")
            review_types.append("temporal_skeleton_review")
        if side != "both":
            if metrics.get(f"{side_prefix}num_points_outside_image", 0.0) >= self.projection_outside_count_threshold:
                reasons.append("projection_outside")
                review_types.append("out_of_frame_review")
            if metrics.get(f"{side_prefix}num_projection_invalid", 0.0) >= self.projection_outside_count_threshold:
                reasons.append("projection_invalid")
                review_types.append("projection_review")
            if metrics.get(f"{side_prefix}num_points_near_border", 0.0) >= self.projection_near_border_count_threshold:
                reasons.append("projection_near_border")
                review_types.append("out_of_frame_review")
            if metrics.get(f"{side_prefix}keypoint_bbox_touches_border", 0.0):
                reasons.append("bbox_touches_border")
                review_types.append("out_of_frame_review")
            center_jump = metrics.get(f"{side_prefix}hand_bbox_center_jump_px", math.nan)
            if math.isfinite(center_jump) and center_jump >= self.projection_center_jump_px_threshold:
                reasons.append("bbox_center_jump")
                review_types.append("projection_review")
            if metrics.get(f"{side_prefix}needs_rotation_mask_review", 0.0):
                reasons.append("rotation_edge_risk")
                review_types.append("rotation_visual_review")
        if not reasons and metrics.get("needs_visual_review", 0.0):
            reasons.append("visual_review")
            review_types.append("projection_review")
        score = self.trigger_priority_score(metrics, side)
        return {
            "episode_idx": result.episode_idx,
            "asset_id": None,
            "hand_side": side,
            "frame_idx": result.frame_idx,
            "priority_score": score,
            "priority": "high" if score >= 80.0 else "medium" if score >= 40.0 else "low",
            "trigger_reason": sorted(set(reasons)),
            "review_type": sorted(set(review_types)),
            "trigger_metrics": self.window_trigger_metrics(metrics, side),
        }

    def trigger_priority_score(self, metrics: dict[str, Any], side: str) -> float:
        score = 0.0
        if metrics.get("skeleton_verdict") == "invalid":
            score += 120.0
        if metrics.get("skeleton_verdict") == "suspect":
            score += 60.0
        if side != "both":
            prefix = f"{side}_"
            score += 20.0 * float(metrics.get(f"{prefix}num_points_outside_image", 0.0))
            score += 15.0 * float(metrics.get(f"{prefix}num_projection_invalid", 0.0))
            score += 4.0 * float(metrics.get(f"{prefix}num_points_near_border", 0.0))
            if metrics.get(f"{prefix}keypoint_bbox_touches_border", 0.0):
                score += 35.0
            if metrics.get(f"{prefix}needs_rotation_mask_review", 0.0):
                score += 45.0
            center_jump = metrics.get(f"{prefix}hand_bbox_center_jump_px", math.nan)
            if math.isfinite(center_jump):
                score += min(40.0, center_jump / max(1.0, self.projection_center_jump_px_threshold) * 30.0)
        return score

    def window_trigger_metrics(self, metrics: dict[str, Any], side: str) -> dict[str, Any]:
        names = [
            "rotation_delta_max",
            "joint_displacement_m_max",
            "joint_acceleration_m_s2_max",
            "joint_angle_change_deg_max",
        ]
        if side != "both":
            prefix = f"{side}_"
            names.extend(
                [
                    f"{prefix}num_points_outside_image",
                    f"{prefix}num_points_near_border",
                    f"{prefix}num_projection_invalid",
                    f"{prefix}hand_bbox_center_jump_px",
                    f"{prefix}hand_bbox_area_2d",
                    f"{prefix}hand_bbox_area_change_ratio",
                ]
            )
        return {name: metrics.get(name) for name in names if name in metrics}

    def merge_trigger_windows(
        self,
        clip: ClipInputs,
        hand_side: str,
        triggers: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not triggers:
            return []
        max_frame = clip.frame_idx_at(clip.num_frames - 1) if clip.num_frames else 0
        sorted_triggers = sorted(triggers, key=lambda item: item["frame_idx"])
        windows: list[dict[str, Any]] = []
        active: dict[str, Any] | None = None
        for trigger in sorted_triggers:
            start = max(0, int(trigger["frame_idx"]) - self.candidate_pre_context_frames)
            end = min(max_frame, int(trigger["frame_idx"]) + self.candidate_post_context_frames)
            if active is None or start > active["end_frame"] + self.candidate_merge_gap_frames:
                if active is not None:
                    windows.append(self.finalize_window(active))
                active = {
                    "episode_idx": trigger["episode_idx"],
                    "asset_id": trigger["asset_id"],
                    "hand_side": hand_side,
                    "start_frame": start,
                    "end_frame": end,
                    "triggers": [trigger],
                }
            else:
                active["end_frame"] = max(active["end_frame"], end)
                active["triggers"].append(trigger)
        if active is not None:
            windows.append(self.finalize_window(active))
        return windows

    def finalize_window(self, window: dict[str, Any]) -> dict[str, Any]:
        seeds = window.get("seeds", window.get("triggers", []))
        peak = max(seeds, key=lambda item: item["priority_score"])
        reasons = sorted({reason for seed in seeds for reason in seed["trigger_reason"]})
        review_types = sorted(
            {kind for seed in seeds for kind in seed.get("review_type", ["temporal_geometry_review"])}
        )
        priority_score = float(peak["priority_score"])
        seed_run_frames = len({int(seed["frame_idx"]) for seed in seeds})
        seed_run_length = int(window.get("seed_run_end", peak["frame_idx"])) - int(
            window.get("seed_run_start", peak["frame_idx"])
        ) + 1
        priority = "high" if (
            priority_score >= 80.0
            or seed_run_frames >= max(5, self.candidate_min_seed_run_frames * 2)
            or seed_run_length >= max(8, self.candidate_min_seed_run_frames * 3)
            or "multi_signal_seed" in reasons
        ) else "medium"
        return {
            "episode_idx": window["episode_idx"],
            "asset_id": window["asset_id"],
            "hand_side": window["hand_side"],
            "start_frame": int(window["start_frame"]),
            "end_frame": int(window["end_frame"]),
            "peak_frame": int(peak["frame_idx"]),
            "seed_run_start": int(window.get("seed_run_start", peak["frame_idx"])),
            "seed_run_end": int(window.get("seed_run_end", peak["frame_idx"])),
            "seed_run_frames": float(seed_run_frames),
            "trigger_reason": reasons,
            "review_type": review_types or ["temporal_geometry_review"],
            "priority": priority,
            "priority_score": priority_score,
            "window_source": "skeleton_quality_temporal_run",
            "trigger_metrics": peak["trigger_metrics"],
        }

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
