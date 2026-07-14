"""Clip-level composite supplier-label and geometry verdict."""

from __future__ import annotations

import numpy as np

from precheck.base import BaseCheck
from precheck.registry import register
from qc_common.types import CheckResult, ClipInputs

from .quality_score import SUMMARY_FRAME_IDX
from .skeleton_quality_score import SkeletonQualityScoreCheck


SUPPLIER_AUDIT_CODES = {
    "supplier_label_absent": 0.0,
    "supplier_zero_weight": 1.0,
    "supplier_partial_weight": 2.0,
    "no_downweight_skeleton_good": 3.0,
    "no_downweight_skeleton_suspect": 4.0,
}


@register
class CompositeFrameVerdictCheck(BaseCheck):
    """Audit optional supplier labels against the common skeleton score."""

    name = "composite_frame_verdict"
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
        self.skeleton_check = SkeletonQualityScoreCheck(config)

    def run(self, clip: ClipInputs) -> list[CheckResult]:
        skeleton_results = [
            result
            for result in self.skeleton_check.run(clip)
            if result.frame_idx != SUMMARY_FRAME_IDX
        ]
        if not skeleton_results:
            return []

        quality = self._quality_or_none(clip.quality_hand)
        num_frames = len(skeleton_results)

        counts = {
            "skeleton_good": 0,
            "skeleton_suspect": 0,
            "supplier_label_absent": 0,
            "supplier_no_downweight": 0,
            "supplier_downweighted": 0,
            "audit_suspect": 0,
        }
        results: list[CheckResult] = []
        for frame_offset, skeleton_result in enumerate(skeleton_results):
            skeleton_metrics = skeleton_result.metrics
            skeleton_suspect = skeleton_result.flag is True
            counts["skeleton_suspect" if skeleton_suspect else "skeleton_good"] += 1
            supplier = self._supplier_metrics(quality, frame_offset)
            counts[supplier["count_key"]] += 1
            audit_suspect = supplier["no_downweight"] and skeleton_suspect
            if audit_suspect:
                counts["audit_suspect"] += 1

            results.append(
                CheckResult(
                    check=self.name,
                    episode_idx=clip.episode_idx,
                    frame_idx=skeleton_result.frame_idx,
                    metrics={
                        "frame_score": float(skeleton_metrics["skeleton_score"]),
                        "weighted_frame_score": float(
                            skeleton_metrics["skeleton_score"]
                        )
                        * supplier["vendor_quality_weight"],
                        "skeleton_score": float(skeleton_metrics["skeleton_score"]),
                        "skeleton_verdict_code": float(
                            skeleton_metrics["skeleton_verdict_code"]
                        ),
                        "joint_angle_change_deg_max": skeleton_metrics[
                            "joint_angle_change_deg_max"
                        ],
                        "rotation_delta_max": skeleton_metrics["rotation_delta_max"],
                        "joint_acceleration_m_s2_max": skeleton_metrics[
                            "joint_acceleration_m_s2_max"
                        ],
                        "joint_displacement_m_max": skeleton_metrics[
                            "joint_displacement_m_max"
                        ],
                        "which_thresholds_exceeded": skeleton_metrics[
                            "which_thresholds_exceeded"
                        ],
                        "vendor_quality_weight": supplier["vendor_quality_weight"],
                        "supplier_quality_left": supplier["left"],
                        "supplier_quality_right": supplier["right"],
                        "supplier_label_available": supplier["label_available"],
                        "supplier_audit_code": (
                            SUPPLIER_AUDIT_CODES[
                                "no_downweight_skeleton_suspect"
                            ]
                            if audit_suspect
                            else supplier["audit_code"]
                        ),
                        "supplier_no_downweight": float(supplier["no_downweight"]),
                        "audit_suspect": float(audit_suspect),
                    },
                    flag=True if audit_suspect else None,
                    reason=self._reason(supplier, skeleton_suspect),
                )
            )

        results.append(self._summary_row(clip, num_frames, counts))
        return results

    def _quality_or_none(self, quality_hand: np.ndarray | None) -> np.ndarray | None:
        if quality_hand is None:
            return None
        quality = np.asarray(quality_hand, dtype=np.float64)
        if quality.ndim != 2 or quality.shape[1] < 2:
            return None
        return quality

    def _supplier_metrics(
        self,
        quality: np.ndarray | None,
        frame_offset: int,
    ) -> dict:
        if quality is None or quality.shape[0] <= frame_offset:
            return {
                "left": float("nan"),
                "right": float("nan"),
                "vendor_quality_weight": 1.0,
                "label_available": 0.0,
                "no_downweight": False,
                "count_key": "supplier_label_absent",
                "audit_code": SUPPLIER_AUDIT_CODES["supplier_label_absent"],
            }
        left = float(quality[frame_offset, 0])
        right = float(quality[frame_offset, 1])
        vendor_quality_weight = min(left, right)
        if vendor_quality_weight == 1.0:
            count_key = "supplier_no_downweight"
            audit_code = SUPPLIER_AUDIT_CODES["no_downweight_skeleton_good"]
            no_downweight = True
        elif vendor_quality_weight == 0.0:
            count_key = "supplier_downweighted"
            audit_code = SUPPLIER_AUDIT_CODES["supplier_zero_weight"]
            no_downweight = False
        else:
            count_key = "supplier_downweighted"
            audit_code = SUPPLIER_AUDIT_CODES["supplier_partial_weight"]
            no_downweight = False
        return {
            "left": left,
            "right": right,
            "vendor_quality_weight": vendor_quality_weight,
            "label_available": 1.0,
            "no_downweight": no_downweight,
            "count_key": count_key,
            "audit_code": audit_code,
        }

    def _reason(self, supplier: dict, skeleton_suspect: bool) -> str:
        if not supplier["label_available"]:
            return (
                "supplier label absent; vendor weight defaults to 1.0 but "
                "skeleton score comes from geometry"
            )
        if supplier["no_downweight"] and skeleton_suspect:
            return "supplier did not downweight; skeleton geometry is suspect"
        if supplier["no_downweight"]:
            return "supplier did not downweight; skeleton geometry is within thresholds"
        return "supplier downweighted frame; skeleton geometry kept as independent score"

    def _summary_row(
        self,
        clip: ClipInputs,
        num_frames: int,
        counts: dict[str, int],
    ) -> CheckResult:
        audit_suspect_count = float(counts.get("audit_suspect", 0))
        pass_ratio = 1.0 - audit_suspect_count / num_frames if num_frames > 0 else 0.0
        return CheckResult(
            check=self.name,
            episode_idx=clip.episode_idx,
            frame_idx=SUMMARY_FRAME_IDX,
            metrics={
                "skeleton_good_count": float(counts.get("skeleton_good", 0)),
                "skeleton_suspect_count": float(counts.get("skeleton_suspect", 0)),
                "supplier_label_absent_count": float(
                    counts.get("supplier_label_absent", 0)
                ),
                "supplier_no_downweight_count": float(
                    counts.get("supplier_no_downweight", 0)
                ),
                "supplier_downweighted_count": float(
                    counts.get("supplier_downweighted", 0)
                ),
                "audit_suspect_count": audit_suspect_count,
                "num_frames": float(num_frames),
                "pass_ratio": pass_ratio,
                "pass_threshold": self.pass_threshold,
            },
            flag=bool(pass_ratio >= self.pass_threshold),
            reason="clip-level supplier-label audit summary",
        )
