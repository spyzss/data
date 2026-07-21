"""Configuration schema for the precheck package."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OverexposureConfig:
    near_saturation: int = 250
    fraction_threshold: float | None = None


@dataclass
class KeypointTemporalConfig:
    sides: list[str] = field(default_factory=lambda: ["left", "right"])
    joint_names: list[str] | None = None
    fps: float | None = None
    project_2d: bool = True
    min_angle_degrees: float = 5.0
    max_angle_degrees: float = 175.0
    temporal_decision_timebase: str = "standardized"
    temporal_target_hz: float = 30.0
    temporal_timestamp_source: str = "auto"
    temporal_max_gap_factor: float = 3.0


@dataclass
class KeypointMissingConfig:
    sides: list[str] = field(default_factory=lambda: ["left", "right"])
    joint_names: list[str] | None = None
    fps: float | None = None
    window_seconds: float = 10.0
    allowed_missing_seconds: float = 1.0


@dataclass
class KeypointMorphologyConfig:
    sides: list[str] = field(default_factory=lambda: ["left", "right"])
    duplicate_joint_distance_m: float = 0.00001
    min_palm_scale_m: float = 0.0001
    max_bone_length_ratio_spread_review: float = 3.0
    max_bone_length_ratio_spread_fail: float = 8.0
    max_normalized_bone_length_review: float = 3.0
    max_normalized_bone_length_fail: float = 6.0
    max_zero_length_bone_count_review: int = 1
    max_zero_length_bone_count_fail: int = 2
    max_duplicate_joint_pair_count_review: int = 1
    max_duplicate_joint_pair_count_fail: int = 3
    min_joint_angle_deg_review: float = 5.0
    min_joint_angle_deg_fail: float = 1.0
    max_joint_angle_violation_fraction_review: float = 0.15
    max_joint_angle_violation_fraction_fail: float = 0.40


@dataclass
class MaskContainmentConfig:
    sides: list[str] = field(default_factory=lambda: ["left", "right"])
    joint_names: list[str] | None = None


@dataclass
class QualityScoreConfig:
    pass_threshold: float = 0.90


@dataclass
class CompositeFrameVerdictConfig:
    joint_angle_change_deg_max_threshold: float = 10.0
    rotation_delta_max_threshold: float = 0.45
    joint_acceleration_m_s2_max_threshold: float = 15.0
    joint_displacement_m_max_threshold: float = 0.05
    pass_threshold: float = 0.90


@dataclass
class SkeletonQualityScoreConfig:
    joint_angle_change_deg_max_threshold: float = 10.0
    rotation_delta_max_threshold: float = 0.45
    joint_acceleration_m_s2_max_threshold: float = 15.0
    joint_displacement_m_max_threshold: float = 0.05
    temporal_decision_timebase: str = "standardized"
    temporal_target_hz: float = 30.0
    temporal_timestamp_source: str = "auto"
    temporal_max_gap_factor: float = 3.0
    decision_mode: str = "any_threshold"
    hard_exceeded_metric_count: int = 3
    strong_acceleration_ratio: float = 2.5
    strong_displacement_ratio: float = 1.8
    rotation_mask_review_ratio: float = 1.0
    rotation_delta_extreme_review_threshold: float | None = None
    palm_camera_angle_review_threshold_deg: float | None = None
    palm_camera_axis: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    palm_camera_angle_min_valid_hands: int = 1
    promote_sustained_review: bool = False
    sustained_review_min_frames: int = 6
    reject_missing_keypoints: bool = True
    reject_low_quality_hand: bool = False
    allowed_missing_keypoints_per_hand: int = 0
    projection_enabled: bool = True
    projection_image_width: int | None = None
    projection_image_height: int | None = None
    projection_fx: float | None = None
    projection_fy: float | None = None
    projection_cx: float | None = None
    projection_cy: float | None = None
    projection_border_margin_px: float = 20.0
    projection_near_border_count_threshold: int = 6
    projection_outside_count_threshold: int = 1
    projection_center_jump_px_threshold: float = 120.0
    projection_bbox_area_change_ratio_threshold: float = 3.0
    candidate_gap_close_frames: int = 2
    candidate_min_seed_run_frames: int = 3
    candidate_pre_context_frames: int = 10
    candidate_post_context_frames: int = 10
    candidate_merge_overlapping_only: bool = True
    pass_threshold: float = 0.90


@dataclass
class TextIntegrityConfig:
    required_fields: list[str] = field(
        default_factory=lambda: ["scene", "task", "text_en"]
    )


@dataclass
class PrecheckConfig:
    output_dir: Path
    enabled_checks: list[str] = field(
        default_factory=lambda: [
            "overexposure",
            "keypoint_temporal",
            "keypoint_missing",
            "mask_containment",
            "quality_score",
        ]
    )
    overwrite: bool = False
    input_paths: list[Path] = field(default_factory=list)
    fps: float | None = None
    overexposure: OverexposureConfig = field(default_factory=OverexposureConfig)
    keypoint_temporal: KeypointTemporalConfig = field(default_factory=KeypointTemporalConfig)
    keypoint_missing: KeypointMissingConfig = field(default_factory=KeypointMissingConfig)
    keypoint_morphology: KeypointMorphologyConfig = field(
        default_factory=KeypointMorphologyConfig
    )
    mask_containment: MaskContainmentConfig = field(default_factory=MaskContainmentConfig)
    quality_score: QualityScoreConfig = field(default_factory=QualityScoreConfig)
    composite_frame_verdict: CompositeFrameVerdictConfig = field(
        default_factory=CompositeFrameVerdictConfig
    )
    skeleton_quality_score: SkeletonQualityScoreConfig = field(
        default_factory=SkeletonQualityScoreConfig
    )
    text_integrity: TextIntegrityConfig = field(default_factory=TextIntegrityConfig)
    log_level: str = "INFO"

    def check_config(self, name: str) -> dict:
        value = getattr(self, name, None)
        return value.__dict__.copy() if value is not None else {}


def load_precheck_config(config_path: Path) -> PrecheckConfig:
    import yaml

    with open(config_path) as handle:
        config_dict = yaml.safe_load(handle) or {}

    config_dict["output_dir"] = Path(config_dict["output_dir"])
    config_dict["input_paths"] = [
        Path(path) for path in config_dict.get("input_paths", [])
    ]

    overexposure = OverexposureConfig(**config_dict.pop("overexposure", {}))
    keypoint_temporal = KeypointTemporalConfig(
        **config_dict.pop("keypoint_temporal", {})
    )
    keypoint_missing = KeypointMissingConfig(
        **config_dict.pop("keypoint_missing", {})
    )
    keypoint_morphology = KeypointMorphologyConfig(
        **config_dict.pop("keypoint_morphology", {})
    )
    mask_containment = MaskContainmentConfig(
        **config_dict.pop("mask_containment", {})
    )
    quality_score = QualityScoreConfig(**config_dict.pop("quality_score", {}))
    composite_frame_verdict = CompositeFrameVerdictConfig(
        **config_dict.pop("composite_frame_verdict", {})
    )
    skeleton_quality_score = SkeletonQualityScoreConfig(
        **config_dict.pop("skeleton_quality_score", {})
    )
    text_integrity = TextIntegrityConfig(**config_dict.pop("text_integrity", {}))
    return PrecheckConfig(
        overexposure=overexposure,
        keypoint_temporal=keypoint_temporal,
        keypoint_missing=keypoint_missing,
        keypoint_morphology=keypoint_morphology,
        mask_containment=mask_containment,
        quality_score=quality_score,
        composite_frame_verdict=composite_frame_verdict,
        skeleton_quality_score=skeleton_quality_score,
        text_integrity=text_integrity,
        **config_dict,
    )
