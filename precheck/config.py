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


@dataclass
class KeypointMissingConfig:
    sides: list[str] = field(default_factory=lambda: ["left", "right"])
    joint_names: list[str] | None = None
    fps: float | None = None
    window_seconds: float = 10.0
    allowed_missing_seconds: float = 1.0


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
        mask_containment=mask_containment,
        quality_score=quality_score,
        composite_frame_verdict=composite_frame_verdict,
        skeleton_quality_score=skeleton_quality_score,
        text_integrity=text_integrity,
        **config_dict,
    )
