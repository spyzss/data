"""Configuration schema for semantic annotation verification."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class InstructionConsistencyConfig:
    sample_frames: int = 8
    model_name: str | None = None


@dataclass
class AnnotationVerifyConfig:
    output_dir: Path
    enabled_checks: list[str] = field(default_factory=lambda: ["instruction_consistency"])
    overwrite: bool = False
    instruction_consistency: InstructionConsistencyConfig = field(
        default_factory=InstructionConsistencyConfig
    )
    log_level: str = "INFO"

    def check_config(self, name: str) -> dict:
        value = getattr(self, name, None)
        return value.__dict__.copy() if value is not None else {}


def load_annotation_verify_config(config_path: Path) -> AnnotationVerifyConfig:
    import yaml

    with open(config_path) as handle:
        config_dict = yaml.safe_load(handle) or {}
    config_dict["output_dir"] = Path(config_dict["output_dir"])
    instruction_consistency = InstructionConsistencyConfig(
        **config_dict.pop("instruction_consistency", {})
    )
    return AnnotationVerifyConfig(
        instruction_consistency=instruction_consistency,
        **config_dict,
    )
