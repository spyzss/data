"""Supplier-specific acceptance manifest adapters."""

from acceptance_pull.supplier_adapters.deepreach import (
    build_deepreach_manifest,
    stage_video_quality_inputs,
)

__all__ = [
    "build_deepreach_manifest",
    "stage_video_quality_inputs",
]
