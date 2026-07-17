"""Supplier-specific acceptance manifest adapters."""

from acceptance_pull.supplier_adapters.deepreach import (
    build_deepreach_manifest,
    stage_video_quality_inputs,
)
from acceptance_pull.supplier_adapters.potentia import build_potentia_manifest

__all__ = [
    "build_deepreach_manifest",
    "build_potentia_manifest",
    "stage_video_quality_inputs",
]
