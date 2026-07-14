"""Adapters from legacy detector outputs to unified QC contracts."""

from .precheck import (
    adapt_hdf5_text_info,
    adapt_quality_hand,
    precheck_config_from_unified,
)

__all__ = [
    "adapt_hdf5_text_info",
    "adapt_quality_hand",
    "precheck_config_from_unified",
]
