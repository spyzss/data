"""Strict source adapters for Canonical QC ingestion."""

from .base import SourceAdapter, SourceInspection
from .standard_hdf5 import StandardHdf5Adapter
from .standard_lerobot import StandardLeRobotAdapter

__all__ = ["SourceAdapter", "SourceInspection", "StandardHdf5Adapter", "StandardLeRobotAdapter"]
