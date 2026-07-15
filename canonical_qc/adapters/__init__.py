"""Strict source adapters for Canonical QC ingestion."""

from .base import SourceAdapter, SourceInspection
from .standard_hdf5 import StandardHdf5Adapter

__all__ = ["SourceAdapter", "SourceInspection", "StandardHdf5Adapter"]
