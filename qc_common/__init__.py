"""Shared contracts and utilities for independent QC modules."""

from .base import BaseCheck
from .config import LoadedQcConfig, load_qc_acceptance_config
from .contracts import EvidenceRef, Issue, ModuleResult, Verdict
from .report import (
    StaleReportRevisionError,
    load_asset_qc_report,
    write_asset_qc_report,
)
from .schema import ReportValidationError, validate_asset_qc_report, validate_qc_config
from .types import CheckResult, ClipInputs

__all__ = [
    "BaseCheck",
    "CheckResult",
    "ClipInputs",
    "EvidenceRef",
    "Issue",
    "LoadedQcConfig",
    "ModuleResult",
    "StaleReportRevisionError",
    "ReportValidationError",
    "Verdict",
    "load_asset_qc_report",
    "load_qc_acceptance_config",
    "validate_asset_qc_report",
    "validate_qc_config",
    "write_asset_qc_report",
]
