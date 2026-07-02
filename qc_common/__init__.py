"""Shared contracts for independent QC modules."""

from .base import BaseCheck
from .types import CheckResult, ClipInputs

__all__ = ["BaseCheck", "CheckResult", "ClipInputs"]
