"""Base check contract shared by QC packages."""

from abc import ABC, abstractmethod
from typing import Literal

from .types import CheckResult, ClipInputs


class BaseCheck(ABC):
    """A clip-level or frame-level QC check."""

    name: str
    granularity: Literal["frame", "clip"]

    @abstractmethod
    def run(self, clip: ClipInputs) -> list[CheckResult]:
        """Run this check and return per-frame results."""
        pass
