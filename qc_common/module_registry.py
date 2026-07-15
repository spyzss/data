"""Generic runner contracts and per-asset module registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from qc_common.config import LoadedQcConfig
from qc_common.contracts import ModuleResult

if TYPE_CHECKING:
    from qc_pipeline.context import AssetContext


ModuleRunner = Callable[["AssetContext", LoadedQcConfig], ModuleResult]


class ModuleUnavailableError(KeyError):
    """Raised when Config names an automatic implementation with no runner."""

    def __init__(self, module: str) -> None:
        self.module = module
        self.message = f"automatic module implementation is unavailable: {module}"
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class ModulePrerequisiteError(RuntimeError):
    """A declared automatic runner lacks a source it needs to execute."""

    def __init__(self, module: str, prerequisite: str) -> None:
        self.module = module
        self.prerequisite = prerequisite
        super().__init__(f"{module} prerequisite unavailable: {prerequisite}")


class _ModuleStatusError(RuntimeError):
    status = "blocked"

    def __init__(self, module: str, reason: str) -> None:
        self.module = module
        self.reason = reason
        super().__init__(f"{module} {self.status}: {reason}")


class ModuleInputError(_ModuleStatusError):
    """A producer input exists but violates its explicit contract."""

    status = "input_invalid"


class ModuleAdapterMissingError(_ModuleStatusError):
    """The supplier has no validated adapter for this producer."""

    status = "adapter_missing"


class ModuleBlockedError(_ModuleStatusError):
    """The producer is valid but cannot run in the current environment."""

    status = "blocked"


class ModuleRegistry:
    """Mutable registry owned by one asset worker."""

    def __init__(self) -> None:
        self._runners: dict[str, ModuleRunner] = {}

    def register(self, name: str, runner: ModuleRunner) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("module implementation name must not be empty")
        if not callable(runner):
            raise TypeError("module runner must be callable")
        if name in self._runners:
            raise ValueError(f"module implementation already registered: {name}")
        self._runners[name] = runner

    def resolve(self, name: str) -> ModuleRunner:
        try:
            return self._runners[name]
        except KeyError:
            raise ModuleUnavailableError(name) from None

    def has(self, name: str) -> bool:
        return name in self._runners


__all__ = [
    "ModuleAdapterMissingError",
    "ModuleBlockedError",
    "ModuleInputError",
    "ModulePrerequisiteError",
    "ModuleRegistry",
    "ModuleRunner",
    "ModuleUnavailableError",
]
