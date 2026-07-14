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
    "ModulePrerequisiteError",
    "ModuleRegistry",
    "ModuleRunner",
    "ModuleUnavailableError",
]
