"""Small registry helper for pluggable QC checks."""

from collections.abc import Callable

from .base import BaseCheck


class CheckRegistry:
    """Map configured check names to concrete check classes."""

    def __init__(self) -> None:
        self._checks: dict[str, type[BaseCheck]] = {}

    def register(self, cls: type[BaseCheck]) -> type[BaseCheck]:
        """Decorator used by check modules."""
        if not getattr(cls, "name", ""):
            raise ValueError(f"{cls.__name__} must define a non-empty name")
        self._checks[cls.name] = cls
        return cls

    def get(self, name: str) -> type[BaseCheck]:
        try:
            return self._checks[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._checks)) or "<none>"
            raise ValueError(f"Unknown check '{name}'. Available: {available}") from exc

    def names(self) -> list[str]:
        return sorted(self._checks)

    def create(self, name: str, config: dict) -> BaseCheck:
        cls = self.get(name)
        return cls(config)


RegisterDecorator = Callable[[type[BaseCheck]], type[BaseCheck]]
