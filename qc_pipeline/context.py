from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


class _FrozenMapping(Mapping[str, Any]):
    """A recursively frozen mapping that remains deepcopy-compatible."""

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_data", MappingProxyType(dict(data)))

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return repr(dict(self._data))

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[str, Any]:
        """Thaw only the caller's copy; the context remains immutable."""
        copied = {
            copy.deepcopy(key, memo): copy.deepcopy(value, memo)
            for key, value in self._data.items()
        }
        memo[id(self)] = copied
        return copied


def _freeze(value: Any) -> Any:
    # CanonicalQcEpisode is already a deeply immutable, validated contract.
    # Copying it would recreate its NumPy arrays as writable and silently break
    # that contract, so preserve the exact object when it is attached to runner
    # metadata by CanonicalQcBridge.
    try:
        from canonical_qc.contracts import CanonicalQcEpisode
    except ImportError:  # pragma: no cover - canonical_qc ships with this package
        CanonicalQcEpisode = ()  # type: ignore[assignment]
    if isinstance(value, CanonicalQcEpisode):
        return value
    if isinstance(value, _FrozenMapping):
        return value
    if isinstance(value, Mapping):
        return _FrozenMapping(
            {key: _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return copy.deepcopy(value)


def _thaw(value: Any) -> Any:
    """Return a mutable copy of a recursively frozen report-bound value."""
    if isinstance(value, Mapping):
        return {
            copy.deepcopy(key): _thaw(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    return copy.deepcopy(value)


def validate_asset_id(asset_id: str) -> None:
    if not isinstance(asset_id, str) or not asset_id.strip():
        raise ValueError("asset_id must not be empty")
    if (
        asset_id in {".", ".."}
        or "/" in asset_id
        or "\\" in asset_id
    ):
        raise ValueError("asset_id must be a safe filename component")


@dataclass(frozen=True)
class AssetContext:
    asset_id: str
    batch_root: Path
    report_path: Path
    source_files: Mapping[str, Any]
    source_range: tuple[int, int] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_asset_id(self.asset_id)

        batch_root = self.batch_root.resolve()
        report_path = self.report_path.resolve()
        try:
            report_path.relative_to(batch_root)
        except ValueError:
            raise ValueError("report_path must be inside batch_root") from None

        source_files = _freeze(self.source_files)
        if not isinstance(source_files, Mapping):
            raise TypeError("source_files must be a mapping")
        for source_name, source in source_files.items():
            if not isinstance(source, Mapping) or source.get("path") is None:
                continue
            source_path = source.get("path")
            if not isinstance(source_path, str) or not source_path.strip():
                raise ValueError(
                    f"source_files.{source_name}.path must be a non-empty relative path"
                )
            relative_path = Path(source_path)
            if relative_path.is_absolute():
                raise ValueError(
                    f"source_files.{source_name}.path must be relative to batch_root"
                )
            try:
                (batch_root / relative_path).resolve().relative_to(batch_root)
            except ValueError:
                raise ValueError(
                    f"source_files.{source_name}.path must stay inside batch_root"
                ) from None
        try:
            json.dumps(_thaw(source_files), ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "source_files must contain JSON-compatible values"
            ) from exc

        if self.source_range is not None:
            start, end = self.source_range
            if start < 0 or end <= start:
                raise ValueError("source_range must be a non-empty half-open frame range")

        object.__setattr__(self, "batch_root", batch_root)
        object.__setattr__(self, "report_path", report_path)
        object.__setattr__(
            self,
            "source_files",
            source_files,
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze(self.metadata),
        )
