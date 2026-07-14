from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class AssetContext:
    asset_id: str
    batch_root: Path
    report_path: Path
    source_files: Mapping[str, Any]
    source_range: tuple[int, int] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.asset_id:
            raise ValueError("asset_id must not be empty")

        batch_root = self.batch_root.resolve()
        report_path = self.report_path.resolve()
        try:
            report_path.relative_to(batch_root)
        except ValueError:
            raise ValueError("report_path must be inside batch_root") from None

        source_files = copy.deepcopy(dict(self.source_files))
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

        if self.source_range is not None:
            start, end = self.source_range
            if start < 0 or end <= start:
                raise ValueError("source_range must be a non-empty half-open frame range")

        object.__setattr__(self, "batch_root", batch_root)
        object.__setattr__(self, "report_path", report_path)
        object.__setattr__(
            self,
            "source_files",
            MappingProxyType(source_files),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(copy.deepcopy(dict(self.metadata))),
        )
