from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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

        if self.source_range is not None:
            start, end = self.source_range
            if start < 0 or end <= start:
                raise ValueError("source_range must be a non-empty half-open frame range")
