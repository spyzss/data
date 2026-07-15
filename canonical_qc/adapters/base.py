"""Common contracts for strict Canonical QC source adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from ..contracts import CanonicalQcEpisode


@dataclass(frozen=True, slots=True)
class SourceInspection:
    adapter_id: str
    adapter_version: str
    source_format: Literal["hdf5", "lerobot"]
    source_schema_version: str
    asset_id: str
    source_root: Path
    hdf5_path: Path
    main_video_path: Path


class SourceAdapter(Protocol):
    adapter_id: str
    adapter_version: str

    def inspect(self, source: Path) -> SourceInspection: ...

    def load(self, source: Path) -> CanonicalQcEpisode: ...
