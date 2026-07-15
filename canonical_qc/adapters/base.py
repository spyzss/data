"""Common contracts for strict Canonical QC source adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from ..contracts import CanonicalQcEpisode, SourceFile


@dataclass(frozen=True, slots=True)
class SourceInspection:
    adapter_id: str
    adapter_version: str
    source_format: Literal["hdf5", "lerobot"]
    source_schema_version: str
    asset_id: str
    source_root: Path
    main_video_path: Path
    hdf5_path: Path | None = None
    info_path: Path | None = None
    episode_metadata_paths: tuple[Path, ...] = ()
    data_path: Path | None = None
    semantics_path: Path | None = None
    episode_index: int | None = None
    frame_count: int | None = None
    data_row_offset: int = 0
    video_frame_offset: int = 0
    layout_version: Literal["v3", "v2.1"] | None = None
    inspection_source_files: tuple[SourceFile, ...] = ()
    data_row_stop: int = 0
    video_frame_stop: int = 0
    video_width_px: int | None = None
    video_height_px: int | None = None


class SourceAdapter(Protocol):
    adapter_id: str
    adapter_version: str

    def inspect(self, source: Path) -> SourceInspection: ...

    def load(self, source: Path) -> CanonicalQcEpisode: ...
