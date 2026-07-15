"""Immutable contracts for gated Curated LeRobot v3 publication."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from canonical_qc.contracts import CanonicalQcEpisode


PUBLISHER_VERSION = "lerobot_v3_curated.v1"


@dataclass(frozen=True, slots=True)
class CanonicalDiagnostic:
    code: str
    stage: str
    field: str | None
    message: str
    retryable: bool


class PublishPrerequisiteError(ValueError):
    """One stable, machine-readable publisher prerequisite diagnostic."""

    def __init__(self, diagnostic: CanonicalDiagnostic) -> None:
        self.diagnostic = diagnostic
        super().__init__(
            f"{diagnostic.code}: {diagnostic.stage}: "
            f"{diagnostic.field or '$'}: {diagnostic.message}"
        )


@dataclass(frozen=True, slots=True)
class PublishRequest:
    episode: CanonicalQcEpisode
    canonical_revision: int
    canonical_source_root: Path
    qc_report_path: Path
    expected_report_revision: int
    release_root: Path

    def __post_init__(self) -> None:
        for name in ("canonical_source_root", "qc_report_path", "release_root"):
            object.__setattr__(self, name, Path(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    relative_path: str
    role: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class PublishPlan:
    request: PublishRequest
    release_id: str
    publisher_version: str
    release_path: Path
    current_path: Path
    semantic_fingerprint: str
    source_fingerprint: str
    qc_report_revision: int
    qc_report_sha256: str
    source_snapshot: tuple[SourceSnapshot, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "release_path", Path(self.release_path))
        object.__setattr__(self, "current_path", Path(self.current_path))
        object.__setattr__(self, "source_snapshot", tuple(self.source_snapshot))


@dataclass(frozen=True, slots=True)
class ManifestFile:
    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ReleaseManifest:
    schema_version: Literal["curated_lerobot_v3_release_manifest.v1"]
    release_id: str
    publisher_version: str
    asset_id: str
    canonical_revision: int
    semantic_fingerprint: str
    source_fingerprint: str
    qc_report_revision: int
    qc_report_sha256: str
    files: tuple[ManifestFile, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", tuple(self.files))


@dataclass(frozen=True, slots=True)
class PublishResult:
    state: Literal["published", "already_published"]
    plan: PublishPlan
    manifest: ReleaseManifest


__all__ = [
    "CanonicalDiagnostic",
    "ManifestFile",
    "PUBLISHER_VERSION",
    "PublishPlan",
    "PublishPrerequisiteError",
    "PublishRequest",
    "PublishResult",
    "ReleaseManifest",
    "SourceSnapshot",
]
