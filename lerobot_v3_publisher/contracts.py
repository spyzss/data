"""Immutable contracts for gated Curated LeRobot v3 publication."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from canonical_qc.contracts import CanonicalQcEpisode

from .toolchain import WriterToolchain, publisher_version_for


PUBLISHER_VERSION = publisher_version_for("lerobot_v3_curated.v1")


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
        if type(self.request) is not PublishRequest:
            raise TypeError("request must be an exact PublishRequest")
        source_snapshot = tuple(self.source_snapshot)
        if any(type(item) is not SourceSnapshot for item in source_snapshot):
            raise TypeError("source_snapshot entries must be exact SourceSnapshot values")
        object.__setattr__(self, "release_path", Path(self.release_path))
        object.__setattr__(self, "current_path", Path(self.current_path))
        object.__setattr__(self, "source_snapshot", source_snapshot)


@dataclass(frozen=True, slots=True)
class ManifestFile:
    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class VideoMaterialization:
    relative_path: str
    source_relative_path: str
    source_frame_range: tuple[int, int]
    source_sha256: str
    target_sha256: str
    method: Literal["verified_copy", "transcoded_frame_range"]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_frame_range", tuple(self.source_frame_range))


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
    video_materialization: VideoMaterialization | None = None
    toolchain: WriterToolchain | None = None

    def __post_init__(self) -> None:
        files = tuple(self.files)
        if any(type(item) is not ManifestFile for item in files):
            raise TypeError("files entries must be exact ManifestFile values")
        if (
            self.video_materialization is not None
            and type(self.video_materialization) is not VideoMaterialization
        ):
            raise TypeError(
                "video_materialization must be an exact VideoMaterialization value"
            )
        if self.toolchain is not None and type(self.toolchain) is not WriterToolchain:
            raise TypeError("toolchain must be an exact WriterToolchain value")
        object.__setattr__(self, "files", files)


@dataclass(frozen=True, slots=True)
class StagedRelease:
    plan: PublishPlan
    root: Path
    transaction_id: str
    manifest: ReleaseManifest
    manifest_sha256: str
    checksums_sha256: str
    root_device: int = 0
    root_inode: int = 0

    def __post_init__(self) -> None:
        if type(self.plan) is not PublishPlan:
            raise TypeError("plan must be an exact PublishPlan")
        if type(self.manifest) is not ReleaseManifest:
            raise TypeError("manifest must be an exact ReleaseManifest")
        object.__setattr__(self, "root", Path(self.root))


@dataclass(frozen=True, slots=True)
class ValidationReport:
    schema_version: Literal["curated_lerobot_v3_validation.v1"]
    release_id: str
    file_count: int
    frame_count: int
    manifest_sha256: str
    official_reader_version: str
    official_reader_versions: tuple[tuple[str, str], ...]
    official_reader_fingerprint: str
    manifest: ReleaseManifest

    def __post_init__(self) -> None:
        if type(self.manifest) is not ReleaseManifest:
            raise TypeError("manifest must be an exact ReleaseManifest")
        object.__setattr__(
            self, "official_reader_versions", tuple(self.official_reader_versions)
        )


@dataclass(frozen=True, slots=True)
class PublishResult:
    state: Literal["published", "already_published"]
    plan: PublishPlan
    manifest: ReleaseManifest

    def __post_init__(self) -> None:
        if type(self.plan) is not PublishPlan:
            raise TypeError("plan must be an exact PublishPlan")
        if type(self.manifest) is not ReleaseManifest:
            raise TypeError("manifest must be an exact ReleaseManifest")


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
    "StagedRelease",
    "ValidationReport",
    "VideoMaterialization",
    "WriterToolchain",
]
