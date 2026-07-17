"""Public contract boundary for canonical QC ingestion."""

from .adapters import SourceAdapter, SourceInspection, StandardHdf5Adapter, StandardLeRobotAdapter
from .contracts import (
    BatchMetadata,
    CameraCalibration,
    CanonicalDataEpisode,
    CanonicalQcEpisode,
    EpisodeIdentity,
    EpisodeSemantics,
    HandObservation,
    ProbedVideo,
    SourceFile,
    SourceProvenance,
    Subtask,
    SupplierEvidence,
    SupplierExtensionField,
    SupplierExtensions,
    SupplierHandQuality,
    TimeAxis,
    VideoStream,
)
from .batch_metadata import load_batch_metadata, with_batch_metadata
from .errors import CanonicalInputError
from .extensions import inventory_hdf5_file
from .bridge import CanonicalQcBridge
from .config import LoadedCanonicalQcConfig, load_canonical_qc_config
from .workflow import CanonicalQcRunResult, load_canonical_source, run_canonical_source_qc
from .source_gate import DeclaredEpisodeIdentity, SourceGateLocator
from .provenance import data_fingerprint, semantic_fingerprint, source_fingerprint
from .revision import (
    CanonicalRevisionArtifact,
    CanonicalRevisionEdit,
    apply_revision_artifact,
    load_revision_artifact,
)
from .validation import validate_episode, validate_video_alignment
from .video_probe import probe_video

__all__ = [
    "CameraCalibration",
    "BatchMetadata",
    "CanonicalDataEpisode",
    "CanonicalInputError",
    "CanonicalQcBridge",
    "CanonicalQcRunResult",
    "DeclaredEpisodeIdentity",
    "LoadedCanonicalQcConfig",
    "CanonicalQcEpisode",
    "CanonicalRevisionArtifact",
    "CanonicalRevisionEdit",
    "EpisodeIdentity",
    "EpisodeSemantics",
    "HandObservation",
    "ProbedVideo",
    "SourceFile",
    "SourceGateLocator",
    "SourceAdapter",
    "SourceInspection",
    "SourceProvenance",
    "StandardHdf5Adapter",
    "StandardLeRobotAdapter",
    "load_canonical_qc_config",
    "load_canonical_source",
    "run_canonical_source_qc",
    "Subtask",
    "SupplierEvidence",
    "SupplierExtensionField",
    "SupplierExtensions",
    "SupplierHandQuality",
    "TimeAxis",
    "VideoStream",
    "semantic_fingerprint",
    "data_fingerprint",
    "apply_revision_artifact",
    "load_revision_artifact",
    "probe_video",
    "source_fingerprint",
    "validate_episode",
    "validate_video_alignment",
    "load_batch_metadata",
    "with_batch_metadata",
    "inventory_hdf5_file",
]
