"""Public contract boundary for canonical QC ingestion."""

from .adapters import SourceAdapter, SourceInspection, StandardHdf5Adapter, StandardLeRobotAdapter
from .contracts import (
    CameraCalibration,
    CanonicalQcEpisode,
    EpisodeIdentity,
    EpisodeSemantics,
    HandObservation,
    ProbedVideo,
    SourceFile,
    SourceProvenance,
    Subtask,
    SupplierEvidence,
    SupplierHandQuality,
    TimeAxis,
    VideoStream,
)
from .errors import CanonicalInputError
from .bridge import CanonicalQcBridge
from .config import LoadedCanonicalQcConfig, load_canonical_qc_config
from .workflow import CanonicalQcRunResult, load_canonical_source, run_canonical_source_qc
from .source_gate import DeclaredEpisodeIdentity, SourceGateLocator
from .provenance import semantic_fingerprint, source_fingerprint
from .validation import validate_episode, validate_video_alignment
from .video_probe import probe_video

__all__ = [
    "CameraCalibration",
    "CanonicalInputError",
    "CanonicalQcBridge",
    "CanonicalQcRunResult",
    "DeclaredEpisodeIdentity",
    "LoadedCanonicalQcConfig",
    "CanonicalQcEpisode",
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
    "SupplierHandQuality",
    "TimeAxis",
    "VideoStream",
    "semantic_fingerprint",
    "probe_video",
    "source_fingerprint",
    "validate_episode",
    "validate_video_alignment",
]
