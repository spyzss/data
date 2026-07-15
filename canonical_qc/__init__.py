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
from .provenance import semantic_fingerprint, source_fingerprint
from .validation import validate_episode, validate_video_alignment
from .video_probe import probe_video

__all__ = [
    "CameraCalibration",
    "CanonicalInputError",
    "CanonicalQcBridge",
    "CanonicalQcEpisode",
    "EpisodeIdentity",
    "EpisodeSemantics",
    "HandObservation",
    "ProbedVideo",
    "SourceFile",
    "SourceAdapter",
    "SourceInspection",
    "SourceProvenance",
    "StandardHdf5Adapter",
    "StandardLeRobotAdapter",
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
