"""Public contract boundary for canonical QC ingestion."""

from .contracts import (
    CameraCalibration,
    CanonicalQcEpisode,
    EpisodeIdentity,
    EpisodeSemantics,
    HandObservation,
    SourceFile,
    SourceProvenance,
    Subtask,
    SupplierEvidence,
    SupplierHandQuality,
    TimeAxis,
    VideoStream,
)
from .errors import CanonicalInputError
from .provenance import semantic_fingerprint, source_fingerprint
from .validation import validate_episode

__all__ = [
    "CameraCalibration",
    "CanonicalInputError",
    "CanonicalQcEpisode",
    "EpisodeIdentity",
    "EpisodeSemantics",
    "HandObservation",
    "SourceFile",
    "SourceProvenance",
    "Subtask",
    "SupplierEvidence",
    "SupplierHandQuality",
    "TimeAxis",
    "VideoStream",
    "semantic_fingerprint",
    "source_fingerprint",
    "validate_episode",
]
