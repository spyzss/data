"""Pure manifest-metadata helpers shared by QC producers and reports."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any


MANIFEST_TEXT_METADATA_FIELDS = (
    "scene",
    "task",
    "task_name",
    "text_en",
    "text_label",
)

_RUNTIME_METADATA_FIELDS = frozenset(
    {
        "canonical_episode",
        "canonical_source_root",
        "clip_inputs",
        "manifest_row",
        "profile",
        "reuse_artifacts",
    }
)


def manifest_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return the source manifest row, excluding runner-only metadata.

    Manifest-backed contexts retain the original row under ``manifest_row``.
    Direct callers that construct an ``AssetContext`` without that field keep
    their supplied metadata except for values owned by the runner itself.
    """
    row = metadata.get("manifest_row")
    if isinstance(row, Mapping):
        return copy.deepcopy(dict(row))
    return {
        key: copy.deepcopy(value)
        for key, value in metadata.items()
        if key not in _RUNTIME_METADATA_FIELDS
    }


def context_metadata_from_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Rehydrate source manifest metadata for a writer resuming a report."""
    stored = report.get("manifest_metadata")
    if not isinstance(stored, Mapping):
        return {}
    return {"manifest_row": copy.deepcopy(dict(stored))}


def normalized_manifest_text_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Return only text-decision fields in their cache-identity form."""
    source = manifest_metadata(metadata)
    return {
        field: (
            value.strip() if isinstance(value := source.get(field), str) else value
        )
        for field in MANIFEST_TEXT_METADATA_FIELDS
    }
