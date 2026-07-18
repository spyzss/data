"""Pure manifest-metadata helpers shared by QC producers and reports."""

from __future__ import annotations

import copy
import math
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
        "cache",
        "canonical_episode",
        "canonical_source_root",
        "clip_inputs",
        "manifest_row",
        "profile",
        "producer",
        "reuse_artifacts",
        "session",
    }
)
_RUNTIME_METADATA_PREFIXES = (
    "cache_",
    "producer_",
    "session_",
)


def canonicalize_manifest_value(value: Any) -> Any:
    """Return one JSON-compatible canonical manifest value.

    CSV readers commonly represent an empty cell as a floating-point NaN.
    Canonical manifest identity uses ``None`` for every missing representation
    so persisted reports never retain a non-reflexive NaN value.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return None if not value.strip() else value
    if isinstance(value, Mapping):
        return {
            key: canonicalize_manifest_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [canonicalize_manifest_value(item) for item in value]
    try:
        if math.isnan(value):
            return None
    except (TypeError, ValueError):
        pass

    # NumPy scalar values can survive Parquet/CSV conversion.  Convert them
    # without making qc_common depend on NumPy at runtime.
    if type(value).__module__.split(".", 1)[0] == "numpy":
        item = getattr(value, "item", None)
        if callable(item):
            return canonicalize_manifest_value(item())
    return copy.deepcopy(value)


def _is_runtime_metadata_field(key: Any) -> bool:
    return isinstance(key, str) and (
        key in _RUNTIME_METADATA_FIELDS
        or key.startswith(_RUNTIME_METADATA_PREFIXES)
    )


def canonical_manifest_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return stable canonical identity for one source manifest row.

    Manifest-backed contexts retain the original row under ``manifest_row``.
    Direct callers that construct an ``AssetContext`` without that field keep
    their supplied metadata except for values owned by the runner itself.
    """
    row = metadata.get("manifest_row")
    manifest_backed = isinstance(row, Mapping)
    source = row if manifest_backed else metadata
    return {
        key: canonicalize_manifest_value(value)
        for key, value in source.items()
        if manifest_backed or not _is_runtime_metadata_field(key)
    }


# Backward-compatible public name used by producer fingerprints and adapters.
# It is an alias, rather than a second implementation, so reports and producers
# share exactly one identity definition.
manifest_metadata = canonical_manifest_metadata


def context_metadata_from_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Rehydrate source manifest metadata for a writer resuming a report."""
    stored = report.get("manifest_metadata")
    if not isinstance(stored, Mapping):
        return {}
    return {"manifest_row": canonical_manifest_metadata({"manifest_row": stored})}


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
