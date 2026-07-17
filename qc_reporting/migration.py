from __future__ import annotations

"""Read-only reconciliation helpers for the v1-to-v2 QC migration.

The canonical reports under ``quality_archive`` are projected first and are
the only source used for verdict comparisons.  Legacy files are read as
evidence; this module never migrates, rewrites, or otherwise mutates either
the reports or the sidecars.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .projection import project_quality_archive


def reconcile_legacy_outputs(
    *,
    quality_archive: Path,
    legacy_inputs: Iterable[Path],
) -> list[dict[str, Any]]:
    """Return only legacy/QC JSON differences in deterministic order.

    ``quality_archive`` is projected once, so an explicit v1 report remains
    readable for migration verification while its current JSON decision (even
    ``None``) remains authoritative.  Each legacy input is passed through the
    existing sidecar reader/comparator.  Rows whose verdict agrees with the QC
    JSON are omitted; every returned difference is explicitly marked with
    ``authoritative_source='asset_qc_json'``.
    """

    if isinstance(legacy_inputs, (str, bytes, Path)):
        raise TypeError("legacy_inputs must be an iterable of paths")

    # Import the existing sidecar reader lazily: importing ``qc_reporting``
    # itself should remain a light-weight projection import and must not load
    # the CLI's optional dataframe/export dependencies.
    from tools.build_qc_json_projection import build_reconciliation_rows

    projection = project_quality_archive(Path(quality_archive))
    differences: list[dict[str, Any]] = []
    for raw_path in legacy_inputs:
        path = Path(raw_path)
        source = path.stem or path.name
        rows = build_reconciliation_rows(projection, {source: path})
        for row in rows:
            if row.get("difference_type") == "match":
                continue
            difference = dict(row)
            difference["authoritative_source"] = "asset_qc_json"
            differences.append(difference)

    return sorted(
        differences,
        key=lambda row: (
            str(row.get("legacy_path") or ""),
            int(row.get("legacy_row_index") or 0),
            str(row.get("asset_id") or ""),
            str(row.get("difference_type") or ""),
        ),
    )


__all__ = ["reconcile_legacy_outputs"]
