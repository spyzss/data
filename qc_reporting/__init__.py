"""Read-only projections of the canonical asset QC reports.

The report JSON under ``quality_archive`` is the source of truth for batch
outputs.  This package intentionally contains projection helpers only; it
does not mutate reports or re-run any detector.
"""

from .projection import (
    BatchProjection,
    iter_asset_reports,
    project_quality_archive,
    project_quality_archive_review_rows,
    project_warn_review_rows,
)
from .aggregate import aggregate_projection
from .migration import reconcile_legacy_outputs

__all__ = [
    "BatchProjection",
    "aggregate_projection",
    "iter_asset_reports",
    "project_quality_archive",
    "project_quality_archive_review_rows",
    "project_warn_review_rows",
    "reconcile_legacy_outputs",
]
