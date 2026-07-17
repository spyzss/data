"""Read-only projections of the canonical asset QC reports.

The report JSON under ``quality_archive`` is the source of truth for batch
outputs.  This package intentionally contains projection helpers only; it
does not mutate reports or re-run any detector.
"""

from .projection import (
    BatchProjection,
    iter_asset_reports,
    project_human_review_rows,
    project_quality_archive,
    project_quality_archive_review_rows,
    project_warn_review_rows,
)
from .aggregate import aggregate_projection
from .export import (
    FORMAL_AGGREGATE_METRICS,
    aggregate_metric_rows,
    write_aggregate_outputs,
)
from .migration import reconcile_legacy_outputs

__all__ = [
    "BatchProjection",
    "aggregate_projection",
    "FORMAL_AGGREGATE_METRICS",
    "aggregate_metric_rows",
    "iter_asset_reports",
    "project_human_review_rows",
    "project_quality_archive",
    "project_quality_archive_review_rows",
    "project_warn_review_rows",
    "reconcile_legacy_outputs",
    "write_aggregate_outputs",
]
