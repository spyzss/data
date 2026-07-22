"""Revision-aware persistence helpers for human-only report state.

Automatic QC modules own the machine observations in an asset report.  This
module deliberately routes every human write through the shared report loader
and atomic writer, and performs a copy-on-write ownership check before the
writer is called.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from qc_common.report import (
    StaleReportRevisionError,
    load_asset_qc_report,
    write_asset_qc_report,
)
from qc_common.schema import validate_asset_qc_report


_HUMAN_MUTABLE_TOP_LEVEL = frozenset(
    {
        "semantic_calibration",
        # The semantic workbench is the external owner of this module block;
        # automatic modules may only write their own result blocks.
        "semantic_consistency",
        "manual_review",
        "pipeline_state",
        "overall_decision",
        "report_revision",
    }
)


def _assert_human_owned_diff(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> None:
    keys = set(before) | set(after)
    for key in keys:
        if key in _HUMAN_MUTABLE_TOP_LEVEL:
            continue
        if key == "execution":
            before_execution = before.get(key)
            after_execution = after.get(key)
            if not isinstance(before_execution, Mapping) or not isinstance(
                after_execution, Mapping
            ):
                if before_execution != after_execution:
                    raise ValueError(
                        "human report mutation may only change allowed human fields; "
                        "execution identity is immutable"
                    )
                continue
            execution_keys = set(before_execution) | set(after_execution)
            for execution_key in execution_keys - {"updated_at"}:
                if before_execution.get(execution_key) != after_execution.get(
                    execution_key
                ) or (execution_key not in before_execution) != (
                    execution_key not in after_execution
                ):
                    raise ValueError(
                        "human report mutation may only change allowed human fields; "
                        "execution identity is immutable"
                    )
            continue
        if before.get(key) != after.get(key) or (key not in before) != (key not in after):
            raise ValueError(
                f"human report mutation may only change allowed human fields; "
                f"machine-owned or unknown field changed: {key}"
            )


def update_human_state(
    report_path: Path,
    expected_revision: int,
    mutate: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    """Apply one expected-revision human mutation through the shared writer."""

    if not callable(mutate):
        raise TypeError("mutate must be callable")
    path = Path(report_path)
    loaded = load_asset_qc_report(path)
    if loaded is None:
        raise FileNotFoundError(path)
    if loaded.get("schema_version") != "asset_qc_report.v2":
        raise ValueError("human report updates require an asset_qc_report.v2 report")
    current_revision = int(loaded.get("report_revision", 0))
    if current_revision != expected_revision:
        raise StaleReportRevisionError(
            f"expected revision {expected_revision}, found {current_revision}: {path}"
        )

    before = copy.deepcopy(loaded)
    candidate = copy.deepcopy(loaded)
    mutate(candidate)
    _assert_human_owned_diff(before, candidate)

    candidate["report_revision"] = expected_revision + 1
    validate_asset_qc_report(candidate)
    execution = candidate.get("execution")
    profile = execution.get("profile") if isinstance(execution, Mapping) else None
    write_asset_qc_report(
        path,
        candidate,
        expected_revision=expected_revision,
        profile=profile if isinstance(profile, str) else None,
    )
    return candidate


__all__ = ["update_human_state"]
