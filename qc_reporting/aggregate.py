"""Deterministic batch statistics for :mod:`qc_reporting.projection`.

The aggregate layer consumes only normalized QC JSON rows.  It intentionally
does not inspect sidecars or the manual queue candidate list, so a report can
be re-projected at any time without changing the machine-quality denominator.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from qc_reporting.projection import BatchProjection


_TERMINAL_STATUSES = frozenset({"completed", "stopped"})
_NON_COVERAGE_STATES = frozenset(
    {"skipped_due_to_fail", "disabled", "not_implemented", "unknown"}
)


def aggregate_projection(projection: BatchProjection) -> dict[str, Any]:
    """Return overall and per-execution-profile QC statistics.

    Asset and issue counts use separate stable keys.  If a caller combines
    projections from multiple revisions, the row with the greatest
    ``report_revision`` wins for each asset/issue/module identity; this makes
    the function safe for incremental rebuilds while preserving one-asset-one-
    denominator semantics.
    """

    assets = tuple(_rows(projection, "asset_rows"))
    issues = tuple(_rows(projection, "issue_rows"))
    executions = tuple(_rows(projection, "execution_rows"))
    profiles = sorted(
        {
            str(row.get("profile") or "unknown")
            for row in assets
        }
    )
    return {
        "overall": _aggregate_group(assets, issues, executions),
        "by_profile": {
            profile: _aggregate_group(
                tuple(row for row in assets if _profile(row) == profile),
                tuple(row for row in issues if _profile(row) == profile),
                tuple(row for row in executions if _profile(row) == profile),
            )
            for profile in profiles
        },
    }


def _rows(projection: Any, name: str) -> tuple[Mapping[str, Any], ...]:
    raw = getattr(projection, name, ())
    if raw is None:
        return ()
    if isinstance(raw, (str, bytes)):
        raise TypeError(f"projection.{name} must be an iterable of row mappings")
    rows: list[Mapping[str, Any]] = []
    for index, row in enumerate(raw):
        if not isinstance(row, Mapping):
            raise TypeError(f"projection.{name}[{index}] must be a mapping")
        rows.append(row)
    return tuple(rows)


def _profile(row: Mapping[str, Any]) -> str:
    return str(row.get("profile") or "unknown")


def _asset_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return _profile(row), str(row.get("asset_id") or "")


def _revision(row: Mapping[str, Any]) -> int:
    try:
        return int(row.get("report_revision", 0))
    except (TypeError, ValueError):
        return 0


def _latest_rows(
    rows: Iterable[Mapping[str, Any]],
    key_fn: Any,
) -> tuple[Mapping[str, Any], ...]:
    latest: dict[Any, Mapping[str, Any]] = {}
    for row in rows:
        key = key_fn(row)
        previous = latest.get(key)
        if previous is None or _revision(row) >= _revision(previous):
            latest[key] = row
    return tuple(latest.values())


def _issue_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (*_asset_key(row), str(row.get("issue_id") or ""))


def _execution_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (*_asset_key(row), str(row.get("module") or ""))


def _aggregate_group(
    asset_rows: Iterable[Mapping[str, Any]],
    issue_rows: Iterable[Mapping[str, Any]],
    execution_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    assets = _latest_rows(asset_rows, _asset_key)
    issues = _latest_rows(issue_rows, _issue_key)
    executions = _latest_rows(execution_rows, _execution_key)
    asset_keys = {_asset_key(row) for row in assets}

    hard_fail_issues = tuple(
        row for row in issues if _machine_severity(row) == "fail"
    )
    warn_issues = tuple(row for row in issues if _machine_severity(row) == "warn")
    hard_fail_assets = {
        _asset_key(row) for row in hard_fail_issues if _asset_key(row) in asset_keys
    }
    warn_assets = {
        _asset_key(row) for row in warn_issues if _asset_key(row) in asset_keys
    }

    completed = tuple(
        row for row in assets if str(row.get("status") or "") == "completed"
    )
    terminal = tuple(
        row for row in assets if str(row.get("status") or "") in _TERMINAL_STATUSES
    )
    incomplete = tuple(
        row for row in assets if str(row.get("status") or "") not in _TERMINAL_STATUSES
    )
    final_pass = {
        _asset_key(row)
        for row in terminal
        if row.get("decision", row.get("overall_decision")) == "pass"
    }
    final_fail = {
        _asset_key(row)
        for row in terminal
        if row.get("decision", row.get("overall_decision")) == "fail"
    }

    human_checked = {
        _issue_key(row)
        for row in warn_issues
        if _is_human_verdict(row.get("human_verdict"))
    }
    human_resolved = {
        _issue_key(row)
        for row in warn_issues
        if _is_human_verdict(row.get("effective_verdict"))
    }
    human_confirmed_fail = {
        _issue_key(row)
        for row in warn_issues
        if row.get("effective_verdict") == "fail"
        or row.get("human_verdict") == "fail"
    }

    module_coverage, module_state_counts = _module_metrics(executions, asset_keys)
    stop_position = Counter(
        str(row.get("stop_position"))
        for row in assets
        if row.get("stop_position") not in {None, ""}
    )
    continued_assets = {
        _asset_key(row)
        for row in executions
        if row.get("continued_after_fail") is True
    }
    runtime_error_assets = {
        _asset_key(row)
        for row in executions
        if row.get("runtime_error") is not None
        or bool(row.get("runtime_errors"))
        or row.get("state") == "runtime_error"
    }

    final_asset_count = len(final_pass | final_fail)
    pass_rate = len(final_pass) / final_asset_count if final_asset_count else 0.0
    asset_count = len(assets)
    return {
        "asset_count": asset_count,
        "total_asset_count": asset_count,
        "issue_count": len(issues),
        "execution_row_count": len(executions),
        "completed_asset_count": len(completed),
        "terminal_asset_count": len(terminal),
        "incomplete_asset_count": len(incomplete),
        "automatic_hard_fail_asset_count": len(hard_fail_assets),
        "automatic_hard_fail_issue_count": len(hard_fail_issues),
        "machine_warn_asset_count": len(warn_assets),
        "machine_warn_issue_count": len(warn_issues),
        "human_checked_warn_issue_count": len(human_checked),
        "human_resolved_warn_issue_count": len(human_resolved),
        "human_confirmed_fail_issue_count": len(human_confirmed_fail),
        "final_pass_asset_count": len(final_pass),
        "final_fail_asset_count": len(final_fail),
        "final_asset_count": final_asset_count,
        "pass_rate": pass_rate,
        "module_coverage": module_coverage,
        "module_state_counts": module_state_counts,
        "stop_position": dict(sorted(stop_position.items())),
        "continued_after_fail_asset_count": len(continued_assets),
        "runtime_error_asset_count": len(runtime_error_assets),
    }


def _machine_severity(row: Mapping[str, Any]) -> str:
    return str(
        row.get("machine_severity")
        or row.get("machine_verdict")
        or row.get("severity")
        or ""
    ).lower()


def _is_human_verdict(value: Any) -> bool:
    return str(value or "").lower() in {"pass", "fail"}


def _module_metrics(
    execution_rows: Iterable[Mapping[str, Any]],
    asset_keys: set[tuple[str, str]],
) -> tuple[dict[str, float], dict[str, dict[str, int]]]:
    by_module: dict[str, set[tuple[str, str]]] = defaultdict(set)
    state_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in execution_rows:
        module = str(row.get("module") or "unknown")
        key = _asset_key(row)
        state = str(row.get("state") or "unknown")
        state_counts[module][state] += 1
        if key in asset_keys and state not in _NON_COVERAGE_STATES:
            by_module[module].add(key)

    denominator = len(asset_keys)
    coverage = {
        module: (len(keys) / denominator if denominator else 0.0)
        for module, keys in sorted(by_module.items())
    }
    normalized_states = {
        module: dict(sorted(counter.items()))
        for module, counter in sorted(state_counts.items())
    }
    return coverage, normalized_states


__all__ = ["aggregate_projection"]
