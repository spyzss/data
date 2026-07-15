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
    human_reviews = tuple(_rows(projection, "human_review_rows"))
    profiles = sorted(
        {
            str(row.get("profile") or "unknown")
            for row in assets
        }
    )
    return {
        "overall": _aggregate_group(assets, issues, executions, human_reviews),
        "by_profile": {
            profile: _aggregate_group(
                tuple(row for row in assets if _profile(row) == profile),
                tuple(row for row in issues if _profile(row) == profile),
                tuple(row for row in executions if _profile(row) == profile),
                tuple(row for row in human_reviews if _profile(row) == profile),
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
    human_review_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    assets = _latest_rows(asset_rows, _asset_key)
    asset_keys = {_asset_key(row) for row in assets}
    current_revisions = {_asset_key(row): _revision(row) for row in assets}
    issues = _latest_rows(
        _at_current_revision(issue_rows, current_revisions), _issue_key
    )
    executions = _latest_rows(
        _at_current_revision(execution_rows, current_revisions), _execution_key
    )
    human_reviews = _latest_rows(
        _at_current_revision(human_review_rows, current_revisions), _asset_key
    )

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

    warn_keys = {_issue_key(row) for row in warn_issues}
    reviewed_verdicts = _reviewed_warn_verdicts(human_reviews, warn_keys)
    if not human_reviews:
        # Compatibility for projections produced before the dedicated human
        # row was introduced.  New formal projections always use the
        # revision-scoped human rows above.
        reviewed_verdicts = {
            _issue_key(row): str(row.get("human_verdict") or "").lower()
            for row in warn_issues
            if _is_human_verdict(row.get("human_verdict"))
        }
    human_checked = set(reviewed_verdicts)
    human_resolved = {
        issue_key for issue_key, verdict in reviewed_verdicts.items() if verdict == "pass"
    }
    human_confirmed_fail = {
        issue_key for issue_key, verdict in reviewed_verdicts.items() if verdict == "fail"
    }

    human_counter_rows = human_reviews if human_reviews else assets
    timeline_edit_count = sum(
        _count_value(row.get("timeline_edit_count", 0), "timeline_edit_count")
        for row in human_counter_rows
    )
    subtask_text_edit_count = sum(
        _count_value(
            row.get("subtask_text_edit_count", 0), "subtask_text_edit_count"
        )
        for row in human_counter_rows
    )

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
    result = {
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
        "timeline_edit_count": timeline_edit_count,
        "subtask_text_edit_count": subtask_text_edit_count,
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
    result.update(
        {
            "auto_fail_assets": len(hard_fail_assets),
            "auto_fail_issues": len(hard_fail_issues),
            "machine_warn_issues": len(warn_issues),
            "human_checked_warn_issues": len(human_checked),
            "human_resolved_warn_issues": len(human_resolved),
            "human_confirmed_fail_issues": len(human_confirmed_fail),
            "final_pass_assets": len(final_pass),
            "final_fail_assets": len(final_fail),
        }
    )
    return result


def _at_current_revision(
    rows: Iterable[Mapping[str, Any]],
    current_revisions: Mapping[tuple[str, str], int],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        row
        for row in rows
        if _asset_key(row) in current_revisions
        and _revision(row) == current_revisions[_asset_key(row)]
    )


def _reviewed_warn_verdicts(
    human_rows: Iterable[Mapping[str, Any]],
    warn_keys: set[tuple[str, str, str]],
) -> dict[tuple[str, str, str], str]:
    verdicts: dict[tuple[str, str, str], str] = {}
    for row in human_rows:
        reviews = row.get("issue_reviews", {})
        if not isinstance(reviews, Mapping):
            raise TypeError("human_review_rows.issue_reviews must be a mapping")
        for issue_id, review in reviews.items():
            if not isinstance(review, Mapping):
                raise TypeError("human review entries must be mappings")
            issue_key = (*_asset_key(row), str(issue_id))
            verdict = str(
                review.get("human_verdict") or review.get("verdict") or ""
            ).lower()
            if issue_key in warn_keys and _is_human_verdict(verdict):
                verdicts[issue_key] = verdict
    return verdicts


def _count_value(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{field} must be a non-negative integer")
    return value


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
