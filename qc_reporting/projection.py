from __future__ import annotations

"""Project canonical asset QC reports into human-review queue rows.

``quality_archive/*.json`` is the authoritative input for the formal manual
queue.  Legacy candidate-window/SAM3/video sidecars are deliberately handled
by the legacy path in :mod:`tools.build_manual_review_queue` and are not read
here.
"""

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qc_common.schema import validate_asset_qc_report


_SEVERITIES = {"low", "medium", "high", "critical"}


@dataclass(frozen=True)
class BatchProjection:
    """Immutable container for the three normalized QC report tables.

    The dictionaries deliberately remain open-ended.  Reports are an
    extension point for the human-QC change, while the tuple boundaries make
    it impossible for a caller to accidentally append a row to one of the
    projection tables in place.
    """

    asset_rows: tuple[dict[str, Any], ...]
    issue_rows: tuple[dict[str, Any], ...]
    execution_rows: tuple[dict[str, Any], ...]
    source_manifest: tuple[dict[str, Any], ...]


def _json_files(quality_archive: Path) -> tuple[Path, ...]:
    root = Path(quality_archive)
    if not root.is_dir():
        raise ValueError(f"quality archive directory does not exist: {root}")
    return tuple(sorted(path for path in root.glob("*.json") if path.is_file()))


def _load_validated_report(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid QC report {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"invalid QC report {path}: root must be an object")
    try:
        validate_asset_qc_report(loaded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid QC report {path}: {exc}") from exc
    return loaded


def _iter_asset_reports_with_paths(
    quality_archive: Path,
) -> Iterator[tuple[Path, dict[str, Any]]]:
    for path in _json_files(quality_archive):
        yield path, _load_validated_report(path)


def iter_asset_reports(quality_archive: Path) -> Iterator[dict[str, Any]]:
    """Yield validated reports in deterministic filename order.

    Every file is validated before it is yielded.  A malformed report raises a
    ``ValueError`` containing the source JSON path so a batch cannot silently
    omit one asset.
    """

    for _path, report in _iter_asset_reports_with_paths(quality_archive):
        yield report


def project_quality_archive(quality_archive: Path) -> BatchProjection:
    """Read every canonical report and flatten it into three QC tables.

    The source JSONs are validated before any row is returned.  This function
    never joins against the manual-review candidate list: candidate selection
    controls the work queue, not batch quality statistics.  Consequently a
    warn issue remains visible in ``issue_rows`` even when it was not selected
    for human review.
    """

    asset_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []
    execution_rows: list[dict[str, Any]] = []
    source_manifest: list[dict[str, Any]] = []

    for path, report in _iter_asset_reports_with_paths(quality_archive):
        try:
            asset_row, issues, execution = _project_report_rows(path, report)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"cannot project QC report {path}: {exc}") from exc
        asset_rows.append(asset_row)
        issue_rows.extend(issues)
        execution_rows.extend(execution)
        source_manifest.append(
            {
                "path": str(path),
                "report_path": str(path),
                "json_path": str(path),
                "asset_id": asset_row["asset_id"],
                "batch_id": asset_row["batch_id"],
                "profile": asset_row["profile"],
                "report_revision": asset_row["report_revision"],
                "schema_version": asset_row["schema_version"],
                "config_hash": asset_row["config_hash"],
            }
        )

    return BatchProjection(
        asset_rows=tuple(asset_rows),
        issue_rows=tuple(issue_rows),
        execution_rows=tuple(execution_rows),
        source_manifest=tuple(source_manifest),
    )


_REPORT_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "asset_id",
        "supplier_id",
        "batch_id",
        "report_revision",
        "qc_config",
        "execution",
        "pipeline_state",
        "overall_decision",
        "source_files",
        "issues",
        "runtime_errors",
        "manual_review",
        "metadata",
        "semantic_calibration",
        "canonical_binding",
        "canonical_qc_range",
        "source_gate",
    }
)


def _project_report_rows(
    path: Path,
    report: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    execution = report.get("execution")
    execution_map = execution if isinstance(execution, Mapping) else {}
    pipeline = report.get("pipeline_state")
    pipeline_map = pipeline if isinstance(pipeline, Mapping) else {}
    config = report.get("qc_config")
    config_map = config if isinstance(config, Mapping) else {}

    asset_id = str(report.get("asset_id") or "")
    if not asset_id:
        raise ValueError("asset_id must be a non-empty string")
    profile = str(execution_map.get("profile") or report.get("profile") or "unknown")
    status = str(pipeline_map.get("status") or "unknown")
    decision = report.get("overall_decision")
    if decision not in {"pass", "fail", None}:
        raise ValueError(f"overall_decision has unsupported value: {decision!r}")

    states = _module_states(report)
    module_coverage = {
        module: 1.0 if _state_counts_as_coverage(state) else 0.0
        for module, state in states.items()
    }
    stop_position = pipeline_map.get("last_completed_module")
    if not isinstance(stop_position, str) or not stop_position:
        stop_position = pipeline_map.get("next_module")
    stop_position = str(stop_position) if stop_position else None

    asset_row: dict[str, Any] = {
        "asset_id": asset_id,
        "batch_id": str(report.get("batch_id") or "unknown"),
        "supplier_id": _supplier_id(report),
        "profile": profile,
        "schema_version": str(report.get("schema_version") or ""),
        "status": status,
        "pipeline_status": status,
        "decision": decision,
        "overall_decision": decision,
        "report_revision": int(report.get("report_revision", 0)),
        "config_hash": str(config_map.get("config_hash") or ""),
        "config_version": str(config_map.get("config_version") or ""),
        "config_path": str(config_map.get("config_path") or ""),
        "module_coverage": module_coverage,
        "stop_position": stop_position,
        "stop_reason": pipeline_map.get("stop_reason"),
        "finalizable": status in {"completed", "stopped"}
        and decision in {"pass", "fail"},
    }

    issue_rows = [
        _project_issue_row(report, issue, asset_row)
        for issue in _issues(report)
    ]
    execution_rows = [
        _project_execution_row(report, module, state, asset_row)
        for module, state in states.items()
    ]
    return asset_row, issue_rows, execution_rows


def _issues(report: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw = report.get("issues")
    if raw is None:
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("issues must be an array")
    result: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"issues[{index}] must be an object")
        issue_id = item.get("issue_id")
        if not isinstance(issue_id, str) or not issue_id:
            raise ValueError(f"issues[{index}].issue_id must be a non-empty string")
        if issue_id in seen:
            # Canonical mutation normally deduplicates issues.  Treating a
            # duplicate as an invalid source prevents ambiguous joins later.
            raise ValueError(f"duplicate issue_id: {issue_id}")
        seen.add(issue_id)
        result.append(item)
    return tuple(result)


def _module_states(report: Mapping[str, Any]) -> dict[str, str]:
    execution = report.get("execution")
    raw_states = execution.get("module_states") if isinstance(execution, Mapping) else None
    states: dict[str, str] = {}
    if isinstance(raw_states, Mapping):
        for module, raw_state in raw_states.items():
            if not isinstance(module, str) or not module:
                continue
            if isinstance(raw_state, Mapping):
                value = raw_state.get("state")
            else:
                value = raw_state
            states[module] = str(value or "unknown")

    # A valid report written by an early v2 producer can contain module blocks
    # before ``execution.module_states`` was introduced.  Infer only blocks
    # with a result gate; arbitrary extension blocks remain opaque.
    for module, block in report.items():
        if module in _REPORT_TOP_LEVEL_KEYS or module in states:
            continue
        if not isinstance(block, Mapping):
            continue
        flow = block.get("flow")
        result_gate = flow.get("result_gate") if isinstance(flow, Mapping) else None
        if isinstance(result_gate, Mapping) and "verdict" in result_gate:
            states[module] = "completed"
    return states


def _state_counts_as_coverage(state: str) -> bool:
    # ``skipped_due_to_fail`` means the module was never run.  A disabled or
    # not-implemented module is also not coverage; runtime errors and external
    # boundaries are reached stages and are retained in coverage metrics.
    return state not in {"skipped_due_to_fail", "disabled", "not_implemented", "unknown"}


def _project_issue_row(
    report: Mapping[str, Any],
    issue: Mapping[str, Any],
    asset_row: Mapping[str, Any],
) -> dict[str, Any]:
    issue_id = str(issue["issue_id"])
    context = issue.get("context")
    context_map = context if isinstance(context, Mapping) else {}
    module = str(issue.get("module") or "unknown")
    machine_severity = str(issue.get("severity") or "")
    human_verdict, effective_verdict, review = _human_issue_verdict(report, issue_id)
    start = _frame_value(
        _first_present(context_map, issue, "start_frame", "window_start_frame", default=None)
    )
    end = _frame_value(
        _first_present(context_map, issue, "end_frame", "window_end_frame", default=None)
    )
    return {
        "asset_id": asset_row["asset_id"],
        "supplier_id": asset_row["supplier_id"],
        "profile": asset_row["profile"],
        "report_revision": asset_row["report_revision"],
        "issue_id": issue_id,
        "module": module,
        "rule_id": str(issue.get("rule_id") or ""),
        "code": str(issue.get("code") or ""),
        "issue_type": str(issue.get("issue_type") or ""),
        "machine_severity": machine_severity,
        "machine_verdict": machine_severity,
        "severity": machine_severity,
        "human_verdict": human_verdict,
        "effective_verdict": effective_verdict,
        "needs_manual_review": issue.get("needs_manual_review"),
        "window_start_frame": start,
        "window_end_frame": end,
        "source_level": str(
            _first_present(context_map, issue, "source_level", default="asset")
        ),
        "review": dict(review) if isinstance(review, Mapping) else None,
    }


def _human_issue_verdict(
    report: Mapping[str, Any], issue_id: str
) -> tuple[str | None, str | None, Mapping[str, Any] | None]:
    manual = report.get("manual_review")
    if not isinstance(manual, Mapping):
        return None, None, None
    reviews = manual.get("issue_reviews")
    review = reviews.get(issue_id) if isinstance(reviews, Mapping) else None
    if not isinstance(review, Mapping):
        return None, None, None
    human = _first_present(review, "human_verdict", "verdict", "decision", default=None)
    effective = _first_present(review, "effective_verdict", "effective_decision", default=None)
    return (
        str(human).lower() if human is not None else None,
        str(effective).lower() if effective is not None else None,
        review,
    )


def _project_execution_row(
    report: Mapping[str, Any],
    module: str,
    state: str,
    asset_row: Mapping[str, Any],
) -> dict[str, Any]:
    block = report.get(module)
    block_map = block if isinstance(block, Mapping) else {}
    runtime = block_map.get("runtime")
    runtime_map = runtime if isinstance(runtime, Mapping) else {}
    duration_sec = _first_present(
        runtime_map,
        block_map,
        "duration_sec",
        "duration_seconds",
        "duration",
        default=None,
    )
    duration_ms = _first_present(runtime_map, block_map, "duration_ms", default=None)
    runtime_errors = tuple(
        error
        for error in report.get("runtime_errors", [])
        if isinstance(error, Mapping) and error.get("module") == module
    )
    return {
        "asset_id": asset_row["asset_id"],
        "supplier_id": asset_row["supplier_id"],
        "profile": asset_row["profile"],
        "report_revision": asset_row["report_revision"],
        "module": module,
        "state": state,
        "duration_sec": duration_sec,
        "duration_ms": duration_ms,
        "duration": duration_sec,
        "continued_after_fail": bool(
            runtime_map.get("continued_after_fail")
            or block_map.get("continued_after_fail")
        ),
        "runtime_error": dict(runtime_errors[0]) if runtime_errors else None,
        "runtime_errors": runtime_errors,
    }


def project_quality_archive_review_rows(
    quality_archive: Path,
) -> list[dict[str, Any]]:
    """Project all queued warn issues from a quality archive.

    Projection is intentionally fail-closed: if a candidate issue reference
    is stale, duplicated, or points to an automatic fail, the source asset
    report path is included in the error and no partial batch result is
    returned.
    """

    rows: list[dict[str, Any]] = []
    for path, report in _iter_asset_reports_with_paths(quality_archive):
        try:
            rows.extend(project_warn_review_rows(report))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"cannot project QC report {path}: {exc}") from exc
    return rows


def project_warn_review_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Project ``manual_review.candidate_issue_ids`` into queue rows.

    Only warn issues explicitly selected by ``candidate_issue_ids`` are
    returned.  All-pass assets and automatic fails therefore produce no row;
    they are represented in the canonical report and later batch statistics.
    """

    if not isinstance(report, Mapping):
        raise ValueError("asset QC report must be an object")
    issues_raw = report.get("issues")
    manual_review = report.get("manual_review")
    if not isinstance(issues_raw, Sequence) or isinstance(issues_raw, (str, bytes)):
        raise ValueError("issues must be an array")
    if not isinstance(manual_review, Mapping):
        raise ValueError("manual_review must be an object")

    issues: dict[str, Mapping[str, Any]] = {}
    for index, raw_issue in enumerate(issues_raw):
        if not isinstance(raw_issue, Mapping):
            raise ValueError(f"issues[{index}] must be an object")
        issue_id = raw_issue.get("issue_id")
        if not isinstance(issue_id, str) or not issue_id:
            raise ValueError(f"issues[{index}].issue_id must be a non-empty string")
        if issue_id in issues:
            raise ValueError(f"duplicate issue_id: {issue_id}")
        issues[issue_id] = raw_issue

    candidate_ids = manual_review.get("candidate_issue_ids")
    if not isinstance(candidate_ids, Sequence) or isinstance(candidate_ids, (str, bytes)):
        raise ValueError("manual_review.candidate_issue_ids must be an array")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_issue_id in candidate_ids:
        if not isinstance(raw_issue_id, str) or not raw_issue_id:
            raise ValueError(
                "manual_review.candidate_issue_ids must contain non-empty strings"
            )
        if raw_issue_id in seen:
            raise ValueError(
                f"manual_review.candidate_issue_ids contains duplicate issue_id: {raw_issue_id}"
            )
        seen.add(raw_issue_id)
        issue = issues.get(raw_issue_id)
        if issue is None:
            raise ValueError(
                f"manual_review.candidate_issue_ids references missing issue: {raw_issue_id}"
            )
        if issue.get("severity") != "warn":
            raise ValueError(
                f"manual candidate {raw_issue_id} severity must be warn, "
                f"got {issue.get('severity')}"
            )
        if issue.get("needs_manual_review") is not True:
            raise ValueError(
                f"manual candidate {raw_issue_id} needs_manual_review must be true"
            )
        rows.append(_review_row_from_issue(report, issue))
    return rows


def _review_row_from_issue(
    report: Mapping[str, Any],
    issue: Mapping[str, Any],
) -> dict[str, Any]:
    issue_id = str(issue["issue_id"])
    module = str(issue.get("module") or "unknown")
    context = issue.get("context")
    context_map = context if isinstance(context, Mapping) else {}
    module_block = report.get(module)
    module_map = module_block if isinstance(module_block, Mapping) else {}
    metrics = module_map.get("metrics")
    metrics_map = dict(metrics) if isinstance(metrics, Mapping) else {}

    evidence_map: dict[str, Mapping[str, Any]] = {}
    evidence = module_map.get("evidence")
    if isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes)):
        for item in evidence:
            if isinstance(item, Mapping) and isinstance(item.get("evidence_id"), str):
                evidence_map[str(item["evidence_id"])] = item

    evidence_item: Mapping[str, Any] = {}
    evidence_ids = issue.get("evidence_ids")
    if isinstance(evidence_ids, Sequence) and not isinstance(evidence_ids, (str, bytes)):
        for evidence_id in evidence_ids:
            if isinstance(evidence_id, str) and evidence_id in evidence_map:
                evidence_item = evidence_map[evidence_id]
                break

    start = _frame_value(
        _first_present(
            context_map,
            evidence_item,
            "start_frame",
            "window_start_frame",
            default=None,
        )
    )
    end = _frame_value(
        _first_present(
            context_map,
            evidence_item,
            "end_frame",
            "window_end_frame",
            default=start,
        )
    )
    representative = _frame_value(
        _first_present(
            context_map,
            evidence_item,
            "peak_frame",
            "representative_frame",
            default=_midpoint(start, end),
        )
    )
    hand_side = _first_present(context_map, evidence_item, "hand_side", default="")
    reason = _first_present(
        context_map,
        issue,
        "reason",
        default=str(issue.get("code") or issue.get("issue_type") or "manual review"),
    )
    suggested_issue_type = str(
        issue.get("issue_type") or issue.get("code") or "unknown"
    )
    severity_suggestion = _severity_suggestion(issue, context_map)
    priority = _normalize_priority(
        _first_present(context_map, issue, "priority", default=severity_suggestion),
        fallback=severity_suggestion,
    )
    metrics_map.update(_issue_metric_context(issue))

    return {
        # ``issue_id`` is the stable join key used by later manual-label
        # mutation.  Keep ``review_id`` equal to it; the legacy path generates
        # synthetic IDs for sidecar rows only.
        "review_id": issue_id,
        "issue_id": issue_id,
        "supplier_id": _supplier_id(report),
        "asset_id": str(report.get("asset_id") or ""),
        "window_start_frame": _empty_if_none(start),
        "window_end_frame": _empty_if_none(end),
        "representative_frame": _empty_if_none(representative),
        "source_level": str(
            _first_present(context_map, issue, "source_level", default="window" if start is not None else "asset")
        ),
        "module": module,
        "rule_id": str(issue.get("rule_id") or ""),
        "hand_side": str(hand_side or ""),
        "auto_verdict": "warn",
        "suggested_issue_type": suggested_issue_type,
        "severity_suggestion": severity_suggestion,
        "priority": priority,
        "key_metrics_json": json.dumps(metrics_map, ensure_ascii=False, sort_keys=True),
        "reason": str(reason),
        "evidence_path": _path_value(evidence_item.get("path")),
        "overlay_path": _path_value(evidence_item.get("overlay_path")),
        "display_overlay_path": "",
        "needs_manual_review": True,
        "sam3_containment_eligible": _bool_or_none(
            _first_present(
                context_map,
                evidence_item,
                "sam3_containment_eligible",
                default=None,
            )
        ),
    }


def _supplier_id(report: Mapping[str, Any]) -> str:
    value = report.get("supplier_id")
    if value is not None:
        supplier_id = str(value).strip()
        if supplier_id:
            return supplier_id
    metadata = report.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("supplier_id", "supplier"):
            value = metadata.get(key)
            if value is None:
                continue
            supplier_id = str(value).strip()
            if supplier_id:
                return supplier_id
    return "unknown"


def _issue_metric_context(issue: Mapping[str, Any]) -> dict[str, Any]:
    """Add scalar issue measurements without replacing module metrics."""

    output: dict[str, Any] = {}
    for source, target in (
        ("observed_value", "issue_observed_value"),
        ("boundary_value", "issue_boundary_value"),
        ("operator", "issue_operator"),
    ):
        value = issue.get(source)
        if value is not None:
            output[target] = value
    return output


def _severity_suggestion(issue: Mapping[str, Any], context: Mapping[str, Any]) -> str:
    value = _first_present(context, issue, "severity_suggestion", "severity", default="medium")
    text = str(value or "").strip().lower()
    return text if text in _SEVERITIES else "medium"


def _normalize_priority(value: Any, *, fallback: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in _SEVERITIES or text == "pass_sample" else fallback


def _first_present(*sources: Mapping[str, Any] | str, default: Any = None) -> Any:
    # Calls use ``_first_present(context, evidence, "start_frame", "end_frame")``.
    # Split trailing key names from the mapping sources while keeping the
    # helper independent from the legacy ledger's permissive scalar helpers.
    if not sources:
        return default
    split_at = len(sources)
    while split_at and isinstance(sources[split_at - 1], str):
        split_at -= 1
    if split_at == len(sources):
        raise TypeError("_first_present requires at least one key")
    mappings = sources[:split_at]
    key_names = tuple(sources[split_at:])
    for mapping in mappings:
        if not isinstance(mapping, Mapping):
            continue
        for key in key_names:
            value = mapping.get(key)
            if value is not None and value != "":
                return value
    return default


def _frame_value(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _midpoint(start: int | None, end: int | None) -> int | None:
    if start is None and end is None:
        return None
    if start is None:
        return end
    if end is None:
        return start
    return int(round((start + end) / 2))


def _empty_if_none(value: Any) -> Any:
    return "" if value is None else value


def _path_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"evidence path must be relative to batch root: {text}")
    return text


def _bool_or_none(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
    return bool(value)
