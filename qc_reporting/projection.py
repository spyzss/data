from __future__ import annotations

"""Project canonical asset QC reports into human-review queue rows.

``quality_archive/*.json`` is the authoritative input for the formal manual
queue.  Legacy candidate-window/SAM3/video sidecars are deliberately handled
by the legacy path in :mod:`tools.build_manual_review_queue` and are not read
here.
"""

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from qc_common.schema import validate_asset_qc_report


_SEVERITIES = {"low", "medium", "high", "critical"}


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
