"""One-time migration of legacy manual-review exports into asset QC reports.

The importer deliberately recognizes only exact ``asset_id`` and ``issue_id``
identities.  A generated legacy ``review_id`` is accepted only when an
explicit asset/review/issue mapping is supplied; frame windows, row order,
comments, and progress JSON are never used to infer an identity.  Once
imported, the asset report is the only authoritative state read by the
workbench.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qc_common.manual_review import mark_semantic_skipped_due_to_fail
from qc_common.report import load_asset_qc_report

from .report_updates import reduce_overall_decision, update_human_state
from .warn_service import effective_issue_verdict


_PASS_OUTCOMES = frozenset({"false_positive", "acceptable_flagged"})
_FAIL_OUTCOMES = frozenset({"true_positive", "partial", "false_negative"})
_LEGACY_FIELDS = (
    "review_id",
    "segment_id",
    "manual_outcome",
    "failure_mode",
    "severity",
    "confidence",
    "acceptance_status",
    "comment",
    "reviewer",
)


@dataclass(frozen=True)
class ImportProblem:
    """One row that could not be mapped without making an inference."""

    row_number: int
    code: str
    message: str
    asset_id: str | None = None
    issue_id: str | None = None


@dataclass(frozen=True)
class ImportResult:
    """Machine-readable summary suitable for dry-run and CLI output."""

    report_path: Path
    asset_id: str
    report_revision: int
    matched_count: int
    unmatched_count: int
    conflict_count: int
    idempotent_count: int
    matched_issue_ids: tuple[str, ...]
    problems: tuple[ImportProblem, ...]
    written: bool
    dry_run: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_path": str(self.report_path),
            "asset_id": self.asset_id,
            "report_revision": self.report_revision,
            "matched": self.matched_count,
            "unmatched": self.unmatched_count,
            "conflicts": self.conflict_count,
            "idempotent": self.idempotent_count,
            "matched_issue_ids": list(self.matched_issue_ids),
            "written": self.written,
            "dry_run": self.dry_run,
            "problems": [
                {
                    "row_number": problem.row_number,
                    "code": problem.code,
                    "message": problem.message,
                    "asset_id": problem.asset_id,
                    "issue_id": problem.issue_id,
                }
                for problem in self.problems
            ],
        }


@dataclass(frozen=True)
class _MatchedRow:
    row_number: int
    issue_id: str
    verdict: str
    reason: str | None
    legacy: Mapping[str, str]


def _clean(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _read_source_bytes(path: Path) -> bytes:
    """Read one immutable source snapshot used for both parsing and hashing."""

    return Path(path).read_bytes()


def _sha256_bytes(value: bytes | None) -> str | None:
    return None if value is None else "sha256:" + hashlib.sha256(value).hexdigest()


def _read_rows(csv_bytes: bytes) -> tuple[list[tuple[int, dict[str, str]]], list[str]]:
    try:
        text = csv_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"legacy CSV must be UTF-8: {exc}") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    fields = list(reader.fieldnames or [])
    if "asset_id" not in fields:
        raise ValueError("legacy CSV must contain asset_id")
    if "issue_id" not in fields and "review_id" not in fields:
        raise ValueError("legacy CSV must contain issue_id or review_id")
    rows = [
        (row_number, {str(key): value or "" for key, value in row.items() if key is not None})
        for row_number, row in enumerate(reader, start=2)
    ]
    return rows, fields


def _read_issue_mapping(
    mapping_bytes: bytes | None,
) -> tuple[dict[tuple[str, str], str], set[tuple[str, str]]]:
    if mapping_bytes is None:
        return {}, set()
    try:
        text = mapping_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"issue mapping CSV must be UTF-8: {exc}") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    required = {"asset_id", "review_id", "issue_id"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError("issue mapping CSV must contain asset_id, review_id, issue_id")
    grouped: dict[tuple[str, str], list[str]] = {}
    for row_number, row in enumerate(reader, start=2):
        asset_id = _clean(row.get("asset_id"))
        review_id = _clean(row.get("review_id"))
        issue_id = _clean(row.get("issue_id"))
        if not asset_id or not review_id or not issue_id:
            raise ValueError(f"issue mapping row {row_number} has an empty identity")
        grouped.setdefault((asset_id, review_id), []).append(issue_id)
    ambiguous = {key for key, values in grouped.items() if len(values) != 1}
    unique = {key: values[0] for key, values in grouped.items() if len(values) == 1}
    return unique, ambiguous


def _row_issue_id(
    row: Mapping[str, str],
    issue_mapping: Mapping[tuple[str, str], str],
    ambiguous_mapping: set[tuple[str, str]],
) -> tuple[str | None, str | None, str | None]:
    issue_id = _clean(row.get("issue_id"))
    review_id = _clean(row.get("review_id"))
    if issue_id:
        return issue_id, None, None
    if not review_id:
        return None, "missing_issue_identity", "issue_id is empty"
    key = (_clean(row.get("asset_id")), review_id)
    if key in ambiguous_mapping:
        return (
            None,
            "ambiguous_issue_mapping",
            f"multiple mapping rows exist for asset_id={key[0]!r}, review_id={review_id!r}",
        )
    mapped = issue_mapping.get(key)
    if mapped is None:
        return (
            None,
            "review_id_requires_mapping",
            f"generated review_id={review_id!r} requires an exact asset/review/issue mapping",
        )
    return mapped, None, None


def _row_verdict(row: Mapping[str, str]) -> tuple[str | None, str | None]:
    direct = _clean(row.get("verdict") or row.get("manual_verdict")).lower()
    if direct:
        if direct in {"pass", "fail"}:
            return direct, None
        return None, f"unsupported verdict {direct!r}"
    outcome = _clean(row.get("manual_outcome")).lower()
    if outcome in _PASS_OUTCOMES:
        return "pass", None
    if outcome in _FAIL_OUTCOMES:
        return "fail", None
    if not outcome:
        return None, "manual_outcome/verdict is empty"
    return None, f"manual_outcome {outcome!r} is not a final pass/fail decision"


def _issue_index(report: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], set[str]]:
    issues = report.get("issues", [])
    if isinstance(issues, (str, bytes, bytearray)) or not isinstance(issues, Sequence):
        raise ValueError("report issues must be a sequence")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for issue in issues:
        if not isinstance(issue, Mapping):
            continue
        issue_id = issue.get("issue_id")
        if isinstance(issue_id, str) and issue_id:
            grouped.setdefault(issue_id, []).append(issue)
    unique = {issue_id: values[0] for issue_id, values in grouped.items() if len(values) == 1}
    ambiguous = {issue_id for issue_id, values in grouped.items() if len(values) > 1}
    return unique, ambiguous


def _review_signature(review: Mapping[str, Any]) -> tuple[object, ...]:
    legacy = review.get("legacy")
    legacy_fields = (
        tuple(sorted((str(key), str(value)) for key, value in legacy.items()))
        if isinstance(legacy, Mapping)
        else None
    )
    return (
        review.get("verdict"),
        review.get("reason"),
        review.get("reviewer"),
        review.get("source"),
        legacy_fields,
    )


def _planned_review(
    row: _MatchedRow,
    issue: Mapping[str, Any],
    reviewer: str,
    reviewed_at: str,
) -> dict[str, Any]:
    machine_verdict = effective_issue_verdict(issue, None)
    return {
        "verdict": row.verdict,
        "effective_verdict": effective_issue_verdict(issue, {"verdict": row.verdict}),
        "machine_verdict": machine_verdict,
        "reason": row.reason,
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
        "source": "legacy_manual_review_import",
        "legacy": dict(row.legacy),
    }


def import_legacy_manual_review(
    report_path: Path,
    csv_path: Path,
    progress_path: Path | None,
    expected_revision: int,
    reviewer: str,
    *,
    dry_run: bool = False,
    issue_mapping_path: Path | None = None,
    asset_scope_only: bool = False,
    _csv_bytes: bytes | None = None,
    _progress_bytes: bytes | None = None,
    _mapping_bytes: bytes | None = None,
) -> ImportResult:
    """Import exact legacy decisions into one report, or report a dry run.

    Valid rows may be imported even when unrelated rows are unmatched.  Rows
    with duplicate or ambiguous identities are always excluded.  Existing
    different human decisions are conflicts and are never overwritten.
    """

    path = Path(report_path)
    csv_file = Path(csv_path)
    progress_file = Path(progress_path) if progress_path is not None else None
    mapping_file = Path(issue_mapping_path) if issue_mapping_path is not None else None
    if isinstance(expected_revision, bool) or not isinstance(expected_revision, int):
        raise TypeError("expected_revision must be an integer")
    reviewer = _clean(reviewer)
    if not reviewer:
        raise ValueError("reviewer must be a non-empty string")
    csv_bytes = _read_source_bytes(csv_file) if _csv_bytes is None else _csv_bytes
    progress_bytes = (
        _read_source_bytes(progress_file)
        if progress_file is not None and _progress_bytes is None
        else _progress_bytes
    )
    mapping_bytes = (
        _read_source_bytes(mapping_file)
        if mapping_file is not None and _mapping_bytes is None
        else _mapping_bytes
    )
    if progress_bytes is not None:
        # Parse only to reject corrupt migration input.  Its contents never
        # supply an identity or verdict.
        try:
            value = json.loads(progress_bytes.decode("utf-8-sig"))
        except UnicodeDecodeError as exc:
            raise ValueError(f"progress JSON must be UTF-8: {exc}") from exc
        if not isinstance(value, (dict, list)):
            raise ValueError("progress JSON must contain an object or array")

    report = load_asset_qc_report(path)
    if report is None:
        raise FileNotFoundError(path)
    if report.get("schema_version") != "asset_qc_report.v2":
        raise ValueError("legacy import requires an asset_qc_report.v2 report")
    asset_id = report.get("asset_id")
    if not isinstance(asset_id, str) or not asset_id:
        raise ValueError("report asset_id must be a non-empty string")
    current_revision = int(report.get("report_revision", 0))
    rows, _fields = _read_rows(csv_bytes)
    issue_mapping, ambiguous_mapping = _read_issue_mapping(mapping_bytes)
    issues, ambiguous_issues = _issue_index(report)

    manual = report.get("manual_review")
    if not isinstance(manual, Mapping):
        raise ValueError("report manual_review must be an object")
    candidates_value = manual.get("candidate_issue_ids", [])
    if isinstance(candidates_value, (str, bytes, bytearray)) or not isinstance(
        candidates_value, Sequence
    ):
        raise ValueError("manual_review.candidate_issue_ids must be a sequence")
    candidate_ids = {item for item in candidates_value if isinstance(item, str)}
    existing_reviews = manual.get("issue_reviews", {})
    if not isinstance(existing_reviews, Mapping):
        raise ValueError("manual_review.issue_reviews must be an object")

    resolved_ids: dict[int, str] = {}
    identity_errors: dict[int, tuple[str, str]] = {}
    keys: list[tuple[str, str]] = []
    for row_number, row in rows:
        issue_id, error_code, error_message = _row_issue_id(
            row, issue_mapping, ambiguous_mapping
        )
        if error_code is not None:
            assert error_message is not None
            identity_errors[row_number] = (error_code, error_message)
            continue
        assert issue_id is not None
        resolved_ids[row_number] = issue_id
        keys.append((_clean(row.get("asset_id")), issue_id))
    duplicate_keys = {key for key, count in Counter(keys).items() if count > 1}

    problems: list[ImportProblem] = []
    matched: list[_MatchedRow] = []
    unmatched_count = 0
    conflict_count = 0
    for row_number, row in rows:
        row_asset_id = _clean(row.get("asset_id"))
        issue_id = resolved_ids.get(row_number)
        if row_asset_id != asset_id:
            if asset_scope_only:
                continue
            unmatched_count += 1
            problems.append(
                ImportProblem(
                    row_number,
                    "asset_id_mismatch",
                    f"row asset_id {row_asset_id!r} does not match report asset_id {asset_id!r}",
                    row_asset_id or None,
                    issue_id,
                )
            )
            continue
        if issue_id is None:
            conflict_count += 1
            error_code, error_message = identity_errors[row_number]
            problems.append(
                ImportProblem(
                    row_number,
                    error_code,
                    error_message,
                    asset_id,
                    None,
                )
            )
            continue
        if (row_asset_id, issue_id) in duplicate_keys:
            conflict_count += 1
            problems.append(
                ImportProblem(
                    row_number,
                    "duplicate_csv_key",
                    f"multiple CSV rows have asset_id={asset_id!r}, issue_id={issue_id!r}",
                    asset_id,
                    issue_id,
                )
            )
            continue
        if issue_id in ambiguous_issues:
            conflict_count += 1
            problems.append(
                ImportProblem(
                    row_number,
                    "ambiguous_report_issue_id",
                    f"report contains multiple issues with issue_id={issue_id!r}",
                    asset_id,
                    issue_id,
                )
            )
            continue
        if issue_id not in issues:
            unmatched_count += 1
            problems.append(
                ImportProblem(
                    row_number,
                    "unknown_issue_id",
                    f"report has no issue_id={issue_id!r}",
                    asset_id,
                    issue_id,
                )
            )
            continue
        if issue_id not in candidate_ids:
            conflict_count += 1
            problems.append(
                ImportProblem(
                    row_number,
                    "issue_not_manual_candidate",
                    f"issue_id={issue_id!r} is not a manual-review candidate",
                    asset_id,
                    issue_id,
                )
            )
            continue
        verdict, verdict_error = _row_verdict(row)
        if verdict_error is not None:
            conflict_count += 1
            problems.append(
                ImportProblem(
                    row_number,
                    "invalid_legacy_verdict",
                    verdict_error,
                    asset_id,
                    issue_id,
                )
            )
            continue
        assert verdict is not None
        reason = _clean(row.get("comment")) or _clean(row.get("reason")) or None
        matched.append(
            _MatchedRow(
                row_number=row_number,
                issue_id=issue_id,
                verdict=verdict,
                reason=reason,
                legacy={field: _clean(row.get(field)) for field in _LEGACY_FIELDS},
            )
        )

    reviewed_at = datetime.now(timezone.utc).isoformat()
    to_write: list[tuple[_MatchedRow, dict[str, Any]]] = []
    idempotent_count = 0
    idempotent_issue_ids: list[str] = []
    for row in matched:
        planned = _planned_review(row, issues[row.issue_id], reviewer, reviewed_at)
        existing = existing_reviews.get(row.issue_id)
        if isinstance(existing, Mapping):
            if _review_signature(existing) == _review_signature(planned):
                idempotent_count += 1
                idempotent_issue_ids.append(row.issue_id)
                continue
            conflict_count += 1
            problems.append(
                ImportProblem(
                    row.row_number,
                    "existing_review_differs",
                    f"report already contains a different review for issue_id={row.issue_id!r}",
                    asset_id,
                    row.issue_id,
                )
            )
            continue
        to_write.append((row, planned))

    successful_issue_ids = {
        row.issue_id for row, _review in to_write
    } | set(idempotent_issue_ids)
    matched_issue_ids = tuple(
        row.issue_id for row in matched if row.issue_id in successful_issue_ids
    )
    if dry_run or not to_write:
        return ImportResult(
            report_path=path,
            asset_id=asset_id,
            report_revision=current_revision,
            matched_count=len(to_write) + idempotent_count,
            unmatched_count=unmatched_count,
            conflict_count=conflict_count,
            idempotent_count=idempotent_count,
            matched_issue_ids=matched_issue_ids,
            problems=tuple(problems),
            written=False,
            dry_run=dry_run,
        )

    csv_hash = _sha256_bytes(csv_bytes)
    progress_hash = _sha256_bytes(progress_bytes)
    mapping_hash = _sha256_bytes(mapping_bytes)

    def mutate(candidate: dict[str, Any]) -> None:
        block = candidate.get("manual_review")
        if not isinstance(block, dict):
            raise ValueError("report manual_review must be an object")
        selected = block.setdefault("selected_issue_ids", [])
        reviews = block.setdefault("issue_reviews", {})
        if not isinstance(selected, list) or not isinstance(reviews, dict):
            raise ValueError("manual_review selected IDs/reviews have invalid types")
        for row, review in to_write:
            if row.issue_id not in selected:
                selected.append(row.issue_id)
            reviews[row.issue_id] = deepcopy(review)
        block["selected_issue_id"] = selected[0] if selected else None
        all_reviewed = bool(selected) and set(selected).issubset(reviews)
        block["state"] = "completed" if all_reviewed else "in_progress"
        block["completed_at"] = reviewed_at if all_reviewed else None
        if all_reviewed:
            has_fail = any(
                isinstance(reviews.get(issue_id), Mapping)
                and reviews[issue_id].get("verdict") == "fail"
                for issue_id in selected
            )
            block["completion_mode"] = "early_fail" if has_fail else "all_reviewed"
            block["failure_reason"] = None
        else:
            block.pop("completion_mode", None)
            block.pop("failure_reason", None)
        audit = block.setdefault("import_audit", [])
        if not isinstance(audit, list):
            raise ValueError("manual_review.import_audit must be a list")
        audit.append(
            {
                "action": "legacy_manual_review_import",
                "reviewer": reviewer,
                "imported_at": reviewed_at,
                "csv_path": str(csv_file),
                "csv_sha256": csv_hash,
                "progress_path": str(progress_file) if progress_file is not None else None,
                "progress_sha256": progress_hash,
                "issue_mapping_path": str(mapping_file) if mapping_file is not None else None,
                "issue_mapping_sha256": mapping_hash,
                "issue_ids": [row.issue_id for row, _review in to_write],
                "row_numbers": [row.row_number for row, _review in to_write],
            }
        )
        if all_reviewed:
            pipeline = candidate.get("pipeline_state")
            if isinstance(pipeline, dict) and pipeline.get("next_module") == "manual_review":
                semantic = candidate.get("semantic_calibration")
                semantic_completed = (
                    isinstance(semantic, dict) and semantic.get("state") == "completed"
                )
                if has_fail:
                    pipeline.update(
                        {
                            "status": "stopped",
                            "last_completed_module": "manual_review",
                            "next_module": None,
                            "stop_reason": "manual_review_failed",
                        }
                    )
                    if not semantic_completed:
                        mark_semantic_skipped_due_to_fail(candidate)
                else:
                    pipeline.update(
                        {
                            "status": (
                                "completed" if semantic_completed else "awaiting_external"
                            ),
                            "last_completed_module": "manual_review",
                            "next_module": (
                                None if semantic_completed else "semantic_consistency"
                            ),
                            "stop_reason": None,
                        }
                    )
            execution = candidate.get("execution")
            if isinstance(execution, dict):
                execution["updated_at"] = reviewed_at
            candidate["overall_decision"] = reduce_overall_decision(candidate)

    updated = update_human_state(path, expected_revision, mutate)
    return ImportResult(
        report_path=path,
        asset_id=asset_id,
        report_revision=int(updated["report_revision"]),
        matched_count=len(to_write) + idempotent_count,
        unmatched_count=unmatched_count,
        conflict_count=conflict_count,
        idempotent_count=idempotent_count,
        matched_issue_ids=matched_issue_ids,
        problems=tuple(problems),
        written=True,
        dry_run=False,
    )


__all__ = [
    "ImportProblem",
    "ImportResult",
    "import_legacy_manual_review",
]
