"""Adapt SAM3 containment window summaries to the unified QC contract."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import mimetypes
from numbers import Integral
from pathlib import Path
from typing import Any, cast

from qc_common.config import LoadedQcConfig
from qc_common.contracts import (
    EvidenceRef,
    Issue,
    ModuleResult,
    Verdict,
    build_issue_id,
    relative_evidence_path,
)
from qc_common.report_mutation import apply_module_result
from qc_pipeline.context import AssetContext


_MODULE = "sam3_containment"
_COORDINATE_SYSTEM = "source_inclusive"
_VERDICT_ORDER = {"pass": 0, "warn": 1, "fail": 2}

SAM3_VERDICT: dict[str, tuple[Verdict, str | None]] = {
    "pass": ("pass", None),
    "acceptable": ("pass", None),
    "side_view_manual_review": ("warn", f"{_MODULE}.side_view_manual_review"),
    "projection_review": ("warn", "sam3_containment.projection_review"),
    "strong_containment_mismatch": ("fail", f"{_MODULE}.strong_containment_mismatch"),
}

_LEGACY_VERDICT = {
    "likely_visible_ok": "pass",
    "acceptable_flagged": "acceptable",
    "containment_fail": "strong_containment_mismatch",
    "rotation_manual_review": "side_view_manual_review",
    "mixed_review": "projection_review",
    "containment_review": "projection_review",
    "mask_missing_or_tiny_review": "projection_review",
    "review": "projection_review",
}
_CONTEXT_FIELDS = (
    "candidate_start_frame",
    "candidate_end_frame",
    "candidate_hand_side",
    "clip_start_frame",
    "clip_end_frame",
    "coordinate_space",
    "roi",
    "roi_id",
    "source_review_type",
    "source_trigger_reason",
    "source_priority",
    "source_window_source",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _integer(value: Any, name: str) -> int:
    if value is None:
        raise ValueError(f"missing {name}")
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed


def _window(row: Mapping[str, Any]) -> tuple[int, int]:
    start = _integer(row.get("window_start_frame"), "window_start_frame")
    end = _integer(row.get("window_end_frame"), "window_end_frame")
    if end < start:
        raise ValueError("window_end_frame must be >= window_start_frame")
    return start, end


def _hand_side(value: Any) -> str | None:
    if value is None:
        return None
    side = str(value).strip().lower()
    if side not in {"left", "right", "both"}:
        raise ValueError(f"invalid hand_side: {value!r}")
    return side


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _evidence_ref(
    *,
    asset_id: str,
    batch_root: Path,
    row: Mapping[str, Any],
) -> EvidenceRef:
    start_frame, end_frame = _window(row)
    hand_side = _hand_side(row.get("hand_side"))
    source_path = row.get("source_path")
    if not isinstance(source_path, str) or not source_path.strip():
        raise ValueError("evidence source_path must be a non-empty string")
    candidate = Path(source_path)
    absolute_path = candidate if candidate.is_absolute() else batch_root / candidate
    relative_path = relative_evidence_path(absolute_path, batch_root)
    if not absolute_path.is_file():
        raise ValueError(f"evidence file does not exist: {absolute_path}")
    kind = row.get("evidence_type")
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("evidence_type must be a non-empty string")
    mime_type, _encoding = mimetypes.guess_type(absolute_path.name)
    return EvidenceRef(
        evidence_id=build_issue_id(
            asset_id=asset_id,
            module=_MODULE,
            rule_id=f"{_MODULE}.evidence",
            source_relative_path=relative_path,
            coordinate_system=_COORDINATE_SYSTEM,
            start_frame=start_frame,
            end_frame=end_frame,
            hand_side=hand_side,
            evidence_kind=kind,
        ),
        kind=kind,
        path=relative_path,
        coordinate_system=_COORDINATE_SYSTEM,
        start_frame=start_frame,
        end_frame=end_frame,
        hand_side=hand_side,
        checksum=_sha256(absolute_path),
        mime_type=mime_type,
    )


def _canonical_verdict(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"window summary has no valid {field}")
    canonical = _LEGACY_VERDICT.get(value, value)
    if canonical not in SAM3_VERDICT:
        raise ValueError(f"unsupported SAM3 containment verdict: {value!r}")
    return canonical


def _normalized_verdict(row: Mapping[str, Any]) -> str:
    authoritative = "window_containment_verdict"
    legacy = "containment_verdict"
    if authoritative in row:
        canonical = _canonical_verdict(row[authoritative], authoritative)
        if legacy in row:
            legacy_canonical = _canonical_verdict(row[legacy], legacy)
            if legacy_canonical != canonical:
                raise ValueError(
                    "contradictory SAM3 containment verdict fields: "
                    f"{canonical!r} != {legacy_canonical!r}"
                )
        return canonical
    if legacy in row:
        return _canonical_verdict(row[legacy], legacy)
    raise ValueError("window summary has no containment verdict")


def _normalized_summaries(
    asset_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], str]]:
    normalized: dict[
        tuple[str, int, int, str | None],
        tuple[Mapping[str, Any], str],
    ] = {}
    for row in rows:
        if row.get("asset_id") != asset_id:
            continue
        start_frame, end_frame = _window(row)
        hand_side = _hand_side(row.get("hand_side"))
        verdict = _normalized_verdict(row)
        key = (asset_id, start_frame, end_frame, hand_side)
        previous = normalized.get(key)
        if previous is not None:
            if previous[1] != verdict:
                raise ValueError(
                    "conflicting SAM3 containment verdicts for "
                    f"{key}: {previous[1]!r} != {verdict!r}"
                )
            continue
        normalized[key] = (row, verdict)
    return [normalized[key] for key in sorted(normalized, key=repr)]


def _matching_evidence_ids(
    *,
    start_frame: int | None,
    end_frame: int | None,
    hand_side: str | None,
    evidence: Sequence[EvidenceRef],
) -> tuple[str, ...]:
    return tuple(
        item.evidence_id
        for item in evidence
        if item.start_frame == start_frame
        and item.end_frame == end_frame
        and (
            hand_side is None
            or item.hand_side is None
            or item.hand_side == "both"
            or item.hand_side == hand_side
        )
    )


def _issue(
    *,
    asset_id: str,
    row: Mapping[str, Any],
    verdict: Verdict,
    rule_id: str,
    evidence_ids: tuple[str, ...],
    config: LoadedQcConfig,
) -> Issue:
    issue_type = rule_id.rsplit(".", 1)[-1]
    configured_rule = config.module_rules(_MODULE).get(issue_type)
    if (
        not isinstance(configured_rule, Mapping)
        or configured_rule.get("rule_id") != rule_id
    ):
        raise ValueError(f"missing configured rule: {rule_id}")
    configured_verdict = configured_rule.get("verdict")
    if configured_verdict != verdict:
        raise ValueError(
            f"configured rule verdict mismatch for {rule_id}: "
            f"{configured_verdict!r} != {verdict!r}"
        )
    start_frame, end_frame = _window(row)
    hand_side = _hand_side(row.get("hand_side"))
    observed = row.get(
        "inside_ratio",
        row.get("inside_ratio_mean", row.get("window_containment_verdict")),
    )
    is_strong = issue_type == "strong_containment_mismatch"
    boundary = (
        config.module_parameters(_MODULE).get("inside_ratio_fail")
        if is_strong
        else None
    )
    context = {
        "coordinate_system": _COORDINATE_SYSTEM,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "hand_side": hand_side,
    }
    for name in _CONTEXT_FIELDS:
        if name in row:
            context[name] = row[name]
    return Issue(
        issue_id=build_issue_id(
            asset_id=asset_id,
            module=_MODULE,
            rule_id=rule_id,
            source_relative_path="window_keypoint_containment_summary.json",
            coordinate_system=_COORDINATE_SYSTEM,
            start_frame=start_frame,
            end_frame=end_frame,
            hand_side=hand_side,
            evidence_kind="window_summary",
        ),
        code=issue_type,
        severity=cast(Any, verdict),
        module=_MODULE,
        issue_type=issue_type,
        metric="inside_ratio",
        observed_value=observed,
        operator="<=" if is_strong else "triggered",
        boundary_value=boundary,
        rule_id=rule_id,
        needs_manual_review=verdict == "warn",
        context=context,
        evidence_ids=evidence_ids,
    )


def adapt_sam3_containment(
    *,
    asset_id: str,
    batch_root: Path,
    window_summaries: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
    config: LoadedQcConfig,
) -> ModuleResult:
    """Translate one asset's structured SAM3 sidecars without rerunning QC."""
    batch_root = Path(batch_root)
    summaries = _normalized_summaries(asset_id, window_summaries)
    evidence = tuple(
        _evidence_ref(asset_id=asset_id, batch_root=batch_root, row=row)
        for row in evidence_rows
        if row.get("asset_id") == asset_id
    )

    issues: list[Issue] = []
    issue_ids: set[str] = set()
    verdicts: list[Verdict] = []
    verdict_counts: Counter[str] = Counter()
    for row, normalized in summaries:
        verdict, rule_id = SAM3_VERDICT[normalized]
        verdicts.append(verdict)
        verdict_counts[normalized] += 1
        if rule_id is None:
            continue
        start_frame, end_frame = _window(row)
        hand_side = _hand_side(row.get("hand_side"))
        issue = _issue(
            asset_id=asset_id,
            row=row,
            verdict=verdict,
            rule_id=rule_id,
            evidence_ids=_matching_evidence_ids(
                start_frame=start_frame,
                end_frame=end_frame,
                hand_side=hand_side,
                evidence=evidence,
            ),
            config=config,
        )
        if issue.issue_id not in issue_ids:
            issue_ids.add(issue.issue_id)
            issues.append(issue)

    verdict: Verdict = (
        max(verdicts, key=_VERDICT_ORDER.__getitem__) if verdicts else "skipped"
    )
    return ModuleResult(
        module=_MODULE,
        verdict=verdict,
        evaluation={
            "decision": verdict,
            "window_count": len(summaries),
            "issue_ids": [issue.issue_id for issue in issues],
        },
        metrics={
            "window_count": len(summaries),
            "window_verdict_counts": dict(sorted(verdict_counts.items())),
            "evidence_count": len(evidence),
        },
        issues=tuple(issues),
        evidence=evidence,
        runtime={"source": "manifest_sam3_containment"},
    )


def write_sam3_asset_result(
    *,
    context: AssetContext,
    window_summaries: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
    config: LoadedQcConfig,
    profile: str,
    expected_revision: int,
    next_module: str | None,
) -> dict[str, Any]:
    """Commit one SAM3 result through the shared revision-aware transaction."""
    result = adapt_sam3_containment(
        asset_id=context.asset_id,
        batch_root=context.batch_root,
        window_summaries=window_summaries,
        evidence_rows=evidence_rows,
        config=config,
    )
    return apply_module_result(
        context.report_path,
        context=context,
        config=config,
        profile=profile,
        result=result,
        expected_revision=expected_revision,
        next_module=next_module,
        now=_utc_now(),
    )
