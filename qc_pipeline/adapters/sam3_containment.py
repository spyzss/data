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

import numpy as np

from canonical_qc.hand_quality import (
    SupplierAgreementResult,
    compare_supplier_and_machine,
)
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

_FRAME_PASS = frozenset({"pass", "acceptable", "likely_visible_ok"})
_FRAME_FAIL = frozenset({"strong_containment_mismatch", "containment_fail"})


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


def _frame_machine_status(row: Mapping[str, Any]) -> str:
    value = row.get("containment_verdict")
    if value is None:
        return "unavailable"
    if not isinstance(value, str) or not value:
        raise ValueError("frame containment_verdict must be a non-empty string")
    if value in _FRAME_PASS:
        return "pass"
    if value in _FRAME_FAIL:
        return "fail"
    if value in _LEGACY_VERDICT or value in SAM3_VERDICT:
        return "review"
    raise ValueError(f"unsupported frame containment verdict: {value!r}")


def _supplier_machine_observations(
    *,
    asset_id: str,
    frame_rows: Sequence[Mapping[str, Any]],
    supplier_status: np.ndarray,
) -> tuple[SupplierAgreementResult, tuple[tuple[int, str, str], ...]]:
    status = np.asarray(supplier_status)
    if status.ndim != 2 or status.shape[1] != 2:
        raise ValueError("supplier hand quality status must have shape [T,2]")
    machine_by_key: dict[tuple[int, str, str], str] = {}
    for row in frame_rows:
        if row.get("asset_id") != asset_id:
            continue
        frame_idx = _integer(row.get("frame_idx"), "frame_idx")
        raw_side = row.get("hand_side")
        if not isinstance(raw_side, str) or raw_side not in {"left", "right"}:
            raise ValueError("frame hand_side must be exactly left or right")
        camera_id = row.get("camera_id")
        if camera_id != "main":
            raise ValueError("frame camera_id must be main")
        if frame_idx >= status.shape[0]:
            raise ValueError(
                "SAM3 logical frame_idx is outside supplier hand quality status"
            )
        key = (frame_idx, raw_side, "main")
        machine = _frame_machine_status(row)
        previous = machine_by_key.get(key)
        if previous is not None and previous != machine:
            raise ValueError(
                f"conflicting SAM3 frame observations for logical alignment key {key}"
            )
        machine_by_key[key] = machine

    keys = tuple(sorted(machine_by_key))
    supplier_values = np.asarray(
        [status[frame, 0 if side == "left" else 1] for frame, side, _camera in keys]
    )
    machine_values = np.asarray([machine_by_key[key] for key in keys])
    agreement = compare_supplier_and_machine(
        supplier_values,
        machine_values,
    )
    false_negative_keys = tuple(
        key
        for key, supplier, machine in zip(
            keys,
            supplier_values.tolist(),
            machine_values.tolist(),
            strict=True,
        )
        if supplier == "good" and machine == "fail"
    )
    return agreement, false_negative_keys


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


def _supplier_disagreement_issue(
    *,
    asset_id: str,
    keys: tuple[tuple[int, str, str], ...],
    agreement: SupplierAgreementResult,
    config: LoadedQcConfig,
) -> Issue:
    rule_id = "sam3_containment.supplier_mask_disagreement"
    configured = config.module_rules(_MODULE).get("supplier_mask_disagreement")
    if (
        not isinstance(configured, Mapping)
        or configured.get("rule_id") != rule_id
        or configured.get("verdict") != "warn"
    ):
        raise ValueError(f"missing configured warn rule: {rule_id}")
    frames = [frame for frame, _side, _camera in keys]
    sides = {side for _frame, side, _camera in keys}
    issue_side = next(iter(sides)) if len(sides) == 1 else None
    start_frame = min(frames)
    end_frame = max(frames)
    return Issue(
        issue_id=build_issue_id(
            asset_id=asset_id,
            module=_MODULE,
            rule_id=rule_id,
            source_relative_path="frame_keypoint_containment.json",
            coordinate_system="canonical_logical",
            start_frame=start_frame,
            end_frame=end_frame,
            hand_side=issue_side,
            evidence_kind="supplier_machine_agreement",
        ),
        code="supplier_mask_disagreement",
        severity="warn",
        module=_MODULE,
        issue_type="supplier_mask_disagreement",
        metric="supplier_hand_quality.status",
        observed_value={
            "supplier_status": "good",
            "machine_status": "fail",
            "count": agreement.supplier_false_negative_count,
        },
        operator="disagrees_with",
        boundary_value="machine_pass",
        rule_id=rule_id,
        needs_manual_review=True,
        context={
            "coordinate_system": "canonical_logical",
            "camera_id": "main",
            "start_frame": start_frame,
            "end_frame": end_frame,
            "hand_side": issue_side,
            "alignment_keys": [
                {
                    "frame_idx": frame,
                    "hand_side": side,
                    "camera_id": camera,
                }
                for frame, side, camera in keys
            ],
        },
    )


def adapt_sam3_containment(
    *,
    asset_id: str,
    batch_root: Path,
    window_summaries: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
    config: LoadedQcConfig,
    frame_rows: Sequence[Mapping[str, Any]] = (),
    supplier_hand_quality_status: np.ndarray | None = None,
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

    supplier_metrics: dict[str, Any] = {"provided": False}
    if supplier_hand_quality_status is not None:
        agreement, false_negative_keys = _supplier_machine_observations(
            asset_id=asset_id,
            frame_rows=frame_rows,
            supplier_status=supplier_hand_quality_status,
        )
        supplier_metrics = agreement.to_metrics()
        if false_negative_keys:
            issue = _supplier_disagreement_issue(
                asset_id=asset_id,
                keys=false_negative_keys,
                agreement=agreement,
                config=config,
            )
            if issue.issue_id not in issue_ids:
                issue_ids.add(issue.issue_id)
                issues.append(issue)
            verdicts.append("warn")

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
            "supplier_hand_quality": supplier_metrics,
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
