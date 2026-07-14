"""Typed in-memory contracts shared by QC module adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Literal


Verdict = Literal["pass", "warn", "fail", "skipped"]


def _to_json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _to_json_safe(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Enum):
        return _to_json_safe(value.value)
    if isinstance(value, Mapping):
        return {key: _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class EvidenceRef:
    evidence_id: str
    kind: str
    path: str
    coordinate_system: str
    start_frame: int | None = None
    end_frame: int | None = None
    hand_side: str | None = None
    checksum: str | None = None
    mime_type: str | None = None
    generator_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _to_json_safe(self)


@dataclass(frozen=True)
class Issue:
    issue_id: str
    code: str
    severity: Literal["warn", "fail"]
    module: str
    issue_type: str
    metric: str
    observed_value: Any
    operator: str
    boundary_value: Any
    rule_id: str
    needs_manual_review: bool
    context: Mapping[str, Any] = field(default_factory=dict)
    evidence_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _to_json_safe(self)


@dataclass(frozen=True)
class ModuleResult:
    module: str
    verdict: Verdict
    evaluation: Mapping[str, Any]
    metrics: Mapping[str, Any]
    issues: tuple[Issue, ...] = ()
    evidence: tuple[EvidenceRef, ...] = ()
    runtime: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_json_safe(self)


def build_issue_id(
    *,
    asset_id: str,
    module: str,
    rule_id: str,
    source_relative_path: str,
    coordinate_system: str,
    start_frame: int | None,
    end_frame: int | None,
    hand_side: str | None,
    evidence_kind: str,
) -> str:
    identity = {
        "asset_id": asset_id,
        "module": module,
        "rule_id": rule_id,
        "source_relative_path": source_relative_path,
        "coordinate_system": coordinate_system,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "hand_side": hand_side,
        "evidence_kind": evidence_kind,
    }
    canonical_identity = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(canonical_identity.encode("utf-8")).hexdigest()[:20]
    rule_name = rule_id.rsplit(".", 1)[-1]
    return f"{module}:{rule_name}:{digest}"


def relative_evidence_path(path: Path, batch_root: Path) -> str:
    resolved_path = path.resolve()
    resolved_root = batch_root.resolve()
    try:
        relative_path = resolved_path.relative_to(resolved_root)
    except ValueError:
        raise ValueError(
            f"evidence path {path} is outside batch root {batch_root}"
        ) from None
    return relative_path.as_posix()
