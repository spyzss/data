"""Typed in-memory contracts shared by QC module adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Literal

from qc_common.frame_survival import FrameExclusion


Verdict = Literal["pass", "warn", "fail", "skipped"]
ModuleExecutionState = Literal[
    "completed",
    "disabled",
    "skipped",
    "not_implemented",
    "runtime_error",
    "awaiting_external",
    "skipped_due_to_fail",
    "not_run_due_to_acceptance_frame_budget",
    "not_run",
    "input_missing",
    "input_invalid",
    "adapter_missing",
    "blocked",
]
_VERDICTS = ("pass", "warn", "fail", "skipped")
_ISSUE_SEVERITIES = ("warn", "fail")
RETRYABLE_RUNTIME_ERROR_TYPES = frozenset(
    {"stale_revision", "process_error", "evidence_integrity_error"}
)


def _to_json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _to_json_safe(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Enum):
        return _to_json_safe(value.value)
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    "mapping keys must be strings, "
                    f"got {type(key).__name__}"
                )
            converted[key] = _to_json_safe(item)
        return converted
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float is not JSON-safe: {value!r}")
        return value
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}")


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

    def __post_init__(self) -> None:
        if self.severity not in _ISSUE_SEVERITIES:
            raise ValueError(
                f"severity must be one of {_ISSUE_SEVERITIES}, "
                f"got {self.severity!r}"
            )

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
    frame_exclusions: tuple[FrameExclusion, ...] = ()
    runtime: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.verdict not in _VERDICTS:
            raise ValueError(
                f"verdict must be one of {_VERDICTS}, got {self.verdict!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return _to_json_safe(self)


@dataclass(frozen=True)
class RuntimeErrorRecord:
    module: str
    error_type: str
    message: str
    occurred_at: str

    def __post_init__(self) -> None:
        for field_name in ("module", "error_type", "message", "occurred_at"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"runtime error {field_name} must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "error_type": self.error_type,
            "message": self.message,
            "occurred_at": self.occurred_at,
            "retryable": self.error_type in RETRYABLE_RUNTIME_ERROR_TYPES,
        }


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


def relative_evidence_path(
    path: Path,
    batch_root: Path,
    *,
    allow_symlinked_sources: bool = False,
) -> str:
    lexical_root = Path(
        os.path.abspath(os.fspath(batch_root))
    )
    candidate = (
        path
        if path.is_absolute()
        else lexical_root / path
    )
    lexical_path = Path(
        os.path.abspath(os.fspath(candidate))
    )

    try:
        relative_path = lexical_path.relative_to(lexical_root)
    except ValueError:
        raise ValueError(
            f"evidence path {path} is outside batch root {batch_root}"
        ) from None

    if not allow_symlinked_sources:
        resolved_path = lexical_path.resolve()
        resolved_root = batch_root.resolve()
        try:
            resolved_path.relative_to(resolved_root)
        except ValueError:
            raise ValueError(
                f"evidence path {path} is outside batch root {batch_root}"
            ) from None

    return relative_path.as_posix()
