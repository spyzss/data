"""Path-oriented package API used by the thin publication CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from canonical_qc.config import load_canonical_qc_config
from canonical_qc.workflow import load_canonical_source, validate_explicit_path

from .contracts import (
    CanonicalDiagnostic,
    PublishPlan,
    PublishPrerequisiteError,
    PublishRequest,
    PublishResult,
)
from .prerequisites import validate_publish_request
from .publisher import publish


def _reject(field: str, message: str) -> None:
    raise PublishPrerequisiteError(
        CanonicalDiagnostic(
            "publish_prerequisite_failed",
            "publish_request",
            field,
            message,
            False,
        )
    )


def _integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _reject(field, "must be a non-negative integer in the final QC report")
    return value


def publish_from_paths(
    *,
    source: Path,
    source_format: str,
    canonical_source_root: Path,
    qc_report_path: Path,
    release_root: Path,
    episode_index: int | None = None,
    canonical_config_path: Path | None = None,
    dry_run: bool = False,
) -> PublishPlan | PublishResult:
    """Derive revisions from the final report and validate or atomically publish."""

    config = load_canonical_qc_config(canonical_config_path)
    explicit_source = validate_explicit_path(source, field="source")
    explicit_source_root = validate_explicit_path(
        canonical_source_root, field="canonical_source_root"
    )
    explicit_report = validate_explicit_path(qc_report_path, field="qc_report")
    explicit_release_root = validate_explicit_path(release_root, field="release_root")
    episode = load_canonical_source(
        source=explicit_source,
        source_format=source_format,
        source_root=explicit_source_root,
        episode_index=episode_index,
        config=config,
    )
    report_path = explicit_report
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _reject("qc_report", f"cannot read final QC report: {exc}")
    if not isinstance(report, dict):
        _reject("qc_report", "final QC report must be a JSON object")
    binding = report.get("canonical_binding")
    if not isinstance(binding, dict):
        _reject("canonical_binding", "final QC report is missing canonical_binding")
    request = PublishRequest(
        episode=episode,
        canonical_revision=_integer(
            binding.get("canonical_revision"),
            field="canonical_binding.canonical_revision",
        ),
        canonical_source_root=explicit_source_root,
        qc_report_path=report_path,
        expected_report_revision=_integer(
            report.get("report_revision"), field="report_revision"
        ),
        release_root=explicit_release_root,
    )
    return validate_publish_request(request) if dry_run else publish(request)


__all__ = ["publish_from_paths"]
