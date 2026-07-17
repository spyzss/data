"""Stable diagnostics raised at the Canonical QC input boundary."""

from __future__ import annotations

from pathlib import Path


class CanonicalInputError(ValueError):
    """A deterministic, field-addressable Canonical input diagnostic."""

    def __init__(
        self,
        code: str,
        field: str,
        detail: str,
        *,
        retryable: bool = False,
    ) -> None:
        self.code = code
        self.field = field
        self.detail = detail
        self.retryable = retryable
        self.report_path: Path | None = None
        self.report_revision: int | None = None
        self.overall_decision: str | None = None
        super().__init__(f"{code}: {field}: {detail}")

    def attach_report(
        self,
        path: Path,
        *,
        revision: int,
        overall_decision: str | None,
    ) -> "CanonicalInputError":
        self.report_path = path
        self.report_revision = revision
        self.overall_decision = overall_decision
        return self
