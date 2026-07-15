"""Stable diagnostics raised at the Canonical QC input boundary."""

from __future__ import annotations


class CanonicalInputError(ValueError):
    """A deterministic, field-addressable Canonical input diagnostic."""

    def __init__(self, code: str, field: str, detail: str) -> None:
        self.code = code
        self.field = field
        self.detail = detail
        super().__init__(f"{code}: {field}: {detail}")
