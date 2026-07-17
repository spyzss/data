"""Canonical supplier identities and backwards-compatible input aliases."""

from __future__ import annotations

from typing import Any


_ALIASES = {
    "dr": ("dr", "DR"),
    "deepreach": ("dr", "DR"),
    "potentia": ("potentia", "Potentia"),
}


def normalize_supplier(value: Any) -> tuple[str, str, str | None]:
    """Return canonical id/display name and the non-canonical source alias."""
    source = "" if value is None else str(value).strip()
    normalized = source.lower()
    canonical_id, display_name = _ALIASES.get(
        normalized,
        (normalized, source or "unknown"),
    )
    alias = source if source and normalized != canonical_id else None
    return canonical_id, display_name, alias


__all__ = ["normalize_supplier"]
