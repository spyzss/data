"""Deterministic and traversal-safe publisher layout helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re


_RELEASE_ID = re.compile(r"lerobot-v3-[0-9a-f]{32}").fullmatch


def derive_release_id(
    *,
    asset_id: str,
    canonical_revision: int,
    semantic_fingerprint: str,
    publisher_version: str,
) -> str:
    payload = json.dumps(
        {
            "asset_id": asset_id,
            "canonical_revision": canonical_revision,
            "publisher_version": publisher_version,
            "semantic_fingerprint": semantic_fingerprint,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"lerobot-v3-{hashlib.sha256(payload).hexdigest()[:32]}"


@dataclass(frozen=True, slots=True)
class ReleaseLayout:
    root: Path
    release_id: str
    release_path: Path
    current_path: Path


def layout_for(release_root: Path, release_id: str) -> ReleaseLayout:
    if _RELEASE_ID(release_id) is None:
        raise ValueError("release_id is not a safe deterministic publisher ID")
    root = Path(release_root)
    return ReleaseLayout(
        root=root,
        release_id=release_id,
        release_path=root / "releases" / release_id,
        current_path=root / "CURRENT.json",
    )


__all__ = ["ReleaseLayout", "derive_release_id", "layout_for"]
