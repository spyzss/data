"""Immutable, explicitly supplied batch metadata for Canonical Data."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from .errors import CanonicalInputError

if TYPE_CHECKING:
    from .contracts import CanonicalQcEpisode


@dataclass(frozen=True, slots=True)
class BatchMetadata:
    schema_version: str
    batch_id: str
    supplier_id: str
    dataset_attributes_json: str
    content_sha256: str

    @property
    def dataset_attributes(self) -> dict[str, object]:
        return json.loads(self.dataset_attributes_json)


def _fail(field: str, detail: str) -> None:
    raise CanonicalInputError("invalid_batch_metadata", field, detail)


def load_batch_metadata(path: Path) -> BatchMetadata:
    source = Path(path)
    try:
        payload_bytes = source.read_bytes()
        payload = json.loads(payload_bytes.decode("utf-8"))
    except OSError as exc:
        raise CanonicalInputError(
            "source_integrity_error",
            "batch_metadata",
            f"cannot read batch metadata: {exc}",
            retryable=True,
        ) from exc
    except (UnicodeError, json.JSONDecodeError) as exc:
        _fail("batch_metadata", f"must contain valid UTF-8 JSON: {exc}")
    if not isinstance(payload, dict):
        _fail("batch_metadata", "must be a JSON object")
    if payload.get("schema_version") != "canonical_batch_metadata.v1":
        _fail(
            "batch_metadata.schema_version",
            "must equal 'canonical_batch_metadata.v1'",
        )
    for name in ("batch_id", "supplier_id"):
        value = payload.get(name)
        if not isinstance(value, str) or not value.strip():
            _fail(f"batch_metadata.{name}", "must be a non-empty string")
    attributes = payload.get("dataset_attributes")
    if not isinstance(attributes, dict):
        _fail("batch_metadata.dataset_attributes", "must be an object")
    try:
        attributes_json = json.dumps(
            attributes,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        _fail(
            "batch_metadata.dataset_attributes",
            f"must be JSON-compatible: {exc}",
        )
    return BatchMetadata(
        schema_version="canonical_batch_metadata.v1",
        batch_id=payload["batch_id"],
        supplier_id=payload["supplier_id"],
        dataset_attributes_json=attributes_json,
        content_sha256=hashlib.sha256(payload_bytes).hexdigest(),
    )


def with_batch_metadata(
    episode: CanonicalQcEpisode, metadata: BatchMetadata
) -> CanonicalQcEpisode:
    for name in ("batch_id", "supplier_id"):
        expected = getattr(episode.identity, name)
        observed = getattr(metadata, name)
        if observed != expected:
            raise CanonicalInputError(
                "batch_metadata_identity_mismatch",
                f"batch_metadata.{name}",
                f"expected {expected!r}, got {observed!r}",
            )
    updated = replace(episode, batch_metadata=metadata)
    from .validation import validate_episode

    validate_episode(updated)
    return updated


__all__ = ["BatchMetadata", "load_batch_metadata", "with_batch_metadata"]
