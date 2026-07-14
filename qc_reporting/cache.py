"""Rebuildable parquet cache for canonical QC projections.

The cache is an acceleration layer only.  The JSON reports under a quality
archive remain the source of truth; a cache is accepted only when its source
manifest exactly matches the current archive and all three projection tables
can be read successfully.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from qc_reporting.projection import BatchProjection


CACHE_FILES = {
    "assets": "assets.parquet",
    "issues": "issues.parquet",
    "execution": "execution.parquet",
}

_TABLE_ROWS = {
    "assets": "asset_rows",
    "issues": "issue_rows",
    "execution": "execution_rows",
}
_JSON_MARKER = "__qc_cache_json__:"
_SCALAR_MARKER = "__qc_cache_scalar__:"
_MANIFEST_FILE = "source_reports.json"
_PROJECTION_MANIFEST_FILE = "projection_source_manifest.json"
_PROJECTION_MANIFEST_HASH_FILE = "projection_source_manifest.sha256"
_TABLE_HASHES_FILE = "projection_table_hashes.json"
_SOURCE_MANIFEST_FIELDS = ("relative_path", "asset_id", "revision", "sha256")


def _json_safe(value: Any) -> Any:
    """Return a JSON-compatible representation of *value*.

    Projection rows contain tuples (for example ``runtime_errors``), so the
    cache encoder carries a small type tag rather than relying on pandas' type
    inference.  This keeps a cache round-trip equal to the original
    :class:`BatchProjection` instead of changing tuples into lists.
    """

    if isinstance(value, Mapping):
        return {
            "__qc_cache_type__": "mapping",
            "value": [[str(key), _json_safe(item)] for key, item in value.items()],
        }
    if isinstance(value, tuple):
        return {"__qc_cache_type__": "tuple", "value": [_json_safe(item) for item in value]}
    if isinstance(value, list):
        return {"__qc_cache_type__": "list", "value": [_json_safe(item) for item in value]}
    if isinstance(value, set):
        return {
            "__qc_cache_type__": "set",
            "value": [_json_safe(item) for item in sorted(value, key=repr)],
        }
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            return None
        return value
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _json_restore(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    if value.get("__qc_cache_type__") == "mapping":
        return {str(key): _json_restore(item) for key, item in value.get("value", ())}
    if value.get("__qc_cache_type__") == "tuple":
        return tuple(_json_restore(item) for item in value.get("value", ()))
    if value.get("__qc_cache_type__") == "list":
        return [_json_restore(item) for item in value.get("value", ())]
    if value.get("__qc_cache_type__") == "set":
        return {_json_restore(item) for item in value.get("value", ())}
    return {str(key): _json_restore(item) for key, item in value.items()}


def _encode_cell(value: Any) -> Any:
    if isinstance(value, (Mapping, tuple, list, set)):
        return _JSON_MARKER + json.dumps(
            _json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    if isinstance(value, str) and (
        value.startswith(_JSON_MARKER) or value.startswith(_SCALAR_MARKER)
    ):
        return _SCALAR_MARKER + value
    return _json_safe(value)


def _decode_cell(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(_SCALAR_MARKER):
        return value[len(_SCALAR_MARKER) :]
    if isinstance(value, str) and value.startswith(_JSON_MARKER):
        try:
            return _json_restore(json.loads(value[len(_JSON_MARKER) :]))
        except (TypeError, ValueError, json.JSONDecodeError):
            raise ValueError("invalid encoded projection value")
    # pandas represents nulls as NaN in object columns after a parquet read.
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _manifest_path(path: Path, quality_archive: Path) -> str:
    try:
        return path.relative_to(quality_archive).as_posix()
    except ValueError as exc:
        raise ValueError(f"source report is outside quality archive: {path}") from exc


def build_source_manifest(quality_archive: Path) -> tuple[dict[str, Any], ...]:
    """Build a deterministic manifest for every canonical report JSON.

    ``relative_path`` is relative to ``quality_archive`` and uses POSIX
    separators so the manifest compares identically across operating systems.
    The file digest catches edits that do not bump ``report_revision``.
    """

    root = Path(quality_archive)
    if not root.is_dir():
        raise ValueError(f"quality archive directory does not exist: {root}")
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json"), key=lambda item: item.name):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid QC report {path}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"invalid QC report {path}: root must be an object")
        asset_id = payload.get("asset_id")
        if asset_id is None or str(asset_id) == "":
            raise ValueError(f"invalid QC report {path}: asset_id must be non-empty")
        revision = payload.get("report_revision", 0)
        try:
            revision = int(revision)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid QC report {path}: report_revision must be an integer") from exc
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(
            {
                "relative_path": _manifest_path(path, root),
                "asset_id": str(asset_id),
                "revision": revision,
                "sha256": digest,
            }
        )
    return tuple(rows)


def _write_atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{path.name}.", dir=path.parent) as temp_dir:
        temp_path = Path(temp_dir) / path.name
        frame.to_parquet(temp_path, index=False)
        digest = hashlib.sha256(temp_path.read_bytes()).hexdigest()
        os.replace(temp_path, path)
        return digest


def _frame_from_rows(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    materialized = [dict(row) for row in rows]
    columns = tuple(dict.fromkeys(key for row in materialized for key in row))
    if not columns:
        return pd.DataFrame()
    return pd.DataFrame(
        [{str(key): _encode_cell(value) for key, value in row.items()} for row in materialized],
        columns=columns,
    )


def write_projection_cache(projection: BatchProjection, cache_dir: Path) -> None:
    """Atomically write all projection tables and their source manifest."""

    if not isinstance(projection, BatchProjection):
        raise TypeError("projection must be a BatchProjection")
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    # The cache manifest is derived from the projection's source paths when
    # available.  CLI callers pass a projection built from the archive, so
    # those paths are sufficient to calculate the same source identity.
    source_manifest: list[dict[str, Any]] = []
    for source in projection.source_manifest:
        if not isinstance(source, Mapping):
            raise ValueError("projection source manifest entries must be mappings")
        path_value = source.get("path") or source.get("report_path") or source.get("json_path")
        if path_value:
            path = Path(str(path_value))
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                asset_id = str(payload.get("asset_id") or source.get("asset_id") or "")
                revision = int(
                    payload.get("report_revision", source.get("report_revision", 0))
                )
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                AttributeError,
                TypeError,
                ValueError,
            ):
                # If a caller provides a projection with a non-file source
                # manifest, retain its identity fields where possible.  The
                # CLI path always has canonical files and therefore takes the
                # branch above.
                asset_id = str(source.get("asset_id") or "")
                revision = int(source.get("report_revision", source.get("revision", 0)) or 0)
                digest = str(source.get("sha256") or "")
            relative_path = str(source.get("relative_path") or path.name)
        else:
            # A projection assembled by another caller may already carry the
            # canonical source manifest fields rather than absolute paths.
            # Preserve those fields instead of silently writing an empty,
            # unverifiable cache.
            missing = [field for field in _SOURCE_MANIFEST_FIELDS if field not in source]
            if missing:
                raise ValueError(
                    "projection source manifest entry is missing: " + ", ".join(missing)
                )
            relative_path = str(source["relative_path"])
            asset_id = str(source["asset_id"])
            revision = int(source["revision"])
            digest = str(source["sha256"])
        source_manifest.append(
            {
                "relative_path": relative_path,
                "asset_id": asset_id,
                "revision": revision,
                "sha256": digest,
            }
        )
    source_manifest.sort(key=lambda row: str(row["relative_path"]))
    _validate_source_manifest(source_manifest)

    # Write the projection identity first and the source manifest last.  The
    # source manifest is the generation barrier: a reader either sees the
    # previous source generation or the complete new generation, never a
    # source manifest paired with partially replaced tables.
    projection_manifest_bytes = (
        json.dumps(
            [dict(row) for row in projection.source_manifest],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    _write_atomic_bytes(
        cache_root / _PROJECTION_MANIFEST_FILE,
        projection_manifest_bytes,
    )
    _write_atomic_bytes(
        cache_root / _PROJECTION_MANIFEST_HASH_FILE,
        (hashlib.sha256(projection_manifest_bytes).hexdigest() + "\n").encode("ascii"),
    )
    table_hashes: dict[str, str] = {}
    for table, attr in _TABLE_ROWS.items():
        table_hashes[table] = _write_parquet_atomic(
            _frame_from_rows(getattr(projection, attr)), cache_root / CACHE_FILES[table]
        )
    _write_atomic_bytes(
        cache_root / _TABLE_HASHES_FILE,
        (
            json.dumps(table_hashes, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
    )
    manifest = json.dumps(
        source_manifest,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    _write_atomic_bytes(
        cache_root / _MANIFEST_FILE,
        manifest.encode("utf-8"),
    )


def _validate_source_manifest(entries: Sequence[Mapping[str, Any]]) -> None:
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise ValueError("cache source manifest must be an array")
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"cache source manifest entry {index} must be an object")
        missing = [field for field in _SOURCE_MANIFEST_FIELDS if field not in entry]
        if missing:
            raise ValueError(
                f"cache source manifest entry {index} missing: {', '.join(missing)}"
            )
        if not isinstance(entry["relative_path"], str) or not entry["relative_path"]:
            raise ValueError(f"cache source manifest entry {index} has invalid relative_path")
        if Path(entry["relative_path"]).is_absolute():
            raise ValueError(f"cache source manifest entry {index} path must be relative")
        if not isinstance(entry["asset_id"], str) or not entry["asset_id"]:
            raise ValueError(f"cache source manifest entry {index} has invalid asset_id")
        if isinstance(entry["revision"], bool):
            raise ValueError(f"cache source manifest entry {index} has invalid revision")
        try:
            int(entry["revision"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"cache source manifest entry {index} has invalid revision") from exc
        digest = entry["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != hashlib.sha256().digest_size * 2
            or any(char not in "0123456789abcdefABCDEF" for char in digest)
        ):
            raise ValueError(f"cache source manifest entry {index} has invalid sha256")


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ValueError("cache manifest must be an array")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, Mapping):
            raise ValueError(f"cache manifest entry {index} must be an object")
        rows.append(dict(row))
    _validate_source_manifest(rows)
    return rows


def _read_projection_manifest(path: Path) -> tuple[dict[str, Any], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ValueError("projection source manifest must be an array")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, Mapping):
            raise ValueError(f"projection source manifest entry {index} must be an object")
        rows.append(dict(row))
    return tuple(rows)


def _read_projection_manifest_with_integrity(
    cache_root: Path,
) -> tuple[dict[str, Any], ...]:
    manifest_path = cache_root / _PROJECTION_MANIFEST_FILE
    manifest_bytes = manifest_path.read_bytes()
    expected_digest = (cache_root / _PROJECTION_MANIFEST_HASH_FILE).read_text(
        encoding="ascii"
    ).strip()
    actual_digest = hashlib.sha256(manifest_bytes).hexdigest()
    if expected_digest != actual_digest:
        raise ValueError("projection source manifest integrity check failed")
    return _read_projection_manifest(manifest_path)


def _read_table_hashes(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("projection table hashes must be an object")
    expected_keys = set(CACHE_FILES)
    if set(payload) != expected_keys:
        raise ValueError("projection table hashes have unexpected tables")
    hashes: dict[str, str] = {}
    for table, digest in payload.items():
        if (
            not isinstance(digest, str)
            or len(digest) != hashlib.sha256().digest_size * 2
            or any(char not in "0123456789abcdefABCDEF" for char in digest)
        ):
            raise ValueError(f"invalid projection table hash for {table}")
        hashes[str(table)] = digest
    return hashes


def _projection_manifest_matches_source(
    projection_manifest: Sequence[Mapping[str, Any]],
    source_manifest: Sequence[Mapping[str, Any]],
) -> bool:
    """Cross-check projection identities against source generation fields."""

    if len(projection_manifest) != len(source_manifest):
        return False
    source_by_path = {str(row["relative_path"]): row for row in source_manifest}
    if len(source_by_path) != len(source_manifest):
        return False
    matched: set[str] = set()
    for row in projection_manifest:
        path_value = row.get("relative_path") or row.get("path") or row.get("report_path") or row.get("json_path")
        asset_id = row.get("asset_id")
        if path_value in {None, ""} or asset_id in {None, ""}:
            return False
        path_text = Path(str(path_value)).as_posix()
        candidates = [
            relative
            for relative in source_by_path
            if path_text == relative or path_text.endswith("/" + relative)
        ]
        if len(candidates) != 1:
            return False
        relative = candidates[0]
        expected = source_by_path[relative]
        if relative in matched or str(asset_id) != str(expected["asset_id"]):
            return False
        raw_revision = row.get("report_revision", row.get("revision", 0))
        try:
            revision = int(raw_revision)
        except (TypeError, ValueError):
            return False
        if revision != int(expected["revision"]):
            return False
        matched.add(relative)
    return len(matched) == len(source_by_path)


def _read_projection_table(path: Path) -> tuple[dict[str, Any], ...]:
    frame = pd.read_parquet(path)
    rows: list[dict[str, Any]] = []
    for raw in frame.to_dict(orient="records"):
        rows.append({str(key): _decode_cell(value) for key, value in raw.items()})
    return tuple(rows)


def _read_projection_table_with_integrity(
    path: Path,
    expected_digest: str,
) -> tuple[dict[str, Any], ...]:
    actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_digest != actual_digest:
        raise ValueError(f"projection table integrity check failed: {path.name}")
    return _read_projection_table(path)


def load_projection_cache(
    cache_dir: Path,
    expected_manifest: Iterable[Mapping[str, Any]],
) -> BatchProjection | None:
    """Load a cache only when its manifest and all tables are valid.

    Any missing, malformed, unreadable, or stale cache artifact is treated as
    a cache miss.  Callers can then rebuild from canonical JSON without
    needing to distinguish the failure mode.
    """

    cache_root = Path(cache_dir)
    try:
        source_manifest = _read_manifest(cache_root / _MANIFEST_FILE)
        expected = [dict(row) for row in expected_manifest]
        _validate_source_manifest(expected)
        if source_manifest != expected:
            return None
        if not all((cache_root / filename).is_file() for filename in CACHE_FILES.values()):
            return None
        projection_manifest = _read_projection_manifest_with_integrity(cache_root)
        if not _projection_manifest_matches_source(projection_manifest, source_manifest):
            return None
        table_hashes = _read_table_hashes(cache_root / _TABLE_HASHES_FILE)
        rows = {
            table: _read_projection_table_with_integrity(
                cache_root / filename, table_hashes[table]
            )
            for table, filename in CACHE_FILES.items()
        }
    except Exception:
        return None
    return BatchProjection(
        asset_rows=rows["assets"],
        issue_rows=rows["issues"],
        execution_rows=rows["execution"],
        source_manifest=tuple(projection_manifest),
    )


__all__ = [
    "CACHE_FILES",
    "build_source_manifest",
    "load_projection_cache",
    "write_projection_cache",
]
