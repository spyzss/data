"""Standalone SAM3 containment window-review contracts and persistence.

This module deliberately does not adapt window IDs into the generic human-QC
``issue_id`` contract.  It consumes the explicit review queue and evidence
manifest, stages only named evidence files, and keeps human decisions scoped
to one source-inclusive window.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import tempfile
from threading import RLock
from typing import Any, Iterable, Iterator, Mapping
from urllib.parse import quote, unquote

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
MANUAL_STATES = frozenset({"fail", "review"})
AUTOMATIC_PASS_STATES = frozenset({"pass", "auto_pass", "completed_pass"})
EXPECTED_EVIDENCE_COUNT = 5
EVIDENCE_SOURCE_MODULE = "sam3_containment"
EVIDENCE_TYPE = "combined_overlay"
FRAME_COORDINATE_SYSTEM = "source_inclusive"
DURATION_RESOLUTION_STATUS = "not_annotated"
STATE_SCHEMA_VERSION = "sam3_window_review_state.v1"
RESULT_COLUMNS = (
    "review_id",
    "supplier_id",
    "asset_id",
    "window_start_frame",
    "window_end_frame",
    "frame_coordinate_system",
    "verdict",
    "reviewer",
    "reviewed_at",
    "revision",
    "source_queue_path",
    "source_queue_sha256",
    "evidence_provenance_json",
    "duration_resolution_status",
)


class ReviewValidationError(ValueError):
    """The requested window mutation violates the review contract."""


class ReviewConflictError(RuntimeError):
    """The requested mutation conflicts with current durable state."""


@contextmanager
def _exclusive_state_lock(path: Path) -> Iterator[None]:
    """Lock a stable sidecar while the authoritative state inode is replaced."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    locked = False
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        locked = True
        yield
    finally:
        try:
            if locked:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


@dataclass(frozen=True)
class ReviewBundle:
    """Normalized queue and staged evidence exposed to the review server."""

    manifest_path: Path
    queue_path: Path
    queue_sha256: str
    evidence_path: Path
    review_dir: Path
    assets_root: Path
    items: tuple[dict[str, Any], ...]
    allowed_asset_paths: frozenset[str]

    def item_map(self) -> dict[str, dict[str, Any]]:
        return {str(item["review_id"]): item for item in self.items}

    def to_dict(self, reviews: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
        saved = reviews or {}
        items: list[dict[str, Any]] = []
        for item in self.items:
            output = _json_copy(item)
            _refresh_staged_evidence(output, self.assets_root, self.allowed_asset_paths)
            review_id = str(item["review_id"])
            review = saved.get(review_id)
            if isinstance(review, Mapping):
                output["manual_review"] = _json_copy(dict(review))
            items.append(output)
        return {
            "schema_version": "sam3_window_review_bundle.v1",
            "frame_coordinate_system": FRAME_COORDINATE_SYSTEM,
            "duration_resolution_status": DURATION_RESOLUTION_STATUS,
            "manifest_path": str(self.manifest_path),
            "source_queue_path": str(self.queue_path),
            "source_queue_sha256": self.queue_sha256,
            "evidence_manifest_path": str(self.evidence_path),
            "items": items,
        }


class Sam3WindowReviewStore:
    """Thread-safe, atomic persistence for independent window decisions."""

    def __init__(self, bundle: ReviewBundle, save_dir: Path) -> None:
        self.bundle = bundle
        self.save_dir = Path(save_dir).resolve()
        self.state_path = self.save_dir / "sam3_window_review_state.json"
        self.results_path = self.save_dir / "sam3_window_review_results.csv"
        self.lock_path = self.save_dir / ".sam3_window_review_state.json.lock"
        self._lock = RLock()
        self._items = bundle.item_map()
        with self._lock, _exclusive_state_lock(self.lock_path):
            self._state = self._load_state()
            if self.state_path.is_file():
                self._sync_results_csv(self._state)

    def _empty_state(self) -> dict[str, Any]:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "source_queue_path": str(self.bundle.queue_path),
            "source_queue_sha256": self.bundle.queue_sha256,
            "evidence_manifest_path": str(self.bundle.evidence_path),
            "reviews": {},
        }

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            return self._empty_state()
        value = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != STATE_SCHEMA_VERSION:
            raise ReviewValidationError("saved state schema_version mismatch")
        if value.get("source_queue_path") != str(self.bundle.queue_path):
            raise ReviewValidationError("saved state source_queue_path mismatch")
        if value.get("source_queue_sha256") != self.bundle.queue_sha256:
            raise ReviewValidationError("saved state queue identity sha256 mismatch")
        reviews = value.get("reviews")
        if not isinstance(reviews, dict):
            raise ReviewValidationError("saved state reviews must be an object")
        unknown = sorted(set(reviews) - set(self._items))
        if unknown:
            raise ReviewValidationError(
                "saved state contains review_id values absent from queue: " + ", ".join(unknown)
            )
        for review_id, review in reviews.items():
            if not isinstance(review, dict):
                raise ReviewValidationError(f"saved review must be an object: {review_id}")
            if review.get("review_id") != review_id or review.get("verdict") not in {"pass", "fail"}:
                raise ReviewValidationError(f"saved review contract mismatch: {review_id}")
        return value

    def snapshot(self) -> dict[str, Any]:
        with self._lock, _exclusive_state_lock(self.lock_path):
            self._state = self._load_state()
            return _json_copy(self._state)

    def _sync_results_csv(self, state: Mapping[str, Any]) -> None:
        expected = _results_csv(state, self.bundle.items)
        try:
            current = (
                self.results_path.read_text(encoding="utf-8")
                if self.results_path.is_file()
                else None
            )
        except (OSError, UnicodeError):
            current = None
        if current == expected:
            return
        try:
            _write_text_files_atomically(
                self.save_dir,
                {self.results_path.name: expected},
            )
        except OSError as exc:
            LOGGER.warning(
                "Could not rebuild derived SAM3 review CSV %s: %s",
                self.results_path,
                exc,
            )

    def save(
        self,
        review_id: str,
        verdict: str,
        reviewer: str,
        *,
        expected_revision: int,
    ) -> dict[str, Any]:
        review_id = _required_text(review_id, "review_id")
        if verdict not in {"pass", "fail"}:
            raise ReviewValidationError("verdict must be pass or fail")
        reviewer = _required_text(reviewer, "reviewer")
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int):
            raise ReviewValidationError("expected_revision must be an integer")
        with self._lock, _exclusive_state_lock(self.lock_path):
            item = self._items.get(review_id)
            if item is None:
                raise KeyError(f"unknown review_id: {review_id}")
            if item.get("can_review") is not True:
                raise ReviewConflictError(
                    f"review evidence is incomplete: {item.get('evidence_error') or 'unavailable'}"
                )
            missing_staged = _missing_staged_evidence(
                item, self.bundle.assets_root, self.bundle.allowed_asset_paths
            )
            if missing_staged:
                raise ReviewConflictError(
                    "review evidence is incomplete: staged evidence missing: "
                    + ", ".join(missing_staged)
                )
            durable_state = self._load_state()
            self._state = durable_state
            reviews = durable_state.get("reviews")
            if not isinstance(reviews, dict):
                raise ReviewValidationError("saved state reviews must be an object")
            prior = reviews.get(review_id)
            current_revision = int(prior.get("revision", 0)) if isinstance(prior, dict) else 0
            if current_revision != expected_revision:
                raise ReviewConflictError(
                    f"expected revision {expected_revision}, found {current_revision}"
                )
            reviewed_at = datetime.now(timezone.utc).isoformat()
            record = {
                "status": "completed",
                "review_id": review_id,
                "supplier_id": item["supplier_id"],
                "asset_id": item["asset_id"],
                "window_start_frame": item["window_start_frame"],
                "window_end_frame": item["window_end_frame"],
                "frame_coordinate_system": item["frame_coordinate_system"],
                "verdict": verdict,
                "reviewer": reviewer,
                "reviewed_at": reviewed_at,
                "revision": current_revision + 1,
                "source_queue_path": str(self.bundle.queue_path),
                "source_queue_sha256": self.bundle.queue_sha256,
                "evidence_provenance": _json_copy(item["evidence_provenance"]),
                "duration_resolution_status": DURATION_RESOLUTION_STATUS,
            }
            candidate = _json_copy(durable_state)
            candidate["reviews"][review_id] = record
            _write_text_files_atomically(
                self.save_dir,
                {
                    self.state_path.name: json.dumps(
                        candidate,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                },
            )
            self._state = candidate
            self._sync_results_csv(candidate)
            return _json_copy(record)


def load_review_bundle(
    *,
    manifest_path: Path,
    queue_path: Path,
    evidence_path: Path,
    review_dir: Path,
) -> ReviewBundle:
    """Load explicit contracts and stage only their named combined overlays."""

    manifest_path = Path(manifest_path).resolve()
    queue_path = Path(queue_path).resolve()
    evidence_path = Path(evidence_path).resolve()
    review_dir = Path(review_dir).resolve()
    assets_root = review_dir / "assets"
    manifest_rows = _read_records(manifest_path)
    queue_rows = _read_records(queue_path)
    evidence_rows = _read_records(evidence_path)

    manifest_assets = {
        _text(row.get("asset_id")) for row in manifest_rows if _text(row.get("asset_id"))
    }
    if not manifest_assets:
        raise ValueError("manifest contains no asset_id values")

    normalized_queue: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row_index, source in enumerate(queue_rows):
        row = {str(key): _missing_to_none(value) for key, value in source.items()}
        state = _text(row.get("sam3_window_state")).lower()
        if state in AUTOMATIC_PASS_STATES:
            continue
        if state and state not in MANUAL_STATES:
            continue
        asset_id = _required_text(row.get("asset_id"), f"review queue row {row_index} asset_id")
        if asset_id not in manifest_assets:
            raise ValueError(f"review queue asset_id is missing from manifest: {asset_id}")
        supplier_id = _text(row.get("supplier_id")) or "jdt"
        start = _strict_int(row.get("window_start_frame"), "window_start_frame")
        end = _strict_int(row.get("window_end_frame"), "window_end_frame")
        if start > end:
            raise ValueError(f"window_start_frame must be <= window_end_frame for {asset_id}")
        coordinate_system = _text(row.get("frame_coordinate_system")) or FRAME_COORDINATE_SYSTEM
        if coordinate_system != FRAME_COORDINATE_SYSTEM:
            raise ValueError(
                f"frame_coordinate_system must be {FRAME_COORDINATE_SYSTEM!r}, got {coordinate_system!r}"
            )
        review_id = _text(row.get("review_id")) or generated_review_id(
            supplier_id=supplier_id,
            asset_id=asset_id,
            start=start,
            end=end,
        )
        if review_id in seen_ids:
            raise ValueError(f"duplicate review_id: {review_id}")
        seen_ids.add(review_id)
        normalized_queue.append(
            {
                "review_id": review_id,
                "supplier_id": supplier_id,
                "asset_id": asset_id,
                "window_start_frame": start,
                "window_end_frame": end,
                "source_start_frame": _int_or(row.get("source_start_frame"), start),
                "source_end_frame": _int_or(row.get("source_end_frame"), end),
                "frame_coordinate_system": coordinate_system,
                "sam3_window_state": state or "review",
                "left_window_containment_verdict": _text(
                    row.get("left_window_containment_verdict")
                ),
                "right_window_containment_verdict": _text(
                    row.get("right_window_containment_verdict")
                ),
                "sampled_frame_indices": _json_field(row.get("sampled_frame_indices_json"), []),
                "video_path": _text(row.get("video_path")),
                "fps": _float_or_none(row.get("fps")),
                "trigger_reason": _json_field(row.get("trigger_reason_json"), []),
                "trigger_metrics": _json_field(row.get("trigger_metrics_json"), {}),
                "source_queue_path": str(queue_path),
                "manual_review": {"status": "unresolved", "verdict": None},
            }
        )

    by_review_id = {row["review_id"]: row for row in normalized_queue}
    by_composite = {
        _composite_key(row): row["review_id"] for row in normalized_queue
    }
    exact: dict[str, list[dict[str, Any]]] = {}
    fallback: dict[str, list[dict[str, Any]]] = {}
    for evidence_index, source in enumerate(evidence_rows):
        row = {str(key): _missing_to_none(value) for key, value in source.items()}
        review_id = _text(row.get("review_id"))
        if review_id:
            if review_id in by_review_id:
                exact.setdefault(review_id, []).append(row)
            continue
        composite = _evidence_composite_key(row, evidence_index)
        target = by_composite.get(composite)
        if target is not None:
            fallback.setdefault(target, []).append(row)

    allowed: set[str] = set()
    completed_items: list[dict[str, Any]] = []
    for row in normalized_queue:
        review_id = str(row["review_id"])
        matched = exact.get(review_id) or fallback.get(review_id) or []
        match_kind = "review_id" if exact.get(review_id) else "composite"
        evidence, staged_paths = _stage_evidence(
            matched,
            row=row,
            evidence_base=evidence_path.parent,
            assets_root=assets_root,
            match_kind=match_kind,
        )
        allowed.update(staged_paths)
        row["evidence"] = evidence
        row["evidence_provenance"] = [
            {
                key: value
                for key, value in item.items()
                if key in {
                    "frame_idx",
                    "source_path",
                    "source_module",
                    "evidence_type",
                    "hand_side",
                    "metadata",
                    "status",
                }
            }
            for item in evidence
        ]
        ready = len(evidence) == EXPECTED_EVIDENCE_COUNT and all(
            item.get("status") == "ready" for item in evidence
        )
        row["evidence_status"] = "ready" if ready else "error"
        row["evidence_error"] = None if ready else _evidence_error(evidence)
        row["can_review"] = ready
        completed_items.append(row)

    return ReviewBundle(
        manifest_path=manifest_path,
        queue_path=queue_path,
        queue_sha256=_file_sha256(queue_path),
        evidence_path=evidence_path,
        review_dir=review_dir,
        assets_root=assets_root,
        items=tuple(completed_items),
        allowed_asset_paths=frozenset(allowed),
    )


def generated_review_id(*, supplier_id: str, asset_id: str, start: int, end: int) -> str:
    supplier = _required_text(supplier_id, "supplier_id")
    asset = _required_text(asset_id, "asset_id")
    prefix = f"{supplier}__"
    base = asset if asset.startswith(prefix) else f"{prefix}{asset}"
    return f"{base}__window_{start}_{end}"


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return []
        return [dict(row) for row in frame.to_dict(orient="records")]
    if suffix in {".parquet", ".pq"}:
        return [dict(row) for row in pd.read_parquet(path).to_dict(orient="records")]
    if suffix == ".jsonl":
        rows = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} must be an object: {path}")
            rows.append(value)
        return rows
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get("records", value.get("items", value.get("rows")))
        if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
            raise ValueError(f"JSON input must contain a list of objects: {path}")
        return [dict(row) for row in value]
    raise ValueError(f"unsupported input format {suffix!r}: {path}")


def _stage_evidence(
    rows: Iterable[dict[str, Any]],
    *,
    row: Mapping[str, Any],
    evidence_base: Path,
    assets_root: Path,
    match_kind: str,
) -> tuple[list[dict[str, Any]], set[str]]:
    review_id = str(row["review_id"])
    token = hashlib.sha256(review_id.encode("utf-8")).hexdigest()[:20]
    output: list[dict[str, Any]] = []
    allowed: set[str] = set()
    sorted_rows = sorted(rows, key=lambda value: _sort_frame(value.get("frame_idx")))
    for index, source in enumerate(sorted_rows, start=1):
        frame_idx = _int_or_none(source.get("frame_idx"))
        source_value = _text(source.get("source_path"))
        source_module = _text(source.get("source_module"))
        evidence_type = _text(source.get("evidence_type"))
        hand_side = _text(source.get("hand_side"))
        metadata = _json_field(source.get("metadata_json"), {})
        item = {
            "frame_idx": frame_idx,
            "source_path": source_value,
            "source_module": source_module,
            "evidence_type": evidence_type,
            "hand_side": hand_side,
            "metadata": metadata,
            "match_kind": match_kind,
            "status": "ready",
            "url": None,
        }
        if (
            source_module != EVIDENCE_SOURCE_MODULE
            or evidence_type != EVIDENCE_TYPE
            or hand_side not in {"", "both"}
        ):
            item["status"] = "invalid_provenance"
            output.append(item)
            continue
        if frame_idx is None or not int(row["window_start_frame"]) <= frame_idx <= int(
            row["window_end_frame"]
        ):
            item["status"] = "invalid_frame_mapping"
            output.append(item)
            continue
        source_path = Path(source_value).expanduser()
        access_path = source_path if source_path.is_absolute() else evidence_base / source_path
        access_path = access_path.resolve()
        if not access_path.is_file():
            item["status"] = "missing"
            output.append(item)
            continue
        suffix = access_path.suffix.lower() or ".bin"
        filename = f"{index:02d}_{frame_idx:09d}{suffix}"
        relative = (Path(token) / filename).as_posix()
        target = assets_root / relative
        _stage_file(access_path, target)
        item["url"] = "/assets/" + quote(relative, safe="/")
        allowed.add(relative)
        output.append(item)
    return output, allowed


def _stage_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".stage-", dir=target.parent)
    os.close(fd)
    temp = Path(temp_name)
    temp.unlink()
    try:
        try:
            os.link(source, temp)
        except OSError:
            shutil.copy2(source, temp)
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def _missing_staged_evidence(
    item: Mapping[str, Any], assets_root: Path, allowed: frozenset[str]
) -> list[str]:
    missing: list[str] = []
    evidence = item.get("evidence", [])
    if not isinstance(evidence, list):
        return ["invalid_evidence_contract"]
    root = assets_root.resolve()
    for row in evidence:
        if not isinstance(row, Mapping) or row.get("status") != "ready":
            continue
        url = row.get("url")
        if not isinstance(url, str) or not url.startswith("/assets/"):
            missing.append("invalid_url")
            continue
        relative_text = unquote(url.removeprefix("/assets/"))
        relative = Path(relative_text)
        if relative_text not in allowed or relative.is_absolute() or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            missing.append(relative_text or "invalid_url")
            continue
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            missing.append(relative_text)
            continue
        if not candidate.is_file():
            missing.append(relative_text)
    return missing


def _refresh_staged_evidence(
    item: dict[str, Any], assets_root: Path, allowed: frozenset[str]
) -> None:
    missing = set(_missing_staged_evidence(item, assets_root, allowed))
    if not missing:
        return
    for row in item.get("evidence", []):
        if not isinstance(row, dict) or row.get("status") != "ready":
            continue
        url = row.get("url")
        relative = unquote(url.removeprefix("/assets/")) if isinstance(url, str) else "invalid_url"
        if relative in missing or "invalid_url" in missing:
            row["status"] = "missing"
    item["can_review"] = False
    item["evidence_status"] = "error"
    item["evidence_error"] = "staged_evidence_missing"


def _results_csv(state: Mapping[str, Any], queue_items: Iterable[Mapping[str, Any]]) -> str:
    reviews = state.get("reviews", {})
    if not isinstance(reviews, Mapping):
        raise ReviewValidationError("saved state reviews must be an object")
    output = tempfile.SpooledTemporaryFile(mode="w+", newline="", encoding="utf-8")
    try:
        writer = csv.DictWriter(output, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for item in queue_items:
            review_id = str(item["review_id"])
            review = reviews.get(review_id)
            if not isinstance(review, Mapping):
                continue
            writer.writerow(
                {
                    "review_id": review_id,
                    "supplier_id": review["supplier_id"],
                    "asset_id": review["asset_id"],
                    "window_start_frame": review["window_start_frame"],
                    "window_end_frame": review["window_end_frame"],
                    "frame_coordinate_system": review["frame_coordinate_system"],
                    "verdict": review["verdict"],
                    "reviewer": review["reviewer"],
                    "reviewed_at": review["reviewed_at"],
                    "revision": review["revision"],
                    "source_queue_path": review["source_queue_path"],
                    "source_queue_sha256": review["source_queue_sha256"],
                    "evidence_provenance_json": json.dumps(
                        review["evidence_provenance"],
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    ),
                    "duration_resolution_status": review["duration_resolution_status"],
                }
            )
        output.seek(0)
        return output.read()
    finally:
        output.close()


def _write_text_files_atomically(save_dir: Path, files: Mapping[str, str]) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    staged: dict[str, Path] = {}
    backups: dict[str, Path] = {}
    installed: list[str] = []
    try:
        for filename, content in files.items():
            if Path(filename).name != filename:
                raise ReviewValidationError(f"unsafe save filename: {filename}")
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", newline="", dir=save_dir, delete=False
            ) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
                staged[filename] = Path(handle.name)
        for filename in files:
            target = save_dir / filename
            if not target.exists():
                continue
            with tempfile.NamedTemporaryFile(dir=save_dir, delete=False) as handle:
                backup = Path(handle.name)
            shutil.copy2(target, backup)
            backups[filename] = backup
        for filename, staged_path in staged.items():
            os.replace(staged_path, save_dir / filename)
            installed.append(filename)
        directory_fd = os.open(save_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        for filename in reversed(installed):
            target = save_dir / filename
            backup = backups.get(filename)
            try:
                if backup is not None and backup.exists():
                    os.replace(backup, target)
                else:
                    target.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    finally:
        for path in (*staged.values(), *backups.values()):
            path.unlink(missing_ok=True)


def _evidence_error(rows: list[dict[str, Any]]) -> str:
    if len(rows) != EXPECTED_EVIDENCE_COUNT:
        return f"expected_{EXPECTED_EVIDENCE_COUNT}_combined_overlays_found_{len(rows)}"
    statuses = sorted({str(row.get("status")) for row in rows if row.get("status") != "ready"})
    return ",".join(statuses) or "evidence_unavailable"


def _composite_key(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return (
        str(row["asset_id"]),
        int(row["window_start_frame"]),
        int(row["window_end_frame"]),
    )


def _evidence_composite_key(row: Mapping[str, Any], row_index: int) -> tuple[str, int, int]:
    asset_id = _required_text(row.get("asset_id"), f"evidence row {row_index} asset_id")
    return (
        asset_id,
        _strict_int(row.get("window_start_frame"), "window_start_frame"),
        _strict_int(row.get("window_end_frame"), "window_end_frame"),
    )


def _missing_to_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _text(value: Any) -> str:
    normalized = _missing_to_none(value)
    return "" if normalized is None else str(normalized).strip()


def _required_text(value: Any, field: str) -> str:
    text = _text(value)
    if not text:
        raise ValueError(f"{field} must be a non-empty string")
    return text


def _strict_int(value: Any, field: str) -> int:
    parsed = _int_or_none(value)
    if parsed is None:
        raise ValueError(f"{field} must be an integer, got {value!r}")
    return parsed


def _int_or_none(value: Any) -> int | None:
    value = _missing_to_none(value)
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    return int(number)


def _int_or(value: Any, default: int) -> int:
    parsed = _int_or_none(value)
    return default if parsed is None else parsed


def _float_or_none(value: Any) -> float | None:
    value = _missing_to_none(value)
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_field(value: Any, default: Any) -> Any:
    value = _missing_to_none(value)
    if value in (None, ""):
        return _json_copy(default)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value
    return _json_copy(value)


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False, default=str))


def _sort_frame(value: Any) -> tuple[int, int]:
    parsed = _int_or_none(value)
    return (0, parsed) if parsed is not None else (1, 0)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "DURATION_RESOLUTION_STATUS",
    "ReviewConflictError",
    "ReviewBundle",
    "ReviewValidationError",
    "Sam3WindowReviewStore",
    "generated_review_id",
    "load_review_bundle",
]
