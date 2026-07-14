"""Recoverable, all-or-nothing replacement of a scalar JSON HDF5 dataset.

The semantic review service edits a copy of an asset HDF5 and only publishes
that copy after it has been reopened, parsed through the canonical source
adapter, and compared recursively with the source file.  The source file is
never opened in write mode during preparation, so a validation or write error
cannot partially update it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import h5py
import numpy as np

from .source_adapters import Hdf5ScalarJsonSubtaskAdapter, encode_canonical_payload


class Hdf5CommitError(RuntimeError):
    """Raised when an HDF5 replacement cannot be safely prepared or committed."""


class RecoveryAction(str, Enum):
    """Action for a caller recovering an interrupted finalization transaction."""

    MARK_REPORT_COMPLETED = "mark_report_completed"
    RETRY_REPLACE = "retry_replace"
    REBUILD_STAGING = "rebuild_staging"
    CONFLICT = "conflict"

    # Readable aliases keep the action contract tolerant of older callers that
    # used shorter names while preserving one canonical enum value per branch.
    REPORT_COMPLETED = MARK_REPORT_COMPLETED
    RETRY_ATOMIC_REPLACE = RETRY_REPLACE
    REBUILD = REBUILD_STAGING
    REBUILD_REQUIRED = REBUILD_STAGING
    ERROR = CONFLICT
    CONFLICT_ERROR = CONFLICT


@dataclass(frozen=True)
class PreparedReplacement:
    """A validated same-directory HDF5 staging artifact."""

    source_path: Path
    staged_path: Path
    old_sha256: str
    new_sha256: str
    transaction_id: str
    _ownership: "_StagingOwnership | None" = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "staged_path", Path(self.staged_path))


@dataclass(frozen=True)
class FinalizingRecord:
    """Durable fields needed to recover a replacement after a process crash."""

    source_path: Path
    staged_path: Path
    old_sha256: str
    new_sha256: str
    transaction_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "staged_path", Path(self.staged_path))


@dataclass(frozen=True)
class _StagingOwnership:
    """In-process proof that a staged path came from ``prepare``."""

    path: Path
    transaction_id: str


__all__ = [
    "FinalizingRecord",
    "Hdf5CommitError",
    "PreparedReplacement",
    "RecoveryAction",
    "assert_only_dataset_changed",
    "commit_hdf5_replacement",
    "prepare_hdf5_replacement",
    "prepared_replacement_from_record",
    "recover_hdf5_replacement",
]


_SAFE_TRANSACTION_ID = re.compile(r"[^A-Za-z0-9_.-]+")


def prepare_hdf5_replacement(
    source_path: Path,
    dataset_path: str,
    payload: Mapping[str, Any],
    transaction_id: str,
) -> PreparedReplacement:
    """Create and validate a same-directory replacement copy.

    The returned staged path is owned by the caller and is consumed by
    :func:`commit_hdf5_replacement`.  Every error removes that staged path and
    leaves ``source_path`` untouched byte-for-byte.
    """

    source = Path(source_path)
    _validate_dataset_path(dataset_path)
    if not isinstance(payload, Mapping):
        raise Hdf5CommitError("payload must be a mapping")
    if not isinstance(transaction_id, str) or not transaction_id:
        raise Hdf5CommitError("transaction_id must be a non-empty string")
    if not source.is_file():
        raise Hdf5CommitError(f"source HDF5 does not exist: {source}")

    staged: Path | None = None
    try:
        old_sha256 = _sha256_file(source)
        staged = _make_staged_path(source, transaction_id)

        # copy2 is deliberately used instead of opening the source in write
        # mode; metadata and all unrelated HDF5 objects remain byte-identical
        # until the target scalar is assigned in the copy.
        shutil.copy2(source, staged)
        serialized = _serialize_payload(payload)
        _write_scalar_dataset(staged, dataset_path, serialized)

        # Reopen through the source adapter and canonical encoder.  This checks
        # UTF-8 JSON, root metadata, closed-frame boundaries, and the explicit
        # canonical field whitelist before any replacement is publishable.
        loaded = Hdf5ScalarJsonSubtaskAdapter(dataset_path).load(staged)
        canonical = encode_canonical_payload(loaded, loaded.timeline)
        if canonical != dict(payload):
            raise Hdf5CommitError(
                "payload does not round-trip to the canonical subtask schema"
            )

        assert_only_dataset_changed(source, staged, dataset_path)
        _fsync_file(staged)
        new_sha256 = _sha256_file(staged)
        return PreparedReplacement(
            source_path=source,
            staged_path=staged,
            old_sha256=old_sha256,
            new_sha256=new_sha256,
            transaction_id=transaction_id,
            _ownership=_StagingOwnership(staged, transaction_id),
        )
    except Hdf5CommitError:
        _unlink_staged(staged, source, transaction_id=transaction_id)
        raise
    except Exception as exc:
        _unlink_staged(staged, source, transaction_id=transaction_id)
        raise Hdf5CommitError(
            f"failed to prepare HDF5 replacement for {source}: {exc}"
        ) from exc


def commit_hdf5_replacement(prepared: PreparedReplacement) -> None:
    """Publish a validated staging artifact with one atomic ``os.replace``."""

    if not isinstance(prepared, PreparedReplacement):
        raise TypeError("prepared must be a PreparedReplacement")

    source = Path(prepared.source_path)
    staged = Path(prepared.staged_path)
    try:
        if not _is_owned_prepared(prepared):
            raise Hdf5CommitError(
                "staged HDF5 path is not an owned prepare() artifact"
            )
        if (
            not source.is_file()
            or not staged.is_file()
            or staged.is_symlink()
            or source.is_symlink()
        ):
            raise Hdf5CommitError("source or staged HDF5 file is missing")

        # Refuse a stale transaction rather than overwriting a source changed
        # by another writer since prepare().
        if _sha256_file(source) != prepared.old_sha256:
            raise Hdf5CommitError("source HDF5 hash changed before commit")
        if _sha256_file(staged) != prepared.new_sha256:
            raise Hdf5CommitError("staged HDF5 hash changed before commit")

        os.replace(staged, source)
        # Verify the replacement itself before making its directory entry
        # durable.  A monkeypatched or unusual replace implementation cannot
        # silently report success while leaving an unexpected source behind.
        if _sha256_file(source) != prepared.new_sha256:
            raise Hdf5CommitError("source HDF5 hash differs after replace")
        _fsync_directory(source.parent)
    except Hdf5CommitError:
        _unlink_owned_prepared(prepared)
        raise
    except Exception as exc:
        _unlink_owned_prepared(prepared)
        raise Hdf5CommitError(f"failed to commit HDF5 replacement: {exc}") from exc


def recover_hdf5_replacement(record: FinalizingRecord) -> RecoveryAction:
    """Classify an interrupted finalization without touching an unknown file.

    ``RETRY_REPLACE`` is an instruction for the caller to pass the durable
    record through :func:`prepared_replacement_from_record` and then invoke
    :func:`commit_hdf5_replacement`.  Recovery itself remains read-only for
    the old-hash branch, which avoids an implicit write in a status/check
    endpoint.  A successfully replaced source may have a leftover valid
    staging file; that artifact is removed before returning
    ``MARK_REPORT_COMPLETED``.
    """

    if not isinstance(record, FinalizingRecord):
        raise TypeError("record must be a FinalizingRecord")

    source = Path(record.source_path)
    staged = Path(record.staged_path)
    try:
        current_sha256 = _sha256_file(source)
    except Exception:
        # A missing/unreadable source cannot be proven to be either side of the
        # transaction.  Never overwrite it from a possibly unrelated staged
        # file.
        return RecoveryAction.CONFLICT

    if current_sha256 == record.new_sha256:
        if _is_managed_staged_path(source, staged, record.transaction_id) and staged.is_file():
            try:
                if _sha256_file(staged) == record.new_sha256:
                    _unlink_staged(
                        staged,
                        source,
                        transaction_id=record.transaction_id,
                    )
            except OSError:
                # The source is already the committed new bytes; failure to
                # clean an ancillary staging file must not downgrade recovery.
                pass
        return RecoveryAction.MARK_REPORT_COMPLETED

    if current_sha256 == record.old_sha256:
        if staged.is_symlink():
            return RecoveryAction.CONFLICT
        if not staged.exists():
            return RecoveryAction.REBUILD_STAGING
        if not _is_managed_staged_path(source, staged, record.transaction_id):
            return RecoveryAction.CONFLICT
        if not staged.is_file() or source.is_symlink():
            return RecoveryAction.REBUILD_STAGING
        try:
            staged_sha256 = _sha256_file(staged)
        except OSError:
            return RecoveryAction.REBUILD_STAGING
        if staged_sha256 == record.new_sha256:
            return RecoveryAction.RETRY_REPLACE
        return RecoveryAction.REBUILD_STAGING

    return RecoveryAction.CONFLICT


def prepared_replacement_from_record(
    record: FinalizingRecord,
) -> PreparedReplacement:
    """Reconstruct an owned commit value from a durable finalizing record.

    Recovery records intentionally do not serialize the private in-process
    ownership marker used to prevent arbitrary ``PreparedReplacement`` values
    from being committed.  This explicit, validating bridge is the only way
    to attach that marker after a process restart.
    """

    if not isinstance(record, FinalizingRecord):
        raise TypeError("record must be a FinalizingRecord")
    if not isinstance(record.transaction_id, str) or not record.transaction_id:
        raise Hdf5CommitError("finalizing record transaction_id must be non-empty")
    source = Path(record.source_path)
    staged = Path(record.staged_path)
    if not _is_managed_staged_path(source, staged, record.transaction_id):
        raise Hdf5CommitError(
            "staged HDF5 path is not an owned prepare() namespace artifact"
        )
    if (
        source.is_symlink()
        or staged.is_symlink()
        or not source.is_file()
        or not staged.is_file()
    ):
        raise Hdf5CommitError("source or staged HDF5 file is missing or symlinked")
    try:
        if _sha256_file(source) != record.old_sha256:
            raise Hdf5CommitError("source HDF5 hash changed before reconstruction")
        if _sha256_file(staged) != record.new_sha256:
            raise Hdf5CommitError("staged HDF5 hash is invalid")
    except Hdf5CommitError:
        raise
    except Exception as exc:
        raise Hdf5CommitError(f"unable to validate durable HDF5 staging: {exc}") from exc

    return PreparedReplacement(
        source_path=source,
        staged_path=staged,
        old_sha256=record.old_sha256,
        new_sha256=record.new_sha256,
        transaction_id=record.transaction_id,
        _ownership=_StagingOwnership(staged, record.transaction_id),
    )


def assert_only_dataset_changed(
    before: Path, after: Path, allowed_dataset_path: str
) -> None:
    """Assert recursive HDF5 structure equality except target dataset content.

    Group/dataset paths, object kinds, dtypes, shapes, attributes, and every
    non-target dataset's content must match.  The target must remain a scalar
    dataset with identical dtype/shape/attributes; only its scalar data may
    differ.  A target absent from the source may be created, together with its
    previously absent ancestor groups, but no unrelated object may be added.
    """

    _validate_dataset_path(allowed_dataset_path)
    before_path = Path(before)
    after_path = Path(after)
    try:
        with h5py.File(before_path, "r") as left, h5py.File(after_path, "r") as right:
            left_objects = _collect_objects(left)
            right_objects = _collect_objects(right)
            left_links = _collect_links(left)
            right_links = _collect_links(right)

            if allowed_dataset_path not in right_objects:
                raise Hdf5CommitError(
                    f"allowed target dataset {allowed_dataset_path!r} is missing after"
                )
            if not isinstance(right_objects[allowed_dataset_path], h5py.Dataset):
                raise Hdf5CommitError(
                    f"allowed target path {allowed_dataset_path!r} is not a dataset"
                )
            if right_objects[allowed_dataset_path].shape != ():
                raise Hdf5CommitError(
                    f"allowed target dataset {allowed_dataset_path!r} must be scalar"
                )
            if right_objects[allowed_dataset_path].dtype.kind not in {"S", "O", "U"}:
                raise Hdf5CommitError(
                    f"allowed target dataset {allowed_dataset_path!r} is not UTF-8 JSON"
                )
            if allowed_dataset_path in left_objects and not isinstance(
                left_objects[allowed_dataset_path], h5py.Dataset
            ):
                raise Hdf5CommitError(
                    f"allowed target path {allowed_dataset_path!r} was not a dataset"
                )
            if allowed_dataset_path in left_objects:
                if left_objects[allowed_dataset_path].shape != ():
                    raise Hdf5CommitError(
                        f"allowed target dataset {allowed_dataset_path!r} must be scalar"
                    )
                if left_objects[allowed_dataset_path].dtype.kind not in {"S", "O", "U"}:
                    raise Hdf5CommitError(
                        f"allowed target dataset {allowed_dataset_path!r} is not UTF-8 JSON"
                    )

            all_paths = set(left_objects) | set(right_objects)
            for path in sorted(all_paths):
                left_obj = left_objects.get(path)
                right_obj = right_objects.get(path)
                if left_obj is None:
                    if path == allowed_dataset_path or _is_target_ancestor(
                        path, allowed_dataset_path
                    ):
                        # A missing parent group can be created as part of a
                        # missing target dataset.  It must have no unrelated
                        # children (which are covered by all_paths).
                        if path != allowed_dataset_path and not isinstance(
                            right_obj, h5py.Group
                        ):
                            raise Hdf5CommitError(
                                f"new target ancestor {path!r} is not a group"
                            )
                        continue
                    raise Hdf5CommitError(f"unexpected HDF5 path added: {path}")
                if right_obj is None:
                    raise Hdf5CommitError(f"HDF5 path removed: {path}")

                if isinstance(left_obj, h5py.Group) != isinstance(right_obj, h5py.Group):
                    raise Hdf5CommitError(f"HDF5 object kind changed at {path}")
                if isinstance(left_obj, h5py.Dataset) != isinstance(right_obj, h5py.Dataset):
                    raise Hdf5CommitError(f"HDF5 object kind changed at {path}")

                if not _attrs_equal(left_obj.attrs, right_obj.attrs):
                    raise Hdf5CommitError(f"HDF5 attribute changed at {path}")

                if isinstance(left_obj, h5py.Dataset):
                    if left_obj.dtype != right_obj.dtype:
                        raise Hdf5CommitError(f"HDF5 dtype changed at {path}")
                    if left_obj.shape != right_obj.shape:
                        raise Hdf5CommitError(f"HDF5 shape changed at {path}")
                    if path != allowed_dataset_path and not _data_equal(
                        left_obj[()], right_obj[()]
                    ):
                        raise Hdf5CommitError(
                            f"non-target dataset content changed at {path}"
                        )

            _assert_links_unchanged(
                left_links,
                right_links,
                allowed_dataset_path=allowed_dataset_path,
            )
    except Hdf5CommitError:
        raise
    except Exception as exc:
        raise Hdf5CommitError(f"failed to compare HDF5 files: {exc}") from exc


def _validate_dataset_path(dataset_path: str) -> None:
    if not isinstance(dataset_path, str) or not dataset_path.startswith("/"):
        raise Hdf5CommitError("dataset_path must be an absolute HDF5 path")
    if dataset_path == "/" or dataset_path.endswith("/") or "//" in dataset_path:
        raise Hdf5CommitError("dataset_path must name a dataset")


def _serialize_payload(payload: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except Exception as exc:
        raise Hdf5CommitError(f"payload is not UTF-8 JSON serializable: {exc}") from exc


def _write_scalar_dataset(path: Path, dataset_path: str, serialized: bytes) -> None:
    with h5py.File(path, "r+") as handle:
        if dataset_path in handle:
            dataset = handle[dataset_path]
            if not isinstance(dataset, h5py.Dataset):
                raise Hdf5CommitError(
                    f"target path {dataset_path!r} is not a dataset"
                )
            if dataset.shape != ():
                raise Hdf5CommitError(
                    f"target dataset {dataset_path!r} must be scalar"
                )
            _assign_scalar_json(dataset, serialized, dataset_path)
        else:
            parent, _, name = dataset_path.rpartition("/")
            group = handle.require_group(parent.strip("/")) if parent else handle
            dataset = group.create_dataset(
                name,
                shape=(),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            dataset[()] = serialized
        handle.flush()


def _assign_scalar_json(dataset: h5py.Dataset, serialized: bytes, path: str) -> None:
    kind = dataset.dtype.kind
    if kind == "S":
        itemsize = dataset.dtype.itemsize
        if itemsize and len(serialized) > itemsize:
            raise Hdf5CommitError(
                f"UTF-8 JSON payload exceeds fixed-width dtype at {path!r}"
            )
        value: object = serialized
    elif kind == "O":
        # h5py's variable-length UTF-8 string dtype is represented as object.
        value = serialized
    elif kind == "U":
        value = serialized.decode("utf-8")
    else:
        raise Hdf5CommitError(
            f"target dataset {path!r} is not a UTF-8 string scalar (dtype={dataset.dtype})"
        )
    try:
        dataset[()] = value
    except Exception as exc:
        raise Hdf5CommitError(f"failed to write target dataset {path!r}: {exc}") from exc


def _make_staged_path(source: Path, transaction_id: str) -> Path:
    safe_id = _safe_transaction_id(transaction_id)
    prefix = f".{source.name}.human-qc-{safe_id}-"
    fd, name = tempfile.mkstemp(prefix=prefix, dir=source.parent)
    staged = Path(name)
    try:
        os.close(fd)
    except Exception:
        _unlink_quiet(staged)
        raise
    return staged


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _unlink_quiet(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _safe_transaction_id(transaction_id: str) -> str:
    return _SAFE_TRANSACTION_ID.sub("_", transaction_id).strip(".") or "tx"


def _is_managed_staged_path(
    source: Path, staged: Path, transaction_id: str
) -> bool:
    """Recognize a temp path belonging to this source/transaction namespace."""

    source = Path(source)
    staged = Path(staged)
    try:
        if source.parent.resolve() != staged.parent.resolve():
            return False
        if source.resolve() == staged.resolve():
            return False
    except OSError:
        return False
    if staged.is_symlink():
        return False
    safe_id = _safe_transaction_id(transaction_id) if transaction_id else ""
    prefix = f".{source.name}.human-qc-"
    if safe_id:
        prefix += f"{safe_id}-"
    if not staged.name.startswith(prefix):
        return False
    return bool(staged.name[len(prefix) :])


def _is_owned_prepared(prepared: PreparedReplacement) -> bool:
    ownership = prepared._ownership
    if not isinstance(ownership, _StagingOwnership):
        return False
    if ownership.path != Path(prepared.staged_path):
        return False
    if ownership.transaction_id != prepared.transaction_id:
        return False
    return _is_managed_staged_path(
        Path(prepared.source_path),
        Path(prepared.staged_path),
        prepared.transaction_id,
    )


def _unlink_staged(
    staged: Path | None,
    source: Path,
    *,
    transaction_id: str,
) -> None:
    """Remove only a managed staging artifact, never source/foreign files."""

    if staged is None:
        return
    staged = Path(staged)
    if not _is_managed_staged_path(Path(source), staged, transaction_id):
        return
    _unlink_quiet(staged)


def _unlink_owned_prepared(prepared: PreparedReplacement) -> None:
    if not _is_owned_prepared(prepared):
        return
    _unlink_staged(
        Path(prepared.staged_path),
        Path(prepared.source_path),
        transaction_id=prepared.transaction_id,
    )


def _collect_objects(handle: h5py.File) -> dict[str, h5py.Group | h5py.Dataset]:
    objects: dict[str, h5py.Group | h5py.Dataset] = {"/": handle}

    def visit(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        objects["/" + name] = obj

    handle.visititems(visit)
    return objects


def _collect_links(handle: h5py.File) -> dict[str, tuple[object, ...]]:
    links: dict[str, tuple[object, ...]] = {}

    def visit(name: str, link: object) -> None:
        links["/" + name] = _link_descriptor(link)

    # ``visititems_links`` includes dangling soft/external links and hard
    # links separately from their targets, which is required for a strict
    # path-level diff.
    handle.visititems_links(visit)
    return links


def _link_descriptor(link: object) -> tuple[object, ...]:
    if isinstance(link, h5py.HardLink):
        return ("hard",)
    if isinstance(link, h5py.SoftLink):
        return ("soft", link.path)
    if isinstance(link, h5py.ExternalLink):
        return ("external", link.filename, link.path)
    return (type(link).__name__, repr(link))


def _assert_links_unchanged(
    before: dict[str, tuple[object, ...]],
    after: dict[str, tuple[object, ...]],
    *,
    allowed_dataset_path: str,
) -> None:
    all_paths = set(before) | set(after)
    for path in sorted(all_paths):
        old = before.get(path)
        new = after.get(path)
        if old is None:
            # Creating a missing target also creates a hard link for the
            # target and, when needed, hard links for ancestor groups.  No
            # soft/external link can be part of that sanctioned change.
            if (
                path == allowed_dataset_path or _is_target_ancestor(path, allowed_dataset_path)
            ) and new is not None and new[0] == "hard":
                continue
            raise Hdf5CommitError(f"HDF5 link/path added at {path}")
        if new is None:
            raise Hdf5CommitError(f"HDF5 link/path removed at {path}")
        if old != new:
            raise Hdf5CommitError(f"HDF5 link changed at {path}")


def _is_target_ancestor(path: str, target: str) -> bool:
    return target.startswith(path.rstrip("/") + "/")


def _attrs_equal(left: h5py.AttributeManager, right: h5py.AttributeManager) -> bool:
    if set(left.keys()) != set(right.keys()):
        return False
    for key in left.keys():
        left_value = left[key]
        right_value = right[key]
        if _attribute_dtype_shape(left_value) != _attribute_dtype_shape(right_value):
            return False
        if not _data_equal(left_value, right_value):
            return False
    return True


def _attribute_dtype_shape(value: object) -> tuple[object, tuple[int, ...]]:
    if isinstance(value, h5py.Empty):
        return ("empty", np.dtype(value.dtype)), ()
    array = np.asarray(value)
    return array.dtype, tuple(array.shape)


def _data_equal(left: object, right: object) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return bool(np.array_equal(left, right, equal_nan=True))
        except TypeError:
            return bool(np.array_equal(left, right))
    if isinstance(left, np.generic) or isinstance(right, np.generic):
        try:
            return bool(
                np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True)
            )
        except TypeError:
            try:
                return bool(np.asarray(left) == np.asarray(right))
            except (TypeError, ValueError):
                return False
        except ValueError:
            return False
    try:
        result = left == right
        return bool(result)
    except (TypeError, ValueError):
        return False
