from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import fcntl
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from qc_common.report_migration import migrate_v1_to_v2
from qc_common.schema import validate_asset_qc_report


class StaleReportRevisionError(RuntimeError):
    pass


@contextmanager
def _exclusive_report_lock(path: Path) -> Iterator[None]:
    """Lock a stable sidecar so replacing the report inode cannot break CAS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    lock_descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    locked = False
    try:
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        locked = True
        yield
    finally:
        try:
            if locked:
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(lock_descriptor)


def load_asset_qc_report(
    path: Path,
    *,
    migrate_to_v2: bool = False,
    config_reference: Mapping[str, str] | None = None,
    profile: str = "acceptance",
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"asset QC report root must be an object: {path}")
    if not migrate_to_v2:
        return loaded
    if loaded.get("schema_version") == "asset_qc_report.v1" and config_reference is None:
        raise ValueError("config_reference is required to migrate an asset QC report v1")
    return migrate_v1_to_v2(
        loaded,
        config_reference=config_reference or {},
        profile=profile,
    )


def write_asset_qc_report(
    path: Path,
    report: dict[str, Any],
    expected_revision: int,
    *,
    profile: str | None = None,
) -> None:
    with _exclusive_report_lock(path):
        current = load_asset_qc_report(path)
        current_revision = 0 if current is None else int(current.get("report_revision", 0))
        if current_revision != expected_revision:
            raise StaleReportRevisionError(
                f"expected revision {expected_revision}, found {current_revision}: {path}"
            )

        report_to_write = report
        config_reference = report.get("qc_config")
        if (
            report.get("schema_version") == "asset_qc_report.v1"
            and isinstance(config_reference, Mapping)
            and config_reference.get("schema_version") == "qc_acceptance_config_schema.v2"
        ):
            if profile is None:
                raise ValueError(
                    "profile is required to promote an asset QC report v1 with Config v2"
                )
            report_to_write = migrate_v1_to_v2(
                report,
                config_reference=config_reference,
                profile=profile,
            )

        next_revision = int(report_to_write.get("report_revision", 0))
        if next_revision != expected_revision + 1:
            raise ValueError(
                f"report_revision must be {expected_revision + 1}, got {next_revision}"
            )

        validate_asset_qc_report(report_to_write)
        payload = json.dumps(report_to_write, ensure_ascii=False, indent=2).encode(
            "utf-8"
        )
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            temporary_path.unlink(missing_ok=True)
