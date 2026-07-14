from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from qc_common.schema import validate_asset_qc_report


class StaleReportRevisionError(RuntimeError):
    pass


def load_asset_qc_report(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"asset QC report root must be an object: {path}")
    return loaded


def write_asset_qc_report(path: Path, report: dict[str, Any], expected_revision: int) -> None:
    current = load_asset_qc_report(path)
    current_revision = 0 if current is None else int(current.get("report_revision", 0))
    if current_revision != expected_revision:
        raise StaleReportRevisionError(f"expected revision {expected_revision}, found {current_revision}: {path}")

    next_revision = int(report.get("report_revision", 0))
    if next_revision != expected_revision + 1:
        raise ValueError(f"report_revision must be {expected_revision + 1}, got {next_revision}")

    validate_asset_qc_report(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
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
    finally:
        temporary_path.unlink(missing_ok=True)
