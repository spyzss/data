from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _validate_with_schema(instance: dict[str, Any], schema_path: Path, label: str) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(instance), key=lambda item: list(item.path))
    if not errors:
        return
    error = errors[0]
    dotted_path = ".".join(str(part) for part in error.absolute_path) or "$"
    raise ValueError(f"{label} validation failed at {dotted_path}: {error.message}")


def validate_qc_config(data: dict[str, Any]) -> None:
    _validate_with_schema(
        data,
        _repo_root() / "schemas" / "qc_acceptance_config.v1.schema.json",
        "QC config",
    )
