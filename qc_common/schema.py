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
    schema_version = data.get("schema_version")
    schema_paths = {
        "qc_acceptance_config_schema.v1": _repo_root()
        / "schemas"
        / "qc_acceptance_config.v1.schema.json",
        "qc_acceptance_config_schema.v2": _repo_root()
        / "schemas"
        / "qc_acceptance_config.v2.schema.json",
    }
    try:
        schema_path = schema_paths[schema_version]
    except KeyError as exc:
        raise ValueError(f"unknown QC config schema_version: {schema_version}") from exc

    _validate_with_schema(
        data,
        schema_path,
        "QC config",
    )

    if schema_version == "qc_acceptance_config_schema.v2":
        legacy_schema = json.loads(
            (_repo_root() / "schemas" / "qc_acceptance_config.v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        video_parameters_schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$ref": "#/$defs/videoParameters",
            "$defs": legacy_schema["$defs"],
        }
        errors = sorted(
            Draft202012Validator(video_parameters_schema).iter_errors(
                data["modules"]["video_quality"]["parameters"]
            ),
            key=lambda item: list(item.path),
        )
        if errors:
            error = errors[0]
            dotted_path = ".".join(str(part) for part in error.absolute_path) or "$"
            raise ValueError(
                f"QC config video parameters validation failed at {dotted_path}: {error.message}"
            )


def validate_asset_qc_report(report: dict[str, Any]) -> None:
    _validate_with_schema(
        report,
        _repo_root() / "schemas" / "asset_qc_report.v1.schema.json",
        "asset QC report",
    )
