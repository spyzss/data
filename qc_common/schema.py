from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

from jsonschema import Draft202012Validator


class ReportValidationError(ValueError):
    """Raised when an asset QC report violates its versioned contract."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _validate_with_schema(instance: dict[str, Any], schema_path: Path, label: str) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(instance), key=lambda item: list(item.path))
    if not errors:
        return
    error = errors[0]
    dotted_path = ".".join(str(part) for part in error.absolute_path) or "$"
    raise ReportValidationError(
        f"{label} validation failed at {dotted_path}: {error.message}"
    )


def _human_validation_error(path: str, message: str) -> None:
    raise ReportValidationError(f"asset QC report validation failed at {path}: {message}")


def _is_string_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _validate_pending_edit(pending_edit: Mapping[str, Any]) -> None:
    kind = pending_edit.get("edit_type")
    if kind is None:
        kind = pending_edit.get("type")
    if kind is None:
        kind = pending_edit.get("kind")
    kind_aliases = {
        "boundary_edit": "boundary",
        "text_edit": "text",
    }
    kind = kind_aliases.get(kind, kind)
    if kind not in {"boundary", "text"}:
        _human_validation_error(
            "semantic_calibration.pending_edit.edit_type",
            "must be 'boundary' or 'text'",
        )

    affected = pending_edit.get("affected_segment_ids")
    if affected is None and "segment_id" in pending_edit:
        affected = [pending_edit.get("segment_id")]
    if not _is_string_sequence(affected) or any(
        not isinstance(item, str) or not item for item in affected
    ):
        _human_validation_error(
            "semantic_calibration.pending_edit.affected_segment_ids",
            "must be a sequence of non-empty strings",
        )

    expected_count = 2 if kind == "boundary" else 1
    if len(affected) != expected_count:
        _human_validation_error(
            "semantic_calibration.pending_edit.affected_segment_ids",
            f"{kind} pending edit must affect exactly {expected_count} segment(s)",
        )
    if kind == "boundary" and len(set(affected)) != 2:
        _human_validation_error(
            "semantic_calibration.pending_edit.affected_segment_ids",
            "boundary pending edit must affect two different segments",
        )

    for field in ("before", "after"):
        value = pending_edit.get(field)
        if kind == "boundary":
            if isinstance(value, Mapping):
                valid_boundary_snapshots = len(value) == 2
            else:
                valid_boundary_snapshots = _is_string_sequence(value) and len(value) == 2
            if not valid_boundary_snapshots:
                _human_validation_error(
                    f"semantic_calibration.pending_edit.{field}",
                    "boundary pending edit must contain exactly two snapshots",
                )
        elif not isinstance(value, Mapping) and (
            not _is_string_sequence(value) or len(value) != 1
        ):
            _human_validation_error(
                f"semantic_calibration.pending_edit.{field}",
                "text pending edit must contain one segment snapshot",
            )


def _validate_semantic_calibration(block: Mapping[str, Any]) -> None:
    required = (
        "state",
        "source_dataset_path",
        "base_hdf5_sha256",
        "final_hdf5_sha256",
        "timeline_edit_count",
        "subtask_text_edit_count",
        "pending_edit",
        "audit",
    )
    for field in required:
        if field not in block:
            _human_validation_error(f"semantic_calibration.{field}", "is required")

    pending_edit = block.get("pending_edit")
    if pending_edit is not None:
        if not isinstance(pending_edit, Mapping):
            _human_validation_error(
                "semantic_calibration.pending_edit", "must be an object or null"
            )
        _validate_pending_edit(pending_edit)


def _validate_manual_review(block: Mapping[str, Any]) -> None:
    state = block.get("state")
    # Reports produced by the automatic pipeline before a human task is
    # created retain the legacy not_evaluated shape.  Once a human state or any
    # of the new fields appears, the complete review sub-contract applies.
    human_fields = {
        "selected_issue_ids",
        "selected_issue_id",
        "issue_reviews",
        "completed_at",
    }
    is_human_block = state in {"required", "queued", "in_progress", "completed"} or bool(
        human_fields.intersection(block)
    )
    if not is_human_block:
        return

    required = (
        "selected_issue_ids",
        "selected_issue_id",
        "issue_reviews",
        "completed_at",
    )
    for field in required:
        if field not in block:
            _human_validation_error(f"manual_review.{field}", "is required")

    candidate_ids = block.get("candidate_issue_ids", [])
    selected_ids = block.get("selected_issue_ids", [])
    if not _is_string_sequence(candidate_ids) or any(
        not isinstance(item, str) or not item for item in candidate_ids
    ):
        _human_validation_error(
            "manual_review.candidate_issue_ids", "must be a sequence of non-empty strings"
        )
    if not _is_string_sequence(selected_ids) or any(
        not isinstance(item, str) or not item for item in selected_ids
    ):
        _human_validation_error(
            "manual_review.selected_issue_ids", "must be a sequence of non-empty strings"
        )
    if len(set(candidate_ids)) != len(candidate_ids):
        _human_validation_error(
            "manual_review.candidate_issue_ids", "must contain unique issue IDs"
        )
    if len(set(selected_ids)) != len(selected_ids):
        _human_validation_error(
            "manual_review.selected_issue_ids", "must contain unique issue IDs"
        )

    candidate_set = set(candidate_ids)
    selected_set = set(selected_ids)
    if not selected_set.issubset(candidate_set):
        _human_validation_error(
            "manual_review.selected_issue_ids",
            "must be a subset of candidate_issue_ids",
        )

    selected_issue_id = block.get("selected_issue_id")
    if selected_issue_id is not None and selected_issue_id not in selected_set:
        _human_validation_error(
            "manual_review.selected_issue_id",
            "must be null or one of selected_issue_ids",
        )

    reviews = block.get("issue_reviews")
    if not isinstance(reviews, Mapping):
        _human_validation_error("manual_review.issue_reviews", "must be an object")
    review_ids = set(reviews)
    if not review_ids.issubset(selected_set):
        _human_validation_error(
            "manual_review.issue_reviews",
            "may only contain selected issue IDs",
        )
    for issue_id, review in reviews.items():
        if not isinstance(issue_id, str) or not isinstance(review, Mapping):
            _human_validation_error(
                f"manual_review.issue_reviews.{issue_id}",
                "must be an object keyed by issue ID",
            )
        verdict = review.get("verdict")
        if verdict not in {"pass", "fail"}:
            _human_validation_error(
                f"manual_review.issue_reviews.{issue_id}.verdict",
                "must be 'pass' or 'fail'",
            )

    completed_at = block.get("completed_at")
    if state == "completed":
        if not isinstance(completed_at, str) or not completed_at:
            _human_validation_error(
                "manual_review.completed_at", "is required when state is completed"
            )
        if not selected_set.issubset(review_ids):
            _human_validation_error(
                "manual_review.issue_reviews",
                "completed review must cover every selected issue ID",
            )
    elif completed_at is not None:
        _human_validation_error(
            "manual_review.completed_at", "must be null before completion"
        )


def _validate_human_blocks(report: Mapping[str, Any]) -> None:
    semantic = report.get("semantic_calibration")
    if semantic is not None:
        if not isinstance(semantic, Mapping):
            _human_validation_error("semantic_calibration", "must be an object")
        semantic_contract_keys = {
            "state",
            "source_dataset_path",
            "base_hdf5_sha256",
            "final_hdf5_sha256",
            "timeline_edit_count",
            "subtask_text_edit_count",
            "pending_edit",
            "audit",
        }
        # Keep pre-contract opaque semantic extensions readable during
        # migration.  As soon as a report opts into any v2 contract key, the
        # complete strict block is required.
        if semantic_contract_keys.intersection(semantic):
            _validate_semantic_calibration(semantic)

    manual = report.get("manual_review")
    if isinstance(manual, Mapping):
        _validate_manual_review(manual)


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
    schema_version = report.get("schema_version")
    schema_paths = {
        "asset_qc_report.v1": _repo_root()
        / "schemas"
        / "asset_qc_report.v1.schema.json",
        "asset_qc_report.v2": _repo_root()
        / "schemas"
        / "asset_qc_report.v2.schema.json",
    }
    try:
        schema_path = schema_paths[schema_version]
    except KeyError as exc:
        raise ValueError(f"unknown asset QC report schema_version: {schema_version}") from exc

    _validate_with_schema(
        report,
        schema_path,
        "asset QC report",
    )
    if schema_version == "asset_qc_report.v2":
        _validate_human_blocks(report)
