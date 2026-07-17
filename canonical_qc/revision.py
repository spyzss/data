"""Format-neutral, immutable semantic revision artifacts for Canonical Data."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
from typing import Literal

from .contracts import CanonicalQcEpisode
from .provenance import semantic_fingerprint
from .validation import validate_episode


_TASK_PATHS = frozenset(
    {
        "semantics.task_cn",
        "semantics.task_en",
        "semantics.description_cn",
        "semantics.description_en",
    }
)
_SUBTASK_PATH = re.compile(
    r"semantics\.subtask_sequence\[(?P<index>[0-9]+)\]\."
    r"(?P<field>description_cn|description_en|start_frame|end_frame_exclusive)"
).fullmatch
_ARTIFACT_KEYS = frozenset(
    {
        "schema_version",
        "asset_id",
        "parent_canonical_revision",
        "canonical_revision",
        "source_fingerprint",
        "before_semantic_fingerprint",
        "after_semantic_fingerprint",
        "qc_report_revision",
        "reviewer",
        "reviewed_at",
        "reason",
        "edit_counts",
        "edits",
    }
)


@dataclass(frozen=True, slots=True)
class CanonicalRevisionEdit:
    path: str
    before: str | int
    after: str | int


@dataclass(frozen=True, slots=True)
class CanonicalRevisionArtifact:
    schema_version: Literal["canonical_revision_artifact.v1"]
    asset_id: str
    parent_canonical_revision: int
    canonical_revision: int
    source_fingerprint: str
    before_semantic_fingerprint: str
    after_semantic_fingerprint: str
    qc_report_revision: int
    reviewer: str
    reviewed_at: str
    reason: str
    timeline_edit_count: int
    subtask_text_edit_count: int
    edits: tuple[CanonicalRevisionEdit, ...]
    content_sha256: str


def _nonempty_string(value: object, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _positive_integer(value: object, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _sha256(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _edit_kind(path: str) -> Literal["timeline", "subtask_text"]:
    if path in _TASK_PATHS:
        return "subtask_text"
    match = _SUBTASK_PATH(path)
    if match is None:
        raise ValueError(f"revision edit path is not allowed: {path}")
    return (
        "timeline"
        if match.group("field") in {"start_frame", "end_frame_exclusive"}
        else "subtask_text"
    )


def load_revision_artifact(path: Path) -> CanonicalRevisionArtifact:
    """Read and validate one revision artifact without modifying its source."""

    artifact_path = Path(path)
    payload_bytes = artifact_path.read_bytes()
    try:
        payload = json.loads(payload_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"revision artifact is not valid UTF-8 JSON: {exc}") from exc
    if type(payload) is not dict or set(payload) != _ARTIFACT_KEYS:
        raise ValueError("revision artifact fields differ from canonical_revision_artifact.v1")
    if payload["schema_version"] != "canonical_revision_artifact.v1":
        raise ValueError("unsupported revision artifact schema_version")
    parent = _positive_integer(
        payload["parent_canonical_revision"], "parent_canonical_revision"
    )
    revision = _positive_integer(payload["canonical_revision"], "canonical_revision")
    if revision != parent + 1:
        raise ValueError("canonical revision must equal parent canonical revision + 1")
    counts = payload["edit_counts"]
    if type(counts) is not dict or set(counts) != {"timeline", "subtask_text"}:
        raise ValueError("edit_counts must contain timeline and subtask_text")
    for name, count in counts.items():
        if type(count) is not int or count < 0:
            raise ValueError(f"edit_counts.{name} must be a non-negative integer")
    rows = payload["edits"]
    if type(rows) is not list or not rows:
        raise ValueError("edits must be a non-empty array")
    edits: list[CanonicalRevisionEdit] = []
    observed_counts = {"timeline": 0, "subtask_text": 0}
    timeline_groups: dict[int, list[CanonicalRevisionEdit]] = {}
    seen_paths: set[str] = set()
    for index, row in enumerate(rows):
        if type(row) is not dict or set(row) != {"path", "before", "after"}:
            raise ValueError(f"edits[{index}] must contain path, before, and after")
        path_value = _nonempty_string(row["path"], f"edits[{index}].path")
        kind = _edit_kind(path_value)
        expected_type = int if kind == "timeline" else str
        if type(row["before"]) is not expected_type or type(row["after"]) is not expected_type:
            raise ValueError(f"edits[{index}] before/after type does not match path")
        if kind == "subtask_text" and (
            not row["before"].strip() or not row["after"].strip()
        ):
            raise ValueError(f"edits[{index}] text values must be non-empty")
        if row["before"] == row["after"]:
            raise ValueError(f"edits[{index}] must not be a no-op")
        if path_value in seen_paths:
            raise ValueError(f"duplicate revision edit path: {path_value}")
        seen_paths.add(path_value)
        edit = CanonicalRevisionEdit(path_value, row["before"], row["after"])
        edits.append(edit)
        if kind == "subtask_text":
            observed_counts[kind] += 1
        else:
            match = _SUBTASK_PATH(path_value)
            assert match is not None
            subtask_index = int(match.group("index"))
            boundary_index = (
                subtask_index + 1
                if match.group("field") == "end_frame_exclusive"
                else subtask_index
            )
            timeline_groups.setdefault(boundary_index, []).append(edit)
    for boundary_index, boundary_edits in timeline_groups.items():
        expected_paths = {
            f"semantics.subtask_sequence[{boundary_index - 1}].end_frame_exclusive",
            f"semantics.subtask_sequence[{boundary_index}].start_frame",
        }
        if (
            boundary_index < 1
            or {edit.path for edit in boundary_edits} != expected_paths
            or len({(edit.before, edit.after) for edit in boundary_edits}) != 1
        ):
            raise ValueError(
                "timeline edits must atomically pair adjacent end/start boundary patches"
            )
    observed_counts["timeline"] = len(timeline_groups)
    if observed_counts != counts:
        raise ValueError("edit_counts do not match typed revision edits")
    reviewed_at = _nonempty_string(payload["reviewed_at"], "reviewed_at")
    if not reviewed_at.endswith("Z"):
        raise ValueError("reviewed_at must be an RFC3339 UTC timestamp ending in Z")
    return CanonicalRevisionArtifact(
        schema_version="canonical_revision_artifact.v1",
        asset_id=_nonempty_string(payload["asset_id"], "asset_id"),
        parent_canonical_revision=parent,
        canonical_revision=revision,
        source_fingerprint=_sha256(payload["source_fingerprint"], "source_fingerprint"),
        before_semantic_fingerprint=_sha256(
            payload["before_semantic_fingerprint"], "before_semantic_fingerprint"
        ),
        after_semantic_fingerprint=_sha256(
            payload["after_semantic_fingerprint"], "after_semantic_fingerprint"
        ),
        qc_report_revision=_positive_integer(
            payload["qc_report_revision"], "qc_report_revision"
        ),
        reviewer=_nonempty_string(payload["reviewer"], "reviewer"),
        reviewed_at=reviewed_at,
        reason=_nonempty_string(payload["reason"], "reason"),
        timeline_edit_count=counts["timeline"],
        subtask_text_edit_count=counts["subtask_text"],
        edits=tuple(edits),
        content_sha256=hashlib.sha256(payload_bytes).hexdigest(),
    )


def apply_revision_artifact(
    episode: CanonicalQcEpisode,
    artifact: CanonicalRevisionArtifact,
    *,
    expected_canonical_revision: int,
    expected_qc_report_revision: int,
    expected_timeline_edit_count: int,
    expected_subtask_text_edit_count: int,
) -> CanonicalQcEpisode:
    """Apply an allowlisted semantic patch as a pure, CAS-checked operation."""

    if type(artifact) is not CanonicalRevisionArtifact:
        raise TypeError("artifact must be an exact CanonicalRevisionArtifact")
    if artifact.asset_id != episode.identity.asset_id:
        raise ValueError("artifact asset_id does not match Canonical Data")
    if artifact.source_fingerprint != episode.provenance.source_fingerprint:
        raise ValueError("artifact source fingerprint does not match Canonical Data")
    if artifact.canonical_revision != expected_canonical_revision:
        raise ValueError("artifact canonical revision does not match publish request")
    if artifact.qc_report_revision != expected_qc_report_revision:
        raise ValueError("artifact QC report revision does not match final report")
    if artifact.timeline_edit_count != expected_timeline_edit_count:
        raise ValueError("artifact timeline edit count does not match final report")
    if artifact.subtask_text_edit_count != expected_subtask_text_edit_count:
        raise ValueError("artifact subtask text edit count does not match final report")
    if semantic_fingerprint(episode) != artifact.before_semantic_fingerprint:
        raise ValueError("artifact before semantic fingerprint does not match Canonical Data")

    semantics = episode.semantics
    subtasks = list(semantics.subtask_sequence)
    task_updates: dict[str, str] = {}
    for edit in artifact.edits:
        if edit.path in _TASK_PATHS:
            field = edit.path.removeprefix("semantics.")
            current = getattr(semantics, field)
            if current != edit.before:
                raise ValueError(f"revision before value mismatch at {edit.path}")
            task_updates[field] = str(edit.after)
            continue
        match = _SUBTASK_PATH(edit.path)
        if match is None:
            raise ValueError(f"revision edit path is not allowed: {edit.path}")
        index = int(match.group("index"))
        if index >= len(subtasks):
            raise ValueError(f"revision subtask index is out of range: {edit.path}")
        field = match.group("field")
        current = getattr(subtasks[index], field)
        if current != edit.before:
            raise ValueError(f"revision before value mismatch at {edit.path}")
        subtasks[index] = replace(subtasks[index], **{field: edit.after})
    revised = replace(
        episode,
        semantics=replace(
            semantics,
            **task_updates,
            subtask_sequence=tuple(subtasks),
        ),
    )
    validate_episode(revised)
    if semantic_fingerprint(revised) != artifact.after_semantic_fingerprint:
        raise ValueError("artifact after semantic fingerprint does not match applied revision")
    return revised


__all__ = [
    "CanonicalRevisionArtifact",
    "CanonicalRevisionEdit",
    "apply_revision_artifact",
    "load_revision_artifact",
]
