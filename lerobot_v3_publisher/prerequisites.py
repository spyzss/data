"""Read-only prerequisites for one immutable Curated LeRobot v3 release."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, NoReturn

from canonical_qc.contracts import SourceFile
from canonical_qc.provenance import semantic_fingerprint, source_fingerprint
from canonical_qc.validation import validate_episode
from qc_common.config import load_qc_acceptance_config
from qc_common.schema import validate_asset_qc_report

from .contracts import (
    CanonicalDiagnostic,
    PUBLISHER_VERSION,
    PublishPlan,
    PublishPrerequisiteError,
    PublishRequest,
    SourceSnapshot,
)
from .layout import derive_release_id, layout_for


_STAGE = "publish_prerequisite"


def _reject(
    field: str | None,
    message: str,
    *,
    code: str = "publish_prerequisite_failed",
    retryable: bool = False,
) -> NoReturn:
    raise PublishPrerequisiteError(
        CanonicalDiagnostic(code, _STAGE, field, message, retryable)
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class _FileSnapshot:
    payload: bytes
    size_bytes: int
    sha256: str
    device: int
    inode: int


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _snapshot_file(
    path: Path,
    *,
    field: str,
    code: str,
    retryable_io: bool,
) -> _FileSnapshot:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        _reject(
            field,
            f"cannot open file without following symlinks: {exc}",
            code=code,
            retryable=retryable_io,
        )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            _reject(field, "must be a regular file", code=code)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        try:
            current_path = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            _reject(
                field,
                f"file path changed while hashing: {exc}",
                code=code,
                retryable=True,
            )
        if (
            _stat_identity(before) != _stat_identity(after)
            or _stat_identity(before) != _stat_identity(current_path)
        ):
            _reject(
                field,
                "file changed while hashing",
                code=code,
                retryable=True,
            )
        payload = b"".join(chunks)
        return _FileSnapshot(
            payload=payload,
            size_bytes=before.st_size,
            sha256=_sha256_bytes(payload),
            device=before.st_dev,
            inode=before.st_ino,
        )
    finally:
        os.close(descriptor)


def _has_symlink_component(path: Path) -> bool:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _safe_absolute_path(
    path: Path,
    field: str,
    *,
    kind: str,
    must_exist: bool,
) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or ".." in candidate.parts:
        _reject(field, "must be an absolute path without traversal components")
    if _has_symlink_component(candidate):
        _reject(field, "symlink path components are forbidden")
    if must_exist:
        if kind == "file" and not candidate.is_file():
            _reject(field, "must name an existing regular file")
        if kind == "directory" and not candidate.is_dir():
            _reject(field, "must name an existing directory")
    elif candidate.exists() and kind == "directory" and not candidate.is_dir():
        _reject(field, "must be a directory when it already exists")
    return candidate


def _safe_relative_source_path(value: str, index: int) -> PurePosixPath:
    field = f"provenance.source_files[{index}]"
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "." in path.parts:
        _reject(
            field,
            "relative_path must be a normalized relative POSIX path",
            code="source_integrity_error",
        )
    return path


def _source_snapshot(request: PublishRequest) -> tuple[SourceSnapshot, ...]:
    root = _safe_absolute_path(
        request.canonical_source_root,
        "canonical_source_root",
        kind="directory",
        must_exist=True,
    )
    root_resolved = root.resolve(strict=True)
    observed: list[SourceFile] = []
    snapshot: list[SourceSnapshot] = []
    for index, source in enumerate(request.episode.provenance.source_files):
        field = f"provenance.source_files[{index}]"
        relative = _safe_relative_source_path(source.relative_path, index)
        path = root.joinpath(*relative.parts)
        if _has_symlink_component(path):
            _reject(
                field,
                "source paths may not contain symlinks",
                code="source_integrity_error",
            )
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root_resolved)
        except (FileNotFoundError, ValueError):
            _reject(
                field,
                "source path is missing or escapes canonical_source_root",
                code="source_integrity_error",
            )
        if not resolved.is_file():
            _reject(
                field,
                "source path is not a regular file",
                code="source_integrity_error",
            )
        file_snapshot = _snapshot_file(
            resolved,
            field=field,
            code="source_integrity_error",
            retryable_io=True,
        )
        if (
            file_snapshot.size_bytes != source.size_bytes
            or file_snapshot.sha256 != source.sha256
        ):
            _reject(
                field,
                "current source size/hash differs from the Canonical provenance",
                code="source_integrity_error",
            )
        observed.append(
            SourceFile(
                source.relative_path,
                source.role,
                file_snapshot.size_bytes,
                file_snapshot.sha256,
            )
        )
        snapshot.append(
            SourceSnapshot(
                source.relative_path,
                source.role,
                file_snapshot.size_bytes,
                file_snapshot.sha256,
            )
        )

    current_fingerprint = source_fingerprint(
        observed,
        source_schema_version=request.episode.identity.source_schema_version,
        adapter_id=request.episode.provenance.adapter_id,
        adapter_version=request.episode.provenance.adapter_version,
        main_video_source_frame_range=request.episode.main_video.source_frame_range,
    )
    if current_fingerprint != request.episode.provenance.source_fingerprint:
        _reject(
            "provenance.source_fingerprint",
            "current source fingerprint differs from the Canonical episode",
            code="source_integrity_error",
        )
    return tuple(snapshot)


def _read_report(request: PublishRequest) -> tuple[dict[str, Any], bytes]:
    path = _safe_absolute_path(
        request.qc_report_path,
        "qc_report_path",
        kind="file",
        must_exist=True,
    )
    file_snapshot = _snapshot_file(
        path,
        field="qc_report_path",
        code="publish_prerequisite_failed",
        retryable_io=True,
    )
    try:
        payload = file_snapshot.payload
        loaded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _reject("qc_report_path", f"cannot read valid QC JSON: {exc}", retryable=True)
    if not isinstance(loaded, dict):
        _reject("qc_report_path", "QC report root must be an object")
    return loaded, payload


def _assert_non_overlapping_roots(request: PublishRequest) -> None:
    source_root = _safe_absolute_path(
        request.canonical_source_root,
        "canonical_source_root",
        kind="directory",
        must_exist=True,
    ).resolve(strict=True)
    release_root = _safe_absolute_path(
        request.release_root,
        "release_root",
        kind="directory",
        must_exist=False,
    ).absolute()
    report_path = _safe_absolute_path(
        request.qc_report_path,
        "qc_report_path",
        kind="file",
        must_exist=True,
    ).resolve(strict=True)

    def contains(parent: Path, child: Path) -> bool:
        try:
            child.relative_to(parent)
        except ValueError:
            return False
        return True

    if contains(source_root, release_root) or contains(release_root, source_root):
        _reject(
            "release_root",
            "must not equal, contain, or be inside canonical_source_root",
        )
    if contains(release_root, report_path):
        _reject("release_root", "must not contain the QC report")

    report_stat = os.stat(report_path, follow_symlinks=False)
    for index, source in enumerate(request.episode.provenance.source_files):
        relative = _safe_relative_source_path(source.relative_path, index)
        source_path = source_root.joinpath(*relative.parts)
        try:
            source_stat = os.stat(source_path, follow_symlinks=False)
        except OSError as exc:
            _reject(
                f"provenance.source_files[{index}]",
                f"cannot stat source path: {exc}",
                code="source_integrity_error",
                retryable=True,
            )
        if (report_stat.st_dev, report_stat.st_ino) == (
            source_stat.st_dev,
            source_stat.st_ino,
        ):
            _reject("qc_report_path", "must not alias a Canonical source file")


def _mapping(
    parent: Mapping[str, Any],
    name: str,
    field: str | None = None,
) -> Mapping[str, Any]:
    value = parent.get(name)
    if not isinstance(value, Mapping):
        _reject(field or name, "must be an object")
    return value


def _verified_config(report: Mapping[str, Any]):
    reference = _mapping(report, "qc_config")
    version = reference.get("config_version")
    if (
        not isinstance(version, str)
        or re.fullmatch(r"qc_acceptance_v\d+\.\d+\.\d+", version) is None
    ):
        _reject("qc_config.config_version", "must name an immutable QC config version")
    root = Path(__file__).resolve().parents[1]
    snapshot_path = root / "configs" / "qc_acceptance" / f"{version}.yaml"
    try:
        loaded = load_qc_acceptance_config(snapshot_path)
        loaded.assert_same_reference(reference)  # type: ignore[arg-type]
    except (OSError, TypeError, ValueError) as exc:
        text = str(exc)
        field = "qc_config.config_hash" if "config_hash" in text else "qc_config"
        _reject(field, f"immutable QC config verification failed: {exc}")
    if reference.get("config_name") != loaded.config_name:
        _reject("qc_config.config_name", "does not match immutable config snapshot")
    return loaded


def _validate_module_states(report: Mapping[str, Any], config: Any) -> None:
    execution = _mapping(report, "execution")
    states = _mapping(execution, "module_states", "execution.module_states")
    for module in config.pipeline_modules:
        state_row = states.get(module)
        if not isinstance(state_row, Mapping):
            _reject(f"execution.module_states.{module}", "final module state is missing")
        state = state_row.get("state")
        enabled = bool(config.module_config(module)["enabled"])
        allowed = {"completed"}
        if module == "quality_hand":
            allowed.add("skipped")
        if not enabled:
            allowed = {"disabled"}
        if state not in allowed:
            _reject(
                f"execution.module_states.{module}",
                f"final state must be one of {sorted(allowed)}, got {state!r}",
            )
        module_config = config.module_config(module)
        if not enabled or module_config.get("execution_kind") == "external":
            continue
        block = report.get(module)
        if not isinstance(block, Mapping):
            _reject(module, "enabled automatic module result block is missing")
        flow = _mapping(block, "flow", f"{module}.flow")
        _mapping(flow, "entry_gate", f"{module}.flow.entry_gate")
        result_gate = _mapping(
            flow,
            "result_gate",
            f"{module}.flow.result_gate",
        )
        _mapping(flow, "exit_gate", f"{module}.flow.exit_gate")
        verdict = result_gate.get("verdict")
        if verdict == "fail" or result_gate.get("has_fail") is True:
            _reject(
                f"{module}.flow.result_gate.verdict",
                "final pass release cannot contain an automatic fail Gate",
            )
        if verdict not in {"pass", "warn", "skipped"}:
            _reject(
                f"{module}.flow.result_gate.verdict",
                "must be pass, warn, or skipped",
            )
        expected_state = "skipped" if verdict == "skipped" else "completed"
        if state != expected_state:
            _reject(
                f"execution.module_states.{module}",
                f"state {state!r} conflicts with result Gate {verdict!r}",
            )


def _validate_manual_review(report: Mapping[str, Any]) -> None:
    manual = _mapping(report, "manual_review")
    state = manual.get("state")
    candidates = manual.get("candidate_issue_ids")
    if not isinstance(candidates, list) or any(
        not isinstance(item, str) for item in candidates
    ):
        _reject("manual_review.candidate_issue_ids", "must be a string array")
    if len(candidates) != len(set(candidates)):
        _reject("manual_review.candidate_issue_ids", "must not contain duplicates")
    selected = manual.get("selected_issue_ids")
    if not isinstance(selected, list) or any(
        not isinstance(item, str) for item in selected
    ):
        _reject("manual_review.selected_issue_ids", "must be a string array")
    if len(selected) != len(set(selected)):
        _reject("manual_review.selected_issue_ids", "must not contain duplicates")
    issues = report.get("issues")
    if not isinstance(issues, list):
        _reject("issues", "must be an array")
    issue_ids: list[str] = []
    for issue in issues:
        if (
            not isinstance(issue, Mapping)
            or not isinstance(issue.get("issue_id"), str)
            or not issue.get("issue_id")
            or issue.get("severity") not in {"warn", "fail"}
            or type(issue.get("needs_manual_review")) is not bool
        ):
            _reject("issues", "each issue must have an ID, severity, and review flag")
        issue_ids.append(issue["issue_id"])
        if issue.get("severity") == "fail":
            _reject("issues", "a pass release cannot contain fail issues")
    if len(issue_ids) != len(set(issue_ids)):
        _reject("issues", "issue IDs must be unique")
    review_required = {
        issue.get("issue_id")
        for issue in issues
        if isinstance(issue, Mapping)
        and issue.get("severity") == "warn"
        and issue.get("needs_manual_review") is True
        and isinstance(issue.get("issue_id"), str)
    }
    if set(candidates) != review_required:
        _reject(
            "manual_review.candidate_issue_ids",
            "must exactly match warn issues requiring human review",
        )
    failures = manual.get("failures_for_batch_stats_issue_ids")
    if failures != []:
        _reject(
            "manual_review.failures_for_batch_stats_issue_ids",
            "must be empty for an overall pass release",
        )
    if state == "not_required":
        if (
            manual.get("required") is not False
            or candidates
            or selected
            or manual.get("reviews") != []
        ):
            _reject("manual_review.state", "not_required requires no candidates")
        return
    if state != "completed" or manual.get("required") is not True or not candidates:
        _reject(
            "manual_review.state",
            "must be completed for warn candidates or not_required when empty",
        )
    if set(selected) != set(candidates):
        _reject(
            "manual_review.selected_issue_ids",
            "final review must cover every candidate issue",
        )
    reviews = manual.get("reviews")
    if not isinstance(reviews, list):
        _reject("manual_review.reviews", "must be an array")
    reviewed_issue_ids: list[str] = []
    review_ids: list[str] = []
    for index, review in enumerate(reviews):
        prefix = f"manual_review.reviews[{index}]"
        if not isinstance(review, Mapping):
            _reject(prefix, "must be an object")
        for field in ("review_id", "issue_id", "reviewer", "reviewed_at"):
            if not isinstance(review.get(field), str) or not review.get(field):
                _reject(f"{prefix}.{field}", "must be a non-empty string")
        if review.get("verdict") not in {
            "accept_issue",
            "reject_issue",
            "unable_to_determine",
        }:
            _reject(f"{prefix}.verdict", "unsupported manual issue verdict")
        action = review.get("asset_action")
        if action not in {
            "accept",
            "accept_with_risk",
            "reject",
            "return_for_rework",
        }:
            _reject(f"{prefix}.asset_action", "unsupported final asset action")
        if action not in {"accept", "accept_with_risk"}:
            _reject(
                f"{prefix}.asset_action",
                "final asset action does not permit publication",
            )
        if not isinstance(review.get("comment"), str):
            _reject(f"{prefix}.comment", "must be a string")
        evidence_paths = review.get("evidence_paths")
        if not isinstance(evidence_paths, list) or any(
            not isinstance(item, str) for item in evidence_paths
        ):
            _reject(f"{prefix}.evidence_paths", "must be a string array")
        reviewed_issue_ids.append(review["issue_id"])
        review_ids.append(review["review_id"])
    if len(review_ids) != len(set(review_ids)):
        _reject("manual_review.reviews", "review IDs must be unique")
    if len(reviewed_issue_ids) != len(set(reviewed_issue_ids)):
        _reject("manual_review.reviews", "each candidate may be reviewed only once")
    if set(reviewed_issue_ids) != set(candidates):
        _reject("manual_review.reviews", "must cover every candidate exactly once")


def _validated_report_binding(
    request: PublishRequest,
    report: Mapping[str, Any],
) -> str:
    if report.get("schema_version") != "asset_qc_report.v2":
        _reject("schema_version", "Publisher requires native asset_qc_report.v2")
    identity = request.episode.identity
    if report.get("asset_id") != identity.asset_id:
        _reject("asset_id", "does not match the Canonical episode")
    if report.get("supplier_id") != identity.supplier_id:
        _reject("supplier_id", "does not match the Canonical episode")
    revision = report.get("report_revision")
    if revision != request.expected_report_revision:
        _reject(
            "report_revision",
            f"expected {request.expected_report_revision}, found {revision!r}",
            retryable=True,
        )
    pipeline = _mapping(report, "pipeline_state")
    if pipeline.get("status") != "completed":
        _reject("pipeline_state.status", "must be completed")
    if report.get("overall_decision") != "pass":
        _reject("overall_decision", "must be pass")
    execution = _mapping(report, "execution")
    if execution.get("profile") != "acceptance":
        _reject("execution.profile", "only the acceptance profile is publishable")
    if pipeline.get("next_module") is not None:
        _reject("pipeline_state.next_module", "must be null after completion")
    if pipeline.get("stop_reason") is not None:
        _reject("pipeline_state.stop_reason", "must be null after successful completion")
    if report.get("runtime_errors") != []:
        _reject("runtime_errors", "must be empty")

    config = _verified_config(report)
    if pipeline.get("last_completed_module") != config.pipeline_modules[-1]:
        _reject(
            "pipeline_state.last_completed_module",
            "must equal the final configured pipeline module",
        )
    _validate_module_states(report, config)
    semantic_stage = _mapping(report, "semantic_consistency")
    if semantic_stage.get("state") != "completed":
        _reject("semantic_consistency.state", "must be completed")
    semantic_calibration = _mapping(report, "semantic_calibration")
    if semantic_calibration.get("state") != "completed":
        _reject("semantic_calibration.state", "must be completed")
    _validate_manual_review(report)

    if type(request.canonical_revision) is not int or request.canonical_revision < 1:
        _reject("canonical_revision", "must be a positive integer")
    if (
        type(request.expected_report_revision) is not int
        or request.expected_report_revision < 1
    ):
        _reject("expected_report_revision", "must be a positive integer")
    binding = _mapping(report, "canonical_binding")
    if binding.get("schema_version") != "canonical_publish_binding.v1":
        _reject("canonical_binding.schema_version", "unsupported binding schema")
    if binding.get("canonical_revision") != request.canonical_revision:
        _reject("canonical_binding.canonical_revision", "does not match the request")
    if semantic_calibration.get("canonical_revision") != request.canonical_revision:
        _reject("semantic_calibration.canonical_revision", "does not match the request")
    if binding.get("qc_report_revision") != revision:
        _reject(
            "canonical_binding.qc_report_revision",
            "must bind this canonical revision to the final QC report revision",
        )
    try:
        validate_episode(request.episode)
        current_semantic = semantic_fingerprint(request.episode)
    except (TypeError, ValueError) as exc:
        _reject("episode", f"Canonical episode validation failed: {exc}")
    if binding.get("semantic_fingerprint") != current_semantic:
        _reject(
            "canonical_binding.semantic_fingerprint",
            "does not match the Canonical episode",
        )
    if binding.get("source_fingerprint") != request.episode.provenance.source_fingerprint:
        _reject(
            "canonical_binding.source_fingerprint",
            "does not match the Canonical episode",
        )
    report_provenance = _mapping(
        _mapping(report, "source_files"),
        "canonical_provenance",
        "source_files.canonical_provenance",
    )
    if report_provenance.get("source_fingerprint") != request.episode.provenance.source_fingerprint:
        _reject(
            "source_files.canonical_provenance.source_fingerprint",
            "does not match the Canonical episode",
        )
    qc_range = _mapping(report, "canonical_qc_range")
    expected_range = {
        "start_frame": 0,
        "end_frame_exclusive": request.episode.time_axis.frame_count,
        "interval_semantics": "half_open",
    }
    if dict(qc_range) != expected_range:
        _reject(
            "canonical_qc_range",
            "Publisher v1 requires the full Canonical [0,T) range",
        )
    try:
        validate_asset_qc_report(dict(report))
    except (TypeError, ValueError) as exc:
        _reject("qc_report_path", f"asset QC report schema validation failed: {exc}")
    return current_semantic


def _build_plan(request: PublishRequest) -> PublishPlan:
    _assert_non_overlapping_roots(request)
    report, report_payload = _read_report(request)
    semantic_hash = _validated_report_binding(request, report)
    snapshot = _source_snapshot(request)
    release_id = derive_release_id(
        asset_id=request.episode.identity.asset_id,
        canonical_revision=request.canonical_revision,
        semantic_fingerprint=semantic_hash,
        publisher_version=PUBLISHER_VERSION,
    )
    layout = layout_for(request.release_root, release_id)
    return PublishPlan(
        request=request,
        release_id=release_id,
        publisher_version=PUBLISHER_VERSION,
        release_path=layout.release_path,
        current_path=layout.current_path,
        semantic_fingerprint=semantic_hash,
        source_fingerprint=request.episode.provenance.source_fingerprint,
        qc_report_revision=request.expected_report_revision,
        qc_report_sha256=_sha256_bytes(report_payload),
        source_snapshot=snapshot,
    )


def validate_publish_request(request: PublishRequest) -> PublishPlan:
    """Return an immutable plan only when all current inputs are publishable."""
    if type(request) is not PublishRequest:
        _reject("request", "must be an exact PublishRequest")
    return _build_plan(request)


def release_id_for(request: PublishRequest) -> str:
    """Return the ID only after validating its report-bound revision inputs."""
    return validate_publish_request(request).release_id


def revalidate_publish_plan(plan: PublishPlan) -> PublishPlan:
    """Recheck CAS and source state before later write/validate/commit phases."""
    current = validate_publish_request(plan.request)
    if current.qc_report_sha256 != plan.qc_report_sha256:
        _reject(
            "qc_report_sha256",
            "QC report content changed after planning",
            retryable=True,
        )
    if current != plan:
        _reject("publish_plan", "validated plan changed after planning", retryable=True)
    return current


__all__ = ["release_id_for", "revalidate_publish_plan", "validate_publish_request"]
