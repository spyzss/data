"""Durable producer-artifact paths, fingerprints, and atomic promotion."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
from uuid import uuid4

from qc_common.config import LoadedQcConfig
from qc_common.contracts import EvidenceRef, Issue, ModuleResult
from qc_common.frame_survival import FrameExclusion
from qc_pipeline.context import AssetContext


RUN_CONFIG_SCHEMA = "qc_producer_run_config.v1"

_REQUIRED_FILES: dict[str, tuple[str, ...]] = {
    "precheck": (
        "check_results.json",
        "clip_aggregates.json",
        "candidate_windows.json",
        "run_config.json",
    ),
    "video_quality": ("video_quality_result.json", "run_config.json"),
    "supplier_data_audit": (
        "supplier_data_audit_result.json",
        "run_config.json",
    ),
    "sam3_containment": (
        "frame_results.json",
        "window_results.json",
        "failures.json",
        "evidence_manifest.json",
        "producer_run_config.json",
        "run_config.json",
    ),
}


@dataclass(frozen=True)
class ProducerArtifact:
    producer: str
    directory: Path
    required_files: tuple[str, ...]

    @property
    def run_config_path(self) -> Path:
        return self.directory / "run_config.json"


def _inside(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    resolved_root = root.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        raise ValueError(f"{label} must stay inside batch_root: {path}") from None
    return resolved


def artifact_for(context: AssetContext, producer: str) -> ProducerArtifact:
    if producer not in _REQUIRED_FILES:
        raise ValueError(f"unknown producer artifact: {producer}")
    module_root = context.batch_root / "module_outputs"
    directory = _inside(
        module_root / context.asset_id / producer,
        module_root,
        label="artifact path",
    )
    return ProducerArtifact(producer, directory, _REQUIRED_FILES[producer])


def file_identity(
    path: Path,
    *,
    batch_root: Path,
    declared: Mapping[str, Any] | None = None,
    allow_symlinked_sources: bool = False,
) -> dict[str, Any]:
    # Keep the declared/staged path relative to batch_root in the
    # fingerprint, while reading size/mtime from the resolved source.
    lexical_root = Path(
        os.path.abspath(os.fspath(batch_root))
    )
    candidate = (
        path
        if path.is_absolute()
        else lexical_root / path
    )
    lexical_path = Path(
        os.path.abspath(os.fspath(candidate))
    )

    try:
        relative_path = lexical_path.relative_to(lexical_root)
    except ValueError:
        raise ValueError(
            f"source path must stay inside batch_root: {path}"
        ) from None

    resolved_root = batch_root.resolve()
    resolved_path = lexical_path.resolve()

    if not allow_symlinked_sources:
        try:
            resolved_path.relative_to(resolved_root)
        except ValueError:
            raise ValueError(
                f"source path must stay inside batch_root: {path}"
            ) from None

    stat = resolved_path.stat()
    identity: dict[str, Any] = {
        "path": relative_path.as_posix(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if declared is not None:
        for key in ("checksum", "etag", "version_id"):
            value = declared.get(key)
            if value is not None and str(value).strip():
                identity[key] = str(value)
    return identity


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def module_result_from_dict(payload: Mapping[str, Any]) -> ModuleResult:
    issues = tuple(
        Issue(
            issue_id=str(item["issue_id"]),
            code=str(item["code"]),
            severity=item["severity"],
            module=str(item["module"]),
            issue_type=str(item["issue_type"]),
            metric=str(item["metric"]),
            observed_value=item.get("observed_value"),
            operator=str(item["operator"]),
            boundary_value=item.get("boundary_value"),
            rule_id=str(item["rule_id"]),
            needs_manual_review=bool(item["needs_manual_review"]),
            context=dict(item.get("context") or {}),
            evidence_ids=tuple(item.get("evidence_ids") or ()),
        )
        for item in payload.get("issues", ())
    )
    evidence = tuple(
        EvidenceRef(
            evidence_id=str(item["evidence_id"]),
            kind=str(item["kind"]),
            path=str(item["path"]),
            coordinate_system=str(item["coordinate_system"]),
            start_frame=item.get("start_frame"),
            end_frame=item.get("end_frame"),
            hand_side=item.get("hand_side"),
            checksum=item.get("checksum"),
            mime_type=item.get("mime_type"),
            generator_version=item.get("generator_version"),
        )
        for item in payload.get("evidence", ())
    )
    frame_exclusions = tuple(
        FrameExclusion(
            start_frame=int(item["start_frame"]),
            end_frame=int(item["end_frame"]),
            module=str(item["module"]),
            reason=str(item["reason"]),
            raw_severity=item["raw_severity"],
            hand_side=item.get("hand_side"),
            first_introduced_stage=str(item["first_introduced_stage"]),
            temporal_pair_start_frame=item.get("temporal_pair_start_frame"),
            temporal_pair_end_frame=item.get("temporal_pair_end_frame"),
            temporal_transition_attribution=item.get(
                "temporal_transition_attribution"
            ),
        )
        for item in payload.get("frame_exclusions", ())
    )
    return ModuleResult(
        module=str(payload["module"]),
        verdict=payload["verdict"],
        evaluation=dict(payload.get("evaluation") or {}),
        metrics=dict(payload.get("metrics") or {}),
        issues=issues,
        evidence=evidence,
        frame_exclusions=frame_exclusions,
        runtime=dict(payload.get("runtime") or {}),
    )


def config_fingerprint(
    config: LoadedQcConfig,
    module_names: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": config.schema_version,
        "config_version": config.config_version,
        "config_hash": config.sha256,
        "modules": {
            name: config.module_config(name)
            for name in module_names
        },
    }


def build_run_fingerprint(
    *,
    context: AssetContext,
    producer: str,
    config: LoadedQcConfig,
    module_names: Sequence[str],
    source_names: Sequence[str],
    implementation_version: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    for name in source_names:
        declared = context.source_files.get(name)
        if not isinstance(declared, Mapping) or not declared.get("path"):
            sources[name] = {"missing": True}
            continue
        path = context.batch_root / str(declared["path"])
        sources[name] = file_identity(
            path,
            batch_root=context.batch_root,
            declared=declared,
            allow_symlinked_sources=context.allow_symlinked_sources,
        )
    fingerprint: dict[str, Any] = {
        "producer": producer,
        "implementation_version": implementation_version,
        "asset_id": context.asset_id,
        "supplier": str(
            context.metadata.get("supplier")
            or context.metadata.get("supplier_id")
            or "unknown"
        ),
        "source_range": (
            list(context.source_range) if context.source_range is not None else None
        ),
        "sources": sources,
        "config": config_fingerprint(config, module_names),
    }
    if extra:
        fingerprint["extra"] = dict(extra)
    return fingerprint


def _valid_json_file(path: Path) -> bool:
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return True


def reusable_artifact(
    artifact: ProducerArtifact,
    expected_fingerprint: Mapping[str, Any],
) -> bool:
    try:
        run_config = json.loads(artifact.run_config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    reusable_run = (
        run_config.get("schema_version") == RUN_CONFIG_SCHEMA
        and run_config.get("producer") == artifact.producer
        and run_config.get("outcome") in {"completed", "no_candidates"}
        and run_config.get("fingerprint") == dict(expected_fingerprint)
    )
    if not reusable_run:
        return False
    for filename in artifact.required_files:
        path = artifact.directory / filename
        if not path.is_file():
            return False
        if path.suffix == ".json" and not _valid_json_file(path):
            return False
    return True


def write_run_config(
    directory: Path,
    *,
    producer: str,
    outcome: str,
    fingerprint: Mapping[str, Any],
    elapsed_seconds: float,
    created_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    payload = {
        "schema_version": RUN_CONFIG_SCHEMA,
        "producer": producer,
        "outcome": outcome,
        "fingerprint": dict(fingerprint),
        "fingerprint_sha256": canonical_sha256(fingerprint),
        "created_at": created_at
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "elapsed_seconds": float(elapsed_seconds),
    }
    if metadata:
        payload.update(dict(metadata))
    path = directory / "run_config.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


@contextmanager
def staged_artifact(artifact: ProducerArtifact) -> Iterator[Path]:
    artifact.directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{artifact.producer}.staging-",
            dir=artifact.directory.parent,
        )
    )
    try:
        yield staging
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def promote_artifact(staging: Path, artifact: ProducerArtifact) -> None:
    if staging.parent.resolve() != artifact.directory.parent.resolve():
        raise ValueError("staging and artifact directories must share a parent")
    missing = [
        filename
        for filename in artifact.required_files
        if not (staging / filename).is_file()
    ]
    if missing:
        raise ValueError(f"artifact staging is incomplete: {', '.join(missing)}")

    target = artifact.directory
    backup = target.parent / f".{artifact.producer}.backup-{uuid4().hex}"
    moved_old = False
    try:
        if target.exists():
            os.replace(target, backup)
            moved_old = True
        os.replace(staging, target)
    except Exception:
        if moved_old and backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


__all__ = [
    "ProducerArtifact",
    "RUN_CONFIG_SCHEMA",
    "artifact_for",
    "build_run_fingerprint",
    "canonical_json",
    "canonical_sha256",
    "config_fingerprint",
    "file_identity",
    "file_sha256",
    "module_result_from_dict",
    "promote_artifact",
    "reusable_artifact",
    "staged_artifact",
    "write_run_config",
]
