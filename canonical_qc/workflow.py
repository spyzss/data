"""Package API for explicit-source Canonical ingest and resumable QC."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import stat
from typing import Callable, Literal

from qc_common.config import load_qc_acceptance_config
from qc_common.module_registry import ModuleRegistry
from qc_pipeline.default_registry import build_default_registry
from qc_pipeline.orchestrator import RunOutcome, run_asset

from .adapters import StandardHdf5Adapter, StandardLeRobotAdapter
from .bridge import CanonicalQcBridge
from .config import LoadedCanonicalQcConfig, load_canonical_qc_config
from .contracts import CanonicalQcEpisode
from .errors import CanonicalInputError


SourceFormat = Literal["hdf5", "lerobot"]


@dataclass(frozen=True, slots=True)
class CanonicalQcRunResult:
    episode: CanonicalQcEpisode
    source: Path
    source_format: SourceFormat
    report_path: Path
    status: str
    overall_decision: str | None
    report_revision: int | None
    executed_modules: tuple[str, ...]
    resumed: bool
    dry_run: bool
    canonical_config_path: Path
    canonical_config_version: str
    canonical_config_hash: str
    qc_config_path: Path
    qc_config_version: str
    qc_config_hash: str
    runtime_error: dict[str, object] | None


def _resolved_inside(path: Path, root: Path, *, field: str) -> Path:
    explicit = validate_explicit_path(path, field=field)
    explicit_root = validate_explicit_path(root, field=f"{field}_root")
    resolved = explicit.resolve()
    resolved_root = explicit_root.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        raise CanonicalInputError(
            "path_outside_root", field, f"{resolved} is outside {resolved_root}"
        ) from None
    return resolved


def validate_explicit_path(path: Path, *, field: str) -> Path:
    """Reject lexical traversal and every existing symlink before normalization."""

    explicit = Path(path).expanduser()
    if not explicit.is_absolute():
        raise CanonicalInputError(
            "path_not_absolute", field, "must be an explicit absolute path"
        )
    if ".." in explicit.parts:
        raise CanonicalInputError(
            "path_traversal", field, "must not contain '..' components"
        )
    current = Path(explicit.anchor)
    for component in explicit.parts[1:]:
        current = current / component
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            break
        except OSError as exc:
            raise CanonicalInputError(
                "source_integrity_error",
                field,
                f"cannot inspect path component: {exc}",
                retryable=True,
            ) from exc
        if stat.S_ISLNK(mode):
            raise CanonicalInputError(
                "path_symlink", field, f"path component must not be a symlink: {current}"
            )
    return explicit


def load_canonical_source(
    *,
    source: Path,
    source_format: str,
    source_root: Path,
    episode_index: int | None = None,
    config: LoadedCanonicalQcConfig | None = None,
) -> CanonicalQcEpisode:
    """Load one explicitly typed source without format or root guessing."""

    loaded = config or load_canonical_qc_config()
    if source_format not in loaded.raw["source"]["supported_formats"]:
        raise CanonicalInputError(
            "format_unsupported", "source_format", f"unsupported format {source_format!r}"
        )
    resolved_source = _resolved_inside(source, source_root, field="source")
    tolerance = loaded.timestamp_tolerance_ns
    if source_format == "hdf5":
        if episode_index is not None:
            raise CanonicalInputError(
                "field_mapping_error",
                "episode_index",
                "episode_index is only valid for LeRobot sources",
            )
        return StandardHdf5Adapter(max_timestamp_delta_ns=tolerance).load(
            resolved_source
        )
    return StandardLeRobotAdapter(max_timestamp_delta_ns=tolerance).load(
        resolved_source, episode_index=episode_index
    )


def run_canonical_source_qc(
    *,
    source: Path,
    source_format: str,
    source_root: Path,
    batch_root: Path,
    quality_archive: Path,
    profile: str,
    canonical_config_path: Path | None = None,
    episode_index: int | None = None,
    resume: bool = True,
    dry_run: bool = False,
    registry_factory: Callable[[object, object], ModuleRegistry] | None = None,
) -> CanonicalQcRunResult:
    """Ingest one Canonical episode and run/resume the package QC orchestrator."""

    canonical_config = load_canonical_qc_config(canonical_config_path)
    if profile not in canonical_config.profiles:
        raise CanonicalInputError(
            "field_mapping_error", "profile", f"unknown profile {profile!r}"
        )
    resolved_batch = validate_explicit_path(batch_root, field="batch_root").resolve()
    resolved_source_root = _resolved_inside(source_root, resolved_batch, field="source_root")
    resolved_source = _resolved_inside(source, resolved_source_root, field="source")
    resolved_archive = _resolved_inside(
        quality_archive, resolved_batch, field="quality_archive"
    )
    try:
        resolved_archive.relative_to(resolved_source_root)
    except ValueError:
        pass
    else:
        raise CanonicalInputError(
            "path_overlap",
            "quality_archive",
            "must not equal or be inside supplier source_root",
        )
    try:
        resolved_source_root.relative_to(resolved_archive)
    except ValueError:
        pass
    else:
        raise CanonicalInputError(
            "path_overlap",
            "quality_archive",
            "must not contain supplier source_root",
        )
    episode = load_canonical_source(
        source=resolved_source,
        source_format=source_format,
        source_root=resolved_source_root,
        episode_index=episode_index,
        config=canonical_config,
    )
    report_path = resolved_archive / f"{episode.identity.asset_id}.json"
    existed = report_path.exists()
    if existed and not resume:
        raise CanonicalInputError(
            "report_exists",
            "quality_archive",
            f"--no-resume requires a fresh report path: {report_path}",
        )
    if dry_run:
        return CanonicalQcRunResult(
            episode=episode,
            source=resolved_source,
            source_format=source_format,  # type: ignore[arg-type]
            report_path=report_path,
            status="validated",
            overall_decision=None,
            report_revision=None,
            executed_modules=(),
            resumed=existed,
            dry_run=True,
            canonical_config_path=canonical_config.path,
            canonical_config_version=canonical_config.config_version,
            canonical_config_hash=canonical_config.sha256,
            qc_config_path=canonical_config.qc_config_path,
            qc_config_version=str(canonical_config.raw["qc"]["config_version"]),
            qc_config_hash=str(canonical_config.raw["qc"]["config_sha256"]),
            runtime_error=None,
        )

    qc_config = load_qc_acceptance_config(canonical_config.qc_config_path)
    context = CanonicalQcBridge(
        episode, source_root=resolved_source_root
    ).asset_context(
        batch_root=resolved_batch,
        report_path=report_path,
    )
    registry = (
        registry_factory(context, qc_config)
        if registry_factory is not None
        else build_default_registry(context, qc_config)
    )
    if not isinstance(registry, ModuleRegistry):
        raise TypeError("registry_factory must return ModuleRegistry")
    outcome: RunOutcome = run_asset(
        context,
        config=qc_config,
        profile=profile,
        registry=registry,
    )
    decision = outcome.report.get("overall_decision")
    runtime_errors = outcome.report.get("runtime_errors")
    runtime_error = (
        dict(runtime_errors[-1])
        if isinstance(runtime_errors, list)
        and runtime_errors
        and isinstance(runtime_errors[-1], dict)
        else None
    )
    return CanonicalQcRunResult(
        episode=episode,
        source=resolved_source,
        source_format=source_format,  # type: ignore[arg-type]
        report_path=report_path,
        status=outcome.status,
        overall_decision=decision if isinstance(decision, str) else None,
        report_revision=int(outcome.report["report_revision"]),
        executed_modules=outcome.executed_modules,
        resumed=existed,
        dry_run=False,
        canonical_config_path=canonical_config.path,
        canonical_config_version=canonical_config.config_version,
        canonical_config_hash=canonical_config.sha256,
        qc_config_path=qc_config.path,
        qc_config_version=qc_config.config_version,
        qc_config_hash=qc_config.sha256,
        runtime_error=runtime_error,
    )


__all__ = [
    "CanonicalQcRunResult",
    "SourceFormat",
    "load_canonical_source",
    "run_canonical_source_qc",
    "validate_explicit_path",
]
