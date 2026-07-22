#!/usr/bin/env python3
"""Serve the revision-aware Warn human-review workbench locally."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from human_qc.http_server import create_http_server  # noqa: E402
from human_qc.media import MediaCatalog  # noqa: E402
from human_qc.sam3_overlay_renderer import (  # noqa: E402
    DEFAULT_OVERLAY_MAX_CACHE_BYTES,
    OverlaySetupError,
    ProductionOverlayRuntime,
    UnavailableOverlayProvider,
    build_production_overlay_runtime,
)
from qc_common.reviewer_lease import LeaseStore  # noqa: E402
from qc_common.manifest_metadata import context_metadata_from_report  # noqa: E402
from human_qc.warn_service import WarnReviewService  # noqa: E402
from human_qc.warn_workbench_service import WarnWorkbenchService  # noqa: E402
from qc_pipeline.context import AssetContext  # noqa: E402


LOGGER = logging.getLogger("serve_human_qc_workbench")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", required=True, type=Path)
    parser.add_argument("--quality-archive", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8897, type=int)
    parser.add_argument("--profile", default="acceptance")
    parser.add_argument("--lease-ttl-seconds", default=900, type=int)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--sam3-model", type=Path, default=None)
    parser.add_argument(
        "--overlay-cache-dir",
        type=Path,
        default=Path(".human_qc/overlay-cache"),
    )
    parser.add_argument("--overlay-workers", type=int, default=1)
    parser.add_argument("--overlay-max-pending", type=int, default=1)
    parser.add_argument(
        "--overlay-max-cache-bytes",
        type=int,
        default=DEFAULT_OVERLAY_MAX_CACHE_BYTES,
    )
    parser.add_argument("--overlay-max-ready-jobs", type=int, default=None)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _inside(root: Path, value: str | Path, *, field: str) -> tuple[str, Path]:
    candidate = Path(value)
    absolute = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        relative = absolute.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{field} must stay inside batch root: {absolute}") from exc
    return relative.as_posix(), absolute


_SHA256 = re.compile(r"sha256:[0-9a-fA-F]{64}").fullmatch


def _source_files(
    report: dict[str, Any], batch_root: Path, asset_id: str
) -> dict[str, dict[str, Any]]:
    raw = report.get("source_files", {})
    result: dict[str, dict[str, Any]] = {}
    if isinstance(raw, dict):
        for name, value in raw.items():
            if not isinstance(value, dict):
                continue
            path_value = value.get("path")
            if not isinstance(path_value, str) or not path_value:
                continue
            relative, _ = _inside(batch_root, path_value, field=f"source_files.{name}.path")
            source: dict[str, Any] = {"path": relative}
            sha256 = value.get("sha256")
            if isinstance(sha256, str) and _SHA256(sha256) is not None:
                source["sha256"] = sha256.lower()
            size_bytes = value.get("size_bytes")
            if (
                isinstance(size_bytes, int)
                and not isinstance(size_bytes, bool)
                and size_bytes >= 0
            ):
                source["size_bytes"] = size_bytes
            result[str(name)] = source
    # Reports produced by older runners sometimes only carry an HDF5 path at
    # the top level.  Keep this fallback source-faithful and containment-safe.
    if "hdf5" not in result:
        for key in ("hdf5_path", "source_hdf5_path"):
            value = report.get(key)
            if isinstance(value, str) and value:
                relative, _ = _inside(batch_root, value, field=key)
                result["hdf5"] = {"path": relative}
                break
    if "video" not in result:
        for key in ("video_path", "primary_video_path"):
            value = report.get(key)
            if isinstance(value, str) and value:
                relative, _ = _inside(batch_root, value, field=key)
                result["video"] = {"path": relative}
                break
    return result


def load_contexts(batch_root: Path, quality_archive: Path) -> list[AssetContext]:
    """Load report metadata into safe contexts for the workbench facade."""

    root = batch_root.resolve()
    archive = quality_archive if quality_archive.is_absolute() else root / quality_archive
    archive = archive.resolve()
    try:
        archive.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"quality archive must stay inside batch root: {archive}") from exc
    if not archive.is_dir():
        raise FileNotFoundError(f"quality archive directory does not exist: {archive}")
    contexts: list[AssetContext] = []
    seen: set[str] = set()
    for report_path in sorted(archive.glob("*.json")):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"unable to load report {report_path}: {exc}") from exc
        if not isinstance(report, dict):
            raise ValueError(f"report root must be an object: {report_path}")
        asset_id = report.get("asset_id") or report_path.stem
        if not isinstance(asset_id, str) or not asset_id.strip():
            raise ValueError(f"report is missing asset_id: {report_path}")
        if asset_id in seen:
            raise ValueError(f"duplicate asset_id in quality archive: {asset_id}")
        seen.add(asset_id)
        source_files = _source_files(report, root, asset_id)
        # Warn review is report/evidence driven.  Preserve an opaque HDF5
        # context path for later pipeline stages, but do not make an unrelated
        # semantic source file a launcher precondition.
        source_range: tuple[int, int] | None = None
        raw_range = report.get("source_range")
        if isinstance(raw_range, (list, tuple)) and len(raw_range) == 2:
            start, end = raw_range
            if isinstance(start, int) and isinstance(end, int):
                source_range = (start, end)
        metadata: dict[str, Any] = {
            "profile": report.get("profile"),
            **context_metadata_from_report(report),
        }
        module = report.get("sam3_containment")
        runtime = module.get("runtime") if isinstance(module, Mapping) else None
        recipe = (
            runtime.get("overlay_input_recipe")
            if isinstance(runtime, Mapping)
            else None
        )
        if isinstance(recipe, Mapping):
            metadata["sam3_overlay_recipe"] = dict(recipe)
        contexts.append(
            AssetContext(
                asset_id=asset_id,
                batch_root=root,
                report_path=report_path,
                source_files=source_files,
                source_range=source_range,
                metadata=metadata,
            )
        )
    if not contexts:
        raise ValueError(f"quality archive contains no JSON reports: {archive}")
    return contexts


@dataclass
class WorkbenchRuntime:
    service: WarnWorkbenchService
    overlay_runtime: ProductionOverlayRuntime | object | None = None

    @property
    def worker(self) -> object | None:
        return (
            None
            if self.overlay_runtime is None
            else getattr(self.overlay_runtime, "worker", None)
        )

    def shutdown(self) -> None:
        if self.overlay_runtime is None:
            return
        close = getattr(self.overlay_runtime, "shutdown", None)
        if callable(close):
            close()
            return
        worker = getattr(self.overlay_runtime, "worker", None)
        shutdown = getattr(worker, "shutdown", None)
        if callable(shutdown):
            shutdown()


def build_workbench_runtime(
    *,
    batch_root: Path,
    quality_archive: Path,
    reviewer: str,
    profile: str = "acceptance",
    lease_ttl_seconds: int = 900,
    sam3_model: Path | None = None,
    overlay_cache_dir: Path = Path(".human_qc/overlay-cache"),
    overlay_workers: int = 1,
    overlay_max_pending: int = 1,
    overlay_max_cache_bytes: int | None = DEFAULT_OVERLAY_MAX_CACHE_BYTES,
    overlay_max_ready_jobs: int | None = None,
) -> WorkbenchRuntime:
    contexts = load_contexts(batch_root, quality_archive)
    reports = {context.asset_id: context.report_path for context in contexts}
    warn = WarnReviewService(reports=reports)
    contexts_by_id = {context.asset_id: context for context in contexts}
    media_catalog = MediaCatalog(contexts_by_id)
    overlay_runtime: ProductionOverlayRuntime | object | None = None
    if sam3_model is None:
        overlay_provider: object = UnavailableOverlayProvider(
            "overlay_model_unavailable"
        )
    else:
        try:
            overlay_runtime = build_production_overlay_runtime(
                contexts=contexts_by_id,
                media_catalog=media_catalog,
                model_path=sam3_model,
                cache_relative=overlay_cache_dir,
                max_workers=overlay_workers,
                max_pending=overlay_max_pending,
                max_cache_bytes=overlay_max_cache_bytes,
                max_ready_jobs=overlay_max_ready_jobs,
            )
        except OverlaySetupError as exc:
            overlay_provider = UnavailableOverlayProvider(exc.code)
        else:
            overlay_provider = overlay_runtime.provider
    try:
        service = WarnWorkbenchService(
            reviewer=reviewer,
            warn_service=warn,
            lease_store=LeaseStore(),
            asset_contexts=contexts_by_id,
            media_catalog=media_catalog,
            overlay_provider=overlay_provider,
            lease_ttl_seconds=lease_ttl_seconds,
            profile=profile,
        )
        # Force report loading at process start so malformed review state fails
        # before the first browser request.
        for context in contexts:
            warn.get_task(context.asset_id)
    except BaseException:
        if overlay_runtime is not None:
            close = getattr(overlay_runtime, "shutdown", None)
            if callable(close):
                close()
            else:
                worker = getattr(overlay_runtime, "worker", None)
                shutdown = getattr(worker, "shutdown", None)
                if callable(shutdown):
                    shutdown()
        raise
    return WorkbenchRuntime(service=service, overlay_runtime=overlay_runtime)


def build_workbench_service(
    *,
    batch_root: Path,
    quality_archive: Path,
    reviewer: str,
    profile: str = "acceptance",
    lease_ttl_seconds: int = 900,
    sam3_model: Path | None = None,
    overlay_cache_dir: Path = Path(".human_qc/overlay-cache"),
    overlay_workers: int = 1,
    overlay_max_pending: int = 1,
    overlay_max_cache_bytes: int | None = DEFAULT_OVERLAY_MAX_CACHE_BYTES,
    overlay_max_ready_jobs: int | None = None,
) -> WarnWorkbenchService:
    """Compatibility wrapper for callers that only need the facade object."""

    runtime = build_workbench_runtime(
        batch_root=batch_root,
        quality_archive=quality_archive,
        reviewer=reviewer,
        profile=profile,
        lease_ttl_seconds=lease_ttl_seconds,
        sam3_model=sam3_model,
        overlay_cache_dir=overlay_cache_dir,
        overlay_workers=overlay_workers,
        overlay_max_pending=overlay_max_pending,
        overlay_max_cache_bytes=overlay_max_cache_bytes,
        overlay_max_ready_jobs=overlay_max_ready_jobs,
    )
    service = runtime.service
    setattr(service, "_workbench_runtime_owner", runtime)
    setattr(service, "shutdown", runtime.shutdown)
    return service


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    runtime: WorkbenchRuntime | None = None
    server = None
    try:
        runtime = build_workbench_runtime(
            batch_root=args.batch_root,
            quality_archive=args.quality_archive,
            reviewer=args.reviewer,
            profile=args.profile,
            lease_ttl_seconds=args.lease_ttl_seconds,
            sam3_model=args.sam3_model,
            overlay_cache_dir=args.overlay_cache_dir,
            overlay_workers=args.overlay_workers,
            overlay_max_pending=args.overlay_max_pending,
            overlay_max_cache_bytes=args.overlay_max_cache_bytes,
            overlay_max_ready_jobs=args.overlay_max_ready_jobs,
        )
        server = create_http_server(args.host, args.port, runtime.service)
        LOGGER.info(
            "Serving human QC workbench at http://%s:%s",
            args.host,
            server.server_port,
        )
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down")
    finally:
        if server is not None:
            server.server_close()
        if runtime is not None:
            runtime.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
