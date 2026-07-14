#!/usr/bin/env python3
"""Serve the revision-aware human semantic/warn QC workbench locally."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from human_qc.evidence import EvidenceService  # noqa: E402
from human_qc.http_server import create_http_server  # noqa: E402
from human_qc.lease import LeaseStore  # noqa: E402
from human_qc.semantic_service import SemanticCalibrationService  # noqa: E402
from human_qc.warn_service import WarnReviewService  # noqa: E402
from human_qc.workbench_service import WorkbenchService  # noqa: E402
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


def _source_files(report: dict[str, Any], batch_root: Path, asset_id: str) -> dict[str, dict[str, str]]:
    raw = report.get("source_files", {})
    result: dict[str, dict[str, str]] = {}
    if isinstance(raw, dict):
        for name, value in raw.items():
            if not isinstance(value, dict):
                continue
            path_value = value.get("path")
            if not isinstance(path_value, str) or not path_value:
                continue
            relative, _ = _inside(batch_root, path_value, field=f"source_files.{name}.path")
            result[str(name)] = {"path": relative}
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
        hdf5_source = source_files.get("hdf5")
        if isinstance(hdf5_source, dict) and isinstance(hdf5_source.get("path"), str):
            hdf5_path = root / hdf5_source["path"]
            if not hdf5_path.is_file():
                raise FileNotFoundError(f"HDF5 source does not exist: {hdf5_path}")
        source_range: tuple[int, int] | None = None
        raw_range = report.get("source_range")
        if isinstance(raw_range, (list, tuple)) and len(raw_range) == 2:
            start, end = raw_range
            if isinstance(start, int) and isinstance(end, int):
                source_range = (start, end)
        contexts.append(
            AssetContext(
                asset_id=asset_id,
                batch_root=root,
                report_path=report_path,
                source_files=source_files,
                source_range=source_range,
                metadata={"profile": report.get("profile")},
            )
        )
    if not contexts:
        raise ValueError(f"quality archive contains no JSON reports: {archive}")
    return contexts


def build_workbench_service(
    *,
    batch_root: Path,
    quality_archive: Path,
    profile: str = "acceptance",
    lease_ttl_seconds: int = 900,
) -> WorkbenchService:
    contexts = load_contexts(batch_root, quality_archive)
    reports = {context.asset_id: context.report_path for context in contexts}
    assets = {
        context.asset_id: context.batch_root / source["path"]
        for context in contexts
        if isinstance((source := context.source_files.get("hdf5")), dict)
        and isinstance(source.get("path"), str)
    }
    semantic = SemanticCalibrationService(assets=assets, reports=reports)
    warn = WarnReviewService(reports=reports)
    evidence = EvidenceService(batch_root / ".human_qc_evidence")
    service = WorkbenchService(
        semantic,
        warn,
        evidence,
        lease_store=LeaseStore(),
        asset_contexts={context.asset_id: context for context in contexts},
        lease_ttl_seconds=lease_ttl_seconds,
        profile=profile,
    )
    # Force report/HDF5 loading at process start so a durable ``finalizing``
    # semantic transaction is recovered before the first browser request.
    for context in contexts:
        if context.asset_id in assets:
            semantic.get_task(context.asset_id)
        warn.get_task(context.asset_id)
    return service


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    service = build_workbench_service(
        batch_root=args.batch_root,
        quality_archive=args.quality_archive,
        profile=args.profile,
        lease_ttl_seconds=args.lease_ttl_seconds,
    )
    server = create_http_server(args.host, args.port, service, evidence_root=args.batch_root)
    LOGGER.info("Serving human QC workbench at http://%s:%s", args.host, server.server_port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
