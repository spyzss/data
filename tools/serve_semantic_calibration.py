#!/usr/bin/env python3
"""Serve the independent semantic-calibration workbench locally."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qc_common.config import load_qc_acceptance_config  # noqa: E402
from qc_common.reviewer_lease import LeaseStore  # noqa: E402
from qc_pipeline.context import AssetContext  # noqa: E402
from semantic_calibration.application import SemanticCalibrationApplication  # noqa: E402
from semantic_calibration.http_server import create_http_server  # noqa: E402
from semantic_calibration.service import SemanticCalibrationService  # noqa: E402


LOGGER = logging.getLogger("serve_semantic_calibration")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", required=True, type=Path)
    parser.add_argument("--quality-archive", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8898, type=int)
    parser.add_argument("--profile", default="acceptance")
    parser.add_argument("--lease-ttl-seconds", default=900, type=int)
    parser.add_argument("--dataset-path", default="/label/subtask_label")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _inside(root: Path, value: str | Path, *, field: str) -> Path:
    candidate = Path(value)
    absolute = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        absolute.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{field} must stay inside batch root") from exc
    return absolute


def _hdf5_path(report: dict[str, Any], batch_root: Path) -> Path:
    sources = report.get("source_files")
    if isinstance(sources, dict):
        hdf5 = sources.get("hdf5")
        path = hdf5.get("path") if isinstance(hdf5, dict) else None
        if isinstance(path, str) and path:
            return _inside(batch_root, path, field="source_files.hdf5.path")
    for field in ("hdf5_path", "source_hdf5_path"):
        value = report.get(field)
        if isinstance(value, str) and value:
            return _inside(batch_root, value, field=field)
    raise ValueError("semantic report does not declare an HDF5 source")


def load_semantic_assets(
    batch_root: Path,
    quality_archive: Path,
) -> tuple[dict[str, Path], dict[str, Path], dict[str, AssetContext]]:
    root = batch_root.resolve()
    archive = _inside(root, quality_archive, field="quality_archive")
    if not archive.is_dir():
        raise FileNotFoundError("quality archive directory does not exist")
    assets: dict[str, Path] = {}
    reports: dict[str, Path] = {}
    contexts: dict[str, AssetContext] = {}
    for report_path in sorted(archive.glob("*.json")):
        value = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("report root must be an object")
        asset_id = value.get("asset_id") or report_path.stem
        if not isinstance(asset_id, str) or not asset_id.strip():
            raise ValueError("report is missing asset_id")
        if asset_id in assets:
            raise ValueError(f"duplicate asset_id: {asset_id}")
        source = _hdf5_path(value, root)
        assets[asset_id] = source
        reports[asset_id] = report_path.resolve()
        contexts[asset_id] = AssetContext(
            asset_id,
            root,
            report_path.resolve(),
            {"hdf5": {"path": source.relative_to(root).as_posix()}},
        )
    return assets, reports, contexts


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    assets, reports, contexts = load_semantic_assets(args.batch_root, args.quality_archive)
    config = load_qc_acceptance_config()
    domain = SemanticCalibrationService(
        assets=assets,
        reports=reports,
        dataset_path=args.dataset_path,
    )
    application = SemanticCalibrationApplication(
        domain,
        lease_store=LeaseStore(),
        lease_ttl_seconds=args.lease_ttl_seconds,
        asset_contexts=contexts,
        profile=args.profile,
        config=config,
    )
    server = create_http_server(args.host, args.port, application)
    LOGGER.info("semantic calibration server listening on http://%s:%s", args.host, server.server_port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
