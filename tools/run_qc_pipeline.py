#!/usr/bin/env python3
"""Run the unified QC pipeline for manifest-defined assets."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qc_common.config import LoadedQcConfig, load_qc_acceptance_config  # noqa: E402
from qc_common.module_registry import ModuleRegistry  # noqa: E402
from qc_pipeline.context import AssetContext  # noqa: E402
from qc_pipeline.orchestrator import (  # noqa: E402
    RunOutcome,
    build_default_registry,
    run_asset,
)
from tools.run_manifest_precheck import read_manifest  # noqa: E402


RegistryFactory = Callable[[AssetContext], ModuleRegistry]
_SOURCE_COLUMNS = {
    "video": ("primary_video_path", "video_path"),
    "hdf5": ("hdf5_path",),
    "parquet": ("parquet_path",),
    "candidate_windows": ("candidate_windows_path", "candidate_windows"),
    "sam3_model": ("sam3_model", "sam3_model_path"),
}


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be an integer") from None
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{field} must be an integer")
    return int(numeric)


def _path_inside_batch(
    value: Any,
    *,
    batch_root: Path,
    manifest_dir: Path,
    field: str,
) -> tuple[str, str]:
    text = _text(value)
    if not text:
        raise ValueError(f"{field} must be a non-empty path")
    candidate = Path(text).expanduser()
    absolute = (
        candidate.resolve()
        if candidate.is_absolute()
        else (manifest_dir / candidate).resolve()
    )
    try:
        relative = absolute.relative_to(batch_root)
    except ValueError:
        raise ValueError(f"{field} is outside batch_root: {absolute}") from None
    return relative.as_posix(), str(absolute)


def contexts_from_manifest(
    manifest: Path,
    *,
    batch_root: Path,
) -> list[AssetContext]:
    """Build source-faithful, independent contexts from manifest rows."""
    manifest = manifest.resolve()
    batch_root = batch_root.resolve()
    rows = read_manifest(manifest)
    seen: set[str] = set()
    for row_index, row in enumerate(rows):
        asset_id = _text(row.get("asset_id"))
        if not asset_id:
            raise ValueError(f"manifest row {row_index} missing asset_id")
        if asset_id in seen:
            raise ValueError(f"duplicate asset_id: {asset_id}")
        seen.add(asset_id)

    contexts: list[AssetContext] = []
    for row_index, source_row in enumerate(rows):
        row = copy.deepcopy(source_row)
        asset_id = _text(row.get("asset_id"))
        source_files: dict[str, Any] = {}
        for source_name, columns in _SOURCE_COLUMNS.items():
            column = next((name for name in columns if _text(row.get(name))), None)
            if column is None:
                continue
            relative, absolute = _path_inside_batch(
                row[column],
                batch_root=batch_root,
                manifest_dir=manifest.parent,
                field=column,
            )
            source_files[source_name] = {"path": relative}
            row[column] = absolute

        start_text = _text(row.get("start_frame"))
        end_text = _text(row.get("end_frame"))
        source_range: tuple[int, int] | None = None
        if start_text or end_text:
            if not start_text or not end_text:
                raise ValueError(
                    f"manifest row {row_index} must define both start_frame and end_frame"
                )
            start = _integer(row.get("start_frame"), "start_frame")
            inclusive_end = _integer(row.get("end_frame"), "end_frame")
            source_range = (start, inclusive_end + 1)

        contexts.append(
            AssetContext(
                asset_id=asset_id,
                batch_root=batch_root,
                report_path=batch_root / "quality_archive" / f"{asset_id}.json",
                source_files=source_files,
                source_range=source_range,
                metadata={
                    **row,
                    "manifest_row": row,
                },
            )
        )
    return contexts


def run_batch(
    contexts: Iterable[AssetContext],
    *,
    config: LoadedQcConfig,
    profile: str,
    registry_factory: RegistryFactory | None = None,
    max_workers: int = 1,
    resume: bool = True,
) -> dict[str, RunOutcome]:
    """Schedule assets concurrently while keeping registry/report state isolated."""
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    context_list = list(contexts)
    seen: set[str] = set()
    for context in context_list:
        if context.asset_id in seen:
            raise ValueError(f"duplicate asset_id: {context.asset_id}")
        seen.add(context.asset_id)
    if not resume:
        existing = [
            context.asset_id
            for context in context_list
            if context.report_path.exists()
        ]
        if existing:
            raise FileExistsError(
                "--no-resume requires fresh report paths: " + ", ".join(existing)
            )

    factory = registry_factory or (
        lambda context: build_default_registry(context, config)
    )

    def run_one(context: AssetContext) -> RunOutcome:
        registry = factory(context)
        if not isinstance(registry, ModuleRegistry):
            raise TypeError("registry_factory must return ModuleRegistry")
        return run_asset(
            context,
            config=config,
            profile=profile,
            registry=registry,
        )

    outcomes: dict[str, RunOutcome] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_asset = {
            pool.submit(run_one, context): context.asset_id
            for context in context_list
        }
        for future in as_completed(future_to_asset):
            asset_id = future_to_asset[future]
            outcomes[asset_id] = future.result()
    return outcomes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--profile",
        required=True,
        choices=("acceptance", "supplier_evaluation"),
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_qc_acceptance_config(args.config)
    contexts = contexts_from_manifest(args.manifest, batch_root=args.batch_root)
    outcomes = run_batch(
        contexts,
        config=config,
        profile=args.profile,
        max_workers=args.max_workers,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                asset_id: {
                    "status": outcome.status,
                    "report_revision": outcome.report["report_revision"],
                    "executed_modules": list(outcome.executed_modules),
                }
                for asset_id, outcome in sorted(outcomes.items())
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
