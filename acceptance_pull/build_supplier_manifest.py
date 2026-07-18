"""Build canonical acceptance manifests from supplier deliverables."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Mapping, Sequence

from acceptance_pull.supplier_adapters.deepreach import (
    SUPPORTED_CAMERAS,
    build_deepreach_manifest,
    stage_video_quality_inputs,
    write_manifest_outputs,
)
from acceptance_pull.supplier_adapters.deepreach_hdf5 import FRAME_DATASETS
from acceptance_pull.supplier_adapters.potentia import (
    build_potentia_manifest,
    write_potentia_manifest,
)


LOGGER = logging.getLogger("build_supplier_manifest")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a canonical supplier manifest for acceptance QC."
    )
    parser.add_argument(
        "--supplier",
        required=True,
        choices=("dr", "deepreach", "potentia"),
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--camera",
        action="append",
        choices=SUPPORTED_CAMERAS,
        help="Camera view to include; repeat for multiple views. Default: head.",
    )
    parser.add_argument("--calib-cache", type=Path)
    parser.add_argument(
        "--calibration-map",
        type=Path,
        help=(
            "Explicit CSV/JSON task_name to content_id mapping for DR calibration; "
            "task_name is never treated as content_id implicitly."
        ),
    )
    parser.add_argument("--max-assets", type=int)
    parser.add_argument(
        "--granularity",
        choices=("task", "task_camera"),
        default="task",
    )
    parser.add_argument(
        "--primary-camera",
        choices=SUPPORTED_CAMERAS,
        default="head",
    )
    parser.add_argument(
        "--hdf5-reference-dataset",
        choices=FRAME_DATASETS,
        help="Required for task-level DR; select only from the confirmed supplier contract.",
    )
    parser.add_argument("--stage-video-quality", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def run(
    *,
    supplier: str,
    root: Path,
    output_dir: Path,
    cameras: tuple[str, ...] = ("head",),
    calib_cache: Path | None = None,
    stage_video_quality: bool = False,
    granularity: str = "task",
    primary_camera: str = "head",
    hdf5_reference_dataset: str | None = None,
    max_assets: int | None = None,
    calibration_mapping: Mapping[str, str] | None = None,
) -> int:
    if supplier == "potentia":
        rows = build_potentia_manifest(root, max_assets=max_assets)
        manifest_path = write_potentia_manifest(rows, output_dir)
        LOGGER.info("Wrote %d rows to %s", len(rows), manifest_path)
        return 0
    if supplier not in {"dr", "deepreach"}:
        raise ValueError(f"unsupported supplier: {supplier}")
    if calib_cache is None:
        raise ValueError("--calib-cache is required for DR")
    if granularity == "task" and hdf5_reference_dataset is None:
        raise ValueError(
            "--hdf5-reference-dataset is required for task-level DR"
        )
    rows = build_deepreach_manifest(
        root=root,
        calib_cache=calib_cache,
        cameras=cameras,
        granularity=granularity,
        primary_camera=primary_camera,
        reference_dataset=hdf5_reference_dataset,
        calibration_mapping=calibration_mapping,
    )
    if max_assets is not None:
        if max_assets < 1:
            raise ValueError("max_assets must be >= 1")
        rows = rows[:max_assets]
    manifest_path, sidecar_path = write_manifest_outputs(rows, output_dir)
    LOGGER.info("Wrote %d rows to %s", len(rows), manifest_path)
    LOGGER.info("Wrote calibration sidecar to %s", sidecar_path)
    if stage_video_quality:
        summary = stage_video_quality_inputs(rows, output_dir)
        LOGGER.info("Video-quality staging: %s", summary)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return run(
        supplier=args.supplier,
        root=args.root,
        output_dir=args.output_dir,
        cameras=tuple(args.camera or ("head",)),
        calib_cache=args.calib_cache,
        stage_video_quality=args.stage_video_quality,
        granularity=args.granularity,
        primary_camera=args.primary_camera,
        hdf5_reference_dataset=args.hdf5_reference_dataset,
        max_assets=args.max_assets,
        calibration_mapping=(
            read_calibration_mapping(args.calibration_map)
            if args.calibration_map is not None
            else None
        ),
    )


def read_calibration_mapping(path: Path) -> dict[str, str]:
    """Read an explicit one-to-one DR task_name -> content_id mapping."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    elif suffix in {".jsonl", ".ndjson"}:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            rows = [
                {"task_name": task_name, "content_id": content_id}
                for task_name, content_id in payload.items()
            ]
        elif isinstance(payload, list):
            rows = payload
        else:
            raise ValueError("calibration mapping JSON must be an object or row list")
    else:
        raise ValueError("calibration mapping must be CSV, JSON, or JSONL")
    mapping: dict[str, str] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"calibration mapping row {index} must be an object")
        task_name = str(row.get("task_name") or "").strip()
        content_id = str(row.get("content_id") or "").strip()
        if not task_name or not content_id:
            raise ValueError(
                f"calibration mapping row {index} requires task_name and content_id"
            )
        if task_name in mapping:
            raise ValueError(f"duplicate calibration mapping task_name: {task_name}")
        mapping[task_name] = content_id
    return mapping


if __name__ == "__main__":
    raise SystemExit(main())
