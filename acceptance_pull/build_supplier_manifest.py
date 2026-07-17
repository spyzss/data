"""Build canonical acceptance manifests from supplier deliverables."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

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
    )


if __name__ == "__main__":
    raise SystemExit(main())
