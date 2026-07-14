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


LOGGER = logging.getLogger("build_supplier_manifest")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a canonical supplier manifest for acceptance QC."
    )
    parser.add_argument("--supplier", required=True, choices=("deepreach",))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--camera",
        action="append",
        choices=SUPPORTED_CAMERAS,
        help="Camera view to include; repeat for multiple views. Default: head.",
    )
    parser.add_argument("--calib-cache", required=True, type=Path)
    parser.add_argument("--stage-video-quality", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def run(
    *,
    supplier: str,
    root: Path,
    output_dir: Path,
    cameras: tuple[str, ...] = ("head",),
    calib_cache: Path,
    stage_video_quality: bool = False,
) -> int:
    if supplier != "deepreach":
        raise ValueError(f"unsupported supplier: {supplier}")
    rows = build_deepreach_manifest(
        root=root,
        calib_cache=calib_cache,
        cameras=cameras,
    )
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
    )


if __name__ == "__main__":
    raise SystemExit(main())
