from __future__ import annotations

import argparse
from pathlib import Path

from acceptance_pull.config import PullConfig
from acceptance_pull.inventory import discover_hdf5, discover_video
from acceptance_pull.manifest import read_manifest
from acceptance_pull.reports import write_reports
from acceptance_pull.sampling import minimum_sample_count, sample_assets
from acceptance_pull.sources import download_oss_batch_inputs, build_source, pull_pairs
from acceptance_pull.validation import validate_ids


def run(config: PullConfig) -> int:
    manifest_path = config.manifest
    readme_path = config.readme
    if config.batch_source is not None and (manifest_path is None or readme_path is None):
        batch_inputs = download_oss_batch_inputs(config.batch_source, config.output / "_inputs")
        manifest_path = manifest_path or batch_inputs.manifest
        readme_path = readme_path or batch_inputs.readme

    if manifest_path is None or not manifest_path.is_file():
        raise FileNotFoundError(f"manifest file not found: {manifest_path}")
    if readme_path is None or not readme_path.is_file():
        raise FileNotFoundError(f"README file not found: {readme_path}")

    manifest = read_manifest(manifest_path)
    if config.hdf5.kind == "local":
        hdf5_inventory = discover_hdf5(config.hdf5.root or Path())
    else:
        hdf5_inventory = {asset_id: Path(f"{asset_id}_hdf5.hdf5") for asset_id in manifest}

    if config.video.kind == "local":
        video_inventory = discover_video(config.video.root or Path())
    else:
        video_inventory = {asset_id: Path(f"{asset_id}_video.mp4") for asset_id in manifest}

    validation = validate_ids(manifest, hdf5_inventory, video_inventory)
    if not validation.valid_ids:
        raise ValueError("no valid IDs available after manifest/hdf5/video matching")

    samples = sample_assets(manifest, validation.valid_ids, config.seed, config.sample_ratio)
    hdf5_source = build_source(config.hdf5, "hdf5")
    video_source = build_source(config.video, "video")
    pulls = pull_pairs(samples, hdf5_source, video_source, config.output, config.workers)
    write_reports(
        output=config.output,
        manifest=manifest,
        validation=validation,
        samples=samples,
        pulls=pulls,
        minimum_count=minimum_sample_count(len(validation.valid_ids), config.sample_ratio),
        seed=config.seed,
        workers=config.workers,
        sample_ratio=config.sample_ratio,
    )
    return 0 if all(result.status == "success" for result in pulls) else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    return run(PullConfig.from_file(args.config))
