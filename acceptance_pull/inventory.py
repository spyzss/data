from __future__ import annotations

import re
from pathlib import Path


HDF5_RE = re.compile(r"^(?P<id>.+)_hdf5\.hdf5$")
VIDEO_RE = re.compile(r"^(?P<id>.+)_video\.mp4$")


def _discover(root: Path, pattern: str, regex: re.Pattern[str]) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"source directory not found: {root}")

    files: dict[str, Path] = {}
    for path in sorted(root.glob(pattern)):
        match = regex.match(path.name)
        if not match:
            continue
        asset_id = match.group("id")
        if asset_id in files:
            raise ValueError(f"duplicate file ID {asset_id} under {root}")
        files[asset_id] = path
    return files


def discover_hdf5(root: Path) -> dict[str, Path]:
    return _discover(root, "*_hdf5.hdf5", HDF5_RE)


def discover_video(root: Path) -> dict[str, Path]:
    return _discover(root, "*_video.mp4", VIDEO_RE)
