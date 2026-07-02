"""Auto-dispatch input adapters for precheck ClipInputs."""

from __future__ import annotations

from pathlib import Path

from qc_common.types import ClipInputs

from .supplier_hdf5 import load_supplier_hdf5_clip

HDF5_SUFFIXES = {".h5", ".hdf5"}


def load_precheck_inputs(
    path: str | Path,
    episode_idx: int | None = None,
    fps: float | None = None,
) -> list[ClipInputs]:
    """
    Load one configured precheck input path.

    A path may be one supplier HDF5 file or a directory. Directories are
    inspected and expanded by adapter type; each resulting file becomes one
    ClipInputs. Future supplier adapters should be added here without changing
    checks or runner logic.
    """
    input_path = Path(path)
    if input_path.is_file():
        return [_load_file(input_path, episode_idx, fps)]
    if input_path.is_dir():
        return _load_directory(input_path, episode_idx, fps)
    raise FileNotFoundError(input_path)


def _load_file(
    path: Path,
    episode_idx: int | None,
    fps: float | None,
) -> ClipInputs:
    if path.suffix.lower() in HDF5_SUFFIXES:
        return load_supplier_hdf5_clip(path, episode_idx=episode_idx, fps=fps)
    raise ValueError(f"No precheck adapter registered for file: {path}")


def _load_directory(
    path: Path,
    episode_idx: int | None,
    fps: float | None,
) -> list[ClipInputs]:
    hdf5_files = sorted(
        file
        for suffix in HDF5_SUFFIXES
        for file in path.rglob(f"*{suffix}")
        if file.is_file()
    )
    if hdf5_files:
        clips = []
        for offset, file in enumerate(hdf5_files):
            clip_episode_idx = None if episode_idx is None else episode_idx + offset
            clips.append(
                load_supplier_hdf5_clip(
                    file,
                    episode_idx=clip_episode_idx,
                    fps=fps,
                )
            )
        return clips

    if _looks_like_lerobot_parquet(path):
        raise NotImplementedError(
            "Detected a LeRobot/parquet-style directory, but the precheck "
            "adapter is not implemented yet. Add a dedicated adapter that maps "
            "episode parquet columns/calibration metadata into ClipInputs."
        )

    if _looks_like_csv_video_bundle(path):
        raise NotImplementedError(
            "Detected a CSV/video bundle directory, but the precheck adapter is "
            "not implemented yet. Add a dedicated adapter that maps aligned.csv, "
            "frames.csv, calibration.json, and video.mp4 into ClipInputs."
        )

    raise ValueError(f"No precheck adapter registered for directory: {path}")


def _looks_like_lerobot_parquet(path: Path) -> bool:
    return (
        (path / "meta").is_dir()
        and (path / "episodes").is_dir()
        and any((path / "episodes").glob("chunk-*/file-*.parquet"))
    )


def _looks_like_csv_video_bundle(path: Path) -> bool:
    return (
        (path / "aligned.csv").exists()
        and (path / "frames.csv").exists()
        and (path / "calibration.json").exists()
        and (path / "video.mp4").exists()
    )
