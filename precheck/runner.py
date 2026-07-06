"""Independent runner for data-trustworthiness prechecks."""

from __future__ import annotations

import logging
import json
from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd

from qc_common.io import (
    aggregate_results,
    dataframe_to_json_records,
    read_existing_dataframe,
    results_to_dataframe,
    write_dataframe,
    write_json_records,
)
from qc_common.types import CheckResult, ClipInputs

from .adapters import load_precheck_inputs
from .config import PrecheckConfig

# Populate registry via decorators.
from . import checks as _checks  # noqa: F401
from .registry import create_check

logger = logging.getLogger(__name__)

ClipLoader = Callable[[Path, int | None, float | None], list[ClipInputs]]


def _infer_asset_id_from_path(path: Path) -> str | None:
    """Infer a human asset id from a configured single-file input path."""
    if not path.is_file():
        return None
    stem = path.stem
    for suffix in ("_hdf5", "-hdf5", "_video", "-video"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem or None


class PrecheckRunner:
    """Run configured prechecks on clips without depending on annotation stages."""

    def __init__(
        self,
        config: PrecheckConfig,
        clip_loader: ClipLoader | None = None,
    ) -> None:
        self.config = config
        self.clip_loader = clip_loader or load_precheck_inputs
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.repair_records: list[dict] = []
        self.candidate_window_records: list[dict] = []

    def run(self, clips: Iterable[ClipInputs] | None = None) -> list[CheckResult]:
        if clips is None:
            clips = self._load_configured_clips()

        all_results: list[CheckResult] = []
        for clip in clips:
            logger.info("Episode %s: running prechecks", clip.episode_idx)
            all_results.extend(self.run_clip(clip))

        self._write_outputs(all_results)
        return all_results

    def run_clip(self, clip: ClipInputs) -> list[CheckResult]:
        results: list[CheckResult] = []
        for check_name in self.config.enabled_checks:
            check = create_check(check_name, self.config.check_config(check_name))
            try:
                check_results = check.run(clip)
                logger.info(
                    "Episode %s: %s emitted %d rows",
                    clip.episode_idx,
                    check_name,
                    len(check_results),
                )
                results.extend(check_results)
                self.repair_records.extend(getattr(check, "repair_records", []))
                self.candidate_window_records.extend(
                    getattr(check, "candidate_windows", [])
                )
            except Exception as exc:
                logger.error(
                    "Episode %s: %s failed: %s",
                    clip.episode_idx,
                    check_name,
                    exc,
                    exc_info=True,
                )
                results.extend(self._failure_rows(clip, check_name, exc))
        return results

    def _load_configured_clips(self) -> list[ClipInputs]:
        clips: list[ClipInputs] = []
        next_episode_idx = 0
        for path in self.config.input_paths:
            loaded_clips = self.clip_loader(path, next_episode_idx, self.config.fps)
            inferred_asset_id = _infer_asset_id_from_path(Path(path))
            if inferred_asset_id is not None and len(loaded_clips) == 1:
                clip = loaded_clips[0]
                if getattr(clip, "asset_id", None) is None:
                    setattr(clip, "asset_id", inferred_asset_id)
            clips.extend(loaded_clips)
            next_episode_idx += len(loaded_clips)
        return clips

    def _failure_rows(
        self,
        clip: ClipInputs,
        check_name: str,
        exc: Exception,
    ) -> list[CheckResult]:
        return [
            CheckResult(
                check=check_name,
                episode_idx=clip.episode_idx,
                frame_idx=clip.frame_idx_at(offset),
                metrics={},
                flag=None,
                reason=f"check failed: {exc}",
            )
            for offset in range(clip.num_frames)
        ]

    def _write_outputs(self, results: list[CheckResult]) -> None:
        if not results:
            logger.warning("No precheck results to write")
            return

        result_df = results_to_dataframe(results)
        result_file = self.output_dir / "check_results.parquet"
        if not self.config.overwrite:
            existing_df = read_existing_dataframe(result_file)
            if existing_df is not None:
                result_df = pd.concat([existing_df, result_df], ignore_index=True)
        result_df = result_df.drop_duplicates(
            subset=["episode_idx", "frame_idx", "check"],
            keep="last" if self.config.overwrite else "first",
        )
        result_path = write_dataframe(result_df, result_file)
        result_json_path = write_json_records(
            dataframe_to_json_records(result_df),
            self.output_dir / "check_results.json",
        )

        aggregate_df = pd.DataFrame(aggregate_results(results))
        aggregate_file = self.output_dir / "clip_aggregates.parquet"
        if not self.config.overwrite:
            existing_aggregate = read_existing_dataframe(aggregate_file)
            if existing_aggregate is not None:
                aggregate_df = pd.concat(
                    [existing_aggregate, aggregate_df],
                    ignore_index=True,
                )
        aggregate_df = aggregate_df.drop_duplicates(
            subset=["episode_idx", "check"],
            keep="last" if self.config.overwrite else "first",
        )
        aggregate_path = write_dataframe(
            aggregate_df,
            aggregate_file,
        )
        aggregate_json_path = write_json_records(
            aggregate_df.to_dict(orient="records"),
            self.output_dir / "clip_aggregates.json",
        )
        if self.repair_records:
            repair_path = self.output_dir / "keypoint_missing_repair_candidates.json"
            repair_path.write_text(
                json.dumps(self.repair_records, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        if self.candidate_window_records:
            candidate_path = self.output_dir / "candidate_windows.json"
            candidate_path.write_text(
                json.dumps(
                    self.candidate_window_records,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            candidate_parquet_path = write_dataframe(
                pd.DataFrame(self.candidate_window_records),
                self.output_dir / "candidate_windows.parquet",
            )
            logger.info(
                "Wrote candidate windows to %s and %s",
                candidate_path,
                candidate_parquet_path,
            )
        logger.info(
            "Wrote precheck results to %s, %s, %s and %s",
            result_path,
            result_json_path,
            aggregate_path,
            aggregate_json_path,
        )
