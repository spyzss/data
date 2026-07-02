"""Independent runner for semantic annotation verification."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from qc_common.io import (
    aggregate_results,
    read_existing_dataframe,
    results_to_dataframe,
    write_dataframe,
)
from qc_common.types import CheckResult, ClipInputs

from . import checks as _checks  # noqa: F401
from .config import AnnotationVerifyConfig
from .registry import create_check

logger = logging.getLogger(__name__)


class AnnotationVerifyRunner:
    """Run semantic consistency checks with optional injected inputs."""

    def __init__(self, config: AnnotationVerifyConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, clips: Iterable[ClipInputs]) -> list[CheckResult]:
        all_results: list[CheckResult] = []
        for clip in clips:
            logger.info("Episode %s: running annotation verification", clip.episode_idx)
            all_results.extend(self.run_clip(clip))

        self._write_outputs(all_results)
        return all_results

    def run_clip(self, clip: ClipInputs) -> list[CheckResult]:
        results: list[CheckResult] = []
        for check_name in self.config.enabled_checks:
            check = create_check(check_name, self.config.check_config(check_name))
            try:
                results.extend(check.run(clip))
            except Exception as exc:
                logger.error(
                    "Episode %s: %s failed: %s",
                    clip.episode_idx,
                    check_name,
                    exc,
                    exc_info=True,
                )
                results.extend(
                    CheckResult(
                        check=check_name,
                        episode_idx=clip.episode_idx,
                        frame_idx=clip.frame_idx_at(offset),
                        metrics={},
                        flag=None,
                        reason=f"check failed: {exc}",
                    )
                    for offset in range(clip.num_frames)
                )
        return results

    def _write_outputs(self, results: list[CheckResult]) -> None:
        if not results:
            logger.warning("No annotation verification results to write")
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
        logger.info("Wrote annotation verification results to %s and %s", result_path, aggregate_path)
