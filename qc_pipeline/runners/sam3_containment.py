"""Real manifest SAM3 producer and unified-adapter runner bridge."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
import tempfile
from typing import Any

from qc_common.config import LoadedQcConfig
from qc_common.contracts import ModuleResult
from qc_common.module_registry import ModulePrerequisiteError, ModuleRunner
from qc_pipeline.context import AssetContext


def _source_entry(context: AssetContext, name: str) -> Mapping[str, Any] | None:
    source = context.source_files.get(name)
    return source if isinstance(source, Mapping) else None


def _source_path(
    context: AssetContext,
    name: str,
    *,
    required: bool = True,
) -> Path | None:
    entry = _source_entry(context, name)
    value = entry.get("path") if entry is not None else None
    if value is None:
        if required:
            raise ModulePrerequisiteError(
                "sam3_containment",
                f"source_files.{name}.path",
            )
        return None
    path = context.batch_root / str(value)
    if not path.is_file():
        raise ModulePrerequisiteError(
            "sam3_containment",
            f"existing source_files.{name}.path",
        )
    return path


def _records_for_asset(path: Path, asset_id: str) -> list[dict[str, Any]]:
    from tools.run_manifest_sam3_containment import read_records

    return [row for row in read_records(path) if str(row.get("asset_id")) == asset_id]


def runner(segmenter_factory: Callable[..., Any] | None) -> ModuleRunner:
    def run(context: AssetContext, config: LoadedQcConfig) -> ModuleResult:
        from qc_pipeline.adapters.sam3_containment import adapt_sam3_containment
        from tools.run_manifest_sam3_containment import (
            read_records,
            run_manifest_sam3_containment,
        )

        canonical_episode = context.metadata.get("canonical_episode")
        bridge = None
        if canonical_episode is not None:
            source_root = context.metadata.get("canonical_source_root")
            if not isinstance(source_root, str) or not source_root:
                raise ModulePrerequisiteError(
                    "sam3_containment", "metadata.canonical_source_root"
                )
            from canonical_qc.bridge import CanonicalQcBridge

            bridge = CanonicalQcBridge(
                canonical_episode,
                source_root=Path(source_root),
            )

        candidate_path = _source_path(context, "candidate_windows")
        assert candidate_path is not None
        staging_parent = context.batch_root / ".qc_pipeline" / context.asset_id / "sam3"
        staging_parent.mkdir(parents=True, exist_ok=True)
        staging_root = Path(tempfile.mkdtemp(prefix="run-", dir=staging_parent))
        manifest_path = _source_path(context, "manifest", required=False)
        if bridge is not None:
            start, end = context.source_range or (
                0,
                canonical_episode.time_axis.frame_count,
            )
            import pandas as pd

            points = canonical_episode.observation.hand_keypoints_2d
            source_data = pd.DataFrame(
                {
                    "canonical_left_hand_2d": [row.reshape(-1) for row in points[:, 0]],
                    "canonical_right_hand_2d": [row.reshape(-1) for row in points[:, 1]],
                }
            )
            canonical_projection = staging_root / "canonical_keypoints_2d.parquet"
            source_data.to_parquet(canonical_projection, index=False)
            manifest_rows = [
                {
                    "asset_id": context.asset_id,
                    "episode_index": 0,
                    "start_frame": start,
                    "end_frame": end - 1,
                    "primary_video_path": str(bridge.video_path()),
                    "parquet_path": str(canonical_projection),
                    "left_hand_2d_field": "canonical_left_hand_2d",
                    "right_hand_2d_field": "canonical_right_hand_2d",
                }
            ]
            manifest_dir = context.batch_root
        elif manifest_path is not None:
            manifest_rows = _records_for_asset(manifest_path, context.asset_id)
            manifest_dir = manifest_path.parent
        else:
            declared_row = context.metadata.get("manifest_row")
            if not isinstance(declared_row, Mapping):
                raise ModulePrerequisiteError(
                    "sam3_containment",
                    "source_files.manifest.path or metadata.manifest_row",
                )
            manifest_rows = [dict(declared_row)]
            manifest_dir = context.batch_root
        if len(manifest_rows) != 1:
            raise ModulePrerequisiteError(
                "sam3_containment",
                "exactly one manifest row for asset_id",
            )
        manifest_row = manifest_rows[0]
        if bridge is None:
            for field, source_name in (
                ("primary_video_path", "video"),
                ("parquet_path", "parquet"),
            ):
                entry = _source_entry(context, source_name)
                if entry is not None:
                    manifest_row[field] = str(context.batch_root / str(entry["path"]))
                elif field in manifest_row:
                    value = Path(str(manifest_row[field]))
                    manifest_row[field] = str(
                        value if value.is_absolute() else manifest_dir / value
                    )
        candidate_rows = _records_for_asset(candidate_path, context.asset_id)

        single_manifest = staging_root / "manifest.jsonl"
        single_candidates = staging_root / "candidate_windows.jsonl"
        single_manifest.write_text(
            json.dumps(manifest_row, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        single_candidates.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for row in candidate_rows
            ),
            encoding="utf-8",
        )
        output_dir = staging_root / "output"
        segmenter = segmenter_factory() if segmenter_factory is not None else None
        model = _source_path(
            context,
            "sam3_model",
            required=segmenter is None,
        )
        source_cache = None
        if bridge is not None:
            from tools.run_manifest_sam3_containment import ManifestSourceCache

            class CanonicalSourceCache(ManifestSourceCache):
                def read_frame(self, path: Path, frame_idx: int):  # type: ignore[no-untyped-def]
                    physical_start, _ = canonical_episode.main_video.source_frame_range
                    return super().read_frame(path, physical_start + frame_idx)

            source_cache = CanonicalSourceCache()

        if bridge is not None:
            bridge.verify_sources()
        try:
            summary = run_manifest_sam3_containment(
                manifest=single_manifest,
                candidate_windows=single_candidates,
                supplier="canonical"
                if bridge is not None
                else str(context.metadata.get("supplier") or "jdt"),
                output_dir=output_dir,
                max_clips=1,
                sam3_model=model,
                overwrite=True,
                segmenter=segmenter,
                source_cache=source_cache,
                config_path=config.path,
                batch_root=staging_root,
                profile=str(context.metadata.get("profile") or "acceptance"),
            )
        finally:
            if source_cache is not None:
                source_cache.close()
        if bridge is not None:
            bridge.verify_sources()
        if int(summary.get("failed_asset_count", 0)):
            raise RuntimeError(f"sam3_containment producer failed: {summary}")
        window_summaries = read_records(
            output_dir / "window_keypoint_containment_summary.json"
        )
        frame_rows = read_records(output_dir / "frame_keypoint_containment.json")
        if bridge is not None:
            frame_rows = [{**row, "camera_id": "main"} for row in frame_rows]
        evidence_rows = read_records(output_dir / "review_evidence_manifest.csv")
        return adapt_sam3_containment(
            asset_id=context.asset_id,
            batch_root=context.batch_root,
            window_summaries=window_summaries,
            evidence_rows=evidence_rows,
            config=config,
            frame_rows=frame_rows,
            supplier_hand_quality_status=(
                None
                if canonical_episode is None
                or canonical_episode.supplier_evidence.hand_quality is None
                else canonical_episode.supplier_evidence.hand_quality.status
            ),
        )

    return run


__all__ = ["runner"]
