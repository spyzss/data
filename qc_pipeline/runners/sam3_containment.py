"""Real manifest SAM3 producer and unified-adapter runner bridge."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
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

        candidate_path = _source_path(context, "candidate_windows")
        assert candidate_path is not None
        manifest_path = _source_path(context, "manifest", required=False)
        if manifest_path is not None:
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

        staging_root = context.batch_root / ".qc_pipeline" / context.asset_id / "sam3"
        staging_root.mkdir(parents=True, exist_ok=True)
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
        summary = run_manifest_sam3_containment(
            manifest=single_manifest,
            candidate_windows=single_candidates,
            supplier=str(context.metadata.get("supplier") or "jdt"),
            output_dir=output_dir,
            max_clips=1,
            sam3_model=model,
            overwrite=True,
            segmenter=segmenter,
            config_path=config.path,
            batch_root=staging_root,
            profile=str(context.metadata.get("profile") or "acceptance"),
        )
        if int(summary.get("failed_asset_count", 0)):
            raise RuntimeError(f"sam3_containment producer failed: {summary}")
        window_summaries = read_records(
            output_dir / "window_keypoint_containment_summary.json"
        )
        evidence_rows = read_records(output_dir / "review_evidence_manifest.csv")
        return adapt_sam3_containment(
            asset_id=context.asset_id,
            batch_root=context.batch_root,
            window_summaries=window_summaries,
            evidence_rows=evidence_rows,
            config=config,
        )

    return run


__all__ = ["runner"]
