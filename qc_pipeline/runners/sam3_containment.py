"""Real manifest SAM3 producer and unified-adapter runner bridge."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
from time import perf_counter
from typing import Any, Literal

from qc_common.config import LoadedQcConfig
from qc_common.contracts import ModuleResult
from qc_common.module_registry import (
    ModuleAdapterMissingError,
    ModuleBlockedError,
    ModuleInputError,
    ModulePrerequisiteError,
    ModuleRunner,
)
from qc_pipeline.context import AssetContext
from qc_pipeline.artifacts import artifact_for
from qc_pipeline.sam3_runtime import SegmenterProvider


_IMPLEMENTATION_VERSION = "sam3-containment-producer-v2"
_MODEL_IDENTITY_FILES = ("config.json", "model.safetensors", "sam3.pt")
_MODEL_HASH_FILES = ("config.json",)


def _source_entry(context: AssetContext, name: str) -> Mapping[str, Any] | None:
    source = context.source_files.get(name)
    return source if isinstance(source, Mapping) else None


def _source_path(
    context: AssetContext,
    name: str,
    *,
    required: bool = True,
    expected_type: Literal["file", "directory"] = "file",
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
    if expected_type == "file":
        valid = path.is_file()
    elif expected_type == "directory":
        valid = path.is_dir()
    else:
        raise ValueError(f"unsupported source path type: {expected_type}")
    if not valid:
        raise ModulePrerequisiteError(
            "sam3_containment",
            f"existing {expected_type} source_files.{name}.path",
        )
    return path


def _records_for_asset(path: Path, asset_id: str) -> list[dict[str, Any]]:
    from tools.run_manifest_sam3_containment import read_records

    return [row for row in read_records(path) if str(row.get("asset_id")) == asset_id]


def _validated_candidates(
    context: AssetContext,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    clip_start: int | None = None
    clip_end: int | None = None
    if context.source_range is not None:
        clip_start, exclusive_end = context.source_range
        clip_end = exclusive_end - 1
    for index, row in enumerate(rows):
        if row.get("sam3_eligible") is not True:
            continue
        if str(row.get("asset_id") or "") != context.asset_id:
            raise ModuleInputError(
                "sam3_containment",
                f"candidate {index} asset_id does not match {context.asset_id}",
            )
        start = row.get("start_frame")
        end = row.get("end_frame")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
        ):
            raise ModuleInputError(
                "sam3_containment",
                f"candidate {index} source bounds must be integers",
            )
        if start > end:
            raise ModuleInputError(
                "sam3_containment",
                f"candidate {index} start_frame exceeds end_frame",
            )
        coordinate_space = str(row.get("coordinate_space") or "source").lower()
        frame_coordinates = str(
            row.get("frame_coordinate_system") or "source_inclusive"
        ).lower()
        if coordinate_space != "source" or frame_coordinates != "source_inclusive":
            raise ModuleInputError(
                "sam3_containment",
                f"candidate {index} must use source_inclusive coordinates",
            )
        if (
            clip_start is not None
            and clip_end is not None
            and not (clip_start <= start <= end <= clip_end)
        ):
            raise ModuleInputError(
                "sam3_containment",
                f"candidate {index} bounds {start}..{end} outside clip "
                f"{clip_start}..{clip_end}",
            )
        if "source_start_frame" in row and row["source_start_frame"] != start:
            raise ModuleInputError(
                "sam3_containment",
                f"candidate {index} source_start_frame disagrees with start_frame",
            )
        if "source_end_frame" in row and row["source_end_frame"] != end:
            raise ModuleInputError(
                "sam3_containment",
                f"candidate {index} source_end_frame disagrees with end_frame",
            )
        validated.append(dict(row))
    return validated


def _with_artifact_runtime(
    result: ModuleResult,
    *,
    state: str,
    elapsed_seconds: float,
    fingerprint_sha256: str,
) -> ModuleResult:
    return replace(
        result,
        runtime={
            **dict(result.runtime),
            "artifact_state": state,
            "elapsed_seconds": float(elapsed_seconds),
            "fingerprint_sha256": fingerprint_sha256,
        },
    )


def _publish_sam3_artifact(
    *,
    context: AssetContext,
    frame_results: list[dict[str, Any]],
    window_summaries: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
    producer_run_config: Mapping[str, Any],
    producer_root: Path,
    fingerprint: Mapping[str, Any],
    elapsed_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from qc_common.io import write_json_records
    from qc_pipeline.artifacts import (
        promote_artifact,
        staged_artifact,
        write_run_config,
    )

    artifact = artifact_for(context, "sam3_containment")
    rewritten: list[dict[str, Any]] = []
    with staged_artifact(artifact) as staging:
        evidence_dir = staging / "evidence"
        for index, source_row in enumerate(evidence_rows):
            row = dict(source_row)
            value = row.get("source_path")
            if not isinstance(value, str) or not value.strip():
                raise ValueError("SAM3 evidence row has no source_path")
            source = Path(value)
            if not source.is_absolute():
                source = producer_root / source
            if not source.is_file():
                raise ValueError(f"SAM3 evidence file does not exist: {source}")
            evidence_dir.mkdir(parents=True, exist_ok=True)
            destination = evidence_dir / f"{index:04d}-{source.name}"
            shutil.copy2(source, destination)
            final_path = artifact.directory / "evidence" / destination.name
            row["source_path"] = final_path.relative_to(context.batch_root).as_posix()
            rewritten.append(row)
        write_json_records(frame_results, staging / "frame_results.json")
        write_json_records(window_summaries, staging / "window_results.json")
        write_json_records(failures, staging / "failures.json")
        write_json_records(rewritten, staging / "evidence_manifest.json")
        (staging / "producer_run_config.json").write_text(
            json.dumps(
                dict(producer_run_config),
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        write_run_config(
            staging,
            producer="sam3_containment",
            outcome="completed",
            fingerprint=fingerprint,
            elapsed_seconds=elapsed_seconds,
        )
        promote_artifact(staging, artifact)
    return window_summaries, rewritten


def _run_canonical(
    context: AssetContext,
    config: LoadedQcConfig,
    segmenter_factory: Callable[..., Any] | None,
    segmenter_provider: SegmenterProvider | None,
) -> ModuleResult:
    """Run SAM3 from CanonicalEpisode without exposing supplier source layouts."""
    import pandas as pd

    from canonical_qc.bridge import CanonicalQcBridge
    from qc_pipeline.adapters.sam3_containment import adapt_sam3_containment
    from tools.run_manifest_sam3_containment import (
        ManifestSourceCache,
        SAM3_CONFIG,
        read_records,
        run_manifest_sam3_containment,
    )

    episode = context.metadata.get("canonical_episode")
    source_root = context.metadata.get("canonical_source_root")
    if not isinstance(source_root, str) or not source_root:
        raise ModulePrerequisiteError(
            "sam3_containment", "metadata.canonical_source_root"
        )
    bridge = CanonicalQcBridge(episode, source_root=Path(source_root))
    candidate_path = _source_path(context, "candidate_windows")
    assert candidate_path is not None
    candidate_rows = _records_for_asset(candidate_path, context.asset_id)

    staging_parent = context.batch_root / ".qc_pipeline" / context.asset_id / "sam3"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix="run-", dir=staging_parent))
    start, end = context.source_range or (0, episode.time_axis.frame_count)
    points = episode.observation.hand_keypoints_2d
    projection = staging_root / "canonical_keypoints_2d.parquet"
    pd.DataFrame(
        {
            "canonical_left_hand_2d": [row.reshape(-1) for row in points[:, 0]],
            "canonical_right_hand_2d": [row.reshape(-1) for row in points[:, 1]],
        }
    ).to_parquet(projection, index=False)
    manifest_row = {
        "asset_id": context.asset_id,
        "episode_index": 0,
        "start_frame": start,
        "end_frame": end - 1,
        "primary_video_path": str(bridge.video_path()),
        "parquet_path": str(projection),
        "left_hand_2d_field": "canonical_left_hand_2d",
        "right_hand_2d_field": "canonical_right_hand_2d",
    }
    single_manifest = staging_root / "manifest.jsonl"
    single_candidates = staging_root / "candidate_windows.jsonl"
    single_manifest.write_text(
        json.dumps(manifest_row, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    single_candidates.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in candidate_rows),
        encoding="utf-8",
    )
    model = _source_path(
        context,
        "sam3_model",
        required=segmenter_factory is None,
        expected_type="directory",
    )
    if segmenter_factory is not None:
        segmenter = segmenter_factory()
    elif segmenter_provider is not None:
        assert model is not None
        segmenter = segmenter_provider(model, dict(SAM3_CONFIG))
    else:
        segmenter = None

    class CanonicalSourceCache(ManifestSourceCache):
        def read_frame(self, path: Path, frame_idx: int):  # type: ignore[no-untyped-def]
            physical_start, _ = episode.main_video.source_frame_range
            return super().read_frame(path, physical_start + frame_idx)

    source_cache = CanonicalSourceCache()
    bridge.verify_sources()
    try:
        summary = run_manifest_sam3_containment(
            manifest=single_manifest,
            candidate_windows=single_candidates,
            supplier="canonical",
            output_dir=staging_root / "output",
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
        source_cache.close()
    bridge.verify_sources()
    if int(summary.get("failed_asset_count", 0)):
        raise RuntimeError(f"sam3_containment producer failed: {summary}")
    output_dir = staging_root / "output"
    window_rows = read_records(
        output_dir / "window_keypoint_containment_summary.json"
    )
    frame_rows = [
        {**row, "camera_id": "main"}
        for row in read_records(output_dir / "frame_keypoint_containment.json")
    ]
    evidence_rows = read_records(output_dir / "review_evidence_manifest.csv")
    hand_quality = episode.supplier_evidence.hand_quality
    return adapt_sam3_containment(
        asset_id=context.asset_id,
        batch_root=context.batch_root,
        window_summaries=window_rows,
        evidence_rows=evidence_rows,
        config=config,
        frame_rows=frame_rows,
        supplier_hand_quality_status=(
            hand_quality.status
            if hand_quality is not None and hand_quality.provided
            else None
        ),
    )


def runner(
    segmenter_factory: Callable[..., Any] | None,
    *,
    segmenter_provider: SegmenterProvider | None = None,
) -> ModuleRunner:
    def run(context: AssetContext, config: LoadedQcConfig) -> ModuleResult:
        if context.metadata.get("canonical_episode") is not None:
            return _run_canonical(
                context,
                config,
                segmenter_factory,
                segmenter_provider,
            )

        from qc_pipeline.adapters.sam3_containment import adapt_sam3_containment
        from tools.run_manifest_sam3_containment import (
            DEFAULT_QUERIES,
            SAM3_CONFIG,
            read_records,
            run_manifest_sam3_containment,
        )
        from qc_pipeline.artifacts import (
            build_run_fingerprint,
            canonical_sha256,
            directory_identity,
            file_sha256,
            reusable_artifact,
        )

        started = perf_counter()

        supplier = str(
            context.metadata.get("supplier")
            or context.metadata.get("supplier_id")
            or "jdt"
        ).lower()
        if supplier in {"dr", "deepreach"}:
            projection_status = str(
                context.metadata.get("projection_validation_status") or ""
            ).lower()
            if projection_status != "validated":
                reason = (
                    "transform_ambiguous"
                    if projection_status == "transform_ambiguous"
                    else "calibration_unverified"
                )
                raise ModuleBlockedError("sam3_containment", reason)
            raise ModuleBlockedError("sam3_containment", "adapter_missing")
        if supplier == "potentia":
            raise ModuleBlockedError("sam3_containment", "no_keypoint_input")

        candidate_path = artifact_for(context, "precheck").directory / "candidate_windows.json"
        if not candidate_path.is_file():
            raise ModulePrerequisiteError(
                "sam3_containment",
                "current precheck candidate_windows.json",
            )
        precheck_run_config = candidate_path.parent / "run_config.json"
        if not precheck_run_config.is_file():
            raise ModulePrerequisiteError(
                "sam3_containment",
                "current precheck run_config.json",
            )
        try:
            precheck_run = json.loads(
                precheck_run_config.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModuleInputError(
                "sam3_containment",
                f"current precheck run_config is unreadable: {exc}",
            ) from exc
        from qc_pipeline.runners.precheck import precheck_fingerprint

        expected_precheck = precheck_fingerprint(context, config)
        completed_precheck_modules = precheck_run.get("completed_modules")
        if (
            precheck_run.get("producer") != "precheck"
            or precheck_run.get("outcome") not in {"completed", "partial"}
            or precheck_run.get("fingerprint") != expected_precheck
            or not isinstance(completed_precheck_modules, list)
            or "keypoint_temporal" not in completed_precheck_modules
        ):
            raise ModuleInputError(
                "sam3_containment",
                "current precheck run does not match this asset/config or lacks temporal output",
            )
        temporal_output = precheck_run.get("temporal_output")
        if (
            not isinstance(temporal_output, Mapping)
            or temporal_output.get("status") != "valid"
            or not isinstance(temporal_output.get("valid_frame_count"), int)
            or int(temporal_output["valid_frame_count"]) <= 0
        ):
            raise ModuleBlockedError(
                "sam3_containment",
                "no_valid_temporal_output",
            )
        try:
            all_candidate_rows = read_records(candidate_path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ModuleInputError(
                "sam3_containment",
                f"candidate artifact is unreadable: {exc}",
            ) from exc
        candidate_rows = _validated_candidates(context, all_candidate_rows)
        if not candidate_rows:
            return ModuleResult(
                module="sam3_containment",
                verdict="skipped",
                evaluation={"decision": "skipped", "reason": "no_candidates"},
                metrics={"window_count": 0},
                runtime={"artifact_state": "no_candidates"},
            )

        if supplier != "jdt":
            raise ModuleAdapterMissingError(
                "sam3_containment",
                f"supplier adapter is not implemented: {supplier}",
            )
        for source_name in ("video", "parquet"):
            _source_path(context, source_name, required=False)
        manifest_path = _source_path(context, "manifest", required=False)
        model = _source_path(
            context,
            "sam3_model",
            required=segmenter_factory is None,
            expected_type="directory",
        )
        source_names = tuple(
            name
            for name in ("video", "parquet")
            if name in context.source_files
        )
        fingerprint = build_run_fingerprint(
            context=context,
            producer="sam3_containment",
            config=config,
            module_names=("sam3_containment",),
            source_names=source_names,
            implementation_version=_IMPLEMENTATION_VERSION,
            extra={
                "candidate_sha256": file_sha256(candidate_path),
                "queries": DEFAULT_QUERIES,
                "sam3_runtime": SAM3_CONFIG,
            },
        )
        if model is not None:
            fingerprint["sources"]["sam3_model"] = directory_identity(
                model,
                batch_root=context.batch_root,
                key_files=_MODEL_IDENTITY_FILES,
                hash_files=_MODEL_HASH_FILES,
                declared=_source_entry(context, "sam3_model"),
                allow_symlinked_sources=context.allow_symlinked_sources,
            )
        fingerprint_sha256 = canonical_sha256(fingerprint)
        artifact = artifact_for(context, "sam3_containment")
        if bool(context.metadata.get("reuse_artifacts", True)) and reusable_artifact(
            artifact, fingerprint
        ):
            window_summaries = read_records(artifact.directory / "window_results.json")
            evidence_rows = read_records(artifact.directory / "evidence_manifest.json")
            result = adapt_sam3_containment(
                asset_id=context.asset_id,
                batch_root=context.batch_root,
                window_summaries=window_summaries,
                evidence_rows=evidence_rows,
                config=config,
            )
            return _with_artifact_runtime(
                result,
                state="reused",
                elapsed_seconds=perf_counter() - started,
                fingerprint_sha256=fingerprint_sha256,
            )
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
        if segmenter_factory is not None:
            segmenter = segmenter_factory()
        elif segmenter_provider is not None:
            assert model is not None
            segmenter = segmenter_provider(model, dict(SAM3_CONFIG))
        else:
            segmenter = None
        summary = run_manifest_sam3_containment(
            manifest=single_manifest,
            candidate_windows=single_candidates,
            supplier=supplier,
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
        frame_path = output_dir / "frame_keypoint_containment.json"
        failures_path = output_dir / "failures.json"
        producer_config_path = output_dir / "run_config.json"
        frame_results = read_records(frame_path) if frame_path.is_file() else []
        failures = read_records(failures_path) if failures_path.is_file() else []
        producer_run_config = (
            json.loads(producer_config_path.read_text(encoding="utf-8"))
            if producer_config_path.is_file()
            else {}
        )
        elapsed = perf_counter() - started
        window_summaries, evidence_rows = _publish_sam3_artifact(
            context=context,
            frame_results=frame_results,
            window_summaries=window_summaries,
            failures=failures,
            evidence_rows=evidence_rows,
            producer_run_config=producer_run_config,
            producer_root=staging_root,
            fingerprint=fingerprint,
            elapsed_seconds=elapsed,
        )
        result = adapt_sam3_containment(
            asset_id=context.asset_id,
            batch_root=context.batch_root,
            window_summaries=window_summaries,
            evidence_rows=evidence_rows,
            config=config,
        )
        return _with_artifact_runtime(
            result,
            state="computed",
            elapsed_seconds=elapsed,
            fingerprint_sha256=fingerprint_sha256,
        )

    return run


__all__ = ["runner"]
