#!/usr/bin/env python3
"""Run the unified QC pipeline for manifest-defined assets."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from dataclasses import replace
import json
import math
import os
import sys
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qc_common.config import LoadedQcConfig, load_qc_acceptance_config  # noqa: E402
from qc_common.manifest_metadata import (  # noqa: E402
    canonical_manifest_metadata,
    canonicalize_manifest_value,
)
from qc_common.module_registry import ModuleRegistry  # noqa: E402
from qc_common.suppliers import normalize_supplier  # noqa: E402
from qc_pipeline.context import AssetContext, validate_asset_id  # noqa: E402
from qc_pipeline.sam3_runtime import Sam3RuntimeProvider  # noqa: E402
from qc_pipeline.orchestrator import (  # noqa: E402
    RunOutcome,
    build_default_registry,
    run_asset,
)
from tools.run_manifest_precheck import read_manifest  # noqa: E402


RegistryFactory = Callable[[AssetContext], ModuleRegistry]
_SOURCE_COLUMNS = {
    "video": ("primary_video_path", "video_path"),
    "head_video": ("head_video_path",),
    "left_wrist_video": ("left_wrist_video_path",),
    "right_wrist_video": ("right_wrist_video_path",),
    "hdf5": ("hdf5_path",),
    "parquet": ("parquet_path",),
    "observations_2d": ("observations_2d_path",),
    "trajectory_3d": ("trajectory_3d_path",),
    "coordinate_system": ("coordinate_system_path",),
    "quality": ("quality_path",),
    "timebase": ("timebase_path",),
    "semantic": ("semantic_path",),
    "episode_manifest": ("episode_manifest_path",),
    "review_video": ("review_video_path",),
    "qy_left_cam_left_video": ("left_cam_left_video_path",),
    "qy_left_cam_right_video": ("left_cam_right_video_path",),
    "qy_mid_cam_left_video": ("mid_cam_left_video_path",),
    "qy_mid_cam_right_video": ("mid_cam_right_video_path",),
    "qy_right_cam_left_video": ("right_cam_left_video_path",),
    "qy_right_cam_right_video": ("right_cam_right_video_path",),
    "lerobot_task": ("lerobot_task_dir",),
    "task_dir": ("task_dir",),
    "calibration": ("calib_path", "calibration_path"),
    "trajectory": ("camera_trajectory_path",),
    "meta": ("meta_path",),
    "frames": ("frames_path",),
    "aligned": ("aligned_path",),
    "imu": ("imu_path",),
    "candidate_windows": ("candidate_windows_path", "candidate_windows"),
    "sam3_model": ("sam3_model", "sam3_model_path"),
}


def _text(value: Any) -> str:
    canonical = canonicalize_manifest_value(value)
    return "" if canonical is None else str(canonical).strip()


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be an integer") from None
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{field} must be an integer")
    return int(numeric)


def _path_inside_batch(
    value: Any,
    *,
    batch_root: Path,
    manifest_dir: Path,
    field: str,
    allow_symlinked_sources: bool = False,
) -> tuple[str, str]:
    text = _text(value)
    if not text:
        raise ValueError(f"{field} must be a non-empty path")

    candidate = Path(text).expanduser()
    candidate_path = (
        candidate
        if candidate.is_absolute()
        else manifest_dir / candidate
    )

    # lexical_absolute preserves a symlink located inside batch_root.
    # resolved_absolute records the real supplier/model source path.
    lexical_absolute = Path(
        os.path.abspath(os.fspath(candidate_path))
    )
    resolved_absolute = lexical_absolute.resolve()

    try:
        logical_relative = lexical_absolute.relative_to(batch_root)
    except ValueError:
        raise ValueError(
            f"{field} is outside batch_root: {resolved_absolute}"
        ) from None

    if not allow_symlinked_sources:
        try:
            resolved_absolute.relative_to(batch_root)
        except ValueError:
            raise ValueError(
                f"{field} is outside batch_root: {resolved_absolute}"
            ) from None

    return logical_relative.as_posix(), str(resolved_absolute)


def contexts_from_manifest(
    manifest: Path,
    *,
    batch_root: Path,
    allow_symlinked_sources: bool = False,
) -> list[AssetContext]:
    """Build source-faithful, independent contexts from manifest rows."""
    manifest = manifest.resolve()
    batch_root = batch_root.resolve()
    rows = read_manifest(manifest)
    seen: set[str] = set()
    for row_index, row in enumerate(rows):
        asset_id = _text(row.get("asset_id"))
        if not asset_id:
            raise ValueError(f"manifest row {row_index} missing asset_id")
        validate_asset_id(asset_id)
        if asset_id in seen:
            raise ValueError(f"duplicate asset_id: {asset_id}")
        seen.add(asset_id)

    contexts: list[AssetContext] = []
    for row_index, source_row in enumerate(rows):
        canonical_row = canonical_manifest_metadata(source_row)
        row = copy.deepcopy(canonical_row)
        raw_manifest_row = copy.deepcopy(canonical_row)
        supplier_value = row.get("supplier") or row.get("supplier_id")
        if _text(supplier_value):
            supplier_id, supplier_name, supplier_alias = normalize_supplier(
                supplier_value
            )
            row["supplier"] = supplier_id
            row["supplier_id"] = supplier_id
            row["supplier_name"] = supplier_name
            if supplier_alias is not None:
                row["supplier_alias"] = supplier_alias
        asset_id = _text(row.get("asset_id"))
        source_files: dict[str, Any] = {}
        for source_name, columns in _SOURCE_COLUMNS.items():
            if source_name == "candidate_windows" and not (
                _text(row.get("canonical_format"))
                or _text(row.get("canonical_source_path"))
            ):
                continue
            column = next((name for name in columns if _text(row.get(name))), None)
            if column is None:
                continue
            relative, absolute = _path_inside_batch(
                row[column],
                batch_root=batch_root,
                manifest_dir=manifest.parent,
                field=column,
                allow_symlinked_sources=allow_symlinked_sources,
            )
            source_files[source_name] = {"path": relative}
            raw_manifest_row[column] = relative
            row[column] = absolute

        canonical_format = _text(row.get("canonical_format")).lower()
        canonical_source = _text(row.get("canonical_source_path"))
        canonical = bool(canonical_format or canonical_source)
        start_text = _text(row.get("start_frame"))
        inclusive_end_text = _text(row.get("end_frame"))
        exclusive_end_text = _text(row.get("end_frame_exclusive"))
        source_range: tuple[int, int] | None = None
        if canonical:
            if inclusive_end_text:
                raise ValueError(
                    f"manifest row {row_index} canonical ranges require "
                    "end_frame_exclusive; end_frame is not allowed"
                )
            if start_text or exclusive_end_text:
                if not start_text or not exclusive_end_text:
                    raise ValueError(
                        f"manifest row {row_index} must define both start_frame "
                        "and end_frame_exclusive"
                    )
                source_range = (
                    _integer(row.get("start_frame"), "start_frame"),
                    _integer(row.get("end_frame_exclusive"), "end_frame_exclusive"),
                )
        else:
            if exclusive_end_text:
                raise ValueError(
                    f"manifest row {row_index} legacy ranges use inclusive end_frame"
                )
            if start_text or inclusive_end_text:
                if not start_text or not inclusive_end_text:
                    raise ValueError(
                        f"manifest row {row_index} must define both start_frame and end_frame"
                    )
                start = _integer(row.get("start_frame"), "start_frame")
                inclusive_end = _integer(row.get("end_frame"), "end_frame")
                source_range = (start, inclusive_end + 1)

        if canonical:
            if canonical_format not in {"hdf5", "lerobot"}:
                raise ValueError(
                    f"manifest row {row_index} canonical_format must be hdf5 or lerobot"
                )
            if not canonical_source:
                raise ValueError(
                    f"manifest row {row_index} missing canonical_source_path"
                )
            relative, absolute = _path_inside_batch(
                canonical_source,
                batch_root=batch_root,
                manifest_dir=manifest.parent,
                field="canonical_source_path",
                allow_symlinked_sources=allow_symlinked_sources,
            )
            raw_manifest_row["canonical_source_path"] = relative
            from canonical_qc import StandardHdf5Adapter, StandardLeRobotAdapter
            from canonical_qc.bridge import CanonicalQcBridge

            if canonical_format == "hdf5":
                episode = StandardHdf5Adapter().load(Path(absolute))
            else:
                episode_index = row.get("episode_index")
                selected = (
                    None
                    if episode_index is None or not _text(episode_index)
                    else _integer(episode_index, "episode_index")
                )
                episode = StandardLeRobotAdapter().load(
                    Path(absolute), episode_index=selected
                )
            if episode.identity.asset_id != asset_id:
                raise ValueError(
                    f"manifest asset_id {asset_id!r} does not match canonical episode "
                    f"{episode.identity.asset_id!r}"
                )
            contexts.append(
                CanonicalQcBridge(
                    episode,
                    source_root=(
                        Path(absolute)
                        if Path(absolute).is_dir()
                        else Path(absolute).parent
                    ),
                ).asset_context(
                    batch_root=batch_root,
                    report_path=batch_root / "quality_archive" / f"{asset_id}.json",
                    source_range=source_range,
                    metadata={**row, "manifest_row": raw_manifest_row},
                    supplemental_source_files=source_files,
                )
            )
            continue

        contexts.append(
            AssetContext(
                asset_id=asset_id,
                batch_root=batch_root,
                report_path=batch_root / "quality_archive" / f"{asset_id}.json",
                source_files=source_files,
                allow_symlinked_sources=allow_symlinked_sources,
                source_range=source_range,
                metadata={
                    **row,
                    "manifest_row": raw_manifest_row,
                },
            )
        )
    return contexts


def apply_dr_heuristic_projection_mapping(
    contexts: Iterable[AssetContext],
    mapping_path: Path,
) -> list[AssetContext]:
    """Overlay user-confirmed DR HFOV choices without mutating manifest identity."""
    context_list = list(contexts)
    mapping_path = Path(mapping_path).resolve()
    if not mapping_path.is_file():
        raise FileNotFoundError(mapping_path)
    rows_by_asset: dict[str, dict[str, Any]] = {}
    for index, source in enumerate(read_manifest(mapping_path)):
        row = canonical_manifest_metadata(source)
        asset_id = _text(row.get("asset_id"))
        if not asset_id:
            raise ValueError(f"DR heuristic mapping row {index} missing asset_id")
        if asset_id in rows_by_asset:
            raise ValueError(f"duplicate DR heuristic mapping asset_id: {asset_id}")
        if (
            _text(row.get("projection_mode")) != "approx_pinhole_from_hfov"
            or _text(row.get("calibration_status")) != "heuristic"
            or _text(row.get("projection_validation_status"))
            != "pending_visual_validation"
            or _text(row.get("selected_by")) != "user_visual_confirmation"
            or _text(row.get("distortion_applied")).lower() not in {"false", "0"}
            or _text(row.get("camera_trajectory_applied")).lower()
            not in {"false", "0"}
        ):
            raise ValueError(
                f"invalid DR heuristic projection contract for asset_id: {asset_id}"
            )
        rows_by_asset[asset_id] = row

    context_ids = {context.asset_id for context in context_list}
    unknown = sorted(set(rows_by_asset) - context_ids)
    if unknown:
        raise ValueError(
            "DR heuristic mapping contains assets absent from manifest: "
            + ", ".join(unknown)
        )

    updated: list[AssetContext] = []
    for context in context_list:
        row = rows_by_asset.get(context.asset_id)
        if row is None:
            updated.append(context)
            continue
        supplier = _text(
            context.metadata.get("supplier")
            or context.metadata.get("supplier_id")
        ).lower()
        if supplier not in {"dr", "deepreach"}:
            raise ValueError(
                f"DR heuristic mapping targets non-DR asset: {context.asset_id}"
            )
        overlay = {
            key: row.get(key)
            for key in (
                "projection_mode",
                "head_hfov_deg",
                "fx",
                "fy",
                "cx",
                "cy",
                "image_width",
                "image_height",
                "calibration_status",
                "projection_validation_status",
                "distortion_applied",
                "camera_trajectory_applied",
                "selected_by",
            )
        }
        overlay["dr_heuristic_projection_mapping_path"] = str(mapping_path)
        updated.append(
            replace(
                context,
                metadata={**context.metadata, **overlay},
            )
        )
    return updated


def run_batch(
    contexts: Iterable[AssetContext],
    *,
    config: LoadedQcConfig,
    profile: str,
    registry_factory: RegistryFactory | None = None,
    max_workers: int = 1,
    resume: bool = True,
) -> dict[str, RunOutcome]:
    """Schedule assets concurrently while keeping registry/report state isolated."""
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    context_list = list(contexts)
    seen: set[str] = set()
    for context in context_list:
        if context.asset_id in seen:
            raise ValueError(f"duplicate asset_id: {context.asset_id}")
        seen.add(context.asset_id)
    if not resume:
        existing = [
            context.asset_id
            for context in context_list
            if context.report_path.exists()
        ]
        if existing:
            raise FileExistsError(
                "--no-resume requires fresh report paths: " + ", ".join(existing)
            )

    if registry_factory is None:
        sam3_runtime = Sam3RuntimeProvider()

        def factory(context: AssetContext) -> ModuleRegistry:
            return build_default_registry(
                context,
                config,
                segmenter_provider=sam3_runtime.get_segmenter,
            )

    else:
        factory = registry_factory

    def run_one(context: AssetContext) -> RunOutcome:
        started = perf_counter()
        runtime_context = replace(
            context,
            metadata={
                **dict(context.metadata),
                "reuse_artifacts": resume,
                "profile": profile,
            },
        )
        registry = factory(runtime_context)
        if not isinstance(registry, ModuleRegistry):
            raise TypeError("registry_factory must return ModuleRegistry")
        outcome = run_asset(
            runtime_context,
            config=config,
            profile=profile,
            registry=registry,
        )
        return replace(outcome, elapsed_seconds=perf_counter() - started)

    outcomes: dict[str, RunOutcome] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_asset = {
            pool.submit(run_one, context): context.asset_id
            for context in context_list
        }
        for future in as_completed(future_to_asset):
            asset_id = future_to_asset[future]
            outcomes[asset_id] = future.result()
    return outcomes


def _module_artifact_state(report: dict[str, Any], module: str) -> str | None:
    block = report.get(module)
    runtime = block.get("runtime") if isinstance(block, dict) else None
    state = runtime.get("artifact_state") if isinstance(runtime, dict) else None
    if isinstance(state, str):
        return "skipped" if state == "no_candidates" else state
    execution = report.get("execution")
    module_states = execution.get("module_states") if isinstance(execution, dict) else None
    recorded = module_states.get(module) if isinstance(module_states, dict) else None
    execution_state = recorded.get("state") if isinstance(recorded, dict) else None
    if execution_state in {"blocked", "adapter_missing", "input_missing"}:
        return "blocked"
    if execution_state in {"runtime_error", "input_invalid", "not_implemented"}:
        return "failed"
    if execution_state == "not_run":
        return "not_run"
    return None


def summarize_outcome(outcome: RunOutcome) -> dict[str, Any]:
    report = outcome.report
    precheck_states = [
        state
        for module in (
            "hdf5_text_info",
            "quality_hand",
            "keypoint_presence",
            "keypoint_morphology",
            "keypoint_temporal",
        )
        if (state := _module_artifact_state(report, module)) is not None
    ]
    producers: dict[str, str] = {}
    if precheck_states:
        producers["precheck"] = (
            "computed"
            if "computed" in precheck_states
            else (
                "reused"
                if all(state == "reused" for state in precheck_states)
                else precheck_states[-1]
            )
        )
    for producer in ("supplier_data_audit", "video_quality", "sam3_containment"):
        state = _module_artifact_state(report, producer)
        if state is not None:
            producers[producer] = state
    counts = Counter(producers.values())
    return {
        "status": outcome.status,
        "report_revision": report.get("report_revision"),
        "executed_modules": list(outcome.executed_modules),
        "producers": producers,
        "producer_counts": dict(sorted(counts.items())),
        "elapsed_seconds": float(outcome.elapsed_seconds),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--allow-symlinked-sources",
        action="store_true",
        help=(
            "Allow manifest source paths to be symlinks located inside "
            "batch_root whose real targets are on external data/model mounts."
        ),
    )
    parser.add_argument(
        "--profile",
        required=True,
        choices=("acceptance", "supplier_evaluation"),
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--dr-heuristic-projection-map",
        type=Path,
        help=(
            "Explicit user-confirmed heuristic DR head projection mapping. "
            "Applied as runtime metadata; the canonical manifest is unchanged."
        ),
    )
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_qc_acceptance_config(args.config)
    contexts = contexts_from_manifest(
        args.manifest,
        batch_root=args.batch_root,
        allow_symlinked_sources=args.allow_symlinked_sources,
    )
    if args.dr_heuristic_projection_map is not None:
        contexts = apply_dr_heuristic_projection_mapping(
            contexts,
            args.dr_heuristic_projection_map,
        )
    outcomes = run_batch(
        contexts,
        config=config,
        profile=args.profile,
        max_workers=args.max_workers,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                asset_id: summarize_outcome(outcome)
                for asset_id, outcome in sorted(outcomes.items())
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
