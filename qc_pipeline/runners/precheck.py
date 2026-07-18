"""Real precheck producer and unified-adapter runner bridges."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from qc_common.config import LoadedQcConfig
from qc_common.contracts import ModuleResult
from qc_common.frame_survival import FrameSurvivalState
from qc_common.manifest_metadata import (
    manifest_metadata,
    normalized_manifest_text_metadata,
)
from qc_common.module_registry import ModuleInputError, ModulePrerequisiteError, ModuleRunner
from qc_common.types import ClipInputs
from qc_pipeline.context import AssetContext


MODULES = (
    "hdf5_text_info",
    "quality_hand",
    "keypoint_presence",
    "keypoint_morphology",
    "keypoint_temporal",
)
_IMPLEMENTATION_VERSION = "precheck-session-v7-calibrated-temporal-validity"
_TEMPORAL_OUTPUT_SCHEMA_VERSION = "keypoint_temporal.output.v2"
_FRAME_SURVIVAL_MODULES = frozenset(
    {"keypoint_presence", "keypoint_morphology", "keypoint_temporal"}
)


@dataclass(frozen=True)
class PrecheckModuleExecution:
    result: ModuleResult
    check_results: tuple[Any, ...]
    candidate_windows: tuple[Mapping[str, Any], ...] = ()


def precheck_fingerprint(
    context: AssetContext,
    config: LoadedQcConfig,
) -> dict[str, Any]:
    from qc_pipeline.artifacts import build_run_fingerprint

    source_names = tuple(
        name for name in ("hdf5", "parquet") if name in context.source_files
    )
    fingerprint = build_run_fingerprint(
        context=context,
        producer="precheck",
        config=config,
        module_names=MODULES,
        source_names=source_names,
        implementation_version=_IMPLEMENTATION_VERSION,
    )
    fingerprint["manifest_text_metadata"] = normalized_manifest_text_metadata(
        context.metadata
    )
    fingerprint["temporal_output_schema_version"] = (
        _TEMPORAL_OUTPUT_SCHEMA_VERSION
    )
    source_contract: dict[str, Any] = {
        "hdf5_reference_dataset": context.metadata.get(
            "hdf5_reference_dataset"
        )
    }
    supplier = str(
        context.metadata.get("supplier")
        or context.metadata.get("supplier_id")
        or ""
    ).lower()
    if supplier in {"dr", "deepreach"}:
        source_contract["supplier_adapter"] = "deepreach-hdf5-precheck-v2"
    fingerprint["source_contract"] = source_contract
    return fingerprint


def _source_entry(context: AssetContext, name: str) -> Mapping[str, Any] | None:
    value = context.source_files.get(name)
    return value if isinstance(value, Mapping) else None


def _source_path(context: AssetContext, module: str, name: str) -> Path:
    entry = _source_entry(context, name)
    value = None if entry is None else entry.get("path")
    if value is None:
        raise ModulePrerequisiteError(module, f"source_files.{name}.path")
    path = context.batch_root / str(value)
    if not path.is_file():
        raise ModulePrerequisiteError(module, f"existing source_files.{name}.path")
    return path


def _source_relative_path(context: AssetContext) -> str:
    for name in ("hdf5", "parquet"):
        entry = _source_entry(context, name)
        if entry is not None and isinstance(entry.get("path"), str):
            return str(entry["path"])
    raise ModulePrerequisiteError("precheck", "hdf5 or parquet source path")


def _slice_clip(clip: Any, source_range: tuple[int, int]) -> Any:
    from qc_common.types import ClipInputs

    start, end = source_range

    def sliced(values: Any) -> Any:
        if values is None:
            return None
        if isinstance(values, Mapping):
            return {name: value[start:end] for name, value in values.items()}
        return values[start:end]

    selected = ClipInputs(
        episode_idx=clip.episode_idx,
        frame_indices=list(range(start, end)),
        keypoints=sliced(clip.keypoints),
        rotations=sliced(clip.rotations),
        confidences=sliced(clip.confidences),
        quality_hand=sliced(clip.quality_hand),
        masks=sliced(clip.masks),
        instruction=clip.instruction,
        text_label=clip.text_label,
        manifest_metadata=clip.manifest_metadata,
        text_label_raw=clip.text_label_raw,
        text_label_parse_error=clip.text_label_parse_error,
        intrinsics=clip.intrinsics,
        supplier_hand_quality_status=sliced(
            clip.supplier_hand_quality_status
        ),
        hand_keypoints_3d=sliced(clip.hand_keypoints_3d),
        hand_joint_valid_3d=sliced(clip.hand_joint_valid_3d),
        timestamps_ns=sliced(clip.timestamps_ns),
        hand_keypoints_2d=sliced(clip.hand_keypoints_2d),
        hand_joint_valid_2d=sliced(clip.hand_joint_valid_2d),
        fps=clip.fps,
    )
    for name in (
        "asset_id",
        "supplier_id",
        "source_path",
        "supplier_quality_signal",
        "morphology_status",
    ):
        if hasattr(clip, name):
            setattr(selected, name, getattr(clip, name))
    setattr(selected, "clip_start_frame", start)
    setattr(selected, "clip_end_frame", end - 1)
    return selected


def _load_clip(context: AssetContext, module: str) -> Any:
    canonical_episode = context.metadata.get("canonical_episode")
    if canonical_episode is not None:
        source_root = context.metadata.get("canonical_source_root")
        if not isinstance(source_root, str) or not source_root:
            raise ModulePrerequisiteError(module, "metadata.canonical_source_root")
        from canonical_qc.bridge import CanonicalQcBridge

        return CanonicalQcBridge(
            canonical_episode,
            source_root=Path(source_root),
        ).clip_inputs(context.source_range)

    declared_clip = context.metadata.get("clip_inputs")
    if declared_clip is not None:
        return declared_clip

    source_range = context.source_range
    supplier = str(
        context.metadata.get("supplier")
        or context.metadata.get("supplier_id")
        or ""
    ).lower()
    if supplier in {"dr", "deepreach"}:
        if source_range is None:
            raise ModulePrerequisiteError(module, "source_range for DeepReach precheck")
        from tools.run_manifest_precheck import load_deepreach_clip
        from acceptance_pull.supplier_adapters.deepreach_hdf5 import (
            DeepReachFrameContractError,
        )

        hdf5 = _source_path(context, module, "hdf5")
        start, end = source_range
        row = {
            **dict(context.metadata),
            "asset_id": context.asset_id,
            "start_frame": start,
            "end_frame": end - 1,
            "hdf5_path": str(hdf5),
        }
        try:
            return load_deepreach_clip(row, episode_idx=0)
        except (DeepReachFrameContractError, OSError, ValueError) as exc:
            raise ModuleInputError(module, str(exc)) from exc

    if supplier == "jdt" or (
        not supplier and _source_entry(context, "parquet") is not None
    ):
        if source_range is None:
            raise ModulePrerequisiteError(module, "source_range for parquet precheck")
        from tools.run_manifest_precheck import load_jdt_clip

        start, end = source_range
        row = {
            **dict(context.metadata),
            "asset_id": context.asset_id,
            "start_frame": start,
            "end_frame": end - 1,
            "parquet_path": str(_source_path(context, module, "parquet")),
        }
        return load_jdt_clip(row, episode_idx=0)

    hdf5 = _source_path(context, module, "hdf5")
    from precheck.adapters import load_precheck_inputs

    clips = load_precheck_inputs(hdf5, episode_idx=0)
    if len(clips) != 1:
        raise ModulePrerequisiteError(module, "one precheck clip per asset")
    clip = clips[0]
    setattr(clip, "asset_id", context.asset_id)
    setattr(clip, "source_path", str(hdf5))
    if source_range is not None:
        clip = _slice_clip(clip, source_range)
    else:
        setattr(clip, "clip_start_frame", 0)
        setattr(clip, "clip_end_frame", max(clip.num_frames - 1, 0))
    return clip


def _normalize_dr_hard_presence_results(
    context: AssetContext,
    module: str,
    results: list[Any],
) -> list[Any]:
    """Make the DR 21-point existence contract explicit at the runner boundary."""
    supplier = str(
        context.metadata.get("supplier")
        or context.metadata.get("supplier_id")
        or ""
    ).lower()
    if supplier not in {"dr", "deepreach"} or module != "keypoint_presence":
        return results

    normalized: list[Any] = []
    for row in results:
        if getattr(row, "check", None) != "keypoint_missing":
            normalized.append(row)
            continue
        metrics = dict(getattr(row, "metrics", {}) or {})
        invalid_sides: list[str] = []
        for side in ("left", "right"):
            missing_value = metrics.get(f"missing_keypoint_count_{side}")
            valid_value = metrics.get(f"valid_keypoint_count_{side}")
            try:
                missing = float(missing_value) > 0.0
            except (TypeError, ValueError):
                missing = False
            try:
                incomplete = float(valid_value) < 21.0
            except (TypeError, ValueError):
                incomplete = False
            invalid = bool(
                metrics.get(f"keypoint_existence_invalid_{side}")
            ) or missing or incomplete
            metrics[f"keypoint_existence_invalid_{side}"] = invalid
            if invalid:
                invalid_sides.append(side)
                metrics.setdefault(
                    f"invalid_reasons_{side}",
                    ["canonical_validity_or_nonfinite"],
                )
        normalized.append(
            replace(
                row,
                metrics=metrics,
                flag=True,
                severity="fail",
            )
            if invalid_sides
            else replace(row, metrics=metrics)
        )
    return normalized


def _run_module_on_clip(
    context: AssetContext,
    config: LoadedQcConfig,
    module: str,
    clip: Any,
) -> PrecheckModuleExecution:
    if module not in MODULES:
        raise ValueError(f"unsupported precheck module: {module}")

    from precheck.runner import PrecheckRunner
    from qc_pipeline.adapters.precheck import precheck_config_from_unified
    from tools.run_manifest_precheck import (
        _map_candidate_window_to_source,
        _result_in_source_coordinates,
        _run_clip_in_local_coordinates,
    )

    output_dir = context.batch_root / ".qc_pipeline" / context.asset_id / module
    producer = PrecheckRunner(
        precheck_config_from_unified(
            config,
            module_names=(
                ["skeleton_quality_score"]
                if module == "keypoint_temporal"
                else [module]
            ),
            output_dir=output_dir,
        )
    )
    results = _run_clip_in_local_coordinates(producer, clip)
    source_start = int(getattr(clip, "clip_start_frame", 0))
    results = [
        _result_in_source_coordinates(
            row,
            clip_start_frame=source_start,
        )
        for row in results
    ]
    results = _normalize_dr_hard_presence_results(context, module, results)
    candidates: list[Mapping[str, Any]] = []
    if module == "keypoint_temporal":
        source_path = _source_relative_path(context)
        candidates = [
            _map_candidate_window_to_source(
                candidate,
                asset_id=context.asset_id,
                supplier=str(context.metadata.get("supplier") or "unknown"),
                source_path=str(getattr(clip, "source_path", source_path)),
                clip_start_frame=source_start,
                clip_end_frame=int(getattr(clip, "clip_end_frame", source_start)),
                clip_frame_count=clip.num_frames,
            )
            for candidate in producer.candidate_window_records
        ]
    result = _adapt_module(
        context,
        config,
        module,
        tuple(results),
        tuple(candidates),
        artifact_state="computed",
    )
    return PrecheckModuleExecution(result, tuple(results), tuple(candidates))


def _adapt_module(
    context: AssetContext,
    config: LoadedQcConfig,
    module: str,
    results: tuple[Any, ...],
    candidates: tuple[Mapping[str, Any], ...],
    *,
    artifact_state: str,
) -> ModuleResult:
    from qc_pipeline.adapters.precheck import (
        adapt_hdf5_text_info,
        adapt_keypoint_morphology,
        adapt_keypoint_presence,
        adapt_keypoint_temporal,
        adapt_quality_hand,
    )
    from qc_pipeline.artifacts import artifact_for

    source_path = _source_relative_path(context)
    common = {
        "asset_id": context.asset_id,
        "source_relative_path": source_path,
        "results": results,
        "config": config,
    }
    adapters = {
        "hdf5_text_info": adapt_hdf5_text_info,
        "quality_hand": adapt_quality_hand,
        "keypoint_presence": adapt_keypoint_presence,
        "keypoint_morphology": adapt_keypoint_morphology,
    }
    if module in adapters:
        result = adapters[module](**common)
    else:
        result = adapt_keypoint_temporal(candidate_windows=candidates, **common)

    artifact = artifact_for(context, "precheck")
    prefix = artifact.directory.relative_to(context.batch_root).as_posix()
    evidence = tuple(
        replace(item, path=f"{prefix}/{item.path}")
        if not item.path.startswith(f"{prefix}/")
        else item
        for item in result.evidence
    )
    runtime = {**dict(result.runtime), "artifact_state": artifact_state}
    return replace(result, evidence=evidence, runtime=runtime)


class PrecheckSession:
    """Lazy per-asset precheck session that shares one decoded source clip."""

    def __init__(self, context: AssetContext, config: LoadedQcConfig) -> None:
        self.context = context
        self.config = config
        self._clip: Any | None = None
        self._results: dict[str, ModuleResult] = {}
        self._raw_results: dict[str, tuple[Any, ...]] = {}
        self._candidate_windows: tuple[Mapping[str, Any], ...] = ()
        self._cache_checked = False
        self._started_at = perf_counter()
        self._frame_survival_state = self._initial_frame_survival_state()

    def _initial_frame_survival_state(self) -> FrameSurvivalState | None:
        profile = str(
            self.context.metadata.get("profile") or self.config.default_profile
        )
        policy = self.config.frame_survival_policy(profile)
        if (
            profile != "acceptance"
            or not bool(policy.get("enabled", False))
        ):
            return None
        source_range = self.context.source_range
        if source_range is None:
            canonical_episode = self.context.metadata.get("canonical_episode")
            if canonical_episode is not None:
                source_range = (0, canonical_episode.time_axis.frame_count)
        if source_range is None:
            return None
        start_frame, end_frame_exclusive = source_range
        return FrameSurvivalState.from_manifest_range(
            start_frame=start_frame,
            end_frame=end_frame_exclusive - 1,
            min_remaining_frame_ratio=float(policy["min_remaining_frame_ratio"]),
            stop_when_below=bool(policy.get("stop_when_below", True)),
        )

    def _prepare_clip_for_module(self, module: str) -> Any:
        if (
            self._frame_survival_state is None
            and module in _FRAME_SURVIVAL_MODULES
            and bool(self.config.frame_survival_policy(
                str(self.context.metadata.get("profile") or self.config.default_profile)
            ).get("enabled", False))
        ):
            raise ModulePrerequisiteError(
                module,
                "manifest source_range for acceptance frame survival",
            )
        if self._clip is None:
            self._clip = _load_clip(self.context, module)
        if isinstance(self._clip, ClipInputs):
            self._clip.manifest_metadata = manifest_metadata(self.context.metadata)
        source_indices = getattr(self._clip, "source_frame_indices", None)
        if source_indices is None:
            frame_indices = getattr(self._clip, "frame_indices", None)
            if frame_indices is None:
                num_frames = getattr(self._clip, "num_frames", None)
                if isinstance(num_frames, int):
                    start = int(getattr(self._clip, "clip_start_frame", 0))
                    frame_indices = list(range(start, start + num_frames))
            if frame_indices is not None:
                setattr(self._clip, "source_frame_indices", tuple(frame_indices))
        if self._frame_survival_state is not None:
            setattr(
                self._clip,
                "eligible_frame_ranges",
                self._frame_survival_state.eligible_ranges,
            )
        return self._clip

    def _apply_frame_survival(
        self,
        module: str,
        result: ModuleResult,
    ) -> ModuleResult:
        if self._frame_survival_state is None or module not in _FRAME_SURVIVAL_MODULES:
            return result
        self._frame_survival_state, update = self._frame_survival_state.apply(
            module=module,
            exclusions=result.frame_exclusions,
        )
        return replace(
            result,
            evaluation={
                **dict(result.evaluation),
                "frame_survival": update.to_dict(),
            },
            runtime={
                **dict(result.runtime),
                "acceptance_frame_survival_handled": True,
            },
        )

    def _fingerprint(self) -> dict[str, Any]:
        return precheck_fingerprint(self.context, self.config)

    def _try_load_cache(self) -> None:
        if self._cache_checked:
            return
        self._cache_checked = True
        if not bool(self.context.metadata.get("reuse_artifacts", True)):
            return
        for source_name in ("hdf5", "parquet"):
            source = _source_entry(self.context, source_name)
            if source is None or source.get("path") is None:
                continue
            if not (self.context.batch_root / str(source["path"])).is_file():
                return
        from qc_common.types import CheckResult
        from qc_pipeline.artifacts import artifact_for, reusable_artifact

        artifact = artifact_for(self.context, "precheck")
        fingerprint = self._fingerprint()
        if not reusable_artifact(artifact, fingerprint):
            return
        run_config = json.loads(artifact.run_config_path.read_text(encoding="utf-8"))
        if run_config.get("manifest_text_metadata") != normalized_manifest_text_metadata(
            self.context.metadata
        ):
            return
        if tuple(run_config.get("completed_modules", ())) != MODULES:
            return
        rows = json.loads(
            (artifact.directory / "check_results.json").read_text(encoding="utf-8")
        )
        grouped: dict[str, list[Any]] = {module: [] for module in MODULES}
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("precheck cached check result must be an object")
            module = str(row.get("pipeline_module") or "")
            if module not in grouped:
                raise ValueError(f"unknown cached precheck module: {module}")
            grouped[module].append(
                CheckResult(
                    check=str(row["check"]),
                    episode_idx=int(row["episode_idx"]),
                    frame_idx=int(row["frame_idx"]),
                    metrics=dict(row.get("metrics") or {}),
                    flag=row.get("flag"),
                    reason=str(row.get("reason") or ""),
                    severity=(
                        str(row["severity"])
                        if row.get("severity") is not None
                        else None
                    ),
                )
            )
        self._raw_results = {
            module: tuple(values) for module, values in grouped.items()
        }
        candidate_payload = json.loads(
            (artifact.directory / "candidate_windows.json").read_text(encoding="utf-8")
        )
        if not isinstance(candidate_payload, list):
            raise ValueError("precheck candidate_windows.json must be an array")
        self._candidate_windows = tuple(
            dict(value) for value in candidate_payload if isinstance(value, Mapping)
        )

    def _publish_if_ready(self, module: str, result: ModuleResult) -> None:
        completed = set(self._raw_results) == set(MODULES)
        if not completed and module != "keypoint_temporal" and not result.evidence:
            return
        from qc_common.io import aggregate_results, write_json_records
        from qc_pipeline.artifacts import (
            artifact_for,
            promote_artifact,
            staged_artifact,
            write_run_config,
        )

        artifact = artifact_for(self.context, "precheck")
        records: list[dict[str, Any]] = []
        raw: list[Any] = []
        for module_name in MODULES:
            for row in self._raw_results.get(module_name, ()):
                raw.append(row)
                records.append({**row.to_record(), "pipeline_module": module_name})
        with staged_artifact(artifact) as staging:
            write_json_records(records, staging / "check_results.json")
            write_json_records(aggregate_results(raw), staging / "clip_aggregates.json")
            write_json_records(
                [dict(value) for value in self._candidate_windows],
                staging / "candidate_windows.json",
            )
            temporal_result = (
                result
                if module == "keypoint_temporal"
                else self._results.get("keypoint_temporal")
            )
            temporal_output: dict[str, Any] | None = None
            if temporal_result is not None:
                evaluation = temporal_result.evaluation
                status = str(evaluation.get("output_status") or "")
                if status in {"valid", "no_valid_output"}:
                    temporal_output = {
                        "schema_version": _TEMPORAL_OUTPUT_SCHEMA_VERSION,
                        "status": status,
                        "valid_frame_count": int(
                            evaluation.get("valid_frame_count", 0)
                        ),
                        "uncalibrated_frame_count": int(
                            evaluation.get("uncalibrated_frame_count", 0)
                        ),
                        "reason": str(
                            evaluation.get("reason")
                            or (
                                "calibrated_temporal_output"
                                if status == "valid"
                                else "no_valid_temporal_output"
                            )
                        ),
                    }
            metadata: dict[str, Any] = {
                "completed_modules": [
                    name for name in MODULES if name in self._raw_results
                ],
                "frame_survival": self._frame_survival_metadata(),
                "manifest_metadata": manifest_metadata(self.context.metadata),
                "manifest_text_metadata": normalized_manifest_text_metadata(
                    self.context.metadata
                ),
            }
            if temporal_output is not None:
                metadata["temporal_output"] = temporal_output
            write_run_config(
                staging,
                producer="precheck",
                outcome="completed" if completed else "partial",
                fingerprint=self._fingerprint(),
                elapsed_seconds=perf_counter() - self._started_at,
                metadata=metadata,
            )
            promote_artifact(staging, artifact)

    def _frame_survival_metadata(self) -> dict[str, Any] | None:
        state = self._frame_survival_state
        if state is None:
            return None
        return {
            "original_frame_count": state.original_frame_count,
            "raw_exclusions": [item.to_dict() for item in state.raw_exclusions],
            "cumulative_excluded_frame_ranges": [
                list(item) for item in state.cumulative_excluded_frame_ranges
            ],
            "eligible_frame_ranges": [list(item) for item in state.eligible_ranges],
            "remaining_frame_count": state.remaining_frame_count,
            "remaining_frame_ratio": state.remaining_frame_ratio,
            "stop_threshold": state.min_remaining_frame_ratio,
        }

    def run_module(self, module: str) -> ModuleResult:
        if module not in MODULES:
            raise ValueError(f"unsupported precheck module: {module}")
        cached = self._results.get(module)
        if cached is not None:
            return cached
        self._try_load_cache()
        if module in self._raw_results:
            candidates = self._candidate_windows if module == "keypoint_temporal" else ()
            result = _adapt_module(
                self.context,
                self.config,
                module,
                self._raw_results[module],
                candidates,
                artifact_state="reused",
            )
            result = self._apply_frame_survival(module, result)
            self._results[module] = result
            return result
        execution = _run_module_on_clip(
            self.context,
            self.config,
            module,
            self._prepare_clip_for_module(module),
        )
        if isinstance(execution, ModuleResult):
            result = self._apply_frame_survival(module, execution)
        else:
            result = self._apply_frame_survival(module, execution.result)
            self._raw_results[module] = execution.check_results
            if module == "keypoint_temporal":
                self._candidate_windows = execution.candidate_windows
            self._publish_if_ready(module, result)
        self._results[module] = result
        return result

    def runner_for(self, module: str) -> ModuleRunner:
        if module not in MODULES:
            raise ValueError(f"unsupported precheck module: {module}")

        def run(
            context: AssetContext,
            config: LoadedQcConfig,
        ) -> ModuleResult:
            if context is not self.context:
                raise ValueError("precheck session cannot be shared across assets")
            if config.sha256 != self.config.sha256:
                raise ValueError("precheck session config drift")
            return self.run_module(module)

        return run


def runner_for(module: str) -> ModuleRunner:
    """Compatibility wrapper; shared registries should create one session."""
    if module not in MODULES:
        raise ValueError(f"unsupported precheck module: {module}")
    session: PrecheckSession | None = None

    def run(context: AssetContext, config: LoadedQcConfig) -> ModuleResult:
        nonlocal session
        if session is None:
            session = PrecheckSession(context, config)
        return session.runner_for(module)(context, config)

    return run


__all__ = [
    "MODULES",
    "PrecheckModuleExecution",
    "PrecheckSession",
    "precheck_fingerprint",
    "runner_for",
]
