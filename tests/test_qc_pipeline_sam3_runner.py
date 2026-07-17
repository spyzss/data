from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
from threading import Lock
import time

import pytest

from qc_common.config import load_qc_acceptance_config
from qc_common.module_registry import (
    ModuleBlockedError,
    ModuleInputError,
    ModulePrerequisiteError,
)
from qc_pipeline.context import AssetContext
from qc_pipeline.runners.sam3_containment import _source_path, runner


def _write_candidates(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = (
        tmp_path
        / "module_outputs"
        / "asset-a"
        / "precheck"
        / "candidate_windows.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def _context(
    tmp_path: Path,
    *,
    supplier: str = "jdt",
    legacy_candidates: Path | None = None,
) -> AssetContext:
    video = tmp_path / "source" / "video.mp4"
    parquet = tmp_path / "source" / "episode.parquet"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"video")
    parquet.write_bytes(b"parquet")
    source_files: dict[str, object] = {
        "video": {"path": "source/video.mp4"},
        "parquet": {"path": "source/episode.parquet"},
    }
    if legacy_candidates is not None:
        source_files["candidate_windows"] = {
            "path": legacy_candidates.relative_to(tmp_path).as_posix()
        }
    manifest_row = {
        "asset_id": "asset-a",
        "episode_index": 0,
        "start_frame": 0,
        "end_frame": 19,
        "primary_video_path": "source/video.mp4",
        "parquet_path": "source/episode.parquet",
        "left_hand_2d_field": "leftcam_left_kp2d",
        "right_hand_2d_field": "leftcam_right_kp2d",
    }
    return AssetContext(
        asset_id="asset-a",
        batch_root=tmp_path,
        report_path=tmp_path / "quality_archive" / "asset-a.json",
        source_files=source_files,
        source_range=(0, 20),
        metadata={"supplier": supplier, "manifest_row": manifest_row},
    )


def _candidate(start: int = 10, end: int = 12) -> dict[str, object]:
    return {
        "asset_id": "asset-a",
        "coordinate_space": "source",
        "frame_coordinate_system": "source_inclusive",
        "start_frame": start,
        "end_frame": end,
        "hand_side": "left",
        "sam3_eligible": True,
    }


def _model_context(tmp_path: Path) -> tuple[AssetContext, Path]:
    base = _context(tmp_path)
    model = tmp_path / "source" / "sam3-model"
    model.mkdir()
    (model / "config.json").write_text('{"model":"sam3"}\n', encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"safetensors-metadata")
    (model / "sam3.pt").write_bytes(b"checkpoint-metadata")
    return (
        replace(
            base,
            source_files={
                **dict(base.source_files),
                "sam3_model": {"path": "source/sam3-model"},
            },
        ),
        model,
    )


def _write_successful_sam3_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "window_keypoint_containment_summary.json").write_text(
        json.dumps(
            [
                {
                    "asset_id": "asset-a",
                    "window_start_frame": 10,
                    "window_end_frame": 12,
                    "hand_side": "left",
                    "window_containment_verdict": "pass",
                }
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "review_evidence_manifest.csv").write_text(
        "asset_id,evidence_type,source_path,window_start_frame,window_end_frame,hand_side\n",
        encoding="utf-8",
    )


def _write_current_run_config(
    context: AssetContext,
    *,
    temporal_status: str = "valid",
    valid_frame_count: int = 19,
) -> None:
    from qc_pipeline.artifacts import write_run_config
    from qc_pipeline.runners.precheck import MODULES, precheck_fingerprint

    directory = (
        context.batch_root
        / "module_outputs"
        / context.asset_id
        / "precheck"
    )
    write_run_config(
        directory,
        producer="precheck",
        outcome="completed",
        fingerprint=precheck_fingerprint(context, load_qc_acceptance_config()),
        elapsed_seconds=0.0,
        metadata={
            "completed_modules": list(MODULES),
            "temporal_output": {
                "status": temporal_status,
                "valid_frame_count": valid_frame_count,
                "uncalibrated_frame_count": 0 if valid_frame_count else 20,
                "reason": (
                    "calibrated_temporal_output"
                    if temporal_status == "valid"
                    else "no_valid_temporal_output"
                ),
            },
        },
    )


def test_runtime_provider_reuses_one_segmenter_for_three_sequential_assets(
    tmp_path: Path,
) -> None:
    from qc_pipeline.sam3_runtime import Sam3RuntimeProvider

    model = tmp_path / "sam3-model"
    model.mkdir()
    created: list[tuple[Path, dict[str, object]]] = []

    def factory(path: Path, config: dict[str, object]) -> object:
        created.append((path, config))
        return object()

    provider = Sam3RuntimeProvider(factory=factory)
    config = {"device": "cuda", "dtype": "bfloat16", "mask_threshold": 0.5}

    segmenters = [provider.get_segmenter(model, config) for _ in range(3)]

    assert len(created) == 1
    assert created[0] == (model.resolve(), config)
    assert segmenters[0] is segmenters[1] is segmenters[2]


def test_runtime_provider_initializes_once_for_concurrent_assets(
    tmp_path: Path,
) -> None:
    from qc_pipeline.sam3_runtime import Sam3RuntimeProvider

    model = tmp_path / "sam3-model"
    model.mkdir()
    creation_count = 0
    count_lock = Lock()

    def factory(path: Path, config: dict[str, object]) -> object:
        nonlocal creation_count
        with count_lock:
            creation_count += 1
        time.sleep(0.03)
        return object()

    provider = Sam3RuntimeProvider(factory=factory)
    config = {"device": "cuda", "dtype": "bfloat16"}
    with ThreadPoolExecutor(max_workers=3) as pool:
        segmenters = list(
            pool.map(lambda _: provider.get_segmenter(model, config), range(3))
        )

    assert creation_count == 1
    assert segmenters[0] is segmenters[1] is segmenters[2]


def test_runtime_provider_serializes_only_shared_segmenter_inference(
    tmp_path: Path,
) -> None:
    from qc_pipeline.sam3_runtime import Sam3RuntimeProvider

    model = tmp_path / "sam3-model"
    model.mkdir()
    active = 0
    max_active = 0
    state_lock = Lock()

    class Segmenter:
        def segment_frame(self, frame: object) -> object:
            nonlocal active, max_active
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.03)
            with state_lock:
                active -= 1
            return frame

    provider = Sam3RuntimeProvider(factory=lambda path, config: Segmenter())
    segmenter = provider.get_segmenter(model, {})
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(segmenter.segment_frame, range(3)))

    assert results == [0, 1, 2]
    assert max_active == 1


def test_runtime_provider_uses_resolved_model_path_in_cache_key(
    tmp_path: Path,
) -> None:
    from qc_pipeline.sam3_runtime import Sam3RuntimeProvider

    model = tmp_path / "sam3-model"
    model.mkdir()
    link = tmp_path / "sam3-model-link"
    link.symlink_to(model, target_is_directory=True)
    calls = 0

    def factory(path: Path, config: dict[str, object]) -> object:
        nonlocal calls
        calls += 1
        return object()

    provider = Sam3RuntimeProvider(factory=factory)

    assert provider.get_segmenter(model, {}) is provider.get_segmenter(link, {})
    assert calls == 1


def test_runtime_provider_separates_different_model_paths(tmp_path: Path) -> None:
    from qc_pipeline.sam3_runtime import Sam3RuntimeProvider

    first_model = tmp_path / "sam3-model-a"
    second_model = tmp_path / "sam3-model-b"
    first_model.mkdir()
    second_model.mkdir()
    calls = 0

    def factory(path: Path, config: dict[str, object]) -> object:
        nonlocal calls
        calls += 1
        return object()

    provider = Sam3RuntimeProvider(factory=factory)

    assert provider.get_segmenter(first_model, {}) is not provider.get_segmenter(
        second_model, {}
    )
    assert calls == 2


def test_runtime_provider_canonicalizes_config_field_order(tmp_path: Path) -> None:
    from qc_pipeline.sam3_runtime import Sam3RuntimeProvider

    model = tmp_path / "sam3-model"
    model.mkdir()
    calls = 0

    def factory(path: Path, config: dict[str, object]) -> object:
        nonlocal calls
        calls += 1
        return object()

    provider = Sam3RuntimeProvider(factory=factory)
    first = provider.get_segmenter(
        model,
        {"device": "cuda", "dtype": "bfloat16", "thresholds": {"b": 2, "a": 1}},
    )
    second = provider.get_segmenter(
        model,
        {"thresholds": {"a": 1, "b": 2}, "dtype": "bfloat16", "device": "cuda"},
    )

    assert first is second
    assert calls == 1


@pytest.mark.parametrize(
    ("changed_config", "field"),
    [
        ({"device": "cpu", "dtype": "bfloat16", "mask_threshold": 0.5}, "device"),
        ({"device": "cuda", "dtype": "float16", "mask_threshold": 0.5}, "dtype"),
        ({"device": "cuda", "dtype": "bfloat16", "mask_threshold": 0.7}, "config"),
    ],
)
def test_runtime_provider_separates_device_dtype_and_runtime_config(
    tmp_path: Path,
    changed_config: dict[str, object],
    field: str,
) -> None:
    from qc_pipeline.sam3_runtime import Sam3RuntimeProvider

    model = tmp_path / "sam3-model"
    model.mkdir()
    calls = 0

    def factory(path: Path, config: dict[str, object]) -> object:
        nonlocal calls
        calls += 1
        return object()

    provider = Sam3RuntimeProvider(factory=factory)
    baseline = provider.get_segmenter(
        model,
        {"device": "cuda", "dtype": "bfloat16", "mask_threshold": 0.5},
    )
    changed = provider.get_segmenter(model, changed_config)

    assert baseline is not changed, field
    assert calls == 2


def test_runtime_provider_retries_after_initialization_failure(tmp_path: Path) -> None:
    from qc_pipeline.sam3_runtime import Sam3RuntimeProvider

    model = tmp_path / "sam3-model"
    model.mkdir()
    attempts = 0
    expected = object()

    def factory(path: Path, config: dict[str, object]) -> object:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("model load failed")
        return expected

    provider = Sam3RuntimeProvider(factory=factory)

    with pytest.raises(RuntimeError, match="model load failed"):
        provider.get_segmenter(model, {})

    assert provider.get_segmenter(model, {}) is not None
    assert attempts == 2


def test_source_path_accepts_model_directory(tmp_path: Path) -> None:
    context, model = _model_context(tmp_path)

    assert _source_path(
        context,
        "sam3_model",
        expected_type="directory",
    ) == model


def test_source_path_accepts_symlink_to_model_directory(tmp_path: Path) -> None:
    context, model = _model_context(tmp_path)
    link = tmp_path / "source" / "sam3-model-link"
    link.symlink_to(model, target_is_directory=True)
    linked_context = replace(
        context,
        source_files={
            **dict(context.source_files),
            "sam3_model": {"path": "source/sam3-model-link"},
        },
    )

    assert _source_path(
        linked_context,
        "sam3_model",
        expected_type="directory",
    ) == link


def test_source_path_rejects_file_as_model_directory(tmp_path: Path) -> None:
    context = _context(tmp_path)
    model_file = tmp_path / "source" / "sam3-model"
    model_file.write_bytes(b"not-a-directory")
    invalid = replace(
        context,
        source_files={
            **dict(context.source_files),
            "sam3_model": {"path": "source/sam3-model"},
        },
    )

    with pytest.raises(
        ModulePrerequisiteError,
        match=r"existing directory source_files\.sam3_model\.path",
    ):
        _source_path(invalid, "sam3_model", expected_type="directory")


def test_source_path_rejects_missing_model_directory(tmp_path: Path) -> None:
    context = _context(tmp_path)
    missing = replace(
        context,
        source_files={
            **dict(context.source_files),
            "sam3_model": {"path": "source/missing-model"},
        },
    )

    with pytest.raises(
        ModulePrerequisiteError,
        match=r"existing directory source_files\.sam3_model\.path",
    ):
        _source_path(missing, "sam3_model", expected_type="directory")


@pytest.mark.parametrize("source_name", ["video", "parquet", "manifest"])
def test_source_path_keeps_file_contract_for_jdt_inputs(
    tmp_path: Path,
    source_name: str,
) -> None:
    context = _context(tmp_path)
    source_path = tmp_path / "source" / source_name
    source_path.mkdir()
    invalid = replace(
        context,
        source_files={
            **dict(context.source_files),
            source_name: {"path": f"source/{source_name}"},
        },
    )

    with pytest.raises(
        ModulePrerequisiteError,
        match=rf"existing file source_files\.{source_name}\.path",
    ):
        _source_path(invalid, source_name)


@pytest.mark.parametrize("source_name", ["video", "parquet"])
def test_jdt_runner_rejects_directory_for_file_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_name: str,
) -> None:
    base = _context(tmp_path)
    source_dir = tmp_path / "source" / f"{source_name}-directory"
    source_dir.mkdir()
    context = replace(
        base,
        source_files={
            **dict(base.source_files),
            source_name: {"path": f"source/{source_name}-directory"},
        },
    )
    _write_candidates(tmp_path, [_candidate()])
    _write_current_run_config(context)
    monkeypatch.setattr(
        "tools.run_manifest_sam3_containment.run_manifest_sam3_containment",
        lambda **kwargs: pytest.fail("invalid file source must fail preflight"),
    )

    with pytest.raises(
        ModulePrerequisiteError,
        match=rf"existing file source_files\.{source_name}\.path",
    ):
        runner(lambda: object())(context, load_qc_acceptance_config())


def test_jdt_model_directory_reaches_manifest_sam3_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qc_pipeline.artifacts import file_sha256

    context, model = _model_context(tmp_path)
    _write_candidates(tmp_path, [_candidate()])
    _write_current_run_config(context)
    captured: dict[str, object] = {}

    def fake_run(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        _write_successful_sam3_outputs(Path(str(kwargs["output_dir"])))
        return {"failed_asset_count": 0}

    monkeypatch.setattr(
        "tools.run_manifest_sam3_containment.run_manifest_sam3_containment",
        fake_run,
    )

    result = runner(None)(context, load_qc_acceptance_config())

    assert result.verdict == "pass"
    assert captured["sam3_model"] == model
    assert captured["segmenter"] is None
    run_config = json.loads(
        (
            tmp_path
            / "module_outputs"
            / "asset-a"
            / "sam3_containment"
            / "run_config.json"
        ).read_text(encoding="utf-8")
    )
    assert run_config["fingerprint"]["implementation_version"] == (
        "sam3-containment-producer-v2"
    )
    model_identity = run_config["fingerprint"]["sources"]["sam3_model"]
    assert model_identity["kind"] == "directory"
    assert model_identity["path"] == "source/sam3-model"
    assert model_identity["resolved_path"] == "source/sam3-model"
    assert model_identity["files"]["config.json"]["sha256"] == file_sha256(
        model / "config.json"
    )
    assert "sha256" not in model_identity["files"]["model.safetensors"]
    assert "sha256" not in model_identity["files"]["sam3.pt"]


def test_jdt_runner_gets_segmenter_from_injected_runtime_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.run_manifest_sam3_containment import SAM3_CONFIG

    context, model = _model_context(tmp_path)
    _write_candidates(tmp_path, [_candidate()])
    _write_current_run_config(context)
    segmenter = object()
    provider_calls: list[tuple[Path, dict[str, object]]] = []
    captured: dict[str, object] = {}

    def provider(path: Path, config: dict[str, object]) -> object:
        provider_calls.append((path, config))
        return segmenter

    def fake_run(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        _write_successful_sam3_outputs(Path(str(kwargs["output_dir"])))
        return {"failed_asset_count": 0}

    monkeypatch.setattr(
        "tools.run_manifest_sam3_containment.run_manifest_sam3_containment",
        fake_run,
    )

    result = runner(None, segmenter_provider=provider)(
        context,
        load_qc_acceptance_config(),
    )

    assert result.verdict == "pass"
    assert provider_calls == [(model, dict(SAM3_CONFIG))]
    assert captured["segmenter"] is segmenter


def test_full_pipeline_uses_current_precheck_artifact_not_legacy_manifest_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = _candidate(10, 12)
    _write_candidates(tmp_path, [current])
    legacy = tmp_path / "legacy_candidates.json"
    legacy.write_text(json.dumps([_candidate(1, 2)]), encoding="utf-8")
    captured: list[dict[str, object]] = []

    def fake_run(**kwargs: object) -> dict[str, object]:
        candidate_path = Path(str(kwargs["candidate_windows"]))
        captured.extend(
            json.loads(line)
            for line in candidate_path.read_text().splitlines()
            if line.strip()
        )
        output_dir = Path(str(kwargs["output_dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "window_keypoint_containment_summary.json").write_text(
            json.dumps(
                [
                    {
                        "asset_id": "asset-a",
                        "window_start_frame": 10,
                        "window_end_frame": 12,
                        "hand_side": "left",
                        "window_containment_verdict": "pass",
                    }
                ]
            ),
            encoding="utf-8",
        )
        (output_dir / "review_evidence_manifest.csv").write_text(
            "asset_id,evidence_type,source_path,window_start_frame,window_end_frame,hand_side\n",
            encoding="utf-8",
        )
        return {"failed_asset_count": 0}

    monkeypatch.setattr(
        "tools.run_manifest_sam3_containment.run_manifest_sam3_containment",
        fake_run,
    )
    context = _context(tmp_path, legacy_candidates=legacy)
    _write_current_run_config(context)
    result = runner(lambda: object())(context, load_qc_acceptance_config())

    assert captured == [current]
    assert result.verdict == "pass"


def test_empty_current_candidates_skip_before_model_or_manifest_lookup(
    tmp_path: Path,
) -> None:
    _write_candidates(tmp_path, [])
    calls = 0

    def forbidden_provider(model: Path, config: dict[str, object]) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("segmenter must not load")

    context = AssetContext(
        asset_id="asset-a",
        batch_root=tmp_path,
        report_path=tmp_path / "quality_archive" / "asset-a.json",
        source_files={},
        source_range=(0, 20),
        metadata={"supplier": "jdt"},
    )
    _write_current_run_config(context)
    result = runner(None, segmenter_provider=forbidden_provider)(
        context,
        load_qc_acceptance_config(),
    )

    assert calls == 0
    assert result.verdict == "skipped"
    assert result.evaluation == {"decision": "skipped", "reason": "no_candidates"}
    assert result.runtime["artifact_state"] == "no_candidates"


def test_only_explicitly_eligible_candidates_reach_sam3(tmp_path: Path) -> None:
    _write_candidates(
        tmp_path,
        [
            {**_candidate(), "sam3_eligible": False},
            {key: value for key, value in _candidate().items() if key != "sam3_eligible"},
        ],
    )
    context = _context(tmp_path)
    _write_current_run_config(context)

    result = runner(
        None,
        segmenter_provider=lambda model, config: pytest.fail(
            "ineligible candidates must not load model"
        ),
    )(context, load_qc_acceptance_config())

    assert result.verdict == "skipped"
    assert result.evaluation == {"decision": "skipped", "reason": "no_candidates"}


def test_uncalibrated_temporal_blocks_sam3_instead_of_no_candidates(
    tmp_path: Path,
) -> None:
    _write_candidates(tmp_path, [])
    context = _context(tmp_path)
    _write_current_run_config(
        context,
        temporal_status="no_valid_output",
        valid_frame_count=0,
    )

    with pytest.raises(
        ModuleBlockedError,
        match="no_valid_temporal_output",
    ) as raised:
        runner(
            None,
            segmenter_provider=lambda model, config: pytest.fail(
                "blocked temporal output must not load model"
            ),
        )(context, load_qc_acceptance_config())

    assert raised.value.reason == "no_valid_temporal_output"


@pytest.mark.parametrize(
    "candidate",
    [
        {**_candidate(), "asset_id": "other"},
        _candidate(12, 10),
        _candidate(10, 20),
        {**_candidate(), "coordinate_space": "local"},
    ],
)
def test_invalid_current_candidate_is_input_invalid_before_model_load(
    tmp_path: Path,
    candidate: dict[str, object],
) -> None:
    _write_candidates(tmp_path, [candidate])
    context = _context(tmp_path)
    _write_current_run_config(context)

    with pytest.raises(ModuleInputError, match="candidate") as raised:
        runner(lambda: pytest.fail("invalid input must not load model"))(
            context,
            load_qc_acceptance_config(),
        )

    assert raised.value.module == "sam3_containment"


def test_dr_is_blocked_by_unverified_calibration_before_candidate_lookup(
    tmp_path: Path,
) -> None:
    _write_candidates(tmp_path, [_candidate()])
    context = _context(tmp_path, supplier="dr")

    with pytest.raises(ModuleBlockedError, match="calibration_unverified") as raised:
        runner(
            None,
            segmenter_provider=lambda model, config: pytest.fail(
                "unsupported adapter must not load model"
            ),
        )(context, load_qc_acceptance_config())

    assert raised.value.reason == "calibration_unverified"


def test_dr_transform_ambiguity_is_preserved_as_block_reason(tmp_path: Path) -> None:
    _write_candidates(tmp_path, [_candidate()])
    base = _context(tmp_path, supplier="deepreach")
    context = AssetContext(
        base.asset_id,
        base.batch_root,
        base.report_path,
        base.source_files,
        source_range=base.source_range,
        metadata={**dict(base.metadata), "projection_validation_status": "transform_ambiguous"},
    )

    with pytest.raises(ModuleBlockedError, match="transform_ambiguous") as raised:
        runner(
            None,
            segmenter_provider=lambda model, config: pytest.fail(
                "blocked adapter must not load model"
            ),
        )(context, load_qc_acceptance_config())

    assert raised.value.reason == "transform_ambiguous"


@pytest.mark.parametrize("candidate_rows", [[], [_candidate()]])
def test_validated_dr_is_blocked_by_adapter_missing_before_candidate_count(
    tmp_path: Path,
    candidate_rows: list[dict[str, object]],
) -> None:
    base = _context(tmp_path, supplier="dr")
    context = AssetContext(
        base.asset_id,
        base.batch_root,
        base.report_path,
        base.source_files,
        source_range=base.source_range,
        metadata={**dict(base.metadata), "projection_validation_status": "validated"},
    )
    _write_candidates(tmp_path, candidate_rows)
    _write_current_run_config(context)

    with pytest.raises(ModuleBlockedError, match="adapter_missing") as raised:
        runner(
            None,
            segmenter_provider=lambda model, config: pytest.fail(
                "missing DR adapter must not load model"
            ),
        )(context, load_qc_acceptance_config())

    assert raised.value.reason == "adapter_missing"


def test_potentia_is_blocked_by_no_keypoint_input_before_candidate_lookup(
    tmp_path: Path,
) -> None:
    _write_candidates(tmp_path, [_candidate()])
    context = _context(tmp_path, supplier="potentia")

    with pytest.raises(ModuleBlockedError, match="no_keypoint_input") as raised:
        runner(
            None,
            segmenter_provider=lambda model, config: pytest.fail(
                "blocked adapter must not load model"
            ),
        )(context, load_qc_acceptance_config())

    assert raised.value.reason == "no_keypoint_input"


def test_stale_precheck_run_config_is_rejected_before_model_load(
    tmp_path: Path,
) -> None:
    _write_candidates(tmp_path, [_candidate()])
    context = _context(tmp_path)
    _write_current_run_config(context)
    (tmp_path / "source" / "episode.parquet").write_bytes(b"changed-parquet")

    with pytest.raises(ModuleInputError, match="current precheck run"):
        runner(lambda: pytest.fail("stale candidates must not load model"))(
            context,
            load_qc_acceptance_config(),
        )
