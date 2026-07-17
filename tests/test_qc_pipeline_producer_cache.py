from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from qc_common.config import load_qc_acceptance_config
from qc_common.contracts import ModuleResult
from qc_pipeline.context import AssetContext


def _video_context(tmp_path: Path) -> AssetContext:
    video = tmp_path / "source" / "video.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"video-v1")
    return AssetContext(
        asset_id="asset-a",
        batch_root=tmp_path,
        report_path=tmp_path / "quality_archive" / "asset-a.json",
        source_files={"video": {"path": "source/video.mp4"}},
        source_range=(0, 20),
        metadata={"supplier": "jdt"},
    )


def test_video_quality_matching_artifact_skips_analyzer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qc_pipeline.runners import video_quality

    calls = 0

    def compute(context: AssetContext, config: object) -> tuple[ModuleResult, object]:
        nonlocal calls
        calls += 1
        return (
            ModuleResult("video_quality", "pass", {"decision": "pass"}, {}),
            {"detector": "raw-v1"},
        )

    monkeypatch.setattr(video_quality, "_compute_video_quality", compute)
    context = _video_context(tmp_path)
    config = load_qc_acceptance_config()

    first = video_quality.run(context, config)
    second = video_quality.run(context, config)

    assert calls == 1
    assert first.runtime["artifact_state"] == "computed"
    assert second.runtime["artifact_state"] == "reused"
    payload = json.loads(
        (
            tmp_path
            / "module_outputs"
            / "asset-a"
            / "video_quality"
            / "video_quality_result.json"
        ).read_text()
    )
    assert payload["raw_result"] == {"detector": "raw-v1"}


def test_failed_video_recompute_preserves_previous_valid_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qc_pipeline.runners import video_quality

    monkeypatch.setattr(
        video_quality,
        "_compute_video_quality",
        lambda context, config: (
            ModuleResult("video_quality", "pass", {"decision": "pass"}, {}),
            {"detector": "raw-v1"},
        ),
    )
    context = _video_context(tmp_path)
    config = load_qc_acceptance_config()
    video_quality.run(context, config)
    artifact = (
        tmp_path / "module_outputs" / "asset-a" / "video_quality"
    )
    before = {
        path.name: path.read_bytes()
        for path in artifact.iterdir()
        if path.is_file()
    }
    (tmp_path / "source" / "video.mp4").write_bytes(b"video-v2-invalidates")

    def fail(context: AssetContext, config: object) -> tuple[ModuleResult, object]:
        raise RuntimeError("decoder crashed")

    monkeypatch.setattr(video_quality, "_compute_video_quality", fail)
    with pytest.raises(RuntimeError, match="decoder crashed"):
        video_quality.run(context, config)

    assert {
        path.name: path.read_bytes()
        for path in artifact.iterdir()
        if path.is_file()
    } == before


@pytest.mark.parametrize(
    "legacy_version",
    [
        "precheck-session-v5-frame-survival-metadata",
        "precheck-session-v6-calibrated-temporal-output",
    ],
)
def test_precheck_legacy_artifact_is_not_reused_by_current_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy_version: str,
) -> None:
    from qc_common.types import CheckResult
    from qc_pipeline.runners import precheck
    from tests.test_qc_pipeline_precheck_session import _config, _context

    context = _context(tmp_path)
    config = _config(tmp_path)
    loads: list[str] = []
    executions: list[str] = []

    monkeypatch.setattr(
        precheck,
        "_load_clip",
        lambda context, module: loads.append(module) or object(),
    )

    def execute(
        context: AssetContext,
        config: object,
        module: str,
        clip: object,
    ) -> precheck.PrecheckModuleExecution:
        executions.append(module)
        return precheck.PrecheckModuleExecution(
            result=ModuleResult(module, "pass", {}, {}),
            check_results=(CheckResult(module, 0, -1, {}, False, "ok"),),
            candidate_windows=(),
        )

    monkeypatch.setattr(precheck, "_run_module_on_clip", execute)
    with monkeypatch.context() as legacy:
        legacy.setattr(
            precheck,
            "_IMPLEMENTATION_VERSION",
            legacy_version,
        )
        old_session = precheck.PrecheckSession(context, config)
        for module in precheck.MODULES:
            old_session.run_module(module)

    old_run_config = json.loads(
        (
            tmp_path
            / "module_outputs"
            / "asset-a"
            / "precheck"
            / "run_config.json"
        ).read_text(encoding="utf-8")
    )
    assert old_run_config["fingerprint"]["implementation_version"] == legacy_version

    loads.clear()
    executions.clear()
    reused_modules: list[str] = []

    def adapt_cached(
        context: AssetContext,
        config: object,
        module: str,
        results: object,
        candidates: object,
        *,
        artifact_state: str,
    ) -> ModuleResult:
        reused_modules.append(module)
        return ModuleResult(
            module,
            "pass",
            {},
            {},
            runtime={"artifact_state": artifact_state},
        )

    monkeypatch.setattr(precheck, "_adapt_module", adapt_cached)
    current_session = precheck.PrecheckSession(context, config)
    current_session.run_module("hdf5_text_info")

    assert loads == ["hdf5_text_info"]
    assert executions == ["hdf5_text_info"]
    assert reused_modules == []


def test_precheck_temporal_output_schema_change_invalidates_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qc_common.types import CheckResult
    from qc_pipeline.runners import precheck
    from tests.test_qc_pipeline_precheck_session import _config, _context

    context = _context(tmp_path)
    config = _config(tmp_path)
    executions: list[str] = []
    monkeypatch.setattr(precheck, "_load_clip", lambda context, module: object())

    def execute(
        context: AssetContext,
        config: object,
        module: str,
        clip: object,
    ) -> precheck.PrecheckModuleExecution:
        executions.append(module)
        return precheck.PrecheckModuleExecution(
            result=ModuleResult(module, "pass", {}, {}),
            check_results=(CheckResult(module, 0, -1, {}, False, "ok"),),
            candidate_windows=(),
        )

    monkeypatch.setattr(precheck, "_run_module_on_clip", execute)
    with monkeypatch.context() as legacy:
        legacy.setattr(
            precheck,
            "_TEMPORAL_OUTPUT_SCHEMA_VERSION",
            "keypoint_temporal.output.v1",
        )
        old_session = precheck.PrecheckSession(context, config)
        for module in precheck.MODULES:
            old_session.run_module(module)

    executions.clear()
    current_session = precheck.PrecheckSession(context, config)
    current_session.run_module("hdf5_text_info")

    assert executions == ["hdf5_text_info"]


def test_precheck_fingerprint_includes_dr_hdf5_reference_dataset(
    tmp_path: Path,
) -> None:
    from qc_pipeline.runners.precheck import precheck_fingerprint
    from tests.test_qc_pipeline_precheck_session import _config, _context

    base = _context(tmp_path)
    timestamp_context = replace(
        base,
        metadata={
            **dict(base.metadata),
            "supplier": "dr",
            "hdf5_reference_dataset": "timestamp",
        },
    )
    joints_context = replace(
        timestamp_context,
        metadata={
            **dict(timestamp_context.metadata),
            "hdf5_reference_dataset": "hand/left/joints3d",
        },
    )

    assert precheck_fingerprint(timestamp_context, _config(tmp_path)) != (
        precheck_fingerprint(joints_context, _config(tmp_path))
    )


def test_sam3_cache_uses_candidate_sha_and_skips_segmenter_on_hit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qc_pipeline.runners.sam3_containment import runner
    from tests.test_qc_pipeline_sam3_runner import (
        _candidate,
        _context,
        _write_candidates,
        _write_current_run_config,
    )

    _write_candidates(tmp_path, [_candidate(10, 12)])
    producer_calls = 0
    factory_calls = 0

    def fake_run(**kwargs: object) -> dict[str, object]:
        nonlocal producer_calls
        producer_calls += 1
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

    def factory() -> object:
        nonlocal factory_calls
        factory_calls += 1
        return object()

    context = _context(tmp_path)
    _write_current_run_config(context)
    config = load_qc_acceptance_config()
    first = runner(factory)(context, config)
    second = runner(lambda: pytest.fail("cache hit must not load segmenter"))(
        context, config
    )

    assert producer_calls == 1
    assert factory_calls == 1
    assert first.runtime["artifact_state"] == "computed"
    assert second.runtime["artifact_state"] == "reused"
    sam3_artifact = (
        tmp_path / "module_outputs" / "asset-a" / "sam3_containment"
    )
    assert json.loads((sam3_artifact / "frame_results.json").read_text()) == []
    assert json.loads((sam3_artifact / "failures.json").read_text()) == []
    assert json.loads((sam3_artifact / "producer_run_config.json").read_text()) == {}

    _write_candidates(tmp_path, [_candidate(13, 14)])
    runner(factory)(context, config)

    assert producer_calls == 2
    assert factory_calls == 2
