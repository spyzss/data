from __future__ import annotations

import json
from pathlib import Path

import pytest

from acceptance_pull.video_quality import (
    Hdf5Alignment,
    VideoQualityResult,
    analyze_video,
    evaluate_video_quality,
    load_video_quality_config,
)
from qc_common.config import LoadedQcConfig, load_qc_acceptance_config
from qc_common.contracts import ModuleResult
from qc_common.report import StaleReportRevisionError
from qc_common.report_mutation import apply_module_result
from qc_pipeline.context import AssetContext
from tests.fixtures import solid_frame, write_test_video


@pytest.fixture
def loaded_v2_config() -> LoadedQcConfig:
    return load_qc_acceptance_config()


@pytest.fixture
def video_result(tmp_path: Path) -> VideoQualityResult:
    path = tmp_path / "video" / "clip.mp4"
    write_test_video(path, [solid_frame(90) for _ in range(20)], fps=22.0)
    config = load_video_quality_config(None)
    metrics = analyze_video(path, config)
    alignment = Hdf5Alignment(
        status="range_not_evaluated",
        hdf5_path=None,
        hdf5_frame_count=None,
        frame_count_match=None,
        reason="logical range alignment validated by manifest bounds",
    )
    evaluation = evaluate_video_quality(metrics, config, alignment=None)
    return VideoQualityResult(metrics, alignment, evaluation)


def test_batch_and_range_video_use_same_module_shape(
    video_result: VideoQualityResult,
    loaded_v2_config: LoadedQcConfig,
    tmp_path: Path,
) -> None:
    from qc_pipeline.adapters.video_quality import adapt_video_quality_result

    batch = adapt_video_quality_result(
        result=video_result,
        config=loaded_v2_config,
        batch_root=tmp_path,
    )
    ranged = adapt_video_quality_result(
        result=video_result,
        config=loaded_v2_config,
        batch_root=tmp_path,
        source_range=(20, 40),
    )

    assert set(batch.to_dict()) == set(ranged.to_dict())
    assert batch.module == ranged.module == "video_quality"
    assert batch.verdict == ranged.verdict == video_result.evaluation.decision
    assert {
        key: value for key, value in batch.evaluation.items() if key != "issue_ids"
    } == {
        key: value for key, value in ranged.evaluation.items() if key != "issue_ids"
    }
    assert batch.metrics == ranged.metrics
    assert batch.runtime == ranged.runtime
    assert ranged.issues[0].context == {
        "coordinate_system": "source_video_inclusive",
        "start_frame": 20,
        "end_frame": 39,
    }
    assert ranged.evidence[0].path == "video/clip.mp4"
    assert ranged.evidence[0].coordinate_system == "source_video_inclusive"
    assert ranged.evidence[0].start_frame == 20
    assert ranged.evidence[0].end_frame == 39
    assert ranged.issues[0].issue_id != batch.issues[0].issue_id
    assert ranged.issues[0].evidence_ids == (ranged.evidence[0].evidence_id,)
    assert ranged.evaluation["issue_ids"] == [
        issue.issue_id for issue in ranged.issues
    ]
    assert all(not issue_id.endswith(":001") for issue_id in ranged.evaluation["issue_ids"])


def test_video_adapter_preserves_explicit_legacy_payload_fields(
    video_result: VideoQualityResult,
    loaded_v2_config: LoadedQcConfig,
    tmp_path: Path,
) -> None:
    from acceptance_pull.video_quality import asset_qc_result_to_json
    from qc_pipeline.adapters.video_quality import adapt_video_quality_result

    legacy = asset_qc_result_to_json(
        video_result,
        load_video_quality_config(loaded_v2_config.path),
    )
    adapted = adapt_video_quality_result(
        result=video_result,
        config=loaded_v2_config,
        batch_root=tmp_path,
    )

    expected_evaluation = dict(legacy["video_quality"]["evaluation"])
    expected_evaluation["issue_ids"] = [issue.issue_id for issue in adapted.issues]
    assert adapted.evaluation == expected_evaluation
    assert adapted.metrics == {
        **legacy["video_quality"]["metrics"],
        "metadata": legacy["video_quality"]["metadata"],
        "sampling": legacy["video_quality"]["sampling"],
        "reference_quality": legacy["reference_quality"],
    }
    assert adapted.runtime == {
        "stage": legacy["video_quality"]["stage"],
        "module_version": legacy["video_quality"]["module_version"],
        "errors": legacy["video_quality"]["errors"],
    }


def _advance_to_video_quality(
    context: AssetContext,
    config: LoadedQcConfig,
) -> int:
    revision = 0
    modules = config.pipeline_modules
    video_index = modules.index("video_quality")
    for index, module in enumerate(modules[:video_index]):
        report = apply_module_result(
            context.report_path,
            context=context,
            config=config,
            profile="acceptance",
            result=ModuleResult(module, "pass", {}, {}),
            expected_revision=revision,
            next_module=modules[index + 1],
            now=f"2026-07-14T00:00:{index:02d}Z",
        )
        revision = report["report_revision"]
    return revision


def test_video_writer_uses_expected_revision_and_preserves_extensions(
    video_result: VideoQualityResult,
    loaded_v2_config: LoadedQcConfig,
    tmp_path: Path,
) -> None:
    from qc_pipeline.adapters.video_quality import write_video_quality_result

    context = AssetContext(
        asset_id="clip",
        batch_root=tmp_path,
        report_path=tmp_path / "quality_archive" / "clip.json",
        source_files={"video": {"path": "video/clip.mp4"}},
    )
    revision = _advance_to_video_quality(context, loaded_v2_config)
    existing = json.loads(context.report_path.read_text(encoding="utf-8"))
    existing["future_extension"] = {"keep": True}
    context.report_path.write_text(json.dumps(existing), encoding="utf-8")

    updated = write_video_quality_result(
        context=context,
        result=video_result,
        config=loaded_v2_config,
        profile="acceptance",
        expected_revision=revision,
        next_module="sam3_containment",
    )

    assert updated["report_revision"] == revision + 1
    assert updated["future_extension"] == {"keep": True}
    assert updated["video_quality"]["flow"]["result_gate"]["verdict"] == (
        video_result.evaluation.decision
    )
    before = context.report_path.read_bytes()
    with pytest.raises(StaleReportRevisionError):
        write_video_quality_result(
            context=context,
            result=video_result,
            config=loaded_v2_config,
            profile="acceptance",
            expected_revision=revision,
            next_module="sam3_containment",
        )
    assert context.report_path.read_bytes() == before
