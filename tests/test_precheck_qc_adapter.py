from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path

import pytest

from precheck.config import (
    CompositeFrameVerdictConfig,
    KeypointMissingConfig,
    KeypointMorphologyConfig,
    KeypointTemporalConfig,
    QualityScoreConfig,
    SkeletonQualityScoreConfig,
    TextIntegrityConfig,
)
from qc_common.types import CheckResult
from qc_pipeline.adapters.precheck import (
    _contiguous_ranges,
    adapt_hdf5_text_info,
    adapt_keypoint_presence,
    adapt_quality_hand,
    precheck_config_from_unified,
)
from tests.qc_report_fixtures import loaded_test_config


def _text_result(
    *,
    metrics: dict[str, object],
    flag: bool | None,
    reason: str,
) -> CheckResult:
    return CheckResult("text_integrity", 0, -1, metrics, flag, reason)


def _quality_frame(
    frame_idx: int,
    *,
    left: object = 1.0,
    right: object = 1.0,
    include_left: bool = True,
    include_right: bool = True,
) -> CheckResult:
    metrics: dict[str, object] = {"frame_score": 1.0}
    if include_left:
        metrics["quality_left"] = left
    if include_right:
        metrics["quality_right"] = right
    if left == 0.0 or right == 0.0:
        metrics["frame_score"] = 0.0
    return CheckResult("quality_score", 0, frame_idx, metrics, None, "per-frame")


def _quality_summary(*, pass_ratio: float, num_frames: int) -> CheckResult:
    return CheckResult(
        "quality_score",
        0,
        -1,
        {"pass_ratio": pass_ratio, "num_frames": float(num_frames)},
        pass_ratio >= 0.9,
        "summary",
    )


def test_text_integrity_missing_field_maps_to_hard_fail() -> None:
    result = adapt_hdf5_text_info(
        asset_id="a",
        source_relative_path="hdf5/a.h5",
        results=[
            _text_result(
                metrics={"missing_field_count": 1.0, "field_present_task": 0.0},
                flag=True,
                reason='{"missing_fields":["task"]}',
            )
        ],
        config=loaded_test_config(),
    )

    assert result.module == "hdf5_text_info"
    assert result.verdict == "fail"
    assert result.issues[0].rule_id == "hdf5_text.missing_required_field"
    assert result.issues[0].needs_manual_review is False


def test_text_integrity_empty_required_field_preserves_detector_fail() -> None:
    result = adapt_hdf5_text_info(
        asset_id="a",
        source_relative_path="hdf5/a.h5",
        results=[
            _text_result(
                metrics={"missing_field_count": 0.0, "empty_field_count": 1.0},
                flag=True,
                reason='{"empty_fields":["task"]}',
            )
        ],
        config=loaded_test_config(),
    )

    assert result.verdict == "fail"
    assert result.issues[0].rule_id == "hdf5_text.missing_required_field"


def test_text_integrity_invalid_json_uses_missing_text_rule() -> None:
    result = adapt_hdf5_text_info(
        asset_id="a",
        source_relative_path="hdf5/a.h5",
        results=[
            _text_result(
                metrics={"missing_field_count": 0.0},
                flag=True,
                reason="text_label not valid JSON",
            )
        ],
        config=loaded_test_config(),
    )

    assert result.verdict == "fail"
    assert result.issues[0].rule_id == "hdf5_text.missing_text_field"


def test_text_integrity_complete_fields_map_to_pass() -> None:
    result = adapt_hdf5_text_info(
        asset_id="a",
        source_relative_path="hdf5/a.h5",
        results=[
            _text_result(
                metrics={"missing_field_count": 0.0, "empty_field_count": 0.0},
                flag=None,
                reason="required text_label fields present and nonempty",
            )
        ],
        config=loaded_test_config(),
    )

    assert result.verdict == "pass"
    assert result.issues == ()


def test_quality_hand_single_side_low_maps_to_warn() -> None:
    rows = [
        _quality_frame(3, left=0.0, right=1.0),
        _quality_summary(pass_ratio=0.5, num_frames=1),
    ]

    result = adapt_quality_hand(
        asset_id="a",
        source_relative_path="hdf5/a.h5",
        results=rows,
        config=loaded_test_config(),
    )

    assert result.verdict == "warn"
    assert result.issues[0].rule_id == "quality_hand.single_hand_low_quality"
    assert result.issues[0].needs_manual_review is True
    assert result.issues[0].context == {
        "coordinate_system": "source_inclusive",
        "start_frame": 3,
        "end_frame": 3,
        "hand_side": "left",
    }


def test_quality_hand_both_sides_low_on_same_frame_maps_to_fail() -> None:
    result = adapt_quality_hand(
        asset_id="a",
        source_relative_path="hdf5/a.h5",
        results=[
            _quality_frame(8, left=0.0, right=0.0),
            _quality_summary(pass_ratio=0.0, num_frames=1),
        ],
        config=loaded_test_config(),
    )

    assert result.verdict == "fail"
    assert len(result.issues) == 1
    assert result.issues[0].rule_id == "quality_hand.both_hands_low_quality"
    assert result.issues[0].needs_manual_review is False
    assert result.issues[0].context["hand_side"] == "both"


def test_quality_hand_invalid_value_maps_to_configured_hard_fail() -> None:
    result = adapt_quality_hand(
        asset_id="a",
        source_relative_path="hdf5/a.h5",
        results=[
            _quality_frame(4, left=0.5, right=1.0),
            _quality_summary(pass_ratio=1.0, num_frames=1),
        ],
        config=loaded_test_config(),
    )

    assert result.verdict == "fail"
    assert result.issues[0].rule_id == "quality_hand.invalid_value"
    assert result.issues[0].needs_manual_review is False
    assert result.issues[0].context["hand_side"] == "left"
    assert result.issues[0].operator == "not in"


def test_quality_hand_invalid_frame_shape_maps_to_configured_hard_fail() -> None:
    result = adapt_quality_hand(
        asset_id="a",
        source_relative_path="hdf5/a.h5",
        results=[
            _quality_frame(5, left=0.0, include_right=False),
            _quality_summary(pass_ratio=0.0, num_frames=1),
        ],
        config=loaded_test_config(),
    )

    assert result.verdict == "fail"
    assert result.issues[0].rule_id == "quality_hand.invalid_shape"
    assert result.issues[0].needs_manual_review is False
    assert result.issues[0].context["start_frame"] == 5
    assert result.issues[0].operator == "!="


def test_quality_hand_absent_source_signal_maps_to_skipped() -> None:
    result = adapt_quality_hand(
        asset_id="a",
        source_relative_path="hdf5/a.h5",
        results=[],
        config=loaded_test_config(),
    )

    assert result.verdict == "skipped"
    assert result.evaluation["reason"] == "source_signal_not_provided"
    assert result.issues == ()


def test_quality_hand_all_valid_frames_map_to_pass() -> None:
    result = adapt_quality_hand(
        asset_id="a",
        source_relative_path="hdf5/a.h5",
        results=[
            _quality_frame(0),
            _quality_frame(1),
            _quality_summary(pass_ratio=1.0, num_frames=2),
        ],
        config=loaded_test_config(),
    )

    assert result.verdict == "pass"
    assert result.issues == ()


def test_quality_hand_issue_ids_are_stable() -> None:
    kwargs = {
        "asset_id": "a",
        "source_relative_path": "hdf5/a.h5",
        "results": [
            _quality_frame(3, left=0.0, right=1.0),
            _quality_summary(pass_ratio=0.5, num_frames=1),
        ],
        "config": loaded_test_config(),
    }

    first = adapt_quality_hand(**kwargs)
    second = adapt_quality_hand(**kwargs)

    assert first.issues[0].issue_id == second.issues[0].issue_id


def test_presence_merges_contiguous_invalid_frames_into_one_issue() -> None:
    rows = [
        CheckResult(
            "skeleton_quality_score",
            0,
            frame,
            {
                "keypoint_presence_invalid": 1.0,
                "valid_keypoint_count_left": 7.0,
                "valid_keypoint_count_right": 21.0,
            },
            True,
            "presence invalid",
        )
        for frame in (10, 11, 12)
    ]

    result = adapt_keypoint_presence(
        asset_id="a",
        source_relative_path="hdf5/a.h5",
        results=rows,
        config=loaded_test_config(),
    )

    assert result.verdict == "fail"
    assert len(result.issues) == 1
    assert result.issues[0].context == {
        "coordinate_system": "source_inclusive",
        "start_frame": 10,
        "end_frame": 12,
        "hand_side": "left",
    }
    repeated = adapt_keypoint_presence(
        asset_id="a",
        source_relative_path="hdf5/a.h5",
        results=rows,
        config=loaded_test_config(),
    )
    assert repeated.issues[0].issue_id == result.issues[0].issue_id


@pytest.mark.parametrize(
    ("invalid_value", "serialized"),
    [(float("nan"), "nan"), (float("inf"), "inf")],
)
def test_presence_nonfinite_valid_count_maps_to_hard_fail(
    invalid_value: float,
    serialized: str,
) -> None:
    row = CheckResult(
        "skeleton_quality_score",
        0,
        5,
        {
            "keypoint_presence_invalid": 1.0,
            "valid_keypoint_count_left": invalid_value,
            "valid_keypoint_count_right": 21.0,
        },
        True,
        "presence invalid",
    )

    result = adapt_keypoint_presence(
        asset_id="a",
        source_relative_path="hdf5/a.h5",
        results=[row],
        config=loaded_test_config(),
    )

    assert result.verdict == "fail"
    assert result.issues[0].rule_id == "keypoint_presence.nan_or_inf"
    assert result.issues[0].observed_value == serialized
    assert result.issues[0].context["hand_side"] == "left"


def test_presence_missing_ratio_warn_becomes_manual_candidate() -> None:
    rows = [
        CheckResult(
            "keypoint_missing",
            0,
            frame,
            {
                "quality_low_left": 1.0,
                "quality_low_right": 0.0,
                "missing_fraction_in_10s_window_left": 0.1,
                "missing_fraction_in_10s_window_right": 0.0,
            },
            True,
            "missing ratio above warn boundary",
        )
        for frame in (20, 21)
    ]

    result = adapt_keypoint_presence(
        asset_id="a",
        source_relative_path="hdf5/a.h5",
        results=rows,
        config=loaded_test_config(),
    )

    assert result.verdict == "warn"
    assert len(result.issues) == 1
    assert result.issues[0].rule_id == "keypoint_presence.high_missing_frame_ratio"
    assert result.issues[0].needs_manual_review is True
    assert result.issues[0].context == {
        "coordinate_system": "source_inclusive",
        "start_frame": 20,
        "end_frame": 21,
        "hand_side": "left",
    }


def test_presence_keeps_only_worst_compatible_failure_per_frame() -> None:
    rows = [
        CheckResult(
            "skeleton_quality_score",
            0,
            10,
            {
                "keypoint_presence_invalid": 1.0,
                "valid_keypoint_count_left": 9.0,
                "valid_keypoint_count_right": 21.0,
            },
            True,
            "count warning",
        ),
        CheckResult(
            "keypoint_missing",
            0,
            10,
            {
                "quality_low_left": 1.0,
                "quality_low_right": 0.0,
                "missing_fraction_in_10s_window_left": 0.25,
                "missing_fraction_in_10s_window_right": 0.0,
            },
            True,
            "ratio failure",
        ),
    ]

    result = adapt_keypoint_presence(
        asset_id="a",
        source_relative_path="hdf5/a.h5",
        results=rows,
        config=loaded_test_config(),
    )

    assert result.verdict == "fail"
    assert len(result.issues) == 1
    assert result.issues[0].rule_id == "keypoint_presence.high_missing_frame_ratio"
    assert result.issues[0].context["start_frame"] == 10
    assert result.issues[0].context["end_frame"] == 10


def test_presence_only_merges_contiguous_compatible_failures() -> None:
    rows = [
        CheckResult(
            "skeleton_quality_score",
            0,
            frame,
            {
                "keypoint_presence_invalid": 1.0,
                "valid_keypoint_count_left": count,
                "valid_keypoint_count_right": 21.0,
            },
            True,
            "presence invalid",
        )
        for frame, count in ((12, 7.0), (10, 7.0), (11, 9.0))
    ]
    rows.extend(
        CheckResult(
            "skeleton_quality_score",
            0,
            frame,
            {
                "keypoint_presence_invalid": 0.0,
                "valid_keypoint_count_left": 21.0,
                "valid_keypoint_count_right": 21.0,
            },
            None,
            "presence valid",
        )
        for frame in range(20)
        if frame not in {10, 11, 12}
    )

    result = adapt_keypoint_presence(
        asset_id="a",
        source_relative_path="hdf5/a.h5",
        results=rows,
        config=loaded_test_config(),
    )

    assert [issue.context["start_frame"] for issue in result.issues] == [
        10,
        11,
        12,
    ]
    assert [issue.severity for issue in result.issues] == ["fail", "warn", "fail"]
    assert all(
        issue.context["start_frame"] == issue.context["end_frame"]
        for issue in result.issues
    )


def test_presence_explicit_missing_keypoint_field_maps_to_hard_fail() -> None:
    result = adapt_keypoint_presence(
        asset_id="a",
        source_relative_path="hdf5/a.h5",
        results=[
            CheckResult(
                "skeleton_quality_score",
                0,
                -1,
                {"keypoint_field_present": 0.0},
                True,
                "explicit source structure signal",
            )
        ],
        config=loaded_test_config(),
    )

    assert result.verdict == "fail"
    assert result.issues[0].rule_id == "keypoint_presence.missing_keypoint_field"
    assert result.issues[0].needs_manual_review is False
    assert result.issues[0].context == {
        "coordinate_system": "source_inclusive",
        "start_frame": None,
        "end_frame": None,
        "hand_side": None,
    }


def test_presence_uses_detector_invalid_frame_ratio_for_warn() -> None:
    rows = [
        CheckResult(
            "skeleton_quality_score",
            0,
            frame,
            {
                "keypoint_presence_invalid": float(frame in {3, 4}),
                "valid_keypoint_count_left": 20.0 if frame in {3, 4} else 21.0,
                "valid_keypoint_count_right": 21.0,
            },
            True if frame in {3, 4} else None,
            "per-frame presence",
        )
        for frame in range(20)
    ]

    result = adapt_keypoint_presence(
        asset_id="a",
        source_relative_path="hdf5/a.h5",
        results=rows,
        config=loaded_test_config(),
    )

    assert result.verdict == "warn"
    assert result.evaluation["checked_frame_count"] == 20
    assert result.evaluation["invalid_frame_count"] == 2
    assert result.evaluation["invalid_frame_ratio"] == pytest.approx(0.1)
    assert result.metrics["invalid_frame_ranges_by_hand"] == {
        "left": ((3, 4),),
        "right": (),
    }
    assert result.issues[0].rule_id == "keypoint_presence.high_missing_frame_ratio"
    assert result.issues[0].context["start_frame"] == 3
    assert result.issues[0].context["end_frame"] == 4


def test_presence_does_not_infer_missing_keypoints_from_absent_quality_hand() -> None:
    result = adapt_keypoint_presence(
        asset_id="a",
        source_relative_path="hdf5/a.h5",
        results=[],
        config=loaded_test_config(),
    )

    assert result.verdict == "skipped"
    assert result.evaluation["reason"] == "source_signal_not_provided"
    assert result.issues == ()


def test_contiguous_ranges_sorts_deduplicates_and_preserves_gaps() -> None:
    assert _contiguous_ranges((8, 5, 6, 6, 10)) == ((5, 6), (8, 8), (10, 10))


def _expected_dataclass_values(cls: type[object], parameters: dict[str, object]) -> dict[str, object]:
    return {item.name: parameters[item.name] for item in fields(cls)}


def test_unified_config_is_injected_field_by_field_into_legacy_precheck(
    tmp_path: Path,
) -> None:
    unified = loaded_test_config()
    runtime = precheck_config_from_unified(
        unified,
        module_names=(
            "text_integrity",
            "quality_score",
            "keypoint_missing",
            "keypoint_morphology",
            "keypoint_temporal",
            "skeleton_quality_score",
            "composite_frame_verdict",
        ),
        output_dir=tmp_path,
    )
    text = unified.module_parameters("hdf5_text_info")
    quality = unified.module_parameters("quality_hand")
    presence = unified.module_parameters("keypoint_presence")
    morphology = unified.module_parameters("keypoint_morphology")
    temporal = unified.module_parameters("keypoint_temporal")

    assert runtime.enabled_checks == [
        "text_integrity",
        "quality_score",
        "keypoint_missing",
        "keypoint_morphology",
        "keypoint_temporal",
        "skeleton_quality_score",
        "composite_frame_verdict",
    ]
    assert asdict(runtime.text_integrity) == _expected_dataclass_values(
        TextIntegrityConfig, text
    )
    assert asdict(runtime.quality_score) == _expected_dataclass_values(
        QualityScoreConfig, quality
    )
    assert asdict(runtime.keypoint_missing) == _expected_dataclass_values(
        KeypointMissingConfig, presence
    )
    assert asdict(runtime.keypoint_morphology) == _expected_dataclass_values(
        KeypointMorphologyConfig, morphology
    )
    assert asdict(runtime.keypoint_temporal) == _expected_dataclass_values(
        KeypointTemporalConfig, temporal
    )
    assert asdict(runtime.skeleton_quality_score) == _expected_dataclass_values(
        SkeletonQualityScoreConfig, temporal
    )
    assert asdict(runtime.composite_frame_verdict) == _expected_dataclass_values(
        CompositeFrameVerdictConfig, temporal
    )


def test_unified_module_names_map_to_existing_precheck_names(tmp_path: Path) -> None:
    runtime = precheck_config_from_unified(
        loaded_test_config(),
        module_names=("hdf5_text_info", "quality_hand", "keypoint_presence"),
        output_dir=tmp_path,
    )

    assert runtime.enabled_checks == [
        "text_integrity",
        "quality_score",
        "keypoint_missing",
    ]


def test_manifest_runtime_keeps_unified_name_mapping(tmp_path: Path) -> None:
    from tools.run_manifest_precheck import _configured_precheck

    runtime = _configured_precheck(
        tmp_path,
        "deepreach",
        ["quality_hand"],
        None,
        False,
    )

    assert runtime.enabled_checks == ["quality_score"]
    assert runtime.quality_score.pass_threshold == loaded_test_config().module_parameters(
        "quality_hand"
    )["pass_threshold"]


def test_manifest_cli_rejects_legacy_and_unified_config_together() -> None:
    from tools.run_manifest_precheck import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--manifest",
                "manifest.csv",
                "--supplier",
                "deepreach",
                "--output-dir",
                "out",
                "--config-path",
                "legacy.yaml",
                "--qc-config",
                "configs/qc_acceptance.yaml",
            ]
        )
