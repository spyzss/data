from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def _sam3_rows() -> list[dict[str, object]]:
    verdicts = ("fail", "review", "pass", "pass", "blocked")
    rows = []
    for source_frame, verdict in enumerate(verdicts):
        rows.append(
            {
                "asset_id": "jdt__episode_1",
                "supplier": "jdt",
                "source_frame": source_frame,
                "video_frame": source_frame + 10,
                "frame_verdict": verdict,
                "frame_reason_codes": json.dumps([f"sam3_{verdict}"]),
                "left_hand_verdict": verdict if verdict != "blocked" else "blocked",
                "right_hand_verdict": "pass" if verdict != "blocked" else "blocked",
                "left_inside_ratio": 0.1 if verdict == "fail" else 0.5,
                "right_inside_ratio": 0.9,
                "left_projected_in_image_ratio": 1.0,
                "right_projected_in_image_ratio": 1.0,
                "left_mask_status": "present" if verdict != "blocked" else "not_run",
                "right_mask_status": "present" if verdict != "blocked" else "not_run",
                "evidence_path": f"overlays/frame_{source_frame}.png"
                if verdict in {"fail", "review"}
                else None,
            }
        )
    return rows


def _precheck_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for frame in range(5):
        for check in (
            "keypoint_missing",
            "keypoint_morphology",
            "keypoint_temporal",
        ):
            flag = (frame == 0 and check == "keypoint_missing") or (
                frame == 2 and check == "keypoint_morphology"
            )
            rows.append(
                {
                    "asset_id": "jdt__episode_1",
                    "source_frame_idx": frame,
                    "check": check,
                    "flag": flag,
                    "reason": f"{check}_{'flag' if flag else 'ok'}",
                    "metrics": {},
                }
            )
        rows.append(
            {
                "asset_id": "jdt__episode_1",
                "source_frame_idx": frame,
                "check": "skeleton_quality_score",
                "flag": False,
                "reason": "geometry_ok",
                "metrics": {
                    "needs_projection_review": 0.0,
                    "needs_out_of_frame_review": 0.0,
                },
            }
        )
    rows.extend(
        [
            {
                "asset_id": "jdt__episode_1",
                "source_frame_idx": 3,
                "check": "composite_frame_verdict",
                "flag": True,
                "reason": "supplier did not downweight",
                "metrics": {"supplier_quality_left": 1},
            },
            {
                "asset_id": "jdt__episode_1",
                "source_frame_idx": 3,
                "check": "text_integrity",
                "flag": True,
                "reason": "text problem",
                "metrics": {},
            },
            {
                "asset_id": "jdt__episode_1",
                "source_frame_idx": 3,
                "check": "video_quality",
                "flag": True,
                "reason": "video problem",
                "metrics": {},
            },
        ]
    )
    return rows


def test_precheck_projection_unions_hands_reasons_and_candidate_overlap_once() -> None:
    from tools.build_precheck_sam3_frame_comparison import project_precheck_frames

    duplicated = _precheck_rows() + [
        {
            "asset_id": "jdt__episode_1",
            "source_frame_idx": 0,
            "check": "keypoint_missing",
            "flag": True,
            "reason": "right hand also missing",
            "metrics": {"hand_side": "right"},
        }
    ]
    candidates = [
        {
            "asset_id": "jdt__episode_1",
            "start_frame": 0,
            "end_frame": 1,
            "review_id": "window-a",
        },
        {
            "asset_id": "jdt__episode_1",
            "start_frame": 1,
            "end_frame": 2,
            "review_id": "window-b",
        },
    ]

    projected = project_precheck_frames(
        sam3_frame_rows=_sam3_rows(),
        check_rows=duplicated,
        candidate_rows=candidates,
    )

    assert projected["source_frame"].tolist() == [0, 1, 2, 3, 4]
    assert projected["precheck_problem_any"].tolist() == [True, False, True, False, False]
    assert projected.loc[0, "keypoint_presence_flag"]
    assert json.loads(projected.loc[0, "precheck_reason_codes"]) == [
        "keypoint_missing_flag",
        "right hand also missing",
    ]
    assert projected["candidate_window_membership"].tolist() == [
        True,
        True,
        True,
        False,
        False,
    ]
    assert json.loads(projected.loc[1, "candidate_review_ids"]) == [
        "window-a",
        "window-b",
    ]


def test_supplier_quality_text_video_and_supplier_composite_are_excluded() -> None:
    from tools.build_precheck_sam3_frame_comparison import project_precheck_frames

    projected = project_precheck_frames(
        sam3_frame_rows=_sam3_rows(),
        check_rows=_precheck_rows(),
        candidate_rows=[],
    )

    assert not projected.loc[3, "precheck_problem_any"]
    assert "composite_frame_verdict" not in projected.loc[3, "precheck_modules"]
    assert "supplier did not downweight" not in projected.loc[3, "precheck_reason_codes"]


def test_binary_join_uses_exact_asset_and_source_frame_confusion_matrix() -> None:
    from tools.build_precheck_sam3_frame_comparison import compare_frames

    comparison = compare_frames(
        sam3_frame_rows=_sam3_rows(),
        precheck_frame_rows=project_precheck_for_test(),
    )

    assert comparison["comparison_class"].tolist() == ["TP", "FN", "FP", "TN", "UNEVALUABLE"]
    assert comparison["sam3_binary_label"].tolist() == [
        "problem",
        "problem",
        "clean",
        "clean",
        "unevaluable",
    ]
    assert comparison.loc[4, "exclusion_reason"] == "sam3_blocked"
    assert comparison["evaluable"].tolist() == [True, True, True, True, False]
    assert comparison["sam3_frame_verdict"].tolist() == [
        "fail",
        "review",
        "pass",
        "pass",
        "blocked",
    ]
    assert json.loads(comparison.loc[0, "sam3_reason_codes"]) == ["sam3_fail"]


def project_precheck_for_test() -> pd.DataFrame:
    from tools.build_precheck_sam3_frame_comparison import project_precheck_frames

    return project_precheck_frames(
        sam3_frame_rows=_sam3_rows(),
        check_rows=_precheck_rows(),
        candidate_rows=[],
    )


def test_summary_formulas_keep_counts_denominators_and_null_zero_divisions() -> None:
    from tools.build_precheck_sam3_frame_comparison import (
        build_comparison_summary,
        compare_frames,
    )

    comparison = compare_frames(
        sam3_frame_rows=_sam3_rows(),
        precheck_frame_rows=project_precheck_for_test(),
    )
    summary = build_comparison_summary(comparison)

    assert (summary["TP"], summary["FN"], summary["FP"], summary["TN"]) == (1, 1, 1, 1)
    assert summary["detection_rate"] == {
        "numerator": 1,
        "denominator": 2,
        "value": 0.5,
    }
    assert summary["miss_rate"]["value"] == 0.5
    assert summary["precision"]["value"] == 0.5
    assert summary["false_discovery_rate"]["value"] == 0.5
    assert summary["false_positive_rate"]["value"] == 0.5
    assert summary["accuracy"]["value"] == 0.5
    assert summary["sam3_problem_frame_rate"] == {
        "numerator": 2,
        "denominator": 4,
        "value": 0.5,
    }
    assert summary["precheck_flagged_frame_rate"] == {
        "numerator": 2,
        "denominator": 4,
        "value": 0.5,
    }
    assert summary["evaluable_coverage"] == {
        "numerator": 4,
        "denominator": 5,
        "value": 0.8,
    }
    assert summary["total_evaluable_frames"] == 4
    assert summary["total_unevaluable_frames"] == 1

    zero = build_comparison_summary(comparison.iloc[[4]])
    assert zero["detection_rate"] == {
        "numerator": 0,
        "denominator": 0,
        "value": None,
        "status": "not_applicable",
    }


def test_contiguous_intervals_are_source_inclusive_and_never_cross_assets() -> None:
    from tools.build_precheck_sam3_frame_comparison import contiguous_intervals

    rows = pd.DataFrame(
        [
            {"asset_id": "a", "source_frame": 2, "comparison_class": "FN"},
            {"asset_id": "a", "source_frame": 3, "comparison_class": "FN"},
            {"asset_id": "a", "source_frame": 5, "comparison_class": "FN"},
            {"asset_id": "b", "source_frame": 0, "comparison_class": "FN"},
        ]
    )

    intervals = contiguous_intervals(rows, class_name="FN")

    assert intervals.to_dict(orient="records") == [
        {"asset_id": "a", "start_frame": 2, "end_frame": 3, "frame_count": 2},
        {"asset_id": "a", "start_frame": 5, "end_frame": 5, "frame_count": 1},
        {"asset_id": "b", "start_frame": 0, "end_frame": 0, "frame_count": 1},
    ]


def test_builder_materializes_required_tables_and_exact_false_negative_rows(
    tmp_path: Path,
) -> None:
    from tools.build_precheck_sam3_frame_comparison import build_comparison_outputs

    output_dir = tmp_path / "comparison"
    summary = build_comparison_outputs(
        sam3_frame_rows=_sam3_rows(),
        check_rows=_precheck_rows(),
        candidate_rows=[],
        output_dir=output_dir,
        manifest_asset_ids=("jdt__episode_1",),
        model_run_lineage={"producer": "test-v1"},
    )

    mandatory = {
        "precheck_frame_flags.parquet",
        "precheck_frame_flags.csv",
        "precheck_vs_sam3_frame_comparison.parquet",
        "precheck_vs_sam3_frame_comparison.csv",
        "false_negative_frames.parquet",
        "false_negative_frames.csv",
        "false_positive_frames.parquet",
        "false_positive_frames.csv",
        "true_positive_frames.parquet",
        "true_positive_frames.csv",
        "false_negative_intervals.csv",
        "false_positive_intervals.csv",
        "sam3_problem_intervals.csv",
        "true_positive_intervals.csv",
        "sam3_fail_intervals.csv",
        "sam3_review_intervals.csv",
        "unevaluable_intervals.csv",
        "precheck_vs_sam3_summary.json",
        "precheck_vs_sam3_per_asset.csv",
        "assets_ranked_by_false_negatives.csv",
        "precheck_module_coverage.csv",
        "precheck_sam3_reason_matrix.csv",
        "review_evidence_manifest.csv",
    }
    assert mandatory.issubset({path.name for path in output_dir.iterdir()})
    false_negatives = pd.read_csv(output_dir / "false_negative_frames.csv")
    assert false_negatives[["asset_id", "source_frame"]].to_dict(orient="records") == [
        {"asset_id": "jdt__episode_1", "source_frame": 1}
    ]
    assert summary["FN"] == 1
    assert summary["raw_frame_recall"]["value"] == 0.5
    assert summary["total_manifest_assets"] == 1
    assert summary["total_source_frames"] == 5
    assert summary["sam3_fail_count"] == 1
    assert summary["sam3_review_count"] == 1
    assert summary["sam3_pass_count"] == 2
    assert summary["sam3_blocked_count"] == 1

    frame_csv = pd.read_csv(
        output_dir / "precheck_vs_sam3_frame_comparison.csv"
    )
    frame_parquet = pd.read_parquet(
        output_dir / "precheck_vs_sam3_frame_comparison.parquet"
    )
    assert frame_csv.columns.tolist() == frame_parquet.columns.tolist()

    per_asset = pd.read_csv(output_dir / "precheck_vs_sam3_per_asset.csv")
    assert int(per_asset.loc[0, "source_frame_count"]) == 5
    assert int(per_asset.loc[0, "FN"]) == 1
    assert int(per_asset.loc[0, "first_fn_frame"]) == 1
    assert int(per_asset.loc[0, "fn_interval_count"]) == 1
    assert int(per_asset.loc[0, "recall_numerator"]) == 1
    assert int(per_asset.loc[0, "recall_denominator"]) == 2
    assert int(per_asset.loc[0, "candidate_window_recall_numerator"]) == 0
    assert int(per_asset.loc[0, "candidate_window_recall_denominator"]) == 2

    module_coverage = pd.read_csv(output_dir / "precheck_module_coverage.csv")
    presence = module_coverage.loc[
        module_coverage["module"] == "keypoint_presence"
    ].iloc[0]
    assert int(presence["sam3_problem_frames_uniquely_caught"]) == 1
    assert int(presence["overlap_with_other_modules_count"]) == 0
    assert int(presence["recall_numerator"]) == 1
    assert int(presence["recall_denominator"]) == 2

    reason_matrix = pd.read_csv(output_dir / "precheck_sam3_reason_matrix.csv")
    assert {
        "precheck_module",
        "precheck_reason",
        "sam3_verdict",
        "sam3_reason",
        "comparison_class",
        "frame_count",
    }.issubset(reason_matrix.columns)
    evidence = pd.read_csv(output_dir / "review_evidence_manifest.csv")
    assert {
        "asset_id",
        "source_frame",
        "video_frame",
        "comparison_class",
        "sam3_frame_verdict",
        "precheck_modules",
        "precheck_reason_codes",
        "overlay_path",
        "left_inside_ratio",
        "right_inside_ratio",
        "model_config_identity",
    }.issubset(evidence.columns)


def test_missing_positive_evidence_is_rejected_when_evidence_is_required(
    tmp_path: Path,
) -> None:
    from tools.build_precheck_sam3_frame_comparison import build_comparison_outputs

    sam3 = _sam3_rows()
    sam3[1]["evidence_path"] = None

    with pytest.raises(ValueError, match="SAM3 problem frames missing evidence"):
        build_comparison_outputs(
            sam3_frame_rows=sam3,
            check_rows=_precheck_rows(),
            candidate_rows=[],
            output_dir=tmp_path / "missing-evidence",
            manifest_asset_ids=("jdt__episode_1",),
            model_run_lineage={},
            require_positive_evidence=True,
        )


def test_all_pass_without_evidence_column_writes_empty_evidence_manifest(
    tmp_path: Path,
) -> None:
    from tools.build_precheck_sam3_frame_comparison import build_comparison_outputs

    sam3_rows = [
        {
            key: value
            for key, value in row.items()
            if key != "evidence_path"
        }
        for row in _sam3_rows()[2:4]
    ]

    summary = build_comparison_outputs(
        sam3_frame_rows=sam3_rows,
        check_rows=[],
        candidate_rows=[],
        output_dir=tmp_path / "all-pass",
        manifest_asset_ids=("jdt__episode_1",),
        model_run_lineage={},
    )

    assert summary["TN"] == 2
    evidence = pd.read_csv(tmp_path / "all-pass" / "review_evidence_manifest.csv")
    assert evidence.empty


def test_reason_matrix_preserves_module_reason_pairs_without_cross_product(
    tmp_path: Path,
) -> None:
    from tools.build_precheck_sam3_frame_comparison import build_comparison_outputs

    sam3 = [_sam3_rows()[0]]
    checks = [
        {
            "asset_id": "jdt__episode_1",
            "source_frame_idx": 0,
            "check": "keypoint_missing",
            "flag": True,
            "reason": "presence_reason",
            "metrics": {},
        },
        {
            "asset_id": "jdt__episode_1",
            "source_frame_idx": 0,
            "check": "keypoint_morphology",
            "flag": True,
            "reason": "morphology_reason",
            "metrics": {},
        },
    ]
    build_comparison_outputs(
        sam3_frame_rows=sam3,
        check_rows=checks,
        candidate_rows=[],
        output_dir=tmp_path / "paired-reasons",
        manifest_asset_ids=("jdt__episode_1",),
        model_run_lineage={},
    )

    matrix = pd.read_csv(
        tmp_path / "paired-reasons" / "precheck_sam3_reason_matrix.csv"
    )
    pairs = set(zip(matrix["precheck_module"], matrix["precheck_reason"]))
    assert pairs == {
        ("keypoint_presence", "presence_reason"),
        ("keypoint_morphology", "morphology_reason"),
    }


def test_precheck_artifact_directory_injects_and_validates_asset_identity(
    tmp_path: Path,
) -> None:
    from tools.build_precheck_sam3_frame_comparison import read_precheck_artifacts

    precheck = tmp_path / "module_outputs" / "jdt__episode_1" / "precheck"
    precheck.mkdir(parents=True)
    (precheck / "check_results.json").write_text(
        json.dumps(
            [
                {
                    "frame_idx": 0,
                    "check": "keypoint_missing",
                    "flag": True,
                    "reason": "missing",
                    "metrics": {},
                }
            ]
        ),
        encoding="utf-8",
    )

    rows = read_precheck_artifacts(
        (tmp_path,), artifact_name="check_results.json"
    )
    assert rows[0]["asset_id"] == "jdt__episode_1"

    (precheck / "check_results.json").write_text(
        json.dumps([{**rows[0], "asset_id": "jdt__wrong"}]), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="asset_id disagrees with artifact path"):
        read_precheck_artifacts((tmp_path,), artifact_name="check_results.json")


def _variant_rows(
    *,
    timebase: str,
    sampled_frames: set[int],
    positive_frames: set[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for frame in range(5):
        rows.append(
            {
                "asset_id": "jdt__episode_1",
                "source_frame_idx": frame,
                "check": "keypoint_missing",
                "flag": False,
                "reason": "present",
                "metrics": {"keypoint_presence_invalid": False},
            }
        )
        sampled = frame in sampled_frames
        rows.append(
            {
                "asset_id": "jdt__episode_1",
                "source_frame_idx": frame,
                "check": "skeleton_quality_score",
                "flag": frame in positive_frames if sampled else None,
                "reason": "temporal",
                "metrics": {
                    "decision_metric_source": (
                        "standardized_30hz"
                        if timebase == "standardized"
                        else "native_source_fps"
                    ),
                    "temporal_output_valid": sampled,
                    "standardized_sample_selected": (
                        sampled if timebase == "standardized" else True
                    ),
                    "joint_position_abs_m_max": 0.5,
                },
            }
        )
    return rows


def test_dual_projection_uses_manifest_universe_and_keeps_unsampled_unknown() -> None:
    from tools.build_precheck_sam3_frame_comparison import (
        build_temporal_frame_comparison,
    )

    frame = build_temporal_frame_comparison(
        manifest_rows=[
            {
                "asset_id": "jdt__episode_1",
                "start_frame": 0,
                "end_frame": 4,
            }
        ],
        sam3_frame_rows=_sam3_rows(),
        native_check_rows=_variant_rows(
            timebase="native",
            sampled_frames={0, 1, 2, 3, 4},
            positive_frames={0, 2},
        ),
        standardized_check_rows=_variant_rows(
            timebase="standardized",
            sampled_frames={0, 2, 4},
            positive_frames={0},
        ),
        native_candidate_rows=[],
        standardized_candidate_rows=[],
        candidate_inputs_provided=True,
        thresholds={
            "finite_extreme_displacement_m": None,
            "finite_extreme_acceleration_m_s2": None,
            "finite_extreme_position_abs_m": None,
        },
    )

    assert frame[["asset_id", "source_frame"]].to_dict(orient="records") == [
        {"asset_id": "jdt__episode_1", "source_frame": frame}
        for frame in range(5)
    ]
    assert pd.isna(frame.loc[1, "standardized_temporal_only_flag"])
    assert pd.isna(frame.loc[3, "standardized_any_precheck_flag"])
    assert not bool(frame.loc[1, "standardized_candidate_window_membership"])
    assert frame.loc[1, "standardized_candidate_window_membership_evaluable"]


def test_native_temporal_flag_excludes_presence_only_score_flag() -> None:
    from tools.build_precheck_sam3_frame_comparison import (
        build_temporal_frame_comparison,
    )

    native_rows = [
        {
            "asset_id": "jdt__episode_1",
            "source_frame_idx": 0,
            "check": "keypoint_presence",
            "flag": True,
            "metrics": {"keypoint_presence_invalid": True},
        },
        {
            "asset_id": "jdt__episode_1",
            "source_frame_idx": 0,
            "check": "skeleton_quality_score",
            "flag": True,
            "metrics": {
                "decision_metric_source": "native_source_fps",
                "temporal_output_valid": True,
                "which_thresholds_exceeded": [],
                "keypoint_presence_invalid": True,
            },
        },
    ]
    frame = build_temporal_frame_comparison(
        manifest_rows=[
            {"asset_id": "jdt__episode_1", "start_frame": 0, "end_frame": 0}
        ],
        sam3_frame_rows=[
            {
                "asset_id": "jdt__episode_1",
                "source_frame": 0,
                "frame_verdict": "fail",
            }
        ],
        native_check_rows=native_rows,
        standardized_check_rows=[],
        native_candidate_rows=[],
        standardized_candidate_rows=[],
        candidate_inputs_provided=False,
        thresholds={},
    )

    assert not bool(frame.loc[0, "native_temporal_only_flag"])
    assert bool(frame.loc[0, "native_hard_existence_morphology_flag"])
    assert bool(frame.loc[0, "native_any_precheck_flag"])


def test_dual_confusion_uses_same_frames_for_native_and_standardized() -> None:
    from tools.build_precheck_sam3_frame_comparison import (
        build_temporal_confusion_summary,
        build_temporal_frame_comparison,
    )

    frame = build_temporal_frame_comparison(
        manifest_rows=[
            {
                "asset_id": "jdt__episode_1",
                "start_frame": 0,
                "end_frame": 4,
            }
        ],
        sam3_frame_rows=_sam3_rows(),
        native_check_rows=_variant_rows(
            timebase="native",
            sampled_frames={0, 1, 2, 3, 4},
            positive_frames={0, 2},
        ),
        standardized_check_rows=_variant_rows(
            timebase="standardized",
            sampled_frames={0, 2, 4},
            positive_frames={0},
        ),
        native_candidate_rows=[],
        standardized_candidate_rows=[],
        candidate_inputs_provided=True,
        thresholds={},
    )
    summary = pd.DataFrame(build_temporal_confusion_summary(frame))
    temporal_broad = summary.loc[
        (summary["comparison_reference"] == "sam3_proxy_broad")
        & (summary["precheck_positive_definition"] == "temporal_only_flag")
    ]

    assert temporal_broad["evaluated_frame_count"].nunique() == 1
    assert set(temporal_broad["baseline"]) == {"native", "standardized"}
    assert int(temporal_broad.iloc[0]["evaluated_frame_count"]) == 2


def test_broad_and_strict_proxy_review_and_blocked_semantics() -> None:
    from tools.build_precheck_sam3_frame_comparison import (
        build_temporal_confusion_summary,
        build_temporal_frame_comparison,
    )

    rows = _variant_rows(
        timebase="native",
        sampled_frames={0, 1, 2, 3, 4},
        positive_frames={0, 2},
    )
    frame = build_temporal_frame_comparison(
        manifest_rows=[
            {
                "asset_id": "jdt__episode_1",
                "start_frame": 0,
                "end_frame": 4,
            }
        ],
        sam3_frame_rows=_sam3_rows(),
        native_check_rows=rows,
        standardized_check_rows=_variant_rows(
            timebase="standardized",
            sampled_frames={0, 1, 2, 3, 4},
            positive_frames={0, 2},
        ),
        native_candidate_rows=[],
        standardized_candidate_rows=[],
        candidate_inputs_provided=True,
        thresholds={},
    )
    summary = pd.DataFrame(build_temporal_confusion_summary(frame))
    broad = summary.loc[
        (summary["baseline"] == "native")
        & (summary["comparison_reference"] == "sam3_proxy_broad")
        & (summary["precheck_positive_definition"] == "temporal_only_flag")
    ].iloc[0]
    strict = summary.loc[
        (summary["baseline"] == "native")
        & (summary["comparison_reference"] == "sam3_proxy_strict")
        & (summary["precheck_positive_definition"] == "temporal_only_flag")
    ].iloc[0]

    assert (broad["TP"], broad["FN"], broad["FP"], broad["TN"]) == (1, 1, 1, 1)
    assert broad["unevaluable_count"] == 1
    assert broad["review_count"] == 1
    assert strict["review_count"] == 1
    assert strict["evaluated_frame_count"] == 3
    assert strict["unevaluable_count"] == 2
    assert strict["precision"] == pytest.approx(0.5)
    assert strict["recall"] == pytest.approx(1.0)


def test_confusion_zero_denominators_are_null() -> None:
    from tools.build_precheck_sam3_frame_comparison import (
        confusion_metrics,
    )

    metrics = confusion_metrics(
        visual_labels=[False, False],
        precheck_labels=[False, False],
        total_universe_count=2,
        review_count=0,
    )

    assert metrics["precision"] is None
    assert metrics["recall"] is None
    assert metrics["false_negative_rate"] is None
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["mcc"] is None


def test_optional_manual_labels_write_separate_confusion_output(
    tmp_path: Path,
) -> None:
    from tools.build_precheck_sam3_frame_comparison import (
        build_temporal_comparison_outputs,
    )

    rows = _variant_rows(
        timebase="native",
        sampled_frames={0, 1, 2, 3, 4},
        positive_frames={0, 2},
    )
    output_dir = tmp_path / "dual-comparison"
    build_temporal_comparison_outputs(
        manifest_rows=[
            {
                "asset_id": "jdt__episode_1",
                "start_frame": 0,
                "end_frame": 4,
            }
        ],
        sam3_frame_rows=_sam3_rows(),
        native_check_rows=rows,
        standardized_check_rows=rows,
        native_candidate_rows=[],
        standardized_candidate_rows=[],
        output_dir=output_dir,
        config_reference={"config_version": "test"},
        thresholds={},
        manual_labels={
            "segments": [
                {
                    "asset_id": "jdt__episode_1",
                    "start": 0,
                    "end": 0,
                    "label": "positive",
                },
                {
                    "asset_id": "jdt__episode_1",
                    "start": 1,
                    "end": 1,
                    "label": "acceptable_flagged",
                },
            ]
        },
        run_metadata={},
    )

    assert (output_dir / "temporal_precheck_vs_sam3_frame_comparison.csv").is_file()
    assert (output_dir / "temporal_precheck_vs_sam3_frame_comparison.json").is_file()
    assert (output_dir / "temporal_precheck_vs_sam3_frame_comparison.parquet").is_file()
    assert (output_dir / "temporal_precheck_vs_sam3_confusion_summary.csv").is_file()
    assert (output_dir / "temporal_precheck_vs_sam3_confusion_summary.json").is_file()
    assert (output_dir / "temporal_precheck_vs_sam3_confusion_summary.parquet").is_file()
    manual_path = output_dir / "temporal_precheck_vs_manual_confusion_summary.json"
    assert manual_path.is_file()
    sam3_summary = json.loads(
        (output_dir / "temporal_precheck_vs_sam3_confusion_summary.json").read_text()
    )
    manual_summary = json.loads(manual_path.read_text())
    assert {row["comparison_reference"] for row in sam3_summary} == {
        "sam3_proxy_broad",
        "sam3_proxy_strict",
    }
    assert {row["comparison_reference"] for row in manual_summary} == {
        "manual_ground_truth"
    }


def test_dual_cli_records_inputs_config_and_audit_only_thresholds(
    tmp_path: Path,
) -> None:
    from tools.build_precheck_sam3_frame_comparison import main

    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "asset_id": "jdt__episode_1",
                "start_frame": 0,
                "end_frame": 4,
            }
        ]
    ).to_csv(manifest, index=False)
    sam3 = tmp_path / "sam3.json"
    sam3.write_text(json.dumps(_sam3_rows()), encoding="utf-8")
    native = tmp_path / "native.json"
    native.write_text(
        json.dumps(
            _variant_rows(
                timebase="native",
                sampled_frames={0, 1, 2, 3, 4},
                positive_frames={0, 2},
            )
        ),
        encoding="utf-8",
    )
    standardized = tmp_path / "standardized.json"
    standardized.write_text(
        json.dumps(
            _variant_rows(
                timebase="standardized",
                sampled_frames={0, 2, 4},
                positive_frames={0},
            )
        ),
        encoding="utf-8",
    )
    native_candidates = tmp_path / "native-candidates.json"
    standardized_candidates = tmp_path / "standardized-candidates.json"
    native_candidates.write_text("[]", encoding="utf-8")
    standardized_candidates.write_text("[]", encoding="utf-8")
    output_dir = tmp_path / "cli-output"

    exit_code = main(
        [
            "--manifest",
            str(manifest),
            "--sam3-frame-results",
            str(sam3),
            "--native-precheck-results",
            str(native),
            "--standardized-precheck-results",
            str(standardized),
            "--native-candidate-windows",
            str(native_candidates),
            "--standardized-candidate-windows",
            str(standardized_candidates),
            "--finite-extreme-position-abs-m",
            "100",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    run_config = json.loads((output_dir / "run_config.json").read_text())
    assert run_config["config_reference"]["config_version"] == "qc_acceptance_v2.5.0"
    assert run_config["frame_universe_definition"] == (
        "manifest_inclusive_source_frames"
    )
    assert run_config["finite_extreme_audit_parameters"][
        "finite_extreme_position_abs_m"
    ] == {
        "value": 100.0,
        "source": "cli",
        "enabled": True,
        "affects_acceptance_decisions": False,
    }
    assert run_config["models_loaded"] == []
    assert run_config["mutates_precheck_outputs"] is False
    assert run_config["standardized_unsampled_source_frame_policy"] == (
        "unknown_not_clean_no_forward_fill"
    )
    assert run_config["input_metadata"]["input_paths"]["manifest"][0][
        "sha256"
    ].startswith("sha256:")
    assert run_config["input_metadata"]["consumed_artifacts"][
        "native_check_results"
    ][0]["sha256"].startswith("sha256:")
    assert run_config["input_metadata"]["variant_validation"]["native"][
        "status"
    ] == "verified_source_with_partial_artifact_lineage"


def test_variant_timebase_validation_rejects_swapped_artifacts() -> None:
    from tools.build_precheck_sam3_frame_comparison import (
        _validate_variant_timebase,
    )

    with pytest.raises(ValueError, match="native.*standardized_30hz"):
        _validate_variant_timebase(
            _variant_rows(
                timebase="standardized",
                sampled_frames={0, 2, 4},
                positive_frames={0},
            ),
            variant="native",
            expected_source="native_source_fps",
            producer_run_configs=[],
        )


def test_nested_pipeline_run_config_lineage_and_config_mismatch_guard(
    tmp_path: Path,
) -> None:
    from tools.build_precheck_sam3_frame_comparison import (
        _producer_run_config_records,
        _validate_config_compatibility,
        _validate_variant_timebase,
    )

    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    (artifact_root / "check_results.json").write_text("[]", encoding="utf-8")
    (artifact_root / "run_config.json").write_text(
        json.dumps(
            {
                "temporal_sampling": {
                    "decision_metric_source": "standardized_30hz",
                    "schema_version": "keypoint_temporal.output.v3",
                },
                "fingerprint": {
                    "temporal_output_schema_version": (
                        "keypoint_temporal.output.v3"
                    ),
                    "config": {"config_hash": "sha256:producer"},
                },
            }
        ),
        encoding="utf-8",
    )
    unrelated = artifact_root / "sam3"
    unrelated.mkdir()
    (unrelated / "run_config.json").write_text(
        json.dumps(
            {
                "temporal_sampling": {
                    "decision_metric_source": "native_source_fps"
                },
                "fingerprint": {
                    "config": {"config_hash": "sha256:unrelated"}
                },
            }
        ),
        encoding="utf-8",
    )

    records = _producer_run_config_records(
        [artifact_root],
        artifact_name="check_results.json",
    )

    assert len(records) == 1
    assert records[0]["decision_metric_source"] == "standardized_30hz"
    assert records[0]["temporal_output_schema_version"] == (
        "keypoint_temporal.output.v3"
    )
    assert records[0]["config_hashes"] == ["sha256:producer"]
    assert records[0]["run_config_missing"] is False
    validation = _validate_variant_timebase(
        [],
        variant="standardized",
        expected_source="standardized_30hz",
        producer_run_configs=records,
    )
    assert validation["status"] == "verified"
    assert validation["temporal_output_schema_status"] == "verified"
    with pytest.raises(ValueError, match="config hash mismatch"):
        _validate_config_compatibility(
            native_run_configs=records,
            standardized_run_configs=records,
            runtime_config_hash="sha256:runtime",
            allow_mismatch=False,
        )
    allowed = _validate_config_compatibility(
        native_run_configs=records,
        standardized_run_configs=records,
        runtime_config_hash="sha256:runtime",
        allow_mismatch=True,
    )
    assert allowed["mismatch_override_enabled"] is True
    assert allowed["native"]["status"] == "mismatch_reported_not_rewritten"

    missing_root = tmp_path / "missing-lineage"
    missing_root.mkdir()
    (missing_root / "check_results.json").write_text("[]", encoding="utf-8")
    missing_records = _producer_run_config_records(
        [missing_root],
        artifact_name="check_results.json",
    )
    partial = _validate_config_compatibility(
        native_run_configs=[*records, *missing_records],
        standardized_run_configs=records,
        runtime_config_hash="sha256:producer",
        allow_mismatch=False,
    )
    assert partial["native"]["status"] == (
        "partially_unverified_missing_config_hash"
    )
    assert partial["native"]["artifact_lineage_coverage"] == pytest.approx(0.5)
