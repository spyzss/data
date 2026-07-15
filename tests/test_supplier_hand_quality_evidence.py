from __future__ import annotations

from types import SimpleNamespace
import copy
import json

import numpy as np
import pytest
import h5py

from canonical_qc import StandardHdf5Adapter, StandardLeRobotAdapter
from canonical_qc.bridge import CanonicalQcBridge
from canonical_qc.hand_quality import compare_supplier_and_machine
from qc_common.config import LoadedQcConfig
from qc_common.report_mutation import apply_module_result
from qc_pipeline.runners.precheck import runner_for
from qc_pipeline.runners.sam3_containment import runner as sam3_runner
from qc_pipeline.adapters.sam3_containment import adapt_sam3_containment
from tests.fixtures import (
    write_standard_hdf5_episode,
    write_standard_lerobot_dataset,
)
from tests.qc_report_fixtures import loaded_test_config


def test_unknown_supplier_status_is_excluded_from_agreement_and_never_fails() -> None:
    result = compare_supplier_and_machine(
        np.array([["unknown", "unknown"]], dtype="U7"),
        np.array([["pass", "fail"]], dtype="U4"),
    )

    assert result.observation_count == 2
    assert result.comparable_count == 0
    assert result.agreement_count == 0
    assert result.agreement_rate is None
    assert result.supplier_unknown_count == 2
    assert result.supplier_false_negative_count == 0
    assert result.supplier_false_positive_count == 0
    assert result.issue_codes == ()


def test_good_machine_fail_warns_and_bad_machine_pass_is_false_positive() -> None:
    result = compare_supplier_and_machine(
        np.array([["good", "bad"]], dtype="U4"),
        np.array([["fail", "pass"]], dtype="U4"),
    )

    assert result.comparable_count == 2
    assert result.agreement_count == 0
    assert result.agreement_rate == 0.0
    assert result.supplier_false_negative_count == 1
    assert result.supplier_false_positive_count == 1
    assert result.supplier_false_negative_numerator == 1
    assert result.supplier_false_negative_denominator == 1
    assert result.supplier_false_positive_numerator == 1
    assert result.supplier_false_positive_denominator == 1
    assert result.issue_codes == ("supplier_mask_disagreement",)


def test_confusion_counts_and_exclusions_are_supplier_perspective() -> None:
    result = compare_supplier_and_machine(
        np.array(
            [
                "good",
                "good",
                "bad",
                "bad",
                "unknown",
                "warning",
                "good",
                "bad",
                "good",
            ]
        ),
        np.array(
            [
                "pass",
                "fail",
                "pass",
                "fail",
                "pass",
                "fail",
                "review",
                "unavailable",
                "skipped",
            ]
        ),
    )

    assert result.true_negative_count == 1
    assert result.supplier_false_negative_count == 1
    assert result.supplier_false_positive_count == 1
    assert result.true_positive_count == 1
    assert result.comparable_count == 4
    assert result.agreement_count == 2
    assert result.agreement_rate == 0.5
    assert result.supplier_unknown_count == 1
    assert result.supplier_warning_count == 1
    assert result.machine_review_count == 1
    assert result.machine_unavailable_count == 1
    assert result.machine_skipped_count == 1


def test_supplier_status_never_guesses_from_raw_numeric_values() -> None:
    with pytest.raises(ValueError, match="supplier status"):
        compare_supplier_and_machine(
            np.array([[0.0, 1.0]], dtype=np.float32),
            np.array([["fail", "pass"]], dtype="U4"),
        )


def test_supplier_and_machine_status_shapes_must_match_exactly() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        compare_supplier_and_machine(
            np.array([["good", "bad"]]),
            np.array(["pass", "fail"]),
        )


def test_sam3_adapter_adds_supplier_disagreement_warn_without_overriding_machine_fail(
    tmp_path,
) -> None:
    result = adapt_sam3_containment(
        asset_id="a",
        batch_root=tmp_path,
        window_summaries=[
            {
                "asset_id": "a",
                "window_start_frame": 0,
                "window_end_frame": 0,
                "hand_side": "left",
                "window_containment_verdict": "strong_containment_mismatch",
                "inside_ratio_mean": 0.1,
            }
        ],
        evidence_rows=[],
        config=loaded_test_config(),
        frame_rows=[
            {
                "asset_id": "a",
                "frame_idx": 0,
                "hand_side": "left",
                "camera_id": "main",
                "containment_verdict": "strong_containment_mismatch",
            },
            {
                "asset_id": "a",
                "frame_idx": 0,
                "hand_side": "left",
                "camera_id": "main",
                "containment_verdict": "strong_containment_mismatch",
            },
        ],
        supplier_hand_quality_status=np.array(
            [["good", "unknown"]],
            dtype="U7",
        ),
    )

    assert result.verdict == "fail"
    disagreement = [
        issue for issue in result.issues if issue.code == "supplier_mask_disagreement"
    ]
    assert len(disagreement) == 1
    assert disagreement[0].severity == "warn"
    assert disagreement[0].needs_manual_review is True
    assert result.metrics["supplier_hand_quality"] == {
        "provided": True,
        "observation_count": 1,
        "comparable_count": 1,
        "agreement_count": 0,
        "disagreement_count": 1,
        "agreement_rate": 0.0,
        "true_positive_count": 0,
        "true_negative_count": 0,
        "supplier_false_positive_count": 0,
        "supplier_false_positive_numerator": 0,
        "supplier_false_positive_denominator": 0,
        "supplier_false_positive_rate": None,
        "supplier_false_negative_count": 1,
        "supplier_false_negative_numerator": 1,
        "supplier_false_negative_denominator": 1,
        "supplier_false_negative_rate": 1.0,
        "supplier_unknown_count": 0,
        "supplier_warning_count": 0,
        "machine_unavailable_count": 0,
        "machine_review_count": 0,
        "machine_skipped_count": 0,
    }


def test_sam3_adapter_records_supplier_false_positive_without_quality_issue(
    tmp_path,
) -> None:
    result = adapt_sam3_containment(
        asset_id="a",
        batch_root=tmp_path,
        window_summaries=[
            {
                "asset_id": "a",
                "window_start_frame": 0,
                "window_end_frame": 0,
                "hand_side": "right",
                "window_containment_verdict": "pass",
                "inside_ratio_mean": 0.95,
            }
        ],
        evidence_rows=[],
        config=loaded_test_config(),
        frame_rows=[
            {
                "asset_id": "a",
                "frame_idx": 0,
                "hand_side": "right",
                "camera_id": "main",
                "containment_verdict": "likely_visible_ok",
            }
        ],
        supplier_hand_quality_status=np.array(
            [["unknown", "bad"]],
            dtype="U7",
        ),
    )

    assert result.verdict == "pass"
    assert result.issues == ()
    assert (
        result.metrics["supplier_hand_quality"][
            "supplier_false_positive_count"
        ]
        == 1
    )
    assert result.metrics["supplier_hand_quality"]["agreement_rate"] == 0.0


def test_sam3_adapter_missing_supplier_evidence_is_cleanly_not_provided(
    tmp_path,
) -> None:
    result = adapt_sam3_containment(
        asset_id="a",
        batch_root=tmp_path,
        window_summaries=[
            {
                "asset_id": "a",
                "window_start_frame": 0,
                "window_end_frame": 0,
                "hand_side": "left",
                "window_containment_verdict": "pass",
            }
        ],
        evidence_rows=[],
        config=loaded_test_config(),
        frame_rows=[],
        supplier_hand_quality_status=None,
    )

    assert result.verdict == "pass"
    assert result.metrics["supplier_hand_quality"] == {"provided": False}


def _load_explicitly_unprovided_episode(tmp_path, source_format: str):
    if source_format == "hdf5":
        source_root = tmp_path / source_format / "asset-001"
        write_standard_hdf5_episode(source_root, hand_quality="false")
        return source_root, StandardHdf5Adapter().load(source_root)

    source_root = tmp_path / source_format
    write_standard_lerobot_dataset(source_root)
    semantics_path = source_root / "meta" / "episode_semantics.jsonl"
    rows = [
        json.loads(line)
        for line in semantics_path.read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        row["supplier_hand_quality"] = {"provided": False}
    semantics_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return source_root, StandardLeRobotAdapter().load(source_root)


@pytest.mark.parametrize("source_format", ["hdf5", "lerobot"])
def test_explicitly_unprovided_supplier_evidence_cleanly_skips_across_qc(
    tmp_path,
    source_format: str,
) -> None:
    source_root, episode = _load_explicitly_unprovided_episode(
        tmp_path,
        source_format,
    )
    bridge = CanonicalQcBridge(episode, source_root=source_root)
    candidates = tmp_path / f"{source_format}-candidate-windows.json"
    candidates.write_text(
        json.dumps(
            [
                {
                    "asset_id": episode.identity.asset_id,
                    "start_frame": 0,
                    "end_frame": 0,
                    "hand_side": "both",
                }
            ]
        ),
        encoding="utf-8",
    )
    context = bridge.asset_context(
        batch_root=tmp_path,
        report_path=tmp_path / "quality_archive" / f"{source_format}.json",
        supplemental_source_files={
            "candidate_windows": {"path": candidates.name}
        },
    )

    clip = bridge.clip_inputs()
    quality_result = runner_for("quality_hand")(context, loaded_test_config())

    class Segmenter:
        def segment_frame(self, frame, queries, config):
            return [
                SimpleNamespace(
                    mask=np.ones(frame.shape[:2], dtype=bool),
                    category="hand",
                )
            ]

    sam3_result = sam3_runner(lambda: Segmenter())(
        context,
        loaded_test_config(),
    )

    assert episode.supplier_evidence.hand_quality is not None
    assert episode.supplier_evidence.hand_quality.provided is False
    assert clip.supplier_hand_quality_status is None
    assert quality_result.verdict == "skipped"
    assert quality_result.evaluation == {
        "decision": "skipped",
        "reason": "source_signal_not_provided",
    }
    assert sam3_result.metrics["supplier_hand_quality"] == {"provided": False}


def test_sam3_supplier_alignment_rejects_both_as_a_frame_hand_side(tmp_path) -> None:
    with pytest.raises(ValueError, match="frame hand_side"):
        adapt_sam3_containment(
            asset_id="a",
            batch_root=tmp_path,
            window_summaries=[],
            evidence_rows=[],
            config=loaded_test_config(),
            frame_rows=[
                {
                    "asset_id": "a",
                    "frame_idx": 0,
                    "hand_side": "both",
                    "camera_id": "main",
                    "containment_verdict": "likely_visible_ok",
                }
            ],
            supplier_hand_quality_status=np.array(
                [["good", "good"]],
                dtype="U4",
            ),
        )


def test_sam3_supplier_alignment_rejects_conflicting_overlapping_frame_rows(
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="conflicting SAM3 frame observations"):
        adapt_sam3_containment(
            asset_id="a",
            batch_root=tmp_path,
            window_summaries=[],
            evidence_rows=[],
            config=loaded_test_config(),
            frame_rows=[
                {
                    "asset_id": "a",
                    "frame_idx": 0,
                    "hand_side": "left",
                    "camera_id": "main",
                    "containment_verdict": "likely_visible_ok",
                },
                {
                    "asset_id": "a",
                    "frame_idx": 0,
                    "hand_side": "left",
                    "camera_id": "main",
                    "containment_verdict": "strong_containment_mismatch",
                },
            ],
            supplier_hand_quality_status=np.array(
                [["good", "unknown"]],
                dtype="U7",
            ),
        )


def test_canonical_quality_hand_records_enum_evidence_without_gating(tmp_path) -> None:
    source_root = tmp_path / "asset-001"
    write_standard_hdf5_episode(source_root, hand_quality="provided")
    episode = StandardHdf5Adapter().load(source_root)
    context = CanonicalQcBridge(episode, source_root=source_root).asset_context(
        batch_root=tmp_path,
        report_path=tmp_path / "quality_archive" / "asset-001.json",
    )

    result = runner_for("quality_hand")(context, loaded_test_config())

    assert result.verdict == "pass"
    assert result.issues == ()
    assert result.metrics["provided"] is True
    assert result.metrics["status_counts"] == {
        "bad": 1,
        "good": 2,
        "unknown": 2,
        "warning": 1,
    }


def test_canonical_keypoint_presence_uses_validity_and_finite_3d_points(
    tmp_path,
) -> None:
    source_root = tmp_path / "asset-001"
    hdf5_path, _video = write_standard_hdf5_episode(source_root)
    with h5py.File(hdf5_path, "r+") as handle:
        handle["/observation/hand_joint_valid_3d"][0, 0, :] = False
        handle["/observation/hand_keypoints_3d"][0, 0, :, :] = np.nan
    episode = StandardHdf5Adapter().load(source_root)
    context = CanonicalQcBridge(episode, source_root=source_root).asset_context(
        batch_root=tmp_path,
        report_path=tmp_path / "quality_archive" / "asset-001.json",
    )

    result = runner_for("keypoint_presence")(context, loaded_test_config())

    assert result.verdict == "fail"
    assert result.evaluation["checked_frame_count"] == 3
    assert result.metrics["min_valid_keypoint_count_left"] == 0.0
    assert result.issues
    assert result.issues[0].context["hand_side"] == "left"


def test_canonical_sam3_runner_persists_frame_aligned_supplier_statistics(
    tmp_path,
) -> None:
    source_root = tmp_path / "asset-001"
    write_standard_hdf5_episode(source_root, hand_quality="provided")
    episode = StandardHdf5Adapter().load(source_root)
    candidates = tmp_path / "candidate-windows.json"
    candidates.write_text(
        '[{"asset_id":"asset-001","start_frame":0,"end_frame":0,'
        '"hand_side":"both"}]',
        encoding="utf-8",
    )
    context = CanonicalQcBridge(episode, source_root=source_root).asset_context(
        batch_root=tmp_path,
        report_path=tmp_path / "quality_archive" / "asset-001.json",
        supplemental_source_files={
            "candidate_windows": {"path": candidates.name}
        },
    )

    class Segmenter:
        def segment_frame(self, frame, queries, config):
            return [
                SimpleNamespace(
                    mask=np.ones(frame.shape[:2], dtype=bool),
                    category="hand",
                )
            ]

    result = sam3_runner(lambda: Segmenter())(
        context,
        loaded_test_config(),
    )

    metrics = result.metrics["supplier_hand_quality"]
    assert metrics["provided"] is True
    assert metrics["observation_count"] == 2
    assert metrics["comparable_count"] == 1
    assert metrics["supplier_false_positive_count"] == 1
    assert metrics["supplier_unknown_count"] == 1

    loaded = loaded_test_config()
    raw = copy.deepcopy(loaded.raw)
    raw["pipeline"]["modules"] = ["sam3_containment"]
    config = LoadedQcConfig(path=loaded.path, raw=raw, sha256=loaded.sha256)
    report = apply_module_result(
        context.report_path,
        context=context,
        config=config,
        profile="supplier_evaluation",
        result=result,
        expected_revision=0,
        next_module=None,
        now="2026-07-15T00:00:00Z",
    )

    assert report["sam3_containment"]["metrics"]["supplier_hand_quality"] == metrics
    assert report["report_revision"] == 1
