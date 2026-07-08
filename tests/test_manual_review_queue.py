import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from tools.build_batch_qc_ledger import load_manifest, read_records
from tools.build_manual_review_queue import (
    FAILURE_MODE_ENUM,
    MANUAL_OUTCOME_ENUM,
    MANUAL_TEMPLATE_COLUMNS,
    REVIEW_QUEUE_COLUMNS,
    assign_review_ids,
    build_review_index_html,
    copy_selected_overlays,
    manual_template_row,
    queue_row,
    rows_from_candidate_windows,
    rows_from_issue_events,
    rows_from_sam3_summary,
    select_review_rows,
)
from tools.convert_manual_labels_csv_to_json import convert_csv_to_patch_records


def _write_csv(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_help_includes_issue_events_argument() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "tools/build_manual_review_queue.py", "--help"],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "--issue-events" in result.stdout


def test_manual_review_queue_uses_fixed_fields_and_enums(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_csv(
        manifest,
        "supplier_id,asset_id,episode_idx\n"
        "supplier_a,100030,0\n"
        "supplier_a,100044,1\n"
        "supplier_a,100560,2\n",
    )
    assets, episode_to_asset = load_manifest(manifest)

    overlay_dir = tmp_path / "overlays"
    overlay_dir.mkdir()
    (overlay_dir / "100044_100_120.png").write_bytes(b"not-really-a-png")

    candidate_path = tmp_path / "candidate_windows.json"
    candidate_rows = [
        {
            "asset_id": "100044",
            "start_frame": 100,
            "end_frame": 120,
            "peak_frame": 111,
            "review_type": ["side_view_manual_review"],
            "trigger_reason": ["side_view_hand_orientation"],
            "priority": "medium",
            "trigger_metrics": {"palm_camera_angle_deg_max": 88.0},
            "needs_manual_review": True,
            "sam3_containment_eligible": False,
        }
    ]
    sam3_path = tmp_path / "sam3_summary.json"
    sam3_rows = [
        {
            "asset_id": "100030",
            "window_start_frame": 464,
            "window_end_frame": 502,
            "window_containment_verdict": "containment_fail",
            "strong_fail_frame_count": 5,
            "inside_ratio_mean": 0.04,
        }
    ]

    rows = []
    rows.extend(
        rows_from_candidate_windows(
            candidate_rows,
            candidate_path,
            assets,
            episode_to_asset,
            overlay_dir,
        )
    )
    rows.extend(
        rows_from_sam3_summary(
            sam3_rows,
            sam3_path,
            assets,
            episode_to_asset,
            overlay_dir,
        )
    )
    selected = assign_review_ids(
        select_review_rows(
            rows,
            assets,
            max_items_per_supplier=10,
            max_side_view_per_supplier=1,
            max_pass_samples_per_supplier=1,
            overlay_dir=overlay_dir,
        )
    )
    original_overlay = next(
        row["overlay_path"]
        for row in selected
        if row["asset_id"] == "100044"
    )
    output_dir = tmp_path / "review"
    copy_selected_overlays(selected, output_dir)

    queue_df = pd.DataFrame(selected, columns=REVIEW_QUEUE_COLUMNS)
    template_df = pd.DataFrame(
        [manual_template_row(row) for row in selected],
        columns=MANUAL_TEMPLATE_COLUMNS,
    )
    html = build_review_index_html(selected)

    assert list(queue_df.columns) == REVIEW_QUEUE_COLUMNS
    assert list(template_df.columns) == MANUAL_TEMPLATE_COLUMNS
    assert "side_view_mask_undersegmentation" in queue_df["suggested_issue_type"].tolist()
    assert "strong_containment_mismatch" in queue_df["suggested_issue_type"].tolist()
    assert "pass_sample" in queue_df["auto_verdict"].tolist()
    assert set(template_df["failure_mode"]).issubset(set(FAILURE_MODE_ENUM))
    assert "manual_outcome" in html
    assert "Auto result" in html
    assert "Human label" in html
    assert "manual_outcome</b> = whether the script flag is correct" in html
    assert "failure_mode</b> = human-confirmed issue type" in html
    assert "severity</b> = impact on sample quality" in html
    assert "confidence</b> = confidence in the human label" in html
    assert "comment optional" in html
    assert "true_positive" in html
    assert "keypoint_raw_invalid" in html
    assert "keypoint_low_quality_window" in html
    assert "visual_skeleton_presence_mismatch" in html
    assert "Export manual_labels.csv" in html
    assert "function exportManualLabelsCsv" in html
    assert "localStorage" in html
    assert "manual_labels_template.csv" in html
    assert "assets/overlays/100044_100_120.png" in html
    for column in (
        "auto_verdict",
        "suggested_issue_type",
        "severity_suggestion",
        "key_metrics_json",
        "reason",
        "manual_outcome",
        "failure_mode",
        "severity",
        "confidence",
        "comment",
        "reviewer",
    ):
        assert column in MANUAL_TEMPLATE_COLUMNS
        assert column in html
    assert (output_dir / "assets" / "overlays" / "100044_100_120.png").exists()
    assert next(row for row in selected if row["asset_id"] == "100044")[
        "overlay_path"
    ] == original_overlay
    assert next(row for row in selected if row["asset_id"] == "100044")[
        "display_overlay_path"
    ] == "assets/overlays/100044_100_120.png"


def test_issue_events_add_asset_level_keypoint_low_quality_item(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_csv(
        manifest,
        "supplier_id,asset_id,episode_idx\n"
        "supplier_a,100030,0\n",
    )
    assets, episode_to_asset = load_manifest(manifest)
    issue_events_path = tmp_path / "issue_events.csv"
    pd.DataFrame(
        [
            {
                "supplier_id": "supplier_a",
                "asset_id": "100030",
                "module": "precheck",
                "issue_type": "keypoint_low_quality_window",
                "severity": "medium",
                "auto_verdict": "review",
                "manual_outcome": "",
                "window_start_frame": "",
                "window_end_frame": "",
                "metric_name": "flagged_frames",
                "metric_value": 4,
                "reason": "keypoint low-quality window exceeded aggregate threshold",
                "evidence_path": str(tmp_path / "clip_aggregates.csv"),
                "needs_manual_review": True,
                "sam3_containment_eligible": "",
            }
        ]
    ).to_csv(issue_events_path, index=False)

    rows = rows_from_issue_events(
        read_records(issue_events_path),
        issue_events_path,
        assets,
        episode_to_asset,
        overlay_dir=None,
    )
    selected = assign_review_ids(
        select_review_rows(
            rows,
            assets,
            max_items_per_supplier=10,
            max_side_view_per_supplier=10,
            max_pass_samples_per_supplier=0,
            overlay_dir=None,
        )
    )
    item = next(
        row
        for row in selected
        if row["suggested_issue_type"] == "keypoint_low_quality_window"
    )

    assert item["source_level"] == "asset"
    assert item["overlay_path"] == ""
    assert item["display_overlay_path"] == ""
    assert "flagged_frames" in item["key_metrics_json"]
    assert "No overlay" in build_review_index_html(selected)


def test_convert_completed_manual_csv_to_patch_records(tmp_path: Path) -> None:
    csv_path = tmp_path / "manual_labels_template.csv"
    row = {column: "" for column in MANUAL_TEMPLATE_COLUMNS}
    row.update(
        {
            "review_id": "supplier_a_100044_100_120_0001",
            "supplier_id": "supplier_a",
            "asset_id": "100044",
            "window_start_frame": 100,
            "window_end_frame": 120,
            "representative_frame": 111,
            "auto_verdict": "review",
            "suggested_issue_type": "side_view_mask_undersegmentation",
            "severity_suggestion": "medium",
            "key_metrics_json": "{\"inside_ratio_mean\": 0.1}",
            "reason": "side-view hand orientation makes SAM3 containment unreliable",
            "manual_outcome": "true_positive",
            "failure_mode": "side_view_mask_undersegmentation",
            "severity": "medium",
            "confidence": "high",
            "comment": "side view mask undersegmentation",
            "reviewer": "nathan",
        }
    )
    pd.DataFrame([row], columns=MANUAL_TEMPLATE_COLUMNS).to_csv(csv_path, index=False)

    records = convert_csv_to_patch_records(csv_path)

    assert len(records) == 1
    record = records[0]
    assert record["source"] == "manual_review_queue"
    assert record["review_id"] == "supplier_a_100044_100_120_0001"
    assert record["asset_id"] == "100044"
    assert record["start"] == 100
    assert record["end"] == 120
    assert record["label"] == "positive"
    assert record["algorithm_outcome"] == "true_positive"
    assert record["manual_outcome"] in MANUAL_OUTCOME_ENUM
    assert record["failure_mode"] == "side_view_mask_undersegmentation"
    assert record["severity_suggestion"] == "medium"
    assert record["key_metrics_json"] == "{\"inside_ratio_mean\": 0.1}"
    assert record["reason"] == "side-view hand orientation makes SAM3 containment unreliable"


def test_convert_manual_csv_preserves_multiple_segments_per_review_id(tmp_path: Path) -> None:
    csv_path = tmp_path / "manual_labels.csv"
    columns = [
        "review_id",
        "segment_id",
        "supplier_id",
        "asset_id",
        "window_start_frame",
        "window_end_frame",
        "representative_frame",
        "affected_start_frame",
        "affected_end_frame",
        "auto_verdict",
        "suggested_issue_type",
        "severity_suggestion",
        "key_metrics_json",
        "reason",
        "manual_outcome",
        "failure_mode",
        "severity",
        "confidence",
        "acceptance_status",
        "reviewer",
        "comment",
    ]
    rows = [
        {
            "review_id": "rq_001",
            "segment_id": "rq_001_seg_001",
            "supplier_id": "supplier_a",
            "asset_id": "100044",
            "window_start_frame": 1,
            "window_end_frame": 100,
            "representative_frame": 50,
            "affected_start_frame": 1,
            "affected_end_frame": 10,
            "auto_verdict": "review",
            "suggested_issue_type": "side_view_mask_undersegmentation",
            "severity_suggestion": "medium",
            "key_metrics_json": "{}",
            "reason": "large candidate window",
            "manual_outcome": "partial",
            "failure_mode": "side_view_mask_undersegmentation",
            "severity": "medium",
            "confidence": "high",
            "acceptance_status": "rejected",
            "reviewer": "nathan",
            "comment": "first bad segment",
        },
        {
            "review_id": "rq_001",
            "segment_id": "rq_001_seg_002",
            "supplier_id": "supplier_a",
            "asset_id": "100044",
            "window_start_frame": 1,
            "window_end_frame": 100,
            "representative_frame": 50,
            "affected_start_frame": 25,
            "affected_end_frame": 88,
            "auto_verdict": "review",
            "suggested_issue_type": "side_view_mask_undersegmentation",
            "severity_suggestion": "medium",
            "key_metrics_json": "{}",
            "reason": "large candidate window",
            "manual_outcome": "partial",
            "failure_mode": "side_view_mask_undersegmentation",
            "severity": "high",
            "confidence": "medium",
            "acceptance_status": "review",
            "reviewer": "nathan",
            "comment": "second bad segment",
        },
    ]
    pd.DataFrame(rows, columns=columns).to_csv(csv_path, index=False)

    records = convert_csv_to_patch_records(csv_path)

    assert len(records) == 2
    assert [record["review_id"] for record in records] == ["rq_001", "rq_001"]
    assert [record["segment_id"] for record in records] == ["rq_001_seg_001", "rq_001_seg_002"]
    assert records[0]["start"] == 1
    assert records[0]["end"] == 10
    assert records[0]["affected_start_frame"] == 1
    assert records[0]["affected_end_frame"] == 10
    assert records[0]["window_start_frame"] == 1
    assert records[0]["window_end_frame"] == 100
    assert records[0]["acceptance_status"] == "rejected"
    assert records[1]["start"] == 25
    assert records[1]["end"] == 88
    assert records[1]["acceptance_status"] == "review"


def test_convert_manual_csv_preserves_false_positive_without_affected_frames(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "manual_labels.csv"
    columns = [
        "review_id",
        "segment_id",
        "supplier_id",
        "asset_id",
        "window_start_frame",
        "window_end_frame",
        "representative_frame",
        "affected_start_frame",
        "affected_end_frame",
        "auto_verdict",
        "suggested_issue_type",
        "severity_suggestion",
        "key_metrics_json",
        "reason",
        "manual_outcome",
        "failure_mode",
        "severity",
        "confidence",
        "acceptance_status",
        "reviewer",
        "comment",
    ]
    row = {
        "review_id": "rq_002",
        "segment_id": "rq_002_false_positive",
        "supplier_id": "supplier_a",
        "asset_id": "100044",
        "window_start_frame": 1,
        "window_end_frame": 100,
        "representative_frame": 50,
        "affected_start_frame": "",
        "affected_end_frame": "",
        "auto_verdict": "review",
        "suggested_issue_type": "side_view_mask_undersegmentation",
        "severity_suggestion": "medium",
        "key_metrics_json": "{}",
        "reason": "large candidate window",
        "manual_outcome": "false_positive",
        "failure_mode": "acceptable_minor_misalignment",
        "severity": "low",
        "confidence": "high",
        "acceptance_status": "accepted",
        "reviewer": "nathan",
        "comment": "not actually bad",
    }
    pd.DataFrame([row], columns=columns).to_csv(csv_path, index=False)

    records = convert_csv_to_patch_records(csv_path)

    assert len(records) == 1
    assert records[0]["start"] is None
    assert records[0]["end"] is None
    assert records[0]["affected_start_frame"] is None
    assert records[0]["affected_end_frame"] is None
    assert records[0]["window_start_frame"] == 1
    assert records[0]["window_end_frame"] == 100
    assert records[0]["acceptance_status"] == "accepted"


def test_convert_manual_csv_preserves_whole_window_affected_segment(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "manual_labels.csv"
    columns = [
        "review_id",
        "segment_id",
        "supplier_id",
        "asset_id",
        "window_start_frame",
        "window_end_frame",
        "representative_frame",
        "affected_start_frame",
        "affected_end_frame",
        "auto_verdict",
        "suggested_issue_type",
        "severity_suggestion",
        "key_metrics_json",
        "reason",
        "manual_outcome",
        "failure_mode",
        "severity",
        "confidence",
        "acceptance_status",
        "reviewer",
        "comment",
    ]
    row = {
        "review_id": "rq_003",
        "segment_id": "rq_003_seg_001",
        "supplier_id": "supplier_a",
        "asset_id": "100044",
        "window_start_frame": 1,
        "window_end_frame": 100,
        "representative_frame": 50,
        "affected_start_frame": 1,
        "affected_end_frame": 100,
        "auto_verdict": "review",
        "suggested_issue_type": "side_view_mask_undersegmentation",
        "severity_suggestion": "medium",
        "key_metrics_json": "{}",
        "reason": "large candidate window",
        "manual_outcome": "true_positive",
        "failure_mode": "side_view_mask_undersegmentation",
        "severity": "high",
        "confidence": "high",
        "acceptance_status": "rejected",
        "reviewer": "nathan",
        "comment": "whole window is bad",
    }
    pd.DataFrame([row], columns=columns).to_csv(csv_path, index=False)

    records = convert_csv_to_patch_records(csv_path)

    assert len(records) == 1
    assert records[0]["segment_id"] == "rq_003_seg_001"
    assert records[0]["start"] == 1
    assert records[0]["end"] == 100
    assert records[0]["affected_start_frame"] == 1
    assert records[0]["affected_end_frame"] == 100
    assert records[0]["window_start_frame"] == 1
    assert records[0]["window_end_frame"] == 100
    assert records[0]["acceptance_status"] == "rejected"


def test_review_queue_selection_caps_side_view_and_keeps_other_issue_types() -> None:
    assets = {
        "asset_side": {"supplier_id": "supplier_a", "asset_id": "asset_side"},
        "asset_strong": {"supplier_id": "supplier_a", "asset_id": "asset_strong"},
        "asset_mixed": {"supplier_id": "supplier_a", "asset_id": "asset_mixed"},
        "asset_quality": {"supplier_id": "supplier_a", "asset_id": "asset_quality"},
    }
    rows = []
    for index in range(20):
        rows.append(
            queue_row(
                supplier_id="supplier_a",
                asset_id=f"asset_side_{index}",
                start=index,
                end=index + 1,
                representative=index,
                source_level="window",
                module="precheck",
                auto_verdict="review",
                suggested_issue_type="side_view_mask_undersegmentation",
                severity_suggestion="medium",
                priority="medium",
                key_metrics={},
                reason="side view",
                evidence_path=None,
                overlay_path=None,
                needs_manual_review=True,
                sam3_containment_eligible=False,
            )
        )
    rows.extend(
        [
            queue_row(
                supplier_id="supplier_a",
                asset_id="asset_mixed",
                start=100,
                end=110,
                representative=105,
                source_level="window",
                module="sam3_containment",
                auto_verdict="mixed_review",
                suggested_issue_type="strong_containment_mismatch",
                severity_suggestion="high",
                priority="high",
                key_metrics={"strong_fail_frame_count": 2},
                reason="mixed strong",
                evidence_path=None,
                overlay_path=None,
                needs_manual_review=True,
                sam3_containment_eligible=True,
            ),
            queue_row(
                supplier_id="supplier_a",
                asset_id="asset_strong",
                start=120,
                end=130,
                representative=125,
                source_level="window",
                module="sam3_containment",
                auto_verdict="containment_fail",
                suggested_issue_type="strong_containment_mismatch",
                severity_suggestion="high",
                priority="high",
                key_metrics={"strong_fail_frame_count": 5},
                reason="strong containment",
                evidence_path=None,
                overlay_path=None,
                needs_manual_review=True,
                sam3_containment_eligible=True,
            ),
            queue_row(
                supplier_id="supplier_a",
                asset_id="asset_quality",
                start=None,
                end=None,
                representative=None,
                source_level="asset",
                module="precheck",
                auto_verdict="review",
                suggested_issue_type="keypoint_low_quality_window",
                severity_suggestion="medium",
                priority="medium",
                key_metrics={"flagged_frames": 3},
                reason="low quality window",
                evidence_path=None,
                overlay_path=None,
                needs_manual_review=True,
                sam3_containment_eligible=None,
            ),
        ]
    )

    selected = select_review_rows(
        rows,
        assets,
        max_items_per_supplier=30,
        max_side_view_per_supplier=10,
        max_pass_samples_per_supplier=2,
        overlay_dir=None,
    )

    issue_counts = pd.Series([row["suggested_issue_type"] for row in selected]).value_counts()
    assert issue_counts["side_view_mask_undersegmentation"] == 10
    assert "keypoint_low_quality_window" in issue_counts
    assert any(row["auto_verdict"] == "mixed_review" for row in selected)
    assert any(row["auto_verdict"] == "containment_fail" for row in selected)


def test_convert_manual_csv_rejects_invalid_enum(tmp_path: Path) -> None:
    csv_path = tmp_path / "manual_labels_template.csv"
    row = {column: "" for column in MANUAL_TEMPLATE_COLUMNS}
    row.update(
        {
            "review_id": "r1",
            "supplier_id": "supplier_a",
            "asset_id": "100044",
            "window_start_frame": 100,
            "window_end_frame": 120,
            "representative_frame": 111,
            "auto_verdict": "review",
            "suggested_issue_type": "unknown",
            "severity_suggestion": "medium",
            "key_metrics_json": "{}",
            "reason": "test row",
            "manual_outcome": "not_an_enum",
            "failure_mode": "unknown",
            "severity": "medium",
            "confidence": "high",
            "reviewer": "nathan",
        }
    )
    pd.DataFrame([row], columns=MANUAL_TEMPLATE_COLUMNS).to_csv(csv_path, index=False)

    try:
        convert_csv_to_patch_records(csv_path)
    except ValueError as exc:
        assert "manual_outcome" in str(exc)
    else:
        raise AssertionError("invalid manual_outcome should fail")
