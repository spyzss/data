import csv
import json
from pathlib import Path

from openpyxl import load_workbook

from tools.build_weekly_supplier_acceptance_report import (
    DETAIL_COLUMNS,
    SUMMARY_COLUMNS,
    build_weekly_report,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _prepare_inputs(run_root: Path) -> None:
    _write_csv(
        run_root / "manifests" / "supplier_manifest_xjgt_100.csv",
        [
            {
                "supplier_id": "xjgt",
                "asset_id": "1001",
                "frame_count": 100,
            },
            {
                "supplier_id": "xjgt",
                "asset_id": "1002",
                "frame_count": 200,
            },
        ],
    )
    _write_csv(
        run_root / "xjgt" / "ledger" / "xjgt_100_asset_ledger.csv",
        [
            {
                "asset_id": "1001",
                "hdf5_text_status": "pass",
                "keypoint_missing_status": "pass",
                "keypoint_morphology_status": "pass",
                "video_quality_status": "pass",
                "temporal_status": "pass",
                "sam3_evidence_status": "pass",
                "manual_review_status": "pass_with_notes",
                "final_verdict": "pass_with_notes",
                "top_issue_types": "acceptable_minor_misalignment",
                "notes": "minor issue accepted",
                "evidence_paths": "manual_labels_patch.json",
            },
            {
                "asset_id": "1002",
                "hdf5_text_status": "pass",
                "keypoint_missing_status": "pass",
                "keypoint_morphology_status": "pass",
                "video_quality_status": "fail",
                "temporal_status": "review",
                "sam3_evidence_status": "review",
                "manual_review_status": "fail",
                "final_verdict": "fail",
                "top_issue_types": "hand_out_of_frame",
                "notes": "confirmed bad segment",
                "evidence_paths": "manual_labels_patch.json",
            },
        ],
    )
    _write_csv(
        run_root / "xjgt" / "ledger" / "xjgt_100_issue_events.csv",
        [
            {
                "asset_id": "1002",
                "source_module": "manual_review",
                "manual_outcome": "true_positive",
                "failure_mode": "hand_out_of_frame",
                "start_frame": 10,
                "end_frame": 109,
            }
        ],
    )
    summary_path = run_root / "xjgt" / "ledger" / "xjgt_100_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "asset_count": 100,
                "final_verdict_counts": {
                    "fail": 21,
                    "review": 64,
                    "pass_with_notes": 15,
                },
            }
        ),
        encoding="utf-8",
    )
    _write_csv(
        run_root / "deepreach" / "manifests" / "supplier_manifest_deepreach.csv",
        [
            {
                "supplier_id": "deepreach",
                "asset_id": f"task_{index:03d}__head",
            }
            for index in range(8)
        ],
    )


def test_minimal_weekly_report_contract(tmp_path: Path) -> None:
    run_root = tmp_path / "acceptance_5x100"
    _prepare_inputs(run_root)

    outputs = build_weekly_report(run_root)

    workbook = load_workbook(outputs.workbook_xlsx, read_only=True)
    assert workbook.sheetnames == [
        "五供应商总览",
        "星际归途",
        "DeepReach",
        "供应商3",
        "供应商4",
        "供应商5",
        "人工与难测问题统计",
    ]
    assert list(
        next(workbook["星际归途"].iter_rows(values_only=True))
    ) == DETAIL_COLUMNS

    with outputs.summary_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    assert list(rows[0]) == SUMMARY_COLUMNS
    xjgt = rows[0]
    assert xjgt["supplier_name"] == "星际归途 / XJGT"
    assert xjgt["sample_clip_count"] == "100"
    assert xjgt["fail_clip_count"] == "21"
    assert xjgt["review_clip_count"] == "64"
    assert xjgt["pass_clip_count"] == "15"
    assert xjgt["pass_clip_ratio"] == "0.15"
    assert xjgt["fail_clip_ratio"] == "0.21"


def test_deepreach_and_placeholders_are_not_reported_as_pass(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "acceptance_5x100"
    _prepare_inputs(run_root)

    outputs = build_weekly_report(run_root)

    workbook = load_workbook(outputs.workbook_xlsx, read_only=True)
    deepreach_rows = list(
        workbook["DeepReach"].iter_rows(min_row=2, values_only=True)
    )
    assert len(deepreach_rows) == 8
    column_index = {name: index for index, name in enumerate(DETAIL_COLUMNS)}
    assert {
        row[column_index["skeleton_missing_status"]]
        for row in deepreach_rows
    } == {"blocked"}
    assert {
        row[column_index["sam3_containment_status"]]
        for row in deepreach_rows
    } == {"blocked"}
    assert {
        row[column_index["video_quality_status"]]
        for row in deepreach_rows
    } == {"not_run"}
    assert {
        row[column_index["final_clip_status"]]
        for row in deepreach_rows
    } == {"blocked"}

    with outputs.summary_csv.open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    deepreach = summary[1]
    assert deepreach["sample_clip_count"] == "8"
    assert deepreach["blocked_modules"] == "precheck|sam3"
    for row in summary[2:]:
        assert row["sample_clip_count"] == "0"
        assert row["expected_clip_count"] == "100"
        assert row["modules_completed"] == ""
        assert row["blocked_modules"] == "missing_input"
        assert row["notes"] == "input manifest not provided"


def test_hard_issue_sheet_contains_required_rows(tmp_path: Path) -> None:
    run_root = tmp_path / "acceptance_5x100"
    _prepare_inputs(run_root)

    outputs = build_weekly_report(run_root)

    workbook = load_workbook(outputs.workbook_xlsx, read_only=True)
    issue_types = {
        row[0]
        for row in workbook["人工与难测问题统计"].iter_rows(
            min_row=2, values_only=True
        )
    }
    assert issue_types == {
        "hand_out_of_frame",
        "severe_keypoint_offset",
        "skeleton_pose_hallucination",
        "visual_skeleton_presence_mismatch",
        "occlusion_or_mask_undersegmentation",
        "projection_review",
        "video_quality_pending_colleague_thresholds",
        "text_check_pending_rules",
    }
