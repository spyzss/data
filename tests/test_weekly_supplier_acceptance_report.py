import csv
import json
from pathlib import Path

from openpyxl import load_workbook

from tools.build_weekly_supplier_acceptance_report import (
    DETAIL_COLUMNS,
    SUMMARY_COLUMNS,
    build_weekly_report,
    recompute_xjgt_final_status,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _prepare_inputs(run_root: Path) -> None:
    _write_csv(
        run_root / "manifests" / "supplier_manifest_xjgt_100.csv",
        [
            {"supplier_id": "xjgt", "asset_id": "1001", "video_path": "/videos/1001.mp4"},
            {"supplier_id": "xjgt", "asset_id": "1002", "video_path": "/videos/1002.mp4"},
            {"supplier_id": "xjgt", "asset_id": "1003", "video_path": ""},
        ],
    )
    common = {
        "hdf5_text_status": "pass",
        "keypoint_missing_status": "pass",
        "keypoint_morphology_status": "pass",
        "video_quality_status": "pass",
        "temporal_status": "review",
        "sam3_evidence_status": "review",
        "manual_review_status": "review",
        "top_issue_types": "temporal_jump",
        "evidence_paths": "old_ledger.csv",
    }
    _write_csv(
        run_root / "xjgt" / "ledger" / "xjgt_100_asset_ledger.csv",
        [
            {
                **common,
                "asset_id": "1001",
                "quality_hand_status": "pass",
                "final_verdict": "review",
            },
            {
                **common,
                "asset_id": "1002",
                "quality_hand_status": "fail",
                "final_verdict": "fail",
            },
            {**common, "asset_id": "1003", "final_verdict": "fail"},
        ],
    )
    _write_csv(
        run_root / "xjgt" / "ledger" / "xjgt_100_issue_events.csv",
        [
            {
                "asset_id": "1001",
                "source_module": "precheck",
                "source_verdict": "fail",
                "failure_mode": "keypoint_missing",
                "start_frame": 80,
                "end_frame": 89,
            },
            {
                "asset_id": "1002",
                "source_module": "sam3_containment",
                "auto_verdict": "fail",
                "failure_mode": "containment_fail",
                "start_frame": 150,
                "end_frame": 169,
            },
            {
                "asset_id": "1002",
                "source_module": "sam3_containment",
                "auto_verdict": "fail",
                "failure_mode": "containment_fail",
                "start_frame": 120,
                "end_frame": 129,
            },
            {
                "asset_id": "1002",
                "source_module": "sam3_containment",
                "auto_verdict": "review",
                "failure_mode": "projection_review",
                "start_frame": 170,
                "end_frame": 199,
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
        run_root
        / "xjgt"
        / "manual_review"
        / "manual_labels_autosave.normalized.csv",
        [
            {
                "asset_id": "1001",
                "manual_outcome": "true_positive",
                "window_start_frame": 0,
                "window_end_frame": 49,
                "affected_start_frame": 0,
                "affected_end_frame": 4,
            },
            {
                "asset_id": "1001",
                "manual_outcome": "true_positive",
                "window_start_frame": 0,
                "window_end_frame": 49,
                "affected_start_frame": 4,
                "affected_end_frame": 9,
            },
            {
                "asset_id": "1001",
                "manual_outcome": "acceptable_flagged",
                "window_start_frame": 50,
                "window_end_frame": 99,
                "affected_start_frame": 10,
                "affected_end_frame": 99,
            },
            {
                "asset_id": "1002",
                "manual_outcome": "true_positive",
                "window_start_frame": 0,
                "window_end_frame": 99,
                "affected_start_frame": 10,
                "affected_end_frame": 19,
            },
            {
                "asset_id": "1002",
                "manual_outcome": "false_positive",
                "window_start_frame": 100,
                "window_end_frame": 149,
                "affected_start_frame": 20,
                "affected_end_frame": 199,
            },
        ],
    )
    _write_csv(
        run_root / "deepreach" / "manifests" / "supplier_manifest_deepreach.csv",
        [
            {"supplier_id": "deepreach", "asset_id": f"task_{index:03d}__head"}
            for index in range(8)
        ],
    )


def _patch_frame_probe(monkeypatch) -> list[Path]:
    observed_paths: list[Path] = []

    def fake_probe(path: Path) -> int:
        observed_paths.append(path)
        return {"1001.mp4": 100, "1002.mp4": 200, "1003_video.mp4": 0}[path.name]

    monkeypatch.setattr(
        "tools.build_weekly_supplier_acceptance_report.probe_video_frame_count",
        fake_probe,
    )
    return observed_paths


def test_xjgt_recomputes_frames_manual_ratio_and_final_status(
    tmp_path: Path, monkeypatch
) -> None:
    run_root = tmp_path / "acceptance_5x100"
    _prepare_inputs(run_root)
    observed_paths = _patch_frame_probe(monkeypatch)

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
    rows = list(workbook["星际归途"].iter_rows(min_row=2, values_only=True))
    columns = {name: index for index, name in enumerate(DETAIL_COLUMNS)}
    assert {
        "skeleton_missing_fail_frame_count",
        "skeleton_missing_fail_frame_ratio",
        "skeleton_morphology_fail_frame_count",
        "skeleton_morphology_fail_frame_ratio",
        "skeleton_static_fail_frame_count",
        "skeleton_static_fail_frame_ratio",
        "skeleton_static_status",
        "abnormal_fail_frame_count",
        "abnormal_fail_frame_ratio",
        "abnormal_frame_status",
        "fail_indicator_count",
        "video_quality_fail_frame_count",
        "video_quality_fail_frame_ratio",
    } <= set(columns)
    by_asset = {row[columns["asset_id"]]: row for row in rows}

    assert by_asset["1001"][columns["total_frames"]] == 100
    assert by_asset["1001"][columns["frame_count_status"]] == "ok"
    assert by_asset["1001"][columns["manual_problem_frame_count"]] == 10
    assert by_asset["1001"][columns["manual_problem_frame_ratio_of_clip"]] == 0.1
    assert by_asset["1001"][columns["manual_reviewed_frame_count"]] == 100
    assert by_asset["1001"][columns["manual_problem_ratio_of_reviewed"]] == 0.1
    assert by_asset["1001"][columns["manual_review_status"]] == "fail"
    assert by_asset["1001"][columns["skeleton_static_fail_frame_count"]] == 10
    assert by_asset["1001"][columns["skeleton_static_fail_frame_ratio"]] == 0.1
    assert by_asset["1001"][columns["skeleton_static_status"]] == "fail"
    assert by_asset["1001"][columns["abnormal_fail_frame_count"]] == 10
    assert by_asset["1001"][columns["abnormal_fail_frame_ratio"]] == 0.1
    assert by_asset["1001"][columns["abnormal_frame_status"]] == "fail"
    assert by_asset["1001"][columns["fail_indicator_count"]] == 2
    assert by_asset["1001"][columns["final_clip_status"]] == "fail"

    assert by_asset["1002"][columns["total_frames"]] == 200
    assert by_asset["1002"][columns["frame_count_status"]] == "ok"
    assert by_asset["1002"][columns["manual_problem_frame_count"]] == 10
    assert by_asset["1002"][columns["manual_problem_frame_ratio_of_clip"]] == 0.05
    assert by_asset["1002"][columns["manual_reviewed_frame_count"]] == 150
    assert by_asset["1002"][columns["manual_problem_ratio_of_reviewed"]] == 10 / 150
    assert by_asset["1002"][columns["manual_review_status"]] == "pass"
    assert by_asset["1002"][columns["skeleton_static_status"]] == "pass"
    assert by_asset["1002"][columns["abnormal_fail_frame_count"]] == 30
    assert by_asset["1002"][columns["abnormal_fail_frame_ratio"]] == 0.15
    assert by_asset["1002"][columns["abnormal_frame_status"]] == "fail"
    assert by_asset["1002"][columns["fail_indicator_count"]] == 1
    # One failed indicator is not enough for final fail.
    assert by_asset["1002"][columns["final_clip_status"]] == "pass"
    assert by_asset["1002"][columns["supplier_quality_signal"]] == "low"

    assert by_asset["1003"][columns["total_frames"]] == 0
    assert by_asset["1003"][columns["frame_count_status"]] == "unreadable"
    assert by_asset["1003"][columns["manual_problem_frame_count"]] == 0
    assert by_asset["1003"][columns["manual_reviewed_frame_count"]] == 0
    assert by_asset["1003"][columns["manual_review_status"]] == "not_reviewed"
    assert by_asset["1003"][columns["abnormal_frame_status"]] == "review"
    assert by_asset["1003"][columns["fail_indicator_count"]] == 0
    assert by_asset["1003"][columns["final_clip_status"]] == "review"
    assert (
        by_asset["1003"][columns["supplier_quality_signal"]]
        == "not_provided"
    )
    assert "frame_count_unreadable" in by_asset["1003"][columns["notes"]]
    assert Path(
        "/mnt/oss/egodata/XJGT_20260629/video/1003_video.mp4"
    ) in observed_paths

    with outputs.summary_csv.open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    xjgt = summary[0]
    assert xjgt["sample_clip_count"] == "3"
    assert xjgt["total_frame_count"] == "300"
    assert xjgt["problem_frame_count"] == "50"
    assert xjgt["problem_frame_ratio"] == str(50 / 300)
    assert xjgt["pass_clip_count"] == "1"
    assert xjgt["fail_clip_count"] == "1"
    assert xjgt["review_clip_count"] == "1"
    assert xjgt["pass_clip_ratio"] == str(1 / 3)
    assert xjgt["fail_clip_ratio"] == str(1 / 3)


def test_single_skeleton_static_indicator_does_not_fail_final_clip() -> None:
    assert (
        recompute_xjgt_final_status(
            text_check_status="pass",
            video_quality_status="pass",
            skeleton_static_status="fail",
            abnormal_frame_status="pass",
            expected_module_missing=False,
        )
        == "pass"
    )


def test_single_abnormal_indicator_does_not_fail_final_clip() -> None:
    assert (
        recompute_xjgt_final_status(
            text_check_status="pass",
            video_quality_status="pass",
            skeleton_static_status="pass",
            abnormal_frame_status="fail",
            expected_module_missing=False,
        )
        == "pass"
    )


def test_static_and_abnormal_indicators_fail_final_clip() -> None:
    assert (
        recompute_xjgt_final_status(
            text_check_status="pass",
            video_quality_status="pass",
            skeleton_static_status="fail",
            abnormal_frame_status="fail",
            expected_module_missing=False,
        )
        == "fail"
    )


def test_video_and_abnormal_indicators_fail_final_clip() -> None:
    assert (
        recompute_xjgt_final_status(
            text_check_status="pass",
            video_quality_status="fail",
            skeleton_static_status="pass",
            abnormal_frame_status="fail",
            expected_module_missing=False,
        )
        == "fail"
    )


def test_text_and_static_indicators_fail_final_clip() -> None:
    assert (
        recompute_xjgt_final_status(
            text_check_status="fail",
            video_quality_status="pass",
            skeleton_static_status="fail",
            abnormal_frame_status="pass",
            expected_module_missing=False,
        )
        == "fail"
    )


def test_deepreach_and_placeholders_remain_honest(
    tmp_path: Path, monkeypatch
) -> None:
    run_root = tmp_path / "acceptance_5x100"
    _prepare_inputs(run_root)
    _patch_frame_probe(monkeypatch)

    outputs = build_weekly_report(run_root)

    workbook = load_workbook(outputs.workbook_xlsx, read_only=True)
    deepreach_rows = list(workbook["DeepReach"].iter_rows(min_row=2, values_only=True))
    columns = {name: index for index, name in enumerate(DETAIL_COLUMNS)}
    assert len(deepreach_rows) == 8
    assert {row[columns["skeleton_missing_status"]] for row in deepreach_rows} == {
        "blocked"
    }
    assert {row[columns["sam3_containment_status"]] for row in deepreach_rows} == {
        "blocked"
    }
    assert {row[columns["video_quality_status"]] for row in deepreach_rows} == {
        "no_valid_output"
    }
    assert {
        row[columns["supplier_quality_signal"]]
        for row in deepreach_rows
    } == {"not_provided"}
    assert {row[columns["final_clip_status"]] for row in deepreach_rows} == {
        "blocked"
    }

    with outputs.summary_csv.open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    assert summary[1]["sample_clip_count"] == "8"
    assert summary[1]["blocked_modules"] == "precheck|sam3"
    assert "no_valid_output" in summary[1]["notes"]
    for row in summary[2:]:
        assert row["sample_clip_count"] == "0"
        assert row["blocked_modules"] == "missing_input"


def test_generation_prints_sanity_checks(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    run_root = tmp_path / "acceptance_5x100"
    _prepare_inputs(run_root)
    _patch_frame_probe(monkeypatch)

    build_weekly_report(run_root)

    output = capsys.readouterr().out
    assert "XJGT total_frame_count=300 ok=True" in output
    assert "XJGT problem_frame_count=50 ok=True" in output
    assert "XJGT manual_problem_ratio_nonzero=True" in output
    assert "XJGT final_clip_status counts={'fail': 1, 'pass': 1, 'review': 1}" in output
    assert "DeepReach sample_clip_count=8" in output
    assert "video_quality=no_valid_output" in output


def test_hard_issue_sheet_contains_required_rows(
    tmp_path: Path, monkeypatch
) -> None:
    run_root = tmp_path / "acceptance_5x100"
    _prepare_inputs(run_root)
    _patch_frame_probe(monkeypatch)

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
