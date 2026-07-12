from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Border, Font, PatternFill, Side

from tools.build_acceptance_ledger import (
    GENERATED_SHEETS,
    build_acceptance_ledger,
    build_supplier_ledger,
    merge_intervals,
)
from tools.build_weekly_supplier_acceptance_report import (
    CORE_DETAIL_COLUMNS,
    detail_columns_for_rows,
)


WEEKLY_SHEETS = [
    "五供应商总览",
    "星际归途",
    "DeepReach",
    "京东JDT",
    "供应商4",
    "供应商5",
    "异常检测拆解",
    "人工问题与阈值",
]

REAL_WEEKLY_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "acceptance_ledger_xjgt_jdt_dr_weekly.yaml"
)
WEEKLY_SUPPLIER_CONFIG_KEYS = {
    "supplier_id",
    "manifest",
    "ledger_asset_ledger",
    "precheck_check_results",
    "precheck_clip_aggregates",
    "candidate_windows",
    "video_quality",
    "sam3_summary",
    "review_queue",
    "manual_labels",
    "blocker",
    "blocked",
    "blocked_policy",
    "text_required",
}


def test_real_xjgt_jdt_deepreach_weekly_config_loads() -> None:
    config = yaml.safe_load(REAL_WEEKLY_CONFIG.read_text(encoding="utf-8"))

    assert config["workbook_mode"] == "weekly_template"
    suppliers = {
        supplier["supplier_id"]: supplier for supplier in config["suppliers"]
    }
    assert set(suppliers) == {"xjgt", "jdt", "deepreach"}
    assert all(
        set(supplier) <= WEEKLY_SUPPLIER_CONFIG_KEYS
        for supplier in suppliers.values()
    )
    assert suppliers["xjgt"]["manifest"] == (
        "outputs/acceptance_5x100/xjgt/ledger/xjgt_video_frame_counts.csv"
    )
    assert suppliers["xjgt"]["ledger_asset_ledger"] == (
        "outputs/acceptance_5x100/xjgt/ledger/xjgt_100_asset_ledger.csv"
    )
    assert suppliers["xjgt"]["precheck_check_results"] == (
        "outputs/acceptance_5x100/xjgt/precheck/check_results.parquet"
    )
    assert suppliers["xjgt"]["video_quality"] == (
        "outputs/acceptance_5x100/xjgt/video_quality/"
        "video_quality_acceptance_summary.csv"
    )
    assert suppliers["xjgt"]["sam3_summary"] == (
        "outputs/acceptance_5x100/xjgt/sam3_containment/"
        "window_keypoint_containment_summary.parquet"
    )
    assert suppliers["xjgt"]["review_queue"] == (
        "outputs/acceptance_5x100/xjgt/review/review_queue.csv"
    )
    assert suppliers["xjgt"]["manual_labels"] == (
        "outputs/acceptance_5x100/xjgt/manual_review/"
        "manual_labels_autosave.normalized.csv"
    )
    assert suppliers["jdt"]["manifest"].endswith(
        "/jdt/manifest/jdt_sample_100_manifest.csv"
    )
    assert suppliers["jdt"]["precheck_check_results"].endswith(
        "/jdt/precheck/check_results.parquet"
    )
    assert suppliers["jdt"]["video_quality"].endswith(
        "/jdt/video_quality/video_quality_results.json"
    )
    assert suppliers["jdt"]["sam3_summary"].endswith(
        "/jdt/sam3_containment/window_keypoint_containment_summary.parquet"
    )
    assert suppliers["jdt"]["review_queue"].endswith(
        "/jdt/review/review_queue.csv"
    )
    assert suppliers["jdt"]["manual_labels"].endswith(
        "/jdt/manual_review/manual_labels.csv"
    )
    assert suppliers["deepreach"]["manifest"].endswith(
        "/deepreach/manifest/deepreach_sample_100_manifest.csv"
    )
    assert suppliers["deepreach"]["precheck_check_results"].endswith(
        "/deepreach/precheck/check_results.parquet"
    )
    assert suppliers["deepreach"]["video_quality"].endswith(
        "/deepreach/video_quality/video_quality_results.json"
    )
    assert suppliers["deepreach"]["blocked"] is True
    assert suppliers["deepreach"]["blocked_policy"] == "all_frames_review"
    assert suppliers["deepreach"]["blocker"].endswith(
        "/deepreach/blockers/sam3_containment_blocker.json"
    )


def _canonical_weekly_header() -> list[str]:
    return detail_columns_for_rows(
        [
            {
                "text_field_present_task_status": "pass",
                "text_field_nonempty_task_status": "pass",
                "video_black_screen_status": "pass",
                "video_underexposure_status": "pass",
                "video_overexposure_status": "pass",
                "video_blur_status": "pass",
                "video_freeze_stutter_status": "pass",
                "video_frame_alignment_status": "pass",
            }
        ]
    )


def _write_weekly_template(tmp_path: Path) -> Path:
    path = tmp_path / "weekly_template.xlsx"
    workbook = Workbook()
    workbook.remove(workbook.active)
    old_names = [
        "五供应商总览",
        "星际归途",
        "DeepReach",
        "供应商3",
        "供应商4",
        "供应商5",
        "异常检测拆解",
        "人工问题与阈值",
    ]
    header = _canonical_weekly_header()
    border = Border(bottom=Side(style="thin", color="44546A"))
    for name in old_names:
        sheet = workbook.create_sheet(name)
        if name in {"星际归途", "DeepReach", "供应商3", "供应商4", "供应商5"}:
            sheet.append(header)
            for cell in sheet[1]:
                cell.fill = PatternFill("solid", fgColor="B4C6E7")
                cell.font = Font(bold=True, color="1F1F1F")
                cell.border = border
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            sheet.row_dimensions[1].height = 37
            sheet.column_dimensions["A"].width = 25
    xjgt = workbook["星际归途"]
    xjgt_row = {column: "" for column in header}
    xjgt_row.update(
        {
            "asset_id": "xjgt-existing",
            "total_frames": 10,
            "text_check_status": "pass",
            "skeleton_static_status": "pass",
            "video_quality_status": "pass",
            "abnormal_frame_status": "pass",
            "fail_indicator_count": 0,
            "acceptance_status": "pass",
            "abnormal_frame_status_v2": "pass",
            "acceptance_status_v2": "pass",
            "review_status_v2": "completed",
        }
    )
    xjgt.append([xjgt_row[column] for column in header])
    workbook["供应商4"]["A2"] = "placeholder-four"
    workbook["供应商5"]["A2"] = "placeholder-five"
    workbook["人工问题与阈值"]["A1"] = (
        "定向复核不是无偏全局召回率估计"
    )
    workbook.save(path)
    return path


def _write_weekly_supplier_inputs(tmp_path: Path) -> tuple[Path, Path]:
    jdt_dir = tmp_path / "jdt"
    jdt_dir.mkdir()
    asset_ids = [f"jdt-{index:03d}" for index in range(100)]
    pd.DataFrame(
        [
            {
                "asset_id": asset_id,
                "start_frame": 0,
                "end_frame": 9,
                "primary_video_path": f"/{asset_id}.mp4",
            }
            for asset_id in asset_ids
        ]
    ).to_csv(jdt_dir / "manifest.csv", index=False)
    check_rows = []
    for asset_id in asset_ids:
        check_rows.extend(
            [
                {
                    "asset_id": asset_id,
                    "check": "skeleton_quality_score",
                    "frame_idx": -1,
                    "flag": False,
                    "metrics": "{}",
                },
                {
                    "asset_id": asset_id,
                    "check": "keypoint_morphology",
                    "frame_idx": -1,
                    "flag": False,
                    "metrics": json.dumps({"morphology_verdict": "pass"}),
                },
                {
                    "asset_id": asset_id,
                    "check": "keypoint_temporal",
                    "frame_idx": -1,
                    "flag": False,
                    "metrics": "{}",
                },
            ]
        )
    pd.DataFrame(check_rows).to_parquet(jdt_dir / "check_results.parquet", index=False)
    pd.DataFrame(
        [{"asset_id": asset_id, "check_count": 3} for asset_id in asset_ids]
    ).to_parquet(jdt_dir / "clip_aggregates.parquet", index=False)
    pd.DataFrame(
        [{"asset_id": asset_id, "status": "pass"} for asset_id in asset_ids]
    ).to_json(jdt_dir / "video_quality.json", orient="records")
    candidate_rows = [
        {
            "review_id": f"review-{index}",
            "asset_id": asset_ids[index],
            "window_start_frame": 0,
            "window_end_frame": 4,
            "source_level": "window",
            "module": "precheck",
        }
        for index in range(3)
    ]
    pd.DataFrame(candidate_rows).to_parquet(
        jdt_dir / "candidate_windows.parquet", index=False
    )
    pd.DataFrame(candidate_rows).to_csv(jdt_dir / "review_queue.csv", index=False)
    pd.DataFrame(
        [
            {
                "asset_id": asset_ids[index],
                "window_start_frame": 0,
                "window_end_frame": 4,
                "window_containment_verdict": "mixed_review",
            }
            for index in range(3)
        ]
    ).to_parquet(jdt_dir / "sam3.parquet", index=False)
    pd.DataFrame(
        [
            {
                "review_id": "review-0",
                "asset_id": asset_ids[0],
                "window_start_frame": 0,
                "window_end_frame": 4,
                "affected_start_frame": 1,
                "affected_end_frame": 2,
                "manual_outcome": "true_positive",
                "failure_mode": "severe_keypoint_offset",
            },
            {
                "review_id": "review-1",
                "asset_id": asset_ids[1],
                "window_start_frame": 0,
                "window_end_frame": 4,
                "affected_start_frame": "",
                "affected_end_frame": "",
                "manual_outcome": "false_positive",
                "failure_mode": "unknown",
            },
        ]
    ).to_csv(jdt_dir / "manual_labels.csv", index=False)

    dr_dir = tmp_path / "deepreach"
    dr_dir.mkdir()
    pd.DataFrame(
        [
            {"asset_id": "dr-000", "start_frame": 0, "end_frame": 9},
            {"asset_id": "dr-001", "start_frame": 0, "end_frame": 19},
        ]
    ).to_csv(dr_dir / "manifest.csv", index=False)
    (dr_dir / "blocker.json").write_text(
        json.dumps(
            {
                "status": "blocked",
                "reason": "missing_head_calibration_lineage",
            }
        ),
        encoding="utf-8",
    )
    return jdt_dir, dr_dir


def _weekly_config(tmp_path: Path) -> Path:
    jdt_dir, dr_dir = _write_weekly_supplier_inputs(tmp_path)
    config = tmp_path / "weekly_ledger.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "workbook_mode": "weekly_template",
                "output_label": "weekly_test",
                "suppliers": [
                    {
                        "supplier_id": "jdt",
                        "manifest": str(jdt_dir / "manifest.csv"),
                        "precheck_check_results": str(
                            jdt_dir / "check_results.parquet"
                        ),
                        "precheck_clip_aggregates": str(
                            jdt_dir / "clip_aggregates.parquet"
                        ),
                        "candidate_windows": str(
                            jdt_dir / "candidate_windows.parquet"
                        ),
                        "video_quality": str(jdt_dir / "video_quality.json"),
                        "sam3_summary": str(jdt_dir / "sam3.parquet"),
                        "review_queue": str(jdt_dir / "review_queue.csv"),
                        "manual_labels": str(jdt_dir / "manual_labels.csv"),
                        "text_required": False,
                        "required_inputs": ["manifest"],
                    },
                    {
                        "supplier_id": "deepreach",
                        "manifest": str(dr_dir / "manifest.csv"),
                        "blocker": str(dr_dir / "blocker.json"),
                        "blocked": True,
                        "blocked_policy": "all_frames_review",
                        "text_required": False,
                        "required_inputs": ["manifest"],
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config


def _write_jdt_inputs(tmp_path: Path) -> dict[str, Path]:
    manifest = tmp_path / "jdt_manifest.csv"
    pd.DataFrame(
        [
            {"asset_id": "asset-a", "start_frame": 0, "end_frame": 99},
            {"asset_id": "asset-b", "start_frame": 0, "end_frame": 49},
            {"asset_id": "asset-c", "start_frame": 0, "end_frame": 29},
            {"asset_id": "asset-d", "start_frame": 0, "end_frame": 19},
        ]
    ).to_csv(manifest, index=False)

    manual_labels = tmp_path / "manual_labels.csv"
    pd.DataFrame(
        [
            {
                "review_id": "review-a",
                "asset_id": "asset-a",
                "window_start_frame": 0,
                "window_end_frame": 29,
                "affected_start_frame": 10,
                "affected_end_frame": 19,
                "manual_outcome": "true_positive",
                "acceptance_status": "rejected",
            },
            {
                "review_id": "review-a",
                "asset_id": "asset-a",
                "window_start_frame": 0,
                "window_end_frame": 29,
                "affected_start_frame": 15,
                "affected_end_frame": 25,
                "manual_outcome": "true_positive",
                "acceptance_status": "rejected",
            },
            {
                "review_id": "review-b",
                "asset_id": "asset-b",
                "window_start_frame": 5,
                "window_end_frame": 9,
                "affected_start_frame": "",
                "affected_end_frame": "",
                "manual_outcome": "true_positive",
                "acceptance_status": "rejected",
            },
        ]
    ).to_csv(manual_labels, index=False)

    review_queue = tmp_path / "review_queue.csv"
    pd.DataFrame(
        [
            {
                "review_id": "review-a",
                "asset_id": "asset-a",
                "window_start_frame": 0,
                "window_end_frame": 29,
            },
            {
                "review_id": "review-a-unresolved",
                "asset_id": "asset-a",
                "window_start_frame": 50,
                "window_end_frame": 59,
            },
            {
                "review_id": "review-b",
                "asset_id": "asset-b",
                "window_start_frame": 5,
                "window_end_frame": 9,
            },
            {
                "review_id": "review-c",
                "asset_id": "asset-c",
                "window_start_frame": 0,
                "window_end_frame": 9,
            },
        ]
    ).to_csv(review_queue, index=False)

    sam3_summary = tmp_path / "sam3_summary.parquet"
    pd.DataFrame(
        [
            {"asset_id": "asset-a", "window_containment_verdict": "review"},
            {"asset_id": "asset-c", "window_containment_verdict": "review"},
        ]
    ).to_parquet(sam3_summary, index=False)
    return {
        "manifest": manifest,
        "manual_labels": manual_labels,
        "review_queue": review_queue,
        "sam3_summary": sam3_summary,
    }


def _write_deepreach_inputs(tmp_path: Path) -> dict[str, Path]:
    manifest = tmp_path / "deepreach_manifest.csv"
    pd.DataFrame(
        [
            {"asset_id": "dr-a", "frame_count": 10},
            {"asset_id": "dr-b", "frame_count": 20},
        ]
    ).to_csv(manifest, index=False)
    blocker = tmp_path / "sam3_containment_blocker.json"
    blocker.write_text(
        json.dumps(
            {
                "status": "blocked",
                "reason": "head calibration lineage missing",
            }
        ),
        encoding="utf-8",
    )
    return {"manifest": manifest, "blocker": blocker}


def _config(tmp_path: Path) -> Path:
    jdt = _write_jdt_inputs(tmp_path)
    deepreach = _write_deepreach_inputs(tmp_path)
    config_path = tmp_path / "ledger.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "output_label": "test_jdt_dr",
                "suppliers": [
                    {
                        "supplier_id": "jdt",
                        **{key: str(value) for key, value in jdt.items()},
                        "blocked": False,
                        "required_inputs": ["manifest"],
                    },
                    {
                        "supplier_id": "deepreach",
                        **{key: str(value) for key, value in deepreach.items()},
                        "blocked": True,
                        "blocked_policy": "all_frames_review",
                        "required_inputs": ["manifest"],
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path


def test_merge_intervals_unions_overlapping_inclusive_ranges() -> None:
    assert merge_intervals([(10, 19), (15, 25), (30, 30)]) == [
        (10, 25),
        (30, 30),
    ]


def test_jdt_manual_labels_compute_fail_review_pass_and_priority(tmp_path: Path) -> None:
    inputs = _write_jdt_inputs(tmp_path)
    result = build_supplier_ledger(
        {
            "supplier_id": "jdt",
            **{key: str(value) for key, value in inputs.items()},
            "blocked": False,
            "required_inputs": ["manifest"],
        },
        config_dir=tmp_path,
    )
    assets = {row["asset_id"]: row for row in result.asset_rows}

    assert assets["asset-a"]["fail_frame_count"] == 16
    assert assets["asset-a"]["review_frame_count"] == 10
    assert assets["asset-a"]["pass_frame_count"] == 74
    assert assets["asset-a"]["final_status"] == "fail"
    assert assets["asset-b"]["fail_frame_count"] == 5
    assert assets["asset-b"]["final_status"] == "fail"
    assert assets["asset-c"]["review_frame_count"] == 10
    assert assets["asset-c"]["final_status"] == "review"
    assert assets["asset-d"]["final_status"] == "pass"

    overview = result.overview
    assert overview["total_clip_count"] == 4
    assert overview["total_frame_count"] == 200
    assert overview["fail_frame_count"] == 21
    assert overview["review_frame_count"] == 20
    assert overview["pass_frame_count"] == 159
    assert overview["fail_clip_count"] == 2
    assert overview["review_clip_count"] == 1
    assert overview["pass_clip_count"] == 1
    assert overview["manual_review_count"] == 2
    assert overview["manual_review_window_count"] == 2
    assert overview["manual_review_asset_count"] == 2
    assert overview["manual_review_label_rows"] == 3
    assert overview["sam3_status"] == "evaluated"
    assert "blocker=not_evaluated" not in overview["note"]


def test_manual_review_counts_windows_assets_and_rows_independently(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([{"asset_id": "asset-a", "frame_count": 20}]).to_csv(
        manifest, index=False
    )
    manual_labels = tmp_path / "manual_labels.csv"
    pd.DataFrame(
        [
            {
                "review_id": "review-001",
                "asset_id": "asset-a",
                "affected_start_frame": 1,
                "affected_end_frame": 2,
                "manual_outcome": "true_positive",
            },
            {
                "review_id": "review-002",
                "asset_id": "asset-a",
                "affected_start_frame": 10,
                "affected_end_frame": 11,
                "manual_outcome": "true_positive",
            },
        ]
    ).to_csv(manual_labels, index=False)

    result = build_supplier_ledger(
        {
            "supplier_id": "jdt",
            "manifest": str(manifest),
            "manual_labels": str(manual_labels),
            "required_inputs": ["manifest"],
        },
        config_dir=tmp_path,
    )

    assert result.overview["manual_review_window_count"] == 2
    assert result.overview["manual_review_asset_count"] == 1
    assert result.overview["manual_review_label_rows"] == 2
    assert result.overview["manual_review_count"] == 2


def test_blocked_all_frames_review_marks_every_clip_and_frame_review(
    tmp_path: Path,
) -> None:
    inputs = _write_deepreach_inputs(tmp_path)
    result = build_supplier_ledger(
        {
            "supplier_id": "deepreach",
            **{key: str(value) for key, value in inputs.items()},
            "blocked": True,
            "blocked_policy": "all_frames_review",
            "required_inputs": ["manifest"],
        },
        config_dir=tmp_path,
    )

    assert result.overview["total_clip_count"] == 2
    assert result.overview["total_frame_count"] == 30
    assert result.overview["review_frame_count"] == 30
    assert result.overview["pass_frame_count"] == 0
    assert result.overview["fail_frame_count"] == 0
    assert result.overview["review_clip_count"] == 2
    assert result.overview["sam3_status"] == "blocked"
    assert "head calibration lineage missing" in result.overview["note"]
    assert all(row["final_status"] == "review" for row in result.asset_rows)


def test_configured_missing_blocker_is_explicit_but_unconfigured_is_silent(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([{"asset_id": "asset", "frame_count": 5}]).to_csv(
        manifest, index=False
    )
    base = {
        "supplier_id": "supplier",
        "manifest": str(manifest),
        "required_inputs": ["manifest"],
    }

    without_blocker = build_supplier_ledger(base, config_dir=tmp_path)
    with_missing_blocker = build_supplier_ledger(
        {**base, "blocker": str(tmp_path / "missing_blocker.json")},
        config_dir=tmp_path,
    )

    assert "blocker=not_evaluated" not in without_blocker.overview["note"]
    assert "blocker=not_evaluated" in with_missing_blocker.overview["note"]


def test_overview_totals_balance_and_workbook_contains_generated_sheets(
    tmp_path: Path,
) -> None:
    output = tmp_path / "ledger.xlsx"
    build_acceptance_ledger(
        config_path=_config(tmp_path),
        output_path=output,
        overwrite=True,
    )

    workbook = load_workbook(output, read_only=True)
    assert workbook.sheetnames == list(GENERATED_SHEETS)
    overview = pd.read_excel(output, sheet_name="Overview")
    assert (
        overview["pass_frame_count"]
        + overview["fail_frame_count"]
        + overview["review_frame_count"]
    ).equals(overview["total_frame_count"])
    assert (
        overview["pass_clip_count"]
        + overview["fail_clip_count"]
        + overview["review_clip_count"]
    ).equals(overview["total_clip_count"])


def test_existing_workbook_preserves_unrelated_sheets_and_replaces_generated(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "existing.xlsx"
    workbook = Workbook()
    notes = workbook.active
    notes.title = "Keep_Me"
    notes["A1"] = "preserved"
    old_overview = workbook.create_sheet("Overview")
    old_overview["A1"] = "stale"
    workbook.save(existing)
    output = tmp_path / "updated.xlsx"

    build_acceptance_ledger(
        config_path=_config(tmp_path),
        output_path=output,
        existing_workbook=existing,
        overwrite=True,
    )

    updated = load_workbook(output)
    assert updated["Keep_Me"]["A1"].value == "preserved"
    assert updated["Overview"]["A1"].value == "supplier_id"
    assert updated.sheetnames == ["Keep_Me", *GENERATED_SHEETS]


def test_missing_required_manifest_raises_but_optional_modules_do_not(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="required input manifest"):
        build_supplier_ledger(
            {
                "supplier_id": "missing",
                "manifest": str(tmp_path / "missing.csv"),
                "required_inputs": ["manifest"],
            },
            config_dir=tmp_path,
        )

    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([{"asset_id": "asset", "frame_count": 5}]).to_csv(
        manifest, index=False
    )
    result = build_supplier_ledger(
        {
            "supplier_id": "optional",
            "manifest": str(manifest),
            "sam3_summary": str(tmp_path / "missing.parquet"),
            "required_inputs": ["manifest"],
        },
        config_dir=tmp_path,
    )

    assert result.overview["sam3_status"] == "not_evaluated"
    assert result.overview["pass_clip_count"] == 1


def test_weekly_template_preserves_canonical_structure_and_maps_jdt_dr(
    tmp_path: Path,
) -> None:
    template = _write_weekly_template(tmp_path)
    output = tmp_path / "weekly_output.xlsx"

    build_acceptance_ledger(
        config_path=_weekly_config(tmp_path),
        output_path=output,
        existing_workbook=template,
        overwrite=True,
    )

    workbook = load_workbook(output)
    assert workbook.sheetnames == WEEKLY_SHEETS
    xjgt = workbook["星际归途"]
    jdt = workbook["京东JDT"]
    deepreach = workbook["DeepReach"]
    canonical_header = [cell.value for cell in xjgt[1]]
    assert canonical_header == _canonical_weekly_header()
    assert [cell.value for cell in jdt[1]] == canonical_header
    assert [cell.value for cell in deepreach[1]] == canonical_header
    assert jdt.max_row == 101
    assert jdt.freeze_panes == xjgt.freeze_panes == "A2"
    assert jdt.row_dimensions[1].height == xjgt.row_dimensions[1].height
    assert jdt.column_dimensions["A"].width == xjgt.column_dimensions["A"].width
    assert jdt["A1"].fill.fgColor.rgb == xjgt["A1"].fill.fgColor.rgb
    assert jdt["A1"].font.bold == xjgt["A1"].font.bold
    assert jdt["A1"].border.bottom.style == xjgt["A1"].border.bottom.style
    assert workbook["供应商4"]["A2"].value == "placeholder-four"
    assert workbook["供应商5"]["A2"].value == "placeholder-five"
    assert "无偏全局召回率" in workbook["人工问题与阈值"]["A1"].value

    columns = {name: index for index, name in enumerate(canonical_header)}
    jdt_rows = {
        row[columns["asset_id"]]: row
        for row in jdt.iter_rows(min_row=2, values_only=True)
    }
    for asset_id, row in jdt_rows.items():
        assert row[columns["text_check_status"]] in {
            "pass",
            "not_applicable",
        }
        assert row[columns["skeleton_static_status"]] == "pass"
        assert row[columns["video_quality_status"]] == "pass"
        assert row[columns["abnormal_frame_status"]] in {
            "pass",
            "fail",
            "review",
        }
        assert row[columns["temporal_detected_frame_count"]] is not None
        assert row[columns["sam3_detected_frame_count"]] is not None
        assert row[columns["manual_true_positive_frame_count"]] is not None

    assert jdt_rows["jdt-000"][columns["acceptance_status"]] == "fail"
    assert jdt_rows["jdt-001"][columns["acceptance_status"]] == "pass"
    assert jdt_rows["jdt-002"][columns["acceptance_status"]] == "review"
    assert jdt_rows["jdt-002"][columns["abnormal_frame_status"]] == "review"
    assert (
        jdt_rows["jdt-002"][columns["unreviewed_submitted_interval_count"]]
        == 1
    )

    dr_rows = list(deepreach.iter_rows(min_row=2, values_only=True))
    assert len(dr_rows) == 2
    assert all(
        row[columns["sam3_containment_status"]] == "blocked"
        for row in dr_rows
    )
    assert all(
        row[columns["abnormal_frame_status"]] == "review" for row in dr_rows
    )
    assert all(row[columns["acceptance_status"]] == "review" for row in dr_rows)
    assert all(row[columns["fail_indicator_count"]] == 0 for row in dr_rows)

    overview = workbook["五供应商总览"]
    overview_rows = list(overview.iter_rows(values_only=True))
    overview_header = list(overview_rows[0])
    overview_by_name = {
        row[overview_header.index("supplier_name")]: dict(zip(overview_header, row))
        for row in overview_rows[1:]
    }
    jdt_summary = overview_by_name["京东JDT / JDT"]
    assert jdt_summary["sample_clip_count"] == 100
    assert jdt_summary["pass_clip_count"] == 98
    assert jdt_summary["fail_clip_count"] == 1
    assert jdt_summary["review_clip_count"] == 1
    assert (
        jdt_summary["pass_clip_count"]
        + jdt_summary["fail_clip_count"]
        + jdt_summary["review_clip_count"]
        == jdt_summary["sample_clip_count"]
    )
    assert overview_by_name["供应商4"]["sample_clip_count"] == 0
    assert overview_by_name["供应商5"]["sample_clip_count"] == 0

    decomposition = workbook["异常检测拆解"]
    decomposition_header = [cell.value for cell in decomposition[1]]
    decomposition_rows = {
        row[0]: row
        for row in decomposition.iter_rows(min_row=2, values_only=True)
    }
    jdt_decomposition = decomposition_rows["京东JDT / JDT"]
    for column in (
        "precheck_temporal_candidate_frame_count",
        "sam3_processed_frame_count",
        "manual_true_positive_frame_count",
        "auto_union_hit_manual_tp_frame_count",
        "abnormal_fail_frame_count",
        "abnormal_review_frame_count",
    ):
        assert jdt_decomposition[decomposition_header.index(column)] is not None
