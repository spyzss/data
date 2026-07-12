from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from openpyxl import Workbook, load_workbook

from tools.build_acceptance_ledger import (
    GENERATED_SHEETS,
    build_acceptance_ledger,
    build_supplier_ledger,
    merge_intervals,
)


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
    assert all(row["final_status"] == "review" for row in result.asset_rows)


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
