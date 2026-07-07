import json
from pathlib import Path

import pandas as pd

from tools.build_batch_qc_ledger import load_manifest
from tools.build_manual_review_queue import (
    FAILURE_MODE_ENUM,
    MANUAL_OUTCOME_ENUM,
    MANUAL_TEMPLATE_COLUMNS,
    REVIEW_QUEUE_COLUMNS,
    assign_review_ids,
    build_review_index_html,
    manual_template_row,
    rows_from_candidate_windows,
    rows_from_sam3_summary,
    select_review_rows,
)
from tools.convert_manual_labels_csv_to_json import convert_csv_to_patch_records


def _write_csv(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
    assert "true_positive" in html
    assert "Fill <code>manual_labels_template.csv</code>" in html
    assert "100044_100_120.png" in html


def test_convert_completed_manual_csv_to_patch_records(tmp_path: Path) -> None:
    csv_path = tmp_path / "manual_labels_template.csv"
    _write_csv(
        csv_path,
        ",".join(MANUAL_TEMPLATE_COLUMNS) + "\n"
        "supplier_a_100044_100_120_0001,supplier_a,100044,100,120,111,"
        "review,side_view_mask_undersegmentation,true_positive,"
        "side_view_mask_undersegmentation,medium,high,"
        "side view mask undersegmentation,nathan\n",
    )

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


def test_convert_manual_csv_rejects_invalid_enum(tmp_path: Path) -> None:
    csv_path = tmp_path / "manual_labels_template.csv"
    _write_csv(
        csv_path,
        ",".join(MANUAL_TEMPLATE_COLUMNS) + "\n"
        "r1,supplier_a,100044,100,120,111,review,unknown,not_an_enum,"
        "unknown,medium,high,,nathan\n",
    )

    try:
        convert_csv_to_patch_records(csv_path)
    except ValueError as exc:
        assert "manual_outcome" in str(exc)
    else:
        raise AssertionError("invalid manual_outcome should fail")
