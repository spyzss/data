import json
from pathlib import Path

from tools.build_batch_qc_ledger import (
    add_candidate_windows,
    add_manual_review,
    add_precheck_aggregates,
    add_sam3_window_summaries,
    build_ledger_rows,
    build_supplier_issue_frequency,
    default_module_statuses,
    load_manifest,
)


def _write_csv(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_build_batch_qc_ledger_combines_modules(tmp_path: Path) -> None:
    manifest = tmp_path / "supplier_sample_manifest.csv"
    _write_csv(
        manifest,
        "supplier_id,asset_id,episode_idx\n"
        "supplier_a,100030,0\n"
        "supplier_a,100044,1\n"
        "supplier_b,100560,2\n",
    )
    assets, episode_to_asset = load_manifest(manifest)
    module_status = {asset_id: default_module_statuses() for asset_id in assets}
    events = []

    precheck_path = tmp_path / "clip_aggregates.json"
    precheck_rows = [
        {
            "episode_idx": 0,
            "check": "keypoint_missing",
            "checked_frames": 10,
            "flagged_frames": 1,
            "clip_flag": True,
        },
        {
            "episode_idx": 1,
            "check": "quality_score",
            "checked_frames": 10,
            "flagged_frames": 0,
            "clip_flag": True,
        },
        {
            "episode_idx": 2,
            "check": "text_integrity",
            "checked_frames": 1,
            "flagged_frames": 0,
            "clip_flag": None,
        },
    ]
    _write_json(precheck_path, precheck_rows)
    add_precheck_aggregates(
        precheck_rows,
        precheck_path,
        assets,
        episode_to_asset,
        module_status,
        events,
    )

    candidate_path = tmp_path / "candidate_windows.json"
    candidate_rows = [
        {
            "asset_id": "100044",
            "episode_idx": 1,
            "start_frame": 100,
            "end_frame": 120,
            "review_type": ["side_view_manual_review"],
            "trigger_reason": ["side_view_hand_orientation"],
            "priority": "medium",
            "needs_manual_review": True,
            "sam3_containment_eligible": False,
            "trigger_metrics": {"palm_camera_angle_deg_max": 87.0},
        }
    ]
    add_candidate_windows(
        candidate_rows,
        candidate_path,
        assets,
        episode_to_asset,
        module_status,
        events,
    )

    sam3_path = tmp_path / "window_keypoint_containment_summary.json"
    sam3_rows = [
        {
            "asset_id": "100044",
            "episode_idx": 1,
            "window_start_frame": 100,
            "window_end_frame": 120,
            "window_containment_verdict": "side_view_manual_review",
            "strong_fail_frame_count": 5,
            "inside_ratio_mean": 0.05,
            "source_review_type": ["side_view_manual_review"],
            "source_needs_manual_review": True,
            "source_sam3_containment_eligible": False,
            "reason": "side-view hand orientation makes SAM3 containment unreliable",
        }
    ]
    add_sam3_window_summaries(
        sam3_rows,
        sam3_path,
        assets,
        episode_to_asset,
        module_status,
        events,
    )

    manual_path = tmp_path / "manual_labels.json"
    manual_rows = [
        {
            "asset_id": "100560",
            "start": 27,
            "end": 147,
            "label": "positive",
            "failure_mode": "keypoints detached from hand",
        }
    ]
    add_manual_review(
        manual_rows,
        manual_path,
        assets,
        episode_to_asset,
        module_status,
        events,
    )

    ledger = {row["asset_id"]: row for row in build_ledger_rows(assets, module_status, events)}

    assert ledger["100030"]["final_verdict"] == "fail"
    assert ledger["100030"]["risk_level"] == "high"
    assert ledger["100030"]["keypoint_missing_status"] == "fail"
    assert ledger["100044"]["final_verdict"] == "review"
    assert ledger["100044"]["risk_level"] == "medium"
    assert ledger["100044"]["side_view_status"] == "review"
    assert ledger["100044"]["sam3_containment_status"] == "review"
    assert ledger["100560"]["final_verdict"] == "fail"
    assert ledger["100560"]["manual_review_status"] == "fail"

    frequency = build_supplier_issue_frequency(assets, events)
    side_view = next(row for row in frequency if row["issue_type"] == "side_view_manual_review")
    assert side_view["supplier_id"] == "supplier_a"
    assert side_view["affected_assets"] == 1
    assert side_view["total_assets"] == 2
    assert side_view["asset_rate"] == 0.5
