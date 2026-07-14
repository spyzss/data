import json
from pathlib import Path

from tools.build_batch_qc_report import build_report, flatten_report, write_csv_row


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_build_batch_qc_report_combines_precheck_and_video_quality(tmp_path: Path) -> None:
    precheck_dir = tmp_path / "outputs" / "precheck_100003"
    _write_json(
        precheck_dir / "clip_aggregates.json",
        [
            {
                "episode_idx": 0,
                "check": "text_integrity",
                "checked_frames": 1,
                "flagged_frames": 0,
                "uncalibrated_frames": 1,
                "clip_flag": None,
            },
            {
                "episode_idx": 0,
                "check": "keypoint_missing",
                "checked_frames": 10,
                "flagged_frames": 1,
                "uncalibrated_frames": 0,
                "clip_flag": True,
            },
        ],
    )
    _write_json(
        precheck_dir / "check_results.json",
        [
            {
                "episode_idx": 0,
                "check": "quality_score",
                "frame_idx": -1,
                "metrics": {"pass_ratio": 1.0, "pass_threshold": 0.9},
                "flag": True,
                "reason": "clip-level quality_hand acceptance verdict",
            },
            {
                "episode_idx": 0,
                "check": "keypoint_missing",
                "frame_idx": 7,
                "metrics": {"missing_frames_in_10s_window_left": 31.0},
                "flag": True,
                "reason": "window exceeded",
            },
        ],
    )

    batch_dir = tmp_path / "sampled_100003"
    _write_json(
        batch_dir / "quality_archive" / "100003.json",
        {
            "schema_version": "asset_qc_report.v1",
            "asset_id": "100003",
            "qc_summary": {
                "status": "passed",
                "passed": True,
                "completed_modules": ["video_quality"],
                "failed_modules": [],
                "reasons": [],
            },
        },
    )

    report = build_report(
        asset_id="100003",
        batch_dir=batch_dir,
        precheck_dir=precheck_dir,
        video_quality_json=None,
    )

    assert report["asset_id"] == "100003"
    assert report["modules"]["precheck"]["derived"]["quality_score_pass"] is True
    assert report["modules"]["precheck"]["derived"]["keypoint_missing_pass"] is False
    assert report["modules"]["video_quality"]["qc_summary"]["passed"] is True
    assert report["final"]["status"] == "failed"
    assert report["final"]["run_sam3_containment_recommended"] is False
    assert report["final"]["hard_fail_reasons"] == ["keypoint_missing_failed"]
    assert report["modules"]["precheck"]["flagged_frame_samples"]["keypoint_missing"][
        0
    ]["frame_idx"] == 7

    csv_path = tmp_path / "batch_qc_100003.csv"
    write_csv_row(csv_path, report)
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "asset_id,final_status" in csv_text
    assert "100003,failed" in csv_text

    flat = flatten_report(report)
    assert flat["video_quality_pass"] is True
