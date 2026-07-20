from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from human_qc.sam3_window_review import load_review_bundle
from tools.build_sam3_review_bundle import build_sam3_review_bundle, parse_args


PNG = b"\x89PNG\r\n\x1a\nfixture"


def _write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_json(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def _evidence(
    root: Path,
    *,
    supplier: str,
    asset_id: str,
    start: int,
    end: int,
    review_id: str = "",
) -> list[dict]:
    rows = []
    for index, frame in enumerate((start, start + 1, (start + end) // 2, end - 1, end)):
        path = root / supplier / asset_id / f"{index}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(PNG)
        rows.append(
            {
                "review_id": review_id,
                "supplier_id": supplier,
                "asset_id": asset_id,
                "window_start_frame": start,
                "window_end_frame": end,
                "frame_idx": frame,
                "source_module": "sam3_containment",
                "evidence_type": "combined_overlay",
                "hand_side": "both",
                "source_path": str(path),
                "metadata_json": "{}",
            }
        )
    return rows


def test_three_supplier_bundle_routes_only_fail_and_review(tmp_path: Path) -> None:
    manifests = _write_csv(
        tmp_path / "manifest.csv",
        [
            {"supplier": "jdt", "asset_id": "jdt__episode_1"},
            {"supplier": "jdt", "asset_id": "jdt__episode_2"},
            {"supplier": "qy", "asset_id": "qy__episode_1"},
            {"supplier": "qy", "asset_id": "qy__episode_2"},
            {"supplier": "dr", "asset_id": "dr__task_1"},
        ],
    )
    jd_queue = _write_csv(
        tmp_path / "jd_queue.csv",
        [
            {
                "review_id": "jdt__episode_1__window_0_10",
                "supplier_id": "jdt",
                "asset_id": "jdt__episode_1",
                "window_start_frame": 0,
                "window_end_frame": 10,
                "sam3_window_state": "fail",
            },
            {
                "supplier_id": "jdt",
                "asset_id": "jdt__episode_2",
                "window_start_frame": 20,
                "window_end_frame": 30,
                "sam3_window_state": "pass",
            },
        ],
    )
    qy_run = tmp_path / "qy_run"
    _write_json(
        qy_run / "module_outputs/qy__episode_1/sam3_containment/window_results.json",
        [{
            "asset_id": "qy__episode_1",
            "start_frame": 100,
            "end_frame": 110,
            "window_containment_verdict": "review",
            "reason": "mixed_review",
        }],
    )
    _write_json(
        qy_run / "module_outputs/qy__episode_1/sam3_containment/evidence_manifest.json",
        _evidence(
            tmp_path / "evidence",
            supplier="qy",
            asset_id="qy__episode_1",
            start=100,
            end=110,
        ),
    )
    _write_json(
        qy_run / "module_outputs/qy__episode_2/sam3_containment/window_results.json",
        [{
            "asset_id": "qy__episode_2",
            "start_frame": 200,
            "end_frame": 210,
            "window_containment_verdict": "pass",
        }],
    )
    _write_json(
        qy_run / "module_outputs/qy__episode_2/sam3_containment/evidence_manifest.json",
        _evidence(
            tmp_path / "evidence",
            supplier="qy",
            asset_id="qy__episode_2",
            start=200,
            end=210,
        ),
    )
    dr_run = tmp_path / "dr_run"
    _write_json(
        dr_run / "module_outputs/dr__task_1/sam3_containment/window_results.json",
        [{
            "asset_id": "dr__task_1",
            "start_frame": 50,
            "end_frame": 60,
            "window_containment_verdict": "projection_review",
            "raw_window_containment_verdict": "pass",
            "calibration_status": "heuristic",
        }],
    )
    _write_json(
        dr_run / "module_outputs/dr__task_1/sam3_containment/evidence_manifest.json",
        _evidence(
            tmp_path / "evidence",
            supplier="dr",
            asset_id="dr__task_1",
            start=50,
            end=60,
        ),
    )
    jd_evidence = _write_csv(
        tmp_path / "jd_evidence.csv",
        _evidence(
            tmp_path / "evidence",
            supplier="jdt",
            asset_id="jdt__episode_1",
            start=0,
            end=10,
            review_id="jdt__episode_1__window_0_10",
        )
        + _evidence(
            tmp_path / "evidence",
            supplier="jdt",
            asset_id="jdt__episode_2",
            start=20,
            end=30,
        ),
    )

    outputs = build_sam3_review_bundle(
        manifest_paths=[manifests],
        review_queue_paths=[jd_queue],
        evidence_paths=[jd_evidence],
        run_roots=[qy_run, dr_run],
        output_dir=tmp_path / "bundle",
    )

    queue = pd.read_csv(outputs["review_queue"])
    assert queue["supplier_id"].tolist() == ["dr", "jdt", "qy"]
    assert set(queue["review_id"]) == {
        "dr__task_1__window_50_60",
        "jdt__episode_1__window_0_10",
        "qy__episode_1__window_100_110",
    }
    assert set(queue["asset_id"]) == {
        "dr__task_1",
        "jdt__episode_1",
        "qy__episode_1",
    }
    dr = queue.loc[queue["supplier_id"] == "dr"].iloc[0]
    assert dr["sam3_window_state"] == "review"
    assert dr["raw_window_containment_verdict"] == "pass"
    assert dr["duration_resolution_status"] == "not_annotated"

    evidence = pd.read_csv(outputs["review_evidence"])
    assert len(evidence) == 15
    assert evidence.groupby("review_id").size().to_dict() == {
        "dr__task_1__window_50_60": 5,
        "jdt__episode_1__window_0_10": 5,
        "qy__episode_1__window_100_110": 5,
    }
    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    assert summary["auto_pass_count"] == 2
    assert summary["human_fail_count"] == 1
    assert summary["human_review_count"] == 2
    assert summary["evidence_count"] == 15
    assert summary["blocked_count"] == 0
    assert summary["supplier_distribution"] == {"dr": 1, "jdt": 1, "qy": 1}
    review_bundle = load_review_bundle(
        manifest_path=outputs["review_manifest"],
        queue_path=outputs["review_queue"],
        evidence_path=outputs["review_evidence"],
        review_dir=tmp_path / "served_review",
    )
    assert len(review_bundle.items) == 3
    assert all(item["can_review"] is True for item in review_bundle.items)


def test_bundle_never_uses_overlap_only_evidence_matching(tmp_path: Path) -> None:
    manifest = _write_csv(
        tmp_path / "manifest.csv",
        [{"supplier": "qy", "asset_id": "qy__episode_1"}],
    )
    queue = _write_csv(
        tmp_path / "queue.csv",
        [{
            "supplier_id": "qy",
            "asset_id": "qy__episode_1",
            "window_start_frame": 100,
            "window_end_frame": 110,
            "sam3_window_state": "review",
        }],
    )
    evidence = _write_csv(
        tmp_path / "evidence.csv",
        _evidence(
            tmp_path / "evidence",
            supplier="qy",
            asset_id="qy__episode_1",
            start=101,
            end=109,
        ),
    )

    with pytest.raises(ValueError, match="expected 5 combined overlays, found 0"):
        build_sam3_review_bundle(
            manifest_paths=[manifest],
            review_queue_paths=[queue],
            evidence_paths=[evidence],
            run_roots=[],
            output_dir=tmp_path / "bundle",
        )


def test_bundle_exact_review_id_evidence_wins_over_exact_window_fallback(
    tmp_path: Path,
) -> None:
    review_id = "qy__episode_1__window_100_110"
    manifest = _write_csv(
        tmp_path / "manifest.csv",
        [{"supplier": "qy", "asset_id": "qy__episode_1"}],
    )
    queue = _write_csv(
        tmp_path / "queue.csv",
        [{
            "review_id": review_id,
            "supplier_id": "qy",
            "asset_id": "qy__episode_1",
            "window_start_frame": 100,
            "window_end_frame": 110,
            "sam3_window_state": "review",
        }],
    )
    exact = _evidence(
        tmp_path / "exact",
        supplier="qy",
        asset_id="qy__episode_1",
        start=100,
        end=110,
        review_id=review_id,
    )
    fallback = _evidence(
        tmp_path / "fallback",
        supplier="qy",
        asset_id="qy__episode_1",
        start=100,
        end=110,
    )
    evidence = _write_csv(tmp_path / "evidence.csv", exact + fallback)

    outputs = build_sam3_review_bundle(
        manifest_paths=[manifest],
        review_queue_paths=[queue],
        evidence_paths=[evidence],
        run_roots=[],
        output_dir=tmp_path / "bundle",
    )

    rows = pd.read_csv(outputs["review_evidence"])
    assert len(rows) == 5
    assert all("/exact/" in path for path in rows["source_path"])


def test_bundle_cli_accepts_explicit_and_run_root_inputs(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--manifest", str(tmp_path / "manifest.csv"),
            "--review-queue", str(tmp_path / "queue.csv"),
            "--evidence-manifest", str(tmp_path / "evidence.csv"),
            "--run-root", str(tmp_path / "run"),
            "--output-dir", str(tmp_path / "bundle"),
        ]
    )

    assert args.manifest == [tmp_path / "manifest.csv"]
    assert args.review_queue == [tmp_path / "queue.csv"]
    assert args.evidence_manifest == [tmp_path / "evidence.csv"]
    assert args.run_root == [tmp_path / "run"]
