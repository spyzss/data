from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from qc_common.config import load_qc_acceptance_config
from qc_common.contracts import ModuleResult
from qc_common.report_mutation import apply_module_result
from qc_pipeline.context import AssetContext
from tests.fixtures import solid_frame, write_test_video


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _result_rows(output_dir: Path) -> list[dict[str, object]]:
    return json.loads(
        (output_dir / "video_quality_results.json").read_text(encoding="utf-8")
    )


def _prerequisite_rows(output_dir: Path) -> list[dict[str, object]]:
    return json.loads(
        (output_dir / "video_quality_prerequisites.json").read_text(
            encoding="utf-8"
        )
    )


def _advance_report_to_video(
    batch_root: Path,
    *,
    asset_id: str,
    video_path: Path,
    source_range: tuple[int, int],
    profile: str = "acceptance",
    manifest_metadata: dict[str, object] | None = None,
) -> None:
    config = load_qc_acceptance_config()
    context = AssetContext(
        asset_id=asset_id,
        batch_root=batch_root,
        report_path=batch_root / "quality_archive" / f"{asset_id}.json",
        source_files={
            "video": {"path": video_path.relative_to(batch_root).as_posix()}
        },
        source_range=source_range,
        metadata=manifest_metadata or {},
    )
    revision = 0
    modules = config.pipeline_modules
    for index, module in enumerate(modules[: modules.index("video_quality")]):
        report = apply_module_result(
            context.report_path,
            context=context,
            config=config,
            profile=profile,
            result=ModuleResult(module, "pass", {}, {}),
            expected_revision=revision,
            next_module=modules[index + 1],
            now=f"2026-07-14T00:00:{index:02d}Z",
        )
        revision = report["report_revision"]


def _completed_manifest_video_report(
    tmp_path: Path,
    *,
    profile: str,
) -> tuple[Path, Path, Path]:
    from tools.run_manifest_video_quality import run_manifest_video_quality

    video = tmp_path / "source.mp4"
    write_test_video(video, [solid_frame(90) for _ in range(5)], fps=10.0)
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [
            {
                "asset_id": "logical-a",
                "primary_video_path": str(video),
                "start_frame": 0,
                "end_frame": 4,
            }
        ],
    )
    _advance_report_to_video(
        tmp_path,
        asset_id="logical-a",
        video_path=video,
        source_range=(0, 5),
        profile=profile,
    )
    output_dir = tmp_path / "quality"
    summary = run_manifest_video_quality(
        manifest,
        output_dir,
        batch_root=tmp_path,
        profile=profile,
    )
    assert summary["qc_report_write_count"] == 1
    return (
        manifest,
        output_dir,
        tmp_path / "quality_archive" / "logical-a.json",
    )


@pytest.mark.parametrize("extension", [".parquet", ".jsonl"])
def test_manifest_video_quality_reads_supported_manifest_formats(
    tmp_path: Path, extension: str
) -> None:
    from tools.run_manifest_video_quality import read_manifest

    path = tmp_path / f"manifest{extension}"
    rows = [{"asset_id": "clip-a", "start_frame": 1, "end_frame": 2}]
    if extension == ".parquet":
        pd.DataFrame(rows).to_parquet(path, index=False)
    else:
        path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")

    assert read_manifest(path) == rows


def test_manifest_video_quality_uses_inclusive_ranges_and_source_coordinates(
    tmp_path: Path,
) -> None:
    from tools.run_manifest_video_quality import run_manifest_video_quality

    video = tmp_path / "source.mp4"
    write_test_video(video, [solid_frame(90) for _ in range(16)], fps=10.0)
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [
            {
                "asset_id": "logical-a",
                "primary_video_path": str(video),
                "start_frame": 2,
                "end_frame": 13,
            }
        ],
    )
    _advance_report_to_video(
        tmp_path,
        asset_id="logical-a",
        video_path=video,
        source_range=(2, 14),
    )

    summary = run_manifest_video_quality(
        manifest,
        tmp_path / "quality",
        batch_root=tmp_path,
        profile="acceptance",
    )

    rows = _result_rows(tmp_path / "quality")
    assert len(rows) == 1
    row = rows[0]
    assert row["asset_id"] == "logical-a"
    assert row["source_video_path"] == str(video)
    assert row["clip_start_frame"] == 2
    assert row["clip_end_frame"] == 13
    assert row["clip_frame_count"] == 12
    assert row["decoded_frame_count"] == 12
    assert row["source_video_frame_count"] == 16
    assert row["sampled_frame_mappings"] == [
        {
            "local_frame_idx": local_frame_idx,
            "source_frame_idx": 2 + local_frame_idx,
        }
        for local_frame_idx in row["sampled_local_frame_indices"]
    ]
    intervals = row["video_quality"]["metrics"]["freeze_metrics"][
        "frozen_intervals"
    ]
    assert intervals[0]["local_start_frame"] == 0
    assert intervals[0]["local_end_frame"] == 11
    assert intervals[0]["start_frame"] == 2
    assert intervals[0]["end_frame"] == 13
    assert intervals[0]["source_start_frame"] == 2
    assert intervals[0]["source_end_frame"] == 13
    assert (tmp_path / "quality" / "video_quality_results.parquet").exists()
    assert (tmp_path / "quality" / "video_quality_decision_summary.csv").exists()
    assert (tmp_path / "quality" / "video_quality_failures.json").exists()
    run_config = json.loads(
        (tmp_path / "quality" / "run_config.json").read_text(encoding="utf-8")
    )
    assert run_config["frame_range_semantics"] == "inclusive_source_frames"
    assert not list((tmp_path / "quality").glob("*.mp4"))
    assert summary["completed_clip_count"] == 1
    assert summary["qc_report_write_count"] == 1
    assert summary["awaiting_pipeline_clip_count"] == 0
    report = json.loads(
        (tmp_path / "quality_archive" / "logical-a.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["schema_version"] == "asset_qc_report.v2"
    assert report["video_quality"]["flow"]["result_gate"]["verdict"] in {
        "pass",
        "warn",
        "fail",
    }
    assert report["video_quality"]["evidence"] == [
        {
            "evidence_id": report["video_quality"]["evidence"][0]["evidence_id"],
            "kind": "source_video",
            "path": "source.mp4",
            "coordinate_system": "source_video_inclusive",
            "start_frame": 2,
            "end_frame": 13,
            "hand_side": None,
            "checksum": None,
            "mime_type": "video/mp4",
            "generator_version": "video_prefilter_v0.3.2",
        }
    ]


def test_manifest_video_quality_fresh_run_records_pipeline_prerequisite(
    tmp_path: Path,
) -> None:
    from tools.run_manifest_video_quality import main, run_manifest_video_quality

    video = tmp_path / "source.mp4"
    write_test_video(video, [solid_frame(90) for _ in range(4)], fps=10.0)
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [
            {
                "asset_id": "logical-a",
                "primary_video_path": str(video),
                "start_frame": 0,
                "end_frame": 3,
            }
        ],
    )

    summary = run_manifest_video_quality(
        manifest,
        tmp_path / "quality",
        batch_root=tmp_path,
    )

    assert summary["completed_clip_count"] == 1
    assert summary["qc_report_write_count"] == 0
    assert summary["awaiting_pipeline_clip_count"] == 1
    assert not (tmp_path / "quality_archive" / "logical-a.json").exists()
    prerequisites = json.loads(
        (tmp_path / "quality" / "video_quality_prerequisites.json").read_text(
            encoding="utf-8"
        )
    )
    assert prerequisites == [
        {
            "asset_id": "logical-a",
            "condition": "awaiting_pipeline",
            "current_next_module": None,
            "reason": "report_missing",
            "required_module": "video_quality",
            "report_path": "quality_archive/logical-a.json",
            "source_range": {
                "coordinate_system": "source_video_inclusive",
                "start_frame": 0,
                "end_frame": 3,
            },
        }
    ]
    assert main(
        [
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "quality"),
            "--batch-root",
            str(tmp_path),
        ]
    ) == 3


def test_manifest_video_quality_supplier_profile_keeps_machine_verdict(
    tmp_path: Path,
) -> None:
    from tools.run_manifest_video_quality import run_manifest_video_quality

    video = tmp_path / "source.mp4"
    write_test_video(video, [solid_frame(90) for _ in range(4)], fps=10.0)
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [
            {
                "asset_id": "logical-a",
                "primary_video_path": str(video),
                "start_frame": 0,
                "end_frame": 3,
            }
        ],
    )
    _advance_report_to_video(
        tmp_path,
        asset_id="logical-a",
        video_path=video,
        source_range=(0, 4),
        profile="supplier_evaluation",
    )

    summary = run_manifest_video_quality(
        manifest,
        tmp_path / "quality",
        batch_root=tmp_path,
        profile="supplier_evaluation",
    )

    assert summary["qc_report_write_count"] == 1
    report = json.loads(
        (tmp_path / "quality_archive" / "logical-a.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["video_quality"]["flow"]["result_gate"]["verdict"] == "fail"
    assert report["video_quality"]["flow"]["exit_gate"] == {
        "state": "continue",
        "continue_to_next_module": True,
        "next_module": "supplier_data_audit",
    }
    assert report["pipeline_state"]["status"] == "running"


def test_manifest_video_quality_reuses_report_manifest_metadata(
    tmp_path: Path,
) -> None:
    from tools.run_manifest_video_quality import run_manifest_video_quality

    video = tmp_path / "source.mp4"
    write_test_video(video, [solid_frame(90) for _ in range(4)], fps=10.0)
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [
            {
                "asset_id": "logical-a",
                "primary_video_path": str(video),
                "start_frame": 0,
                "end_frame": 3,
            }
        ],
    )
    manifest_metadata = {
        "scene": "manifest scene",
        "task": "manifest task",
        "task_name": "pick_cup",
        "text_en": "Pick the cup.",
        "text_label": "display label",
    }
    _advance_report_to_video(
        tmp_path,
        asset_id="logical-a",
        video_path=video,
        source_range=(0, 4),
        manifest_metadata=manifest_metadata,
    )

    summary = run_manifest_video_quality(
        manifest,
        tmp_path / "quality",
        batch_root=tmp_path,
    )

    assert summary["qc_report_write_count"] == 1
    report = json.loads(
        (tmp_path / "quality_archive" / "logical-a.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["manifest_metadata"] == manifest_metadata


def test_manifest_video_quality_rejects_legacy_report_without_manifest_metadata(
    tmp_path: Path,
) -> None:
    from tools.run_manifest_video_quality import run_manifest_video_quality

    video = tmp_path / "source.mp4"
    write_test_video(video, [solid_frame(90) for _ in range(4)], fps=10.0)
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [
            {
                "asset_id": "logical-a",
                "primary_video_path": str(video),
                "start_frame": 0,
                "end_frame": 3,
            }
        ],
    )
    _advance_report_to_video(
        tmp_path,
        asset_id="logical-a",
        video_path=video,
        source_range=(0, 4),
    )
    report_path = tmp_path / "quality_archive" / "logical-a.json"
    legacy = json.loads(report_path.read_text(encoding="utf-8"))
    legacy.pop("manifest_metadata")
    report_path.write_text(json.dumps(legacy), encoding="utf-8")
    before = report_path.read_bytes()

    summary = run_manifest_video_quality(
        manifest,
        tmp_path / "quality",
        batch_root=tmp_path,
    )

    assert summary["qc_report_write_count"] == 0
    assert summary["failed_clip_count"] == 1
    assert report_path.read_bytes() == before


def test_manifest_video_quality_handles_repeated_source_ranges_independently(
    tmp_path: Path,
) -> None:
    from tools.run_manifest_video_quality import run_manifest_video_quality

    video = tmp_path / "shared.mp4"
    write_test_video(
        video,
        [solid_frame(40 + frame_idx * 10) for frame_idx in range(10)],
        fps=10.0,
    )
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [
            {
                "asset_id": "left-range",
                "primary_video_path": str(video),
                "start_frame": 0,
                "end_frame": 2,
            },
            {
                "asset_id": "right-range",
                "primary_video_path": str(video),
                "start_frame": 6,
                "end_frame": 9,
            },
        ],
    )

    run_manifest_video_quality(manifest, tmp_path / "quality")

    rows = {row["asset_id"]: row for row in _result_rows(tmp_path / "quality")}
    assert rows["left-range"]["decoded_frame_count"] == 3
    assert rows["left-range"]["clip_frame_count"] == 3
    assert rows["right-range"]["decoded_frame_count"] == 4
    assert rows["right-range"]["clip_frame_count"] == 4


def test_manifest_video_rejects_completed_block_without_valid_exit_gate(
    tmp_path: Path,
) -> None:
    from tools.run_manifest_video_quality import run_manifest_video_quality

    video = tmp_path / "source.mp4"
    write_test_video(video, [solid_frame(90) for _ in range(5)], fps=10.0)
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [
            {
                "asset_id": "logical-a",
                "primary_video_path": str(video),
                "start_frame": 0,
                "end_frame": 4,
            }
        ],
    )
    _advance_report_to_video(
        tmp_path,
        asset_id="logical-a",
        video_path=video,
        source_range=(0, 5),
    )
    output_dir = tmp_path / "quality"
    run_manifest_video_quality(manifest, output_dir, batch_root=tmp_path)
    report_path = tmp_path / "quality_archive" / "logical-a.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    del report["video_quality"]["flow"]["exit_gate"]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    before = report_path.read_bytes()

    summary = run_manifest_video_quality(
        manifest,
        output_dir,
        batch_root=tmp_path,
        overwrite=True,
    )

    assert summary["skipped_clip_count"] == 0
    assert summary["qc_report_write_count"] == 0
    assert summary["awaiting_pipeline_clip_count"] == 1
    assert report_path.read_bytes() == before
    prerequisite = _prerequisite_rows(output_dir)[0]
    assert prerequisite["condition"] == "invalid_report"
    assert prerequisite["reason"] == "video_quality_exit_gate_invalid"


def test_manifest_video_rejects_completed_source_range_mismatch(
    tmp_path: Path,
) -> None:
    from tools.run_manifest_video_quality import run_manifest_video_quality

    video = tmp_path / "source.mp4"
    write_test_video(video, [solid_frame(90) for _ in range(5)], fps=10.0)
    first_manifest = _write_manifest(
        tmp_path / "first.csv",
        [
            {
                "asset_id": "logical-a",
                "primary_video_path": str(video),
                "start_frame": 0,
                "end_frame": 2,
            }
        ],
    )
    _advance_report_to_video(
        tmp_path,
        asset_id="logical-a",
        video_path=video,
        source_range=(0, 3),
    )
    output_dir = tmp_path / "quality"
    run_manifest_video_quality(first_manifest, output_dir, batch_root=tmp_path)
    report_path = tmp_path / "quality_archive" / "logical-a.json"
    before = report_path.read_bytes()
    second_manifest = _write_manifest(
        tmp_path / "second.csv",
        [
            {
                "asset_id": "logical-a",
                "primary_video_path": str(video),
                "start_frame": 1,
                "end_frame": 3,
            }
        ],
    )

    summary = run_manifest_video_quality(
        second_manifest,
        output_dir,
        batch_root=tmp_path,
        overwrite=True,
    )

    assert summary["qc_report_write_count"] == 0
    assert summary["awaiting_pipeline_clip_count"] == 1
    assert report_path.read_bytes() == before
    prerequisite = _prerequisite_rows(output_dir)[0]
    assert prerequisite["condition"] == "invalid_report"
    assert prerequisite["reason"] == "video_quality_source_range_mismatch"


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("null_successor", "video_quality_exit_gate_invalid"),
        ("wrong_successor", "video_quality_exit_gate_invalid"),
        ("exit_pipeline_mismatch", "video_quality_pipeline_state_invalid"),
        ("invalid_continue_state", "video_quality_pipeline_state_invalid"),
        ("invalid_stop_state", "video_quality_exit_gate_invalid"),
    ],
)
def test_manifest_video_rejects_inconsistent_completed_flow(
    tmp_path: Path,
    case: str,
    reason: str,
) -> None:
    from tools.run_manifest_video_quality import run_manifest_video_quality

    manifest, output_dir, report_path = _completed_manifest_video_report(
        tmp_path,
        profile="supplier_evaluation",
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    exit_gate = report["video_quality"]["flow"]["exit_gate"]
    if case == "null_successor":
        exit_gate["next_module"] = None
    elif case == "wrong_successor":
        exit_gate["next_module"] = "semantic_consistency"
    elif case == "exit_pipeline_mismatch":
        report["pipeline_state"]["next_module"] = "semantic_consistency"
    elif case == "invalid_continue_state":
        report["pipeline_state"].update(
            {
                "status": "stopped",
                "next_module": None,
                "stop_reason": "quality_fail:video_quality",
            }
        )
        report["overall_decision"] = "fail"
    else:
        exit_gate.update(
            {
                "state": "stop_qc",
                "continue_to_next_module": False,
                "next_module": None,
            }
        )
        result_gate = report["video_quality"]["flow"]["result_gate"]
        result_gate.update(
            {"verdict": "pass", "has_fail": False, "has_warn": False}
        )
        report["video_quality"]["evaluation"]["decision"] = "pass"
        report["pipeline_state"].update(
            {
                "status": "stopped",
                "next_module": None,
                "stop_reason": "quality_fail:video_quality",
            }
        )
        report["overall_decision"] = "fail"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    before = report_path.read_bytes()

    summary = run_manifest_video_quality(
        manifest,
        output_dir,
        batch_root=tmp_path,
        profile="supplier_evaluation",
    )

    assert summary["skipped_clip_count"] == 0
    assert summary["qc_report_write_count"] == 0
    assert summary["awaiting_pipeline_clip_count"] == 1
    assert report_path.read_bytes() == before
    prerequisite = _prerequisite_rows(output_dir)[0]
    assert prerequisite["condition"] == "invalid_report"
    assert prerequisite["reason"] == reason


@pytest.mark.parametrize(
    ("profile", "exit_state", "pipeline_status", "exit_next"),
    [
        ("supplier_evaluation", "continue", "running", "supplier_data_audit"),
        ("acceptance", "continue", "running", "supplier_data_audit"),
    ],
)
def test_manifest_video_skips_consistent_completed_outcomes(
    tmp_path: Path,
    profile: str,
    exit_state: str,
    pipeline_status: str,
    exit_next: str | None,
) -> None:
    from tools.run_manifest_video_quality import run_manifest_video_quality

    manifest, output_dir, report_path = _completed_manifest_video_report(
        tmp_path,
        profile=profile,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["video_quality"]["flow"]["exit_gate"]["state"] == exit_state
    assert report["video_quality"]["flow"]["exit_gate"]["next_module"] == exit_next
    assert report["pipeline_state"]["status"] == pipeline_status
    before = report_path.read_bytes()

    summary = run_manifest_video_quality(
        manifest,
        output_dir,
        batch_root=tmp_path,
        profile=profile,
    )

    assert summary["skipped_clip_count"] == 1
    assert summary["qc_report_write_count"] == 0
    assert report_path.read_bytes() == before


def test_manifest_video_quality_rejects_source_drift_before_report_write(
    tmp_path: Path,
) -> None:
    from tools.run_manifest_video_quality import run_manifest_video_quality

    original = tmp_path / "original.mp4"
    replacement = tmp_path / "replacement.mp4"
    frames = [solid_frame(90) for _ in range(5)]
    write_test_video(original, frames, fps=10.0)
    write_test_video(replacement, frames, fps=10.0)
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [
            {
                "asset_id": "logical-a",
                "primary_video_path": str(replacement),
                "start_frame": 0,
                "end_frame": 4,
            }
        ],
    )
    _advance_report_to_video(
        tmp_path,
        asset_id="logical-a",
        video_path=original,
        source_range=(0, 5),
    )
    report_path = tmp_path / "quality_archive" / "logical-a.json"
    before = report_path.read_bytes()

    summary = run_manifest_video_quality(
        manifest,
        tmp_path / "quality",
        batch_root=tmp_path,
    )

    assert summary["failed_clip_count"] == 1
    assert summary["qc_report_write_count"] == 0
    assert report_path.read_bytes() == before
    failures = json.loads(
        (tmp_path / "quality" / "video_quality_failures.json").read_text(
            encoding="utf-8"
        )
    )
    assert "source_files.video.path mismatch" in failures[0]["error"]


def test_manifest_video_quality_isolates_invalid_ranges(tmp_path: Path) -> None:
    from tools.run_manifest_video_quality import run_manifest_video_quality

    video = tmp_path / "source.mp4"
    write_test_video(video, [solid_frame(100) for _ in range(5)], fps=10.0)
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [
            {
                "asset_id": "valid",
                "primary_video_path": str(video),
                "start_frame": 1,
                "end_frame": 3,
            },
            {
                "asset_id": "invalid",
                "primary_video_path": str(video),
                "start_frame": 4,
                "end_frame": 9,
            },
        ],
    )

    summary = run_manifest_video_quality(manifest, tmp_path / "quality")

    assert summary["completed_clip_count"] == 1
    assert summary["failed_clip_count"] == 1
    assert [row["asset_id"] for row in _result_rows(tmp_path / "quality")] == [
        "valid"
    ]
    failures = json.loads(
        (tmp_path / "quality" / "video_quality_failures.json").read_text(
            encoding="utf-8"
        )
    )
    assert failures[0]["asset_id"] == "invalid"
    assert failures[0]["source_video_path"] == str(video)
    assert failures[0]["clip_start_frame"] == 4
    assert failures[0]["clip_end_frame"] == 9
    assert "outside source frame count" in failures[0]["error"]


def test_manifest_video_quality_dry_run_writes_no_producer_outputs(
    tmp_path: Path,
) -> None:
    from tools.run_manifest_video_quality import run_manifest_video_quality

    video = tmp_path / "source.mp4"
    write_test_video(video, [solid_frame(100) for _ in range(5)], fps=10.0)
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [
            {
                "asset_id": "valid",
                "primary_video_path": str(video),
                "start_frame": 0,
                "end_frame": 4,
            }
        ],
    )
    output_dir = tmp_path / "quality"

    summary = run_manifest_video_quality(manifest, output_dir, dry_run=True)

    assert summary["validated_clip_count"] == 1
    assert not (output_dir / "video_quality_results.json").exists()
    assert not (output_dir / "video_quality_results.parquet").exists()
    assert not (output_dir / "video_quality_decision_summary.csv").exists()
    assert not (output_dir / "run_config.json").exists()


def test_manifest_video_quality_overwrites_only_valid_completed_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.run_manifest_video_quality as runner

    video = tmp_path / "source.mp4"
    write_test_video(video, [solid_frame(100) for _ in range(5)], fps=10.0)
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [
            {
                "asset_id": "clip-a",
                "primary_video_path": str(video),
                "start_frame": 0,
                "end_frame": 4,
            }
        ],
    )
    output_dir = tmp_path / "quality"
    _advance_report_to_video(
        tmp_path,
        asset_id="clip-a",
        video_path=video,
        source_range=(0, 5),
    )
    runner.run_manifest_video_quality(manifest, output_dir)
    report_path = tmp_path / "quality_archive" / "clip-a.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    revision = report["report_revision"]
    report["video_quality"]["review_marker"] = "replace"
    report["hdf5_text_info"]["review_marker"] = "keep"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    calls = 0
    original = runner.analyze_video_frame_range

    def counted(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "analyze_video_frame_range", counted)
    skipped = runner.run_manifest_video_quality(manifest, output_dir)
    assert calls == 0
    assert skipped["skipped_clip_count"] == 1

    overwritten = runner.run_manifest_video_quality(
        manifest,
        output_dir,
        overwrite=True,
    )
    assert calls == 1
    assert overwritten["skipped_clip_count"] == 0
    assert overwritten["qc_report_write_count"] == 1
    updated = json.loads(report_path.read_text(encoding="utf-8"))
    assert updated["report_revision"] == revision + 1
    assert "review_marker" not in updated["video_quality"]
    assert updated["hdf5_text_info"]["review_marker"] == "keep"
