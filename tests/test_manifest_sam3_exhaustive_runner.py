from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def test_exhaustive_runner_is_an_independent_tool() -> None:
    assert importlib.util.find_spec("tools.run_manifest_sam3_exhaustive") is not None


@pytest.mark.parametrize(
    ("script_name", "expected_option"),
    [
        ("run_manifest_sam3_exhaustive.py", "--checkpoint-every-frames"),
        ("build_precheck_sam3_frame_comparison.py", "--sam3-frame-results"),
        ("build_sam3_comparison_evidence.py", "--false-positive-frames"),
        ("merge_sam3_exhaustive_shards.py", "--shard-dir"),
    ],
)
def test_new_cli_help_runs_as_direct_script(
    script_name: str,
    expected_option: str,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [sys.executable, str(repo_root / "tools" / script_name), "--help"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert expected_option in completed.stdout


def test_enumeration_emits_every_inclusive_source_frame_once_at_stride_one() -> None:
    from tools.run_manifest_sam3_exhaustive import enumerate_asset_frames

    rows = enumerate_asset_frames(
        {
            "asset_id": "jdt__episode_000001",
            "start_frame": 3,
            "end_frame": 6,
        }
    )

    assert [row["source_frame"] for row in rows] == [3, 4, 5, 6]
    assert [row["local_frame"] for row in rows] == [0, 1, 2, 3]
    assert [row["video_frame"] for row in rows] == [3, 4, 5, 6]
    assert len({(row["asset_id"], row["source_frame"]) for row in rows}) == 4


@pytest.mark.parametrize(
    ("raw_verdict", "audit_verdict"),
    [
        ("strong_containment_mismatch", "fail"),
        ("containment_review", "review"),
        ("projection_review", "review"),
        ("mask_missing_or_tiny_review", "review"),
        ("likely_visible_ok", "pass"),
    ],
)
def test_raw_hand_verdict_is_preserved_and_mapped_for_audit(
    raw_verdict: str,
    audit_verdict: str,
) -> None:
    from tools.run_manifest_sam3_exhaustive import classify_audit_hand

    assert classify_audit_hand(raw_verdict) == audit_verdict


@pytest.mark.parametrize(
    ("hands", "expected"),
    [
        ({"left": "fail", "right": "pass"}, "fail"),
        ({"left": "review", "right": "pass"}, "review"),
        ({"left": "pass", "right": "pass"}, "pass"),
        ({"left": "blocked", "right": "pass"}, "blocked"),
        ({}, "blocked"),
    ],
)
def test_frame_verdict_uses_required_hand_precedence(
    hands: dict[str, str],
    expected: str,
) -> None:
    from tools.run_manifest_sam3_exhaustive import aggregate_frame_verdict

    result = aggregate_frame_verdict(hands, required_hands=("left", "right"))

    assert result["frame_verdict"] == expected
    assert result["required_hand_count"] == 2
    assert result["evaluable_hand_count"] == sum(
        verdict in {"pass", "review", "fail"} for verdict in hands.values()
    )
    assert result["blocked_hand_count"] == 2 - result["evaluable_hand_count"]


def _flat_hand(x_offset: float = 0.0) -> list[float]:
    return np.asarray(
        [[x_offset + 5 + index % 5, 5 + index // 5] for index in range(21)],
        dtype=np.float64,
    ).reshape(-1).tolist()


def _write_manifest_inputs(
    tmp_path: Path,
    *,
    start_frame: int = 1,
    end_frame: int = 4,
) -> tuple[Path, Path, Path]:
    video_path = tmp_path / "episode.mp4"
    video_path.write_bytes(b"synthetic-video-placeholder")
    parquet_path = tmp_path / "episode.parquet"
    pd.DataFrame(
        [
            {
                "leftcam_left_kp2d": _flat_hand(0.0),
                "leftcam_right_kp2d": _flat_hand(10.0),
            }
            for _ in range(end_frame + 2)
        ]
    ).to_parquet(parquet_path, index=False)
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "asset_id": "jdt__episode_000001",
                "supplier": "jdt",
                "episode_index": 1,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "primary_video_path": str(video_path),
                "parquet_path": str(parquet_path),
                "left_hand_2d_field": "leftcam_left_kp2d",
                "right_hand_2d_field": "leftcam_right_kp2d",
            }
        ]
    ).to_csv(manifest_path, index=False)
    return manifest_path, video_path, parquet_path


class _FakeSequentialSource:
    def __init__(self, *, eof_after: int | None = None) -> None:
        self.eof_after = eof_after
        self.requested_video_frames: list[int] = []
        self.iter_frames_calls = 0
        self.closed = False

    def read_parquet(self, path: Path) -> pd.DataFrame:
        return pd.read_parquet(path)

    def iter_frames(self, path: Path, video_frames: list[int]):
        del path
        self.iter_frames_calls += 1
        for offset, video_frame in enumerate(video_frames):
            self.requested_video_frames.append(video_frame)
            if self.eof_after is not None and offset >= self.eof_after:
                return
            yield video_frame, np.zeros((64, 96, 3), dtype=np.uint8), 0.001

    def video_metadata(self, path: Path) -> dict[str, float | int]:
        del path
        return {"frame_count": 100, "width": 96, "height": 64, "fps": 30.0}

    def close(self) -> None:
        self.closed = True


class _FullMaskSegmenter:
    def __init__(self) -> None:
        self.calls = 0

    def segment_frame(self, frame, queries, config):
        del queries, config
        self.calls += 1
        return [
            SimpleNamespace(
                mask=np.ones(frame.shape[:2], dtype=bool),
                category="hand",
            )
        ]


class _EmptyMaskSegmenter(_FullMaskSegmenter):
    def segment_frame(self, frame, queries, config):
        del frame, queries, config
        self.calls += 1
        return []


def test_exhaustive_producer_does_not_require_candidate_windows(
    tmp_path: Path,
) -> None:
    from tools.run_manifest_sam3_exhaustive import run_manifest_sam3_exhaustive

    manifest, _, _ = _write_manifest_inputs(tmp_path)
    source = _FakeSequentialSource()
    segmenter = _FullMaskSegmenter()
    output_dir = tmp_path / "exhaustive"

    summary = run_manifest_sam3_exhaustive(
        manifest=manifest,
        supplier="jdt",
        sam3_model=None,
        output_dir=output_dir,
        source_reader=source,
        segmenter=segmenter,
        checkpoint_every_frames=2,
    )

    frames = pd.read_parquet(output_dir / "sam3_exhaustive_frame_results.parquet")
    hand_rows = [
        json.loads(line)
        for line in (
            output_dir / "sam3_exhaustive_hand_results.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert frames["source_frame"].tolist() == [1, 2, 3, 4]
    assert frames["video_frame"].tolist() == [1, 2, 3, 4]
    assert frames["frame_verdict"].tolist() == ["pass"] * 4
    assert len(hand_rows) == 8
    assert {row["raw_containment_verdict"] for row in hand_rows} == {
        "likely_visible_ok"
    }
    assert segmenter.calls == 4
    assert source.requested_video_frames == [1, 2, 3, 4]
    assert source.closed is True
    assert summary["total_source_frames"] == 4
    assert summary["completed_assets"] == 1


def test_early_eof_emits_blocked_rows_for_every_remaining_requested_frame(
    tmp_path: Path,
) -> None:
    from tools.run_manifest_sam3_exhaustive import run_manifest_sam3_exhaustive

    manifest, _, _ = _write_manifest_inputs(tmp_path)
    output_dir = tmp_path / "eof"

    run_manifest_sam3_exhaustive(
        manifest=manifest,
        supplier="jdt",
        sam3_model=None,
        output_dir=output_dir,
        source_reader=_FakeSequentialSource(eof_after=2),
        segmenter=_FullMaskSegmenter(),
        checkpoint_every_frames=4,
    )

    frames = pd.read_csv(output_dir / "sam3_exhaustive_frame_results.csv")
    assert frames["source_frame"].tolist() == [1, 2, 3, 4]
    assert frames["frame_verdict"].tolist() == [
        "pass",
        "pass",
        "blocked",
        "blocked",
    ]
    assert frames.loc[2:, "frame_reason_codes"].str.contains("video_early_eof").all()
    failures = [
        json.loads(line)
        for line in (
            output_dir / "sam3_exhaustive_failures.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert any(row["reason"] == "video_early_eof" for row in failures)


def test_cli_uses_conservative_single_accelerator_defaults(tmp_path: Path) -> None:
    from tools.run_manifest_sam3_exhaustive import build_parser

    args = build_parser().parse_args(
        [
            "--manifest",
            str(tmp_path / "manifest.csv"),
            "--supplier",
            "jdt",
            "--sam3-model",
            str(tmp_path / "sam3"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert args.gpu_inference_workers == 1
    assert args.asset_inference_concurrency == 1
    assert args.decode_workers == 4
    assert args.prefetch_frames == 32
    assert args.writer_workers == 1


def test_cli_main_forwards_explicit_file_contract_and_runtime_controls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import tools.run_manifest_sam3_exhaustive as module

    captured: dict[str, object] = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"completed_assets": 1}

    monkeypatch.setattr(module, "run_manifest_sam3_exhaustive", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_manifest_sam3_exhaustive.py",
            "--manifest",
            str(tmp_path / "manifest.csv"),
            "--supplier",
            "jdt",
            "--sam3-model",
            str(tmp_path / "model"),
            "--output-dir",
            str(tmp_path / "out"),
            "--asset-ids",
            "jdt__a",
            "jdt__b",
            "--resume",
            "--decode-workers",
            "2",
            "--prefetch-frames",
            "8",
        ],
    )

    assert module.main() == 0
    assert captured["asset_ids"] == ["jdt__a", "jdt__b"]
    assert captured["resume"] is True
    assert captured["decode_workers"] == 2
    assert captured["prefetch_frames"] == 8
    assert json.loads(capsys.readouterr().out)["completed_assets"] == 1


def test_cuda_probe_uses_runtime_api_without_vendor_name_assumption() -> None:
    from tools.run_manifest_sam3_exhaustive import cuda_runtime_status

    class Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

    runtime = cuda_runtime_status(SimpleNamespace(cuda=Cuda()))

    assert runtime == {"cuda_available": True, "cuda_device_count": 1}


def test_producer_writes_the_complete_raw_artifact_contract(tmp_path: Path) -> None:
    from tools.run_manifest_sam3_exhaustive import run_manifest_sam3_exhaustive

    manifest, _, _ = _write_manifest_inputs(tmp_path, start_frame=0, end_frame=1)
    output_dir = tmp_path / "contract"

    run_manifest_sam3_exhaustive(
        manifest=manifest,
        supplier="jdt",
        sam3_model=None,
        output_dir=output_dir,
        source_reader=_FakeSequentialSource(),
        segmenter=_FullMaskSegmenter(),
        checkpoint_every_frames=1,
    )

    expected = {
        "sam3_exhaustive_hand_results.parquet",
        "sam3_exhaustive_hand_results.jsonl",
        "sam3_exhaustive_frame_results.parquet",
        "sam3_exhaustive_frame_results.csv",
        "sam3_exhaustive_failures.jsonl",
        "review_evidence_manifest.csv",
        "progress.json",
        "run_config.json",
    }
    assert expected.issubset({path.name for path in output_dir.iterdir()})
    run_config = json.loads((output_dir / "run_config.json").read_text())
    assert run_config["producer"] == "jdt-sam3-exhaustive-producer-v1"
    assert run_config["candidate_independent"] is True
    assert run_config["source_frame_stride"] == 1
    assert run_config["required_hands"] == ["left", "right"]
    assert run_config["thresholds"]["projected_in_image_ratio_threshold"] == 0.8
    assert run_config["thresholds"]["strong_inside_ratio_threshold"] == 0.2
    assert run_config["thresholds"]["acceptable_inside_ratio_threshold"] == 0.6
    timing = run_config["timing"]
    assert {
        "decode_seconds",
        "inference_seconds",
        "write_seconds",
        "checkpoint_write_seconds",
        "evidence_write_seconds",
        "final_materialization_write_seconds",
        "frames_per_second",
    } <= set(timing)
    assert timing["write_seconds"] == pytest.approx(
        timing["checkpoint_write_seconds"]
        + timing["evidence_write_seconds"]
        + timing["final_materialization_write_seconds"]
    )
    assert run_config["summary"]["overlay_bytes"] == 0
    assert run_config["summary"]["result_bytes"] > 0
    assert run_config["summary"]["per_asset_timing"][0]["asset_id"] == (
        "jdt__episode_000001"
    )
    assert "write_seconds" in run_config["summary"]["per_asset_timing"][0]
    progress = json.loads((output_dir / "progress.json").read_text())
    assert progress["status"] == "completed"
    hands = pd.read_parquet(output_dir / "sam3_exhaustive_hand_results.parquet")
    assert {
        "producer_version",
        "model_config_identity",
        "source_mapping_identity",
    }.issubset(hands.columns)
    frames = pd.read_parquet(output_dir / "sam3_exhaustive_frame_results.parquet")
    assert frames.loc[0, "source_video_path"].endswith("episode.mp4")
    assert frames.loc[0, "direct_2d_path"].endswith("episode.parquet")
    assert frames.loc[0, "coordinate_space"] == "source"
    assert float(frames.loc[0, "processing_seconds"]) >= 0.0


def test_invalid_direct_2d_required_hand_is_blocked_not_pass(
    tmp_path: Path,
) -> None:
    from tools.run_manifest_sam3_exhaustive import run_manifest_sam3_exhaustive

    manifest, _, parquet = _write_manifest_inputs(tmp_path, start_frame=0, end_frame=0)
    data = pd.read_parquet(parquet)
    data.at[0, "leftcam_left_kp2d"] = [1.0] * 40
    data.to_parquet(parquet, index=False)
    output_dir = tmp_path / "invalid"

    run_manifest_sam3_exhaustive(
        manifest=manifest,
        supplier="jdt",
        sam3_model=None,
        output_dir=output_dir,
        source_reader=_FakeSequentialSource(),
        segmenter=_FullMaskSegmenter(),
    )

    frame = pd.read_csv(output_dir / "sam3_exhaustive_frame_results.csv").iloc[0]
    assert frame["left_hand_verdict"] == "blocked"
    assert frame["right_hand_verdict"] == "pass"
    assert frame["frame_verdict"] == "blocked"
    assert "expected 42" in frame["frame_reason_codes"]


def test_matching_resume_reuses_completed_chunks_without_duplicate_inference(
    tmp_path: Path,
) -> None:
    from tools.run_manifest_sam3_exhaustive import run_manifest_sam3_exhaustive

    manifest, _, _ = _write_manifest_inputs(tmp_path)
    output_dir = tmp_path / "resume"
    run_manifest_sam3_exhaustive(
        manifest=manifest,
        supplier="jdt",
        sam3_model=None,
        output_dir=output_dir,
        source_reader=_FakeSequentialSource(),
        segmenter=_FullMaskSegmenter(),
        checkpoint_every_frames=2,
    )
    second_segmenter = _FullMaskSegmenter()

    summary = run_manifest_sam3_exhaustive(
        manifest=manifest,
        supplier="jdt",
        sam3_model=None,
        output_dir=output_dir,
        source_reader=_FakeSequentialSource(),
        segmenter=second_segmenter,
        checkpoint_every_frames=2,
        resume=True,
    )

    assert second_segmenter.calls == 0
    assert summary["reused_frame_count"] == 4
    frames = pd.read_csv(output_dir / "sam3_exhaustive_frame_results.csv")
    assert frames[["asset_id", "source_frame"]].drop_duplicates().shape[0] == 4


def test_multiple_checkpoint_chunks_use_one_sequential_video_decode_session(
    tmp_path: Path,
) -> None:
    from tools.run_manifest_sam3_exhaustive import run_manifest_sam3_exhaustive

    manifest, _, _ = _write_manifest_inputs(tmp_path, start_frame=0, end_frame=4)
    source = _FakeSequentialSource()

    run_manifest_sam3_exhaustive(
        manifest=manifest,
        supplier="jdt",
        sam3_model=None,
        output_dir=tmp_path / "one-decode-session",
        source_reader=source,
        segmenter=_FullMaskSegmenter(),
        checkpoint_every_frames=2,
    )

    assert source.iter_frames_calls == 1
    assert source.requested_video_frames == [0, 1, 2, 3, 4]


def test_changed_direct_2d_source_rejects_stale_resume(tmp_path: Path) -> None:
    from tools.run_manifest_sam3_exhaustive import (
        StaleResumeError,
        run_manifest_sam3_exhaustive,
    )

    manifest, _, parquet = _write_manifest_inputs(tmp_path)
    output_dir = tmp_path / "stale-source"
    kwargs = {
        "manifest": manifest,
        "supplier": "jdt",
        "sam3_model": None,
        "output_dir": output_dir,
        "source_reader": _FakeSequentialSource(),
        "segmenter": _FullMaskSegmenter(),
        "checkpoint_every_frames": 2,
    }
    run_manifest_sam3_exhaustive(**kwargs)
    source = pd.read_parquet(parquet)
    source.at[1, "leftcam_left_kp2d"] = _flat_hand(1.0)
    source.to_parquet(parquet, index=False)

    with pytest.raises(StaleResumeError, match="fingerprint"):
        run_manifest_sam3_exhaustive(
            **{
                **kwargs,
                "source_reader": _FakeSequentialSource(),
                "segmenter": _FullMaskSegmenter(),
                "resume": True,
            }
        )


def test_changed_model_identity_rejects_stale_resume(tmp_path: Path) -> None:
    from tools.run_manifest_sam3_exhaustive import (
        StaleResumeError,
        run_manifest_sam3_exhaustive,
    )

    manifest, _, _ = _write_manifest_inputs(tmp_path)
    model = tmp_path / "sam3"
    model.mkdir()
    (model / "config.json").write_text('{"version": 1}', encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights-metadata-v1")
    output_dir = tmp_path / "stale-model"
    run_manifest_sam3_exhaustive(
        manifest=manifest,
        supplier="jdt",
        sam3_model=model,
        output_dir=output_dir,
        source_reader=_FakeSequentialSource(),
        segmenter=_FullMaskSegmenter(),
        checkpoint_every_frames=2,
    )
    (model / "config.json").write_text('{"version": 2}', encoding="utf-8")

    with pytest.raises(StaleResumeError, match="fingerprint"):
        run_manifest_sam3_exhaustive(
            manifest=manifest,
            supplier="jdt",
            sam3_model=model,
            output_dir=output_dir,
            source_reader=_FakeSequentialSource(),
            segmenter=_FullMaskSegmenter(),
            checkpoint_every_frames=2,
            resume=True,
        )


def test_partial_asset_resume_recomputes_only_incomplete_chunk(tmp_path: Path) -> None:
    from tools.run_manifest_sam3_exhaustive import run_manifest_sam3_exhaustive

    manifest, _, _ = _write_manifest_inputs(tmp_path)
    output_dir = tmp_path / "partial"
    run_manifest_sam3_exhaustive(
        manifest=manifest,
        supplier="jdt",
        sam3_model=None,
        output_dir=output_dir,
        source_reader=_FakeSequentialSource(),
        segmenter=_FullMaskSegmenter(),
        checkpoint_every_frames=2,
    )
    chunks = sorted((output_dir / "asset_chunks").glob("*/chunk_*"))
    assert len(chunks) == 2
    (chunks[1] / "completion.json").unlink()
    resumed_segmenter = _FullMaskSegmenter()

    summary = run_manifest_sam3_exhaustive(
        manifest=manifest,
        supplier="jdt",
        sam3_model=None,
        output_dir=output_dir,
        source_reader=_FakeSequentialSource(),
        segmenter=resumed_segmenter,
        checkpoint_every_frames=2,
        resume=True,
    )

    assert resumed_segmenter.calls == 2
    assert summary["reused_frame_count"] == 2
    assert summary["computed_frame_count"] == 2
    frames = pd.read_csv(output_dir / "sam3_exhaustive_frame_results.csv")
    assert frames["source_frame"].tolist() == [1, 2, 3, 4]


def test_resume_recomputes_chunk_with_wrong_frame_identity_even_if_counts_match(
    tmp_path: Path,
) -> None:
    from tools.run_manifest_sam3_exhaustive import run_manifest_sam3_exhaustive

    manifest, _, _ = _write_manifest_inputs(tmp_path, start_frame=0, end_frame=1)
    output_dir = tmp_path / "wrong-frame-identity"
    run_manifest_sam3_exhaustive(
        manifest=manifest,
        supplier="jdt",
        sam3_model=None,
        output_dir=output_dir,
        source_reader=_FakeSequentialSource(),
        segmenter=_FullMaskSegmenter(),
        checkpoint_every_frames=2,
    )
    chunk = next((output_dir / "asset_chunks").glob("*/chunk_*"))
    frame_path = chunk / "frame_results.jsonl"
    rows = _read_jsonl_for_test(frame_path)
    rows[0]["source_frame"] = 99
    frame_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    segmenter = _FullMaskSegmenter()

    summary = run_manifest_sam3_exhaustive(
        manifest=manifest,
        supplier="jdt",
        sam3_model=None,
        output_dir=output_dir,
        source_reader=_FakeSequentialSource(),
        segmenter=segmenter,
        checkpoint_every_frames=2,
        resume=True,
    )

    assert segmenter.calls == 2
    assert summary["reused_frame_count"] == 0
    assert pd.read_csv(output_dir / "sam3_exhaustive_frame_results.csv")[
        "source_frame"
    ].tolist() == [0, 1]


def test_fail_and_review_frames_get_deterministic_evidence_but_pass_does_not(
    tmp_path: Path,
) -> None:
    from tools.run_manifest_sam3_exhaustive import run_manifest_sam3_exhaustive

    manifest, _, _ = _write_manifest_inputs(tmp_path, start_frame=0, end_frame=1)
    output_dir = tmp_path / "evidence"
    writes: list[tuple[str, int]] = []

    def overlay_writer(*, clip_id, frame_idx, output_dir, **kwargs):
        del kwargs
        writes.append((clip_id, frame_idx))
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{clip_id}_source_{frame_idx:012d}.png"
        path.write_bytes(b"png")
        return path

    run_manifest_sam3_exhaustive(
        manifest=manifest,
        supplier="jdt",
        sam3_model=None,
        output_dir=output_dir,
        source_reader=_FakeSequentialSource(),
        segmenter=_EmptyMaskSegmenter(),
        overlay_writer=overlay_writer,
        save_positive_overlays=True,
    )

    evidence = pd.read_csv(output_dir / "review_evidence_manifest.csv")
    assert evidence["source_frame"].tolist() == [0, 1]
    assert evidence["sam3_frame_verdict"].tolist() == ["review", "review"]
    assert len(writes) == 2
    assert "__source_000000000000__video_000000000000" in writes[0][0]
    assert "__source_000000000001__video_000000000001" in writes[1][0]
    assert evidence["overlay_path"].map(Path).map(Path.is_file).all()

    pass_output = tmp_path / "pass-evidence"
    run_manifest_sam3_exhaustive(
        manifest=manifest,
        supplier="jdt",
        sam3_model=None,
        output_dir=pass_output,
        source_reader=_FakeSequentialSource(),
        segmenter=_FullMaskSegmenter(),
        overlay_writer=overlay_writer,
        save_positive_overlays=True,
    )
    assert pd.read_csv(pass_output / "review_evidence_manifest.csv").empty


def test_sam3_runtime_error_is_unevaluable_and_does_not_abort_asset(
    tmp_path: Path,
) -> None:
    from tools.run_manifest_sam3_exhaustive import run_manifest_sam3_exhaustive

    class FailingOnceSegmenter(_FullMaskSegmenter):
        def segment_frame(self, frame, queries, config):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("synthetic inference failure")
            return super().segment_frame(frame, queries, config)

    manifest, _, _ = _write_manifest_inputs(tmp_path, start_frame=0, end_frame=1)
    output_dir = tmp_path / "runtime-error"
    run_manifest_sam3_exhaustive(
        manifest=manifest,
        supplier="jdt",
        sam3_model=None,
        output_dir=output_dir,
        source_reader=_FakeSequentialSource(),
        segmenter=FailingOnceSegmenter(),
    )

    frames = pd.read_csv(output_dir / "sam3_exhaustive_frame_results.csv")
    assert frames["frame_verdict"].tolist() == ["blocked", "pass"]
    failures = _read_jsonl_for_test(output_dir / "sam3_exhaustive_failures.jsonl")
    assert failures[0]["stage"] == "sam3_inference"
    assert "synthetic inference failure" in failures[0]["reason"]


def _read_jsonl_for_test(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_asset_selection_and_shards_are_stable_disjoint_and_complete() -> None:
    from tools.run_manifest_sam3_exhaustive import select_asset_ids

    assets = ["jdt__z", "jdt__a", "jdt__m", "jdt__b", "jdt__q"]
    shard_zero = select_asset_ids(
        assets, requested=None, max_assets=None, num_shards=2, shard_index=0
    )
    shard_one = select_asset_ids(
        assets, requested=None, max_assets=None, num_shards=2, shard_index=1
    )

    assert shard_zero == ["jdt__a", "jdt__m", "jdt__z"]
    assert shard_one == ["jdt__b", "jdt__q"]
    assert set(shard_zero).isdisjoint(shard_one)
    assert set(shard_zero) | set(shard_one) == set(assets)


def test_asset_ids_and_max_assets_limit_before_model_loading(tmp_path: Path) -> None:
    from tools.run_manifest_sam3_exhaustive import run_manifest_sam3_exhaustive

    manifest, video, parquet = _write_manifest_inputs(tmp_path, start_frame=0, end_frame=0)
    original = pd.read_csv(manifest).iloc[0].to_dict()
    rows = []
    for suffix in ("c", "a", "b"):
        rows.append(
            {
                **original,
                "asset_id": f"jdt__{suffix}",
                "primary_video_path": str(video),
                "parquet_path": str(parquet),
            }
        )
    pd.DataFrame(rows).to_csv(manifest, index=False)
    segmenter = _FullMaskSegmenter()

    summary = run_manifest_sam3_exhaustive(
        manifest=manifest,
        supplier="jdt",
        sam3_model=None,
        output_dir=tmp_path / "limited",
        source_reader=_FakeSequentialSource(),
        segmenter=segmenter,
        asset_ids=("jdt__b", "jdt__a"),
        max_assets=1,
    )

    assert summary["selected_asset_ids"] == ["jdt__a"]
    assert segmenter.calls == 1


def test_video_frame_out_of_range_is_blocked_before_inference(tmp_path: Path) -> None:
    from tools.run_manifest_sam3_exhaustive import run_manifest_sam3_exhaustive

    class ShortVideoSource(_FakeSequentialSource):
        def video_metadata(self, path: Path) -> dict[str, float | int]:
            del path
            return {"frame_count": 2, "width": 96, "height": 64, "fps": 30.0}

    manifest, _, _ = _write_manifest_inputs(tmp_path, start_frame=1, end_frame=3)
    segmenter = _FullMaskSegmenter()
    output_dir = tmp_path / "short-video"
    run_manifest_sam3_exhaustive(
        manifest=manifest,
        supplier="jdt",
        sam3_model=None,
        output_dir=output_dir,
        source_reader=ShortVideoSource(),
        segmenter=segmenter,
    )

    frames = pd.read_csv(output_dir / "sam3_exhaustive_frame_results.csv")
    assert frames["frame_verdict"].tolist() == ["pass", "blocked", "blocked"]
    assert frames.loc[1:, "frame_reason_codes"].str.contains(
        "video_frame_out_of_range"
    ).all()
    assert segmenter.calls == 1


def test_cli_allows_safe_runtime_controls_to_be_overridden(tmp_path: Path) -> None:
    from tools.run_manifest_sam3_exhaustive import build_parser

    args = build_parser().parse_args(
        [
            "--manifest",
            str(tmp_path / "manifest.csv"),
            "--supplier",
            "jdt",
            "--sam3-model",
            str(tmp_path / "model"),
            "--output-dir",
            str(tmp_path / "out"),
            "--decode-workers",
            "2",
            "--prefetch-frames",
            "8",
            "--writer-workers",
            "2",
            "--gpu-inference-workers",
            "2",
            "--asset-inference-concurrency",
            "2",
        ]
    )

    assert args.decode_workers == 2
    assert args.prefetch_frames == 8
    assert args.writer_workers == 2
    assert args.gpu_inference_workers == 2
    assert args.asset_inference_concurrency == 2


def test_sequential_source_seeks_once_then_decodes_forward() -> None:
    from tools.run_manifest_sam3_exhaustive import SequentialVideoSource

    class Capture:
        def __init__(self) -> None:
            self.position = 0
            self.set_calls: list[tuple[int, int]] = []
            self.read_count = 0
            self.released = False

        def isOpened(self) -> bool:
            return True

        def set(self, prop: int, value: int) -> bool:
            self.set_calls.append((prop, value))
            self.position = value
            return True

        def read(self):
            value = self.position
            self.position += 1
            self.read_count += 1
            frame = np.zeros((2, 3, 3), dtype=np.uint8)
            frame[..., 0] = value  # B
            frame[..., 2] = value + 10  # R
            return True, frame

        def get(self, prop: int) -> float:
            return {7: 20.0, 3: 3.0, 4: 2.0, 5: 30.0}[prop]

        def release(self) -> None:
            self.released = True

    captures: list[Capture] = []

    def factory(path: Path):
        del path
        capture = Capture()
        captures.append(capture)
        return capture

    source = SequentialVideoSource(
        decode_workers=4,
        prefetch_frames=2,
        capture_factory=factory,
    )
    decoded = list(source.iter_frames(Path("video.mp4"), [3, 5]))

    assert [row[0] for row in decoded] == [3, 5]
    assert captures[0].set_calls == [(1, 3)]
    assert captures[0].read_count == 3
    assert decoded[0][1][0, 0].tolist() == [13, 0, 3]
    assert captures[0].released is True


def test_run_config_records_requested_and_effective_runtime_controls(
    tmp_path: Path,
) -> None:
    from tools.run_manifest_sam3_exhaustive import run_manifest_sam3_exhaustive

    manifest, _, _ = _write_manifest_inputs(tmp_path, start_frame=0, end_frame=0)
    output_dir = tmp_path / "runtime-controls"
    run_manifest_sam3_exhaustive(
        manifest=manifest,
        supplier="jdt",
        sam3_model=None,
        output_dir=output_dir,
        source_reader=_FakeSequentialSource(),
        segmenter=_FullMaskSegmenter(),
        decode_workers=2,
        prefetch_frames=8,
        writer_workers=2,
        gpu_inference_workers=2,
        asset_inference_concurrency=2,
    )

    runtime = json.loads((output_dir / "run_config.json").read_text())["runtime"]
    assert runtime["requested"] == {
        "decode_workers": 2,
        "prefetch_frames": 8,
        "writer_workers": 2,
        "gpu_inference_workers": 2,
        "asset_inference_concurrency": 2,
    }
    assert runtime["effective"]["gpu_inference_workers"] == 1
    assert runtime["effective"]["asset_inference_concurrency"] == 1
    assert runtime["effective"]["writer_workers"] == 2
    assert runtime["single_model_instance"] is True
