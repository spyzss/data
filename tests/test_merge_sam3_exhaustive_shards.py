from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tests.test_manifest_sam3_exhaustive_runner import (
    _FakeSequentialSource,
    _FullMaskSegmenter,
    _write_manifest_inputs,
)


def _write_shard(
    root: Path,
    *,
    shard_index: int,
    assets: tuple[str, ...],
    config_hash: str = "config-a",
) -> None:
    root.mkdir(parents=True)
    frame_rows = [
        {
            "asset_id": asset_id,
            "supplier": "jdt",
            "source_frame": frame,
            "video_frame": frame,
            "frame_verdict": "pass",
        }
        for asset_id in assets
        for frame in (0, 1)
    ]
    hand_rows = [
        {**row, "hand": hand, "hand_verdict": "pass"}
        for row in frame_rows
        for hand in ("left", "right")
    ]
    pd.DataFrame(frame_rows).to_parquet(
        root / "sam3_exhaustive_frame_results.parquet", index=False
    )
    pd.DataFrame(frame_rows).to_csv(
        root / "sam3_exhaustive_frame_results.csv", index=False
    )
    pd.DataFrame(hand_rows).to_parquet(
        root / "sam3_exhaustive_hand_results.parquet", index=False
    )
    (root / "sam3_exhaustive_hand_results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in hand_rows), encoding="utf-8"
    )
    (root / "sam3_exhaustive_failures.jsonl").write_text("", encoding="utf-8")
    pd.DataFrame(
        columns=[
            "asset_id",
            "source_frame",
            "video_frame",
            "sam3_frame_verdict",
            "source_video",
            "overlay_path",
            "left_inside_ratio",
            "right_inside_ratio",
            "model_config_identity",
        ]
    ).to_csv(root / "review_evidence_manifest.csv", index=False)
    run_config = {
        "producer": "jdt-sam3-exhaustive-producer-v1",
        "status": "completed",
        "candidate_independent": True,
        "source_frame_stride": 1,
        "fingerprint": {
            "producer": "jdt-sam3-exhaustive-producer-v1",
            "manifest": {"sha256": "manifest-hash"},
            "selected_asset_ids": list(assets),
            "assets": {asset_id: {"identity": asset_id} for asset_id in assets},
            "model": {"resolved_path": "/model", "files": {}},
            "config_hash": config_hash,
            "frame_thresholds": {"acceptable": 0.6},
            "sam3_runtime_config": {"confidence_threshold": 0.5},
            "queries": ["hand"],
            "required_hands": ["left", "right"],
            "source_video_mapping": "jd_identity_source_frame_equals_video_frame_v1",
            "checkpoint_every_frames": 1000,
            "evidence": {"save_positive_overlays": True},
            "fingerprint_sha256": f"shard-{shard_index}",
        },
        "summary": {
            "selected_asset_ids": list(assets),
            "num_shards": 2,
            "shard_index": shard_index,
            "completed_assets": len(assets),
            "failed_assets": 0,
            "total_source_frames": len(frame_rows),
        },
        "timing": {
            "decode_seconds": 1.0,
            "inference_seconds": 2.0,
            "write_seconds": 0.5,
            "total_seconds": 3.5,
        },
    }
    (root / "run_config.json").write_text(
        json.dumps(run_config), encoding="utf-8"
    )


def _manifest(path: Path, assets: tuple[str, ...]) -> Path:
    pd.DataFrame(
        [
            {"asset_id": value, "start_frame": 0, "end_frame": 1}
            for value in assets
        ]
    ).to_csv(path, index=False)
    return path


def test_merge_shards_is_deterministic_disjoint_and_covers_manifest(tmp_path: Path) -> None:
    from tools.merge_sam3_exhaustive_shards import merge_shards

    shard0 = tmp_path / "shard0"
    shard1 = tmp_path / "shard1"
    _write_shard(shard0, shard_index=0, assets=("jdt__a", "jdt__c"))
    _write_shard(shard1, shard_index=1, assets=("jdt__b",))
    output = tmp_path / "merged"

    summary = merge_shards(
        shard_dirs=(shard1, shard0),
        manifest=_manifest(tmp_path / "manifest.csv", ("jdt__a", "jdt__b", "jdt__c")),
        output_dir=output,
    )

    frames = pd.read_csv(output / "sam3_exhaustive_frame_results.csv")
    assert frames[["asset_id", "source_frame"]].to_dict(orient="records") == [
        {"asset_id": "jdt__a", "source_frame": 0},
        {"asset_id": "jdt__a", "source_frame": 1},
        {"asset_id": "jdt__b", "source_frame": 0},
        {"asset_id": "jdt__b", "source_frame": 1},
        {"asset_id": "jdt__c", "source_frame": 0},
        {"asset_id": "jdt__c", "source_frame": 1},
    ]
    assert summary["completed_assets"] == 3
    assert summary["total_source_frames"] == 6
    merged_config = json.loads((output / "run_config.json").read_text())
    assert merged_config["merge"]["compatible_config_identity"]


def test_merge_rejects_overlapping_asset_frame_rows(tmp_path: Path) -> None:
    from tools.merge_sam3_exhaustive_shards import merge_shards

    shard0 = tmp_path / "shard0"
    shard1 = tmp_path / "shard1"
    _write_shard(shard0, shard_index=0, assets=("jdt__a",))
    _write_shard(shard1, shard_index=1, assets=("jdt__a",))

    with pytest.raises(ValueError, match="overlapping asset/source-frame"):
        merge_shards(
            shard_dirs=(shard0, shard1),
            manifest=_manifest(tmp_path / "manifest.csv", ("jdt__a",)),
            output_dir=tmp_path / "merged",
        )


def test_merge_rejects_missing_manifest_assets(tmp_path: Path) -> None:
    from tools.merge_sam3_exhaustive_shards import merge_shards

    shard = tmp_path / "shard0"
    _write_shard(shard, shard_index=0, assets=("jdt__a",))

    with pytest.raises(ValueError, match="missing manifest assets.*jdt__b"):
        merge_shards(
            shard_dirs=(shard,),
            manifest=_manifest(tmp_path / "manifest.csv", ("jdt__a", "jdt__b")),
            output_dir=tmp_path / "merged",
        )


def test_merge_rejects_missing_manifest_source_frames(tmp_path: Path) -> None:
    from tools.merge_sam3_exhaustive_shards import merge_shards

    shard = tmp_path / "shard0"
    _write_shard(shard, shard_index=0, assets=("jdt__a",))
    frames = pd.read_parquet(shard / "sam3_exhaustive_frame_results.parquet")
    frames.loc[frames["source_frame"] == 0].to_parquet(
        shard / "sam3_exhaustive_frame_results.parquet", index=False
    )

    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [{"asset_id": "jdt__a", "start_frame": 0, "end_frame": 1}]
    ).to_csv(manifest, index=False)

    with pytest.raises(ValueError, match="missing manifest source frames"):
        merge_shards(
            shard_dirs=(shard,),
            manifest=manifest,
            output_dir=tmp_path / "merged",
        )


def test_merge_rejects_one_asset_split_across_shards(tmp_path: Path) -> None:
    from tools.merge_sam3_exhaustive_shards import merge_shards

    shard0 = tmp_path / "shard0"
    shard1 = tmp_path / "shard1"
    _write_shard(shard0, shard_index=0, assets=("jdt__a",))
    _write_shard(shard1, shard_index=1, assets=("jdt__a",))
    for shard, source_frame in ((shard0, 0), (shard1, 1)):
        frames = pd.read_parquet(shard / "sam3_exhaustive_frame_results.parquet")
        frames.loc[frames["source_frame"] == source_frame].to_parquet(
            shard / "sam3_exhaustive_frame_results.parquet", index=False
        )
        hands = pd.read_parquet(shard / "sam3_exhaustive_hand_results.parquet")
        hands.loc[hands["source_frame"] == source_frame].to_parquet(
            shard / "sam3_exhaustive_hand_results.parquet", index=False
        )

    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [{"asset_id": "jdt__a", "start_frame": 0, "end_frame": 1}]
    ).to_csv(manifest, index=False)

    with pytest.raises(ValueError, match="asset appears in multiple shards"):
        merge_shards(
            shard_dirs=(shard0, shard1),
            manifest=manifest,
            output_dir=tmp_path / "merged",
        )


def test_merge_rejects_missing_required_hand_rows(tmp_path: Path) -> None:
    from tools.merge_sam3_exhaustive_shards import merge_shards

    shard = tmp_path / "shard0"
    _write_shard(shard, shard_index=0, assets=("jdt__a",))
    hands = pd.read_parquet(shard / "sam3_exhaustive_hand_results.parquet")
    hands.loc[
        ~((hands["source_frame"] == 1) & (hands["hand"] == "right"))
    ].to_parquet(shard / "sam3_exhaustive_hand_results.parquet", index=False)

    with pytest.raises(ValueError, match="missing required hand rows"):
        merge_shards(
            shard_dirs=(shard,),
            manifest=_manifest(tmp_path / "manifest.csv", ("jdt__a",)),
            output_dir=tmp_path / "merged",
        )


def test_merge_rejects_incompatible_model_or_config_identity(tmp_path: Path) -> None:
    from tools.merge_sam3_exhaustive_shards import merge_shards

    shard0 = tmp_path / "shard0"
    shard1 = tmp_path / "shard1"
    _write_shard(shard0, shard_index=0, assets=("jdt__a",), config_hash="a")
    _write_shard(shard1, shard_index=1, assets=("jdt__b",), config_hash="b")

    with pytest.raises(ValueError, match="incompatible shard run configuration"):
        merge_shards(
            shard_dirs=(shard0, shard1),
            manifest=_manifest(tmp_path / "manifest.csv", ("jdt__a", "jdt__b")),
            output_dir=tmp_path / "merged",
        )


def test_real_small_shard_merge_rows_equal_unsharded_producer(tmp_path: Path) -> None:
    from tools.merge_sam3_exhaustive_shards import merge_shards
    from tools.run_manifest_sam3_exhaustive import run_manifest_sam3_exhaustive

    manifest, video, parquet = _write_manifest_inputs(
        tmp_path, start_frame=0, end_frame=1
    )
    template = pd.read_csv(manifest).iloc[0].to_dict()
    pd.DataFrame(
        [
            {
                **template,
                "asset_id": asset_id,
                "episode_index": index,
                "primary_video_path": str(video),
                "parquet_path": str(parquet),
            }
            for index, asset_id in enumerate(("jdt__a", "jdt__b", "jdt__c"))
        ]
    ).to_csv(manifest, index=False)
    common = {
        "manifest": manifest,
        "supplier": "jdt",
        "sam3_model": None,
        "segmenter": _FullMaskSegmenter(),
    }
    unsharded = tmp_path / "unsharded"
    run_manifest_sam3_exhaustive(
        **common,
        output_dir=unsharded,
        source_reader=_FakeSequentialSource(),
    )
    shard_root = tmp_path / "shards"
    for shard_index in (0, 1):
        run_manifest_sam3_exhaustive(
            **{**common, "segmenter": _FullMaskSegmenter()},
            output_dir=shard_root,
            source_reader=_FakeSequentialSource(),
            num_shards=2,
            shard_index=shard_index,
        )
    merged = tmp_path / "merged-real"
    merge_shards(
        shard_dirs=(
            shard_root / "shard-00000-of-00002",
            shard_root / "shard-00001-of-00002",
        ),
        manifest=manifest,
        output_dir=merged,
    )

    unsharded_frames = pd.read_csv(
        unsharded / "sam3_exhaustive_frame_results.csv"
    )
    merged_frames = pd.read_csv(merged / "sam3_exhaustive_frame_results.csv")
    semantic_columns = [
        "asset_id",
        "source_frame",
        "video_frame",
        "local_frame",
        "left_hand_verdict",
        "right_hand_verdict",
        "frame_verdict",
        "frame_reason_codes",
        "required_hand_count",
        "evaluable_hand_count",
        "blocked_hand_count",
    ]
    pd.testing.assert_frame_equal(
        merged_frames[semantic_columns], unsharded_frames[semantic_columns]
    )
    assert merged_frames["model_config_identity"].nunique() == 1
