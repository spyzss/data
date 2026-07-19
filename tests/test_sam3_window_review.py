from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from human_qc.sam3_window_review import load_review_bundle


PNG_BYTES = b"\x89PNG\r\n\x1a\nfixture"


def _write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _queue_row(
    review_id: str,
    *,
    asset_id: str = "episode_000001",
    start: int = 0,
    end: int = 20,
    state: str = "review",
) -> dict:
    return {
        "review_id": review_id,
        "supplier_id": "jdt",
        "asset_id": asset_id,
        "window_start_frame": start,
        "window_end_frame": end,
        "source_start_frame": start,
        "source_end_frame": end,
        "frame_coordinate_system": "source_inclusive",
        "sam3_window_state": state,
        "left_window_containment_verdict": "review",
        "right_window_containment_verdict": "pass",
        "sampled_frame_indices_json": json.dumps([start, start + 5, start + 10, start + 15, end]),
        "video_path": "source/jdt/video.mp4",
        "fps": 30.0,
        "trigger_reason_json": json.dumps(["containment_mismatch"]),
        "trigger_metrics_json": json.dumps({"outside_ratio": 0.4}),
    }


def _evidence_rows(
    root: Path,
    review_id: str,
    *,
    asset_id: str = "episode_000001",
    start: int = 0,
    end: int = 20,
    exact: bool = True,
    frames: tuple[int, ...] = (0, 5, 10, 15, 20),
) -> list[dict]:
    rows = []
    for frame_idx in frames:
        source = root / "cloud evidence with spaces" / review_id / f"frame {frame_idx}.png"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(PNG_BYTES)
        rows.append(
            {
                "review_id": review_id if exact else "",
                "supplier_id": "jdt",
                "asset_id": asset_id,
                "window_start_frame": start,
                "window_end_frame": end,
                "frame_idx": frame_idx,
                "source_module": "sam3_containment",
                "evidence_type": "combined_overlay",
                "hand_side": "both",
                "source_path": str(source),
                "metadata_json": json.dumps({"source_frame_idx": frame_idx}),
            }
        )
    return rows


def _load(
    tmp_path: Path,
    queue_rows: list[dict],
    evidence_rows: list[dict],
):
    manifest_assets = sorted({str(row["asset_id"]) for row in queue_rows})
    manifest = _write_csv(
        tmp_path / "manifest.csv",
        [
            {"supplier_id": "jdt", "asset_id": asset_id, "schema_version": "jdt.v1"}
            for asset_id in manifest_assets
        ],
    )
    queue = _write_csv(tmp_path / "review_queue.csv", queue_rows)
    evidence = _write_csv(tmp_path / "evidence.csv", evidence_rows)
    return load_review_bundle(
        manifest_path=manifest,
        queue_path=queue,
        evidence_path=evidence,
        review_dir=tmp_path / "review dir",
    )


def test_filters_381_automatic_pass_rows_and_preserves_86_review_ids(tmp_path: Path) -> None:
    queue = [
        _queue_row(f"jdt__episode_{index:06d}__window_0_20", asset_id=f"episode_{index:06d}")
        for index in range(86)
    ]
    queue.extend(
        _queue_row(
            f"jdt__episode_{index:06d}__window_0_20",
            asset_id=f"episode_{index:06d}",
            state="pass",
        )
        for index in range(86, 467)
    )

    bundle = _load(tmp_path, queue, [])

    assert len(bundle.items) == 86
    assert [item["review_id"] for item in bundle.items] == [row["review_id"] for row in queue[:86]]
    assert all(item["manual_review"]["verdict"] is None for item in bundle.items)
    assert all(item["manual_review"]["status"] == "unresolved" for item in bundle.items)


def test_missing_review_id_is_generated_once_without_double_supplier_prefix(tmp_path: Path) -> None:
    queue = [_queue_row("", asset_id="jdt__episode_000001", start=0, end=5760)]
    evidence = _evidence_rows(
        tmp_path,
        "ignored",
        asset_id="jdt__episode_000001",
        start=0,
        end=5760,
        exact=False,
        frames=(0, 1440, 2880, 4320, 5760),
    )

    bundle = _load(tmp_path, queue, evidence)

    assert bundle.items[0]["review_id"] == "jdt__episode_000001__window_0_5760"
    assert "jdt__jdt__" not in bundle.items[0]["review_id"]


def test_duplicate_review_ids_are_rejected_even_for_same_asset(tmp_path: Path) -> None:
    queue = [
        _queue_row("jdt__episode_000001__window_0_20"),
        _queue_row(" jdt__episode_000001__window_0_20 ", start=30, end=50),
    ]

    with pytest.raises(ValueError, match="duplicate review_id"):
        _load(tmp_path, queue, [])


def test_exact_review_id_evidence_wins_over_exact_composite_fallback(tmp_path: Path) -> None:
    review_id = "jdt__episode_000001__window_0_20"
    exact = _evidence_rows(tmp_path, review_id, frames=(0, 5, 10, 15, 20))
    fallback = _evidence_rows(tmp_path, "fallback", exact=False, frames=(1, 6, 11, 16, 19))

    bundle = _load(tmp_path, [_queue_row(review_id)], [*fallback, *exact])

    item = bundle.items[0]
    assert [row["frame_idx"] for row in item["evidence"]] == [0, 5, 10, 15, 20]
    assert all(row["match_kind"] == "review_id" for row in item["evidence"])


def test_composite_fallback_is_exact_and_never_uses_overlapping_window(tmp_path: Path) -> None:
    review_id = "jdt__episode_000001__window_0_20"
    exact_fallback = _evidence_rows(tmp_path, "fallback", exact=False)
    overlapping = _evidence_rows(
        tmp_path,
        "overlap",
        exact=False,
        start=10,
        end=30,
        frames=(10, 15, 20, 25, 30),
    )

    bundle = _load(tmp_path, [_queue_row(review_id)], [*overlapping, *exact_fallback])

    assert [row["frame_idx"] for row in bundle.items[0]["evidence"]] == [0, 5, 10, 15, 20]


def test_same_asset_windows_remain_independent_queue_items(tmp_path: Path) -> None:
    first = "jdt__episode_000001__window_0_20"
    second = "jdt__episode_000001__window_30_50"
    queue = [_queue_row(first), _queue_row(second, start=30, end=50)]
    evidence = [
        *_evidence_rows(tmp_path, first),
        *_evidence_rows(tmp_path, second, start=30, end=50, frames=(30, 35, 40, 45, 50)),
    ]

    bundle = _load(tmp_path, queue, evidence)

    assert [item["review_id"] for item in bundle.items] == [first, second]
    assert bundle.items[0]["asset_id"] == bundle.items[1]["asset_id"]
    assert bundle.items[0]["window_start_frame"] != bundle.items[1]["window_start_frame"]


def test_stages_five_whitelisted_overlays_and_preserves_source_provenance(tmp_path: Path) -> None:
    review_id = "jdt__episode_000001__window_0_20"
    evidence = _evidence_rows(tmp_path, review_id)

    bundle = _load(tmp_path, [_queue_row(review_id)], evidence)

    item = bundle.items[0]
    assert item["evidence_status"] == "ready"
    assert item["can_review"] is True
    assert len(item["evidence"]) == 5
    assert len(bundle.allowed_asset_paths) == 5
    for source, staged in zip(evidence, item["evidence"], strict=True):
        assert staged["status"] == "ready"
        assert staged["source_path"] == source["source_path"]
        assert staged["url"].startswith("/assets/")
        assert "/mnt/" not in staged["url"]
        relative = staged["url"].removeprefix("/assets/")
        assert relative in bundle.allowed_asset_paths
        assert (bundle.assets_root / relative).read_bytes() == PNG_BYTES


def test_missing_or_wrong_provenance_evidence_blocks_review_without_becoming_pass(
    tmp_path: Path,
) -> None:
    review_id = "jdt__episode_000001__window_0_20"
    evidence = _evidence_rows(tmp_path, review_id)
    Path(evidence[0]["source_path"]).unlink()
    evidence[1]["source_module"] = "keypoint_temporal"

    bundle = _load(tmp_path, [_queue_row(review_id)], evidence)

    item = bundle.items[0]
    assert item["evidence_status"] == "error"
    assert item["can_review"] is False
    assert item["manual_review"] == {"status": "unresolved", "verdict": None}
    assert item["evidence"][0]["status"] == "missing"
    assert item["evidence"][1]["status"] == "invalid_provenance"
