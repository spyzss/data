#!/usr/bin/env python3
"""Validate and merge deterministic JD exhaustive SAM3 process shards."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from qc_pipeline.artifacts import canonical_sha256
from tools.run_manifest_sam3_containment import read_records
from tools.run_manifest_sam3_exhaustive import EVIDENCE_COLUMNS, _atomic_write_json, _write_jsonl


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _compatibility_identity(run_config: Mapping[str, Any]) -> dict[str, Any]:
    fingerprint = dict(run_config.get("fingerprint", {}))
    for key in ("selected_asset_ids", "assets", "fingerprint_sha256"):
        fingerprint.pop(key, None)
    return {
        "producer": run_config.get("producer"),
        "candidate_independent": run_config.get("candidate_independent"),
        "source_frame_stride": run_config.get("source_frame_stride"),
        "fingerprint_common": fingerprint,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _reject_duplicates(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    duplicate = frame.duplicated(columns, keep=False)
    if bool(duplicate.any()):
        examples = frame.loc[duplicate, columns].head(5).to_dict(orient="records")
        raise ValueError(f"overlapping {label} rows: {examples}")


def _manifest_ranges(manifest: Path) -> dict[str, tuple[int, int]]:
    ranges: dict[str, tuple[int, int]] = {}
    for row in read_records(Path(manifest)):
        asset_id = str(row.get("asset_id", "")).strip()
        if not asset_id:
            raise ValueError("manifest row is missing asset_id")
        if asset_id in ranges:
            raise ValueError(f"duplicate manifest asset_id: {asset_id}")
        try:
            start = int(row["start_frame"])
            end = int(row["end_frame"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"manifest asset requires integer start_frame/end_frame: {asset_id}"
            ) from exc
        if end < start:
            raise ValueError(f"manifest source-frame range is reversed: {asset_id}")
        ranges[asset_id] = (start, end)
    return ranges


def _validate_full_coverage(
    frames: pd.DataFrame,
    hands: pd.DataFrame,
    *,
    manifest_ranges: Mapping[str, tuple[int, int]],
) -> None:
    for asset_id, (start, end) in sorted(manifest_ranges.items()):
        actual = frames.loc[frames["asset_id"].astype(str) == asset_id, "source_frame"]
        numeric = pd.to_numeric(actual, errors="coerce")
        if bool(numeric.isna().any()) or bool((numeric % 1 != 0).any()):
            raise ValueError(f"invalid source frames for manifest asset: {asset_id}")
        values = sorted(int(value) for value in numeric.tolist())
        expected_count = end - start + 1
        complete = (
            len(values) == expected_count
            and bool(values)
            and values[0] == start
            and values[-1] == end
        )
        if not complete:
            actual_set = set(values)
            missing = [
                frame for frame in range(start, end + 1) if frame not in actual_set
            ][:5]
            raise ValueError(
                "missing manifest source frames: "
                f"asset_id={asset_id}, expected={start}..{end}, "
                f"actual_count={len(values)}, missing_examples={missing}"
            )

    valid_hands = hands["hand"].astype(str).isin({"left", "right"})
    if not bool(valid_hands.all()):
        invalid = sorted(set(hands.loc[~valid_hands, "hand"].astype(str)))
        raise ValueError(f"unexpected hand labels in shard rows: {invalid[:5]}")
    hand_counts = (
        hands.groupby(["asset_id", "source_frame"], sort=False)["hand"]
        .nunique()
        .rename("required_hand_count")
        .reset_index()
    )
    frame_keys = frames[["asset_id", "source_frame"]]
    coverage = frame_keys.merge(
        hand_counts,
        on=["asset_id", "source_frame"],
        how="left",
        validate="one_to_one",
    )
    missing_hands = coverage["required_hand_count"].fillna(0).astype(int) != 2
    extra_hand_frames = hands[["asset_id", "source_frame"]].drop_duplicates().merge(
        frame_keys,
        on=["asset_id", "source_frame"],
        how="left",
        indicator=True,
    )
    extra_hand_frames = extra_hand_frames[extra_hand_frames["_merge"] == "left_only"]
    if bool(missing_hands.any()) or not extra_hand_frames.empty:
        examples = coverage.loc[
            missing_hands, ["asset_id", "source_frame", "required_hand_count"]
        ].head(5).to_dict(orient="records")
        examples.extend(
            extra_hand_frames[["asset_id", "source_frame"]]
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(f"missing required hand rows: {examples}")


def merge_shards(
    *,
    shard_dirs: Sequence[Path],
    manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if not shard_dirs:
        raise ValueError("at least one shard directory is required")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"merge output directory is not empty: {output_dir}")

    shards: list[tuple[Path, dict[str, Any]]] = []
    compatibility: dict[str, Any] | None = None
    for directory in sorted((Path(value) for value in shard_dirs), key=str):
        config = _load_json(directory / "run_config.json")
        if config.get("status") != "completed":
            raise ValueError(f"shard is not completed: {directory}")
        current = _compatibility_identity(config)
        if compatibility is None:
            compatibility = current
        elif current != compatibility:
            raise ValueError(
                f"incompatible shard run configuration: {directory}"
            )
        shards.append((directory, config))
    assert compatibility is not None

    frame_tables = [
        pd.read_parquet(directory / "sam3_exhaustive_frame_results.parquet")
        for directory, _ in shards
    ]
    hand_tables = [
        pd.read_parquet(directory / "sam3_exhaustive_hand_results.parquet")
        for directory, _ in shards
    ]
    frames = pd.concat(frame_tables, ignore_index=True)
    hands = pd.concat(hand_tables, ignore_index=True)
    _reject_duplicates(
        frames, ["asset_id", "source_frame"], "asset/source-frame"
    )
    _reject_duplicates(
        hands, ["asset_id", "source_frame", "hand"], "asset/source-frame/hand"
    )
    asset_owners: dict[str, Path] = {}
    for (directory, _), frame_table in zip(shards, frame_tables, strict=True):
        for asset_id in sorted(set(frame_table["asset_id"].astype(str))):
            previous = asset_owners.get(asset_id)
            if previous is not None:
                raise ValueError(
                    "asset appears in multiple shards: "
                    f"{asset_id} ({previous}, {directory})"
                )
            asset_owners[asset_id] = directory

    manifest_ranges = _manifest_ranges(Path(manifest))
    expected_assets = set(manifest_ranges)
    actual_assets = {str(value) for value in frames["asset_id"].unique()}
    missing = sorted(expected_assets - actual_assets)
    extra = sorted(actual_assets - expected_assets)
    if missing:
        raise ValueError("missing manifest assets: " + ", ".join(missing))
    if extra:
        raise ValueError("shards contain assets absent from manifest: " + ", ".join(extra))
    _validate_full_coverage(
        frames,
        hands,
        manifest_ranges=manifest_ranges,
    )

    frames = frames.sort_values(
        ["asset_id", "source_frame"], kind="stable"
    ).reset_index(drop=True)
    hands = hands.sort_values(
        ["asset_id", "source_frame", "hand"], kind="stable"
    ).reset_index(drop=True)
    failures = [
        row
        for directory, _ in shards
        for row in _read_jsonl(directory / "sam3_exhaustive_failures.jsonl")
    ]
    failures.sort(
        key=lambda row: (
            str(row.get("asset_id", "")),
            int(row.get("source_frame", -1)),
            str(row.get("stage", "")),
        )
    )
    evidence_tables = [
        pd.read_csv(directory / "review_evidence_manifest.csv")
        for directory, _ in shards
    ]
    evidence = pd.concat(evidence_tables, ignore_index=True)
    if not evidence.empty:
        _reject_duplicates(
            evidence, ["asset_id", "source_frame"], "evidence asset/source-frame"
        )
        evidence = evidence.sort_values(
            ["asset_id", "source_frame"], kind="stable"
        ).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    hands.to_parquet(output_dir / "sam3_exhaustive_hand_results.parquet", index=False)
    _write_jsonl(
        output_dir / "sam3_exhaustive_hand_results.jsonl",
        hands.to_dict(orient="records"),
    )
    frames.to_parquet(output_dir / "sam3_exhaustive_frame_results.parquet", index=False)
    frames.to_csv(output_dir / "sam3_exhaustive_frame_results.csv", index=False)
    _write_jsonl(output_dir / "sam3_exhaustive_failures.jsonl", failures)
    evidence.reindex(columns=list(EVIDENCE_COLUMNS)).to_csv(
        output_dir / "review_evidence_manifest.csv", index=False
    )

    summary = {
        "producer": "jdt-sam3-exhaustive-shard-merge-v1",
        "status": "completed",
        "total_manifest_assets": len(expected_assets),
        "completed_assets": len(actual_assets),
        "failed_assets": 0,
        "total_source_frames": len(frames),
        "total_hand_rows": len(hands),
        "failure_row_count": len(failures),
        "evidence_row_count": len(evidence),
        "source_shards": [str(directory.resolve()) for directory, _ in shards],
    }
    merge_config = {
        "producer": summary["producer"],
        "status": "completed",
        "candidate_independent": True,
        "source_frame_stride": 1,
        "merge": {
            "compatible_config_identity": compatibility,
            "compatible_config_sha256": canonical_sha256(compatibility),
            "source_shards": summary["source_shards"],
            "manifest": str(Path(manifest).resolve()),
        },
        "summary": summary,
        "timing": {
            key: sum(
                float(config.get("timing", {}).get(key, 0.0))
                for _, config in shards
            )
            for key in (
                "decode_seconds",
                "inference_seconds",
                "write_seconds",
                "total_seconds",
            )
        },
    }
    _atomic_write_json(output_dir / "run_config.json", merge_config)
    _atomic_write_json(output_dir / "progress.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and merge exhaustive JD SAM3 shard outputs."
    )
    parser.add_argument("--shard-dir", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = merge_shards(
        shard_dirs=args.shard_dir,
        manifest=args.manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
