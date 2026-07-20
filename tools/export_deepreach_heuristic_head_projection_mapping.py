#!/usr/bin/env python3
"""Export user-selected heuristic DR head projection mappings."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _parse_selection(value: str) -> tuple[str, float]:
    asset_id, separator, raw_hfov = value.partition("=")
    if not separator or not asset_id.strip():
        raise ValueError("--select must use ASSET_ID=HFOV_DEG")
    try:
        hfov = float(raw_hfov)
    except ValueError:
        raise ValueError("--select HFOV must be numeric") from None
    if not math.isfinite(hfov):
        raise ValueError("--select HFOV must be finite")
    return asset_id.strip(), hfov


def export_heuristic_mapping(
    *,
    audit: Path,
    output: Path,
    selections: Mapping[str, float],
) -> int:
    if not selections:
        raise ValueError("at least one explicit --select ASSET_ID=HFOV_DEG is required")
    payload = json.loads(Path(audit).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "dr_approximate_head_projection_audit.v1":
        raise ValueError("unsupported DR approximate audit schema")
    candidates = {
        (str(row["asset_id"]), float(row["head_hfov_deg"])): row
        for asset in payload.get("assets", [])
        for row in asset.get("candidates", [])
    }
    rows: list[dict[str, object]] = []
    for asset_id, hfov in selections.items():
        source = candidates.get((str(asset_id), float(hfov)))
        if source is None:
            raise ValueError(f"selected HFOV was not audited: {asset_id}={hfov:g}")
        rows.append(
            {
                "asset_id": asset_id,
                "projection_mode": "approx_pinhole_from_hfov",
                "head_hfov_deg": float(hfov),
                "fx": source["fx"],
                "fy": source["fy"],
                "cx": source["cx"],
                "cy": source["cy"],
                "image_width": source["image_width"],
                "image_height": source["image_height"],
                "calibration_status": "heuristic",
                "projection_validation_status": "pending_visual_validation",
                "distortion_applied": "false",
                "camera_trajectory_applied": "false",
                "selected_by": "user_visual_confirmation",
            }
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = tuple(rows[0])
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--select", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selections = dict(_parse_selection(value) for value in args.select)
    count = export_heuristic_mapping(
        audit=args.audit,
        output=args.output,
        selections=selections,
    )
    print(json.dumps({"mapping_count": count}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
