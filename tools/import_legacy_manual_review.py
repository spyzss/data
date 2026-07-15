#!/usr/bin/env python3
"""Import legacy manual-review CSV decisions into authoritative QC JSON."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from human_qc.legacy_import import ImportResult, import_legacy_manual_review  # noqa: E402
from qc_common.report import load_asset_qc_report  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One-time import of exact legacy asset/issue decisions into "
            "quality_archive JSON. CSV and progress JSON remain non-authoritative."
        )
    )
    parser.add_argument("--quality-archive", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--progress-json", type=Path)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument(
        "--expected-revision",
        type=int,
        help="Required for an actual write; dry-run reads the current revision.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report exact matches/conflicts without writing QC JSON.",
    )
    return parser.parse_args(argv)


def _csv_asset_ids(path: Path) -> tuple[list[str], int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "asset_id" not in (reader.fieldnames or []):
            raise ValueError("legacy CSV must contain asset_id")
        values = [str(row.get("asset_id") or "").strip() for row in reader]
    asset_ids = list(dict.fromkeys(value for value in values if value))
    return asset_ids, sum(1 for value in values if not value)


def _report_paths(quality_archive: Path, csv_path: Path) -> tuple[list[Path], list[str], int]:
    if quality_archive.is_file():
        return [quality_archive], [], 0
    if not quality_archive.is_dir():
        raise FileNotFoundError(quality_archive)
    asset_ids, empty_asset_rows = _csv_asset_ids(csv_path)
    paths: list[Path] = []
    unknown_assets: list[str] = []
    for asset_id in asset_ids:
        path = quality_archive / f"{asset_id}.json"
        if path.is_file():
            paths.append(path)
        else:
            unknown_assets.append(asset_id)
    return paths, unknown_assets, empty_asset_rows


def _result_output(result: ImportResult) -> dict[str, Any]:
    return result.to_dict()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.dry_run and args.expected_revision is None:
        raise SystemExit("--expected-revision is required unless --dry-run is used")

    paths, unknown_assets, empty_asset_rows = _report_paths(
        args.quality_archive, args.csv
    )
    results: list[ImportResult] = []
    known_asset_ids: set[str] = set()
    for path in paths:
        report = load_asset_qc_report(path)
        if report is None:
            continue
        asset_id = report.get("asset_id")
        if isinstance(asset_id, str):
            known_asset_ids.add(asset_id)
        current_revision = int(report.get("report_revision", 0))
        result = import_legacy_manual_review(
            path,
            args.csv,
            args.progress_json,
            expected_revision=(
                current_revision if args.dry_run else int(args.expected_revision)
            ),
            reviewer=args.reviewer,
            dry_run=args.dry_run,
        )
        results.append(result)

    # Each per-report import sees rows for other reports as mismatches.  They
    # are excluded from totals when that exact asset has its own report.
    unmatched = len(unknown_assets) + empty_asset_rows
    conflicts = 0
    for result in results:
        for problem in result.problems:
            if (
                problem.code == "asset_id_mismatch"
                and problem.asset_id in known_asset_ids
            ):
                continue
            if problem.code in {"asset_id_mismatch", "unknown_issue_id"}:
                unmatched += 1
            else:
                conflicts += 1
    output = {
        "dry_run": bool(args.dry_run),
        "quality_archive": str(args.quality_archive),
        "csv": str(args.csv),
        "progress_json": str(args.progress_json) if args.progress_json else None,
        "totals": {
            "matched": sum(result.matched_count for result in results),
            "unmatched": unmatched,
            "conflicts": conflicts,
        },
        "unknown_asset_ids": unknown_assets,
        "results": [_result_output(result) for result in results],
    }
    sys.stdout.write(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    return 0 if conflicts == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
