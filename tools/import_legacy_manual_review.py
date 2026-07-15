#!/usr/bin/env python3
"""Import legacy manual-review CSV decisions into authoritative QC JSON."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from human_qc.legacy_import import (  # noqa: E402
    ImportResult,
    import_legacy_manual_review,
)
from qc_common.report import StaleReportRevisionError, load_asset_qc_report  # noqa: E402


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
    parser.add_argument(
        "--issue-mapping",
        type=Path,
        help="Authoritative CSV with asset_id, review_id, issue_id for generated review IDs.",
    )
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


def _csv_asset_counts(csv_bytes: bytes) -> tuple[Counter[str], int]:
    try:
        text = csv_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"legacy CSV must be UTF-8: {exc}") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if "asset_id" not in (reader.fieldnames or []):
        raise ValueError("legacy CSV must contain asset_id")
    values = [str(row.get("asset_id") or "").strip() for row in reader]
    return Counter(value for value in values if value), sum(1 for value in values if not value)


def _report_paths(
    quality_archive: Path, asset_counts: Counter[str]
) -> tuple[list[tuple[str | None, Path]], list[str]]:
    if quality_archive.is_file():
        return [(None, quality_archive)], []
    if not quality_archive.is_dir():
        raise FileNotFoundError(quality_archive)
    paths: list[tuple[str | None, Path]] = []
    unknown_assets: list[str] = []
    for asset_id in asset_counts:
        path = quality_archive / f"{asset_id}.json"
        if path.is_file():
            paths.append((asset_id, path))
        else:
            unknown_assets.append(asset_id)
    return paths, unknown_assets


def _result_output(result: ImportResult) -> dict[str, Any]:
    return result.to_dict()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.dry_run and args.expected_revision is None:
        raise SystemExit("--expected-revision is required unless --dry-run is used")

    csv_bytes = args.csv.read_bytes()
    progress_bytes = args.progress_json.read_bytes() if args.progress_json else None
    mapping_bytes = args.issue_mapping.read_bytes() if args.issue_mapping else None
    asset_counts, empty_asset_rows = _csv_asset_counts(csv_bytes)
    targets, unknown_assets = _report_paths(args.quality_archive, asset_counts)
    loaded_targets: list[tuple[str | None, Path, dict[str, Any]]] = []
    for requested_asset_id, path in targets:
        report = load_asset_qc_report(path)
        if report is None:
            continue
        report_asset_id = report.get("asset_id")
        if requested_asset_id is not None and report_asset_id != requested_asset_id:
            raise ValueError(
                f"report {path} asset_id={report_asset_id!r} does not match requested "
                f"asset_id={requested_asset_id!r}"
            )
        loaded_targets.append((requested_asset_id, path, report))

    if not args.dry_run:
        expected = int(args.expected_revision)
        stale = [
            (path, int(report.get("report_revision", 0)))
            for _asset_id, path, report in loaded_targets
            if int(report.get("report_revision", 0)) != expected
        ]
        if stale:
            details = ", ".join(f"{path}={revision}" for path, revision in stale)
            raise StaleReportRevisionError(
                f"batch preflight expected revision {expected}; mismatches: {details}"
            )

    results: list[ImportResult] = []
    directory_mode = args.quality_archive.is_dir()
    for _requested_asset_id, path, report in loaded_targets:
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
            issue_mapping_path=args.issue_mapping,
            asset_scope_only=directory_mode,
            _csv_bytes=csv_bytes,
            _progress_bytes=progress_bytes,
            _mapping_bytes=mapping_bytes,
        )
        results.append(result)

    unmatched = empty_asset_rows + sum(asset_counts[asset_id] for asset_id in unknown_assets)
    unmatched += sum(result.unmatched_count for result in results)
    conflicts = sum(result.conflict_count for result in results)
    output = {
        "dry_run": bool(args.dry_run),
        "quality_archive": str(args.quality_archive),
        "csv": str(args.csv),
        "progress_json": str(args.progress_json) if args.progress_json else None,
        "issue_mapping": str(args.issue_mapping) if args.issue_mapping else None,
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
