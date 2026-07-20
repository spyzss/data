#!/usr/bin/env python3
"""Merge supplier SAM3 outputs into the standalone window-review contract."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from human_qc.sam3_window_review import generated_review_id


FRAME_COORDINATE_SYSTEM = "source_inclusive"
DURATION_RESOLUTION_STATUS = "not_annotated"
_PASS_STATES = frozenset({"pass", "auto_pass", "completed_pass", "acceptable"})
_FAIL_STATES = frozenset({"fail", "containment_fail", "strong_fail"})
_REVIEW_STATES = frozenset(
    {
        "review",
        "warn",
        "mixed_review",
        "projection_review",
        "visual_conflict_review",
        "contained_but_truncated_review",
    }
)
_BLOCKED_STATES = frozenset(
    {"blocked", "input_missing", "not_run", "adapter_missing"}
)


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if type(value).__module__.split(".", 1)[0] == "numpy":
        item = getattr(value, "item", None)
        if callable(item):
            return _clean(item())
    return value


def _text(value: Any) -> str:
    value = _clean(value)
    return "" if value is None else str(value).strip()


def _integer(value: Any, field: str) -> int:
    value = _clean(value)
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be an integer") from None
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{field} must be an integer")
    return int(number)


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return []
        return [dict(row) for row in frame.to_dict(orient="records")]
    if suffix in {".parquet", ".pq"}:
        return [dict(row) for row in pd.read_parquet(path).to_dict(orient="records")]
    if suffix in {".jsonl", ".ndjson"}:
        rows: list[dict[str, Any]] = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number} must be a JSON object")
            rows.append(value)
        return rows
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, Mapping):
            value = value.get("records", value.get("items", value.get("rows")))
        if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
            raise ValueError(f"{path} must contain a JSON row array")
        return [dict(row) for row in value]
    raise ValueError(f"unsupported input format: {path}")


def _source_state(row: Mapping[str, Any]) -> str:
    return _text(
        row.get("sam3_window_state")
        or row.get("window_containment_verdict")
        or row.get("containment_verdict")
        or row.get("verdict")
    ).lower()


def _routing_state(value: str) -> str:
    if value in _PASS_STATES:
        return "pass"
    if value in _FAIL_STATES:
        return "fail"
    if value in _REVIEW_STATES:
        return "review"
    if value in _BLOCKED_STATES:
        return "blocked"
    raise ValueError(f"unsupported SAM3 window state: {value or '<missing>'}")


def _window_bounds(row: Mapping[str, Any]) -> tuple[int, int]:
    start = _integer(
        row.get("window_start_frame", row.get("start_frame")),
        "window_start_frame",
    )
    end = _integer(
        row.get("window_end_frame", row.get("end_frame")),
        "window_end_frame",
    )
    if start > end:
        raise ValueError("window_start_frame must not exceed window_end_frame")
    return start, end


def _json_field(value: Any, fallback: Any) -> str:
    value = _clean(value)
    if value is None:
        payload = fallback
    elif isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            payload = value
    else:
        payload = value
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _discover_run_root(root: Path) -> tuple[list[Path], list[Path]]:
    window_paths = sorted(
        root.glob("module_outputs/*/sam3_containment/window_results.json")
    )
    evidence_paths: list[Path] = []
    for window_path in window_paths:
        evidence_path = window_path.with_name("evidence_manifest.json")
        if not evidence_path.is_file():
            raise FileNotFoundError(evidence_path)
        evidence_paths.append(evidence_path)
    return window_paths, evidence_paths


def _normalize_source_path(value: Any, *, base: Path) -> str:
    text = _text(value)
    if not text:
        raise ValueError("evidence source_path is required")
    path = Path(text).expanduser()
    access = path if path.is_absolute() else base / path
    access = access.resolve()
    if not access.is_file():
        raise ValueError(f"evidence source_path does not exist: {access}")
    return str(access)


def build_sam3_review_bundle(
    *,
    manifest_paths: Sequence[Path],
    review_queue_paths: Sequence[Path],
    evidence_paths: Sequence[Path],
    run_roots: Sequence[Path],
    output_dir: Path,
    expected_evidence_count: int = 5,
) -> dict[str, Path]:
    if expected_evidence_count < 1:
        raise ValueError("expected_evidence_count must be >= 1")
    if not manifest_paths:
        raise ValueError("at least one manifest is required")
    output_dir = Path(output_dir).resolve()

    manifest_by_asset: dict[str, dict[str, Any]] = {}
    for path in manifest_paths:
        for source in _read_records(Path(path)):
            row = {str(key): _clean(value) for key, value in source.items()}
            asset_id = _text(row.get("asset_id"))
            if not asset_id:
                raise ValueError(f"manifest row has no asset_id: {path}")
            supplier = _text(row.get("supplier") or row.get("supplier_id")).lower()
            if not supplier:
                raise ValueError(f"manifest row has no supplier: {asset_id}")
            row["supplier"] = supplier
            row["supplier_id"] = supplier
            prior = manifest_by_asset.get(asset_id)
            if prior is not None and prior != row:
                raise ValueError(f"conflicting manifest rows for asset_id: {asset_id}")
            manifest_by_asset[asset_id] = row

    queue_sources: list[tuple[dict[str, Any], Path]] = []
    evidence_sources: list[tuple[dict[str, Any], Path]] = []
    for path in review_queue_paths:
        path = Path(path).resolve()
        queue_sources.extend((row, path.parent) for row in _read_records(path))
    for path in evidence_paths:
        path = Path(path).resolve()
        evidence_sources.extend((row, path.parent) for row in _read_records(path))
    for root_value in run_roots:
        root = Path(root_value).resolve()
        windows, evidence = _discover_run_root(root)
        for path in windows:
            queue_sources.extend((row, root) for row in _read_records(path))
        for path in evidence:
            evidence_sources.extend((row, root) for row in _read_records(path))
    if not queue_sources:
        raise ValueError("no review queues or SAM3 window outputs were found")

    counts = Counter()
    manual_rows: list[dict[str, Any]] = []
    seen_review_ids: set[str] = set()
    for source, source_base in queue_sources:
        row = {str(key): _clean(value) for key, value in source.items()}
        state = _routing_state(_source_state(row))
        counts[state] += 1
        if state in {"pass", "blocked"}:
            continue
        asset_id = _text(row.get("asset_id"))
        manifest = manifest_by_asset.get(asset_id)
        if manifest is None:
            raise ValueError(f"SAM3 asset_id is absent from manifests: {asset_id}")
        supplier = _text(row.get("supplier_id") or row.get("supplier")).lower()
        supplier = supplier or _text(manifest.get("supplier_id")).lower()
        if supplier != _text(manifest.get("supplier_id")).lower():
            raise ValueError(f"supplier mismatch for asset_id: {asset_id}")
        start, end = _window_bounds(row)
        review_id = _text(row.get("review_id")) or generated_review_id(
            supplier_id=supplier,
            asset_id=asset_id,
            start=start,
            end=end,
        )
        if review_id in seen_review_ids:
            raise ValueError(f"duplicate review_id: {review_id}")
        seen_review_ids.add(review_id)
        raw_verdict = _text(
            row.get("raw_window_containment_verdict")
            or row.get("window_containment_verdict")
        )
        sampled = row.get("sampled_frame_indices_json", row.get("sampled_frame_indices"))
        manual_rows.append(
            {
                "review_id": review_id,
                "supplier_id": supplier,
                "asset_id": asset_id,
                "window_start_frame": start,
                "window_end_frame": end,
                "source_start_frame": start,
                "source_end_frame": end,
                "frame_coordinate_system": FRAME_COORDINATE_SYSTEM,
                "sam3_window_state": state,
                "raw_window_containment_verdict": raw_verdict,
                "left_window_containment_verdict": _text(
                    row.get("left_window_containment_verdict")
                ),
                "right_window_containment_verdict": _text(
                    row.get("right_window_containment_verdict")
                ),
                "sampled_frame_indices_json": _json_field(sampled, []),
                "video_path": _text(row.get("video_path") or manifest.get("primary_video_path")),
                "fps": _clean(row.get("fps") or manifest.get("fps")),
                "trigger_reason_json": _json_field(
                    row.get("trigger_reason_json", row.get("reason")), []
                ),
                "trigger_metrics_json": _json_field(
                    row.get("trigger_metrics_json", row), {}
                ),
                "calibration_status": _text(row.get("calibration_status")),
                "duration_resolution_status": DURATION_RESOLUTION_STATUS,
                "source_queue_path": str(source_base),
            }
        )
    manual_rows.sort(
        key=lambda row: (
            row["supplier_id"],
            row["asset_id"],
            row["window_start_frame"],
            row["window_end_frame"],
            row["review_id"],
        )
    )

    by_review_id = {row["review_id"]: row for row in manual_rows}
    by_exact_window = {
        (
            row["supplier_id"],
            row["asset_id"],
            row["window_start_frame"],
            row["window_end_frame"],
        ): row["review_id"]
        for row in manual_rows
    }
    matched_exact: dict[str, list[dict[str, Any]]] = {
        key: [] for key in by_review_id
    }
    matched_fallback: dict[str, list[dict[str, Any]]] = {
        key: [] for key in by_review_id
    }
    for source, base in evidence_sources:
        row = {str(key): _clean(value) for key, value in source.items()}
        source_review_id = _text(row.get("review_id"))
        review_id = source_review_id if source_review_id in by_review_id else ""
        match_kind = "exact" if review_id else ""
        if not review_id and not source_review_id:
            asset_id = _text(row.get("asset_id"))
            manifest = manifest_by_asset.get(asset_id)
            if manifest is None:
                continue
            supplier = _text(row.get("supplier_id") or row.get("supplier")).lower()
            supplier = supplier or _text(manifest.get("supplier_id")).lower()
            try:
                start, end = _window_bounds(row)
            except ValueError:
                continue
            review_id = by_exact_window.get((supplier, asset_id, start, end), "")
            match_kind = "fallback" if review_id else ""
        if not review_id:
            continue
        normalized = dict(row)
        normalized["review_id"] = review_id
        normalized["supplier_id"] = by_review_id[review_id]["supplier_id"]
        normalized["asset_id"] = by_review_id[review_id]["asset_id"]
        normalized["window_start_frame"] = by_review_id[review_id]["window_start_frame"]
        normalized["window_end_frame"] = by_review_id[review_id]["window_end_frame"]
        normalized["source_path"] = _normalize_source_path(
            row.get("source_path"), base=base
        )
        if match_kind == "exact":
            matched_exact[review_id].append(normalized)
        else:
            matched_fallback[review_id].append(normalized)

    review_evidence: list[dict[str, Any]] = []
    for row in manual_rows:
        review_id = row["review_id"]
        rows = matched_exact[review_id] or matched_fallback[review_id]
        if len(rows) != expected_evidence_count:
            raise ValueError(
                f"{review_id} expected {expected_evidence_count} combined overlays, "
                f"found {len(rows)}"
            )
        review_evidence.extend(
            sorted(rows, key=lambda item: _integer(item.get("frame_idx"), "frame_idx"))
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_output = output_dir / "review_manifest.csv"
    queue_output = output_dir / "review_queue.csv"
    evidence_output = output_dir / "review_evidence.csv"
    summary_output = output_dir / "review_contract_summary.json"
    pd.DataFrame(
        [manifest_by_asset[key] for key in sorted(manifest_by_asset)]
    ).to_csv(manifest_output, index=False)
    pd.DataFrame(manual_rows).to_csv(queue_output, index=False)
    pd.DataFrame(review_evidence).to_csv(evidence_output, index=False)
    supplier_distribution = Counter(row["supplier_id"] for row in manual_rows)
    summary_output.write_text(
        json.dumps(
            {
                "schema_version": "sam3_multi_supplier_review_bundle.v1",
                "frame_coordinate_system": FRAME_COORDINATE_SYSTEM,
                "duration_resolution_status": DURATION_RESOLUTION_STATUS,
                "expected_evidence_count_per_review": expected_evidence_count,
                "review_queue_count": len(manual_rows),
                "auto_pass_count": counts["pass"],
                "human_fail_count": counts["fail"],
                "human_review_count": counts["review"],
                "evidence_count": len(review_evidence),
                "blocked_count": counts["blocked"],
                "supplier_distribution": dict(sorted(supplier_distribution.items())),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "review_manifest": manifest_output,
        "review_queue": queue_output,
        "review_evidence": evidence_output,
        "summary": summary_output,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True, type=Path)
    parser.add_argument("--review-queue", action="append", default=[], type=Path)
    parser.add_argument("--evidence-manifest", action="append", default=[], type=Path)
    parser.add_argument("--run-root", action="append", default=[], type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-evidence-count", type=int, default=5)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = build_sam3_review_bundle(
        manifest_paths=args.manifest,
        review_queue_paths=args.review_queue,
        evidence_paths=args.evidence_manifest,
        run_roots=args.run_root,
        output_dir=args.output_dir,
        expected_evidence_count=args.expected_evidence_count,
    )
    for name, path in outputs.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
