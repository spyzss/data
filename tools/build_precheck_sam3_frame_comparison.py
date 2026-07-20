#!/usr/bin/env python3
"""Compare raw Precheck frame flags with candidate-independent SAM3 results."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from tools.run_manifest_sam3_containment import read_records


_PRIMARY_CHECKS = {
    "keypoint_missing": "keypoint_presence",
    "keypoint_presence": "keypoint_presence",
    "keypoint_morphology": "keypoint_morphology",
    "keypoint_temporal": "keypoint_temporal",
}
_FLAG_COLUMNS = (
    "keypoint_presence_flag",
    "keypoint_morphology_flag",
    "keypoint_temporal_flag",
    "projection_flag",
    "composite_skeleton_flag",
)
_PROJECTION_KEYS = (
    "needs_projection_review",
    "needs_out_of_frame_review",
    "left_needs_projection_review",
    "right_needs_projection_review",
    "left_needs_out_of_frame_review",
    "right_needs_out_of_frame_review",
)


def read_precheck_artifacts(
    paths: Sequence[Path], *, artifact_name: str
) -> list[dict[str, Any]]:
    """Read explicit artifacts or roots and recover per-asset directory identity."""

    files: list[Path] = []
    for value in paths:
        path = Path(value)
        if path.is_dir():
            files.extend(path.rglob(artifact_name))
        else:
            files.append(path)
    unique_files = sorted({path.resolve() for path in files}, key=str)
    if not unique_files:
        raise ValueError(f"no {artifact_name} artifacts found")
    rows: list[dict[str, Any]] = []
    for path in unique_files:
        inferred_asset_id = (
            path.parent.parent.name if path.parent.name == "precheck" else None
        )
        for raw in read_records(path):
            row = dict(raw)
            existing = row.get("asset_id")
            if inferred_asset_id is not None:
                if existing is not None and str(existing) != inferred_asset_id:
                    raise ValueError(
                        "asset_id disagrees with artifact path: "
                        f"{existing!r} != {inferred_asset_id!r} ({path})"
                    )
                row["asset_id"] = inferred_asset_id
            elif existing is None or not str(existing).strip():
                raise ValueError(
                    f"artifact row lacks asset_id and path has no precheck identity: {path}"
                )
            row.setdefault("source_artifact_path", str(path))
            rows.append(row)
    return rows


def _json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        return _json_list(decoded)
    if isinstance(value, Mapping):
        return [json.dumps(dict(value), sort_keys=True)]
    if isinstance(value, Iterable):
        return [str(item) for item in value if item is not None and str(item)]
    return [str(value)]


def _metrics(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, Mapping) else {}
    return {}


def _truthy(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1"}
    return False


def _source_frame(row: Mapping[str, Any]) -> int | None:
    value = row.get("source_frame")
    if value is None:
        value = row.get("source_frame_idx", row.get("frame_idx"))
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def _stable_json(values: Iterable[str]) -> str:
    return json.dumps(sorted(set(values)), separators=(",", ":"))


def _candidate_identity(row: Mapping[str, Any]) -> str:
    for key in ("review_id", "candidate_id", "window_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    asset_id = str(row.get("asset_id", ""))
    start = row.get("start_frame", row.get("window_start_frame"))
    end = row.get("end_frame", row.get("window_end_frame"))
    return f"{asset_id}:{start}:{end}"


def _candidate_bounds(row: Mapping[str, Any]) -> tuple[int, int] | None:
    start = row.get("start_frame", row.get("window_start_frame"))
    end = row.get("end_frame", row.get("window_end_frame"))
    try:
        return int(start), int(end)
    except (TypeError, ValueError, OverflowError):
        return None


def project_precheck_frames(
    *,
    sam3_frame_rows: Sequence[Mapping[str, Any]],
    check_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Project raw check rows onto the exact exhaustive SAM3 frame universe."""

    universe: dict[tuple[str, int], dict[str, Any]] = {}
    for row in sam3_frame_rows:
        asset_id = str(row.get("asset_id", ""))
        frame = _source_frame(row)
        if not asset_id or frame is None:
            raise ValueError("SAM3 frame row requires asset_id and source_frame")
        key = (asset_id, frame)
        if key in universe:
            raise ValueError(f"duplicate SAM3 frame key: {key}")
        universe[key] = {
            "asset_id": asset_id,
            "source_frame": frame,
            **{column: False for column in _FLAG_COLUMNS},
            "precheck_modules": set(),
            "precheck_reason_codes": set(),
            "precheck_module_reasons": defaultdict(set),
            "candidate_window_membership": False,
            "candidate_review_ids": set(),
        }

    for check_row in check_rows:
        asset_id = str(check_row.get("asset_id", ""))
        frame = _source_frame(check_row)
        projected = universe.get((asset_id, frame)) if frame is not None else None
        if projected is None:
            continue
        check = str(check_row.get("check", ""))
        metrics = _metrics(check_row.get("metrics"))
        flag = _truthy(check_row.get("flag"))
        module = _PRIMARY_CHECKS.get(check)
        module_flag = False
        if module is not None and flag:
            projected[f"{module}_flag"] = True
            module_flag = True
        elif check == "skeleton_quality_score":
            projection_flag = any(_truthy(metrics.get(key)) for key in _PROJECTION_KEYS)
            if projection_flag:
                projected["projection_flag"] = True
                module_flag = True
            if flag:
                projected["composite_skeleton_flag"] = True
                module_flag = True
            module = "skeleton_quality_score"
        if not module_flag:
            continue
        projected["precheck_modules"].add(module)
        reason = check_row.get("reason")
        if reason is not None and str(reason).strip():
            projected["precheck_reason_codes"].add(str(reason))
            projected["precheck_module_reasons"][module].add(str(reason))

    candidates_by_asset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidate_rows:
        candidates_by_asset[str(candidate.get("asset_id", ""))].append(candidate)
    for (asset_id, frame), projected in universe.items():
        for candidate in candidates_by_asset.get(asset_id, ()):
            bounds = _candidate_bounds(candidate)
            if bounds is None:
                continue
            start, end = bounds
            if start <= frame <= end:
                projected["candidate_window_membership"] = True
                projected["candidate_review_ids"].add(_candidate_identity(candidate))

    records: list[dict[str, Any]] = []
    for key in sorted(universe):
        row = universe[key]
        row["precheck_problem_any"] = any(bool(row[column]) for column in _FLAG_COLUMNS)
        row["precheck_modules"] = _stable_json(row["precheck_modules"])
        row["precheck_reason_codes"] = _stable_json(row["precheck_reason_codes"])
        row["precheck_module_reasons"] = json.dumps(
            {
                module: sorted(reasons)
                for module, reasons in sorted(row["precheck_module_reasons"].items())
            },
            separators=(",", ":"),
        )
        row["candidate_review_ids"] = _stable_json(row["candidate_review_ids"])
        records.append(row)
    return pd.DataFrame.from_records(records)


def compare_frames(
    *,
    sam3_frame_rows: Sequence[Mapping[str, Any]],
    precheck_frame_rows: pd.DataFrame | Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Build an exact source-frame comparison with explicit unevaluable rows."""

    sam3 = pd.DataFrame.from_records(sam3_frame_rows)
    if sam3.empty:
        raise ValueError("SAM3 frame results are empty")
    if "source_frame" not in sam3.columns:
        raise ValueError("SAM3 frame results require source_frame")
    keys = ["asset_id", "source_frame"]
    if sam3.duplicated(keys).any():
        raise ValueError("SAM3 frame results contain duplicate asset/source-frame keys")
    precheck = (
        precheck_frame_rows.copy()
        if isinstance(precheck_frame_rows, pd.DataFrame)
        else pd.DataFrame.from_records(precheck_frame_rows)
    )
    if precheck.duplicated(keys).any():
        raise ValueError("Precheck projection contains duplicate asset/source-frame keys")
    joined = sam3.merge(precheck, on=keys, how="left", validate="one_to_one")
    for column in (*_FLAG_COLUMNS, "precheck_problem_any", "candidate_window_membership"):
        if column in joined:
            joined[column] = joined[column].fillna(False).astype(bool)
    for column in (
        "precheck_modules",
        "precheck_reason_codes",
        "candidate_review_ids",
    ):
        if column in joined:
            joined[column] = joined[column].fillna("[]")
    if "precheck_module_reasons" in joined:
        joined["precheck_module_reasons"] = joined[
            "precheck_module_reasons"
        ].fillna("{}")

    labels: list[str] = []
    classes: list[str] = []
    exclusions: list[str | None] = []
    for row in joined.to_dict(orient="records"):
        verdict = str(row.get("frame_verdict", "")).strip().lower()
        precheck_problem = bool(row.get("precheck_problem_any", False))
        if verdict in {"fail", "review"}:
            label = "problem"
            comparison = "TP" if precheck_problem else "FN"
            exclusion = None
        elif verdict == "pass":
            label = "clean"
            comparison = "FP" if precheck_problem else "TN"
            exclusion = None
        else:
            label = "unevaluable"
            comparison = "UNEVALUABLE"
            exclusion = f"sam3_{verdict or 'unknown'}"
        labels.append(label)
        classes.append(comparison)
        exclusions.append(exclusion)
    joined["sam3_binary_label"] = labels
    joined["comparison_class"] = classes
    joined["exclusion_reason"] = exclusions
    joined["evaluable"] = joined["comparison_class"] != "UNEVALUABLE"
    joined["sam3_frame_verdict"] = joined["frame_verdict"]
    joined["sam3_reason_codes"] = joined["frame_reason_codes"]
    return joined.sort_values(keys, kind="stable").reset_index(drop=True)


def _ratio(numerator: int, denominator: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "value": float(numerator / denominator) if denominator else None,
    }
    if not denominator:
        result["status"] = "not_applicable"
    return result


def build_comparison_summary(comparison: pd.DataFrame) -> dict[str, Any]:
    counts = comparison["comparison_class"].value_counts().to_dict()
    tp = int(counts.get("TP", 0))
    fn = int(counts.get("FN", 0))
    fp = int(counts.get("FP", 0))
    tn = int(counts.get("TN", 0))
    evaluable = tp + fn + fp + tn
    unevaluable = int(counts.get("UNEVALUABLE", 0))
    return {
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "total_evaluable_frames": evaluable,
        "total_unevaluable_frames": unevaluable,
        "detection_rate": _ratio(tp, tp + fn),
        "raw_frame_recall": _ratio(tp, tp + fn),
        "miss_rate": _ratio(fn, tp + fn),
        "precision": _ratio(tp, tp + fp),
        "false_discovery_rate": _ratio(fp, tp + fp),
        "false_positive_rate": _ratio(fp, fp + tn),
        "accuracy": _ratio(tp + tn, evaluable),
        "sam3_problem_frame_rate": _ratio(tp + fn, evaluable),
        "precheck_flagged_frame_rate": _ratio(tp + fp, evaluable),
        "evaluable_coverage": _ratio(evaluable, evaluable + unevaluable),
    }


def contiguous_intervals(rows: pd.DataFrame, *, class_name: str) -> pd.DataFrame:
    selected = rows.loc[rows["comparison_class"] == class_name, ["asset_id", "source_frame"]]
    records: list[dict[str, Any]] = []
    for asset_id, group in selected.groupby("asset_id", sort=True):
        frames = sorted({int(value) for value in group["source_frame"].tolist()})
        if not frames:
            continue
        start = previous = frames[0]
        for frame in frames[1:]:
            if frame != previous + 1:
                records.append(
                    {"asset_id": asset_id, "start_frame": start, "end_frame": previous, "frame_count": previous - start + 1}
                )
                start = frame
            previous = frame
        records.append(
            {"asset_id": asset_id, "start_frame": start, "end_frame": previous, "frame_count": previous - start + 1}
        )
    return pd.DataFrame.from_records(
        records, columns=["asset_id", "start_frame", "end_frame", "frame_count"]
    )


def _write_table_pair(frame: pd.DataFrame, output_dir: Path, stem: str) -> None:
    frame.to_parquet(output_dir / f"{stem}.parquet", index=False)
    frame.to_csv(output_dir / f"{stem}.csv", index=False)


def _per_asset(comparison: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for asset_id, group in comparison.groupby("asset_id", sort=True):
        summary = build_comparison_summary(group)
        verdicts = group["sam3_frame_verdict"].value_counts().to_dict()
        fn_frames = sorted(
            int(value)
            for value in group.loc[group["comparison_class"] == "FN", "source_frame"]
        )
        fn_intervals = contiguous_intervals(group, class_name="FN")
        candidate_problem = group[
            (group["sam3_binary_label"] == "problem")
            & group["candidate_window_membership"]
        ]
        sam3_problem_count = int((group["sam3_binary_label"] == "problem").sum())
        candidate_window_recall = _ratio(
            len(candidate_problem), sam3_problem_count
        )
        processing_column = next(
            (
                column
                for column in ("processing_seconds", "elapsed_seconds")
                if column in group.columns
            ),
            None,
        )
        records.append(
            {
                "asset_id": asset_id,
                "source_frame_count": len(group),
                "evaluable_count": int(group["evaluable"].sum()),
                "sam3_fail_count": int(verdicts.get("fail", 0)),
                "sam3_review_count": int(verdicts.get("review", 0)),
                "sam3_pass_count": int(verdicts.get("pass", 0)),
                "sam3_blocked_count": int(
                    len(group) - verdicts.get("fail", 0) - verdicts.get("review", 0) - verdicts.get("pass", 0)
                ),
                "TP": summary["TP"],
                "FN": summary["FN"],
                "FP": summary["FP"],
                "TN": summary["TN"],
                "recall": summary["detection_rate"]["value"],
                "recall_numerator": summary["detection_rate"]["numerator"],
                "recall_denominator": summary["detection_rate"]["denominator"],
                "miss_rate": summary["miss_rate"]["value"],
                "miss_rate_numerator": summary["miss_rate"]["numerator"],
                "miss_rate_denominator": summary["miss_rate"]["denominator"],
                "precision": summary["precision"]["value"],
                "precision_numerator": summary["precision"]["numerator"],
                "precision_denominator": summary["precision"]["denominator"],
                "false_positive_rate": summary["false_positive_rate"]["value"],
                "false_positive_rate_numerator": summary["false_positive_rate"][
                    "numerator"
                ],
                "false_positive_rate_denominator": summary["false_positive_rate"][
                    "denominator"
                ],
                "candidate_window_recall": candidate_window_recall["value"],
                "candidate_window_recall_numerator": candidate_window_recall[
                    "numerator"
                ],
                "candidate_window_recall_denominator": candidate_window_recall[
                    "denominator"
                ],
                "first_fn_frame": fn_frames[0] if fn_frames else None,
                "last_fn_frame": fn_frames[-1] if fn_frames else None,
                "fn_interval_count": len(fn_intervals),
                "processing_seconds": (
                    float(group[processing_column].fillna(0.0).sum())
                    if processing_column
                    else None
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _module_coverage(comparison: pd.DataFrame) -> pd.DataFrame:
    sam3_problem = comparison["sam3_binary_label"] == "problem"
    flags = comparison[list(_FLAG_COLUMNS)].astype(bool)
    flag_count = flags.sum(axis=1)
    records: list[dict[str, Any]] = []
    for column in _FLAG_COLUMNS:
        module_flag = flags[column]
        caught = int((module_flag & sam3_problem).sum())
        problem_count = int(sam3_problem.sum())
        recall = _ratio(caught, problem_count)
        false_positive = int((module_flag & (comparison["sam3_binary_label"] == "clean")).sum())
        records.append(
            {
                "module": column.removesuffix("_flag"),
                "flagged_frame_count": int(module_flag.sum()),
                "sam3_problem_caught_count": caught,
                "sam3_problem_frames_uniquely_caught": int(
                    (module_flag & sam3_problem & (flag_count == 1)).sum()
                ),
                "sam3_problem_frame_count": problem_count,
                "recall": recall["value"],
                "recall_numerator": recall["numerator"],
                "recall_denominator": recall["denominator"],
                "overlap_with_other_modules_count": int(
                    (module_flag & (flag_count > 1)).sum()
                ),
                "false_positive_count": false_positive,
            }
        )
    return pd.DataFrame.from_records(records)


def _reason_matrix(comparison: pd.DataFrame) -> pd.DataFrame:
    counts: dict[tuple[str, str, str, str, str], int] = defaultdict(int)
    for row in comparison.to_dict(orient="records"):
        raw_pairs = row.get("precheck_module_reasons")
        try:
            module_reasons = (
                json.loads(raw_pairs)
                if isinstance(raw_pairs, str) and raw_pairs.strip()
                else raw_pairs
            )
        except json.JSONDecodeError:
            module_reasons = None
        if not isinstance(module_reasons, Mapping) or not module_reasons:
            module_reasons = {"none": ["none"]}
        sam3_reasons = _json_list(row.get("frame_reason_codes")) or ["none"]
        for module, reasons in sorted(module_reasons.items()):
            for precheck_reason in _json_list(reasons) or ["none"]:
                for sam3_reason in sam3_reasons:
                    counts[
                        (
                            module,
                            precheck_reason,
                            str(row.get("sam3_frame_verdict", "")),
                            sam3_reason,
                            str(row.get("comparison_class", "")),
                        )
                    ] += 1
    return pd.DataFrame.from_records(
        [
            {
                "precheck_module": key[0],
                "precheck_reason": key[1],
                "sam3_verdict": key[2],
                "sam3_reason": key[3],
                "comparison_class": key[4],
                "frame_count": count,
            }
            for key, count in sorted(counts.items())
        ],
        columns=[
            "precheck_module",
            "precheck_reason",
            "sam3_verdict",
            "sam3_reason",
            "comparison_class",
            "frame_count",
        ],
    )


def _evidence_manifest(
    comparison: pd.DataFrame, *, require_positive_evidence: bool
) -> pd.DataFrame:
    positives = comparison.loc[comparison["sam3_binary_label"] == "problem"].copy()
    if "evidence_path" not in positives:
        positives["evidence_path"] = None
    missing = positives["evidence_path"].isna() | (positives["evidence_path"].astype(str).str.strip() == "")
    if require_positive_evidence and bool(missing.any()):
        missing_keys = positives.loc[missing, ["asset_id", "source_frame"]].to_dict(orient="records")
        raise ValueError(f"SAM3 problem frames missing evidence: {missing_keys}")
    positives["overlay_path"] = positives["evidence_path"]
    positives["source_video"] = positives.get("source_video_path")
    for column in (
        "supplier",
        "sam3_frame_verdict",
        "sam3_reason_codes",
        "precheck_modules",
        "precheck_reason_codes",
        "left_inside_ratio",
        "right_inside_ratio",
        "model_config_identity",
        "source_video",
    ):
        if column not in positives:
            positives[column] = None
    columns = [
        "asset_id",
        "supplier",
        "source_frame",
        "video_frame",
        "comparison_class",
        "sam3_frame_verdict",
        "sam3_reason_codes",
        "precheck_modules",
        "precheck_reason_codes",
        "source_video",
        "overlay_path",
        "left_inside_ratio",
        "right_inside_ratio",
        "model_config_identity",
    ]
    return positives.loc[~missing, columns].reset_index(drop=True)


def build_comparison_outputs(
    *,
    sam3_frame_rows: Sequence[Mapping[str, Any]],
    check_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    manifest_asset_ids: Sequence[str],
    model_run_lineage: Mapping[str, Any],
    require_positive_evidence: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"comparison output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    precheck = project_precheck_frames(
        sam3_frame_rows=sam3_frame_rows,
        check_rows=check_rows,
        candidate_rows=candidate_rows,
    )
    comparison = compare_frames(
        sam3_frame_rows=sam3_frame_rows,
        precheck_frame_rows=precheck,
    )
    evidence = _evidence_manifest(
        comparison, require_positive_evidence=require_positive_evidence
    )
    summary = build_comparison_summary(comparison)
    producer_summary = model_run_lineage.get("summary", {})
    if not isinstance(producer_summary, Mapping):
        producer_summary = {}
    compared_assets = int(comparison["asset_id"].nunique())
    summary.update(
        {
            "schema_version": "precheck_vs_sam3_summary.v1",
            "total_manifest_assets": len(set(manifest_asset_ids)),
            "compared_asset_count": compared_assets,
            "completed_assets": int(
                producer_summary.get("completed_assets", compared_assets)
            ),
            "failed_assets": int(
                producer_summary.get(
                    "failed_assets",
                    max(0, len(set(manifest_asset_ids)) - compared_assets),
                )
            ),
            "total_source_frames": len(comparison),
            "lineage": dict(model_run_lineage),
        }
    )
    verdict_counts = comparison["sam3_frame_verdict"].value_counts().to_dict()
    summary.update(
        {
            "sam3_fail_count": int(verdict_counts.get("fail", 0)),
            "sam3_review_count": int(verdict_counts.get("review", 0)),
            "sam3_pass_count": int(verdict_counts.get("pass", 0)),
            "sam3_blocked_count": int(
                len(comparison)
                - verdict_counts.get("fail", 0)
                - verdict_counts.get("review", 0)
                - verdict_counts.get("pass", 0)
            ),
            "sam3_error_count": int(
                comparison["sam3_reason_codes"]
                .astype(str)
                .str.contains("runtime_error|input_invalid|input_missing", regex=True)
                .sum()
            ),
            "sam3_unevaluable_verdict_counts": {
                str(key): int(value)
                for key, value in sorted(
                    comparison.loc[
                        comparison["comparison_class"] == "UNEVALUABLE",
                        "sam3_frame_verdict",
                    ]
                    .value_counts()
                    .to_dict()
                    .items()
                )
            },
        }
    )
    candidate_problem = comparison[
        (comparison["sam3_binary_label"] == "problem")
        & comparison["candidate_window_membership"]
    ]
    problem_count = int((comparison["sam3_binary_label"] == "problem").sum())
    summary["candidate_frame_recall"] = _ratio(len(candidate_problem), problem_count)

    _write_table_pair(precheck, output_dir, "precheck_frame_flags")
    _write_table_pair(comparison, output_dir, "precheck_vs_sam3_frame_comparison")
    for class_name, stem in (
        ("FN", "false_negative_frames"),
        ("FP", "false_positive_frames"),
        ("TP", "true_positive_frames"),
    ):
        _write_table_pair(
            comparison.loc[comparison["comparison_class"] == class_name].reset_index(drop=True),
            output_dir,
            stem,
        )
        contiguous_intervals(comparison, class_name=class_name).to_csv(
            output_dir / f"{stem.removesuffix('_frames')}_intervals.csv", index=False
        )
    sam3_problem_rows = comparison.loc[
        comparison["sam3_binary_label"] == "problem"
    ].copy()
    sam3_problem_rows["comparison_class"] = "SAM3_PROBLEM"
    contiguous_intervals(sam3_problem_rows, class_name="SAM3_PROBLEM").to_csv(
        output_dir / "sam3_problem_intervals.csv", index=False
    )
    for verdict, filename in (
        ("fail", "sam3_fail_intervals.csv"),
        ("review", "sam3_review_intervals.csv"),
    ):
        selected = comparison.loc[
            comparison["sam3_frame_verdict"] == verdict
        ].copy()
        selected["comparison_class"] = verdict.upper()
        contiguous_intervals(selected, class_name=verdict.upper()).to_csv(
            output_dir / filename, index=False
        )
    unevaluable = comparison.loc[~comparison["evaluable"]].copy()
    unevaluable["comparison_class"] = "UNEVALUABLE"
    contiguous_intervals(unevaluable, class_name="UNEVALUABLE").to_csv(
        output_dir / "unevaluable_intervals.csv", index=False
    )
    per_asset = _per_asset(comparison)
    per_asset.to_csv(output_dir / "precheck_vs_sam3_per_asset.csv", index=False)
    ranking = per_asset.sort_values(
        ["FN", "asset_id"], ascending=[False, True], kind="stable"
    ) if not per_asset.empty else per_asset
    ranking.to_csv(output_dir / "assets_ranked_by_false_negatives.csv", index=False)
    _module_coverage(comparison).to_csv(
        output_dir / "precheck_module_coverage.csv", index=False
    )
    _reason_matrix(comparison).to_csv(
        output_dir / "precheck_sam3_reason_matrix.csv", index=False
    )
    evidence.to_csv(output_dir / "review_evidence_manifest.csv", index=False)
    (output_dir / "precheck_vs_sam3_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare raw Precheck frame flags with exhaustive SAM3 results."
    )
    parser.add_argument("--sam3-frame-results", type=Path, required=True)
    parser.add_argument("--precheck-results", type=Path, action="append", required=True)
    parser.add_argument("--candidate-windows", type=Path, action="append")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--producer-run-config", type=Path)
    parser.add_argument("--allow-missing-positive-evidence", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    sam3_rows = read_records(args.sam3_frame_results)
    check_rows = read_precheck_artifacts(
        args.precheck_results, artifact_name="check_results.json"
    )
    candidate_rows = (
        read_precheck_artifacts(
            args.candidate_windows, artifact_name="candidate_windows.json"
        )
        if args.candidate_windows
        else []
    )
    manifest_rows = read_records(args.manifest) if args.manifest else sam3_rows
    lineage = (
        json.loads(args.producer_run_config.read_text(encoding="utf-8"))
        if args.producer_run_config
        else {}
    )
    build_comparison_outputs(
        sam3_frame_rows=sam3_rows,
        check_rows=check_rows,
        candidate_rows=candidate_rows,
        output_dir=args.output_dir,
        manifest_asset_ids=tuple(str(row.get("asset_id", "")) for row in manifest_rows),
        model_run_lineage=lineage,
        require_positive_evidence=not args.allow_missing_positive_evidence,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
