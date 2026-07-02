from __future__ import annotations

import csv
import json
from pathlib import Path

from acceptance_pull.models import ManifestAsset, PullResult, SampleRow, ValidationResult


def write_reports(
    output: Path,
    manifest: dict[str, ManifestAsset],
    validation: ValidationResult,
    samples: list[SampleRow],
    pulls: list[PullResult],
    minimum_count: int,
    seed: int,
    workers: int,
    sample_ratio: float,
) -> None:
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    with (reports / "id_consistency.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["issue_type", "asset_id", "detail"])
        for issue in validation.issues:
            writer.writerow([issue.issue_type, issue.asset_id, issue.detail])

    pulls_by_asset_type = {(result.asset_id, result.file_type): result for result in pulls}
    with (reports / "sample_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "asset_id",
                "scene",
                "task",
                "reason",
                "hdf5_source",
                "hdf5_target",
                "video_source",
                "video_target",
            ]
        )
        for row in samples:
            hdf5_pull = pulls_by_asset_type.get((row.asset_id, "hdf5"))
            video_pull = pulls_by_asset_type.get((row.asset_id, "video"))
            writer.writerow(
                [
                    row.asset_id,
                    row.scene,
                    row.task,
                    row.reason,
                    hdf5_pull.source if hdf5_pull else "",
                    str(hdf5_pull.target) if hdf5_pull else "",
                    video_pull.source if video_pull else "",
                    str(video_pull.target) if video_pull else "",
                ]
            )

    with (reports / "pull_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["asset_id", "file_type", "source", "target", "status", "error"])
        for result in pulls:
            writer.writerow(
                [result.asset_id, result.file_type, result.source, str(result.target), result.status, result.error]
            )

    summary = {
        "manifest_count": len(manifest),
        "valid_id_count": len(validation.valid_ids),
        "minimum_sample_count": minimum_count,
        "actual_sample_count": len(samples),
        "scene_count": len({row.scene for row in samples}),
        "task_count": len({row.task for row in samples}),
        "seed": seed,
        "workers": workers,
        "sample_ratio": sample_ratio,
        "pull_success_count": sum(1 for result in pulls if result.status == "success"),
        "pull_failed_count": sum(1 for result in pulls if result.status != "success"),
    }
    (reports / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
