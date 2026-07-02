from __future__ import annotations

from pathlib import Path

from acceptance_pull.models import IdIssue, ManifestAsset, ValidationResult


def validate_ids(
    manifest: dict[str, ManifestAsset],
    hdf5: dict[str, Path],
    video: dict[str, Path],
) -> ValidationResult:
    manifest_ids = set(manifest)
    hdf5_ids = set(hdf5)
    video_ids = set(video)
    valid_ids = manifest_ids & hdf5_ids & video_ids
    issues: list[IdIssue] = []

    for asset_id in sorted(manifest_ids - hdf5_ids - video_ids):
        issues.append(IdIssue("manifest_only", asset_id, "present in manifest only"))
    for asset_id in sorted(hdf5_ids - manifest_ids - video_ids):
        issues.append(IdIssue("hdf5_only", asset_id, "present in hdf5 only"))
    for asset_id in sorted(video_ids - manifest_ids - hdf5_ids):
        issues.append(IdIssue("video_only", asset_id, "present in video only"))
    for asset_id in sorted((manifest_ids & hdf5_ids) - video_ids):
        issues.append(IdIssue("missing_video", asset_id, "manifest and hdf5 exist, video missing"))
    for asset_id in sorted((manifest_ids & video_ids) - hdf5_ids):
        issues.append(IdIssue("missing_hdf5", asset_id, "manifest and video exist, hdf5 missing"))
    for asset_id in sorted((hdf5_ids & video_ids) - manifest_ids):
        issues.append(IdIssue("missing_manifest", asset_id, "hdf5 and video exist, manifest missing"))

    return ValidationResult(valid_ids=valid_ids, issues=issues)
