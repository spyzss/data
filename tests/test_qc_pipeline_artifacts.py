from __future__ import annotations

import json
from pathlib import Path

import pytest

from qc_common.config import LoadedQcConfig
from qc_common.contracts import ModuleResult
from qc_common.frame_survival import FrameExclusion
from qc_pipeline.context import AssetContext


def _context(tmp_path: Path, asset_id: str = "asset-a") -> AssetContext:
    source = tmp_path / "source" / "clip.bin"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"source-v1")
    return AssetContext(
        asset_id=asset_id,
        batch_root=tmp_path,
        report_path=tmp_path / "quality_archive" / f"{asset_id}.json",
        source_files={"video": {"path": "source/clip.bin", "etag": "etag-v1"}},
        source_range=(10, 21),
        metadata={"supplier": "jdt"},
    )


def _config(tmp_path: Path) -> LoadedQcConfig:
    return LoadedQcConfig(
        path=tmp_path / "qc.yaml",
        raw={
            "schema_version": "qc_acceptance_config_schema.v2",
            "config_version": "qc_acceptance_v2.1.0",
            "config_name": "test",
            "pipeline": {"default_profile": "acceptance", "modules": ["video_quality"]},
            "execution_profiles": {},
            "modules": {
                "video_quality": {
                    "enabled": True,
                    "implementation": "video_quality.unified",
                    "module_version": "video-v1",
                    "parameters": {"threshold": 2},
                    "rules": {},
                }
            },
        },
        sha256="sha256:" + "1" * 64,
    )


def test_artifact_layout_and_file_identity_are_source_faithful(tmp_path: Path) -> None:
    from qc_pipeline.artifacts import artifact_for, file_identity

    context = _context(tmp_path)
    artifact = artifact_for(context, "video_quality")
    identity = file_identity(
        tmp_path / "source" / "clip.bin",
        batch_root=tmp_path,
        declared=context.source_files["video"],
    )

    assert artifact.directory == (
        tmp_path / "module_outputs" / "asset-a" / "video_quality"
    )
    assert artifact.required_files == ("video_quality_result.json", "run_config.json")
    assert identity == {
        "path": "source/clip.bin",
        "size": len(b"source-v1"),
        "mtime_ns": (tmp_path / "source" / "clip.bin").stat().st_mtime_ns,
        "etag": "etag-v1",
    }


def test_artifact_paths_and_source_identity_reject_batch_escape(tmp_path: Path) -> None:
    from qc_pipeline.artifacts import artifact_for, file_identity

    with pytest.raises(ValueError, match="asset_id must be a safe filename component"):
        artifact_for(_context(tmp_path, "../escape"), "precheck")

    outside = tmp_path.parent / "outside.bin"
    outside.write_bytes(b"outside")
    with pytest.raises(ValueError, match="source path must stay inside batch_root"):
        file_identity(outside, batch_root=tmp_path)


def test_run_fingerprint_is_canonical_and_changes_with_inputs(tmp_path: Path) -> None:
    from qc_pipeline.artifacts import build_run_fingerprint, canonical_sha256

    context = _context(tmp_path)
    config = _config(tmp_path)
    first = build_run_fingerprint(
        context=context,
        producer="video_quality",
        config=config,
        module_names=("video_quality",),
        source_names=("video",),
        implementation_version="producer-v1",
    )
    reordered = {key: first[key] for key in reversed(first)}
    assert canonical_sha256(first) == canonical_sha256(reordered)

    changed = build_run_fingerprint(
        context=context,
        producer="video_quality",
        config=config,
        module_names=("video_quality",),
        source_names=("video",),
        implementation_version="producer-v2",
    )
    assert canonical_sha256(first) != canonical_sha256(changed)


def test_module_result_round_trip_preserves_temporal_exclusion_lineage() -> None:
    from qc_pipeline.artifacts import module_result_from_dict

    original = ModuleResult(
        "keypoint_temporal",
        "fail",
        {"decision": "fail"},
        {},
        frame_exclusions=(
            FrameExclusion(
                start_frame=42,
                end_frame=42,
                module="keypoint_temporal",
                reason="keypoint_temporal.strong_temporal_failure",
                raw_severity="fail",
                hand_side="both",
                first_introduced_stage="keypoint_temporal",
                temporal_pair_start_frame=41,
                temporal_pair_end_frame=42,
                temporal_transition_attribution="target_frame",
            ),
        ),
    )

    restored = module_result_from_dict(original.to_dict())

    assert restored.frame_exclusions == original.frame_exclusions


def test_reusable_artifact_requires_exact_fingerprint_and_valid_files(
    tmp_path: Path,
) -> None:
    from qc_pipeline.artifacts import ProducerArtifact, reusable_artifact

    directory = tmp_path / "artifact"
    directory.mkdir()
    artifact = ProducerArtifact(
        producer="precheck",
        directory=directory,
        required_files=("check_results.json", "candidate_windows.json", "run_config.json"),
    )
    fingerprint = {"producer": "precheck", "value": 1}
    (directory / "check_results.json").write_text("[]", encoding="utf-8")
    (directory / "candidate_windows.json").write_text("[]", encoding="utf-8")
    (directory / "run_config.json").write_text(
        json.dumps(
            {
                "schema_version": "qc_producer_run_config.v1",
                "producer": "precheck",
                "outcome": "completed",
                "fingerprint": fingerprint,
            }
        ),
        encoding="utf-8",
    )

    assert reusable_artifact(artifact, fingerprint)
    assert not reusable_artifact(artifact, {"producer": "precheck", "value": 2})

    (directory / "candidate_windows.json").unlink()
    assert not reusable_artifact(artifact, fingerprint)
    (directory / "candidate_windows.json").write_text("{broken", encoding="utf-8")
    assert not reusable_artifact(artifact, fingerprint)


def test_staged_artifact_exception_preserves_previous_release(tmp_path: Path) -> None:
    from qc_pipeline.artifacts import ProducerArtifact, staged_artifact

    directory = tmp_path / "module_outputs" / "asset-a" / "precheck"
    directory.mkdir(parents=True)
    (directory / "check_results.json").write_text('[{"version": 1}]', encoding="utf-8")
    artifact = ProducerArtifact(
        producer="precheck",
        directory=directory,
        required_files=("check_results.json",),
    )
    before = (directory / "check_results.json").read_bytes()

    with pytest.raises(RuntimeError, match="producer failed"):
        with staged_artifact(artifact) as staging:
            (staging / "check_results.json").write_text(
                '[{"version": 2}]', encoding="utf-8"
            )
            raise RuntimeError("producer failed")

    assert (directory / "check_results.json").read_bytes() == before
    assert list(directory.parent.glob(".precheck.staging-*")) == []


def test_promote_artifact_replaces_complete_directory(tmp_path: Path) -> None:
    from qc_pipeline.artifacts import (
        ProducerArtifact,
        promote_artifact,
        staged_artifact,
    )

    directory = tmp_path / "module_outputs" / "asset-a" / "precheck"
    directory.mkdir(parents=True)
    (directory / "check_results.json").write_text('[{"version": 1}]', encoding="utf-8")
    artifact = ProducerArtifact(
        producer="precheck",
        directory=directory,
        required_files=("check_results.json",),
    )

    with staged_artifact(artifact) as staging:
        (staging / "check_results.json").write_text('[{"version": 2}]', encoding="utf-8")
        promote_artifact(staging, artifact)

    assert json.loads((directory / "check_results.json").read_text()) == [
        {"version": 2}
    ]
    assert list(directory.parent.glob(".precheck.backup-*")) == []



def test_file_identity_allows_opted_in_symlinked_source(
    tmp_path: Path,
) -> None:
    from qc_pipeline.artifacts import file_identity

    outside_dir = (
        tmp_path.parent
        / f"{tmp_path.name}-outside-source"
    )
    outside_dir.mkdir()
    target = outside_dir / "clip.bin"
    target.write_bytes(b"external-source-v1")

    link = tmp_path / "source" / "clip.bin"
    link.parent.mkdir(parents=True)
    link.symlink_to(target)

    # Strict/default behavior still rejects a symlink whose
    # resolved target leaves batch_root.
    with pytest.raises(
        ValueError,
        match="source path must stay inside batch_root",
    ):
        file_identity(
            link,
            batch_root=tmp_path,
        )

    identity = file_identity(
        link,
        batch_root=tmp_path,
        allow_symlinked_sources=True,
    )

    assert identity == {
        "path": "source/clip.bin",
        "size": len(b"external-source-v1"),
        "mtime_ns": target.stat().st_mtime_ns,
    }

    # Opt-in allows only links lexically staged under batch_root;
    # it must not allow direct external paths.
    with pytest.raises(
        ValueError,
        match="source path must stay inside batch_root",
    ):
        file_identity(
            target,
            batch_root=tmp_path,
            allow_symlinked_sources=True,
        )
