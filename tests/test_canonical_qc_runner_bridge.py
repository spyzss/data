from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from canonical_qc import StandardHdf5Adapter, source_fingerprint
from canonical_qc.bridge import CanonicalQcBridge
from qc_common.config import load_qc_acceptance_config
from qc_common.types import ClipInputs
from qc_pipeline.context import AssetContext
from qc_pipeline.runners import precheck, sam3_containment, video_quality
from tests.fixtures import solid_frame, write_standard_hdf5_episode, write_test_video


def _bridge_context(tmp_path: Path, *, hand_quality: str = "missing"):
    source_root = tmp_path / "asset-001"
    write_standard_hdf5_episode(source_root, hand_quality=hand_quality)
    episode = StandardHdf5Adapter().load(source_root)
    bridge = CanonicalQcBridge(episode, source_root=source_root)
    return bridge, bridge.asset_context(
        batch_root=tmp_path,
        report_path=tmp_path / "quality_archive" / "asset-001.json",
    )


def test_precheck_canonical_wins_over_poisoned_declared_clip_and_legacy_falls_back(
    tmp_path: Path,
) -> None:
    bridge, canonical = _bridge_context(tmp_path / "canonical")
    poison = ClipInputs(episode_idx=99, frame_indices=[])
    canonical_metadata = dict(canonical.metadata)
    canonical_metadata["clip_inputs"] = poison
    preferred = AssetContext(
        canonical.asset_id,
        canonical.batch_root,
        canonical.report_path,
        canonical.source_files,
        metadata=canonical_metadata,
    )

    loaded = precheck._load_clip(preferred, "keypoint_presence")

    assert loaded is not poison
    assert loaded.num_frames == bridge.episode.time_axis.frame_count
    legacy = AssetContext(
        "legacy",
        tmp_path,
        tmp_path / "quality_archive" / "legacy.json",
        {},
        metadata={"clip_inputs": poison},
    )
    legacy_loaded = precheck._load_clip(legacy, "keypoint_presence")
    assert legacy_loaded.episode_idx == poison.episode_idx
    assert legacy_loaded.frame_indices == poison.frame_indices


def test_video_runner_composes_logical_range_with_physical_video_offset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bridge, context = _bridge_context(tmp_path)
    video_path = bridge.video_path()
    write_test_video(
        video_path,
        [solid_frame(20 + index * 10) for index in range(6)],
        fps=10.0,
    )
    video_bytes = video_path.read_bytes()
    import hashlib

    digest = hashlib.sha256(video_bytes).hexdigest()
    files = tuple(
        replace(item, size_bytes=len(video_bytes), sha256=digest)
        if item.role == "main_video"
        else item
        for item in bridge.episode.provenance.source_files
    )
    provenance = replace(
        bridge.episode.provenance,
        source_files=files,
        source_fingerprint=source_fingerprint(
            files,
            source_schema_version=bridge.episode.identity.source_schema_version,
            adapter_id=bridge.episode.provenance.adapter_id,
            adapter_version=bridge.episode.provenance.adapter_version,
        ),
    )
    shifted = replace(
        bridge.episode,
        provenance=provenance,
        main_video=replace(
            bridge.episode.main_video,
            sha256=digest,
            source_frame_range=(3, 6),
        ),
    )
    shifted_bridge = CanonicalQcBridge(shifted, source_root=bridge.source_root)
    shifted_context = shifted_bridge.asset_context(
        batch_root=tmp_path,
        report_path=context.report_path,
    )

    from acceptance_pull import video_quality as producer

    original = producer.analyze_video_frame_range
    calls: list[tuple[int, int]] = []

    def recording(*args, **kwargs):
        calls.append((int(args[2]), int(args[3])))
        return original(*args, **kwargs)

    monkeypatch.setattr(producer, "analyze_video_frame_range", recording)

    result = video_quality.run(shifted_context, load_qc_acceptance_config())

    assert result.module == "video_quality"
    assert calls == [(3, 5)]


def test_sam3_runner_uses_derived_canonical_projection_not_source_hdf5(
    tmp_path: Path,
) -> None:
    bridge, base = _bridge_context(tmp_path)
    candidates = tmp_path / "candidate-windows.json"
    candidates.write_text(
        json.dumps(
            [
                {
                    "asset_id": "asset-001",
                    "start_frame": 0,
                    "end_frame": 0,
                    "hand_side": "both",
                }
            ]
        ),
        encoding="utf-8",
    )
    sources = dict(base.source_files)
    sources["candidate_windows"] = {"path": "candidate-windows.json"}
    context = AssetContext(
        base.asset_id,
        base.batch_root,
        base.report_path,
        sources,
        metadata={**dict(base.metadata), "manifest_row": {"parquet_path": "poison"}},
    )

    class Segmenter:
        def segment_frame(self, frame, queries, config):
            return [
                SimpleNamespace(
                    mask=np.ones(frame.shape[:2], dtype=bool),
                    category="hand",
                )
            ]

    result = sam3_containment.runner(lambda: Segmenter())(
        context, load_qc_acceptance_config()
    )

    projection = (
        tmp_path
        / ".qc_pipeline"
        / "asset-001"
        / "sam3"
        / "canonical_keypoints_2d.parquet"
    )
    manifest = json.loads(
        (
            tmp_path
            / ".qc_pipeline"
            / "asset-001"
            / "sam3"
            / "manifest.jsonl"
        ).read_text(encoding="utf-8")
    )
    assert result.module == "sam3_containment"
    assert projection.is_file()
    assert manifest["parquet_path"] == str(projection)
    source_data = next(
        item
        for item in bridge.episode.provenance.source_files
        if item.role == "episode_data"
    )
    assert manifest["parquet_path"] != str(
        source_data.relative_path
    )
