from __future__ import annotations

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from canonical_qc import StandardHdf5Adapter, source_fingerprint
from canonical_qc import CanonicalInputError
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
            main_video_source_frame_range=(3, 6),
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


def test_shifted_video_report_rebases_all_producer_frame_coordinates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "asset-001"
    write_standard_hdf5_episode(source_root, frame_count=10)
    episode = StandardHdf5Adapter().load(source_root)
    video_path = source_root / "main.mp4"
    write_test_video(video_path, [solid_frame(80) for _ in range(15)], fps=10.0)
    digest = hashlib.sha256(video_path.read_bytes()).hexdigest()
    files = tuple(
        replace(
            item,
            size_bytes=video_path.stat().st_size,
            sha256=digest,
        )
        if item.role == "main_video"
        else item
        for item in episode.provenance.source_files
    )
    shifted = replace(
        episode,
        provenance=replace(
            episode.provenance,
            source_files=files,
            source_fingerprint=source_fingerprint(
                files,
                source_schema_version=episode.identity.source_schema_version,
                adapter_id=episode.provenance.adapter_id,
                adapter_version=episode.provenance.adapter_version,
                main_video_source_frame_range=(5, 15),
            ),
        ),
        main_video=replace(
            episode.main_video,
            sha256=digest,
            source_frame_range=(5, 15),
        ),
    )
    bridge = CanonicalQcBridge(shifted, source_root=source_root)
    context = bridge.asset_context(
        batch_root=tmp_path,
        report_path=tmp_path / "quality_archive" / "asset-001.json",
    )

    from acceptance_pull import video_quality as producer

    original = producer.analyze_video_frame_range
    calls: list[tuple[int, int]] = []

    def physical_analysis(*args, **kwargs):
        calls.append((int(args[2]), int(args[3])))
        analysis = original(*args, **kwargs)
        interval = producer.FrozenInterval(
            start_frame=5,
            end_frame=14,
            frame_count=10,
            start_time_sec=0.5,
            end_time_sec=1.5,
            duration_sec=1.0,
            duration_ms=1000.0,
        )
        return replace(
            analysis,
            metrics=replace(
                analysis.metrics,
                frozen_intervals=(interval,),
                errors=("range_decode_failed:14",),
            ),
        )

    monkeypatch.setattr(producer, "analyze_video_frame_range", physical_analysis)

    result = video_quality.run(context, load_qc_acceptance_config())

    assert calls == [(5, 14)]
    interval = result.metrics["freeze_metrics"]["frozen_intervals"][0]
    assert interval["start_frame"] == 0
    assert interval["end_frame"] == 9
    assert interval["start_time_sec"] == 0.0
    assert interval["end_time_sec"] == 1.0
    assert result.runtime["errors"] == ["range_decode_failed:9"]
    assert result.evidence[0].start_frame == 0
    assert result.evidence[0].end_frame == 9
    assert all(issue.context["start_frame"] == 0 for issue in result.issues)
    assert all(issue.context["end_frame"] == 9 for issue in result.issues)


def test_video_runner_fails_if_nonvideo_source_mutates_during_producer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bridge, context = _bridge_context(tmp_path)
    hdf5 = bridge.source_root / "asset-001.h5"
    from acceptance_pull import video_quality as producer

    original = producer.analyze_video

    def mutating(*args, **kwargs):
        metrics = original(*args, **kwargs)
        with hdf5.open("ab") as stream:
            stream.write(b"drift")
        return metrics

    monkeypatch.setattr(producer, "analyze_video", mutating)

    with pytest.raises(CanonicalInputError, match="source"):
        video_quality.run(context, load_qc_acceptance_config())


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

    staging_parent = tmp_path / ".qc_pipeline" / "asset-001" / "sam3"
    runs = list(staging_parent.glob("run-*"))
    assert len(runs) == 1
    projection = runs[0] / "canonical_keypoints_2d.parquet"
    manifest = json.loads(
        (runs[0] / "manifest.jsonl").read_text(encoding="utf-8")
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


def test_canonical_sam3_runner_closes_external_source_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _bridge, base = _bridge_context(tmp_path)
    candidates = tmp_path / "candidate-windows.json"
    candidates.write_text(
        '[{"asset_id":"asset-001","start_frame":0,"end_frame":0,'
        '"hand_side":"both"}]',
        encoding="utf-8",
    )
    context = AssetContext(
        base.asset_id,
        base.batch_root,
        base.report_path,
        {**dict(base.source_files), "candidate_windows": {"path": candidates.name}},
        metadata=base.metadata,
    )
    from tools import run_manifest_sam3_containment as producer

    closed: list[object] = []
    original_close = producer.ManifestSourceCache.close

    def recording_close(cache):
        closed.append(cache)
        return original_close(cache)

    monkeypatch.setattr(producer.ManifestSourceCache, "close", recording_close)

    class Segmenter:
        def segment_frame(self, frame, queries, config):
            return [SimpleNamespace(mask=np.ones(frame.shape[:2], dtype=bool), category="hand")]

    sam3_containment.runner(lambda: Segmenter())(
        context, load_qc_acceptance_config()
    )

    assert len(closed) == 1


def test_canonical_sam3_concurrent_runs_use_distinct_staging_directories(
    tmp_path: Path,
) -> None:
    _bridge, base = _bridge_context(tmp_path)
    candidates = tmp_path / "candidate-windows.json"
    candidates.write_text(
        '[{"asset_id":"asset-001","start_frame":0,"end_frame":0,'
        '"hand_side":"both"}]',
        encoding="utf-8",
    )
    context = AssetContext(
        base.asset_id,
        base.batch_root,
        base.report_path,
        {**dict(base.source_files), "candidate_windows": {"path": candidates.name}},
        metadata=base.metadata,
    )

    class Segmenter:
        def segment_frame(self, frame, queries, config):
            return [SimpleNamespace(mask=np.ones(frame.shape[:2], dtype=bool), category="hand")]

    def execute():
        return sam3_containment.runner(lambda: Segmenter())(
            context, load_qc_acceptance_config()
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: execute(), range(2)))

    runs = sorted(
        (tmp_path / ".qc_pipeline" / "asset-001" / "sam3").glob("run-*")
    )
    assert len(results) == 2
    assert len(runs) == 2
    assert len({path.name for path in runs}) == 2
    assert all((path / "canonical_keypoints_2d.parquet").is_file() for path in runs)


def test_shifted_canonical_sam3_reads_physical_frame_and_reports_logical_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "asset-001"
    write_standard_hdf5_episode(source_root)
    episode = StandardHdf5Adapter().load(source_root)
    video = source_root / "main.mp4"
    write_test_video(video, [solid_frame(40 + index * 10) for index in range(6)], fps=10.0)
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    files = tuple(
        replace(item, size_bytes=video.stat().st_size, sha256=digest)
        if item.role == "main_video"
        else item
        for item in episode.provenance.source_files
    )
    shifted_range = (3, 6)
    shifted = replace(
        episode,
        provenance=replace(
            episode.provenance,
            source_files=files,
            source_fingerprint=source_fingerprint(
                files,
                source_schema_version=episode.identity.source_schema_version,
                adapter_id=episode.provenance.adapter_id,
                adapter_version=episode.provenance.adapter_version,
                main_video_source_frame_range=shifted_range,
            ),
        ),
        main_video=replace(
            episode.main_video,
            sha256=digest,
            source_frame_range=shifted_range,
        ),
    )
    bridge = CanonicalQcBridge(shifted, source_root=source_root)
    candidates = tmp_path / "candidate-windows.json"
    candidates.write_text(
        '[{"asset_id":"asset-001","start_frame":0,"end_frame":0,'
        '"hand_side":"both"}]',
        encoding="utf-8",
    )
    context = bridge.asset_context(
        batch_root=tmp_path,
        report_path=tmp_path / "quality_archive" / "asset-001.json",
        supplemental_source_files={
            "candidate_windows": {"path": candidates.name}
        },
    )
    from tools import run_manifest_sam3_containment as producer

    physical_reads: list[int] = []
    original_read = producer.ManifestSourceCache.read_frame

    def recording_read(cache, path, frame_idx):
        physical_reads.append(int(frame_idx))
        return original_read(cache, path, frame_idx)

    monkeypatch.setattr(producer.ManifestSourceCache, "read_frame", recording_read)

    class Segmenter:
        def segment_frame(self, frame, queries, config):
            return [SimpleNamespace(mask=np.ones(frame.shape[:2], dtype=bool), category="hand")]

    result = sam3_containment.runner(lambda: Segmenter())(
        context, load_qc_acceptance_config()
    )

    run_dir = next(
        (tmp_path / ".qc_pipeline" / "asset-001" / "sam3").glob("run-*")
    )
    rows = json.loads(
        (run_dir / "output" / "frame_keypoint_containment.json").read_text(
            encoding="utf-8"
        )
    )
    assert physical_reads == [3]
    assert {row["frame_idx"] for row in rows} == {0}
    assert {row["source_frame_idx"] for row in rows} == {0}
    assert result.module == "sam3_containment"


def test_canonical_sam3_fails_if_source_mutates_during_producer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, base = _bridge_context(tmp_path)
    candidates = tmp_path / "candidate-windows.json"
    candidates.write_text(
        '[{"asset_id":"asset-001","start_frame":0,"end_frame":0,'
        '"hand_side":"both"}]',
        encoding="utf-8",
    )
    context = AssetContext(
        base.asset_id,
        base.batch_root,
        base.report_path,
        {**dict(base.source_files), "candidate_windows": {"path": candidates.name}},
        metadata=base.metadata,
    )
    from tools import run_manifest_sam3_containment as producer

    original = producer.run_manifest_sam3_containment

    def mutating(**kwargs):
        summary = original(**kwargs)
        with (bridge.source_root / "asset-001.h5").open("ab") as stream:
            stream.write(b"drift")
        return summary

    monkeypatch.setattr(producer, "run_manifest_sam3_containment", mutating)

    class Segmenter:
        def segment_frame(self, frame, queries, config):
            return [SimpleNamespace(mask=np.ones(frame.shape[:2], dtype=bool), category="hand")]

    with pytest.raises(CanonicalInputError, match="provenance.source_files"):
        sam3_containment.runner(lambda: Segmenter())(
            context, load_qc_acceptance_config()
        )
