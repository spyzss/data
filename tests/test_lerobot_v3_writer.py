from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import pytest

from canonical_qc import CanonicalQcBridge, StandardHdf5Adapter, StandardLeRobotAdapter
from canonical_qc.provenance import semantic_fingerprint
from canonical_qc.video_probe import probe_video
from lerobot_v3_publisher import (
    PublishPrerequisiteError,
    PublishRequest,
    StagedRelease,
    validate_publish_request,
    write_staging,
)
import lerobot_v3_publisher.writer as publisher_writer
from tests.fixtures import (
    solid_frame,
    write_standard_hdf5_episode,
    write_standard_lerobot_dataset,
    write_test_video,
)
from tests.test_lerobot_v3_publish_prerequisites import _write_publish_fixture


EXPECTED_PAYLOADS = {
    "meta/info.json",
    "meta/episodes/chunk-000/file-000.parquet",
    "meta/tasks.parquet",
    "meta/subtask.parquet",
    "meta/stats.json",
    "meta/episode_semantics.jsonl",
    "data/chunk-000/file-000.parquet",
    "videos/observation.images.main/chunk-000/file-000.mp4",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _plan_for_episode(tmp_path: Path, source_root: Path, episode: object):
    baseline_request, report = _write_publish_fixture(tmp_path / "report-fixture")
    report_path = baseline_request.qc_report_path
    batch_root = Path(
        os.path.commonpath((source_root.resolve(), report_path.resolve()))
    )
    context = CanonicalQcBridge(episode, source_root=source_root).asset_context(
        batch_root=batch_root,
        report_path=report_path,
    )
    report["asset_id"] = episode.identity.asset_id
    report["supplier_id"] = episode.identity.supplier_id
    report["source_files"] = copy.deepcopy(dict(context.source_files))
    report["canonical_binding"].update(  # type: ignore[union-attr]
        semantic_fingerprint=semantic_fingerprint(episode),
        source_fingerprint=episode.provenance.source_fingerprint,
    )
    report["canonical_qc_range"]["end_frame_exclusive"] = (  # type: ignore[index]
        episode.time_axis.frame_count
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    request = PublishRequest(
        episode=episode,
        canonical_revision=3,
        canonical_source_root=source_root,
        qc_report_path=report_path,
        expected_report_revision=9,
        release_root=(tmp_path / "curated").absolute(),
    )
    return validate_publish_request(request)


def _all_artifact_bytes(staged: StagedRelease) -> dict[str, bytes]:
    return {
        path.relative_to(staged.root).as_posix(): path.read_bytes()
        for path in staged.root.rglob("*")
        if path.is_file()
    }


def _assert_no_absolute_path_strings(value: object) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _assert_no_absolute_path_strings(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_absolute_path_strings(child)
    elif isinstance(value, str):
        assert not value.startswith("/")


def test_write_staging_emits_complete_registered_curated_v3_layout(
    tmp_path: Path,
) -> None:
    request, _report = _write_publish_fixture(tmp_path)
    plan = validate_publish_request(request)

    staged = write_staging(plan, request.release_root / ".staging")

    assert isinstance(staged, StagedRelease)
    assert staged.plan is plan
    assert staged.root.parent == request.release_root / ".staging"
    assert staged.root.name == staged.transaction_id
    with pytest.raises(FrozenInstanceError):
        staged.transaction_id = "changed"  # type: ignore[misc]
    paths = {
        path.relative_to(staged.root).as_posix()
        for path in staged.root.rglob("*")
        if path.is_file()
    }
    assert paths == EXPECTED_PAYLOADS | {"release_manifest.json", "checksums.sha256"}

    info = json.loads((staged.root / "meta/info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["data_path"] == "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    assert info["video_path"] == (
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    )
    assert info["total_episodes"] == 1
    assert info["total_frames"] == request.episode.time_axis.frame_count
    assert info["total_tasks"] == 1
    assert info["features"]["index"] == {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    }
    assert info["features"]["timestamp"] == {
        "dtype": "float64",
        "shape": [1],
        "names": None,
    }
    assert info["features"]["timestamp_ns"]["dtype"] == "int64"
    assert info["fps_num"] == request.episode.time_axis.fps_num
    assert info["fps_den"] == request.episode.time_axis.fps_den

    episode_row = pq.read_table(
        staged.root / "meta/episodes/chunk-000/file-000.parquet"
    ).to_pylist()[0]
    assert episode_row["tasks"] == [request.episode.semantics.task_en]
    assert episode_row["dataset_from_index"] == 0
    assert episode_row["dataset_to_index"] == request.episode.time_axis.frame_count
    assert episode_row["data/chunk_index"] == 0
    assert episode_row["data/file_index"] == 0
    assert episode_row["videos/observation.images.main/from_timestamp"] == 0.0
    assert episode_row["videos/observation.images.main/to_timestamp"] == pytest.approx(0.3)
    tasks = pd.read_parquet(staged.root / "meta/tasks.parquet")
    assert tasks.index.name == "task"
    assert tasks.index.tolist() == [request.episode.semantics.task_en]
    assert tasks["task_index"].tolist() == [0]

    manifest_payload = json.loads((staged.root / "release_manifest.json").read_text())
    assert manifest_payload["schema_version"] == "curated_lerobot_v3_release_manifest.v1"
    assert {row["relative_path"] for row in manifest_payload["files"]} == EXPECTED_PAYLOADS
    assert manifest_payload["video_materialization"] == {
        "relative_path": "videos/observation.images.main/chunk-000/file-000.mp4",
        "source_relative_path": request.episode.main_video.path,
        "source_frame_range": list(request.episode.main_video.source_frame_range),
        "source_sha256": request.episode.main_video.sha256,
        "target_sha256": request.episode.main_video.sha256,
        "method": "verified_copy",
    }
    assert manifest_payload["toolchain"]["schema_version"] == (
        "curated_lerobot_v3_toolchain.v1"
    )
    assert manifest_payload["toolchain"]["python_version"]
    assert manifest_payload["toolchain"]["numpy_version"] == np.__version__
    assert len(manifest_payload["toolchain"]["ffmpeg_signature_sha256"]) == 64
    assert len(manifest_payload["toolchain"]["libx264_signature_sha256"]) == 64
    assert plan.publisher_version.endswith(
        manifest_payload["toolchain"]["fingerprint"][:16]
    )
    assert staged.manifest_sha256 == _sha256(staged.root / "release_manifest.json")
    assert staged.checksums_sha256 == _sha256(staged.root / "checksums.sha256")

    checksum_rows = (staged.root / "checksums.sha256").read_text().splitlines()
    registered = {line.split("  ", 1)[1] for line in checksum_rows}
    assert registered == EXPECTED_PAYLOADS | {"release_manifest.json"}
    for line in checksum_rows:
        digest, relative = line.split("  ", 1)
        assert digest == _sha256(staged.root / relative)


def test_official_lerobot_reader_version_is_frozen() -> None:
    requirements = (Path(__file__).parents[1] / "requirements.txt").read_text()
    assert "lerobot[dataset]==0.6.0" in requirements.splitlines()


def test_frame_table_uses_fixed_schema_authoritative_ns_and_half_open_subtasks(
    tmp_path: Path,
) -> None:
    request, _report = _write_publish_fixture(tmp_path)
    episode = request.episode
    subtasks = (
        replace(episode.semantics.subtask_sequence[0], end_frame_exclusive=1),
        replace(
            episode.semantics.subtask_sequence[0],
            subtask_id="subtask_002",
            start_frame=1,
            description_cn="放置物体",
            description_en="place the object",
        ),
    )
    episode = replace(episode, semantics=replace(episode.semantics, subtask_sequence=subtasks))
    plan = _plan_for_episode(tmp_path / "split", request.canonical_source_root, episode)

    staged = write_staging(plan, plan.request.release_root / ".staging")
    table = pq.read_table(staged.root / "data/chunk-000/file-000.parquet")

    assert table.schema == pa.schema(
        [
            pa.field("index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("frame_index", pa.int64()),
            pa.field("timestamp", pa.float64()),
            pa.field("timestamp_ns", pa.int64()),
            pa.field("observation.hand_keypoints_3d", pa.list_(pa.list_(pa.list_(pa.float32(), 3), 21), 2)),
            pa.field("observation.hand_joint_valid_3d", pa.list_(pa.list_(pa.bool_(), 21), 2)),
            pa.field("observation.hand_keypoints_2d", pa.list_(pa.list_(pa.list_(pa.float32(), 2), 21), 2)),
            pa.field("observation.hand_joint_valid_2d", pa.list_(pa.list_(pa.bool_(), 21), 2)),
            pa.field("task_index", pa.int64()),
            pa.field("subtask_index", pa.int64()),
        ]
    )
    assert table["timestamp_ns"].to_pylist() == episode.time_axis.timestamps_ns.tolist()
    expected_float = (
        episode.time_axis.timestamps_ns - episode.time_axis.timestamps_ns[0]
    ) / 1_000_000_000
    assert table["timestamp"].to_pylist() == expected_float.tolist()
    assert table["subtask_index"].to_pylist() == [0, 1, 1]
    assert pq.read_table(staged.root / "meta/subtask.parquet").to_pylist() == [
        {
            "episode_index": 0,
            "subtask_index": 0,
            "subtask_id": "subtask_001",
            "start_frame": 0,
            "end_frame_exclusive": 1,
            "description_cn": "拿起物体",
            "description_en": "pick up the object",
        },
        {
            "episode_index": 0,
            "subtask_index": 1,
            "subtask_id": "subtask_002",
            "start_frame": 1,
            "end_frame_exclusive": 3,
            "description_cn": "放置物体",
            "description_en": "place the object",
        },
    ]


def test_stats_cover_published_numeric_features_without_fabricating_video_stats(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset-001"
    write_standard_hdf5_episode(source, hand_quality="provided")
    episode = StandardHdf5Adapter().load(source)
    plan = _plan_for_episode(tmp_path / "publish", source, episode)

    staged = write_staging(plan, plan.request.release_root / ".staging")
    stats = json.loads((staged.root / "meta/stats.json").read_text())

    assert set(stats) == {
        "episode_index",
        "frame_index",
        "index",
        "observation.hand_joint_valid_2d",
        "observation.hand_joint_valid_3d",
        "observation.hand_keypoints_2d",
        "observation.hand_keypoints_3d",
        "subtask_index",
        "supplier.hand_quality.normalized_score",
        "supplier.hand_quality.raw_value",
        "task_index",
        "timestamp",
        "timestamp_ns",
    }
    assert set(stats["timestamp"]) == {"count", "min", "max"}
    assert set(stats["timestamp_ns"]) == {"count", "min", "max"}
    for key in ("index", "episode_index", "frame_index", "task_index", "subtask_index"):
        assert set(stats[key]) == {"count", "min", "max"}
    for key in (
        "observation.hand_joint_valid_2d",
        "observation.hand_joint_valid_3d",
    ):
        assert set(stats[key]) == {"count", "min", "max", "true_count"}
    for key in (
        "observation.hand_keypoints_2d",
        "observation.hand_keypoints_3d",
        "supplier.hand_quality.normalized_score",
        "supplier.hand_quality.raw_value",
    ):
        assert set(stats[key]) == {"count", "min", "max", "mean", "std"}
    assert "observation.images.main" not in stats
    assert stats["observation.hand_keypoints_3d"]["count"] == [
        episode.time_axis.frame_count
    ]


def test_revalidates_before_creating_any_staging_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _report = _write_publish_fixture(tmp_path)
    plan = validate_publish_request(request)
    staging_base = request.release_root / ".staging"
    sentinel = RuntimeError("revalidation sentinel")

    def fail_revalidation(candidate: object) -> None:
        assert candidate is plan
        raise sentinel

    monkeypatch.setattr(publisher_writer, "revalidate_publish_plan", fail_revalidation)
    with pytest.raises(RuntimeError, match="revalidation sentinel"):
        write_staging(plan, staging_base)
    assert not staging_base.exists()


def test_source_bytes_remain_unchanged_and_video_never_shares_inode(
    tmp_path: Path,
) -> None:
    request, _report = _write_publish_fixture(tmp_path)
    plan = validate_publish_request(request)
    before = _source_bytes(request.canonical_source_root)
    source_video = request.canonical_source_root / request.episode.main_video.path

    staged = write_staging(plan, request.release_root / ".staging")
    target_video = staged.root / "videos/observation.images.main/chunk-000/file-000.mp4"

    assert _source_bytes(request.canonical_source_root) == before
    assert _sha256(target_video) == _sha256(source_video)
    assert (target_video.stat().st_dev, target_video.stat().st_ino) != (
        source_video.stat().st_dev,
        source_video.stat().st_ino,
    )


def test_lerobot_source_is_rewritten_but_full_aligned_video_is_verified_copy(
    tmp_path: Path,
) -> None:
    source = write_standard_lerobot_dataset(tmp_path / "supplier")
    episode = StandardLeRobotAdapter().load(source)
    plan = _plan_for_episode(tmp_path / "publish", source, episode)
    source_data = next((source / "data").rglob("*.parquet"))
    source_info = source / "meta/info.json"
    source_video = next((source / "videos").rglob("*.mp4"))

    staged = write_staging(plan, plan.request.release_root / ".staging")

    assert (staged.root / "meta/info.json").read_bytes() != source_info.read_bytes()
    assert (staged.root / "data/chunk-000/file-000.parquet").read_bytes() != source_data.read_bytes()
    target_video = staged.root / "videos/observation.images.main/chunk-000/file-000.mp4"
    assert _sha256(target_video) == _sha256(source_video)
    assert target_video.stat().st_ino != source_video.stat().st_ino
    assert staged.manifest.video_materialization.method == "verified_copy"


def test_official_v3_staging_roundtrips_through_standard_adapter(
    tmp_path: Path,
) -> None:
    request, _report = _write_publish_fixture(tmp_path)
    plan = validate_publish_request(request)

    staged = write_staging(plan, request.release_root / ".staging")
    loaded = StandardLeRobotAdapter().load(staged.root)

    assert loaded.identity.source_format == "lerobot"
    assert loaded.identity.source_schema_version == "lerobot_v3.0"
    assert loaded.time_axis.timestamps_ns.tolist() == request.episode.time_axis.timestamps_ns.tolist()
    assert loaded.main_video.source_frame_range == (0, request.episode.time_axis.frame_count)
    assert semantic_fingerprint(loaded) == semantic_fingerprint(request.episode)


def test_rational_fps_and_nonzero_nanosecond_origin_roundtrip_without_float_loss(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset-001"
    _hdf5_path, video_path = write_standard_hdf5_episode(
        source, fps_num=30_000, fps_den=1_001
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=32x24:r=30000/1001",
            "-frames:v",
            "3",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(video_path),
        ],
        check=True,
    )
    episode = StandardHdf5Adapter().load(source)
    origin = 1_700_000_000_000_000_000
    episode = replace(
        episode,
        time_axis=replace(
            episode.time_axis,
            timestamps_ns=episode.time_axis.timestamps_ns + origin,
        ),
    )
    plan = _plan_for_episode(tmp_path / "publish", source, episode)

    staged = write_staging(plan, plan.request.release_root / ".staging")
    table = pq.read_table(staged.root / "data/chunk-000/file-000.parquet")
    loaded = StandardLeRobotAdapter().load(staged.root)

    assert table["timestamp_ns"].to_pylist() == episode.time_axis.timestamps_ns.tolist()
    floats = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
    expected = (
        episode.time_axis.timestamps_ns - episode.time_axis.timestamps_ns[0]
    ) / 1_000_000_000
    assert np.max(np.abs((floats - expected) * 1_000_000_000)) <= 1
    assert loaded.time_axis.fps_num == 30_000
    assert loaded.time_axis.fps_den == 1_001
    assert loaded.time_axis.timestamps_ns.tolist() == episode.time_axis.timestamps_ns.tolist()


def test_shared_lerobot_shifted_video_range_is_trimmed_to_canonical_episode(
    tmp_path: Path,
) -> None:
    source = write_standard_lerobot_dataset(
        tmp_path / "supplier", episodes=((3, "asset-three"), (7, "asset-seven"))
    )
    data_paths = sorted((source / "data").rglob("*.parquet"))
    first, second = (pq.read_table(path) for path in data_paths)
    pq.write_table(pa.concat_tables([first, second]), data_paths[0])
    data_paths[1].unlink()
    episode_path = next((source / "meta/episodes").rglob("*.parquet"))
    rows = pq.read_table(episode_path).to_pylist()
    rows[1].update(
        {
            "data/file_index": 0,
            "videos/observation.images.main/file_index": 0,
            "dataset_from_index": 3,
            "dataset_to_index": 6,
            "videos/observation.images.main/from_index": 3,
            "videos/observation.images.main/to_index": 6,
        }
    )
    pq.write_table(pa.Table.from_pylist(rows), episode_path)
    videos = sorted((source / "videos").rglob("*.mp4"))
    write_test_video(videos[0], [solid_frame(20 + i * 10) for i in range(6)], fps=10.0)
    videos[1].unlink()
    episode = StandardLeRobotAdapter().load(source, episode_index=7)
    assert episode.main_video.source_frame_range == (3, 6)
    plan = _plan_for_episode(tmp_path / "publish", source, episode)
    source_before = _source_bytes(source)

    staged = write_staging(plan, plan.request.release_root / ".staging")
    second = write_staging(plan, plan.request.release_root / ".staging")
    target = staged.root / "videos/observation.images.main/chunk-000/file-000.mp4"
    probed = probe_video(target)

    assert probed.frame_count == 3
    relative_pts = np.asarray(probed.timestamps_ns) - probed.timestamps_ns[0]
    assert np.max(np.abs(relative_pts - episode.time_axis.timestamps_ns)) <= 1_000_000
    assert staged.manifest.video_materialization.method == "transcoded_frame_range"
    assert staged.manifest.video_materialization.source_sha256 == episode.main_video.sha256
    assert staged.manifest.video_materialization.target_sha256 == _sha256(target)
    capture = cv2.VideoCapture(str(target))
    ok, frame = capture.read()
    capture.release()
    assert ok
    assert 40 < float(frame.mean()) < 60
    assert _source_bytes(source) == source_before
    assert _all_artifact_bytes(staged) == _all_artifact_bytes(second)
    roundtrip = StandardLeRobotAdapter().load(staged.root)
    assert roundtrip.identity.asset_id == episode.identity.asset_id
    assert roundtrip.time_axis.timestamps_ns.tolist() == episode.time_axis.timestamps_ns.tolist()
    assert np.array_equal(
        roundtrip.observation.hand_keypoints_3d,
        episode.observation.hand_keypoints_3d,
        equal_nan=True,
    )
    assert roundtrip.semantics == episode.semantics
    assert roundtrip.main_video.source_frame_range == (0, episode.time_axis.frame_count)
    assert roundtrip.main_video.sha256 == _sha256(target)


def test_same_plan_produces_byte_identical_artifacts_without_tx_or_absolute_paths(
    tmp_path: Path,
) -> None:
    request, _report = _write_publish_fixture(tmp_path)
    plan = validate_publish_request(request)

    first = write_staging(plan, request.release_root / ".staging")
    second = write_staging(plan, request.release_root / ".staging")

    assert first.transaction_id != second.transaction_id
    assert _all_artifact_bytes(first) == _all_artifact_bytes(second)
    for relative in ("meta/info.json", "meta/stats.json", "meta/episode_semantics.jsonl", "release_manifest.json"):
        path = first.root / relative
        for line in path.read_text(encoding="utf-8").splitlines():
            _assert_no_absolute_path_strings(json.loads(line))


def test_separate_writer_processes_produce_identical_registered_bytes(
    tmp_path: Path,
) -> None:
    script = """
import hashlib
import json
from pathlib import Path
import sys

from lerobot_v3_publisher import validate_publish_request, write_staging
from tests.test_lerobot_v3_publish_prerequisites import _write_publish_fixture

root = Path(sys.argv[1])
request, _ = _write_publish_fixture(root)
plan = validate_publish_request(request)
staged = write_staging(plan, request.release_root / '.staging')
result = {}
for path in sorted(staged.root.rglob('*')):
    if path.is_file():
        result[path.relative_to(staged.root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
print(json.dumps(result, sort_keys=True))
"""

    outputs = []
    for name in ("first", "second"):
        completed = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path / name)],
            cwd=Path(__file__).parents[1],
            shell=False,
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        outputs.append(json.loads(completed.stdout))

    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("unsafe_kind", ["outside", "symlink"])
def test_staging_base_must_be_exact_safe_release_local_directory(
    tmp_path: Path, unsafe_kind: str
) -> None:
    request, _report = _write_publish_fixture(tmp_path)
    plan = validate_publish_request(request)
    if unsafe_kind == "outside":
        staging = tmp_path / "outside"
    else:
        target = tmp_path / "actual-staging"
        target.mkdir()
        staging = request.release_root / ".staging"
        staging.parent.mkdir(parents=True)
        staging.symlink_to(target, target_is_directory=True)

    with pytest.raises(PublishPrerequisiteError) as raised:
        write_staging(plan, staging)

    assert raised.value.diagnostic.stage == "publish_staging"
    assert raised.value.diagnostic.field == "staging_root"


def test_staging_failure_removes_only_its_private_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _report = _write_publish_fixture(tmp_path)
    plan = validate_publish_request(request)
    staging_base = request.release_root / ".staging"
    existing = staging_base / "tx-existing"
    existing.mkdir(parents=True)
    marker = existing / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    def fail_after_video(*args: object, **kwargs: object) -> None:
        raise RuntimeError("writer failure sentinel")

    monkeypatch.setattr(publisher_writer, "_write_dataset_files", fail_after_video)
    with pytest.raises(RuntimeError, match="writer failure sentinel"):
        write_staging(plan, staging_base)

    assert marker.read_text(encoding="utf-8") == "keep"
    assert sorted(path.name for path in staging_base.iterdir()) == ["tx-existing"]
    assert not plan.release_path.exists()
    assert not plan.current_path.exists()


def test_plan_is_revalidated_again_after_all_staging_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _report = _write_publish_fixture(tmp_path)
    plan = validate_publish_request(request)
    staging_base = request.release_root / ".staging"
    original = publisher_writer.revalidate_publish_plan
    calls = 0

    def fail_final_revalidation(candidate: object):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("final revalidation sentinel")
        return original(candidate)

    monkeypatch.setattr(
        publisher_writer, "revalidate_publish_plan", fail_final_revalidation
    )
    with pytest.raises(RuntimeError, match="final revalidation sentinel"):
        write_staging(plan, staging_base)

    assert calls == 2
    assert list(staging_base.iterdir()) == []
    assert not plan.release_path.exists()
    assert not plan.current_path.exists()


def test_writer_rejects_toolchain_drift_after_plan_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _report = _write_publish_fixture(tmp_path)
    plan = validate_publish_request(request)
    original = publisher_writer.current_toolchain()
    drifted = replace(original, fingerprint="f" * 64)
    monkeypatch.setattr(publisher_writer, "current_toolchain", lambda: drifted)

    with pytest.raises(PublishPrerequisiteError) as raised:
        write_staging(plan, request.release_root / ".staging")

    assert raised.value.diagnostic.field == "publisher_version"
    assert list((request.release_root / ".staging").iterdir()) == []


def test_ffmpeg_timeout_budget_scales_with_duration_and_has_hard_cap() -> None:
    assert publisher_writer._ffmpeg_timeout_seconds(3, 10, 1) == 60
    assert 120 < publisher_writer._ffmpeg_timeout_seconds(36_000, 10, 1) <= 3_600
    assert publisher_writer._ffmpeg_timeout_seconds(10_000_000, 10, 1) == 3_600


def test_target_symlink_injection_is_rejected_without_touching_external_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _report = _write_publish_fixture(tmp_path)
    plan = validate_publish_request(request)
    external = tmp_path / "external.mp4"
    external.write_bytes(b"external-safe")
    original = publisher_writer._materialize_video

    def inject_symlink(candidate: object, root: Path):
        target = root / "videos/observation.images.main/chunk-000/file-000.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(external)
        return original(candidate, root)

    monkeypatch.setattr(publisher_writer, "_materialize_video", inject_symlink)
    with pytest.raises(PublishPrerequisiteError):
        write_staging(plan, request.release_root / ".staging")

    assert external.read_bytes() == b"external-safe"


def test_ffmpeg_output_path_swap_cannot_overwrite_external_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.mp4"
    write_test_video(source, [solid_frame(20 + index * 10) for index in range(6)])
    transaction_root = tmp_path / "transaction"
    transaction_root.mkdir(mode=0o700)
    root_fd = os.open(transaction_root, os.O_RDONLY)
    source_fd = os.open(source, os.O_RDONLY)
    external = tmp_path / "external.mp4"
    external.write_bytes(b"external-safe")
    original = publisher_writer.subprocess.run

    def swap_temp_before_ffmpeg(argv: list[str], *args: object, **kwargs: object):
        if argv and argv[0] == "ffmpeg" and any("trim=start_frame" in item for item in argv):
            temp = next(transaction_root.glob(".video-*.tmp"))
            temp.unlink()
            temp.symlink_to(external)
        return original(argv, *args, **kwargs)

    monkeypatch.setattr(publisher_writer.subprocess, "run", swap_temp_before_ffmpeg)
    try:
        with pytest.raises(PublishPrerequisiteError):
            publisher_writer._transcode_range(
                source_fd,
                publisher_writer._SafeRoot(transaction_root, root_fd),
                "output.mp4",
                1,
                4,
                10,
                1,
            )
    finally:
        os.close(source_fd)
        os.close(root_fd)

    assert external.read_bytes() == b"external-safe"


def test_release_root_symlink_swap_cannot_redirect_staging_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _report = _write_publish_fixture(tmp_path)
    plan = validate_publish_request(request)
    release_root = request.release_root
    moved = tmp_path / "moved-release-root"
    outside = tmp_path / "outside"
    original = publisher_writer._write_dataset_files

    def swap_ancestor(root: Path, *args: object, **kwargs: object) -> None:
        release_root.rename(moved)
        outside.mkdir()
        release_root.symlink_to(outside, target_is_directory=True)
        original(root, *args, **kwargs)

    monkeypatch.setattr(publisher_writer, "_write_dataset_files", swap_ancestor)
    with pytest.raises(PublishPrerequisiteError) as raised:
        write_staging(plan, release_root / ".staging")

    assert raised.value.diagnostic.field == "staging_root"
    assert list(outside.rglob("*")) == []
    assert list((moved / ".staging").iterdir()) == []


def test_payload_subdirectory_swap_cannot_redirect_manifest_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _report = _write_publish_fixture(tmp_path)
    plan = validate_publish_request(request)
    outside = tmp_path / "outside-data"
    outside_payload = outside / "chunk-000/file-000.parquet"
    outside_payload.parent.mkdir(parents=True)
    outside_payload.write_bytes(b"external-safe")
    moved = tmp_path / "moved-data"
    original = publisher_writer._manifest

    def swap_payload_directory(*args: object, **kwargs: object):
        root = args[1]
        data = root.path / "data"
        data.rename(moved)
        data.symlink_to(outside, target_is_directory=True)
        return original(*args, **kwargs)

    monkeypatch.setattr(publisher_writer, "_manifest", swap_payload_directory)
    with pytest.raises(PublishPrerequisiteError):
        write_staging(plan, request.release_root / ".staging")

    assert outside_payload.read_bytes() == b"external-safe"
