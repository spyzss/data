from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import multiprocessing
import numpy as np

import pyarrow.parquet as pq
import pyarrow as pa
import pytest

from annotation.lerobot_v3_dataset import LeRobotV3Dataset
from lerobot_v3_publisher import (
    PublishPrerequisiteError,
    ValidationReport,
    publish,
    validate_publish_request,
    validate_staged_release,
    write_staging,
)
import lerobot_v3_publisher.publisher as publisher_module
import lerobot_v3_publisher.validation as validation_module
from tests.test_lerobot_v3_publish_prerequisites import _write_publish_fixture
from tests.fixtures import solid_frame, write_test_video


_ORIGINAL_OFFICIAL_READER = validation_module._validate_with_official_reader


def _process_publish(request: object, start: object, queue: object) -> None:
    import lerobot_v3_publisher.validation as child_validation

    child_validation._validate_with_official_reader = lambda root, release_id, expected: (
        "0.6.0-process-double",
        (("lerobot", "0.6.0-process-double"),),
        "a" * 64,
    )

    def restore_readonly(value: object) -> None:
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
        elif is_dataclass(value):
            for field in fields(value):
                restore_readonly(getattr(value, field.name))
        elif isinstance(value, (tuple, list)):
            for item in value:
                restore_readonly(item)

    restore_readonly(request)
    start.wait()
    try:
        result = publish(request)
        queue.put(("ok", result.state, result.plan.release_id))
    except BaseException as exc:
        queue.put(("error", type(exc).__name__, str(exc)))


@pytest.fixture(autouse=True)
def _official_reader_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        validation_module,
        "_validate_with_official_reader",
        lambda root, release_id, expected: (
            "0.6.0-test-double",
            (("lerobot", "0.6.0-test-double"),),
            "f" * 64,
        ),
    )


def _staged(tmp_path: Path):
    request, _report = _write_publish_fixture(tmp_path)
    plan = validate_publish_request(request)
    staged = write_staging(plan, request.release_root / ".staging")
    return request, staged


def _refresh_inventory(staged: object, relative: str) -> None:
    manifest_path = staged.root / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    artifact = staged.root / relative
    payload = artifact.read_bytes()
    for row in manifest["files"]:
        if row["relative_path"] == relative:
            row["size_bytes"] = len(payload)
            row["sha256"] = hashlib.sha256(payload).hexdigest()
            break
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )
    checksums = {
        row["relative_path"]: row["sha256"] for row in manifest["files"]
    }
    checksums["release_manifest.json"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    (staged.root / "checksums.sha256").write_text(
        "".join(f"{digest}  {path}\n" for path, digest in sorted(checksums.items()))
    )


def test_validate_staged_release_independently_reopens_complete_payload(
    tmp_path: Path,
) -> None:
    request, staged = _staged(tmp_path)

    report = validate_staged_release(staged, request.episode)

    assert isinstance(report, ValidationReport)
    assert report.release_id == staged.plan.release_id
    assert report.frame_count == request.episode.time_axis.frame_count
    assert report.file_count == len(staged.manifest.files)
    assert report.manifest_sha256 == staged.manifest_sha256
    assert report.official_reader_version == "0.6.0-test-double"
    assert report.official_reader_fingerprint == "f" * 64
    with pytest.raises(FrozenInstanceError):
        report.frame_count = 100  # type: ignore[misc]


def test_validator_ignores_writer_memory_and_rejects_actual_checksum_drift(
    tmp_path: Path,
) -> None:
    request, staged = _staged(tmp_path)
    data_path = staged.root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(data_path)
    pq.write_table(table.slice(0, 2), data_path)

    with pytest.raises(PublishPrerequisiteError) as raised:
        validate_staged_release(staged, request.episode)

    assert raised.value.diagnostic.code == "validation_failed"
    assert raised.value.diagnostic.stage == "publish_validation"


@pytest.mark.parametrize(
    "kind",
    [
        "row_count",
        "dtype",
        "index",
        "timestamp_ns",
        "float_timestamp",
        "keypoint",
        "validity",
        "task_index",
        "subtask",
    ],
)
def test_validator_rejects_internally_rechecksummed_frame_data_drift(
    tmp_path: Path, kind: str
) -> None:
    request, staged = _staged(tmp_path)
    relative = "data/chunk-000/file-000.parquet"
    path = staged.root / relative
    table = pq.read_table(path)
    if kind == "row_count":
        table = table.slice(0, 2)
    elif kind == "dtype":
        position = table.schema.get_field_index("index")
        table = table.set_column(position, "index", pa.array([0, 1, 2], type=pa.int32()))
    elif kind == "index":
        position = table.schema.get_field_index("index")
        table = table.set_column(position, "index", pa.array([0, 2, 2], type=pa.int64()))
    elif kind == "timestamp_ns":
        position = table.schema.get_field_index("timestamp_ns")
        table = table.set_column(
            position, "timestamp_ns", pa.array([0, 100_000_000, 250_000_000], type=pa.int64())
        )
    elif kind == "float_timestamp":
        position = table.schema.get_field_index("timestamp")
        table = table.set_column(
            position, "timestamp", pa.array([0.0, 0.1, 0.25], type=pa.float64())
        )
    elif kind in {"keypoint", "validity"}:
        name = (
            "observation.hand_keypoints_3d"
            if kind == "keypoint"
            else "observation.hand_joint_valid_3d"
        )
        position = table.schema.get_field_index(name)
        values = table[name].to_pylist()
        if kind == "keypoint":
            values[0][0][0][0] += 1.0
        else:
            values[0][0][0] = not values[0][0][0]
        table = table.set_column(
            position, name, pa.array(values, type=table.schema.field(name).type)
        )
    elif kind == "task_index":
        position = table.schema.get_field_index("task_index")
        table = table.set_column(
            position, "task_index", pa.array([0, 1, 0], type=pa.int64())
        )
    else:
        position = table.schema.get_field_index("subtask_index")
        table = table.set_column(
            position, "subtask_index", pa.array([0, 1, 1], type=pa.int64())
        )
    pq.write_table(table, path)
    _refresh_inventory(staged, relative)

    with pytest.raises(PublishPrerequisiteError) as raised:
        validate_staged_release(staged, request.episode)

    assert raised.value.diagnostic.code == "validation_failed"


@pytest.mark.parametrize("kind", ["subtask", "semantics", "video"])
def test_validator_rejects_rechecksummed_semantic_or_video_drift(
    tmp_path: Path, kind: str
) -> None:
    request, staged = _staged(tmp_path)
    if kind == "subtask":
        relative = "meta/subtask.parquet"
        path = staged.root / relative
        rows = pq.read_table(path).to_pylist()
        rows[0]["description_en"] = "wrong"
        pq.write_table(pa.Table.from_pylist(rows), path)
    elif kind == "semantics":
        relative = "meta/episode_semantics.jsonl"
        path = staged.root / relative
        payload = json.loads(path.read_text())
        payload["task_en"] = "wrong"
        path.write_text(json.dumps(payload) + "\n")
    else:
        relative = "videos/observation.images.main/chunk-000/file-000.mp4"
        path = staged.root / relative
        write_test_video(path, [solid_frame(10), solid_frame(20)], fps=10.0)
    _refresh_inventory(staged, relative)

    with pytest.raises(PublishPrerequisiteError):
        validate_staged_release(staged, request.episode)


@pytest.mark.parametrize(
    ("kind", "expected_field"),
    [
        ("info", "meta/info.json"),
        ("robot_type", "meta/info.json"),
        ("stats", "meta/stats.json"),
        ("tasks", "meta/tasks.parquet"),
        ("episode", "meta/episodes.length"),
        ("episode_extra", "meta/episodes"),
        ("extra_column", "data"),
        ("calibration", "meta/episode_semantics.jsonl"),
    ],
)
def test_validator_rejects_rechecksummed_registered_metadata_drift(
    tmp_path: Path, kind: str, expected_field: str
) -> None:
    request, staged = _staged(tmp_path)
    if kind in {"info", "robot_type"}:
        relative = "meta/info.json"
        if kind == "info":
            (staged.root / relative).write_text("{}\n")
        else:
            payload = json.loads((staged.root / relative).read_text())
            payload["robot_type"] = "wrong"
            (staged.root / relative).write_text(json.dumps(payload) + "\n")
    elif kind == "stats":
        relative = "meta/stats.json"
        (staged.root / relative).write_text("{}\n")
    elif kind == "tasks":
        relative = "meta/tasks.parquet"
        path = staged.root / relative
        rows = pq.read_table(path).to_pylist()
        rows[0]["task_index"] = 1
        pq.write_table(pa.Table.from_pylist(rows), path)
    elif kind in {"episode", "episode_extra"}:
        relative = "meta/episodes/chunk-000/file-000.parquet"
        path = staged.root / relative
        rows = pq.read_table(path).to_pylist()
        if kind == "episode":
            rows[0]["length"] = 2
        else:
            rows[0]["unexpected"] = 1
        pq.write_table(pa.Table.from_pylist(rows), path)
    elif kind == "extra_column":
        relative = "data/chunk-000/file-000.parquet"
        path = staged.root / relative
        table = pq.read_table(path).append_column(
            "unexpected", pa.array([1, 2, 3], type=pa.int64())
        )
        pq.write_table(table, path)
    else:
        relative = "meta/episode_semantics.jsonl"
        path = staged.root / relative
        payload = json.loads(path.read_text())
        payload["calibration"]["pixel_origin"] = "wrong"
        path.write_text(json.dumps(payload) + "\n")
    _refresh_inventory(staged, relative)

    with pytest.raises(PublishPrerequisiteError) as raised:
        validate_staged_release(staged, request.episode)

    assert raised.value.diagnostic.field == expected_field


def test_publish_commits_immutable_release_then_updates_current_atomically(
    tmp_path: Path,
) -> None:
    request, _report = _write_publish_fixture(tmp_path)

    result = publish(request)

    assert result.state == "published"
    assert result.plan.release_path.is_dir()
    current = json.loads(result.plan.current_path.read_text())
    assert current == {
        "schema_version": "curated_lerobot_v3_current.v1",
        "release_id": result.plan.release_id,
        "manifest_sha256": result.manifest.files
        and publisher_module._sha256(result.plan.release_path / "release_manifest.json"),
    }
    assert list((request.release_root / ".staging").iterdir()) == []

    repeated = publish(request)
    assert repeated.state == "already_published"
    assert repeated.plan.release_id == result.plan.release_id
    current_identity = (
        result.plan.current_path.stat().st_ino,
        result.plan.current_path.stat().st_mtime_ns,
    )

    third = publish(request)

    assert third.state == "already_published"
    assert (
        result.plan.current_path.stat().st_ino,
        result.plan.current_path.stat().st_mtime_ns,
    ) == current_identity


def test_annotation_reader_uses_official_task_subtask_fields_and_video_span(
    tmp_path: Path,
) -> None:
    request, staged = _staged(tmp_path)
    dataset = LeRobotV3Dataset(
        staged.root,
        camera_names=["observation.images.main"],
        instruction_config={"instruction_source": "episode_field"},
        frame_indices=[0],
        load_frames=True,
    )

    episode = dataset.get_episode(0)

    assert episode["instruction"] == request.episode.semantics.task_en
    assert episode["frame_metadata"][0]["subtask_id"] == "subtask_001"
    assert episode["frame_metadata"][0]["description_en"] == "pick up the object"
    assert episode["frames"]["observation.images.main"][0].shape == (24, 32, 3)


def test_official_reader_version_or_load_failure_is_a_hard_validation_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, staged = _staged(tmp_path)
    monkeypatch.setattr(
        validation_module, "_validate_with_official_reader", _ORIGINAL_OFFICIAL_READER
    )
    original_run = subprocess.run

    def reader_response(*args: object, **kwargs: object):
        argv = args[0]
        if argv[0] != validation_module.sys.executable:
            return original_run(*args, **kwargs)
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(
                {
                    "versions": {
                        **validation_module._OFFICIAL_READER_VERSIONS,
                        "lerobot": "0.5.0",
                    }
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(
        validation_module.subprocess,
        "run",
        reader_response,
    )

    with pytest.raises(PublishPrerequisiteError) as raised:
        validate_staged_release(staged, request.episode)

    assert raised.value.diagnostic.field == "official_reader"
    assert "official reader environment mismatch" in raised.value.diagnostic.message


def test_frozen_official_lerobot_reader_reopens_first_and_last_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, staged = _staged(tmp_path)
    monkeypatch.setattr(
        validation_module, "_validate_with_official_reader", _ORIGINAL_OFFICIAL_READER
    )

    report = validate_staged_release(staged, request.episode)

    assert report.official_reader_version == "0.6.0"
    assert dict(report.official_reader_versions) == validation_module._OFFICIAL_READER_VERSIONS
    assert len(report.official_reader_fingerprint) == 64


@pytest.mark.parametrize("failure", ["timeout", "signal"])
def test_official_reader_timeout_or_signal_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    request, staged = _staged(tmp_path)
    monkeypatch.setattr(
        validation_module, "_validate_with_official_reader", _ORIGINAL_OFFICIAL_READER
    )
    original_run = subprocess.run

    def fail_reader(*args: object, **kwargs: object):
        argv = args[0]
        if argv[0] != validation_module.sys.executable:
            return original_run(*args, **kwargs)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, 180)
        return subprocess.CompletedProcess(argv, -9, stdout="", stderr="killed")

    monkeypatch.setattr(validation_module.subprocess, "run", fail_reader)

    with pytest.raises(PublishPrerequisiteError) as raised:
        validate_staged_release(staged, request.episode)

    assert raised.value.diagnostic.field == "official_reader"
    if failure == "timeout":
        assert raised.value.diagnostic.retryable is True
        assert "timed out" in raised.value.diagnostic.message
    else:
        assert "signal 9" in raised.value.diagnostic.message


def test_official_reader_rejects_partial_success_payload_and_forces_offline_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, staged = _staged(tmp_path)
    monkeypatch.setattr(
        validation_module, "_validate_with_official_reader", _ORIGINAL_OFFICIAL_READER
    )
    original_run = subprocess.run
    captured_environment: dict[str, str] = {}

    def partial_response(*args: object, **kwargs: object):
        argv = args[0]
        if argv[0] != validation_module.sys.executable:
            return original_run(*args, **kwargs)
        captured_environment.update(kwargs["env"])
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(
                {
                    "versions": validation_module._OFFICIAL_READER_VERSIONS,
                    "length": request.episode.time_axis.frame_count - 1,
                    "boundary_rows": [],
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(validation_module.subprocess, "run", partial_response)

    with pytest.raises(PublishPrerequisiteError) as raised:
        validate_staged_release(staged, request.episode)

    assert raised.value.diagnostic.field == "official_reader"
    assert "does not prove boundary rows" in raised.value.diagnostic.message
    assert captured_environment["HF_HUB_OFFLINE"] == "1"
    assert captured_environment["HF_DATASETS_OFFLINE"] == "1"


def test_official_reader_dependency_constraints_are_frozen() -> None:
    constraints = (
        Path(__file__).parents[1] / "constraints/lerobot-validator-v0.6.0.txt"
    ).read_text()
    for name, version in validation_module._OFFICIAL_READER_PACKAGES.items():
        package = "lerobot[dataset]" if name == "lerobot" else name
        assert f"{package}=={version}" in constraints.splitlines()
    assert "Python 3.12.13" in constraints
    assert "pyav backend" in constraints
    validator_requirements = (
        Path(__file__).parents[1] / "requirements-validator.txt"
    ).read_text().splitlines()
    assert "-c constraints/lerobot-validator-v0.6.0.txt" in validator_requirements


@pytest.mark.parametrize(
    "failure_point",
    ["writer", "validator", "fsync", "rename"],
)
def test_precommit_failure_cleans_only_fresh_staging_and_preserves_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    request, _report = _write_publish_fixture(tmp_path)
    request.release_root.mkdir(parents=True)
    current = request.release_root / "CURRENT.json"
    current.write_bytes(b'{"old":"current"}\n')
    sentinel = RuntimeError(f"{failure_point} sentinel")
    if failure_point == "writer":
        monkeypatch.setattr(publisher_module, "write_staging", lambda *args: (_ for _ in ()).throw(sentinel))
    elif failure_point == "validator":
        monkeypatch.setattr(publisher_module, "validate_staged_release", lambda *args: (_ for _ in ()).throw(sentinel))
    elif failure_point == "fsync":
        monkeypatch.setattr(
            publisher_module,
            "_fsync_tree",
            lambda *args, **kwargs: (_ for _ in ()).throw(sentinel),
        )
    else:
        monkeypatch.setattr(publisher_module, "_rename_noreplace", lambda *args: (_ for _ in ()).throw(sentinel))

    with pytest.raises(RuntimeError, match=f"{failure_point} sentinel"):
        publish(request)

    assert current.read_bytes() == b'{"old":"current"}\n'
    assert list((request.release_root / "releases").iterdir()) == []
    staging = request.release_root / ".staging"
    assert not staging.exists() or list(staging.iterdir()) == []


def test_current_failure_keeps_complete_orphan_and_retry_recovers_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _report = _write_publish_fixture(tmp_path)
    request.release_root.mkdir(parents=True)
    current = request.release_root / "CURRENT.json"
    current.write_bytes(b'{"old":"current"}\n')
    old_release = request.release_root / "releases/old-release"
    old_release.mkdir(parents=True)
    old_marker = old_release / "marker"
    old_marker.write_bytes(b"old-release-safe")
    original = publisher_module._replace_current
    monkeypatch.setattr(
        publisher_module,
        "_replace_current",
        lambda *args: (_ for _ in ()).throw(RuntimeError("CURRENT sentinel")),
    )

    with pytest.raises(RuntimeError, match="CURRENT sentinel"):
        publish(request)

    plan = validate_publish_request(request)
    assert plan.release_path.is_dir()
    orphan_bytes = {
        path.relative_to(plan.release_path): path.read_bytes()
        for path in plan.release_path.rglob("*")
        if path.is_file()
    }
    assert current.read_bytes() == b'{"old":"current"}\n'
    assert old_marker.read_bytes() == b"old-release-safe"

    monkeypatch.setattr(publisher_module, "_replace_current", original)
    recovered = publish(request)

    assert recovered.state == "already_published"
    assert {
        path.relative_to(plan.release_path): path.read_bytes()
        for path in plan.release_path.rglob("*")
        if path.is_file()
    } == orphan_bytes
    assert old_marker.read_bytes() == b"old-release-safe"
    assert json.loads(current.read_text())["release_id"] == plan.release_id


def test_invalid_existing_release_is_commit_conflict_and_current_is_unchanged(
    tmp_path: Path,
) -> None:
    request, _report = _write_publish_fixture(tmp_path)
    plan = validate_publish_request(request)
    plan.release_path.mkdir(parents=True)
    (plan.release_path / "garbage").write_text("conflict")
    current = request.release_root / "CURRENT.json"
    current.write_bytes(b'{"old":"current"}\n')

    with pytest.raises(PublishPrerequisiteError) as raised:
        publish(request)

    assert raised.value.diagnostic.code == "commit_conflict"
    assert current.read_bytes() == b'{"old":"current"}\n'


def test_concurrent_noreplace_winner_is_independently_validated_then_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _report = _write_publish_fixture(tmp_path)

    def concurrent_winner(source: Path, target: Path, *args: object) -> None:
        source.rename(target)
        raise FileExistsError("simulated no-replace loser")

    monkeypatch.setattr(publisher_module, "_rename_noreplace", concurrent_winner)

    result = publish(request)

    assert result.state == "already_published"
    assert result.plan.release_path.is_dir()
    assert json.loads(result.plan.current_path.read_text())["release_id"] == result.plan.release_id


def test_publish_durability_and_revalidation_order_precedes_current_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _report = _write_publish_fixture(tmp_path)
    events: list[str] = []
    originals = {
        "validate": publisher_module.validate_staged_release,
        "revalidate": publisher_module.revalidate_publish_plan,
        "fsync": publisher_module._fsync_tree,
        "rename": publisher_module._rename_noreplace,
        "sync_dirs": publisher_module._sync_commit_directories,
        "current": publisher_module._replace_current,
    }

    def wrap(name: str):
        def invoke(*args: object, **kwargs: object):
            events.append(name)
            return originals[name](*args, **kwargs)

        return invoke

    for name, target in (
        ("validate", "validate_staged_release"),
        ("revalidate", "revalidate_publish_plan"),
        ("fsync", "_fsync_tree"),
        ("rename", "_rename_noreplace"),
        ("sync_dirs", "_sync_commit_directories"),
        ("current", "_replace_current"),
    ):
        monkeypatch.setattr(publisher_module, target, wrap(name))

    publish(request)

    assert events == [
        "validate",
        "revalidate",
        "validate",
        "fsync",
        "validate",
        "revalidate",
        "rename",
        "sync_dirs",
        "revalidate",
        "current",
    ]


def test_fsync_time_payload_drift_is_revalidated_before_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _report = _write_publish_fixture(tmp_path)
    request.release_root.mkdir(parents=True)
    current = request.release_root / "CURRENT.json"
    current.write_bytes(b'{"old":"current"}\n')
    original = publisher_module._fsync_tree

    def mutate_after_fsync(root: Path, *args: object, **kwargs: object) -> None:
        original(root, *args, **kwargs)
        data = root / "data/chunk-000/file-000.parquet"
        data.write_bytes(data.read_bytes() + b"drift")

    monkeypatch.setattr(publisher_module, "_fsync_tree", mutate_after_fsync)

    with pytest.raises(PublishPrerequisiteError):
        publish(request)

    assert current.read_bytes() == b'{"old":"current"}\n'
    assert list((request.release_root / "releases").iterdir()) == []


def test_release_root_swap_after_rename_cannot_claim_publication_or_touch_outside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _report = _write_publish_fixture(tmp_path)
    release_root = request.release_root
    moved = tmp_path / "moved-release-root"
    outside = tmp_path / "outside"
    original = publisher_module._rename_noreplace

    def rename_then_swap(*args: object, **kwargs: object) -> None:
        original(*args, **kwargs)
        release_root.rename(moved)
        outside.mkdir()
        release_root.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(publisher_module, "_rename_noreplace", rename_then_swap)

    with pytest.raises((OSError, PublishPrerequisiteError)):
        publish(request)

    assert list(outside.rglob("*")) == []
    assert (moved / "releases").is_dir()


def test_release_reader_identity_failures_do_not_leak_file_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "release"
    other = tmp_path / "other"
    root.mkdir()
    other.mkdir()
    original_stat = validation_module.os.stat

    def swapped_stat(path: object, *args: object, **kwargs: object):
        if Path(path) == root:
            return original_stat(other, *args, **kwargs)
        return original_stat(path, *args, **kwargs)

    before = len(os.listdir("/dev/fd"))
    monkeypatch.setattr(validation_module.os, "stat", swapped_stat)
    for _ in range(50):
        with pytest.raises(PublishPrerequisiteError):
            validation_module._ReleaseReader(root)
    after = len(os.listdir("/dev/fd"))

    assert after == before


def test_directory_sync_mid_open_failure_does_not_leak_file_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _report = _write_publish_fixture(tmp_path)
    request.release_root.mkdir(parents=True)
    (request.release_root / ".staging").mkdir()
    (request.release_root / "releases").mkdir()
    plan = validate_publish_request(request)
    original_open = publisher_module.os.open

    def fail_second(path: object, *args: object, **kwargs: object):
        if Path(path) == request.release_root / "releases":
            raise OSError("sync open sentinel")
        return original_open(path, *args, **kwargs)

    before = len(os.listdir("/dev/fd"))
    monkeypatch.setattr(publisher_module.os, "open", fail_second)
    with pytest.raises(OSError, match="sync open sentinel"):
        publisher_module._sync_commit_directories(plan)
    after = len(os.listdir("/dev/fd"))

    assert after == before


@pytest.mark.parametrize("same_release", [True, False])
def test_fresh_root_real_processes_serialize_same_or_different_release_ids(
    tmp_path: Path, same_release: bool
) -> None:
    first, _ = _write_publish_fixture(tmp_path / "first")
    shared_root = tmp_path / "shared-curated"
    first = replace(first, release_root=shared_root)
    if same_release:
        second = first
    else:
        second, second_report = _write_publish_fixture(tmp_path / "second")
        second_report["canonical_binding"]["canonical_revision"] = 4
        second_report["semantic_calibration"]["canonical_revision"] = 4
        second.qc_report_path.write_text(json.dumps(second_report))
        second = replace(second, canonical_revision=4, release_root=shared_root)
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    queue = context.Queue()
    processes = [
        context.Process(target=_process_publish, args=(request, start, queue))
        for request in (first, second)
    ]
    for process in processes:
        process.start()
    start.set()
    results = [queue.get(timeout=120) for _ in processes]
    for process in processes:
        process.join(timeout=120)
        assert process.exitcode == 0

    assert all(row[0] == "ok" for row in results), results
    states = sorted(row[1] for row in results)
    if same_release:
        assert states == ["already_published", "published"]
        assert len(list((shared_root / "releases").iterdir())) == 1
    else:
        assert states == ["published", "published"]
        assert len(list((shared_root / "releases").iterdir())) == 2
