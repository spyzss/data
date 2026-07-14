from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from human_qc.hdf5_commit import (
    FinalizingRecord,
    Hdf5CommitError,
    PreparedReplacement,
    RecoveryAction,
    assert_only_dataset_changed,
    commit_hdf5_replacement,
    prepare_hdf5_replacement,
    finalizing_record_from_prepared,
    prepared_replacement_from_record,
    recover_hdf5_replacement,
)


DATASET_PATH = "/label/subtask_label"

UPDATED = {
    "id": "asset-1",
    "scene": "kitchen",
    "task": "pick up the cup",
    "fps": 30.0,
    "frame_count": 4,
    "annotations": [
        {
            "start_frame": 0,
            "end_frame": 1,
            "start_time_sec": 0.0,
            "end_time_sec": 1 / 30,
            "subtask_cn": "拿起杯子",
            "subtask_en": "pick up cup",
            "verb": "pick",
            "object": "cup",
            "target": "cup",
            "hand": "right",
            "phase": "approach",
            "evidence_frames": [0, 1],
            "confidence": 0.9,
            "status": "confirmed",
        },
        {
            "start_frame": 2,
            "end_frame": 3,
            "start_time_sec": 2 / 30,
            "end_time_sec": 3 / 30,
            "subtask_cn": "放下杯子",
            "subtask_en": "put down cup",
            "verb": "place",
            "object": "cup",
            "target": "table",
            "hand": "right",
            "phase": "release",
            "evidence_frames": [2, 3],
            "confidence": 0.8,
            "status": "confirmed",
        },
    ],
}

INVALID = {"id": "asset-1", "annotations": []}


def _write_scalar_dataset(
    group: h5py.Group,
    name: str,
    payload: object,
    *,
    dtype: np.dtype | str = "S16384",
    attrs: dict[str, object] | None = None,
) -> h5py.Dataset:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    dataset = group.create_dataset(name, shape=(), dtype=dtype)
    dataset[()] = data
    for key, value in (attrs or {}).items():
        dataset.attrs[key] = value
    return dataset


def write_complex_hdf5(path: Path, payload: object = UPDATED) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["root_attr"] = "preserve-me"
        label = handle.create_group("label")
        label.attrs["group_attr"] = np.asarray([1, 2, 3], dtype=np.int32)
        _write_scalar_dataset(
            label,
            "subtask_label",
            payload,
            attrs={"encoding": "utf-8", "revision": np.int64(3)},
        )
        quality = label.create_dataset(
            "quality_hand", data=np.asarray([[1.0, 0.5], [0.9, 0.8]], dtype=np.float32)
        )
        quality.attrs["units"] = "score"
        data = handle.create_group("data")
        data.create_dataset("frames", data=np.arange(8, dtype=np.int64).reshape(2, 4))
        data.create_dataset("text", data=np.asarray([b"unchanged", b"bytes"]))
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepared_record(prepared: PreparedReplacement) -> FinalizingRecord:
    return finalizing_record_from_prepared(prepared)


def make_record_after_replace_before_report_commit(tmp_path: Path) -> FinalizingRecord:
    source = write_complex_hdf5(tmp_path / "asset.hdf5")
    prepared = prepare_hdf5_replacement(source, DATASET_PATH, UPDATED, "tx-recovery")
    record = _prepared_record(prepared)
    os.replace(prepared.staged_path, source)
    return record


def test_prepare_changes_only_subtask_scalar_dataset(tmp_path: Path) -> None:
    source = write_complex_hdf5(tmp_path / "asset.hdf5")

    prepared = prepare_hdf5_replacement(source, DATASET_PATH, UPDATED, "tx-1")

    assert_only_dataset_changed(source, prepared.staged_path, DATASET_PATH)
    assert source.read_bytes() != prepared.staged_path.read_bytes()
    assert prepared.old_sha256 == _sha256(source)
    assert prepared.new_sha256 == _sha256(prepared.staged_path)
    assert prepared.staged_path.parent == source.parent
    assert not list(tmp_path.glob("*.bak"))


def test_validation_failure_leaves_original_bytes_unchanged(tmp_path: Path) -> None:
    source = write_complex_hdf5(tmp_path / "asset.hdf5")
    before = source.read_bytes()

    with pytest.raises(Hdf5CommitError):
        prepare_hdf5_replacement(source, DATASET_PATH, INVALID, "tx-2")

    assert source.read_bytes() == before
    assert not list(tmp_path.glob(".asset.hdf5.human-qc-*"))


def test_target_dtype_and_attrs_are_preserved(tmp_path: Path) -> None:
    source = write_complex_hdf5(tmp_path / "asset.hdf5")

    prepared = prepare_hdf5_replacement(source, DATASET_PATH, UPDATED, "tx-attrs")

    with h5py.File(source, "r") as before, h5py.File(prepared.staged_path, "r") as after:
        old = before[DATASET_PATH]
        new = after[DATASET_PATH]
        assert new.dtype == old.dtype
        assert new.shape == old.shape == ()
        assert dict(new.attrs) == dict(old.attrs)
        assert before["label/quality_hand"].dtype == after["label/quality_hand"].dtype
        np.testing.assert_array_equal(
            before["label/quality_hand"][()], after["label/quality_hand"][()]
        )


def test_missing_target_can_be_added_without_touching_unrelated_objects(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset-without-subtask.hdf5"
    with h5py.File(source, "w") as handle:
        handle.create_group("other").create_dataset("values", data=[1, 2, 3])

    prepared = prepare_hdf5_replacement(source, DATASET_PATH, UPDATED, "tx-add")

    assert_only_dataset_changed(source, prepared.staged_path, DATASET_PATH)
    with h5py.File(prepared.staged_path, "r") as handle:
        assert handle[DATASET_PATH].shape == ()
        assert handle[DATASET_PATH].dtype.kind in {"O", "S", "U"}


def test_commit_replaces_atomically_and_removes_staged_file(tmp_path: Path) -> None:
    source = write_complex_hdf5(tmp_path / "asset.hdf5")
    before = source.read_bytes()
    prepared = prepare_hdf5_replacement(source, DATASET_PATH, UPDATED, "tx-commit")

    commit_hdf5_replacement(prepared)

    assert source.read_bytes() != before
    assert _sha256(source) == prepared.new_sha256
    assert not prepared.staged_path.exists()
    assert not list(tmp_path.glob("*.bak"))


def test_commit_rejects_staged_hash_drift_without_touching_original(tmp_path: Path) -> None:
    source = write_complex_hdf5(tmp_path / "asset.hdf5")
    before = source.read_bytes()
    prepared = prepare_hdf5_replacement(source, DATASET_PATH, UPDATED, "tx-drift")
    prepared.staged_path.write_bytes(prepared.staged_path.read_bytes() + b"drift")

    with pytest.raises(Hdf5CommitError):
        commit_hdf5_replacement(prepared)

    assert source.read_bytes() == before
    assert not prepared.staged_path.exists()


def test_commit_never_cleans_source_when_staged_path_is_malformed(tmp_path: Path) -> None:
    source = write_complex_hdf5(tmp_path / "asset.hdf5")
    before = source.read_bytes()
    malformed = PreparedReplacement(
        source_path=source,
        staged_path=source,
        old_sha256="0" * 64,
        new_sha256="1" * 64,
        transaction_id="tx-malformed",
    )

    with pytest.raises(Hdf5CommitError):
        commit_hdf5_replacement(malformed)

    assert source.read_bytes() == before


def test_assert_only_dataset_changed_rejects_non_target_content_change(tmp_path: Path) -> None:
    before = write_complex_hdf5(tmp_path / "before.hdf5")
    after = tmp_path / "after.hdf5"
    after.write_bytes(before.read_bytes())
    with h5py.File(after, "r+") as handle:
        handle["data/frames"][0, 0] = 99

    with pytest.raises(Hdf5CommitError, match="data/frames"):
        assert_only_dataset_changed(before, after, DATASET_PATH)


def test_assert_only_dataset_changed_rejects_target_schema_change(tmp_path: Path) -> None:
    before = write_complex_hdf5(tmp_path / "before.hdf5")
    after = tmp_path / "after.hdf5"
    after.write_bytes(before.read_bytes())
    with h5py.File(after, "r+") as handle:
        del handle[DATASET_PATH]
        handle["label"].create_dataset("subtask_label", data=np.bytes_(b"{}"))

    with pytest.raises(Hdf5CommitError, match="dtype|shape|attribute"):
        assert_only_dataset_changed(before, after, DATASET_PATH)


def test_recovery_finishes_when_current_hash_matches_staged_hash(tmp_path: Path) -> None:
    record = make_record_after_replace_before_report_commit(tmp_path)

    assert recover_hdf5_replacement(record) == RecoveryAction.MARK_REPORT_COMPLETED


def test_recovery_requests_replace_when_old_and_valid_staged_are_present(
    tmp_path: Path,
) -> None:
    source = write_complex_hdf5(tmp_path / "asset.hdf5")
    prepared = prepare_hdf5_replacement(source, DATASET_PATH, UPDATED, "tx-retry")

    assert recover_hdf5_replacement(_prepared_record(prepared)) == RecoveryAction.RETRY_REPLACE
    assert source.read_bytes() != prepared.staged_path.read_bytes()


def test_recovery_retry_can_reconstruct_prepared_and_commit_after_restart(
    tmp_path: Path,
) -> None:
    source = write_complex_hdf5(tmp_path / "asset.hdf5")
    prepared = prepare_hdf5_replacement(source, DATASET_PATH, UPDATED, "tx-restart")
    record = _prepared_record(prepared)

    assert recover_hdf5_replacement(record) == RecoveryAction.RETRY_REPLACE
    reconstructed = prepared_replacement_from_record(record)

    commit_hdf5_replacement(reconstructed)

    assert _sha256(source) == record.new_sha256
    assert not record.staged_path.exists()


def test_reconstruction_normalizes_json_list_stage_identity(tmp_path: Path) -> None:
    source = write_complex_hdf5(tmp_path / "asset-list-identity.hdf5")
    prepared = prepare_hdf5_replacement(source, DATASET_PATH, UPDATED, "tx-list-id")
    durable = _prepared_record(prepared)
    persisted = FinalizingRecord(
        source_path=durable.source_path,
        staged_path=durable.staged_path,
        old_sha256=durable.old_sha256,
        new_sha256=durable.new_sha256,
        transaction_id=durable.transaction_id,
        staged_identity=list(durable.staged_identity or ()),
    )

    reconstructed = prepared_replacement_from_record(persisted)
    commit_hdf5_replacement(reconstructed)

    assert _sha256(source) == durable.new_sha256


def test_reconstruction_rejects_foreign_stage_without_touching_it(
    tmp_path: Path,
) -> None:
    source = write_complex_hdf5(tmp_path / "asset.hdf5")
    prepared = prepare_hdf5_replacement(source, DATASET_PATH, UPDATED, "tx-reconstruct")
    foreign = tmp_path / "foreign-reconstruct.hdf5"
    foreign.write_bytes(prepared.staged_path.read_bytes())
    prepared.staged_path.unlink()
    before = foreign.read_bytes()
    record = FinalizingRecord(
        source_path=source,
        staged_path=foreign,
        old_sha256=prepared.old_sha256,
        new_sha256=prepared.new_sha256,
        transaction_id=prepared.transaction_id,
        staged_identity=prepared.staged_identity,
    )

    with pytest.raises(Hdf5CommitError, match="owned|staging|namespace"):
        prepared_replacement_from_record(record)

    assert foreign.read_bytes() == before


def test_relative_source_path_can_reconstruct_and_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source = write_complex_hdf5(Path("asset-relative.hdf5"))
    prepared = prepare_hdf5_replacement(
        source, DATASET_PATH, UPDATED, "tx-relative"
    )
    record = _prepared_record(prepared)

    reconstructed = prepared_replacement_from_record(record)
    commit_hdf5_replacement(reconstructed)

    assert _sha256(source) == prepared.new_sha256


def test_reconstruction_requires_transaction_id(tmp_path: Path) -> None:
    source = write_complex_hdf5(tmp_path / "asset-no-tx.hdf5")
    prepared = prepare_hdf5_replacement(source, DATASET_PATH, UPDATED, "tx-required")
    record = FinalizingRecord(
        source_path=prepared.source_path,
        staged_path=prepared.staged_path,
        old_sha256=prepared.old_sha256,
        new_sha256=prepared.new_sha256,
    )
    before = prepared.staged_path.read_bytes()

    with pytest.raises(Hdf5CommitError, match="transaction_id"):
        prepared_replacement_from_record(record)

    assert prepared.staged_path.read_bytes() == before
    prepared.staged_path.unlink()


def test_spoofed_same_prefix_stage_survives_current_new_recovery(
    tmp_path: Path,
) -> None:
    source = write_complex_hdf5(tmp_path / "asset-spoof-new.hdf5")
    prepared = prepare_hdf5_replacement(source, DATASET_PATH, UPDATED, "tx-spoof")
    record = finalizing_record_from_prepared(prepared)
    os.replace(prepared.staged_path, source)
    spoof = Path(record.staged_path)
    spoof.write_bytes(source.read_bytes())
    before = spoof.read_bytes()

    assert recover_hdf5_replacement(record) == RecoveryAction.CONFLICT
    assert spoof.read_bytes() == before


def test_spoofed_same_prefix_stage_cannot_be_retried_or_committed(
    tmp_path: Path,
) -> None:
    source = write_complex_hdf5(tmp_path / "asset-spoof-old.hdf5")
    prepared = prepare_hdf5_replacement(source, DATASET_PATH, UPDATED, "tx-spoof-old")
    record = finalizing_record_from_prepared(prepared)
    spoof = Path(record.staged_path)
    original = spoof.read_bytes()
    spoof.unlink()
    spoof.write_bytes(original)

    assert recover_hdf5_replacement(record) == RecoveryAction.CONFLICT
    with pytest.raises(Hdf5CommitError):
        commit_hdf5_replacement(prepared)
    assert spoof.read_bytes() == original


def test_recovery_requires_persisted_stage_identity(tmp_path: Path) -> None:
    source = write_complex_hdf5(tmp_path / "asset-identity.hdf5")
    prepared = prepare_hdf5_replacement(source, DATASET_PATH, UPDATED, "tx-identity")
    record = finalizing_record_from_prepared(prepared)
    missing_identity = FinalizingRecord(
        source_path=record.source_path,
        staged_path=record.staged_path,
        old_sha256=record.old_sha256,
        new_sha256=record.new_sha256,
        transaction_id=record.transaction_id,
    )
    assert recover_hdf5_replacement(missing_identity) == RecoveryAction.CONFLICT
    prepared.staged_path.unlink()


def test_prepare_rejects_non_string_transaction_id_with_controlled_error(
    tmp_path: Path,
) -> None:
    source = write_complex_hdf5(tmp_path / "asset-bad-tx.hdf5")

    with pytest.raises(Hdf5CommitError, match="transaction_id"):
        prepare_hdf5_replacement(source, DATASET_PATH, UPDATED, 123)  # type: ignore[arg-type]


def test_recovery_classifies_non_string_transaction_id_without_raw_type_error(
    tmp_path: Path,
) -> None:
    source = write_complex_hdf5(tmp_path / "asset-bad-record-tx.hdf5")
    prepared = prepare_hdf5_replacement(source, DATASET_PATH, UPDATED, "tx-record")
    record = FinalizingRecord(
        source_path=prepared.source_path,
        staged_path=prepared.staged_path,
        old_sha256=prepared.old_sha256,
        new_sha256=prepared.new_sha256,
        transaction_id=123,  # type: ignore[arg-type]
        staged_identity=prepared.staged_identity,
    )

    assert recover_hdf5_replacement(record) == RecoveryAction.CONFLICT
    prepared.staged_path.unlink()


def test_prepare_preserves_target_hidden_behind_hard_link_alias(tmp_path: Path) -> None:
    source = write_complex_hdf5(tmp_path / "asset-hard-alias.hdf5")
    with h5py.File(source, "r+") as handle:
        original = handle[DATASET_PATH]
        handle.require_group("metadata")["canonical_subtask"] = original
        del handle[DATASET_PATH]
        handle[DATASET_PATH] = handle["metadata/canonical_subtask"]

    prepared = prepare_hdf5_replacement(source, DATASET_PATH, UPDATED, "tx-alias")

    assert_only_dataset_changed(source, prepared.staged_path, DATASET_PATH)
    with h5py.File(prepared.staged_path, "r") as handle:
        np.testing.assert_array_equal(
            handle[DATASET_PATH][()], handle["metadata/canonical_subtask"][()]
        )


def test_assert_only_dataset_changed_rejects_hard_link_retarget(tmp_path: Path) -> None:
    before = write_complex_hdf5(tmp_path / "before-hard-link.hdf5")
    with h5py.File(before, "r+") as handle:
        group = handle.require_group("data")
        group.create_dataset("first", data=np.asarray([7], dtype=np.int64))
        group.create_dataset("second", data=np.asarray([7], dtype=np.int64))
        group["alias"] = group["first"]
    after = tmp_path / "after-hard-link.hdf5"
    after.write_bytes(before.read_bytes())
    with h5py.File(after, "r+") as handle:
        del handle["data/alias"]
        handle["data/alias"] = handle["data/second"]

    with pytest.raises(Hdf5CommitError, match="link|path|content"):
        assert_only_dataset_changed(before, after, DATASET_PATH)


def test_assert_only_dataset_changed_rejects_attrs_on_new_target_ancestors(
    tmp_path: Path,
) -> None:
    before = tmp_path / "before-new-target.hdf5"
    with h5py.File(before, "w") as handle:
        handle.create_group("other").create_dataset("values", data=[1, 2, 3])
    after = tmp_path / "after-new-target.hdf5"
    after.write_bytes(before.read_bytes())
    with h5py.File(after, "r+") as handle:
        label = handle.create_group("label")
        label.attrs["unexpected"] = "must-reject"
        target = label.create_dataset("subtask_label", shape=(), dtype="S16384")
        target[()] = json.dumps(UPDATED, ensure_ascii=False).encode("utf-8")

    with pytest.raises(Hdf5CommitError, match="attribute"):
        assert_only_dataset_changed(before, after, DATASET_PATH)


def test_recovery_requests_rebuild_when_old_hash_has_no_staged_file(tmp_path: Path) -> None:
    source = write_complex_hdf5(tmp_path / "asset.hdf5")
    prepared = prepare_hdf5_replacement(source, DATASET_PATH, UPDATED, "tx-rebuild")
    record = _prepared_record(prepared)
    prepared.staged_path.unlink()

    assert recover_hdf5_replacement(record) == RecoveryAction.REBUILD_STAGING


def test_recovery_requests_rebuild_when_managed_staged_hash_is_corrupt(
    tmp_path: Path,
) -> None:
    source = write_complex_hdf5(tmp_path / "asset.hdf5")
    prepared = prepare_hdf5_replacement(source, DATASET_PATH, UPDATED, "tx-corrupt")
    record = _prepared_record(prepared)
    prepared.staged_path.write_bytes(prepared.staged_path.read_bytes() + b"corrupt")

    assert recover_hdf5_replacement(record) == RecoveryAction.REBUILD_STAGING


def test_recovery_does_not_touch_foreign_stage_on_old_hash_conflict(tmp_path: Path) -> None:
    source = write_complex_hdf5(tmp_path / "asset.hdf5")
    prepared = prepare_hdf5_replacement(source, DATASET_PATH, UPDATED, "tx-foreign")
    foreign = tmp_path / "foreign-stage.bin"
    foreign.write_bytes(prepared.staged_path.read_bytes())
    prepared.staged_path.unlink()
    before = foreign.read_bytes()
    record = FinalizingRecord(
        source_path=source,
        staged_path=foreign,
        old_sha256=prepared.old_sha256,
        new_sha256=prepared.new_sha256,
        transaction_id=prepared.transaction_id,
    )

    assert recover_hdf5_replacement(record) == RecoveryAction.CONFLICT
    assert foreign.read_bytes() == before


def test_recovery_does_not_remove_unmanaged_file_when_current_is_new(
    tmp_path: Path,
) -> None:
    source = write_complex_hdf5(tmp_path / "asset.hdf5")
    prepared = prepare_hdf5_replacement(source, DATASET_PATH, UPDATED, "tx-unmanaged")
    record = _prepared_record(prepared)
    os.replace(prepared.staged_path, source)
    foreign = tmp_path / "foreign-stage.bin"
    foreign.write_bytes(source.read_bytes())
    before = foreign.read_bytes()
    record = FinalizingRecord(
        source_path=record.source_path,
        staged_path=foreign,
        old_sha256=record.old_sha256,
        new_sha256=record.new_sha256,
        transaction_id=record.transaction_id,
        staged_identity=record.staged_identity,
    )

    assert recover_hdf5_replacement(record) == RecoveryAction.CONFLICT
    assert foreign.read_bytes() == before


def test_commit_rejects_manually_crafted_same_directory_stage(tmp_path: Path) -> None:
    source = write_complex_hdf5(tmp_path / "asset.hdf5")
    prepared = prepare_hdf5_replacement(source, DATASET_PATH, UPDATED, "tx-owned")
    foreign = tmp_path / "arbitrary-stage.hdf5"
    foreign.write_bytes(prepared.staged_path.read_bytes())
    crafted = PreparedReplacement(
        source_path=source,
        staged_path=foreign,
        old_sha256=prepared.old_sha256,
        new_sha256=prepared.new_sha256,
        transaction_id=prepared.transaction_id,
    )
    before_source = source.read_bytes()
    before_foreign = foreign.read_bytes()

    with pytest.raises(Hdf5CommitError, match="owned|staging"):
        commit_hdf5_replacement(crafted)

    assert source.read_bytes() == before_source
    assert foreign.read_bytes() == before_foreign
    prepared.staged_path.unlink()


def test_assert_only_dataset_changed_rejects_attribute_dtype_drift(tmp_path: Path) -> None:
    before = write_complex_hdf5(tmp_path / "before.hdf5")
    after = tmp_path / "after.hdf5"
    after.write_bytes(before.read_bytes())
    with h5py.File(after, "r+") as handle:
        handle[DATASET_PATH].attrs["revision"] = np.float64(3.0)

    with pytest.raises(Hdf5CommitError, match="attribute"):
        assert_only_dataset_changed(before, after, DATASET_PATH)


def test_assert_only_dataset_changed_rejects_added_soft_link(tmp_path: Path) -> None:
    before = write_complex_hdf5(tmp_path / "before.hdf5")
    after = tmp_path / "after.hdf5"
    after.write_bytes(before.read_bytes())
    with h5py.File(after, "r+") as handle:
        handle["dangling_review_link"] = h5py.SoftLink("/missing/target")

    with pytest.raises(Hdf5CommitError, match="link|path"):
        assert_only_dataset_changed(before, after, DATASET_PATH)


def test_commit_parent_directory_fsync_failure_keeps_published_new_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_complex_hdf5(tmp_path / "asset.hdf5")
    prepared = prepare_hdf5_replacement(source, DATASET_PATH, UPDATED, "tx-dir-fsync")
    expected_new = prepared.staged_path.read_bytes()
    import human_qc.hdf5_commit as module

    monkeypatch.setattr(
        module,
        "_fsync_directory",
        lambda *_: (_ for _ in ()).throw(OSError("directory fsync injected")),
    )

    with pytest.raises(Hdf5CommitError, match="fsync|commit"):
        commit_hdf5_replacement(prepared)

    assert source.read_bytes() == expected_new
    assert not prepared.staged_path.exists()


def test_recovery_reports_conflict_for_unknown_current_hash(tmp_path: Path) -> None:
    source = write_complex_hdf5(tmp_path / "asset.hdf5")
    prepared = prepare_hdf5_replacement(source, DATASET_PATH, UPDATED, "tx-conflict")
    with h5py.File(source, "r+") as handle:
        handle["data/frames"][0, 0] = 999

    before = source.read_bytes()
    assert recover_hdf5_replacement(_prepared_record(prepared)) == RecoveryAction.CONFLICT
    assert source.read_bytes() == before
    assert prepared.staged_path.exists()


@pytest.mark.parametrize(
    "failure_point", ["copy", "write", "h5py_write", "reopen", "diff", "fsync"]
)
def test_prepare_failure_injection_cleans_staged_and_preserves_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_point: str
) -> None:
    source = write_complex_hdf5(tmp_path / "asset.hdf5")
    before = source.read_bytes()

    if failure_point == "copy":
        import human_qc.hdf5_commit as module

        def fail_copy(*args: object, **kwargs: object) -> None:
            raise OSError("copy injected")

        monkeypatch.setattr(module.shutil, "copy2", fail_copy)
    elif failure_point == "write":
        import human_qc.hdf5_commit as module

        original_dump = module.json.dumps

        def fail_dump(*args: object, **kwargs: object) -> str:
            if kwargs.pop("_test_only", False):
                return original_dump(*args, **kwargs)
            raise ValueError("write injected")

        monkeypatch.setattr(module.json, "dumps", fail_dump)
    elif failure_point == "h5py_write":
        import human_qc.hdf5_commit as module

        monkeypatch.setattr(
            module,
            "_assign_scalar_json",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                OSError("h5py write injected")
            ),
        )
    elif failure_point == "reopen":
        import human_qc.hdf5_commit as module

        original_load = module.Hdf5ScalarJsonSubtaskAdapter.load
        calls = 0

        def fail_reopen(*args: object, **kwargs: object):
            nonlocal calls
            calls += 1
            if calls >= 1:
                raise OSError("reopen injected")
            return original_load(*args, **kwargs)

        monkeypatch.setattr(module.Hdf5ScalarJsonSubtaskAdapter, "load", fail_reopen)
    elif failure_point == "diff":
        import human_qc.hdf5_commit as module

        monkeypatch.setattr(
            module,
            "assert_only_dataset_changed",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("diff injected")),
        )
    else:
        import human_qc.hdf5_commit as module

        monkeypatch.setattr(module.os, "fsync", lambda *_: (_ for _ in ()).throw(OSError("fsync injected")))

    with pytest.raises(Hdf5CommitError):
        prepare_hdf5_replacement(source, DATASET_PATH, UPDATED, f"tx-{failure_point}")

    assert source.read_bytes() == before
    assert not list(tmp_path.glob(".asset.hdf5.human-qc-*"))


def test_commit_replace_failure_leaves_original_and_cleans_staged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_complex_hdf5(tmp_path / "asset.hdf5")
    before = source.read_bytes()
    prepared = prepare_hdf5_replacement(source, DATASET_PATH, UPDATED, "tx-replace-fail")
    import human_qc.hdf5_commit as module

    monkeypatch.setattr(module.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("replace injected")))

    with pytest.raises(Hdf5CommitError):
        commit_hdf5_replacement(prepared)

    assert source.read_bytes() == before
    assert not prepared.staged_path.exists()
