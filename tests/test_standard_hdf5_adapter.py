from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from canonical_qc import CanonicalInputError, StandardHdf5Adapter
from canonical_qc.adapters import _standard_hdf5_readers as hdf5_readers
from tests.fixtures import solid_frame, write_standard_hdf5_episode, write_test_video


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_error(
    action: object, *, code: str, field: str
) -> CanonicalInputError:
    assert callable(action)
    with pytest.raises(CanonicalInputError) as raised:
        action()
    assert raised.value.code == code
    assert raised.value.field == field
    assert str(raised.value).startswith(f"{code}: {field}:")
    return raised.value


def test_semantics_scalar_io_error_is_retryable() -> None:
    class FailingScalar:
        shape = ()
        dtype = h5py.string_dtype(encoding="utf-8")

        def asstr(self) -> "FailingScalar":
            return self

        def __getitem__(self, _key: object) -> object:
            raise OSError("temporary HDF5 dataset read failure")

    with pytest.raises(CanonicalInputError) as caught:
        hdf5_readers._json_string(FailingScalar())  # type: ignore[arg-type]

    assert caught.value.code == "source_integrity_error"
    assert caught.value.field == "/semantics/annotation_json"
    assert caught.value.retryable is True


@pytest.mark.parametrize("source_kind", ["directory", "hdf5"])
def test_load_reads_real_standard_episode_without_mutating_sources(
    tmp_path: Path,
    source_kind: str,
) -> None:
    episode_dir = tmp_path / "asset-001"
    hdf5_path, video_path = write_standard_hdf5_episode(
        episode_dir, hand_quality="provided"
    )
    before = (_sha256(hdf5_path), _sha256(video_path))

    adapter = StandardHdf5Adapter()
    inspection = adapter.inspect(
        episode_dir if source_kind == "directory" else hdf5_path
    )
    episode = adapter.load(
        episode_dir if source_kind == "directory" else hdf5_path
    )

    assert inspection.adapter_id == "standard_hdf5"
    assert inspection.adapter_version == "1.0.0"
    assert inspection.source_format == "hdf5"
    assert inspection.source_schema_version == "egodata_hdf5_qc_input.v1"
    assert inspection.asset_id == "asset-001"
    assert inspection.source_root == episode_dir.resolve()
    assert inspection.hdf5_path == hdf5_path.resolve()
    assert inspection.main_video_path == video_path.resolve()
    assert episode.identity.asset_id == "asset-001"
    assert episode.identity.batch_id == "batch-001"
    assert episode.time_axis.timestamps_ns.tolist() == [0, 100_000_000, 200_000_000]
    assert episode.main_video.path == "main.mp4"
    assert episode.main_video.sha256 == before[1]
    assert [item.relative_path for item in episode.provenance.source_files] == [
        "asset-001.h5",
        "main.mp4",
    ]
    assert [item.sha256 for item in episode.provenance.source_files] == list(before)
    assert [item.size_bytes for item in episode.provenance.source_files] == [
        hdf5_path.stat().st_size,
        video_path.stat().st_size,
    ]
    quality = episode.supplier_evidence.hand_quality
    assert quality is not None and quality.provided is True
    assert quality.raw_value is not None and quality.raw_value.dtype == np.int16
    assert quality.normalized_score is not None
    assert quality.status is not None
    assert quality.status.tolist() == [
        ["unknown", "bad"],
        ["warning", "good"],
        ["good", "unknown"],
    ]
    assert quality.mapping_version == "supplier-001.hand-quality.v1"
    assert (_sha256(hdf5_path), _sha256(video_path)) == before


def test_load_keeps_missing_and_explicitly_unprovided_quality_optional(
    tmp_path: Path,
) -> None:
    missing_dir = tmp_path / "asset-missing"
    false_dir = tmp_path / "asset-false"
    write_standard_hdf5_episode(
        missing_dir, asset_id="asset-missing", hand_quality="missing"
    )
    write_standard_hdf5_episode(
        false_dir, asset_id="asset-false", hand_quality="false"
    )

    missing = StandardHdf5Adapter().load(missing_dir)
    unprovided = StandardHdf5Adapter().load(false_dir)

    assert missing.supplier_evidence.hand_quality is None
    quality = unprovided.supplier_evidence.hand_quality
    assert quality is not None and quality.provided is False
    assert quality.raw_value is None
    assert quality.normalized_score is None
    assert quality.mapping_version is None
    assert quality.status is not None
    assert np.all(quality.status == "unknown")


def test_provided_quality_fixture_covers_the_full_episode(tmp_path: Path) -> None:
    episode_dir = tmp_path / "asset-001"
    write_standard_hdf5_episode(
        episode_dir,
        frame_count=4,
        hand_quality="provided",
    )

    episode = StandardHdf5Adapter().load(episode_dir)

    quality = episode.supplier_evidence.hand_quality
    assert quality is not None and quality.status is not None
    assert quality.status.shape == (4, 2)


def test_directory_resolution_requires_one_exact_asset_hdf5_and_main_mp4(
    tmp_path: Path,
) -> None:
    episode_dir = tmp_path / "asset-001"
    hdf5_path, video_path = write_standard_hdf5_episode(episode_dir)
    (episode_dir / "other.h5").write_bytes(hdf5_path.read_bytes())

    _assert_error(
        lambda: StandardHdf5Adapter().inspect(episode_dir),
        code="source_integrity_error",
        field="source",
    )

    (episode_dir / "other.h5").unlink()
    video_path.rename(episode_dir / "video.mp4")
    _assert_error(
        lambda: StandardHdf5Adapter().inspect(episode_dir),
        code="source_integrity_error",
        field="main_video.path",
    )


def test_directory_resolution_rejects_main_video_symlink_alias(tmp_path: Path) -> None:
    episode_dir = tmp_path / "asset-001"
    _, video_path = write_standard_hdf5_episode(episode_dir)
    alias_target = episode_dir / "other.mp4"
    video_path.rename(alias_target)
    video_path.symlink_to(alias_target.name)

    _assert_error(
        lambda: StandardHdf5Adapter().inspect(episode_dir),
        code="source_integrity_error",
        field="main_video.path",
    )


def test_inspect_rejects_a_symlink_episode_directory_source(tmp_path: Path) -> None:
    real_episode = tmp_path / "real" / "asset-001"
    write_standard_hdf5_episode(real_episode)
    linked_episode = tmp_path / "asset-001"
    linked_episode.symlink_to(real_episode, target_is_directory=True)

    _assert_error(
        lambda: StandardHdf5Adapter().inspect(linked_episode),
        code="source_integrity_error",
        field="source",
    )


def test_inspect_rejects_an_explicit_hdf5_beneath_a_symlink_parent(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real-parent"
    real_episode = real_parent / "asset-001"
    write_standard_hdf5_episode(real_episode)
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    explicit_hdf5 = linked_parent / "asset-001" / "asset-001.h5"
    assert explicit_hdf5.is_symlink() is False

    _assert_error(
        lambda: StandardHdf5Adapter().inspect(explicit_hdf5),
        code="source_integrity_error",
        field="source",
    )


@pytest.mark.parametrize(
    ("mutation", "code", "field"),
    [
        ("missing_dataset", "schema_missing", "/time/timestamps_ns"),
        ("wrong_dtype", "field_mapping_error", "/time/timestamps_ns"),
        ("wrong_shape", "field_mapping_error", "/observation/hand_keypoints_2d"),
        ("bad_json", "field_mapping_error", "/semantics/annotation_json"),
        ("missing_semantic", "field_mapping_error", "semantics.task_en"),
        ("bad_status", "field_mapping_error", "/supplier/hand_quality/status"),
        ("missing_mapping", "field_mapping_error", "/supplier/hand_quality@mapping_version"),
        ("payload_when_false", "field_mapping_error", "/supplier/hand_quality"),
    ],
)
def test_load_rejects_schema_and_mapping_defects_without_repair(
    tmp_path: Path,
    mutation: str,
    code: str,
    field: str,
) -> None:
    episode_dir = tmp_path / "asset-001"
    hdf5_path, _ = write_standard_hdf5_episode(
        episode_dir, hand_quality="provided"
    )
    with h5py.File(hdf5_path, "r+") as handle:
        if mutation == "missing_dataset":
            del handle["/time/timestamps_ns"]
            handle.create_dataset(
                "/timestamp",
                data=np.array([0, 100_000_000, 200_000_000], dtype=np.int64),
            )
        elif mutation == "wrong_dtype":
            del handle["/time/timestamps_ns"]
            handle.create_dataset(
                "/time/timestamps_ns",
                data=np.array([0, 0.1, 0.2], dtype=np.float64),
            )
        elif mutation == "wrong_shape":
            del handle["/observation/hand_keypoints_2d"]
            handle.create_dataset(
                "/observation/hand_keypoints_2d",
                data=np.ones((3, 42, 2), dtype=np.float32),
            )
        elif mutation in {"bad_json", "missing_semantic"}:
            dataset = handle["/semantics/annotation_json"]
            if mutation == "bad_json":
                payload = "{not-json"
            else:
                decoded = json.loads(dataset.asstr()[()])
                del decoded["task_en"]
                payload = json.dumps(decoded, ensure_ascii=False)
            del handle["/semantics/annotation_json"]
            handle.create_dataset(
                "/semantics/annotation_json",
                data=payload,
                dtype=h5py.string_dtype("utf-8"),
            )
        elif mutation == "bad_status":
            handle["/supplier/hand_quality/status"][0, 0] = 4
        elif mutation == "missing_mapping":
            del handle["/supplier/hand_quality"].attrs["mapping_version"]
        elif mutation == "payload_when_false":
            handle["/supplier/hand_quality"].attrs["provided"] = False

    _assert_error(
        lambda: StandardHdf5Adapter().load(episode_dir),
        code=code,
        field=field,
    )


@pytest.mark.parametrize(
    ("mutation", "field"),
    [
        ("frame_count", "/@frame_count"),
        ("image_width", "/camera/main@image_width_px"),
        ("status", "/supplier/hand_quality/status"),
    ],
)
def test_load_requires_exact_hdf5_attribute_and_dataset_dtypes(
    tmp_path: Path,
    mutation: str,
    field: str,
) -> None:
    episode_dir = tmp_path / "asset-001"
    hdf5_path, _ = write_standard_hdf5_episode(
        episode_dir, hand_quality="provided"
    )
    with h5py.File(hdf5_path, "r+") as handle:
        if mutation == "frame_count":
            handle.attrs["frame_count"] = np.int32(3)
        elif mutation == "image_width":
            handle["/camera/main"].attrs["image_width_px"] = np.int64(32)
        else:
            status = handle["/supplier/hand_quality/status"][()]
            del handle["/supplier/hand_quality/status"]
            handle.create_dataset(
                "/supplier/hand_quality/status", data=status.astype(np.int16)
            )

    _assert_error(
        lambda: StandardHdf5Adapter().load(episode_dir),
        code="field_mapping_error",
        field=field,
    )


def test_load_rejects_non_scalar_or_non_utf8_annotation_json(tmp_path: Path) -> None:
    episode_dir = tmp_path / "asset-001"
    hdf5_path, _ = write_standard_hdf5_episode(episode_dir)
    with h5py.File(hdf5_path, "r+") as handle:
        del handle["/semantics/annotation_json"]
        handle.create_dataset(
            "/semantics/annotation_json",
            data=np.array([b"{}", b"{}"], dtype="S2"),
        )

    _assert_error(
        lambda: StandardHdf5Adapter().load(episode_dir),
        code="field_mapping_error",
        field="/semantics/annotation_json",
    )


@pytest.mark.parametrize(
    "linked_path",
    ["dataset", "dataset_parent", "camera_group", "quality_group"],
)
def test_load_rejects_hdf5_soft_link_aliases(
    tmp_path: Path,
    linked_path: str,
) -> None:
    episode_dir = tmp_path / "asset-001"
    hdf5_path, _ = write_standard_hdf5_episode(
        episode_dir, hand_quality="provided"
    )
    with h5py.File(hdf5_path, "r+") as handle:
        if linked_path == "dataset":
            handle.create_dataset(
                "/legacy_timestamps_ns",
                data=handle["/time/timestamps_ns"][()],
            )
            del handle["/time/timestamps_ns"]
            handle["/time/timestamps_ns"] = h5py.SoftLink(
                "/legacy_timestamps_ns"
            )
            field = "/time/timestamps_ns"
        elif linked_path == "dataset_parent":
            handle.copy("/time", "/legacy_time")
            del handle["/time"]
            handle["/time"] = h5py.SoftLink("/legacy_time")
            field = "/time/timestamps_ns"
        elif linked_path == "camera_group":
            handle.copy("/camera/main", "/legacy_camera")
            del handle["/camera/main"]
            handle["/camera/main"] = h5py.SoftLink("/legacy_camera")
            field = "/camera/main"
        else:
            handle.copy("/supplier/hand_quality", "/legacy_quality")
            del handle["/supplier/hand_quality"]
            handle["/supplier/hand_quality"] = h5py.SoftLink(
                "/legacy_quality"
            )
            field = "/supplier/hand_quality"

    _assert_error(
        lambda: StandardHdf5Adapter().load(episode_dir),
        code="field_mapping_error",
        field=field,
    )


def test_load_uses_configurable_timestamp_tolerance_and_never_fps_fill(
    tmp_path: Path,
) -> None:
    episode_dir = tmp_path / "asset-001"
    hdf5_path, _ = write_standard_hdf5_episode(episode_dir)
    with h5py.File(hdf5_path, "r+") as handle:
        handle["/time/timestamps_ns"][:] = [0, 100_500_000, 201_000_000]

    StandardHdf5Adapter(max_timestamp_delta_ns=1_000_000).load(episode_dir)
    _assert_error(
        lambda: StandardHdf5Adapter(max_timestamp_delta_ns=499_999).load(
            episode_dir
        ),
        code="timebase_invalid",
        field="main_video.timestamps_ns[1]",
    )


def test_load_rejects_real_video_frame_count_mismatch(tmp_path: Path) -> None:
    episode_dir = tmp_path / "asset-001"
    _, video_path = write_standard_hdf5_episode(episode_dir)
    write_test_video(
        video_path,
        [solid_frame(10), solid_frame(20)],
        fps=10.0,
    )

    _assert_error(
        lambda: StandardHdf5Adapter().load(episode_dir),
        code="timebase_invalid",
        field="main_video.frame_count",
    )


def test_load_maps_episode_validation_failures_to_stable_input_categories(
    tmp_path: Path,
) -> None:
    episode_dir = tmp_path / "asset-001"
    hdf5_path, _ = write_standard_hdf5_episode(episode_dir)
    with h5py.File(hdf5_path, "r+") as handle:
        handle["/observation/hand_joint_valid_3d"][0, 0, 0] = False

    _assert_error(
        lambda: StandardHdf5Adapter().load(episode_dir),
        code="field_mapping_error",
        field="observation.hand_keypoints_3d",
    )
