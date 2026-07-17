from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

import canonical_qc as cq
from canonical_qc.extensions import lerobot_extension_inventory
from tests.test_canonical_qc_contracts import make_episode
from tests.fixtures import write_standard_hdf5_episode


def test_canonical_data_episode_alias_preserves_legacy_constructor_defaults() -> None:
    assert hasattr(cq, "CanonicalDataEpisode")
    assert cq.CanonicalDataEpisode is cq.CanonicalQcEpisode

    episode = make_episode()

    assert episode.batch_metadata is None
    assert episode.supplier_extensions.fields == ()


def test_batch_manifest_is_canonicalized_hashed_and_identity_bound(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "batch_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "canonical_batch_metadata.v1",
                "batch_id": "batch-001",
                "supplier_id": "supplier-001",
                "dataset_attributes": {
                    "sensors": ["rgb", "force"],
                    "robot_platform": "franka",
                    "languages": ["zh", "en"],
                    "modalities": ["video", "force"],
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    metadata = cq.load_batch_metadata(manifest)
    episode = cq.with_batch_metadata(make_episode(), metadata)

    assert metadata.dataset_attributes == {
        "languages": ["zh", "en"],
        "modalities": ["video", "force"],
        "robot_platform": "franka",
        "sensors": ["rgb", "force"],
    }
    assert len(metadata.content_sha256) == 64
    assert episode.batch_metadata is metadata
    cq.validate_episode(episode)

    wrong = replace(metadata, batch_id="another-batch")
    with pytest.raises(cq.CanonicalInputError) as caught:
        cq.with_batch_metadata(make_episode(), wrong)
    assert caught.value.code == "batch_metadata_identity_mismatch"
    assert caught.value.field == "batch_metadata.batch_id"


def test_supplier_extension_values_are_immutable_and_validated() -> None:
    values = np.arange(6, dtype=np.float32).reshape(3, 2)
    field = cq.SupplierExtensionField(
        published_name="observation.force",
        source_path="/sensors/force",
        values=values,
        time_alignment="frame",
        metadata={"unit": "newton"},
    )
    episode = replace(
        make_episode(),
        supplier_extensions=cq.SupplierExtensions(fields=(field,)),
    )

    values[:] = -1
    assert field.values.flags.writeable is False
    assert field.values.tolist() == [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]
    assert field.metadata == {"unit": "newton"}
    cq.validate_episode(episode)

    with pytest.raises(ValueError):
        field.values[0, 0] = 99
    with pytest.raises(FrozenInstanceError):
        field.published_name = "changed"  # type: ignore[misc]


def test_supplier_extensions_reject_duplicate_names_and_bad_frame_alignment() -> None:
    first = cq.SupplierExtensionField(
        published_name="observation.force",
        source_path="/sensors/force",
        values=np.ones((3, 2), dtype=np.float32),
        time_alignment="frame",
    )
    duplicate = cq.SupplierExtensionField(
        published_name="observation.force",
        source_path="/backup/force",
        values=np.ones((3, 2), dtype=np.float32),
        time_alignment="frame",
    )
    bad_length = cq.SupplierExtensionField(
        published_name="observation.tactile",
        source_path="/sensors/tactile",
        values=np.ones((2, 4), dtype=np.float32),
        time_alignment="frame",
    )
    core_collision = cq.SupplierExtensionField(
        published_name="timestamp",
        source_path="timestamp",
        values=np.arange(3, dtype=np.float64),
        time_alignment="frame",
    )

    for fields, code, field_name in (
        ((first, duplicate), "duplicate_extension_name", "supplier_extensions.fields"),
        ((bad_length,), "invalid_extension_shape", "supplier_extensions.fields[0].values"),
        ((core_collision,), "extension_name_collision", "supplier_extensions.fields[0].published_name"),
    ):
        episode = replace(
            make_episode(),
            supplier_extensions=cq.SupplierExtensions(fields=fields),
        )
        with pytest.raises(cq.CanonicalInputError) as caught:
            cq.validate_episode(episode)
        assert caught.value.code == code
        assert caught.value.field == field_name


def test_data_fingerprint_changes_with_batch_metadata_and_extensions(
    tmp_path: Path,
) -> None:
    base = make_episode()
    extension = cq.SupplierExtensionField(
        published_name="action",
        source_path="action",
        values=np.arange(3, dtype=np.float32),
        time_alignment="frame",
    )
    extended = replace(
        base,
        supplier_extensions=cq.SupplierExtensions(fields=(extension,)),
    )
    manifest = tmp_path / "batch_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "canonical_batch_metadata.v1",
                "batch_id": "batch-001",
                "supplier_id": "supplier-001",
                "dataset_attributes": {"robot_platform": "franka"},
            }
        ),
        encoding="utf-8",
    )
    attributed = cq.with_batch_metadata(base, cq.load_batch_metadata(manifest))

    assert cq.data_fingerprint(base) != cq.data_fingerprint(extended)
    assert cq.data_fingerprint(base) != cq.data_fingerprint(attributed)
    assert cq.semantic_fingerprint(base) == cq.semantic_fingerprint(extended)


def test_path_loader_overlays_explicit_batch_manifest(tmp_path: Path) -> None:
    source_root = (tmp_path / "source").absolute()
    episode_root = source_root / "asset-001"
    write_standard_hdf5_episode(episode_root)
    manifest = (tmp_path / "batch_manifest.json").absolute()
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "canonical_batch_metadata.v1",
                "batch_id": "batch-001",
                "supplier_id": "supplier-001",
                "dataset_attributes": {"robot_platform": "franka"},
            }
        ),
        encoding="utf-8",
    )

    episode = cq.load_canonical_source(
        source=episode_root,
        source_format="hdf5",
        source_root=source_root,
        batch_metadata_path=manifest,
    )

    assert episode.batch_metadata is not None
    assert episode.batch_metadata.dataset_attributes == {"robot_platform": "franka"}


def test_lerobot_inventory_accepts_regular_variable_encoded_lists() -> None:
    table = pa.table(
        {
            "action": pa.array(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                type=pa.list_(pa.float32()),
            )
        }
    )

    extensions = lerobot_extension_inventory(
        table,
        features={"action": {"dtype": "float32", "shape": [2]}},
    )

    assert extensions.fields[0].values.dtype == np.dtype("float32")
    assert extensions.fields[0].values.shape == (3, 2)
