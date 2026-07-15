from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import numpy as np

from canonical_qc import CanonicalInputError, StandardLeRobotAdapter
from annotation.lerobot_v3_dataset import LeRobotV3Dataset
from tests.fixtures import solid_frame, write_standard_lerobot_dataset, write_test_video


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_error(action: object, *, code: str, field: str) -> CanonicalInputError:
    assert callable(action)
    with pytest.raises(CanonicalInputError) as raised:
        action()
    assert raised.value.code == code
    assert raised.value.field == field
    return raised.value


@pytest.mark.parametrize("layout", ["v3", "v2.1"])
def test_load_reads_registered_lerobot_layout_without_mutation(
    tmp_path: Path, layout: str
) -> None:
    root = write_standard_lerobot_dataset(tmp_path / layout, layout=layout)
    before = {p.relative_to(root).as_posix(): _sha256(p) for p in root.rglob("*") if p.is_file()}

    episode = StandardLeRobotAdapter().load(root)

    assert episode.identity.asset_id == "asset-001"
    assert episode.identity.source_format == "lerobot"
    assert episode.identity.source_schema_version == "egodata_lerobot_qc_input.v1"
    assert episode.time_axis.timestamps_ns.tolist() == [0, 100_000_000, 200_000_000]
    assert episode.observation.hand_keypoints_3d.shape == (3, 2, 21, 3)
    assert episode.main_video.path.endswith(".mp4")
    assert {item.role for item in episode.provenance.source_files} == {
        "dataset_info", "episode_index", "episode_data", "episode_semantics", "main_video"
    }
    assert {p.relative_to(root).as_posix(): _sha256(p) for p in root.rglob("*") if p.is_file()} == before


def test_selector_is_explicit_for_multiple_episodes_and_never_selects_first(
    tmp_path: Path,
) -> None:
    root = write_standard_lerobot_dataset(
        tmp_path / "dataset", episodes=((3, "asset-three"), (7, "asset-seven"))
    )
    _assert_error(
        lambda: StandardLeRobotAdapter().load(root),
        code="field_mapping_error", field="episode_index",
    )

    selected = StandardLeRobotAdapter().load(root, episode_index=7)
    assert selected.identity.asset_id == "asset-seven"
    _assert_error(
        lambda: StandardLeRobotAdapter().load(root, episode_index=99),
        code="field_mapping_error", field="episode_index",
    )


@pytest.mark.parametrize("column", ["timestamp_ns", "observation.hand_keypoints_3d"])
def test_load_rejects_missing_or_wrong_fixed_parquet_field(
    tmp_path: Path, column: str
) -> None:
    root = write_standard_lerobot_dataset(tmp_path / "dataset")
    path = next((root / "data").rglob("*.parquet"))
    table = pq.read_table(path)
    if column == "timestamp_ns":
        table = table.drop([column])
    else:
        index = table.schema.get_field_index(column)
        wrong = pa.array([[[0.0] * 3] * 42] * table.num_rows)
        table = table.set_column(index, column, wrong)
    pq.write_table(table, path)

    _assert_error(
        lambda: StandardLeRobotAdapter().load(root),
        code="schema_missing" if column == "timestamp_ns" else "field_mapping_error",
        field=column,
    )


def test_float_timestamp_is_cross_checked_and_never_authoritative(tmp_path: Path) -> None:
    root = write_standard_lerobot_dataset(tmp_path / "dataset")
    path = next((root / "data").rglob("*.parquet"))
    table = pq.read_table(path)
    index = table.schema.get_field_index("timestamp")
    table = table.set_column(index, "timestamp", pa.array([0.0, 0.15, 0.2], type=pa.float64()))
    pq.write_table(table, path)

    _assert_error(
        lambda: StandardLeRobotAdapter(max_timestamp_delta_ns=1_000_000).load(root),
        code="timebase_invalid", field="timestamp[1]",
    )


def test_rejects_info_path_template_escape_and_filesystem_symlink(tmp_path: Path) -> None:
    root = write_standard_lerobot_dataset(tmp_path / "dataset")
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["data_path"] = "../outside/{episode_index}.parquet"
    info_path.write_text(json.dumps(info), encoding="utf-8")
    _assert_error(
        lambda: StandardLeRobotAdapter().load(root),
        code="source_integrity_error", field="info.data_path",
    )

    root = write_standard_lerobot_dataset(tmp_path / "symlink-dataset")
    video = next((root / "videos").rglob("*.mp4"))
    target = video.with_name("target.mp4")
    video.rename(target)
    video.symlink_to(target.name)
    _assert_error(
        lambda: StandardLeRobotAdapter().load(root),
        code="source_integrity_error", field="main_video.path",
    )


def test_load_detects_source_mutation_during_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = write_standard_lerobot_dataset(tmp_path / "dataset")
    data_path = next((root / "data").rglob("*.parquet"))
    original = pq.read_table

    def mutating_read(*args: object, **kwargs: object) -> pa.Table:
        table = original(*args, **kwargs)
        if Path(args[0]) == data_path:
            data_path.write_bytes(data_path.read_bytes() + b"mutation")
        return table

    monkeypatch.setattr(pq, "read_table", mutating_read)
    _assert_error(
        lambda: StandardLeRobotAdapter().load(root),
        code="source_integrity_error", field="source",
    )


def test_v3_shared_shards_use_metadata_row_and_video_offsets(tmp_path: Path) -> None:
    root = write_standard_lerobot_dataset(
        tmp_path / "dataset", episodes=((3, "asset-three"), (7, "asset-seven"))
    )
    data_paths = sorted((root / "data").rglob("*.parquet"))
    first, second = (pq.read_table(path) for path in data_paths)
    points_index = second.schema.get_field_index("observation.hand_keypoints_3d")
    second = second.set_column(
        points_index,
        "observation.hand_keypoints_3d",
        pa.array(
            np.full((3, 2, 21, 3), 7.0, dtype=np.float32).tolist(),
            type=second.schema.field("observation.hand_keypoints_3d").type,
        ),
    )
    pq.write_table(pa.concat_tables([first, second]), data_paths[0])
    data_paths[1].unlink()
    episode_path = next((root / "meta" / "episodes").rglob("*.parquet"))
    rows = pq.read_table(episode_path).to_pylist()
    rows[1].update({
        "data/file_index": 0,
        "videos/observation.images.main/file_index": 0,
        "dataset_from_index": 3,
        "dataset_to_index": 6,
        "videos/observation.images.main/from_index": 3,
        "videos/observation.images.main/to_index": 6,
    })
    pq.write_table(pa.Table.from_pylist(rows), episode_path)
    videos = sorted((root / "videos").rglob("*.mp4"))
    write_test_video(videos[0], [solid_frame(20 + index * 10) for index in range(6)], fps=10.0)
    videos[1].unlink()

    episode = StandardLeRobotAdapter().load(root, episode_index=7)

    assert np.all(episode.observation.hand_keypoints_3d == 7.0)
    assert episode.time_axis.timestamps_ns.tolist() == [0, 100_000_000, 200_000_000]


def test_v3_fixture_remains_readable_by_legacy_annotation_reader(tmp_path: Path) -> None:
    root = write_standard_lerobot_dataset(tmp_path / "dataset")
    dataset = LeRobotV3Dataset(
        root,
        camera_names=["observation.images.main"],
        instruction_config={"instruction_source": "none", "default_instruction": "pick"},
        load_frames=False,
    )

    loaded = dataset.get_episode(0)
    assert loaded["episode_index"] == 0
    assert loaded["num_frames"] == 3
    assert loaded["frame_metadata"][2]["timestamp"] == pytest.approx(0.2)


def test_semantic_match_is_unique_and_per_frame_indices_follow_boundaries(tmp_path: Path) -> None:
    root = write_standard_lerobot_dataset(tmp_path / "dataset")
    semantics_path = root / "meta" / "episode_semantics.jsonl"
    semantics_path.write_text(
        semantics_path.read_text(encoding="utf-8") * 2, encoding="utf-8"
    )
    _assert_error(
        lambda: StandardLeRobotAdapter().load(root),
        code="field_mapping_error", field="episode_semantics",
    )

    root = write_standard_lerobot_dataset(tmp_path / "bad-boundary")
    data_path = next((root / "data").rglob("*.parquet"))
    table = pq.read_table(data_path)
    index = table.schema.get_field_index("subtask_index")
    pq.write_table(
        table.set_column(index, "subtask_index", pa.array([0, 1, 0], type=pa.int64())),
        data_path,
    )
    _assert_error(
        lambda: StandardLeRobotAdapter().load(root),
        code="field_mapping_error", field="subtask_index",
    )


def test_provided_supplier_quality_requires_declared_features_and_preserves_state(tmp_path: Path) -> None:
    root = write_standard_lerobot_dataset(tmp_path / "dataset")
    data_path = next((root / "data").rglob("*.parquet"))
    table = pq.read_table(data_path)
    table = table.append_column(
        "supplier.hand_quality.raw_value",
        pa.array([[0, 1], [2, 3], [3, 0]], type=pa.list_(pa.int16(), 2)),
    ).append_column(
        "supplier.hand_quality.normalized_score",
        pa.array([[0.75, 0.75]] * 3, type=pa.list_(pa.float32(), 2)),
    ).append_column(
        "supplier.hand_quality.status",
        pa.array([["unknown", "bad"], ["warning", "good"], ["good", "unknown"]], type=pa.list_(pa.string(), 2)),
    )
    pq.write_table(table, data_path)
    semantics_path = root / "meta" / "episode_semantics.jsonl"
    semantic = json.loads(semantics_path.read_text(encoding="utf-8"))
    semantic["supplier_hand_quality"] = {
        "provided": True, "mapping_version": "supplier-001.hand-quality.v1"
    }
    semantics_path.write_text(json.dumps(semantic) + "\n", encoding="utf-8")

    _assert_error(
        lambda: StandardLeRobotAdapter().load(root),
        code="schema_missing", field="info.features.supplier.hand_quality.status",
    )

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"].update({
        "supplier.hand_quality.raw_value": {"dtype": "int16", "shape": [2]},
        "supplier.hand_quality.normalized_score": {"dtype": "float32", "shape": [2]},
        "supplier.hand_quality.status": {"dtype": "string", "shape": [2]},
    })
    info_path.write_text(json.dumps(info), encoding="utf-8")

    quality = StandardLeRobotAdapter().load(root).supplier_evidence.hand_quality
    assert quality is not None and quality.provided is True
    assert quality.raw_value is not None and quality.raw_value.dtype == np.int16
    assert quality.status is not None and quality.status.tolist()[1] == ["warning", "good"]


def test_fixed_feature_metadata_constants_are_not_inferred(tmp_path: Path) -> None:
    root = write_standard_lerobot_dataset(tmp_path / "dataset")
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"]["observation.hand_keypoints_3d"]["hand_order"] = ["right", "left"]
    info_path.write_text(json.dumps(info), encoding="utf-8")
    _assert_error(
        lambda: StandardLeRobotAdapter().load(root),
        code="field_mapping_error",
        field="info.features.observation.hand_keypoints_3d.hand_order",
    )


def test_v21_episode_metadata_needs_no_file_index_not_used_by_template(tmp_path: Path) -> None:
    root = write_standard_lerobot_dataset(tmp_path / "dataset", layout="v2.1")
    episodes_path = root / "meta" / "episodes.jsonl"
    row = json.loads(episodes_path.read_text(encoding="utf-8"))
    assert "episode_file" not in row
    episodes_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    assert StandardLeRobotAdapter().load(root).identity.asset_id == "asset-001"


def test_selector_rejects_zero_and_duplicate_episode_metadata_rows(tmp_path: Path) -> None:
    root = write_standard_lerobot_dataset(tmp_path / "empty", layout="v2.1")
    (root / "meta" / "episodes.jsonl").write_text("", encoding="utf-8")
    _assert_error(
        lambda: StandardLeRobotAdapter().load(root),
        code="field_mapping_error", field="episode_index",
    )

    root = write_standard_lerobot_dataset(tmp_path / "duplicate", layout="v2.1")
    path = root / "meta" / "episodes.jsonl"
    path.write_text(path.read_text(encoding="utf-8") * 2, encoding="utf-8")
    _assert_error(
        lambda: StandardLeRobotAdapter().load(root, episode_index=0),
        code="field_mapping_error", field="episode_index",
    )


def _rewrite_semantic(root: Path, transform: object) -> None:
    assert callable(transform)
    path = root / "meta" / "episode_semantics.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    transformed = transform(rows)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in transformed),
        encoding="utf-8",
    )


def _add_quality_column(root: Path, name: str, values: pa.Array) -> None:
    path = next((root / "data").rglob("*.parquet"))
    pq.write_table(pq.read_table(path).append_column(name, values), path)


def _declare_feature(root: Path, name: str, declaration: dict[str, object]) -> None:
    path = root / "meta" / "info.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    info["features"][name] = declaration
    path.write_text(json.dumps(info), encoding="utf-8")


def test_provided_quality_allows_independently_optional_raw_and_score(tmp_path: Path) -> None:
    root = write_standard_lerobot_dataset(tmp_path / "status-only")
    _add_quality_column(
        root,
        "supplier.hand_quality.status",
        pa.array([["unknown", "bad"], ["warning", "good"], ["good", "unknown"]], type=pa.list_(pa.string(), 2)),
    )
    _declare_feature(root, "supplier.hand_quality.status", {"dtype": "string", "shape": [2]})
    _rewrite_semantic(
        root,
        lambda rows: [dict(row, supplier_hand_quality={"provided": True, "mapping_version": "supplier.status.v1"}) for row in rows],
    )

    status_only = StandardLeRobotAdapter().load(root).supplier_evidence.hand_quality
    assert status_only is not None and status_only.provided
    assert status_only.raw_value is None
    assert status_only.normalized_score is None

    root = write_standard_lerobot_dataset(tmp_path / "uint32-raw")
    _add_quality_column(
        root,
        "supplier.hand_quality.status",
        pa.array([["unknown", "bad"], ["warning", "good"], ["good", "unknown"]], type=pa.list_(pa.string(), 2)),
    )
    _add_quality_column(
        root,
        "supplier.hand_quality.raw_value",
        pa.array([[1, 2], [3, 4], [5, 6]], type=pa.list_(pa.uint32(), 2)),
    )
    _declare_feature(root, "supplier.hand_quality.status", {"dtype": "string", "shape": [2]})
    _declare_feature(root, "supplier.hand_quality.raw_value", {"dtype": "uint32", "shape": [2]})
    _rewrite_semantic(
        root,
        lambda rows: [dict(row, supplier_hand_quality={"provided": True, "mapping_version": "supplier.raw.v1"}) for row in rows],
    )

    raw = StandardLeRobotAdapter().load(root).supplier_evidence.hand_quality
    assert raw is not None and raw.raw_value is not None
    assert raw.raw_value.dtype == np.uint32


def test_quality_status_is_validated_before_fixed_width_conversion(tmp_path: Path) -> None:
    root = write_standard_lerobot_dataset(tmp_path / "dataset")
    _add_quality_column(
        root,
        "supplier.hand_quality.status",
        pa.array([["unknown", "bad"], ["warning-extra", "good"], ["good", "unknown"]], type=pa.list_(pa.string(), 2)),
    )
    _declare_feature(root, "supplier.hand_quality.status", {"dtype": "string", "shape": [2]})
    _rewrite_semantic(
        root,
        lambda rows: [dict(row, supplier_hand_quality={"provided": True, "mapping_version": "supplier.status.v1"}) for row in rows],
    )
    _assert_error(
        lambda: StandardLeRobotAdapter().load(root),
        code="field_mapping_error", field="supplier.hand_quality.status",
    )


@pytest.mark.parametrize(
    ("declared_dtype", "arrow_type", "numpy_dtype", "values"),
    [
        ("float32", pa.float32(), np.float32, [[0.125, 0.25], [0.375, 0.5], [0.625, 0.75]]),
        ("float64", pa.float64(), np.float64, [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
    ],
)
def test_supplier_raw_float_contract_names_preserve_exact_dtype_and_values(
    tmp_path: Path,
    declared_dtype: str,
    arrow_type: pa.DataType,
    numpy_dtype: object,
    values: list[list[float]],
) -> None:
    root = write_standard_lerobot_dataset(tmp_path / declared_dtype)
    _add_quality_column(
        root,
        "supplier.hand_quality.status",
        pa.array([["unknown", "bad"], ["warning", "good"], ["good", "unknown"]], type=pa.list_(pa.string(), 2)),
    )
    _add_quality_column(
        root,
        "supplier.hand_quality.raw_value",
        pa.array(values, type=pa.list_(arrow_type, 2)),
    )
    _declare_feature(root, "supplier.hand_quality.status", {"dtype": "string", "shape": [2]})
    _declare_feature(root, "supplier.hand_quality.raw_value", {"dtype": declared_dtype, "shape": [2]})
    _rewrite_semantic(
        root,
        lambda rows: [dict(row, supplier_hand_quality={"provided": True, "mapping_version": "supplier.float.v1"}) for row in rows],
    )

    quality = StandardLeRobotAdapter().load(root).supplier_evidence.hand_quality

    assert quality is not None and quality.raw_value is not None
    assert quality.raw_value.dtype == np.dtype(numpy_dtype)
    np.testing.assert_array_equal(
        quality.raw_value,
        np.asarray(values, dtype=numpy_dtype),
    )


@pytest.mark.parametrize(
    ("declared_dtype", "arrow_type", "numpy_dtype", "values"),
    [
        ("int32", pa.int32(), np.int32, [[-2, 3], [4, -5], [6, 7]]),
        ("bool", pa.bool_(), np.bool_, [[True, False], [False, True], [True, True]]),
        ("string", pa.string(), str, [["a", "bb"], ["ccc", "d"], ["e", "ff"]]),
    ],
)
def test_supplier_raw_explicit_primitive_mappings_preserve_values(
    tmp_path: Path,
    declared_dtype: str,
    arrow_type: pa.DataType,
    numpy_dtype: object,
    values: list[list[object]],
) -> None:
    root = write_standard_lerobot_dataset(tmp_path / declared_dtype)
    _add_quality_column(
        root,
        "supplier.hand_quality.status",
        pa.array([["unknown", "bad"], ["warning", "good"], ["good", "unknown"]], type=pa.list_(pa.string(), 2)),
    )
    _add_quality_column(
        root,
        "supplier.hand_quality.raw_value",
        pa.array(values, type=pa.list_(arrow_type, 2)),
    )
    _declare_feature(root, "supplier.hand_quality.status", {"dtype": "string", "shape": [2]})
    _declare_feature(root, "supplier.hand_quality.raw_value", {"dtype": declared_dtype, "shape": [2]})
    _rewrite_semantic(
        root,
        lambda rows: [dict(row, supplier_hand_quality={"provided": True, "mapping_version": "supplier.primitive.v1"}) for row in rows],
    )

    quality = StandardLeRobotAdapter().load(root).supplier_evidence.hand_quality

    assert quality is not None and quality.raw_value is not None
    np.testing.assert_array_equal(
        quality.raw_value,
        np.asarray(values, dtype=numpy_dtype),
    )


def test_supplier_raw_binary_is_rejected_without_lossy_numpy_conversion(
    tmp_path: Path,
) -> None:
    root = write_standard_lerobot_dataset(tmp_path / "binary")
    _add_quality_column(
        root,
        "supplier.hand_quality.status",
        pa.array([["unknown", "bad"], ["warning", "good"], ["good", "unknown"]], type=pa.list_(pa.string(), 2)),
    )
    _add_quality_column(
        root,
        "supplier.hand_quality.raw_value",
        pa.array(
            [[b"a\x00", b"a"], [b"b\x00\x00", b"b"], [b"c\x00", b"c"]],
            type=pa.list_(pa.binary(), 2),
        ),
    )
    _declare_feature(root, "supplier.hand_quality.status", {"dtype": "string", "shape": [2]})
    _declare_feature(root, "supplier.hand_quality.raw_value", {"dtype": "binary", "shape": [2]})
    _rewrite_semantic(
        root,
        lambda rows: [dict(row, supplier_hand_quality={"provided": True, "mapping_version": "supplier.binary.v1"}) for row in rows],
    )

    _assert_error(
        lambda: StandardLeRobotAdapter().load(root),
        code="field_mapping_error",
        field="supplier.hand_quality.raw_value",
    )


def test_video_feature_shape_is_dynamic_and_matches_calibration_and_probe(tmp_path: Path) -> None:
    root = write_standard_lerobot_dataset(tmp_path / "dataset")
    video = next((root / "videos").rglob("*.mp4"))
    write_test_video(
        video,
        [solid_frame(20 + index * 10, width=64, height=48) for index in range(3)],
        fps=10.0,
    )
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"]["observation.images.main"]["shape"] = [48, 64, 3]
    info_path.write_text(json.dumps(info), encoding="utf-8")
    _rewrite_semantic(
        root,
        lambda rows: [
            dict(row, calibration=dict(row["calibration"], image_width_px=64, image_height_px=48))
            for row in rows
        ],
    )

    loaded = StandardLeRobotAdapter().load(root)
    assert (loaded.main_video.width_px, loaded.main_video.height_px) == (64, 48)

    info["features"]["observation.images.main"]["shape"] = [48, 32, 3]
    info_path.write_text(json.dumps(info), encoding="utf-8")
    _assert_error(
        lambda: StandardLeRobotAdapter().load(root),
        code="field_mapping_error", field="info.features.observation.images.main.shape",
    )


def test_mutation_during_inspect_is_rejected_and_not_rebased_into_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = write_standard_lerobot_dataset(tmp_path / "dataset")
    info_path = root / "meta" / "info.json"
    original = Path.read_text
    mutated = False

    def mutating_read(path: Path, *args: object, **kwargs: object) -> str:
        nonlocal mutated
        value = original(path, *args, **kwargs)
        if path == info_path and not mutated:
            mutated = True
            info_path.write_text(value + " ", encoding="utf-8")
        return value

    monkeypatch.setattr(Path, "read_text", mutating_read)
    _assert_error(
        lambda: StandardLeRobotAdapter().load(root),
        code="source_integrity_error", field="source",
    )


def test_episode_metadata_mutation_during_selection_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = write_standard_lerobot_dataset(tmp_path / "dataset")
    episode_path = next((root / "meta" / "episodes").rglob("*.parquet"))
    original = pq.read_table
    mutated = False

    def mutating_read(*args: object, **kwargs: object) -> pa.Table:
        nonlocal mutated
        table = original(*args, **kwargs)
        if Path(args[0]) == episode_path and not mutated:
            mutated = True
            episode_path.write_bytes(episode_path.read_bytes() + b"mutation")
        return table

    monkeypatch.setattr(pq, "read_table", mutating_read)
    _assert_error(
        lambda: StandardLeRobotAdapter().inspect(root),
        code="source_integrity_error", field="source",
    )


def test_load_rejects_episode_shard_added_after_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = write_standard_lerobot_dataset(tmp_path / "dataset")
    existing = next((root / "meta" / "episodes").rglob("*.parquet"))
    added = root / "meta" / "episodes" / "chunk-001" / "file-001.parquet"
    adapter = StandardLeRobotAdapter()
    original_inspect = adapter.inspect

    def inspect_then_add(*args: object, **kwargs: object) -> object:
        inspection = original_inspect(*args, **kwargs)
        added.parent.mkdir(parents=True, exist_ok=True)
        added.write_bytes(existing.read_bytes())
        return inspection

    monkeypatch.setattr(adapter, "inspect", inspect_then_add)
    _assert_error(
        lambda: adapter.load(root),
        code="source_integrity_error", field="source",
    )


def test_load_rejects_episode_shard_added_during_decode_before_final_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = write_standard_lerobot_dataset(tmp_path / "dataset")
    existing = next((root / "meta" / "episodes").rglob("*.parquet"))
    data_path = next((root / "data").rglob("*.parquet"))
    added = root / "meta" / "episodes" / "chunk-001" / "file-001.parquet"
    original = pq.read_table
    mutated = False

    def read_then_add(*args: object, **kwargs: object) -> pa.Table:
        nonlocal mutated
        table = original(*args, **kwargs)
        if Path(args[0]) == data_path and not mutated:
            mutated = True
            added.parent.mkdir(parents=True, exist_ok=True)
            added.write_bytes(existing.read_bytes())
        return table

    monkeypatch.setattr(pq, "read_table", read_then_add)
    _assert_error(
        lambda: StandardLeRobotAdapter().load(root),
        code="source_integrity_error", field="source",
    )


@pytest.mark.parametrize(
    ("key", "value", "field"),
    [
        ("dataset_from_index", -1, "episode.dataset_from_index"),
        ("dataset_to_index", None, "episode.dataset_to_index"),
        ("videos/observation.images.main/from_index", -1, "episode.video_from_index"),
        ("videos/observation.images.main/to_index", None, "episode.video_to_index"),
    ],
)
def test_v3_requires_explicit_nonnegative_exact_data_and_video_offsets(
    tmp_path: Path, key: str, value: int | None, field: str
) -> None:
    root = write_standard_lerobot_dataset(tmp_path / key.replace("/", "_"))
    path = next((root / "meta" / "episodes").rglob("*.parquet"))
    rows = pq.read_table(path).to_pylist()
    if value is None:
        del rows[0][key]
    else:
        rows[0][key] = value
    pq.write_table(pa.Table.from_pylist(rows), path)
    _assert_error(
        lambda: StandardLeRobotAdapter().load(root),
        code="schema_missing" if value is None else "field_mapping_error",
        field=field,
    )


def test_v21_episode_files_must_not_hide_shared_rows_or_frames(tmp_path: Path) -> None:
    root = write_standard_lerobot_dataset(tmp_path / "rows", layout="v2.1")
    data_path = next((root / "data").rglob("*.parquet"))
    table = pq.read_table(data_path)
    pq.write_table(pa.concat_tables([table, table.slice(0, 1)]), data_path)
    _assert_error(
        lambda: StandardLeRobotAdapter().load(root),
        code="field_mapping_error", field="episode.length",
    )

    root = write_standard_lerobot_dataset(tmp_path / "frames", layout="v2.1")
    video_path = next((root / "videos").rglob("*.mp4"))
    write_test_video(
        video_path,
        [solid_frame(20 + index * 10) for index in range(4)],
        fps=10.0,
    )
    _assert_error(
        lambda: StandardLeRobotAdapter().load(root),
        code="timebase_invalid", field="main_video.frame_count",
    )


def test_semantics_keys_are_globally_unique_typed_and_subtask_ids_are_sequential(tmp_path: Path) -> None:
    root = write_standard_lerobot_dataset(
        tmp_path / "duplicate", episodes=((0, "asset-zero"), (1, "asset-one"))
    )
    _rewrite_semantic(root, lambda rows: rows + [dict(rows[1])])
    _assert_error(
        lambda: StandardLeRobotAdapter().load(root, episode_index=0),
        code="field_mapping_error", field="episode_semantics",
    )

    root = write_standard_lerobot_dataset(tmp_path / "bool-index")
    _rewrite_semantic(root, lambda rows: [dict(rows[0], episode_index=True)])
    _assert_error(
        lambda: StandardLeRobotAdapter().load(root),
        code="field_mapping_error", field="episode_semantics[0].episode_index",
    )

    root = write_standard_lerobot_dataset(tmp_path / "subtask")
    data_path = next((root / "data").rglob("*.parquet"))
    table = pq.read_table(data_path)
    index = table.schema.get_field_index("subtask_index")
    pq.write_table(table.set_column(index, "subtask_index", pa.array([4, 4, 4], type=pa.int64())), data_path)
    _rewrite_semantic(
        root,
        lambda rows: [dict(rows[0], subtask_sequence=[dict(rows[0]["subtask_sequence"][0], subtask_index=4)])],
    )
    _assert_error(
        lambda: StandardLeRobotAdapter().load(root),
        code="field_mapping_error", field="semantics.subtask_sequence[0].subtask_index",
    )


@pytest.mark.parametrize(("name", "value"), [("fps_num", 0), ("fps_den", -1), ("fps_num", True)])
def test_public_inspect_rejects_invalid_fps_with_structured_error(
    tmp_path: Path, name: str, value: object
) -> None:
    root = write_standard_lerobot_dataset(tmp_path / f"{name}-{value}")
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info[name] = value
    info_path.write_text(json.dumps(info), encoding="utf-8")
    _assert_error(
        lambda: StandardLeRobotAdapter().inspect(root),
        code="field_mapping_error", field=f"info.{name}",
    )
