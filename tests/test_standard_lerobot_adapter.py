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
        code="schema_missing", field="info.features.supplier.hand_quality.raw_value",
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
