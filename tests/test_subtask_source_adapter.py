from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from human_qc.contracts import BoundaryError
from human_qc.source_adapters import (
    Hdf5ScalarJsonSubtaskAdapter,
    encode_canonical_payload,
)
from human_qc.timeline import SharedBoundaryTimeline


FIXTURE = Path(__file__).parent / "fixtures" / "human_qc" / "subtasks_closed.json"
CANONICAL_FIELDS = {
    "start_frame",
    "end_frame",
    "start_time_sec",
    "end_time_sec",
    "subtask_cn",
    "subtask_en",
    "verb",
    "object",
    "target",
    "hand",
    "phase",
    "evidence_frames",
    "confidence",
    "status",
}
ROOT_FIELDS = {"id", "scene", "task", "fps", "frame_count", "annotations"}


def read_fixture() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def write_scalar_json_hdf5(
    path: Path,
    payload: object,
    dataset_path: str = "/label/subtask_label",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    with h5py.File(path, "w") as handle:
        parent, _, name = dataset_path.rpartition("/")
        group = handle.require_group(parent.strip("/")) if parent else handle
        group.create_dataset(name, data=np.bytes_(serialized))
    return path


def write_json_sidecar(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_hdf5_closed_segments_become_shared_half_open_boundaries(tmp_path: Path) -> None:
    source = write_scalar_json_hdf5(tmp_path / "617856.hdf5", read_fixture())

    loaded = Hdf5ScalarJsonSubtaskAdapter().load(source)

    assert loaded.asset_id == "617856"
    assert loaded.source_kind == "hdf5"
    assert loaded.dataset_path == "/label/subtask_label"
    assert loaded.root_payload["scene"] == "商场"
    assert [
        (segment.start_frame, segment.end_frame_exclusive)
        for segment in loaded.timeline.segments
    ] == [(0, 51), (51, 123), (123, 195)]


def test_sidecar_is_used_only_when_explicit_and_dataset_is_missing(tmp_path: Path) -> None:
    source = tmp_path / "617856.hdf5"
    with h5py.File(source, "w"):
        pass
    sidecar = write_json_sidecar(tmp_path / "subtasks.json", read_fixture())

    loaded = Hdf5ScalarJsonSubtaskAdapter().load(source, sidecar_path=sidecar)

    assert loaded.source_kind == "sidecar"
    assert loaded.asset_id == "617856"

    with pytest.raises(ValueError, match="dataset"):
        Hdf5ScalarJsonSubtaskAdapter().load(source)


def test_hdf5_dataset_takes_precedence_over_explicit_sidecar(tmp_path: Path) -> None:
    source_payload = read_fixture()
    source_payload["scene"] = "source"
    sidecar_payload = read_fixture()
    sidecar_payload["scene"] = "sidecar"
    source = write_scalar_json_hdf5(tmp_path / "617856.hdf5", source_payload)
    sidecar = write_json_sidecar(tmp_path / "subtasks.json", sidecar_payload)

    loaded = Hdf5ScalarJsonSubtaskAdapter().load(source, sidecar_path=sidecar)

    assert loaded.source_kind == "hdf5"
    assert loaded.root_payload["scene"] == "source"


def test_internal_ids_are_stable_and_not_canonical_fields(tmp_path: Path) -> None:
    source = write_scalar_json_hdf5(tmp_path / "617856.hdf5", read_fixture())
    loaded = Hdf5ScalarJsonSubtaskAdapter().load(source)

    expected = hashlib.sha256(b"617856:0:0:50").hexdigest()[:16]
    assert loaded.timeline.segments[0].internal_id == expected
    assert "internal_id" not in loaded.timeline.segments[0].canonical_record


def test_encoder_strips_working_review_helper_fields(tmp_path: Path) -> None:
    payload = read_fixture()
    for index, row in enumerate(payload["annotations"]):
        row.update(
            {
                "notes": [f"review note {index}"],
                "playback_end_frame": row["end_frame"],
                "object_cn": "红枣",
                "source_container": "红枣堆",
                "order": index + 1,
            }
        )
    source = write_scalar_json_hdf5(tmp_path / "617856.hdf5", payload)
    loaded = Hdf5ScalarJsonSubtaskAdapter().load(source)

    encoded = encode_canonical_payload(loaded, loaded.timeline)

    assert set(encoded) == ROOT_FIELDS
    assert all(set(row) == CANONICAL_FIELDS for row in encoded["annotations"])
    assert "notes" not in encoded["annotations"][0]
    assert "playback_end_frame" not in encoded["annotations"][0]
    assert "object_cn" not in encoded["annotations"][0]
    assert "source_container" not in encoded["annotations"][0]
    assert "order" not in encoded["annotations"][0]


def test_encoder_recomputes_closed_frames_and_times_after_boundary_edit(
    tmp_path: Path,
) -> None:
    source = write_scalar_json_hdf5(tmp_path / "617856.hdf5", read_fixture())
    loaded = Hdf5ScalarJsonSubtaskAdapter().load(source)
    moved, _ = loaded.timeline.move_boundary(
        1,
        60,
        actor_segment_id=loaded.timeline.segments[0].internal_id,
        reviewer="alice",
        now=datetime(2026, 7, 14),
    )

    encoded = encode_canonical_payload(loaded, moved)

    assert encoded["fps"] == loaded.root_payload["fps"] == 30.0
    assert encoded["frame_count"] == loaded.root_payload["frame_count"] == 195
    assert [
        (row["start_frame"], row["end_frame"])
        for row in encoded["annotations"]
    ] == [(0, 59), (60, 122), (123, 194)]
    assert encoded["annotations"][0]["start_time_sec"] == pytest.approx(0.0)
    assert encoded["annotations"][0]["end_time_sec"] == pytest.approx(59 / 30.0)
    assert encoded["annotations"][1]["start_time_sec"] == pytest.approx(2.0)
    assert encoded["annotations"][1]["end_time_sec"] == pytest.approx(122 / 30.0)


def test_encoder_rejects_timeline_with_foreign_root_metadata(tmp_path: Path) -> None:
    source = write_scalar_json_hdf5(tmp_path / "617856.hdf5", read_fixture())
    loaded = Hdf5ScalarJsonSubtaskAdapter().load(source)
    foreign_segment = replace(
        loaded.timeline.segments[0],
        start_frame=0,
        end_frame_exclusive=10,
    )
    foreign_timeline = SharedBoundaryTimeline(
        frame_count=10,
        fps=60.0,
        segments=(foreign_segment,),
    )

    with pytest.raises(ValueError, match="metadata"):
        encode_canonical_payload(loaded, foreign_timeline)


def test_loaded_segments_do_not_alias_source_records(tmp_path: Path) -> None:
    payload = read_fixture()
    source = write_scalar_json_hdf5(tmp_path / "617856.hdf5", payload)
    loaded = Hdf5ScalarJsonSubtaskAdapter().load(source)

    loaded.timeline.segments[0].canonical_record["evidence_frames"].append(999)
    loaded.root_payload["scene"] = "mutated working root"
    encoded = encode_canonical_payload(loaded, loaded.timeline)

    assert 999 in loaded.timeline.segments[0].canonical_record["evidence_frames"]
    assert 999 not in payload["annotations"][0]["evidence_frames"]
    assert encoded["annotations"][0]["evidence_frames"][-1] == 999
    assert payload["scene"] == "商场"


@pytest.mark.parametrize(
    "payload, match",
    [
        ([{"not": "an object"}], "JSON root"),
        ({"id": "a"}, "root field"),
        (
            {
                **read_fixture(),
                "annotations": [
                    {key: value for key, value in read_fixture()["annotations"][0].items() if key != "phase"}
                ],
            },
            "annotation",
        ),
    ],
)
def test_invalid_payload_shapes_are_rejected(
    tmp_path: Path, payload: object, match: str
) -> None:
    source = write_scalar_json_hdf5(tmp_path / "bad.hdf5", payload)

    with pytest.raises(ValueError, match=match):
        Hdf5ScalarJsonSubtaskAdapter().load(source)


def test_missing_dataset_and_non_scalar_dataset_are_rejected(tmp_path: Path) -> None:
    missing = tmp_path / "missing.hdf5"
    with h5py.File(missing, "w"):
        pass
    with pytest.raises(ValueError, match="dataset"):
        Hdf5ScalarJsonSubtaskAdapter().load(missing)

    non_scalar = tmp_path / "non-scalar.hdf5"
    with h5py.File(non_scalar, "w") as handle:
        handle.create_dataset(
            "/label/subtask_label",
            data=np.asarray([json.dumps(read_fixture())], dtype="S4096"),
        )
    with pytest.raises(ValueError, match="scalar"):
        Hdf5ScalarJsonSubtaskAdapter().load(non_scalar)


def test_non_object_json_and_invalid_sidecar_are_rejected(tmp_path: Path) -> None:
    source = write_scalar_json_hdf5(tmp_path / "list.hdf5", [1, 2, 3])
    with pytest.raises(ValueError, match="JSON root"):
        Hdf5ScalarJsonSubtaskAdapter().load(source)

    missing_dataset = tmp_path / "missing.hdf5"
    with h5py.File(missing_dataset, "w"):
        pass
    invalid_sidecar = tmp_path / "bad-sidecar.json"
    invalid_sidecar.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar"):
        Hdf5ScalarJsonSubtaskAdapter().load(
            missing_dataset, sidecar_path=invalid_sidecar
        )


@pytest.mark.parametrize(
    "annotation_mutator, match",
    [
        (lambda rows: rows.__setitem__(1, {**rows[1], "start_frame": 52}), "continuous"),
        (lambda rows: rows.__setitem__(1, {**rows[1], "start_frame": 50}), "continuous"),
    ],
)
def test_gaps_and_overlaps_are_rejected(
    tmp_path: Path, annotation_mutator, match: str
) -> None:
    payload = read_fixture()
    annotation_mutator(payload["annotations"])
    source = write_scalar_json_hdf5(tmp_path / "invalid.hdf5", payload)

    with pytest.raises(BoundaryError, match=match):
        Hdf5ScalarJsonSubtaskAdapter().load(source)


def test_frame_count_mismatch_is_rejected(tmp_path: Path) -> None:
    payload = read_fixture()
    payload["frame_count"] = 196
    source = write_scalar_json_hdf5(tmp_path / "invalid.hdf5", payload)

    with pytest.raises(BoundaryError, match="last segment"):
        Hdf5ScalarJsonSubtaskAdapter().load(source)


def test_round_trip_encoded_payload_can_be_loaded_again(tmp_path: Path) -> None:
    source = write_scalar_json_hdf5(tmp_path / "617856.hdf5", read_fixture())
    loaded = Hdf5ScalarJsonSubtaskAdapter().load(source)
    encoded = encode_canonical_payload(loaded, loaded.timeline)
    round_trip_source = write_scalar_json_hdf5(tmp_path / "round-trip.hdf5", encoded)

    round_trip = Hdf5ScalarJsonSubtaskAdapter().load(round_trip_source)

    assert round_trip.asset_id == loaded.asset_id
    assert round_trip.root_payload == loaded.root_payload
    assert encode_canonical_payload(round_trip, round_trip.timeline) == encoded
