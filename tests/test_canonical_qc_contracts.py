from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
from types import SimpleNamespace

import numpy as np
import pytest

from canonical_qc import (
    CameraCalibration,
    CanonicalInputError,
    CanonicalQcEpisode,
    EpisodeIdentity,
    EpisodeSemantics,
    HandObservation,
    SourceFile,
    SourceProvenance,
    Subtask,
    SupplierEvidence,
    SupplierHandQuality,
    TimeAxis,
    VideoStream,
    semantic_fingerprint,
    source_fingerprint,
    validate_episode,
)


def _source_files(*, prefix: str = "data/asset-001") -> tuple[SourceFile, ...]:
    return (
        SourceFile(
            relative_path=f"{prefix}/asset-001.h5",
            role="episode_data",
            size_bytes=1024,
            sha256="1" * 64,
        ),
        SourceFile(
            relative_path=f"{prefix}/main.mp4",
            role="main_video",
            size_bytes=2048,
            sha256="2" * 64,
        ),
    )


def _provenance(
    *,
    source_files: tuple[SourceFile, ...] | None = None,
    source_schema_version: str = "egodata_hdf5_qc_input.v1",
    adapter_id: str = "standard_hdf5",
    adapter_version: str = "1.0.0",
) -> SourceProvenance:
    files = _source_files() if source_files is None else source_files
    return SourceProvenance(
        source_files=files,
        source_fingerprint=source_fingerprint(
            files,
            source_schema_version=source_schema_version,
            adapter_id=adapter_id,
            adapter_version=adapter_version,
            main_video_source_frame_range=(0, 3),
        ),
        adapter_id=adapter_id,
        adapter_version=adapter_version,
    )


def make_episode(*, frame_count: int = 3) -> CanonicalQcEpisode:
    timestamps_ns = np.array([0, 20_000_000, 55_000_000], dtype=np.int64)
    assert frame_count == 3
    keypoints_3d = np.arange(frame_count * 2 * 21 * 3, dtype=np.float32).reshape(
        frame_count, 2, 21, 3
    )

    keypoints_2d = np.arange(frame_count * 2 * 21 * 2, dtype=np.float32).reshape(
        frame_count, 2, 21, 2
    )
    valid = np.ones((frame_count, 2, 21), dtype=np.bool_)
    identity = EpisodeIdentity(
        asset_id="asset-001",
        batch_id="batch-001",
        supplier_id="supplier-001",
        source_format="hdf5",
        source_schema_version="egodata_hdf5_qc_input.v1",
    )
    return CanonicalQcEpisode(
        schema_version="canonical_qc_episode.v1",
        profile="human_ego_hand_pose.v1",
        identity=identity,
        provenance=_provenance(
            source_schema_version=identity.source_schema_version
        ),
        time_axis=TimeAxis(
            frame_count=frame_count,
            timestamps_ns=timestamps_ns,
            fps_num=30,
            fps_den=1,
        ),
        main_video=VideoStream(
            path="data/asset-001/main.mp4",
            sha256="2" * 64,
            frame_count=frame_count,
            width_px=1920,
            height_px=1080,
            fps_num=30,
            fps_den=1,
            codec="h264",
            pixel_format="yuv420p",
        ),
        observation=HandObservation(
            hand_keypoints_3d=keypoints_3d,
            hand_joint_valid_3d=valid,
            hand_keypoints_2d=keypoints_2d,
            hand_joint_valid_2d=valid.copy(),
        ),
        calibration=CameraCalibration(
            intrinsic_matrix=np.array(
                [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            distortion_model="none",
            distortion_coefficients=np.array([], dtype=np.float64),
            image_width_px=1920,
            image_height_px=1080,
        ),
        semantics=EpisodeSemantics(
            scene_id="kitchen",
            task_id="pick-object",
            task_category="manipulation",
            task_cn="拿起物体",
            task_en="pick up object",
            description_cn="拿起物体并放到桌面中央",
            description_en="pick up the object and place it at the center",
            subtask_sequence=(
                Subtask(
                    subtask_id="subtask_001",
                    start_frame=0,
                    end_frame_exclusive=1,
                    description_cn="接近物体",
                    description_en="approach the object",
                ),
                Subtask(
                    subtask_id="subtask_002",
                    start_frame=1,
                    end_frame_exclusive=3,
                    description_cn="拿起物体",
                    description_en="pick up the object",
                ),
            ),
        ),
        supplier_evidence=SupplierEvidence(),
    )


@pytest.mark.parametrize(
    "source_range",
    [(-1, 2), (0, 0), (0, 2), (1, 5), [0, 3]],
)
def test_video_source_range_is_strict_half_open_physical_placement(
    source_range: object,
) -> None:
    episode = make_episode()
    malformed = replace(
        episode,
        main_video=replace(episode.main_video, source_frame_range=source_range),
    )

    with pytest.raises(CanonicalInputError, match="main_video.source_frame_range"):
        validate_episode(malformed)


def test_physical_video_range_participates_in_source_and_semantic_identity() -> None:
    episode = make_episode()
    shifted_range = (5, 8)
    shifted_provenance = replace(
        episode.provenance,
        source_fingerprint=source_fingerprint(
            episode.provenance.source_files,
            source_schema_version=episode.identity.source_schema_version,
            adapter_id=episode.provenance.adapter_id,
            adapter_version=episode.provenance.adapter_version,
            main_video_source_frame_range=shifted_range,
        ),
    )
    shifted = replace(
        episode,
        provenance=shifted_provenance,
        main_video=replace(
            episode.main_video,
            source_frame_range=shifted_range,
        ),
    )

    validate_episode(shifted)
    assert shifted.provenance.source_fingerprint != episode.provenance.source_fingerprint
    assert semantic_fingerprint(shifted) != semantic_fingerprint(episode)

    collision = replace(
        episode,
        main_video=replace(
            episode.main_video,
            source_frame_range=shifted_range,
        ),
    )
    _assert_error(
        collision,
        code="source_fingerprint_mismatch",
        field="provenance.source_fingerprint",
    )

def _assert_error(
    episode: CanonicalQcEpisode, *, code: str, field: str
) -> CanonicalInputError:
    with pytest.raises(CanonicalInputError) as caught:
        validate_episode(episode)
    assert caught.value.code == code
    assert caught.value.field == field
    assert str(caught.value).startswith(f"{code}: {field}:")
    return caught.value


def _duck(value: object) -> SimpleNamespace:
    return SimpleNamespace(
        **{item.name: getattr(value, item.name) for item in fields(value)}
    )


def _subclass_copy(value: object) -> object:
    subclass = type(
        f"{type(value).__name__}Subclass",
        (type(value),),
        {},
    )
    return subclass(
        **{item.name: getattr(value, item.name) for item in fields(value)}
    )


def _episode_with_duck(field: str) -> CanonicalQcEpisode:
    episode = make_episode()
    if field == "episode":
        return _duck(episode)
    if field == "identity":
        return replace(episode, identity=_duck(episode.identity))
    if field == "provenance":
        return replace(episode, provenance=_duck(episode.provenance))
    if field == "provenance.source_files[0]":
        source_files = (
            _duck(episode.provenance.source_files[0]),
            episode.provenance.source_files[1],
        )
        return replace(
            episode,
            provenance=replace(episode.provenance, source_files=source_files),
        )
    if field == "time_axis":
        return replace(episode, time_axis=_duck(episode.time_axis))
    if field == "main_video":
        return replace(episode, main_video=_duck(episode.main_video))
    if field == "observation":
        return replace(episode, observation=_duck(episode.observation))
    if field == "calibration":
        return replace(episode, calibration=_duck(episode.calibration))
    if field == "semantics":
        return replace(episode, semantics=_duck(episode.semantics))
    if field == "semantics.subtask_sequence[0]":
        subtasks = (
            _duck(episode.semantics.subtask_sequence[0]),
            episode.semantics.subtask_sequence[1],
        )
        return replace(
            episode,
            semantics=replace(episode.semantics, subtask_sequence=subtasks),
        )
    if field == "supplier_evidence":
        return replace(
            episode,
            supplier_evidence=_duck(episode.supplier_evidence),
        )
    if field == "supplier_evidence.hand_quality":
        quality = SupplierHandQuality(
            provided=True,
            status=np.full((3, 2), "unknown"),
            mapping_version="supplier-001.hand-quality.v1",
        )
        return replace(
            episode,
            supplier_evidence=SupplierEvidence(hand_quality=_duck(quality)),
        )
    raise AssertionError(f"unknown test field: {field}")


def _episode_with_contract_subclass(field: str) -> CanonicalQcEpisode:
    episode = make_episode()
    if field == "episode":
        return _subclass_copy(episode)
    if field == "identity":
        return replace(episode, identity=_subclass_copy(episode.identity))
    if field == "provenance":
        return replace(
            episode,
            provenance=_subclass_copy(episode.provenance),
        )
    if field == "provenance.source_files[0]":
        source_files = (
            _subclass_copy(episode.provenance.source_files[0]),
            episode.provenance.source_files[1],
        )
        return replace(
            episode,
            provenance=replace(episode.provenance, source_files=source_files),
        )
    if field == "time_axis":
        return replace(episode, time_axis=_subclass_copy(episode.time_axis))
    if field == "main_video":
        return replace(episode, main_video=_subclass_copy(episode.main_video))
    if field == "observation":
        return replace(
            episode,
            observation=_subclass_copy(episode.observation),
        )
    if field == "calibration":
        return replace(
            episode,
            calibration=_subclass_copy(episode.calibration),
        )
    if field == "semantics":
        return replace(
            episode,
            semantics=_subclass_copy(episode.semantics),
        )
    if field == "semantics.subtask_sequence[0]":
        subtasks = (
            _subclass_copy(episode.semantics.subtask_sequence[0]),
            episode.semantics.subtask_sequence[1],
        )
        return replace(
            episode,
            semantics=replace(episode.semantics, subtask_sequence=subtasks),
        )
    if field == "supplier_evidence":
        return replace(
            episode,
            supplier_evidence=_subclass_copy(episode.supplier_evidence),
        )
    if field == "supplier_evidence.hand_quality":
        quality = SupplierHandQuality(
            provided=True,
            status=np.full((3, 2), "unknown"),
            mapping_version="supplier-001.hand-quality.v1",
        )
        return replace(
            episode,
            supplier_evidence=SupplierEvidence(
                hand_quality=_subclass_copy(quality)
            ),
        )
    raise AssertionError(f"unknown test field: {field}")


def test_minimal_episode_is_valid_and_keeps_authoritative_timestamps() -> None:
    episode = make_episode()

    validate_episode(episode)

    # The non-uniform deltas are authoritative; FPS must not rebuild timestamps.
    assert episode.time_axis.timestamps_ns.tolist() == [0, 20_000_000, 55_000_000]
    assert episode.time_axis.frame_index_base == 0
    assert episode.time_axis.interval_semantics == "half_open"


@pytest.mark.parametrize(
    "field",
    [
        "episode",
        "identity",
        "provenance",
        "provenance.source_files[0]",
        "time_axis",
        "main_video",
        "observation",
        "calibration",
        "semantics",
        "semantics.subtask_sequence[0]",
        "supplier_evidence",
        "supplier_evidence.hand_quality",
    ],
)
def test_validator_rejects_every_nested_duck_typed_contract(field: str) -> None:
    episode = _episode_with_duck(field)

    _assert_error(
        episode,
        code="invalid_contract_type",
        field=field,
    )


@pytest.mark.parametrize(
    "field",
    [
        "episode",
        "identity",
        "provenance",
        "provenance.source_files[0]",
        "time_axis",
        "main_video",
        "observation",
        "calibration",
        "semantics",
        "semantics.subtask_sequence[0]",
        "supplier_evidence",
        "supplier_evidence.hand_quality",
    ],
)
def test_validator_rejects_contract_subclasses(field: str) -> None:
    episode = _episode_with_contract_subclass(field)

    _assert_error(
        episode,
        code="invalid_contract_type",
        field=field,
    )


def test_contracts_and_all_arrays_are_deeply_immutable() -> None:
    source = np.zeros((3, 2, 21, 3), dtype=np.float32)
    observation = HandObservation(
        hand_keypoints_3d=source,
        hand_joint_valid_3d=np.ones((3, 2, 21), dtype=np.bool_),
        hand_keypoints_2d=np.zeros((3, 2, 21, 2), dtype=np.float32),
        hand_joint_valid_2d=np.ones((3, 2, 21), dtype=np.bool_),
    )
    source[0, 0, 0, 0] = 99.0

    assert observation.hand_keypoints_3d[0, 0, 0, 0] == 0.0
    assert observation.hand_keypoints_3d.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        observation.hand_keypoints_3d[0, 0, 0, 0] = 1.0
    with pytest.raises(ValueError):
        observation.hand_keypoints_3d.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        observation.hand_order = ("right", "left")
    with pytest.raises((FrozenInstanceError, TypeError)):
        observation.vendor_temporary_field = "not allowed"


def test_object_dtype_raw_evidence_is_rejected_before_it_can_be_mutable() -> None:
    raw_value = np.empty((3, 2), dtype=object)
    raw_value.fill("supplier-value")

    with pytest.raises(CanonicalInputError) as caught:
        SupplierHandQuality(provided=True, raw_value=raw_value)

    assert caught.value.code == "invalid_dtype"
    assert caught.value.field == "supplier_evidence.hand_quality.raw_value"


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        (
            "hand_keypoints_3d",
            np.zeros((3, 2, 20, 3), dtype=np.float32),
            "invalid_shape",
        ),
        (
            "hand_keypoints_2d",
            np.zeros((3, 2, 21, 2), dtype=np.float64),
            "invalid_dtype",
        ),
        (
            "hand_joint_valid_3d",
            np.ones((3, 2, 21), dtype=np.uint8),
            "invalid_dtype",
        ),
    ],
)
def test_observation_shape_and_dtype_are_strict(
    field: str, value: np.ndarray, code: str
) -> None:
    episode = make_episode()
    changed = replace(
        episode,
        observation=replace(episode.observation, **{field: value}),
    )

    _assert_error(changed, code=code, field=f"observation.{field}")


def test_valid_joint_requires_all_finite_coordinates() -> None:
    episode = make_episode()
    points = episode.observation.hand_keypoints_3d.copy()
    points[1, 0, 4, 2] = np.nan
    changed = replace(
        episode,
        observation=replace(episode.observation, hand_keypoints_3d=points),
    )

    _assert_error(
        changed,
        code="invalid_coordinate_validity",
        field="observation.hand_keypoints_3d",
    )


def test_invalid_joint_requires_all_nan_coordinates() -> None:
    episode = make_episode()
    valid = episode.observation.hand_joint_valid_2d.copy()
    valid[1, 1, 4] = False
    changed = replace(
        episode,
        observation=replace(episode.observation, hand_joint_valid_2d=valid),
    )

    _assert_error(
        changed,
        code="invalid_coordinate_validity",
        field="observation.hand_keypoints_2d",
    )


@pytest.mark.parametrize(
    "timestamps",
    [
        np.array([0, 20_000_000, 20_000_000], dtype=np.int64),
        np.array([0, 20_000_000, 10_000_000], dtype=np.int64),
    ],
)
def test_timestamps_must_be_strictly_increasing(timestamps: np.ndarray) -> None:
    episode = make_episode()
    changed = replace(
        episode,
        time_axis=replace(episode.time_axis, timestamps_ns=timestamps),
    )

    _assert_error(
        changed,
        code="timestamps_not_strictly_increasing",
        field="time_axis.timestamps_ns",
    )


def test_timestamp_order_check_cannot_overflow_int64() -> None:
    episode = make_episode()
    timestamps = np.array(
        [np.iinfo(np.int64).max, np.iinfo(np.int64).min, -1],
        dtype=np.int64,
    )
    changed = replace(
        episode,
        time_axis=replace(episode.time_axis, timestamps_ns=timestamps),
    )

    _assert_error(
        changed,
        code="timestamps_not_strictly_increasing",
        field="time_axis.timestamps_ns",
    )


@pytest.mark.parametrize(
    "frame_index_base",
    [False, 0.0, np.bool_(False), np.float32(0.0)],
)
def test_frame_index_base_rejects_equal_zero_non_integral_types(
    frame_index_base: object,
) -> None:
    episode = make_episode()
    changed = replace(
        episode,
        time_axis=replace(
            episode.time_axis,
            frame_index_base=frame_index_base,
        ),
    )

    _assert_error(
        changed,
        code="invalid_integer",
        field="time_axis.frame_index_base",
    )


@pytest.mark.parametrize(
    ("calibration", "code", "field"),
    [
        (
            {"intrinsic_matrix": np.eye(3, dtype=np.float32)},
            "invalid_dtype",
            "calibration.intrinsic_matrix",
        ),
        (
            {"distortion_coefficients": np.zeros(1, dtype=np.float64)},
            "invalid_distortion_coefficients",
            "calibration.distortion_coefficients",
        ),
        (
            {"image_width_px": 1280},
            "calibration_video_mismatch",
            "calibration.image_width_px",
        ),
    ],
)
def test_calibration_contract_is_strict(
    calibration: dict[str, object], code: str, field: str
) -> None:
    episode = make_episode()
    changed = replace(
        episode,
        calibration=replace(episode.calibration, **calibration),
    )

    _assert_error(changed, code=code, field=field)


@pytest.mark.parametrize(
    "subtasks",
    [
        (
            Subtask("one", 1, 2, "一", "one"),
            Subtask("two", 2, 3, "二", "two"),
        ),
        (
            Subtask("one", 0, 1, "一", "one"),
            Subtask("two", 2, 3, "二", "two"),
        ),
        (
            Subtask("one", 0, 2, "一", "one"),
            Subtask("two", 1, 3, "二", "two"),
        ),
        (
            Subtask("one", 0, 1, "一", "one"),
            Subtask("two", 1, 2, "二", "two"),
        ),
        (Subtask("one", 0, 0, "一", "one"),),
    ],
)
def test_subtasks_must_contiguously_cover_half_open_episode(
    subtasks: tuple[Subtask, ...],
) -> None:
    episode = make_episode()
    changed = replace(
        episode,
        semantics=replace(episode.semantics, subtask_sequence=subtasks),
    )

    _assert_error(
        changed,
        code="invalid_subtask_sequence",
        field="semantics.subtask_sequence",
    )


def test_missing_optional_hand_quality_is_valid() -> None:
    episode = make_episode()

    assert episode.supplier_evidence.hand_quality is None
    validate_episode(episode)


def test_provided_false_allows_absent_or_all_unknown_status() -> None:
    episode = make_episode()
    for status in (None, np.full((3, 2), "unknown")):
        changed = replace(
            episode,
            supplier_evidence=SupplierEvidence(
                hand_quality=SupplierHandQuality(provided=False, status=status)
            ),
        )
        validate_episode(changed)


@pytest.mark.parametrize(
    "evidence",
    [
        {"raw_value": np.zeros((3, 2), dtype=np.uint8)},
        {"normalized_score": np.zeros((3, 2), dtype=np.float32)},
        {"mapping_version": "supplier-001.hand-quality.v1"},
    ],
)
def test_provided_false_rejects_actual_hand_quality_evidence(
    evidence: dict[str, object],
) -> None:
    episode = make_episode()
    changed = replace(
        episode,
        supplier_evidence=SupplierEvidence(
            hand_quality=SupplierHandQuality(
                provided=False,
                status=np.full((3, 2), "unknown"),
                **evidence,
            )
        ),
    )

    _assert_error(
        changed,
        code="invalid_hand_quality_state",
        field="supplier_evidence.hand_quality",
    )


@pytest.mark.parametrize(
    ("status", "mapping_version", "field"),
    [
        (None, "supplier-001.hand-quality.v1", "supplier_evidence.hand_quality.status"),
        (np.full((3, 2), "unknown"), None, "supplier_evidence.hand_quality.mapping_version"),
        (np.full((3, 2), "unknown"), "", "supplier_evidence.hand_quality.mapping_version"),
    ],
)
def test_provided_true_requires_status_and_nonempty_mapping_version(
    status: np.ndarray | None,
    mapping_version: str | None,
    field: str,
) -> None:
    episode = make_episode()
    changed = replace(
        episode,
        supplier_evidence=SupplierEvidence(
            hand_quality=SupplierHandQuality(
                provided=True,
                status=status,
                mapping_version=mapping_version,
            )
        ),
    )

    _assert_error(
        changed,
        code="invalid_hand_quality_state",
        field=field,
    )


def test_provided_true_allows_status_and_mapping_without_score_or_raw() -> None:
    episode = make_episode()
    changed = replace(
        episode,
        supplier_evidence=SupplierEvidence(
            hand_quality=SupplierHandQuality(
                provided=True,
                status=np.full((3, 2), "unknown"),
                mapping_version="supplier-001.hand-quality.v1",
            )
        ),
    )

    validate_episode(changed)


def test_hand_quality_validates_only_its_own_optional_contract() -> None:
    episode = make_episode()
    hand_quality = SupplierHandQuality(
        provided=True,
        raw_value=np.array([[0, 1], [1, 1], [0, 0]], dtype=np.uint8),
        normalized_score=np.array(
            [[0.2, 1.0], [0.5, 0.6], [0.0, 0.8]], dtype=np.float32
        ),
        status=np.array(
            [["warning", "good"], ["good", "good"], ["bad", "unknown"]]
        ),
        mapping_version="supplier-001.hand-quality.v1",
    )
    changed = replace(
        episode,
        supplier_evidence=SupplierEvidence(hand_quality=hand_quality),
    )

    validate_episode(changed)

    bad_scores = hand_quality.normalized_score.copy()
    bad_scores[0, 0] = 1.1
    invalid = replace(
        changed,
        supplier_evidence=SupplierEvidence(
            hand_quality=replace(hand_quality, normalized_score=bad_scores)
        ),
    )
    _assert_error(
        invalid,
        code="invalid_hand_quality_score",
        field="supplier_evidence.hand_quality.normalized_score",
    )


def test_source_fingerprint_is_order_independent_and_adapter_sensitive() -> None:
    files = _source_files()
    kwargs = {
        "source_schema_version": "egodata_hdf5_qc_input.v1",
        "adapter_id": "standard_hdf5",
        "adapter_version": "1.0.0",
        "main_video_source_frame_range": (0, 3),
    }

    first = source_fingerprint(files, **kwargs)
    reordered = source_fingerprint(tuple(reversed(files)), **kwargs)
    upgraded = source_fingerprint(files, **{**kwargs, "adapter_version": "1.0.1"})

    assert first == reordered
    assert len(first) == 64
    assert first != upgraded


def test_source_fingerprint_requires_physical_video_range() -> None:
    with pytest.raises(TypeError, match="main_video_source_frame_range"):
        source_fingerprint(
            _source_files(),
            source_schema_version="egodata_hdf5_qc_input.v1",
            adapter_id="standard_hdf5",
            adapter_version="1.0.0",
        )


@pytest.mark.parametrize(
    "source_range",
    [None, [0, 3], (False, 3), (0, 0), (-1, 2), (0, 2, 3)],
)
def test_source_fingerprint_rejects_invalid_physical_video_range(
    source_range: object,
) -> None:
    with pytest.raises(ValueError, match="main_video_source_frame_range"):
        source_fingerprint(
            _source_files(),
            source_schema_version="egodata_hdf5_qc_input.v1",
            adapter_id="standard_hdf5",
            adapter_version="1.0.0",
            main_video_source_frame_range=source_range,
        )


def test_source_fingerprint_normalizes_numpy_integral_size() -> None:
    native = _source_files()
    numpy_sized = (
        replace(native[0], size_bytes=np.int64(native[0].size_bytes)),
        replace(native[1], size_bytes=np.int64(native[1].size_bytes)),
    )
    kwargs = {
        "source_schema_version": "egodata_hdf5_qc_input.v1",
        "adapter_id": "standard_hdf5",
        "adapter_version": "1.0.0",
        "main_video_source_frame_range": (0, 3),
    }

    assert source_fingerprint(native, **kwargs) == source_fingerprint(
        numpy_sized, **kwargs
    )


def test_validator_rejects_source_fingerprint_drift() -> None:
    episode = make_episode()
    changed = replace(
        episode,
        provenance=replace(episode.provenance, source_fingerprint="0" * 64),
    )

    _assert_error(
        changed,
        code="source_fingerprint_mismatch",
        field="provenance.source_fingerprint",
    )


def test_validator_requires_main_video_in_source_fingerprint() -> None:
    episode = make_episode()
    files = (episode.provenance.source_files[0],)
    changed = replace(
        episode,
        provenance=_provenance(
            source_files=files,
            source_schema_version=episode.identity.source_schema_version,
        ),
    )

    _assert_error(
        changed,
        code="missing_main_video_source",
        field="provenance.source_files",
    )


def test_validator_requires_main_video_source_path_and_hash_to_match() -> None:
    episode = make_episode()
    files = (
        episode.provenance.source_files[0],
        replace(episode.provenance.source_files[1], sha256="3" * 64),
    )
    changed = replace(
        episode,
        provenance=_provenance(
            source_files=files,
            source_schema_version=episode.identity.source_schema_version,
        ),
    )

    _assert_error(
        changed,
        code="main_video_source_mismatch",
        field="provenance.source_files",
    )


def test_invalid_utf8_hand_quality_status_has_stable_diagnostic() -> None:
    episode = make_episode()
    status = np.full((3, 2), b"\xff", dtype="S1")
    changed = replace(
        episode,
        supplier_evidence=SupplierEvidence(
            hand_quality=SupplierHandQuality(
                provided=True,
                status=status,
                mapping_version="supplier-001.hand-quality.v1",
            )
        ),
    )

    _assert_error(
        changed,
        code="invalid_hand_quality_status",
        field="supplier_evidence.hand_quality.status",
    )


def test_semantic_fingerprint_ignores_source_format_and_paths() -> None:
    first = make_episode()
    files = _source_files(prefix="different/layout")
    second_identity = replace(
        first.identity,
        source_format="lerobot",
        source_schema_version="egodata_lerobot_qc_input.v1",
    )
    second = replace(
        first,
        identity=second_identity,
        provenance=_provenance(
            source_files=files,
            source_schema_version=second_identity.source_schema_version,
            adapter_id="standard_lerobot",
        ),
        main_video=replace(first.main_video, path="different/layout/main.mp4"),
    )

    validate_episode(second)
    assert semantic_fingerprint(first) == semantic_fingerprint(second)


def test_semantic_fingerprint_changes_with_core_array_value() -> None:
    first = make_episode()
    points = first.observation.hand_keypoints_3d.copy()
    points[0, 0, 0, 0] += np.float32(1.0)
    second = replace(
        first,
        observation=replace(first.observation, hand_keypoints_3d=points),
    )

    assert semantic_fingerprint(first) != semantic_fingerprint(second)


def test_semantic_fingerprint_canonicalizes_invalid_nan_payload_bits() -> None:
    first = make_episode()
    second = make_episode()
    first_points = first.observation.hand_keypoints_3d.copy()
    second_points = second.observation.hand_keypoints_3d.copy()
    first_valid = first.observation.hand_joint_valid_3d.copy()
    second_valid = second.observation.hand_joint_valid_3d.copy()
    first_nan = np.array([0x7FC00000], dtype=np.uint32).view(np.float32)[0]
    second_nan = np.array([0x7FC00001], dtype=np.uint32).view(np.float32)[0]
    first_points[0, 0, 0] = first_nan
    second_points[0, 0, 0] = second_nan
    first_valid[0, 0, 0] = False
    second_valid[0, 0, 0] = False
    first = replace(
        first,
        observation=replace(
            first.observation,
            hand_keypoints_3d=first_points,
            hand_joint_valid_3d=first_valid,
        ),
    )
    second = replace(
        second,
        observation=replace(
            second.observation,
            hand_keypoints_3d=second_points,
            hand_joint_valid_3d=second_valid,
        ),
    )

    assert semantic_fingerprint(first) == semantic_fingerprint(second)
    assert first.observation.hand_keypoints_3d[0, 0, 0, 0].view(np.uint32) == 0x7FC00000
    assert second.observation.hand_keypoints_3d[0, 0, 0, 0].view(np.uint32) == 0x7FC00001


def test_semantic_fingerprint_canonicalizes_signed_zero_without_mutation() -> None:
    first = make_episode()
    second = make_episode()
    first_points = first.observation.hand_keypoints_3d.copy()
    second_points = second.observation.hand_keypoints_3d.copy()
    first_points[0, 0, 0, 0] = np.float32(0.0)
    second_points[0, 0, 0, 0] = np.float32(-0.0)
    first = replace(
        first,
        observation=replace(first.observation, hand_keypoints_3d=first_points),
    )
    second = replace(
        second,
        observation=replace(second.observation, hand_keypoints_3d=second_points),
    )

    assert semantic_fingerprint(first) == semantic_fingerprint(second)
    assert np.signbit(second.observation.hand_keypoints_3d[0, 0, 0, 0])


def test_semantic_fingerprint_normalizes_all_numpy_integral_contract_fields() -> None:
    native = make_episode()
    source_files = tuple(
        replace(item, size_bytes=np.int64(item.size_bytes))
        for item in native.provenance.source_files
    )
    numpy_episode = replace(
        native,
        provenance=_provenance(
            source_files=source_files,
            source_schema_version=native.identity.source_schema_version,
        ),
        time_axis=replace(
            native.time_axis,
            frame_count=np.int64(native.time_axis.frame_count),
            fps_num=np.int64(native.time_axis.fps_num),
            fps_den=np.int64(native.time_axis.fps_den),
            frame_index_base=np.int64(native.time_axis.frame_index_base),
        ),
        main_video=replace(
            native.main_video,
            frame_count=np.int64(native.main_video.frame_count),
            width_px=np.int64(native.main_video.width_px),
            height_px=np.int64(native.main_video.height_px),
            fps_num=np.int64(native.main_video.fps_num),
            fps_den=np.int64(native.main_video.fps_den),
        ),
        calibration=replace(
            native.calibration,
            image_width_px=np.int64(native.calibration.image_width_px),
            image_height_px=np.int64(native.calibration.image_height_px),
        ),
        semantics=replace(
            native.semantics,
            subtask_sequence=tuple(
                replace(
                    item,
                    start_frame=np.int64(item.start_frame),
                    end_frame_exclusive=np.int64(item.end_frame_exclusive),
                )
                for item in native.semantics.subtask_sequence
            ),
        ),
    )

    validate_episode(numpy_episode)
    assert semantic_fingerprint(native) == semantic_fingerprint(numpy_episode)
