from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest

from canonical_qc import CanonicalInputError, StandardHdf5Adapter
from canonical_qc.bridge import CanonicalQcBridge
from tests.fixtures import write_standard_hdf5_episode
from tools.run_qc_pipeline import contexts_from_manifest


_JOINT_NAMES = tuple(
    f"{side}{base}"
    for side in ("left", "right")
    for base in (
        "Hand",
        "ThumbKnuckle",
        "ThumbIntermediateBase",
        "ThumbIntermediateTip",
        "ThumbTip",
        "IndexFingerKnuckle",
        "IndexFingerIntermediateBase",
        "IndexFingerIntermediateTip",
        "IndexFingerTip",
        "MiddleFingerKnuckle",
        "MiddleFingerIntermediateBase",
        "MiddleFingerIntermediateTip",
        "MiddleFingerTip",
        "RingFingerKnuckle",
        "RingFingerIntermediateBase",
        "RingFingerIntermediateTip",
        "RingFingerTip",
        "LittleFingerKnuckle",
        "LittleFingerIntermediateBase",
        "LittleFingerIntermediateTip",
        "LittleFingerTip",
    )
)


def _episode_with_invalid_joint(tmp_path: Path):
    root = tmp_path / "asset-001"
    hdf5_path, _ = write_standard_hdf5_episode(root, hand_quality="provided")
    with h5py.File(hdf5_path, "r+") as handle:
        valid = handle["/observation/hand_joint_valid_3d"]
        points = handle["/observation/hand_keypoints_3d"]
        for hand_index in range(2):
            for joint_index in range(21):
                points[:, hand_index, joint_index] = np.array(
                    [
                        hand_index * 1000 + joint_index * 10 + axis
                        for axis in range(3)
                    ],
                    dtype=np.float32,
                )
        valid[1, 0, 4] = False
        points[1, 0, 4] = np.nan
    return root, StandardHdf5Adapter().load(root)


def test_bridge_projects_exact_legacy_clip_without_mutating_episode(tmp_path: Path) -> None:
    source_root, episode = _episode_with_invalid_joint(tmp_path)
    before = episode.observation.hand_keypoints_3d.tobytes()
    bridge = CanonicalQcBridge(episode, source_root=source_root)

    clip = bridge.clip_inputs((1, 3))

    expected_names = _JOINT_NAMES
    assert tuple(clip.keypoints or {}) == expected_names
    assert clip.frame_indices == [1, 2]
    assert clip.fps == 10.0
    assert clip.intrinsics is not None
    assert clip.intrinsics.dtype == np.float64
    for hand_index, side in enumerate(("left", "right")):
        for joint_index, name in enumerate(expected_names[hand_index * 21 : (hand_index + 1) * 21]):
            projected = clip.keypoints[name]
            expected = episode.observation.hand_keypoints_3d[1:3, hand_index, joint_index]
            np.testing.assert_array_equal(projected, expected)
            assert projected.dtype == np.float32
            assert projected.flags.writeable is False
    assert np.isnan(clip.keypoints["leftThumbTip"][0]).all()
    np.testing.assert_array_equal(
        clip.keypoints["leftIndexFingerKnuckle"],
        np.tile(np.array([50, 51, 52], dtype=np.float32), (2, 1)),
    )
    np.testing.assert_array_equal(
        clip.keypoints["rightLittleFingerTip"],
        np.tile(np.array([1200, 1201, 1202], dtype=np.float32), (2, 1)),
    )
    assert clip.quality_hand is None
    assert clip.supplier_hand_quality_status is not None
    assert clip.supplier_hand_quality_status.tolist() == [["warning", "good"], ["good", "unknown"]]
    assert clip.hand_joint_valid_3d is not None
    assert clip.hand_joint_valid_3d.flags.writeable is False
    assert clip.hand_joint_valid_3d[0, 0, 4] == np.bool_(False)
    assert clip.hand_joint_valid_2d is not None
    assert clip.hand_joint_valid_2d.flags.writeable is False
    assert clip.timestamps_ns is not None and clip.timestamps_ns.flags.writeable is False
    assert clip.hand_keypoints_2d is not None and clip.hand_keypoints_2d.flags.writeable is False
    assert episode.observation.hand_keypoints_3d.tobytes() == before
    assert episode.observation.hand_keypoints_3d.flags.writeable is False


def test_bridge_preserves_semantics_and_resolves_only_declared_video(tmp_path: Path) -> None:
    source_root = tmp_path / "asset-001"
    _, video = write_standard_hdf5_episode(source_root)
    episode = StandardHdf5Adapter().load(source_root)
    bridge = CanonicalQcBridge(episode, source_root=source_root)

    payload = bridge.semantic_payload()

    assert payload == {
        "scene_id": "kitchen",
        "task_id": "pick-object",
        "task_category": "manipulation",
        "task_cn": "拿起物体",
        "task_en": "pick up object",
        "description_cn": "拿起物体并放到桌面中央",
        "description_en": "pick up the object and place it at the center",
        "subtask_sequence": (
            {
                "subtask_id": "subtask_001",
                "start_frame": 0,
                "end_frame_exclusive": 3,
                "description_cn": "拿起物体",
                "description_en": "pick up the object",
            },
        ),
    }
    assert bridge.video_path() == video.resolve()

    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(CanonicalInputError, match="source_root"):
        CanonicalQcBridge(episode, source_root=outside).video_path()


def test_bridge_asset_context_keeps_canonical_contract_and_half_open_range(tmp_path: Path) -> None:
    source_root = tmp_path / "batch" / "asset-001"
    write_standard_hdf5_episode(source_root)
    episode = StandardHdf5Adapter().load(source_root)
    bridge = CanonicalQcBridge(episode, source_root=source_root)

    context = bridge.asset_context(
        batch_root=tmp_path / "batch",
        report_path=tmp_path / "batch" / "quality_archive" / "asset-001.json",
        source_range=(1, 3),
        metadata={"profile": "acceptance"},
    )

    assert context.metadata["canonical_episode"] is episode
    assert context.metadata["canonical_episode"].observation.hand_keypoints_3d.flags.writeable is False
    assert context.metadata["canonical_source_root"] == str(source_root.resolve())
    assert context.source_range == (1, 3)
    assert context.source_files["video"]["path"] == "asset-001/main.mp4"
    assert context.source_files["hdf5"]["path"] == "asset-001/asset-001.h5"
    assert context.source_files["canonical_provenance"]["source_fingerprint"] == episode.provenance.source_fingerprint
    assert len(context.source_files["canonical_provenance"]["source_files"]) == 2
    assert hashlib.sha256(bridge.video_path().read_bytes()).hexdigest() == episode.main_video.sha256


def test_manifest_entrypoint_builds_canonical_context_without_legacy_loading(tmp_path: Path) -> None:
    batch = tmp_path / "batch"
    source_root = batch / "asset-001"
    write_standard_hdf5_episode(source_root)
    manifest = batch / "manifest.jsonl"
    candidates = batch / "candidate-windows.json"
    candidates.write_text("[]", encoding="utf-8")
    manifest.write_text(
        '{"asset_id":"asset-001","canonical_format":"hdf5",'
        '"canonical_source_path":"asset-001","candidate_windows_path":'
        '"candidate-windows.json","start_frame":1,"end_frame":2}\n',
        encoding="utf-8",
    )

    contexts = contexts_from_manifest(manifest, batch_root=batch)

    assert len(contexts) == 1
    context = contexts[0]
    assert context.source_range == (1, 3)
    assert context.metadata["canonical_episode"].identity.source_format == "hdf5"
    assert context.source_files["video"]["path"] == "asset-001/main.mp4"
    assert context.source_files["candidate_windows"]["path"] == "candidate-windows.json"


def test_bridge_rejects_symlink_source_root(tmp_path: Path) -> None:
    source_root = tmp_path / "asset-001"
    write_standard_hdf5_episode(source_root)
    episode = StandardHdf5Adapter().load(source_root)
    alias = tmp_path / "alias"
    alias.symlink_to(source_root, target_is_directory=True)

    with pytest.raises(CanonicalInputError, match="source_root"):
        CanonicalQcBridge(episode, source_root=alias)


def test_video_resolution_rejects_post_load_symlink_and_hash_drift(tmp_path: Path) -> None:
    symlink_root = tmp_path / "symlink" / "asset-001"
    write_standard_hdf5_episode(symlink_root)
    symlink_episode = StandardHdf5Adapter().load(symlink_root)
    video = symlink_root / "main.mp4"
    target = symlink_root / "target.mp4"
    video.rename(target)
    video.symlink_to(target.name)

    with pytest.raises(CanonicalInputError, match="main_video.path"):
        CanonicalQcBridge(symlink_episode, source_root=symlink_root).video_path()

    drift_root = tmp_path / "drift" / "asset-001"
    write_standard_hdf5_episode(drift_root)
    drift_episode = StandardHdf5Adapter().load(drift_root)
    with (drift_root / "main.mp4").open("ab") as stream:
        stream.write(b"drift")

    with pytest.raises(CanonicalInputError, match="main_video"):
        CanonicalQcBridge(drift_episode, source_root=drift_root).video_path()


def test_canonical_reserved_sources_override_supplemental_poison(tmp_path: Path) -> None:
    batch = tmp_path / "batch"
    source_root = batch / "asset-001"
    write_standard_hdf5_episode(source_root)
    episode = StandardHdf5Adapter().load(source_root)
    candidate = batch / "candidate.json"
    candidate.write_text("[]", encoding="utf-8")

    context = CanonicalQcBridge(episode, source_root=source_root).asset_context(
        batch_root=batch,
        report_path=batch / "quality_archive" / "asset-001.json",
        supplemental_source_files={
            "video": {"path": "poison.mp4"},
            "hdf5": {"path": "poison.h5"},
            "canonical_provenance": {"poison": True},
            "candidate_windows": {"path": "candidate.json"},
        },
    )

    assert context.source_files["video"]["path"] == "asset-001/main.mp4"
    assert context.source_files["hdf5"]["path"] == "asset-001/asset-001.h5"
    assert "poison" not in context.source_files["canonical_provenance"]
    assert context.source_files["candidate_windows"]["path"] == "candidate.json"


@pytest.mark.parametrize("source_range", [(-1, 1), (1, 1), (0, 4)])
def test_clip_inputs_rejects_invalid_half_open_range(tmp_path: Path, source_range: tuple[int, int]) -> None:
    source_root = tmp_path / "asset-001"
    write_standard_hdf5_episode(source_root)
    bridge = CanonicalQcBridge(StandardHdf5Adapter().load(source_root), source_root=source_root)

    with pytest.raises(ValueError, match="half-open"):
        bridge.clip_inputs(source_range)
