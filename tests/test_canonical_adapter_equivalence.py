from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np

from canonical_qc import StandardHdf5Adapter, StandardLeRobotAdapter, semantic_fingerprint
from tests.fixtures import write_standard_hdf5_episode, write_standard_lerobot_dataset


def test_hdf5_and_lerobot_normalize_to_identical_logical_episode(tmp_path: Path) -> None:
    hdf5_dir = tmp_path / "asset-001"
    _, video = write_standard_hdf5_episode(hdf5_dir)
    lerobot_root = write_standard_lerobot_dataset(
        tmp_path / "lerobot", main_video_source=video
    )

    hdf5 = StandardHdf5Adapter().load(hdf5_dir)
    lerobot = StandardLeRobotAdapter().load(lerobot_root)

    assert semantic_fingerprint(hdf5) == semantic_fingerprint(lerobot)
    assert hdf5.identity.asset_id == lerobot.identity.asset_id
    assert hdf5.identity.batch_id == lerobot.identity.batch_id
    assert hdf5.identity.supplier_id == lerobot.identity.supplier_id
    for name in ("time_axis", "observation", "calibration", "semantics", "supplier_evidence"):
        left, right = getattr(hdf5, name), getattr(lerobot, name)
        for item in fields(left):
            lv, rv = getattr(left, item.name), getattr(right, item.name)
            if isinstance(lv, np.ndarray):
                np.testing.assert_equal(lv, rv)
            else:
                assert lv == rv
