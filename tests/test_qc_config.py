from pathlib import Path

import pytest

from qc_common.config import load_qc_acceptance_config


def test_default_qc_config_loads_and_hashes() -> None:
    loaded = load_qc_acceptance_config()

    assert loaded.config_version == "qc_acceptance_v1.1.0"
    assert loaded.module_rules("video_quality")["fps_below_min"]["verdict"] == "fail"
    assert loaded.json_reference()["config_hash"].startswith("sha256:")


def test_video_only_yaml_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "video-only.yaml"
    path.write_text("decode:\n  max_sample_frames: 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unified qc_acceptance config"):
        load_qc_acceptance_config(path)


def test_released_config_archive_matches_canonical() -> None:
    canonical = Path("configs/qc_acceptance.yaml").read_bytes()
    archived = Path("configs/qc_acceptance/qc_acceptance_v1.1.0.yaml").read_bytes()

    assert archived == canonical
