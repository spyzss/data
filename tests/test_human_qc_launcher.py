from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.serve_human_qc_workbench import load_contexts, parse_args


def test_warn_only_launcher_accepts_report_when_declared_hdf5_is_missing(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "quality_archive"
    archive.mkdir()
    report = {
        "asset_id": "asset-1",
        "source_files": {"hdf5": {"path": "missing/asset-1.hdf5"}},
    }
    (archive / "asset-1.json").write_text(json.dumps(report), encoding="utf-8")

    contexts = load_contexts(tmp_path, archive)

    assert len(contexts) == 1
    assert contexts[0].source_files["hdf5"]["path"] == "missing/asset-1.hdf5"


def test_warn_launcher_requires_fixed_reviewer_identity(tmp_path: Path) -> None:
    argv = [
        "--batch-root",
        str(tmp_path),
        "--quality-archive",
        "quality_archive",
    ]

    with pytest.raises(SystemExit):
        parse_args(argv)

    parsed = parse_args([*argv, "--reviewer", "alice"])
    assert parsed.reviewer == "alice"


def test_launcher_preserves_safe_source_hash_metadata(tmp_path: Path) -> None:
    archive = tmp_path / "quality_archive"
    archive.mkdir()
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    report = {
        "asset_id": "asset-1",
        "source_files": {
            "video": {
                "path": "video.mp4",
                "sha256": "sha256:" + "a" * 64,
                "size_bytes": 5,
                "internal_path": "/private/secret",
            }
        },
    }
    (archive / "asset-1.json").write_text(json.dumps(report), encoding="utf-8")

    contexts = load_contexts(tmp_path, archive)

    assert contexts[0].source_files["video"] == {
        "path": "video.mp4",
        "sha256": "sha256:" + "a" * 64,
        "size_bytes": 5,
    }
