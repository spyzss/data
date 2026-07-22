from __future__ import annotations

import json
from pathlib import Path

from tools.serve_human_qc_workbench import load_contexts


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
