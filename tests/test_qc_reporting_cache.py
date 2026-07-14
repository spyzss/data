from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from qc_reporting.cache import (
    build_source_manifest,
    load_projection_cache,
    write_projection_cache,
)
from qc_reporting.projection import project_quality_archive
from tools.build_qc_json_projection import run_projection_cli
from tests.test_qc_reporting_projection import _write_report


def _make_archive(root: Path) -> Path:
    archive = root / "quality_archive"
    _write_report(archive, "a", "acceptance", "pass")
    _write_report(archive, "b", "acceptance", "fail")
    return archive


def test_cache_can_be_deleted_and_rebuilt_identically(tmp_path: Path) -> None:
    archive = _make_archive(tmp_path)
    cache_dir = tmp_path / "cache"
    manifest = build_source_manifest(archive)

    first = project_quality_archive(archive)
    write_projection_cache(first, cache_dir)
    first_loaded = load_projection_cache(cache_dir, manifest)

    assert first_loaded == first
    shutil.rmtree(cache_dir)

    second = project_quality_archive(archive)
    write_projection_cache(second, cache_dir)
    second_loaded = load_projection_cache(cache_dir, build_source_manifest(archive))

    assert second_loaded == second
    assert first_loaded == second_loaded


def test_revision_change_invalidates_cache(tmp_path: Path) -> None:
    archive = _make_archive(tmp_path)
    cache_dir = tmp_path / "cache"
    projection = project_quality_archive(archive)
    write_projection_cache(projection, cache_dir)

    report_path = archive / "a.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["report_revision"] = int(report.get("report_revision", 0)) + 1
    report_path.write_text(json.dumps(report), encoding="utf-8")

    assert load_projection_cache(cache_dir, build_source_manifest(archive)) is None


def test_cache_corruption_or_missing_table_returns_none(tmp_path: Path) -> None:
    archive = _make_archive(tmp_path)
    cache_dir = tmp_path / "cache"
    projection = project_quality_archive(archive)
    write_projection_cache(projection, cache_dir)
    manifest = build_source_manifest(archive)

    (cache_dir / "issues.parquet").write_bytes(b"not parquet")
    assert load_projection_cache(cache_dir, manifest) is None

    write_projection_cache(projection, cache_dir)
    (cache_dir / "execution.parquet").unlink()
    assert load_projection_cache(cache_dir, manifest) is None


def test_cli_rebuilds_missing_or_stale_cache_from_json(tmp_path: Path) -> None:
    archive = _make_archive(tmp_path)
    cache_dir = tmp_path / "cache"
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    run_projection_cli(archive, first_output, formats=("csv",), cache_dir=cache_dir)
    assert (cache_dir / "source_reports.json").is_file()

    (cache_dir / "assets.parquet").unlink()
    run_projection_cli(archive, second_output, formats=("csv",), cache_dir=cache_dir)

    assert (cache_dir / "assets.parquet").is_file()
    assert pd.read_csv(first_output / "assets.csv").to_dict("records") == pd.read_csv(
        second_output / "assets.csv"
    ).to_dict("records")

