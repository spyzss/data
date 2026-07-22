from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import serve_human_qc_workbench as launcher
from tools.serve_human_qc_workbench import load_contexts, parse_args
from tests.qc_report_fixtures import make_v2_report


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


def test_launcher_rehydrates_manifest_metadata_for_strict_overlay_mapping(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "quality_archive"
    archive.mkdir()
    report = {
        "asset_id": "asset-1",
        "profile": "supplier",
        "source_files": {"video": {"path": "video.mp4"}},
        "manifest_metadata": {
            "supplier": "jdt",
            "primary_camera": "cam_left",
            "left_hand_2d_field": "left_points",
            "right_hand_2d_field": "right_points",
        },
    }
    (tmp_path / "video.mp4").write_bytes(b"video")
    (archive / "asset-1.json").write_text(json.dumps(report), encoding="utf-8")

    context = load_contexts(tmp_path, archive)[0]

    assert context.metadata == {
        "profile": "supplier",
        "manifest_row": report["manifest_metadata"],
    }


def _current_overlay_recipe(asset_id: str = "asset-1") -> dict[str, object]:
    return {
        "schema_version": "sam3_overlay_input.v2",
        "asset_id": asset_id,
        "producer_fingerprint_sha256": "sha256:" + "b" * 64,
        "video_source": "video",
        "video_identity": "sha256:" + "a" * 64,
        "candidate_intervals": [[0, 3]],
        "source_mapping": {
            "schema_version": "linear_ranges.v1",
            "ranges": [
                {
                    "start_frame": 0,
                    "end_frame_exclusive": 3,
                    "video_start_frame": 0,
                }
            ],
        },
        "keypoints_2d_reference": {
            "schema_version": "json_keypoints.v1",
            "relative_path": ".qc_pipeline/asset-1/overlay-inputs/points.json",
            "sha256": "sha256:" + "c" * 64,
            "size_bytes": 123,
        },
    }


def _current_report_with_recipe() -> dict[str, object]:
    recipe = _current_overlay_recipe()
    report = make_v2_report()
    report.update(
        asset_id="asset-1",
        source_files={"video": {"path": "video.mp4"}},
        sam3_containment={
            "flow": {
                "entry_gate": {"state": "ready", "eligible": True},
                "result_gate": {"verdict": "warn"},
                "exit_gate": {"state": "continue", "continue_to_next_module": True},
            },
            "runtime": {
                "artifact_state": "computed",
                "fingerprint_sha256": "sha256:" + "b" * 64,
                "overlay_input_recipe": recipe,
            },
        },
    )
    return report


def test_launcher_accepts_recipe_only_from_current_completed_v2_sam3_report(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "quality_archive"
    archive.mkdir()
    (tmp_path / "video.mp4").write_bytes(b"video")
    report = _current_report_with_recipe()
    (archive / "asset-1.json").write_text(json.dumps(report), encoding="utf-8")

    context = load_contexts(tmp_path, archive)[0]

    loaded = context.metadata["sam3_overlay_recipe"]
    assert loaded["schema_version"] == "sam3_overlay_input.v2"
    assert loaded["asset_id"] == "asset-1"
    assert loaded["candidate_intervals"] == ((0, 3),)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda report: report.update(schema_version="asset_qc_report.v1"),
        lambda report: report["sam3_containment"]["runtime"].update(  # type: ignore[index]
            artifact_state="stale"
        ),
        lambda report: report["sam3_containment"]["flow"]["exit_gate"].update(  # type: ignore[index]
            state="blocked", continue_to_next_module=False
        ),
        lambda report: report["sam3_containment"]["runtime"].update(  # type: ignore[index]
            fingerprint_sha256="sha256:" + "d" * 64
        ),
        lambda report: report["sam3_containment"]["runtime"][  # type: ignore[index]
            "overlay_input_recipe"
        ].update(
            keypoints_2d={"0": {"left": [[1.0, 2.0]]}}
        ),
        lambda report: report["sam3_containment"]["runtime"][  # type: ignore[index]
            "overlay_input_recipe"
        ]["source_mapping"].update(
            ranges=[
                {
                    "start_frame": 0,
                    "end_frame_exclusive": 2,
                    "video_start_frame": 0,
                }
            ]
        ),
    ],
)
def test_launcher_rejects_historical_stale_ambiguous_or_gapped_overlay_recipe(
    tmp_path: Path,
    mutation,
) -> None:
    archive = tmp_path / "quality_archive"
    archive.mkdir()
    (tmp_path / "video.mp4").write_bytes(b"video")
    report = _current_report_with_recipe()
    mutation(report)
    (archive / "asset-1.json").write_text(json.dumps(report), encoding="utf-8")

    context = load_contexts(tmp_path, archive)[0]

    assert "sam3_overlay_recipe" not in context.metadata


def test_launcher_accepts_explicit_model_and_bounded_overlay_inputs(
    tmp_path: Path,
) -> None:
    args = parse_args(
        [
            "--batch-root",
            str(tmp_path),
            "--quality-archive",
            "quality_archive",
            "--reviewer",
            "alice",
            "--sam3-model",
            str(tmp_path / "sam3-model"),
            "--overlay-cache-dir",
            ".human_qc/overlay-cache",
            "--overlay-workers",
            "1",
            "--overlay-max-pending",
            "3",
            "--overlay-max-cache-bytes",
            "1048576",
            "--overlay-max-ready-jobs",
            "8",
        ]
    )

    assert args.sam3_model == tmp_path / "sam3-model"
    assert args.overlay_cache_dir == Path(".human_qc/overlay-cache")
    assert args.overlay_workers == 1
    assert args.overlay_max_pending == 3
    assert args.overlay_max_cache_bytes == 1048576
    assert args.overlay_max_ready_jobs == 8


def test_launcher_has_a_non_none_safe_default_batch_cache_limit(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--batch-root",
            str(tmp_path),
            "--quality-archive",
            "quality_archive",
            "--reviewer",
            "alice",
        ]
    )

    assert isinstance(args.overlay_max_cache_bytes, int)
    assert args.overlay_max_cache_bytes > 0


def test_production_launcher_builds_one_shared_runtime_worker_and_injects_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = SimpleNamespace(
        asset_id="asset-1",
        batch_root=tmp_path.resolve(),
        report_path=tmp_path / "quality_archive" / "asset-1.json",
    )
    model = tmp_path / "sam3-model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    captured: dict[str, object] = {}
    worker = SimpleNamespace(shutdown_calls=0)

    def shutdown(*, wait: bool = True) -> None:
        assert wait is True
        worker.shutdown_calls += 1

    worker.shutdown = shutdown
    provider = object()
    freeze_calls: list[str] = []
    media_catalog = SimpleNamespace(
        freeze_sources=lambda: freeze_calls.append("freeze")
    )

    monkeypatch.setattr(launcher, "load_contexts", lambda *_args: [context])
    monkeypatch.setattr(launcher, "MediaCatalog", lambda _contexts: media_catalog)
    monkeypatch.setattr(
        launcher,
        "WarnReviewService",
        lambda **_kwargs: SimpleNamespace(get_task=lambda _asset_id: {}),
    )

    class FakeWorkbench:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(launcher, "WarnWorkbenchService", FakeWorkbench)

    calls = 0

    def build_overlay_runtime(**kwargs: object):
        nonlocal calls
        calls += 1
        captured["overlay_runtime_kwargs"] = kwargs
        return SimpleNamespace(worker=worker, provider=provider)

    monkeypatch.setattr(
        launcher,
        "build_production_overlay_runtime",
        build_overlay_runtime,
    )

    runtime = launcher.build_workbench_runtime(
        batch_root=tmp_path,
        quality_archive=tmp_path / "quality_archive",
        reviewer="alice",
        sam3_model=model,
        overlay_cache_dir=Path(".human_qc/overlay-cache"),
        overlay_workers=1,
        overlay_max_pending=2,
        overlay_max_cache_bytes=2048,
        overlay_max_ready_jobs=5,
    )

    assert calls == 1
    assert freeze_calls == ["freeze"]
    assert captured["overlay_provider"] is provider
    build_kwargs = captured["overlay_runtime_kwargs"]
    assert build_kwargs["contexts"] == {"asset-1": context}
    assert build_kwargs["model_path"] == model
    assert build_kwargs["cache_relative"] == Path(".human_qc/overlay-cache")
    assert build_kwargs["max_workers"] == 1
    assert build_kwargs["max_pending"] == 2
    assert build_kwargs["max_cache_bytes"] == 2048
    assert build_kwargs["max_ready_jobs"] == 5
    runtime.shutdown()
    assert worker.shutdown_calls == 1


def test_launcher_without_model_fails_closed_instead_of_leaving_overlay_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = SimpleNamespace(
        asset_id="asset-1",
        batch_root=tmp_path.resolve(),
        report_path=tmp_path / "quality_archive" / "asset-1.json",
    )
    captured: dict[str, object] = {}
    freeze_calls: list[str] = []
    monkeypatch.setattr(launcher, "load_contexts", lambda *_args: [context])
    monkeypatch.setattr(
        launcher,
        "MediaCatalog",
        lambda _contexts: SimpleNamespace(
            freeze_sources=lambda: freeze_calls.append("freeze")
        ),
    )
    monkeypatch.setattr(
        launcher,
        "WarnReviewService",
        lambda **_kwargs: SimpleNamespace(get_task=lambda _asset_id: {}),
    )

    class FakeWorkbench:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(launcher, "WarnWorkbenchService", FakeWorkbench)

    runtime = launcher.build_workbench_runtime(
        batch_root=tmp_path,
        quality_archive=tmp_path / "quality_archive",
        reviewer="alice",
    )
    overlay_provider = captured["overlay_provider"]
    selected = (
        SimpleNamespace(
            issue_id="issue-1",
            frame_range=SimpleNamespace(start_frame=10, end_frame_exclusive=20),
        ),
    )

    result = overlay_provider.get_asset_overlays("asset-1", selected)

    assert result["issue-1"].status == "failed"
    assert result["issue-1"].code == "overlay_model_unavailable"
    assert runtime.worker is None
    assert freeze_calls == ["freeze"]


def test_main_closes_overlay_worker_in_finally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    runtime = SimpleNamespace(
        service=object(),
        shutdown=lambda: events.append("worker_shutdown"),
    )

    class FakeServer:
        server_port = 8897

        def serve_forever(self) -> None:
            raise KeyboardInterrupt

        def server_close(self) -> None:
            events.append("server_close")

    monkeypatch.setattr(
        launcher,
        "parse_args",
        lambda _argv=None: SimpleNamespace(
            batch_root=tmp_path,
            quality_archive=tmp_path / "quality_archive",
            reviewer="alice",
            profile="acceptance",
            lease_ttl_seconds=900,
            host="127.0.0.1",
            port=8897,
            sam3_model=tmp_path / "sam3-model",
            overlay_cache_dir=Path(".human_qc/overlay-cache"),
            overlay_workers=1,
            overlay_max_pending=1,
            overlay_max_cache_bytes=None,
            overlay_max_ready_jobs=None,
        ),
    )
    monkeypatch.setattr(launcher, "build_workbench_runtime", lambda **_kwargs: runtime)
    monkeypatch.setattr(
        launcher,
        "create_http_server",
        lambda _host, _port, service: FakeServer(),
    )

    assert launcher.main([]) == 0
    assert events == ["server_close", "worker_shutdown"]


def test_compatibility_service_retains_and_exposes_its_shutdown_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_calls: list[str] = []
    runtime = SimpleNamespace(
        service=SimpleNamespace(),
        shutdown=lambda: shutdown_calls.append("shutdown"),
    )
    monkeypatch.setattr(
        launcher,
        "build_workbench_runtime",
        lambda **_kwargs: runtime,
    )

    service = launcher.build_workbench_service(
        batch_root=tmp_path,
        quality_archive=tmp_path / "quality_archive",
        reviewer="alice",
        sam3_model=tmp_path / "model",
    )

    assert service._workbench_runtime_owner is runtime
    service.shutdown()
    assert shutdown_calls == ["shutdown"]


def test_runtime_preload_failure_shuts_down_overlay_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = SimpleNamespace(
        asset_id="asset-1",
        batch_root=tmp_path,
        report_path=tmp_path / "quality_archive" / "asset-1.json",
    )
    shutdown_calls: list[str] = []
    overlay_runtime = SimpleNamespace(
        provider=object(),
        shutdown=lambda: shutdown_calls.append("shutdown"),
    )
    monkeypatch.setattr(launcher, "load_contexts", lambda *_args: [context])
    monkeypatch.setattr(
        launcher,
        "MediaCatalog",
        lambda _contexts: SimpleNamespace(freeze_sources=lambda: None),
    )
    monkeypatch.setattr(
        launcher,
        "build_production_overlay_runtime",
        lambda **_kwargs: overlay_runtime,
    )
    monkeypatch.setattr(
        launcher,
        "WarnReviewService",
        lambda **_kwargs: SimpleNamespace(
            get_task=lambda _asset_id: (_ for _ in ()).throw(RuntimeError("preload"))
        ),
    )
    monkeypatch.setattr(
        launcher,
        "WarnWorkbenchService",
        lambda **_kwargs: SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="preload"):
        launcher.build_workbench_runtime(
            batch_root=tmp_path,
            quality_archive=tmp_path / "quality_archive",
            reviewer="alice",
            sam3_model=tmp_path / "model",
        )

    assert shutdown_calls == ["shutdown"]
