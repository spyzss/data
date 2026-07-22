from __future__ import annotations

from dataclasses import replace
import importlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest


def _renderer_module():
    try:
        return importlib.import_module("human_qc.sam3_overlay_renderer")
    except ModuleNotFoundError:
        pytest.fail("production SAM3 overlay renderer is not implemented")


class FakeFrameProvider:
    def __init__(
        self,
        *,
        mapping_identity: str = "mapping-v1",
        fail_source_frame: int | None = None,
    ) -> None:
        self.mapping_identity = mapping_identity
        self.fail_source_frame = fail_source_frame
        self.source_frames: list[int] = []

    @property
    def input_identity(self) -> dict[str, object]:
        return {
            "recipe_version": 1,
            "mapping": self.mapping_identity,
            "video": "trusted-video-v1",
            "keypoints": "trusted-keypoints-v1",
        }

    def read_frame(self, source_frame: int):
        module = _renderer_module()
        self.source_frames.append(source_frame)
        if source_frame == self.fail_source_frame:
            raise module.OverlaySetupError("overlay_decode_failed")
        return module.OverlayFrame(
            source_frame=source_frame,
            video_frame=source_frame + 1000,
            frame_rgb=np.full((4, 6, 3), source_frame % 255, dtype=np.uint8),
            keypoints={
                "left": np.asarray([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
            },
        )


class RecordingSegmenter:
    def __init__(self, *, fail_call: int | None = None) -> None:
        self.fail_call = fail_call
        self.calls: list[int] = []

    def segment_frame(self, frame, queries, config):
        self.calls.append(int(frame[0, 0, 0]))
        if self.fail_call is not None and len(self.calls) == self.fail_call:
            raise RuntimeError("private SAM3 traceback")
        return [
            SimpleNamespace(
                mask=np.ones(frame.shape[:2], dtype=bool),
                category="hand",
            )
        ]


class RecordingRuntimeProvider:
    def __init__(self, segmenter: RecordingSegmenter) -> None:
        self.segmenter = segmenter
        self.calls: list[tuple[Path, dict[str, object]]] = []

    def get_segmenter(self, model_path: Path, runtime_config: dict[str, object]):
        self.calls.append((Path(model_path), dict(runtime_config)))
        return self.segmenter


class RecordingEncoder:
    def __init__(self, output_path: Path, *, fail_write: int | None = None) -> None:
        self.output_path = output_path
        self.fail_write = fail_write
        self.frames: list[np.ndarray] = []
        self.closed = False

    def write(self, frame: np.ndarray) -> None:
        if self.fail_write is not None and len(self.frames) + 1 == self.fail_write:
            raise OSError("private encoder command failed")
        self.frames.append(np.asarray(frame).copy())

    def close(self) -> None:
        self.closed = True
        self.output_path.write_bytes(b"mp4:" + str(len(self.frames)).encode("ascii"))


def _renderer(
    tmp_path: Path,
    *,
    provider: FakeFrameProvider | None = None,
    segmenter: RecordingSegmenter | None = None,
    encoder: RecordingEncoder | None = None,
):
    module = _renderer_module()
    frame_provider = provider or FakeFrameProvider()
    concrete_segmenter = segmenter or RecordingSegmenter()
    runtime = RecordingRuntimeProvider(concrete_segmenter)
    observed: dict[str, RecordingEncoder] = {}

    def encoder_factory(output_path: Path, fps: float, frame_size: tuple[int, int]):
        assert fps == 30.0
        assert frame_size == (6, 4)
        value = encoder or RecordingEncoder(output_path)
        value.output_path = output_path
        observed["encoder"] = value
        return value

    renderer = module.Sam3OverlayRenderer(
        frame_provider=frame_provider,
        runtime_provider=runtime,
        model_path=tmp_path / "sam3-model",
        runtime_config={"confidence_threshold": 0.5},
        queries=("hand", "left hand", "right hand"),
        encoder_factory=encoder_factory,
        frame_composer=lambda sample, masks: sample.frame_rgb,
        media_probe=lambda _path: SimpleNamespace(
            frame_count=len(observed["encoder"].frames),
            width_px=6,
            height_px=4,
            fps_num=30,
            fps_den=1,
            codec="mpeg4",
            container_format="mov,mp4,m4a,3gp,3g2,mj2",
            container_major_brand="isom",
        ),
    )
    return renderer, frame_provider, runtime, concrete_segmenter, observed


def _request(tmp_path: Path, renderer: object, *, interval=(120, 182)):
    from human_qc.overlay_worker import OverlayRequest

    return OverlayRequest(
        asset_id="asset-1",
        cache_root=tmp_path / ".human_qc" / "overlay-cache" / "asset-1",
        source_sha256="sha256:" + "a" * 64,
        intervals=(interval,),
        fps=30.0,
        total_frames=500,
        model_hash="sha256:" + "b" * 64,
        config_hash="sha256:" + "c" * 64,
        input_fingerprint_hash="sha256:" + "d" * 64,
        renderer_version="sam3-overlay-renderer-v1",
        renderer=renderer,
    )


def _wait(worker: object, request: object):
    import time

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        view = worker.get(request)
        if view.status in {"ready", "failed"}:
            return view
        time.sleep(0.01)
    pytest.fail("overlay render did not become terminal")


def _v2_recipe_with_json_sidecar(
    tmp_path: Path,
    video: Path,
    *,
    source_frame: int,
    video_frame: int,
    asset_id: str = "asset-a",
) -> tuple[dict[str, object], Path, tuple[int, int, int, int]]:
    from qc_pipeline.artifacts import file_sha256

    sidecar = tmp_path / f"{asset_id}-points-{source_frame}.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": "sam3_overlay_keypoints.v1",
                "asset_id": asset_id,
                "frames": [
                    {
                        "source_frame": source_frame,
                        "keypoints": {"left": [[1.0, 2.0]]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    stat = video.stat()
    recipe: dict[str, object] = {
        "schema_version": "sam3_overlay_input.v2",
        "asset_id": asset_id,
        "producer_fingerprint_sha256": "sha256:" + "f" * 64,
        "video_source": "video",
        "video_identity": file_sha256(video),
        "candidate_intervals": [[source_frame, source_frame + 1]],
        "source_mapping": {
            "schema_version": "linear_ranges.v1",
            "ranges": [
                {
                    "start_frame": source_frame,
                    "end_frame_exclusive": source_frame + 1,
                    "video_start_frame": video_frame,
                }
            ],
        },
        "keypoints_2d_reference": {
            "schema_version": "json_keypoints.v1",
            "relative_path": sidecar.name,
            "sha256": file_sha256(sidecar),
            "size_bytes": sidecar.stat().st_size,
        },
    }
    identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    return recipe, sidecar, identity


def test_renderer_consumes_every_explicit_source_frame_once_and_uses_shared_runtime(
    tmp_path: Path,
) -> None:
    renderer, provider, runtime, segmenter, observed = _renderer(tmp_path)
    output = tmp_path / "segment.mp4"

    metadata = renderer.render_interval(
        _request(tmp_path, renderer),
        120,
        182,
        output,
    )

    assert provider.source_frames == list(range(120, 182))
    assert len(segmenter.calls) == 62
    assert runtime.calls == [
        (tmp_path / "sam3-model", {"confidence_threshold": 0.5})
    ]
    assert len(observed["encoder"].frames) == 62
    assert observed["encoder"].closed is True
    assert metadata == {
        "frame_count": 62,
        "fps": 30.0,
        "width_px": 6,
        "height_px": 4,
        "codec": "mpeg4",
        "container_format": "mov,mp4,m4a,3gp,3g2,mj2",
        "container_major_brand": "isom",
        "first_source_frame": 120,
        "end_source_frame_exclusive": 182,
        "mapping": "explicit",
        "renderer_version": "sam3-overlay-renderer-v1",
    }
    assert output.read_bytes() == b"mp4:62"


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        ("mapping", "overlay_mapping_unavailable"),
        ("decode", "overlay_decode_failed"),
        ("inference", "overlay_inference_failed"),
        ("encoder", "overlay_encoder_failed"),
    ],
)
def test_renderer_failures_are_stable_and_never_publish_a_partial_ready_segment(
    tmp_path: Path,
    failure: str,
    expected_code: str,
) -> None:
    module = _renderer_module()
    provider = FakeFrameProvider(
        mapping_identity="" if failure == "mapping" else "mapping-v1",
        fail_source_frame=122 if failure == "decode" else None,
    )
    segmenter = RecordingSegmenter(fail_call=3 if failure == "inference" else None)
    encoder = RecordingEncoder(
        tmp_path / "unused.mp4", fail_write=3 if failure == "encoder" else None
    )
    renderer, _, _, _, _ = _renderer(
        tmp_path,
        provider=provider,
        segmenter=segmenter,
        encoder=encoder,
    )
    request = _request(tmp_path, renderer, interval=(120, 125))
    from human_qc.overlay_worker import BoundedOverlayWorker

    worker = BoundedOverlayWorker(max_workers=1, max_pending=0)
    try:
        worker.submit(request)
        failed = _wait(worker, request)
    finally:
        worker.shutdown()

    assert failed.status == "failed"
    assert failed.code == expected_code
    assert all(segment.status == "failed" for segment in failed.segments)
    assert all(segment.path is None for segment in failed.segments)
    assert not list(request.cache_root.rglob("*.partial.mp4"))
    assert expected_code in module.PUBLIC_OVERLAY_FAILURE_CODES


def test_request_cache_identity_changes_for_source_model_config_mapping_or_renderer(
    tmp_path: Path,
) -> None:
    module = _renderer_module()
    context = SimpleNamespace(asset_id="asset-1", batch_root=tmp_path)
    source = SimpleNamespace(
        path=tmp_path / "video.mp4",
        etag="sha256:" + "a" * 64,
        fps=30.0,
        total_frames=500,
    )
    selected = (
        SimpleNamespace(
            frame_range=SimpleNamespace(start_frame=120, end_frame_exclusive=182)
        ),
    )
    provider = FakeFrameProvider()
    renderer, *_ = _renderer(tmp_path, provider=provider)

    baseline = module.build_overlay_request(
        context=context,
        source=source,
        selected=selected,
        cache_root=tmp_path / ".human_qc" / "overlay-cache" / "asset-1",
        renderer=renderer,
        model_hash="model-v1",
        runtime_config={"confidence_threshold": 0.5},
        renderer_inputs={"queries": ["hand"], "style": "mask-and-keypoints-v1"},
    )

    changed = (
        replace(baseline, source_sha256="sha256:" + "e" * 64),
        replace(baseline, model_hash="model-v2"),
        replace(baseline, config_hash="config-v2"),
        replace(baseline, input_fingerprint_hash="mapping-v2"),
        replace(baseline, renderer_version="renderer-v2"),
    )
    assert all(item.cache_key != baseline.cache_key for item in changed)
    assert baseline.intervals == ((120, 182),)
    assert baseline.input_fingerprint_hash.startswith("sha256:")


def test_request_factory_rejects_unproved_mapping_instead_of_guessing_direct_frames(
    tmp_path: Path,
) -> None:
    module = _renderer_module()
    context = SimpleNamespace(asset_id="asset-1", batch_root=tmp_path)
    source = SimpleNamespace(
        path=tmp_path / "video.mp4",
        etag="sha256:" + "a" * 64,
        fps=30.0,
        total_frames=500,
    )
    selected = (
        SimpleNamespace(
            frame_range=SimpleNamespace(start_frame=120, end_frame_exclusive=182)
        ),
    )
    provider = FakeFrameProvider(mapping_identity="")
    renderer, *_ = _renderer(tmp_path, provider=provider)

    with pytest.raises(module.OverlaySetupError, match="overlay_mapping_unavailable"):
        module.build_overlay_request(
            context=context,
            source=source,
            selected=selected,
            cache_root=tmp_path / ".human_qc" / "overlay-cache" / "asset-1",
            renderer=renderer,
            model_hash="model-v1",
            runtime_config={"confidence_threshold": 0.5},
            renderer_inputs={"queries": ["hand"]},
        )


@pytest.mark.parametrize("failure", ["seek_false", "wrong_position", "out_of_bounds"])
def test_explicit_recipe_provider_rejects_untrusted_video_seek_or_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    import cv2

    module = _renderer_module()

    class Capture:
        position = 0.0

        @staticmethod
        def isOpened() -> bool:
            return True

        @staticmethod
        def release() -> None:
            return None

        def set(self, prop: int, value: float) -> bool:
            assert prop == cv2.CAP_PROP_POS_FRAMES
            if failure == "seek_false":
                return False
            self.position = float(value)
            return True

        def get(self, prop: int) -> float:
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 5.0
            if prop == cv2.CAP_PROP_POS_FRAMES:
                if failure == "wrong_position":
                    return self.position + 2.0
                return self.position
            return 0.0

        def read(self):
            self.position += 1.0
            return True, np.zeros((4, 6, 3), dtype=np.uint8)

    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: Capture())
    video_frame = 5 if failure == "out_of_bounds" else 2
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    recipe, sidecar, identity = _v2_recipe_with_json_sidecar(
        tmp_path,
        video_path,
        source_frame=10,
        video_frame=video_frame,
    )
    provider = module.ExplicitRecipeFrameProvider(
        video_path=video_path,
        recipe=recipe,
        keypoint_reference_path=sidecar,
        expected_video_identity=identity,
    )

    with pytest.raises(module.OverlaySetupError, match="overlay_decode_failed"):
        provider.read_frame(10)


def test_runtime_recreates_frame_provider_without_retaining_per_identity_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qc_pipeline.context import AssetContext
    from human_qc.warn_workbench_service import FrameRangeDto, OverlayIssueInput

    module = _renderer_module()
    context = AssetContext(
        asset_id="asset-a",
        batch_root=tmp_path,
        report_path=tmp_path / "quality_archive" / "asset-a.json",
        source_files={},
        source_range=(0, 1),
        metadata={},
    )
    identity = ["sha256:" + "a" * 64]

    class Catalog:
        @staticmethod
        def source(_asset_id: str):
            return SimpleNamespace(etag=identity[0], fps=10.0, total_frames=1)

    factory_calls: list[str] = []

    def frame_provider_factory(_context: object, source: object):
        factory_calls.append(source.etag)
        return FakeFrameProvider(mapping_identity=source.etag)

    class Renderer:
        renderer_version = "source-revalidation-test-v1"

        def __init__(self, *, frame_provider: object, **_kwargs: object) -> None:
            self.frame_provider = frame_provider

        @staticmethod
        def render_interval(
            _request: object,
            _start: int,
            _end: int,
            output_path: Path,
        ) -> dict[str, object]:
            output_path.write_bytes(b"mp4")
            return {"frame_count": 1, "fps": 10.0}

    monkeypatch.setattr(module, "Sam3OverlayRenderer", Renderer)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    runtime = module.build_production_overlay_runtime(
        contexts={"asset-a": context},
        media_catalog=Catalog(),
        model_path=model,
        max_cache_bytes=1024 * 1024,
        frame_provider_factory=frame_provider_factory,
    )
    selected = (OverlayIssueInput("issue", FrameRangeDto(0, 1)),)

    def wait_ready() -> None:
        import time

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            view = runtime.provider.get_asset_overlays("asset-a", selected)["issue"]
            if view.status == "ready":
                return
            if view.status == "failed":
                pytest.fail(f"unexpected failed overlay: {view.code}")
            time.sleep(0.01)
        pytest.fail("overlay did not become ready")

    try:
        wait_ready()
        identity[0] = "sha256:" + "b" * 64
        wait_ready()
    finally:
        runtime.shutdown()

    assert factory_calls[0] == "sha256:" + "a" * 64
    assert factory_calls[-1] == "sha256:" + "b" * 64
    assert set(factory_calls) == {
        "sha256:" + "a" * 64,
        "sha256:" + "b" * 64,
    }
    assert runtime.frame_providers == {}


def test_explicit_recipe_provider_rejects_source_file_substitution_before_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cv2
    from qc_pipeline.artifacts import file_sha256

    module = _renderer_module()
    video = tmp_path / "video.mp4"
    video.write_bytes(b"original")
    recipe, sidecar, identity = _v2_recipe_with_json_sidecar(
        tmp_path,
        video,
        source_frame=0,
        video_frame=0,
    )
    provider = module.ExplicitRecipeFrameProvider(
        video_path=video,
        recipe=recipe,
        keypoint_reference_path=sidecar,
        expected_video_identity=identity,
    )
    video.write_bytes(b"replaced")
    monkeypatch.setattr(
        cv2,
        "VideoCapture",
        lambda _path: pytest.fail("substituted source must fail before decoder open"),
    )

    with pytest.raises(module.OverlaySetupError, match="overlay_source_unavailable"):
        provider.read_frame(0)


def test_v2_sidecar_provider_decodes_from_verified_open_file_identity_not_reopened_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cv2
    from qc_pipeline.artifacts import file_sha256

    module = _renderer_module()
    video = tmp_path / "video.mp4"
    video.write_bytes(b"source-bytes")
    sidecar = tmp_path / "points.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": "sam3_overlay_keypoints.v1",
                "asset_id": "asset-a",
                "frames": [
                    {
                        "source_frame": 0,
                        "keypoints": {"left": [[1.0, 2.0]]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    recipe = {
        "schema_version": "sam3_overlay_input.v2",
        "asset_id": "asset-a",
        "producer_fingerprint_sha256": "sha256:" + "f" * 64,
        "video_source": "video",
        "video_identity": file_sha256(video),
        "candidate_intervals": [[0, 1]],
        "source_mapping": {
            "schema_version": "linear_ranges.v1",
            "ranges": [
                {
                    "start_frame": 0,
                    "end_frame_exclusive": 1,
                    "video_start_frame": 0,
                }
            ],
        },
        "keypoints_2d_reference": {
            "schema_version": "json_keypoints.v1",
            "relative_path": "points.json",
            "sha256": file_sha256(sidecar),
            "size_bytes": sidecar.stat().st_size,
        },
    }

    class Capture:
        position = 0.0

        @staticmethod
        def isOpened() -> bool:
            return True

        @staticmethod
        def release() -> None:
            return None

        def get(self, prop: int) -> float:
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 1.0
            return self.position

        def set(self, _prop: int, value: float) -> bool:
            self.position = value
            return True

        def read(self):
            self.position += 1
            return True, np.zeros((4, 6, 3), dtype=np.uint8)

    opened_paths: list[str] = []

    def open_capture(path: str):
        opened_paths.append(path)
        assert path.startswith("/dev/fd/")
        return Capture()

    monkeypatch.setattr(cv2, "VideoCapture", open_capture)
    provider = module.ExplicitRecipeFrameProvider(
        video_path=video,
        recipe=recipe,
        keypoint_reference_path=sidecar,
        expected_video_identity=(
            video.stat().st_dev,
            video.stat().st_ino,
            video.stat().st_size,
            video.stat().st_mtime_ns,
        ),
    )

    frame = provider.read_frame(0)
    provider.close()

    assert frame.video_frame == 0
    assert opened_paths and opened_paths[0].startswith("/dev/fd/")


def test_request_time_frame_provider_creation_does_not_read_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd
    from qc_pipeline.artifacts import file_sha256
    from qc_pipeline.context import AssetContext

    module = _renderer_module()
    video = tmp_path / "video.mp4"
    parquet = tmp_path / "points.parquet"
    video.write_bytes(b"video")
    parquet.write_bytes(b"parquet")
    recipe = {
        "schema_version": "sam3_overlay_input.v2",
        "asset_id": "asset-a",
        "producer_fingerprint_sha256": "sha256:" + "f" * 64,
        "video_source": "video",
        "video_identity": file_sha256(video),
        "candidate_intervals": [[0, 1]],
        "source_mapping": {
            "schema_version": "linear_ranges.v1",
            "ranges": [
                {
                    "start_frame": 0,
                    "end_frame_exclusive": 1,
                    "video_start_frame": 0,
                }
            ],
        },
        "keypoints_2d_reference": {
            "schema_version": "parquet_columns.v2",
            "source": "parquet",
            "sha256": file_sha256(parquet),
            "size_bytes": parquet.stat().st_size,
            "row_mapping": "source_frame_index",
            "fields": {"left": "left_points"},
        },
    }
    context = AssetContext(
        asset_id="asset-a",
        batch_root=tmp_path,
        report_path=tmp_path / "quality_archive" / "asset-a.json",
        source_files={
            "video": {"path": "video.mp4"},
            "parquet": {"path": "points.parquet"},
        },
        metadata={"sam3_overlay_recipe": recipe},
    )
    stat = video.stat()
    source = SimpleNamespace(
        path=video,
        etag=file_sha256(video),
        identity=(stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns),
    )
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda *_args, **_kwargs: pytest.fail("task GET must not read Parquet"),
    )

    provider = module.frame_provider_from_context(context, source)

    assert provider.input_identity["keypoints"].startswith("sha256:")


def test_model_hash_covers_all_directory_content_even_when_size_and_mtime_are_preserved(
    tmp_path: Path,
) -> None:
    import os

    module = _renderer_module()
    model = tmp_path / "model"
    shard_dir = model / "weights"
    shard_dir.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    shard = shard_dir / "model-00001-of-00002.safetensors"
    shard.write_bytes(b"AAAA")
    before_stat = shard.stat()
    before_hash = module._model_hash(model)

    shard.write_bytes(b"BBBB")
    os.utime(shard, ns=(before_stat.st_atime_ns, before_stat.st_mtime_ns))
    after_stat = shard.stat()
    after_hash = module._model_hash(model)

    assert after_stat.st_size == before_stat.st_size
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns
    assert after_hash != before_hash


def test_default_encoder_output_is_probed_as_exact_decodable_mp4(tmp_path: Path) -> None:
    from canonical_qc.video_probe import probe_video

    module = _renderer_module()
    provider = FakeFrameProvider()
    segmenter = RecordingSegmenter()
    renderer = module.Sam3OverlayRenderer(
        frame_provider=provider,
        runtime_provider=RecordingRuntimeProvider(segmenter),
        model_path=tmp_path / "model",
        runtime_config={"confidence_threshold": 0.5},
        queries=("hand",),
    )
    request = _request(tmp_path, renderer, interval=(0, 3))
    output = tmp_path / "real-overlay.mp4"

    metadata = renderer.render_interval(request, 0, 3, output)
    probed = probe_video(output)

    assert probed.frame_count == 3
    assert (probed.width_px, probed.height_px) == (6, 4)
    assert probed.fps_num / probed.fps_den == pytest.approx(30.0)
    assert probed.codec == "mpeg4"
    assert metadata["frame_count"] == 3
    assert metadata["width_px"] == 6
    assert metadata["height_px"] == 4
    assert metadata["codec"] == "mpeg4"


def test_renderer_rejects_bad_mp4_product_after_encoder_close(tmp_path: Path) -> None:
    module = _renderer_module()

    class JunkEncoder:
        def __init__(self, output_path: Path) -> None:
            self.output_path = output_path

        @staticmethod
        def write(_frame: np.ndarray) -> None:
            return None

        def close(self) -> None:
            self.output_path.write_bytes(b"not-an-mp4")

    renderer = module.Sam3OverlayRenderer(
        frame_provider=FakeFrameProvider(),
        runtime_provider=RecordingRuntimeProvider(RecordingSegmenter()),
        model_path=tmp_path / "model",
        runtime_config={},
        queries=("hand",),
        encoder_factory=lambda path, _fps, _size: JunkEncoder(path),
    )
    output = tmp_path / "bad-overlay.mp4"

    with pytest.raises(module.OverlayRenderError, match="overlay_encoder_failed"):
        renderer.render_interval(
            _request(tmp_path, renderer, interval=(0, 2)),
            0,
            2,
            output,
        )

    assert not output.exists()


def test_renderer_rejects_mpeg4_stream_inside_non_mp4_container(tmp_path: Path) -> None:
    import cv2

    module = _renderer_module()

    class AviNamedMp4Encoder:
        def __init__(self, output_path: Path, fps: float, size: tuple[int, int]) -> None:
            self.output_path = output_path
            self.avi_path = output_path.with_suffix(".avi")
            self.writer = cv2.VideoWriter(
                str(self.avi_path),
                cv2.VideoWriter_fourcc(*"XVID"),
                fps,
                size,
            )
            assert self.writer.isOpened()

        def write(self, frame: np.ndarray) -> None:
            self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        def close(self) -> None:
            self.writer.release()
            self.avi_path.replace(self.output_path)

    renderer = module.Sam3OverlayRenderer(
        frame_provider=FakeFrameProvider(),
        runtime_provider=RecordingRuntimeProvider(RecordingSegmenter()),
        model_path=tmp_path / "model",
        runtime_config={},
        queries=("hand",),
        encoder_factory=AviNamedMp4Encoder,
    )
    output = tmp_path / "avi-disguised-as-mp4.mp4"

    with pytest.raises(module.OverlayRenderError, match="overlay_encoder_failed"):
        renderer.render_interval(
            _request(tmp_path, renderer, interval=(0, 2)),
            0,
            2,
            output,
        )

    assert not output.exists()


@pytest.mark.parametrize(
    ("muxer", "forced_brand", "accepted", "expected_brand"),
    [
        pytest.param("mov", None, False, "qt  ", id="quicktime-mov-renamed-mp4"),
        pytest.param("3gp", None, False, "3gp4", id="3gp-renamed-mp4"),
        pytest.param("mp4", "M4A ", False, "M4A ", id="m4a-brand-with-video"),
        pytest.param("mp4", None, True, "isom", id="normal-mp4"),
    ],
)
def test_renderer_requires_an_mp4_compatible_major_brand_from_real_ffmpeg(
    tmp_path: Path,
    muxer: str,
    forced_brand: str | None,
    accepted: bool,
    expected_brand: str,
) -> None:
    module = _renderer_module()

    class FfmpegMuxEncoder:
        def __init__(self, output_path: Path, fps: float, size: tuple[int, int]) -> None:
            self.output_path = output_path
            self.fps = fps
            self.size = size
            self.frame_count = 0

        def write(self, _frame: np.ndarray) -> None:
            self.frame_count += 1

        def close(self) -> None:
            argv = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"color=c=black:s={self.size[0]}x{self.size[1]}:r={self.fps}",
                "-frames:v",
                str(self.frame_count),
                "-c:v",
                "mpeg4",
            ]
            if forced_brand is not None:
                argv.extend(("-brand", forced_brand))
            argv.extend(("-f", muxer, str(self.output_path)))
            subprocess.run(
                argv,
                shell=False,
                capture_output=True,
                check=True,
            )

    renderer = module.Sam3OverlayRenderer(
        frame_provider=FakeFrameProvider(),
        runtime_provider=RecordingRuntimeProvider(RecordingSegmenter()),
        model_path=tmp_path / "model",
        runtime_config={},
        queries=("hand",),
        encoder_factory=FfmpegMuxEncoder,
    )
    output = tmp_path / f"{muxer}-container-named.mp4"
    request = _request(tmp_path, renderer, interval=(0, 2))

    if not accepted:
        with pytest.raises(module.OverlayRenderError, match="overlay_encoder_failed"):
            renderer.render_interval(request, 0, 2, output)
        assert not output.exists()
        return

    metadata = renderer.render_interval(request, 0, 2, output)
    assert metadata["container_major_brand"] == expected_brand
    assert output.is_file()


def test_renderer_closes_frame_provider_after_each_bounded_interval(tmp_path: Path) -> None:
    class ClosableFrameProvider(FakeFrameProvider):
        def __init__(self) -> None:
            super().__init__()
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    provider = ClosableFrameProvider()
    renderer, *_ = _renderer(tmp_path, provider=provider)

    renderer.render_interval(
        _request(tmp_path, renderer, interval=(0, 2)),
        0,
        2,
        tmp_path / "bounded-provider.mp4",
    )

    assert provider.close_calls == 1


def test_malformed_frame_array_maps_to_stable_decode_failure(tmp_path: Path) -> None:
    module = _renderer_module()

    class MalformedFrameProvider(FakeFrameProvider):
        def read_frame(self, source_frame: int):
            return module.OverlayFrame(
                source_frame=source_frame,
                video_frame=source_frame,
                frame_rgb=[[[1, 2, 3]], [[4, 5]]],
                keypoints={"left": np.asarray([[1.0, 2.0]], dtype=np.float32)},
            )

    renderer, *_ = _renderer(tmp_path, provider=MalformedFrameProvider())

    with pytest.raises(module.OverlayRenderError, match="overlay_decode_failed"):
        renderer.render_interval(
            _request(tmp_path, renderer, interval=(0, 1)),
            0,
            1,
            tmp_path / "malformed.mp4",
        )


def test_historical_inline_recipe_is_rejected_from_frozen_report_context(tmp_path: Path) -> None:
    from qc_pipeline.artifacts import file_sha256
    from qc_pipeline.context import AssetContext

    module = _renderer_module()
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    recipe = {
        "schema_version": "sam3_overlay_input.v1",
        "video_source": "video",
        "video_identity": file_sha256(video),
        "source_to_video": {"0": 0},
        "keypoints_2d": {"0": {"left": [[1.0, 2.0]]}},
    }
    context = AssetContext(
        asset_id="asset-a",
        batch_root=tmp_path,
        report_path=tmp_path / "quality_archive" / "asset-a.json",
        source_files={"video": {"path": "video.mp4"}},
        source_range=(0, 1),
        metadata={"sam3_overlay_recipe": recipe},
    )
    source = SimpleNamespace(path=video, etag=file_sha256(video))

    with pytest.raises(module.OverlaySetupError, match="overlay_mapping_unavailable"):
        module.frame_provider_from_context(context, source)


def test_production_provider_never_exposes_ready_segments_from_a_failed_multi_interval_job(
    tmp_path: Path,
) -> None:
    import time

    module = _renderer_module()
    provider = FakeFrameProvider(fail_source_frame=125)
    renderer, *_ = _renderer(tmp_path, provider=provider)
    request = replace(
        _request(tmp_path, renderer, interval=(120, 122)),
        intervals=((120, 122), (124, 126)),
    )
    from human_qc.overlay_worker import BoundedOverlayWorker
    from human_qc.warn_workbench_service import FrameRangeDto, OverlayIssueInput

    selected = (
        OverlayIssueInput("issue-1", FrameRangeDto(120, 122)),
        OverlayIssueInput("issue-2", FrameRangeDto(124, 126)),
    )
    worker = BoundedOverlayWorker(max_workers=1, max_pending=0)
    production = module.ProductionWorkerOverlayProvider(
        worker=worker,
        request_factory=lambda _asset_id, _selected: request,
    )
    try:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            values = production.get_asset_overlays("asset-1", selected)
            if all(value.status in {"ready", "failed"} for value in values.values()):
                break
            time.sleep(0.01)
        else:
            pytest.fail("multi-interval job did not become terminal")
    finally:
        worker.shutdown()

    assert {value.status for value in values.values()} == {"failed"}
    assert {value.code for value in values.values()} == {"overlay_decode_failed"}
    assert all(
        not value.segments
        or all(segment.path is None for segment in value.segments)
        for value in values.values()
    )


def test_production_provider_retries_the_same_asset_union_request_once(
    tmp_path: Path,
) -> None:
    module = _renderer_module()
    renderer, *_ = _renderer(tmp_path)
    request = replace(
        _request(tmp_path, renderer, interval=(120, 169)),
        intervals=((120, 169), (390, 427)),
    )
    from human_qc.overlay_worker import OverlayJobView, OverlaySegmentView
    from human_qc.warn_workbench_service import FrameRangeDto, OverlayIssueInput

    class FakeWorker:
        def __init__(self) -> None:
            self.retry_requests: list[object] = []
            self.failed = OverlayJobView(
                request.cache_key,
                "failed",
                (
                    OverlaySegmentView(120, 169, "failed", "opaque-a", code="overlay_render_failed", retryable=True),
                    OverlaySegmentView(390, 427, "failed", "opaque-b", code="overlay_render_failed", retryable=True),
                ),
                code="overlay_render_failed",
                retryable=True,
            )
            self.pending = OverlayJobView(
                request.cache_key,
                "pending",
                (
                    OverlaySegmentView(120, 169, "pending", "opaque-a"),
                    OverlaySegmentView(390, 427, "pending", "opaque-b"),
                ),
            )

        def submit(self, value):
            assert value is request
            return self.failed

        def get(self, value):
            assert value is request
            return self.failed

        def retry(self, value):
            self.retry_requests.append(value)
            return self.pending

    worker = FakeWorker()
    selected = (
        OverlayIssueInput("issue-a", FrameRangeDto(120, 169)),
        OverlayIssueInput("issue-b", FrameRangeDto(390, 427)),
    )
    production = module.ProductionWorkerOverlayProvider(
        worker=worker,
        request_factory=lambda asset_id, actual_selected: (
            request if asset_id == "asset-1" and actual_selected == selected else None
        ),
    )

    values = production.retry_asset_overlays("asset-1", selected)

    assert worker.retry_requests == [request]
    assert {issue_id: value.status for issue_id, value in values.items()} == {
        "issue-a": "pending",
        "issue-b": "pending",
    }


def test_ready_overlay_url_is_pinned_until_catalog_lease_expiry_across_asset_eviction(
    tmp_path: Path,
) -> None:
    import time

    from human_qc.media import MediaCatalog, MediaNotFoundError
    from human_qc.overlay_worker import BoundedOverlayWorker
    from human_qc.warn_workbench_service import FrameRangeDto, OverlayIssueInput
    from qc_pipeline.context import AssetContext

    module = _renderer_module()
    contexts = {
        asset_id: AssetContext(
            asset_id=asset_id,
            batch_root=tmp_path,
            report_path=tmp_path / "quality_archive" / f"{asset_id}.json",
            source_files={},
        )
        for asset_id in ("asset-a", "asset-b")
    }
    catalog = MediaCatalog(contexts)
    shared_root = tmp_path / ".human_qc" / "overlay-cache"
    renderer_a, *_ = _renderer(tmp_path)
    renderer_b, *_ = _renderer(tmp_path)
    requests = {
        "asset-a": replace(
            _request(tmp_path, renderer_a, interval=(0, 1)),
            asset_id="asset-a",
            cache_root=shared_root,
            source_sha256="sha256:" + "1" * 64,
        ),
        "asset-b": replace(
            _request(tmp_path, renderer_b, interval=(0, 1)),
            asset_id="asset-b",
            cache_root=shared_root,
            source_sha256="sha256:" + "2" * 64,
        ),
    }
    worker = BoundedOverlayWorker(
        max_workers=1,
        max_pending=1,
        max_ready_jobs=1,
        pin_lease_seconds=0.08,
    )
    production = module.ProductionWorkerOverlayProvider(
        worker=worker,
        request_factory=lambda asset_id, _selected: requests[asset_id],
        media_catalog=catalog,
        pin_lease_seconds=0.08,
    )
    selected = (OverlayIssueInput("issue", FrameRangeDto(0, 1)),)
    try:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            ready_a = production.get_asset_overlays("asset-a", selected)["issue"]
            if ready_a.status == "ready":
                break
            time.sleep(0.01)
        else:
            pytest.fail("asset-a overlay did not become ready")
        overlay_id = ready_a.segments[0].overlay_id
        assert overlay_id is not None
        assert catalog.overlay("asset-a", overlay_id).path.is_file()

        production.get_asset_overlays("asset-b", selected)
        blocked_b = _wait(worker, requests["asset-b"])
        assert blocked_b.code == "overlay_cache_full"
        assert catalog.overlay("asset-a", overlay_id).path.is_file()

        time.sleep(0.11)
        worker.retry(requests["asset-b"])
        assert _wait(worker, requests["asset-b"]).status == "ready"
        with pytest.raises(MediaNotFoundError):
            catalog.overlay("asset-a", overlay_id)
    finally:
        production.release_all()
        worker.shutdown()


def test_production_cache_quota_is_shared_across_assets_and_keeps_asset_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qc_pipeline.context import AssetContext
    from human_qc.warn_workbench_service import FrameRangeDto, OverlayIssueInput

    module = _renderer_module()
    contexts = {
        asset_id: AssetContext(
            asset_id=asset_id,
            batch_root=tmp_path,
            report_path=tmp_path / "quality_archive" / f"{asset_id}.json",
            source_files={},
            source_range=(0, 1),
            metadata={},
        )
        for asset_id in ("asset-a", "asset-b")
    }

    class Catalog:
        @staticmethod
        def source(_asset_id: str):
            return SimpleNamespace(
                etag="sha256:" + "a" * 64,
                fps=10.0,
                total_frames=1,
            )

    rendered: list[tuple[str, Path]] = []

    class LargeRenderer:
        renderer_version = "shared-quota-test-v1"

        def __init__(self, *, frame_provider: object, **_kwargs: object) -> None:
            self.frame_provider = frame_provider

        def render_interval(
            self,
            request: object,
            _start: int,
            _end: int,
            output_path: Path,
        ) -> dict[str, object]:
            rendered.append((request.asset_id, request.cache_root))
            output_path.write_bytes(b"x" * (1024 * 1024))
            return {"frame_count": 1, "fps": 10.0}

    monkeypatch.setattr(module, "Sam3OverlayRenderer", LargeRenderer)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    runtime = module.build_production_overlay_runtime(
        contexts=contexts,
        media_catalog=Catalog(),
        model_path=model,
        max_cache_bytes=1024 * 1024 + 64 * 1024,
        frame_provider_factory=lambda _context, _source: FakeFrameProvider(),
    )
    selected = (OverlayIssueInput("issue", FrameRangeDto(0, 1)),)

    def wait_ready(asset_id: str) -> None:
        import time

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            view = runtime.provider.get_asset_overlays(asset_id, selected)["issue"]
            if view.status == "ready":
                return
            if view.status == "failed":
                pytest.fail(f"unexpected failed overlay: {view.code}")
            time.sleep(0.01)
        pytest.fail("overlay did not become ready")

    try:
        wait_ready("asset-a")
        wait_ready("asset-b")
    finally:
        runtime.shutdown()

    shared_root = tmp_path / ".human_qc" / "overlay-cache"
    assert rendered == [
        ("asset-a", shared_root.resolve()),
        ("asset-b", shared_root.resolve()),
    ]
    ready_manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in shared_root.glob("*/manifest.json")
        if json.loads(path.read_text(encoding="utf-8")).get("status") == "ready"
    ]
    assert len(ready_manifests) == 1
