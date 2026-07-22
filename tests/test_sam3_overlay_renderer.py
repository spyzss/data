from __future__ import annotations

from dataclasses import replace
import importlib
from pathlib import Path
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
