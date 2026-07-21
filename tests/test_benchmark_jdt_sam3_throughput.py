from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


class _BatchFeature(dict):
    def to(self, device: str):
        del device
        return self

    def __getattr__(self, name: str):
        return self[name]


class _FakeProcessor:
    def __init__(self, resolution: int, *, center_masks: bool = False) -> None:
        self.image_processor = SimpleNamespace(
            size={"height": resolution, "width": resolution}
        )
        self.target_size = resolution
        self.center_masks = center_masks
        self.image_calls: list[tuple[int, int, int]] = []
        self.text_calls = 0

    def __call__(self, *, images=None, text=None, return_tensors=None):
        assert return_tensors == "pt"
        if images is not None:
            batch = np.stack(images).transpose(0, 3, 1, 2)
            self.image_calls.append(
                (int(batch.shape[0]), int(batch.shape[-2]), int(batch.shape[-1]))
            )
            return _BatchFeature(
                pixel_values=batch,
                original_sizes=np.asarray(
                    [[image.shape[0], image.shape[1]] for image in images],
                    dtype=np.int64,
                ),
            )
        assert text is not None
        prompts = [text] if isinstance(text, str) else list(text)
        self.text_calls += 1
        return _BatchFeature(
            input_ids=np.ones((len(prompts), 4), dtype=np.int64),
            attention_mask=np.ones((len(prompts), 4), dtype=np.int64),
        )

    def post_process_instance_segmentation(
        self,
        outputs,
        *,
        threshold,
        mask_threshold,
        target_sizes,
    ):
        del threshold, mask_threshold
        results = []
        for height, width in target_sizes:
            mask = np.zeros((1, int(height), int(width)), dtype=bool)
            if self.center_masks:
                mask[0, int(height) // 2, int(width) // 2] = True
            else:
                mask[:] = True
            results.append(
                {
                    "masks": mask,
                    "boxes": np.asarray([[0, 0, width, height]], dtype=np.float64),
                    "scores": np.asarray([0.99], dtype=np.float64),
                }
            )
        assert len(results) == outputs.pair_count
        return results


class _FakeModel:
    def __init__(self, resolution: int) -> None:
        self.device = "cuda"
        self.config = SimpleNamespace(
            image_size=resolution,
            vision_config=SimpleNamespace(image_size=resolution),
        )
        self.vision_forward_calls = 0
        self.model_forward_calls = 0
        self.text_feature_calls = 0
        self.vision_batch_sizes: list[int] = []
        self.detector_pair_batch_sizes: list[int] = []

    def get_vision_features(self, *, pixel_values):
        self.vision_forward_calls += 1
        self.vision_batch_sizes.append(int(pixel_values.shape[0]))
        return {"last_hidden_state": np.asarray(pixel_values[:, :1, :1, :1])}

    def get_text_features(self, *, input_ids, attention_mask):
        del attention_mask
        self.text_feature_calls += 1
        prompt_count = int(input_ids.shape[0])
        return SimpleNamespace(
            pooler_output=np.ones((prompt_count, 4, 3), dtype=np.float32)
        )

    def __call__(self, *, vision_embeds, text_embeds, attention_mask):
        del attention_mask
        pair_count = int(text_embeds.shape[0])
        assert int(vision_embeds["last_hidden_state"].shape[0]) == pair_count
        self.model_forward_calls += 1
        self.detector_pair_batch_sizes.append(pair_count)
        return SimpleNamespace(pair_count=pair_count)


class _TaggedProcessor(_FakeProcessor):
    def __call__(self, *, images=None, text=None, return_tensors=None):
        if images is not None:
            return super().__call__(
                images=images, text=text, return_tensors=return_tensors
            )
        assert return_tensors == "pt"
        prompts = [text] if isinstance(text, str) else list(text)
        self.text_calls += 1
        prompt_ids = np.arange(len(prompts), dtype=np.int64)[:, None]
        return _BatchFeature(
            input_ids=np.repeat(prompt_ids, 4, axis=1),
            attention_mask=np.repeat(prompt_ids + 1, 4, axis=1),
        )


class _TaggedModelBase(_FakeModel):
    def __init__(self, resolution: int, *, output_kind: str) -> None:
        super().__init__(resolution)
        self.output_kind = output_kind
        self.pair_orders: list[list[tuple[int, int]]] = []
        self.attention_orders: list[list[int]] = []

    def get_text_features(self, *, input_ids, attention_mask):
        del attention_mask
        self.text_feature_calls += 1
        pooled = np.asarray(input_ids[:, :1, None], dtype=np.float32)
        if self.output_kind == "model_output":
            return SimpleNamespace(pooler_output=pooled)
        return pooled

    def _record_forward(self, *, vision_embeds, pooled, attention_mask):
        vision_values = np.asarray(
            vision_embeds["last_hidden_state"][:, 0, 0, 0], dtype=np.int64
        )
        prompt_values = np.asarray(pooled[:, 0, 0], dtype=np.int64)
        attention_values = np.asarray(attention_mask[:, 0], dtype=np.int64)
        assert len(vision_values) == len(prompt_values) == len(attention_values)
        self.pair_orders.append(list(zip(vision_values.tolist(), prompt_values.tolist())))
        self.attention_orders.append(attention_values.tolist())
        self.model_forward_calls += 1
        self.detector_pair_batch_sizes.append(len(prompt_values))
        return SimpleNamespace(pair_count=len(prompt_values))

    def __call__(self, *, vision_embeds, text_embeds, attention_mask):
        return self.forward(
            vision_embeds=vision_embeds,
            text_embeds=text_embeds,
            attention_mask=attention_mask,
        )


class _TensorContractModel(_TaggedModelBase):
    def forward(self, *, vision_embeds, text_embeds, attention_mask):
        return self._record_forward(
            vision_embeds=vision_embeds,
            pooled=text_embeds,
            attention_mask=attention_mask,
        )


class _ModelOutputContractModel(_TaggedModelBase):
    def forward(self, *, vision_embeds, text_embeds, attention_mask):
        return self._record_forward(
            vision_embeds=vision_embeds,
            pooled=text_embeds.pooler_output,
            attention_mask=attention_mask,
        )


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def reset_peak_memory_stats() -> None:
        return None

    @staticmethod
    def max_memory_allocated() -> int:
        return 1234

    @staticmethod
    def synchronize() -> None:
        return None

    @staticmethod
    def empty_cache() -> None:
        return None


class _FakeTorch:
    cuda = _FakeCuda()

    @staticmethod
    def no_grad():
        return nullcontext()


class _FakeSequentialSource:
    def __init__(self, frame_count: int) -> None:
        self.frame_count = frame_count
        self.iter_frames_calls = 0
        self.requested_frames: list[int] = []
        self.closed = False

    def read_parquet(self, path: Path) -> pd.DataFrame:
        return pd.read_parquet(path)

    def iter_frames(self, path: Path, video_frames: list[int]):
        del path
        self.iter_frames_calls += 1
        for frame in video_frames[: self.frame_count]:
            self.requested_frames.append(frame)
            yield frame, np.zeros((64, 96, 3), dtype=np.uint8), 0.01

    def close(self) -> None:
        self.closed = True


def _points(x: float = 48.0, y: float = 32.0) -> list[float]:
    return np.tile(np.asarray([[x, y]], dtype=np.float64), (21, 1)).reshape(-1).tolist()


def _write_inputs(tmp_path: Path, frame_count: int = 5) -> Path:
    video = tmp_path / "episode.mp4"
    video.write_bytes(b"video-placeholder")
    parquet = tmp_path / "episode.parquet"
    pd.DataFrame(
        [
            {
                "leftcam_left_kp2d": _points(),
                "leftcam_right_kp2d": _points(),
            }
            for _ in range(frame_count)
        ]
    ).to_parquet(parquet, index=False)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "asset_id": "jdt__episode_000001",
                "supplier": "jdt",
                "episode_index": 1,
                "start_frame": 0,
                "end_frame": frame_count - 1,
                "primary_video_path": str(video),
                "parquet_path": str(parquet),
                "left_hand_2d_field": "leftcam_left_kp2d",
                "right_hand_2d_field": "leftcam_right_kp2d",
            }
        ]
    ).to_csv(manifest, index=False)
    return manifest


def _backend(resolution: int, *, center_masks: bool = False):
    from tools.benchmark_jdt_sam3_throughput import HuggingFaceSam3BatchBackend

    model = _FakeModel(resolution)
    processor = _FakeProcessor(resolution, center_masks=center_masks)
    backend = HuggingFaceSam3BatchBackend(
        model=model,
        processor=processor,
        torch_module=_FakeTorch(),
        device="cuda",
        requested_resolution=None,
        effective_resolution=(resolution, resolution),
    )
    return backend, model, processor


def _contract_backend(
    resolution: int,
    *,
    forward_contract: str,
    output_kind: str,
):
    from tools.benchmark_jdt_sam3_throughput import HuggingFaceSam3BatchBackend

    model_type = (
        _ModelOutputContractModel
        if forward_contract == "model_output"
        else _TensorContractModel
    )
    model = model_type(resolution, output_kind=output_kind)
    processor = _TaggedProcessor(resolution)
    backend = HuggingFaceSam3BatchBackend(
        model=model,
        processor=processor,
        torch_module=_FakeTorch(),
        device="cuda",
        requested_resolution=None,
        effective_resolution=(resolution, resolution),
        model_output_factory=SimpleNamespace,
        transformers_version="test-transformers",
    )
    return backend, model, processor


def test_huggingface_backend_uses_one_true_batch_forward_and_caches_queries() -> None:
    backend, model, processor = _backend(32)
    frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(3)]
    queries = ["hand", "left hand", "right hand"]

    first = backend.segment_batch(frames, queries, {"confidence_threshold": 0.5})
    second = backend.segment_batch(frames[:2], queries, {"confidence_threshold": 0.5})

    assert len(first.masks_by_frame) == 3
    assert len(second.masks_by_frame) == 2
    assert model.vision_forward_calls == 2
    assert model.model_forward_calls == 2
    assert model.vision_batch_sizes == [3, 2]
    assert model.detector_pair_batch_sizes == [9, 6]
    assert model.text_feature_calls == 1
    assert processor.text_calls == 1
    assert processor.image_calls == [(3, 32, 32), (2, 32, 32)]


def test_model_output_forward_contract_preserves_pooler_container() -> None:
    backend, model, processor = _contract_backend(
        32,
        forward_contract="model_output",
        output_kind="model_output",
    )
    frames = [
        np.full((32, 32, 3), fill_value=value, dtype=np.uint8)
        for value in (10, 20)
    ]

    backend.segment_batch(frames, ["q0", "q1", "q2"], {})

    assert backend.text_embedding_forward_contract == "base_model_output_with_pooling"
    assert model.pair_orders == [
        [(10, 0), (10, 1), (10, 2), (20, 0), (20, 1), (20, 2)]
    ]
    assert model.attention_orders == [[1, 2, 3, 1, 2, 3]]
    assert model.text_feature_calls == 1
    assert processor.text_calls == 1


def test_model_output_forward_contract_can_wrap_tensor_feature_output() -> None:
    backend, model, _processor = _contract_backend(
        32,
        forward_contract="model_output",
        output_kind="tensor",
    )

    backend.segment_batch(
        [np.zeros((32, 32, 3), dtype=np.uint8)],
        ["q0", "q1"],
        {},
    )

    assert model.pair_orders == [[(0, 0), (0, 1)]]
    assert model.text_feature_calls == 1


def test_tensor_forward_contract_accepts_pooled_tensor() -> None:
    backend, model, _processor = _contract_backend(
        32,
        forward_contract="tensor",
        output_kind="model_output",
    )

    backend.segment_batch(
        [np.zeros((32, 32, 3), dtype=np.uint8)],
        ["q0", "q1"],
        {},
    )

    assert backend.text_embedding_forward_contract == "tensor"
    assert model.pair_orders == [[(0, 0), (0, 1)]]


def test_batch_sizes_preserve_frame_major_query_and_attention_order() -> None:
    queries = ["q0", "q1", "q2"]
    for batch_size in (1, 2, 4):
        backend, model, _processor = _contract_backend(
            32,
            forward_contract="model_output",
            output_kind="model_output",
        )
        frames = [
            np.full((32, 32, 3), fill_value=index + 10, dtype=np.uint8)
            for index in range(batch_size)
        ]

        backend.segment_batch(frames, queries, {})

        expected_pairs = [
            (frame_index + 10, query_index)
            for frame_index in range(batch_size)
            for query_index in range(len(queries))
        ]
        assert model.pair_orders == [expected_pairs]
        assert model.attention_orders == [
            [query_index + 1 for _frame in frames for query_index in range(3)]
        ]
        assert backend.expanded_attention_mask_shapes == [
            (batch_size * len(queries), 4)
        ]
        assert model.model_forward_calls == 1
        assert model.text_feature_calls == 1


def test_benchmark_batches_forward_calls_once_per_batch_and_decodes_once(
    tmp_path: Path,
) -> None:
    from tools.benchmark_jdt_sam3_throughput import benchmark_jdt_sam3_throughput

    manifest = _write_inputs(tmp_path, frame_count=5)
    source = _FakeSequentialSource(frame_count=5)
    created: dict[str, tuple[object, _FakeModel, _FakeProcessor]] = {}

    def factory(model_path: Path, requested_resolution: int | None):
        del model_path
        resolution = 32 if requested_resolution is None else requested_resolution
        created[str(requested_resolution)] = _backend(resolution)
        return created[str(requested_resolution)][0]

    summary = benchmark_jdt_sam3_throughput(
        manifest=manifest,
        asset_id="jdt__episode_000001",
        sam3_model=tmp_path / "model",
        config_path=None,
        output_dir=tmp_path / "benchmark",
        max_frames=5,
        warmup_frames=0,
        batch_sizes=(1, 2, 4),
        input_resolutions=("baseline", "16"),
        source_reader=source,
        backend_factory=factory,
    )

    baseline_model = created["None"][1]
    reduced_model = created["16"][1]
    expected_source_batches = [1, 1, 1, 1, 1, 2, 2, 1, 4, 1]
    assert baseline_model.vision_batch_sizes == expected_source_batches
    assert reduced_model.vision_batch_sizes == expected_source_batches
    assert baseline_model.model_forward_calls == 5 + 3 + 2
    assert reduced_model.model_forward_calls == 5 + 3 + 2
    assert source.iter_frames_calls == 1
    assert source.requested_frames == [0, 1, 2, 3, 4]
    assert source.closed
    assert summary["configuration_count"] == 6
    results = pd.read_csv(tmp_path / "benchmark" / "benchmark_results.csv")
    assert set(results.loc[results["batch_size"] == 1, "model_forward_call_count"]) == {
        5
    }
    assert set(results.loc[results["batch_size"] == 2, "model_forward_call_count"]) == {
        3
    }
    assert set(results.loc[results["batch_size"] == 4, "model_forward_call_count"]) == {
        2
    }


def test_resolution_scales_frames_and_keypoints_into_same_mask_space(
    tmp_path: Path,
) -> None:
    from tools.benchmark_jdt_sam3_throughput import benchmark_jdt_sam3_throughput

    manifest = _write_inputs(tmp_path, frame_count=2)

    def factory(model_path: Path, requested_resolution: int | None):
        del model_path
        resolution = 32 if requested_resolution is None else requested_resolution
        return _backend(resolution, center_masks=True)[0]

    benchmark_jdt_sam3_throughput(
        manifest=manifest,
        asset_id="jdt__episode_000001",
        sam3_model=tmp_path / "model",
        config_path=None,
        output_dir=tmp_path / "scaled",
        max_frames=2,
        warmup_frames=0,
        batch_sizes=(1,),
        input_resolutions=("baseline", "16"),
        source_reader=_FakeSequentialSource(frame_count=2),
        backend_factory=factory,
    )

    details = pd.read_parquet(tmp_path / "scaled" / "comparison_details.parquet")
    assert set(details["current_inside_ratio"]) == {1.0}
    assert set(details["current_hand_verdict"]) == {"pass"}
    reduced = details[details["requested_input_resolution"] == "16"]
    assert set(reduced["effective_input_height"]) == {16}
    assert set(reduced["effective_input_width"]) == {16}


def test_outputs_record_actual_resolution_agreement_timing_and_coverage(
    tmp_path: Path,
) -> None:
    from tools.benchmark_jdt_sam3_throughput import benchmark_jdt_sam3_throughput

    manifest = _write_inputs(tmp_path, frame_count=3)

    def factory(model_path: Path, requested_resolution: int | None):
        del model_path
        resolution = 36 if requested_resolution is None else requested_resolution
        return _backend(resolution)[0]

    benchmark_jdt_sam3_throughput(
        manifest=manifest,
        asset_id="jdt__episode_000001",
        sam3_model=tmp_path / "model",
        config_path=None,
        output_dir=tmp_path / "outputs",
        max_frames=3,
        warmup_frames=0,
        batch_sizes=(1, 2),
        input_resolutions=("baseline", "18"),
        source_reader=_FakeSequentialSource(frame_count=3),
        backend_factory=factory,
    )

    output = tmp_path / "outputs"
    assert {
        "benchmark_results.csv",
        "benchmark_results.json",
        "baseline_frame_results.parquet",
        "comparison_details.parquet",
        "run_config.json",
    } <= {path.name for path in output.iterdir()}
    results = pd.read_csv(output / "benchmark_results.csv")
    required = {
        "batch_size",
        "requested_input_resolution",
        "effective_input_height",
        "effective_input_width",
        "processed_frame_count",
        "warmup_frame_count",
        "decode_seconds",
        "inference_seconds",
        "postprocess_seconds",
        "total_seconds",
        "inference_frames_per_second",
        "end_to_end_frames_per_second",
        "peak_accelerator_memory_bytes",
        "fail_count",
        "review_count",
        "pass_count",
        "frame_verdict_agreement",
        "per_hand_verdict_agreement",
        "inside_ratio_mean_absolute_difference",
        "inside_ratio_max_absolute_difference",
        "source_frame_order_valid",
        "source_frame_coverage_valid",
    }
    assert required <= set(results.columns)
    baseline = results[
        (results["batch_size"] == 1)
        & (results["requested_input_resolution"] == "baseline")
    ].iloc[0]
    assert int(baseline["effective_input_height"]) == 36
    assert int(baseline["effective_input_width"]) == 36
    assert baseline["frame_verdict_agreement"] == 1.0
    assert baseline["per_hand_verdict_agreement"] == 1.0
    assert bool(results["source_frame_order_valid"].all())
    assert bool(results["source_frame_coverage_valid"].all())
    assert set(results["processed_frame_count"]) == {3}
    assert set(results["pass_count"]) == {3}
    assert not list(output.rglob("*.png"))
    assert not list(output.rglob("*.npy"))

    baseline_frames = pd.read_parquet(output / "baseline_frame_results.parquet")
    assert baseline_frames["source_frame"].tolist() == [0, 1, 2]
    details = pd.read_parquet(output / "comparison_details.parquet")
    assert len(details) == 4 * 3 * 2
    run_config = json.loads((output / "run_config.json").read_text())
    assert run_config["baseline"]["effective_input_resolution"] == [36, 36]
    assert run_config["model_lifecycle"]["precision"] == "unchanged"
    assert run_config["artifacts"]["raw_masks_saved"] is False
    assert run_config["artifacts"]["overlays_saved"] is False
    assert run_config["transformers_version"] == "unavailable"
    assert run_config["text_embedding_output_type"] == "types.SimpleNamespace"
    assert run_config["text_embedding_forward_contract"] == "tensor"
    assert run_config["text_embedding_pooler_shape"] == [5, 4, 3]
    assert run_config["expanded_attention_mask_shape"]["baseline/batch_2"] == [
        [10, 4],
        [5, 4],
    ]


def test_cli_supports_required_benchmark_matrix() -> None:
    from tools.benchmark_jdt_sam3_throughput import build_parser

    args = build_parser().parse_args(
        [
            "--manifest",
            "manifest.csv",
            "--asset-id",
            "jdt__episode_000001",
            "--sam3-model",
            "/model",
            "--config",
            "configs/qc_acceptance.yaml",
            "--output-dir",
            "/tmp/benchmark",
            "--max-frames",
            "100",
            "--warmup-frames",
            "5",
            "--batch-sizes",
            "1",
            "2",
            "4",
            "8",
            "--input-resolutions",
            "baseline",
            "768",
            "640",
            "--decode-workers",
            "4",
            "--prefetch-frames",
            "32",
        ]
    )

    assert args.batch_sizes == [1, 2, 4, 8]
    assert args.input_resolutions == ["baseline", "768", "640"]
    assert args.decode_workers == 4
    assert args.prefetch_frames == 32
