"""Process-local SAM3 lifecycle owned by one threaded QC batch run."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from qc_pipeline.artifacts import canonical_json


SegmenterFactory = Callable[[Path, dict[str, Any]], Any]
SegmenterProvider = Callable[[Path, Mapping[str, Any]], Any]


@dataclass(frozen=True)
class Sam3RuntimeCacheKey:
    resolved_model_path: str
    device: str
    dtype: str
    canonical_runtime_config: str


class _InferenceLockedSegmenter:
    def __init__(self, segmenter: Any, inference_lock: Lock) -> None:
        self._segmenter = segmenter
        self._inference_lock = inference_lock

    def segment_frame(self, *args: Any, **kwargs: Any) -> Any:
        with self._inference_lock:
            return self._segmenter.segment_frame(*args, **kwargs)


def _default_factory(model_path: Path, runtime_config: dict[str, Any]) -> Any:
    from tools.sam3_keypoint_containment import create_sam3_segmenter

    return create_sam3_segmenter(model_path, runtime_config)


class Sam3RuntimeProvider:
    """Lazily cache keyed segmenters and serialize their inference calls."""

    def __init__(self, *, factory: SegmenterFactory | None = None) -> None:
        self._factory = factory or _default_factory
        self._cache: dict[Sam3RuntimeCacheKey, _InferenceLockedSegmenter] = {}
        self._initialization_lock = Lock()
        self._inference_lock = Lock()

    @staticmethod
    def cache_key(
        model_path: Path,
        runtime_config: Mapping[str, Any],
    ) -> Sam3RuntimeCacheKey:
        config = deepcopy(dict(runtime_config))
        return Sam3RuntimeCacheKey(
            resolved_model_path=str(Path(model_path).expanduser().resolve()),
            device=canonical_json(config.get("device", "auto")),
            dtype=canonical_json(config.get("dtype", "default")),
            canonical_runtime_config=canonical_json(config),
        )

    def get_segmenter(
        self,
        model_path: Path,
        runtime_config: Mapping[str, Any],
    ) -> Any:
        config = deepcopy(dict(runtime_config))
        key = self.cache_key(model_path, config)
        with self._initialization_lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            segmenter = self._factory(Path(key.resolved_model_path), config)
            wrapped = _InferenceLockedSegmenter(segmenter, self._inference_lock)
            self._cache[key] = wrapped
            return wrapped


__all__ = [
    "Sam3RuntimeCacheKey",
    "Sam3RuntimeProvider",
    "SegmenterFactory",
    "SegmenterProvider",
]
