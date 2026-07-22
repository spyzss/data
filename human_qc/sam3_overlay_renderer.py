"""Strict server-side SAM3 rendering for continuous Warn overlay jobs.

The generic worker owns scheduling and atomic publication.  This module owns
only production input proof, locked SAM3 inference, frame composition, and MP4
encoding.  It deliberately refuses to infer a source-to-video mapping from
frame numbers: a report must carry a versioned explicit recipe or the public
result is a stable unavailable code.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from threading import RLock
from typing import Any, Protocol

import numpy as np

from qc_pipeline.artifacts import canonical_sha256, file_sha256
from qc_pipeline.context import AssetContext
from qc_pipeline.sam3_runtime import Sam3RuntimeProvider

from .media import MediaCatalog, MediaError, SourceMedia
from .overlay_worker import (
    BoundedOverlayWorker,
    OverlayRenderError,
    OverlayRequest,
)
from .warn_workbench_service import (
    OverlayHandle,
    OverlayIssueInput,
    WorkerOverlayProvider,
)


RENDERER_VERSION = "sam3-overlay-renderer-v1"
RECIPE_SCHEMA = "sam3_overlay_input.v1"
REFERENCE_SCHEMA = "parquet_columns.v1"
DEFAULT_OVERLAY_MAX_CACHE_BYTES = 2 * 1024 * 1024 * 1024
PUBLIC_OVERLAY_FAILURE_CODES = frozenset(
    {
        "overlay_model_unavailable",
        "overlay_source_unavailable",
        "overlay_mapping_unavailable",
        "overlay_input_unavailable",
        "overlay_decode_failed",
        "overlay_inference_failed",
        "overlay_encoder_failed",
        "overlay_render_failed",
    }
)


class OverlaySetupError(RuntimeError):
    """A public-safe failure discovered before or during request creation."""

    def __init__(self, code: str) -> None:
        if code not in PUBLIC_OVERLAY_FAILURE_CODES:
            code = "overlay_input_unavailable"
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class OverlayFrame:
    """One source frame whose physical video mapping is already proven."""

    source_frame: int
    video_frame: int
    frame_rgb: np.ndarray
    keypoints: Mapping[str, np.ndarray]


class StrictFrameProvider(Protocol):
    @property
    def input_identity(self) -> Mapping[str, object]: ...

    def read_frame(self, source_frame: int) -> OverlayFrame: ...


class OverlayEncoder(Protocol):
    def write(self, frame: np.ndarray) -> None: ...

    def close(self) -> None: ...


EncoderFactory = Callable[[Path, float, tuple[int, int]], OverlayEncoder]
FrameComposer = Callable[[OverlayFrame, Sequence[object]], np.ndarray]
FrameProviderFactory = Callable[[AssetContext, SourceMedia], StrictFrameProvider]
MediaProbe = Callable[[Path], object]


def _finite_int(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise OverlaySetupError("overlay_mapping_unavailable")
    return value


def _non_empty_identity(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not value:
        raise OverlaySetupError("overlay_mapping_unavailable")
    mapping = value.get("mapping")
    if not isinstance(mapping, str) or not mapping.strip():
        raise OverlaySetupError("overlay_mapping_unavailable")
    try:
        canonical_sha256(dict(value))
    except (TypeError, ValueError):
        raise OverlaySetupError("overlay_mapping_unavailable") from None
    return value


def _plain_identity(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_identity(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain_identity(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise OverlaySetupError("overlay_input_unavailable")


def _frame_array(value: object) -> np.ndarray:
    try:
        frame = np.asarray(value)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            raise OverlaySetupError("overlay_decode_failed")
        if frame.dtype != np.uint8:
            if not np.issubdtype(frame.dtype, np.number) or not np.isfinite(frame).all():
                raise OverlaySetupError("overlay_decode_failed")
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(frame)
    except OverlaySetupError:
        raise
    except Exception:
        raise OverlaySetupError("overlay_decode_failed") from None


class _Cv2Mp4Encoder:
    def __init__(self, output_path: Path, fps: float, frame_size: tuple[int, int]) -> None:
        import cv2

        self._size = frame_size
        self._writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            frame_size,
        )
        if not self._writer.isOpened():
            self._writer.release()
            raise OverlaySetupError("overlay_encoder_failed")

    def write(self, frame: np.ndarray) -> None:
        import cv2

        rgb = _frame_array(frame)
        if (int(rgb.shape[1]), int(rgb.shape[0])) != self._size:
            raise OverlaySetupError("overlay_encoder_failed")
        self._writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        self._writer.release()


def _default_encoder_factory(
    output_path: Path,
    fps: float,
    frame_size: tuple[int, int],
) -> OverlayEncoder:
    return _Cv2Mp4Encoder(output_path, fps, frame_size)


def _default_media_probe(path: Path) -> object:
    from canonical_qc.video_probe import probe_video

    return probe_video(path)


def _mask_array(value: object, shape: tuple[int, int]) -> np.ndarray | None:
    raw = (
        value.get("mask")
        if isinstance(value, Mapping)
        else getattr(value, "mask", None)
    )
    if raw is None:
        return None
    mask = np.asarray(raw)
    if mask.shape != shape:
        raise OverlaySetupError("overlay_inference_failed")
    return mask.astype(bool, copy=False)


def _default_frame_composer(
    sample: OverlayFrame,
    masks: Sequence[object],
) -> np.ndarray:
    import cv2

    frame = _frame_array(sample.frame_rgb).copy()
    union = np.zeros(frame.shape[:2], dtype=bool)
    usable = False
    for item in masks:
        mask = _mask_array(item, frame.shape[:2])
        if mask is None:
            continue
        usable = True
        union |= mask
    if not usable or not union.any():
        raise OverlaySetupError("overlay_inference_failed")
    tint = np.zeros_like(frame)
    tint[..., 0] = 235
    tint[..., 2] = 210
    frame[union] = (
        frame[union].astype(np.float32) * 0.55
        + tint[union].astype(np.float32) * 0.45
    ).astype(np.uint8)
    colors = {"left": (40, 220, 90), "right": (40, 130, 255)}
    for side, raw_points in sample.keypoints.items():
        points = np.asarray(raw_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise OverlaySetupError("overlay_input_unavailable")
        color = colors.get(str(side), (255, 255, 255))
        for x_raw, y_raw in points:
            if not math.isfinite(float(x_raw)) or not math.isfinite(float(y_raw)):
                continue
            x, y = int(round(float(x_raw))), int(round(float(y_raw)))
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                cv2.circle(frame, (x, y), 3, color, -1, cv2.LINE_AA)
    return frame


class Sam3OverlayRenderer:
    """Render exactly one worker-owned half-open source interval."""

    renderer_version = RENDERER_VERSION

    def __init__(
        self,
        *,
        frame_provider: StrictFrameProvider,
        runtime_provider: Sam3RuntimeProvider | object,
        model_path: Path,
        runtime_config: Mapping[str, object],
        queries: Sequence[str],
        encoder_factory: EncoderFactory | None = None,
        frame_composer: FrameComposer | None = None,
        media_probe: MediaProbe | None = None,
    ) -> None:
        if not callable(getattr(frame_provider, "read_frame", None)):
            raise TypeError("frame_provider must expose read_frame")
        if not callable(getattr(runtime_provider, "get_segmenter", None)):
            raise TypeError("runtime_provider must expose get_segmenter")
        query_tuple = tuple(
            str(query).strip() for query in queries if str(query).strip()
        )
        if not query_tuple:
            raise ValueError("queries must not be empty")
        self.frame_provider = frame_provider
        self._runtime_provider = runtime_provider
        self._model_path = Path(model_path)
        self._runtime_config = dict(runtime_config)
        self._queries = query_tuple
        self._encoder_factory = encoder_factory or _default_encoder_factory
        self._frame_composer = frame_composer or _default_frame_composer
        self._media_probe = media_probe or _default_media_probe

    @staticmethod
    def _render_error(
        code: str,
        exc: BaseException | None = None,
    ) -> OverlayRenderError:
        stable = code if code in PUBLIC_OVERLAY_FAILURE_CODES else "overlay_render_failed"
        error = OverlayRenderError(stable)
        if exc is not None:
            error.__cause__ = exc
        return error

    @staticmethod
    def _remove_partial(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    def render_interval(
        self,
        request: OverlayRequest,
        start_frame: int,
        end_frame_exclusive: int,
        output_path: Path,
    ) -> Mapping[str, object]:
        try:
            _non_empty_identity(self.frame_provider.input_identity)
        except OverlaySetupError as exc:
            raise self._render_error(exc.code, exc) from None
        try:
            segmenter = self._runtime_provider.get_segmenter(
                self._model_path,
                dict(self._runtime_config),
            )
        except BaseException as exc:
            raise self._render_error("overlay_model_unavailable", exc) from None

        encoder: OverlayEncoder | None = None
        expected_size: tuple[int, int] | None = None
        count = 0
        failure: OverlayRenderError | None = None
        try:
            for source_frame in range(start_frame, end_frame_exclusive):
                try:
                    sample = self.frame_provider.read_frame(source_frame)
                except OverlaySetupError as exc:
                    raise self._render_error(exc.code, exc) from None
                except BaseException as exc:
                    raise self._render_error("overlay_decode_failed", exc) from None
                if sample.source_frame != source_frame:
                    raise self._render_error("overlay_mapping_unavailable")
                if (
                    isinstance(sample.video_frame, bool)
                    or not isinstance(sample.video_frame, int)
                    or sample.video_frame < 0
                ):
                    raise self._render_error("overlay_mapping_unavailable")
                try:
                    frame = _frame_array(sample.frame_rgb)
                except OverlaySetupError as exc:
                    raise self._render_error(exc.code, exc) from None
                frame_size = (int(frame.shape[1]), int(frame.shape[0]))
                if expected_size is None:
                    expected_size = frame_size
                    try:
                        encoder = self._encoder_factory(
                            output_path,
                            request.fps,
                            expected_size,
                        )
                    except BaseException as exc:
                        raise self._render_error("overlay_encoder_failed", exc) from None
                elif frame_size != expected_size:
                    raise self._render_error("overlay_decode_failed")
                try:
                    masks = segmenter.segment_frame(
                        frame,
                        list(self._queries),
                        dict(self._runtime_config),
                    )
                except BaseException as exc:
                    raise self._render_error("overlay_inference_failed", exc) from None
                if (
                    isinstance(masks, (str, bytes, bytearray, Mapping))
                    or not isinstance(masks, Sequence)
                    or not masks
                ):
                    raise self._render_error("overlay_inference_failed")
                try:
                    composed = self._frame_composer(sample, masks)
                except OverlaySetupError as exc:
                    raise self._render_error(exc.code, exc) from None
                except BaseException as exc:
                    raise self._render_error("overlay_render_failed", exc) from None
                assert encoder is not None
                try:
                    encoder.write(_frame_array(composed))
                except BaseException as exc:
                    raise self._render_error("overlay_encoder_failed", exc) from None
                count += 1
        except OverlayRenderError as exc:
            failure = exc
        finally:
            if encoder is not None:
                try:
                    encoder.close()
                except BaseException as exc:
                    if failure is None:
                        failure = self._render_error("overlay_encoder_failed", exc)
        if failure is not None:
            self._remove_partial(output_path)
            raise failure
        if count != end_frame_exclusive - start_frame or not output_path.is_file():
            self._remove_partial(output_path)
            raise self._render_error("overlay_encoder_failed")
        try:
            probed = self._media_probe(output_path)
            probed_count = getattr(probed, "frame_count")
            width = getattr(probed, "width_px")
            height = getattr(probed, "height_px")
            fps_num = getattr(probed, "fps_num")
            fps_den = getattr(probed, "fps_den")
            codec = getattr(probed, "codec")
            actual_fps = float(fps_num) / float(fps_den)
        except Exception as exc:
            self._remove_partial(output_path)
            raise self._render_error("overlay_encoder_failed", exc) from None
        if (
            probed_count != count
            or (width, height) != expected_size
            or not math.isclose(
                actual_fps,
                float(request.fps),
                rel_tol=1e-6,
                abs_tol=1e-3,
            )
            or codec != "mpeg4"
        ):
            self._remove_partial(output_path)
            raise self._render_error("overlay_encoder_failed")
        return {
            "frame_count": count,
            "fps": request.fps,
            "width_px": width,
            "height_px": height,
            "codec": codec,
            "first_source_frame": start_frame,
            "end_source_frame_exclusive": end_frame_exclusive,
            "mapping": "explicit",
            "renderer_version": self.renderer_version,
        }


class ExplicitRecipeFrameProvider:
    """Decode only report-declared source-to-video lookup entries.

    Recipe schema (``sam3_overlay_input.v1``) stores an explicit mapping table
    and per-source-frame 2D points.  The renderer never substitutes an identity
    or offset mapping when an entry is absent.
    """

    def __init__(
        self,
        *,
        video_path: Path,
        recipe: Mapping[str, object],
        keypoint_reference_path: Path | None = None,
    ) -> None:
        if recipe.get("schema_version") != RECIPE_SCHEMA:
            raise OverlaySetupError("overlay_mapping_unavailable")
        raw_mapping = recipe.get("source_to_video")
        raw_keypoints = recipe.get("keypoints_2d")
        raw_reference = recipe.get("keypoints_2d_reference")
        if not isinstance(raw_mapping, Mapping) or (
            not isinstance(raw_keypoints, Mapping)
            and not isinstance(raw_reference, Mapping)
        ):
            raise OverlaySetupError("overlay_mapping_unavailable")
        mapping: dict[int, int] = {}
        for raw_source, raw_video in raw_mapping.items():
            try:
                source_frame = int(raw_source)
            except (TypeError, ValueError):
                raise OverlaySetupError("overlay_mapping_unavailable") from None
            if str(source_frame) != str(raw_source) and raw_source != source_frame:
                raise OverlaySetupError("overlay_mapping_unavailable")
            mapping[source_frame] = _finite_int(
                raw_video, name="video_frame", minimum=0
            )
        if not mapping:
            raise OverlaySetupError("overlay_mapping_unavailable")
        self._video_path = Path(video_path).resolve()
        self._mapping = mapping
        self._keypoints = dict(raw_keypoints) if isinstance(raw_keypoints, Mapping) else None
        self._keypoint_table: Any | None = None
        self._keypoint_fields: dict[str, str] = {}
        keypoint_identity: object = raw_keypoints
        if isinstance(raw_reference, Mapping):
            if (
                raw_reference.get("schema_version") != REFERENCE_SCHEMA
                or raw_reference.get("row_mapping") != "source_frame_index"
                or keypoint_reference_path is None
                or raw_reference.get("sha256") != file_sha256(keypoint_reference_path)
            ):
                raise OverlaySetupError("overlay_input_unavailable")
            raw_fields = raw_reference.get("fields")
            if not isinstance(raw_fields, Mapping) or not raw_fields:
                raise OverlaySetupError("overlay_input_unavailable")
            for side, field in raw_fields.items():
                if not isinstance(side, str) or not isinstance(field, str) or not field:
                    raise OverlaySetupError("overlay_input_unavailable")
                self._keypoint_fields[side] = field
            try:
                import pandas as pd

                self._keypoint_table = pd.read_parquet(keypoint_reference_path)
            except Exception:
                raise OverlaySetupError("overlay_input_unavailable") from None
            if any(
                field not in self._keypoint_table.columns
                for field in self._keypoint_fields.values()
            ):
                raise OverlaySetupError("overlay_input_unavailable")
            keypoint_identity = {
                "schema_version": REFERENCE_SCHEMA,
                "source": str(raw_reference.get("source") or ""),
                "sha256": str(raw_reference.get("sha256") or ""),
                "row_mapping": "source_frame_index",
                "fields": dict(self._keypoint_fields),
            }
        else:
            keypoint_identity = _plain_identity(keypoint_identity)
        self._identity = {
            "recipe_version": RECIPE_SCHEMA,
            "mapping": canonical_sha256(mapping),
            "video": str(recipe.get("video_identity") or ""),
            "keypoints": canonical_sha256(keypoint_identity),
        }
        if not self._identity["video"]:
            raise OverlaySetupError("overlay_source_unavailable")
        self._video_identity = str(self._identity["video"])
        self._capture: Any | None = None
        self._lock = RLock()

    @property
    def input_identity(self) -> Mapping[str, object]:
        return dict(self._identity)

    def _open(self) -> Any:
        import cv2

        try:
            actual_identity = file_sha256(self._video_path)
        except OSError:
            raise OverlaySetupError("overlay_source_unavailable") from None
        if actual_identity != self._video_identity:
            raise OverlaySetupError("overlay_source_unavailable")
        capture = cv2.VideoCapture(str(self._video_path))
        if not capture.isOpened():
            capture.release()
            raise OverlaySetupError("overlay_source_unavailable")
        return capture

    def read_frame(self, source_frame: int) -> OverlayFrame:
        import cv2

        video_frame = self._mapping.get(source_frame)
        if self._keypoints is not None:
            raw_points = self._keypoints.get(
                str(source_frame), self._keypoints.get(source_frame)
            )
        elif (
            self._keypoint_table is not None
            and 0 <= source_frame < len(self._keypoint_table)
        ):
            row = self._keypoint_table.iloc[source_frame]
            raw_points = {
                side: row[field] for side, field in self._keypoint_fields.items()
            }
        else:
            raw_points = None
        if video_frame is None or not isinstance(raw_points, Mapping):
            raise OverlaySetupError("overlay_mapping_unavailable")
        points: dict[str, np.ndarray] = {}
        for side, value in raw_points.items():
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    raise OverlaySetupError("overlay_input_unavailable") from None
            try:
                array = np.asarray(value, dtype=np.float32)
            except (TypeError, ValueError):
                raise OverlaySetupError("overlay_input_unavailable") from None
            if self._keypoints is None:
                if array.size != 42:
                    raise OverlaySetupError("overlay_input_unavailable")
                array = array.reshape(21, 2)
            if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
                raise OverlaySetupError("overlay_input_unavailable")
            points[str(side)] = array
        if not points:
            raise OverlaySetupError("overlay_input_unavailable")
        with self._lock:
            if self._capture is None:
                self._capture = self._open()
            frame_count = float(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if (
                not math.isfinite(frame_count)
                or frame_count <= 0
                or video_frame >= int(round(frame_count))
            ):
                raise OverlaySetupError("overlay_decode_failed")
            if not self._capture.set(cv2.CAP_PROP_POS_FRAMES, video_frame):
                raise OverlaySetupError("overlay_decode_failed")
            positioned = float(self._capture.get(cv2.CAP_PROP_POS_FRAMES))
            if not math.isfinite(positioned) or abs(positioned - video_frame) > 0.5:
                raise OverlaySetupError("overlay_decode_failed")
            ok, frame_bgr = self._capture.read()
            after_read = float(self._capture.get(cv2.CAP_PROP_POS_FRAMES))
        if not ok or frame_bgr is None:
            raise OverlaySetupError("overlay_decode_failed")
        if not math.isfinite(after_read) or abs(after_read - (video_frame + 1)) > 0.5:
            raise OverlaySetupError("overlay_decode_failed")
        return OverlayFrame(
            source_frame=source_frame,
            video_frame=video_frame,
            frame_rgb=np.asarray(frame_bgr)[..., ::-1].copy(),
            keypoints=points,
        )

    def close(self) -> None:
        with self._lock:
            if self._capture is not None:
                self._capture.release()
                self._capture = None


def frame_provider_from_context(
    context: AssetContext,
    source: SourceMedia,
) -> StrictFrameProvider:
    recipe = context.metadata.get("sam3_overlay_recipe")
    if not isinstance(recipe, Mapping):
        manifest = context.metadata.get("manifest_row")
        recipe = (
            manifest.get("sam3_overlay_recipe")
            if isinstance(manifest, Mapping)
            else None
        )
    if not isinstance(recipe, Mapping):
        raise OverlaySetupError("overlay_mapping_unavailable")
    video_source = recipe.get("video_source")
    if not isinstance(video_source, str) or not video_source:
        raise OverlaySetupError("overlay_source_unavailable")
    declared = context.source_files.get(video_source)
    declared_path = declared.get("path") if isinstance(declared, Mapping) else None
    if not isinstance(declared_path, str) or not declared_path:
        raise OverlaySetupError("overlay_source_unavailable")
    candidate = (context.batch_root / declared_path).resolve()
    if candidate != source.path.resolve():
        raise OverlaySetupError("overlay_source_unavailable")
    expected_identity = recipe.get("video_identity")
    if expected_identity != source.etag:
        raise OverlaySetupError("overlay_source_unavailable")
    reference_path: Path | None = None
    raw_reference = recipe.get("keypoints_2d_reference")
    if isinstance(raw_reference, Mapping):
        reference_source = raw_reference.get("source")
        if not isinstance(reference_source, str) or not reference_source:
            raise OverlaySetupError("overlay_input_unavailable")
        reference_entry = context.source_files.get(reference_source)
        reference_value = (
            reference_entry.get("path")
            if isinstance(reference_entry, Mapping)
            else None
        )
        if not isinstance(reference_value, str) or not reference_value:
            raise OverlaySetupError("overlay_input_unavailable")
        reference_path = (context.batch_root / reference_value).resolve()
        try:
            reference_path.relative_to(context.batch_root.resolve())
        except ValueError:
            raise OverlaySetupError("overlay_input_unavailable") from None
        if not reference_path.is_file():
            raise OverlaySetupError("overlay_input_unavailable")
    return ExplicitRecipeFrameProvider(
        video_path=source.path,
        recipe=recipe,
        keypoint_reference_path=reference_path,
    )


def build_overlay_request(
    *,
    context: AssetContext | object,
    source: SourceMedia | object,
    selected: Sequence[OverlayIssueInput | object],
    cache_root: Path,
    renderer: Sam3OverlayRenderer | object,
    model_hash: str,
    runtime_config: Mapping[str, object],
    renderer_inputs: Mapping[str, object],
) -> OverlayRequest:
    asset_id = getattr(context, "asset_id", None)
    batch_root = getattr(context, "batch_root", None)
    if not isinstance(asset_id, str) or not asset_id or not isinstance(batch_root, Path):
        raise OverlaySetupError("overlay_input_unavailable")
    resolved_cache = Path(cache_root).resolve()
    try:
        resolved_cache.relative_to(batch_root.resolve())
    except ValueError:
        raise OverlaySetupError("overlay_input_unavailable") from None
    source_hash = getattr(source, "etag", None)
    fps = getattr(source, "fps", None)
    total_frames = getattr(source, "total_frames", None)
    if not isinstance(source_hash, str) or not source_hash:
        raise OverlaySetupError("overlay_source_unavailable")
    intervals: list[tuple[int, int]] = []
    for item in selected:
        frame_range = getattr(item, "frame_range", None)
        start = getattr(frame_range, "start_frame", None)
        end = getattr(frame_range, "end_frame_exclusive", None)
        intervals.append(
            (
                _finite_int(start, name="start_frame"),
                _finite_int(end, name="end_frame_exclusive", minimum=1),
            )
        )
    provider = getattr(renderer, "frame_provider", None)
    identity = _non_empty_identity(getattr(provider, "input_identity", None))
    renderer_version = getattr(renderer, "renderer_version", None)
    if not isinstance(renderer_version, str) or not renderer_version:
        raise OverlaySetupError("overlay_input_unavailable")
    try:
        config_hash = canonical_sha256(dict(runtime_config))
        input_hash = canonical_sha256(
            {
                "asset_id": asset_id,
                "frame_provider": dict(identity),
                "renderer_inputs": dict(renderer_inputs),
                "renderer_version": renderer_version,
            }
        )
    except (TypeError, ValueError):
        raise OverlaySetupError("overlay_input_unavailable") from None
    return OverlayRequest(
        asset_id=asset_id,
        cache_root=resolved_cache,
        source_sha256=source_hash,
        intervals=tuple(intervals),
        fps=fps,
        total_frames=total_frames,
        model_hash=model_hash,
        config_hash=config_hash,
        input_fingerprint_hash=input_hash,
        renderer_version=renderer_version,
        renderer=renderer,
    )


class UnavailableOverlayProvider:
    """Immediate terminal provider used when process-level inputs are absent."""

    def __init__(self, code: str) -> None:
        self.code = (
            code
            if code in PUBLIC_OVERLAY_FAILURE_CODES
            else "overlay_input_unavailable"
        )

    def get_asset_overlays(
        self,
        asset_id: str,
        selected: tuple[OverlayIssueInput, ...],
    ) -> Mapping[str, object]:
        return {
            item.issue_id: OverlayHandle(
                "failed",
                code=self.code,
                retryable=self.code != "overlay_model_unavailable",
            )
            for item in selected
        }


class ProductionWorkerOverlayProvider(WorkerOverlayProvider):
    """Worker adapter that turns strict setup failures into terminal views."""

    def __init__(
        self,
        *,
        worker: BoundedOverlayWorker,
        request_factory: Callable[
            [str, tuple[OverlayIssueInput, ...]], OverlayRequest
        ],
    ) -> None:
        self._production_worker = worker
        self._production_request_factory = request_factory
        super().__init__(worker=worker, request_factory=request_factory)

    def get_asset_overlays(
        self,
        asset_id: str,
        selected: tuple[OverlayIssueInput, ...],
    ) -> Mapping[str, object]:
        try:
            request = self._production_request_factory(asset_id, selected)
        except OverlaySetupError as exc:
            return UnavailableOverlayProvider(exc.code).get_asset_overlays(
                asset_id, selected
            )

        worker = self._production_worker

        class AtomicJobViewWorker:
            """Keep a failed asset job from exposing earlier ready segments."""

            @staticmethod
            def submit(value: OverlayRequest) -> object:
                view = worker.submit(value)
                if getattr(view, "status", None) != "failed":
                    return view
                code = getattr(view, "code", None) or "overlay_render_failed"
                retryable = getattr(view, "retryable", False) is True
                segments = tuple(
                    replace(
                        segment,
                        status="failed",
                        path=None,
                        code=code,
                        retryable=retryable,
                        content_sha256=None,
                        metadata=None,
                    )
                    for segment in view.segments
                )
                return replace(view, segments=segments)

        delegate = WorkerOverlayProvider(
            worker=AtomicJobViewWorker(),
            request_factory=lambda requested_asset, requested_selected: request,
        )
        return delegate.get_asset_overlays(asset_id, selected)


def _model_hash(model_path: Path) -> str:
    resolved = Path(model_path).expanduser().resolve()
    if resolved.is_file():
        return file_sha256(resolved)
    if not resolved.is_dir():
        raise OverlaySetupError("overlay_model_unavailable")
    files: dict[str, str] = {}
    try:
        candidates = sorted(
            (path for path in resolved.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(resolved).as_posix(),
        )
        for path in candidates:
            files[path.relative_to(resolved).as_posix()] = file_sha256(path)
    except OSError:
        raise OverlaySetupError("overlay_model_unavailable") from None
    if not files:
        raise OverlaySetupError("overlay_model_unavailable")
    return canonical_sha256({"path": str(resolved), "files": files})


@dataclass
class ProductionOverlayRuntime:
    runtime_provider: Sam3RuntimeProvider
    worker: BoundedOverlayWorker
    provider: WorkerOverlayProvider
    frame_providers: dict[tuple[str, str], StrictFrameProvider]

    def shutdown(self) -> None:
        self.worker.shutdown()
        for provider in self.frame_providers.values():
            close = getattr(provider, "close", None)
            if callable(close):
                close()


def _batch_cache_root(
    contexts: Mapping[str, AssetContext], cache_relative: Path
) -> Path:
    if cache_relative.is_absolute() or ".." in cache_relative.parts:
        raise OverlaySetupError("overlay_input_unavailable")
    roots = {context.batch_root.resolve() for context in contexts.values()}
    if len(roots) != 1:
        raise OverlaySetupError("overlay_input_unavailable")
    batch_root = next(iter(roots))
    base = (batch_root / cache_relative).resolve()
    try:
        base.relative_to(batch_root)
    except ValueError:
        raise OverlaySetupError("overlay_input_unavailable") from None
    return base


def build_production_overlay_runtime(
    *,
    contexts: Mapping[str, AssetContext],
    media_catalog: MediaCatalog,
    model_path: Path,
    cache_relative: Path = Path(".human_qc/overlay-cache"),
    max_workers: int = 1,
    max_pending: int = 1,
    max_cache_bytes: int | None = DEFAULT_OVERLAY_MAX_CACHE_BYTES,
    max_ready_jobs: int | None = None,
    runtime_config: Mapping[str, object] | None = None,
    frame_provider_factory: FrameProviderFactory = frame_provider_from_context,
) -> ProductionOverlayRuntime:
    from tools.run_manifest_sam3_containment import DEFAULT_QUERIES, SAM3_CONFIG

    config = dict(SAM3_CONFIG if runtime_config is None else runtime_config)
    queries = tuple(
        value.strip() for value in DEFAULT_QUERIES.split(",") if value.strip()
    )
    resolved_model = Path(model_path).expanduser().resolve()
    model_hash = _model_hash(resolved_model)
    cache_root = _batch_cache_root(contexts, cache_relative)
    runtime_provider = Sam3RuntimeProvider()
    worker = BoundedOverlayWorker(
        max_workers=max_workers,
        max_pending=max_pending,
        max_cache_bytes=(
            DEFAULT_OVERLAY_MAX_CACHE_BYTES
            if max_cache_bytes is None
            else max_cache_bytes
        ),
        max_ready_jobs=max_ready_jobs,
    )
    providers: dict[tuple[str, str], StrictFrameProvider] = {}
    provider_lock = RLock()

    def request_factory(
        asset_id: str,
        selected: tuple[OverlayIssueInput, ...],
    ) -> OverlayRequest:
        context = contexts.get(asset_id)
        if context is None:
            raise OverlaySetupError("overlay_source_unavailable")
        try:
            source = media_catalog.source(asset_id)
        except MediaError:
            raise OverlaySetupError("overlay_source_unavailable") from None
        source_identity = source.etag
        if not isinstance(source_identity, str) or not source_identity:
            raise OverlaySetupError("overlay_source_unavailable")
        provider_key = (asset_id, source_identity)
        with provider_lock:
            frame_provider = providers.get(provider_key)
            if frame_provider is None:
                frame_provider = frame_provider_factory(context, source)
                providers[provider_key] = frame_provider
        renderer_inputs = {
            "queries": list(queries),
            "style": "mask-and-keypoints-v1",
        }
        renderer = Sam3OverlayRenderer(
            frame_provider=frame_provider,
            runtime_provider=runtime_provider,
            model_path=resolved_model,
            runtime_config=config,
            queries=queries,
        )
        return build_overlay_request(
            context=context,
            source=source,
            selected=selected,
            cache_root=cache_root,
            renderer=renderer,
            model_hash=model_hash,
            runtime_config=config,
            renderer_inputs=renderer_inputs,
        )

    provider = ProductionWorkerOverlayProvider(
        worker=worker,
        request_factory=request_factory,
    )
    return ProductionOverlayRuntime(runtime_provider, worker, provider, providers)


__all__ = [
    "ExplicitRecipeFrameProvider",
    "DEFAULT_OVERLAY_MAX_CACHE_BYTES",
    "OverlayFrame",
    "OverlaySetupError",
    "PUBLIC_OVERLAY_FAILURE_CODES",
    "ProductionOverlayRuntime",
    "ProductionWorkerOverlayProvider",
    "RENDERER_VERSION",
    "Sam3OverlayRenderer",
    "UnavailableOverlayProvider",
    "build_overlay_request",
    "build_production_overlay_runtime",
    "frame_provider_from_context",
]
