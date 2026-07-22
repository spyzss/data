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
import hashlib
import json
import math
import os
from pathlib import Path
from threading import RLock
import tempfile
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
    merge_frame_intervals,
)
from .warn_workbench_service import (
    OverlayHandle,
    OverlayIssueInput,
    WorkerOverlayProvider,
)


RENDERER_VERSION = "sam3-overlay-renderer-v1"
RECIPE_SCHEMA = "sam3_overlay_input.v2"
MAPPING_SCHEMA = "linear_ranges.v1"
PARQUET_REFERENCE_SCHEMA = "parquet_columns.v2"
JSON_REFERENCE_SCHEMA = "json_keypoints.v1"
KEYPOINT_SIDECAR_SCHEMA = "sam3_overlay_keypoints.v1"
DEFAULT_OVERLAY_MAX_CACHE_BYTES = 2 * 1024 * 1024 * 1024
_MP4_VIDEO_MAJOR_BRANDS = frozenset(
    {
        "isom",
        "iso2",
        "iso3",
        "iso4",
        "iso5",
        "iso6",
        "iso7",
        "iso8",
        "iso9",
        "mp41",
        "mp42",
        "avc1",
        "m4v",
    }
)
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
            close_provider = getattr(self.frame_provider, "close", None)
            if callable(close_provider):
                try:
                    close_provider()
                except BaseException as exc:
                    if failure is None:
                        failure = self._render_error("overlay_render_failed", exc)
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
            container_format = getattr(probed, "container_format")
            container_major_brand = getattr(probed, "container_major_brand")
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
            or not isinstance(container_format, str)
            or "mp4" not in {
                item.strip().lower()
                for item in container_format.split(",")
                if item.strip()
            }
            or not isinstance(container_major_brand, str)
            or container_major_brand.strip().lower() not in _MP4_VIDEO_MAJOR_BRANDS
        ):
            self._remove_partial(output_path)
            raise self._render_error("overlay_encoder_failed")
        return {
            "frame_count": count,
            "fps": request.fps,
            "width_px": width,
            "height_px": height,
            "codec": codec,
            "container_format": container_format,
            "container_major_brand": container_major_brand,
            "first_source_frame": start_frame,
            "end_source_frame_exclusive": end_frame_exclusive,
            "mapping": "explicit",
            "renderer_version": self.renderer_version,
        }


def _sha256_identity(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
    ):
        raise OverlaySetupError("overlay_input_unavailable")
    try:
        int(value[7:], 16)
    except ValueError:
        raise OverlaySetupError("overlay_input_unavailable") from None
    return value.lower()


def validate_overlay_recipe(
    value: object,
    *,
    asset_id: str | None = None,
    producer_fingerprint_sha256: str | None = None,
) -> dict[str, object]:
    """Validate one compact current recipe without reading referenced payloads."""

    if not isinstance(value, Mapping) or value.get("schema_version") != RECIPE_SCHEMA:
        raise OverlaySetupError("overlay_mapping_unavailable")
    if "source_to_video" in value or "keypoints_2d" in value:
        raise OverlaySetupError("overlay_input_unavailable")
    recipe_asset = value.get("asset_id")
    if not isinstance(recipe_asset, str) or not recipe_asset:
        raise OverlaySetupError("overlay_input_unavailable")
    if asset_id is not None and recipe_asset != asset_id:
        raise OverlaySetupError("overlay_input_unavailable")
    fingerprint = _sha256_identity(value.get("producer_fingerprint_sha256"))
    if (
        producer_fingerprint_sha256 is not None
        and fingerprint != producer_fingerprint_sha256
    ):
        raise OverlaySetupError("overlay_input_unavailable")
    video_source = value.get("video_source")
    if not isinstance(video_source, str) or not video_source:
        raise OverlaySetupError("overlay_source_unavailable")
    video_identity = _sha256_identity(value.get("video_identity"))

    raw_intervals = value.get("candidate_intervals")
    if (
        isinstance(raw_intervals, (str, bytes, bytearray, Mapping))
        or not isinstance(raw_intervals, Sequence)
    ):
        raise OverlaySetupError("overlay_mapping_unavailable")
    try:
        intervals = merge_frame_intervals(tuple(tuple(item) for item in raw_intervals))
    except (TypeError, ValueError):
        raise OverlaySetupError("overlay_mapping_unavailable") from None
    if not intervals:
        raise OverlaySetupError("overlay_mapping_unavailable")

    raw_mapping = value.get("source_mapping")
    if (
        not isinstance(raw_mapping, Mapping)
        or raw_mapping.get("schema_version") != MAPPING_SCHEMA
    ):
        raise OverlaySetupError("overlay_mapping_unavailable")
    raw_ranges = raw_mapping.get("ranges")
    if (
        isinstance(raw_ranges, (str, bytes, bytearray, Mapping))
        or not isinstance(raw_ranges, Sequence)
    ):
        raise OverlaySetupError("overlay_mapping_unavailable")
    ranges: list[dict[str, int]] = []
    for raw in raw_ranges:
        if not isinstance(raw, Mapping):
            raise OverlaySetupError("overlay_mapping_unavailable")
        start = _finite_int(raw.get("start_frame"), name="start_frame")
        end = _finite_int(
            raw.get("end_frame_exclusive"),
            name="end_frame_exclusive",
            minimum=1,
        )
        video_start = _finite_int(
            raw.get("video_start_frame"),
            name="video_start_frame",
        )
        if end <= start:
            raise OverlaySetupError("overlay_mapping_unavailable")
        ranges.append(
            {
                "start_frame": start,
                "end_frame_exclusive": end,
                "video_start_frame": video_start,
            }
        )
    if tuple((item["start_frame"], item["end_frame_exclusive"]) for item in ranges) != intervals:
        raise OverlaySetupError("overlay_mapping_unavailable")

    reference = value.get("keypoints_2d_reference")
    if not isinstance(reference, Mapping):
        raise OverlaySetupError("overlay_input_unavailable")
    reference_schema = reference.get("schema_version")
    size_bytes = reference.get("size_bytes")
    if (
        isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes <= 0
    ):
        raise OverlaySetupError("overlay_input_unavailable")
    reference_sha = _sha256_identity(reference.get("sha256"))
    normalized_reference: dict[str, object] = {
        "schema_version": reference_schema,
        "sha256": reference_sha,
        "size_bytes": size_bytes,
    }
    if reference_schema == PARQUET_REFERENCE_SCHEMA:
        source = reference.get("source")
        fields = reference.get("fields")
        if (
            not isinstance(source, str)
            or not source
            or reference.get("row_mapping") != "source_frame_index"
            or not isinstance(fields, Mapping)
            or not fields
            or any(
                not isinstance(side, str)
                or not side
                or not isinstance(field, str)
                or not field
                for side, field in fields.items()
            )
        ):
            raise OverlaySetupError("overlay_input_unavailable")
        normalized_reference.update(
            {
                "source": source,
                "row_mapping": "source_frame_index",
                "fields": {str(side): str(field) for side, field in fields.items()},
            }
        )
    elif reference_schema == JSON_REFERENCE_SCHEMA:
        relative_path = reference.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            raise OverlaySetupError("overlay_input_unavailable")
        normalized_reference["relative_path"] = relative_path
    else:
        raise OverlaySetupError("overlay_input_unavailable")

    normalized = {
        "schema_version": RECIPE_SCHEMA,
        "asset_id": recipe_asset,
        "producer_fingerprint_sha256": fingerprint,
        "video_source": video_source,
        "video_identity": video_identity,
        "candidate_intervals": [list(interval) for interval in intervals],
        "source_mapping": {
            "schema_version": MAPPING_SCHEMA,
            "ranges": ranges,
        },
        "keypoints_2d_reference": normalized_reference,
    }
    canonical_sha256(normalized)
    return normalized


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
        expected_video_identity: tuple[int, int, int, int] | None = None,
    ) -> None:
        normalized = validate_overlay_recipe(recipe)
        self._recipe_asset_id = str(normalized["asset_id"])
        raw_mapping = normalized["source_mapping"]
        assert isinstance(raw_mapping, Mapping)
        raw_ranges = raw_mapping["ranges"]
        assert isinstance(raw_ranges, list)
        self._video_path = Path(video_path).resolve()
        self._expected_video_identity = expected_video_identity
        self._mapping_ranges = tuple(
            (
                int(item["start_frame"]),
                int(item["end_frame_exclusive"]),
                int(item["video_start_frame"]),
            )
            for item in raw_ranges
            if isinstance(item, Mapping)
        )
        raw_reference = normalized["keypoints_2d_reference"]
        assert isinstance(raw_reference, Mapping)
        self._reference = dict(raw_reference)
        self._reference_path = (
            None if keypoint_reference_path is None else Path(keypoint_reference_path).resolve()
        )
        if self._reference_path is None:
            raise OverlaySetupError("overlay_input_unavailable")
        self._keypoint_table: Any | None = None
        self._json_keypoints: dict[int, Mapping[str, object]] | None = None
        self._keypoint_fields: dict[str, str] = {}
        if raw_reference.get("schema_version") == PARQUET_REFERENCE_SCHEMA:
            raw_fields = raw_reference.get("fields")
            assert isinstance(raw_fields, Mapping)
            for side, field in raw_fields.items():
                self._keypoint_fields[str(side)] = str(field)
        self._identity = {
            "recipe_version": RECIPE_SCHEMA,
            "mapping": canonical_sha256(normalized["source_mapping"]),
            "video": str(normalized["video_identity"]),
            "keypoints": str(raw_reference["sha256"]),
        }
        self._video_identity = str(self._identity["video"])
        self._capture: Any | None = None
        self._video_handle: Any | None = None
        self._parquet_snapshot: Any | None = None
        self._lock = RLock()

    @property
    def input_identity(self) -> Mapping[str, object]:
        return dict(self._identity)

    @staticmethod
    def _opened_identity(handle: Any) -> tuple[int, int, int, int]:
        stat = os.fstat(handle.fileno())
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)

    @classmethod
    def _verified_payload(
        cls,
        path: Path,
        *,
        expected_sha256: str,
        expected_size: int,
    ) -> bytes:
        try:
            with path.open("rb") as handle:
                before = cls._opened_identity(handle)
                if before[2] != expected_size:
                    raise OverlaySetupError("overlay_input_unavailable")
                digest = hashlib.sha256()
                chunks: list[bytes] = []
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                    chunks.append(chunk)
                if cls._opened_identity(handle) != before:
                    raise OverlaySetupError("overlay_input_unavailable")
        except OverlaySetupError:
            raise
        except OSError:
            raise OverlaySetupError("overlay_input_unavailable") from None
        if "sha256:" + digest.hexdigest() != expected_sha256:
            raise OverlaySetupError("overlay_input_unavailable")
        return b"".join(chunks)

    def _materialize_keypoints(self) -> None:
        if self._keypoint_table is not None or self._json_keypoints is not None:
            return
        payload = self._verified_payload(
            self._reference_path,
            expected_sha256=str(self._reference["sha256"]),
            expected_size=int(self._reference["size_bytes"]),
        )
        schema = self._reference.get("schema_version")
        if schema == PARQUET_REFERENCE_SCHEMA:
            snapshot: Any | None = None
            try:
                import pandas as pd

                snapshot = tempfile.NamedTemporaryFile(mode="w+b", suffix=".parquet")
                snapshot.write(payload)
                snapshot.flush()
                snapshot.seek(0)
                self._keypoint_table = pd.read_parquet(snapshot.name)
                self._parquet_snapshot = snapshot
            except Exception:
                if snapshot is not None:
                    snapshot.close()
                raise OverlaySetupError("overlay_input_unavailable") from None
            if any(
                field not in self._keypoint_table.columns
                for field in self._keypoint_fields.values()
            ):
                raise OverlaySetupError("overlay_input_unavailable")
            return
        try:
            raw = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise OverlaySetupError("overlay_input_unavailable") from None
        if (
            not isinstance(raw, Mapping)
            or raw.get("schema_version") != KEYPOINT_SIDECAR_SCHEMA
            or raw.get("asset_id") != self._recipe_asset_id
            or not isinstance(raw.get("frames"), list)
        ):
            raise OverlaySetupError("overlay_input_unavailable")
        keypoints: dict[int, Mapping[str, object]] = {}
        for row in raw["frames"]:
            if not isinstance(row, Mapping):
                raise OverlaySetupError("overlay_input_unavailable")
            source_frame = _finite_int(row.get("source_frame"), name="source_frame")
            points = row.get("keypoints")
            if source_frame in keypoints or not isinstance(points, Mapping):
                raise OverlaySetupError("overlay_input_unavailable")
            keypoints[source_frame] = points
        self._json_keypoints = keypoints

    def _open(self) -> Any:
        import cv2

        try:
            handle = self._video_path.open("rb")
        except OSError:
            raise OverlaySetupError("overlay_source_unavailable") from None
        try:
            identity = self._opened_identity(handle)
            if self._expected_video_identity is not None and identity != self._expected_video_identity:
                raise OverlaySetupError("overlay_source_unavailable")
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            if self._opened_identity(handle) != identity:
                raise OverlaySetupError("overlay_source_unavailable")
            if "sha256:" + digest.hexdigest() != self._video_identity:
                raise OverlaySetupError("overlay_source_unavailable")
            handle.seek(0)
        except Exception:
            handle.close()
            raise
        try:
            capture = cv2.VideoCapture(f"/dev/fd/{handle.fileno()}")
        except Exception:
            handle.close()
            raise OverlaySetupError("overlay_source_unavailable") from None
        if not capture.isOpened():
            capture.release()
            handle.close()
            raise OverlaySetupError("overlay_source_unavailable")
        self._video_handle = handle
        return capture

    def _video_frame(self, source_frame: int) -> int | None:
        for start, end, video_start in self._mapping_ranges:
            if start <= source_frame < end:
                return video_start + source_frame - start
        return None

    def read_frame(self, source_frame: int) -> OverlayFrame:
        import cv2

        video_frame = self._video_frame(source_frame)
        with self._lock:
            self._materialize_keypoints()
        if self._json_keypoints is not None:
            raw_points = self._json_keypoints.get(source_frame)
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
            if self._keypoint_table is not None:
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
            if self._video_handle is not None:
                self._video_handle.close()
                self._video_handle = None
            if self._parquet_snapshot is not None:
                self._parquet_snapshot.close()
                self._parquet_snapshot = None
            self._keypoint_table = None
            self._json_keypoints = None


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
    recipe = validate_overlay_recipe(recipe, asset_id=context.asset_id)
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
        if raw_reference.get("schema_version") == PARQUET_REFERENCE_SCHEMA:
            reference_source = raw_reference.get("source")
            if not isinstance(reference_source, str) or not reference_source:
                raise OverlaySetupError("overlay_input_unavailable")
            reference_entry = context.source_files.get(reference_source)
            reference_value = (
                reference_entry.get("path")
                if isinstance(reference_entry, Mapping)
                else None
            )
        else:
            reference_value = raw_reference.get("relative_path")
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
        expected_video_identity=getattr(source, "identity", None),
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

    def retry_asset_overlays(
        self,
        asset_id: str,
        selected: tuple[OverlayIssueInput, ...],
    ) -> Mapping[str, object]:
        """Remain non-blocking when process-level setup is still unavailable."""

        return self.get_asset_overlays(asset_id, selected)


class ProductionWorkerOverlayProvider(WorkerOverlayProvider):
    """Worker adapter that turns strict setup failures into terminal views."""

    def __init__(
        self,
        *,
        worker: BoundedOverlayWorker,
        request_factory: Callable[
            [str, tuple[OverlayIssueInput, ...]], OverlayRequest
        ],
        media_catalog: MediaCatalog | None = None,
        pin_lease_seconds: float | None = None,
    ) -> None:
        self._production_worker = worker
        self._production_request_factory = request_factory
        self._media_catalog = media_catalog
        self._pin_lease_seconds = pin_lease_seconds
        self._lease_lock = RLock()
        self._leased_requests: dict[str, OverlayRequest] = {}
        self._leased_overlay_ids: dict[str, set[str]] = {}
        super().__init__(worker=worker, request_factory=request_factory)

    def _renew_request(self, request: OverlayRequest) -> float:
        return self._production_worker.pin(
            request,
            lease_seconds=self._pin_lease_seconds,
        )

    def _release_request(
        self,
        asset_id: str,
        request: OverlayRequest,
        overlay_id: str,
    ) -> None:
        with self._lease_lock:
            current = self._leased_requests.get(asset_id)
            if current is not request:
                return
            remaining = self._leased_overlay_ids.get(asset_id)
            if remaining is not None:
                remaining.discard(overlay_id)
                if remaining:
                    return
                self._leased_overlay_ids.pop(asset_id, None)
            self._leased_requests.pop(asset_id, None)
        self._production_worker.unpin(request)

    def _publish_ready_lease(
        self,
        asset_id: str,
        request: OverlayRequest,
        view: object,
    ) -> None:
        catalog = self._media_catalog
        allow_overlay = getattr(catalog, "allow_overlay", None)
        if not callable(allow_overlay):
            return
        raw_segments = getattr(view, "segments", ())
        ready_segments = tuple(
            segment
            for segment in raw_segments
            if getattr(segment, "status", None) == "ready"
            and isinstance(getattr(segment, "overlay_id", None), str)
            and isinstance(getattr(segment, "path", None), Path)
        )
        if not ready_segments:
            return
        with self._lease_lock:
            previous = self._leased_requests.get(asset_id)
        if previous is not None and previous.cache_key != request.cache_key:
            release_asset = getattr(catalog, "release_asset_overlays", None)
            if callable(release_asset):
                release_asset(asset_id)
            self._production_worker.unpin(previous)
        expiry = self._renew_request(request)
        with self._lease_lock:
            self._leased_requests[asset_id] = request
            self._leased_overlay_ids[asset_id] = {
                str(segment.overlay_id) for segment in ready_segments
            }

        for segment in ready_segments:
            overlay_id = str(segment.overlay_id)
            path = Path(segment.path)
            allow_overlay(
                asset_id,
                overlay_id,
                path,
                lease_expires_at=expiry,
                renew_lease=lambda request=request: self._renew_request(request),
                release_lease=lambda asset_id=asset_id, request=request, overlay_id=overlay_id: self._release_request(
                    asset_id, request, overlay_id
                ),
            )

    def release_asset(self, asset_id: str) -> None:
        catalog = self._media_catalog
        release_asset = getattr(catalog, "release_asset_overlays", None)
        if callable(release_asset):
            release_asset(asset_id)
        with self._lease_lock:
            request = self._leased_requests.pop(asset_id, None)
            self._leased_overlay_ids.pop(asset_id, None)
        if request is not None:
            self._production_worker.unpin(request)

    def release_all(self) -> None:
        with self._lease_lock:
            asset_ids = tuple(self._leased_requests)
        for asset_id in asset_ids:
            self.release_asset(asset_id)

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
                    if getattr(view, "status", None) == "ready":
                        self._publish_ready_lease(asset_id, request, view)
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

    def retry_asset_overlays(
        self,
        asset_id: str,
        selected: tuple[OverlayIssueInput, ...],
    ) -> Mapping[str, object]:
        """Retry exactly the same asset-level union request without blocking."""

        try:
            request = self._production_request_factory(asset_id, selected)
        except OverlaySetupError as exc:
            return UnavailableOverlayProvider(exc.code).get_asset_overlays(
                asset_id, selected
            )

        worker = self._production_worker

        class AtomicRetryViewWorker:
            @staticmethod
            def retry(value: OverlayRequest) -> object:
                view = worker.retry(value)
                if getattr(view, "status", None) != "failed":
                    if getattr(view, "status", None) == "ready":
                        self._publish_ready_lease(asset_id, request, view)
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

            # The generic facade validates the narrow worker protocol at
            # construction time; retry remains the only operation invoked.
            submit = retry

        delegate = WorkerOverlayProvider(
            worker=AtomicRetryViewWorker(),
            request_factory=lambda requested_asset, requested_selected: request,
        )
        return delegate.retry_asset_overlays(asset_id, selected)


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
        release_all = getattr(self.provider, "release_all", None)
        if callable(release_all):
            release_all()
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
    freeze_sources = getattr(media_catalog, "freeze_sources", None)
    if callable(freeze_sources):
        try:
            freeze_sources()
        except MediaError:
            raise OverlaySetupError("overlay_source_unavailable") from None
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
        # Keep request construction metadata-only.  The provider constructor
        # validates compact recipe structure/path containment; hashing,
        # Parquet/sidecar materialization and decoder open happen in the
        # bounded worker's render_interval call.
        frame_provider = frame_provider_factory(context, source)
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
        media_catalog=media_catalog,
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
    "validate_overlay_recipe",
]
