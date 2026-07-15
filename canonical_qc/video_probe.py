"""Read authoritative per-frame video timing and stream metadata with ffprobe."""

from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import subprocess
from typing import Any

from .contracts import ProbedVideo
from .errors import CanonicalInputError


_FFPROBE_TIMEOUT_SECONDS = 30


def _fail(code: str, field: str, detail: str) -> None:
    raise CanonicalInputError(code, field, detail)


def _fraction(value: object, *, field: str) -> Fraction:
    if not isinstance(value, str):
        _fail("timebase_invalid", field, "must be an ffprobe rational string")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError):
        _fail("timebase_invalid", field, f"invalid ffprobe rational {value!r}")
    if result <= 0:
        _fail("timebase_invalid", field, "must be greater than zero")
    return result


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        _fail("source_integrity_error", field, "must be a positive integer")
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        _fail("source_integrity_error", field, "must be a positive integer")
    if result <= 0:
        _fail("source_integrity_error", field, "must be a positive integer")
    return result


def _nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("source_integrity_error", field, "must be a non-empty string")
    return value


def _timestamp_seconds(
    frame: dict[str, Any], time_base: Fraction | None
) -> Fraction | None:
    raw_timestamp = frame.get("best_effort_timestamp")
    if time_base is not None and raw_timestamp not in (None, "N/A"):
        try:
            timestamp = int(raw_timestamp)
        except (TypeError, ValueError, OverflowError):
            timestamp = None
        if timestamp is not None:
            return timestamp * time_base

    raw_seconds = frame.get("best_effort_timestamp_time")
    if raw_seconds in (None, "N/A"):
        return None
    try:
        return Fraction(str(raw_seconds))
    except (ValueError, ZeroDivisionError):
        return None


def _probe_payload(path: Path) -> dict[str, Any]:
    argv = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "stream=codec_name,codec_type,width,height,pix_fmt,avg_frame_rate,"
            "time_base:frame=media_type,best_effort_timestamp,"
            "best_effort_timestamp_time"
        ),
        "-show_streams",
        "-show_frames",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            argv,
            shell=False,
            capture_output=True,
            text=True,
            timeout=_FFPROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError as exc:
        _fail(
            "source_integrity_error",
            "main_video.path",
            f"ffprobe unavailable: {exc}",
        )
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr
        if isinstance(stderr, bytes):
            stderr_text = stderr.decode("utf-8", errors="replace").strip()
        elif isinstance(stderr, str):
            stderr_text = stderr.strip()
        else:
            stderr_text = ""
        detail = f"ffprobe timed out after {_FFPROBE_TIMEOUT_SECONDS} seconds"
        if stderr_text:
            detail = f"{detail}: {stderr_text}"
        _fail(
            "source_integrity_error",
            "main_video.path",
            detail,
        )
    except (OSError, UnicodeError) as exc:
        _fail(
            "source_integrity_error",
            "main_video.path",
            f"ffprobe execution failed: {exc}",
        )

    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "no stderr"
        _fail(
            "source_integrity_error",
            "main_video.path",
            f"ffprobe exited with status {completed.returncode}: {stderr}",
        )
    try:
        payload = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        _fail(
            "source_integrity_error",
            "main_video.path",
            f"ffprobe returned invalid JSON: {exc}",
        )
    if not isinstance(payload, dict):
        _fail(
            "source_integrity_error",
            "main_video.path",
            "ffprobe JSON root must be an object",
        )
    return payload


def probe_video(path: Path) -> ProbedVideo:
    """Probe one video without hashing it or fabricating missing frame timestamps."""

    payload = _probe_payload(path)
    streams = payload.get("streams")
    frames = payload.get("frames")
    if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
        _fail(
            "source_integrity_error",
            "main_video.path",
            "ffprobe returned no video stream",
        )
    if not isinstance(frames, list):
        _fail(
            "source_integrity_error",
            "main_video.path",
            "ffprobe returned no frame list",
        )

    stream = streams[0]
    fps = _fraction(stream.get("avg_frame_rate"), field="main_video.fps")
    raw_time_base = stream.get("time_base")
    time_base: Fraction | None = None
    if isinstance(raw_time_base, str):
        try:
            candidate = Fraction(raw_time_base)
        except (ValueError, ZeroDivisionError):
            candidate = Fraction(0, 1)
        if candidate > 0:
            time_base = candidate

    video_frames: list[dict[str, Any]] = []
    for frame in frames:
        if not isinstance(frame, dict):
            _fail(
                "source_integrity_error",
                "main_video.path",
                "ffprobe frame entries must be objects",
            )
        if frame.get("media_type", "video") == "video":
            video_frames.append(frame)

    absolute_timestamps = [
        timestamp
        for frame in video_frames
        if (timestamp := _timestamp_seconds(frame, time_base)) is not None
    ]
    if absolute_timestamps:
        first_timestamp = absolute_timestamps[0]
        timestamps_ns = tuple(
            round((timestamp - first_timestamp) * 1_000_000_000)
            for timestamp in absolute_timestamps
        )
    else:
        timestamps_ns = ()

    return ProbedVideo(
        frame_count=len(video_frames),
        width_px=_positive_int(stream.get("width"), field="main_video.width_px"),
        height_px=_positive_int(stream.get("height"), field="main_video.height_px"),
        fps_num=fps.numerator,
        fps_den=fps.denominator,
        codec=_nonempty_string(stream.get("codec_name"), field="main_video.codec"),
        pixel_format=_nonempty_string(
            stream.get("pix_fmt"), field="main_video.pixel_format"
        ),
        timestamps_ns=timestamps_ns,
    )
