"""Read authoritative per-frame video timing and stream metadata with ffprobe."""

from __future__ import annotations

from fractions import Fraction
import json
from numbers import Integral
from pathlib import Path
import re
import subprocess
from typing import Any

from .contracts import ProbedVideo
from .errors import CanonicalInputError


_FFPROBE_TIMEOUT_SECONDS = 30
_SIGNED_INTEGER = re.compile(r"[+-]?\d+").fullmatch
_DECIMAL_SECONDS = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)").fullmatch


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
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        _fail("source_integrity_error", field, "must be a positive integer")
    return int(value)


def _nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("source_integrity_error", field, "must be a non-empty string")
    return value


def _timestamp_seconds(
    frame: dict[str, Any],
    time_base: Fraction | None,
    *,
    frame_index: int,
) -> Fraction | None:
    raw_timestamp = frame.get("best_effort_timestamp")
    if raw_timestamp not in (None, "N/A"):
        field = f"main_video.frames[{frame_index}].best_effort_timestamp"
        if isinstance(raw_timestamp, bool):
            _fail("timebase_invalid", field, "must be an exact integer PTS")
        if isinstance(raw_timestamp, Integral):
            timestamp = int(raw_timestamp)
        elif isinstance(raw_timestamp, str) and _SIGNED_INTEGER(raw_timestamp):
            timestamp = int(raw_timestamp)
        else:
            _fail("timebase_invalid", field, "must be an exact integer PTS")
        if time_base is not None:
            return timestamp * time_base

    raw_seconds = frame.get("best_effort_timestamp_time")
    if raw_seconds in (None, "N/A"):
        return None
    field = f"main_video.frames[{frame_index}].best_effort_timestamp_time"
    if not isinstance(raw_seconds, str) or _DECIMAL_SECONDS(raw_seconds) is None:
        _fail("timebase_invalid", field, "must be an exact decimal seconds string")
    try:
        return Fraction(raw_seconds)
    except (ValueError, ZeroDivisionError):
        _fail("timebase_invalid", field, "must be an exact decimal seconds string")


def _probe_payload(path: Path, *, pass_fds: tuple[int, ...] = ()) -> dict[str, Any]:
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
            pass_fds=pass_fds,
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


def probe_video(path: Path, *, file_descriptor: int | None = None) -> ProbedVideo:
    """Probe one video without hashing it or fabricating missing frame timestamps."""

    if file_descriptor is None:
        probe_path = Path(path)
        pass_fds: tuple[int, ...] = ()
    else:
        probe_path = Path(f"/dev/fd/{file_descriptor}")
        pass_fds = (file_descriptor,)
    payload = _probe_payload(probe_path, pass_fds=pass_fds)
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
    if raw_time_base not in (None, "N/A"):
        time_base = _fraction(raw_time_base, field="main_video.time_base")

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

    absolute_timestamps: list[Fraction] = []
    for frame_index, frame in enumerate(video_frames):
        timestamp = _timestamp_seconds(
            frame,
            time_base,
            frame_index=frame_index,
        )
        if timestamp is not None:
            absolute_timestamps.append(timestamp)
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
