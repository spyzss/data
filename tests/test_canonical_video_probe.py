from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from canonical_qc import (
    CanonicalInputError,
    ProbedVideo,
    TimeAxis,
    probe_video,
    validate_video_alignment,
)


def _completed_probe(
    payload: object,
    *,
    returncode: int = 0,
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["ffprobe"],
        returncode=returncode,
        stdout=json.dumps(payload),
        stderr=stderr,
    )


def _ffprobe_payload() -> dict[str, object]:
    """A complete subset of real ``ffprobe -show_streams -show_frames`` JSON."""

    return {
        "frames": [
            {
                "media_type": "video",
                "best_effort_timestamp": "900000",
                "best_effort_timestamp_time": "10.000000",
            },
            {
                "media_type": "video",
                "best_effort_timestamp": "903003",
                "best_effort_timestamp_time": "10.033367",
            },
            {
                "media_type": "video",
                "best_effort_timestamp": "906006",
                "best_effort_timestamp_time": "10.066733",
            },
        ],
        "streams": [
            {
                "codec_name": "h264",
                "codec_type": "video",
                "width": 640,
                "height": 480,
                "pix_fmt": "yuv420p",
                "avg_frame_rate": "60000/2002",
                "time_base": "1/90000",
            }
        ],
    }


def _time_axis(*timestamps_ns: int) -> TimeAxis:
    return TimeAxis(
        frame_count=len(timestamps_ns),
        timestamps_ns=np.asarray(timestamps_ns, dtype=np.int64),
        fps_num=30000,
        fps_den=1001,
    )


def _video(*timestamps_ns: int, frame_count: int | None = None) -> ProbedVideo:
    return ProbedVideo(
        frame_count=len(timestamps_ns) if frame_count is None else frame_count,
        width_px=640,
        height_px=480,
        fps_num=30000,
        fps_den=1001,
        codec="h264",
        pixel_format="yuv420p",
        timestamps_ns=timestamps_ns,
    )


def _assert_error(
    action: object,
    *,
    code: str,
    field: str,
) -> CanonicalInputError:
    assert callable(action)
    with pytest.raises(CanonicalInputError) as raised:
        action()
    assert raised.value.code == code
    assert raised.value.field == field
    return raised.value


def test_probe_video_reads_metadata_rational_fps_and_normalized_pts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed["argv"] = argv
        observed["kwargs"] = kwargs
        return _completed_probe(_ffprobe_payload())

    monkeypatch.setattr("canonical_qc.video_probe.subprocess.run", fake_run)
    path = tmp_path / "clip;touch-not-executed.mp4"

    video = probe_video(path)

    assert video.frame_count == 3
    assert (video.width_px, video.height_px) == (640, 480)
    assert (video.fps_num, video.fps_den) == (30000, 1001)
    assert video.codec == "h264"
    assert video.pixel_format == "yuv420p"
    assert video.timestamps_ns == (0, 33_366_667, 66_733_333)
    argv = observed["argv"]
    kwargs = observed["kwargs"]
    assert isinstance(argv, list)
    assert argv[-1] == str(path)
    assert "-show_frames" in argv
    assert kwargs["shell"] is False
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert isinstance(kwargs["timeout"], (int, float))
    assert kwargs["timeout"] > 0


def test_probe_video_never_synthesizes_a_missing_pts_from_fps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _ffprobe_payload()
    frames = payload["frames"]
    assert isinstance(frames, list)
    frames[1] = {"media_type": "video"}
    monkeypatch.setattr(
        "canonical_qc.video_probe.subprocess.run",
        lambda *_args, **_kwargs: _completed_probe(payload),
    )

    video = probe_video(tmp_path / "missing-pts.mp4")

    assert video.frame_count == 3
    assert video.timestamps_ns == (0, 66_733_333)


def test_probe_video_accepts_best_effort_timestamp_time_without_float_rounding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _ffprobe_payload()
    stream = payload["streams"]
    frames = payload["frames"]
    assert isinstance(stream, list)
    assert isinstance(frames, list)
    stream[0].pop("time_base")
    for frame in frames:
        frame.pop("best_effort_timestamp")
    monkeypatch.setattr(
        "canonical_qc.video_probe.subprocess.run",
        lambda *_args, **_kwargs: _completed_probe(payload),
    )

    video = probe_video(tmp_path / "timestamp-time.mp4")

    assert video.timestamps_ns == (0, 33_367_000, 66_733_000)


def test_probe_video_normalizes_pts_before_rounding_to_integer_nanoseconds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _ffprobe_payload()
    stream = payload["streams"]
    frames = payload["frames"]
    assert isinstance(stream, list)
    assert isinstance(frames, list)
    stream[0]["time_base"] = "1/3"
    frames[:] = [
        {"media_type": "video", "best_effort_timestamp": "1"},
        {"media_type": "video", "best_effort_timestamp": "2"},
    ]
    monkeypatch.setattr(
        "canonical_qc.video_probe.subprocess.run",
        lambda *_args, **_kwargs: _completed_probe(payload),
    )

    video = probe_video(tmp_path / "fractional-nanoseconds.mp4")

    assert video.timestamps_ns == (0, 333_333_333)


@pytest.mark.parametrize(
    ("failure", "detail"),
    [
        (FileNotFoundError("ffprobe"), "unavailable"),
        (
            subprocess.TimeoutExpired(cmd=["ffprobe"], timeout=30),
            "timed out",
        ),
    ],
)
def test_probe_video_distinguishes_tool_launch_failures_from_bad_timebase(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: OSError | subprocess.TimeoutExpired,
    detail: str,
) -> None:
    def fail_run(*_args: object, **_kwargs: object) -> None:
        raise failure

    monkeypatch.setattr("canonical_qc.video_probe.subprocess.run", fail_run)

    error = _assert_error(
        lambda: probe_video(tmp_path / "clip.mp4"),
        code="source_integrity_error",
        field="main_video.path",
    )
    assert detail in error.detail


def test_probe_video_preserves_timeout_stderr_in_runtime_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def timeout(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired(
            cmd=["ffprobe"],
            timeout=30,
            stderr="partial decoder diagnostic",
        )

    monkeypatch.setattr("canonical_qc.video_probe.subprocess.run", timeout)

    error = _assert_error(
        lambda: probe_video(tmp_path / "slow.mp4"),
        code="source_integrity_error",
        field="main_video.path",
    )
    assert "partial decoder diagnostic" in error.detail


def test_probe_video_reports_invalid_rational_fps_as_bad_timebase(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _ffprobe_payload()
    streams = payload["streams"]
    assert isinstance(streams, list)
    streams[0]["avg_frame_rate"] = "0/0"
    monkeypatch.setattr(
        "canonical_qc.video_probe.subprocess.run",
        lambda *_args, **_kwargs: _completed_probe(payload),
    )

    _assert_error(
        lambda: probe_video(tmp_path / "bad-fps.mp4"),
        code="timebase_invalid",
        field="main_video.fps",
    )


def test_probe_video_reports_nonzero_ffprobe_with_captured_stderr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "canonical_qc.video_probe.subprocess.run",
        lambda *_args, **_kwargs: _completed_probe(
            {}, returncode=1, stderr="moov atom not found"
        ),
    )

    error = _assert_error(
        lambda: probe_video(tmp_path / "broken.mp4"),
        code="source_integrity_error",
        field="main_video.path",
    )
    assert "moov atom not found" in error.detail


def test_probe_video_reports_invalid_ffprobe_json_as_tool_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "canonical_qc.video_probe.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["ffprobe"], returncode=0, stdout="not-json", stderr=""
        ),
    )

    _assert_error(
        lambda: probe_video(tmp_path / "clip.mp4"),
        code="source_integrity_error",
        field="main_video.path",
    )


def test_validate_video_alignment_normalizes_both_authoritative_timelines() -> None:
    time_axis = _time_axis(5_000_000_000, 5_033_366_667, 5_066_733_333)
    video = _video(10_000_000_000, 10_033_366_667, 10_066_733_333)

    validate_video_alignment(time_axis, video, max_delta_ns=0)


@pytest.mark.parametrize(
    "video",
    [
        _video(0, 33_366_667, frame_count=3),
        _video(0, 33_366_667, frame_count=2),
    ],
)
def test_validate_video_alignment_rejects_insufficient_frame_pts(
    video: ProbedVideo,
) -> None:
    error = _assert_error(
        lambda: validate_video_alignment(
            _time_axis(0, 33_366_667, 66_733_333),
            video,
            max_delta_ns=1_000,
        ),
        code="timebase_invalid",
        field="main_video.timestamps_ns",
    )
    assert "expected 3" in error.detail


@pytest.mark.parametrize(
    "timestamps_ns",
    [
        (0, 33_366_667, 33_366_667),
        (0, 33_366_667, 20_000_000),
    ],
)
def test_validate_video_alignment_rejects_non_increasing_pts(
    timestamps_ns: tuple[int, ...],
) -> None:
    _assert_error(
        lambda: validate_video_alignment(
            _time_axis(0, 33_366_667, 66_733_333),
            _video(*timestamps_ns),
            max_delta_ns=1_000,
        ),
        code="timebase_invalid",
        field="main_video.timestamps_ns",
    )


def test_validate_video_alignment_rejects_delta_above_explicit_tolerance() -> None:
    error = _assert_error(
        lambda: validate_video_alignment(
            _time_axis(4_000_000_000, 4_033_366_667, 4_066_733_333),
            _video(7_000_000_000, 7_033_366_667, 7_066_735_334),
            max_delta_ns=2_000,
        ),
        code="timebase_invalid",
        field="main_video.timestamps_ns[2]",
    )
    assert "2001 ns" in error.detail


def test_validate_video_alignment_accepts_delta_at_tolerance_boundary() -> None:
    validate_video_alignment(
        _time_axis(4_000_000_000, 4_033_366_667, 4_066_733_333),
        _video(7_000_000_000, 7_033_366_667, 7_066_735_333),
        max_delta_ns=2_000,
    )


@pytest.mark.parametrize("invalid", [33_366_667.9, "33366667", True])
def test_validate_video_alignment_never_coerces_video_pts_to_integer(
    invalid: object,
) -> None:
    video = _video(0, invalid, 66_733_333)  # type: ignore[arg-type]

    _assert_error(
        lambda: validate_video_alignment(
            _time_axis(0, 33_366_667, 66_733_333),
            video,
            max_delta_ns=1_000,
        ),
        code="timebase_invalid",
        field="main_video.timestamps_ns[1]",
    )


def test_validate_video_alignment_never_coerces_canonical_pts_to_integer() -> None:
    time_axis = TimeAxis(
        frame_count=3,
        timestamps_ns=np.asarray([0.0, 33_366_667.9, 66_733_333.0]),
        fps_num=30000,
        fps_den=1001,
    )

    _assert_error(
        lambda: validate_video_alignment(
            time_axis,
            _video(0, 33_366_667, 66_733_333),
            max_delta_ns=1_000,
        ),
        code="timebase_invalid",
        field="time_axis.timestamps_ns[0]",
    )


@pytest.mark.parametrize("max_delta_ns", [-1, 1.5, True])
def test_validate_video_alignment_requires_nonnegative_integer_tolerance(
    max_delta_ns: object,
) -> None:
    _assert_error(
        lambda: validate_video_alignment(
            _time_axis(0, 33_366_667, 66_733_333),
            _video(0, 33_366_667, 66_733_333),
            max_delta_ns=max_delta_ns,  # type: ignore[arg-type]
        ),
        code="invalid_integer",
        field="max_delta_ns",
    )


def test_validate_video_alignment_rejects_canonical_timestamp_count_mismatch() -> None:
    time_axis = TimeAxis(
        frame_count=3,
        timestamps_ns=np.asarray([0, 33_366_667], dtype=np.int64),
        fps_num=30000,
        fps_den=1001,
    )

    _assert_error(
        lambda: validate_video_alignment(
            time_axis,
            _video(0, 33_366_667, 66_733_333),
            max_delta_ns=1_000,
        ),
        code="timebase_invalid",
        field="time_axis.timestamps_ns",
    )


def test_validate_video_alignment_rejects_empty_canonical_time_axis_cleanly() -> None:
    time_axis = TimeAxis(
        frame_count=0,
        timestamps_ns=np.asarray([], dtype=np.int64),
        fps_num=30000,
        fps_den=1001,
    )
    video = _video(frame_count=0)

    _assert_error(
        lambda: validate_video_alignment(time_axis, video, max_delta_ns=1_000),
        code="invalid_integer",
        field="time_axis.frame_count",
    )
