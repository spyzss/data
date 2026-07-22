from __future__ import annotations

from dataclasses import replace
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
        "format": {
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "tags": {"major_brand": "isom"},
        },
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
    assert video.container_format == "mov,mp4,m4a,3gp,3g2,mj2"
    assert video.container_major_brand == "isom"
    assert video.timestamps_ns == (0, 33_366_667, 66_733_333)
    argv = observed["argv"]
    kwargs = observed["kwargs"]
    assert isinstance(argv, list)
    assert argv[-1] == str(path)
    assert "-show_frames" in argv
    show_entries = argv[argv.index("-show_entries") + 1]
    assert "format_tags=major_brand" in show_entries
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


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("width", 640.0),
        ("width", "640"),
        ("height", 480.0),
        ("height", "480"),
    ],
)
def test_probe_video_rejects_coerced_json_dimensions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    invalid: object,
) -> None:
    payload = _ffprobe_payload()
    streams = payload["streams"]
    assert isinstance(streams, list)
    streams[0][field] = invalid
    monkeypatch.setattr(
        "canonical_qc.video_probe.subprocess.run",
        lambda *_args, **_kwargs: _completed_probe(payload),
    )

    _assert_error(
        lambda: probe_video(tmp_path / "bad-dimension.mp4"),
        code="source_integrity_error",
        field=f"main_video.{field}_px",
    )


@pytest.mark.parametrize(
    "invalid",
    [True, 900_000.0, "900000.0", " 900000"],
)
def test_probe_video_rejects_invalid_present_best_effort_timestamp_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    invalid: object,
) -> None:
    payload = _ffprobe_payload()
    frames = payload["frames"]
    assert isinstance(frames, list)
    frames[0]["best_effort_timestamp"] = invalid
    assert frames[0]["best_effort_timestamp_time"] == "10.000000"
    monkeypatch.setattr(
        "canonical_qc.video_probe.subprocess.run",
        lambda *_args, **_kwargs: _completed_probe(payload),
    )

    _assert_error(
        lambda: probe_video(tmp_path / "bad-pts.mp4"),
        code="timebase_invalid",
        field="main_video.frames[0].best_effort_timestamp",
    )


def test_probe_video_accepts_exact_signed_decimal_integer_timestamp_strings(
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
        {"media_type": "video", "best_effort_timestamp": "-3"},
        {"media_type": "video", "best_effort_timestamp": "+0"},
    ]
    monkeypatch.setattr(
        "canonical_qc.video_probe.subprocess.run",
        lambda *_args, **_kwargs: _completed_probe(payload),
    )

    video = probe_video(tmp_path / "signed-pts.mp4")

    assert video.timestamps_ns == (0, 1_000_000_000)


@pytest.mark.parametrize("missing", [None, "N/A"])
def test_probe_video_uses_timestamp_time_only_when_integer_pts_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    missing: object,
) -> None:
    payload = _ffprobe_payload()
    frames = payload["frames"]
    assert isinstance(frames, list)
    frames[0]["best_effort_timestamp"] = missing
    frames[1]["best_effort_timestamp"] = missing
    frames[2]["best_effort_timestamp"] = missing
    monkeypatch.setattr(
        "canonical_qc.video_probe.subprocess.run",
        lambda *_args, **_kwargs: _completed_probe(payload),
    )

    video = probe_video(tmp_path / "fallback-pts.mp4")

    assert video.timestamps_ns == (0, 33_367_000, 66_733_000)


def test_probe_video_rejects_non_string_timestamp_time_without_coercion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _ffprobe_payload()
    frames = payload["frames"]
    assert isinstance(frames, list)
    frames[0]["best_effort_timestamp"] = None
    frames[0]["best_effort_timestamp_time"] = 10.0
    monkeypatch.setattr(
        "canonical_qc.video_probe.subprocess.run",
        lambda *_args, **_kwargs: _completed_probe(payload),
    )

    _assert_error(
        lambda: probe_video(tmp_path / "bad-timestamp-time.mp4"),
        code="timebase_invalid",
        field="main_video.frames[0].best_effort_timestamp_time",
    )


def test_probe_video_rejects_invalid_present_stream_time_base(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _ffprobe_payload()
    streams = payload["streams"]
    assert isinstance(streams, list)
    streams[0]["time_base"] = "0/0"
    monkeypatch.setattr(
        "canonical_qc.video_probe.subprocess.run",
        lambda *_args, **_kwargs: _completed_probe(payload),
    )

    _assert_error(
        lambda: probe_video(tmp_path / "bad-time-base.mp4"),
        code="timebase_invalid",
        field="main_video.time_base",
    )


def test_validate_video_alignment_normalizes_both_authoritative_timelines() -> None:
    time_axis = _time_axis(5_000_000_000, 5_033_366_667, 5_066_733_333)
    video = _video(10_000_000_000, 10_033_366_667, 10_066_733_333)

    validate_video_alignment(time_axis, video, max_delta_ns=0)


@pytest.mark.parametrize(
    ("video", "field"),
    [
        (_video(0, 33_366_667, frame_count=3), "main_video.timestamps_ns"),
        (_video(0, 33_366_667, frame_count=2), "main_video.frame_count"),
    ],
)
def test_validate_video_alignment_rejects_insufficient_frame_pts(
    video: ProbedVideo,
    field: str,
) -> None:
    error = _assert_error(
        lambda: validate_video_alignment(
            _time_axis(0, 33_366_667, 66_733_333),
            video,
            max_delta_ns=1_000,
        ),
        code="timebase_invalid",
        field=field,
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


def test_validate_video_alignment_rejects_scalar_canonical_timestamps_cleanly() -> None:
    time_axis = TimeAxis(
        frame_count=1,
        timestamps_ns=np.asarray(0, dtype=np.int64),
        fps_num=30000,
        fps_den=1001,
    )

    _assert_error(
        lambda: validate_video_alignment(time_axis, _video(0), max_delta_ns=0),
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
        code="timebase_invalid",
        field="time_axis.frame_count",
    )


@pytest.mark.parametrize(
    ("time_axis", "video", "field"),
    [
        (object(), _video(0), "time_axis"),
        (_time_axis(0), object(), "main_video"),
    ],
)
def test_validate_video_alignment_checks_exact_contract_types_before_dereference(
    time_axis: object,
    video: object,
    field: str,
) -> None:
    _assert_error(
        lambda: validate_video_alignment(  # type: ignore[arg-type]
            time_axis,
            video,
            max_delta_ns=0,
        ),
        code="invalid_contract_type",
        field=field,
    )


@pytest.mark.parametrize("invalid", [True, 0, -1])
def test_validate_video_alignment_rejects_invalid_probed_frame_count(
    invalid: object,
) -> None:
    video = replace(_video(0, 33_366_667, 66_733_333), frame_count=invalid)

    _assert_error(
        lambda: validate_video_alignment(
            _time_axis(0, 33_366_667, 66_733_333),
            video,
            max_delta_ns=0,
        ),
        code="timebase_invalid",
        field="main_video.frame_count",
    )


@pytest.mark.parametrize("invalid", [True, -1])
def test_validate_video_alignment_rejects_invalid_canonical_frame_count(
    invalid: object,
) -> None:
    time_axis = replace(_time_axis(0), frame_count=invalid)

    _assert_error(
        lambda: validate_video_alignment(time_axis, _video(0), max_delta_ns=0),
        code="timebase_invalid",
        field="time_axis.frame_count",
    )


@pytest.mark.parametrize(
    ("owner", "field", "invalid"),
    [
        ("time_axis", "fps_num", True),
        ("time_axis", "fps_num", 0),
        ("time_axis", "fps_den", -1),
        ("main_video", "fps_num", True),
        ("main_video", "fps_num", 0),
        ("main_video", "fps_den", -1),
    ],
)
def test_validate_video_alignment_rejects_invalid_rational_fps_components(
    owner: str,
    field: str,
    invalid: object,
) -> None:
    time_axis = _time_axis(0, 33_366_667, 66_733_333)
    video = _video(0, 33_366_667, 66_733_333)
    if owner == "time_axis":
        time_axis = replace(time_axis, **{field: invalid})
    else:
        video = replace(video, **{field: invalid})

    _assert_error(
        lambda: validate_video_alignment(time_axis, video, max_delta_ns=0),
        code="timebase_invalid",
        field=f"{owner}.{field}",
    )


def test_validate_video_alignment_rejects_rational_fps_mismatch() -> None:
    video = replace(_video(0, 33_366_667, 66_733_333), fps_num=25, fps_den=1)

    _assert_error(
        lambda: validate_video_alignment(
            _time_axis(0, 33_366_667, 66_733_333),
            video,
            max_delta_ns=0,
        ),
        code="timebase_invalid",
        field="main_video.fps",
    )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("width_px", True),
        ("width_px", 0),
        ("width_px", -1),
        ("width_px", 640.0),
        ("height_px", 0),
    ],
)
def test_validate_video_alignment_rejects_invalid_probed_dimensions(
    field: str,
    invalid: object,
) -> None:
    video = replace(_video(0, 33_366_667, 66_733_333), **{field: invalid})

    _assert_error(
        lambda: validate_video_alignment(
            _time_axis(0, 33_366_667, 66_733_333),
            video,
            max_delta_ns=0,
        ),
        code="source_integrity_error",
        field=f"main_video.{field}",
    )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("codec", ""),
        ("codec", "   "),
        ("pixel_format", ""),
        ("pixel_format", None),
    ],
)
def test_validate_video_alignment_rejects_invalid_probed_strings(
    field: str,
    invalid: object,
) -> None:
    video = replace(_video(0, 33_366_667, 66_733_333), **{field: invalid})

    _assert_error(
        lambda: validate_video_alignment(
            _time_axis(0, 33_366_667, 66_733_333),
            video,
            max_delta_ns=0,
        ),
        code="source_integrity_error",
        field=f"main_video.{field}",
    )


@pytest.mark.parametrize(
    "timestamps_ns",
    [
        (0, 33_366_667, 33_366_667),
        (0, 33_366_667, 20_000_000),
    ],
)
def test_validate_video_alignment_rejects_non_increasing_canonical_timestamps(
    timestamps_ns: tuple[int, ...],
) -> None:
    _assert_error(
        lambda: validate_video_alignment(
            _time_axis(*timestamps_ns),
            _video(0, 33_366_667, 66_733_333),
            max_delta_ns=100_000_000,
        ),
        code="timebase_invalid",
        field="time_axis.timestamps_ns",
    )


@pytest.mark.parametrize(
    "invalid",
    [1, -1, False, 0.0, np.float32(0.0)],
)
def test_validate_video_alignment_requires_exact_zero_frame_index_base(
    invalid: object,
) -> None:
    time_axis = replace(_time_axis(0), frame_index_base=invalid)

    _assert_error(
        lambda: validate_video_alignment(time_axis, _video(0), max_delta_ns=0),
        code="timebase_invalid",
        field="time_axis.frame_index_base",
    )


@pytest.mark.parametrize("invalid", ["closed", "", False, None])
def test_validate_video_alignment_requires_half_open_interval_semantics(
    invalid: object,
) -> None:
    time_axis = replace(_time_axis(0), interval_semantics=invalid)

    _assert_error(
        lambda: validate_video_alignment(time_axis, _video(0), max_delta_ns=0),
        code="timebase_invalid",
        field="time_axis.interval_semantics",
    )


class _IntervalString(str):
    pass


@pytest.mark.parametrize(
    "invalid",
    [
        np.str_("half_open"),
        ["half_open"],
        np.asarray("half_open"),
        np.asarray(["half_open"]),
        np.asarray(["half_open", "half_open"]),
        _IntervalString("half_open"),
    ],
    ids=["numpy-scalar", "list", "0d-array", "1d-one", "1d-many", "str-subclass"],
)
def test_validate_video_alignment_rejects_non_exact_string_interval_semantics(
    invalid: object,
) -> None:
    time_axis = replace(_time_axis(0), interval_semantics=invalid)

    _assert_error(
        lambda: validate_video_alignment(time_axis, _video(0), max_delta_ns=0),
        code="timebase_invalid",
        field="time_axis.interval_semantics",
    )
