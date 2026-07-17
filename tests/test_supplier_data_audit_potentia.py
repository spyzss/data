from __future__ import annotations

import json
from pathlib import Path

import pytest

from acceptance_pull.supplier_audit import audit_supplier_data
from acceptance_pull.supplier_adapters.structured_audit import audit_csv_timeline
from qc_pipeline.context import AssetContext
from tests.fixtures import solid_frame, write_test_video


def write_sources(tmp_path: Path) -> AssetContext:
    source = tmp_path / "source"
    source.mkdir()
    write_test_video(
        source / "video.mp4",
        [solid_frame(value, width=192, height=144) for value in (40, 80, 120)],
        fps=60.0,
    )
    (source / "meta.json").write_text(
        json.dumps({"task": {"name": "pick"}, "qc": {"score": 0}}),
        encoding="utf-8",
    )
    (source / "frames.csv").write_text(
        "frame_index,timestamp\n1,0.0\n2,0.0166667\n3,0.0333334\n",
        encoding="utf-8",
    )
    (source / "aligned.csv").write_text(
        "frame_index,timestamp,value\n1,0.0,1\n2,0.0166667,2\n3,0.0333334,3\n",
        encoding="utf-8",
    )
    (source / "imu.csv").write_text(
        "timestamp,ax,ay\n0.0,1.0,2.0\n0.01,1.1,2.1\n0.02,1.2,2.2\n0.20,1.3,2.3\n",
        encoding="utf-8",
    )
    (source / "calibration.json").write_text(
        json.dumps(
            {
                "capture": {
                    "width": 192,
                    "height": 108,
                    "K": [[100, 0, 96], [0, 100, 54], [0, 0, 1]],
                },
                "scaled": {
                    "width": 192,
                    "height": 144,
                    "K": [[100, 0, 96], [0, 133.333, 72], [0, 0, 1]],
                },
            }
        ),
        encoding="utf-8",
    )
    return AssetContext(
        "potentia__task-a",
        tmp_path,
        tmp_path / "quality_archive" / "potentia__task-a.json",
        source_files={
            name: {"path": f"source/{filename}"}
            for name, filename in {
                "video": "video.mp4",
                "meta": "meta.json",
                "frames": "frames.csv",
                "aligned": "aligned.csv",
                "imu": "imu.csv",
                "calibration": "calibration.json",
            }.items()
        },
        metadata={"supplier": "potentia"},
    )


def mapping() -> dict[str, object]:
    return {
        "suppliers": {
            "potentia": {
                "mapping_status": "verified",
                "max_timestamp_gap_factor": 3.0,
                "mapping": {
                    "meta": {
                        "task_path": "task.name",
                        "qc_path": "qc.score",
                    },
                    "frames": {
                        "frame_index_column": "frame_index",
                        "timestamp_column": "timestamp",
                        "timestamp_unit": "s",
                    },
                    "aligned": {
                        "frame_index_column": "frame_index",
                        "timestamp_column": "timestamp",
                        "timestamp_unit": "s",
                        "required_columns": ["value"],
                    },
                    "imu": {
                        "timestamp_column": "timestamp",
                        "timestamp_unit": "s",
                        "numeric_columns": ["ax", "ay"],
                    },
                    "calibration": {
                        "raw_intrinsics_matrix_path": "capture.K",
                        "raw_width_path": "capture.width",
                        "raw_height_path": "capture.height",
                        "scaled_intrinsics_matrix_path": "scaled.K",
                        "scaled_width_path": "scaled.width",
                        "scaled_height_path": "scaled.height",
                    },
                },
            }
        }
    }


def test_potentia_preserves_one_based_frames_and_both_fps_sources(
    tmp_path: Path,
) -> None:
    raw = audit_supplier_data(write_sources(tmp_path), mapping())

    frames = raw["structured"]["frames"]
    assert frames["frame_index_min"] == 1
    assert frames["frame_index_start"] == 1
    assert frames["frame_index_monotonic"] is True
    assert frames["frame_index_duplicate_count"] == 0
    assert frames["frame_index_gap_count"] == 0
    assert frames["timestamp_derived_fps"] == pytest.approx(60.0, rel=1e-4)
    assert raw["structured"]["video_metadata"] == {
        "status": "pass",
        "frame_count": 3,
        "width": 192,
        "height": 144,
        "fps": pytest.approx(60.0),
        "duration_sec": pytest.approx(0.05),
    }
    assert raw["structured"]["video_timeline_comparison"] == {
        "video_fps": pytest.approx(60.0),
        "timestamp_derived_fps": pytest.approx(60.0, rel=1e-4),
        "fps_delta": pytest.approx(0.0, abs=1e-3),
        "fps_relative_delta": pytest.approx(0.0, abs=1e-3),
        "effective_fps": pytest.approx(60.0),
        "effective_fps_source": "video_metadata",
    }
    assert raw["structured"]["alignment"] == {
        "status": "pass",
        "frames_vs_aligned_row_delta": 0,
        "video_vs_frames_row_delta": 0,
        "aligned_covers_frame_timeline": True,
    }


def test_potentia_audits_imu_calibration_and_supplier_signal_without_hard_fail(
    tmp_path: Path,
) -> None:
    raw = audit_supplier_data(write_sources(tmp_path), mapping())

    imu = raw["structured"]["imu"]
    calibration = raw["structured"]["calibration"]
    assert imu["sampling_rate_hz"] == pytest.approx(100.0)
    assert imu["timestamp_gap_count"] == 1
    assert imu["covers_frame_timeline"] is True
    assert imu["nonfinite_value_count"] == 0
    assert calibration["raw_resolution"] == [192, 108]
    assert calibration["scaled_resolution"] == [192, 144]
    assert calibration["video_resolution"] == [192, 144]
    assert calibration["selected_intrinsics"] == "scaled"
    assert calibration["resolution_scale"] == pytest.approx([1.0, 4.0 / 3.0])
    assert calibration["scaling_interpretable"] is True
    assert calibration["scaled_intrinsics_max_abs_error"] == pytest.approx(
        1.0 / 3000.0,
        rel=1e-3,
    )
    assert raw["supplier_quality_signal"] == {
        "status": "provided",
        "value": 0,
        "source": "meta.qc",
    }
    assert not any(
        issue["code"] == "supplier_quality_signal"
        and issue["severity"] == "fail"
        for issue in raw["issues"]
    )


def test_potentia_reports_duplicate_gap_and_nonmonotonic_timestamps(
    tmp_path: Path,
) -> None:
    context = write_sources(tmp_path)
    (tmp_path / "source" / "frames.csv").write_text(
        "frame_index,timestamp\n1,0.0\n2,0.02\n2,0.02\n5,0.01\n",
        encoding="utf-8",
    )

    raw = audit_supplier_data(context, mapping())
    frames = raw["structured"]["frames"]

    assert frames["frame_index_duplicate_count"] == 1
    assert frames["frame_index_gap_count"] == 1
    assert frames["timestamp_duplicate_count"] == 1
    assert frames["timestamp_monotonic"] is False
    assert raw["decision"] == "fail"
    assert any(issue["code"] == "timeline_invalid" for issue in raw["issues"])


def test_potentia_invalid_meta_is_a_structure_failure(tmp_path: Path) -> None:
    context = write_sources(tmp_path)
    (tmp_path / "source" / "meta.json").write_text("{broken", encoding="utf-8")

    raw = audit_supplier_data(context, mapping())

    assert raw["structured"]["meta"]["status"] == "invalid"
    assert any(issue["code"] == "structure_invalid" for issue in raw["issues"])
    assert not any(
        issue["code"] == "timeline_invalid" and issue["source_name"] == "meta"
        for issue in raw["issues"]
    )


def test_potentia_structured_metrics_and_supplier_signal_reach_module_result(
    tmp_path: Path,
) -> None:
    from qc_common.config import load_qc_acceptance_config
    from qc_pipeline.adapters.supplier_data_audit import adapt_supplier_data_audit

    raw = audit_supplier_data(write_sources(tmp_path), mapping())
    result = adapt_supplier_data_audit(raw, load_qc_acceptance_config())

    assert result.metrics["structured"]["frames"]["frame_index_start"] == 1
    assert result.metrics["supplier_quality_signal"]["value"] == 0


def test_potentia_missing_video_is_input_failure_without_crashing_mapped_audit(
    tmp_path: Path,
) -> None:
    context = write_sources(tmp_path)
    (tmp_path / "source" / "video.mp4").unlink()

    raw = audit_supplier_data(context, mapping())

    assert raw["decision"] == "fail"
    assert raw["structured"]["video_metadata"] == {
        "status": "input_missing",
        "reason": "required_source_missing",
    }
    assert "video" in raw["missing_sources"]


def test_potentia_unverified_mapping_still_records_structural_evidence(
    tmp_path: Path,
) -> None:
    raw = audit_supplier_data(
        write_sources(tmp_path),
        {
            "suppliers": {
                "potentia": {"mapping_status": "unverified", "mapping": {}}
            }
        },
    )

    assert raw["structured"]["video_metadata"]["status"] == "pass"
    assert raw["structured"]["meta_structure"]["status"] == "pass"
    assert raw["structured"]["frames_structure"]["row_count"] == 3
    assert raw["structured"]["calibration_structure"]["status"] == "pass"
    assert raw["structured"]["frames"]["status"] == "unverified"
    assert raw["structured"]["frames"]["reason"] == "mapping_missing"


@pytest.mark.parametrize(
    ("unit", "timestamps"),
    [
        ("s", [0.0, 1.0 / 60.0, 2.0 / 60.0]),
        ("ms", [0.0, 1000.0 / 60.0, 2000.0 / 60.0]),
        ("us", [0.0, 1_000_000.0 / 60.0, 2_000_000.0 / 60.0]),
        ("ns", [0.0, 1_000_000_000.0 / 60.0, 2_000_000_000.0 / 60.0]),
    ],
)
def test_csv_timeline_units_normalize_to_identical_seconds_and_fps(
    tmp_path: Path,
    unit: str,
    timestamps: list[float],
) -> None:
    path = tmp_path / f"timeline-{unit}.csv"
    path.write_text(
        "frame_index,timestamp\n"
        + "".join(
            f"{index},{timestamp}\n"
            for index, timestamp in enumerate(timestamps)
        ),
        encoding="utf-8",
    )

    result = audit_csv_timeline(
        path,
        {
            "frame_index_column": "frame_index",
            "timestamp_column": "timestamp",
            "timestamp_unit": unit,
        },
    )

    assert result["status"] == "pass"
    assert result["timestamp_unit"] == unit
    assert result["timestamp_scale_to_seconds"] == pytest.approx(
        {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}[unit]
    )
    assert result["normalized_timestamp_min_sec"] == pytest.approx(0.0)
    assert result["normalized_timestamp_max_sec"] == pytest.approx(2.0 / 60.0)
    assert result["timestamp_median_interval_sec"] == pytest.approx(1.0 / 60.0)
    assert result["timestamp_derived_fps"] == pytest.approx(60.0)


def test_csv_timeline_accepts_explicit_numeric_scale(tmp_path: Path) -> None:
    path = tmp_path / "timeline.csv"
    path.write_text(
        "frame_index,timestamp\n0,0\n1,1000\n2,2000\n",
        encoding="utf-8",
    )

    result = audit_csv_timeline(
        path,
        {
            "frame_index_column": "frame_index",
            "timestamp_column": "timestamp",
            "timestamp_scale_to_seconds": 0.001,
        },
    )

    assert result["status"] == "pass"
    assert result["timestamp_unit"] == "explicit_scale"
    assert result["timestamp_scale_to_seconds"] == pytest.approx(0.001)
    assert result["normalized_timestamp_max_sec"] == pytest.approx(2.0)
    assert result["timestamp_derived_fps"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("mapping_override", "reason"),
    [
        ({}, "timestamp_unit_unverified"),
        ({"timestamp_unit": "minutes"}, "timestamp_mapping_invalid"),
        (
            {"timestamp_unit": "ms", "timestamp_scale_to_seconds": 1.0},
            "timestamp_mapping_invalid",
        ),
    ],
)
def test_csv_timeline_requires_one_explicit_valid_timestamp_scale(
    tmp_path: Path,
    mapping_override: dict[str, object],
    reason: str,
) -> None:
    path = tmp_path / "timeline.csv"
    path.write_text(
        "frame_index,timestamp\n0,0\n1,1\n",
        encoding="utf-8",
    )

    result = audit_csv_timeline(
        path,
        {
            "frame_index_column": "frame_index",
            "timestamp_column": "timestamp",
            **mapping_override,
        },
    )

    assert result["status"] in {"unverified", "mapping_invalid"}
    assert result["reason"] == reason
    assert result.get("timestamp_derived_fps") is None
    assert result.get("timestamp_median_interval_sec") is None


def test_potentia_uninterpretable_scaling_defaults_to_review(tmp_path: Path) -> None:
    context = write_sources(tmp_path)
    (tmp_path / "source" / "imu.csv").write_text(
        "timestamp,ax,ay\n0.0,1.0,2.0\n0.01,1.1,2.1\n0.02,1.2,2.2\n0.03,1.3,2.3\n",
        encoding="utf-8",
    )
    calibration_path = tmp_path / "source" / "calibration.json"
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    payload["scaled"]["K"][0][0] = 40.0
    calibration_path.write_text(json.dumps(payload), encoding="utf-8")

    raw = audit_supplier_data(context, mapping())
    calibration = raw["structured"]["calibration"]

    assert calibration["status"] == "unverified"
    assert calibration["reason"] == "scaling_unverified"
    assert calibration["scaling_interpretable"] is False
    assert calibration["raw_resolution"] == [192, 108]
    assert calibration["scaled_resolution"] == [192, 144]
    assert calibration["video_resolution"] == [192, 144]
    assert calibration["selected_intrinsics"] == "scaled"
    assert raw["decision"] == "warn"
    assert not any(issue["code"] == "calibration_invalid" for issue in raw["issues"])


def test_potentia_scaling_mismatch_only_fails_when_explicitly_configured(
    tmp_path: Path,
) -> None:
    context = write_sources(tmp_path)
    calibration_path = tmp_path / "source" / "calibration.json"
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    payload["scaled"]["K"][0][0] = 40.0
    calibration_path.write_text(json.dumps(payload), encoding="utf-8")
    parameters = mapping()
    parameters["suppliers"]["potentia"]["scaling_mismatch_action"] = "fail"

    raw = audit_supplier_data(context, parameters)

    assert raw["structured"]["calibration"]["status"] == "invalid"
    assert raw["decision"] == "fail"
    assert any(issue["code"] == "calibration_invalid" for issue in raw["issues"])


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("matrix_shape", "intrinsics_matrix_invalid"),
        ("nonpositive_focal", "intrinsics_parameters_invalid"),
    ],
)
def test_potentia_structurally_invalid_intrinsics_fail(
    tmp_path: Path,
    mutation: str,
    reason: str,
) -> None:
    context = write_sources(tmp_path)
    calibration_path = tmp_path / "source" / "calibration.json"
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    if mutation == "matrix_shape":
        payload["scaled"]["K"] = [[1, 0], [0, 1]]
    else:
        payload["scaled"]["K"][0][0] = -1.0
    calibration_path.write_text(json.dumps(payload), encoding="utf-8")

    raw = audit_supplier_data(context, mapping())

    assert raw["structured"]["calibration"]["status"] == "invalid"
    assert raw["structured"]["calibration"]["reason"] == reason
    assert raw["decision"] == "fail"


def test_potentia_missing_video_keeps_calibration_unverified_without_crash(
    tmp_path: Path,
) -> None:
    context = write_sources(tmp_path)
    (tmp_path / "source" / "video.mp4").unlink()

    raw = audit_supplier_data(context, mapping())

    calibration = raw["structured"]["calibration"]
    assert calibration["status"] == "unverified"
    assert calibration["reason"] == "video_metadata_missing"
    assert calibration["video_resolution"] is None
