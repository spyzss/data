from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tests.test_manifest_sam3_exhaustive_runner import (
    _FakeSequentialSource,
    _FullMaskSegmenter,
    _write_manifest_inputs,
)


def test_fp_evidence_materializer_runs_only_exact_selected_frames(
    tmp_path: Path,
) -> None:
    from tools.build_sam3_comparison_evidence import build_false_positive_evidence

    manifest, _, _ = _write_manifest_inputs(tmp_path, start_frame=0, end_frame=4)
    false_positive_rows = [
        {
            "asset_id": "jdt__episode_000001",
            "source_frame": 1,
            "video_frame": 1,
            "comparison_class": "FP",
            "precheck_problem_any": True,
            "precheck_modules": json.dumps(["keypoint_temporal"]),
            "precheck_reason_codes": json.dumps(["jump"]),
            "sam3_frame_verdict": "pass",
        },
        {
            "asset_id": "jdt__episode_000001",
            "source_frame": 3,
            "video_frame": 3,
            "comparison_class": "FP",
            "precheck_problem_any": True,
            "precheck_modules": json.dumps(["keypoint_morphology"]),
            "precheck_reason_codes": json.dumps(["shape"]),
            "sam3_frame_verdict": "pass",
        },
    ]
    source = _FakeSequentialSource()
    segmenter = _FullMaskSegmenter()
    writes: list[tuple[str, int]] = []

    def overlay_writer(*, clip_id, frame_idx, output_dir, **kwargs):
        del kwargs
        writes.append((clip_id, frame_idx))
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"frame_{frame_idx:012d}.png"
        path.write_bytes(b"png")
        return path

    summary = build_false_positive_evidence(
        manifest=manifest,
        false_positive_rows=false_positive_rows,
        sam3_model=None,
        output_dir=tmp_path / "fp-evidence",
        source_reader=source,
        segmenter=segmenter,
        overlay_writer=overlay_writer,
    )

    assert source.requested_video_frames == [1, 3]
    assert source.iter_frames_calls == 1
    assert segmenter.calls == 2
    assert [frame for _clip_id, frame in writes] == [1, 3]
    assert "__source_000000000001__video_000000000001" in writes[0][0]
    assert "__source_000000000003__video_000000000003" in writes[1][0]
    manifest_rows = pd.read_csv(
        tmp_path / "fp-evidence" / "false_positive_evidence_manifest.csv"
    )
    assert manifest_rows["source_frame"].tolist() == [1, 3]
    assert manifest_rows["comparison_class"].tolist() == ["FP", "FP"]
    assert manifest_rows["rerun_frame_verdict"].tolist() == ["pass", "pass"]
    assert summary["requested_frame_count"] == 2
    assert summary["materialized_evidence_count"] == 2


def test_fp_evidence_materializer_rejects_non_fp_or_duplicate_keys(
    tmp_path: Path,
) -> None:
    import pytest

    from tools.build_sam3_comparison_evidence import build_false_positive_evidence

    manifest, _, _ = _write_manifest_inputs(tmp_path, start_frame=0, end_frame=1)
    base = {
        "asset_id": "jdt__episode_000001",
        "source_frame": 0,
        "video_frame": 0,
        "comparison_class": "FN",
    }

    with pytest.raises(ValueError, match="only FP rows"):
        build_false_positive_evidence(
            manifest=manifest,
            false_positive_rows=[base],
            sam3_model=None,
            output_dir=tmp_path / "wrong-class",
            source_reader=_FakeSequentialSource(),
            segmenter=_FullMaskSegmenter(),
        )

    with pytest.raises(ValueError, match="duplicate asset/source-frame"):
        build_false_positive_evidence(
            manifest=manifest,
            false_positive_rows=[{**base, "comparison_class": "FP"}] * 2,
            sam3_model=None,
            output_dir=tmp_path / "duplicate",
            source_reader=_FakeSequentialSource(),
            segmenter=_FullMaskSegmenter(),
        )
