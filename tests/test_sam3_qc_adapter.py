from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from qc_common.contracts import ModuleResult
from qc_common.report_mutation import apply_module_result
from qc_pipeline.adapters.sam3_containment import adapt_sam3_containment
from qc_pipeline.context import AssetContext
from tests.qc_report_fixtures import loaded_test_config


def test_sam3_adapter_maps_fail_and_overlay_reference(tmp_path: Path) -> None:
    overlay = tmp_path / "sam3" / "combined_overlays" / "a_10.png"
    overlay.parent.mkdir(parents=True)
    overlay.write_bytes(b"png")

    result = adapt_sam3_containment(
        asset_id="a",
        batch_root=tmp_path,
        window_summaries=[
            {
                "asset_id": "a",
                "window_start_frame": 10,
                "window_end_frame": 20,
                "hand_side": "left",
                "containment_verdict": "strong_containment_mismatch",
                "inside_ratio": 0.1,
            }
        ],
        evidence_rows=[
            {
                "asset_id": "a",
                "window_start_frame": 10,
                "window_end_frame": 20,
                "hand_side": "both",
                "evidence_type": "combined_overlay",
                "source_path": str(overlay),
            }
        ],
        config=loaded_test_config(),
    )

    assert result.verdict == "fail"
    assert result.evidence[0].path == "sam3/combined_overlays/a_10.png"
    assert result.issues[0].evidence_ids == (result.evidence[0].evidence_id,)


def test_missing_overlay_is_integrity_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="evidence file does not exist"):
        adapt_sam3_containment(
            asset_id="a",
            batch_root=tmp_path,
            window_summaries=[],
            evidence_rows=[
                {
                    "asset_id": "a",
                    "evidence_type": "combined_overlay",
                    "source_path": str(tmp_path / "missing.png"),
                }
            ],
            config=loaded_test_config(),
        )


def test_sam3_evidence_is_stable_hashed_and_source_inclusive(
    tmp_path: Path,
) -> None:
    overlay = tmp_path / "sam3" / "combined_overlays" / "a_10.png"
    overlay.parent.mkdir(parents=True)
    overlay.write_bytes(b"png")
    kwargs = {
        "asset_id": "a",
        "batch_root": tmp_path,
        "window_summaries": [
            {
                "asset_id": "a",
                "window_start_frame": 10,
                "window_end_frame": 20,
                "hand_side": "left",
                "containment_verdict": "projection_review",
                "inside_ratio": 0.4,
                "mask": [[True]],
                "video_frames": ["must-not-be-inlined"],
            }
        ],
        "evidence_rows": [
            {
                "asset_id": "a",
                "window_start_frame": 10,
                "window_end_frame": 20,
                "hand_side": "both",
                "evidence_type": "combined_overlay",
                "source_path": "sam3/combined_overlays/a_10.png",
            }
        ],
        "config": loaded_test_config(),
    }

    first = adapt_sam3_containment(**kwargs)
    second = adapt_sam3_containment(**kwargs)

    assert first.evidence == second.evidence
    assert first.evidence[0].checksum == f"sha256:{hashlib.sha256(b'png').hexdigest()}"
    assert first.evidence[0].mime_type == "image/png"
    assert first.evidence[0].coordinate_system == "source_inclusive"
    assert first.evidence[0].start_frame == 10
    assert first.evidence[0].end_frame == 20
    assert first.issues[0].context["start_frame"] == 10
    assert first.issues[0].context["end_frame"] == 20
    payload = json.dumps(first.to_dict(), sort_keys=True)
    assert "must-not-be-inlined" not in payload
    assert '"mask"' not in payload


def test_sam3_adapter_rejects_absolute_evidence_outside_batch(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / "outside.png"
    outside.write_bytes(b"png")

    with pytest.raises(ValueError, match="outside batch root"):
        adapt_sam3_containment(
            asset_id="a",
            batch_root=tmp_path,
            window_summaries=[],
            evidence_rows=[
                {
                    "asset_id": "a",
                    "evidence_type": "combined_overlay",
                    "source_path": str(outside),
                }
            ],
            config=loaded_test_config(),
        )


def test_sam3_adapter_emits_at_most_one_issue_per_window_hand(
    tmp_path: Path,
) -> None:
    summary = {
        "asset_id": "a",
        "window_start_frame": 10,
        "window_end_frame": 20,
        "hand_side": "left",
        "containment_verdict": "strong_containment_mismatch",
        "inside_ratio": 0.1,
    }

    result = adapt_sam3_containment(
        asset_id="a",
        batch_root=tmp_path,
        window_summaries=[summary, dict(summary)],
        evidence_rows=[],
        config=loaded_test_config(),
    )

    assert len(result.issues) == 1


@pytest.mark.parametrize(
    ("producer_verdict", "expected_verdict", "expected_rule"),
    [
        ("likely_visible_ok", "pass", None),
        ("acceptable_flagged", "pass", None),
        (
            "containment_fail",
            "fail",
            "sam3_containment.strong_containment_mismatch",
        ),
        (
            "side_view_manual_review",
            "warn",
            "sam3_containment.side_view_manual_review",
        ),
        (
            "rotation_manual_review",
            "warn",
            "sam3_containment.side_view_manual_review",
        ),
        (
            "projection_review",
            "warn",
            "sam3_containment.projection_review",
        ),
        ("mixed_review", "warn", "sam3_containment.projection_review"),
        ("containment_review", "warn", "sam3_containment.projection_review"),
        (
            "mask_missing_or_tiny_review",
            "warn",
            "sam3_containment.projection_review",
        ),
        ("review", "warn", "sam3_containment.projection_review"),
    ],
)
def test_sam3_adapter_maps_every_structured_producer_verdict(
    producer_verdict: str,
    expected_verdict: str,
    expected_rule: str | None,
    tmp_path: Path,
) -> None:
    result = adapt_sam3_containment(
        asset_id="a",
        batch_root=tmp_path,
        window_summaries=[
            {
                "asset_id": "a",
                "window_start_frame": 10,
                "window_end_frame": 20,
                "hand_side": "left",
                "window_containment_verdict": producer_verdict,
                "reason": "strong mismatch text must never drive the mapping",
            }
        ],
        evidence_rows=[],
        config=loaded_test_config(),
    )

    assert result.verdict == expected_verdict
    assert [issue.rule_id for issue in result.issues] == (
        [] if expected_rule is None else [expected_rule]
    )


def _advance_to_sam3(context: AssetContext) -> tuple[int, str]:
    config = loaded_test_config()
    modules = config.pipeline_modules
    target_index = modules.index("sam3_containment")
    revision = 0
    for index, module in enumerate(modules[:target_index]):
        report = apply_module_result(
            context.report_path,
            context=context,
            config=config,
            profile="acceptance",
            result=ModuleResult(module, "pass", {}, {}),
            expected_revision=revision,
            next_module=modules[index + 1],
            now=f"2026-07-14T00:00:{index:02d}Z",
        )
        revision = int(report["report_revision"])
    return revision, modules[target_index + 1]


def test_sam3_writer_commits_only_summary_and_relative_evidence(
    tmp_path: Path,
) -> None:
    from qc_pipeline.adapters.sam3_containment import write_sam3_asset_result

    overlay = tmp_path / "sam3" / "combined_overlays" / "a_10.png"
    overlay.parent.mkdir(parents=True)
    overlay.write_bytes(b"png")
    context = AssetContext(
        asset_id="a",
        batch_root=tmp_path,
        report_path=tmp_path / "quality_archive" / "a.json",
        source_files={},
    )
    revision, next_module = _advance_to_sam3(context)

    updated = write_sam3_asset_result(
        context=context,
        window_summaries=[
            {
                "asset_id": "a",
                "window_start_frame": 10,
                "window_end_frame": 20,
                "hand_side": "left",
                "containment_verdict": "acceptable",
                "mask": [[True]],
            }
        ],
        evidence_rows=[
            {
                "asset_id": "a",
                "window_start_frame": 10,
                "window_end_frame": 20,
                "hand_side": "both",
                "evidence_type": "combined_overlay",
                "source_path": str(overlay),
            }
        ],
        config=loaded_test_config(),
        profile="acceptance",
        expected_revision=revision,
        next_module=next_module,
    )

    block = updated["sam3_containment"]
    assert updated["report_revision"] == revision + 1
    assert block["flow"]["result_gate"]["verdict"] == "pass"
    assert block["metrics"]["window_count"] == 1
    assert block["evidence"][0]["path"] == "sam3/combined_overlays/a_10.png"
    assert not Path(block["evidence"][0]["path"]).is_absolute()
    assert "mask" not in json.dumps(block, sort_keys=True)
