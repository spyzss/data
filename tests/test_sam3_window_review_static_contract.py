from __future__ import annotations

from pathlib import Path
import subprocess

from tools.serve_sam3_window_review import parse_args


STATIC = Path(__file__).parents[1] / "human_qc" / "static"


def _source(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def test_page_exposes_only_window_level_pass_fail_controls() -> None:
    source = "\n".join(
        _source(name)
        for name in (
            "sam3_window_review.html",
            "sam3_window_review.js",
            "sam3_window_review.css",
        )
    )

    assert 'data-action="verdict-pass"' in source
    assert 'data-action="verdict-fail"' in source
    assert ">Pass<" in source
    assert ">Fail<" in source
    for forbidden in (
        "false_positive",
        "true_positive",
        "rejected segment",
        "Human affected segments",
        "Accept whole window",
        "affected_start_frame",
        "affected_end_frame",
    ):
        assert forbidden not in source
    assert "localStorage" not in source
    assert "duration_resolution_status" in source
    assert "not_annotated" in source


def test_browser_render_has_five_overlays_navigation_progress_and_missing_guard() -> None:
    script = r'''
      import assert from "node:assert/strict";
      import {
        Sam3WindowReviewApp,
        renderReviewMarkup,
        reviewProgress,
      } from "./human_qc/static/sam3_window_review.js";

      const evidence = [0, 5, 10, 15, 20].map((frame) => ({
        frame_idx: frame,
        status: "ready",
        url: `/assets/token/frame-${frame}.png`,
        source_module: "sam3_containment",
        evidence_type: "combined_overlay",
      }));
      const item = {
        review_id: "jdt__episode_000001__window_0_20",
        asset_id: "episode_000001",
        supplier_id: "jdt",
        window_start_frame: 0,
        window_end_frame: 20,
        frame_coordinate_system: "source_inclusive",
        fps: 30,
        left_window_containment_verdict: "fail",
        right_window_containment_verdict: "review",
        trigger_reason: ["containment_mismatch"],
        trigger_metrics: { outside_ratio: 0.4 },
        evidence,
        evidence_status: "ready",
        evidence_error: null,
        can_review: true,
        manual_review: { status: "completed", verdict: "pass", reviewer: "alice", revision: 1 },
      };
      const markup = renderReviewMarkup(item, 0, 2);
      assert.equal((markup.match(/<img /g) || []).length, 5);
      assert.match(markup, /SAM3 containment combined overlay/);
      assert.match(markup, /source inclusive/);
      assert.match(markup, /1 \/ 2/);
      assert.match(markup, /Saved: pass/);
      assert.deepEqual(reviewProgress([item, { ...item, manual_review: { status: "unresolved", verdict: null } }]), { completed: 1, total: 2 });

      const missing = { ...item, can_review: false, evidence_status: "error", evidence_error: "missing", manual_review: { status: "unresolved", verdict: null } };
      const missingMarkup = renderReviewMarkup(missing, 0, 1);
      assert.match(missingMarkup, /Evidence unavailable/);
      assert.match(missingMarkup, /data-action="verdict-pass" disabled/);
      assert.match(missingMarkup, /data-action="verdict-fail" disabled/);

      const app = new Sam3WindowReviewApp({ fetcher: async () => ({ ok: true, json: async () => ({ bundle: { items: [item, missing] } }) }) });
      await app.load();
      assert.equal(app.currentItem().review_id, item.review_id);
      app.next();
      assert.equal(app.currentItem().review_id, missing.review_id);
      app.previous();
      assert.equal(app.currentItem().review_id, item.review_id);
    '''
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=STATIC.parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_cli_requires_explicit_contract_paths_and_has_no_run_root_defaults(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--manifest",
            str(tmp_path / "manifest.csv"),
            "--review-queue",
            str(tmp_path / "queue.csv"),
            "--evidence-manifest",
            str(tmp_path / "evidence.csv"),
            "--review-dir",
            str(tmp_path / "review"),
            "--save-dir",
            str(tmp_path / "save"),
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
        ]
    )

    assert args.manifest == tmp_path / "manifest.csv"
    assert args.review_queue == tmp_path / "queue.csv"
    assert args.evidence_manifest == tmp_path / "evidence.csv"
    assert args.review_dir == tmp_path / "review"
    assert args.save_dir == tmp_path / "save"
    assert args.host == "0.0.0.0"
    assert args.port == 9000
