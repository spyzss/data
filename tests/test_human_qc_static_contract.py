from __future__ import annotations

import re
from pathlib import Path
import subprocess


STATIC = Path(__file__).parents[1] / "human_qc" / "static"
ALLOWED_ACTIONS = {
    "${action}",
    "acquire-lease",
    "cancel-pending",
    "complete-semantic",
    "complete-warn",
    "confirm-pending",
    "next-asset",
    "play",
    "previous-asset",
    "refresh",
    "save-text",
    "select-issue",
    "step-back",
    "step-forward",
    "toggle-overlay",
    "verdict-fail",
    "verdict-pass",
}


def _read(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def test_semantic_markup_exposes_only_internal_shared_boundary_drag_handles() -> None:
    adapter = _read("semantic_adapter.js")
    static_source = "".join(
        _read(name) for name in ("semantic_adapter.js", "warn_adapter.js", "app.js", "index.html")
    )
    action_names = set(
        re.findall(r'data-action\s*=\s*["\']([^"\']+)', static_source)
    )

    assert "segments.slice(1).map" in adapter
    assert 'querySelectorAll?.(".boundary-handle")' in adapter
    assert "boundaryIndex < 1 || boundaryIndex >= segmentCount" in adapter
    assert 'class="timeline-segment" draggable' not in adapter
    assert "dragstart" not in adapter
    assert action_names == ALLOWED_ACTIONS
    for event in ("pointerdown", "pointermove", "pointerup", "pointercancel"):
        assert adapter.count(f'handle.addEventListener("{event}"') == 1
    assert not re.search(r'\b(?:segment|track)\.addEventListener\("pointer', adapter)


def test_internal_half_open_ranges_are_rendered_with_inclusive_ui_end() -> None:
    adapter = _read("semantic_adapter.js")
    warn = _read("warn_adapter.js")

    assert "exclusive - 1" in adapter
    assert "end_frame_inclusive: endExclusive - 1" in adapter
    assert "model.window.endFrameExclusive - 1" in warn
    assert "[start, end)" in warn


def test_pending_state_locks_mutation_navigation_and_completion_without_local_storage() -> None:
    app = _read("app.js")
    adapter = _read("semantic_adapter.js")
    static_source = "".join(
        _read(name) for name in ("semantic_adapter.js", "warn_adapter.js", "app.js", "index.html")
    )

    assert "pending edit must be confirmed or cancelled before navigation" in app
    assert 'querySelectorAll?.("[data-mutation-control], [data-navigation-control]")' in app
    assert "if (this.isPending()) throw new Error" in adapter
    assert "pending edit locks text editing" in adapter
    assert "localStorage" not in static_source


def test_workbench_app_restores_server_pending_task_and_relocks_controls() -> None:
    script = r'''
      import assert from "node:assert/strict";
      import { WorkbenchApp, mutationControlsDisabled } from "./human_qc/static/app.js";

      const controls = [
        { dataset: { action: "next-asset" }, disabled: false },
        { dataset: { action: "complete-semantic" }, disabled: false },
        { dataset: { action: "confirm-pending" }, disabled: false },
        { dataset: { action: "cancel-pending" }, disabled: false },
      ];
      const stage = {};
      const root = {
        querySelector(selector) {
          return selector === "[data-workbench-stage]" ? stage : null;
        },
        querySelectorAll(selector) {
          assert.equal(selector, "[data-mutation-control], [data-navigation-control]");
          return controls;
        },
      };
      const pendingTask = {
        asset_id: "asset-pending",
        revision: 2,
        task_type: "semantic_calibration",
        semantic: {
          pending_edit: {
            edit_type: "boundary",
            affected_segment_ids: ["s1", "s2"],
            before: [{ internal_id: "s1" }, { internal_id: "s2" }],
            after: [{ internal_id: "s1" }, { internal_id: "s2" }],
          },
        },
      };
      const app = new WorkbenchApp({
        documentRef: {},
        root,
        fetcher: async () => ({
          ok: true,
          status: 200,
          json: async () => ({ task: pendingTask }),
        }),
        semanticAdapterFactory: (options) => ({
          render(task) { options.onLockChange(Boolean(task.semantic.pending_edit)); },
        }),
      });
      const restored = await app.loadAsset("asset-pending");
      assert.equal(restored, pendingTask);
      assert.equal(mutationControlsDisabled(app.task), true);
      assert.equal(controls[0].disabled, true);
      assert.equal(controls[1].disabled, true);
      assert.equal(controls[2].disabled, false);
      assert.equal(controls[3].disabled, false);
    '''
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=STATIC.parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_warn_stage_keeps_machine_fields_read_only_and_exposes_only_human_verdicts() -> None:
    warn = _read("warn_adapter.js")

    assert "data-machine-reason" in warn
    assert "data-machine-metrics" in warn
    assert "data-machine-threshold" in warn
    assert not re.search(r"<(?:input|textarea)[^>]+data-machine-", warn)
    assert 'data-action="verdict-pass"' in warn
    assert 'data-action="verdict-fail"' in warn
    assert "all selected issues require a verdict before completion" in warn


def test_static_application_has_no_hidden_machine_issue_mutation_endpoint() -> None:
    app = _read("app.js")
    warn = _read("warn_adapter.js")

    assert "/warn/${encodeURIComponent(issueId)}/verdict" in app
    assert "/issues/" not in app
    assert "issue.severity =" not in warn
    assert "issue.observed_value =" not in warn


def test_warn_stage_has_complete_visual_and_responsive_css_contract() -> None:
    css = _read("workbench.css")
    required = {
        '[data-task-type="warn_review"] .workspace-grid',
        ".warn-review-head",
        ".warn-rationale",
        ".warn-metrics",
        ".warn-overlay-gallery",
        ".warn-overlay-sample",
        ".warn-decision",
        ".warn-verdict-actions",
        '[data-action="verdict-pass"]',
        '[data-action="verdict-fail"]',
    }
    assert all(selector in css for selector in required)
    assert "@media (max-width: 700px)" in css


def test_hidden_video_placeholder_never_covers_loaded_video() -> None:
    css = _read("workbench.css")
    assert ".video-placeholder[hidden]" in css
    assert "display: none" in css
