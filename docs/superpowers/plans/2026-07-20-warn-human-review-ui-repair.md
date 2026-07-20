# Warn Human Review UI Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the Warn human-review page into the approved video-first vertical workflow, make evidence generation reliable, and keep internal errors out of the browser while preserving the canonical JSON review outcome.

**Architecture:** Keep the framework-free workbench and its existing `WarnReviewAdapter`. The app shell selects a Warn presentation mode, the adapter renders one issue at a time below the shared video player, and `EvidenceService` returns only stable evidence status codes. The canonical report remains the sole source of truth for per-issue verdicts and the final `overall_decision`.

**Tech Stack:** Python 3.11+, pytest, vanilla ES modules, Node.js built-in test runner, HTML/CSS, FFmpeg.

## Global Constraints

- Use the existing workbench video element; do not add a second player or a frontend framework.
- Warn uses a vertical order: video, review reason/evidence, PASS/FAIL.
- Every Warn plays only the raw issue-frame clip. SAM3/skeleton Warns show the existing sampled skeleton/mask overlay images below the video and never synthesize a continuous overlay video.
- Machine fields remain read-only; only human reason and verdict are mutable.
- Never expose command lines, absolute paths, or Python exception text in the task DTO or browser.
- Preserve `asset_qc_report.v2`: `manual_review.state`, `manual_review.issue_reviews`, `manual_review.completed_at`, `pipeline_state`, and `overall_decision` remain authoritative.
- Human PASS cannot downgrade a machine hard fail.
- Keep all work on `codex/human-qc-impl`; do not merge to `main`.

---

## File Structure

- Modify `human_qc/evidence.py`: preserve media suffixes for atomic generation and emit stable overlay failure codes.
- Modify `human_qc/workbench_service.py`: sanitize clip-generation failures before creating the browser task DTO.
- Modify `human_qc/static/app.js`: expose the active task type to CSS and update Warn-specific workbench chrome.
- Modify `human_qc/static/warn_adapter.js`: render one focused Warn in the approved vertical information order and advance to the next unresolved issue.
- Modify `human_qc/static/workbench.css`: add the complete Warn presentation and responsive rules.
- Modify `tests/test_review_evidence.py`: cover MP4-compatible temporary paths.
- Modify `tests/test_human_qc_recovery.py`: cover safe clip/overlay degradation codes.
- Modify `human_qc/static/workbench.test.mjs`: cover focused issue progression, safe error copy, and vertical markup.
- Modify `tests/test_human_qc_static_contract.py`: require Warn CSS coverage and the task-type styling hook.
- Modify `docs/reviewer-guide.md`: document the visual workflow and final JSON markers.

---

### Task 1: Reliable and Safe Evidence Projection

**Files:**
- Modify: `tests/test_review_evidence.py`
- Modify: `tests/test_human_qc_recovery.py`
- Modify: `human_qc/evidence.py`
- Modify: `human_qc/workbench_service.py`

**Interfaces:**
- Consumes: `EvidenceService.resolve(issue: Mapping[str, Any], asset_context: AssetContext) -> EvidenceView`
- Produces: `EvidenceView.clip_url` for the raw bounded issue MP4 and ordered `overlay_urls` for every existing SAM3 sampled overlay image. `EvidenceView.generation_error` is `None` or `"overlay_unavailable"`; workbench fallback rows use `"clip_unavailable"`.

- [ ] **Step 1: Write the failing MP4 temporary-path regression test**

Add a test that records both the command output argument and the `output` callback argument:

```python
def test_generated_clip_temporary_path_keeps_mp4_suffix(tmp_path: Path) -> None:
    context = _context(tmp_path)
    issue = _issue()
    observed: dict[str, Path] = {}

    def generate(command, output: Path) -> None:
        observed["command_output"] = Path(command[-1])
        observed["callback_output"] = output
        output.write_bytes(b"mp4")

    service = EvidenceService(tmp_path / "cache", ffmpeg_runner=generate)
    view = service.resolve(issue, context)

    assert observed["command_output"].suffix == ".mp4"
    assert observed["callback_output"].suffix == ".mp4"
    assert view.clip_url.endswith(".mp4")
```

- [ ] **Step 2: Run the MP4 regression test and verify RED**

Run: `.venv/bin/python -m pytest tests/test_review_evidence.py::test_generated_clip_temporary_path_keeps_mp4_suffix -q`

Expected: FAIL because the atomic temporary file currently ends in `.tmp`.

- [ ] **Step 3: Write failing safe-error projection tests**

Change the overlay failure assertion and add a clip failure assertion:

```python
assert task["evidence"][0]["generation_error"] == "overlay_unavailable"
assert "renderer unavailable" not in json.dumps(task["evidence"])
```

```python
def test_clip_failure_exposes_stable_code_without_internal_exception(tmp_path: Path) -> None:
    asset = build_file_asset(tmp_path, asset_id="asset-clip-error", hard_fail=False)
    semantic = _semantic(asset)
    semantic.complete(asset.asset_id, expected_revision=1, lease_token=LEASE)

    def fail_clip(*_args: object) -> None:
        raise RuntimeError(f"ffmpeg failed for {asset.video_path}")

    workbench = WorkbenchService(
        semantic,
        WarnReviewService(reports={asset.asset_id: asset.report_path}, leases={asset.asset_id: LEASE}),
        EvidenceService(asset.root / "cache", ffmpeg_runner=fail_clip),
        asset_contexts={asset.asset_id: asset.context},
    )
    task = workbench.get_asset_task(asset.asset_id)

    assert task["evidence"][0]["generation_error"] == "clip_unavailable"
    assert str(asset.video_path) not in json.dumps(task["evidence"])
```

- [ ] **Step 4: Run safe-error tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_human_qc_recovery.py -k 'overlay_failure or clip_failure' -q`

Expected: FAIL because raw exception strings are currently projected.

- [ ] **Step 5: Write the failing sampled-overlay collection test**

Create three existing SAM3 overlay PNG evidence rows with distinct frame indexes and assert the task projection retains all three in frame order while `clip_url` remains the raw bounded MP4:

```python
assert [row["frame"] for row in view.overlay_images] == [120, 144, 188]
assert all(row["url"].endswith(".png") for row in view.overlay_images)
assert view.clip_url.endswith(".mp4")
```

- [ ] **Step 6: Run the sampled-overlay test and verify RED**

Run: `.venv/bin/python -m pytest tests/test_review_evidence.py -k 'sampled_overlay' -q`

Expected: FAIL because the current evidence resolver retains only one overlay path.

- [ ] **Step 7: Preserve the final suffix in atomic generation**

Change the temporary name in `_atomic_generate`:

```python
temporary = output.with_name(
    f".{output.stem}.{uuid.uuid4().hex}.tmp{output.suffix}"
)
```

This creates paths ending in `.mp4` or `.png`, so FFmpeg and image renderers can infer the intended format before `os.replace` publishes the final file.

- [ ] **Step 8: Retain every sampled overlay image**

Replace the single existing-overlay slot with an ordered collection containing URL and representative frame. Filter evidence to image kinds `overlay`, `skeleton_overlay`, and `combined_overlay`, keep only files inside the batch root, sort by `start_frame`, and never invoke a dynamic overlay-video renderer. Preserve the first image as `overlay_url` only for backward compatibility; new UI code consumes the full collection.

- [ ] **Step 9: Replace browser-facing exception strings with stable codes**

In `EvidenceService.resolve`, log the overlay exception server-side and set:

```python
generation_error = "overlay_unavailable"
```

In `WorkbenchService._evidence`, log the clip exception server-side and append:

```python
result.append({
    "issue_id": issue.get("issue_id"),
    "generation_error": "clip_unavailable",
})
```

Use module loggers created with `logging.getLogger(__name__)`; exception details may appear in server logs but never in the DTO.

- [ ] **Step 10: Run evidence tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_review_evidence.py tests/test_human_qc_recovery.py -q`

Expected: all tests pass and no DTO assertion contains internal paths or exception messages.

- [ ] **Step 11: Commit the evidence repair**

```bash
git add human_qc/evidence.py human_qc/workbench_service.py tests/test_review_evidence.py tests/test_human_qc_recovery.py
git commit -m "fix(human-qc): harden warn evidence generation"
```

---

### Task 2: Focused Video-First Warn Markup and Progression

**Files:**
- Modify: `human_qc/static/workbench.test.mjs`
- Modify: `human_qc/static/warn_adapter.js`
- Modify: `human_qc/static/app.js`

**Interfaces:**
- Consumes: task DTO fields `warn.selected_issue_ids`, `warn.issue_reviews`, `evidence[].generation_error`, `evidence[].overlay_images`, and the shared `[data-video]` element.
- Produces: `nextReviewIssueId(task, currentIssueId) -> string | null` and a vertical `.warn-review` DOM with `.warn-rationale` before `.warn-decision`.

- [ ] **Step 1: Write failing vertical-markup tests**

Add `nextReviewIssueId` to the existing `warn_adapter.js` import list, then add Node assertions:

```javascript
test("warn markup follows video-first rationale-then-decision flow", () => {
  const markup = renderWarnMarkup(warnTask, "warn-1");
  assert.match(markup, /class="warn-rationale"/);
  assert.match(markup, /class="warn-decision"/);
  assert.match(markup, /data-action="verdict-pass"/);
  assert.match(markup, /data-action="verdict-fail"/);
  assert.ok(markup.indexOf("warn-rationale") < markup.indexOf("warn-decision"));
  assert.doesNotMatch(markup, /class="warn-layout"/);
  assert.doesNotMatch(markup, /<pre data-machine-metrics>/);
});
```

Add safe-copy assertions:

```javascript
test("warn evidence codes render safe reviewer copy", () => {
  const degraded = structuredClone(warnTask);
  degraded.evidence = [{ issue_id: "warn-1", generation_error: "clip_unavailable" }];
  const markup = renderWarnMarkup(degraded, "warn-1");
  assert.match(markup, /问题片段暂不可用/);
  assert.doesNotMatch(markup, /ffmpeg|Command|\/private\//i);
});
```

Add an overlay-gallery assertion:

```javascript
test("warn markup renders every sampled SAM3 overlay image", () => {
  const sampled = structuredClone(warnTask);
  sampled.evidence[0].overlay_images = [
    { frame: 120, url: "/evidence/frame-120.png" },
    { frame: 144, url: "/evidence/frame-144.png" },
    { frame: 188, url: "/evidence/frame-188.png" },
  ];
  const markup = renderWarnMarkup(sampled, "warn-1");
  assert.equal((markup.match(/class="warn-overlay-sample"/g) || []).length, 3);
  assert.ok(markup.indexOf("frame-120.png") < markup.indexOf("frame-188.png"));
});
```

- [ ] **Step 2: Write the failing next-unreviewed issue test**

```javascript
test("reviewed warn advances to the next unresolved issue", () => {
  const updated = structuredClone(warnTask);
  updated.warn.issue_reviews = { "warn-1": { verdict: "pass" } };
  assert.equal(nextReviewIssueId(updated, "warn-1"), "warn-2");
  updated.warn.issue_reviews["warn-2"] = { verdict: "fail" };
  assert.equal(nextReviewIssueId(updated, "warn-2"), "warn-2");
});
```

- [ ] **Step 3: Run the Node tests and verify RED**

Run: `node --test human_qc/static/workbench.test.mjs`

Expected: FAIL because the vertical classes and `nextReviewIssueId` do not exist.

- [ ] **Step 4: Implement focused issue selection and safe evidence copy**

Export this selection helper from `warn_adapter.js`:

```javascript
export function nextReviewIssueId(task, currentIssueId = null) {
  const ids = selectedIds(task);
  const reviews = objectOrEmpty(task?.warn?.issue_reviews);
  const unresolved = ids.find((id) => !["pass", "fail"].includes(reviews[id]?.verdict));
  if (unresolved) return unresolved;
  return ids.includes(String(currentIssueId)) ? String(currentIssueId) : ids.at(-1) ?? null;
}
```

In `render`, advance only when the current issue has a persisted verdict. Map evidence codes to reviewer-safe copy and render every `overlay_images` entry as a frame-labelled thumbnail linking to the original image:

```javascript
const evidenceMessage = {
  clip_unavailable: "问题片段暂不可用，正在使用原视频定位问题区间。",
  overlay_unavailable: "骨架 overlay 暂不可用，仍可使用问题视频完成判断。",
}[model.generationError] ?? "";
```

- [ ] **Step 5: Replace the old Warn layout markup**

Render one issue with these structural regions:

```html
<section class="warn-review">
  <header class="warn-review-head">...</header>
  <section class="warn-rationale">...</section>
  <section class="warn-decision">
    <label class="warn-reason">...</label>
    <div class="warn-verdict-actions">...</div>
  </section>
  <div class="warn-complete-row">...</div>
</section>
```

Render metrics as escaped `<dl class="warn-metrics">` key/value rows instead of raw JSON `<pre>`. Keep `data-machine-reason`, `data-machine-metrics`, and `data-machine-threshold` markers on read-only elements for contract compatibility.

- [ ] **Step 6: Expose active task type to the app shell**

In `WorkbenchApp.renderStatus`, set the root attribute before updating labels:

```javascript
this.root.dataset.taskType = currentTaskType || "idle";
```

For Warn, update the stage label and inspector copy to concise operator language. Do not change mutation endpoints or JSON payloads.

- [ ] **Step 7: Run Node tests and verify GREEN**

Run: `node --test human_qc/static/workbench.test.mjs`

Expected: all Node tests pass.

- [ ] **Step 8: Commit markup and behavior**

```bash
git add human_qc/static/app.js human_qc/static/warn_adapter.js human_qc/static/workbench.test.mjs
git commit -m "feat(human-qc): add focused warn review flow"
```

---

### Task 3: Complete Warn Visual System and Responsive Layout

**Files:**
- Modify: `tests/test_human_qc_static_contract.py`
- Modify: `human_qc/static/workbench.css`

**Interfaces:**
- Consumes: `[data-task-type="warn_review"]`, `.warn-review-head`, `.warn-rationale`, `.warn-metrics`, `.warn-decision`, `.warn-verdict-actions`.
- Produces: full-width video-first Warn presentation on desktop and a single-column decision stack below 700px.

- [ ] **Step 1: Write the failing CSS coverage test**

```python
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
```

- [ ] **Step 2: Run the CSS contract test and verify RED**

Run: `.venv/bin/python -m pytest tests/test_human_qc_static_contract.py::test_warn_stage_has_complete_visual_and_responsive_css_contract -q`

Expected: FAIL because Warn selectors are absent.

- [ ] **Step 3: Add Warn workbench-mode styles**

Add CSS that:

- changes the Warn workspace grid to one column and hides the semantic inspector;
- gives the video stage a restrained orange evidence border;
- lays out the rationale as a two-column reason/evidence region;
- renders SAM3 sampled overlay images as a compact, horizontally scrollable thumbnail strip with frame labels and full-image links;
- renders PASS and FAIL as equal-height, full-width decision targets;
- styles textarea, progress, safe degradation notice, completion control, focus-visible, hover, selected, loading, and disabled states;
- uses the existing typography, surfaces, and spacing variables instead of introducing a second design system.

The core layout declarations must be:

```css
[data-task-type="warn_review"] .workspace-grid { grid-template-columns: minmax(0, 1fr); }
[data-task-type="warn_review"] .inspector-panel { display: none; }
.warn-rationale { display: grid; grid-template-columns: minmax(0, 1fr) minmax(280px, .9fr); }
.warn-verdict-actions { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); }
```

- [ ] **Step 4: Add responsive and accessibility rules**

At `@media (max-width: 700px)`, stack `.warn-rationale` and `.warn-verdict-actions`. Include visible `:focus-visible` outlines, minimum 48px decision targets, `overflow-wrap: anywhere` for safe notices, and no horizontal overflow.

- [ ] **Step 5: Run static and Node tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_human_qc_static_contract.py -q && node --test human_qc/static/workbench.test.mjs`

Expected: all tests pass.

- [ ] **Step 6: Commit the visual system**

```bash
git add human_qc/static/workbench.css tests/test_human_qc_static_contract.py
git commit -m "style(human-qc): finish warn review workspace"
```

---

### Task 4: JSON Contract Documentation and End-to-End Verification

**Files:**
- Modify: `docs/reviewer-guide.md`
- Verify: `schemas/asset_qc_report.v2.schema.json`
- Verify: `tests/test_warn_review_service.py`
- Verify: `tests/test_human_qc_end_to_end.py`

**Interfaces:**
- Consumes: persisted `asset_qc_report.v2` after Warn completion.
- Produces: reviewer documentation that identifies the exact human-review and final-decision fields.

- [ ] **Step 1: Add the reviewer-facing JSON example**

Document this successful terminal shape without introducing a new flag:

```json
{
  "manual_review": {
    "state": "completed",
    "completed_at": "2026-07-20T12:00:00Z",
    "issue_reviews": {
      "warn-1": {
        "verdict": "pass",
        "effective_verdict": "pass",
        "reviewer": "reviewer-id",
        "reviewed_at": "2026-07-20T11:59:30Z"
      }
    }
  },
  "pipeline_state": {
    "status": "completed",
    "last_completed_module": "manual_review",
    "next_module": null
  },
  "overall_decision": "pass"
}
```

Explain that there is intentionally no duplicate `human_qc_pass: true`: `overall_decision` is the canonical asset-level outcome, while `manual_review` preserves human provenance.

- [ ] **Step 2: Run canonical JSON contract tests**

Run: `.venv/bin/python -m pytest tests/test_warn_review_service.py tests/test_human_qc_end_to_end.py tests/test_human_qc_report_schema.py -q`

Expected: all tests pass, including `manual_review.state == "completed"` and `overall_decision == "pass"` cases.

- [ ] **Step 3: Run the focused full repair suite**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_review_evidence.py \
  tests/test_human_qc_recovery.py \
  tests/test_human_qc_static_contract.py \
  tests/test_warn_review_service.py \
  tests/test_human_qc_end_to_end.py \
  tests/test_human_qc_report_schema.py -q
node --test human_qc/static/workbench.test.mjs
```

Expected: every selected Python and Node test passes.

- [ ] **Step 4: Run repository integrity checks**

Run: `.venv/bin/python -m pytest -q`

Expected: full suite exits 0.

Run: `git diff --check`

Expected: exit 0 with no output.

- [ ] **Step 5: Perform desktop browser visual acceptance**

Open the local workbench with a Warn task and verify:

- the shared video occupies the top full-width region;
- the problem range and current frame are visible with the video;
- reason and evidence appear directly below the video;
- PASS and FAIL are equal, prominent actions below the reason;
- no native unstyled controls or internal FFmpeg command appears;
- submitting a verdict advances to the next unresolved Warn and updates JSON revision.

- [ ] **Step 6: Perform narrow-screen browser visual acceptance**

Set the viewport below 700px and verify the reason blocks and PASS/FAIL stack without horizontal overflow, clipped copy, or hidden controls.

- [ ] **Step 7: Commit documentation and final verification record**

```bash
git add docs/reviewer-guide.md
git commit -m "docs(human-qc): document warn review outcomes"
```

- [ ] **Step 8: Push the feature branch without merging**

Run: `git push origin codex/human-qc-impl`

Verify: `git ls-remote --heads origin codex/human-qc-impl` returns the exact local `git rev-parse HEAD` SHA. Do not merge or update `main`.
