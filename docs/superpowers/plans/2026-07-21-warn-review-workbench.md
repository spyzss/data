---
change: add-human-semantic-warn-review
design-doc: docs/superpowers/specs/2026-07-21-warn-review-workbench-design.md
base-ref: 2a63925d8f11be9685347f8539afe9e9f18f1be6
---

# Warn 人工复核工作台实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有共用人工工作台重构为 Warn-only 的整段视频复核页面，支持 early-fail、资产级人工原因、真实帧时间轴和 SAM3 问题区间连续 overlay，并把语义校准完整迁入独立模块。

**Architecture:** `human_qc` 与新 `semantic_calibration` 只通过 `asset_qc_report.v2` 通信。后端先固化报告合同与流水线状态，再拆分服务；前端使用原生 ES modules，把应用、视频、时间轴和复核面板分开；SAM3 连续 overlay 由有界后台 evidence worker 预生成并缓存，浏览器仅同步播放产物。

**Tech Stack:** Python 3、pytest、标准库 HTTP server、JSON Schema、原生 ES modules、Node test runner、HTML/CSS、FFmpeg/OpenCV、现有 SAM3 runtime provider。

## Global Constraints

- 保持 `asset_qc_report.v2` 为唯一正式业务状态源，机器 issue、指标、阈值和 evidence 不可被人工操作改写。
- 内部问题区间统一使用半开区间 `[start_frame, end_frame_exclusive)`；UI 显示闭区间。
- `human_qc` 不得导入 `semantic_calibration`，`semantic_calibration` 不得导入 `human_qc`。
- 浏览器和 HTTP 请求线程不得运行 SAM3 推理；连续 overlay 只覆盖问题帧并集。
- localStorage 只保存倍速等非业务偏好。
- 所有写请求继续使用 reviewer lease 和 expected report revision。
- 每完成一个任务，同步勾选 `openspec/changes/add-human-semantic-warn-review/tasks.md` 对应条目并单独提交。

---

### Task 1: 扩展人工复核报告合同和批次统计

**Files:**
- Modify: `qc_common/schema.py`
- Modify: `qc_common/report_migration.py`
- Modify: `qc_common/projection.py`
- Modify: `tests/test_human_qc_report_schema.py`
- Modify: `tests/test_human_qc_aggregation.py`
- Modify: `tests/test_qc_migration_reconciliation.py`

**Interfaces:**
- Produces: `manual_review.completion_mode: null | "all_reviewed" | "early_fail"`
- Produces: `manual_review.failure_reason: null | {mode, reason_codes, other_text}`
- Produces: aggregation field `unreviewed_selected_warn_count`
- Preserves: missing new fields remain readable for historical reports; any new write emits the new canonical shape.

- [x] **Step 1: Write failing schema and aggregation tests**

Add tests that construct a completed early-fail report with three selected issues and only one Fail review, then assert:

```python
validate_asset_qc_report(report)
assert report["manual_review"]["completion_mode"] == "early_fail"
assert report["manual_review"]["failure_reason"] == {
    "mode": "manual",
    "reason_codes": ["occlusion", "other"],
    "other_text": "手被工具完全遮挡",
}
assert projected["human_reviewed_warn_count"] == 1
assert projected["human_confirmed_fail_count"] == 1
assert projected["unreviewed_selected_warn_count"] == 2
```

Also assert that `other` with blank `other_text`, `early_fail` without an actual Fail, and `all_reviewed` with a missing review raise `ReportValidationError`.

- [x] **Step 2: Run tests and verify Red**

Run:

```bash
pytest -q tests/test_human_qc_report_schema.py tests/test_human_qc_aggregation.py tests/test_qc_migration_reconciliation.py
```

Expected: FAIL because the current schema requires every selected issue review and has no completion/failure-reason validation or unreviewed aggregate.

- [x] **Step 3: Implement the canonical validators and projection**

In `qc_common/schema.py`, validate the new blocks with explicit helpers:

```python
_COMPLETION_MODES = frozenset({"all_reviewed", "early_fail"})

def _validate_failure_reason(value: object) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping) or value.get("mode") != "manual":
        _human_validation_error("manual_review.failure_reason", "must use manual mode")
    codes = value.get("reason_codes")
    if not _is_string_sequence(codes) or len(set(codes)) != len(codes):
        _human_validation_error("manual_review.failure_reason.reason_codes", "must contain unique codes")
    other = str(value.get("other_text") or "").strip()
    if "other" in codes and not other:
        _human_validation_error("manual_review.failure_reason.other_text", "is required for other")
```

For completed blocks, derive Pass/Fail counts from `issue_reviews`; enforce `all_reviewed` and `early_fail` invariants. Update the v1/v2 compatibility migration to add `completion_mode="all_reviewed"` only when a historical completed report has all selected reviews, otherwise retain the legacy block as read-only. Count unreviewed IDs as `selected_issue_ids - issue_reviews.keys()` in `qc_common/projection.py`.

- [x] **Step 4: Run focused tests and verify Green**

Run the Step 2 command. Expected: PASS.

- [x] **Step 5: Commit Task 1**

```bash
git add qc_common/schema.py qc_common/report_migration.py qc_common/projection.py tests/test_human_qc_report_schema.py tests/test_human_qc_aggregation.py tests/test_qc_migration_reconciliation.py openspec/changes/add-human-semantic-warn-review/tasks.md
git commit -m "feat(human-qc): add early-fail report contract"
```

### Task 2: 实现 Warn verdict、原因和完成状态机

**Files:**
- Modify: `human_qc/warn_service.py`
- Modify: `human_qc/report_updates.py`
- Modify: `tests/test_warn_review_service.py`
- Modify: `tests/test_human_qc_end_to_end.py`

**Interfaces:**
- Change: `WarnReviewService.submit_verdict(..., failure_reason: Mapping[str, object] | None = None)`
- Change: `WarnReviewService.complete(..., completion_mode: Literal["all_reviewed", "early_fail"], failure_reason: Mapping[str, object] | None = None)`
- Produces: completion transition to semantic for all-reviewed Pass, or stopped/skipped semantic for early Fail.

- [x] **Step 1: Write failing service tests**

Add tests named:

```python
def test_first_fail_can_complete_without_reviewing_remaining_selected_issues(...): ...
def test_last_fail_changed_to_pass_recloses_early_fail_gate(...): ...
def test_manual_reasons_replace_all_machine_default_reasons(...): ...
def test_other_reason_requires_non_empty_text_on_verdict_and_complete(...): ...
def test_completed_asset_rejects_verdict_changes(...): ...
```

The first test must assert that unreviewed issue IDs are absent from `issue_reviews`, `semantic_calibration.state == "skipped_due_to_fail"`, pipeline status is `stopped`, and `overall_decision == "fail"`.

- [x] **Step 2: Run tests and verify Red**

```bash
pytest -q tests/test_warn_review_service.py tests/test_human_qc_end_to_end.py
```

Expected: FAIL because `complete()` still requires every selected issue and advances directly to a completed pipeline.

- [x] **Step 3: Implement verdict and completion invariants**

Introduce normalization with stable types:

```python
FailureReason = dict[str, object]

def normalize_failure_reason(value: Mapping[str, object] | None) -> FailureReason | None:
    if value is None or not value.get("reason_codes"):
        return None
    codes = list(dict.fromkeys(str(code).strip() for code in value["reason_codes"] if str(code).strip()))
    other_text = str(value.get("other_text") or "").strip() or None
    if "other" in codes and other_text is None:
        raise WarnStateError("other_text is required when other is selected")
    return {"mode": "manual", "reason_codes": codes, "other_text": other_text}
```

`submit_verdict()` must audit replacements and atomically persist the current reason version with a Fail. `complete()` must re-read the current report under expected revision, derive actual Fail reviews, validate the requested mode, and write exactly one of the two pipeline transitions described in the design. It must never synthesize reviews for unreviewed IDs.

- [x] **Step 4: Run focused tests and verify Green**

Run the Step 2 command. Expected: PASS.

- [x] **Step 5: Commit Task 2**

```bash
git add human_qc/warn_service.py human_qc/report_updates.py tests/test_warn_review_service.py tests/test_human_qc_end_to_end.py openspec/changes/add-human-semantic-warn-review/tasks.md
git commit -m "feat(human-qc): support explicit early-fail completion"
```

### Task 3: 反转流水线门禁并兼容既有 Profile

**Files:**
- Modify: `qc_common/config.py`
- Modify: `qc_common/manual_review.py`
- Modify: `qc_pipeline/orchestrator.py`
- Modify: `tests/test_human_qc_profile_routing.py`
- Modify: `tests/test_qc_pipeline_profiles_e2e.py`
- Modify: `tests/test_report_mutation.py`

**Interfaces:**
- Produces: automatic successor order `manual_review -> semantic_consistency`
- Produces: server-side `semantic_eligibility(report) -> Literal["ready", "blocked", "skipped_due_to_fail"]`
- Preserves: acceptance automatic hard-stop and supplier-evaluation hard-fail precedence.

- [x] **Step 1: Write failing routing tests**

Cover four exact traces:

```python
assert trace_no_warn == ("manual_review:not_required", "semantic_consistency:awaiting_external")
assert trace_all_pass == ("manual_review:completed/all_reviewed", "semantic_consistency:awaiting_external")
assert trace_early_fail == ("manual_review:completed/early_fail", "pipeline:stopped")
assert report_early_fail["semantic_calibration"]["state"] == "skipped_due_to_fail"
```

Also assert that a direct semantic task request cannot bypass an incomplete manual review.

- [x] **Step 2: Run tests and verify Red**

```bash
pytest -q tests/test_human_qc_profile_routing.py tests/test_qc_pipeline_profiles_e2e.py tests/test_report_mutation.py
```

Expected: FAIL because the current order and Warn service guard require semantic completion first.

- [x] **Step 3: Implement the state transition order**

Update configured external-module order and `resume_after_external()` so that:

```python
def semantic_eligibility(report: Mapping[str, Any]) -> str:
    manual = report["manual_review"]
    if manual.get("state") == "not_required":
        return "ready"
    if manual.get("state") != "completed":
        return "blocked"
    if manual.get("completion_mode") == "early_fail":
        return "skipped_due_to_fail"
    return "ready"
```

Remove `_assert_semantic_ready()` from Warn mutation paths and replace it with a manual-review cursor guard. Make transitions idempotent under restart and expected revision.

- [x] **Step 4: Run focused tests and verify Green**

Run the Step 2 command. Expected: PASS.

- [x] **Step 5: Commit Task 3**

```bash
git add qc_common/config.py qc_common/manual_review.py qc_pipeline/orchestrator.py human_qc/warn_service.py tests/test_human_qc_profile_routing.py tests/test_qc_pipeline_profiles_e2e.py tests/test_report_mutation.py openspec/changes/add-human-semantic-warn-review/tasks.md
git commit -m "refactor(qc): place warn review before semantic calibration"
```

### Task 4: 将语义校准迁入独立模块和服务

**Files:**
- Create: `semantic_calibration/__init__.py`
- Create: `semantic_calibration/contracts.py`
- Create: `semantic_calibration/service.py`
- Create: `semantic_calibration/source_adapters.py`
- Create: `semantic_calibration/timeline.py`
- Create: `semantic_calibration/hdf5_commit.py`
- Create: `semantic_calibration/lease.py`
- Create: `semantic_calibration/http_server.py`
- Create: `semantic_calibration/static/index.html`
- Create: `semantic_calibration/static/app.js`
- Create: `semantic_calibration/static/semantic_adapter.js`
- Create: `semantic_calibration/static/semantic_adapter.test.mjs`
- Create: `semantic_calibration/static/workbench.css`
- Create: `semantic_calibration/static/package.json`
- Create: `tools/serve_semantic_calibration.py`
- Modify: `tests/test_semantic_service.py`
- Modify: `tests/test_shared_timeline.py`
- Modify: `tests/test_hdf5_semantic_commit.py`
- Create: `tests/test_semantic_calibration_http_server.py`

**Interfaces:**
- Moves existing semantic domain APIs without semantic behavior changes.
- Exposes only `/api/semantic/...` routes.
- Depends on `qc_common`; never imports `human_qc`.

- [x] **Step 1: Write failing import-boundary and route tests**

Add tests that import every module under both packages, inspect imports, and assert:

```python
assert not imports_between("semantic_calibration", "human_qc")
assert not imports_between("human_qc", "semantic_calibration")
assert request(semantic_server, "GET", "/api/warn/assets").status == 404
assert request(semantic_server, "GET", "/api/semantic/assets").status == 200
```

Retain all existing shared-boundary, pending-edit and HDF5 atomicity assertions under the new package imports.

- [x] **Step 2: Run tests and verify Red**

```bash
pytest -q tests/test_semantic_service.py tests/test_shared_timeline.py tests/test_hdf5_semantic_commit.py tests/test_semantic_calibration_http_server.py
```

Expected: FAIL because `semantic_calibration` does not exist.

- [x] **Step 3: Move the semantic implementation and build the dedicated entrypoint**

Move the existing semantic-only modules and adapter with `git mv`, update imports to `semantic_calibration.*`, and keep shared lease primitives either duplicated behind the same contract or moved into a neutral `qc_common` module. The server route table must be explicit:

```python
SEMANTIC_ROUTES = {
    "GET /api/semantic/assets",
    "GET /api/semantic/assets/{asset_id}/task",
    "POST /api/semantic/assets/{asset_id}/lease/acquire",
    "POST /api/semantic/assets/{asset_id}/lease/renew",
    "POST /api/semantic/assets/{asset_id}/lease/release",
    "POST /api/semantic/assets/{asset_id}/boundary/pending",
    "POST /api/semantic/assets/{asset_id}/text/pending",
    "POST /api/semantic/assets/{asset_id}/pending/confirm",
    "POST /api/semantic/assets/{asset_id}/pending/cancel",
    "POST /api/semantic/assets/{asset_id}/complete",
}
```

The semantic task projection must call `semantic_eligibility()` before returning task data or accepting a write.

- [x] **Step 4: Run semantic regression tests and verify Green**

Run the Step 2 command plus:

```bash
node --test semantic_calibration/static/*.test.mjs
```

Expected: all PASS.

- [x] **Step 5: Commit Task 4**

```bash
git add semantic_calibration tools/serve_semantic_calibration.py human_qc tests/test_semantic_service.py tests/test_shared_timeline.py tests/test_hdf5_semantic_commit.py tests/test_semantic_calibration_http_server.py openspec/changes/add-human-semantic-warn-review/tasks.md
git commit -m "refactor(semantic): split calibration into independent service"
```

### Task 4A: 完成 early-fail 未查看 Warn 的正式批次投影

**OpenSpec mapping:** `7.3 更新批次投影，仅统计实际 issue review，并单独统计 early-fail 后未查看 Warn`

**Files:**
- Modify: `qc_reporting/projection.py`
- Modify: `qc_reporting/aggregate.py`
- Modify: `qc_reporting/export.py`
- Modify: `tests/test_human_qc_aggregation.py`
- Modify: `tests/test_human_qc_reporting_outputs.py`
- Modify when RED proves it needed: `tests/test_qc_reporting_entrypoints.py`

**Interfaces:**
- Preserve only actual `manual_review.issue_reviews` as reviewed results; never synthesize a verdict for early-fail unviewed issues.
- Project `selected_issue_ids` and `completion_mode` into revision-scoped human-review rows.
- Count `unreviewed_selected_warn_issues` only for terminal, completed `early_fail` reports, after current-revision de-duplication and intersecting with current machine-Warn issues.
- Add the metric to all formal aggregate formats; do not introduce Feishu/Lark scope.

- [x] **Step 1: Write failing early-fail aggregation and export tests**

Cover a three-selected/one-Fail early-fail report, an in-progress report that must not count as terminal unviewed, latest-revision de-duplication, invalid selected/review relations, and formal CSV/Parquet/XLSX/Markdown parity.

- [x] **Step 2: Run aggregation tests and verify Red**

```bash
pytest -q tests/test_human_qc_aggregation.py tests/test_human_qc_reporting_outputs.py tests/test_qc_reporting_projection.py tests/test_qc_reporting_aggregate.py
```

Expected: FAIL because the formal projection drops selected IDs/completion mode and the aggregate/export do not publish the metric.

- [x] **Step 3: Implement formal early-fail projection and aggregate contract**

Keep the source-of-truth fields in the human-review row, calculate the derived count only after terminal/latest-revision gating, emit both the descriptive internal key and formal `unreviewed_selected_warn_issues` alias, and make every formal exporter require the new metric.

- [x] **Step 4: Run focused aggregation/export tests and verify Green**

Run the Step 2 command plus any CLI entrypoint coverage added by RED. Expected: all PASS.

- [x] **Step 5: Commit Task 4A**

```bash
git add qc_reporting/projection.py qc_reporting/aggregate.py qc_reporting/export.py tests/test_human_qc_aggregation.py tests/test_human_qc_reporting_outputs.py tests/test_qc_reporting_entrypoints.py openspec/changes/add-human-semantic-warn-review/tasks.md
git commit -m "feat(reporting): count early-fail unreviewed warns"
```

### Task 5: 建立 Warn-only DTO、媒体 API 和自动 Lease

**Files:**
- Create: `human_qc/warn_workbench_service.py`
- Modify: `human_qc/http_server.py`
- Modify: `tools/serve_human_qc_workbench.py`
- Modify: `tests/test_human_qc_http_server.py`
- Modify: `tests/test_human_qc_workbench.py`
- Modify: `tests/test_review_evidence.py`

**Interfaces:**
- Produces: `WarnTaskDto` with safe source-video URL, `fps`, `total_frames`, half-open issue intervals, thresholds, reason options, review state and overlay status.
- Exposes: `/api/warn/assets`, `/api/warn/assets/{id}/task`, lease, verdict, complete and overlay-status routes.
- Exposes: Range-capable `/media/assets/{id}/source` and allowlisted overlay URLs.

- [x] **Step 1: Write failing API tests**

Assert:

```python
assert task["video"] == {"url": "/media/assets/asset-1/source", "fps": 30.0, "total_frames": 1800}
assert task["issues"][0]["frame_range"] == {"start_frame": 120, "end_frame_exclusive": 169}
assert "source_path" not in json.dumps(task)
assert range_response.status == 206
assert range_response.headers["Content-Range"].startswith("bytes 0-99/")
assert acquire_called_automatically_on_task_load is True
```

Also assert 409 is sanitized and 423 returns a stable read-only code.

- [x] **Step 2: Run tests and verify Red**

```bash
pytest -q tests/test_human_qc_http_server.py tests/test_human_qc_workbench.py tests/test_review_evidence.py
```

Expected: FAIL because the current facade is shared, returns issue clips, and lacks Range media DTOs.

- [x] **Step 3: Implement the Warn DTO and route boundary**

Define immutable DTO types and explicit normalization:

```python
@dataclass(frozen=True)
class FrameRangeDto:
    start_frame: int
    end_frame_exclusive: int

@dataclass(frozen=True)
class VideoDto:
    url: str
    fps: float
    total_frames: int
```

Probe media once per source hash, clamp issue ranges to `[0, total_frames)`, and reject empty ranges. Implement byte Range parsing for a single range and stream only the requested bytes. Task load acquires or renews the current reviewer lease in the facade; lease collision returns a task with `read_only=true` rather than exposing an edit-lock button.

- [x] **Step 4: Run focused tests and verify Green**

Run the Step 2 command. Expected: PASS.

- [x] **Step 5: Commit Task 5**

```bash
git add human_qc/warn_workbench_service.py human_qc/http_server.py tools/serve_human_qc_workbench.py tests/test_human_qc_http_server.py tests/test_human_qc_workbench.py tests/test_review_evidence.py openspec/changes/add-human-semantic-warn-review/tasks.md
git commit -m "feat(human-qc): expose warn-only full-video API"
```

### Task 6: 实现时间轴纯模型和重叠投影

**Files:**
- Create: `human_qc/static/warning_timeline.js`
- Create: `human_qc/static/warning_timeline.test.mjs`
- Modify: `human_qc/static/package.json`

**Interfaces:**
- Produces: `normalizeWarnings(issues, totalFrames)`
- Produces: `mergeWarningIntervals(warnings)`
- Produces: `activeWarningsAtFrame(warnings, frame)`
- Produces: `frameToPercent(frame, totalFrames)` and `pointerToFrame(event, track, totalFrames)`
- Produces: `WarningTimeline` DOM controller with `onSeek(frame)` callback.

- [x] **Step 1: Write failing Node tests**

Use the canonical overlap fixture:

```javascript
const warnings = normalizeWarnings([
  { id: "exposure", start_frame: 120, end_frame_exclusive: 169 },
  { id: "shake", start_frame: 142, end_frame_exclusive: 182 },
  { id: "late", start_frame: 390, end_frame_exclusive: 427 },
], 1800);
assert.deepEqual(mergeWarningIntervals(warnings).map(({startFrame, endFrameExclusive}) => [startFrame, endFrameExclusive]), [[120, 182], [390, 427]]);
assert.deepEqual(activeWarningsAtFrame(warnings, 142).map(x => x.id), ["exposure", "shake"]);
assert.deepEqual(activeWarningsAtFrame(warnings, 169).map(x => x.id), ["shake"]);
```

Assert narrow blocks return `showLabel=false`, popover rows stay clickable, Warning clicks seek to their own start, and track/playhead dragging emits exact frames.

- [x] **Step 2: Run tests and verify Red**

```bash
node --test human_qc/static/warning_timeline.test.mjs
```

Expected: FAIL because the module does not exist.

- [x] **Step 3: Implement interval sweep and DOM controller**

Sort by `(startFrame, selectedIndex)`, merge only when `next.startFrame < current.endFrameExclusive`, calculate `leftPercent` and `widthPercent` from `totalFrames`, and retain all child Warning objects on each visual group. Keep the popover open while either block or popover has pointer/focus; clicking a row calls only `onSeek(row.startFrame)`.

- [x] **Step 4: Run Node tests and verify Green**

Run the Step 2 command. Expected: PASS.

- [x] **Step 5: Commit Task 6**

```bash
git add human_qc/static/warning_timeline.js human_qc/static/warning_timeline.test.mjs human_qc/static/package.json openspec/changes/add-human-semantic-warn-review/tasks.md
git commit -m "feat(human-qc): add frame-accurate warning timeline"
```

### Task 7: 实现视频控制器、逐帧和倍速

**Files:**
- Create: `human_qc/static/video_controller.js`
- Create: `human_qc/static/video_controller.test.mjs`

**Interfaces:**
- Produces: `PLAYBACK_RATES = [0.25, 0.5, 1, 1.5, 2, 3]`
- Produces: `VideoController.seekToFrame(frame)`, `stepFrame(delta)`, `setRate(rate)`, `increaseRate()`, `decreaseRate()`
- Emits: `onFrameChange(frame)` and `onPlaybackStateChange(state)`.

- [x] **Step 1: Write failing controller tests**

Assert frame clamping, exact rate sequence, preference restore, and focus gating:

```javascript
controller.seekToFrame(120);
assert.equal(video.currentTime, 4);
controller.stepFrame(-1);
assert.equal(controller.currentFrame, 119);
assert.deepEqual(stepRates(1, +1, 5), [1.5, 2, 3, 3, 3]);
assert.equal(handleArrow({key: "ArrowRight", target: reasonInput}, false), false);
```

- [x] **Step 2: Run tests and verify Red**

```bash
node --test human_qc/static/video_controller.test.mjs
```

Expected: FAIL because the module does not exist.

- [x] **Step 3: Implement frame/rate synchronization**

Use `frame / fps` for seek, clamp to `0..totalFrames-1`, update current frame from `Math.round(video.currentTime * fps)` after `timeupdate/seeked`, and only register arrow stepping while the video container owns focus. Persist only the numeric rate under `human-qc.playback-rate.v1`.

- [x] **Step 4: Run tests and verify Green**

Run the Step 2 command. Expected: PASS.

- [x] **Step 5: Commit Task 7**

```bash
git add human_qc/static/video_controller.js human_qc/static/video_controller.test.mjs openspec/changes/add-human-semantic-warn-review/tasks.md
git commit -m "feat(human-qc): add focused frame and speed controls"
```

### Task 8: 实现复核面板状态机和 Warn-only 页面

**Files:**
- Create: `human_qc/static/review_panel.js`
- Rewrite: `human_qc/static/app.js`
- Rewrite: `human_qc/static/index.html`
- Rewrite: `human_qc/static/workbench.css`
- Rewrite: `human_qc/static/workbench.test.mjs`
- Delete: `human_qc/static/semantic_adapter.js`
- Delete: `human_qc/static/warn_adapter.js`
- Modify: `tests/test_human_qc_static_contract.py`

**Interfaces:**
- Produces: `nextDecisionTarget(task, activeIssueIds, explicitIssueId)`
- Produces: `completionGate(task)`
- Produces: `normalizeReasonDraft(options, selectedCodes, otherText)`
- Produces: `ReviewPanel` and `WarnReviewApp`.
- Consumes: `VideoController`, `WarningTimeline`, Warn-only DTO/API.

- [ ] **Step 1: Write failing state and DOM contract tests**

Assert:

```javascript
assert.equal(nextDecisionTarget(task, ["exposure", "shake"], null), "exposure");
const afterFirstPass = applySavedReview(task, "exposure", "pass");
assert.equal(nextDecisionTarget(afterFirstPass, ["exposure", "shake"], null), "shake");
assert.deepEqual(completionGate(afterFirstPass), {enabled: false, mode: null});
assert.deepEqual(completionGate(withFail), {enabled: true, mode: "early_fail"});
```

Static tests must prove: title is “Warn 复核”; no lock button, score body, semantic adapter, or `±1` buttons; current frame appears once in the video; `other` reveals a required text input; bottom order is Pass/Fail → 完成复核 → 上一条/下一条.

- [ ] **Step 2: Run tests and verify Red**

```bash
node --test human_qc/static/workbench.test.mjs
pytest -q tests/test_human_qc_static_contract.py
```

Expected: FAIL because the existing shared shell and adapter implement the old layout and completion logic.

- [ ] **Step 3: Implement `ReviewPanel` and `WarnReviewApp`**

Render every issue active at the current frame. Use the earliest pending active issue as the default target; only a click on the lower status row may set `explicitIssueId`. Reason chips update draft state only. On Fail, submit the current normalized reason atomically; on Pass, ignore the draft for verdict state and select the next pending issue after a successful response. Completion sends the server-derived mode and then loads the next asset. Direct asset navigation calls `resetDraft()` before loading the target.

- [ ] **Step 4: Run Node and static tests and verify Green**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit Task 8**

```bash
git add human_qc/static tests/test_human_qc_static_contract.py openspec/changes/add-human-semantic-warn-review/tasks.md
git commit -m "feat(human-qc): build warn-only review interface"
```

### Task 9: 生成并缓存 SAM3 问题区间连续 Overlay

**Files:**
- Create: `human_qc/overlay_worker.py`
- Create: `tests/test_sam3_overlay_worker.py`
- Modify: `human_qc/evidence.py`
- Modify: `qc_pipeline/sam3_runtime.py`
- Modify: `tests/test_review_evidence.py`
- Modify: `tests/test_qc_pipeline_sam3_runner.py`

**Interfaces:**
- Produces: `merge_frame_intervals(intervals) -> tuple[tuple[int, int], ...]`
- Produces: immutable `OverlayCacheKey(source_sha256, intervals, model_hash, config_hash, renderer_version)`
- Produces: `BoundedOverlayWorker.submit(request) -> OverlayJobView`
- Produces statuses: `pending | generating | ready | failed` and allowlisted result path.

- [ ] **Step 1: Write failing worker tests**

Use a fake renderer that records frame IDs and assert:

```python
assert merge_frame_intervals(((120, 169), (142, 182), (390, 427))) == ((120, 182), (390, 427))
worker.submit(request)
wait_until_ready(worker, request.cache_key)
assert renderer.frames == list(range(120, 182)) + list(range(390, 427))
assert worker.submit(request).cache_hit is True
```

Assert model/config/renderer version changes miss cache; errors become `failed` with a stable public code; max concurrent renderer calls never exceeds the configured worker count.

- [ ] **Step 2: Run tests and verify Red**

```bash
pytest -q tests/test_sam3_overlay_worker.py tests/test_review_evidence.py tests/test_qc_pipeline_sam3_runner.py
```

Expected: FAIL because continuous overlay jobs and cache keys do not exist.

- [ ] **Step 3: Implement bounded background generation**

Use `ThreadPoolExecutor(max_workers=configured_limit)` only inside `BoundedOverlayWorker`. The HTTP layer may enqueue but never call the renderer directly. Iterate each merged half-open interval exactly once, obtain the already inference-locked segmenter from `Sam3RuntimeProvider`, write a temporary video, fsync and `os.replace` into the cache. Persist a compact manifest with cache key, intervals, status and safe relative output path.

- [ ] **Step 4: Run worker tests and verify Green**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit Task 9**

```bash
git add human_qc/overlay_worker.py human_qc/evidence.py qc_pipeline/sam3_runtime.py tests/test_sam3_overlay_worker.py tests/test_review_evidence.py tests/test_qc_pipeline_sam3_runner.py openspec/changes/add-human-semantic-warn-review/tasks.md
git commit -m "feat(sam3): cache continuous issue-window overlays"
```

### Task 10: 同步播放 Overlay 并实施局部就绪门禁

**Files:**
- Modify: `human_qc/warn_workbench_service.py`
- Modify: `human_qc/http_server.py`
- Create: `human_qc/static/overlay_controller.js`
- Create: `human_qc/static/overlay_controller.test.mjs`
- Modify: `human_qc/static/video_controller.js`
- Modify: `human_qc/static/review_panel.js`
- Modify: `human_qc/static/app.js`
- Modify: `tests/test_human_qc_http_server.py`

**Interfaces:**
- Produces: `OverlayController.setEvidence(job)`, `syncFromBase()`, `updateForFrame(frame)`.
- Exposes: overlay status polling and retry routes.
- Enforces: only the affected SAM3 issue is non-reviewable until `ready`.

- [ ] **Step 1: Write failing synchronization and API tests**

Assert base/overlay parity for play, pause, seek, frame step and speed:

```javascript
base.currentTime = 4.75;
base.playbackRate = 1.5;
overlayController.syncFromBase();
assert.equal(overlay.currentTime, 4.75 - 4.0);
assert.equal(overlay.playbackRate, 1.5);
assert.equal(overlay.hidden, false);
overlayController.updateForFrame(200);
assert.equal(overlay.hidden, true);
```

Assert the review panel disables only the generating SAM3 issue; polling `ready` unlocks without reloading the asset; failed status exposes retry but no traceback.

- [ ] **Step 2: Run tests and verify Red**

```bash
node --test human_qc/static/overlay_controller.test.mjs human_qc/static/video_controller.test.mjs human_qc/static/workbench.test.mjs
pytest -q tests/test_human_qc_http_server.py
```

Expected: FAIL because no overlay video controller or readiness route exists.

- [ ] **Step 3: Implement overlay synchronization and readiness polling**

Maintain one muted, no-controls overlay video element. Translate base time to overlay-local time with `(currentFrame - startFrame) / fps`; hide outside the half-open interval. Correct drift greater than one frame on `timeupdate`, and hard-sync on `seeked`, `ratechange`, frame step and source change. `ReviewPanel.canSubmit(issue)` must require `overlay.status === "ready"` only when the issue declares continuous SAM3 evidence.

- [ ] **Step 4: Run focused tests and verify Green**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit Task 10**

```bash
git add human_qc/warn_workbench_service.py human_qc/http_server.py human_qc/static tests/test_human_qc_http_server.py openspec/changes/add-human-semantic-warn-review/tasks.md
git commit -m "feat(human-qc): synchronize interval overlay evidence"
```

### Task 11: 完成 API、浏览器和恢复验收

**Files:**
- Modify: `tests/test_human_qc_end_to_end.py`
- Modify: `tests/test_human_qc_profile_routing.py`
- Create: `tests/test_warn_review_browser_contract.py`
- Modify: `tests/test_human_qc_recovery.py`
- Modify: `tests/test_human_qc_workbench.py`

**Interfaces:**
- Verifies the complete workflow; does not introduce new production interfaces.

- [ ] **Step 1: Add end-to-end scenarios**

Cover one full desktop fixture with Warning ranges 120–168, 142–181 and 390–426. Assert timeline block count, popover choice, exact seek frames, drag seek, first-Pass order, return-and-modify, reason draft behavior, 409/423 recovery, overlay lock/unlock, explicit completion and automatic next asset.

- [ ] **Step 2: Run scenarios and verify Red for remaining integration gaps**

```bash
pytest -q tests/test_warn_review_browser_contract.py tests/test_human_qc_end_to_end.py tests/test_human_qc_profile_routing.py tests/test_human_qc_recovery.py tests/test_human_qc_workbench.py
```

Expected before integration fixes: at least one FAIL identifies a missing real route, state restoration or DOM connection; no test may use an implementation-only shortcut.

- [ ] **Step 3: Fix only the identified integration seams**

Wire the already defined APIs and controllers; do not add alternative state paths. Ensure task reload uses server revision, lease renew survives asset changes, completion response provides next asset or explicit empty-queue state, and direct navigation clears only unsaved reason draft.

- [ ] **Step 4: Run integration suites and verify Green**

Run the Step 2 command and all static Node tests. Expected: PASS.

- [ ] **Step 5: Commit Task 11**

```bash
git add human_qc semantic_calibration tests/test_warn_review_browser_contract.py tests/test_human_qc_end_to_end.py tests/test_human_qc_profile_routing.py tests/test_human_qc_recovery.py tests/test_human_qc_workbench.py openspec/changes/add-human-semantic-warn-review/tasks.md
git commit -m "test(human-qc): cover complete warn review workflow"
```

### Task 12: 同步文档并执行完整验证

**Files:**
- Modify: `docs/PRD-asset-qc-pipeline.md`
- Modify: `docs/asset-qc-json-format.md`
- Modify: `docs/reviewer-guide.md`
- Modify: `tools/serve_human_qc_workbench.py`
- Modify: `openspec/changes/add-human-semantic-warn-review/tasks.md`

**Interfaces:**
- Documents the final contract and operating commands.

- [ ] **Step 1: Update user and data-contract documentation**

Document the exact flow `自动 QC → Warn 人工复核 → 语义校准`, `completion_mode`, unreviewed selected issues, failure-reason precedence, full-video controls, overlap popover, auto lease, SAM3 readiness and both independent server commands. Remove instructions for the shared workbench, edit-lock button, issue-only clips and score body.

- [ ] **Step 2: Run focused documentation and OpenSpec validation**

```bash
pytest -q tests/test_qc_docs_contract.py tests/test_human_qc_static_contract.py
openspec validate add-human-semantic-warn-review --strict
git diff --check
```

Expected: all PASS and no whitespace errors.

- [ ] **Step 3: Run complete Python and Node suites**

```bash
pytest -q
node --test human_qc/static/*.test.mjs semantic_calibration/static/*.test.mjs
```

Expected: all PASS with no skipped test introduced for this change.

- [ ] **Step 4: Perform final dependency and browser-source checks**

```bash
python - <<'PY'
from pathlib import Path

for root, forbidden in ((Path("human_qc"), "semantic_calibration"), (Path("semantic_calibration"), "human_qc")):
    hits = [str(path) for path in root.rglob("*.py") if forbidden in path.read_text(encoding="utf-8")]
    assert not hits, hits
PY
if rg -n "获取编辑锁|语义与 Warn 复核|[+-]1 帧|machine score" human_qc/static docs/reviewer-guide.md; then exit 1; fi
```

Expected: dependency script exits 0; `rg` returns no matches.

- [ ] **Step 5: Mark tasks complete and commit documentation**

```bash
git add docs tools/serve_human_qc_workbench.py openspec/changes/add-human-semantic-warn-review/tasks.md
git commit -m "docs: publish warn review workflow"
```

## Plan Self-Review

- Spec coverage: Tasks 1–3 cover report and pipeline contracts; Task 4 covers module separation; Tasks 5–8 cover full-video Warn UI and interaction; Tasks 9–10 cover continuous overlay and readiness; Tasks 11–12 cover recovery, browser acceptance, documentation and full verification.
- Placeholder scan: the plan contains no deferred field, unnamed test command or unspecified implementation step.
- Type consistency: all frame ranges are half-open; all UI seeks use source-frame integers; `completion_mode`, `failure_reason`, overlay statuses and component method names are consistent across producer, DTO, API and browser tasks.
