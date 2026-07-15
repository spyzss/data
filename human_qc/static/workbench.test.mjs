import test from "node:test";
import assert from "node:assert/strict";

import {
  SemanticCalibrationAdapter,
  buildTimelineModel,
  linkedBoundaryPreview,
  makeBoundaryPayload,
  pendingPresentation,
  renderTimelineMarkup,
} from "./semantic_adapter.js";
import {
  WarnReviewAdapter,
  allSelectedIssuesReviewed,
  buildWarnIssueModel,
  renderWarnMarkup,
} from "./warn_adapter.js";
import { WorkbenchApp, mutationControlsDisabled } from "./app.js";

const segments = [
  { internal_id: "s1", start_frame: 0, end_frame_exclusive: 51, text_cn: "接近杯子", text_en: "approach" },
  { internal_id: "s2", start_frame: 51, end_frame_exclusive: 123, text_cn: "拿起杯子", text_en: "pick up" },
  { internal_id: "s3", start_frame: 123, end_frame_exclusive: 195, text_cn: "放入托盘", text_en: "place" },
];

const task = {
  asset_id: "asset-1",
  revision: 4,
  task_type: "semantic_calibration",
  semantic: {
    report_revision: 4,
    report_state: "in_progress",
    timeline: { frame_count: 195, fps: 30, segments },
    pending_edit: null,
  },
};

test("N segments render exactly N-1 internal shared-boundary handles", () => {
  const model = buildTimelineModel(task.semantic);
  assert.equal(model.segments.length, 3);
  assert.equal(model.handles.length, 2);
  assert.deepEqual(model.handles.map((item) => item.boundary_index), [1, 2]);

  const markup = renderTimelineMarkup(model);
  assert.equal((markup.match(/class="boundary-handle/g) || []).length, 2);
  assert.equal((markup.match(/class="timeline-segment/g) || []).length, 3);
  assert.doesNotMatch(markup, /class="timeline-segment[^>]*draggable=/);
  assert.match(markup, /data-end-frame="50"/);
  assert.match(markup, /data-end-frame="122"/);
  assert.doesNotMatch(markup, /data-end-frame="123"/);
});

test("boundary payload carries shared boundary index, actor segment, and exclusive frame", () => {
  assert.deepEqual(
    makeBoundaryPayload({
      boundaryIndex: 1,
      actorSegmentId: "s2",
      frameExclusive: 60,
      expectedRevision: 4,
      leaseToken: "lease-1",
    }),
    {
      boundary_index: 1,
      actor_segment_id: "s2",
      new_frame_exclusive: 60,
      expected_revision: 4,
      lease_token: "lease-1",
    },
  );
});

test("drag preview resizes exactly the two adjacent segments", () => {
  const model = buildTimelineModel(task.semantic);
  const preview = linkedBoundaryPreview(model, 1, 60);
  assert.deepEqual(preview.previous, {
    startFrame: 0,
    endFrameExclusive: 60,
    widthPercent: (60 / 195) * 100,
  });
  assert.deepEqual(preview.following, {
    startFrame: 60,
    endFrameExclusive: 123,
    widthPercent: (63 / 195) * 100,
  });
  assert.equal(model.segments[2].start_frame, 123);
});

test("pending presentation includes both affected before/after snapshots and locks edits", () => {
  const pendingTask = {
    ...task,
    semantic: {
      ...task.semantic,
      pending_edit: {
        edit_type: "boundary",
        affected_segment_ids: ["s1", "s2"],
        before: [
          { internal_id: "s1", start_frame: 0, end_frame_exclusive: 51 },
          { internal_id: "s2", start_frame: 51, end_frame_exclusive: 123 },
        ],
        after: [
          { internal_id: "s1", start_frame: 0, end_frame_exclusive: 60 },
          { internal_id: "s2", start_frame: 60, end_frame_exclusive: 123 },
        ],
      },
    },
  };
  const view = pendingPresentation(pendingTask.semantic.pending_edit);
  assert.deepEqual(view.affectedSegmentIds, ["s1", "s2"]);
  assert.deepEqual(view.before.map((item) => item.end_frame_exclusive), [51, 123]);
  assert.deepEqual(view.after.map((item) => item.start_frame), [0, 60]);
  assert.equal(mutationControlsDisabled(pendingTask), true);
});

test("adapter accepts boundary drag payloads only for internal handles", () => {
  const adapter = new SemanticCalibrationAdapter({
    postPending: () => {},
  });
  adapter.model = buildTimelineModel(task.semantic);
  assert.throws(() => adapter.beginBoundaryDrag(0, "s1", 20), /internal boundary/);
  assert.throws(() => adapter.beginBoundaryDrag(3, "s3", 170), /internal boundary/);
  assert.deepEqual(adapter.beginBoundaryDrag(1, "s2", 60), {
    boundary_index: 1,
    actor_segment_id: "s2",
    new_frame_exclusive: 60,
  });
});

test("WorkbenchApp keeps server-conflict errors visible without overwriting the task", async () => {
  const app = new WorkbenchApp({
    fetcher: async () => ({
      ok: false,
      status: 409,
      json: async () => ({ error: { code: "stale_revision", message: "refresh" } }),
    }),
  });
  app.applyServerTask(task);
  let statusRenders = 0;
  app.renderStatus = () => { statusRenders += 1; };
  await assert.rejects(() => app.requestTask("asset-1"), /refresh/);
  assert.equal(app.task.revision, 4);
  assert.equal(app.lastError.code, "stale_revision");
  assert.equal(statusRenders, 1);
});

test("loading a different asset clears the prior asset lease", async () => {
  const app = new WorkbenchApp({ fetcher: null });
  app.task = task;
  app.assetId = "asset-1";
  app.lease = { token: "old-token", expires_at: "later" };
  app.leaseTimer = setInterval(() => {}, 60_000);
  app.leaseTimer.unref?.();
  app.requestTask = async (assetId) => {
    assert.equal(assetId, "asset-2");
    assert.equal(app.lease, null);
    assert.equal(app.leaseTimer, null);
    return { asset_id: assetId };
  };
  await app.loadAsset("asset-2");
});

const warnTask = {
  asset_id: "asset-1",
  revision: 7,
  task_type: "warn_review",
  semantic: { state: "completed" },
  warn: {
    state: "in_progress",
    selected_issue_ids: ["warn-1", "warn-2"],
    selected_issue_id: "warn-1",
    selected_issues: {
      "warn-1": {
        issue_id: "warn-1",
        code: "motion_spike",
        reason: "关节速度超过阈值",
        metric: "joint_velocity_max",
        observed_value: 1.4,
        operator: ">",
        boundary_value: 1.0,
        context: { start_frame: 30, end_frame: 42 },
      },
      "warn-2": {
        issue_id: "warn-2",
        code: "blur_warning",
        reason: "画面清晰度偏低",
        metrics: { blur_score: 0.2 },
        threshold: { operator: "<", value: 0.5 },
      },
    },
    issue_reviews: {
      "warn-1": { verdict: "pass", reason: "动作本身正常" },
    },
  },
  evidence: [
    {
      issue_id: "warn-1",
      start_frame: 30,
      end_frame_exclusive: 43,
      clip_url: "/evidence/asset-1/warn-1.mp4",
      overlay_url: "/evidence/asset-1/warn-1.png",
      generation_error: null,
    },
  ],
};

test("warn model keeps machine reason metrics threshold and half-open evidence read-only", () => {
  const model = buildWarnIssueModel(warnTask, "warn-1");
  assert.equal(model.issueId, "warn-1");
  assert.equal(model.reason, "关节速度超过阈值");
  assert.deepEqual(model.metrics, { joint_velocity_max: 1.4 });
  assert.deepEqual(model.threshold, { operator: ">", value: 1.0 });
  assert.deepEqual(model.window, { startFrame: 30, endFrameExclusive: 43 });
  assert.equal(model.overlayUrl, "/evidence/asset-1/warn-1.png");

  const markup = renderWarnMarkup(warnTask, "warn-1");
  assert.match(markup, /关节速度超过阈值/);
  assert.match(markup, /joint_velocity_max/);
  assert.match(markup, /30–42/);
  assert.match(markup, /data-action="toggle-overlay"/);
  assert.match(markup, /data-action="verdict-pass"/);
  assert.match(markup, /data-action="verdict-fail"/);
  assert.doesNotMatch(markup, /timeline-track|semantic-text-slot|boundary-handle/);
});

test("warn completion stays disabled until every selected issue has a verdict", () => {
  assert.equal(allSelectedIssuesReviewed(warnTask), false);
  assert.match(renderWarnMarkup(warnTask, "warn-1"), /data-action="complete-warn"[^>]*disabled/);
  const complete = structuredClone(warnTask);
  complete.warn.issue_reviews["warn-2"] = { verdict: "fail", reason: "confirmed" };
  assert.equal(allSelectedIssuesReviewed(complete), true);
  assert.doesNotMatch(renderWarnMarkup(complete, "warn-1"), /data-action="complete-warn"[^>]*disabled/);
});

test("warn adapter submits verdict for the explicitly selected issue", async () => {
  const submissions = [];
  const adapter = new WarnReviewAdapter({
    onVerdict: async (...args) => submissions.push(args),
  });
  adapter.render(warnTask);
  adapter.selectIssue("warn-2");
  await adapter.submitVerdict("warn-2", "fail", "confirmed");
  assert.deepEqual(submissions, [["warn-2", "fail", "confirmed"]]);
  await assert.rejects(() => adapter.submitVerdict("not-selected", "pass", ""), /selected/);
});

test("task_type switches mutually exclusively between semantic, warn, and completed", () => {
  let semanticConstructed = 0;
  let warnConstructed = 0;
  const renders = [];
  const stage = { innerHTML: "", querySelector: () => null };
  const root = {
    querySelector(selector) {
      return selector === "[data-workbench-stage]" ? stage : null;
    },
  };
  const app = new WorkbenchApp({
    documentRef: {},
    root,
    semanticAdapterFactory: () => {
      semanticConstructed += 1;
      return { render: () => renders.push("semantic") };
    },
    warnAdapterFactory: () => {
      warnConstructed += 1;
      return { render: () => renders.push("warn") };
    },
  });

  app.advanceStage(task);
  assert.deepEqual([semanticConstructed, warnConstructed, renders.at(-1)], [1, 0, "semantic"]);
  app.advanceStage(warnTask);
  assert.deepEqual([semanticConstructed, warnConstructed, renders.at(-1)], [1, 1, "warn"]);
  app.advanceStage({ ...warnTask, task_type: "completed", warn: { state: "completed", selected_issue_ids: [] } });
  assert.equal(app.adapter, null);
  assert.match(stage.innerHTML, /已完成/);
});

test("warn tasks advance only for explicitly empty candidates or a terminal state", () => {
  let warnConstructed = 0;
  const stage = { innerHTML: "" };
  const root = { querySelector: (selector) => selector === "[data-workbench-stage]" ? stage : null };
  const app = new WorkbenchApp({
    documentRef: {},
    root,
    warnAdapterFactory: () => {
      warnConstructed += 1;
      return { render: () => {} };
    },
  });
  assert.equal(app.advanceStage({
    ...warnTask,
    warn: {
      ...warnTask.warn,
      candidate_issue_ids: [],
      selected_issue_ids: [],
      selected_issues: {},
    },
  }), "completed");
  assert.equal(warnConstructed, 0);
  assert.match(stage.innerHTML, /已完成/);
  assert.equal(app.advanceStage({
    ...warnTask,
    warn: { ...warnTask.warn, state: "completed" },
  }), "completed");
  assert.equal(warnConstructed, 0);
});

test("non-empty warn candidates do not auto-complete when selection is temporarily empty", () => {
  let warnConstructed = 0;
  const stage = { innerHTML: "" };
  const root = { querySelector: (selector) => selector === "[data-workbench-stage]" ? stage : null };
  const app = new WorkbenchApp({
    documentRef: {},
    root,
    warnAdapterFactory: () => {
      warnConstructed += 1;
      return { render: () => {} };
    },
  });
  const queuedTask = {
    ...warnTask,
    warn: {
      ...warnTask.warn,
      candidate_issue_ids: ["warn-1"],
      selected_issue_ids: [],
      selected_issues: {},
    },
  };
  assert.equal(app.advanceStage(queuedTask), "warn_review");
  assert.equal(warnConstructed, 1);
  const markup = renderWarnMarkup(queuedTask);
  assert.match(markup, /data-warn-queued/);
  assert.match(markup, /1 个.*等待选择/);
  assert.doesNotMatch(markup, /任务已完成|data-action="complete-warn"/);
});

test("configured warn evidence video hides the opaque media placeholder", () => {
  const video = {
    src: "",
    currentTime: -1,
    dataset: {},
    addEventListener() {},
    removeEventListener() {},
  };
  const videoPlaceholder = { hidden: false };
  const adapter = new WarnReviewAdapter({ video, videoPlaceholder });
  adapter.task = warnTask;
  adapter.configureVideo(buildWarnIssueModel(warnTask, "warn-1"));
  assert.equal(video.src, "/evidence/asset-1/warn-1.mp4");
  assert.equal(video.currentTime, 0);
  assert.equal(videoPlaceholder.hidden, true);
});

test("switching from a clipped issue to an issue without a clip clears stale video", () => {
  let loadCount = 0;
  let pauseCount = 0;
  let removedSource = false;
  const video = {
    src: "",
    currentTime: -1,
    dataset: {},
    addEventListener() {},
    removeEventListener() {},
    removeAttribute(name) {
      if (name === "src") {
        removedSource = true;
        this.src = "";
      }
    },
    load() { loadCount += 1; },
    pause() { pauseCount += 1; },
  };
  const videoPlaceholder = { hidden: false };
  const adapter = new WarnReviewAdapter({ video, videoPlaceholder });
  const sequentialTask = structuredClone(warnTask);
  sequentialTask.evidence.push({
    issue_id: "warn-2",
    generation_error: "clip generation failed",
  });
  adapter.task = sequentialTask;
  adapter.configureVideo(buildWarnIssueModel(sequentialTask, "warn-1"));
  assert.equal(video.src, "/evidence/asset-1/warn-1.mp4");
  assert.equal(videoPlaceholder.hidden, true);

  adapter.configureVideo(buildWarnIssueModel(sequentialTask, "warn-2"));
  assert.equal(removedSource, true);
  assert.equal(video.src, "");
  assert.equal(loadCount, 1);
  assert.equal(pauseCount, 1);
  assert.equal(videoPlaceholder.hidden, false);
});

test("rejected Pass or Fail saves are caught and shown in the warn error region", async () => {
  const visibleError = { textContent: "" };
  const adapter = new WarnReviewAdapter({ onVerdict: async () => { throw new Error("lease expired"); } });
  adapter.task = warnTask;
  adapter.selectedIssueId = "warn-1";
  adapter.root = {
    querySelector(selector) {
      if (selector === "[data-review-reason]") return { value: "reviewed" };
      if (selector === ".warn-error") return visibleError;
      return null;
    },
  };
  assert.equal(await adapter.submitCurrentVerdict("pass"), null);
  assert.equal(visibleError.textContent, "lease expired");
});

test("overlay image load errors visibly degrade while preserving video evidence", () => {
  let overlayErrorHandler = null;
  const overlay = {
    hidden: false,
    addEventListener(type, handler) {
      if (type === "error") overlayErrorHandler = handler;
    },
  };
  const toggle = { checked: true, disabled: false, addEventListener() {} };
  const degraded = { hidden: true, textContent: "" };
  const root = {
    innerHTML: "",
    querySelectorAll: () => [],
    querySelector(selector) {
      if (selector === "[data-warn-overlay]") return overlay;
      if (selector === '[data-action="toggle-overlay"]') return toggle;
      if (selector === "[data-overlay-error]") return degraded;
      return null;
    },
  };
  const adapter = new WarnReviewAdapter();
  adapter.render(warnTask, root);
  assert.equal(typeof overlayErrorHandler, "function");
  overlayErrorHandler();
  assert.equal(overlay.hidden, true);
  assert.equal(toggle.disabled, true);
  assert.equal(toggle.checked, false);
  assert.equal(degraded.hidden, false);
  assert.match(degraded.textContent, /overlay.*加载失败/);
  assert.match(root.innerHTML, /打开问题窗口视频/);
});

test("WorkbenchApp uses the issue-id verdict endpoint and server revision refresh", async () => {
  let request = null;
  const app = new WorkbenchApp({ fetcher: async (path, options) => {
    request = { path, options };
    return { ok: true, status: 200, json: async () => ({ task: { ...warnTask, revision: 8 } }) };
  } });
  app.task = warnTask;
  app.assetId = "asset-1";
  app.lease = { token: "lease-1" };
  await app.submitWarnVerdict("warn-1", "pass", "normal motion");
  assert.equal(request.path, "/api/assets/asset-1/warn/warn-1/verdict");
  assert.deepEqual(JSON.parse(request.options.body), {
    verdict: "pass",
    reason: "normal motion",
    expected_revision: 7,
    lease_token: "lease-1",
  });
  assert.equal(app.revision(), 8);
});
