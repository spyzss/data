import test from "node:test";
import assert from "node:assert/strict";

import {
  WarnReviewAdapter,
  allSelectedIssuesReviewed,
  buildWarnIssueModel,
  nextReviewIssueId,
  renderWarnMarkup,
} from "./warn_adapter.js";
import { WorkbenchApp, stageTypeForTask } from "./app.js";


const task = {
  asset_id: "asset-1",
  revision: 7,
  task_type: "warn_review",
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
  evidence: [{
    issue_id: "warn-1",
    start_frame: 30,
    end_frame_exclusive: 43,
    clip_url: "/media/asset-1/warn-1.mp4",
    overlay_url: "/media/asset-1/warn-1.png",
    generation_error: null,
  }],
};


test("warn model keeps machine fields and half-open evidence read-only", () => {
  const model = buildWarnIssueModel(task, "warn-1");
  assert.equal(model.issueId, "warn-1");
  assert.equal(model.reason, "关节速度超过阈值");
  assert.deepEqual(model.metrics, { joint_velocity_max: 1.4 });
  assert.deepEqual(model.threshold, { operator: ">", value: 1.0 });
  assert.deepEqual(model.window, { startFrame: 30, endFrameExclusive: 43 });
  assert.equal(model.overlayUrl, "/media/asset-1/warn-1.png");
  const markup = renderWarnMarkup(task, "warn-1");
  assert.match(markup, /30–42/);
  assert.match(markup, /关节速度超过阈值/);
  assert.match(markup, /joint_velocity_max/);
  assert.match(markup, /data-action="toggle-overlay"/);
  assert.match(markup, /data-action="verdict-pass"/);
  assert.match(markup, /data-action="verdict-fail"/);
  assert.doesNotMatch(markup, /timeline-track|semantic-text-slot|boundary-handle/);
});


test("warn markup follows video-first rationale-then-decision flow", () => {
  const markup = renderWarnMarkup(task, "warn-1");
  assert.match(markup, /class="warn-rationale"/);
  assert.match(markup, /class="warn-decision"/);
  assert.ok(markup.indexOf("warn-rationale") < markup.indexOf("warn-decision"));
  assert.match(markup, /data-action="verdict-pass"/);
  assert.match(markup, /data-action="verdict-fail"/);
  assert.doesNotMatch(markup, /class="warn-layout"/);
  assert.doesNotMatch(markup, /<pre[^>]*data-machine-metrics/);
});


test("warn markup renders every sampled SAM3 overlay image in server order", () => {
  const sampled = structuredClone(task);
  sampled.evidence[0].overlay_images = [
    { frame: 120, url: "/media/frame-120.png" },
    { frame: 144, url: "/media/frame-144.png" },
    { frame: 188, url: "/media/frame-188.png" },
  ];
  const markup = renderWarnMarkup(sampled, "warn-1");
  assert.equal((markup.match(/class="warn-overlay-sample"/g) || []).length, 3);
  assert.ok(markup.indexOf("frame-120.png") < markup.indexOf("frame-188.png"));
});


test("warn model converts inclusive canonical context end when evidence projection degrades", () => {
  const degraded = structuredClone(task);
  degraded.evidence = [];
  const issue = degraded.warn.selected_issues["warn-1"];
  delete issue.start_frame;
  delete issue.end_frame_exclusive;
  issue.context = { start_frame: 30, end_frame: 42 };
  assert.deepEqual(buildWarnIssueModel(degraded, "warn-1").window, {
    startFrame: 30,
    endFrameExclusive: 43,
  });
});


test("reviewed issue advances to the first unresolved issue", () => {
  assert.equal(nextReviewIssueId(task), "warn-2");
  assert.equal(allSelectedIssuesReviewed(task), false);
  const complete = structuredClone(task);
  complete.warn.issue_reviews["warn-2"] = { verdict: "pass" };
  assert.equal(allSelectedIssuesReviewed(complete), true);
});


test("adapter submits verdict for the explicitly selected issue", async () => {
  const calls = [];
  const adapter = new WarnReviewAdapter({
    onVerdict: async (...args) => calls.push(args),
  });
  adapter.task = task;
  adapter.selectedIssueId = "warn-2";
  await adapter.submitVerdict("warn-2", "pass", "clear");
  assert.deepEqual(calls, [["warn-2", "pass", "clear"]]);
});


test("human application treats only warning work as editable", () => {
  assert.equal(stageTypeForTask(task), "warn_review");
  assert.equal(stageTypeForTask({ task_type: "completed", warn: { state: "completed" } }), "completed");
});


test("status exposes the active task type on the app shell", () => {
  const root = { dataset: {}, querySelector: () => null };
  const app = new WorkbenchApp({ root });
  app.task = task;
  app.renderStatus();
  assert.equal(root.dataset.taskType, "warn_review");
});


test("application uses the explicit issue-id verdict endpoint", async () => {
  const calls = [];
  const app = new WorkbenchApp({
    fetcher: async (path, options) => {
      calls.push([path, JSON.parse(options.body)]);
      return { ok: true, status: 200, json: async () => ({ task: { ...task, revision: 8 } }) };
    },
  });
  app.task = task;
  app.assetId = task.asset_id;
  app.lease = { token: "lease-1" };
  await app.submitWarnVerdict("warn-2", "pass", "clear");
  assert.equal(calls[0][0], "/api/assets/asset-1/warn/warn-2/verdict");
  assert.equal(calls[0][1].expected_revision, 7);
  assert.equal(app.task.revision, 8);
});


test("409 conflict never overwrites the current task", async () => {
  const app = new WorkbenchApp({
    fetcher: async () => ({
      ok: false,
      status: 409,
      json: async () => ({ error: { code: "stale_revision", message: "refresh" } }),
    }),
  });
  app.applyServerTask(task);
  app.renderStatus = () => {};
  await assert.rejects(() => app.requestTask("asset-1"), /refresh/);
  assert.equal(app.task.revision, 7);
  assert.equal(app.lastError.code, "stale_revision");
});


test("switching assets clears the old lease", async () => {
  const app = new WorkbenchApp({ fetcher: null });
  app.task = task;
  app.assetId = "asset-1";
  app.lease = { token: "old-token" };
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


test("evidence degradation uses stable safe operator copy", () => {
  const degraded = structuredClone(task);
  degraded.evidence = [{ issue_id: "warn-1", generation_error: "clip_unavailable" }];
  const markup = renderWarnMarkup(degraded, "warn-1");
  assert.match(markup, /问题片段暂不可用/);
  assert.doesNotMatch(markup, /ffmpeg|Command|\/private\//i);
});


test("one overlay sample failure is isolated from video and healthy samples", () => {
  let failedImageError = null;
  const failedSample = { hidden: false };
  const failedImage = {
    hidden: false,
    closest: () => failedSample,
    addEventListener(type, handler) { if (type === "error") failedImageError = handler; },
  };
  const healthySample = { hidden: false };
  const healthyImage = { hidden: false, closest: () => healthySample, addEventListener() {} };
  const degradation = { hidden: true, textContent: "" };
  const root = {
    innerHTML: "",
    querySelectorAll(selector) {
      if (selector === '[data-action="select-issue"]') return [];
      if (selector === "[data-warn-overlay-sample]") return [failedImage, healthyImage];
      return [];
    },
    querySelector(selector) {
      if (selector === "[data-overlay-sample-error]") return degradation;
      return null;
    },
  };
  const sampled = structuredClone(task);
  sampled.warn.issue_reviews = {};
  sampled.evidence[0].overlay_images = [
    { frame: 120, url: "/evidence/frame-120.png" },
    { frame: 144, url: "/evidence/frame-144.png" },
  ];
  const video = { src: "", currentTime: -1, dataset: {}, addEventListener() {}, removeEventListener() {} };
  const adapter = new WarnReviewAdapter({ video });
  adapter.render(sampled, root);
  failedImageError();
  assert.equal(failedImage.hidden, true);
  assert.equal(healthyImage.hidden, false);
  assert.equal(video.src, sampled.evidence[0].clip_url);
  assert.match(degradation.textContent, /部分骨架抽样图加载失败/);
});


test("missing clip clears stale video and reveals the placeholder", () => {
  const video = {
    src: "/old.mp4",
    currentTime: 9,
    dataset: {},
    pause() {},
    load() {},
    removeAttribute(name) { if (name === "src") this.src = ""; },
    addEventListener() {},
    removeEventListener() {},
  };
  const placeholder = { hidden: true };
  const adapter = new WarnReviewAdapter({ video, videoPlaceholder: placeholder });
  const noClip = structuredClone(task);
  noClip.evidence = [];
  adapter.task = noClip;
  adapter.configureVideo(buildWarnIssueModel(noClip, "warn-1"));
  assert.equal(video.src, "");
  assert.equal(video.currentTime, 0);
  assert.equal(placeholder.hidden, false);
});


test("configured warn evidence video hides the opaque media placeholder", () => {
  const video = {
    src: "",
    currentTime: 9,
    dataset: {},
    addEventListener() {},
    removeEventListener() {},
  };
  const placeholder = { hidden: false };
  const adapter = new WarnReviewAdapter({ video, videoPlaceholder: placeholder });
  adapter.task = task;
  adapter.configureVideo(buildWarnIssueModel(task, "warn-1"));
  assert.equal(video.src, "/media/asset-1/warn-1.mp4");
  assert.equal(video.currentTime, 0);
  assert.equal(placeholder.hidden, true);
});


test("queued candidates and empty candidates render distinct stable states", () => {
  const queued = { task_type: "warn_review", warn: { candidate_issue_ids: ["w1"], selected_issue_ids: [] } };
  const empty = { task_type: "warn_review", warn: { candidate_issue_ids: [], selected_issue_ids: [] } };
  assert.match(renderWarnMarkup(queued), /data-warn-queued/);
  assert.match(renderWarnMarkup(empty), /data-warn-empty/);
});


test("completion remains gated until every selected warning has a verdict", async () => {
  const adapter = new WarnReviewAdapter({ onComplete: async () => "done" });
  adapter.task = task;
  await assert.rejects(() => adapter.complete(), /all selected issues/);
  const completed = structuredClone(task);
  completed.warn.issue_reviews["warn-2"] = { verdict: "pass" };
  adapter.task = completed;
  assert.equal(await adapter.complete(), "done");
});


test("Pass and Fail save errors remain visible to the operator", async () => {
  const errorBox = { textContent: "" };
  const root = {
    querySelector(selector) {
      if (selector === "[data-review-reason]") return { value: "reason" };
      if (selector === ".warn-error") return errorBox;
      return null;
    },
  };
  const adapter = new WarnReviewAdapter({ onVerdict: async () => { throw new Error("save failed"); } });
  adapter.task = task;
  adapter.root = root;
  adapter.selectedIssueId = "warn-2";
  assert.equal(await adapter.submitCurrentVerdict("pass"), null);
  assert.match(errorBox.textContent, /save failed/);
  assert.equal(await adapter.submitCurrentVerdict("fail"), null);
  assert.match(errorBox.textContent, /save failed/);
});
