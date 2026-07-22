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
  assert.deepEqual(model.metrics, { joint_velocity_max: 1.4 });
  assert.deepEqual(model.threshold, { operator: ">", value: 1.0 });
  assert.deepEqual(model.window, { startFrame: 30, endFrameExclusive: 43 });
  const markup = renderWarnMarkup(task, "warn-1");
  assert.match(markup, /30–42/);
  assert.match(markup, /data-action="verdict-pass"/);
  assert.match(markup, /data-action="verdict-fail"/);
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
