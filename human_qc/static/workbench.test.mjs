import test from "node:test";
import assert from "node:assert/strict";

import {
  ReviewPanel,
  applySavedReview,
  completionGate,
  nextDecisionTarget,
  normalizeReasonDraft,
} from "./review_panel.js";
import { WarnReviewApp } from "./app.js";


function canonicalTask(overrides = {}) {
  return {
    asset_id: "asset-1",
    report_revision: 7,
    manual_review_state: "in_progress",
    completion_mode: null,
    failure_reason: null,
    can_complete: false,
    video: {
      url: "/media/assets/asset-1/source",
      fps: 30,
      total_frames: 600,
    },
    issues: [
      {
        id: "exposure",
        display_name: "曝光异常",
        frame_range: { start_frame: 120, end_frame_exclusive: 169 },
        default_reason: "曝光异常",
        threshold: { operator: ">", value: 0.7 },
        evidence_type: "source_video",
        review: null,
        overlay: null,
      },
      {
        id: "shake",
        display_name: "画面抖动",
        frame_range: { start_frame: 142, end_frame_exclusive: 182 },
        default_reason: "画面抖动",
        threshold: { operator: ">", value: 0.5 },
        evidence_type: "source_video",
        review: null,
        overlay: null,
      },
      {
        id: "late",
        display_name: "画面抖动",
        frame_range: { start_frame: 390, end_frame_exclusive: 427 },
        default_reason: "画面抖动",
        threshold: { operator: ">", value: 0.5 },
        evidence_type: "source_video",
        review: null,
        overlay: null,
      },
    ],
    reason_options: [
      { code: "occlusion", display_name: "遮挡", requires_text: false },
      { code: "action_unrecognizable", display_name: "动作不可辨", requires_text: false },
      { code: "inaccurate_interval", display_name: "标注区间不准确", requires_text: false },
      { code: "other", display_name: "其他", requires_text: true },
    ],
    lease: { read_only: false, token: "lease-1", expires_at: "2099-01-01T00:00:00Z", code: null },
    ...overrides,
  };
}


test("the earliest pending active warning wins even when active ids arrive in another order", () => {
  const task = canonicalTask();
  assert.equal(nextDecisionTarget(task, ["shake", "exposure"], null), "exposure");

  const afterFirstPass = applySavedReview(task, "exposure", "pass");
  assert.equal(nextDecisionTarget(afterFirstPass, ["exposure", "shake"], null), "shake");
  assert.equal(nextDecisionTarget(afterFirstPass, ["shake"], "exposure"), "exposure");
});


test("completion only opens for a saved fail or when every warning has passed", () => {
  const onePass = applySavedReview(canonicalTask(), "exposure", "pass");
  assert.deepEqual(completionGate(onePass), { enabled: false, mode: null });

  const withFail = applySavedReview(onePass, "shake", "fail");
  assert.deepEqual(completionGate(withFail), { enabled: true, mode: "early_fail" });

  const allPass = ["exposure", "shake", "late"].reduce(
    (task, issueId) => applySavedReview(task, issueId, "pass"),
    canonicalTask(),
  );
  assert.deepEqual(completionGate(allPass), { enabled: true, mode: "all_reviewed" });
});


test("manual reason choices are multi-selectable, cancellable, and require trimmed Other text", () => {
  const task = canonicalTask();
  assert.deepEqual(
    normalizeReasonDraft(task.reason_options, ["occlusion", "other"], "  遮住手部  "),
    {
      reasonCodes: ["occlusion", "other"],
      otherText: "遮住手部",
      hasManualReason: true,
      requiresText: true,
      valid: true,
    },
  );
  assert.deepEqual(
    normalizeReasonDraft(task.reason_options, ["other"], "   "),
    {
      reasonCodes: ["other"],
      otherText: "",
      hasManualReason: true,
      requiresText: true,
      valid: false,
    },
  );
  assert.deepEqual(
    normalizeReasonDraft(task.reason_options, [], "ignored"),
    {
      reasonCodes: [],
      otherText: null,
      hasManualReason: false,
      requiresText: false,
      valid: true,
    },
  );
});


test("review panel keeps reason chips as a cancellable local draft until a Fail payload is built", () => {
  const panel = new ReviewPanel();
  const task = canonicalTask();
  panel.setTask(task);
  panel.toggleReason("occlusion");
  panel.toggleReason("other");
  panel.setOtherText("  操作员补充  ");

  assert.deepEqual(panel.failurePayloadFor("exposure"), {
    failure_reason: {
      mode: "manual",
      reason_codes: ["occlusion", "other"],
      other_text: "操作员补充",
    },
  });

  panel.toggleReason("occlusion");
  panel.toggleReason("other");
  assert.deepEqual(panel.failurePayloadFor("exposure"), { reason: "曝光异常" });
});


test("review panel blocks an Other Fail until text is present and Pass ignores the unsaved reason draft", () => {
  const panel = new ReviewPanel();
  panel.setTask(canonicalTask());
  panel.toggleReason("other");
  assert.throws(() => panel.failurePayloadFor("exposure"), /其他原因/);
  panel.setOtherText("可见但不清楚");
  assert.deepEqual(panel.payloadForVerdict("exposure", "pass"), { verdict: "pass" });
  assert.equal(panel.reasonDraft().reasonCodes.length, 1);
});


test("WarnReviewApp uses only canonical task, verdict, and completion routes with revision and lease token", async () => {
  const calls = [];
  const updated = applySavedReview(canonicalTask({ report_revision: 8 }), "exposure", "pass");
  const app = new WarnReviewApp({
    fetcher: async (path, options = {}) => {
      calls.push([path, options.method ?? "GET", options.body ? JSON.parse(options.body) : null]);
      if (path === "/api/warn/assets") return { ok: true, status: 200, json: async () => ({ assets: ["asset-1", "asset-2"] }) };
      if (path.endsWith("/task")) return { ok: true, status: 200, json: async () => ({ task: canonicalTask() }) };
      if (path.includes("/verdict")) return { ok: true, status: 200, json: async () => ({ task: updated }) };
      if (path.endsWith("/complete")) return { ok: true, status: 200, json: async () => ({ task: canonicalTask({ report_revision: 9, manual_review_state: "completed" }) }) };
      throw new Error(`unexpected path ${path}`);
    },
  });

  await app.loadAsset("asset-1");
  app.setCurrentFrame(142);
  await app.submitVerdict("exposure", "pass");
  assert.deepEqual(calls[0], ["/api/warn/assets/asset-1/task", "GET", null]);
  assert.deepEqual(calls[1], [
    "/api/warn/assets/asset-1/issues/exposure/verdict",
    "POST",
    { expected_revision: 7, lease_token: "lease-1", verdict: "pass" },
  ]);
  assert.equal(app.defaultDecisionTarget(), "shake");
});


test("a pass always uses the earliest active pending warning, rather than a later simultaneous one", async () => {
  const submitted = [];
  const app = new WarnReviewApp({
    fetcher: async (path, options = {}) => {
      if (path.endsWith("/task")) return { ok: true, status: 200, json: async () => ({ task: canonicalTask() }) };
      if (path.includes("/verdict")) {
        submitted.push(path);
        return { ok: true, status: 200, json: async () => ({ task: applySavedReview(canonicalTask({ report_revision: 8 }), "exposure", "pass") }) };
      }
      throw new Error(`unexpected path ${path}`);
    },
  });
  await app.loadAsset("asset-1");
  app.setCurrentFrame(142);
  await app.submitDefaultVerdict("pass");
  assert.deepEqual(submitted, ["/api/warn/assets/asset-1/issues/exposure/verdict"]);
});


test("a pass at a non-overlapping interval advances playback to the next pending warning start", async () => {
  const app = new WarnReviewApp({
    fetcher: async (path) => {
      if (path.endsWith("/task")) return { ok: true, status: 200, json: async () => ({ task: canonicalTask() }) };
      if (path.includes("/verdict")) {
        return {
          ok: true,
          status: 200,
          json: async () => ({ task: applySavedReview(canonicalTask({ report_revision: 8 }), "exposure", "pass") }),
        };
      }
      throw new Error(`unexpected path ${path}`);
    },
  });
  await app.loadAsset("asset-1");
  app.setCurrentFrame(120);
  await app.submitDefaultVerdict("pass");
  assert.equal(app.currentFrame, 142);
  assert.equal(app.explicitIssueId, null);
  assert.equal(app.defaultDecisionTarget(), "shake");
});


test("timeline seeking never sets an explicit decision target, while status-row selection can reopen saved work", () => {
  const app = new WarnReviewApp();
  app.applyServerTask(applySavedReview(canonicalTask(), "exposure", "pass"));
  app.setCurrentFrame(142);
  app.seekFromTimeline(142);
  assert.equal(app.explicitIssueId, null);
  assert.equal(app.defaultDecisionTarget(), "shake");
  assert.equal(app.selectIssueFromStatus("exposure"), "exposure");
  assert.equal(app.defaultDecisionTarget(), "exposure");
});


test("a successful manual completion loads the next asset but direct navigation discards unsaved reason choices", async () => {
  const completed = ["exposure", "shake", "late"].reduce(
    (task, issueId) => applySavedReview(task, issueId, "pass"),
    canonicalTask({ report_revision: 11 }),
  );
  const calls = [];
  const app = new WarnReviewApp({
    fetcher: async (path, options = {}) => {
      calls.push(path);
      if (path === "/api/warn/assets") return { ok: true, status: 200, json: async () => ({ assets: ["asset-1", "asset-2"] }) };
      if (path === "/api/warn/assets/asset-2/task") return { ok: true, status: 200, json: async () => ({ task: canonicalTask({ asset_id: "asset-2" }) }) };
      if (path.endsWith("/complete")) return { ok: true, status: 200, json: async () => ({ task: completed }) };
      throw new Error(`unexpected path ${path}`);
    },
  });
  app.applyServerTask(completed);
  app.panel.toggleReason("occlusion");
  await app.completeReview();
  assert.deepEqual(calls, ["/api/warn/assets/asset-1/complete", "/api/warn/assets", "/api/warn/assets/asset-2/task"]);
  assert.equal(app.assetId, "asset-2");
  assert.deepEqual(app.panel.reasonDraft().reasonCodes, []);
});


test("409 retains the safe server snapshot and reports the server-safe error", async () => {
  const current = canonicalTask();
  const app = new WarnReviewApp({
    fetcher: async () => ({
      ok: false,
      status: 409,
      json: async () => ({ error: { code: "stale_revision", message: "请刷新任务" } }),
    }),
  });
  app.applyServerTask(current);
  await assert.rejects(() => app.submitVerdict("exposure", "pass"), /请刷新任务/);
  assert.equal(app.task, current);
  assert.deepEqual(app.lastError, { code: "stale_revision", message: "请刷新任务", status: 409 });
});
