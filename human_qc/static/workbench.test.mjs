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


function fakePanelRoot() {
  const listeners = new Map();
  return {
    innerHTML: "",
    addEventListener(type, listener) {
      const values = listeners.get(type) ?? [];
      values.push(listener);
      listeners.set(type, values);
    },
    removeEventListener(type, listener) {
      const values = listeners.get(type) ?? [];
      listeners.set(type, values.filter((candidate) => candidate !== listener));
    },
    listenerCount(type) {
      return (listeners.get(type) ?? []).length;
    },
  };
}


function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}


test("the earliest pending active warning wins even when active ids arrive in another order", () => {
  const task = canonicalTask();
  assert.equal(nextDecisionTarget(task, ["shake", "exposure"], null), "exposure");

  const afterFirstPass = applySavedReview(task, "exposure", "pass");
  assert.equal(nextDecisionTarget(afterFirstPass, ["exposure", "shake"], null), "shake");
  assert.equal(nextDecisionTarget(afterFirstPass, ["shake"], "exposure"), "shake");
});


test("only a non-ready continuous overlay locks its own issue without changing first-pass order", () => {
  const task = canonicalTask({
    issues: canonicalTask().issues.map((issue) => (
      issue.id === "exposure"
        ? { ...issue, overlay: { status: "generating", segments: [] } }
        : issue
    )),
  });
  const panel = new ReviewPanel();
  panel.setTask(task);
  panel.setCurrentFrame(142);
  assert.equal(panel.decisionTargetId(), "exposure");
  assert.equal(panel.canSubmit(task.issues[0]), false);
  assert.equal(panel.canSubmit(task.issues[1]), true);
  panel.setExplicitIssueId("shake");
  assert.equal(panel.decisionTargetId(), "shake");
});


test("overlay polling replaces only the selected issue status without reloading task or clearing reason draft", async () => {
  const initial = canonicalTask({
    issues: canonicalTask().issues.map((issue) => (
      issue.id === "exposure"
        ? { ...issue, overlay: { status: "generating", segments: [] } }
        : issue
    )),
  });
  const calls = [];
  const timers = [];
  const app = new WarnReviewApp({
    scheduler: { setTimeout: (fn) => { timers.push(fn); return fn; }, clearTimeout() {} },
    random: () => 0.5,
    fetcher: async (path) => {
      calls.push(path);
      if (path.endsWith("/task")) return { ok: true, status: 200, json: async () => ({ task: initial }) };
      if (path.endsWith("/status")) return {
        ok: true,
        status: 200,
        json: async () => ({
          asset_id: "asset-1",
          issue_id: "exposure",
          overlay: {
            status: "ready",
            segments: [{ start_frame: 120, end_frame_exclusive: 169, status: "ready", url: "/media/overlay" }],
          },
        }),
      };
      throw new Error(`unexpected path ${path}`);
    },
  });
  await app.loadAsset("asset-1");
  app.setCurrentFrame(120);
  app.panel.toggleReason("occlusion");
  assert.equal(timers.length, 1);
  await timers.shift()();
  assert.equal(app.task.report_revision, 7);
  assert.equal(app.task.issues.find((issue) => issue.id === "exposure").overlay.status, "ready");
  assert.deepEqual(app.panel.reasonDraft().reasonCodes, ["occlusion"]);
  assert.deepEqual(calls.filter((path) => path.endsWith("/task")), ["/api/warn/assets/asset-1/task"]);
  app.destroy();
});


test("an unchanged generating overlay uses bounded exponential backoff instead of resetting each poll", async () => {
  const initial = canonicalTask({
    issues: canonicalTask().issues.map((issue) => (
      issue.id === "exposure"
        ? { ...issue, overlay: { status: "generating", segments: [] } }
        : issue
    )),
  });
  const timers = [];
  const app = new WarnReviewApp({
    scheduler: {
      setTimeout(fn, delay) {
        const timer = { fn, delay };
        timers.push(timer);
        return timer;
      },
      clearTimeout(timer) {
        const index = timers.indexOf(timer);
        if (index >= 0) timers.splice(index, 1);
      },
    },
    random: () => 0.5,
    fetcher: async (path) => {
      if (path.endsWith("/task")) return { ok: true, status: 200, json: async () => ({ task: initial }) };
      if (path.endsWith("/status")) return {
        ok: true,
        status: 200,
        json: async () => ({
          asset_id: "asset-1",
          issue_id: "exposure",
          overlay: { status: "generating", segments: [] },
        }),
      };
      throw new Error(`unexpected path ${path}`);
    },
  });

  await app.loadAsset("asset-1");
  assert.deepEqual(timers.map((timer) => timer.delay), [1000]);
  await timers.shift().fn();
  assert.deepEqual(timers.map((timer) => timer.delay), [2000]);
  await timers.shift().fn();
  assert.deepEqual(timers.map((timer) => timer.delay), [4000]);
  await timers.shift().fn();
  assert.deepEqual(timers.map((timer) => timer.delay), [5000]);
  app.destroy();
});


test("a retry 503 honours Retry-After without replacing the local failed view or issuing a second POST", async () => {
  const initial = canonicalTask({
    issues: canonicalTask().issues.map((issue) => (
      issue.id === "exposure"
        ? { ...issue, overlay: { status: "failed", retryable: true, segments: [] } }
        : issue
    )),
  });
  const timers = [];
  const calls = [];
  let now = 0;
  const app = new WarnReviewApp({
    now: () => now,
    scheduler: {
      setTimeout(fn, delay) {
        const timer = { fn, delay };
        timers.push(timer);
        return timer;
      },
      clearTimeout(timer) {
        const index = timers.indexOf(timer);
        if (index >= 0) timers.splice(index, 1);
      },
    },
    random: () => 0.5,
    fetcher: async (path, options = {}) => {
      calls.push({ path, method: options.method });
      if (path.endsWith("/task")) return { ok: true, status: 200, json: async () => ({ task: initial }) };
      if (path.endsWith("/retry")) return {
        ok: false,
        status: 503,
        headers: { get: (name) => (name === "Retry-After" ? "2" : null) },
        json: async () => ({ error: { code: "internal", message: "private worker error" } }),
      };
      if (path.endsWith("/status")) return {
        ok: true,
        status: 200,
        json: async () => ({
          asset_id: "asset-1",
          issue_id: "exposure",
          overlay: { status: "ready", segments: [] },
        }),
      };
      throw new Error(`unexpected path ${path}`);
    },
  });

  await app.loadAsset("asset-1");
  await app.retryOverlay("exposure");
  assert.equal(app.task.issues[0].overlay.status, "failed");
  assert.equal(app._overlayRetrying.has("exposure"), true);
  assert.deepEqual(timers.map((timer) => timer.delay), [2000]);
  await app.retryOverlay("exposure");
  assert.equal(calls.filter(({ path }) => path.endsWith("/retry")).length, 1);

  now = 2000;
  await timers.shift().fn();
  assert.equal(app.task.issues[0].overlay.status, "ready");
  assert.equal(app._overlayRetrying.has("exposure"), false);
  assert.equal(calls.filter(({ path }) => path.endsWith("/retry")).length, 1);
  app.destroy();
});


test("status transport failures and 5xx show a safe retry notice without clearing the local reason draft or reloading", async () => {
  const initial = canonicalTask({
    issues: canonicalTask().issues.map((issue) => (
      issue.id === "exposure"
        ? { ...issue, overlay: { status: "generating", segments: [] } }
        : issue
    )),
  });
  const timers = [];
  const calls = [];
  let statusAttempts = 0;
  const app = new WarnReviewApp({
    scheduler: {
      setTimeout(fn, delay) {
        const timer = { fn, delay };
        timers.push(timer);
        return timer;
      },
      clearTimeout(timer) {
        const index = timers.indexOf(timer);
        if (index >= 0) timers.splice(index, 1);
      },
    },
    random: () => 0.5,
    fetcher: async (path) => {
      calls.push(path);
      if (path.endsWith("/task")) return { ok: true, status: 200, json: async () => ({ task: initial }) };
      if (path.endsWith("/status")) {
        statusAttempts += 1;
        if (statusAttempts === 1) throw new Error("private upstream failure");
        return {
          ok: false,
          status: 503,
          json: async () => ({ error: { code: "internal", message: "private worker failure" } }),
        };
      }
      throw new Error(`unexpected path ${path}`);
    },
  });

  await app.loadAsset("asset-1");
  app.panel.toggleReason("occlusion");
  await timers.shift().fn();
  assert.deepEqual(app.lastError, {
    code: "overlay_status_unavailable",
    message: "状态暂时无法更新，正在重试",
    status: 0,
  });
  assert.deepEqual(app.panel.reasonDraft().reasonCodes, ["occlusion"]);
  assert.deepEqual(calls.filter((path) => path.endsWith("/task")), ["/api/warn/assets/asset-1/task"]);
  assert.deepEqual(timers.map((timer) => timer.delay), [2000]);
  await timers.shift().fn();
  assert.deepEqual(app.lastError, {
    code: "overlay_status_unavailable",
    message: "状态暂时无法更新，正在重试",
    status: 503,
  });
  assert.deepEqual(app.panel.reasonDraft().reasonCodes, ["occlusion"]);
  assert.deepEqual(timers.map((timer) => timer.delay), [4000]);
  app.destroy();
});


test("a retry queues behind a suspended status poll instead of starting a concurrent same-generation fetch", async () => {
  const initial = canonicalTask({
    issues: canonicalTask().issues.map((issue) => {
      if (issue.id === "exposure") return { ...issue, overlay: { status: "generating", segments: [] } };
      if (issue.id === "shake") return { ...issue, overlay: { status: "failed", retryable: true, segments: [] } };
      return issue;
    }),
  });
  const firstStatus = deferred();
  const timers = [];
  let statusCalls = 0;
  const app = new WarnReviewApp({
    scheduler: {
      setTimeout(fn, delay) {
        const timer = { fn, delay };
        timers.push(timer);
        return timer;
      },
      clearTimeout(timer) {
        const index = timers.indexOf(timer);
        if (index >= 0) timers.splice(index, 1);
      },
    },
    random: () => 0.5,
    fetcher: async (path) => {
      if (path.endsWith("/task")) return { ok: true, status: 200, json: async () => ({ task: initial }) };
      if (path.endsWith("/retry")) return {
        ok: true,
        status: 202,
        json: async () => ({
          asset_id: "asset-1",
          issue_id: "shake",
          overlay: { status: "ready", segments: [] },
        }),
      };
      if (path.endsWith("/status")) {
        statusCalls += 1;
        if (statusCalls === 1) return firstStatus.promise;
        return {
          ok: true,
          status: 200,
          json: async () => ({
            asset_id: "asset-1",
            issue_id: "exposure",
            overlay: { status: "generating", segments: [] },
          }),
        };
      }
      throw new Error(`unexpected path ${path}`);
    },
  });

  await app.loadAsset("asset-1");
  const pollA = timers.shift().fn();
  await Promise.resolve();
  assert.equal(statusCalls, 1);

  await app.retryOverlay("shake");
  assert.equal(statusCalls, 1);
  assert.equal(timers.length, 0);

  firstStatus.resolve({
    ok: true,
    status: 200,
    json: async () => ({
      asset_id: "asset-1",
      issue_id: "exposure",
      overlay: { status: "generating", segments: [] },
    }),
  });
  await pollA;
  await Promise.resolve();
  assert.equal(statusCalls, 1);
  assert.deepEqual(timers.map((timer) => timer.delay), [1000]);

  await timers.shift().fn();
  assert.equal(statusCalls, 2);
  app.destroy();
});


test("asset changes and destroy clear queued state from a suspended overlay poll", async () => {
  const initial = canonicalTask({
    issues: canonicalTask().issues.map((issue) => (
      issue.id === "exposure" ? { ...issue, overlay: { status: "generating", segments: [] } } : issue
    )),
  });
  const successor = canonicalTask({
    asset_id: "asset-2",
    video: { ...canonicalTask().video, url: "/media/assets/asset-2/source" },
  });
  const timers = [];
  const pendingStatus = deferred();
  const destroyedStatus = deferred();
  let statusRequests = 0;
  const app = new WarnReviewApp({
    scheduler: {
      setTimeout(fn, delay) {
        const timer = { fn, delay };
        timers.push(timer);
        return timer;
      },
      clearTimeout(timer) {
        const index = timers.indexOf(timer);
        if (index >= 0) timers.splice(index, 1);
      },
    },
    random: () => 0.5,
    fetcher: async (path) => {
      if (path.endsWith("/assets/asset-1/task")) return { ok: true, status: 200, json: async () => ({ task: initial }) };
      if (path.endsWith("/assets/asset-2/task")) return { ok: true, status: 200, json: async () => ({ task: successor }) };
      if (path.endsWith("/status")) {
        statusRequests += 1;
        return statusRequests === 1 ? pendingStatus.promise : destroyedStatus.promise;
      }
      throw new Error(`unexpected path ${path}`);
    },
  });

  await app.loadAsset("asset-1");
  const pendingPoll = timers.shift().fn();
  await Promise.resolve();
  await app.loadAsset("asset-2");
  assert.equal(app._overlayPollInFlight, null);
  assert.equal(timers.length, 0);

  pendingStatus.resolve({
    ok: true,
    status: 200,
    json: async () => ({
      asset_id: "asset-1",
      issue_id: "exposure",
      overlay: { status: "generating", segments: [] },
    }),
  });
  await pendingPoll;
  await Promise.resolve();
  assert.equal(app.assetId, "asset-2");
  assert.equal(app._overlayPollInFlight, null);
  assert.equal(timers.length, 0);

  await app.loadAsset("asset-1");
  const destroyedPoll = timers.shift().fn();
  await Promise.resolve();
  app.destroy();
  assert.equal(app._overlayPollInFlight, null);
  assert.equal(app._overlayPollTimer, null);
  assert.equal(timers.length, 0);

  destroyedStatus.resolve({
    ok: true,
    status: 200,
    json: async () => ({
      asset_id: "asset-1",
      issue_id: "exposure",
      overlay: { status: "generating", segments: [] },
    }),
  });
  await destroyedPoll;
  await Promise.resolve();
  assert.equal(app._overlayPollInFlight, null);
  assert.equal(timers.length, 0);
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


test("an explicit status-row target is ignored outside its active frame and becomes active after its start is seeked", () => {
  const reviewed = applySavedReview(canonicalTask(), "exposure", "pass");
  assert.equal(nextDecisionTarget(reviewed, ["late"], "exposure"), "late");

  const app = new WarnReviewApp();
  app.applyServerTask(reviewed);
  app.setCurrentFrame(390);
  let explicitIssueIdDuringSeek = "not-called";
  app.videoController = {
    seekToFrame() { explicitIssueIdDuringSeek = app.explicitIssueId; },
  };
  assert.equal(app.selectIssueFromStatus("exposure"), "exposure");
  assert.equal(explicitIssueIdDuringSeek, null);
  assert.equal(app.currentFrame, 120);
  assert.equal(app.defaultDecisionTarget(), "exposure");
});


test("an in-flight verdict is single-flight and blocks direct asset navigation", async () => {
  const calls = [];
  let resolveVerdict;
  const pendingVerdict = new Promise((resolve) => { resolveVerdict = resolve; });
  const saved = applySavedReview(canonicalTask({ report_revision: 8 }), "exposure", "pass");
  const app = new WarnReviewApp({
    fetcher: async (path) => {
      calls.push(path);
      if (path === "/api/warn/assets/asset-1/task") {
        return { ok: true, status: 200, json: async () => ({ task: canonicalTask() }) };
      }
      if (path.includes("/verdict")) return pendingVerdict;
      if (path === "/api/warn/assets") {
        return { ok: true, status: 200, json: async () => ({ assets: ["asset-1", "asset-2"] }) };
      }
      if (path === "/api/warn/assets/asset-2/task") {
        return { ok: true, status: 200, json: async () => ({ task: canonicalTask({ asset_id: "asset-2" }) }) };
      }
      throw new Error(`unexpected path ${path}`);
    },
  });

  await app.loadAsset("asset-1");
  app.setCurrentFrame(120);
  const first = app.submitDefaultVerdict("pass");
  const second = app.submitDefaultVerdict("pass");

  assert.equal(app.isBusy, true);
  assert.equal(calls.filter((path) => path.includes("/verdict")).length, 1);
  assert.equal(await app.loadNextAsset(), null);
  assert.equal(calls.filter((path) => path === "/api/warn/assets").length, 0);

  resolveVerdict({ ok: true, status: 200, json: async () => ({ task: saved }) });
  await Promise.all([first, second]);
  assert.equal(app.isBusy, false);
  assert.equal(app.assetId, "asset-1");
});


test("a Pass preserves a selected reason draft for the next active Fail", async () => {
  const submitted = [];
  const afterPass = applySavedReview(canonicalTask({ report_revision: 8 }), "exposure", "pass");
  const afterFail = applySavedReview(
    canonicalTask({
      report_revision: 9,
      failure_reason: { mode: "manual", reason_codes: ["occlusion"], other_text: null },
    }),
    "shake",
    "fail",
  );
  const app = new WarnReviewApp({
    fetcher: async (path, options = {}) => {
      if (path.endsWith("/task")) return { ok: true, status: 200, json: async () => ({ task: canonicalTask() }) };
      if (path.includes("/verdict")) {
        submitted.push(JSON.parse(options.body));
        const task = path.includes("/exposure/") ? afterPass : afterFail;
        return { ok: true, status: 200, json: async () => ({ task }) };
      }
      throw new Error(`unexpected path ${path}`);
    },
  });

  await app.loadAsset("asset-1");
  app.setCurrentFrame(120);
  app.panel.toggleReason("occlusion");
  await app.submitDefaultVerdict("pass");
  assert.deepEqual(app.panel.reasonDraft().reasonCodes, ["occlusion"]);

  await app.submitDefaultVerdict("fail");
  assert.deepEqual(submitted[1].failure_reason, {
    mode: "manual",
    reason_codes: ["occlusion"],
    other_text: null,
  });
  assert.deepEqual(app.panel.reasonDraft().reasonCodes, ["occlusion"]);
});


test("early-fail completion restores and submits the current normalized manual reason draft", async () => {
  const failed = applySavedReview(
    canonicalTask({
      report_revision: 8,
      failure_reason: { mode: "manual", reason_codes: ["occlusion"], other_text: null },
    }),
    "exposure",
    "fail",
  );
  let completionBody = null;
  const app = new WarnReviewApp({
    fetcher: async (path, options = {}) => {
      if (path.endsWith("/task")) return { ok: true, status: 200, json: async () => ({ task: failed }) };
      if (path.endsWith("/complete")) {
        completionBody = JSON.parse(options.body);
        return { ok: true, status: 200, json: async () => ({ task: failed }) };
      }
      if (path === "/api/warn/assets") return { ok: true, status: 200, json: async () => ({ assets: ["asset-1"] }) };
      throw new Error(`unexpected path ${path}`);
    },
  });

  await app.loadAsset("asset-1");
  assert.deepEqual(app.panel.reasonDraft().reasonCodes, ["occlusion"]);
  app.panel.toggleReason("occlusion");
  app.panel.toggleReason("other");
  app.panel.setOtherText("  补充文字  ");

  await app.completeReview();
  assert.deepEqual(completionBody.failure_reason, {
    mode: "manual",
    reason_codes: ["other"],
    other_text: "补充文字",
  });
});


test("early-fail completion rejects blank Other text without issuing a request", async () => {
  const failed = applySavedReview(canonicalTask({ report_revision: 8 }), "exposure", "fail");
  let requests = 0;
  const app = new WarnReviewApp({
    fetcher: async () => {
      requests += 1;
      throw new Error("completion must not request with blank Other text");
    },
  });
  app.applyServerTask(failed);
  app.setCurrentFrame(120);
  app.panel.toggleReason("other");

  await assert.rejects(() => app.completeReview(), /其他原因/);
  assert.equal(requests, 0);
  assert.deepEqual(app.panel.reasonDraft().reasonCodes, ["other"]);
});


test("the active warning card exposes saved verdicts and accessible reason and threshold controls", () => {
  const root = fakePanelRoot();
  const panel = new ReviewPanel();
  const task = applySavedReview(
    applySavedReview(canonicalTask(), "exposure", "pass"),
    "shake",
    "fail",
  );
  panel.mount(root);
  panel.setTask(task);
  panel.setCurrentFrame(142);

  assert.match(root.innerHTML, /data-review-status="pass"[\s\S]*✓ 已通过/);
  assert.match(root.innerHTML, /data-review-status="fail"[\s\S]*已失败/);
  assert.match(root.innerHTML, /aria-pressed="false"/);
  assert.match(root.innerHTML, /aria-describedby="threshold-/);
  assert.match(root.innerHTML, /role="tooltip"/);
  assert.doesNotMatch(root.innerHTML, /data-reason-code="occlusion"[^>]*disabled/);
});


test("review panel destroy unbinds its delegated DOM listeners", () => {
  const root = fakePanelRoot();
  const panel = new ReviewPanel();
  panel.mount(root);
  assert.equal(root.listenerCount("click"), 1);
  assert.equal(root.listenerCount("input"), 1);

  panel.destroy();
  assert.equal(root.listenerCount("click"), 0);
  assert.equal(root.listenerCount("input"), 0);
  assert.equal(panel.root, null);
});


test("app destroy releases chrome, panel, timeline, and video controller listeners", () => {
  const chrome = fakePanelRoot();
  let panelDestroyed = 0;
  let timelineDestroyed = 0;
  let videoDestroyed = 0;
  const app = new WarnReviewApp({
    panelFactory: () => ({
      destroy() { panelDestroyed += 1; },
      setBusy() {},
    }),
  });
  const listener = () => {};
  chrome.addEventListener("click", listener);
  app._chromeListeners = [[chrome, "click", listener]];
  app._chromeBound = true;
  app.root = chrome;
  app.timeline = { destroy() { timelineDestroyed += 1; } };
  app.videoController = { destroy() { videoDestroyed += 1; } };

  app.destroy();
  assert.equal(chrome.listenerCount("click"), 0);
  assert.equal(panelDestroyed, 1);
  assert.equal(timelineDestroyed, 1);
  assert.equal(videoDestroyed, 1);
  assert.equal(app.root, null);
  assert.equal(app._chromeBound, false);
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
