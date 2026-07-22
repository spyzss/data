import test from "node:test";
import assert from "node:assert/strict";

import { SemanticCalibrationApp } from "./app.js";


function response(value, ok = true, status = 200) {
  return { ok, status, json: async () => value };
}


test("deep link loads the requested eligible semantic asset", async () => {
  const calls = [];
  const app = new SemanticCalibrationApp({
    locationRef: { search: "?asset_id=asset-2" },
    reviewer: "alice",
    fetcher: async (path) => {
      calls.push(path);
      if (path === "/api/semantic/assets") {
        return response({ assets: [{ asset_id: "asset-1" }, { asset_id: "asset-2" }] });
      }
      return response({ task: { asset_id: "asset-2", revision: 4, semantic: { pending_edit: null } } });
    },
  });
  await app.start();
  assert.equal(app.assetId, "asset-2");
  assert.deepEqual(calls, [
    "/api/semantic/assets",
    "/api/semantic/assets/asset-2/task",
    "/api/semantic/assets/asset-2/lease/acquire",
  ]);
});


test("no query loads the first eligible asset and an empty list is terminal", async () => {
  const first = new SemanticCalibrationApp({
    locationRef: { search: "" },
    reviewer: "alice",
    fetcher: async (path) => path.endsWith("/task")
      ? response({ task: { asset_id: "asset-1", revision: 1, semantic: {} } })
      : response({ assets: [{ asset_id: "asset-1" }] }),
  });
  await first.start();
  assert.equal(first.assetId, "asset-1");

  const empty = new SemanticCalibrationApp({
    locationRef: { search: "" },
    reviewer: "alice",
    fetcher: async () => response({ assets: [] }),
  });
  await empty.start();
  assert.equal(empty.state, "empty");
  assert.equal(empty.task, null);
});


test("blocked deep links retain a stable server error", async () => {
  const app = new SemanticCalibrationApp({
    locationRef: { search: "?asset_id=blocked" },
    reviewer: "alice",
    fetcher: async (path) => path === "/api/semantic/assets"
      ? response({ assets: [{ asset_id: "asset-1" }] })
      : response({ error: { code: "semantic_not_ready", message: "semantic task is not ready" } }, false, 409),
  });
  await assert.rejects(() => app.start(), /not ready/);
  assert.equal(app.error.code, "semantic_not_ready");
});


test("completed and reportless deep links render read-only without acquiring a lease", async () => {
  for (const state of ["completed", "preview"]) {
    const calls = [];
    let rendered = null;
    const stage = {};
    const root = { querySelector(selector) { return selector === "[data-semantic-stage]" ? stage : null; } };
    const app = new SemanticCalibrationApp({
      root,
      reviewer: "alice",
      locationRef: { search: `?asset_id=${state}` },
      fetcher: async (path) => {
        calls.push(path);
        if (path === "/api/semantic/assets") return response({ assets: [] });
        if (path.endsWith("/task")) {
          return response({ task: { asset_id: state, revision: 4, editable: false, semantic: { report_state: state } } });
        }
        throw new Error(`unexpected ${path}`);
      },
      adapterFactory: () => ({ render(task) { rendered = task; } }),
    });
    await app.start();
    assert.deepEqual(calls, [
      "/api/semantic/assets",
      `/api/semantic/assets/${state}/task`,
    ]);
    assert.equal(app.lease, null);
    assert.equal(app.state, "read_only");
    assert.equal(rendered.editable, false);
  }
});


test("task load auto-acquires, periodically renews, and complete finally releases", async () => {
  const calls = [];
  let renewCallback = null;
  let adapterOptions = null;
  const stage = {};
  const root = {
    querySelector(selector) {
      return selector === "[data-semantic-stage]" ? stage : null;
    },
  };
  const app = new SemanticCalibrationApp({
    root,
    reviewer: "alice",
    leaseRenewIntervalMs: 100,
    setIntervalFn(callback) { renewCallback = callback; return 17; },
    clearIntervalFn() {},
    fetcher: async (path, options = {}) => {
      calls.push([path, options.method ?? "GET", options.body ? JSON.parse(options.body) : null]);
      if (path === "/api/semantic/assets") return response({ assets: [{ asset_id: "asset-1" }] });
      if (path.endsWith("/task")) {
        return response({ task: { asset_id: "asset-1", revision: 4, video_url: "/api/semantic/assets/asset-1/video", semantic: {} } });
      }
      if (path.endsWith("/lease/acquire") || path.endsWith("/lease/renew")) {
        return response({ lease: { token: "lease-token", reviewer: "alice", expires_at: "later" } });
      }
      if (path.endsWith("/complete")) {
        return response({ task: { asset_id: "asset-1", revision: 5, semantic: { report_state: "completed" } } });
      }
      if (path.endsWith("/lease/release")) return response({ released: true });
      throw new Error(`unexpected ${path}`);
    },
    adapterFactory(options) {
      adapterOptions = options;
      return { render() {} };
    },
  });

  await app.start();
  assert.equal(app.lease.reviewer, "alice");
  assert.equal(typeof renewCallback, "function");
  await renewCallback();
  await adapterOptions.onComplete();

  assert.deepEqual(calls.map(([path]) => path), [
    "/api/semantic/assets",
    "/api/semantic/assets/asset-1/task",
    "/api/semantic/assets/asset-1/lease/acquire",
    "/api/semantic/assets/asset-1/lease/renew",
    "/api/semantic/assets/asset-1/complete",
    "/api/semantic/assets/asset-1/lease/release",
  ]);
  assert.equal(app.lease, null);
});


test("renew conflicts clear the stale lease and allow reacquiring without dropping the task", async () => {
  const calls = [];
  const rendered = [];
  const stage = {};
  const root = { querySelector(selector) { return selector === "[data-semantic-stage]" ? stage : null; } };
  const app = new SemanticCalibrationApp({
    root,
    reviewer: "alice",
    fetcher: async (path) => {
      calls.push(path);
      if (path.endsWith("/lease/renew")) {
        return response({ error: { code: "lease_invalid", message: "stale" } }, false, 423);
      }
      if (path.endsWith("/lease/acquire")) {
        return response({ lease: { token: "fresh-token", reviewer: "alice" } });
      }
      throw new Error(`unexpected ${path}`);
    },
    adapterFactory: () => ({ render(task) { rendered.push(task); } }),
  });
  app.assetId = "asset-1";
  app.task = { asset_id: "asset-1", revision: 4, editable: true, semantic: { pending_edit: null } };
  app.lease = { token: "stale-token", reviewer: "alice" };
  app.render();

  await assert.rejects(() => app.renewLease(), /stale/);
  assert.equal(app.task.asset_id, "asset-1");
  assert.equal(app.lease, null);
  assert.equal(app.state, "read_only");
  assert.equal(rendered.at(-1).editable, false);

  await app.acquireLease();
  assert.equal(app.lease.token, "fresh-token");
  assert.equal(app.state, "ready");
  assert.equal(rendered.at(-1).editable, true);
  assert.deepEqual(calls, [
    "/api/semantic/assets/asset-1/lease/renew",
    "/api/semantic/assets/asset-1/lease/acquire",
  ]);
});


test("navigation releases the old lease and unload attempts a keepalive release", async () => {
  const calls = [];
  const eventHandlers = {};
  const app = new SemanticCalibrationApp({
    reviewer: "alice",
    eventTarget: { addEventListener(name, handler) { eventHandlers[name] = handler; } },
    fetcher: async (path, options = {}) => {
      calls.push([path, options]);
      if (path.includes("/lease/release")) return response({ released: true });
      if (path.endsWith("/task")) return response({ task: { asset_id: "asset-2", revision: 1, semantic: {} } });
      if (path.endsWith("/lease/acquire")) return response({ lease: { token: "new-token", reviewer: "alice" } });
      throw new Error(`unexpected ${path}`);
    },
  });
  app.assetId = "asset-1";
  app.task = { asset_id: "asset-1", revision: 9, semantic: {} };
  app.lease = { token: "old-token", reviewer: "alice" };
  app.bindLifecycle();

  await app.loadAsset("asset-2");
  assert.match(calls[0][0], /asset-1\/lease\/release$/);
  assert.match(calls[1][0], /asset-2\/task$/);
  assert.match(calls[2][0], /asset-2\/lease\/acquire$/);

  await eventHandlers.pagehide();
  const unload = calls.at(-1);
  assert.match(unload[0], /asset-2\/lease\/release$/);
  assert.equal(unload[1].keepalive, true);
});


test("video task URL and current frame stay synchronized with playback", () => {
  let timeupdate = null;
  const video = {
    src: "",
    currentTime: 0,
    getAttribute() { return this.src; },
    addEventListener(name, handler) { if (name === "timeupdate") timeupdate = handler; },
  };
  const frame = { textContent: "" };
  const stage = {};
  const root = {
    querySelector(selector) {
      if (selector === "[data-semantic-stage]") return stage;
      if (selector === "[data-semantic-video]") return video;
      if (selector === "[data-current-frame]") return frame;
      return null;
    },
  };
  const app = new SemanticCalibrationApp({
    root,
    adapterFactory: () => ({ render() {} }),
  });
  app.assetId = "asset-1";
  app.task = {
    asset_id: "asset-1",
    revision: 1,
    video_url: "/api/semantic/assets/asset-1/video",
    semantic: { timeline: { fps: 30, frame_count: 100, segments: [] } },
  };
  app.render();
  assert.equal(video.src, "/api/semantic/assets/asset-1/video");
  video.currentTime = 1.5;
  timeupdate();
  assert.equal(frame.textContent, "45");
});


test("asset selector renders server asset IDs as text, never HTML", () => {
  const options = [];
  const selector = {
    value: "",
    replaceChildren(...children) { options.push(...children); },
    set innerHTML(_) { throw new Error("asset IDs must not be assigned as HTML"); },
  };
  const documentRef = {
    createElement() { return { value: "", textContent: "" }; },
  };
  const root = { querySelector(selectorName) { return selectorName === "[data-asset-select]" ? selector : null; } };
  const app = new SemanticCalibrationApp({ documentRef, root });
  app.assets = [{ asset_id: '<img src=x onerror="boom">' }];
  app.render();
  assert.equal(options[0].textContent, '<img src=x onerror="boom">');
});


test("adapter actions are wired to every semantic mutation route", async () => {
  const calls = [];
  let adapterOptions = null;
  const stage = {};
  const root = {
    querySelector(selector) {
      return selector === "[data-semantic-stage]" ? stage : null;
    },
  };
  const app = new SemanticCalibrationApp({
    root,
    fetcher: async (path, options) => {
      calls.push([path, JSON.parse(options.body)]);
      return response({ task: { asset_id: "asset-1", revision: calls.length + 4, semantic: {} } });
    },
    adapterFactory(options) {
      adapterOptions = options;
      return { render() {} };
    },
  });
  app.assetId = "asset-1";
  app.task = { asset_id: "asset-1", revision: 4, semantic: {} };
  app.lease = { token: "lease-token" };
  app.render();

  await adapterOptions.onTextPending({ segment_id: "s1", text_cn: "甲", text_en: "a" });
  await adapterOptions.onConfirm();
  await adapterOptions.onCancel();
  await adapterOptions.onComplete();

  assert.deepEqual(calls.map(([path]) => path), [
    "/api/semantic/assets/asset-1/text/pending",
    "/api/semantic/assets/asset-1/pending/confirm",
    "/api/semantic/assets/asset-1/pending/cancel",
    "/api/semantic/assets/asset-1/complete",
    "/api/semantic/assets/asset-1/lease/release",
  ]);
});
