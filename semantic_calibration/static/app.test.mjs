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
  ]);
});


test("no query loads the first eligible asset and an empty list is terminal", async () => {
  const first = new SemanticCalibrationApp({
    locationRef: { search: "" },
    fetcher: async (path) => path.endsWith("/task")
      ? response({ task: { asset_id: "asset-1", revision: 1, semantic: {} } })
      : response({ assets: [{ asset_id: "asset-1" }] }),
  });
  await first.start();
  assert.equal(first.assetId, "asset-1");

  const empty = new SemanticCalibrationApp({
    locationRef: { search: "" },
    fetcher: async () => response({ assets: [] }),
  });
  await empty.start();
  assert.equal(empty.state, "empty");
  assert.equal(empty.task, null);
});


test("blocked deep links retain a stable server error", async () => {
  const app = new SemanticCalibrationApp({
    locationRef: { search: "?asset_id=blocked" },
    fetcher: async (path) => path === "/api/semantic/assets"
      ? response({ assets: [{ asset_id: "asset-1" }] })
      : response({ error: { code: "semantic_not_ready", message: "semantic task is not ready" } }, false, 409),
  });
  await assert.rejects(() => app.start(), /not ready/);
  assert.equal(app.error.code, "semantic_not_ready");
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
  ]);
});
