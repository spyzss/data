import { EventEmitter } from "node:events";
import { spawn } from "node:child_process";
import { chmod, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";

import {
  Cdp,
  NetworkHealth,
  cdpTimeoutFromEnvironment,
  installShutdownSignalHandlers,
  resolveChromeBin,
  shutdownResources,
} from "./cdp_driver.mjs";


const sleep = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

const until = async (predicate, timeoutMs = 2_000) => {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await predicate()) return;
    await sleep(10);
  }
  throw new Error("timed out waiting for test condition");
};


class FakeChild extends EventEmitter {
  constructor({ exitOn = "SIGTERM" } = {}) {
    super();
    this.exitOn = exitOn;
    this.exitCode = null;
    this.signalCode = null;
    this.signals = [];
  }

  kill(signal) {
    this.signals.push(signal);
    if (signal === this.exitOn) {
      queueMicrotask(() => {
        this.signalCode = signal;
        this.emit("exit", null, signal);
        this.emit("close", null, signal);
      });
    }
    return true;
  }
}


test("CDP timeout uses the configured five-to-ten-second production range and clears a timed-out request", async () => {
  assert.equal(cdpTimeoutFromEnvironment({}), 8000);
  assert.equal(cdpTimeoutFromEnvironment({ HUMAN_QC_CDP_TIMEOUT_MS: "5000" }), 5000);
  assert.equal(cdpTimeoutFromEnvironment({ HUMAN_QC_CDP_TIMEOUT_MS: "10000" }), 10000);
  assert.throws(
    () => cdpTimeoutFromEnvironment({ HUMAN_QC_CDP_TIMEOUT_MS: "4999" }),
    /between 5000 and 10000/,
  );

  // A tiny injected deadline keeps the unit test fast; the environment parser
  // above is what constrains the real driver to the approved 5-10 second range.
  const cdp = new Cdp("ws://fixture", { requestTimeoutMs: 8 });
  cdp.socket = { send() {} };
  await assert.rejects(cdp.send("Runtime.enable"), /CDP Runtime.enable/);
  assert.equal(cdp.pending.size, 0);
});


test("CHROME_BIN wins over explicit and platform discovery, with a diagnostic missing-browser error", () => {
  const macChrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
  const exists = (candidate) => candidate === "/env/chrome" || candidate === macChrome;
  assert.equal(
    resolveChromeBin({
      environment: { CHROME_BIN: "/env/chrome" },
      explicit: "/explicit/chrome",
      exists,
    }),
    "/env/chrome",
  );
  assert.equal(
    resolveChromeBin({
      environment: {},
      exists,
      findOnPath: () => null,
    }),
    macChrome,
  );
  assert.throws(
    () => resolveChromeBin({ environment: { CHROME_BIN: "/missing/chrome" }, exists: () => false }),
    /CHROME_BIN does not point to an executable browser/,
  );
  assert.throws(
    () => resolveChromeBin({ environment: {}, exists: () => false, findOnPath: () => null }),
    /Chrome\/Chromium was not found.*Set CHROME_BIN/,
  );
});


test("a timed-out driver operation reaps both supervised Chrome and server children, escalating only when TERM did not exit", async () => {
  const chrome = new FakeChild({ exitOn: "SIGKILL" });
  const server = new FakeChild({ exitOn: "SIGTERM" });
  const cdp = new Cdp("ws://fixture", { requestTimeoutMs: 8 });
  cdp.socket = { send() {} };
  let stopped;
  await assert.rejects(async () => {
    try {
      await cdp.send("Page.navigate");
    } finally {
      stopped = await shutdownResources({
        cdp: { close: async () => {} },
        children: [chrome, server],
        stopOptions: { termGraceMs: 2, killGraceMs: 2 },
      });
    }
  }, /CDP Page.navigate/);

  assert.deepEqual(stopped, ["sigkill", "sigterm"]);
  assert.deepEqual(chrome.signals, ["SIGTERM", "SIGKILL"]);
  assert.deepEqual(server.signals, ["SIGTERM"]);
  assert.equal(chrome.signalCode, "SIGKILL");
  assert.equal(server.signalCode, "SIGTERM");
});


test("SIGTERM and SIGINT await cleanup before exiting and only handle the first signal", async () => {
  const processRef = new EventEmitter();
  const calls = [];
  const remove = installShutdownSignalHandlers({
    processRef,
    shutdown: async () => {
      await sleep(1);
      calls.push("shutdown");
    },
    exit: (code) => calls.push(`exit:${code}`),
  });

  processRef.emit("SIGTERM");
  processRef.emit("SIGINT");
  await sleep(10);
  remove();

  assert.deepEqual(calls, ["shutdown", "exit:143"]);
});


test("an operating-system SIGTERM from a Python-style parent reaps the launched Chrome child", async () => {
  const sandbox = await mkdtemp(join(tmpdir(), "human-qc-cdp-signal-"));
  const pidFile = join(sandbox, "fake-chrome.pid");
  const fakeChrome = join(sandbox, "fake-chrome.mjs");
  const driver = fileURLToPath(new URL("./cdp_driver.mjs", import.meta.url));
  const profile = join(sandbox, "profile");
  const source = `#!/usr/bin/env node\nimport { writeFileSync } from 'node:fs';\nwriteFileSync(${JSON.stringify(pidFile)}, String(process.pid));\nprocess.on('SIGTERM', () => process.exit(0));\nsetInterval(() => {}, 1000);\n`;
  let child = null;
  try {
    await writeFile(fakeChrome, source, "utf8");
    await chmod(fakeChrome, 0o755);
    child = spawn(process.execPath, [driver, "http://127.0.0.1:9/", profile], {
      env: { ...process.env, CHROME_BIN: fakeChrome },
      stdio: "ignore",
    });
    await until(async () => {
      try { return Number(await readFile(pidFile, "utf8")) > 0; } catch { return false; }
    });
    const chromePid = Number(await readFile(pidFile, "utf8"));
    child.kill("SIGTERM");
    const outcome = await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error("driver did not exit after SIGTERM")), 3_000);
      child.once("exit", (code, signal) => {
        clearTimeout(timer);
        resolve({ code, signal });
      });
    });
    assert.deepEqual(outcome, { code: 143, signal: null });
    await until(() => {
      try {
        process.kill(chromePid, 0);
        return false;
      } catch (error) {
        return error?.code === "ESRCH";
      }
    });
  } finally {
    if (child?.exitCode == null && child?.signalCode == null) child.kill("SIGKILL");
    await rm(sandbox, { recursive: true, force: true });
  }
});


test("Network.loadingFailed reports actual failures by request URL but keeps canceled media/navigation separate", () => {
  const health = new NetworkHealth("http://fixture/");
  health.request({ requestId: "navigation", request: { url: "http://fixture/" } });
  health.loadingFailed({ requestId: "navigation", errorText: "net::ERR_ABORTED", canceled: true });
  health.request({ requestId: "media", request: { url: "http://fixture/media/source" } });
  health.loadingFailed({ requestId: "media", errorText: "net::ERR_ABORTED", canceled: true });
  health.request({ requestId: "transport", request: { url: "http://fixture/api/warn/assets" } });
  health.loadingFailed({ requestId: "transport", errorText: "net::ERR_CONNECTION_REFUSED", canceled: false });
  health.request({ requestId: "blocked", request: { url: "http://fixture/media/overlay" } });
  health.loadingFailed({ requestId: "blocked", blockedReason: "inspector", canceled: false });
  health.request({ requestId: "external", request: { url: "https://external.invalid/script.js" } });
  health.loadingFailed({ requestId: "external", errorText: "net::ERR_NAME_NOT_RESOLVED", canceled: false });

  assert.deepEqual(health.canceled.map(({ url }) => url), ["http://fixture/", "http://fixture/media/source"]);
  assert.deepEqual(health.failed, [
    {
      url: "http://fixture/api/warn/assets",
      error_text: "net::ERR_CONNECTION_REFUSED",
      blocked_reason: null,
    },
    {
      url: "http://fixture/media/overlay",
      error_text: null,
      blocked_reason: "inspector",
    },
  ]);
});
