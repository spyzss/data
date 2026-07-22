#!/usr/bin/env node
/*
 * Tiny zero-dependency Chrome DevTools Protocol helper for the Task 11
 * browser contract.  It deliberately owns only browser process/event/input
 * plumbing; assertions remain in the Python test and no application state is
 * written through Runtime.evaluate.
 */

import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { spawn, spawnSync } from "node:child_process";
import { setTimeout as delay } from "node:timers/promises";
import { pathToFileURL } from "node:url";

export const DEFAULT_CDP_TIMEOUT_MS = 8_000;
export const MIN_CDP_TIMEOUT_MS = 5_000;
export const MAX_CDP_TIMEOUT_MS = 10_000;
export const TERM_GRACE_MS = 1_000;
export const KILL_GRACE_MS = 1_000;

export const cdpTimeoutFromEnvironment = (environment = process.env) => {
  const raw = environment.HUMAN_QC_CDP_TIMEOUT_MS;
  if (raw === undefined || raw === "") return DEFAULT_CDP_TIMEOUT_MS;
  const value = Number(raw);
  if (!Number.isInteger(value) || value < MIN_CDP_TIMEOUT_MS || value > MAX_CDP_TIMEOUT_MS) {
    throw new Error(
      `HUMAN_QC_CDP_TIMEOUT_MS must be an integer between ${MIN_CDP_TIMEOUT_MS} and ${MAX_CDP_TIMEOUT_MS}`,
    );
  }
  return value;
};

export const resolveChromeBin = ({
  environment = process.env,
  explicit = null,
  exists = existsSync,
  findOnPath = (name) => {
    const result = spawnSync("which", [name], { encoding: "utf8" });
    return result.status === 0 ? result.stdout.trim() : null;
  },
} = {}) => {
  const override = environment.CHROME_BIN;
  if (typeof override === "string" && override.trim()) {
    const candidate = override.trim();
    if (exists(candidate)) return candidate;
    throw new Error(`CHROME_BIN does not point to an executable browser: ${candidate}`);
  }
  if (typeof explicit === "string" && explicit.trim() && exists(explicit.trim())) return explicit.trim();
  const macCandidates = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
  ];
  for (const candidate of macCandidates) if (exists(candidate)) return candidate;
  for (const command of ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"]) {
    const candidate = findOnPath(command);
    if (candidate && exists(candidate)) return candidate;
  }
  throw new Error(
    "Chrome/Chromium was not found. Set CHROME_BIN to an executable browser path; "
      + "checked macOS Google Chrome/Chromium and google-chrome, google-chrome-stable, chromium, chromium-browser on PATH.",
  );
};

export const parseDriverArguments = (argumentsList = process.argv.slice(2)) => {
  const [baseUrl, first, second] = argumentsList;
  if (!baseUrl || !first || (second && argumentsList.length !== 3) || (!second && argumentsList.length !== 2)) {
    throw new Error("usage: cdp_driver.mjs <base-url> <profile-dir> (legacy: <base-url> <chrome-bin> <profile-dir>)");
  }
  return second
    ? { baseUrl, profileDir: second, explicitChromeBin: first }
    : { baseUrl, profileDir: first, explicitChromeBin: null };
};

export const withTimeout = async (promise, { timeoutMs, label } = {}) => {
  const timeout = Number.isInteger(timeoutMs) && timeoutMs > 0 ? timeoutMs : DEFAULT_CDP_TIMEOUT_MS;
  let timer;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(`timed out after ${timeout}ms waiting for ${label ?? "operation"}`)), timeout);
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
};

const timeoutAt = (milliseconds) => Date.now() + milliseconds;

const until = async (predicate, { timeout = 8000, label = "condition" } = {}) => {
  const deadline = timeoutAt(timeout);
  let lastError = null;
  while (Date.now() < deadline) {
    try {
      const value = await predicate();
      if (value) return value;
    } catch (error) {
      lastError = error;
    }
    await delay(25);
  }
  throw new Error(`timed out waiting for ${label}${lastError ? `: ${lastError.message}` : ""}`);
};

export class Cdp {
  constructor(url, { requestTimeoutMs = cdpTimeoutFromEnvironment() } = {}) {
    this.url = url;
    this.requestTimeoutMs = requestTimeoutMs;
    this.socket = null;
    this.sequence = 0;
    this.pending = new Map();
    this.listeners = new Map();
  }

  async connect() {
    await withTimeout(new Promise((resolve, reject) => {
      const socket = new WebSocket(this.url);
      this.socket = socket;
      socket.addEventListener("open", resolve, { once: true });
      socket.addEventListener("error", (event) => reject(event.error ?? new Error("CDP socket error")), { once: true });
      socket.addEventListener("message", (event) => this._message(event));
      socket.addEventListener("close", () => {
        for (const { reject: pendingReject } of this.pending.values()) {
          pendingReject(new Error("CDP socket closed"));
        }
        this.pending.clear();
      });
    }), { timeoutMs: this.requestTimeoutMs, label: "CDP WebSocket connection" });
    return this;
  }

  _message(event) {
    const message = JSON.parse(String(event.data));
    if (message.id) {
      const pending = this.pending.get(message.id);
      if (!pending) return;
      this.pending.delete(message.id);
      if (message.error) pending.reject(new Error(`${message.error.message} (${message.error.code})`));
      else pending.resolve(message.result ?? {});
      return;
    }
    const callbacks = this.listeners.get(message.method) ?? [];
    for (const callback of callbacks) callback(message.params ?? {});
  }

  on(method, callback) {
    const callbacks = this.listeners.get(method) ?? [];
    callbacks.push(callback);
    this.listeners.set(method, callbacks);
  }

  send(method, params = {}) {
    const id = ++this.sequence;
    const pending = new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      try {
        this.socket.send(JSON.stringify({ id, method, params }));
      } catch (error) {
        this.pending.delete(id);
        reject(error);
      }
    });
    return withTimeout(pending, {
      timeoutMs: this.requestTimeoutMs,
      label: `CDP ${method}`,
    }).finally(() => this.pending.delete(id));
  }

  async evaluate(expression, { awaitPromise = false } = {}) {
    const response = await this.send("Runtime.evaluate", {
      expression,
      returnByValue: true,
      awaitPromise,
      userGesture: true,
    });
    if (response.exceptionDetails) {
      throw new Error(response.exceptionDetails.exception?.description ?? response.exceptionDetails.text ?? "Runtime.evaluate failed");
    }
    return response.result?.value;
  }

  async close() {
    const socket = this.socket;
    if (!socket) return;
    try {
      if (socket.readyState === WebSocket.CLOSED) return;
      await withTimeout(new Promise((resolve) => {
        socket.addEventListener("close", resolve, { once: true });
        socket.close();
      }), { timeoutMs: this.requestTimeoutMs, label: "CDP WebSocket close" });
    } catch { /* process cleanup owns the fallback */ }
  }
}

export const waitForChildExit = (child, timeoutMs) => {
  if (!child || child.exitCode != null || child.signalCode != null) return Promise.resolve(true);
  return new Promise((resolve) => {
    let timer;
    const finish = () => {
      if (timer) clearTimeout(timer);
      child.removeListener?.("exit", finish);
      child.removeListener?.("close", finish);
      resolve(true);
    };
    child.once?.("exit", finish);
    child.once?.("close", finish);
    timer = setTimeout(() => {
      child.removeListener?.("exit", finish);
      child.removeListener?.("close", finish);
      resolve(false);
    }, timeoutMs);
  });
};

export const stopChild = async (
  child,
  { termGraceMs = TERM_GRACE_MS, killGraceMs = KILL_GRACE_MS } = {},
) => {
  if (!child || child.exitCode != null || child.signalCode != null) return "already_exited";
  child.kill?.("SIGTERM");
  if (await waitForChildExit(child, termGraceMs)) return "sigterm";
  child.kill?.("SIGKILL");
  if (await waitForChildExit(child, killGraceMs)) return "sigkill";
  throw new Error("Chrome process did not exit after SIGTERM and SIGKILL");
};

export const shutdownResources = async ({ cdp = null, children = [], stopOptions = {} } = {}) => {
  await cdp?.close?.();
  return Promise.all(children.filter(Boolean).map((child) => stopChild(child, stopOptions)));
};

export const installShutdownSignalHandlers = ({
  processRef = process,
  shutdown,
  exit = (code) => process.exit(code),
} = {}) => {
  let handled = false;
  const handler = (signal) => {
    if (handled) return;
    handled = true;
    void Promise.resolve(shutdown?.()).catch(() => undefined).finally(() => {
      exit(signal === "SIGINT" ? 130 : 143);
    });
  };
  processRef.once("SIGTERM", handler);
  processRef.once("SIGINT", handler);
  return () => {
    processRef.removeListener?.("SIGTERM", handler);
    processRef.removeListener?.("SIGINT", handler);
  };
};

export class NetworkHealth {
  constructor(baseUrl) {
    this.baseUrl = baseUrl;
    this.urls = new Map();
    this.failed = [];
    this.canceled = [];
  }

  _relevant(url) {
    return typeof url === "string" && url.startsWith(this.baseUrl);
  }

  request({ requestId, request }) {
    if (requestId && typeof request?.url === "string") this.urls.set(requestId, request.url);
    return request?.url;
  }

  response({ requestId, response }) {
    const url = response?.url ?? this.urls.get(requestId);
    if (requestId && typeof url === "string") this.urls.set(requestId, url);
    if (this._relevant(url) && Number(response?.status) >= 400) this.failed.push({ status: Number(response.status), url });
  }

  finished({ requestId }) {
    if (requestId) this.urls.delete(requestId);
  }

  loadingFailed({ requestId, errorText, blockedReason, canceled = false }) {
    const url = this.urls.get(requestId);
    if (requestId) this.urls.delete(requestId);
    if (!this._relevant(url)) return;
    const record = {
      url,
      error_text: typeof errorText === "string" && errorText ? errorText : null,
      blocked_reason: typeof blockedReason === "string" && blockedReason ? blockedReason : null,
    };
    if (canceled) {
      this.canceled.push(record);
      return;
    }
    this.failed.push(record);
  }
}

let baseUrl = null;
let profileDir = null;
let chrome = null;
let cdp = null;
let shutdownPromise = null;

const launchChrome = (chromeBin) => spawn(chromeBin, [
  "--headless=new",
  "--remote-debugging-port=0",
  `--user-data-dir=${profileDir}`,
  "--no-first-run",
  "--no-default-browser-check",
  "--autoplay-policy=no-user-gesture-required",
  "--window-size=1280,1100",
], { stdio: "ignore" });

const uncaught = [];
const consoleErrors = [];
const mutationRequests = [];

const shutdown = async () => {
  shutdownPromise ??= shutdownResources({ cdp, children: [chrome] });
  return shutdownPromise;
};

const devtoolsPort = async () => {
  const active = `${profileDir}/DevToolsActivePort`;
  return until(() => {
    if (!existsSync(active)) return null;
    const port = Number(readFileSync(active, "utf8").split(/\r?\n/, 1)[0]);
    return Number.isInteger(port) && port > 0 ? port : null;
  }, { label: "Chrome DevToolsActivePort" });
};

const point = async (selector, index = 0, { scroll = true } = {}) => cdp.evaluate(`(() => {
  const element = [...document.querySelectorAll(${JSON.stringify(selector)})][${index}];
  if (!element) return null;
  ${scroll ? "element.scrollIntoView({ block: 'center', inline: 'center' });" : ""}
  const rect = element.getBoundingClientRect();
  return { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2, left: rect.left, top: rect.top, width: rect.width, height: rect.height };
})()`);

const mouse = async (type, x, y, extra = {}) => cdp.send("Input.dispatchMouseEvent", {
  type,
  x,
  y,
  button: extra.button ?? "none",
  buttons: extra.buttons ?? 0,
  clickCount: extra.clickCount ?? 0,
});

const move = async (target) => mouse("mouseMoved", target.x, target.y);

const click = async (target) => {
  await move(target);
  await mouse("mousePressed", target.x, target.y, { button: "left", buttons: 1, clickCount: 1 });
  await mouse("mouseReleased", target.x, target.y, { button: "left", clickCount: 1 });
};

const key = async (name, code, keyCode) => {
  await cdp.send("Input.dispatchKeyEvent", { type: "keyDown", key: name, code, windowsVirtualKeyCode: keyCode, nativeVirtualKeyCode: keyCode });
  await cdp.send("Input.dispatchKeyEvent", { type: "keyUp", key: name, code, windowsVirtualKeyCode: keyCode, nativeVirtualKeyCode: keyCode });
};

const currentFrame = () => cdp.evaluate("document.querySelector('[data-current-frame]')?.textContent?.trim() ?? ''");

const waitFrame = async (frame) => {
  let observed = "";
  try {
    return await until(async () => {
      observed = await currentFrame();
      return observed === String(frame);
    }, { label: `frame ${frame}` });
  } catch (error) {
    throw new Error(`${error.message}; observed frame ${observed}`);
  }
};

const dragToFrame = async (frame) => {
  const head = await point(".warning-timeline__playhead");
  const track = await point(".warning-timeline__track", 0, { scroll: false });
  if (!head || !track) throw new Error("timeline playhead/track is unavailable");
  // pointerToFrame uses floor(progress * totalFrames); the interior offset
  // avoids a device-pixel boundary rounding into its neighbouring source frame.
  const x = track.left + track.width * ((frame + 0.25) / 1800);
  const y = track.top + track.height / 2;
  await cdp.evaluate("window.__task11PointerEvents = []; document.addEventListener('pointerdown', (event) => window.__task11PointerEvents.push({ type: event.type, x: Math.round(event.clientX), y: Math.round(event.clientY), buttons: event.buttons, target: event.target?.className ?? '' }), { once: true }); document.addEventListener('pointermove', (event) => window.__task11PointerEvents.push({ type: event.type, x: Math.round(event.clientX), y: Math.round(event.clientY), buttons: event.buttons, target: event.target?.className ?? '' }), { once: true });");
  await move(head);
  await mouse("mousePressed", head.x, head.y, { button: "left", buttons: 1, clickCount: 1 });
  await mouse("mouseMoved", x, y, { button: "left", buttons: 1 });
  await mouse("mouseReleased", x, y, { button: "left", clickCount: 1 });
  try {
    await waitFrame(frame);
  } catch (error) {
    const debug = await cdp.evaluate(`(() => ({
      head: ${JSON.stringify(head)}, track: ${JSON.stringify(track)}, target: { x: ${x}, y: ${y} },
      element: document.elementFromPoint(${head.x}, ${head.y})?.className ?? null,
      events: window.__task11PointerEvents ?? [],
    }))()`);
    throw new Error(`${error.message}; pointer debug ${JSON.stringify(debug)}`);
  }
};

export const runDriver = async ({
  argumentsList = process.argv.slice(2),
  environment = process.env,
} = {}) => {
  const parsed = parseDriverArguments(argumentsList);
  baseUrl = parsed.baseUrl;
  profileDir = parsed.profileDir;
  const requestTimeoutMs = cdpTimeoutFromEnvironment(environment);
  const chromeBin = resolveChromeBin({
    environment,
    explicit: parsed.explicitChromeBin,
  });
  chrome = launchChrome(chromeBin);
  const removeSignalHandlers = installShutdownSignalHandlers({ shutdown });
  try {
  mkdirSync(profileDir, { recursive: true });
  const port = await devtoolsPort();
  const pages = await until(async () => {
    const response = await fetch(`http://127.0.0.1:${port}/json/list`);
    const values = await response.json();
    return values.find((item) => item.type === "page" && item.webSocketDebuggerUrl) ?? null;
  }, { label: "Chrome page target" });
  cdp = await new Cdp(pages.webSocketDebuggerUrl, { requestTimeoutMs }).connect();
  const networkHealth = new NetworkHealth(baseUrl);
  cdp.on("Runtime.exceptionThrown", ({ exceptionDetails }) => {
    uncaught.push(exceptionDetails?.exception?.description ?? exceptionDetails?.text ?? "runtime exception");
  });
  cdp.on("Runtime.consoleAPICalled", ({ type, args }) => {
    if (type === "error") consoleErrors.push(args?.map((value) => value.value ?? value.description ?? "").join(" ") ?? "console error");
  });
  cdp.on("Log.entryAdded", ({ entry }) => {
    if (entry?.level === "error") consoleErrors.push(entry.text ?? "log error");
  });
  cdp.on("Network.responseReceived", (params) => {
    networkHealth.response(params);
  });
  cdp.on("Network.requestWillBeSent", (params) => {
    networkHealth.request(params);
    const { request } = params;
    if (request?.method === "POST" && typeof request.url === "string" && request.url.startsWith(baseUrl)) {
      mutationRequests.push(request.url);
    }
  });
  cdp.on("Network.loadingFinished", (params) => networkHealth.finished(params));
  cdp.on("Network.loadingFailed", (params) => networkHealth.loadingFailed(params));
  await cdp.send("Runtime.enable");
  await cdp.send("Log.enable");
  await cdp.send("Network.enable");
  await cdp.send("Page.enable");
  await cdp.send("Page.navigate", { url: baseUrl });
  await until(() => cdp.evaluate("Boolean(window.humanQcWarnReview?.task?.asset_id === 'asset-overlap' && document.querySelector('[data-video]')?.readyState >= 1)"), { timeout: 12000, label: "Warn app and H264 video metadata" });
  // A headless media element can retain an implementation-default playback
  // transition while metadata settles.  Begin each pointer assertion from the
  // operator's explicitly paused state; this changes only native media state,
  // not the review task or any controller private state.
  await cdp.evaluate("document.querySelector('[data-video]').pause()");
  await until(() => cdp.evaluate("document.querySelector('[data-video]').paused === true"), { label: "paused source-video baseline" });
  await cdp.evaluate("window.__task11MediaEvents = []; ['play', 'pause', 'seeking', 'seeked', 'timeupdate'].forEach((type) => document.querySelector('[data-video]').addEventListener(type, () => window.__task11MediaEvents.push({ type, paused: document.querySelector('[data-video]').paused, time: document.querySelector('[data-video]').currentTime })))");

  const initial = await cdp.evaluate(`(() => ({
    asset_id: window.humanQcWarnReview.task.asset_id,
    video_ready: document.querySelector('[data-video]').readyState >= 1,
    blocks: [...document.querySelectorAll('.warning-timeline__block')].map((block) => ({
      start: Number(block.dataset.startFrame), end: Number(block.dataset.endFrameExclusive), label: block.dataset.showLabel === 'true',
    })),
  }))()`);

  const firstBlock = await point(".warning-timeline__block", 0);
  if (!firstBlock) throw new Error("overlap block is absent");
  await move(firstBlock);
  await until(() => cdp.evaluate("!document.querySelector('.warning-timeline__popover')?.hidden"), { label: "overlap popover" });
  const popoverFrames = await cdp.evaluate("[...document.querySelector('.warning-timeline__popover')?.querySelectorAll('.warning-timeline__popover-row') ?? []].map((row) => Number(row.dataset.startFrame))");
  const secondRow = await point(".warning-timeline__popover-row", 1, { scroll: false });
  if (!secondRow) throw new Error("second overlap row is absent");
  // This real pointer path crosses the block/popover corridor; it must not
  // close during the 80ms leave grace period before the row can be clicked.
  await move(secondRow);
  const popoverTarget = await cdp.evaluate(`(() => ({
    hidden: document.querySelector('.warning-timeline__popover')?.hidden,
    target: document.elementFromPoint(${secondRow.x}, ${secondRow.y})?.className ?? null,
  }))()`);
  await click(secondRow);
  try {
    await waitFrame(142);
  } catch (error) {
    const debug = await cdp.evaluate(`(() => ({
      popoverTarget: ${JSON.stringify(popoverTarget)},
      hidden: document.querySelector('.warning-timeline__popover')?.hidden,
      row: document.elementFromPoint(${secondRow.x}, ${secondRow.y})?.className ?? null,
      current: document.querySelector('[data-current-frame]')?.textContent?.trim(),
      media: { paused: document.querySelector('[data-video]')?.paused, time: document.querySelector('[data-video]')?.currentTime },
      events: window.__task11MediaEvents ?? [],
    }))()`);
    throw new Error(`${error.message}; popover debug ${JSON.stringify(debug)}`);
  }
  const popoverSeekFrame = Number(await currentFrame());

  await dragToFrame(390);
  const dragFrame = Number(await currentFrame());
  await dragToFrame(142);

  const video = await point("[data-video]");
  if (!video) throw new Error("base video is absent");
  // The root deliberately gains frame-step focus on pointerdown.  Move before
  // release so the native video control does not also toggle play/pause; this
  // isolates the keyboard-focus contract from the browser's own click control.
  const videoFocusPoint = { x: video.x, y: video.y - Math.min(20, video.height / 4) };
  await move(videoFocusPoint);
  await mouse("mousePressed", videoFocusPoint.x, videoFocusPoint.y, { button: "left", buttons: 1, clickCount: 1 });
  await mouse("mouseMoved", videoFocusPoint.x + 20, videoFocusPoint.y, { button: "left", buttons: 1 });
  await mouse("mouseReleased", videoFocusPoint.x + 20, videoFocusPoint.y, { button: "left", clickCount: 1 });
  await until(() => cdp.evaluate("document.activeElement === document.querySelector('[data-video-root]')"), { label: "video-root focus" });
  await key("ArrowRight", "ArrowRight", 39);
  await waitFrame(143);
  const afterRight = Number(await currentFrame());
  await key("ArrowLeft", "ArrowLeft", 37);
  await waitFrame(142);
  const afterLeft = Number(await currentFrame());

  const other = await point('[data-action="toggle-reason"][data-reason-code="other"]');
  if (!other) throw new Error("Other reason chip is absent");
  await click(other);
  const input = await point("[data-reason-other]");
  if (!input) throw new Error("Other reason input did not appear");
  await click(input);
  await key("ArrowRight", "ArrowRight", 39);
  const inputFocusedFrame = Number(await currentFrame());

  const mutationsBeforeBlankOther = mutationRequests.length;
  const blankOtherFail = await point('[data-action="verdict-fail"]');
  if (!blankOtherFail) throw new Error("Fail action is absent");
  await click(blankOtherFail);
  await until(() => cdp.evaluate("document.querySelector('.panel-error')?.hidden === false"), { label: "blank Other validation" });
  const blankOtherSubmissionBlocked = mutationRequests.length === mutationsBeforeBlankOther;

  const pass = await point('[data-action="verdict-pass"]');
  if (!pass) throw new Error("Pass action is absent");
  await click(pass);
  await until(() => cdp.evaluate("window.humanQcWarnReview.task.issues.find((issue) => issue.id === 'exposure')?.review?.verdict === 'pass'"), { label: "first durable Pass" });
  const firstPassIssue = await cdp.evaluate("window.humanQcWarnReview.task.issues.find((issue) => issue.review?.verdict === 'pass')?.id ?? null");
  const savedPassMarker = await cdp.evaluate("Boolean(document.querySelector('[data-passed-marker]'))");

  await until(() => cdp.evaluate("window.humanQcWarnReview.task.issues.find((issue) => issue.id === 'shake-a')?.overlay?.status === 'ready'"), { timeout: 8000, label: "formal overlay status ready" });
  const boundaries = {};
  for (const frame of [141, 142, 181, 182]) {
    await dragToFrame(frame);
    if (frame === 142 || frame === 181) {
      await until(() => cdp.evaluate("document.querySelector('[data-overlay-video]')?.hidden === false"), { timeout: 8000, label: `overlay visible at ${frame}` });
    }
    boundaries[String(frame)] = await cdp.evaluate("document.querySelector('[data-overlay-video]')?.hidden === false");
  }
  await dragToFrame(142);
  await until(() => cdp.evaluate("document.querySelector('[data-overlay-video]')?.hidden === false"), { label: "overlay reload at 142" });
  const increase = await point('[data-action="rate-increase"]');
  if (!increase) throw new Error("rate increase control is absent");
  await click(increase);
  await until(() => cdp.evaluate("document.querySelector('[data-overlay-video]').playbackRate === document.querySelector('[data-video]').playbackRate"), { label: "overlay playback-rate sync" });
  const overlaySync = await cdp.evaluate("(() => { const source = document.querySelector('[data-video]'); const overlay = document.querySelector('[data-overlay-video]'); return Boolean(source && overlay && Math.abs(overlay.currentTime - Math.max(0, source.currentTime - 142 / 30)) <= 1 / 30 + 0.03 && overlay.playbackRate === source.playbackRate); })()");

  // Give the fetch/render queue one turn to report a deterministic transport
  // failure, rather than reading an event stream mid-dispatch.
  await delay(50);
  process.stdout.write(`${JSON.stringify({
    asset_id: initial.asset_id,
    video_ready: initial.video_ready,
    timeline_blocks: initial.blocks,
    popover_frames: popoverFrames,
    popover_seek_frame: popoverSeekFrame,
    drag_frame: dragFrame,
    keyboard_frames: [afterRight, afterLeft, inputFocusedFrame],
    blank_other_submission_blocked: blankOtherSubmissionBlocked,
    first_pass_issue: firstPassIssue,
    saved_pass_marker: savedPassMarker,
    overlay_boundaries: boundaries,
    overlay_sync: overlaySync,
    uncaught,
    console_errors: consoleErrors,
    failed_network: networkHealth.failed,
    canceled_network: networkHealth.canceled,
  })}\n`);
} finally {
  removeSignalHandlers();
  await shutdown();
}
};

const invokedDirectly = process.argv[1]
  && import.meta.url === pathToFileURL(process.argv[1]).href;
if (invokedDirectly) {
  runDriver().catch((error) => {
    process.stderr.write(`${error?.stack ?? error}\n`);
    process.exitCode = 1;
  });
}
