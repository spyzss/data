#!/usr/bin/env node
/*
 * Tiny zero-dependency Chrome DevTools Protocol helper for the Task 11
 * browser contract.  It deliberately owns only browser process/event/input
 * plumbing; assertions remain in the Python test and no application state is
 * written through Runtime.evaluate.
 */

import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { spawn } from "node:child_process";
import { setTimeout as delay } from "node:timers/promises";

const [baseUrl, chromeBin, profileDir] = process.argv.slice(2);
if (!baseUrl || !chromeBin || !profileDir) {
  throw new Error("usage: cdp_driver.mjs <base-url> <chrome-bin> <profile-dir>");
}

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

class Cdp {
  constructor(url) {
    this.url = url;
    this.socket = null;
    this.sequence = 0;
    this.pending = new Map();
    this.listeners = new Map();
  }

  async connect() {
    await new Promise((resolve, reject) => {
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
    });
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
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.socket.send(JSON.stringify({ id, method, params }));
    });
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
    try { this.socket?.close(); } catch { /* process cleanup owns the fallback */ }
  }
}

const chrome = spawn(chromeBin, [
  "--headless=new",
  "--remote-debugging-port=0",
  `--user-data-dir=${profileDir}`,
  "--no-first-run",
  "--no-default-browser-check",
  "--autoplay-policy=no-user-gesture-required",
  "--window-size=1280,1100",
], { stdio: "ignore" });

let cdp = null;
const uncaught = [];
const consoleErrors = [];
const failedNetwork = [];
const mutationRequests = [];

const shutdown = async () => {
  await cdp?.close();
  if (!chrome.killed) chrome.kill("SIGTERM");
  await Promise.race([
    new Promise((resolve) => chrome.once("exit", resolve)),
    delay(2000),
  ]);
  if (!chrome.killed) chrome.kill("SIGKILL");
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

try {
  mkdirSync(profileDir, { recursive: true });
  const port = await devtoolsPort();
  const pages = await until(async () => {
    const response = await fetch(`http://127.0.0.1:${port}/json/list`);
    const values = await response.json();
    return values.find((item) => item.type === "page" && item.webSocketDebuggerUrl) ?? null;
  }, { label: "Chrome page target" });
  cdp = await new Cdp(pages.webSocketDebuggerUrl).connect();
  cdp.on("Runtime.exceptionThrown", ({ exceptionDetails }) => {
    uncaught.push(exceptionDetails?.exception?.description ?? exceptionDetails?.text ?? "runtime exception");
  });
  cdp.on("Runtime.consoleAPICalled", ({ type, args }) => {
    if (type === "error") consoleErrors.push(args?.map((value) => value.value ?? value.description ?? "").join(" ") ?? "console error");
  });
  cdp.on("Log.entryAdded", ({ entry }) => {
    if (entry?.level === "error") consoleErrors.push(entry.text ?? "log error");
  });
  cdp.on("Network.responseReceived", ({ response }) => {
    if (typeof response?.url === "string" && response.url.startsWith(baseUrl) && Number(response.status) >= 400) {
      failedNetwork.push({ status: Number(response.status), url: response.url });
    }
  });
  cdp.on("Network.requestWillBeSent", ({ request }) => {
    if (request?.method === "POST" && typeof request.url === "string" && request.url.startsWith(baseUrl)) {
      mutationRequests.push(request.url);
    }
  });
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
    failed_network: failedNetwork,
  })}\n`);
} finally {
  await shutdown();
}
