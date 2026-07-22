import test from "node:test";
import assert from "node:assert/strict";

import {
  WarningTimeline,
  activeWarningsAtFrame,
  frameToPercent,
  mergeWarningIntervals,
  normalizeWarnings,
  pointerToFrame,
} from "./warning_timeline.js";


class FakeEventTarget {
  constructor() {
    this.listeners = new Map();
  }

  addEventListener(type, listener) {
    const listeners = this.listeners.get(type) ?? [];
    listeners.push(listener);
    this.listeners.set(type, listeners);
  }

  removeEventListener(type, listener) {
    const listeners = this.listeners.get(type) ?? [];
    this.listeners.set(type, listeners.filter((candidate) => candidate !== listener));
  }

  dispatch(type, properties = {}) {
    const event = {
      type,
      currentTarget: this,
      target: this,
      defaultPrevented: false,
      preventDefault() { this.defaultPrevented = true; },
      ...properties,
    };
    for (const listener of [...(this.listeners.get(type) ?? [])]) listener(event);
    return event;
  }

  listenerCount(type) {
    return (this.listeners.get(type) ?? []).length;
  }
}


class FakeElement extends FakeEventTarget {
  constructor(tagName = "div") {
    super();
    this.tagName = tagName.toUpperCase();
    this.children = [];
    this.parentNode = null;
    this.className = "";
    this.dataset = {};
    this.style = {};
    this.hidden = false;
    this.textContent = "";
    this.type = "";
    this.tabIndex = -1;
    this.attributes = new Map();
    this.rect = { left: 0, width: 100 };
    this.focused = false;
    this.capturedPointerIds = [];
    this.releasedPointerIds = [];
  }

  append(...children) {
    for (const child of children) this.appendChild(child);
  }

  appendChild(child) {
    child.parentNode = this;
    this.children.push(child);
    return child;
  }

  replaceChildren(...children) {
    this.children = [];
    this.append(...children);
  }

  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }

  getAttribute(name) {
    return this.attributes.get(name) ?? null;
  }

  contains(candidate) {
    if (candidate === this) return true;
    return this.children.some((child) => child.contains(candidate));
  }

  getBoundingClientRect() {
    return { ...this.rect };
  }

  focus() {
    this.focused = true;
  }

  setPointerCapture(pointerId) {
    this.capturedPointerIds.push(pointerId);
  }

  releasePointerCapture(pointerId) {
    this.releasedPointerIds.push(pointerId);
  }
}


class FakeDocument extends FakeEventTarget {
  createElement(tagName) {
    return new FakeElement(tagName);
  }
}


function findAll(root, predicate, found = []) {
  if (predicate(root)) found.push(root);
  for (const child of root.children) findAll(child, predicate, found);
  return found;
}


const canonicalIssues = [
  {
    id: "exposure",
    display_name: "曝光异常",
    frame_range: { start_frame: 120, end_frame_exclusive: 169 },
    threshold: { operator: ">", value: 0.7 },
  },
  {
    id: "shake",
    display_name: "画面抖动",
    frame_range: { start_frame: 142, end_frame_exclusive: 182 },
    threshold: { operator: ">", value: 0.5 },
  },
  {
    id: "late",
    display_name: "画面抖动",
    frame_range: { start_frame: 390, end_frame_exclusive: 427 },
  },
];


test("normalizes canonical nested ranges and projects true overlap against the full video", () => {
  const warnings = normalizeWarnings(canonicalIssues, 1800);

  assert.deepEqual(
    warnings.map(({ id, startFrame, endFrameExclusive, selectedIndex }) => [id, startFrame, endFrameExclusive, selectedIndex]),
    [["exposure", 120, 169, 0], ["shake", 142, 182, 1], ["late", 390, 427, 2]],
  );
  assert.deepEqual(
    mergeWarningIntervals(warnings).map(({ startFrame, endFrameExclusive }) => [startFrame, endFrameExclusive]),
    [[120, 182], [390, 427]],
  );
  assert.deepEqual(activeWarningsAtFrame(warnings, 142).map(({ id }) => id), ["exposure", "shake"]);
  assert.deepEqual(activeWarningsAtFrame(warnings, 169).map(({ id }) => id), ["shake"]);
  assert.equal(frameToPercent(900, 1800), 50);
});


test("keeps selected-index ties deterministic, clamps fixture aliases, and does not merge adjacent ranges", () => {
  const warnings = normalizeWarnings([
    { id: "later-tie", start_frame: 10, end_frame_exclusive: 20 },
    { id: "first-tie", start_frame: 10, end_frame_exclusive: 15 },
    { id: "adjacent", start_frame: 20, end_frame_exclusive: 30 },
    { id: "clamped", start_frame: -10, end_frame_exclusive: 4 },
  ], 100);

  assert.deepEqual(warnings.map(({ id }) => id), ["clamped", "later-tie", "first-tie", "adjacent"]);
  const groups = mergeWarningIntervals(warnings);
  assert.deepEqual(groups.map(({ startFrame, endFrameExclusive }) => [startFrame, endFrameExclusive]), [[0, 4], [10, 20], [20, 30]]);
  assert.deepEqual(groups[1].warnings.map(({ id }) => id), ["later-tie", "first-tie"]);
  assert.equal(groups[1].leftPercent, 10);
  assert.equal(groups[1].widthPercent, 10);
});


test("preserves original selected ordering when an already-normalized model is mounted", () => {
  const normalized = normalizeWarnings([
    { id: "selected-first", frame_range: { start_frame: 30, end_frame_exclusive: 40 } },
    { id: "selected-second", frame_range: { start_frame: 10, end_frame_exclusive: 20 } },
  ], 100);
  const timeline = new WarningTimeline({ warnings: normalized, totalFrames: 100 });

  assert.deepEqual(
    timeline.warnings.map(({ id, selectedIndex }) => [id, selectedIndex]),
    [["selected-second", 1], ["selected-first", 0]],
  );
});


test("uses exact clamped frames for free track and needle mapping", () => {
  const track = { getBoundingClientRect: () => ({ left: 100, width: 200 }) };
  assert.equal(pointerToFrame({ clientX: 100 }, track, 100), 0);
  assert.equal(pointerToFrame({ clientX: 200 }, track, 100), 50);
  assert.equal(pointerToFrame({ clientX: 300 }, track, 100), 99);
  assert.equal(pointerToFrame({ clientX: -10 }, track, 100), 0);
  assert.equal(pointerToFrame({ clientX: 1000 }, track, 100), 99);
});


test("uses one half-open coordinate system for high frame positions and playhead placement", () => {
  const track = { getBoundingClientRect: () => ({ left: 0, width: 100 }) };

  assert.equal(pointerToFrame({ clientX: 90 }, track, 100), 90);
  assert.equal(pointerToFrame({ clientX: 99 }, track, 100), 99);
  assert.equal(frameToPercent(99, 100), 99);
  assert.equal(frameToPercent(100, 100), 99);
});


test("floors controlled current frames so a fractional playback value cannot jump ahead", () => {
  const timeline = new WarningTimeline({
    warnings: [{ id: "only", frame_range: { start_frame: 10, end_frame_exclusive: 20 } }],
    totalFrames: 100,
  });

  assert.equal(timeline.setCurrentFrame(50.9), 50);
  assert.equal(timeline.currentFrame, 50);
});


test("rejects malformed selected warnings instead of silently dropping them", () => {
  assert.throws(() => normalizeWarnings([], 0), RangeError);
  assert.throws(() => normalizeWarnings("not-an-array", 100), TypeError);
  assert.throws(() => normalizeWarnings([
    { frame_range: { start_frame: 1, end_frame_exclusive: 2 } },
  ], 100), TypeError);
  assert.throws(() => normalizeWarnings([
    { id: "duplicate", frame_range: { start_frame: 1, end_frame_exclusive: 2 } },
    { id: "duplicate", frame_range: { start_frame: 3, end_frame_exclusive: 4 } },
  ], 100), RangeError);
  assert.throws(() => normalizeWarnings([
    { id: "empty", frame_range: { start_frame: 9, end_frame_exclusive: 9 } },
  ], 100), RangeError);
  assert.throws(() => normalizeWarnings([
    { id: "outside", frame_range: { start_frame: 100, end_frame_exclusive: 101 } },
  ], 100), RangeError);
  assert.throws(() => normalizeWarnings([
    {
      id: "malformed-nested-range",
      frame_range: null,
      start_frame: 1,
      end_frame_exclusive: 2,
    },
  ], 100), TypeError);
});


test("keeps the mounted model intact when a replacement warning payload is rejected", () => {
  const timeline = new WarningTimeline({
    warnings: [{ id: "stable", frame_range: { start_frame: 10, end_frame_exclusive: 20 } }],
    totalFrames: 100,
    currentFrame: 12,
  });

  assert.throws(() => timeline.setWarnings([
    { id: "replacement", frame_range: { start_frame: 30, end_frame_exclusive: 40 } },
  ], 0), RangeError);
  assert.equal(timeline.totalFrames, 100);
  assert.equal(timeline.currentFrame, 12);
  assert.deepEqual(timeline.warnings.map(({ id }) => id), ["stable"]);
});


test("fits labels to actual track pixels, recomputes layout, and retains full block accessibility text", () => {
  const documentRef = new FakeDocument();
  const root = new FakeElement("section");
  const timeline = new WarningTimeline({
    documentRef,
    warnings: [{ id: "short", display_name: "短问题", frame_range: { start_frame: 10, end_frame_exclusive: 20 } }],
    totalFrames: 100,
    measureLabel: () => 30,
  });

  timeline.mount(root);
  const block = findAll(root, (element) => element.className.includes("warning-timeline__block"))[0];
  const track = findAll(root, (element) => element.className.includes("warning-timeline__track"))[0];
  assert.equal(block.dataset.showLabel, "false");
  assert.equal(block.textContent, "");
  assert.match(block.getAttribute("aria-label"), /短问题.*10.*19/);

  track.rect = { left: 0, width: 400 };
  timeline.refreshLayout();
  assert.equal(block.dataset.showLabel, "true");
  assert.equal(block.textContent, "短问题");
});


test("observes track resize for label fit and releases the observer on destroy", () => {
  const documentRef = new FakeDocument();
  const root = new FakeElement("section");
  const observers = [];
  class FakeResizeObserver {
    constructor(callback) {
      this.callback = callback;
      this.observed = null;
      this.disconnected = false;
      observers.push(this);
    }

    observe(element) {
      this.observed = element;
    }

    disconnect() {
      this.disconnected = true;
    }
  }
  const timeline = new WarningTimeline({
    documentRef,
    warnings: [{ id: "short", display_name: "短问题", frame_range: { start_frame: 10, end_frame_exclusive: 20 } }],
    totalFrames: 100,
    measureLabel: () => 30,
    resizeObserverFactory: FakeResizeObserver,
  });

  timeline.mount(root);
  const block = findAll(root, (element) => element.className.includes("warning-timeline__block"))[0];
  const track = findAll(root, (element) => element.className.includes("warning-timeline__track"))[0];
  assert.equal(observers.length, 1);
  assert.equal(observers[0].observed, track);
  assert.equal(block.dataset.showLabel, "false");

  track.rect = { left: 0, width: 400 };
  observers[0].callback();
  assert.equal(block.dataset.showLabel, "true");

  timeline.destroy();
  assert.equal(observers[0].disconnected, true);
});


test("lets keyboard users enter a popover and escape back to its warning block", () => {
  const documentRef = new FakeDocument();
  const root = new FakeElement("section");
  const timeline = new WarningTimeline({
    documentRef,
    warnings: canonicalIssues,
    totalFrames: 1800,
  });

  timeline.mount(root);
  const block = findAll(root, (element) => element.className.includes("warning-timeline__block"))[0];
  const popover = findAll(root, (element) => element.className.includes("warning-timeline__popover"))[0];
  const row = findAll(root, (element) => element.className.includes("warning-timeline__popover-row"))[0];
  const enter = block.dispatch("keydown", { key: "ArrowDown" });
  assert.equal(enter.defaultPrevented, true);
  assert.equal(popover.hidden, false);
  assert.equal(row.focused, true);

  const escape = row.dispatch("keydown", { key: "Escape" });
  assert.equal(escape.defaultPrevented, true);
  assert.equal(popover.hidden, true);
  assert.equal(block.focused, true);
});


test("renders narrow color-only blocks and keeps a hover popover enterable for per-warning seeks", () => {
  const documentRef = new FakeDocument();
  const root = new FakeElement("section");
  const seeks = [];
  const timeline = new WarningTimeline({
    documentRef,
    warnings: normalizeWarnings(canonicalIssues, 1800),
    totalFrames: 1800,
    currentFrame: 120,
    onSeek: (frame) => seeks.push(frame),
  });

  timeline.mount(root);
  const blocks = findAll(root, (element) => element.className.includes("warning-timeline__block"));
  assert.equal(blocks.length, 2);
  assert.equal(blocks[0].dataset.showLabel, "false");
  assert.equal(blocks[0].textContent, "");

  const popover = findAll(root, (element) => element.className.includes("warning-timeline__popover"))[0];
  const rows = findAll(root, (element) => element.className.includes("warning-timeline__popover-row"));
  blocks[0].dispatch("pointerenter");
  assert.equal(popover.hidden, false);
  blocks[0].dispatch("pointerleave");
  popover.dispatch("pointerenter");
  assert.equal(popover.hidden, false);
  rows[1].dispatch("click");
  assert.deepEqual(seeks, [142]);
  assert.equal(timeline.currentFrame, 142);
});


test("block clicks and both drag paths seek only to concrete clamped frames and clean up listeners", () => {
  const documentRef = new FakeDocument();
  const root = new FakeElement("section");
  const seeks = [];
  const warnings = normalizeWarnings([
    { id: "first", start_frame: 10, end_frame_exclusive: 30 },
    { id: "second", start_frame: 15, end_frame_exclusive: 40 },
  ], 100);
  const timeline = new WarningTimeline({
    documentRef,
    warnings,
    totalFrames: 100,
    onSeek: (frame) => seeks.push(frame),
  });

  timeline.mount(root);
  const block = findAll(root, (element) => element.className.includes("warning-timeline__block"))[0];
  const track = findAll(root, (element) => element.className.includes("warning-timeline__track"))[0];
  const needle = findAll(root, (element) => element.className.includes("warning-timeline__playhead"))[0];
  track.rect = { left: 0, width: 100 };

  let blockStopped = false;
  let needleStopped = false;
  block.dispatch("pointerdown", { stopPropagation() { blockStopped = true; } });
  needle.dispatch("pointerdown", { clientX: 20, pointerId: 2, stopPropagation() { needleStopped = true; } });
  documentRef.dispatch("pointerup", { clientX: 20, pointerId: 2 });
  assert.equal(blockStopped, true);
  assert.equal(needleStopped, true);

  block.dispatch("click");
  track.dispatch("pointerdown", { clientX: 50, pointerId: 1 });
  documentRef.dispatch("pointermove", { clientX: 100, pointerId: 1 });
  documentRef.dispatch("pointerup", { clientX: -100, pointerId: 1 });
  needle.dispatch("pointerdown", { clientX: 20, pointerId: 2 });
  documentRef.dispatch("pointermove", { clientX: 100, pointerId: 2 });
  documentRef.dispatch("pointerup", { clientX: 100, pointerId: 2 });

  assert.deepEqual(seeks, [20, 10, 50, 99, 0, 20, 99]);
  assert.equal(timeline.currentFrame, 99);
  assert.equal(documentRef.listenerCount("pointermove"), 0);
  assert.equal(documentRef.listenerCount("pointerup"), 0);

  timeline.destroy();
  track.dispatch("pointerdown", { clientX: 5, pointerId: 3 });
  assert.deepEqual(seeks, [20, 10, 50, 99, 0, 20, 99]);
});


test("pointer cancellation releases the drag without inventing a frame-zero seek", () => {
  const documentRef = new FakeDocument();
  const root = new FakeElement("section");
  const seeks = [];
  const timeline = new WarningTimeline({
    documentRef,
    warnings: normalizeWarnings([{ id: "only", start_frame: 10, end_frame_exclusive: 20 }], 100),
    totalFrames: 100,
    onSeek: (frame) => seeks.push(frame),
  });

  timeline.mount(root);
  const track = findAll(root, (element) => element.className.includes("warning-timeline__track"))[0];
  track.rect = { left: 0, width: 100 };
  track.dispatch("pointerdown", { clientX: 50, pointerId: 1 });
  documentRef.dispatch("pointercancel", { pointerId: 1 });

  assert.deepEqual(seeks, [50]);
  assert.equal(documentRef.listenerCount("pointermove"), 0);
  assert.equal(documentRef.listenerCount("pointerup"), 0);
});


test("captures and releases the active pointer for up, cancel, and destroy cleanup", () => {
  const documentRef = new FakeDocument();
  const root = new FakeElement("section");
  const timeline = new WarningTimeline({
    documentRef,
    warnings: [{ id: "only", frame_range: { start_frame: 10, end_frame_exclusive: 20 } }],
    totalFrames: 100,
  });

  timeline.mount(root);
  const track = findAll(root, (element) => element.className.includes("warning-timeline__track"))[0];
  track.rect = { left: 0, width: 100 };

  track.dispatch("pointerdown", { clientX: 20, pointerId: 7 });
  assert.deepEqual(track.capturedPointerIds, [7]);
  documentRef.dispatch("pointerup", { clientX: 20, pointerId: 7 });
  assert.deepEqual(track.releasedPointerIds, [7]);

  track.dispatch("pointerdown", { clientX: 30, pointerId: 8 });
  documentRef.dispatch("pointercancel", { pointerId: 8 });
  assert.deepEqual(track.releasedPointerIds, [7, 8]);

  track.dispatch("pointerdown", { clientX: 40, pointerId: 9 });
  timeline.destroy();
  assert.deepEqual(track.releasedPointerIds, [7, 8, 9]);
  assert.equal(documentRef.listenerCount("pointermove"), 0);
});
