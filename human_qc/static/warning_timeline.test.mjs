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
    { id: "invalid", start_frame: 9, end_frame_exclusive: 9 },
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
