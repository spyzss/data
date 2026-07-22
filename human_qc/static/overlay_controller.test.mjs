import test from "node:test";
import assert from "node:assert/strict";

import { OverlayController } from "./overlay_controller.js";


class FakeEventTarget {
  constructor() { this.listeners = new Map(); }
  addEventListener(type, listener) {
    const values = this.listeners.get(type) ?? [];
    values.push(listener);
    this.listeners.set(type, values);
  }
  removeEventListener(type, listener) {
    this.listeners.set(type, (this.listeners.get(type) ?? []).filter((value) => value !== listener));
  }
  dispatch(type) {
    for (const listener of [...(this.listeners.get(type) ?? [])]) listener({ type, target: this });
  }
  listenerCount(type) { return (this.listeners.get(type) ?? []).length; }
}


class FakeVideo extends FakeEventTarget {
  constructor() {
    super();
    this.currentTime = 0;
    this.playbackRate = 1;
    this.paused = true;
    this.hidden = true;
    this.src = "";
    this.playCalls = 0;
    this.pauseCalls = 0;
  }
  play() { this.playCalls += 1; this.paused = false; return Promise.resolve(); }
  pause() { this.pauseCalls += 1; this.paused = true; }
  load() {}
  removeAttribute(name) { if (name === "src") this.src = ""; }
}


const evidence = () => [{
  id: "exposure",
  overlay: {
    status: "ready",
    segments: [{ start_frame: 120, end_frame_exclusive: 182, status: "ready", url: "/media/a" }],
  },
}, {
  id: "shake",
  overlay: {
    status: "ready",
    segments: [{ start_frame: 120, end_frame_exclusive: 182, status: "ready", url: "/media/a" }],
  },
}, {
  id: "late",
  overlay: {
    status: "ready",
    segments: [{ start_frame: 390, end_frame_exclusive: 427, status: "ready", url: "/media/b" }],
  },
}];


test("a single overlay follows the base source time across half-open segments", () => {
  const base = new FakeVideo();
  const overlay = new FakeVideo();
  const controller = new OverlayController({ baseVideo: base, overlayVideo: overlay, fps: 30 });
  controller.setEvidence(evidence());

  base.currentTime = 4.75;
  base.playbackRate = 1.5;
  controller.updateForFrame(120);
  assert.equal(overlay.src, "/media/a");
  overlay.dispatch("canplay");
  controller.syncFromBase({ hard: true });
  assert.equal(overlay.currentTime, 0.75);
  assert.equal(overlay.playbackRate, 1.5);
  assert.equal(overlay.hidden, false);

  controller.updateForFrame(182);
  assert.equal(overlay.hidden, true);
  controller.updateForFrame(390);
  assert.equal(overlay.src, "/media/b");
  assert.equal(overlay.currentTime, 0);
  assert.equal(overlay.hidden, true);
  overlay.dispatch("canplay");
  assert.equal(overlay.hidden, false);
});


test("it mirrors base media events without adding keyboard ownership and cleans up", async () => {
  const base = new FakeVideo();
  const overlay = new FakeVideo();
  const availability = [];
  const controller = new OverlayController({
    baseVideo: base,
    overlayVideo: overlay,
    fps: 30,
    onAvailabilityChange: (value) => availability.push(value),
  });
  controller.setEvidence(evidence());
  base.currentTime = 4;
  base.dispatch("seeked");
  overlay.dispatch("canplay");
  base.paused = false;
  base.dispatch("play");
  await Promise.resolve();
  assert.equal(overlay.playCalls, 1);
  base.dispatch("pause");
  assert.equal(overlay.pauseCalls > 0, true);
  assert.equal(base.listenerCount("keydown"), 0);
  overlay.dispatch("error");
  assert.equal(overlay.hidden, true);
  assert.equal(availability.at(-1).available, false);

  controller.destroy();
  assert.equal(base.listenerCount("timeupdate"), 0);
  assert.equal(overlay.listenerCount("canplay"), 0);
  assert.equal(overlay.src, "");
});
