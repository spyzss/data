import test from "node:test";
import assert from "node:assert/strict";

import {
  PLAYBACK_RATES,
  PLAYBACK_RATE_STORAGE_KEY,
  VideoController,
} from "./video_controller.js";


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


class FakeRoot extends FakeEventTarget {
  constructor(documentRef) {
    super();
    this.documentRef = documentRef;
    this.tabIndex = -1;
  }

  focus() {
    this.documentRef.activeElement = this;
    this.dispatch("focus");
  }

  blur() {
    if (this.documentRef.activeElement === this) this.documentRef.activeElement = null;
    this.dispatch("blur");
  }
}


class FakeVideo extends FakeEventTarget {
  constructor() {
    super();
    this.currentTime = 0;
    this.playbackRate = 1;
    this.defaultPlaybackRate = 1;
    this.src = "";
    this.paused = true;
    this.pauseCalls = 0;
    this.readyState = 1;
  }

  pause() {
    this.pauseCalls += 1;
    this.paused = true;
  }
}


class FakeStorage {
  constructor(entries = {}) {
    this.entries = new Map(Object.entries(entries));
  }

  getItem(key) {
    return this.entries.get(key) ?? null;
  }

  setItem(key, value) {
    this.entries.set(key, String(value));
  }
}


function createController({ storage = new FakeStorage(), onFrameChange, onPlaybackStateChange } = {}) {
  const documentRef = { activeElement: null };
  const root = new FakeRoot(documentRef);
  const video = new FakeVideo();
  const controller = new VideoController({
    video,
    root,
    storage,
    documentRef,
    onFrameChange,
    onPlaybackStateChange,
  });
  return { controller, documentRef, root, storage, video };
}


function canonicalVideo(overrides = {}) {
  return {
    url: "/media/original.mp4",
    fps: 30,
    total_frames: 1800,
    ...overrides,
  };
}


test("uses the canonical full-video DTO for exact frame seeks and frame clamping", () => {
  const frames = [];
  const { controller, root, video } = createController({ onFrameChange: (frame) => frames.push(frame) });

  controller.setMedia(canonicalVideo());
  assert.equal(root.tabIndex, 0);
  assert.equal(video.src, "/media/original.mp4");

  controller.seekToFrame(120);
  assert.equal(video.currentTime, 4);
  assert.equal(controller.currentFrame, 120);

  controller.seekToFrame(-10);
  assert.equal(video.currentTime, 0);
  assert.equal(controller.currentFrame, 0);

  controller.seekToFrame(9999);
  assert.equal(controller.currentFrame, 1799);
  assert.equal(video.currentTime, 1799 / 30);
  assert.equal(frames.at(-1), 1799);
});


test("synchronizes rounded, clamped source frames from timeupdate and seeked", () => {
  const frames = [];
  const { controller, video } = createController({ onFrameChange: (frame) => frames.push(frame) });
  controller.setMedia(canonicalVideo({ total_frames: 150 }));

  video.currentTime = 4.016;
  video.dispatch("timeupdate");
  assert.equal(controller.currentFrame, 120);

  video.currentTime = 99;
  video.dispatch("seeked");
  assert.equal(controller.currentFrame, 149);
  assert.equal(frames.at(-1), 149);
});


test("steps one frame at a time only for a focused video root and never consumes input or page arrows", () => {
  const { controller, root, video } = createController();
  controller.setMedia(canonicalVideo({ total_frames: 200 }));
  controller.seekToFrame(120);

  assert.equal(controller.stepFrame(-1), 119);
  assert.equal(video.currentTime, 119 / 30);

  const unfocused = {
    key: "ArrowRight",
    target: root,
    prevented: false,
    preventDefault() { this.prevented = true; },
  };
  assert.equal(controller.handleKeydown(unfocused), false);
  assert.equal(unfocused.prevented, false);
  assert.equal(controller.currentFrame, 119);

  root.focus();
  const focused = root.dispatch("keydown", { key: "ArrowRight", target: root });
  assert.equal(focused.defaultPrevented, true);
  assert.equal(controller.currentFrame, 120);

  const reasonInput = { tagName: "INPUT" };
  const inputArrow = root.dispatch("keydown", { key: "ArrowRight", target: reasonInput });
  assert.equal(inputArrow.defaultPrevented, false);
  assert.equal(controller.currentFrame, 120);

  const pageArrow = {
    key: "ArrowLeft",
    target: { tagName: "BODY" },
    prevented: false,
    preventDefault() { this.prevented = true; },
  };
  assert.equal(controller.handleKeydown(pageArrow), false);
  assert.equal(pageArrow.prevented, false);
  assert.equal(controller.currentFrame, 120);
});


test("a direct video click grants frame-step focus without stealing a reason-input click", () => {
  const { controller, root, video } = createController();
  controller.setMedia(canonicalVideo({ total_frames: 200 }));

  root.dispatch("pointerdown", { target: video });
  const afterVideoClick = root.dispatch("keydown", { key: "ArrowRight", target: root });
  assert.equal(afterVideoClick.defaultPrevented, true);
  assert.equal(controller.currentFrame, 1);

  root.blur();
  const reasonInput = { tagName: "INPUT" };
  root.dispatch("pointerdown", { target: reasonInput });
  const afterReasonClick = root.dispatch("keydown", { key: "ArrowRight", target: root });
  assert.equal(afterReasonClick.defaultPrevented, false);
  assert.equal(controller.currentFrame, 1);
});


test("uses the exact fixed rate ladder, persists only the selected numeric rate, and keeps it across assets", () => {
  const storage = new FakeStorage();
  const { controller, video } = createController({ storage });
  controller.setMedia(canonicalVideo());

  assert.deepEqual(PLAYBACK_RATES, [0.25, 0.5, 1, 1.5, 2, 3]);
  assert.deepEqual(
    [controller.increaseRate(), controller.increaseRate(), controller.increaseRate(), controller.increaseRate(), controller.increaseRate()],
    [1.5, 2, 3, 3, 3],
  );
  assert.equal(video.playbackRate, 3);
  assert.equal(storage.getItem(PLAYBACK_RATE_STORAGE_KEY), "3");
  assert.deepEqual(
    [controller.decreaseRate(), controller.decreaseRate(), controller.decreaseRate(), controller.decreaseRate(), controller.decreaseRate()],
    [2, 1.5, 1, 0.5, 0.25],
  );
  assert.equal(controller.decreaseRate(), 0.25);

  controller.setRate(2);
  controller.setMedia(canonicalVideo({ url: "/media/next-original.mp4", total_frames: 300 }));
  assert.equal(video.src, "/media/next-original.mp4");
  assert.equal(video.playbackRate, 2);
  assert.equal(controller.playbackRate, 2);
});


test("restores a valid stored speed and tolerates malformed or unavailable storage", () => {
  const restoredStorage = new FakeStorage({ [PLAYBACK_RATE_STORAGE_KEY]: "1.5" });
  const restored = createController({ storage: restoredStorage });
  restored.controller.setMedia(canonicalVideo());
  assert.equal(restored.controller.playbackRate, 1.5);
  assert.equal(restored.video.playbackRate, 1.5);

  const malformed = createController({ storage: new FakeStorage({ [PLAYBACK_RATE_STORAGE_KEY]: "2.5" }) });
  malformed.controller.setMedia(canonicalVideo());
  assert.equal(malformed.controller.playbackRate, 1);

  const unavailableStorage = {
    getItem() { throw new Error("storage disabled"); },
    setItem() { throw new Error("storage disabled"); },
  };
  const unavailable = createController({ storage: unavailableStorage });
  unavailable.controller.setMedia(canonicalVideo());
  assert.equal(unavailable.controller.setRate(0.5), 0.5);
});


test("emits playback state without owning warning or verdict state", () => {
  const states = [];
  const { controller, video } = createController({ onPlaybackStateChange: (state) => states.push(state) });
  controller.setMedia(canonicalVideo());

  video.paused = false;
  video.dispatch("play");
  video.dispatch("ratechange");
  video.paused = true;
  video.dispatch("pause");

  assert.equal(states.at(-1).isPlaying, false);
  assert.equal(states.at(-1).playbackRate, 1);
  assert.equal(states.at(-1).currentFrame, controller.currentFrame);
  assert.equal("verdict" in states.at(-1), false);
});


test("releases all DOM listeners when destroyed", () => {
  const { controller, root, video } = createController();
  controller.setMedia(canonicalVideo());
  controller.destroy();

  assert.equal(root.listenerCount("keydown"), 0);
  assert.equal(root.listenerCount("focus"), 0);
  assert.equal(video.listenerCount("timeupdate"), 0);
  assert.equal(video.listenerCount("seeked"), 0);
  assert.equal(video.listenerCount("play"), 0);
});


test("pauses native playback before stepping exactly one source frame", () => {
  const { controller, video } = createController();
  controller.setMedia(canonicalVideo({ total_frames: 200 }));
  controller.seekToFrame(120);
  video.paused = false;

  assert.equal(controller.stepFrame(1), 121);
  assert.equal(video.pauseCalls, 1);
  assert.equal(video.paused, true);
  assert.equal(video.currentTime, 121 / 30);
});


test("rejects zero, fractional, and non-numeric frame-step deltas without pausing or seeking", () => {
  const { controller, video } = createController();
  controller.setMedia(canonicalVideo({ total_frames: 200 }));
  controller.seekToFrame(120);
  const initialTime = video.currentTime;

  for (const invalidDelta of [0, 0.5, -0.25, Number.NaN, "1"]) {
    assert.throws(() => controller.stepFrame(invalidDelta), RangeError);
  }

  assert.equal(controller.currentFrame, 120);
  assert.equal(video.currentTime, initialTime);
  assert.equal(video.pauseCalls, 0);
});


test("rejects non-numeric and unsupported rates before mutating controller, video, or storage", () => {
  const storage = new FakeStorage();
  const states = [];
  const { controller, video } = createController({
    storage,
    onPlaybackStateChange: (state) => states.push(state),
  });
  controller.setMedia(canonicalVideo());
  controller.setRate(1.5);
  const stateCount = states.length;

  for (const invalidRate of ["1.5", Number.NaN, 0.75]) {
    assert.throws(() => controller.setRate(invalidRate), RangeError);
  }

  assert.equal(controller.playbackRate, 1.5);
  assert.equal(video.playbackRate, 1.5);
  assert.equal(video.defaultPlaybackRate, 1);
  assert.equal(storage.getItem(PLAYBACK_RATE_STORAGE_KEY), "1.5");
  assert.equal(states.length, stateCount);
});


test("reapplies the saved legal rate to both native rate fields after metadata resets a source", () => {
  const states = [];
  const { controller, video } = createController({
    onPlaybackStateChange: (state) => states.push(state),
  });
  controller.setMedia(canonicalVideo());
  controller.setRate(2);

  video.playbackRate = 1;
  video.defaultPlaybackRate = 1;
  video.dispatch("loadedmetadata");

  assert.equal(video.playbackRate, 2);
  assert.equal(video.defaultPlaybackRate, 2);
  assert.equal(states.at(-1).playbackRate, 2);
});


test("keeps the newest pending seek visible while older timeupdate and seeked events arrive", () => {
  const frames = [];
  const { controller, video } = createController({ onFrameChange: (frame) => frames.push(frame) });
  controller.setMedia(canonicalVideo({ total_frames: 300 }));

  controller.seekToFrame(120);
  controller.seekToFrame(142);
  assert.equal(controller.currentFrame, 142);

  video.currentTime = 120 / 30;
  video.dispatch("timeupdate");
  video.dispatch("seeked");
  assert.equal(controller.currentFrame, 142);
  assert.equal(frames.at(-1), 142);

  video.currentTime = 142 / 30;
  video.dispatch("seeked");
  assert.equal(controller.currentFrame, 142);

  video.currentTime = 143 / 30;
  video.dispatch("timeupdate");
  assert.equal(controller.currentFrame, 143);
});


test("queues the newest source-frame seek until metadata is available", () => {
  const frames = [];
  const { controller, video } = createController({ onFrameChange: (frame) => frames.push(frame) });
  video.readyState = 0;
  controller.setMedia(canonicalVideo({ total_frames: 300 }));

  controller.seekToFrame(120);
  controller.seekToFrame(142);
  assert.equal(video.currentTime, 0);
  assert.equal(controller.currentFrame, 142);
  assert.deepEqual(frames, [0, 120, 142]);

  video.readyState = 1;
  video.dispatch("loadedmetadata");
  assert.equal(video.currentTime, 142 / 30);
  assert.equal(controller.currentFrame, 142);

  video.dispatch("seeked");
  assert.equal(controller.currentFrame, 142);
  assert.deepEqual(frames, [0, 120, 142]);
});
