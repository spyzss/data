/**
 * Full-source-video frame and playback controls for the Warn workbench.
 *
 * This controller deliberately knows nothing about warnings, verdicts, or
 * overlay state.  It receives the canonical Task 5 `video` DTO and emits only
 * source frames plus playback state for the application to fan out.
 */

export const PLAYBACK_RATES = Object.freeze([0.25, 0.5, 1, 1.5, 2, 3]);
export const PLAYBACK_RATE_STORAGE_KEY = "human-qc.playback-rate.v1";

const DEFAULT_PLAYBACK_RATE = 1;

const clamp = (value, minimum, maximum) => Math.min(Math.max(value, minimum), maximum);

const isEventTarget = (value) => Boolean(value && typeof value.addEventListener === "function");

const supportedRate = (value) => {
  const numeric = Number(value);
  return PLAYBACK_RATES.includes(numeric) ? numeric : null;
};

const safeLocalStorage = () => {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
};

const readStoredRate = (storage) => {
  if (!storage || typeof storage.getItem !== "function") return DEFAULT_PLAYBACK_RATE;
  try {
    return supportedRate(storage.getItem(PLAYBACK_RATE_STORAGE_KEY)) ?? DEFAULT_PLAYBACK_RATE;
  } catch {
    return DEFAULT_PLAYBACK_RATE;
  }
};

const frameBounds = (totalFrames) => Number.isInteger(totalFrames) && totalFrames > 0
  ? totalFrames
  : 0;

const clampFrame = (frame, totalFrames) => {
  const count = frameBounds(totalFrames);
  if (!count || !Number.isFinite(Number(frame))) return 0;
  return clamp(Math.trunc(Number(frame)), 0, count - 1);
};

const normalizeMedia = (media) => {
  if (!media || typeof media !== "object" || Array.isArray(media)) {
    throw new TypeError("video media must be a Task 5 video DTO");
  }
  const url = typeof media.url === "string" ? media.url.trim() : "";
  const fps = Number(media.fps);
  const totalFrames = Number(media.total_frames);
  if (!url) throw new TypeError("video.url must be a non-empty string");
  if (!Number.isFinite(fps) || fps <= 0) throw new RangeError("video.fps must be positive");
  if (!Number.isInteger(totalFrames) || totalFrames <= 0) {
    throw new RangeError("video.total_frames must be a positive integer");
  }
  return { url, fps, totalFrames };
};

/**
 * Controller for a single full original-video element.
 *
 * `root` must be the focusable video region. Arrow keys are intentionally not
 * registered globally: a reason input, review chip, or the page cannot steal
 * the frame-step shortcut.
 */
export class VideoController {
  constructor({
    video,
    root = video,
    storage = safeLocalStorage(),
    documentRef = globalThis.document,
    onFrameChange = null,
    onPlaybackStateChange = null,
  } = {}) {
    if (!isEventTarget(video)) throw new TypeError("VideoController requires an event-capable video element");
    if (!isEventTarget(root)) throw new TypeError("VideoController requires an event-capable video root");

    this.video = video;
    this.root = root;
    this.storage = storage;
    this.document = documentRef;
    this.onFrameChange = typeof onFrameChange === "function" ? onFrameChange : () => {};
    this.onPlaybackStateChange = typeof onPlaybackStateChange === "function"
      ? onPlaybackStateChange
      : () => {};
    this.fps = 0;
    this.totalFrames = 0;
    this.currentFrame = 0;
    this.playbackRate = readStoredRate(this.storage);
    this.media = null;
    this._rootOwnsFocus = false;
    this._listeners = [];

    this._makeRootFocusable();
    this._bindEvents();
    this._applyPlaybackRate();
  }

  /** Attach the full original video from a canonical Task 5 `task.video`. */
  setMedia(media) {
    const normalized = normalizeMedia(media);
    this.media = normalized;
    this.fps = normalized.fps;
    this.totalFrames = normalized.totalFrames;
    this.currentFrame = 0;
    this.video.src = normalized.url;
    this.video.currentTime = 0;
    this._applyPlaybackRate();
    this._emitFrameChange(true);
    this._emitPlaybackState();
    return this.playbackState;
  }

  /** Seek using a source-frame coordinate, never a visual timeline percentage. */
  seekToFrame(frame) {
    const target = clampFrame(frame, this.totalFrames);
    this.currentFrame = target;
    if (this.fps > 0) this.video.currentTime = target / this.fps;
    this._emitFrameChange();
    return target;
  }

  /** Advance or rewind by whole source frames. */
  stepFrame(delta) {
    if (!Number.isFinite(Number(delta))) return this.currentFrame;
    return this.seekToFrame(this.currentFrame + Math.trunc(Number(delta)));
  }

  /** Set one of the discrete supported rates; unsupported values leave state unchanged. */
  setRate(rate) {
    const nextRate = supportedRate(rate);
    if (nextRate === null) return this.playbackRate;
    this.playbackRate = nextRate;
    this._applyPlaybackRate();
    this._persistRate();
    this._emitPlaybackState();
    return this.playbackRate;
  }

  increaseRate() {
    const currentIndex = Math.max(0, PLAYBACK_RATES.indexOf(this.playbackRate));
    return this.setRate(PLAYBACK_RATES[Math.min(currentIndex + 1, PLAYBACK_RATES.length - 1)]);
  }

  decreaseRate() {
    const currentIndex = Math.max(0, PLAYBACK_RATES.indexOf(this.playbackRate));
    return this.setRate(PLAYBACK_RATES[Math.max(currentIndex - 1, 0)]);
  }

  /** Let an application explicitly move keyboard ownership to the video region. */
  focus() {
    this.root.focus?.();
    if (this.document?.activeElement === this.root) this._rootOwnsFocus = true;
  }

  /**
   * Return true only when an arrow step was handled.  Callers can use that
   * result without guessing whether `preventDefault()` was invoked.
   */
  handleKeydown(event) {
    if (!this._videoRootOwnsKeyboard(event)) return false;
    const delta = event?.key === "ArrowLeft" ? -1 : event?.key === "ArrowRight" ? 1 : null;
    if (delta === null) return false;
    event.preventDefault?.();
    this.stepFrame(delta);
    return true;
  }

  get playbackState() {
    return {
      currentFrame: this.currentFrame,
      playbackRate: this.playbackRate,
      isPlaying: this.video.paused === false && this.video.ended !== true,
      fps: this.fps,
      totalFrames: this.totalFrames,
    };
  }

  destroy() {
    for (const [target, type, listener] of this._listeners.splice(0)) {
      target.removeEventListener?.(type, listener);
    }
    this._rootOwnsFocus = false;
  }

  _makeRootFocusable() {
    this.root.tabIndex = 0;
    this.root.setAttribute?.("tabindex", "0");
  }

  _bindEvents() {
    this._listen(this.video, "timeupdate", () => this._syncFrameFromVideo());
    this._listen(this.video, "seeked", () => this._syncFrameFromVideo());
    this._listen(this.video, "play", () => this._emitPlaybackState());
    this._listen(this.video, "pause", () => this._emitPlaybackState());
    this._listen(this.video, "ended", () => this._emitPlaybackState());
    this._listen(this.video, "ratechange", () => this._syncNativeRate());
    this._listen(this.root, "focus", (event) => {
      if (event.target === this.root) this._rootOwnsFocus = true;
    });
    this._listen(this.root, "blur", (event) => {
      if (event.target === this.root) this._rootOwnsFocus = false;
    });
    this._listen(this.root, "pointerdown", (event) => {
      // A direct click on the video region enables the intentional frame-step
      // shortcut.  Descendant form controls keep their native focus behavior.
      if (event.target === this.root || event.target === this.video) this.focus();
    });
    this._listen(this.root, "keydown", (event) => this.handleKeydown(event));
  }

  _listen(target, type, listener) {
    target.addEventListener(type, listener);
    this._listeners.push([target, type, listener]);
  }

  _videoRootOwnsKeyboard(event) {
    const target = event?.target;
    const directVideoTarget = this.root === this.video && target === this.video;
    if (target !== this.root && !directVideoTarget) return false;
    if (this.document && "activeElement" in this.document) {
      return this.document.activeElement === this.root;
    }
    return this._rootOwnsFocus;
  }

  _syncFrameFromVideo() {
    if (!this.fps || !this.totalFrames) return this.currentFrame;
    const seconds = Number(this.video.currentTime);
    const sourceFrame = Number.isFinite(seconds) ? Math.round(seconds * this.fps) : 0;
    const nextFrame = clampFrame(sourceFrame, this.totalFrames);
    if (nextFrame === this.currentFrame) return this.currentFrame;
    this.currentFrame = nextFrame;
    this._emitFrameChange();
    return this.currentFrame;
  }

  _syncNativeRate() {
    const nativeRate = supportedRate(this.video.playbackRate);
    if (nativeRate !== null && nativeRate !== this.playbackRate) {
      this.playbackRate = nativeRate;
      this._persistRate();
    }
    this._emitPlaybackState();
  }

  _applyPlaybackRate() {
    this.video.playbackRate = this.playbackRate;
  }

  _persistRate() {
    if (!this.storage || typeof this.storage.setItem !== "function") return;
    try {
      this.storage.setItem(PLAYBACK_RATE_STORAGE_KEY, String(this.playbackRate));
    } catch {
      // A private browser context may reject storage; controls stay functional.
    }
  }

  _emitFrameChange(force = false) {
    if (force || this.totalFrames > 0) this.onFrameChange(this.currentFrame);
  }

  _emitPlaybackState() {
    this.onPlaybackStateChange(this.playbackState);
  }
}
