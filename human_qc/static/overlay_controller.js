/**
 * Keep one muted interval-overlay video in lockstep with the original source.
 *
 * The base video stays the sole owner of controls, keyboard and playback-rate
 * persistence.  This controller only follows base media events and never
 * drives them back.
 */

const isEventTarget = (value) => Boolean(value && typeof value.addEventListener === "function");

const finiteFrame = (value) => Number.isInteger(value) && value >= 0;

const segmentKey = (segment) => `${segment.start}:${segment.end}`;

const asFrame = (time, fps) => Math.max(0, Math.round(Number(time) * fps) || 0);

const sourceMatches = (video, url) => {
  if (typeof url !== "string" || !url) return false;
  const assigned = typeof video?.src === "string" ? video.src : "";
  if (assigned === url) return true;
  try {
    const base = video?.ownerDocument?.baseURI ?? globalThis.document?.baseURI ?? "http://localhost/";
    return new URL(assigned, base).href === new URL(url, base).href;
  } catch {
    return false;
  }
};

const safePause = (video) => {
  try { video.pause?.(); } catch { /* media teardown is best effort */ }
};

const readySegment = (issue, overlay, segment) => {
  const start = Number(segment?.start_frame ?? segment?.frame_range?.start_frame ?? overlay?.frame_range?.start_frame);
  const end = Number(segment?.end_frame_exclusive ?? segment?.frame_range?.end_frame_exclusive ?? overlay?.frame_range?.end_frame_exclusive);
  const status = segment?.status ?? overlay?.status;
  const url = segment?.url ?? overlay?.url;
  if (!finiteFrame(start) || !finiteFrame(end) || end <= start || typeof issue?.id !== "string") return null;
  return { start, end, status, url: typeof url === "string" && url ? url : null, issueId: issue.id };
};

/** A tiny single-element state machine for Task 9 interval segments. */
export class OverlayController {
  constructor({ baseVideo, overlayVideo, fps, onAvailabilityChange = () => {} } = {}) {
    if (!isEventTarget(baseVideo) || !isEventTarget(overlayVideo)) {
      throw new TypeError("OverlayController requires event-capable base and overlay videos");
    }
    if (!Number.isFinite(Number(fps)) || Number(fps) <= 0) {
      throw new RangeError("OverlayController fps must be positive");
    }
    this.baseVideo = baseVideo;
    this.overlayVideo = overlayVideo;
    this.fps = Number(fps);
    this.onAvailabilityChange = typeof onAvailabilityChange === "function" ? onAvailabilityChange : () => {};
    this.segments = new Map();
    this.current = null;
    this.currentFrame = asFrame(baseVideo.currentTime, this.fps);
    this.loaded = new Set();
    this.failed = new Set();
    this.sourceGeneration = 0;
    this.destroyed = false;
    this.listeners = [];
    this.sourceListeners = [];
    this.overlayVideo.muted = true;
    this.overlayVideo.controls = false;
    this._bind();
  }

  setEvidence(issues) {
    if (this.destroyed) return;
    const next = new Map();
    for (const issue of Array.isArray(issues) ? issues : []) {
      const overlay = issue?.overlay;
      if (!overlay || typeof overlay !== "object") continue;
      const rawSegments = Array.isArray(overlay.segments) && overlay.segments.length
        ? overlay.segments
        : [overlay];
      for (const raw of rawSegments) {
        const parsed = readySegment(issue, overlay, raw);
        if (!parsed) continue;
        const key = segmentKey(parsed);
        const existing = next.get(key);
        if (existing) {
          existing.issueIds.add(parsed.issueId);
          if (existing.status !== "ready" && parsed.status === "ready") {
            existing.status = parsed.status;
            existing.url = parsed.url;
          }
        } else {
          next.set(key, { ...parsed, key, issueIds: new Set([parsed.issueId]) });
        }
      }
    }
    const sameCurrent = this.current && next.get(this.current.key)?.url === this.current.url;
    this.segments = next;
    this.loaded = new Set([...this.loaded].filter((key) => next.has(key)));
    this.failed = new Set([...this.failed].filter((key) => next.has(key)));
    if (!sameCurrent) this._clearSource();
    this._notifyAllAvailability();
    this.updateForFrame(this.currentFrame);
  }

  updateForFrame(frame) {
    if (this.destroyed) return;
    this.currentFrame = finiteFrame(frame) ? frame : asFrame(this.baseVideo.currentTime, this.fps);
    const next = [...this.segments.values()].find((segment) => (
      segment.start <= this.currentFrame && this.currentFrame < segment.end
    )) ?? null;
    if (!next || next.status !== "ready" || !next.url) {
      this.current = null;
      this._hide();
      this._notifyAllAvailability();
      return;
    }
    if (!this.current || this.current.key !== next.key || this.current.url !== next.url) {
      this.current = next;
      this._load(next);
    }
    if (this.loaded.has(next.key) && !this.failed.has(next.key)) {
      this.syncFromBase({ hard: true });
      this.overlayVideo.hidden = false;
    }
    this._notifyAllAvailability();
  }

  syncFromBase({ hard = false } = {}) {
    const segment = this.current;
    if (!segment || !this.loaded.has(segment.key) || this.failed.has(segment.key)) return;
    const expected = Math.max(0, Number(this.baseVideo.currentTime) - segment.start / this.fps);
    const drift = Math.abs(Number(this.overlayVideo.currentTime) - expected);
    this.overlayVideo.playbackRate = Number(this.baseVideo.playbackRate) || 1;
    if (hard || drift > 1 / this.fps) this.overlayVideo.currentTime = expected;
  }

  destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const [target, type, listener] of this.listeners.splice(0)) target.removeEventListener?.(type, listener);
    this._unbindSourceEvents();
    this._clearSource();
    this.segments.clear();
  }

  _bind() {
    this._listen(this.baseVideo, "timeupdate", () => {
      this.updateForFrame(asFrame(this.baseVideo.currentTime, this.fps));
      this.syncFromBase();
    });
    this._listen(this.baseVideo, "seeking", () => this._hide());
    this._listen(this.baseVideo, "seeked", () => {
      this.updateForFrame(asFrame(this.baseVideo.currentTime, this.fps));
      this.syncFromBase({ hard: true });
    });
    this._listen(this.baseVideo, "play", () => this._playOverlay());
    this._listen(this.baseVideo, "pause", () => {
      safePause(this.overlayVideo);
      this.syncFromBase({ hard: true });
    });
    this._listen(this.baseVideo, "ratechange", () => this.syncFromBase({ hard: true }));
    this._listen(this.baseVideo, "ended", () => safePause(this.overlayVideo));
  }

  _listen(target, type, listener) {
    target.addEventListener(type, listener);
    this.listeners.push([target, type, listener]);
  }

  _load(segment) {
    this.sourceGeneration += 1;
    const generation = this.sourceGeneration;
    this._unbindSourceEvents();
    this.loaded.delete(segment.key);
    this.failed.delete(segment.key);
    safePause(this.overlayVideo);
    this.overlayVideo.hidden = true;
    this.overlayVideo.currentTime = 0;
    this.overlayVideo.src = segment.url;
    this._bindSourceEvents(generation);
    this.overlayVideo.load?.();
  }

  _clearSource() {
    this.sourceGeneration += 1;
    this._unbindSourceEvents();
    safePause(this.overlayVideo);
    this.overlayVideo.hidden = true;
    this.current = null;
    this.overlayVideo.removeAttribute?.("src");
    if ("src" in this.overlayVideo) this.overlayVideo.src = "";
    this.overlayVideo.load?.();
  }

  _bindSourceEvents(generation) {
    const bind = (type, callback) => {
      const listener = () => {
        if (!this.destroyed && generation === this.sourceGeneration) callback();
      };
      this.overlayVideo.addEventListener(type, listener);
      this.sourceListeners.push([type, listener]);
    };
    bind("canplay", () => this._onCanPlay());
    bind("loadedmetadata", () => this._onCanPlay());
    bind("error", () => this._onMediaError());
    bind("abort", () => this._onMediaError());
    bind("ended", () => {
      if (this.current && this.current.start <= this.currentFrame && this.currentFrame < this.current.end) {
        this._onMediaError();
      }
    });
  }

  _unbindSourceEvents() {
    for (const [type, listener] of this.sourceListeners.splice(0)) {
      this.overlayVideo.removeEventListener?.(type, listener);
    }
  }

  _hide() {
    this.overlayVideo.hidden = true;
    safePause(this.overlayVideo);
  }

  _onCanPlay() {
    const segment = this.current;
    // Native HTMLVideoElement.src is normalized to an absolute URL, while the
    // canonical Warn DTO intentionally carries a same-origin relative route.
    if (this.destroyed || !segment || !sourceMatches(this.overlayVideo, segment.url)) return;
    this.loaded.add(segment.key);
    this.failed.delete(segment.key);
    this.syncFromBase({ hard: true });
    this.overlayVideo.hidden = false;
    if (!this.baseVideo.paused) this._playOverlay();
    this._notifyAllAvailability();
  }

  _onMediaError() {
    const segment = this.current;
    if (!segment || this.destroyed) return;
    this.failed.add(segment.key);
    this._hide();
    this._notifyAllAvailability();
  }

  _playOverlay() {
    const segment = this.current;
    if (!segment || !this.loaded.has(segment.key) || this.failed.has(segment.key) || this.overlayVideo.hidden) return;
    this.syncFromBase({ hard: true });
    const attempt = this.overlayVideo.play?.();
    Promise.resolve(attempt).catch(() => this._onMediaError());
  }

  _notifyAllAvailability() {
    for (const segment of this.segments.values()) {
      const available = segment.status === "ready" && this.loaded.has(segment.key) && !this.failed.has(segment.key);
      for (const issueId of segment.issueIds) {
        this.onAvailabilityChange({ issueId, available, status: segment.status, code: null });
      }
    }
  }
}

export default OverlayController;
