/**
 * Frame-accurate, navigation-only projection for Warn intervals.
 *
 * The workbench passes the canonical Task 5 issue DTO to `normalizeWarnings`.
 * This module deliberately has no knowledge of verdicts, review selection, or
 * server transport: every user interaction is reduced to `onSeek(frame)`.
 */

const asInteger = (value) => Number.isInteger(value) ? value : null;

const asObject = (value) => value && typeof value === "object" && !Array.isArray(value) ? value : null;

const clamp = (value, minimum, maximum) => Math.min(Math.max(value, minimum), maximum);

const validTotalFrames = (totalFrames) => Number.isInteger(totalFrames) && totalFrames > 0 ? totalFrames : 0;

const clampFrameToTotal = (frame, totalFrames) => {
  if (!validTotalFrames(totalFrames) || !Number.isFinite(frame)) return 0;
  return clamp(Math.floor(Number(frame)), 0, totalFrames - 1);
};

const requireTotalFrames = (totalFrames) => {
  if (!validTotalFrames(totalFrames)) {
    throw new RangeError("totalFrames must be a positive integer");
  }
  return totalFrames;
};

const compareWarnings = (left, right) => (
  left.startFrame - right.startFrame
  || left.selectedIndex - right.selectedIndex
);

const rangeForIssue = (issue) => {
  if (Object.prototype.hasOwnProperty.call(issue, "frame_range")) {
    const nested = asObject(issue.frame_range);
    if (!nested) return { start: null, end: null };
    return {
      start: asInteger(nested.start_frame),
      end: asInteger(nested.end_frame_exclusive),
    };
  }

  // Top-level range fields are deliberately limited to fixture compatibility.
  // Production callers must provide `issue.frame_range` from the Warn DTO.
  return {
    start: asInteger(issue?.start_frame) ?? asInteger(issue?.startFrame),
    end: asInteger(issue?.end_frame_exclusive) ?? asInteger(issue?.endFrameExclusive),
  };
};

const closedRangeLabel = (startFrame, endFrameExclusive) => `${startFrame}–${endFrameExclusive - 1}`;

const groupLabelText = (group) => {
  const first = group.warnings[0];
  return group.warnings.length > 1
    ? `${first.displayName} · ${group.warnings.length}`
    : first.displayName;
};

const groupAriaLabel = (group) => (
  group.warnings
    .map((warning) => `${warning.displayName}，帧 ${closedRangeLabel(warning.startFrame, warning.endFrameExclusive)}`)
    .join("；")
);

const defaultLabelWidth = (text) => Array.from(String(text)).length * 16 + 24;

/**
 * Normalize canonical issues into the only shape consumed by the timeline.
 * Invalid selected warnings are rejected rather than silently omitted. Retained
 * values are half-open and clamped to the video extent.
 */
export function normalizeWarnings(issues, totalFrames) {
  const frameCount = requireTotalFrames(totalFrames);
  if (!Array.isArray(issues)) throw new TypeError("issues must be an array");
  const seenIds = new Set();

  const warnings = issues.map((issue, selectedIndex) => {
    if (!asObject(issue)) throw new TypeError(`warning at index ${selectedIndex} must be an object`);
    const id = typeof issue.id === "string" ? issue.id.trim() : "";
    if (!id) throw new TypeError(`warning at index ${selectedIndex} requires a non-empty id`);
    if (seenIds.has(id)) throw new RangeError(`warning id ${id} must be unique`);
    seenIds.add(id);

    const { start, end } = rangeForIssue(issue);
    if (start === null || end === null) {
      throw new TypeError(`warning ${id} requires integer frame bounds`);
    }
    if (end <= start) throw new RangeError(`warning ${id} has an empty frame range`);

    const startFrame = clamp(start, 0, frameCount);
    const endFrameExclusive = clamp(end, 0, frameCount);
    if (endFrameExclusive <= startFrame) {
      throw new RangeError(`warning ${id} is empty after frame clamping`);
    }

    return {
      id,
      displayName: String(issue.display_name ?? issue.displayName ?? issue.code ?? id),
      startFrame,
      endFrameExclusive,
      selectedIndex: asInteger(issue.selectedIndex) ?? selectedIndex,
      totalFrames: frameCount,
      threshold: asObject(issue.threshold),
      source: issue,
    };
  });

  return warnings.sort(compareWarnings);
}

/**
 * Merge strictly overlapping warnings for visual rendering. Adjacent ranges
 * remain distinct because an exact boundary is meaningful to an operator.
 */
export function mergeWarningIntervals(warnings) {
  const sorted = (Array.isArray(warnings) ? warnings : [])
    .filter((warning) => warning && Number.isInteger(warning.startFrame) && Number.isInteger(warning.endFrameExclusive))
    .slice()
    .sort(compareWarnings);
  const groups = [];

  for (const warning of sorted) {
    const previous = groups.at(-1);
    if (previous && warning.startFrame < previous.endFrameExclusive) {
      previous.endFrameExclusive = Math.max(previous.endFrameExclusive, warning.endFrameExclusive);
      previous.warnings.push(warning);
      previous.children = previous.warnings;
      continue;
    }
    groups.push({
      startFrame: warning.startFrame,
      endFrameExclusive: warning.endFrameExclusive,
      warnings: [warning],
      children: null,
    });
    groups.at(-1).children = groups.at(-1).warnings;
  }

  return groups.map((group) => {
    const totalFrames = validTotalFrames(group.warnings[0]?.totalFrames);
    const leftPercent = totalFrames ? (group.startFrame / totalFrames) * 100 : 0;
    const widthPercent = totalFrames
      ? ((group.endFrameExclusive - group.startFrame) / totalFrames) * 100
      : 0;
    return {
      ...group,
      leftPercent,
      widthPercent,
      // The controller decides this after measuring the actual track width.
      // A conservative model value ensures a narrow range never overflows.
      showLabel: false,
    };
  });
}

/** Return every active warning under the canonical [start, end) convention. */
export function activeWarningsAtFrame(warnings, frame) {
  if (!Number.isInteger(frame)) return [];
  return (Array.isArray(warnings) ? warnings : [])
    .filter((warning) => warning.startFrame <= frame && frame < warning.endFrameExclusive)
    .slice()
    .sort(compareWarnings);
}

/** Position a frame relative to the full video, not to a visual block width. */
export function frameToPercent(frame, totalFrames) {
  const frameCount = validTotalFrames(totalFrames);
  if (!frameCount || !Number.isFinite(frame)) return 0;
  return (clamp(Math.floor(Number(frame)), 0, frameCount - 1) / frameCount) * 100;
}

/** Map a pointer on the entire track to an exact, clamped source-video frame. */
export function pointerToFrame(event, track, totalFrames) {
  const frameCount = validTotalFrames(totalFrames);
  const rect = track?.getBoundingClientRect?.();
  if (!frameCount || !rect || !(Number(rect.width) > 0) || !Number.isFinite(event?.clientX)) return 0;
  const progress = clamp((Number(event.clientX) - Number(rect.left)) / Number(rect.width), 0, 1);
  return clamp(Math.floor(progress * frameCount), 0, frameCount - 1);
}

const setClassState = (element, className, enabled) => {
  const names = new Set(String(element.className || "").split(/\s+/).filter(Boolean));
  if (enabled) names.add(className);
  else names.delete(className);
  element.className = [...names].join(" ");
};

/**
 * Small DOM controller. It mounts into a caller-owned root and only emits
 * source frame navigation. The caller owns video playback and review state.
 */
export class WarningTimeline {
  constructor({
    warnings = [],
    totalFrames = 0,
    currentFrame = 0,
    onSeek = null,
    documentRef = globalThis.document,
    measureLabel = null,
    resizeObserverFactory = globalThis.ResizeObserver,
  } = {}) {
    this.document = documentRef;
    this.totalFrames = validTotalFrames(totalFrames);
    this.warnings = normalizeWarnings(warnings, this.totalFrames);
    this.groups = mergeWarningIntervals(this.warnings);
    this.currentFrame = this.clampFrame(currentFrame);
    this.onSeek = typeof onSeek === "function" ? onSeek : () => {};
    this.root = null;
    this.element = null;
    this.track = null;
    this.playhead = null;
    this._boundListeners = [];
    this._groupStates = [];
    this._dragCleanup = null;
    this.measureLabel = typeof measureLabel === "function" ? measureLabel : defaultLabelWidth;
    this.resizeObserverFactory = typeof resizeObserverFactory === "function" ? resizeObserverFactory : null;
    this._layoutObserver = null;
  }

  clampFrame(frame) {
    return clampFrameToTotal(frame, this.totalFrames);
  }

  setWarnings(warnings, totalFrames = this.totalFrames) {
    const nextTotalFrames = requireTotalFrames(totalFrames);
    const nextWarnings = normalizeWarnings(warnings, nextTotalFrames);
    const nextGroups = mergeWarningIntervals(nextWarnings);
    const nextCurrentFrame = clampFrameToTotal(this.currentFrame, nextTotalFrames);

    this.totalFrames = nextTotalFrames;
    this.warnings = nextWarnings;
    this.groups = nextGroups;
    this.currentFrame = nextCurrentFrame;
    if (this.root) this.mount(this.root);
    return this.groups;
  }

  setCurrentFrame(frame) {
    this.currentFrame = this.clampFrame(frame);
    this._renderCurrentFrame();
    return this.currentFrame;
  }

  seekToFrame(frame) {
    const target = this.setCurrentFrame(frame);
    this.onSeek(target);
    return target;
  }

  mount(root) {
    if (!root || typeof this.document?.createElement !== "function") {
      throw new Error("WarningTimeline requires a DOM root and document");
    }
    this.destroy();
    this.root = root;

    const element = this.document.createElement("section");
    element.className = "warning-timeline";
    element.setAttribute?.("aria-label", "整段视频时间轴");
    const track = this.document.createElement("div");
    track.className = "warning-timeline__track";
    track.setAttribute?.("role", "slider");
    track.setAttribute?.("aria-label", "视频帧位置");
    track.tabIndex = 0;
    const rail = this.document.createElement("div");
    rail.className = "warning-timeline__rail";
    track.append(rail);

    for (const group of this.groups) this._appendGroup(track, element, group);

    const playhead = this.document.createElement("button");
    playhead.type = "button";
    playhead.className = "warning-timeline__playhead";
    playhead.setAttribute?.("aria-label", "当前帧，可拖动");
    track.append(playhead);
    element.append(track);
    root.replaceChildren?.(element);

    this.element = element;
    this.track = track;
    this.playhead = playhead;
    this._listen(track, "pointerdown", (event) => this._beginDrag(event));
    this._listen(playhead, "pointerdown", (event) => {
      event.preventDefault?.();
      event.stopPropagation?.();
      this._beginDrag(event);
    });
    this._renderCurrentFrame();
    this.refreshLayout();
    this._observeTrackLayout();
    return this;
  }

  /** Recalculate label visibility after a track/layout width change. */
  refreshLayout() {
    const trackWidth = Number(this.track?.getBoundingClientRect?.()?.width);
    for (const state of this._groupStates) {
      const availableWidth = Number.isFinite(trackWidth) && trackWidth > 0
        ? (state.group.widthPercent / 100) * trackWidth
        : 0;
      const measured = Number(this.measureLabel(state.labelText, {
        group: state.group,
        block: state.block,
        availableWidth,
        trackWidth,
      }));
      const requiredWidth = Number.isFinite(measured) && measured >= 0
        ? measured
        : defaultLabelWidth(state.labelText);
      const showLabel = availableWidth > 0 && requiredWidth <= availableWidth;
      state.group.showLabel = showLabel;
      state.block.dataset.showLabel = String(showLabel);
      state.block.textContent = showLabel ? state.labelText : "";
    }
    return this.groups;
  }

  _observeTrackLayout() {
    if (!this.resizeObserverFactory || !this.track) return;
    this._layoutObserver = new this.resizeObserverFactory(() => this.refreshLayout());
    this._layoutObserver.observe?.(this.track);
  }

  _appendGroup(track, element, group) {
    const block = this.document.createElement("button");
    block.type = "button";
    block.className = "warning-timeline__block";
    block.dataset.startFrame = String(group.startFrame);
    block.dataset.endFrameExclusive = String(group.endFrameExclusive);
    block.dataset.showLabel = String(group.showLabel);
    block.style.left = `${group.leftPercent}%`;
    block.style.width = `${group.widthPercent}%`;
    block.setAttribute?.("aria-haspopup", "dialog");
    block.setAttribute?.("aria-expanded", "false");
    block.setAttribute?.("aria-label", groupAriaLabel(group));

    const popover = this.document.createElement("div");
    popover.className = "warning-timeline__popover";
    popover.hidden = true;
    popover.setAttribute?.("role", "dialog");
    popover.setAttribute?.("aria-label", "重叠 Warn 选择");
    const rows = [];
    for (const warning of group.warnings) {
      const row = this.document.createElement("button");
      row.type = "button";
      row.className = "warning-timeline__popover-row";
      row.dataset.issueId = warning.id;
      row.dataset.startFrame = String(warning.startFrame);
      row.textContent = `${warning.displayName} · ${warning.startFrame}–${warning.endFrameExclusive - 1}`;
      this._listen(row, "click", () => this.seekToFrame(warning.startFrame));
      popover.append(row);
      rows.push(row);
    }

    const state = {
      group,
      block,
      popover,
      rows,
      labelText: groupLabelText(group),
      pointerOnBlock: false,
      pointerOnPopover: false,
      focusInBlock: false,
      focusInPopover: false,
      closeTimer: null,
      suppressBlockFocusOpen: false,
    };
    this._groupStates.push(state);
    const open = () => this._openPopover(state);
    this._listen(block, "pointerenter", () => {
      state.pointerOnBlock = true;
      open();
    });
    this._listen(block, "pointerleave", () => {
      state.pointerOnBlock = false;
      this._schedulePopoverClose(state);
    });
    this._listen(popover, "pointerenter", () => {
      state.pointerOnPopover = true;
      open();
    });
    this._listen(popover, "pointerleave", () => {
      state.pointerOnPopover = false;
      this._schedulePopoverClose(state);
    });
    this._listen(block, "focusin", () => {
      state.focusInBlock = true;
      if (state.suppressBlockFocusOpen) {
        state.suppressBlockFocusOpen = false;
        return;
      }
      open();
    });
    this._listen(block, "focusout", () => {
      state.focusInBlock = false;
      this._schedulePopoverClose(state);
    });
    this._listen(popover, "focusin", () => {
      state.focusInPopover = true;
      open();
    });
    this._listen(popover, "focusout", () => {
      state.focusInPopover = false;
      this._schedulePopoverClose(state);
    });
    this._listen(block, "keydown", (event) => this._handleBlockKeydown(state, event));
    for (const row of rows) {
      this._listen(row, "keydown", (event) => this._handlePopoverRowKeydown(state, event));
    }
    this._listen(block, "pointerdown", (event) => event.stopPropagation?.());
    this._listen(block, "click", () => this.seekToFrame(group.warnings[0].startFrame));
    track.append(block);
    element.append(popover);
  }

  _handleBlockKeydown(state, event) {
    const key = event?.key;
    if (key === "ArrowDown" || key === "Enter" || key === " " || key === "Spacebar") {
      event.preventDefault?.();
      this._openPopover(state);
      state.rows[0]?.focus?.();
      return;
    }
    if (key === "Escape") {
      event.preventDefault?.();
      this._closePopover(state);
    }
  }

  _handlePopoverRowKeydown(state, event) {
    if (event?.key !== "Escape") return;
    event.preventDefault?.();
    this._closePopover(state, { returnFocus: true });
  }

  _openPopover(state) {
    if (state.closeTimer !== null) {
      globalThis.clearTimeout?.(state.closeTimer);
      state.closeTimer = null;
    }
    state.popover.hidden = false;
    state.block.setAttribute?.("aria-expanded", "true");
  }

  _closePopover(state, { returnFocus = false } = {}) {
    if (state.closeTimer !== null) {
      globalThis.clearTimeout?.(state.closeTimer);
      state.closeTimer = null;
    }
    state.popover.hidden = true;
    state.block.setAttribute?.("aria-expanded", "false");
    if (returnFocus) {
      state.suppressBlockFocusOpen = true;
      state.block.focus?.();
    }
  }

  _schedulePopoverClose(state) {
    if (state.closeTimer !== null) globalThis.clearTimeout?.(state.closeTimer);
    state.closeTimer = globalThis.setTimeout?.(() => {
      state.closeTimer = null;
      if (state.pointerOnBlock || state.pointerOnPopover || state.focusInBlock || state.focusInPopover) return;
      this._closePopover(state);
    }, 80) ?? null;
  }

  _listen(target, type, listener) {
    target.addEventListener?.(type, listener);
    this._boundListeners.push([target, type, listener]);
  }

  _beginDrag(event) {
    if (!this.track) return;
    this._endDrag();
    const pointerId = event?.pointerId;
    const captureTarget = event?.currentTarget;
    const canCapture = Number.isInteger(pointerId)
      && typeof captureTarget?.setPointerCapture === "function";
    let captured = false;
    if (canCapture) {
      try {
        captureTarget.setPointerCapture(pointerId);
        captured = true;
      } catch {
        // Pointer capture may be unavailable on a detached test or browser node.
        captured = false;
      }
    }
    let lastFrame = null;
    const ownsPointer = (candidate) => pointerId === undefined || candidate?.pointerId === undefined || candidate.pointerId === pointerId;
    const seek = (candidate) => {
      if (!ownsPointer(candidate) || !Number.isFinite(candidate?.clientX)) return;
      const frame = pointerToFrame(candidate, this.track, this.totalFrames);
      if (frame === lastFrame) return;
      lastFrame = frame;
      this.seekToFrame(frame);
    };
    seek(event);
    const move = (candidate) => seek(candidate);
    const end = (candidate) => {
      if (!ownsPointer(candidate)) return;
      if (candidate?.type !== "pointercancel") seek(candidate);
      this._endDrag();
    };
    const lostCapture = (candidate) => {
      if (ownsPointer(candidate)) this._endDrag();
    };
    this.document?.addEventListener?.("pointermove", move);
    this.document?.addEventListener?.("pointerup", end);
    this.document?.addEventListener?.("pointercancel", end);
    captureTarget?.addEventListener?.("lostpointercapture", lostCapture);
    this._dragCleanup = () => {
      this.document?.removeEventListener?.("pointermove", move);
      this.document?.removeEventListener?.("pointerup", end);
      this.document?.removeEventListener?.("pointercancel", end);
      captureTarget?.removeEventListener?.("lostpointercapture", lostCapture);
      if (captured) {
        try {
          captureTarget.releasePointerCapture?.(pointerId);
        } catch {
          // A browser can release capture itself before pointercancel is handled.
        }
      }
      this._dragCleanup = null;
    };
  }

  _endDrag() {
    this._dragCleanup?.();
  }

  _renderCurrentFrame() {
    if (this.playhead) {
      this.playhead.style.left = `${frameToPercent(this.currentFrame, this.totalFrames)}%`;
      this.playhead.dataset.frame = String(this.currentFrame);
    }
    for (const { block } of this._groupStates) {
      const startFrame = Number(block.dataset.startFrame);
      const endFrameExclusive = Number(block.dataset.endFrameExclusive);
      setClassState(block, "is-active", startFrame <= this.currentFrame && this.currentFrame < endFrameExclusive);
    }
  }

  destroy() {
    this._endDrag();
    this._layoutObserver?.disconnect?.();
    this._layoutObserver = null;
    for (const [target, type, listener] of this._boundListeners) target.removeEventListener?.(type, listener);
    this._boundListeners = [];
    for (const state of this._groupStates) {
      if (state.closeTimer !== null) globalThis.clearTimeout?.(state.closeTimer);
    }
    this._groupStates = [];
    this.root = null;
    this.element = null;
    this.track = null;
    this.playhead = null;
  }
}

export default WarningTimeline;
