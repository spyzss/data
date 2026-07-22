/**
 * Frame-accurate, navigation-only projection for Warn intervals.
 *
 * The workbench passes the canonical Task 5 issue DTO to `normalizeWarnings`.
 * This module deliberately has no knowledge of verdicts, review selection, or
 * server transport: every user interaction is reduced to `onSeek(frame)`.
 */

const LABEL_MIN_WIDTH_PERCENT = 8;

const asInteger = (value) => Number.isInteger(value) ? value : null;

const asObject = (value) => value && typeof value === "object" && !Array.isArray(value) ? value : null;

const clamp = (value, minimum, maximum) => Math.min(Math.max(value, minimum), maximum);

const validTotalFrames = (totalFrames) => Number.isInteger(totalFrames) && totalFrames > 0 ? totalFrames : 0;

const compareWarnings = (left, right) => (
  left.startFrame - right.startFrame
  || left.selectedIndex - right.selectedIndex
);

const rangeForIssue = (issue) => {
  const nested = asObject(issue?.frame_range);
  if (nested) {
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

/**
 * Normalize canonical issues into the only shape consumed by the timeline.
 * Invalid or empty intervals are omitted; retained values are half-open and
 * clamped to the video extent.
 */
export function normalizeWarnings(issues, totalFrames) {
  const frameCount = validTotalFrames(totalFrames);
  if (!frameCount || !Array.isArray(issues)) return [];

  return issues.flatMap((issue, selectedIndex) => {
    const id = typeof issue?.id === "string" && issue.id.trim() ? issue.id : null;
    const { start, end } = rangeForIssue(issue);
    if (!id || start === null || end === null) return [];

    const startFrame = clamp(start, 0, frameCount - 1);
    const endFrameExclusive = clamp(end, 0, frameCount);
    if (endFrameExclusive <= startFrame) return [];

    return [{
      id,
      displayName: String(issue.display_name ?? issue.displayName ?? issue.code ?? id),
      startFrame,
      endFrameExclusive,
      selectedIndex: asInteger(issue.selectedIndex) ?? selectedIndex,
      totalFrames: frameCount,
      threshold: asObject(issue.threshold),
      source: issue,
    }];
  }).sort(compareWarnings);
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
      showLabel: widthPercent >= LABEL_MIN_WIDTH_PERCENT,
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
  return (clamp(Number(frame), 0, frameCount) / frameCount) * 100;
}

/** Map a pointer on the entire track to an exact, clamped source-video frame. */
export function pointerToFrame(event, track, totalFrames) {
  const frameCount = validTotalFrames(totalFrames);
  const rect = track?.getBoundingClientRect?.();
  if (!frameCount || !rect || !(Number(rect.width) > 0) || !Number.isFinite(event?.clientX)) return 0;
  const progress = clamp((Number(event.clientX) - Number(rect.left)) / Number(rect.width), 0, 1);
  return Math.round(progress * (frameCount - 1));
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
  }

  clampFrame(frame) {
    if (!this.totalFrames || !Number.isFinite(frame)) return 0;
    return clamp(Math.round(Number(frame)), 0, this.totalFrames - 1);
  }

  setWarnings(warnings, totalFrames = this.totalFrames) {
    this.totalFrames = validTotalFrames(totalFrames);
    this.warnings = normalizeWarnings(warnings, this.totalFrames);
    this.groups = mergeWarningIntervals(this.warnings);
    this.currentFrame = this.clampFrame(this.currentFrame);
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
    return this;
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
    if (group.showLabel) {
      const first = group.warnings[0];
      block.textContent = group.warnings.length > 1
        ? `${first.displayName} · ${group.warnings.length}`
        : first.displayName;
    }

    const popover = this.document.createElement("div");
    popover.className = "warning-timeline__popover";
    popover.hidden = true;
    popover.setAttribute?.("role", "dialog");
    popover.setAttribute?.("aria-label", "重叠 Warn 选择");
    for (const warning of group.warnings) {
      const row = this.document.createElement("button");
      row.type = "button";
      row.className = "warning-timeline__popover-row";
      row.dataset.issueId = warning.id;
      row.dataset.startFrame = String(warning.startFrame);
      row.textContent = `${warning.displayName} · ${warning.startFrame}–${warning.endFrameExclusive - 1}`;
      this._listen(row, "click", () => this.seekToFrame(warning.startFrame));
      popover.append(row);
    }

    const state = { block, popover, pointerOnBlock: false, pointerOnPopover: false, focusInBlock: false, focusInPopover: false, closeTimer: null };
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
    this._listen(block, "pointerdown", (event) => event.stopPropagation?.());
    this._listen(block, "click", () => this.seekToFrame(group.warnings[0].startFrame));
    track.append(block);
    element.append(popover);
  }

  _openPopover(state) {
    if (state.closeTimer !== null) {
      globalThis.clearTimeout?.(state.closeTimer);
      state.closeTimer = null;
    }
    state.popover.hidden = false;
    state.block.setAttribute?.("aria-expanded", "true");
  }

  _schedulePopoverClose(state) {
    if (state.closeTimer !== null) globalThis.clearTimeout?.(state.closeTimer);
    state.closeTimer = globalThis.setTimeout?.(() => {
      state.closeTimer = null;
      if (state.pointerOnBlock || state.pointerOnPopover || state.focusInBlock || state.focusInPopover) return;
      state.popover.hidden = true;
      state.block.setAttribute?.("aria-expanded", "false");
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
    this.document?.addEventListener?.("pointermove", move);
    this.document?.addEventListener?.("pointerup", end);
    this.document?.addEventListener?.("pointercancel", end);
    this._dragCleanup = () => {
      this.document?.removeEventListener?.("pointermove", move);
      this.document?.removeEventListener?.("pointerup", end);
      this.document?.removeEventListener?.("pointercancel", end);
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
