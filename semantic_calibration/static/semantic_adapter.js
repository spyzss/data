/**
 * Semantic calibration adapter for the framework-free human QC workbench.
 *
 * The timeline is intentionally a shared-boundary model: segments are
 * ordinary layout elements and only the N-1 internal handles receive pointer
 * listeners. A boundary move is previewed locally and submitted once on
 * pointerup, as one transaction that changes the two adjacent segments.
 */

const asSegments = (semantic) => {
  const timeline = semantic?.timeline ?? semantic?.working_timeline ?? semantic ?? {};
  const segments = timeline.segments ?? timeline;
  return Array.isArray(segments) ? segments : [];
};

export function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

export function displayEndFrame(segment) {
  const exclusive = Number(segment?.end_frame_exclusive);
  return Number.isFinite(exclusive) ? exclusive - 1 : "—";
}

export function buildTimelineModel(semantic) {
  const timeline = semantic?.timeline ?? semantic?.working_timeline ?? {};
  const segments = asSegments(semantic).map((segment, index) => {
    const start = Number(segment?.start_frame ?? 0);
    const endExclusive = Number(segment?.end_frame_exclusive ?? segment?.end_frame ?? start + 1);
    return {
      ...segment,
      internal_id: String(segment?.internal_id ?? segment?.id ?? `segment-${index + 1}`),
      start_frame: start,
      end_frame_exclusive: endExclusive,
      end_frame_inclusive: endExclusive - 1,
      index,
      text_cn: String(segment?.text_cn ?? segment?.subtask_cn ?? ""),
      text_en: String(segment?.text_en ?? segment?.subtask_en ?? ""),
    };
  });
  const frameCount = Number(timeline.frame_count ?? segments.at(-1)?.end_frame_exclusive ?? 0);
  const handles = segments.slice(1).map((segment, offset) => {
    const boundaryIndex = offset + 1;
    const previous = segments[offset];
    const frameExclusive = Number(previous.end_frame_exclusive);
    return {
      boundary_index: boundaryIndex,
      frame_exclusive: frameExclusive,
      previous_segment_id: previous.internal_id,
      next_segment_id: segment.internal_id,
      actor_segment_id: segment.internal_id,
      percent: frameCount > 0 ? (frameExclusive / frameCount) * 100 : 0,
    };
  });
  return {
    frameCount,
    fps: Number(timeline.fps ?? 0),
    segments,
    handles,
  };
}

export function makeBoundaryPayload({
  boundaryIndex,
  actorSegmentId,
  frameExclusive,
  expectedRevision,
  leaseToken,
} = {}) {
  const payload = {
    boundary_index: boundaryIndex,
    actor_segment_id: actorSegmentId,
    new_frame_exclusive: frameExclusive,
  };
  if (expectedRevision !== undefined) payload.expected_revision = expectedRevision;
  if (leaseToken !== undefined) payload.lease_token = leaseToken;
  return payload;
}

export function pendingPresentation(pending) {
  const before = Array.isArray(pending?.before) ? pending.before : [];
  const after = Array.isArray(pending?.after) ? pending.after : [];
  const affected = Array.isArray(pending?.affected_segment_ids)
    ? pending.affected_segment_ids.map(String)
    : [...new Set([...before, ...after].map((item) => item?.internal_id).filter(Boolean))];
  return {
    editType: pending?.edit_type ?? pending?.type ?? "boundary",
    affectedSegmentIds: affected,
    before,
    after,
    boundaryIndex: pending?.boundary_index ?? null,
  };
}

export function linkedBoundaryPreview(model, boundaryIndex, frameExclusive) {
  const previous = model?.segments?.[boundaryIndex - 1];
  const following = model?.segments?.[boundaryIndex];
  if (!previous || !following || !Number.isInteger(frameExclusive)) return null;
  if (!(previous.start_frame < frameExclusive && frameExclusive < following.end_frame_exclusive)) return null;
  const frameCount = Number(model.frameCount || 0);
  return {
    previous: {
      startFrame: previous.start_frame,
      endFrameExclusive: frameExclusive,
      widthPercent: frameCount > 0 ? ((frameExclusive - previous.start_frame) / frameCount) * 100 : 0,
    },
    following: {
      startFrame: frameExclusive,
      endFrameExclusive: following.end_frame_exclusive,
      widthPercent: frameCount > 0 ? ((following.end_frame_exclusive - frameExclusive) / frameCount) * 100 : 0,
    },
  };
}

export function renderTimelineMarkup(model, { pending = null, preview = null } = {}) {
  const affected = new Set(pendingPresentation(pending).affectedSegmentIds);
  const previewBoundary = preview?.boundary_index;
  const previewFrame = preview?.frame_exclusive;
  const segments = model.segments
    .map((segment, index) => {
      const className = ["timeline-segment", affected.has(segment.internal_id) ? "pending-affected" : ""]
        .filter(Boolean)
        .join(" ");
      const width = model.frameCount > 0
        ? Math.max(0, ((segment.end_frame_exclusive - segment.start_frame) / model.frameCount) * 100)
        : 0;
      const text = escapeHtml(segment.text_cn || segment.text_en || `Segment ${index + 1}`);
      const endFrame = displayEndFrame(segment);
      let html = `<div class="${className}" data-segment-id="${escapeHtml(segment.internal_id)}" data-start-frame="${segment.start_frame}" data-end-frame="${endFrame}" style="--segment-width:${width}%">`;
      html += `<div class="segment-title">${text}</div><div class="segment-range">${segment.start_frame}–${endFrame}</div></div>`;
      const handle = model.handles[index];
      if (handle) {
        const isPreview = handle.boundary_index === previewBoundary;
        const frame = isPreview ? previewFrame : handle.frame_exclusive;
        html += `<button type="button" class="boundary-handle${isPreview ? " is-preview" : ""}" data-boundary-index="${handle.boundary_index}" data-actor-segment-id="${escapeHtml(handle.actor_segment_id)}" data-frame-exclusive="${frame}" aria-label="调整第 ${handle.boundary_index} 个共享边界"${pending ? " disabled" : ""}></button>`;
      }
      return html;
    })
    .join("");
  return `<div class="timeline-track" data-frame-count="${model.frameCount}">${segments}</div>`;
}

function frameFromPointer(event, track) {
  const rect = track?.getBoundingClientRect?.();
  if (!rect || !rect.width) return null;
  const fraction = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
  return Math.round(fraction * Number(track.dataset.frameCount || 0));
}

export class SemanticCalibrationAdapter {
  constructor({
    api = null,
    postPending = null,
    onPending = null,
    onConfirm = null,
    onCancel = null,
    onTextPending = null,
    onComplete = null,
    onLockChange = null,
  } = {}) {
    this.api = api;
    this.postPending = postPending;
    this.onPending = onPending;
    this.onConfirm = onConfirm;
    this.onCancel = onCancel;
    this.onTextPending = onTextPending;
    this.onComplete = onComplete;
    this.onLockChange = onLockChange;
    this.task = null;
    this.root = null;
    this.model = null;
    this.preview = null;
    this.drag = null;
  }

  render(task, root = this.root) {
    this.task = task;
    this.root = root;
    const semantic = task?.semantic ?? task;
    this.model = buildTimelineModel(semantic);
    const pending = semantic?.pending_edit ?? task?.pending_edit ?? null;
    if (!root) return this.model;
    try {
      root.innerHTML = `<section class="semantic-calibration" data-pending="${pending ? "true" : "false"}">
      <div class="timeline-toolbar"><span class="timeline-caption">时间轴（帧）</span><span class="timeline-total">0–${displayEndFrame({ end_frame_exclusive: this.model.frameCount })}</span></div>
      <div class="timeline-host">${renderTimelineMarkup(this.model, { pending, preview: this.preview })}</div>
      <div class="semantic-pending-slot"></div>
      <div class="semantic-error" role="alert" aria-live="polite"></div>
      <div class="semantic-text-slot"></div>
      <div class="semantic-complete-row"><button type="button" class="semantic-complete" data-action="complete-semantic" data-mutation-control>完成语义校准</button></div>
    </section>`;
    } catch {
      return this.model;
    }
    this.bindHandles();
    this.renderPending(pending);
    this.renderTextEditor(pending);
    this.root.querySelector?.('[data-action="complete-semantic"]')?.addEventListener("click", () => {
      const result = this.complete();
      if (result?.catch) result.catch((error) => this.showError(error));
    });
    return this.model;
  }

  bindHandles() {
    const handles = this.root?.querySelectorAll?.(".boundary-handle") ?? [];
    handles.forEach((handle) => {
      handle.addEventListener("pointerdown", (event) => this.handlePointerDown(event, handle));
      handle.addEventListener("pointermove", (event) => this.handlePointerMove(event, handle));
      handle.addEventListener("pointerup", (event) => this.handlePointerUp(event, handle));
      handle.addEventListener("pointercancel", (event) => this.handlePointerUp(event, handle, true));
      handle.addEventListener("keydown", (event) => this.handleHandleKeydown(event, handle));
    });
  }

  handlePointerDown(event, handle) {
    if (this.isPending()) return;
    const boundaryIndex = Number(handle.dataset.boundaryIndex);
    const actorSegmentId = handle.dataset.actorSegmentId;
    const frameExclusive = Number(handle.dataset.frameExclusive);
    this.beginBoundaryDrag(boundaryIndex, actorSegmentId, frameExclusive);
    this.drag = { handle, boundaryIndex, actorSegmentId, frameExclusive };
    handle.setPointerCapture?.(event.pointerId);
    event.preventDefault?.();
  }

  handlePointerMove(event, handle) {
    if (!this.drag || this.drag.handle !== handle) return;
    const track = this.root?.querySelector?.(".timeline-track");
    let next = frameFromPointer(event, track);
    if (!Number.isInteger(next)) return;
    const boundary = this.model.handles.find((item) => item.boundary_index === this.drag.boundaryIndex);
    const previous = this.model.segments[this.drag.boundaryIndex - 1];
    const following = this.model.segments[this.drag.boundaryIndex];
    if (boundary && previous && following) {
      next = Math.max(previous.start_frame + 1, Math.min(following.end_frame_exclusive - 1, next));
    }
    this.drag.frameExclusive = next;
    handle.dataset.frameExclusive = String(next);
    handle.classList?.add("is-preview");
    handle.setAttribute?.("aria-valuenow", String(next));
    const previewLabel = this.root?.querySelector?.(".semantic-preview-frame");
    if (previewLabel) previewLabel.textContent = `预览边界 ${next}`;
    this.updatePreviewRanges(this.drag.boundaryIndex, next);
  }

  handlePointerUp(event, handle, cancelled = false) {
    if (!this.drag || this.drag.handle !== handle) return;
    handle.releasePointerCapture?.(event.pointerId);
    const drag = this.drag;
    this.drag = null;
    if (cancelled) {
      this.preview = null;
      this.render(this.task, this.root);
      return;
    }
    const payload = makeBoundaryPayload({
      boundaryIndex: drag.boundaryIndex,
      actorSegmentId: drag.actorSegmentId,
      frameExclusive: drag.frameExclusive,
    });
    const result = this.postPending?.(payload) ?? this.onPending?.(payload);
    if (result?.then) result.catch((error) => {
      this.preview = null;
      this.render(this.task, this.root);
      this.showError(error);
    });
  }

  handleHandleKeydown(event, handle) {
    if (this.isPending()) return;
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault?.();
    const boundaryIndex = Number(handle.dataset.boundaryIndex);
    const actorSegmentId = handle.dataset.actorSegmentId;
    const current = Number(handle.dataset.frameExclusive);
    const previous = this.model?.segments?.[boundaryIndex - 1];
    const following = this.model?.segments?.[boundaryIndex];
    if (!previous || !following) return;
    const next = Math.max(previous.start_frame + 1, Math.min(following.end_frame_exclusive - 1, current + (event.key === "ArrowRight" ? 1 : -1)));
    const payload = makeBoundaryPayload({ boundaryIndex, actorSegmentId, frameExclusive: next });
    const result = this.postPending?.(payload) ?? this.onPending?.(payload);
    if (result?.then) result.catch((error) => this.showError(error));
  }

  beginBoundaryDrag(boundaryIndex, actorSegmentId, frameExclusive) {
    const segmentCount = this.model?.segments?.length ?? 0;
    if (!Number.isInteger(boundaryIndex) || boundaryIndex < 1 || boundaryIndex >= segmentCount) {
      throw new Error("only an internal boundary can be dragged");
    }
    const adjacent = this.model.segments[boundaryIndex - 1]?.internal_id;
    const following = this.model.segments[boundaryIndex]?.internal_id;
    if (![adjacent, following].includes(actorSegmentId)) {
      throw new Error("actor segment must be adjacent to the internal boundary");
    }
    return makeBoundaryPayload({ boundaryIndex, actorSegmentId, frameExclusive });
  }

  beginTextEdit(segmentId, textCn, textEn) {
    if (this.isPending()) throw new Error("pending edit locks text editing");
    const payload = { segment_id: segmentId, text_cn: textCn, text_en: textEn };
    const result = this.onTextPending?.(payload);
    if (result?.then) result.catch((error) => this.showError(error));
    return payload;
  }

  renderTextEditor(pending) {
    const slot = this.root?.querySelector?.(".semantic-text-slot");
    if (!slot) return;
    const locked = Boolean(pending);
    const rows = this.model.segments.map((segment) => `<div class="text-editor-row" data-segment-id="${escapeHtml(segment.internal_id)}">
      <label><span>中文</span><input data-mutation-control data-text-cn="${escapeHtml(segment.internal_id)}" value="${escapeHtml(segment.text_cn)}" ${locked ? "disabled" : ""}></label>
      <label><span>English</span><input data-mutation-control data-text-en="${escapeHtml(segment.internal_id)}" value="${escapeHtml(segment.text_en)}" ${locked ? "disabled" : ""}></label>
      <button type="button" data-mutation-control data-action="save-text" ${locked ? "disabled" : ""}>暂存文字</button>
    </div>`).join("");
    slot.innerHTML = `<div class="text-editor-heading"><span>任务文字</span><span>修改后需确认</span></div>${rows}`;
    slot.querySelectorAll?.('[data-action="save-text"]').forEach((button) => {
      button.addEventListener("click", () => {
        const row = button.closest?.(".text-editor-row");
        const segmentId = row?.dataset.segmentId;
        const cn = row?.querySelector?.("[data-text-cn]")?.value ?? "";
        const en = row?.querySelector?.("[data-text-en]")?.value ?? "";
        if (segmentId) this.beginTextEdit(segmentId, cn, en);
      });
    });
  }

  updatePreviewRanges(boundaryIndex, frameExclusive) {
    const preview = linkedBoundaryPreview(this.model, boundaryIndex, frameExclusive);
    if (!preview) return;
    const segments = this.root?.querySelectorAll?.(".timeline-segment") ?? [];
    const previous = segments[boundaryIndex - 1];
    const following = segments[boundaryIndex];
    if (!previous || !following) return;
    previous.classList?.add("preview-affected");
    following.classList?.add("preview-affected");
    previous.style?.setProperty?.("--segment-width", `${preview.previous.widthPercent}%`);
    following.style?.setProperty?.("--segment-width", `${preview.following.widthPercent}%`);
    previous.dataset.endFrame = String(frameExclusive - 1);
    following.dataset.startFrame = String(frameExclusive);
    previous.querySelector?.(".segment-range") && (previous.querySelector(".segment-range").textContent = `${previous.dataset.startFrame}–${frameExclusive - 1}`);
    following.querySelector?.(".segment-range") && (following.querySelector(".segment-range").textContent = `${frameExclusive}–${following.dataset.endFrame}`);
  }

  renderPending(pending) {
    const slot = this.root?.querySelector?.(".semantic-pending-slot");
    if (!slot) return;
    if (!pending) {
      slot.innerHTML = "";
      this.onLockChange?.(false);
      return;
    }
    const view = pendingPresentation(pending);
    const rows = view.affectedSegmentIds.map((id, index) => {
      const before = view.before[index] ?? {};
      const after = view.after[index] ?? {};
      return `<div class="pending-row" data-segment-id="${escapeHtml(id)}"><span class="pending-segment-id">${escapeHtml(id)}</span><span class="pending-before">修改前：${before.start_frame}–${displayEndFrame(before)}</span><span class="pending-after">修改后：${after.start_frame}–${displayEndFrame(after)}</span></div>`;
    }).join("");
    slot.innerHTML = `<div class="semantic-pending" role="status"><div class="pending-title">待确认修改</div><div class="pending-rows">${rows}</div><span class="semantic-preview-frame" aria-live="polite"></span><div class="pending-actions"><button type="button" data-action="confirm-pending">确认并保存</button><button type="button" data-action="cancel-pending">取消本次修改</button></div></div>`;
    slot.querySelector?.('[data-action="confirm-pending"]')?.addEventListener("click", () => this.confirmPending());
    slot.querySelector?.('[data-action="cancel-pending"]')?.addEventListener("click", () => this.cancelPending());
    this.onLockChange?.(true);
  }

  isPending() {
    const semantic = this.task?.semantic ?? this.task;
    return Boolean(semantic?.pending_edit);
  }

  async confirmPending() {
    if (!this.isPending()) return null;
    return this.onConfirm?.(this.task?.semantic?.pending_edit ?? this.task?.pending_edit);
  }

  async cancelPending() {
    if (!this.isPending()) return null;
    return this.onCancel?.(this.task?.semantic?.pending_edit ?? this.task?.pending_edit);
  }

  async complete() {
    if (this.isPending()) throw new Error("pending edit must be confirmed or cancelled first");
    return this.onComplete?.();
  }

  showError(error) {
    const element = this.root?.querySelector?.(".semantic-error");
    if (element) element.textContent = String(error?.message ?? error);
  }
}

export default SemanticCalibrationAdapter;
