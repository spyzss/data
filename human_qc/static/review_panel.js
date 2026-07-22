/**
 * Warn-review state and panel rendering.
 *
 * This module deliberately treats the Task 5 DTO as the only source of
 * review state.  Timeline and video interactions only move the source-frame
 * cursor; saved verdicts are changed exclusively through the canonical
 * verdict endpoint supplied by the application shell.
 */

const SAVED_VERDICTS = new Set(["pass", "fail"]);

const asIssues = (task) => (Array.isArray(task?.issues) ? task.issues : []);

const asFrameRange = (issue) => {
  const range = issue?.frame_range;
  const start = Number(range?.start_frame);
  const end = Number(range?.end_frame_exclusive);
  if (!Number.isInteger(start) || !Number.isInteger(end) || end <= start) return null;
  return { start, end };
};

const savedVerdict = (issue) => (
  SAVED_VERDICTS.has(issue?.review?.verdict) ? issue.review.verdict : null
);

const escapeHtml = (value) => String(value ?? "")
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")
  .replaceAll("'", "&#39;");

const thresholdTitle = (threshold) => {
  if (!threshold || typeof threshold !== "object") return "未提供阈值";
  const operator = typeof threshold.operator === "string" ? threshold.operator : "阈值";
  const value = threshold.value ?? threshold.boundary_value ?? threshold.threshold;
  return value === undefined || value === null ? `阈值：${operator}` : `阈值：${operator} ${value}`;
};

const issueLabel = (issue) => (
  typeof issue?.display_name === "string" && issue.display_name.trim()
    ? issue.display_name.trim()
    : "未命名问题"
);

const frameLabel = (issue) => {
  const range = asFrameRange(issue);
  return range ? `${range.start}–${range.end - 1}` : "—";
};

const selectedOptionMap = (options) => new Map(
  (Array.isArray(options) ? options : [])
    .filter((option) => option && typeof option.code === "string" && option.code.trim())
    .map((option) => [option.code, option]),
);

/**
 * Choose the warning a Pass/Fail should affect.
 *
 * A deliberate selection in the lower status row is allowed to reopen a
 * reviewed warning.  Without that selection, only pending warnings currently
 * active at the source-frame cursor participate, ordered by frame start and
 * the server-selected issue order for deterministic ties.
 */
export function nextDecisionTarget(task, activeIssueIds, explicitIssueId = null) {
  const issues = asIssues(task);
  const byId = new Map(issues.map((issue) => [issue?.id, issue]));
  const active = new Set(Array.isArray(activeIssueIds) ? activeIssueIds : []);
  if (
    typeof explicitIssueId === "string"
    && active.has(explicitIssueId)
    && byId.has(explicitIssueId)
  ) return explicitIssueId;

  return issues
    .map((issue, selectedIndex) => ({ issue, selectedIndex, range: asFrameRange(issue) }))
    .filter(({ issue, range }) => active.has(issue?.id) && range && savedVerdict(issue) === null)
    .sort((left, right) => (
      left.range.start - right.range.start || left.selectedIndex - right.selectedIndex
    ))[0]?.issue?.id ?? null;
}

/** Return the server-derived completion mode the UI is allowed to request. */
export function completionGate(task) {
  const issues = asIssues(task);
  if (issues.some((issue) => savedVerdict(issue) === "fail")) {
    return { enabled: true, mode: "early_fail" };
  }
  if (issues.length > 0 && issues.every((issue) => savedVerdict(issue) === "pass")) {
    return { enabled: true, mode: "all_reviewed" };
  }
  return { enabled: false, mode: null };
}

/**
 * Normalize a local-only manual failure-reason draft.  Only server-provided
 * options are retained, and their server order gives stable payload order.
 */
export function normalizeReasonDraft(options, selectedCodes, otherText = "") {
  const requested = new Set(
    selectedCodes instanceof Set
      ? selectedCodes
      : (Array.isArray(selectedCodes) ? selectedCodes : []),
  );
  const validOptions = Array.isArray(options) ? options : [];
  const reasonCodes = validOptions
    .filter((option) => option && typeof option.code === "string" && requested.has(option.code))
    .map((option) => option.code);
  const requiresText = validOptions.some((option) => (
    reasonCodes.includes(option.code) && option.requires_text === true
  ));
  const trimmed = typeof otherText === "string" ? otherText.trim() : "";
  return {
    reasonCodes,
    otherText: requiresText ? trimmed : null,
    hasManualReason: reasonCodes.length > 0,
    requiresText,
    valid: !requiresText || trimmed.length > 0,
  };
}

/** Immutable helper for DOM-free state tests and returned server task fixtures. */
export function applySavedReview(task, issueId, verdict) {
  if (!SAVED_VERDICTS.has(verdict)) throw new RangeError("verdict must be pass or fail");
  const issues = asIssues(task);
  if (!issues.some((issue) => issue?.id === issueId)) throw new Error("unknown issue");
  return {
    ...task,
    issues: issues.map((issue) => (
      issue.id === issueId ? { ...issue, review: { ...(issue.review ?? {}), verdict } } : issue
    )),
  };
}

export class ReviewPanel {
  constructor({
    documentRef = globalThis.document,
    onVerdict = null,
    onComplete = null,
    onSelectIssue = null,
    onRetryOverlay = null,
    canSubmitIssue = () => true,
  } = {}) {
    this.document = documentRef;
    this.onVerdict = typeof onVerdict === "function" ? onVerdict : async () => {};
    this.onComplete = typeof onComplete === "function" ? onComplete : async () => {};
    this.onSelectIssue = typeof onSelectIssue === "function" ? onSelectIssue : () => {};
    this.onRetryOverlay = typeof onRetryOverlay === "function" ? onRetryOverlay : async () => {};
    this.canSubmitIssue = typeof canSubmitIssue === "function" ? canSubmitIssue : () => true;
    this.root = null;
    this.task = null;
    this.currentFrame = 0;
    this.explicitIssueId = null;
    this.selectedReasonCodes = new Set();
    this.otherText = "";
    this.localError = "";
    this.isBusy = false;
    this.overlayAvailability = new Map();
    this.retryingOverlayIssues = new Set();
    this._bound = false;
    this._clickListener = (event) => this._handleClick(event);
    this._inputListener = (event) => this._handleInput(event);
  }

  mount(root) {
    const nextRoot = root ?? null;
    if (this.root && this.root !== nextRoot) this.destroy();
    this.root = nextRoot;
    if (!this.root) return this;
    if (!this._bound) {
      this.root.addEventListener?.("click", this._clickListener);
      this.root.addEventListener?.("input", this._inputListener);
      this._bound = true;
    }
    this.render();
    return this;
  }

  destroy() {
    if (this.root && this._bound) {
      this.root.removeEventListener?.("click", this._clickListener);
      this.root.removeEventListener?.("input", this._inputListener);
    }
    this._bound = false;
    this.root = null;
  }

  setTask(task) {
    this.task = task && typeof task === "object" ? task : null;
    if (this.explicitIssueId && !asIssues(this.task).some((issue) => issue.id === this.explicitIssueId)) {
      this.explicitIssueId = null;
    }
    this.render();
    return this.task;
  }

  /** Restore only the normalized, server-saved asset-level manual reason. */
  setDraftFromFailureReason(failureReason) {
    const options = Array.isArray(this.task?.reason_options) ? this.task.reason_options : [];
    const allowedCodes = selectedOptionMap(options);
    const requestedCodes = failureReason?.mode === "manual" && Array.isArray(failureReason.reason_codes)
      ? failureReason.reason_codes
      : [];
    this.selectedReasonCodes = new Set(
      requestedCodes.filter((code) => typeof code === "string" && allowedCodes.has(code)),
    );
    const normalized = normalizeReasonDraft(options, this.selectedReasonCodes, failureReason?.other_text);
    this.otherText = normalized.requiresText ? normalized.otherText : "";
    this.localError = "";
    this.render();
    return this.reasonDraft();
  }

  setCurrentFrame(frame) {
    this.currentFrame = Number.isFinite(Number(frame)) ? Math.max(0, Math.trunc(Number(frame))) : 0;
    this.render();
    return this.currentFrame;
  }

  setExplicitIssueId(issueId) {
    this.explicitIssueId = asIssues(this.task).some((issue) => issue.id === issueId) ? issueId : null;
    this.render();
    return this.explicitIssueId;
  }

  setBusy(busy) {
    this.isBusy = Boolean(busy);
    this.render();
    return this.isBusy;
  }

  setOverlayAvailability(issueId, available, code = null) {
    if (typeof issueId !== "string" || !issueId) return;
    this.overlayAvailability.set(issueId, { available: available === true, code });
    this.render();
  }

  clearOverlayAvailability() {
    this.overlayAvailability.clear();
    this.render();
  }

  setRetryPending(issueId, pending) {
    if (typeof issueId !== "string" || !issueId) return;
    if (pending) this.retryingOverlayIssues.add(issueId);
    else this.retryingOverlayIssues.delete(issueId);
    this.render();
  }

  /** Issue-local readiness guard.  It never changes the decision ordering. */
  canSubmit(issue) {
    if (!this.canSubmitIssue(issue)) return false;
    const overlay = issue?.overlay;
    if (!overlay || typeof overlay !== "object") return true;
    if (overlay.status !== "ready") return false;
    return this.overlayAvailability.get(issue.id)?.available !== false;
  }

  resetDraft() {
    this.selectedReasonCodes.clear();
    this.otherText = "";
    this.localError = "";
    this.render();
  }

  toggleReason(code) {
    const options = selectedOptionMap(this.task?.reason_options);
    if (!options.has(code)) return this.reasonDraft();
    if (this.selectedReasonCodes.has(code)) this.selectedReasonCodes.delete(code);
    else this.selectedReasonCodes.add(code);
    if (!this.selectedReasonCodes.has("other")) this.otherText = "";
    this.localError = "";
    this.render();
    return this.reasonDraft();
  }

  setOtherText(value) {
    this.otherText = typeof value === "string" ? value : "";
    this.localError = "";
    this.render();
    return this.reasonDraft();
  }

  reasonDraft() {
    return normalizeReasonDraft(this.task?.reason_options, this.selectedReasonCodes, this.otherText);
  }

  activeIssues() {
    return asIssues(this.task).filter((issue) => {
      const range = asFrameRange(issue);
      return range && range.start <= this.currentFrame && this.currentFrame < range.end;
    });
  }

  decisionTargetId() {
    return nextDecisionTarget(
      this.task,
      this.activeIssues().map((issue) => issue.id),
      this.explicitIssueId,
    );
  }

  failurePayloadFor(issueId) {
    const issue = asIssues(this.task).find((candidate) => candidate.id === issueId);
    if (!issue) throw new Error("unknown issue");
    const draft = this.reasonDraft();
    if (draft.requiresText && !draft.valid) {
      throw new Error("请选择“其他”后填写其他原因");
    }
    if (draft.hasManualReason) {
      return {
        failure_reason: {
          mode: "manual",
          reason_codes: draft.reasonCodes,
          other_text: draft.otherText,
        },
      };
    }
    return typeof issue.default_reason === "string" && issue.default_reason.trim()
      ? { reason: issue.default_reason.trim() }
      : {};
  }

  completionPayload() {
    const draft = this.reasonDraft();
    if (draft.requiresText && !draft.valid) {
      throw new Error("请选择“其他”后填写其他原因");
    }
    if (!draft.hasManualReason) return {};
    return {
      failure_reason: {
        mode: "manual",
        reason_codes: draft.reasonCodes,
        other_text: draft.otherText,
      },
    };
  }

  payloadForVerdict(issueId, verdict) {
    if (!SAVED_VERDICTS.has(verdict)) throw new RangeError("verdict must be pass or fail");
    if (verdict === "pass") return { verdict };
    return { verdict, ...this.failurePayloadFor(issueId) };
  }

  async submit(issueId, verdict) {
    if (this.isBusy) return false;
    try {
      const payload = this.payloadForVerdict(issueId, verdict);
      this.localError = "";
      await this.onVerdict(issueId, verdict, payload);
      return true;
    } catch (error) {
      this.localError = String(error?.message ?? error);
      this.render();
      return false;
    }
  }

  async complete() {
    if (this.isBusy) return false;
    try {
      this.localError = "";
      await this.onComplete();
      return true;
    } catch (error) {
      this.localError = String(error?.message ?? error);
      this.render();
      return false;
    }
  }

  render() {
    if (!this.root) return;
    const task = this.task;
    if (!task) {
      this.root.innerHTML = '<div class="review-empty">加载一个资产后开始 Warn 复核。</div>';
      return;
    }
    const active = this.activeIssues();
    const targetId = this.decisionTargetId();
    const target = asIssues(task).find((issue) => issue.id === targetId) ?? null;
    const readOnly = task?.lease?.read_only === true;
    const gate = completionGate(task);
    const draft = this.reasonDraft();
    const targetDisabled = readOnly || target === null || this.isBusy || !this.canSubmit(target);
    const reasonDisabled = readOnly || this.isBusy || (target === null && gate.mode !== "early_fail");
    const interval = active.length
      ? `${Math.min(...active.map((issue) => asFrameRange(issue).start))}–${Math.max(...active.map((issue) => asFrameRange(issue).end - 1))}`
      : "当前帧没有问题 Warn";
    const issueRows = active.length
      ? active.map((issue, activeIndex) => {
        const isTarget = issue.id === targetId;
        const verdict = savedVerdict(issue);
        const tooltipId = `threshold-${activeIndex}`;
        const status = verdict === "pass"
          ? '<strong class="saved-warning-status" data-review-status="pass">✓ 已通过</strong>'
          : verdict === "fail"
            ? '<strong class="saved-warning-status" data-review-status="fail">已失败</strong>'
            : isTarget
              ? '<span class="current-warning" data-review-status="pending">当前</span>'
              : '<span class="same-window" data-review-status="pending">同区间待复核</span>';
        return `<li class="active-warning${isTarget ? " is-target" : ""}" data-review-status="${verdict ?? "pending"}">
          <span>${escapeHtml(issueLabel(issue))}</span>
          <span class="threshold-anchor">
            <button type="button" class="threshold-question" aria-describedby="${tooltipId}" aria-label="${escapeHtml(issueLabel(issue))} 的阈值">?</button>
            <span id="${tooltipId}" role="tooltip" data-threshold-tooltip>${escapeHtml(thresholdTitle(issue.threshold))}</span>
          </span>
          ${status}
        </li>`;
      }).join("")
      : '<li class="active-warning-empty">拖动时间针或点击色块查看相应问题。</li>';
    const statusRows = asIssues(task).map((issue, selectedIndex) => {
      const verdict = savedVerdict(issue);
      const status = verdict === "pass" ? "已通过" : verdict === "fail" ? "已失败" : "待复核";
      const marker = verdict === "pass" ? '<span data-passed-marker aria-label="已通过">✓</span>' : "";
      const selected = issue.id === targetId;
      return `<button type="button" class="warning-status warning-status--${verdict ?? "pending"}${selected ? " is-selected" : ""}" data-action="select-status-issue" data-issue-id="${escapeHtml(issue.id)}" ${this.isBusy ? "disabled" : ""}>
        <span>${marker}${selectedIndex + 1} ${escapeHtml(issueLabel(issue))} · ${frameLabel(issue)} · ${status}</span>
      </button>`;
    }).join("");
    const chips = (Array.isArray(task.reason_options) ? task.reason_options : []).map((option) => {
      const code = option?.code;
      if (typeof code !== "string") return "";
      const selected = draft.reasonCodes.includes(code);
      const label = option.display_name ?? option.label ?? code;
      return `<button type="button" class="reason-chip${selected ? " is-selected" : ""}" data-action="toggle-reason" data-reason-code="${escapeHtml(code)}" aria-pressed="${selected ? "true" : "false"}" ${reasonDisabled ? "disabled" : ""}>${selected ? "✓ " : ""}${escapeHtml(label)}</button>`;
    }).join("");
    const otherField = draft.requiresText
      ? `<label class="other-reason-label">请填写其他原因（必填）
          <input data-reason-other type="text" required value="${escapeHtml(this.otherText)}" placeholder="请输入其他原因" ${reasonDisabled ? "disabled" : ""}>
        </label>`
      : "";
    const overlay = target?.overlay;
    const localOverlay = target ? this.overlayAvailability.get(target.id) : null;
    const overlayNotice = overlay && (
      overlay.status !== "ready" || localOverlay?.available === false
    )
      ? `<p class="overlay-state" data-overlay-state>${escapeHtml(this._overlayMessage(overlay, localOverlay))}</p>`
      : "";
    const retryButton = overlay?.status === "failed" && overlay.retryable === true
      ? `<button type="button" class="overlay-retry" data-action="retry-overlay" data-issue-id="${escapeHtml(target.id)}" ${this.retryingOverlayIssues.has(target.id) || readOnly || this.isBusy ? "disabled" : ""}>重试 Overlay</button>`
      : "";

    this.root.innerHTML = `<section class="warning-status-list" aria-label="Warning 复核状态">${statusRows}</section>
      <section class="active-warning-card" aria-label="当前问题">
        <h2>问题帧区间：${escapeHtml(interval)}</h2>
        <ul>${issueRows}</ul>
      </section>
      <section class="reason-section" aria-label="Fail 原因">
        <div class="reason-heading"><h2>Fail 原因</h2><span>可多选，可随时取消</span></div>
        <div class="reason-chips">${chips}</div>
        ${otherField}
        ${overlayNotice}
        ${retryButton}
      </section>
      <section class="verdict-actions" aria-label="判定操作">
        <button type="button" class="pass-button" data-action="verdict-pass" ${targetDisabled ? "disabled" : ""}>Pass</button>
        <button type="button" class="fail-button" data-action="verdict-fail" ${targetDisabled ? "disabled" : ""}>Fail</button>
      </section>
      <section class="completion-action">
        <button type="button" class="complete-button" data-action="complete-review" ${readOnly || !gate.enabled || this.isBusy ? "disabled" : ""}>完成复核</button>
        <span>${gate.enabled ? (gate.mode === "early_fail" ? "Fail 后可提前完成复核" : "全部通过后可完成复核") : "请先完成当前 Warn 判定"}</span>
      </section>
      <p class="panel-error" role="alert" ${this.localError ? "" : "hidden"}>${escapeHtml(this.localError)}</p>`;
  }

  _handleClick(event) {
    const actionElement = event?.target?.closest?.("[data-action]");
    if (!actionElement || !this.root?.contains?.(actionElement)) return;
    const action = actionElement.dataset.action;
    if (this.isBusy) return;
    if (action === "toggle-reason") {
      this.toggleReason(actionElement.dataset.reasonCode);
      return;
    }
    if (action === "select-status-issue") {
      const selected = this.setExplicitIssueId(actionElement.dataset.issueId);
      this.onSelectIssue(selected);
      return;
    }
    if (action === "retry-overlay") {
      const issueId = actionElement.dataset.issueId;
      if (issueId) this.onRetryOverlay(issueId);
      return;
    }
    if (action === "verdict-pass" || action === "verdict-fail") {
      const issueId = this.decisionTargetId();
      if (issueId) this.submit(issueId, action === "verdict-pass" ? "pass" : "fail");
      return;
    }
    if (action === "complete-review") this.complete();
  }

  _handleInput(event) {
    if (!this.isBusy && event?.target?.matches?.("[data-reason-other]")) this.setOtherText(event.target.value);
  }

  _overlayMessage(overlay, local) {
    if (overlay?.status === "pending") return "Overlay 已排队，当前 Warning 暂不可判定。";
    if (overlay?.status === "generating") return "Overlay 生成中，当前 Warning 暂不可判定。";
    if (overlay?.status === "failed") {
      return overlay.retryable === true
        ? "Overlay 暂不可用，其他 Warning 可继续复核。"
        : "Overlay 无法生成，请联系管理员。";
    }
    if (local?.available === false) return "Overlay 加载中，当前 Warning 暂不可判定。";
    return "当前 Warning 的 Overlay 暂不可用。";
  }
}

export default ReviewPanel;
