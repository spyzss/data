/**
 * Warning-review adapter for the framework-free human QC workbench.
 *
 * Machine issue fields and generated evidence are read-only.  This adapter
 * owns only issue selection, the reviewer reason/verdict controls, and the
 * final completion gate.
 */

const escapeHtml = (value) => String(value ?? "")
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")
  .replaceAll("'", "&#39;");

const objectOrEmpty = (value) => value && typeof value === "object" && !Array.isArray(value) ? value : {};

const selectedIds = (task) => {
  const ids = task?.warn?.selected_issue_ids;
  return Array.isArray(ids) ? ids.map(String) : [];
};

const issueMap = (task) => {
  const warn = objectOrEmpty(task?.warn);
  const selected = objectOrEmpty(warn.selected_issues);
  if (Object.keys(selected).length) return selected;
  const rows = Array.isArray(warn.issues) ? warn.issues : [];
  return Object.fromEntries(rows
    .filter((row) => row && typeof row.issue_id === "string")
    .map((row) => [row.issue_id, row]));
};

const evidenceFor = (task, issueId) => {
  const rows = Array.isArray(task?.evidence) ? task.evidence : [];
  return objectOrEmpty(rows.find((row) => row?.issue_id === issueId));
};

const issueMetrics = (issue) => {
  if (issue.metrics && typeof issue.metrics === "object" && !Array.isArray(issue.metrics)) {
    return issue.metrics;
  }
  if (typeof issue.metric === "string" && issue.metric) {
    return { [issue.metric]: issue.observed_value ?? null };
  }
  const context = objectOrEmpty(issue.context);
  return objectOrEmpty(context.trigger_metrics ?? context.metrics);
};

const issueThreshold = (issue) => {
  const explicit = objectOrEmpty(issue.threshold);
  if (Object.keys(explicit).length) {
    return {
      operator: explicit.operator ?? issue.operator ?? "",
      value: explicit.value ?? explicit.boundary_value ?? issue.boundary_value ?? null,
    };
  }
  return { operator: issue.operator ?? "", value: issue.boundary_value ?? null };
};

const frameWindow = (issue, evidence) => {
  const context = objectOrEmpty(issue.context);
  const nested = objectOrEmpty(issue.window);
  const start = evidence.start_frame ?? issue.start_frame ?? nested.start_frame ?? context.start_frame;
  const end = evidence.end_frame_exclusive
    ?? issue.end_frame_exclusive
    ?? nested.end_frame_exclusive
    ?? issue.end_frame
    ?? nested.end_frame
    ?? context.end_frame_exclusive
    ?? context.end_frame;
  return {
    startFrame: Number.isInteger(start) ? start : null,
    endFrameExclusive: Number.isInteger(end) ? end : null,
  };
};

const jsonText = (value) => {
  if (!value || !Object.keys(value).length) return "—";
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
};

export function buildWarnIssueModel(task, issueId) {
  const ids = selectedIds(task);
  const selectedIssueId = issueId ?? task?.warn?.selected_issue_id ?? ids[0] ?? null;
  if (!selectedIssueId || !ids.includes(String(selectedIssueId))) {
    throw new Error("issue must be selected for human review");
  }
  const issue = objectOrEmpty(issueMap(task)[selectedIssueId]);
  const evidence = evidenceFor(task, selectedIssueId);
  const context = objectOrEmpty(issue.context);
  return {
    issueId: String(selectedIssueId),
    code: String(issue.code ?? issue.issue_type ?? selectedIssueId),
    reason: String(issue.reason ?? issue.message ?? context.reason ?? issue.code ?? "—"),
    metrics: issueMetrics(issue),
    threshold: issueThreshold(issue),
    window: frameWindow(issue, evidence),
    clipUrl: typeof evidence.clip_url === "string" ? evidence.clip_url : null,
    overlayUrl: typeof evidence.overlay_url === "string" ? evidence.overlay_url : null,
    generationError: typeof evidence.generation_error === "string" ? evidence.generation_error : null,
    review: objectOrEmpty(task?.warn?.issue_reviews?.[selectedIssueId]),
  };
}

export function allSelectedIssuesReviewed(task) {
  const ids = selectedIds(task);
  const reviews = objectOrEmpty(task?.warn?.issue_reviews);
  return ids.length > 0 && ids.every((issueId) => ["pass", "fail"].includes(reviews[issueId]?.verdict));
}

export function renderWarnMarkup(task, issueId = null) {
  const ids = selectedIds(task);
  if (!ids.length) {
    const candidates = Array.isArray(task?.warn?.candidate_issue_ids)
      ? task.warn.candidate_issue_ids
      : [];
    if (candidates.length) {
      return `<div class="empty-stage" data-warn-queued><span>${candidates.length} 个 Warn 候选，等待选择人工复核项。</span></div>`;
    }
    return '<div class="empty-stage" data-warn-empty>没有需要人工复核的 Warn。</div>';
  }
  const model = buildWarnIssueModel(task, issueId);
  const reviews = objectOrEmpty(task?.warn?.issue_reviews);
  const reviewedCount = ids.filter((id) => ["pass", "fail"].includes(reviews[id]?.verdict)).length;
  const issueButtons = ids.map((id, index) => {
    const issue = objectOrEmpty(issueMap(task)[id]);
    const verdict = reviews[id]?.verdict;
    const active = id === model.issueId;
    return `<button type="button" class="warn-issue${active ? " is-active" : ""}" data-action="select-issue" data-issue-id="${escapeHtml(id)}" aria-current="${active ? "true" : "false"}"><span>${index + 1}. ${escapeHtml(issue.code ?? id)}</span><strong>${escapeHtml(verdict ?? "待复核")}</strong></button>`;
  }).join("");
  const threshold = `${model.threshold.operator || ""} ${model.threshold.value ?? "—"}`.trim();
  const windowLabel = Number.isInteger(model.window.startFrame) && Number.isInteger(model.window.endFrameExclusive)
    ? `${model.window.startFrame}–${model.window.endFrameExclusive - 1}`
    : "—";
  const overlay = model.overlayUrl
    ? `<label class="warn-overlay-toggle"><input type="checkbox" data-action="toggle-overlay">显示骨架 overlay</label><img class="warn-overlay" data-warn-overlay src="${escapeHtml(model.overlayUrl)}" alt="${escapeHtml(model.issueId)} 骨架 overlay" hidden>`
    : '<label class="warn-overlay-toggle"><input type="checkbox" data-action="toggle-overlay" disabled>没有可用 overlay</label>';
  const degradationMessage = model.generationError
    ? `overlay 生成失败，已保留原始视频：${model.generationError}`
    : "";
  const degradation = `<div class="warn-evidence-degraded" data-overlay-error role="status"${degradationMessage ? "" : " hidden"}>${escapeHtml(degradationMessage)}</div>`;
  const clip = model.clipUrl
    ? `<a class="warn-clip-link" href="${escapeHtml(model.clipUrl)}" target="_blank" rel="noopener">打开问题窗口视频</a>`
    : '<span class="warn-clip-unavailable">问题窗口视频暂不可用</span>';
  const previousReason = String(model.review.reason ?? "");
  const completeDisabled = allSelectedIssuesReviewed(task) ? "" : " disabled";
  return `<section class="warn-review" data-selected-issue-id="${escapeHtml(model.issueId)}">
    <div class="warn-progress"><span>Warn 人工复核</span><strong>${reviewedCount} / ${ids.length}</strong></div>
    <div class="warn-layout">
      <nav class="warn-issue-list" aria-label="待复核问题">${issueButtons}</nav>
      <article class="warn-issue-detail">
        <header><span class="warn-code">${escapeHtml(model.code)}</span><strong>${escapeHtml(model.issueId)}</strong></header>
        <dl class="warn-machine-fields">
          <div><dt>机器原因</dt><dd data-machine-reason>${escapeHtml(model.reason)}</dd></div>
          <div><dt>机器指标</dt><dd><pre data-machine-metrics>${escapeHtml(jsonText(model.metrics))}</pre></dd></div>
          <div><dt>机器阈值</dt><dd data-machine-threshold>${escapeHtml(threshold)}</dd></div>
          <div><dt>证据窗口</dt><dd data-evidence-window>${escapeHtml(windowLabel)} <small>[start, end)</small></dd></div>
        </dl>
        <div class="warn-evidence">${clip}${overlay}${degradation}</div>
        <label class="warn-reason"><span>人工判定原因（可选）</span><textarea data-review-reason>${escapeHtml(previousReason)}</textarea></label>
        <div class="warn-verdict-actions">
          <button type="button" data-action="verdict-pass" data-mutation-control>Pass</button>
          <button type="button" data-action="verdict-fail" data-mutation-control>Fail</button>
        </div>
      </article>
    </div>
    <div class="warn-error" role="alert" aria-live="polite"></div>
    <div class="warn-complete-row"><button type="button" data-action="complete-warn" data-mutation-control${completeDisabled}>完成 Warn 复核</button></div>
  </section>`;
}

export class WarnReviewAdapter {
  constructor({ onVerdict = null, onComplete = null, video = null, videoPlaceholder = null } = {}) {
    this.onVerdict = onVerdict;
    this.onComplete = onComplete;
    this.video = video;
    this.videoPlaceholder = videoPlaceholder;
    this.task = null;
    this.root = null;
    this.selectedIssueId = null;
    this.boundTimeUpdate = null;
  }

  render(task, root = this.root) {
    this.task = task;
    this.root = root;
    const ids = selectedIds(task);
    if (!ids.includes(this.selectedIssueId)) {
      const preferred = task?.warn?.selected_issue_id;
      this.selectedIssueId = ids.includes(preferred) ? preferred : ids[0] ?? null;
    }
    const markup = renderWarnMarkup(task, this.selectedIssueId);
    if (!root) return this.selectedIssueId ? buildWarnIssueModel(task, this.selectedIssueId) : null;
    root.innerHTML = markup;
    if (!this.selectedIssueId) return null;
    root.querySelectorAll?.('[data-action="select-issue"]').forEach((button) => {
      button.addEventListener("click", () => this.selectIssue(button.dataset.issueId));
    });
    root.querySelector?.('[data-action="toggle-overlay"]')?.addEventListener("change", (event) => {
      const overlay = root.querySelector?.("[data-warn-overlay]");
      if (overlay) overlay.hidden = !event.currentTarget.checked;
    });
    root.querySelector?.("[data-warn-overlay]")?.addEventListener("error", () => this.handleOverlayError());
    root.querySelector?.('[data-action="verdict-pass"]')?.addEventListener("click", () => this.submitCurrentVerdict("pass"));
    root.querySelector?.('[data-action="verdict-fail"]')?.addEventListener("click", () => this.submitCurrentVerdict("fail"));
    root.querySelector?.('[data-action="complete-warn"]')?.addEventListener("click", () => {
      const result = this.complete();
      if (result?.catch) result.catch((error) => this.showError(error));
    });
    const model = buildWarnIssueModel(task, this.selectedIssueId);
    this.configureVideo(model);
    return model;
  }

  selectIssue(issueId) {
    if (!selectedIds(this.task).includes(String(issueId))) {
      throw new Error("issue must be selected for human review");
    }
    this.selectedIssueId = String(issueId);
    return this.render(this.task, this.root);
  }

  async submitCurrentVerdict(verdict) {
    const reason = this.root?.querySelector?.("[data-review-reason]")?.value ?? "";
    try {
      return await this.submitVerdict(this.selectedIssueId, verdict, reason);
    } catch (error) {
      this.showError(error);
      return null;
    }
  }

  async submitVerdict(issueId, verdict, reason = "") {
    if (!selectedIds(this.task).includes(String(issueId))) {
      throw new Error("issue must be selected for human review");
    }
    if (!["pass", "fail"].includes(verdict)) throw new Error("verdict must be pass or fail");
    return this.onVerdict?.(String(issueId), verdict, String(reason ?? ""));
  }

  async complete() {
    if (!allSelectedIssuesReviewed(this.task)) {
      throw new Error("all selected issues require a verdict before completion");
    }
    return this.onComplete?.();
  }

  configureVideo(model) {
    const video = this.video;
    if (!video || !model) return;
    if (this.boundTimeUpdate) video.removeEventListener?.("timeupdate", this.boundTimeUpdate);
    const fps = Number(this.task?.warn?.fps ?? this.task?.semantic?.timeline?.fps ?? this.task?.fps ?? 0);
    const startFrame = model.window.startFrame;
    const endFrame = model.window.endFrameExclusive;
    if (model.clipUrl) {
      video.src = model.clipUrl;
      video.currentTime = 0;
      if (this.videoPlaceholder) this.videoPlaceholder.hidden = true;
    } else {
      video.pause?.();
      if (typeof video.removeAttribute === "function") video.removeAttribute("src");
      else video.src = "";
      video.load?.();
      video.currentTime = 0;
      if (this.videoPlaceholder) this.videoPlaceholder.hidden = false;
    }
    video.dataset && (video.dataset.windowStartFrame = String(startFrame ?? ""));
    video.dataset && (video.dataset.windowEndFrameExclusive = String(endFrame ?? ""));
    if (!(fps > 0) || !Number.isInteger(startFrame) || !Number.isInteger(endFrame)) return;
    const endSeconds = model.clipUrl ? (endFrame - startFrame) / fps : endFrame / fps;
    this.boundTimeUpdate = () => {
      if (video.currentTime >= endSeconds) {
        video.pause?.();
        video.currentTime = model.clipUrl ? 0 : startFrame / fps;
      }
    };
    video.addEventListener?.("timeupdate", this.boundTimeUpdate);
  }

  handleOverlayError() {
    const overlay = this.root?.querySelector?.("[data-warn-overlay]");
    const toggle = this.root?.querySelector?.('[data-action="toggle-overlay"]');
    const degraded = this.root?.querySelector?.("[data-overlay-error]");
    if (overlay) overlay.hidden = true;
    if (toggle) {
      toggle.checked = false;
      toggle.disabled = true;
    }
    if (degraded) {
      degraded.hidden = false;
      degraded.textContent = "overlay 图片加载失败，已保留原始视频证据。";
    }
  }

  showError(error) {
    const element = this.root?.querySelector?.(".warn-error");
    if (element) element.textContent = String(error?.message ?? error);
  }

  destroy() {
    if (this.video && this.boundTimeUpdate) {
      this.video.removeEventListener?.("timeupdate", this.boundTimeUpdate);
    }
    this.boundTimeUpdate = null;
  }
}

export default WarnReviewAdapter;
