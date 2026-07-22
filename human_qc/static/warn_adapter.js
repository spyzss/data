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
  const exclusiveEnd = evidence.end_frame_exclusive
    ?? issue.end_frame_exclusive
    ?? nested.end_frame_exclusive
    ?? issue.end_frame
    ?? nested.end_frame
    ?? context.end_frame_exclusive;
  const end = Number.isInteger(exclusiveEnd)
    ? exclusiveEnd
    : (Number.isInteger(context.end_frame) ? context.end_frame + 1 : null);
  return {
    startFrame: Number.isInteger(start) ? start : null,
    endFrameExclusive: Number.isInteger(end) ? end : null,
  };
};

const displayValue = (value) => {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    return JSON.stringify(value);
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
    overlayImages: (Array.isArray(evidence.overlay_images) ? evidence.overlay_images : [])
      .filter((item) => item && typeof item.url === "string")
      .map((item) => ({
        frame: Number.isInteger(item.frame) ? item.frame : null,
        url: item.url,
      })),
    generationError: typeof evidence.generation_error === "string" ? evidence.generation_error : null,
    review: objectOrEmpty(task?.warn?.issue_reviews?.[selectedIssueId]),
  };
}

export function allSelectedIssuesReviewed(task) {
  const ids = selectedIds(task);
  const reviews = objectOrEmpty(task?.warn?.issue_reviews);
  return ids.length > 0 && ids.every((issueId) => ["pass", "fail"].includes(reviews[issueId]?.verdict));
}

export function nextReviewIssueId(task, currentIssueId = null) {
  const ids = selectedIds(task);
  const reviews = objectOrEmpty(task?.warn?.issue_reviews);
  const unresolved = ids.find((id) => !["pass", "fail"].includes(reviews[id]?.verdict));
  if (unresolved) return unresolved;
  return ids.includes(String(currentIssueId)) ? String(currentIssueId) : ids.at(-1) ?? null;
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
  const metrics = Object.entries(model.metrics).map(([name, value]) => `<div><dt>${escapeHtml(name)}</dt><dd>${escapeHtml(displayValue(value))}</dd></div>`).join("")
    || "<div><dt>指标</dt><dd>—</dd></div>";
  const evidenceMessage = {
    clip_unavailable: "问题片段暂不可用，正在使用原视频定位问题区间。",
    overlay_unavailable: "骨架 overlay 暂不可用，仍可使用问题视频完成判断。",
  }[model.generationError] ?? "";
  const overlays = model.overlayImages.length
    ? `<div class="warn-overlay-gallery" aria-label="SAM3 骨架抽样图">${model.overlayImages.map((item) => `<a class="warn-overlay-sample" href="${escapeHtml(item.url)}" target="_blank" rel="noopener"><img data-warn-overlay-sample src="${escapeHtml(item.url)}" alt="${escapeHtml(model.issueId)} 骨架抽样帧 ${escapeHtml(item.frame ?? "—")}"><span>帧 ${escapeHtml(item.frame ?? "—")}</span></a>`).join("")}</div>`
    : "";
  const legacyOverlay = !model.overlayImages.length && model.overlayUrl
    ? `<label class="warn-overlay-toggle"><input type="checkbox" data-action="toggle-overlay">显示单帧骨架图</label><img class="warn-overlay" data-warn-overlay src="${escapeHtml(model.overlayUrl)}" alt="${escapeHtml(model.issueId)} 骨架图" hidden>`
    : "";
  const clip = model.clipUrl
    ? `<a class="warn-clip-link" href="${escapeHtml(model.clipUrl)}" target="_blank" rel="noopener">在新窗口打开问题片段</a>`
    : '<span class="warn-clip-unavailable">问题片段暂不可用</span>';
  const previousReason = String(model.review.reason ?? "");
  const completeDisabled = allSelectedIssuesReviewed(task) ? "" : " disabled";
  return `<section class="warn-review" data-selected-issue-id="${escapeHtml(model.issueId)}">
    <header class="warn-review-head">
      <div class="warn-progress"><span>Warn 人工复核</span><strong>${reviewedCount} / ${ids.length}</strong></div>
      <nav class="warn-issue-list" aria-label="待复核问题">${issueButtons}</nav>
    </header>
    <section class="warn-rationale">
      <header><span class="warn-code">${escapeHtml(model.code)}</span><strong>${escapeHtml(model.issueId)}</strong></header>
      <dl class="warn-machine-fields">
        <div><dt>机器原因</dt><dd data-machine-reason>${escapeHtml(model.reason)}</dd></div>
        <div><dt>机器阈值</dt><dd data-machine-threshold>${escapeHtml(threshold)}</dd></div>
        <div><dt>证据窗口</dt><dd data-evidence-window>${escapeHtml(windowLabel)} <small>[start, end)</small></dd></div>
      </dl>
      <dl class="warn-metrics" data-machine-metrics>${metrics}</dl>
      <div class="warn-evidence">${clip}${evidenceMessage ? `<p class="warn-evidence-degraded" role="status">${escapeHtml(evidenceMessage)}</p>` : ""}${overlays}${legacyOverlay}<div class="warn-evidence-degraded" data-overlay-sample-error role="status" hidden></div><div class="warn-evidence-degraded" data-overlay-error role="status" hidden></div></div>
    </section>
    <section class="warn-decision">
        <label class="warn-reason"><span>人工判定原因（可选）</span><textarea data-review-reason>${escapeHtml(previousReason)}</textarea></label>
        <div class="warn-verdict-actions">
          <button type="button" data-action="verdict-pass" data-mutation-control>Pass</button>
          <button type="button" data-action="verdict-fail" data-mutation-control>Fail</button>
        </div>
      <div class="warn-error" role="alert" aria-live="polite"></div>
    </section>
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
    const preferred = task?.warn?.selected_issue_id;
    const currentIssueId = ids.includes(this.selectedIssueId)
      ? this.selectedIssueId
      : (ids.includes(preferred) ? preferred : ids[0] ?? null);
    const currentReview = objectOrEmpty(task?.warn?.issue_reviews?.[currentIssueId]);
    if (["pass", "fail"].includes(currentReview.verdict)) {
      this.selectedIssueId = nextReviewIssueId(task, currentIssueId);
    } else {
      this.selectedIssueId = currentIssueId;
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
    root.querySelectorAll?.("[data-warn-overlay-sample]").forEach((image) => {
      image.addEventListener("error", () => this.handleOverlaySampleError(image));
    });
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
    const fps = Number(this.task?.warn?.fps ?? this.task?.fps ?? 0);
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
      degraded.textContent = "骨架图片加载失败，仍可使用问题视频完成判断。";
    }
  }

  handleOverlaySampleError(image) {
    if (image) image.hidden = true;
    const sample = image?.closest?.(".warn-overlay-sample");
    if (sample) sample.hidden = true;
    const degraded = this.root?.querySelector?.("[data-overlay-sample-error]");
    if (degraded) {
      degraded.hidden = false;
      degraded.textContent = "部分骨架抽样图加载失败，仍可使用其他抽样图和问题视频完成判断。";
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
