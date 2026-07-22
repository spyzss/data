/**
 * Canonical Warn-only application shell.
 *
 * The transport boundary is intentionally narrow: Task 5 owns the DTO and
 * lease lifecycle, while this browser shell only consumes the safe `/api/warn`
 * projection.  A timeline seek cannot mutate review state; verdict mutations
 * always carry the revision and lease token returned by the current task.
 */

import { ReviewPanel, completionGate, nextDecisionTarget } from "./review_panel.js";
import { VideoController } from "./video_controller.js";
import { WarningTimeline } from "./warning_timeline.js";

const isObject = (value) => value !== null && typeof value === "object" && !Array.isArray(value);

const clampFrame = (frame, totalFrames) => {
  const total = Number(totalFrames);
  if (!Number.isInteger(total) || total <= 0) return 0;
  const requested = Number(frame);
  const normalized = Number.isFinite(requested) ? Math.trunc(requested) : 0;
  return Math.min(Math.max(normalized, 0), total - 1);
};

const responseError = (response, body) => {
  const detail = isObject(body?.error) ? body.error : {};
  const error = new Error(
    typeof detail.message === "string" && detail.message ? detail.message : `request failed (${response.status})`,
  );
  error.code = typeof detail.code === "string" ? detail.code : "request_failed";
  error.status = Number(response.status) || 0;
  return error;
};

const taskFromEnvelope = (value) => (isObject(value?.task) ? value.task : value);

const canonicalTask = (task) => (
  isObject(task)
  && typeof task.asset_id === "string"
  && Number.isInteger(task.report_revision)
  && isObject(task.video)
  && Array.isArray(task.issues)
  && isObject(task.lease)
);

const issueRange = (issue) => {
  const start = Number(issue?.frame_range?.start_frame);
  const end = Number(issue?.frame_range?.end_frame_exclusive);
  return Number.isInteger(start) && Number.isInteger(end) && end > start ? { start, end } : null;
};

const savedVerdict = (issue) => (
  issue?.review?.verdict === "pass" || issue?.review?.verdict === "fail"
    ? issue.review.verdict
    : null
);

const earliestPendingIssue = (task) => (Array.isArray(task?.issues) ? task.issues : [])
  .map((issue, selectedIndex) => ({ issue, selectedIndex, range: issueRange(issue) }))
  .filter(({ issue, range }) => range && savedVerdict(issue) === null)
  .sort((left, right) => left.range.start - right.range.start || left.selectedIndex - right.selectedIndex)[0]?.issue ?? null;

export function activeIssueIdsAtFrame(task, frame) {
  const target = Number.isFinite(Number(frame)) ? Math.trunc(Number(frame)) : 0;
  return (Array.isArray(task?.issues) ? task.issues : [])
    .filter((issue) => {
      const range = issueRange(issue);
      return range && range.start <= target && target < range.end;
    })
    .map((issue) => issue.id);
}

export class WarnReviewApp {
  constructor({
    baseUrl = "",
    fetcher = globalThis.fetch?.bind(globalThis),
    documentRef = globalThis.document,
    root = null,
    panelFactory = (options) => new ReviewPanel(options),
    timelineFactory = (options) => new WarningTimeline(options),
    videoControllerFactory = (options) => new VideoController(options),
  } = {}) {
    this.baseUrl = String(baseUrl).replace(/\/$/, "");
    this.fetcher = fetcher;
    this.document = documentRef;
    this.root = root;
    this.panelFactory = panelFactory;
    this.timelineFactory = timelineFactory;
    this.videoControllerFactory = videoControllerFactory;
    this.task = null;
    this.assetId = null;
    this.assets = [];
    this.currentFrame = 0;
    this.explicitIssueId = null;
    this.lastError = null;
    this.videoController = null;
    this.timeline = null;
    this.panel = this.panelFactory({
      documentRef: this.document,
      onVerdict: (issueId, verdict, payload) => this.submitVerdict(issueId, verdict, payload),
      onComplete: () => this.completeReview(),
      onSelectIssue: (issueId) => this._acceptStatusSelection(issueId),
    });
    this._mediaKey = null;
    this._chromeBound = false;
  }

  endpoint(path) {
    return `${this.baseUrl}${path}`;
  }

  async requestJson(path, { method = "GET", body = undefined } = {}) {
    if (typeof this.fetcher !== "function") throw new Error("fetch is not available");
    const options = { method, headers: { Accept: "application/json" } };
    if (body !== undefined) {
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(body);
    }
    let response;
    try {
      response = await this.fetcher(this.endpoint(path), options);
    } catch (error) {
      this._setError("network_error", String(error?.message ?? error), 0);
      throw error;
    }
    let value = null;
    try {
      value = await response.json();
    } catch {
      value = null;
    }
    if (!response.ok) {
      const error = responseError(response, value);
      this._setError(error.code, error.message, error.status);
      // A 409/423 response is only an error envelope.  Never replace a safe
      // local server snapshot with one while the operator is looking at it.
      throw error;
    }
    this.lastError = null;
    return value;
  }

  async listAssets() {
    const value = await this.requestJson("/api/warn/assets");
    const assets = Array.isArray(value?.assets)
      ? value.assets.filter((assetId) => typeof assetId === "string" && assetId)
      : [];
    this.assets = [...new Set(assets)];
    this.render();
    return this.assets;
  }

  async requestTask(assetId) {
    const value = await this.requestJson(`/api/warn/assets/${encodeURIComponent(assetId)}/task`);
    const task = taskFromEnvelope(value);
    this.applyServerTask(task);
    return task;
  }

  /** Direct asset switching intentionally discards only local unsaved drafts. */
  async loadAsset(assetId) {
    if (typeof assetId !== "string" || !assetId) throw new Error("assetId is required");
    this.resetDraft();
    return this.requestTask(assetId);
  }

  async start() {
    const assets = await this.listAssets();
    if (assets.length) await this.loadAsset(assets[0]);
    return this.task;
  }

  applyServerTask(task) {
    if (!canonicalTask(task)) throw new TypeError("server task must be a canonical Warn task DTO");
    this.task = task;
    this.assetId = task.asset_id;
    if (!Array.isArray(this.assets) || !this.assets.includes(this.assetId)) {
      this.assets = [...this.assets, this.assetId];
    }
    if (!task.issues.some((issue) => issue.id === this.explicitIssueId)) this.explicitIssueId = null;
    this._configureMedia(task);
    this._configureTimeline(task);
    this.panel.setTask(task);
    this.panel.setExplicitIssueId(this.explicitIssueId);
    this.panel.setCurrentFrame(this.currentFrame);
    this.render();
    return task;
  }

  setCurrentFrame(frame, { fromVideo = false } = {}) {
    const totalFrames = this.task?.video?.total_frames;
    this.currentFrame = clampFrame(frame, totalFrames);
    if (!fromVideo) this.videoController?.seekToFrame(this.currentFrame);
    this.timeline?.setCurrentFrame(this.currentFrame);
    this.panel.setCurrentFrame(this.currentFrame);
    this._renderVideoState();
    return this.currentFrame;
  }

  /** Timeline and popover calls are seek-only by contract. */
  seekFromTimeline(frame) {
    return this.setCurrentFrame(frame);
  }

  _acceptStatusSelection(issueId) {
    this.explicitIssueId = this.task?.issues?.some((issue) => issue.id === issueId) ? issueId : null;
    this.panel.setExplicitIssueId(this.explicitIssueId);
    return this.explicitIssueId;
  }

  selectIssueFromStatus(issueId) {
    return this._acceptStatusSelection(issueId);
  }

  defaultDecisionTarget() {
    return nextDecisionTarget(
      this.task,
      activeIssueIdsAtFrame(this.task, this.currentFrame),
      this.explicitIssueId,
    );
  }

  revision() {
    return Number(this.task?.report_revision ?? 0);
  }

  mutationBody(payload) {
    const lease = this.task?.lease;
    if (lease?.read_only === true || typeof lease?.token !== "string" || !lease.token) {
      throw new Error("当前任务为只读，不能提交判定");
    }
    return {
      expected_revision: this.revision(),
      lease_token: lease.token,
      ...payload,
    };
  }

  async submitDefaultVerdict(verdict) {
    const issueId = this.defaultDecisionTarget();
    if (!issueId) throw new Error("当前帧没有待复核的 Warn");
    return this.submitVerdict(issueId, verdict);
  }

  async submitVerdict(issueId, verdict, suppliedPayload = null) {
    if (!this.task || !this.assetId) throw new Error("尚未加载 Warn 任务");
    const target = this.task.issues.find((issue) => issue.id === issueId);
    if (!target) throw new Error("当前 Warn 不存在");
    const payload = suppliedPayload ?? this.panel.payloadForVerdict(issueId, verdict);
    const value = await this.requestJson(
      `/api/warn/assets/${encodeURIComponent(this.assetId)}/issues/${encodeURIComponent(issueId)}/verdict`,
      { method: "POST", body: this.mutationBody(payload) },
    );
    if (verdict === "pass") this.explicitIssueId = null;
    this.applyServerTask(taskFromEnvelope(value));
    // Pass should expose the next pending active warning after the returned
    // snapshot is consumed.  Moving the source cursor—not an explicit status
    // selection—to that warning's start preserves the timeline's seek-only
    // semantics.  Fail remains inspectable until the operator explicitly
    // presses 完成复核.
    if (verdict === "pass") {
      const next = earliestPendingIssue(this.task);
      const range = issueRange(next);
      if (range) this.setCurrentFrame(range.start);
    }
    this.panel.setExplicitIssueId(this.explicitIssueId);
    this.panel.resetDraft();
    this.render();
    return this.task;
  }

  async completeReview() {
    if (!this.task || !this.assetId) throw new Error("尚未加载 Warn 任务");
    const gate = completionGate(this.task);
    if (!gate.enabled || !gate.mode) throw new Error("请先完成当前 Warn 判定");
    const payload = { completion_mode: gate.mode };
    const failureReason = this.task.failure_reason;
    if (gate.mode === "early_fail" && isObject(failureReason) && Array.isArray(failureReason.reason_codes)) {
      payload.failure_reason = {
        reason_codes: failureReason.reason_codes,
        other_text: typeof failureReason.other_text === "string" ? failureReason.other_text : null,
      };
    }
    const value = await this.requestJson(
      `/api/warn/assets/${encodeURIComponent(this.assetId)}/complete`,
      { method: "POST", body: this.mutationBody(payload) },
    );
    this.applyServerTask(taskFromEnvelope(value));
    this.resetDraft();
    await this.loadNextAsset();
    return this.task;
  }

  async loadNextAsset() {
    // Navigation never carries an unsaved local reason draft across assets,
    // including while the lightweight asset-list request is in flight.
    this.resetDraft();
    const assets = await this.listAssets();
    const currentIndex = assets.indexOf(this.assetId);
    const next = currentIndex >= 0 ? assets[currentIndex + 1] : assets[0];
    if (!next) return null;
    return this.loadAsset(next);
  }

  async loadPreviousAsset() {
    this.resetDraft();
    const assets = await this.listAssets();
    const currentIndex = assets.indexOf(this.assetId);
    const previous = currentIndex > 0 ? assets[currentIndex - 1] : null;
    if (!previous) return null;
    return this.loadAsset(previous);
  }

  resetDraft() {
    this.explicitIssueId = null;
    this.panel.setExplicitIssueId(null);
    this.panel.resetDraft();
  }

  mount(root = this.root ?? this.document?.querySelector?.("#app")) {
    this.root = root ?? null;
    if (!this.root) return this;
    const video = this.root.querySelector?.("[data-video]");
    const videoRoot = this.root.querySelector?.("[data-video-root]");
    const panelRoot = this.root.querySelector?.("[data-review-panel]");
    const timelineRoot = this.root.querySelector?.("[data-warning-timeline]");
    if (video && videoRoot && !this.videoController) {
      this.videoController = this.videoControllerFactory({
        video,
        root: videoRoot,
        documentRef: this.document,
        onFrameChange: (frame) => this.setCurrentFrame(frame, { fromVideo: true }),
        onPlaybackStateChange: () => this._renderVideoState(),
      });
    }
    if (panelRoot) this.panel.mount(panelRoot);
    if (timelineRoot && this.task) this._configureTimeline(this.task);
    this._bindChrome();
    this.render();
    return this;
  }

  render() {
    if (!this.root) return;
    const setText = (selector, text) => {
      const element = this.root.querySelector?.(selector);
      if (element) element.textContent = text;
    };
    setText("[data-asset-id]", this.assetId ?? "未加载资产");
    setText("[data-revision]", this.task ? `revision ${this.revision()}` : "revision —");
    const error = this.root.querySelector?.("[data-save-error]");
    if (error) {
      error.textContent = this.lastError?.message ?? "";
      error.hidden = !this.lastError;
    }
    const previous = this.root.querySelector?.('[data-action="previous-asset"]');
    const next = this.root.querySelector?.('[data-action="next-asset"]');
    const currentIndex = this.assets.indexOf(this.assetId);
    if (previous) previous.disabled = currentIndex <= 0;
    if (next) next.disabled = currentIndex < 0 || currentIndex >= this.assets.length - 1;
    this._renderVideoState();
  }

  _setError(code, message, status) {
    this.lastError = { code, message, status };
    this.render();
  }

  _configureMedia(task) {
    if (!this.videoController || !task?.video) return;
    const mediaKey = `${task.asset_id}|${task.video.url}|${task.video.fps}|${task.video.total_frames}`;
    if (mediaKey === this._mediaKey) return;
    this._mediaKey = mediaKey;
    this.currentFrame = 0;
    this.videoController.setMedia(task.video);
    const placeholder = this.root?.querySelector?.("[data-video-placeholder]");
    if (placeholder) placeholder.hidden = true;
  }

  _configureTimeline(task) {
    const root = this.root?.querySelector?.("[data-warning-timeline]");
    const totalFrames = task?.video?.total_frames;
    if (!root || !Number.isInteger(totalFrames) || totalFrames <= 0) return;
    const options = {
      warnings: task.issues,
      totalFrames,
      currentFrame: this.currentFrame,
      onSeek: (frame) => this.seekFromTimeline(frame),
      documentRef: this.document,
    };
    if (!this.timeline) {
      this.timeline = this.timelineFactory(options);
      this.timeline.mount(root);
    } else {
      this.timeline.setWarnings(task.issues, totalFrames);
      this.timeline.setCurrentFrame(this.currentFrame);
    }
  }

  _renderVideoState() {
    if (!this.root) return;
    const rate = this.videoController?.playbackRate ?? 1;
    const frame = this.task ? String(this.currentFrame) : "—";
    const rateElement = this.root.querySelector?.("[data-playback-rate]");
    const frameElement = this.root.querySelector?.("[data-current-frame]");
    if (rateElement) rateElement.textContent = `${rate}×`;
    if (frameElement) frameElement.textContent = frame;
  }

  _bindChrome() {
    if (this._chromeBound || !this.root) return;
    this._chromeBound = true;
    const bind = (selector, handler) => this.root.querySelector?.(selector)?.addEventListener("click", () => {
      Promise.resolve(handler()).catch((error) => this._setError(error?.code ?? "request_failed", String(error?.message ?? error), error?.status ?? 0));
    });
    bind('[data-action="refresh"]', () => this.assetId ? this.requestTask(this.assetId) : this.start());
    bind('[data-action="rate-decrease"]', () => this.videoController?.decreaseRate());
    bind('[data-action="rate-increase"]', () => this.videoController?.increaseRate());
    bind('[data-action="previous-asset"]', () => this.loadPreviousAsset());
    bind('[data-action="next-asset"]', () => this.loadNextAsset());
  }
}

export default WarnReviewApp;

if (typeof window !== "undefined" && window.document) {
  window.WarnReviewApp = WarnReviewApp;
  window.addEventListener("DOMContentLoaded", () => {
    const root = window.document.querySelector("#app");
    if (!root) return;
    const app = new WarnReviewApp({ documentRef: window.document, root });
    window.humanQcWarnReview = app;
    app.mount(root);
    app.start().catch((error) => app._setError(error?.code ?? "request_failed", String(error?.message ?? error), error?.status ?? 0));
  }, { once: true });
}
