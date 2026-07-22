/**
 * Small application shell for the human QC workbench.
 *
 * WorkbenchApp owns transport, reviewer lease and the current server task.
 * Domain adapters receive a task snapshot and never reach into another
 * adapter's DOM. A failed 409/423 response is retained as an error; the
 * previous task is deliberately left untouched until the reviewer refreshes.
 */

import { WarnReviewAdapter } from "./warn_adapter.js";

export function mutationControlsDisabled(task) {
  return false;
}

export function stageTypeForTask(task) {
  const taskType = String(task?.task_type ?? "");
  if (taskType !== "warn_review") return taskType;
  const warn = task?.warn ?? {};
  const candidates = Array.isArray(warn.candidate_issue_ids) ? warn.candidate_issue_ids : null;
  if (["completed", "not_required", "skipped_due_to_fail"].includes(warn.state)) return "completed";
  if (candidates && !candidates.length) return "completed";
  return taskType;
}

function responseError(response, body) {
  const detail = body?.error ?? {};
  const error = new Error(detail.message || `request failed (${response.status})`);
  error.code = detail.code || (response.status === 423 ? "lease_invalid" : "request_failed");
  error.status = response.status;
  error.currentRevision = detail.current_revision ?? null;
  return error;
}

export class WorkbenchApp {
  constructor({
    baseUrl = "",
    fetcher = globalThis.fetch?.bind(globalThis),
    documentRef = globalThis.document,
    root = null,
    reviewer = "",
    adapterFactory = null,
    warnAdapterFactory = null,
    leaseRenewIntervalMs = 240000,
    onNavigate = null,
  } = {}) {
    this.baseUrl = String(baseUrl).replace(/\/$/, "");
    this.fetcher = fetcher;
    this.document = documentRef;
    this.root = root;
    this.reviewer = reviewer;
    this.warnAdapterFactory = warnAdapterFactory ?? adapterFactory ?? ((options) => new WarnReviewAdapter(options));
    this.leaseRenewIntervalMs = Number(leaseRenewIntervalMs) || 0;
    this.onNavigate = onNavigate;
    this.task = null;
    this.assetId = null;
    this.lease = null;
    this.adapter = null;
    this.adapterType = null;
    this.lastError = null;
    this.loading = false;
    this.onTask = null;
    this.leaseTimer = null;
    this.chromeBound = false;
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
      this.lastError = { code: "network_error", message: String(error?.message ?? error) };
      this.renderStatus();
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
      this.lastError = {
        code: error.code,
        message: error.message,
        status: error.status,
        current_revision: error.currentRevision,
      };
      this.renderStatus();
      // Do not call applyServerTask here: a stale/conflicting response must
      // never replace the reviewer's current task snapshot.
      throw error;
    }
    this.lastError = null;
    return value;
  }

  async requestTask(assetId) {
    const value = await this.requestJson(`/api/assets/${encodeURIComponent(assetId)}/task`);
    const task = value?.task ?? value;
    this.applyServerTask(task);
    return task;
  }

  async loadAsset(assetId) {
    if (!assetId) throw new Error("assetId is required");
    if (mutationControlsDisabled(this.task)) {
      const error = new Error("pending edit must be confirmed or cancelled before navigation");
      error.code = "pending_lock";
      this.lastError = { code: error.code, message: error.message };
      throw error;
    }
    const nextAssetId = String(assetId);
    if (this.assetId !== nextAssetId) {
      this.stopLeaseRenewal();
      this.lease = null;
      this.renderStatus();
    }
    return this.requestTask(nextAssetId);
  }

  applyServerTask(task) {
    if (!task || typeof task !== "object") throw new Error("server task must be an object");
    this.task = task;
    this.assetId = task.asset_id ?? this.assetId;
    if (task.lease_token && !this.lease) this.lease = { token: task.lease_token };
    this.advanceStage(task);
    this.onTask?.(task);
    return task;
  }

  advanceStage(serverTask = this.task) {
    if (!serverTask || typeof serverTask !== "object") throw new Error("server task must be an object");
    this.task = serverTask;
    this.assetId = serverTask.asset_id ?? this.assetId;
    const nextType = stageTypeForTask(serverTask);
    if (nextType !== this.adapterType) {
      this.adapter?.destroy?.();
      this.adapter = null;
      this.adapterType = nextType;
    }
    this.render();
    return nextType;
  }

  async acquireLease(reviewer = this.reviewer) {
    if (!reviewer) throw new Error("reviewer is required");
    const value = await this.requestJson(`/api/assets/${encodeURIComponent(this.assetId)}/lease/acquire`, {
      method: "POST",
      body: { reviewer },
    });
    this.lease = value?.lease ?? null;
    this.startLeaseRenewal();
    this.renderStatus();
    return this.lease;
  }

  async renewLease(ttlSeconds = undefined) {
    if (!this.assetId || !this.lease?.token || !this.task) throw new Error("lease is not acquired");
    const body = {
      expected_revision: this.revision(),
      lease_token: this.lease.token,
    };
    if (ttlSeconds !== undefined) body.ttl_seconds = ttlSeconds;
    const value = await this.requestJson(`/api/assets/${encodeURIComponent(this.assetId)}/lease/renew`, {
      method: "POST",
      body,
    });
    this.lease = value?.lease ?? this.lease;
    this.renderStatus();
    return this.lease;
  }

  revision() {
    return Number(this.task?.revision ?? this.task?.report_revision ?? 0);
  }

  mutationBody(payload = {}) {
    if (!this.lease?.token) throw new Error("lease is not acquired");
    return {
      ...payload,
      expected_revision: this.revision(),
      lease_token: this.lease.token,
    };
  }

  async mutate(path, payload = {}) {
    const value = await this.requestJson(path, { method: "POST", body: this.mutationBody(payload) });
    const task = value?.task ?? value;
    this.applyServerTask(task);
    return task;
  }

  async submitWarnVerdict(issueId, verdict, reason = "") {
    return this.mutate(
      `/api/assets/${encodeURIComponent(this.assetId)}/warn/${encodeURIComponent(issueId)}/verdict`,
      { verdict, reason },
    );
  }

  async completeWarn() {
    return this.mutate(`/api/assets/${encodeURIComponent(this.assetId)}/warn/complete`);
  }

  mount(root = this.root ?? this.document?.querySelector?.("#app")) {
    this.root = root;
    if (!root) return this;
    this.bindChrome();
    this.render();
    return this;
  }

  render() {
    if (!this.root || !this.document) return;
    const task = this.task;
    const stage = this.root.querySelector?.("[data-workbench-stage]");
    if (!stage) return;
    const taskType = this.adapterType ?? stageTypeForTask(task);
    if (taskType === "warn_review") {
      this.adapter ??= this.warnAdapterFactory({
        onVerdict: (issueId, verdict, reason) => this.submitWarnVerdict(issueId, verdict, reason),
        onComplete: () => this.completeWarn(),
        video: this.root.querySelector?.("[data-video]") ?? null,
        videoPlaceholder: this.root.querySelector?.("[data-video-placeholder]") ?? null,
      });
      this.adapter.render(task, stage);
    } else if (taskType === "completed") {
      stage.innerHTML = '<div class="empty-stage" data-task-completed>当前资产的人工复核已完成，可以进入下一条。</div>';
    } else if (taskType === "error") {
      stage.innerHTML = '<div class="empty-stage" data-task-error>当前资产处理失败，请查看服务端错误后刷新。</div>';
    } else {
      stage.innerHTML = '<div class="empty-stage">当前任务没有可用的复核阶段。</div>';
    }
    this.renderStatus();
  }

  renderStatus() {
    if (!this.root) return;
    const currentTaskType = this.task ? stageTypeForTask(this.task) : "";
    if (this.root.dataset) this.root.dataset.taskType = currentTaskType || "idle";
    const set = (selector, value) => {
      const element = this.root.querySelector?.(selector);
      if (element) element.textContent = value;
    };
    set("[data-asset-id]", this.assetId || "未加载资产");
    set("[data-revision]", this.task ? `revision ${this.revision()}` : "revision —");
    set("[data-reviewer]", this.reviewer || "reviewer —");
    set("[data-lease-status]", this.lease?.expires_at ? "已获取" : "未获取");
    const progress = this.root.querySelector?.("[data-progress]");
    if (progress && this.task) progress.textContent = this.task.task_type || "—";
    const taskLabels = {
      warn_review: "Warn Review",
      completed: "Completed",
      error: "Error",
    };
    set("[data-task-kind]", taskLabels[currentTaskType] || "—");
    set("[data-stage-title]", {
      warn_review: "Warn 复核",
      completed: "人工复核已完成",
      error: "任务处理失败",
    }[currentTaskType] || "人工复核");
    set("[data-inspector-note]", "观看问题片段后选择 Pass 或 Fail；机器信息仅供参考。");
    const error = this.root.querySelector?.("[data-save-error]");
    if (error) {
      error.textContent = this.lastError?.message || "";
      error.hidden = !this.lastError;
    }
  }

  setMutationLock(locked) {
    if (!this.root) return;
    this.root.querySelectorAll?.("[data-mutation-control], [data-navigation-control]").forEach((element) => {
      if (element.dataset.action === "confirm-pending" || element.dataset.action === "cancel-pending") return;
      element.disabled = Boolean(locked);
    });
  }

  bindChrome() {
    if (this.chromeBound || !this.root) return;
    this.chromeBound = true;
    this.root.querySelector?.('[data-action="refresh"]')?.addEventListener("click", () => {
      if (this.assetId) this.requestTask(this.assetId).catch(() => {});
    });
    this.root.querySelector?.('[data-action="acquire-lease"]')?.addEventListener("click", () => {
      this.acquireLease().catch(() => {});
    });
    ["previous-asset", "next-asset"].forEach((action) => {
      this.root.querySelector?.(`[data-action="${action}"]`)?.addEventListener("click", (event) => {
        const id = event.currentTarget?.dataset?.assetId;
        if (id) this.onNavigate?.(id, action);
      });
    });
  }

  startLeaseRenewal() {
    this.stopLeaseRenewal();
    if (!(this.leaseRenewIntervalMs > 0)) return;
    this.leaseTimer = setInterval(() => {
      this.renewLease().catch((error) => {
        // A lease conflict or stale revision requires a refresh instead of a
        // background retry that could overwrite the reviewer's task.
        if ([409, 423].includes(error?.status)) this.stopLeaseRenewal();
      });
    }, this.leaseRenewIntervalMs);
    this.leaseTimer?.unref?.();
  }

  stopLeaseRenewal() {
    if (this.leaseTimer) clearInterval(this.leaseTimer);
    this.leaseTimer = null;
  }
}

export default WorkbenchApp;

// Browser bootstrap. Keeping this at the edge means the class remains usable
// from tests and embedding applications without a global singleton.
if (typeof window !== "undefined" && window.document) {
  window.WorkbenchApp = WorkbenchApp;
  window.addEventListener("DOMContentLoaded", () => {
    const root = window.document.querySelector("#app");
    if (!root) return;
    const app = new WorkbenchApp({
      documentRef: window.document,
      root,
      reviewer: root.dataset.reviewer || "",
    });
    window.humanQcWorkbench = app;
    app.mount(root);
  }, { once: true });
}
