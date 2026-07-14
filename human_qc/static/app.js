/**
 * Small application shell for the human QC workbench.
 *
 * WorkbenchApp owns transport, reviewer lease and the current server task.
 * Domain adapters receive a task snapshot and never reach into another
 * adapter's DOM. A failed 409/423 response is retained as an error; the
 * previous task is deliberately left untouched until the reviewer refreshes.
 */

import { SemanticCalibrationAdapter } from "./semantic_adapter.js";

export function mutationControlsDisabled(task) {
  const semantic = task?.semantic ?? task;
  return Boolean(semantic?.pending_edit ?? task?.pending_edit);
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
    adapterFactory = (options) => new SemanticCalibrationAdapter(options),
    leaseRenewIntervalMs = 240000,
    onNavigate = null,
  } = {}) {
    this.baseUrl = String(baseUrl).replace(/\/$/, "");
    this.fetcher = fetcher;
    this.document = documentRef;
    this.root = root;
    this.reviewer = reviewer;
    this.adapterFactory = adapterFactory;
    this.leaseRenewIntervalMs = Number(leaseRenewIntervalMs) || 0;
    this.onNavigate = onNavigate;
    this.task = null;
    this.assetId = null;
    this.lease = null;
    this.adapter = null;
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
    this.assetId = String(assetId);
    return this.requestTask(this.assetId);
  }

  applyServerTask(task) {
    if (!task || typeof task !== "object") throw new Error("server task must be an object");
    this.task = task;
    this.assetId = task.asset_id ?? this.assetId;
    if (task.lease_token && !this.lease) this.lease = { token: task.lease_token };
    this.render();
    this.onTask?.(task);
    return task;
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

  async submitBoundary(payload) {
    return this.mutate(`/api/assets/${encodeURIComponent(this.assetId)}/semantic/boundary/pending`, payload);
  }

  async submitText(payload) {
    return this.mutate(`/api/assets/${encodeURIComponent(this.assetId)}/semantic/text/pending`, payload);
  }

  async confirmPending() {
    return this.mutate(`/api/assets/${encodeURIComponent(this.assetId)}/semantic/pending/confirm`);
  }

  async cancelPending() {
    return this.mutate(`/api/assets/${encodeURIComponent(this.assetId)}/semantic/pending/cancel`);
  }

  async completeSemantic() {
    return this.mutate(`/api/assets/${encodeURIComponent(this.assetId)}/semantic/complete`);
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
    const semantic = task?.semantic;
    const stage = this.root.querySelector?.("[data-workbench-stage]");
    if (!stage) return;
    if (semantic) {
      this.adapter ??= this.adapterFactory({
        postPending: (payload) => this.submitBoundary(payload),
        onTextPending: (payload) => this.submitText(payload),
        onConfirm: () => this.confirmPending(),
        onCancel: () => this.cancelPending(),
        onComplete: () => this.completeSemantic(),
        onLockChange: (locked) => this.setMutationLock(locked),
      });
      this.adapter.render(task, stage);
    } else {
      stage.innerHTML = `<div class="empty-stage">当前任务没有可用的语义校准数据。</div>`;
    }
    this.renderStatus();
  }

  renderStatus() {
    if (!this.root) return;
    const set = (selector, value) => {
      const element = this.root.querySelector?.(selector);
      if (element) element.textContent = value;
    };
    set("[data-asset-id]", this.assetId || "未加载资产");
    set("[data-revision]", this.task ? `revision ${this.revision()}` : "revision —");
    set("[data-reviewer]", this.reviewer || "reviewer —");
    set("[data-lease-status]", this.lease?.expires_at ? "已获取" : "未获取");
    const timelineCount = this.task?.semantic?.timeline_edit_count;
    const textCount = this.task?.semantic?.subtask_text_edit_count;
    set("[data-timeline-count]", Number.isInteger(timelineCount) ? `${timelineCount} 次` : "—");
    set("[data-text-count]", Number.isInteger(textCount) ? `${textCount} 次` : "—");
    const progress = this.root.querySelector?.("[data-progress]");
    if (progress && this.task) progress.textContent = this.task.task_type || "—";
    set("[data-task-kind]", this.task?.task_type || "Semantic");
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
