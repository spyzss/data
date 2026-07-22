import { SemanticCalibrationAdapter } from "./semantic_adapter.js";


function responseError(response, body) {
  const detail = body?.error ?? {};
  const error = new Error(detail.message || `request failed (${response.status})`);
  error.code = detail.code || "request_failed";
  error.status = response.status;
  error.currentRevision = detail.current_revision ?? null;
  return error;
}


export class SemanticCalibrationApp {
  constructor({
    baseUrl = "",
    fetcher = globalThis.fetch?.bind(globalThis),
    documentRef = globalThis.document,
    locationRef = globalThis.location ?? { search: "" },
    root = null,
    reviewer = "",
    adapterFactory = null,
  } = {}) {
    this.baseUrl = String(baseUrl).replace(/\/$/, "");
    this.fetcher = fetcher;
    this.document = documentRef;
    this.location = locationRef;
    this.root = root;
    this.reviewer = reviewer;
    this.adapterFactory = adapterFactory ?? ((options) => new SemanticCalibrationAdapter(options));
    this.assets = [];
    this.assetId = null;
    this.task = null;
    this.lease = null;
    this.adapter = null;
    this.error = null;
    this.state = "idle";
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
    const response = await this.fetcher(this.endpoint(path), options);
    const value = await response.json().catch(() => null);
    if (!response.ok) {
      const error = responseError(response, value);
      this.error = { code: error.code, message: error.message, status: error.status };
      this.state = "error";
      this.render();
      throw error;
    }
    this.error = null;
    return value;
  }

  async start() {
    this.state = "loading";
    this.render();
    const value = await this.requestJson("/api/semantic/assets");
    this.assets = Array.isArray(value?.assets) ? value.assets : [];
    const requested = new URLSearchParams(this.location?.search ?? "").get("asset_id");
    const assetId = requested || this.assets[0]?.asset_id || null;
    if (!assetId) {
      this.state = "empty";
      this.task = null;
      this.render();
      return null;
    }
    return this.loadAsset(assetId);
  }

  async loadAsset(assetId) {
    if (this.task?.semantic?.pending_edit) {
      const error = new Error("pending edit must be confirmed or cancelled before navigation");
      error.code = "pending_lock";
      throw error;
    }
    const value = await this.requestJson(`/api/semantic/assets/${encodeURIComponent(assetId)}/task`);
    this.assetId = String(assetId);
    this.task = value?.task ?? value;
    this.state = "ready";
    this.render();
    return this.task;
  }

  revision() {
    return Number(this.task?.revision ?? this.task?.report_revision ?? 0);
  }

  async acquireLease(reviewer = this.reviewer) {
    if (!this.assetId || !reviewer) throw new Error("asset and reviewer are required");
    const value = await this.requestJson(
      `/api/semantic/assets/${encodeURIComponent(this.assetId)}/lease/acquire`,
      { method: "POST", body: { reviewer } },
    );
    this.lease = value?.lease ?? null;
    this.render();
    return this.lease;
  }

  async releaseLease() {
    if (!this.lease?.token) return false;
    await this.requestJson(
      `/api/semantic/assets/${encodeURIComponent(this.assetId)}/lease/release`,
      {
        method: "POST",
        body: { expected_revision: this.revision(), lease_token: this.lease.token },
      },
    );
    this.lease = null;
    this.render();
    return true;
  }

  async mutate(path, payload = {}) {
    if (!this.lease?.token) throw new Error("lease is not acquired");
    const value = await this.requestJson(path, {
      method: "POST",
      body: {
        ...payload,
        expected_revision: this.revision(),
        lease_token: this.lease.token,
      },
    });
    this.task = value?.task ?? value;
    this.state = "ready";
    this.render();
    return this.task;
  }

  render() {
    if (!this.root) return;
    const stage = this.root.querySelector?.("[data-semantic-stage]");
    const status = this.root.querySelector?.("[data-status]");
    const asset = this.root.querySelector?.("[data-asset-id]");
    if (asset) asset.textContent = this.assetId ?? "—";
    if (status) {
      status.textContent = this.error?.message
        ?? (this.state === "empty" ? "没有待处理的语义任务" : this.state);
    }
    if (!stage || !this.task) return;
    if (!this.adapter) {
      this.adapter = this.adapterFactory({
        root: stage,
        postPending: (payload) => this.mutate(
          `/api/semantic/assets/${encodeURIComponent(this.assetId)}/boundary/pending`,
          payload,
        ),
        onTextPending: (payload) => this.mutate(
          `/api/semantic/assets/${encodeURIComponent(this.assetId)}/text/pending`,
          payload,
        ),
        onConfirm: () => this.mutate(
          `/api/semantic/assets/${encodeURIComponent(this.assetId)}/pending/confirm`,
        ),
        onCancel: () => this.mutate(
          `/api/semantic/assets/${encodeURIComponent(this.assetId)}/pending/cancel`,
        ),
        onComplete: () => this.mutate(
          `/api/semantic/assets/${encodeURIComponent(this.assetId)}/complete`,
        ),
      });
    }
    this.adapter.render?.(this.task);
  }
}


export function bootstrapSemanticCalibration(documentRef = globalThis.document) {
  const root = documentRef?.querySelector?.("[data-semantic-app]");
  if (!root) return null;
  const app = new SemanticCalibrationApp({
    documentRef,
    root,
    reviewer: root.dataset?.reviewer ?? "",
  });
  app.start().catch(() => {});
  return app;
}


if (typeof document !== "undefined") {
  document.addEventListener("DOMContentLoaded", () => bootstrapSemanticCalibration(document), { once: true });
}
