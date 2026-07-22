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
    leaseRenewIntervalMs = 240000,
    setIntervalFn = globalThis.setInterval?.bind(globalThis),
    clearIntervalFn = globalThis.clearInterval?.bind(globalThis),
    eventTarget = globalThis.window,
    storage = null,
  } = {}) {
    this.baseUrl = String(baseUrl).replace(/\/$/, "");
    this.fetcher = fetcher;
    this.document = documentRef;
    this.location = locationRef;
    this.root = root;
    const queryReviewer = new URLSearchParams(this.location?.search ?? "").get("reviewer");
    this.reviewer = String(reviewer || queryReviewer || storage?.getItem?.("semantic_reviewer") || "").trim();
    this.adapterFactory = adapterFactory ?? ((options) => new SemanticCalibrationAdapter(options));
    this.assets = [];
    this.assetId = null;
    this.task = null;
    this.lease = null;
    this.adapter = null;
    this.error = null;
    this.state = "idle";
    this.leaseRenewIntervalMs = Number(leaseRenewIntervalMs) || 0;
    this.setIntervalFn = setIntervalFn;
    this.clearIntervalFn = clearIntervalFn;
    this.eventTarget = eventTarget;
    this.storage = storage;
    this.leaseTimer = null;
    this.lifecycleBound = false;
    this.chromeBound = false;
    this.boundVideo = null;
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
    this.bindChrome();
    this.bindLifecycle();
    if (!this.reviewer) {
      this.state = "awaiting_reviewer";
      this.render();
      return null;
    }
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
    const nextAssetId = String(assetId);
    if (this.assetId && this.assetId !== nextAssetId && this.lease?.token) {
      await this.releaseLease();
    }
    const value = await this.requestJson(`/api/semantic/assets/${encodeURIComponent(nextAssetId)}/task`);
    this.assetId = nextAssetId;
    this.task = value?.task ?? value;
    this.state = "ready";
    this.render();
    await this.acquireLease();
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
    this.startLeaseRenewal();
    this.render();
    return this.lease;
  }

  async renewLease() {
    if (!this.assetId || !this.lease?.token) throw new Error("lease is not acquired");
    const value = await this.requestJson(
      `/api/semantic/assets/${encodeURIComponent(this.assetId)}/lease/renew`,
      {
        method: "POST",
        body: { expected_revision: this.revision(), lease_token: this.lease.token },
      },
    );
    this.lease = value?.lease ?? this.lease;
    this.render();
    return this.lease;
  }

  async releaseLease({ keepalive = false } = {}) {
    if (!this.lease?.token) return false;
    const assetId = this.assetId;
    const token = this.lease.token;
    this.stopLeaseRenewal();
    try {
      if (keepalive) {
        const response = await this.fetcher(
          this.endpoint(`/api/semantic/assets/${encodeURIComponent(assetId)}/lease/release`),
          {
            method: "POST",
            headers: { Accept: "application/json", "Content-Type": "application/json" },
            body: JSON.stringify({ lease_token: token }),
            keepalive: true,
          },
        );
        if (!response.ok) throw responseError(response, await response.json().catch(() => null));
      } else {
        await this.requestJson(
          `/api/semantic/assets/${encodeURIComponent(assetId)}/lease/release`,
          { method: "POST", body: { lease_token: token } },
        );
      }
      return true;
    } finally {
      if (this.assetId === assetId && this.lease?.token === token) this.lease = null;
      this.render();
    }
  }

  startLeaseRenewal() {
    this.stopLeaseRenewal();
    if (!(this.leaseRenewIntervalMs > 0) || typeof this.setIntervalFn !== "function") return;
    this.leaseTimer = this.setIntervalFn(() => this.renewLease().catch((error) => {
      if ([409, 423].includes(error?.status)) this.stopLeaseRenewal();
    }), this.leaseRenewIntervalMs);
    this.leaseTimer?.unref?.();
  }

  stopLeaseRenewal() {
    if (this.leaseTimer != null && typeof this.clearIntervalFn === "function") {
      this.clearIntervalFn(this.leaseTimer);
    }
    this.leaseTimer = null;
  }

  bindLifecycle() {
    if (this.lifecycleBound || !this.eventTarget?.addEventListener) return;
    this.lifecycleBound = true;
    const release = () => this.releaseLease({ keepalive: true }).catch(() => false);
    this.eventTarget.addEventListener("pagehide", release);
    this.eventTarget.addEventListener("beforeunload", release);
  }

  bindChrome() {
    if (!this.root || this.chromeBound) return;
    this.chromeBound = true;
    const start = this.root.querySelector?.('[data-action="start-calibration"]');
    start?.addEventListener?.("click", () => {
      const input = this.root.querySelector?.("[data-reviewer-input]");
      const reviewer = String(input?.value ?? "").trim();
      if (!reviewer) {
        this.error = { code: "reviewer_required", message: "请填写操作员身份" };
        this.render();
        return;
      }
      this.reviewer = reviewer;
      this.storage?.setItem?.("semantic_reviewer", reviewer);
      this.start().catch(() => {});
    });
    this.root.querySelector?.("[data-asset-select]")?.addEventListener?.("change", (event) => {
      if (event.currentTarget?.value) this.loadAsset(event.currentTarget.value).catch(() => {});
    });
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

  async completeAndRelease() {
    try {
      return await this.mutate(
        `/api/semantic/assets/${encodeURIComponent(this.assetId)}/complete`,
      );
    } finally {
      await this.releaseLease().catch(() => false);
    }
  }

  render() {
    if (!this.root) return;
    const stage = this.root.querySelector?.("[data-semantic-stage]");
    const status = this.root.querySelector?.("[data-status]");
    const asset = this.root.querySelector?.("[data-asset-id]");
    const reviewer = this.root.querySelector?.("[data-reviewer-current]");
    const reviewerGate = this.root.querySelector?.("[data-reviewer-gate]");
    const workbench = this.root.querySelector?.("[data-workbench]");
    const selector = this.root.querySelector?.("[data-asset-select]");
    if (asset) asset.textContent = this.assetId ?? "—";
    if (reviewer) reviewer.textContent = this.reviewer || "—";
    if (reviewerGate) reviewerGate.hidden = Boolean(this.reviewer);
    if (workbench) workbench.hidden = !this.reviewer;
    if (selector && this.assets.length && this.document?.createElement) {
      const options = this.assets.map((item) => {
        const option = this.document.createElement("option");
        option.value = String(item.asset_id);
        option.textContent = String(item.asset_id);
        return option;
      });
      selector.replaceChildren?.(...options);
      selector.value = this.assetId ?? "";
    }
    if (status) {
      status.textContent = this.error?.message
        ?? (this.state === "empty" ? "没有待处理的语义任务" : this.state);
    }
    if (!stage || !this.task) return;
    const video = this.root.querySelector?.("[data-semantic-video]");
    if (video && this.task.video_url && video.getAttribute?.("src") !== this.task.video_url) {
      video.src = this.task.video_url;
    }
    if (video && this.boundVideo !== video) {
      this.boundVideo = video;
      video.addEventListener?.("timeupdate", () => {
        const fps = Number(this.task?.semantic?.timeline?.fps ?? 0);
        const total = Number(this.task?.semantic?.timeline?.frame_count ?? 0);
        const current = fps > 0 ? Math.max(0, Math.floor(Number(video.currentTime || 0) * fps)) : 0;
        const frame = total > 0 ? Math.min(total - 1, current) : current;
        const output = this.root?.querySelector?.("[data-current-frame]");
        if (output) output.textContent = String(frame);
      });
    }
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
        onComplete: () => this.completeAndRelease(),
      });
    }
    this.adapter.render?.(this.task, stage);
  }
}


export function bootstrapSemanticCalibration(documentRef = globalThis.document) {
  const root = documentRef?.querySelector?.("[data-semantic-app]");
  if (!root) return null;
  const app = new SemanticCalibrationApp({
    documentRef,
    root,
    reviewer: root.dataset?.reviewer ?? "",
    storage: globalThis.localStorage,
  });
  app.start().catch(() => {});
  return app;
}


if (typeof document !== "undefined") {
  document.addEventListener("DOMContentLoaded", () => bootstrapSemanticCalibration(document), { once: true });
}
