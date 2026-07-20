function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function jsonText(value) {
  return JSON.stringify(value ?? null, null, 2);
}

export function resolveAppUrl(path, baseURI = globalThis.document?.baseURI) {
  const value = String(path ?? "").trim();
  if (!value) return "";
  try {
    return new URL(value).toString();
  } catch (_error) {
    if (!baseURI) return value;
    const pageDirectory = new URL(".", baseURI);
    return new URL(value.replace(/^\/+/, ""), pageDirectory).toString();
  }
}

export function reviewProgress(items) {
  const rows = Array.isArray(items) ? items : [];
  return {
    completed: rows.filter((item) => ["pass", "fail"].includes(item?.manual_review?.verdict)).length,
    total: rows.length,
  };
}

function evidenceMarkup(item, baseURI) {
  const evidence = Array.isArray(item?.evidence) ? item.evidence : [];
  if (!evidence.length) return '<div class="evidence-error">Evidence unavailable: no combined overlays</div>';
  return evidence.map((row, index) => {
    const frame = escapeHtml(row.frame_idx ?? "—");
    if (row.status !== "ready" || !row.url) {
      return `<article class="evidence-card evidence-card-error"><strong>Sample ${index + 1}</strong><span>source frame ${frame}</span><span>Evidence unavailable: ${escapeHtml(row.status)}</span></article>`;
    }
    const url = resolveAppUrl(row.url, baseURI);
    return `<article class="evidence-card" data-evidence-index="${index}"><header><strong>Sample ${index + 1}</strong><span>source frame ${frame}</span></header><img src="${escapeHtml(url)}" alt="SAM3 containment combined overlay at source frame ${frame}" loading="eager" data-source-frame="${frame}"></article>`;
  }).join("");
}

export function renderReviewMarkup(item, index, total, baseURI = globalThis.document?.baseURI) {
  if (!item) return '<div class="empty-state">No SAM3 windows require human review.</div>';
  const review = item.manual_review ?? { status: "unresolved", verdict: null };
  const completed = ["pass", "fail"].includes(review.verdict);
  const saved = completed
    ? `Saved: ${escapeHtml(review.verdict)} by ${escapeHtml(review.reviewer)}`
    : "Unresolved — no server verdict saved";
  const disabled = item.can_review === true ? "" : " disabled";
  const firstFrame = item.evidence?.find((row) => row.status === "ready")?.frame_idx ?? "—";
  const evidenceError = item.can_review === true
    ? ""
    : `<div class="evidence-error" role="alert">Evidence unavailable: ${escapeHtml(item.evidence_error || item.evidence_status)}</div>`;
  return `<article class="review-window" data-review-id="${escapeHtml(item.review_id)}">
    <nav class="window-nav">
      <button type="button" data-action="previous-window"${index <= 0 ? " disabled" : ""}>Previous</button>
      <strong>${index + 1} / ${total}</strong>
      <button type="button" data-action="next-window"${index + 1 >= total ? " disabled" : ""}>Next</button>
    </nav>
    <section class="identity-grid">
      <div><span>review_id</span><strong>${escapeHtml(item.review_id)}</strong></div>
      <div><span>asset_id</span><strong>${escapeHtml(item.asset_id)}</strong></div>
      <div><span>source inclusive window</span><strong>${escapeHtml(item.window_start_frame)}–${escapeHtml(item.window_end_frame)}</strong></div>
      <div><span>fps</span><strong>${escapeHtml(item.fps ?? "—")}</strong></div>
      <div><span>current sampled source frame</span><strong data-current-source-frame>${escapeHtml(firstFrame)}</strong></div>
      <div><span>duration_resolution_status</span><strong>not_annotated</strong></div>
    </section>
    <section class="sam3-summary">
      <h2>SAM3 containment combined overlay evidence</h2>
      <div class="hand-verdicts"><span>Left: <b>${escapeHtml(item.left_window_containment_verdict || "—")}</b></span><span>Right: <b>${escapeHtml(item.right_window_containment_verdict || "—")}</b></span></div>
      <div class="trigger-grid"><div><h3>Trigger reason</h3><pre>${escapeHtml(jsonText(item.trigger_reason))}</pre></div><div><h3>Key metrics</h3><pre>${escapeHtml(jsonText(item.trigger_metrics))}</pre></div></div>
      ${evidenceError}
      <div class="evidence-grid">${evidenceMarkup(item, baseURI)}</div>
    </section>
    <footer class="decision-bar">
      <div><span>Server state</span><strong data-save-status>${saved}</strong></div>
      <div class="decision-actions">
        <button type="button" class="pass-action" data-action="verdict-pass"${disabled}>Pass</button>
        <button type="button" class="fail-action" data-action="verdict-fail"${disabled}>Fail</button>
      </div>
    </footer>
  </article>`;
}

export class Sam3WindowReviewApp {
  constructor({
    fetcher = globalThis.fetch?.bind(globalThis),
    root = null,
    baseURI = globalThis.document?.baseURI,
  } = {}) {
    this.fetcher = fetcher;
    this.root = root;
    this.baseURI = baseURI;
    this.items = [];
    this.index = 0;
    this.lastError = null;
  }

  async load() {
    const response = await this.fetcher(resolveAppUrl("api/review-bundle", this.baseURI), { cache: "no-store" });
    if (!response.ok) throw new Error(`Unable to load review queue: HTTP ${response.status}`);
    const payload = await response.json();
    this.items = Array.isArray(payload?.bundle?.items) ? payload.bundle.items : [];
    this.index = Math.min(this.index, Math.max(this.items.length - 1, 0));
    this.render();
    return this.items;
  }

  currentItem() {
    return this.items[this.index] ?? null;
  }

  next() {
    if (this.index + 1 < this.items.length) this.index += 1;
    this.render();
    return this.currentItem();
  }

  previous() {
    if (this.index > 0) this.index -= 1;
    this.render();
    return this.currentItem();
  }

  reviewer() {
    return String(this.root?.querySelector?.("[data-reviewer]")?.value ?? "").trim();
  }

  async submit(verdict) {
    if (!["pass", "fail"].includes(verdict)) throw new Error("verdict must be pass or fail");
    const item = this.currentItem();
    if (!item || item.can_review !== true) throw new Error("Evidence unavailable; review cannot be completed");
    const reviewer = this.reviewer();
    if (!reviewer) throw new Error("Reviewer is required");
    const expectedRevision = Number(item.manual_review?.revision ?? 0);
    const response = await this.fetcher(resolveAppUrl(`api/reviews/${encodeURIComponent(item.review_id)}`, this.baseURI), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ verdict, reviewer, expected_revision: expectedRevision }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload?.error?.message || `Save failed: HTTP ${response.status}`);
    item.manual_review = payload.review;
    this.lastError = null;
    this.render();
    return payload.review;
  }

  render() {
    if (!this.root) return renderReviewMarkup(this.currentItem(), this.index, this.items.length, this.baseURI);
    const stage = this.root.querySelector?.("[data-review-stage]");
    if (stage) stage.innerHTML = renderReviewMarkup(this.currentItem(), this.index, this.items.length, this.baseURI);
    const progress = reviewProgress(this.items);
    const progressNode = this.root.querySelector?.("[data-global-progress]");
    if (progressNode) progressNode.textContent = `${progress.completed} / ${progress.total}`;
    const error = this.root.querySelector?.("[data-error]");
    if (error) {
      error.textContent = this.lastError?.message || "";
      error.hidden = !this.lastError;
    }
    stage?.querySelector?.('[data-action="previous-window"]')?.addEventListener("click", () => this.previous());
    stage?.querySelector?.('[data-action="next-window"]')?.addEventListener("click", () => this.next());
    stage?.querySelector?.('[data-action="verdict-pass"]')?.addEventListener("click", () => this._submit("pass"));
    stage?.querySelector?.('[data-action="verdict-fail"]')?.addEventListener("click", () => this._submit("fail"));
    stage?.querySelectorAll?.("[data-source-frame]").forEach((image) => {
      image.addEventListener("click", () => {
        const current = stage.querySelector?.("[data-current-source-frame]");
        if (current) current.textContent = image.dataset.sourceFrame;
      });
    });
    return stage?.innerHTML ?? "";
  }

  async _submit(verdict) {
    try {
      await this.submit(verdict);
    } catch (error) {
      this.lastError = error;
      this.render();
    }
  }
}

if (typeof window !== "undefined" && window.document) {
  window.addEventListener("DOMContentLoaded", () => {
    const root = window.document.querySelector("#app");
    const app = new Sam3WindowReviewApp({ root });
    window.sam3WindowReview = app;
    app.load().catch((error) => {
      app.lastError = error;
      app.render();
    });
  }, { once: true });
}
