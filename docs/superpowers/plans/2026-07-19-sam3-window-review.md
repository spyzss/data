# SAM3 Window Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development. Execute in the current workspace because the user explicitly prohibited branch/worktree changes and commits.

**Goal:** Add a standalone JD SAM3 containment window-review entry that consumes canonical manifest/queue/evidence files, exposes only Pass/Fail decisions, stages whitelisted overlays for HTTP, and persists window-level decisions without changing QC or rejected-duration semantics.

**Architecture:** `human_qc.sam3_window_review` owns immutable input normalization, exact review/evidence matching, controlled asset staging, and the atomic result store. `human_qc.sam3_window_review_server` exposes a small JSON/static/evidence HTTP surface; `tools/serve_sam3_window_review.py` wires explicit CLI paths into one process. Dedicated static files render the queue and autosave one `review_id` at a time. Existing legacy manual-review tools and the generic report-revision workbench remain unchanged.

**Tech Stack:** Python 3 standard-library HTTP server and atomic filesystem replacement, pandas for CSV/Parquet input, vanilla browser JavaScript, pytest.

## Global Constraints

- Work only on `codex/human-qc-impl` at HEAD `1550b65fbf664d8ba06c6835d8cd577094909f71` plus the existing intentional QY workspace changes.
- Do not reset, restore, clean, stash, stage, commit, or push.
- Do not modify `tools/build_video_review_clips.py` or `tools/serve_manual_review.py`.
- Verdict is exactly `pass`, `fail`, or unresolved; never add `false`, `false_positive`, `true_positive`, or affected intervals.
- A window fail does not fail an asset and does not resolve rejected duration. Persist `duration_resolution_status=not_annotated`.
- Source-frame windows are inclusive; do not transform frame coordinates.
- Do not access real supplier data or hard-code machine paths.

---

### Task 1: Normalize queue and evidence contracts

**Files:**
- Create: `human_qc/sam3_window_review.py`
- Create: `tests/test_sam3_window_review.py`

**Interfaces:**
- Produces: `load_review_bundle(manifest_path, queue_path, evidence_path, review_dir) -> ReviewBundle`
- Produces: queue DTOs keyed by normalized `review_id`, with exact composite fallback only when evidence lacks a review ID.

- [ ] Write tests for stable `review_id`, the no-double-`jdt` invariant, duplicate IDs, exact composite fallback, exact-ID precedence, exclusion of automatic-pass rows, multiple windows per asset, and unresolved defaults.
- [ ] Run the new test module and confirm RED because the module/API does not exist.
- [ ] Implement suffix-aware CSV/Parquet/JSON record loading, NaN normalization, strict inclusive integer windows, and manifest asset validation.
- [ ] Implement exact evidence matching; never match overlaps.
- [ ] Run the test module and confirm GREEN.

### Task 2: Stage only whitelisted evidence

**Files:**
- Modify: `human_qc/sam3_window_review.py`
- Modify: `tests/test_sam3_window_review.py`

**Interfaces:**
- Produces: staged overlay DTOs with `url`, `source_path`, `frame_idx`, provenance, and `status`.
- Produces: a frozen allowlist of staged relative asset paths for the HTTP layer.

- [ ] Write failing tests for five combined overlays, absolute source conversion to controlled URLs, filenames with spaces, missing files, wrong evidence type/source module, and path traversal-resistant staged names.
- [ ] Run those tests and confirm expected RED assertions.
- [ ] Hardlink each regular whitelisted image into `review_dir/assets/<safe-review-token>/`, falling back to `shutil.copy2` only for unsupported cross-device/filesystem links.
- [ ] Preserve original `source_path` only as provenance and mark any missing/non-file input as `missing` without aborting the whole queue.
- [ ] Run the test module and confirm GREEN.

### Task 3: Persist independent window decisions

**Files:**
- Modify: `human_qc/sam3_window_review.py`
- Create: `tests/test_sam3_window_review_server.py`

**Interfaces:**
- Produces: `Sam3WindowReviewStore` with `snapshot()` and `save(review_id, verdict, reviewer, expected_revision)`.
- Persists: `sam3_window_review_state.json` and completed-only `sam3_window_review_results.csv` under `save_dir`.

- [ ] Write failing tests for pass/fail validation, unresolved omission from completed CSV, reviewer/review ID preservation, independent same-asset windows, refresh recovery, optimistic revision conflict, missing-evidence rejection, and `duration_resolution_status=not_annotated`.
- [ ] Run the server/store tests and confirm RED.
- [ ] Implement lock-protected read-modify-write with temporary files, fsync, and `os.replace`; only write cache state after all output text is prepared.
- [ ] Store queue/evidence provenance and server timestamps; never emit rejected frame counts or affected intervals.
- [ ] Run the tests and confirm GREEN.

### Task 4: Serve safe HTTP APIs and overlays

**Files:**
- Create: `human_qc/sam3_window_review_server.py`
- Modify: `tests/test_sam3_window_review_server.py`

**Interfaces:**
- GET `/api/review-bundle`
- POST `/api/reviews/<review_id>`
- GET `/assets/<whitelisted-relative-path>`
- GET `/`, `/static/sam3_window_review.js`, `/static/sam3_window_review.css`

- [ ] Write failing live-server tests for image 200/content-type, unlisted file 404, encoded traversal 404, malformed mutation 400, stale revision 409, missing-evidence conflict, and POST-then-GET recovery.
- [ ] Run and confirm RED.
- [ ] Implement a `ThreadingHTTPServer` handler that serves only package static files and exact staged allowlist entries.
- [ ] Decode URL path components once, reject empty/dot/dot-dot components, and verify resolved staged files remain inside `review_dir/assets`.
- [ ] Run and confirm GREEN.

### Task 5: Add the Pass/Fail browser workbench and CLI

**Files:**
- Create: `human_qc/static/sam3_window_review.html`
- Create: `human_qc/static/sam3_window_review.js`
- Create: `human_qc/static/sam3_window_review.css`
- Create: `tools/serve_sam3_window_review.py`
- Create: `tests/test_sam3_window_review_static_contract.py`
- Modify: `tests/test_sam3_window_review_server.py`

**Interfaces:**
- CLI requires `--manifest --review-queue --evidence-manifest --review-dir --save-dir` and supports `--host --port`.
- Browser submits `{verdict, reviewer, expected_revision}` and reloads authoritative state from the server.

- [ ] Write static contract tests proving only Pass/Fail controls exist and all forbidden legacy terms are absent.
- [ ] Write a Node-backed behavior test for navigation, progress, saved-state restoration, disabled decisions when evidence is missing, and five overlay cards.
- [ ] Run and confirm RED.
- [ ] Implement a no-localStorage browser app showing the required queue, SAM3 provenance, source-inclusive window, per-frame carousel/grid, left/right verdicts, trigger details, reviewer, progress, save state, and previous/next controls.
- [ ] Implement CLI wiring without any fixed run-root assumptions.
- [ ] Run the static/server tests and confirm GREEN.

### Task 6: Regression and final audit

**Files:**
- No production changes unless a failing regression demonstrates a defect in the new files.

- [ ] Run all new SAM3 window-review tests.
- [ ] Run existing legacy manual-review and generic human-QC workbench tests.
- [ ] Run focused JD/QY/DR manifest, precheck, and SAM3 runner tests that exist in the repository.
- [ ] Run `python3 -m compileall tools human_qc acceptance_pull qc_pipeline qc_common`.
- [ ] Run `git diff --check`.
- [ ] Inspect `git status --short`, confirm old manual-review tools and existing QY modifications were not overwritten, and report only fresh verification evidence.
