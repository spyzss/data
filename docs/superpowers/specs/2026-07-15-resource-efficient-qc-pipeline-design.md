# Resource-Efficient QC Pipeline Design

**Date:** 2026-07-15

**Branch:** `codex/human-qc-impl`

**Status:** Implemented and audited; see `docs/reviews/2026-07-15-resource-efficient-qc-pipeline-audit.md`

Implementation note: the final precheck implementation shares one decoded source
through a per-asset session but still constructs lightweight module-specific
`PrecheckRunner` instances on demand. This preserves real acceptance fail-fast
behavior while eliminating the repeated source loads that motivated the change.

## 1. Problem statement

The current unified QC entry point gives one asset a shared report and a fixed module order, but its producer execution is inefficient and partly coupled through the wrong interface:

- The five precheck modules each construct a separate `PrecheckRunner` and reload the same JD Parquet or DeepReach HDF5 source.
- `keypoint_temporal` produces candidate-window information in memory and in the v2 report, while the SAM3 runner ignores it and requires a pre-existing `candidate_windows_path` in the input manifest.
- Existing outputs are not validated and reused through a stable cache contract, so rerunning an asset can repeat precheck, video-quality, and SAM3 work unnecessarily.
- Producer outputs and the v2 report are not cleanly separated. The report can contain evidence paths that do not correspond to durable files.
- Runtime states such as missing input, unsupported adapter, model unavailable, and not executed are not represented consistently enough to distinguish them from quality failures.

The goal is to keep the convenient one-command pipeline while restoring explicit file boundaries and making computation resource-aware. This design does not turn all checks into one monolithic module: producers exchange durable sidecars, and each QC module retains its own raw status and verdict.

## 2. Goals

1. Load each asset's precheck source once per producer run and execute all configured precheck checks through that shared session.
2. Preserve five independent precheck module results in the v2 report.
3. Persist canonical producer sidecars before adapting them into the report.
4. Make SAM3 containment consume the canonical candidate-window artifact produced by the current temporal run.
5. Reuse a producer sidecar only when its inputs, configuration, and implementation identity still match.
6. Avoid loading the SAM3 model when there are no candidate windows.
7. Preserve the behavioral distinction between acceptance and supplier-evaluation profiles without treating missing work as pass.
8. Continue processing independent assets and modules after record-level failures when the selected profile permits it.
9. Keep standalone manifest runners usable as explicit file-contract tools.
10. Provide enough run metadata to estimate cloud runtime and explain which work was reused or recomputed.

## 3. Non-goals

- This change does not implement the DeepReach head projection/calibration adapter. DeepReach SAM3 remains explicitly blocked or adapter-missing until its calibration lineage is validated.
- This change does not unify JDT, XJGT, and DeepReach raw schemas. Supplier adapters remain separate.
- This change does not make precheck import or load SAM3, DA3, Qwen, or other VLMs.
- This change does not replace the existing batch ledger, manual-review queue, or acceptance-workbook tools.
- This change does not use candidate-window duration as rejected duration.
- This change does not run cloud data locally and does not add supplier data or generated outputs to Git.

## 4. Architecture

### 4.1 File-level workflow

```text
canonical manifest
  |
  +--> precheck producer session
  |      - load source once
  |      - execute configured checks in order
  |      - write check results, aggregates, candidates, run config
  |      - adapt five independent module blocks into v2 report
  |
  +--> video-quality producer
  |      - validate/reuse or compute sidecar
  |      - adapt independent module block into v2 report
  |
  +--> SAM3 containment producer
         - read current precheck candidate artifact
         - skip without model load when the artifact is valid and empty
         - validate/reuse or compute containment sidecar
         - adapt independent module block into v2 report

durable module sidecars
  +--> per-asset v2 report
  +--> existing ledger/manual-review consumers
```

The entry point orchestrates producers, but producers communicate through files rather than hidden Python state. In-memory values may be used within a producer session, but a downstream producer must be able to reproduce its input from the recorded sidecar path.

### 4.2 Artifact layout

For a pipeline output root, durable outputs use the following structure:

```text
module_outputs/<asset_id>/
  precheck/
    check_results.json
    clip_aggregates.json
    candidate_windows.json
    run_config.json
  video_quality/
    video_quality_result.json
    run_config.json
  sam3_containment/
    frame_results.jsonl
    window_results.jsonl
    evidence/
    run_config.json

quality_archive/<asset_id>.json
```

An empty but valid candidate set is represented by an existing `candidate_windows.json` containing an empty JSON list. Absence of the file means the producer contract was not completed; it must not be interpreted as “no candidates.”

Producer output is written transactionally: files are prepared in a temporary sibling directory and promoted only after the full artifact validates. A failed recomputation must not overwrite an older valid artifact.

### 4.3 Shared precheck session

The precheck pipeline runner changes from “one source load per module” to “one source load per asset producer run.” The shared session:

1. Resolves the supplier adapter and manifest slice.
2. Loads the source clip once.
3. Constructs one `PrecheckRunner` with the checks required by the selected profile/configuration.
4. Executes checks in the configured order.
5. Persists all raw results and candidate-window records.
6. Adapts each check result into its existing independent v2 module block.

This optimization shares input decoding and producer setup, not verdicts. `hdf5_text_info`, `quality_hand`, `keypoint_presence`, `keypoint_morphology`, and `keypoint_temporal` remain separately addressable module states with their own reasons, metrics, thresholds, and evidence.

For acceptance mode, a quality failure may stop later scheduled checks. Every unexecuted required check is recorded as `not_run` with a blocking reason; it is never synthesized as pass. For supplier-evaluation mode, a check failure or check-level runtime error does not prevent independent later checks from executing.

## 5. Candidate-window and SAM3 contract

### 5.1 Full pipeline

In the full pipeline, the only candidate input accepted by SAM3 is the canonical artifact generated or validated for the current precheck run:

```text
module_outputs/<asset_id>/precheck/candidate_windows.json
```

The manifest's historical `candidate_windows_path` is not used to override this artifact. This removes the ambiguous case where the report describes fresh temporal candidates but SAM3 evaluates a stale file from a previous run.

The SAM3 runner validates that each candidate:

- belongs to the current `asset_id`;
- has `start_frame <= end_frame`;
- declares or can be normalized to source-frame coordinates;
- remains within the inclusive manifest clip bounds;
- has not already received an extra local-to-source offset.

Invalid records are written to failure output and yield `input_invalid`; they are not silently clamped.

### 5.2 Standalone runner

`tools/run_manifest_sam3_containment.py` remains a standalone file-contract runner and keeps an explicit candidate-window input option. This supports manual or external orchestration without importing the unified pipeline.

### 5.3 Empty candidates

When a valid candidate artifact contains zero records, SAM3 returns a non-error terminal state equivalent to `skipped/no_candidates`. The implementation must decide this before constructing or loading the segmenter. This state means containment work was unnecessary; it does not change the raw temporal status.

## 6. Cache and invalidation

### 6.1 Run-config fingerprint

Every producer writes a versioned `run_config.json` containing JSON-safe values sufficient to explain reuse. The common fingerprint includes:

- schema version and producer name;
- canonical `asset_id` and supplier;
- source path recorded relative to an allowed root when possible;
- inclusive source-frame start and end;
- source size and `mtime_ns` for local files;
- supplier checksum, object ETag, or version ID when the adapter provides one;
- normalized module configuration and its SHA-256 hash;
- implementation/module version;
- creation time and producer outcome.

SAM3 additionally records:

- SHA-256 of the exact candidate-window artifact;
- video identity and supplier-2D-keypoint identity;
- model/checkpoint identity;
- prompt queries and containment thresholds.

File modification time alone is not treated as a strong remote-data identity. A supplied checksum or ETag takes precedence when available.

### 6.2 Reuse rules

With resume enabled, a sidecar is reused only if:

1. its required files exist and pass schema validation;
2. the previous producer outcome is reusable;
3. every current fingerprint field matches;
4. all evidence paths referenced by the artifact remain available where required.

Any mismatch causes that producer to recompute. A precheck invalidation invalidates downstream SAM3 reuse because the candidate SHA changes or is revalidated. Video-quality remains independent and is not invalidated by a precheck-only configuration change.

With resume disabled, producers recompute even if a matching artifact exists. Transactional promotion still protects the previous valid artifact from a failed new attempt.

## 7. Status and profile semantics

### 7.1 Producer/module states

The report contract must distinguish at least these conditions:

| Condition | Module state/status | Quality verdict implication |
|---|---|---|
| Check completed and met policy | `completed/pass` | pass evidence |
| Check completed and violated policy | `completed/fail` | fail evidence |
| Valid empty candidate set | `skipped/no_candidates` | no SAM3 quality evidence required |
| Required input absent | `input_missing` | not ready; never pass |
| Input present but invalid | `input_invalid` | not ready/review according to policy |
| Supplier adapter unavailable | `adapter_missing` or `blocked` | not ready; never pass |
| Model/runtime dependency unavailable | `blocked` | not ready; never pass |
| Producer crashed | `runtime_error` | not ready; preserve failure reason |
| Required module stopped by profile | `not_run` with blocker | not ready; never pass |
| Module genuinely not required | `not_applicable` | excluded by explicit policy |

Raw statuses are immutable evidence. Final verdict calculation reads them but does not rewrite them.

### 7.2 Acceptance profile

Acceptance prioritizes fast gating. After a configured hard quality failure, later expensive work that has not already produced a valid reusable artifact is not scheduled. Required modules that were not run are recorded as `not_run`, and the asset cannot become pass through their absence. Missing, blocked, or runtime-error required work produces a not-ready/review outcome rather than a fabricated quality fail.

### 7.3 Supplier-evaluation profile

Supplier evaluation prioritizes complete diagnostic coverage. A module quality fail or module-level runtime error is recorded and independent later producers continue. Completed quality failures still contribute fail evidence to the final machine verdict. A required module that remains missing, blocked, invalid, or not run leaves the overall readiness/verdict unresolved according to ledger policy; it is not converted to pass.

This is the recommended profile for the user's current temporary supplier-facing run because it measures all available checks and exposes coverage gaps.

## 8. Supplier boundaries

- JDT continues to read 2D hand keypoints directly from Parquet. No fake camera calibration is introduced.
- DeepReach source parsing and head-video selection remain supplier-adapter responsibilities.
- Until DeepReach head intrinsics/extrinsics and projection lineage are implemented and validated, DeepReach SAM3 containment reports `adapter_missing`/`blocked` with an actionable reason. It must not report pass or skeleton fail.
- XJGT/JDT/DeepReach source identities and raw schemas remain separate; only canonical manifest and sidecar contracts are shared.

## 9. Human QC and ledger compatibility

The optimization does not redefine manual-review semantics:

- Queue-to-label matching prefers normalized non-empty `review_id`.
- Exact `asset_id + window_start_frame + window_end_frame` is only a fallback.
- One review may produce multiple `affected_segments`.
- Rejected/review duration comes only from confirmed affected intervals, after merging overlaps per asset.
- Candidate windows are evidence intervals, not rejected duration.
- `acceptable_flagged` and `false_positive` remain calibration evidence rather than problem frames.

The v2 report may reference new durable sidecar paths, but it does not replace raw module outputs or silently change the existing ledger schema. Any necessary ledger adapter change must be tested as an explicit compatibility change.

## 10. CLI behavior

The unified entry point retains explicit profile selection and adds or formalizes:

- an output root for `module_outputs/` and `quality_archive/`;
- resume/reuse behavior;
- a force/no-resume option for recomputation;
- stable reporting of `reused`, `computed`, `skipped`, `blocked`, and `failed` producer counts;
- per-producer elapsed time and total asset elapsed time in run metadata.

The command does not require manifest `candidate_windows_path` for full-pipeline SAM3. If that legacy field is present, it is ignored by the full pipeline and may be reported as deprecated input metadata; it remains usable only through the standalone SAM3 workflow where explicitly selected.

## 11. Error handling

- A record-level error is written with asset, module, exception category, and actionable reason; processing continues for other assets.
- Supplier-evaluation mode also continues to independent later modules for the same asset.
- Acceptance mode may stop scheduling later modules according to its gate policy, but it must materialize their `not_run` states.
- JSON serialization converts NumPy scalars and arrays to JSON-safe Python values.
- Evidence paths stored in reports are relative to the declared batch root and must resolve to actual promoted artifacts unless the schema explicitly marks them external.
- Partial temporary output is never advertised as a completed sidecar.

## 12. Implementation surface

Expected changes are deliberately concentrated at producer/adaptation boundaries:

- add `qc_pipeline/artifacts.py` for artifact paths, fingerprints, validation, and transactional promotion;
- refactor `qc_pipeline/runners/precheck.py` into one shared per-asset producer session;
- update `qc_pipeline/adapters/precheck.py` to reference real canonical artifacts;
- add reuse/invalidation integration to `qc_pipeline/runners/video_quality.py`;
- make `qc_pipeline/runners/sam3_containment.py` read the current precheck artifact and short-circuit empty candidates;
- update `qc_pipeline/orchestrator.py` for profile-specific scheduling and explicit unexecuted states;
- extend `qc_common/contracts.py`, `qc_common/report_mutation.py`, and the v2 schema only as needed for the status vocabulary and evidence validation;
- update `tools/run_qc_pipeline.py` and versioned configs for output-root/resume behavior;
- preserve standalone runner interfaces unless a backward-compatible option is added.

Parallel implementations of ledgers, review queues, or supplier schemas are out of scope.

## 13. Test strategy

Implementation follows test-driven development. Tests must demonstrate behavior, not only internal call structure.

### Precheck execution

- All five supplier-evaluation checks load one source exactly once.
- Acceptance hard failure prevents later check execution and records each unexecuted required module as `not_run`.
- Supplier-evaluation failure does not prevent later independent checks.
- All executed checks retain separate module result blocks.

### Candidate/SAM3 handoff

- Temporal output always writes a canonical candidate JSON file, including the empty-list case.
- Full-pipeline SAM3 reads the current canonical artifact and ignores a stale manifest legacy path.
- Candidate content changes invalidate SAM3 reuse by SHA-256.
- Empty candidates return `skipped/no_candidates` without constructing the segmenter.
- Corrupt or out-of-bounds candidates return `input_invalid` and are not clamped.

### Cache behavior

- Matching artifacts make producer call counts zero during resume.
- Source identity, frame range, configuration, implementation version, and candidate changes invalidate only the affected producers and dependents.
- A failed recomputation leaves the previous valid promoted artifact intact.

### Profiles and statuses

- Both profiles cover quality fail, runtime error, missing input, adapter missing, blocked, and not-run paths.
- Required incomplete modules cannot produce overall pass.
- SAM3 blocked does not fail unrelated assets or the entire batch.
- Raw module statuses survive final verdict calculation unchanged.

### Supplier and compatibility regression

- JDT direct Parquet 2D-keypoint behavior and inclusive source coordinates remain correct.
- DeepReach SAM3 remains explicit blocked/adapter-missing until its adapter exists.
- Standalone manifest precheck, video-quality, and SAM3 runners retain their file-contract behavior.
- Existing manual-review, review-id matching, affected-segment duration, ledger, and v2 schema tests remain green.

## 14. Verification and final audit

After implementation, run the repository's targeted pipeline, manifest-runner, ledger/manual-review, smoke, compile, and `git diff --check` validations. Model-dependent cloud execution is not claimed from this local environment.

The final review must separately report:

1. eliminated repeated source loads and model-load avoidance;
2. remaining computation that is intentionally repeated;
3. residual orchestration coupling and artifact/schema risks;
4. human-QC, `review_id`, `affected_segments`, and duration semantics;
5. compatibility between v2 reports and existing ledgers;
6. JDT/JD video-level manifest implications;
7. the remaining DeepReach head SAM3 adapter gap;
8. cloud timing fields and commands the user can use to estimate the five-check runtime.

## 15. Acceptance criteria

The design is complete when all of the following are true:

- One asset precheck run performs at most one raw source load for its configured checks.
- Five precheck checks remain five independently reported module outcomes.
- Full-pipeline SAM3 consumes the current precheck candidate artifact, never a legacy manifest path.
- Valid empty candidates avoid SAM3 model initialization.
- Resume uses only fingerprint-valid sidecars and recomputes stale ones.
- Failed recomputation cannot destroy the previous valid artifact.
- Acceptance and supplier-evaluation scheduling/status semantics are covered by tests.
- Missing, blocked, adapter-missing, input-invalid, runtime-error, and not-run are never serialized as pass.
- Supplier boundaries and existing manual-review/ledger semantics remain intact.
- Local verification passes, with cloud/model-dependent gaps stated explicitly.
