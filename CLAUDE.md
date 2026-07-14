# Marmalade Annotation Agents - Execution Contract

This repository is a multi-module robotics vision annotation system. Agents must
follow the current architecture below and must not assume the old single
`annotation/`-only design.

## Architecture

Root modules:

- `annotation/` - visual annotation only
- `precheck/` - data trust and signal-quality checks only
- `annotation_verify/` - semantic consistency verification only
- `qc_common/` - shared contracts, schemas, metrics helpers, and IO utilities only

There is no enforced workflow order in code. A possible external workflow may
run precheck, annotation, and verification in sequence, but that orchestration
belongs outside these modules.

## Hard Boundaries

### `annotation/`

Responsibility:

- Visual annotation pipeline only.
- Discovery, segmentation, depth, storage, and annotation QC visualization.
- SAM3 and DA3 model integration belongs here.

Allowed:

- Load annotation datasets.
- Run discovery, segmentation, depth, storage.
- Write masks/depth and annotation manifests.

Forbidden:

- Do not import from `precheck/`.
- Do not import from `annotation_verify/`.
- Do not run data-trust checks.
- Do not run semantic verification.
- Do not treat `precheck -> annotation -> verify` as an internal workflow.

### `precheck/`

Responsibility:

- Data trust, signal quality, and statistical/geometric checks only.
- Current checks: `overexposure`, `keypoint_temporal`, `keypoint_missing`,
  `mask_containment`.

Allowed:

- Consume raw frames, supplier skeleton HDF5 inputs, optional injected masks,
  camera intrinsics, FPS, and supplier quality signals.
- Use adapters under `precheck/adapters/` for supplier-specific loading.
- Skip checks when optional inputs are absent.

Forbidden:

- Do not perform semantic reasoning.
- Do not decide whether an instruction matches video content.
- Do not load SAM3 or DA3.
- Do not depend on `annotation/` runtime internals.
- Do not correct, snap, or repair labels; detection and reporting only.

### `annotation_verify/`

Responsibility:

- Semantic consistency checks only.
- Compare supplier instruction/description with video content.
- VLM call is a stub for now.

Allowed:

- Define verification contracts, config, registry, runner, and output rows.
- Accept frames/instruction through `ClipInputs`.

Forbidden:

- Do not implement real VLM/model logic until explicitly requested.
- Do not modify annotation outputs.
- Do not run signal-quality checks.
- Do not import from `annotation/` or `precheck/` internals.

### `qc_common/`

Responsibility:

- Shared contracts and reusable definitions only.
- `CheckResult`, `ClipInputs`, `BaseCheck`, generic registry helper, official
  hand keypoint schema/topology, result IO helpers.

Allowed:

- Define schemas, typed contracts, shared constants, and pure helper utilities.

Forbidden:

- Do not contain pipeline orchestration.
- Do not contain supplier-specific loading adapters.
- Do not import from `annotation/`, `precheck/`, or `annotation_verify/`.
- Do not instantiate checks or runners.
- Do not load models.

## Execution Priority Rules

When asked to make changes:

1. Identify the target module by responsibility before editing.
2. Preserve module independence.
3. Prefer config-driven behavior over hardcoded paths or thresholds.
4. Add new checks through registry entries; do not change runners for each new
   check unless the runner contract itself changes.
5. Keep optional inputs optional. Missing optional inputs should skip the check,
   not fail the clip.
6. Do not introduce a global workflow order.
7. Do not add shared global state.
8. Do not move model loading outside `annotation/`.
9. Do not use confidence as keypoint absence/missing signal.
10. If a requested change crosses module boundaries, stop and explain the
    boundary issue before editing.

## Failure Policy

All independent runners must isolate failures:

- A failed frame must not crash the full clip or job when frame-level isolation
  is possible.
- A failed check must be reported without preventing other checks from running.
- Missing optional inputs should produce no result rows for that check unless a
  check-specific contract says otherwise.
- Uncalibrated metrics must emit `flag=None`.
- Only explicitly specified customer rules may set `flag=True` or `flag=False`.
- Do not silently infer thresholds from observed data.
- Output rows must remain keyed by `(episode_idx, frame_idx)` when frame-aligned
  results are emitted.

## Keypoint System

Use the supplier's official hand acceptance topology. This is not MediaPipe,
OpenPose, or any external standard.

Each hand has exactly 21 acceptance keypoints:

- `Hand`
- Thumb: `ThumbKnuckle`, `ThumbIntermediateBase`, `ThumbIntermediateTip`,
  `ThumbTip`
- Index: `IndexFingerKnuckle`, `IndexFingerIntermediateBase`,
  `IndexFingerIntermediateTip`, `IndexFingerTip`
- Middle: `MiddleFingerKnuckle`, `MiddleFingerIntermediateBase`,
  `MiddleFingerIntermediateTip`, `MiddleFingerTip`
- Ring: `RingFingerKnuckle`, `RingFingerIntermediateBase`,
  `RingFingerIntermediateTip`, `RingFingerTip`
- Little: `LittleFingerKnuckle`, `LittleFingerIntermediateBase`,
  `LittleFingerIntermediateTip`, `LittleFingerTip`

Both hands total 42 acceptance keypoints.

Rules:

- No Metacarpal joint is an acceptance keypoint.
- No body, torso, leg, or non-hand joint is an acceptance keypoint.
- Metacarpal transforms may exist in supplier data but must be excluded from
  default keypoint metrics.
- `confidence == 0` is not a missing/absence signal.
- Missing/low-quality judgement is based on `label/quality_hand`.

## Check Semantics

### `keypoint_temporal`

Purpose:

- Detect temporal drift/jitter in supplier hand skeleton signals.

Primary signal:

- Per-joint rotation-block frame-to-frame delta.

Secondary signal:

- Joint-angle frame-to-frame change.

Sanity-only signal:

- Bone length and bone-length change. The supplier skeleton is a rigid rig, so
  bone length variance is expected to be near zero and is not a drift signal.

Flag policy:

- Thresholds are uncalibrated. Emit raw metrics with `flag=None`.

### `keypoint_missing`

Purpose:

- Apply the customer acceptance rule for missing/low-quality hand keypoints.

Rule:

- In any configurable 10-second window, low-quality/missing frames must not
  exceed configurable 1 second total.

Signal:

- Use `label/quality_hand` per hand.
- `quality_hand < 0.5` means that hand is low-quality for that frame.

Flag policy:

- This check may set flags because the customer supplied the threshold.

### `mask_containment`

Purpose:

- Optional gross-error gate for projected keypoints against injected hand masks.

Rules:

- If no mask is injected, skip cleanly.
- Measure only. Do not move or repair keypoints.

### `instruction_consistency`

Purpose:

- Semantic consistency between supplier text and video content.

Current status:

- Stub only. Do not implement real VLM logic unless explicitly requested.

## Runner Requirements

Every root module runner must be independently executable:

- `run_annotate.py` / annotation runner for visual annotation.
- `run_precheck.py` / `precheck.runner.PrecheckRunner` for data trust checks.
- `run_annotation_verify.py` / `annotation_verify.runner.AnnotationVerifyRunner`
  for semantic verification.

Runners must not assume another module has already run. Optional products such
as masks must be injected through `ClipInputs` or configuration, and checks that
require absent optional inputs must skip cleanly.

## Output Contracts

`CheckResult` fields:

- `check: str`
- `episode_idx: int`
- `frame_idx: int`
- `metrics: dict[str, float]`
- `flag: bool | None`
- `reason: str`

`ClipInputs` is the shared optional input bundle. Fields may be absent or lazily
loaded. Checks must verify required inputs locally.

## Agent Editing Rules

Before editing:

- State which module owns the requested change.
- State whether the request risks crossing module boundaries.

When editing:

- Keep changes minimal.
- Do not refactor unrelated modules.
- Do not introduce hidden coupling.
- Do not add new dependencies unless required by the owning module.
- Do not modify generated outputs.

Before finishing:

- Run the narrowest relevant smoke/compile check if code changed.
- Report skipped verification and why.
- Report any files intentionally left untouched.

## Files Agents Should Load First

For architecture/context:

- `CLAUDE.md`
- `qc_common/types.py`
- `qc_common/base.py`
- `qc_common/keypoints.py`

For precheck work:

- `precheck/config.py`
- `precheck/runner.py`
- `precheck/registry.py`
- Relevant `precheck/checks/*.py`
- Relevant `precheck/adapters/*.py`

For annotation work:

- `annotation/config.py`
- `pipeline.py`
- Relevant `annotation/*` submodule

For semantic verification work:

- `annotation_verify/config.py`
- `annotation_verify/runner.py`
- Relevant `annotation_verify/checks/*.py`

## Current Priority

Keep the architecture clean while adding checks incrementally. The most common
mistake is to reintroduce the old single-pipeline mental model. Do not do that.
