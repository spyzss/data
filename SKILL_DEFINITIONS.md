# Agent Skill Definitions

These skill definitions are intended for Claude/GPT Project instructions. They
describe when an agent should operate in each part of the repository. They are
documentation-only and do not implement runtime behavior.

## Skill: `marmalade-architecture-guardian`

Use when:

- Reviewing or changing module boundaries.
- Adding a new root module, runner, check, adapter, or shared contract.
- A request mentions workflow order, orchestration, gating, skipping, or
  cross-module dependencies.

Rules:

- Enforce root module separation: `annotation/`, `precheck/`,
  `annotation_verify/`, `qc_common/`.
- Never encode `precheck -> annotation -> verify` as code-level order.
- Keep orchestration external.
- Reject or redesign changes that make `annotation/` depend on `precheck/` or
  `annotation_verify/`.
- Keep `qc_common/` free of pipeline logic, supplier-specific adapters, model
  loading, and runner instantiation.

Expected output:

- Module ownership decision.
- Boundary risk assessment.
- Minimal safe edit plan.

## Skill: `marmalade-annotation`

Use when:

- Working on visual annotation generation.
- Editing discovery, segmentation, depth, storage, or annotation QC.
- Working with SAM3, DA3, masks, depth maps, or LeRobot annotation outputs.

Owned paths:

- `annotation/`
- `pipeline.py`
- `run_annotate.py`
- annotation configs under `configs/`

Rules:

- Do not import from `precheck/`.
- Do not import from `annotation_verify/`.
- Do not run semantic verification.
- Do not run data-trust checks.
- Keep segmentation and depth stage decoupling intact.
- Do not load SAM3 in depth-only paths.
- Do not load DA3 in segmentation-only paths.

Expected output:

- Visual annotation behavior changes only.
- Verification command scoped to annotation behavior.

## Skill: `marmalade-precheck`

Use when:

- Working on data trust, signal quality, keypoint quality, geometric/statistical
  checks, overexposure, mask containment, missing-keypoint detection, or HDF5
  supplier skeleton inputs.

Owned paths:

- `precheck/`
- `run_precheck.py`
- `keypoint_temporal_probe.py`
- precheck configs under `configs/`

Rules:

- No semantic reasoning.
- Do not load SAM3 or DA3.
- Optional masks may be injected; absence of masks must skip mask-dependent
  checks cleanly.
- Add checks by creating a check file and registry entry.
- Keep supplier-specific loading under `precheck/adapters/`.
- Do not treat `confidence == 0` as keypoint missing.
- Use `label/quality_hand` for missing/low-quality hand judgement.

Keypoint rules:

- Default hand acceptance set is 42 joints total, 21 per hand.
- No Metacarpals.
- No body/torso/leg joints.
- `keypoint_temporal` primary signal is rotation delta.
- `keypoint_temporal` secondary signal is joint-angle frame-to-frame change.
- Bone length is rigid-rig sanity only.
- `keypoint_missing` implements the configurable 10s-window/1s-allowed rule.

Expected output:

- Per-check results as `CheckResult` rows.
- No changes to annotation outputs.

## Skill: `marmalade-annotation-verify`

Use when:

- Working on semantic consistency between supplier text and video content.
- Editing verification contracts, config, registry, runner, or VLM stub.

Owned paths:

- `annotation_verify/`
- `run_annotation_verify.py`
- semantic verification configs under `configs/`

Rules:

- VLM logic is stub-only until explicitly requested.
- Do not affect annotation outputs.
- Do not run signal-quality checks.
- Do not import annotation or precheck internals.
- Consume frames and text through `ClipInputs`.

Expected output:

- Verification result rows only.
- Clear TODO if real VLM logic is requested but out of scope.

## Skill: `marmalade-qc-common`

Use when:

- Editing shared contracts, keypoint topology, generic registry helper, or
  result IO helpers.

Owned paths:

- `qc_common/`

Rules:

- Contracts and pure shared utilities only.
- No module orchestration.
- No supplier-specific adapters.
- No model loading.
- No runner creation.
- No imports from `annotation/`, `precheck/`, or `annotation_verify/`.

Expected output:

- Stable shared interfaces.
- Compatibility note for all modules that consume the changed contract.

## Priority Order For Agents

When multiple skills seem relevant:

1. `marmalade-architecture-guardian`
2. The owning module skill
3. `marmalade-qc-common`, only if a shared contract must change

If a request spans multiple modules, state the boundary explicitly and keep each
edit inside its owning module. Do not create an implicit global workflow.
