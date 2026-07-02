# Current Status

## Top-Level

The repository now contains four independent root modules plus one cloud sidecar utility:

- `precheck/`: implemented and smoke-tested for supplier HDF5/text/quality/skeleton checks.
- `tools/sam3_keypoint_containment.py`: implemented as a SAM3-loading cloud utility for sampled mask containment scoring.
- `annotation/`: existing visual annotation pipeline with discovery, SAM3 segmentation, DA3 depth, storage, QC, and stage decoupling.
- `annotation_verify/`: contract and runner stub for future semantic verification.
- `qc_common/`: shared contracts and helper utilities.

The authoritative interface document is [WORKFLOW_INTERFACE.md](WORKFLOW_INTERFACE.md).

Testing policy:

- Default tests stay module-local.
- Temporary coupled tests are allowed under `tests/` or `tools/`.
- Coupled tests must connect modules through config/files/JSON outputs, not cross-module runtime imports.

## Precheck

Implemented checks:

- `text_integrity`
- `quality_score`
- `keypoint_missing`
- `keypoint_temporal`
- `skeleton_quality_score`
- `composite_frame_verdict`
- `mask_containment`
- `overexposure`

Important semantics:

- `quality_hand` is optional supplier signal, not a universal truth source.
- `skeleton_quality_score` is vendor-agnostic and based on geometry.
- `confidence == 0` is not treated as missing.
- Raw continuous metrics are preserved in JSON outputs.
- Heavy models are not loaded by `precheck/`.

Outputs:

```text
check_results.json
clip_aggregates.json
check_results.parquet or check_results.csv
```

## SAM3 Keypoint Containment Sidecar

Implemented script:

```text
tools/sam3_keypoint_containment.py
```

Purpose:

```text
sample mp4 frames -> run SAM3 masks -> project HDF5 hand keypoints -> compute inside-mask ratios
```

Main clip metric:

```text
clip_keypoint_inside_ratio = inside_keypoints / total_expected_keypoints
```

This is intentionally outside `precheck/checks` because it loads SAM3.

## Annotation

The annotation pipeline remains responsible for:

- Discovery
- SAM3 segmentation
- DA3 depth
- Storage
- Annotation QC visualization

Stage decoupling exists:

```bash
python run_annotate.py configs/seg_only.yaml --stage segmentation
python run_annotate.py configs/depth_only.yaml --stage depth
```

Annotation should not read precheck results or assume precheck has run.

## Annotation Verify

Current status:

- `instruction_consistency` is a stub.
- Runner/config/registry are present.
- Real VLM logic is not implemented.

This module must remain semantic-only.

## Validation

Last local validation commands used:

```bash
python -m pytest tests/test_qc_modules_smoke.py
python -m compileall qc_common precheck annotation_verify annotation tools
```

## Known Follow-Ups

- Add real clip loading to `annotation_verify` only when VLM requirements are clear.
- Decide whether SAM3 containment JSON should later be ingested by a batch ledger or by a `ClipInputs.masks` adapter.
- Calibrate skeleton thresholds with confirmed bad samples.
- Keep supplier HDF5/mp4/model files out of git.
