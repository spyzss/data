# Implementation Summary

This file is a high-level implementation index. For the canonical workflow and interfaces, use [WORKFLOW_INTERFACE.md](WORKFLOW_INTERFACE.md).

## Implemented Modules

### `precheck/`

Data-trust and signal-quality package:

- HDF5 adapter: `precheck/adapters/supplier_hdf5.py`
- Checks:
  - `text_integrity`
  - `quality_score`
  - `keypoint_missing`
  - `keypoint_temporal`
  - `skeleton_quality_score`
  - `composite_frame_verdict`
  - `mask_containment`
  - `overexposure`
- Runner output:
  - `check_results.json`
  - `clip_aggregates.json`
  - parquet/csv fallback

### `tools/sam3_keypoint_containment.py`

Cloud-side utility for mentor-facing containment validation:

```text
HDF5 + mp4 + SAM3 -> sampled masks -> projected 21 hand keypoints -> clip inside ratio
```

The script outputs frame-level and clip-level JSON. It loads SAM3, so it intentionally stays outside `precheck/`.

### `annotation/`

Visual annotation pipeline:

- discovery
- SAM3 segmentation
- DA3 depth
- mask/depth storage
- annotation QC visualization
- segmentation/depth stage decoupling

Implementation details remain in:

- [SAM3_IMPLEMENTATION.md](SAM3_IMPLEMENTATION.md)
- [DA3_IMPLEMENTATION.md](DA3_IMPLEMENTATION.md)
- [DA3_DIAGNOSIS_REPORT.md](DA3_DIAGNOSIS_REPORT.md)

### `annotation_verify/`

Semantic verification scaffold:

- config
- registry
- runner
- `instruction_consistency` stub

Real VLM logic is intentionally not implemented yet.

### `qc_common/`

Shared model-free contracts and helpers:

- `ClipInputs`
- `CheckResult`
- official 21-per-hand keypoint topology
- registry helpers
- JSON/parquet result helpers
- safe HDF5 scalar JSON loader

## Validation

```bash
python -m pytest tests/test_qc_modules_smoke.py
python -m compileall qc_common precheck annotation_verify annotation tools
```
