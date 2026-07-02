# Project Structure

当前仓库按模块边界组织，而不是单一 annotation pipeline。

```text
marmalade_annotation/
├── qc_common/                 # shared contracts and pure helpers
│   ├── types.py               # ClipInputs, CheckResult
│   ├── keypoints.py           # official 21-per-hand acceptance topology
│   ├── io.py                  # dataframe/json result writers
│   └── registry.py
│
├── precheck/                  # data trust and signal-quality checks
│   ├── adapters/
│   │   └── supplier_hdf5.py   # supplier HDF5 -> ClipInputs
│   ├── checks/
│   │   ├── text_integrity.py
│   │   ├── quality_score.py
│   │   ├── keypoint_missing.py
│   │   ├── keypoint_temporal.py
│   │   ├── skeleton_quality_score.py
│   │   ├── composite_frame_verdict.py
│   │   ├── mask_containment.py
│   │   └── overexposure.py
│   ├── config.py
│   ├── registry.py
│   └── runner.py
│
├── annotation/                # visual annotation only
│   ├── config.py
│   ├── lerobot_v3_dataset.py
│   ├── discovery/
│   ├── segmentation/
│   ├── depth/
│   ├── storage/
│   └── qc/
│
├── annotation_verify/         # semantic verification contract
│   ├── checks/
│   │   └── instruction_consistency.py
│   ├── config.py
│   ├── registry.py
│   └── runner.py
│
├── tools/
│   └── sam3_keypoint_containment.py
│
├── configs/
│   ├── precheck_example.yaml
│   ├── annotation_verify_example.yaml
│   ├── anygrasp_full.yaml
│   ├── seg_only.yaml
│   └── depth_only.yaml
│
├── run_precheck.py
├── run_annotate.py
├── run_annotation_verify.py
├── run_dryrun.py
│
├── WORKFLOW_INTERFACE.md      # top-level workflow/interface contract
├── PRECHECK_INTERFACE.md      # precheck details
├── README.md
├── QUICKSTART.md
└── STATUS.md
```

## Boundary Summary

| Path | Owns | Must Not Own |
| --- | --- | --- |
| `precheck/` | HDF5/text/quality/skeleton/mask-consumption checks | SAM3/DA3/VLM loading |
| `annotation/` | discovery/segmentation/depth/storage/QC | precheck verdicts, semantic verification |
| `annotation_verify/` | semantic consistency rows | signal-quality checks |
| `qc_common/` | shared contracts and pure helpers | root-module runtime internals |
| `tools/` | external workflow utilities | implicit root-module coupling |

## Runtime Outputs

All generated outputs should stay under `outputs/` or another ignored path.

```text
outputs/
├── precheck_example/
│   ├── check_results.json
│   └── clip_aggregates.json
├── sam3_keypoint_containment/
│   ├── frame_keypoint_containment.json
│   └── clip_keypoint_containment.json
└── anygrasp_full/
    ├── masks.parquet
    ├── depth/
    ├── sampling_manifest.parquet
    └── qc/
```

## Extension Rules

- New data-trust checks go under `precheck/checks/` and register via `precheck/registry.py`.
- New shared schemas/helpers go under `qc_common/` only if they are model-free and supplier-neutral.
- New semantic checks go under `annotation_verify/checks/`.
- New model-backed sidecar tools can live under `tools/`, but must document their inputs/outputs clearly.
