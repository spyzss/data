# Supplier Acceptance Tools

This repository contains two supplier-acceptance workflows beside the
SAM3/DA3 annotation pipeline:

- `acceptance_pull`: validate a supplier manifest, sample IDs by scene/task,
  and pull paired HDF5/video files into a local batch.
- `acceptance_video_quality`: run the low-cost no-reference video prefilter and
  update one QC archive per asset.

The durable per-asset report is:

```text
<batch>/quality_archive/<asset_id>.json
```

Other QC modules are described in `docs/PRD-qc-gated-json.md`. Their source
code is owned by the corresponding module developers; this repository's
current implementation change only covers video QC and its shared config/report
contract.

## Batch Sampling And Pull

Example local config:

```yaml
manifest: XJGT_20260616.xlsx
readme: README.txt
output: sampled/XJGT_20260616
workers: 8
seed: 20260701
sample_ratio: 0.01
hdf5:
  kind: local
  root: hdf5
video:
  kind: local
  root: video
```

Run:

```bash
python run_acceptance_pull.py --config pull.yaml
```

The pull workflow writes:

```text
sampled/XJGT_20260616/
  hdf5/
  video/
  reports/
    id_consistency.csv
    sample_manifest.csv
    pull_report.csv
    summary.json
```

Sampling uses the manifest/video/HDF5 ID intersection, keeps full scene
coverage, and fills the remaining quota with task diversity. `sample_ratio`
defaults to `0.01` and uses `ceil(valid_id_count * sample_ratio)` as the minimum
sample size.

## Manual Inputs And Confirmation Points

The pull workflow intentionally keeps these choices human-owned:

- `output`: local destination for the sampled batch.
- `sample_ratio`: acceptance sampling ratio; default `0.01`.
- `workers`: pull concurrency for the current network and disk; default `8`.
- `seed`: explicit value for a reproducible historical sample, or blank to use
  the run date.
- Local mode: `manifest`, `readme`, `hdf5.root`, and `video.root`.
- OSS mode: `batch_uri` and `region`. Do not put credentials or browser login
  state in YAML.
- Acceptance policy: the reviewed, immutable `qc_config.config_version` used
  to initialize every asset JSON in one pipeline run.

After pulling, review:

- `reports/id_consistency.csv` for missing or extra IDs.
- `reports/pull_report.csv` for failed pull operations.
- `reports/summary.json` for actual sample size and coverage.
- `quality_archive/<asset_id>.json` for the asset's accumulated QC evidence.

## OSS Batch Input

For local OSS testing with `oss-browser2`, keep credentials outside YAML and
logs. Use a batch-level URI:

```yaml
batch_uri: oss://xingjiguitu/egodata/XJGT_20260616
region: beijing
output: sampled/XJGT_20260616
workers: 8
sample_ratio: 0.01
```

The tool derives `<batch prefix>/hdf5` and `<batch prefix>/video`, then downloads
`README.txt` and the first `.xlsx` manifest when they are not provided.

## Unified QC Config

Video QC accepts only the unified versioned config:

```text
configs/qc_acceptance.yaml
```

The immutable copy for the current release is:

```text
configs/qc_acceptance/qc_acceptance_v1.1.0.yaml
```

The video parameters live under:

```yaml
schema_version: qc_acceptance_config_schema.v1
config_version: qc_acceptance_v1.1.0
modules:
  video_quality:
    module_version: video_prefilter_v0.3.2
    parameters:
      decode: {}
      exposure: {}
      sharpness_global: {}
      freeze: {}
      defects: {}
      hdf5_alignment: {}
```

Do not pass a legacy video-only YAML. To test another reviewed config, copy the
whole unified document, bump `config_version`, edit
`modules.video_quality.parameters`, and run with `--config`.

## Video Quality Check

Run the default reviewed config:

```bash
python run_acceptance_video_quality.py --batch sampled/XJGT_20260616
```

Run a different complete unified config:

```bash
python run_acceptance_video_quality.py \
  --batch sampled/XJGT_20260616 \
  --config configs/qc_acceptance/qc_acceptance_v1.1.0.yaml
```

Output:

```text
sampled/XJGT_20260616/
  hdf5/
  video/
  quality_archive/
    <asset_id>.json
```

The command updates only video-owned content in the existing asset JSON and
preserves unknown fields and blocks written by other modules. It increments
`report_revision`, validates `asset_qc_report.v1`, writes a temporary file, and
atomically replaces the archive.

The top-level `qc_config` records the actual loaded config's schema version,
config version, path, and SHA-256 hash. Thresholds and per-issue config versions
are not copied into the asset JSON. Every warn/fail item is one object in the
top-level `issues` array with a stable `issue_id`, `rule_id`, actual value,
operator, boundary value, and context.

Video flow rules:

- `pass`: continue to the next configured module.
- `warn`: keep issue IDs in `manual_review.candidate_issue_ids` and continue.
- `fail`: stop automated QC, route to `batch_statistics`, and do not run later
  high-cost modules.
- `video_quality.flow.exit_gate.continue_to_next_module` is authoritative.
- `video_quality.evaluation.should_run_mask_qc` is a compatibility alias for
  the same video gate and should not become the cross-module orchestrator.

## Video Prefilter Scope

`video_prefilter_v0.3.2` is a practical no-reference prefilter for robot
pretraining video. It checks:

- open/stream/codec/metadata health;
- FPS and minimum display resolution;
- PTS monotonicity, estimated missing frames, interval p99, and maximum gap;
- sampled decode completeness;
- black, over-dark, and over-exposed frames;
- normalized global Laplacian and Tenengrad indicators;
- low-motion, freeze candidates, confirmed freezes, and continuous intervals;
- combined defect-duration ratio;
- video/HDF5 frame-count alignment.

Drop detection prefers real per-frame PTS from `ffprobe`, then PyAV. OpenCV
`CAP_PROP_POS_MSEC` is only a fallback and records
`drop_detection_reliable=false`. `drop_frame_ratio` is based on
`estimated_missing_frames`.

`adjacent_near_duplicate_ratio` is a low-motion indicator, not a reject rule.
Frames 0.5 seconds apart must still be near-duplicates to become a freeze
candidate; 1.0 second is required for confirmed freeze. A confirmed visual
freeze with clear HDF5 keypoint/action/camera-pose motion is recorded as
`video_state_conflict` and evaluated more strictly in critical interaction
windows.

Sharpness is calibrated for content that may later be downsampled to roughly
448x256. The objective is to distinguish edges and objects and reject extreme
blur, not to demand high-definition imagery. Hand ROI quality is not computed.

Video QC does not perform keypoint accuracy validation, keypoint-mask matching,
trajectory checks, mask IoU, semantic consistency, or subtask acceptance.

## Contracts

- Asset JSON format: `docs/asset-qc-json-format.md`
- Gate and colleague implementation PRD: `docs/PRD-qc-gated-json.md`
- Unified config PRD: `docs/PRD-qc-unified-config.md`
- Config schema: `schemas/qc_acceptance_config.v1.schema.json`
- Asset report schema: `schemas/asset_qc_report.v1.schema.json`
