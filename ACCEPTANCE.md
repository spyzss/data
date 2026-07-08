# Supplier Acceptance Tools

This repository also includes two supplier-acceptance workflows that sit beside
the SAM3/DA3 annotation pipeline:

- `acceptance_pull`: validate a supplier manifest, sample IDs by scene/task,
  and pull paired HDF5/video files into a local batch.
- `acceptance_video_quality`: run no-reference video quality checks on a sampled
  batch and write per-asset QC JSON archives.

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
coverage, and fills remaining quota with task diversity. `sample_ratio` defaults
to `0.01`, using `ceil(valid_id_count * sample_ratio)` as the minimum size.

## Manual Inputs And Confirmation Points

The workflow intentionally keeps several choices as human-owned inputs:

- `output`: choose the local destination directory for the sampled batch.
- `sample_ratio`: choose the acceptance sampling ratio. The default is `0.01`
  for 1%; set values such as `0.02` or `0.005` when the batch policy changes.
- `workers`: choose pull concurrency for the current network and disk
  environment. The default is `8`.
- `seed`: leave blank to use the run date, or set an explicit value when a
  historical sample must be reproduced.
- Local source mode: provide `manifest`, `readme`, `hdf5.root`, and
  `video.root`.
- OSS source mode: provide `batch_uri` and `region`; do not put access keys,
  tokens, or browser login state in YAML.
- Video quality mode: confirm `hdf5_alignment.mode`, threshold overrides, and
  the `decision` / `should_run_mask_qc` pipeline flags before using the result
  as an acceptance gate.

After each run, a reviewer should check:

- `reports/id_consistency.csv` for missing or extra manifest/HDF5/video IDs.
- `reports/pull_report.csv` for failed pull operations.
- `reports/summary.json` for actual sample size, scene/task coverage, seed,
  worker count, and sampling ratio.
- `quality_archive/<asset_id>.json` for per-asset machine-readable evidence
  when a sample needs closer review.

## OSS Batch Input

For local OSS testing with `oss-browser2`, keep credentials outside YAML and log
output. Open `oss-browser2`, sign in, then use a business-level batch URI:

```yaml
batch_uri: oss://xingjiguitu/egodata/XJGT_20260616
region: beijing
output: sampled/XJGT_20260616
workers: 8
sample_ratio: 0.01
```

The tool derives:

```text
hdf5 prefix: <batch prefix>/hdf5
video prefix: <batch prefix>/video
```

and downloads `README.txt` plus the first `.xlsx` manifest from the batch root
when they are not provided explicitly.

## Video Quality Check

After a batch is sampled and pulled:

```bash
python run_acceptance_video_quality.py --batch sampled/XJGT_20260616
```

Optional config:

```yaml
threshold_version: video_prefilter_v0.2.9
decode:
  max_sample_frames: 300
hdf5_alignment:
  mode: fail
resolution:
  min_short_side_fail: 720
  min_long_side_fail: 1280
exposure:
  black:
    max_frame_count_fail: 10
    ratio_pass: 0.01
    ratio_warn: 0.90
  over_dark:
    ratio_pass: 0.05
    ratio_warn: 0.90
  over_exposed:
    ratio_pass: 0.05
    ratio_warn: 0.90
sharpness_global:
  target_short_side: 720
  laplacian_p10_pass: 35
  laplacian_p10_warn: 0
  laplacian_median_pass: 50
  laplacian_median_warn: 0
  laplacian_under_100_ratio_pass: 0.50
  laplacian_under_100_ratio_warn: 1.00
  tenengrad_p10_pass: 12
  tenengrad_p10_warn: 6
  tenengrad_median_pass: 13
  tenengrad_median_warn: 8
freeze:
  frozen_frame_ratio_pass: 0.05
  frozen_frame_ratio_warn: 0.10
  min_interval_frames: 6
defects:
  max_duration_ratio_fail: 0.10
  duration_ratio_warn: 0.05
hand_roi:
  enabled: false
  mode: warn_except_severe_fail
```

The video check writes:

```text
sampled/XJGT_20260616/
  hdf5/
  video/
  quality_archive/
    <asset_id>.json
```

The video quality command writes each asset's QC result into
`quality_archive/<asset_id>.json`, alongside `hdf5/` and `video/`. The schema is
documented in `docs/asset-qc-json-format.md`; future batch-level summaries or
tables can be generated from these archive files. The video block stores
`decision: pass|warn|fail` and `should_run_mask_qc`; downstream high-cost QC
should run only when `should_run_mask_qc` is `true`.

The batch pull workflow still writes pull/sampling reports under:

```text
sampled/XJGT_20260616/
  reports/
    id_consistency.csv
    sample_manifest.csv
    pull_report.csv
    summary.json
```

It is a `video_prefilter_v0.2.9` low-cost prefilter. It uses practical
no-reference indicators: open/decode health, fps, resolution, timeline
continuity, sampled-frame decode ratio, black/over-dark/over-exposure ratios,
global sharpness at a normalized short side, frozen-frame risk, drop-frame risk,
total defect-duration ratio, HDF5 frame-count alignment, and continuous frozen
intervals with at least 6 frames for downstream trimming. Hand ROI is disabled by
default in this prefilter because the rough bbox is too noisy for gating. It is
calibrated for robot pretraining videos that may be downsampled to low resolution,
so sharpness mostly produces warnings unless edges are nearly unreadable. It does
not perform keypoint accuracy validation, keypoint-mask matching,
hand-object mask IoU, trajectory jump checks, semantic consistency, or subtask
acceptance.
