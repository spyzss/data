# Supplier Acceptance Tools

This repository also includes two supplier-acceptance workflows that sit beside
the SAM3/DA3 annotation pipeline:

- `acceptance_pull`: validate a supplier manifest, sample IDs by scene/task,
  and pull paired HDF5/video files into a local batch.
- `acceptance_video_quality`: run no-reference video quality checks on a sampled
  batch and write CSV/JSON reports.

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
sample_count: 10
alignment_mode: warn
thresholds:
  min_fps: 1
  min_width: 1
  min_height: 1
  min_sample_decode_ratio: 1.0
  max_mean_over_dark_ratio: 0.10
  max_mean_over_exposed_ratio: 0.05
  max_black_frame_ratio: 0.05
  max_frozen_frame_ratio: 0.8
```

The video check writes:

```text
sampled/XJGT_20260616/
  reports/
    video_quality.csv
    video_quality_summary.json
```

It uses practical no-reference indicators: open/decode health, frame count,
fps, duration, resolution, sampled-frame decode ratio, brightness, over-dark and
over-exposure ratios, blur proxy, black-frame risk, frozen-frame risk, and
optional HDF5 frame-count alignment.
