# Quick Start - Marmalade Visual Annotation

This guide reflects the current verified AnyGrasp pipeline state.

## Environment

```bash
cd /mnt/workspace/spy/marmalade/marmalade_annotation
source .venv_annotation/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
```

The verified environment is Python 3.12 with torch cu121, SAM3 weights at `models/sam3`, and DA3Metric loaded from `depth-anything/DA3METRIC-LARGE`.

## 1. Dry-Run Discovery

Dry-run reads real AnyGrasp `expand_task` metadata and writes instruction -> queries mappings. It does not decode video frames or run models.

```bash
python run_dryrun.py configs/anygrasp_dryrun.yaml
cat outputs/anygrasp_dryrun/discovery_queries.jsonl
```

Expected ep0 instruction:

```text
Pick the green bottle from the table and place it inside the blue basket.
```

With the temporary rule extractor, expected queries include `green bottle`, `blue basket`, `robot hand`, and possibly one noisy long phrase.

## 2. Full Annotation

```bash
HF_HOME=/tmp/hf-home MPLCONFIGDIR=/tmp/matplotlib \
python run_annotate.py configs/anygrasp_full.yaml
```

Outputs:

```text
<output_dir>/masks.parquet
<output_dir>/depth/observation.images.top/episode_000000/frame_000000.png
<output_dir>/depth/observation.images.top/episode_000000/frame_000000.json
<output_dir>/qc/qc_ep000000_frame000000.png
<output_dir>/.checkpoints/completed_episodes.json
```

Resume is automatic when the same config/output directory is rerun. Current CLI initializes models before entering the pipeline, so checkpoint skip avoids frame processing but not model startup cost.

## 3. Delivery Checks

Small multi-episode robustness check:

```bash
HF_HOME=/tmp/hf-home MPLCONFIGDIR=/tmp/matplotlib \
python run_annotate.py configs/delivery_rule_small_batch.yaml
```

Clean-query ceiling check:

```bash
HF_HOME=/tmp/hf-home MPLCONFIGDIR=/tmp/matplotlib \
python run_annotate.py configs/delivery_rule_clean_compare.yaml
HF_HOME=/tmp/hf-home MPLCONFIGDIR=/tmp/matplotlib \
python run_annotate.py configs/delivery_manual_clean_queries.yaml
```

Comparison images:

```text
outputs/delivery_checks/qc_comparisons/compare_ep000000_frame000170.png
outputs/delivery_checks/qc_comparisons/compare_ep000001_frame000198.png
outputs/delivery_checks/qc_comparisons/compare_ep000014_frame000180.png
```

## 4. Key Config Fields

```yaml
dataset_type: lerobot_v3
episode_indices: [0, 1, 2]          # optional subset
sample_frames_per_episode: 5        # optional uniform sampling

discovery:
  extractor: rule                   # rule | manual | qwen | mock
  instruction_source: episode_field
  instruction_field: expand_task
  always_include:
    - robot hand
    - gripper

segmentation:
  model_path: models/sam3

depth:
  model_path: depth-anything/DA3METRIC-LARGE
  output_metric: true
  fx: 369.81
  fy: 341.14
  calibration_width: 832
  calibration_height: 480
```

For another dataset, update YAML first. Do not hardcode dataset-specific paths in the pipeline.

Instruction source options:

- `episode_field`: read a text field from `meta/episodes/...parquet`.
- `join_file`: read a separate parquet table and join by configured keys.
- `none`: no language available; logs warning and falls back to defaults/vocab.

## 5. Intern Handoff

The pipeline wiring is done. The main remaining work is discovery quality.

- Implement real Qwen loading and generation in `annotation/discovery/qwen_extractor.py`.
- Keep `discover_objects(instruction, config) -> list[str]` unchanged.
- Use `annotation/discovery/factory.py` and `discovery.extractor: qwen` to switch it on.
- The rule extractor is intentionally temporary and can emit noisy broad/long queries.
- Use `manual` extractor configs as an oracle reference for expected clean object names.

## 6. Reading Outputs

```python
import pandas as pd

df = pd.read_parquet('outputs/delivery_checks/manual_clean_queries/masks.parquet')
print(df[['episode_idx', 'frame_idx', 'category', 'score', 'bbox']].head())
```

Depth PNGs are metric millimeters. Divide uint16 pixel values by 1000 to recover meters; the JSON sidecar records original min/max and conversion notes.
