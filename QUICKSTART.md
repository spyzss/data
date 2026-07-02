# Quick Start

本指南按当前统一 workflow 组织：

```text
precheck -> optional SAM3 containment sidecar -> annotation -> annotation_verify
```

各模块可以独立运行。不要把这个顺序写死进 root modules。临时端到端测试可以串联这些命令，但应通过输出文件和配置对接。

## 0. Setup

```bash
cd /path/to/marmalade_annotation
source .venv/bin/activate  # 或你的云端环境
```

如需跑 SAM3 / DA3：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 1. Precheck

先改 `configs/precheck_example.yaml` 的 `input_paths` 和 `output_dir`。

```bash
python run_precheck.py configs/precheck_example.yaml
```

输出：

```text
outputs/precheck_example/check_results.json
outputs/precheck_example/clip_aggregates.json
```

常看字段：

- `text_integrity.missing_field_count`
- `quality_score.pass_ratio`
- `skeleton_quality_score.pass_ratio`
- `skeleton_quality_score.mean_skeleton_score`
- `composite_frame_verdict.count_suspect`

## 2. SAM3 Keypoint Containment 抽检

云端 GPU 上跑：

```bash
python tools/sam3_keypoint_containment.py \
  --hdf5-dir /path/to/hdf5 \
  --video-dir /path/to/mp4 \
  --sam3-model /path/to/sam3 \
  --output-dir outputs/sam3_keypoint_containment_smoke \
  --sample-fraction 0.10 \
  --max-clips 2 \
  --max-sampled-frames-per-clip 5 \
  --projection-mode auto
```

确认 OK 后去掉 smoke 限制：

```bash
python tools/sam3_keypoint_containment.py \
  --hdf5-dir /path/to/hdf5 \
  --video-dir /path/to/mp4 \
  --sam3-model /path/to/sam3 \
  --output-dir outputs/sam3_keypoint_containment \
  --sample-fraction 0.10 \
  --projection-mode auto
```

输出：

```text
frame_keypoint_containment.json
clip_keypoint_containment.json
run_manifest.json
```

核心字段：

```text
clip_keypoint_inside_ratio
valid_projected_inside_ratio
```

## 3. Annotation

Dry-run discovery：

```bash
python run_dryrun.py configs/anygrasp_dryrun.yaml
```

完整 annotation：

```bash
python run_annotate.py configs/anygrasp_full.yaml
```

只跑 SAM3 segmentation：

```bash
python run_annotate.py configs/seg_only.yaml --stage segmentation
```

只跑 DA3 depth：

```bash
python run_annotate.py configs/depth_only.yaml --stage depth
```

Mock 验证：

```bash
python run_annotate.py configs/anygrasp_full.yaml --use-mock
```

## 4. Annotation Verify

当前是 semantic verification stub：

```bash
python run_annotation_verify.py configs/annotation_verify_example.yaml
```

未来接 VLM 时仍只做语义一致性，不做 precheck signal-quality。

## 5. Validation

轻量 smoke：

```bash
python -m pytest tests/test_qc_modules_smoke.py
python -m compileall qc_common precheck annotation_verify annotation tools
```

## 6. Cloud Pull

云端直接拉当前 feature branch：

```bash
git clone git@github.com:spyzss/data.git
cd data
git checkout feat/qc-modules
```

如果已经 clone：

```bash
git fetch origin
git checkout feat/qc-modules
git pull origin feat/qc-modules
```
