# Marmalade Data QC and Annotation

本仓库现在不是单一 annotation pipeline，而是面向机器人/VLA 数据的多模块验收与标注系统：

```text
supplier adapter / canonical manifest
  -> precheck and video_quality (independent producers)
  -> candidate windows
  -> optional SAM3 containment sidecar
  -> ledger and manual review

annotation discovery -> SAM3 segmentation -> DA3 depth -> annotation QC
annotation outputs -> annotation_verify
```

这个顺序属于外部 workflow。代码层面四个 root modules 保持独立；测试可以临时串联多个模块，但只能通过配置/文件/JSON 输出连接，不能把模块互相 import 成隐式 workflow。

- `precheck/`：数据可信度、HDF5 文本字段、`quality_hand`、骨骼点 existence/morphology/temporal 与 candidate-window 生成；不加载视觉模型。
- `annotation/`：视觉标注，包含 discovery、SAM3 segmentation、DA3 depth、storage、annotation QC。
- `annotation_verify/`：语义一致性验证契约，目前 VLM 仍是 stub。
- `qc_common/`：共享契约、schema、keypoint topology、registry 和纯工具。

主接口文档见 [WORKFLOW_INTERFACE.md](WORKFLOW_INTERFACE.md)。

## 快速入口

### 1. HDF5 / skeleton precheck

```bash
python run_precheck.py configs/precheck_example.yaml
```

主要输出：

```text
<output_dir>/check_results.json
<output_dir>/clip_aggregates.json
```

### 2. 云端 SAM3 骨骼点区域占比抽检

```bash
python tools/sam3_keypoint_containment.py \
  --hdf5-dir /path/to/hdf5 \
  --video-dir /path/to/mp4 \
  --sam3-model /path/to/sam3 \
  --output-dir outputs/sam3_keypoint_containment \
  --sample-fraction 0.10 \
  --projection-mode auto
```

核心指标：

```text
clip_keypoint_inside_ratio = inside_keypoints / total_expected_keypoints
```

### 3. Manifest 统一 QC 入口（云端）

当前需要尽可能跑完整检查并保留供应商诊断时，使用 `supplier_evaluation`：

```bash
python tools/run_qc_pipeline.py \
  --batch-root /path/to/qc_run \
  --manifest /path/to/qc_run/manifest.jsonl \
  --profile supplier_evaluation \
  --max-workers 1 \
  --resume
```

同一个 asset 的五项 precheck 共用一次源数据加载，但仍分别写入五个 module result。Producer sidecar 位于：

```text
<batch-root>/module_outputs/<asset_id>/precheck/
<batch-root>/module_outputs/<asset_id>/video_quality/
<batch-root>/module_outputs/<asset_id>/sam3_containment/
<batch-root>/quality_archive/<asset_id>.json
```

`--resume` 只复用输入、范围、配置和实现指纹完全匹配的 sidecar。CLI JSON 摘要会报告每个 asset 的 `computed/reused/skipped/blocked/failed` producer 状态和总耗时。要强制 producer 重算可使用 `--no-resume`；若已有 QC report，建议使用新的 `batch-root` 保留旧运行，而不是删除或覆盖旧 report。

统一入口的 SAM3 只读取本轮 precheck 写出的 `module_outputs/<asset_id>/precheck/candidate_windows.json`，不要求 manifest 预填 `candidate_windows_path`。空候选不会加载 SAM3。JDT 继续直接读取 Parquet 2D keypoints；DeepReach head calibration/projection adapter 尚未完成时会明确标记 `adapter_missing/blocked`。

### 4. Annotation

```bash
python run_annotate.py configs/anygrasp_full.yaml
python run_annotate.py configs/seg_only.yaml --stage segmentation
python run_annotate.py configs/depth_only.yaml --stage depth
```

输出：

```text
masks.parquet
depth/<camera>/episode_<idx>/frame_<idx>.png
sampling_manifest.parquet
qc/*.png
```

### 5. Annotation verification

```bash
python run_annotation_verify.py configs/annotation_verify_example.yaml
```

当前只验证 runner/config/stub contract，不执行真实 VLM。

## 当前重点能力

- HDF5 `label/text_label` 完整性检查。
- `quality_hand` 客户规则检查。
- 21 hand acceptance keypoints 的连续性、角速度/旋转 delta、加速度、位移指标。
- vendor-agnostic `skeleton_quality_score`。
- supplier label + geometry 的 `composite_frame_verdict` 审计。
- JSON first output contract，方便台账和人工质检系统消费。
- SAM3 sidecar 抽样计算骨骼点落在手部 mask 内的比例。

## 文档索引

- [WORKFLOW_INTERFACE.md](WORKFLOW_INTERFACE.md)：统一 workflow 和接口。
- [PRECHECK_INTERFACE.md](PRECHECK_INTERFACE.md)：precheck/blue-circle 详细接口。
- [QUICKSTART.md](QUICKSTART.md)：常用命令。
- [STRUCTURE.md](STRUCTURE.md)：目录结构。
- [STATUS.md](STATUS.md)：当前状态。
- [SAM3_IMPLEMENTATION.md](SAM3_IMPLEMENTATION.md)：annotation SAM3 实现说明。
- [DA3_DIAGNOSIS_REPORT.md](DA3_DIAGNOSIS_REPORT.md)：DA3 诊断记录。

## 环境

基础依赖见 `requirements.txt`。SAM3 / DA3 / VLM 属于 heavy model 依赖，按对应模块或云端 sidecar 需要单独安装。

本地数据、模型、输出都不进 git：

```text
models/
outputs/
*.h5
*.hdf5
*.parquet
*.mp4
```
