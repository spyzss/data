# Marmalade Data QC and Annotation

本仓库现在不是单一 annotation pipeline，而是面向机器人/VLA 数据的多模块验收与标注系统：

```text
precheck -> optional SAM3 containment sidecar -> annotation -> annotation_verify
```

这个顺序属于外部 workflow。代码层面四个 root modules 保持独立；测试可以临时串联多个模块，但只能通过配置/文件/JSON 输出连接，不能把模块互相 import 成隐式 workflow。

- `precheck/`：数据可信度、HDF5 文本字段、`quality_hand`、骨骼点连续性、基础画质、mask containment 消费端。
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

### 3. Annotation

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

### 4. Annotation verification

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
