# Marmalade Data QC and Annotation

本仓库现在不是单一 annotation pipeline，而是面向机器人/VLA 数据的多模块验收与标注系统：

```text
supplier adapter / canonical manifest
  -> precheck -> video_quality -> supplier_data_audit (independent producers)
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
<batch-root>/module_outputs/<asset_id>/supplier_data_audit/
<batch-root>/module_outputs/<asset_id>/sam3_containment/
<batch-root>/quality_archive/<asset_id>.json
```

`--resume` 只复用输入、范围、配置和实现指纹完全匹配的 sidecar。CLI JSON 摘要会报告每个 asset 的 `computed/reused/skipped/blocked/failed` producer 状态和总耗时。要强制 producer 重算可使用 `--no-resume`；若已有 QC report，建议使用新的 `batch-root` 保留旧运行，而不是删除或覆盖旧 report。

统一入口的实际自动模块顺序是 `precheck -> video_quality -> supplier_data_audit -> sam3_containment`。`video_quality` 与 `supplier_data_audit` 都是独立 producer，顺序不表示前者向后者传值；两者只通过各自 artifact 和共享 QC report 交付结果。

SAM3 只读取本轮 precheck 写出的 `module_outputs/<asset_id>/precheck/candidate_windows.json`，不要求 manifest 预填 `candidate_windows_path`，并且只消费显式 `sam3_eligible=true` 的候选。无有效 temporal output 时为 `blocked/no_valid_temporal_output`；有效 temporal 但无 eligible candidate 时为 `skipped/no_candidates`。JDT 继续直接读取 Parquet 2D keypoints。DR/head adapter 仅在 task→content_id 映射、head 内参、分辨率、HDF5/video 帧对齐和 direct head-camera 变换链均显式验证后，从 DR HDF5 读取 `hand/<side>/joints3d` 并投影到 2D；未验证时保留 `calibration_unverified` / `mapping_missing` / `transform_ambiguous` / `resolution_mismatch` / `frame_alignment_unverified`，不加载模型也不写 pass。

DR 与 Potentia 的 `supplier_evaluation` 通过独立 `supplier_data_audit` producer 接入。DR 默认一个 task 一个 asset，并在同一 manifest 行保留 head/left_wrist/right_wrist 三路视频。DR HDF5 manifest 要求通过 CLI/config 显式选择 reference dataset；当前仓库不替供应商猜默认值，云端确认正式契约后可明确选择 `timestamp`。`joints3d` 是必需 dataset，`valid` 是 optional hand-level validity；存在时必须与 reference 等长，缺失时以 finite joint values 建立 validity。DR precheck fingerprint 额外记录 `deepreach-hdf5-precheck-v2`，防止复用旧 adapter artifact，同时不改 JD 的 v7 identity。Potentia 的 package 只是运输分区，一个 task 目录一个 asset。CSV timestamp 必须通过 `s/ms/us/ns` 或单一显式 scale 归一化为秒；标定缩放无法解释时默认 review，只有供应商配置明确为 `fail` 才升级。supplier audit 只做文件、CSV、IMU、标定和轨迹结构审计，不修改 precheck/video 原始输出，也不生成第二套文本 verdict。

当前 cache identity 为 `precheck-session-v8-standardized-temporal-timebase` 和 `supplier-data-audit-producer-v3`；外层仍是 `qc_producer_run_config.v1`。Temporal output schema 为 `keypoint_temporal.output.v3`：原生 FPS 相邻帧指标仅作诊断保留，score 和 candidate window 使用按 timestamp（无 timestamp 时按 source frame / resolved FPS）最近且不重复采样的 30 Hz standardized metrics。Run config 同时记录 sample→source-frame lineage 和异常时间戳计数； supplier audit 仍独立记录 raw schema identity。

云端小样本构建、pipeline、projection overlay 和状态审计命令见 `docs/dr_potentia_supplier_evaluation_cloud_smoke_zh.md`。

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
