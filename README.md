# Marmalade Data Acceptance and Annotation

本仓库用于机器人/VLA supplier 数据验收、人工复核和正式视觉标注。系统由多个独立模块组成，通过 manifest、JSON、CSV、Parquet 和 XLSX 交换结果；它不是一个必须按 Python import 串起来的单体 pipeline。

## 两条独立工作流

### Supplier acceptance

```text
supplier adapter / canonical manifest
  -> precheck + video_quality
  -> candidate windows
  -> optional SAM3 containment sidecar
  -> batch ledger / issue events
  -> manual review
  -> final acceptance ledger / weekly workbook
```

### Visual annotation

```text
annotation discovery
  -> SAM3 segmentation
  -> DA3 depth
  -> annotation QC
  -> annotation_verify semantic validation
```

两条 workflow 可以共享文件契约，但互不要求对方先运行。

## 模块目录

| 路径 | 职责 | 主要输出 |
| --- | --- | --- |
| `precheck/` | 文本、keypoint existence、static morphology、temporal metrics、signal checks | `check_results.*`、`clip_aggregates.*`、`candidate_windows.*` |
| `acceptance_pull/` | supplier sampling/pull、adapters、manifest、video quality | manifests、quality archives、video-quality summaries |
| `tools/run_manifest_*.py` | 按 manifest 的 inclusive source-frame ranges 运行 precheck/video quality/SAM3 sidecar | module JSON/CSV/Parquet、failures、`run_config.json` |
| `tools/build_*.py` | ledger、review queue、人工页面和 XLSX 报告 | issue events、review CSV/HTML、acceptance workbooks |
| `annotation/` | SAM3/DA3 正式视觉标注与 QC | masks、depth、sampling manifests、QC images |
| `annotation_verify/` | instruction/video semantic consistency | verification results |
| `qc_common/` | 共享 types、schema、topology、projection 和 IO helpers | 无独立业务 job |

详细边界见 [AGENTS.md](AGENTS.md)。

## 环境

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

SAM3、DA3、Qwen/VLM 等模型依赖按运行方式另行安装。`precheck/` 本身不得加载这些模型。

快速验证：

```bash
python3 -m compileall precheck qc_common tools acceptance_pull annotation_verify
.venv/bin/python -m pytest tests/test_qc_modules_smoke.py -q
```

## Acceptance 快速入口

以下命令使用占位路径。先用 `--dry-run` 或小批量参数验证 manifest 和坐标范围，再运行完整批次。

### 1. 原生 precheck

适用于已由 `precheck/adapters/` 支持的 HDF5 输入：

```bash
.venv/bin/python run_precheck.py configs/precheck_example.yaml
```

典型输出：

```text
<output_dir>/check_results.json
<output_dir>/check_results.parquet
<output_dir>/clip_aggregates.json
<output_dir>/clip_aggregates.parquet
<output_dir>/candidate_windows.json
<output_dir>/candidate_windows.parquet
```

### 2. Manifest-aware precheck

JDT/DeepReach 通过 manifest adapter 读取 supplier source ranges：

```bash
.venv/bin/python -m tools.run_manifest_precheck \
  --manifest /path/to/supplier_manifest.csv \
  --supplier jdt \
  --output-dir outputs/run/precheck \
  --dry-run
```

`start_frame`/`end_frame` 是 inclusive source-frame coordinates。runner 导出 local/source frame fields，并将无效行隔离到 failures。

### 3. Manifest-aware video quality

```bash
.venv/bin/python -m tools.run_manifest_video_quality \
  --manifest /path/to/supplier_manifest.csv \
  --output-dir outputs/run/video_quality \
  --video-column primary_video_path \
  --dry-run
```

完整运行输出包括：

```text
video_quality_results.json
video_quality_results.parquet
video_quality_decision_summary.csv
video_quality_failures.json
run_config.json
```

旧式 sampled batch 入口和统一 video-quality config 见 [ACCEPTANCE.md](ACCEPTANCE.md)。

### 4. SAM3 containment sidecar

当前 manifest runner 支持 JDT 直接从 parquet 读取 2D hand keypoints：

```bash
.venv/bin/python -m tools.run_manifest_sam3_containment \
  --manifest /path/to/jdt_manifest.csv \
  --candidate-windows outputs/run/precheck/candidate_windows.parquet \
  --supplier jdt \
  --output-dir outputs/run/sam3_containment \
  --frames-per-window 3 \
  --sam3-model /path/to/sam3 \
  --dry-run
```

正式运行输出：

```text
frame_keypoint_containment.json/parquet
window_keypoint_containment_summary.json/parquet
review_evidence_manifest.csv/parquet
failures.json
run_config.json
```

SAM3 是独立 sidecar，不会被 `precheck/` import。DeepReach 只有在 calibration/projection lineage 确认后才能运行对应 containment；blocked 不得写成 pass。

### 5. Batch ledger 与人工队列

```bash
.venv/bin/python -m tools.build_batch_qc_ledger \
  --supplier-sample-manifest /path/to/manifest.csv \
  --precheck-clip-aggregates outputs/run/precheck/clip_aggregates.parquet \
  --precheck-candidate-windows outputs/run/precheck/candidate_windows.parquet \
  --video-quality-results outputs/run/video_quality/video_quality_results.json \
  --sam3-window-summary outputs/run/sam3_containment/window_keypoint_containment_summary.parquet \
  --output-dir outputs/run/ledger
```

```bash
.venv/bin/python -m tools.build_manual_review_queue \
  --manifest /path/to/manifest.csv \
  --candidate-windows outputs/run/precheck/candidate_windows.parquet \
  --sam3-window-summary outputs/run/sam3_containment/window_keypoint_containment_summary.parquet \
  --issue-events outputs/run/ledger/issue_events.csv \
  --video-quality outputs/run/video_quality/video_quality_results.json \
  --output-dir outputs/run/review
```

### 6. Overlay review 与云端 autosave

```bash
.venv/bin/python -m tools.build_video_review_clips \
  --manifest /path/to/manifest.csv \
  --review-queue outputs/run/review/review_queue.csv \
  --evidence-manifest outputs/run/sam3_containment/review_evidence_manifest.csv \
  --output-dir outputs/run/video_review
```

```bash
.venv/bin/python -m tools.serve_manual_review \
  --review-dir outputs/run/video_review \
  --save-dir outputs/run/manual_review \
  --host 0.0.0.0 \
  --port 8896
```

人工导出的 rejected duration 只来自 confirmed affected segments，不直接使用整个 candidate window。重叠 segments 在统计前按 asset 合并。

### 7. Acceptance workbook

Generic/weekly workbook 共用一个 builder：

```bash
.venv/bin/python -m tools.build_acceptance_ledger \
  --config configs/acceptance_ledger_xjgt_jdt_dr_weekly.yaml \
  --existing-workbook /path/to/weekly_template.xlsx \
  --output outputs/acceptance_5x100/weekly_supplier_acceptance_report.xlsx \
  --overwrite
```

`workbook_mode: weekly_template` 使用周度模板；不应把周度 final policy 偷换成所有 ledger 的通用规则。

## Annotation 快速入口

```bash
.venv/bin/python run_annotate.py configs/anygrasp_full.yaml
.venv/bin/python run_annotate.py configs/seg_only.yaml --stage segmentation
.venv/bin/python run_annotate.py configs/depth_only.yaml --stage depth
```

Annotation verification：

```bash
.venv/bin/python run_annotation_verify.py configs/annotation_verify_example.yaml
```

模型可以是本地部署或 endpoint。运行前先确认部署模式、模型路径/API 和对应环境依赖。

## 关键语义

- `quality_hand` 是 supplier-provided signal，不是五供应商通用 hard fail。
- existence、morphology 和 temporal 是三个独立 keypoint checks。
- temporal suspect 通常生成 candidate window，不等于整 clip discard。
- SAM3 containment 是视觉证据；mask 内外不能脱离 projection/遮挡语义直接判 skeleton 对错。
- `missing`、`not_run`、`blocked`、`adapter_missing`、`no_valid_output` 不得写成 pass。
- final verdict 在 ledger join 后按所选 policy 计算，raw module statuses 和 metrics 必须保留。

## 文档入口

- [AGENTS.md](AGENTS.md)：Agent 开发边界与稳定规则。
- [WORKFLOW_INTERFACE.md](WORKFLOW_INTERFACE.md)：模块接口总览。
- [PRECHECK_INTERFACE.md](PRECHECK_INTERFACE.md)：precheck 输入输出契约。
- [ACCEPTANCE.md](ACCEPTANCE.md)：supplier pull、video quality 和 asset QC archive。
- [QUICKSTART.md](QUICKSTART.md)：基础运行示例。
- [STRUCTURE.md](STRUCTURE.md)：仓库目录说明。
- [docs/asset-qc-json-format.md](docs/asset-qc-json-format.md)：asset QC JSON schema。
- [docs/deepreach_supplier_adapter_zh.md](docs/deepreach_supplier_adapter_zh.md)：DeepReach manifest adapter。

## Git 中禁止的数据

以下内容只保存在本地/云端数据目录，不进入 Git：

```text
outputs/
output/
*.h5
*.hdf5
*.parquet
*.mp4
*.xlsx
model weights
calibration caches
supplier archives
generated HTML/PDF
```

提交前运行：

```bash
git status --short --branch
git diff --check
```
