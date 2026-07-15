# Marmalade Annotation Agents

本仓库是模块化 VLA / 机器人数据验收与视觉标注系统。模块之间只能通过 manifest、JSON、CSV、Parquet、XLSX 或其他显式文件契约通信，不得演变成隐式耦合的 monolithic pipeline。

## 事实来源

发生冲突时按以下优先级判断当前行为：

1. 当前分支代码和自动化测试。
2. 实际运行使用的 YAML/config、CLI 参数和 schema。
3. 已验证的 `run_config.json`、module outputs 和 ledger。
4. `AGENTS.md` 中的开发约束。
5. `README.md` 和其他说明文档。

说明性 MD、PDF 和历史 handoff 只是某一时间点的快照，不能反向证明代码行为。修改阈值、字段或决策规则时，必须同时更新 config、测试和输出元数据，不能只改文档。

## 外部验收流程

当前推荐的外部编排是：

```text
supplier data
  -> supplier adapter / canonical manifest
  -> precheck and video_quality (independent modules)
  -> candidate windows
  -> optional SAM3 containment sidecar
  -> batch ledger / issue events
  -> manual review queue and evidence page
  -> manual labels
  -> final ledger / weekly workbook
```

这是文件级 workflow，不是 Python import 顺序。任何模块都不能因为前一模块的 verdict 而跳过自己的必要检查；最终 verdict 只能在所有要求的独立输出完成 join 后计算。

正式视觉标注是另一条独立 workflow：

```text
annotation discovery -> SAM3 segmentation -> DA3 depth -> annotation QC
annotation outputs -> annotation_verify semantic validation
```

`annotation/` 不得要求 acceptance precheck 已经运行。

## 模块边界

### `precheck/`

负责可低成本批量运行的数据可信度和信号检查：

- HDF5/text integrity。
- keypoint existence：missing、NaN、inf、有效点不足。
- static keypoint morphology。
- keypoint temporal metrics 和 candidate-window 生成。
- supplier signal 记录，例如 `quality_hand`。
- optional projection metrics，但只在坐标系、内外参和图像尺寸明确时运行。

禁止：

- 加载 SAM3、DA3、Qwen 或其他 VLM。
- import `annotation/` 或 `annotation_verify/` runtime internals。
- 修复、移动或 snap supplier keypoints。
- 把 `quality_hand` 当成跨供应商通用 skeleton hard fail。
- 把 temporal review signal 伪装成 static morphology fail。

### `acceptance_pull/`

负责 supplier inventory、sampling、pull、canonical manifest、supplier adapters 和独立 video quality。

- `video_quality` 只评估视频/流/画质/冻结/时序元数据及可用的 HDF5 对齐证据。
- 不得反向修改 precheck、SAM3 或人工标签原始输出。
- supplier-specific schema 必须通过 adapter 显式转换。

### `tools/run_manifest_*.py`

负责 manifest-aware 外部编排：

- `run_manifest_precheck.py`：对 manifest 中的 inclusive source-frame ranges 运行 precheck。
- `run_manifest_video_quality.py`：对原始视频的 source-frame ranges 运行 video quality，不物理切视频。
- `run_manifest_sam3_containment.py`：读取 candidate windows 和 supplier 2D keypoints，运行独立 SAM3 containment sidecar。

这些 runner 可以读取大文件或加载 sidecar 模型，但不得把模型依赖塞回 `precheck/`。

### `tools/build_*.py` 与人工复核工具

负责消费已有 module outputs，不重跑模型：

- `build_batch_qc_ledger.py`：合并 manifest、precheck、video quality、SAM3 和 optional manual labels。
- `build_manual_review_queue.py`：从候选证据生成最终人工队列。
- `build_video_review_clips.py`：生成 sampled overlay review HTML 和稳定 CSV schema。
- `serve_manual_review.py`：提供本地 HTTP autosave endpoint。
- `build_acceptance_ledger.py`：生成 generic 或 `weekly_template` XLSX ledger。

### `annotation/`

负责正式视觉标注，包括 discovery、SAM3 segmentation、DA3/depth、storage 和 annotation QC。不得 import precheck 或 acceptance ledger runtime 逻辑。

### `annotation_verify/`

只负责 instruction/video semantic consistency 和外部 HTTP VLM/Qwen integration。不得执行 HDF5 temporal、keypoint morphology 或 video-quality checks，也不得修改 annotation outputs。

### `qc_common/`

只放稳定共享契约和纯工具，例如：

- `ClipInputs`、`CheckResult`。
- keypoint topology 和 projection helpers。
- schema/config validation。
- JSON/CSV/Parquet IO helpers。

禁止加载模型、读取批量大数据或写业务 outputs。

## Supplier 与坐标契约

- XJGT、JDT、DeepReach 不共享同一个原始 HDF5/parquet schema，必须经过各自 adapter。
- canonical `asset_id` 必须稳定、可回连 supplier source，并在所有 module outputs 中保持一致。
- manifest frame ranges 使用 inclusive source-frame coordinates。
- sliced clip 内部可使用 local frame index，但导出 candidate window 时必须同时保留 local/source 字段，并明确 `coordinate_space`。
- local window 只能偏移一次；已经标记为 source coordinates 的窗口不得再次 offset。
- `start_frame <= end_frame`，并且导出范围必须落在 manifest clip bounds 内。无效记录应隔离到 failures，不能静默 clamp。
- JDT 直接读取 parquet 里的 2D hand keypoints；不得为此伪造 calibration。
- DeepReach 的 calibration/projection lineage 未确认时，SAM3 containment 应标记 blocked/review，不得伪造 pass 或 skeleton fail。

## Keypoint 与验收语义

- 每只手的 acceptance topology 期望 21 个点。双手通常为 42 个点，但不能只按 flat array 长度推断左右手语义。
- missing、NaN、inf、短坐标或有效点不足属于 existence invalid，不应同时记为 morphology fail。
- morphology 是单帧静态几何；temporal 是相邻帧变化，两者必须保持独立字段和 verdict。
- `quality_hand` 是 supplier-provided evidence。存在时可记录 `low/provided_ok`，不存在时为 `not_provided/not_applicable`；不得作为五供应商通用 hard fail。
- temporal suspect 通常生成 candidate window，进入 SAM3 或人工仲裁；不得仅凭 rotation/acceleration/displacement 就删除整 clip。
- SAM3 mask 只描述可见区域。inside mask 不证明完整可见，outside mask 也不能脱离 projection、遮挡和 mask 边界直接证明 skeleton 错误。

窗口级推荐解析顺序：

```text
precheck hard invalid -> hard evidence, no SAM3 repair
precheck review/suspect -> SAM3 candidate
SAM3 pass -> window resolved pass
SAM3 fail/review -> manual review
manual true_positive/rejected -> affected interval fail
manual false_positive/acceptable_flagged/accepted -> window pass
manual incomplete -> review
SAM3 blocked -> only unresolved candidate windows review
```

最终 asset/clip verdict 必须由对应 ledger policy 计算。不得用上述窗口状态覆盖 raw module statuses，也不得把某一种周报 policy 偷换成所有 workflow 的通用规则。

## Manual Review 契约

- 优先按 normalized non-empty `review_id` 匹配 queue 与 labels。
- `review_id` 缺失时，才回退到 exact `asset_id + window_start_frame + window_end_frame`。
- 不得因为同一 asset 上另一个 interval overlap 就把当前窗口标记为已审核。
- 一个 `review_id` 可以导出多个 affected segments。
- reject/review duration 只来自人工确认的 affected intervals；candidate window 本身不是 reject duration。
- 统计 frame count 前必须按 asset 合并重叠 intervals。
- `acceptable_flagged` 和 `false_positive` 是 calibration evidence，不计入 problem frames。
- 浏览器 localStorage 不是唯一持久化。云端复核应使用 `serve_manual_review.py` 写 autosave CSV/JSON，并保留导出备份。

## Status 与输出规则

- 保留 raw statuses、reasons、metrics、threshold metadata、issue counts 和 evidence paths。
- `missing`、`not_run`、`blocked`、`adapter_missing`、`input_missing`、`no_valid_output` 不得写成 `pass`。
- required module 未运行通常进入 review/not_ready；真正不要求的模块才可为 `not_applicable`。
- 单个 asset/窗口失败必须写 failures 并继续其他记录，除非输入契约整体无效。
- JSON metrics 必须是 JSON-safe Python scalars/containers，不能泄漏 NumPy scalar 或 ndarray。
- 运行使用的阈值必须进入 versioned config 或 `run_config.json`；代码默认值不能冒充某次运行的实际 config。

## 数据与 Git 规则

不得提交：

- `outputs/`、`output/` 或生成的 XLSX/HTML/PDF。
- videos、HDF5、large parquet datasets、supplier archives。
- model weights、calibration caches、extracted supplier data。
- browser/localStorage exports、temporary config、cloud credentials。

开始修改前：

1. 运行 `git status --short --branch`。
2. 确认当前 worktree、branch 和 owning module。
3. 检查是否已有用户改动，不得擅自 reset、restore 或删除。
4. 优先使用已有 schema/adapter/helper，不创建并行实现。

完成后：

1. 只报告和 stage intentional files。
2. 运行与 blast radius 匹配的测试和 `git diff --check`。
3. 明确未运行的测试、环境缺失和残余风险。
4. 除非用户明确要求，不得 stage、commit 或 push。

## 标准验证

```bash
python3 -m compileall precheck qc_common tools acceptance_pull annotation_verify
.venv/bin/python -m pytest tests/test_qc_modules_smoke.py -q
git diff --check
```

Manifest runner 改动至少运行：

```bash
.venv/bin/python -m pytest \
  tests/test_manifest_precheck_runner.py \
  tests/test_manifest_video_quality_runner.py \
  tests/test_manifest_sam3_containment_runner.py \
  -q
```

Ledger/manual review 改动至少运行对应的：

```bash
.venv/bin/python -m pytest \
  tests/test_batch_qc_ledger.py \
  tests/test_manual_review_queue.py \
  tests/test_video_review_clips.py \
  tests/test_acceptance_ledger.py \
  -q
```

## 参考入口

- [README.md](README.md)
- [WORKFLOW_INTERFACE.md](WORKFLOW_INTERFACE.md)
- [PRECHECK_INTERFACE.md](PRECHECK_INTERFACE.md)
- [ACCEPTANCE.md](ACCEPTANCE.md)
- [docs/asset-qc-json-format.md](docs/asset-qc-json-format.md)
- [docs/deepreach_supplier_adapter_zh.md](docs/deepreach_supplier_adapter_zh.md)

Last updated: 2026-07-14
