## Why

当前静态人工复核页面只围绕独立 review queue、CSV 和浏览器进度文件工作，既没有语义时间轴校准，也没有将人工判定作为 warn 的最终处置写回单资产 QC JSON。人工操作因此无法成为可审计、可统计、可恢复的正式流水线步骤。

需要在 `unify-qc-dataflow` 的统一报告合同之上，提供两个运行时彻底分离的应用：先对累计 warn 做人工 Pass/Fail，只有人工全 Pass 或无需人工时才进入独立语义校准。人工 Fail 必须终止资产，不能通过直接 URL 或 API 绕过门禁。

## What Changes

- 新增独立 `semantic_calibration` 包、HTTP server、静态页面和启动入口，只提供 `/api/semantic/...`。
- 人工质检与语义校准不共享工作台、service facade、HTTP 路由、前端 adapter 或 lease 实例，只通过 `asset_qc_report.v2` 衔接。
- 流水线改为自动 QC → warn 人工质检 → 语义校准；人工 Fail 直接终止，`not_required` 直接进入语义。
- 每次时间轴拖动或单条 subtask 文字修改都必须立即确认或取消；未确认时锁定其他段落和下一阶段操作。
- 确认后的语义修改在样本语义阶段完成时一次性原子替换 HDF5；不保存本地原 HDF5 备份。
- QC JSON 只统计 `timeline_edit_count` 和 `subtask_text_edit_count`，同时允许保留 before/after 审计明细。
- 自动检查无 warn 候选时，先将人工质检标记为 `not_required`，再进入语义校准。
- 机器 warn 的人工判定是该 warn 的最终处置：人工 Pass 消解 warn，人工 Fail 将其升级为确认失败；机器原始观测仍保留用于审计。
- 准入模式下，自动 hard fail 数据不进入人工或语义阶段；供应商测评模式下，自动 fail 不截断，仍先执行适用的 warn 人工质检，再执行语义校准。
- 本次不实现模型语义检查、语义 Pass/Fail、自动 hard-fail 申诉或正常样本人工抽检。

## Capabilities

### New Capabilities

- `semantic-calibration`: 规定人工 subtask 时间轴/文字校准、逐次确认、HDF5 原子替换和修改次数审计。
- `warn-human-review`: 规定 warn 候选展示、人工 Pass/Fail、effective verdict 和最终资产结论。
- `human-qc-workbench`: 规定人工质检与独立语义校准的运行时边界、人工前置门禁和服务端状态源。

### Modified Capabilities

- `qc-pipeline-execution`: 在 `unify-qc-dataflow` 的统一编排能力上补充人工语义与 warn 复核阶段的完成条件。
- `qc-json-batch-aggregation`: 补充人工 Pass/Fail、误报率和语义修改次数的统计来源。

## Impact

- 主要影响独立语义服务/页面、人工复核到语义的状态迁移、HDF5 文本写回、QC JSON Schema 和批次统计字段。
- 依赖 `unify-qc-dataflow` 先提供统一单资产报告、profile 和路由接口。
- 现有 `manual_labels.csv` 和 progress JSON 可提供迁移读取，但不再是最终事实来源。
