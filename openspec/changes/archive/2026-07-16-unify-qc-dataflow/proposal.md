## Why

当前仓库已经合入 Precheck、关键点质量、视频质量、SAM3、人工复核证据和多套批次报表，但除 `video_quality` 外，其余模块仍以独立 JSON、Parquet 或 CSV 作为事实来源，未按现有 PRD 写回单资产 `quality_archive/<asset_id>.json`，也没有统一 Gate 编排。这导致模块结论、人工结果和批次统计存在多条互相绕开的数据流，无法稳定回答每条数据为什么通过或失败。

现在需要在不重写同事算法的前提下，将已合入模块统一到版本化配置、单资产 QC JSON、原子写回和可重建批次统计合同，并支持准入与供应商测评两种执行策略。

## What Changes

- 为现有自动质检模块建立统一的模块结果、issue、evidence 和 flow 写回合同。
- 将 `hdf5_text_info`、`quality_hand`、`keypoint_presence`、`keypoint_morphology`、`keypoint_temporal`、`video_quality`、`sam3_containment`、`duplicate_check`、`content_validity`、`effective_duration` 纳入统一编排。
- 保留各模块现有 sidecar 作为可追溯证据，但取消其作为最终决策和批次统计事实来源的职责。
- 新增 `acceptance` 执行 profile：自动 hard fail 立即停止，失败数据不进入后续语义校准或 warn 人工质检。
- 新增 `supplier_evaluation` 执行 profile：自动 fail 被记录但不控制流转，所有数据执行完整流程；最终结论仍反映已记录的 fail。
- 统一更新单资产报告 revision、pipeline state、module block、顶层 issues、manual review 路由状态和最终二元结论。
- 批次 CSV、XLSX、Markdown 和统计指标仅从最终 `quality_archive/*.json` 重建。
- **BREAKING**：现有 ledger/report 工具不再直接把 candidate windows、SAM3 summary、video sidecar 或 manual CSV 作为最终事实来源。

## Capabilities

### New Capabilities

- `asset-qc-module-contract`: 规定所有自动模块如何以稳定 issue、evidence、flow 和 module block 原子更新单资产 QC JSON。
- `qc-pipeline-execution`: 规定统一模块顺序、准入/供应商测评执行 profile、fail 截断与最终二元结论。
- `qc-json-batch-aggregation`: 规定批次统计只聚合单资产 QC JSON，并区分自动 fail、人工确认 fail 和人工消解 warn。

### Modified Capabilities

当前仓库没有既有 OpenSpec capability；本 change 将现有 PRD 转化为首批可验证 capability。

## Impact

- 主要影响 `qc_common/`、`precheck/`、`acceptance_pull/video_quality.py`、`tools/run_manifest_*.py`、批次 ledger/report 工具、统一配置、JSON Schema 和对应测试。
- 现有算法阈值和 sidecar 生成能力原则上保持不变；新增适配层、统一写回层和编排层。
- `add-human-semantic-warn-review` 依赖本 change 提供的报告合同、执行 profile 和人工路由状态。
