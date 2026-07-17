# qc-json-batch-aggregation Specification

## Purpose
定义仅基于单资产 QC JSON 的批次投影与统计，规范按 execution profile 分组，并确保派生缓存可从源报告完整重建。
## Requirements
### Requirement: 批次统计只读取单资产 QC JSON
批次聚合器 SHALL 仅遍历目标批次 `quality_archive/*.json` 生成 CSV、XLSX、Markdown 或缓存表。聚合器 MUST NOT 通过重新读取候选窗口、SAM3 summary、视频结果 sidecar、manual CSV 或浏览器进度文件推导最终质量结论。

#### Scenario: sidecar 与 QC JSON 结论不一致
- **WHEN** 遗留 sidecar 显示 warn 但对应 QC JSON 已记录人工 Pass
- **THEN** 批次聚合结果使用 QC JSON 的有效结论
- **THEN** sidecar 不改变最终统计

### Requirement: 自动失败与人工失败分别统计
批次统计 MUST 分别计算自动 hard-fail 资产/issue 数、机器 warn 数、人工消解 warn 数、人工确认 fail 数、最终 fail 资产数和最终通过率。一个资产内多个 issue MUST NOT 被错误计算为多个资产。

#### Scenario: 一个资产包含两个自动 fail issue
- **WHEN** 同一资产报告包含两个自动 hard-fail issue
- **THEN** 自动 fail issue 数增加二
- **THEN** 自动 fail 资产数和最终 fail 资产数各增加一

### Requirement: 不同执行 profile 可分别聚合
每份 QC JSON MUST 记录执行 profile，批次聚合器 MUST 支持按 profile 分组，且不得将准入模式的截断覆盖率与供应商测评模式的全流程覆盖率直接混为一个指标。

#### Scenario: 同一批次包含两种 profile
- **WHEN** 聚合目录中同时存在 acceptance 和 supplier_evaluation 报告
- **THEN** 输出分别给出两种 profile 的资产数、完成率和结果统计

### Requirement: 派生缓存可完全重建
任何用于加速查询的 Parquet、CSV 或索引 MUST 标记为派生产物，并能够仅凭 QC JSON 全量删除后重建。缓存缺失或过期不得影响源报告真实性。

#### Scenario: 删除所有批次缓存
- **WHEN** 操作者删除派生 Parquet 和 CSV 后重新运行聚合
- **THEN** 系统仅从 `quality_archive/*.json` 生成与删除前语义一致的统计结果
