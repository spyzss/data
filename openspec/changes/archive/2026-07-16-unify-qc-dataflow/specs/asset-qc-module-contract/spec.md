## ADDED Requirements

### Requirement: 单资产 QC JSON 是唯一质量事实源
系统 SHALL 为每个抽样资产维护且只维护一份 `quality_archive/<asset_id>.json` 主报告。本 change MUST 发布 `asset_qc_report.v2` 以承载通用模块、execution profile、runtime error 和二元 `overall_decision`；自动模块 MUST 将最终模块结论、顶层 issue 引用、流程状态和证据索引写入该报告；模块 sidecar、CSV、Parquet 和批次工作簿 MUST NOT 作为最终质量结论的事实源。

#### Scenario: 模块同时生成主报告和证据 sidecar
- **WHEN** 自动模块完成一个资产的检查并生成逐帧或逐窗口证据
- **THEN** 模块将汇总结论和证据相对路径写入单资产 QC JSON，并允许将大体量明细保留在 sidecar
- **THEN** 下游决策只读取单资产 QC JSON

#### Scenario: 读取历史 v1 报告
- **WHEN** 迁移工具或聚合器读取 `asset_qc_report.v1`
- **THEN** 系统以只读兼容方式将其投影为 v2 语义
- **THEN** 只有后续正式写回时才生成经过 v2 Schema 校验的新 revision

### Requirement: 模块写回边界明确
每个自动模块 MUST 只拥有自己的顶层 module block、自己生成的 issue 和相应 evidence 引用。模块 MUST 保留其他模块和未知扩展字段，MUST NOT 复制其他模块的完整 issue 或代写其他模块结论。

#### Scenario: 后运行模块更新已有报告
- **WHEN** `sam3_containment` 更新已含 `video_quality` 和关键点模块结果的报告
- **THEN** 更新后的报告保留所有既有模块字段和 issue
- **THEN** 仅新增或替换 `sam3_containment` 拥有的字段和引用

#### Scenario: 报告已包含共享边界语义修订扩展
- **WHEN** 自动模块更新的报告包含由后续人工 change 写入的一次共享边界事务及其两个相邻片段 before/after
- **THEN** Schema 与模块写回事务原样保留该扩展
- **THEN** 自动模块不得拆分、覆盖或把它降格成单片段独立编辑

#### Scenario: external-stage payload 使用半开帧边界
- **WHEN** 语义扩展以内边界 411 表示界面闭区间结束帧 410，并将该边界从 411 修改为 429
- **THEN** 统一数据流原样保留内部边界 429，使当前段界面结束帧为 428、下一段开始帧为 429
- **THEN** 报告写回和投影不得把 429 再执行一次闭区间或半开区间换算

### Requirement: Issue 标识与字段稳定
模块产生的每个 warn 或 fail MUST 使用单资产内稳定且唯一的 `issue_id`，并包含统一配置中登记的全局唯一 `rule_id`、机器 verdict、可序列化观测值、边界值、证据上下文和 `needs_manual_review`。同一资产同一规则同一证据区间重跑时 MUST 产生相同 `issue_id`。

#### Scenario: 同一异常窗口重跑
- **WHEN** 同一资产、规则、手侧和帧区间被确定性重跑
- **THEN** 模块生成与前次相同的 `issue_id`
- **THEN** 报告不会因重跑累积重复 issue

### Requirement: 模块使用统一 Flow 合同
每个已执行模块 MUST 写入 `entry_gate`、`result_gate` 和 `exit_gate`。`result_gate` MUST 表达机器检查所得的 pass、warn、fail 或 skipped；`exit_gate` MUST 表达执行 profile 应用后的实际流转动作，二者不得混为一个字段。

#### Scenario: 供应商测评模式记录 fail 后继续
- **WHEN** 模块机器结果为 fail 且执行 profile 为 `supplier_evaluation`
- **THEN** `result_gate.verdict` 保持 fail
- **THEN** `exit_gate` 记录实际继续到下一模块

### Requirement: QC JSON 原子且并发安全地更新
所有模块写回 MUST 先校验 JSON Schema 和期望 revision，再以临时文件、落盘同步和原子替换更新报告。revision 不匹配 MUST 拒绝写入且不得覆盖新版本。

#### Scenario: 两个写入者使用同一旧 revision
- **WHEN** 第一个写入者成功将 revision 从 N 更新为 N+1，第二个写入者仍声明期望 N
- **THEN** 第二次写入失败并报告 stale revision
- **THEN** revision N+1 的报告保持完整

### Requirement: 未实现模块不得伪装为通过
统一编排器 MUST 区分 disabled、skipped、not_implemented 和执行错误。配置为 disabled 的模块 MUST 在 `execution.module_states` 写入 `state=disabled`；只有模块已启用、实现可用且经入口条件判断为明确不适用时，才可写入 `state=skipped`。配置为 enabled 但没有注册实现的模块 MUST 使本次运行保持未完成或显式失败，MUST NOT 自动写成 pass。

#### Scenario: 配置禁用模块
- **WHEN** 当前活跃配置将 `effective_duration` 设置为 `enabled: false`
- **THEN** 编排器写入 `execution.module_states.effective_duration.state=disabled` 和禁用原因
- **THEN** 编排器不得将该模块写成 `skipped` 或 pass

#### Scenario: 已启用模块明确不适用
- **WHEN** 已启用且有注册实现的模块通过入口条件确定当前资产不适用，并且该判断没有输入错误或运行错误
- **THEN** 编排器可写入 `state=skipped`
- **THEN** 该状态不得被用于表示配置禁用或实现缺失

#### Scenario: 配置启用尚未实现的 duplicate_check
- **WHEN** 当前代码没有 `duplicate_check` 实现但运行配置将其启用
- **THEN** 编排器记录结构化 unavailable 状态并停止形成最终 Pass
- **THEN** 报告不会包含伪造的 duplicate_check 通过结论
