# Comet Design Handoff

- Change: unify-qc-dataflow
- Phase: design
- Mode: compact
- Context hash: 5b7e3f9faa26b3ba9a0a723d8cc70bd5d4861de63c7f23dcad825a65dbadb8b6

Generated-by: comet-handoff.sh

OpenSpec remains the canonical capability spec. This handoff is a deterministic, source-traceable context pack, not an agent-authored summary.

## openspec/changes/unify-qc-dataflow/proposal.md

- Source: openspec/changes/unify-qc-dataflow/proposal.md
- Lines: 1-34
- SHA256: 38b03a1e73484a97757a3076ce0dc9a462e5aaff4cdac9581bd7d1e6b5ec1bf8

```md
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
```

## openspec/changes/unify-qc-dataflow/design.md

- Source: openspec/changes/unify-qc-dataflow/design.md
- Lines: 1-77
- SHA256: 46515ea8f05f7d23a7d3374e3688efe331957cdbc58aed58eb467782e1f6b07f

```md
## Context

仓库中已经存在两类相互脱节的实现：`video_quality` 使用 `qc_common.report` 原子更新单资产 QC JSON；Precheck、manifest 视频、SAM3、人工复核和多套 ledger/report 则各自读写 JSON、Parquet、CSV 或 HTML 进度文件。统一配置已经声明模块顺序和 `aggregate_from_quality_archive_only`，但没有一个编排器实际执行这些合同，Schema 也只对视频模块建立了专用 block。

本 change 先统一自动模块和数据流，不重写同事已经合入的检测算法。人工语义校准和 warn 复核由依赖 change `add-human-semantic-warn-review` 实现。

## Goals / Non-Goals

**Goals:**

- 将现有模块结果归一为单资产 QC JSON 的 module block、稳定 issue、evidence 和 flow。
- 建立可恢复、可按资产运行的统一编排器。
- 支持 fail 截断的准入模式和 fail 不截断的供应商测评模式。
- 使所有批次统计只依赖单资产 QC JSON。
- 保持现有 sidecar 作为大体量逐帧/逐窗口证据和调试产物。

**Non-Goals:**

- 不重新实现或重新定义各检测算法阈值。
- 不在本 change 实现语义时间轴工作台、HDF5 修改或人工 Pass/Fail UI。
- 不将配置中尚无代码实现的 duplicate/content/effective-duration 模块伪造成已实现。
- 不支持人工推翻自动 hard fail。

## Decisions

### 1. 使用适配层统一结果，不重写检测器

为现有 Precheck aggregate/candidate、video result 和 SAM3 summary 建立模块 adapter。adapter 将遗留输出转换为统一 `ModuleResult`，包含模块名、机器 verdict、metrics、issues、evidence 和执行元数据。检测器仍可生成现有 sidecar，adapter 负责主报告写回。

选择该方案是为了保留已合入算法和测试。替代方案是要求每个检测器直接重写为统一接口，但会扩大回归范围并把数据流改造与算法重构混在一起。

### 2. 共享报告事务统一处理 revision、所有权和派生状态

在 `qc_common` 增加报告 mutation/transaction API：加载当前 revision，移除本模块旧 issue，合并新 module block，重建顶层 issue 索引和人工候选，计算 pipeline state，Schema 校验后原子替换。adapter 不直接手写完整 JSON。

该事务层使用模块所有权映射，确保重跑只替换本模块数据并保留其他模块和未来扩展字段。

### 3. 机器结果与实际流转分离

`result_gate.verdict` 永远反映机器规则结果；`exit_gate` 由 `execution_profile` 决定。acceptance 下 fail 产生 `stop_qc`；supplier_evaluation 下同一个 fail 产生 `continue`，同时在执行轨迹记录 `continued_after_fail=true`。这样测评模式不会通过篡改 verdict 实现全流程。

### 4. 编排器按资产和配置注册表执行

新增资产级 orchestrator，读取版本化 config 的模块顺序，通过 registry 找到实现并传入 manifest/source_files。每完成一个模块立即提交一次报告 revision，使中断后可以从 `pipeline_state.next_module` 恢复。

配置为 disabled 的模块写 skipped；配置为 enabled 但 registry 无实现的模块写运行错误并保持 final decision 为 null。当前只接入仓库已有实现；未来模块通过同一 registry 扩展。

### 5. Sidecar 是引用证据，不是第二份结论

逐帧结果、候选窗口、overlay 和模型明细继续保留。QC JSON 只保存相对路径、内容类型、坐标系、帧区间和可选 checksum。人工队列与批次报表从 QC JSON 的 issue/evidence 索引生成，不再反向解释 sidecar 得出不同结论。

### 6. 批次报表改为投影层

新增统一 QC JSON reader 和扁平化投影，现有 ledger/report 输出逐步迁移到该投影。所有比率明确区分资产、issue、区间和帧四种分母，并按 execution profile 分组。Parquet/CSV 只作为可重建缓存。

## Risks / Trade-offs

- **遗留输出字段不一致** → 为每个 adapter 建立固定映射和 golden fixture，无法映射的值进入结构化 runtime error，禁止静默 pass。
- **多个模块并发更新同一资产** → 默认资产内串行，报告写入继续使用 expected revision 防止丢失更新。
- **Schema 一次扩展过大** → 先增加通用 module block 定义和已接入模块的严格字段，再用模块级测试逐步收紧。
- **旧报表与新投影短期不一致** → 迁移期对相同 fixture 同时运行旧/新报表并记录差异，最终只保留 QC JSON 投影为正式入口。
- **供应商测评执行成本较高** → profile 明确是显式选择，不改变准入模式的 fail-fast 默认行为。

## Migration Plan

1. 扩展配置和 Schema，加入 execution profile、通用 module contract 和运行错误状态。
2. 实现报告事务、稳定 issue ID 和 adapter 基础接口。
3. 依次接入 Precheck 模块、manifest 视频兼容路径和 SAM3。
4. 增加 orchestrator，验证 acceptance 与 supplier_evaluation 的相同输入不同流转。
5. 将 manual queue 输入改为 QC JSON issue/evidence 投影，为后续人工 change 提供接口。
6. 将批次 ledger/report 切换为 QC JSON 聚合，并保留一次性遗留输入迁移/对账工具。

回滚时可保留 sidecar 生成和现有独立 runner，但不得让新旧路径同时写同一资产报告；通过 feature flag 选择旧 runner 或统一 orchestrator。

## Open Questions

无阻塞问题。尚未实现的自动模块只建立注册合同，不在本 change 内虚构算法结果。
```

## openspec/changes/unify-qc-dataflow/tasks.md

- Source: openspec/changes/unify-qc-dataflow/tasks.md
- Lines: 1-35
- SHA256: 7c7442a62003908bae267899022e6faf3ac22853040ad2e6d2c93d411624cb38

```md
## 1. 统一合同与 Schema

- [ ] 1.1 为 execution profile、通用 module flow、runtime error、evidence 和二元 final decision 增加配置及 JSON Schema 测试
- [ ] 1.2 在 `qc_common` 定义统一 `ModuleResult`、`Issue`、`EvidenceRef` 和稳定 issue ID 接口
- [ ] 1.3 实现模块所有权感知的 revision-aware 报告 mutation，并验证重跑去重与未知字段保留

## 2. 已合入自动模块适配

- [ ] 2.1 将 text integrity 与 quality_hand 结果适配为 `hdf5_text_info` 和 `quality_hand` module block
- [ ] 2.2 将 keypoint missing/skeleton quality 适配为 `keypoint_presence` module block 和稳定帧区间 issue
- [ ] 2.3 将 keypoint morphology 适配为 `keypoint_morphology` module block、metrics 和 evidence
- [ ] 2.4 将 keypoint temporal/candidate windows 适配为 `keypoint_temporal` module block 和人工候选 issue
- [ ] 2.5 统一 batch 视频与 manifest range 视频路径，使两者通过同一 `video_quality` 报告 mutation 写回
- [ ] 2.6 将 SAM3 window summary、overlay 和 containment 结果适配为 `sam3_containment` module block 和 evidence 引用

## 3. 统一编排与执行策略

- [ ] 3.1 实现按版本化配置和 registry 运行的资产级 orchestrator，并支持从 `next_module` 恢复
- [ ] 3.2 实现 acceptance profile 的 hard-fail 截断、后续阶段跳过和最终 fail
- [ ] 3.3 实现 supplier_evaluation profile 的 fail 记录后继续、完整执行轨迹和最终 fail 保留
- [ ] 3.4 对 disabled、skipped、not_implemented 和 runtime error 建立互不混淆的状态与测试

## 4. 人工路由输入与批次投影

- [ ] 4.1 从 QC JSON issues/evidence 生成 warn 人工队列输入，停止直接拼接多套 sidecar 结论
- [ ] 4.2 实现仅遍历 `quality_archive/*.json` 的批次投影和按 profile 分组统计
- [ ] 4.3 将现有 batch ledger/weekly report 正式入口迁移到统一投影，并保留遗留结果对账测试
- [ ] 4.4 增加可删除重建的 Parquet/CSV 缓存并验证缓存不参与事实判定

## 5. 文档与端到端验证

- [ ] 5.1 同步 PRD、统一配置、Schema 文档和 reviewer 指南中的双 profile 与唯一事实源规则
- [ ] 5.2 使用 pass、warn、hard fail、runtime error 四类 fixture 验证单资产完整 revision 轨迹
- [ ] 5.3 使用同一批输入端到端验证 acceptance 截断与 supplier_evaluation 全流程的差异
- [ ] 5.4 运行全量测试并记录旧 sidecar 到 QC JSON 的迁移与回滚说明
```

## openspec/changes/unify-qc-dataflow/specs/asset-qc-module-contract/spec.md

- Source: openspec/changes/unify-qc-dataflow/specs/asset-qc-module-contract/spec.md
- Lines: 1-54
- SHA256: 21f4b25e541630ccb1088eed2427054af83ed4539eaccfc380a61a51bf5a968d

```md
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
统一编排器 MUST 区分 disabled、skipped、not_implemented 和执行错误。配置为 enabled 但没有注册实现的模块 MUST 使本次运行保持未完成或显式失败，MUST NOT 自动写成 pass。

#### Scenario: 配置启用尚未实现的 duplicate_check
- **WHEN** 当前代码没有 `duplicate_check` 实现但运行配置将其启用
- **THEN** 编排器记录结构化 unavailable 状态并停止形成最终 Pass
- **THEN** 报告不会包含伪造的 duplicate_check 通过结论
```

## openspec/changes/unify-qc-dataflow/specs/qc-json-batch-aggregation/spec.md

- Source: openspec/changes/unify-qc-dataflow/specs/qc-json-batch-aggregation/spec.md
- Lines: 1-31
- SHA256: 2c3abc07c223048cbc15ee56cdd01fa35c4b7ae9369568ad16dca32e066969cb

```md
## ADDED Requirements

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
```

## openspec/changes/unify-qc-dataflow/specs/qc-pipeline-execution/spec.md

- Source: openspec/changes/unify-qc-dataflow/specs/qc-pipeline-execution/spec.md
- Lines: 1-64
- SHA256: 449342f9c1d450aff2100d63462f80b1f8241a3b7fd3b8b575684835d16511a7

```md
## ADDED Requirements

### Requirement: 流程顺序来自不可变版本化配置
统一编排器 SHALL 按单资产报告顶层 `qc_config` 指向的版本化配置执行模块。本 change MUST 发布 `qc_acceptance_config_schema.v2` 和 `qc_acceptance_v2.0.0`，并保持历史 v1.1.0 文件不可变。一个资产从初始化到最终完成 MUST 使用同一 config version 和 hash；运行中不得切换阈值或模块顺序。

#### Scenario: 配置文件在运行中被修改
- **WHEN** 资产报告已记录配置 hash 后磁盘配置内容发生变化
- **THEN** 后续模块拒绝以新 hash 继续更新该资产
- **THEN** 报告保留原配置引用和未完成状态

#### Scenario: 历史配置保留
- **WHEN** v2 配置发布
- **THEN** `qc_acceptance_v1.1.0.yaml` 的内容和 hash 保持不变
- **THEN** 新运行默认使用 `qc_acceptance_v2.0.0`

### Requirement: 并发边界按资产隔离
统一编排器 MUST 保证同一资产内的模块按配置顺序串行写回，同时 SHALL 允许不同资产并行执行。任何跨资产并行不得共享可变报告状态或绕过每个资产的 revision 校验。

#### Scenario: 两个资产并行运行
- **WHEN** 资产 A 和资产 B 同时进入自动质检
- **THEN** 两个资产可以并行执行各自模块
- **THEN** 每个资产内部的 report revision 仍按模块完成顺序单调递增

### Requirement: 准入模式由 hard fail 截断
`acceptance` profile MUST 在任何自动模块产生 hard fail 时停止后续正常流转，将资产最终结论设为 fail，并跳过该资产的语义校准和 warn 人工质检。

#### Scenario: 关键点模块 hard fail
- **WHEN** `keypoint_presence` 在准入模式产生 fail
- **THEN** 后续自动模块不再执行
- **THEN** 语义校准和 warn 人工质检状态为因 fail 跳过
- **THEN** 最终结论为 fail

### Requirement: 供应商测评模式记录 fail 但不截断
`supplier_evaluation` profile MUST 保留所有自动 fail 及其证据，但 MUST NOT 使用 fail 控制模块流转。只要没有运行时错误，所有配置且有实现的模块和后续人工阶段均应执行，以形成完整供应商能力报告。

#### Scenario: 视频模块 fail 后继续 SAM3
- **WHEN** `video_quality` 在供应商测评模式产生 fail
- **THEN** 报告保留视频 fail issue
- **THEN** 编排器继续执行 `sam3_containment` 和后续适用阶段
- **THEN** 最终结论仍为 fail

### Requirement: Warn 累积但不立即截断
自动 warn MUST 追加到人工候选集合并继续执行后续模块。候选集合 MUST 去重并保留 issue 来源、问题帧区间和证据路径；机器 warn 在人工处理前不得形成最终 Pass 或 Fail。

#### Scenario: 多模块产生 warn
- **WHEN** 同一资产的关键点时序和 SAM3 模块分别产生 warn
- **THEN** 两个稳定 issue ID 均出现在候选集合
- **THEN** 流程继续到语义及人工路由阶段

### Requirement: 最终业务结论只有 Pass 和 Fail
资产在流程未完成时 `overall_decision` MUST 为 null。资产形成最终结论后只能为 pass 或 fail：存在任一自动 hard fail或人工确认 fail 时为 fail；否则在所有必需阶段完成后为 pass。warn、review 和 accept_with_risk MUST NOT 作为最终业务状态。

#### Scenario: 所有机器检查通过且无 warn
- **WHEN** 所有必需自动模块完成且无 hard fail、无 warn 候选，后续必需语义阶段也完成
- **THEN** warn 人工质检被标记为不需要
- **THEN** 最终业务结论为 pass

### Requirement: 运行错误不是质量结论
输入缺失、实现不可用、进程异常或报告写入冲突 MUST 记录为运行错误或未完成状态，MUST NOT 被转换为资产质量 pass/fail。只有成功执行的质量规则才能产生质量结论。

#### Scenario: SAM3 模型加载失败
- **WHEN** SAM3 运行因模型文件不可用而失败
- **THEN** 报告记录运行错误和最后成功模块
- **THEN** `overall_decision` 保持 null
```

