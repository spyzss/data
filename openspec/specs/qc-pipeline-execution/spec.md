# qc-pipeline-execution Specification

## Purpose
定义基于不可变版本化配置的 QC 流程执行，规范资产内顺序、CAS 恢复，以及双 execution profile 与 external 阶段的编排语义。
## Requirements
### Requirement: 流程顺序来自不可变版本化配置
统一编排器 SHALL 按单资产报告顶层 `qc_config` 指向的版本化配置执行模块。本 change MUST 发布 `qc_acceptance_config_schema.v2` 和不可变版本快照，并保持已经发布的历史快照不可变。新运行 MUST 使用活跃入口当前声明的 `config_version`，且活跃入口 MUST 与该 `config_version` 对应的不可变快照字节一致。一个资产从初始化到最终完成 MUST 使用同一 config version 和 hash；运行中不得切换阈值或模块顺序。

#### Scenario: 配置文件在运行中被修改
- **WHEN** 资产报告已记录配置 hash 后磁盘配置内容发生变化
- **THEN** 后续模块拒绝以新 hash 继续更新该资产
- **THEN** 报告保留原配置引用和未完成状态

#### Scenario: 活跃入口选择当前快照
- **WHEN** 新资产开始运行且 `configs/qc_acceptance.yaml` 声明 `config_version: qc_acceptance_v2.1.0`
- **THEN** 新运行使用 `qc_acceptance_v2.1.0`
- **THEN** 活跃入口与 `configs/qc_acceptance/qc_acceptance_v2.1.0.yaml` 字节一致

#### Scenario: 历史配置保留
- **WHEN** 新配置版本发布并成为活跃版本
- **THEN** `qc_acceptance_v1.1.0.yaml` 和 `qc_acceptance_v2.0.0.yaml` 等已发布快照的内容和 hash 保持不变
- **THEN** 新运行不得因历史文档示例而回退到 `qc_acceptance_v2.0.0`

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
