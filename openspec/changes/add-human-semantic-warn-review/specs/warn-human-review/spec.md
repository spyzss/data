## ADDED Requirements

### Requirement: 人工质检只处理累计 Warn
warn 人工质检 SHALL 只为自动模块写入且被路由选中的 warn issue 创建任务。自动检查全部 Pass 且语义校准完成、候选集合为空时，系统 MUST 跳过人工质检并继续形成最终 Pass；本次不得抽取正常样本额外送审。

#### Scenario: 全部自动检查 Pass
- **WHEN** 资产没有自动 hard fail且 warn 候选集合为空，语义校准已完成
- **THEN** manual review 标记为 not_required
- **THEN** 资产无需人工 Pass/Fail 即可进入最终完成流程

### Requirement: 人工结论是机器 Warn 的最终处置
每个选中的 warn issue MUST 获得且只能获得一个最终人工 Pass 或 Fail。人工 Pass MUST 将该 issue 的 effective verdict 设为 pass；人工 Fail MUST 将 effective verdict 设为 fail。机器 verdict、指标、阈值和证据 MUST 保留且不得被人工结论删除或改写。

#### Scenario: 人工确认机器误报
- **WHEN** 机器 warn 经人工判定为 Pass
- **THEN** issue 保留 `machine_verdict=warn` 和原始观测
- **THEN** issue 的人工 verdict 与 effective verdict 为 pass
- **THEN** 该 issue 不导致资产最终失败

#### Scenario: 人工确认真实问题
- **WHEN** 机器 warn 经人工判定为 Fail
- **THEN** issue 的 effective verdict 为 fail
- **THEN** 资产最终结论为 fail

### Requirement: 所有候选完成后才能结束人工质检
manual review MUST 在所有 selected issue 都有最终人工结论后才能标记 completed。存在未判定 issue 时，系统 MUST 禁止完成样本或生成最终业务结论。

#### Scenario: 两个 Warn 只完成一个
- **WHEN** 资产有两个 selected issue且仅一个已人工判定
- **THEN** manual review 保持 in_progress
- **THEN** `final_decision` 保持 null

### Requirement: 自动 Hard Fail 不进入本次人工申诉
本 change 的人工 Pass/Fail MUST NOT 覆盖自动 hard fail。准入模式的 hard fail 已在人工阶段前停止；供应商测评模式即使继续全流程，自动 hard fail 仍保留并使最终资产结论为 fail。

#### Scenario: 测评模式同时有自动 Fail 和人工 Warn Pass
- **WHEN** 供应商测评资产包含一个自动 hard fail且所有 warn 均被人工判定为 Pass
- **THEN** 最终资产结论仍为 fail
- **THEN** 报告同时保留自动 fail 和人工消解 warn 的统计
