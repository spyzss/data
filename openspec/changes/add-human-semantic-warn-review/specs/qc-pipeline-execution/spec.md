## MODIFIED Requirements

### Requirement: 准入模式由 hard fail 截断
`acceptance` profile MUST 在任何自动模块产生 hard fail 时停止后续正常流转，将资产最终结论设为 fail，并跳过该资产的 warn 人工质检和语义校准。未被 hard fail 截断的资产 MUST 先根据累计 warn 进入或跳过人工质检；人工全 Pass 或 `not_required` 后才进入语义校准。

#### Scenario: 关键点模块 hard fail
- **WHEN** `keypoint_presence` 在准入模式产生 fail
- **THEN** 后续自动模块不再执行
- **THEN** 语义校准和 warn 人工质检状态为因 fail 跳过
- **THEN** 最终结论为 fail

#### Scenario: 自动检查全部 Pass
- **WHEN** 资产完成自动阶段且无 hard fail、无 warn
- **THEN** 系统将人工质检标记为 `not_required` 并进入语义校准
- **THEN** 语义完成后形成最终 pass

### Requirement: 供应商测评模式记录 fail 但不截断
`supplier_evaluation` profile MUST 保留所有自动 fail 及其证据，但 MUST NOT 使用机器 fail 控制模块流转。只要没有运行时错误，所有配置且有实现的自动模块、适用的 warn 人工质检和语义校准均应按此顺序执行，以形成完整供应商能力报告；人工 Warn Fail 仍按业务要求终止语义。

#### Scenario: 视频模块 fail 后继续完整流程
- **WHEN** `video_quality` 在供应商测评模式产生 fail
- **THEN** 报告保留视频 fail issue
- **THEN** 编排器继续后续自动模块并进入适用的 warn 人工质检
- **THEN** 人工 Pass/not_required 后继续语义校准
- **THEN** 最终结论仍为 fail

### Requirement: 最终业务结论只有 Pass 和 Fail
资产在流程未完成时 `final_decision` MUST 为 null。资产形成最终结论后只能为 pass 或 fail：存在任一自动 hard fail或人工确认 fail 时为 fail；否则在所有必需自动模块、语义校准和适用的人工质检完成后为 pass。warn、review 和 accept_with_risk MUST NOT 作为最终业务状态。

#### Scenario: 机器 Warn 被人工消解
- **WHEN** 资产没有自动 hard fail且所有机器 warn 均被人工判定为 Pass
- **THEN** 所有必需阶段完成后最终业务结论为 pass
