## MODIFIED Requirements

### Requirement: 自动失败与人工失败分别统计
批次统计 MUST 分别计算自动 hard-fail 资产/issue 数、机器 warn 数、人工检查 warn 数、人工消解 warn 数、人工确认 fail 数、最终 fail 资产数、最终通过率、时间轴修改次数和 subtask 文字修改次数。一个资产内多个 issue MUST NOT 被错误计算为多个资产，取消的语义修改不得计数。

#### Scenario: 人工处理两个 Warn 并修改一次语义
- **WHEN** 一个资产的两个机器 warn 分别被人工判定为 Pass 和 Fail，且确认一次时间轴修改
- **THEN** 人工检查 warn 数增加二
- **THEN** 人工消解 warn 数和人工确认 fail 数各增加一
- **THEN** 时间轴修改次数增加一
- **THEN** 最终 fail 资产数增加一
