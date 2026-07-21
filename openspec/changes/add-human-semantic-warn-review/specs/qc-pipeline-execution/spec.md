## MODIFIED Requirements

### Requirement: 准入模式由自动 Hard Fail 截断
`acceptance` profile MUST 在任何自动模块产生 hard fail 时按既有准入规则停止后续正常流转，将资产最终结论设为 fail，并跳过 Warn 人工复核和语义校准。未被自动 hard fail 截断的资产 MUST 先完成或跳过 Warn 人工复核，再决定是否进入语义校准。

#### Scenario: 自动模块 Hard Fail
- **WHEN** `keypoint_presence` 在准入模式产生 fail
- **THEN** 后续自动模块按既有规则停止
- **THEN** Warn 人工复核和语义校准状态为因 fail 跳过
- **THEN** 最终结论为 fail

### Requirement: 人工质检前置于语义校准
未被自动门禁截断的资产 MUST 按 `自动 QC → Warn 人工复核 → 语义校准` 顺序流转。没有人工候选或所有 selected Warn 最终均为 Pass 时 MUST 进入语义校准；任一人工 Fail 以 `early_fail` 完成时 MUST 停止流水线并将语义校准标记为 `skipped_due_to_fail`。浏览器参数不得绕过该服务端门禁。

#### Scenario: 自动检查无 Warn
- **WHEN** 自动阶段完成且没有人工候选
- **THEN** `manual_review.state=not_required`
- **THEN** 资产进入语义校准

#### Scenario: 所有 Warn 人工 Pass
- **WHEN** manual review 以 `completion_mode=all_reviewed` 完成
- **THEN** pipeline `next_module=semantic_consistency`
- **THEN** 语义服务可以创建或恢复该资产任务

#### Scenario: 人工 Fail 后提前完成
- **WHEN** manual review 以 `completion_mode=early_fail` 完成
- **THEN** pipeline 状态为 stopped 且 `next_module=null`
- **THEN** `semantic_calibration.state=skipped_due_to_fail`

### Requirement: 供应商测评模式保留完整自动结果
`supplier_evaluation` profile MUST 保留所有自动 fail 及其证据，并按该 profile 的既有策略决定是否继续采集人工证据；无论是否继续，人工 Pass MUST NOT 覆盖自动 hard fail。只有未被最终失败门禁终止且完成或跳过 Warn 复核的资产才可进入语义校准。

#### Scenario: 自动 Fail 后继续人工取证
- **WHEN** profile 配置为自动 fail 后仍继续采集 Warn 人工证据
- **THEN** 报告保留自动 fail 和后续人工 review
- **THEN** 最终结论仍为 fail

### Requirement: 最终业务结论只有 Pass 和 Fail
资产在必需阶段未完成时 `final_decision` MUST 为 null。最终结论只能为 pass 或 fail：存在任一自动 hard fail 或人工确认 fail 时为 fail；否则在所有必需自动模块、适用的人工复核和语义校准完成后为 pass。Warn、review 和 accept_with_risk MUST NOT 作为最终业务状态。

#### Scenario: 机器 Warn 被人工消解并完成语义
- **WHEN** 资产没有自动 hard fail、所有 selected Warn 均被人工判定为 Pass，且语义校准完成
- **THEN** 最终业务结论为 pass
