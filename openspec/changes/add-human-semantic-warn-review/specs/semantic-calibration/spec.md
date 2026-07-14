## ADDED Requirements

### Requirement: 语义校准的进入条件由执行 profile 决定
准入模式下，只有未产生自动 hard fail 的资产 SHALL 进入语义校准；供应商测评模式下，只要自动阶段没有运行错误，包含自动 fail 的资产也 SHALL 进入语义校准。

#### Scenario: 准入模式自动 fail
- **WHEN** 资产在 acceptance profile 的自动阶段产生 hard fail
- **THEN** 语义校准状态标记为因自动 fail 跳过
- **THEN** 不创建该资产的语义编辑任务

#### Scenario: 供应商测评模式自动 fail
- **WHEN** 同一资产在 supplier_evaluation profile 产生 hard fail但自动阶段正常结束
- **THEN** 系统仍创建语义校准任务

### Requirement: 每次语义修改必须独立确认
时间轴边界拖动或一条 subtask 文字修改 MUST 立即进入 pending confirmation。存在 pending confirmation 时，工作台 MUST 禁止修改其他段落、切换任务或完成样本，直到操作者确认或取消本次修改。

#### Scenario: 拖动时间轴后尝试编辑另一条文字
- **WHEN** 操作者拖动一个 subtask 的边界但尚未确认
- **THEN** 另一条 subtask 的文字编辑和时间轴拖动被禁用
- **THEN** 界面只允许确认或取消当前修改

### Requirement: 确认后的修改形成轻量审计
每次确认 MUST 保存编辑类型、subtask 标识、操作者、时间和 before/after。QC JSON MUST 维护独立的 `timeline_edit_count` 与 `subtask_text_edit_count`，且统计次数只由已确认编辑增加，取消操作不得计数。

#### Scenario: 修改后取消
- **WHEN** 操作者拖动边界后点击取消
- **THEN** 时间轴恢复确认前值
- **THEN** `timeline_edit_count` 不增加

### Requirement: 样本完成时原子替换 HDF5
语义阶段完成时，系统 MUST 将所有已确认时间轴和 subtask 文本一次性写入临时 HDF5，完整校验后原子替换原 HDF5。系统 MUST NOT 在本地保留原 HDF5 备份；写入失败时原文件必须保持不变。

#### Scenario: 临时 HDF5 校验失败
- **WHEN** 已确认编辑写入临时文件后结构校验失败
- **THEN** 系统删除临时文件并保留原 HDF5
- **THEN** QC JSON 不得将语义阶段标记为完成

### Requirement: 语义校准不产生 Pass/Fail
当前人工语义校准 SHALL 只产生 completed、skipped 或 error 等流程状态以及修改审计，MUST NOT 产生语义质量 pass/fail 或改变资产最终质量结论。

#### Scenario: 操作者修改多个 subtask
- **WHEN** 操作者确认修改并完成语义阶段
- **THEN** 系统记录完成状态和修改次数
- **THEN** 不生成 semantic fail issue
