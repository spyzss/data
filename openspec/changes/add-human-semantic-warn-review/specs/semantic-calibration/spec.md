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
时间轴共享边界拖动或一条 subtask 文字修改 MUST 立即进入 pending confirmation。存在 pending confirmation 时，工作台 MUST 禁止修改其他段落、切换任务或完成样本，直到操作者确认或取消本次修改。

#### Scenario: 拖动时间轴后尝试编辑另一条文字
- **WHEN** 操作者拖动一个 subtask 的边界但尚未确认
- **THEN** 另一条 subtask 的文字编辑和时间轴拖动被禁用
- **THEN** 界面只允许确认或取消当前修改

### Requirement: 时间轴使用共享边界联动相邻 Subtask
语义时间轴 MUST 将相邻 subtask 建模为共享边界序列；只有相邻段之间的内部边界可以拖动，整个 subtask 区间不得平移。拖动一个内部边界 MUST 在同一个 pending transaction 中同时更新前一段的结束边界和后一段的起始边界。首段最左边界和末段最右边界 MUST 固定。

#### Scenario: 拖动中间 Subtask 的右边界
- **WHEN** 操作者将第二段与第三段之间的内部半开边界从 411 拖到 429，对应界面将第二段闭区间结束帧从 410 改为 428
- **THEN** 第二段的界面结束帧从 410 更新为 428，第三段的界面起始帧从 411 更新为 429，并处于同一个 pending transaction
- **THEN** 工作台同时展示并高亮第二段和第三段的 before/after
- **THEN** 第一段和其他边界保持不变

#### Scenario: 确认共享边界修改
- **WHEN** 操作者确认一个影响两段的共享边界 pending transaction
- **THEN** 两段的新边界作为一个不可拆分的操作提交
- **THEN** `timeline_edit_count` 只增加一而不是按受影响段数增加

#### Scenario: 取消共享边界修改
- **WHEN** 操作者取消一个共享边界 pending transaction
- **THEN** 两个受影响 subtask 的边界同时恢复到修改前
- **THEN** `timeline_edit_count` 不增加

### Requirement: 共享边界不得破坏时间轴连续性
内部时间轴 MUST 使用严格递增的半开边界序列 `[b_i, b_{i+1})`，并保持原时间轴连续覆盖。工作台和服务端 MUST 拒绝会产生空档、重叠、逆序或零长度区间的边界值；源格式使用闭区间时，source adapter MUST 负责无歧义的帧边界转换。

#### Scenario: 边界越过相邻外侧边界
- **WHEN** 操作者尝试将内部边界拖过前一段起点或后一段终点
- **THEN** 工作台拒绝该边界值且不得创建可确认的 pending transaction
- **THEN** 服务端即使收到等价无效请求也拒绝写入

### Requirement: 确认后的修改形成轻量审计
每次确认 MUST 保存编辑类型、操作者、时间和 before/after。共享边界编辑还 MUST 保存稳定 `boundary_id`、两个受影响的 subtask 标识和两段各自的 before/after。QC JSON MUST 维护独立的 `timeline_edit_count` 与 `subtask_text_edit_count`；统计次数按已确认的用户编辑事务计算，而不是按受影响记录数计算，取消操作不得计数。

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
