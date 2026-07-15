# 人工 QC 工作台操作指南

## 1. 工作顺序

每个资产按服务端状态串行处理：自动 QC → 语义校准 → Warn 复核。页面只显示
当前可执行阶段；任务状态、编辑计数和最终结论都以
`quality_archive/<asset_id>.json` 为准。

开始工作前填写 reviewer 并获取资产 lease。lease 期间其他 reviewer 不能修改
同一资产。页面刷新不会丢失已确认或 pending 的服务端状态。

## 2. 语义校准

时间轴只允许拖动相邻段之间的内部边界手柄。首尾边界固定，不能拖整段色块，
也不能拆分、合并、删除或平移一个 subtask。

拖动一个边界时会同时发生两项联动：

1. 左侧 subtask 的结束边界变化；
2. 右侧 subtask 的起始边界变化。

界面必须同时展示两段 before/after。内部区间是半开区间 `[start, end)`，界面与
HDF5 的闭区间结束帧显示为 `end - 1`。例如内部 `[241, 429)` 显示为帧
241–428。

一次只能存在一个 pending edit。pending 期间其他边界、文字输入、模式切换、
完成样本和下一资产全部锁定：

- “确认并原子写入”会一次性确认两段联动，只增加一次时间轴修改计数；
- “取消本次修改”会同时恢复两段，不增加计数；
- 文字修改也必须逐次确认或取消，确认后只增加一次文字修改计数。

语义完成时，服务端通过同目录临时 HDF5 副本验证并原子替换原文件，只允许
`/label/subtask_label` 的 canonical annotation 变化。系统不保留 `.bak`；若
失败，原文件保持不变并由服务端恢复状态。

## 3. Warn 复核

Warn 页面只展示 QC JSON 已选择的机器问题、问题区间、原因和可用 evidence。
机器指标、阈值、原始 verdict 和 evidence 都不可编辑。

- Pass：认定该机器 Warn 可消解；
- Fail：确认该机器 Warn 是真实失败；
- 自动 hard fail：不能通过人工 Pass 改为通过。

必须为所有 selected warn 选择 Pass 或 Fail 后才能完成资产。一个问题重复提交时
以当前 revision 的最后一次服务端结果为准，并保留审计记录。

## 4. 冲突与恢复

- 409：页面持有的 `expected_revision` 已过期。刷新当前资产，核对最新状态后再
  重新操作，不能覆盖新 revision。
- 423：lease 缺失、过期或属于其他 reviewer。重新获取 lease 后继续。
- evidence/overlay 失败：仍可查看可用的问题信息或短片；不要据此修改机器记录。
- 浏览器刷新：重新读取服务端 task DTO；不得用 localStorage 恢复正式结论。

## 5. 遗留数据边界

旧 manual CSV 和 progress JSON 只允许经一次性导入工具迁移。先执行 dry-run，
确认 matched、unmatched 和 conflict，再带 expected revision 写入。导入完成后，
正式复核、最终结论和批次统计仍只读取 `quality_archive/*.json`。
