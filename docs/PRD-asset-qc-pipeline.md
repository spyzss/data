# 资产 QC 串行流水线 PRD

## 1. 目标与事实源

系统为每个资产维护唯一的 `quality_archive/<asset_id>.json`，把机器检测、语义
校准、Warn 人工复核和最终二元结论串成可恢复流程。CSV、XLSX、Markdown、
Parquet、review queue、overlay 和浏览器缓存都是从 QC JSON 派生的视图或证据，
不能成为最终事实源。

## 2. 串行流程

1. 自动模块生成不可变机器 issue，severity 为 warn 或 fail。
2. 适用时进入 `semantic_calibration`，只校准 subtask 共享边界和中英文文字。
3. 语义完成后进入 `manual_review`，逐条判定 selected machine warn。
4. 全部适用阶段完成后，`overall_decision` 只能是 pass 或 fail；运行中、等待人工
   或错误状态必须为 null。

每次业务写入携带 lease token 与 expected report revision。服务端成功写回后
revision 加一；过期 revision 返回 409，lease 冲突或过期返回 423。

## 3. Profile 规则

| Profile | 自动 hard fail | 语义校准 | Warn 复核 | 最终结论 |
|---|---|---|---|---|
| `acceptance` | 立即停止 | 不创建 | 不创建 | fail |
| `supplier_evaluation` | 保留机器 fail 并继续 | 按配置执行 | 复核适用 warn | 仍为 fail，人工 Pass 不得覆盖机器 fail |

没有 machine warn 候选时，Warn 阶段标记 `not_required` 并完成为 pass。

## 4. 语义校准合同

- 内部时间轴使用严格递增共享边界和半开区间；HDF5 对外使用闭区间。
- 只允许拖内部边界手柄；首尾固定，整段不可拖，不支持拆分/合并/删除/平移。
- 一个边界 transaction 必须包含左右两段 before/after；确认和取消都原子作用于
  两段，确认一次只增加一次 `timeline_edit_count`。
- 同一资产最多一个 pending edit；存在 pending 时锁定其他编辑、切换和完成操作。
- 文字修改独立增加 `subtask_text_edit_count`，取消操作不计数。
- 完成时写同目录临时 HDF5，校验 source fidelity、fsync 后 `os.replace`；不生成
  持久备份。

## 5. Warn 复核合同

- `candidate_issue_ids` 保留完整机器候选池；`selected_issue_ids` 是当前人工任务快照。
- 当前 `all_candidates` 策略在 selection 为空时全量快照候选，并记录
  `selection_policy=all_candidates`；已有非空 selection 不覆盖。
- policy seam 后续可换成抽样、风险或预算选择，但不得删改候选池。
- 工作台只消费当前 QC JSON 的 `manual_review.selected_issue_ids`。
- 人工 Pass 表示消解 warn；人工 Fail 表示确认 fail。
- 人工记录写入 `manual_review.issue_reviews[issue_id]`，不得改写顶层机器 issue。
- 所有 selected issue 必须有 verdict 才能完成；自动 hard fail 优先于人工结论。

## 6. 正式批次指标

聚合器只读取当前 QC JSON revision，按 profile 输出：

- `auto_fail_assets`、`auto_fail_issues`；
- `machine_warn_issues`；
- `human_checked_warn_issues`、`human_resolved_warn_issues`、
  `human_confirmed_fail_issues`；
- `timeline_edit_count`、`subtask_text_edit_count`；
- `final_pass_assets`、`final_fail_assets` 和最终通过率。

资产与 issue 分别去重。同一资产存在多个 hard-fail issue 时只增加一个
`auto_fail_assets`；旧 revision 的 issue 和人工计数不得混入当前 revision。
未完成、等待人工或错误资产保留在 incomplete 统计中，但不进入最终通过率分母。

## 7. 迁移和派生输出

旧 manual CSV/progress JSON 只能通过一次性 importer 导入当前 QC JSON。默认
dry-run 并报告 matched/unmatched/conflict，实际写入必须使用 expected revision。
所有批次 CSV、Parquet、Markdown 和 XLSX 使用同一投影字段，不能自行读取旧
人工文件重新计算结论。

正式聚合导出通过 `qc_reporting.export.write_aggregate_outputs` 完成。该 API 从
同一个 `scope/profile/metric/value_json` 长表写出 CSV、Parquet、XLSX `Metrics`
sheet 和 Markdown，禁止为某个格式单独重算指标。缺少任一正式指标时整次导出
失败，以避免跨格式列漂移。
