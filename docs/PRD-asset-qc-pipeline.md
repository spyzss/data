# 资产 QC 串行流水线 PRD

## 1. 目标与事实源

系统为每个资产维护唯一的 `quality_archive/<asset_id>.json`，把机器检测、SAM3、
Warn 人工复核、语义校准和最终二元结论串成可恢复流程。CSV、XLSX、Markdown、
Parquet、review queue、overlay 和浏览器缓存都是从 QC JSON 派生的视图或证据，
不能成为最终事实源。

## 2. 串行流程

1. 自动/SAM3 生成不可变机器 issue，severity 为 warn 或 fail。
2. 自动/SAM3 → Warn 人工复核 → 语义：适用的 `manual_review` 必须先于
   `semantic_consistency`；Warn 全部通过或 `not_required` 后才可进入语义。
3. `early_fail` 结束资产，不再进入语义；未查看的 Warn 保持原状，不得伪造 verdict。
4. 全部适用阶段完成后，`overall_decision` 只能是 pass 或 fail；运行中、等待人工
   或错误状态必须为 null。

每次业务写入携带 lease token 与 expected report revision。服务端成功写回后
revision 加一；过期 revision 返回 409，lease 冲突或过期返回 423。

## 3. Profile 规则

| Profile | 自动 hard fail | Warn 复核 | 语义校准 | 最终结论 |
|---|---|---|---|---|
| `acceptance` | 立即停止 | 不创建 | 不创建 | fail |
| `supplier_evaluation` | 保留机器 fail 并继续 | 先复核适用 warn | Warn `all_reviewed` 或 `not_required` 后按配置执行 | 仍为 fail，人工 Pass 不得覆盖机器 fail |

没有 machine warn 候选时，Warn 阶段不创建逐 issue 判定；它以 `not_required` 满足
Warn Gate，语义阶段随即可执行。

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
- 人工 Pass 表示消解 warn：当同一时刻多个 Warn 命中时，Pass 必须先保存最早的待复核
  Warn；已判定 Warn 可以返回修改。
- 点击 Fail 立即写入当前 issue 的 `manual_review.issue_reviews[issue_id].verdict=fail`；
  未查看的 Warn 保持原状。
- 人工记录写入 `manual_review.issue_reviews[issue_id]`，不得改写顶层机器 issue。
- `completion_mode=all_reviewed` 要求全部 selected issue 有 Pass；
  `completion_mode=early_fail` 要求已有 Fail。`not_required` 不创建正常 Pass 抽检。
- `manual_review.issue_reviews` 是 Warn 服务的逐 issue 映射；Publisher audit 使用独立的
  `manual_review.reviews[]`，不得互相推导或复用。
- Warn 帧区间用半开 `[start_frame, end_frame_exclusive)`；操作员显示结束帧
  `end_frame_exclusive - 1`。该约定只适用于 Warn，不得改写 legacy freeze 语义。
- Warn 卡片正文只显示问题帧区间；同一区间有多个 Warn 时逐条同时列出。每个 Warn 名称
  后的“？”悬停时才展示该 Warn 阈值，正文不显示机器指标或阈值字段。
- Warn 工作台播放整条视频：时间轴色块按真实帧宽度绘制，重叠色块在 popover 中可选，
  点击色块跳转到起始帧，且可拖动的播放针始终可用。显示空间不足时色块仅显示颜色；
  不堆叠色块，重叠 Warn 仍由 popover 选择。
- SAM3 仅在问题帧区间实时 overlay；pending/failed 只锁定对应 Warn，可轮询和重试。
- 视频聚焦后左右键逐帧；速率为 0.25×、0.5×、1×、1.5×、2×、3× 并记住上次选择。
- Fail 原因可预选、多选和取消，Other 为必填且覆盖默认原因；只选原因不改变 verdict。

Warn Gate 状态合同：

- `not_required`：`manual_review.state=not_required`，不写 `completion_mode`；Warn Gate 已满足，语义为 ready。
- `all_reviewed`：`manual_review.state=completed` 且 `completion_mode=all_reviewed`；Warn Gate 已满足，语义为 ready。
- `early_fail`：`manual_review.state=completed` 且 `completion_mode=early_fail`；资产 `pipeline_state.status=stopped`，`semantic_calibration.state=skipped_due_to_fail`。

点击 Fail 立即写入当前 issue 的 `manual_review.issue_reviews[issue_id].verdict=fail`。
人工原因预选、多选、取消或填写 Other 只更新本地草稿，不改变 issue verdict 或资产状态。
点击“完成复核”才写资产级 `completion_mode=early_fail`，停止资产并自动跳转下一条。

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
