# 人工 QC 工作台操作指南

## 1. 工作顺序

每个资产按服务端状态串行处理：自动/SAM3 → Warn 人工复核 → 语义。页面只显示
当前可执行阶段；任务状态、编辑计数和最终结论都以
`quality_archive/<asset_id>.json` 为准。

Warn 使用独立的 8897 服务，语义使用独立的 8898 服务。Warn 页面自动申请并续租
当前 reviewer 的 lease；不需要手动获取锁。lease 期间其他 reviewer 不能修改
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

当前任务选择策略为 `all_candidates`：进入人工阶段时，系统将完整机器候选池
`candidate_issue_ids` 全量快照到 `selected_issue_ids`，并在 JSON 中记录
`selection_policy=all_candidates`。`candidate_issue_ids` 始终保留完整候选池；
`selected_issue_ids` 是本次人工任务快照，已有非空快照不会被自动覆盖。后续可以把
策略替换为抽样、风险或预算选择，但不得因此删改候选池。

问题卡片正文只显示问题帧区间；同一区间的多个 Warn 必须同时逐条列出。每个 Warn 名称后
的“？”悬停时才显示该 Warn 阈值，正文不显示机器指标或阈值字段。

### 3.1 视频、时间轴和 overlay

页面始终播放整条视频，而不是裁剪证据。每个 Warn 使用半开帧区间
`[start_frame, end_frame_exclusive)`，页面显示 `end_frame_exclusive - 1`；这不改变
legacy freeze 的既有语义。时间轴色块的宽度按整段视频帧数计算：点击色块跳转到起始帧，
重叠区域会展开可进入的 popover 以选择具体 Warn；可拖动的播放针可随时移动到任意帧。
空间不足时色块只保留颜色，不堆叠多个色块；重叠 Warn 仍在 popover 中选择。

SAM3 只在问题帧区间实时 overlay，离开区间即隐藏。`pending/failed` 只锁定对应 Warn，
不阻塞其他 Warn；页面会轮询，失败项可重试。不要将 overlay、辅助视觉产物或裁剪文件当成
正式结论。

点击视频区域后，左右键才逐帧移动；输入 Other 文本等表单获得焦点时不会抢占方向键。
速率固定为 0.25×、0.5×、1×、1.5×、2×、3×，页面持续显示当前速率并记住上一次选择。

### 3.2 判定、原因和完成

- Pass：认定机器 Warn 可消解。多个 Warn 同时点亮时，Pass 保存最早的待复核 Warn；
  已通过或已 Fail 的项可从状态条返回修改。
- Fail：点击 Fail 立即保存当前 Warn 的 Fail verdict，但不立即结束资产；操作员仍可返回
  修改已判定项。
- 人工原因可预选、多选和取消；选中 Other 后必须填写非空文字。任何人工补充原因
  覆盖默认原因。
- “完成复核”有两种终态：全部 selected Warn 已 Pass 时提交 `all_reviewed`；已有
  Fail 时提交 `early_fail`。early fail 后未查看的 Warn 保持原状，并自动进入下一条资产。
- 自动 hard fail：不能通过人工 Pass 改为通过。

Warn 逐 issue 结果写入 `manual_review.issue_reviews[issue_id]`，一个问题重复提交时以
当前 revision 的最后一次服务端结果为准，并保留审计记录。Publisher audit 使用独立的
`manual_review.reviews[]`，不能把该列表当作 Warn 映射。完成状态以 JSON 的
`manual_review.state=completed`、`completion_mode`、`completed_at`、`issue_reviews` 和
顶层 `overall_decision` 为准；系统不新增 `human_qc_pass` 标志位。

Warn Gate 状态合同：

- `not_required`：`manual_review.state=not_required`，不写 `completion_mode`；Warn Gate 已满足，语义为 ready。
- `all_reviewed`：`manual_review.state=completed` 且 `completion_mode=all_reviewed`；Warn Gate 已满足，语义为 ready。
- `early_fail`：`manual_review.state=completed` 且 `completion_mode=early_fail`；资产 `pipeline_state.status=stopped`，`semantic_calibration.state=skipped_due_to_fail`。

点击 Fail 立即写入当前 issue 的 `manual_review.issue_reviews[issue_id].verdict=fail`。
人工原因预选、多选、取消或填写 Other 只更新本地草稿，不改变 issue verdict 或资产状态。
点击“完成复核”才写资产级 `completion_mode=early_fail`，停止资产并自动跳转下一条。

### 3.3 启动独立服务

Warn 操作员工作台使用 8897 端口，启动时必须指定批次根目录、QC JSON 目录和当前
reviewer；SAM3 模型路径为可选项：

```bash
python tools/serve_human_qc_workbench.py \
  --batch-root /path/to/batch \
  --quality-archive quality_archive \
  --reviewer reviewer-id \
  --sam3-model /path/to/sam3-model
```

不需要 SAM3 overlay 时省略 `--sam3-model`。语义校准服务独立运行在 8898 端口；Warn `all_reviewed` 或 `not_required` 后才可进入语义，不通过 Warn 页面启动或代替语义校准。

## 4. 冲突与恢复

- 409：页面持有的 `expected_revision` 已过期。刷新当前资产，核对最新状态后再
  重新操作，不能覆盖新 revision。
- 423：lease 缺失、过期或属于其他 reviewer。重新获取 lease 后继续。
- evidence/overlay 失败：仍可查看可用的问题信息或整条视频；不要据此修改机器记录。
- 浏览器刷新：重新读取服务端 task DTO；不得用 localStorage 恢复正式结论。

## 5. 遗留数据边界

旧 manual CSV 和 progress JSON 只允许经一次性导入工具迁移。先执行 dry-run，
确认 matched、unmatched 和 conflict，再带 expected revision 写入。导入完成后，
正式复核、最终结论和批次统计仍只读取 `quality_archive/*.json`。
