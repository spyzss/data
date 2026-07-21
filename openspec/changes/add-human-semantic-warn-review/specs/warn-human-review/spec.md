## ADDED Requirements

### Requirement: 人工质检只处理累计 Warn
Warn 人工质检 SHALL 只为自动模块写入且被路由选中的 Warn issue 创建任务。系统 SHALL 保留 `candidate_issue_ids` 作为完整机器候选池，并以 `selected_issue_ids` 保存任务快照。当前 `all_candidates` 策略 MUST 在 selection 为空时全量选择候选，但 MUST NOT 覆盖已有非空 selection。

#### Scenario: 非空候选进入人工复核
- **WHEN** 自动 QC 完成且存在两个 machine Warn candidates，当前 selection 为空
- **THEN** 两个 candidate ID 都写入 `selected_issue_ids`
- **THEN** `selection_policy` 为 `all_candidates`，候选池保持不变

#### Scenario: 没有人工候选
- **WHEN** 资产没有 selected Warn
- **THEN** `manual_review.state` 标记为 `not_required`
- **THEN** 资产无需人工 Pass/Fail 即可进入语义校准

### Requirement: 人工结论可在完成前修改
每个已处理的 selected Warn MUST 保存且只能保存一个当前人工 Pass 或 Fail。人工 Pass MUST 将该 issue 的 effective verdict 设为 pass；人工 Fail MUST 将其设为 fail。完成前重新判定同一 issue MUST 替换当前判定并将旧版本写入审计。机器 verdict、指标、阈值、区间和 evidence MUST 保留且不得被改写。

#### Scenario: 修改已通过的 Warning
- **WHEN** 操作者从下方状态行选中已 Pass 的 Warning 并改为 Fail
- **THEN** 当前 issue review 更新为 Fail
- **THEN** 原 Pass 记录进入审计，机器观测保持不变

#### Scenario: 点击时间轴或原因
- **WHEN** 操作者点击时间轴色块、重叠列表或人工原因
- **THEN** 视频位置或原因草稿可以变化
- **THEN** 任何 Warning 的人工 verdict 都不得因此变化

### Requirement: 全部 Pass 或已有 Fail 时可显式完成
`manual_review.state=completed` MUST 同时写入 `completion_mode`。`all_reviewed` MUST 要求所有 selected issue 均已判定且全部为 Pass；`early_fail` MUST 要求至少一个已判定 issue 为 Fail，并 MAY 保留其他 selected issue 未判定。完成操作必须由操作员显式触发。

#### Scenario: 全部 Warning 通过
- **WHEN** 所有 selected issue 都已有 Pass 且操作员点击完成复核
- **THEN** `completion_mode=all_reviewed`
- **THEN** 资产进入语义校准，页面自动进入下一条资产

#### Scenario: 首个 Fail 后提前完成
- **WHEN** 至少一个 selected issue 已被判定为 Fail，其他 issue 尚未查看，操作员点击完成复核
- **THEN** `completion_mode=early_fail`
- **THEN** 未查看 issue 不创建人工 review，也不计入人工检查、Pass 或 Fail
- **THEN** 流水线停止，语义校准标记为 `skipped_due_to_fail`

#### Scenario: 最后一个 Fail 改回 Pass
- **WHEN** 操作者在完成前将最后一个 Fail 改为 Pass，且仍有未查看 issue
- **THEN** `early_fail` 完成门禁失效
- **THEN** manual review 保持 in_progress

### Requirement: 默认原因与资产级人工补充原因互斥生效
系统 MUST 为 Fail 保留对应机器 Warning 的默认原因。`manual_review.failure_reason.reason_codes` SHALL 支持多选并统一绑定到资产；只要选择任一人工补充原因，所选人工原因 MUST 整体替代所有 Fail 的默认原因。原因可在 Fail 前选择或取消，原因选择本身 MUST NOT 改变 verdict。

#### Scenario: 未选择人工补充原因后 Fail
- **WHEN** 操作者未选择人工补充原因并提交 Fail
- **THEN** 该 Fail 使用机器 Warning 的默认原因

#### Scenario: 多选人工补充原因后 Fail
- **WHEN** 操作者选择一个或多个人工原因并提交 Fail
- **THEN** 资产保存统一的人工原因集合
- **THEN** 所有 Fail 的机器默认原因均不再作为有效 Fail 原因

#### Scenario: 选择其他
- **WHEN** 操作者选择 `other`
- **THEN** 页面立即显示文字输入框
- **THEN** 输入去空白后为空时，服务端拒绝 Fail 和完成提交

### Requirement: 整段时间轴按真实帧区间投影 Warning
工作台 MUST 根据服务端提供的 `total_frames` 和半开问题区间计算每个色块的真实位置与宽度。实际相交的 Warning MAY 合并为一个视觉色块，但底层 issue、区间和人工状态 MUST 保持独立。窄色块空间不足时 MUST 只显示颜色，不得堆叠多个色块或让文字溢出。

#### Scenario: 两个问题区间部分重叠
- **WHEN** 曝光异常覆盖 `[120,169)`，画面抖动覆盖 `[142,182)`
- **THEN** 时间轴显示一个覆盖 `[120,182)` 的合并视图色块
- **THEN** 120–141 只激活曝光异常，142–168 同时激活两个 Warning，169–181 只激活画面抖动

#### Scenario: 悬停重叠色块
- **WHEN** 操作者悬停或聚焦包含多个 Warning 的色块
- **THEN** 页面纵向展开可进入、可点击的 Warning 列表
- **THEN** 操作者可以把鼠标移入弹层并选择任一 Warning 跳到其起始帧

### Requirement: 视频定位与判定状态互相独立
点击 Warning 色块或重叠列表 MUST 将视频和蓝色时间针定位到所选 Warning 的起始帧。蓝色时间针 MUST 支持轨道点击和拖动到任意帧。所有定位操作 MUST NOT 自动 Pass、Fail 或切换 issue verdict。

#### Scenario: 点击重叠色块中的第二个 Warning
- **WHEN** 操作者点击起始帧为 142 的 Warning
- **THEN** 视频当前帧和蓝色时间针都定位到 142
- **THEN** 起始帧为 120 的 Warning 状态保持不变

### Requirement: 同时命中时按确定顺序逐项判定
当前帧命中的 Warning MUST 同时显示。默认判定目标 MUST 是其中起始帧最早的未判定 Warning；起始帧相同时按 selected issue 原始顺序。第一次 Pass MUST 处理第一个目标并自动进入下一个待复核 Warning，已 Pass 项 MUST 显示明确通过标志。

#### Scenario: 当前帧同时命中两个未判定 Warning
- **WHEN** 两个 Warning 同时点亮且操作者第一次点击 Pass
- **THEN** 起始帧更早的第一个 Warning 被判定为 Pass
- **THEN** 第二个 Warning 保持未判定并成为下一个目标

### Requirement: 支持聚焦后的逐帧和持久倍速
视频获得点击或键盘焦点后，左、右键 MUST 分别移动一帧；未聚焦时不得拦截页面方向键。页面 SHALL 提供 `0.25、0.5、1、1.5、2、3` 六档逐级速度，持续显示当前倍速并在资产切换后保留上一次本机设置。页面 MUST NOT 显示独立的 `-1 帧`、`+1 帧`按钮。

#### Scenario: 视频未聚焦时按方向键
- **WHEN** 焦点位于人工原因输入框或页面其他控件
- **THEN** 左右键不会改变视频帧

### Requirement: SAM3 仅在问题区间使用连续 Overlay
SAM3 Warning MUST 使用后台 worker 预生成并缓存只覆盖问题帧并集的连续 overlay 视频；重叠帧 MUST 去重处理。浏览器和 HTTP 请求 MUST NOT 执行 SAM3 推理。主播放器 MUST 始终播放整段原视频，overlay 作为无控制条同步层仅在对应问题区间显示。

#### Scenario: 两个 SAM3 区间重叠
- **WHEN** 后台任务生成相交区间的连续 overlay
- **THEN** 重叠帧只推理和编码一次
- **THEN** overlay 与原视频的播放、暂停、seek、逐帧和倍速保持同步

#### Scenario: Overlay 尚未就绪
- **WHEN** 当前 SAM3 Warning 的 overlay 状态为 pending、generating 或 failed
- **THEN** 仅该 Warning 的 Pass/Fail 被禁用
- **THEN** 页面其他 Warning、视频、时间轴和资产导航仍可使用
- **THEN** 状态变为 ready 后该 Warning 自动解锁

### Requirement: 自动 Hard Fail 不进入本次人工申诉
人工 Pass/Fail MUST NOT 覆盖自动 hard fail。供应商测评模式即使继续采集人工证据，自动 hard fail 仍保留并使最终资产结论为 fail。

#### Scenario: 自动 Fail 与人工 Warn Pass 并存
- **WHEN** 资产包含自动 hard fail 且所有已处理 Warn 均被人工判定为 Pass
- **THEN** 最终资产结论仍为 fail
- **THEN** 报告同时保留自动 fail 和人工消解 Warn 的统计
