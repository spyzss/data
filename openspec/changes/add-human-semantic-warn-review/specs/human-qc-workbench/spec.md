## ADDED Requirements

### Requirement: 语义校准与人工质检运行时完全分离
语义校准和 warn 人工质检 MUST 使用独立 application facade、HTTP server、API 命名空间和静态资源。两个领域包 MUST NOT 互相导入 service；唯一衔接面 SHALL 为服务端持久化的 `asset_qc_report.v2`。

#### Scenario: 打开语义校准任务
- **WHEN** 操作者打开独立语义服务根地址
- **THEN** 界面显示视频、字幕、subtask 时间轴和文字编辑器
- **THEN** 不渲染 warn 原因和 Pass/Fail 控件
- **THEN** 所有请求只使用 `/api/semantic/...`

#### Scenario: 打开 Warn 质检任务
- **WHEN** 当前任务类型为 warn_review
- **THEN** 界面显示问题证据、warn 原因和 Pass/Fail 控件
- **THEN** 不渲染时间轴拖动和 subtask 文字编辑器
- **THEN** 人工质检 server 不提供语义编辑路由

### Requirement: Warn 任务展示最小相关证据
视频类 warn MUST 默认播放 issue context 指定的起止帧片段。21 点骨骼问题 MUST 播放同一问题窗口的原始片段，并展示已有的抽样 overlay PNG；工作台不得合成整段 overlay 视频。

#### Scenario: 打开骨骼连续性 Warn
- **WHEN** issue 包含起止帧和 overlay evidence
- **THEN** 播放器只加载对应问题窗口并显示已有的 21 点抽样 overlay PNG
- **THEN** 操作者可以查看 warn 原因和原始机器指标

### Requirement: 人工质检完成后才能进入语义校准
同一资产 MUST 先完成或跳过 warn 人工质检。全部 selected issue 为 Pass 时 MUST 以 `all_reviewed` 推进语义；候选为空时 MUST 以 `not_required` 推进语义；任一 Fail MUST 终止资产并禁止语义读取和写入。

#### Scenario: 人工质检全部 Pass
- **WHEN** 所有 selected warn issue 都已人工判定为 Pass
- **THEN** QC JSON 原子记录 manual completion 并将 cursor 推进到 `semantic_consistency`
- **THEN** 独立语义服务开始列出该资产

#### Scenario: 人工质检确认 Fail
- **WHEN** 任一 selected warn issue 被人工判定为 Fail 并以 early-fail 完成
- **THEN** pipeline 停止且语义状态为 `skipped_due_to_fail`
- **THEN** 独立语义服务不得返回该资产的 task DTO

### Requirement: 服务端保存是正式状态源
工作台所有确认操作 MUST 通过服务端 revision-aware API 写入 QC JSON 或语义工作状态。localStorage、导出 JSON 和 CSV 可以作为临时恢复或迁移工具，但 MUST NOT 作为最终事实来源。

#### Scenario: 浏览器刷新
- **WHEN** 操作者在已确认部分编辑后刷新页面
- **THEN** 工作台从服务端最新 revision 恢复已确认状态
- **THEN** 不依赖本地浏览器存储才能继续
