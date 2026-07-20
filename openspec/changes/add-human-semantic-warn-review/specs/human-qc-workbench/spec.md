## ADDED Requirements

### Requirement: 两个任务共用外壳但组件互斥
人工工作台 SHALL 共用资产导航、视频播放器、身份、保存状态和 QC JSON 会话，但 MUST 通过独立 task adapter 渲染语义校准或 warn 人工质检。语义模式不得显示 warn 判定组件；warn 模式不得显示时间轴或 subtask 编辑组件。

#### Scenario: 打开语义校准任务
- **WHEN** 当前任务类型为 semantic_calibration
- **THEN** 界面显示视频、字幕、subtask 时间轴和文字编辑器
- **THEN** 不渲染 warn 原因和 Pass/Fail 控件

#### Scenario: 打开 Warn 质检任务
- **WHEN** 当前任务类型为 warn_review
- **THEN** 界面显示问题证据、warn 原因和 Pass/Fail 控件
- **THEN** 不渲染时间轴拖动和 subtask 文字编辑器

### Requirement: Warn 任务展示最小相关证据
视频类 warn MUST 默认播放 issue context 指定的起止帧片段。21 点骨骼问题 MUST 播放同一问题窗口的原始片段，并展示已有的抽样 overlay PNG；工作台不得合成整段 overlay 视频。

#### Scenario: 打开骨骼连续性 Warn
- **WHEN** issue 包含起止帧和 overlay evidence
- **THEN** 播放器只加载对应问题窗口并显示已有的 21 点抽样 overlay PNG
- **THEN** 操作者可以查看 warn 原因和原始机器指标

### Requirement: 阶段按语义后 Warn 串行切换
同一资产 MUST 只在语义校准完成且没有 pending confirmation 后进入 warn 人工质检。候选为空时 MUST 跳过 warn 模式；候选存在时 MUST 进入 warn 模式并逐项判定。

#### Scenario: 语义阶段完成且存在候选
- **WHEN** HDF5 原子替换成功且 QC JSON 记录语义完成，资产存在 selected warn issues
- **THEN** 工作台切换到 warn_review task adapter

### Requirement: 服务端保存是正式状态源
工作台所有确认操作 MUST 通过服务端 revision-aware API 写入 QC JSON 或语义工作状态。localStorage、导出 JSON 和 CSV 可以作为临时恢复或迁移工具，但 MUST NOT 作为最终事实来源。

#### Scenario: 浏览器刷新
- **WHEN** 操作者在已确认部分编辑后刷新页面
- **THEN** 工作台从服务端最新 revision 恢复已确认状态
- **THEN** 不依赖本地浏览器存储才能继续
