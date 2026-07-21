## ADDED Requirements

### Requirement: Warn 复核与语义校准使用独立工作台
系统 MUST 将 Warn 人工复核和语义校准实现为独立 Python 模块、Service、HTTP API、静态页面和启动入口。`human_qc` SHALL 只承载 Warn 复核；`semantic_calibration` SHALL 只承载 subtask 时间轴和文字校准。两者 MUST NOT 共享业务 Service、HTTP facade、前端 adapter 或 DOM，唯一业务通信面 MUST 是持久化 `asset_qc_report.v2`。

#### Scenario: 打开 Warn 复核入口
- **WHEN** 操作者打开 Warn 人工复核服务
- **THEN** 页面标题为“Warn 复核”并只渲染 Warn 视频、时间轴、原因和判定控件
- **THEN** 页面不加载语义时间轴、subtask 文字编辑器或 HDF5 写回 API

#### Scenario: 打开语义校准入口
- **WHEN** 操作者打开语义校准服务
- **THEN** 页面只渲染语义视频、subtask 时间轴、文字编辑和确认控件
- **THEN** 页面不加载 Warn evidence、人工原因或 Pass/Fail 控件

### Requirement: Warn 页面播放整段原视频
Warn 工作台 MUST 播放整段原视频并使用服务端验证的 FPS 与总帧数映射业务帧。页面 MUST 在播放器区域右下角显示唯一的当前帧信息；MUST NOT 只加载问题片段，也不得在标题栏重复显示当前帧。

#### Scenario: 打开包含多个问题区间的资产
- **WHEN** 资产包含位于不同帧段的多个 selected Warn
- **THEN** 主播放器可连续播放整段原视频
- **THEN** 页面可以跳转到任意问题的起始帧，也可以拖动到任意非问题帧

### Requirement: 页面自动管理编辑租约
Warn 工作台 MUST 在加载可编辑资产时自动获取并续期 reviewer lease，且 MUST 保留 expected revision 冲突保护。页面 MUST NOT 显示“获取编辑锁”按钮；租约冲突时 SHALL 进入只读占用态。

#### Scenario: 资产已被其他操作员占用
- **WHEN** 自动获取 lease 返回占用冲突
- **THEN** 页面允许查看视频和已有状态
- **THEN** 页面禁止 Pass、Fail 和完成操作并显示稳定的占用提示

### Requirement: 服务端保存是正式状态源
工作台所有 Pass、Fail、原因和完成操作 MUST 通过 revision-aware API 原子写入 QC JSON。localStorage MUST NOT 作为业务状态源，只可保存倍速等非业务偏好。直接切换资产 MUST 丢弃未提交草稿，并从目标资产的服务端状态开始。

#### Scenario: 浏览器刷新
- **WHEN** 操作者在已提交部分判定后刷新页面
- **THEN** 工作台从服务端最新 revision 恢复已提交状态
- **THEN** 未提交原因草稿不会被当成正式人工结论
