# Brainstorm Summary

- Change: add-human-semantic-warn-review
- Date: 2026-07-21

## 确认的技术方案

### 模块边界与流水线

- 采用“先完成模块切割，再落地 Warn 原型”的方案。
- `human_qc` 成为 Warn-only 模块；`semantic_calibration` 使用独立 Python 包、Service、HTTP API、静态页面和启动入口。
- 最终流水线为 `自动 QC → Warn 人工复核 → 语义校准 → 后续模块/完成`。
- 两个模块只通过持久化 `asset_qc_report.v2` 通信，不共享业务 Service、HTTP facade、前端 adapter 或 DOM。
- 人工全部 Pass 后才进入语义校准；任一人工 Fail 且操作员显式完成后，流水线停止，`semantic_calibration.state=skipped_due_to_fail`。
- 页面移除显式“获取编辑锁”按钮，但保留服务端 lease/revision。加载资产时自动获取和续期；冲突时页面进入只读占用态。

### Warn 完成与原因合同

- `manual_review.state=completed` 支持 `completion_mode=all_reviewed` 与 `completion_mode=early_fail`。
- `all_reviewed` 要求所有 selected issue 都已判定且全部为 Pass。
- `early_fail` 要求至少一个 Fail，允许其他 selected issue 未判定。
- 未判定 issue 不创建 `issue_reviews`，机器 Warn 原始状态保持不变，也不计入人工检查、Pass 或 Fail 数。
- 最后一个 Fail 在完成前被改回 Pass 且仍有未判定 issue 时，完成门禁重新关闭。
- 已判定 issue 在资产完成前可重复修改并写入审计；资产完成后为终态，重新打开不在本次范围。
- 人工补充原因存放在资产级 `manual_review.failure_reason`，包含 `mode`、多选 `reason_codes` 和可选 `other_text`。
- 未选择人工原因时，Fail 使用对应机器 Warn 默认原因；任一人工原因被选择后，统一替换所有 Fail 的默认原因。
- `other` 被选中时文字必填。原因选项由服务端 DTO 下发，前端通过默认配置 seam 消费，后续接正式 config。
- 原因选择作为页面草稿；点击 Fail 时与 issue verdict 原子保存，点击完成时再次提交最终版本。直接资产导航只丢弃未提交原因草稿，不回滚已经保存的 Pass/Fail。

### 前端与时间轴

- 前端按 `WarnReviewApp`、`VideoController`、`WarningTimeline`、`ReviewPanel` 分责，继续使用原生 ES modules，不引入新框架。
- 服务端 DTO 提供安全媒体 URL、FPS、总帧数和规范化半开 Warn 区间；页面显示闭区间并按真实总帧数计算位置与宽度。
- 只有实际相交的区间合并为一个视图色块；底层 issue 状态独立。窄色块隐藏文字，悬停或聚焦弹出可进入、可点击的纵向 Warn 列表。
- 时间轴 Warn 点击只定位到起始帧，不改变 verdict；蓝色时间针可拖动和点击轨道定位。
- 视频获得焦点后左右键逐帧；倍速按 0.25、0.5、1、1.5、2、3 逐级切换，并只作为本机偏好保存。
- 当前帧命中多个 Warn 时同时展示。默认判定目标是起始帧最早的未判定 Warn；第一次 Pass 始终处理第一个并自动进入下一个。
- Fail 处理当前判定目标并启用完成按钮。时间轴选择与判定选择分离；只有下方 Warn 状态行能明确选择某个 Warn 进行修改。
- 页面标题为“Warn 复核”，移除锁按钮；上一条、下一条放在页面底部。直接切换资产不保存未提交草稿。
- 正文不再显示机器检测分数和阈值；每个 Warn 名称后的问号悬停时显示该 Warn 阈值。

### SAM3 连续 Overlay

- 当前 SAM3 producer 每个候选窗口只抽样 3–5 帧并生成静态 `combined_overlay` PNG；连续 overlay 需要扩展 evidence 产物。
- 只对 SAM3 Warn 的问题帧区间生成连续 overlay，不处理整条视频；重叠区间按帧并集合并并只推理和编码一次。
- 禁止在浏览器主线程或 HTTP 请求中运行 SAM3。在自动 QC / evidence worker 中异步预生成并缓存 overlay 视频。
- 缓存键绑定源视频 hash、问题区间、模型/config hash 和 renderer 版本；任务 DTO 只暴露安全 URL、起止帧和稳定状态码。
- 主播放器始终承载整段原视频；SAM3 evidence 使用无控制条的同步视频层，仅在对应区间覆盖显示，并同步播放、暂停、seek、逐帧和倍速。
- 只预加载当前和下一个 SAM3 区间。
- 连续 overlay 未生成或生成失败时，对应 SAM3 Warn 禁止 Pass/Fail；页面其他浏览、时间轴和资产导航保持可用，生成成功后自动解锁。

### 错误、恢复与性能

- 409 revision conflict 不覆盖服务端状态；保留当前原因草稿并要求刷新。
- 423 lease conflict 进入只读占用态。
- 原视频不可访问时阻断当前资产判定；SAM3 evidence 非 ready 只阻断对应 Warn。
- `other` 为空、完成模式不合法或 Fail 被改回 Pass 后，服务端必须再次拒绝提交。
- API 不暴露命令、绝对路径或异常原文。
- 原视频支持 HTTP Range。SAM3 overlay 使用有界后台 worker 和现有模型锁，绝不随网页并发重复加载模型。
- 浏览器只预加载当前和相邻 overlay，并设置缓存淘汰上限。

## 关键取舍与风险

- Fail 提前完成改变既有“所有 selected issue 都必须判定”的终态不变量；通过显式 `completion_mode` 区分全 Pass 与提前 Fail，避免把未判定隐式算作 Pass。
- 人工原因是资产级多选，而 verdict 是 issue 级；原因单独放在 `manual_review.failure_reason`，避免复制成多个不一致的 issue 字符串。
- 全视频 URL、FPS 和总帧数必须由服务端 DTO 提供并验证，前端不得只依赖浏览器 duration 推断业务帧号。
- 时间轴重叠合并只属于视图投影，不能改写底层 issue 区间或判定。
- 连续 SAM3 overlay 比当前抽样证据昂贵。以 120–181 帧为例，需要 62 帧推理，约为 5 帧抽样的 12.4 倍，但比 60 秒、30 fps 的整段 1800 帧处理少约 29 倍。成本放在离线 worker，不阻塞网页主线程。
- 严格 evidence 就绪策略保障 SAM3 判定依据完整，但可能让单个 Warn 等待；页面和其他任务仍保持可用。
- 采用先切割后落地会增加前期改动量，但避免在旧共用外壳上实现一次、语义切割后再返工一次。

## 测试策略

- Python 合同与服务测试：全 Pass、提前 Fail、未判定保留、人工原因覆盖、Other 必填、revision/lease、人工 Fail 后语义门禁、旧报告兼容与迁移。
- SAM3/evidence 测试：只处理问题区间、重叠帧去重、异步状态、缓存命中/失效、生成失败和安全错误投影。
- Node 单元测试：区间合并、帧命中、动态宽度、Pass 目标顺序、原因草稿、完成门禁、已判定项修改、基础视频与 overlay 同步。
- DOM/静态合同测试：完整视频控制、时间轴、可进入的重叠 popover、底部操作顺序、无锁按钮、无机器分数正文。
- API 测试：HTTP Range、自动 lease、overlay 状态轮询、内部错误脱敏和 revision 冲突。
- 浏览器验收：桌面完整流程、重叠区间 120–181、时间针拖动、键盘逐帧、倍速持久化、Fail 原因与 Other 必填、SAM3 锁定/解锁、完成后自动下一条。
- 最终运行 Python 全量测试、Node 全量测试、OpenSpec strict 验证和 `git diff --check`。

## Spec Patch

- 修改 warn-human-review 的完成约束：全 Pass 路径仍要求全部判定；任一 Fail 后允许显式 `early_fail` 完成，未判定 issue 保持未判定且不得计入人工检查数。
- 新增资产级人工补充原因多选、默认原因替换、Other 必填和原因草稿提交场景。
- 新增整段视频媒体 DTO、真实帧数时间轴、重叠区间投影、问题跳转、时间针拖动、逐帧键盘和倍速场景。
- 新增 SAM3 问题区间连续 overlay 的异步生成、缓存、就绪门禁和局部失败场景。
- 修改 human-qc-workbench：人工质检为 Warn-only 页面；语义校准使用独立模块、API 和页面。
- 修改 qc-pipeline-execution：流水线顺序为自动 QC → 人工质检 → 语义校准；人工全 Pass/无候选进入语义，任一人工 Fail 完成后停止。
- 保持机器 issue、指标、阈值和 evidence 在正式报告中不可变；UI 只移除分数正文，阈值通过问号提示展示。
- 本次不实现已完成资产重新打开、正式原因 config 管理后台、SAM3 模型算法调整、非问题区间 overlay 或合并到 main。
