---
comet_change: add-human-semantic-warn-review
role: technical-design
canonical_spec: openspec
---

# Warn 人工复核工作台与区间 Overlay 设计

日期：2026-07-21

## 背景

现有 `human_qc` 同时承载语义校准和 Warn 人工复核，页面以问题片段和抽样图片为主，服务端还要求语义先完成、全部 Warning 都判定后才能完成。这些约束与已确认的最终原型不一致。

本设计在《语义校准独立化与人工质检前置设计》的模块切割基础上，完成 Warn-only 工作台的新交互和数据合同。它取代 `2026-07-20-warn-human-review-ui-design.md` 中与本设计冲突的页面和完成规则；语义校准的独立化设计继续有效。

最终流水线为：

`自动 QC → Warn 人工复核 → 语义校准 → 后续模块/完成`

## 目标

- 将 `human_qc` 收敛为 Warn-only 的服务、API 和页面，不再托管语义编辑能力。
- 在整段原视频上显示按真实帧区间定位的 Warning 时间轴，并支持重叠区间的可选弹层。
- 让多个同时命中的 Warning 仍按确定顺序逐个 Pass/Fail，并允许完成前返回修改。
- 支持有 Fail 后提前完成，同时保持尚未查看的 Warning 为原始机器状态。
- 支持资产级人工补充原因；选择人工原因后统一替代所有 Fail 的机器默认原因。
- 对 SAM3 Warning 只在问题区间播放预生成的连续 overlay，生成期间不阻塞整个页面。
- 保留服务端 lease、revision、审计和正式 `asset_qc_report.v2` 状态源。

## 非目标

- 不实现已完成资产的重新打开或撤销完成。
- 不实现人工原因配置管理后台；本次只保留服务端配置接缝和默认选项。
- 不修改 SAM3 模型算法，也不在浏览器或 HTTP 请求中执行模型推理。
- 不生成非问题区间或整段视频的 overlay。
- 不改变机器 issue、指标、阈值和原始 evidence；页面隐藏不等于删除报告字段。
- 不在本次合并到 `main`。

## 模块边界

### `human_qc`

`human_qc` 只负责：

- 从 `asset_qc_report.v2` 创建 Warn 任务投影。
- 资产队列、自动 lease、revision-aware Pass/Fail 和完成操作。
- 原视频与 Warn evidence 的安全媒体路由。
- SAM3 连续 overlay 的任务状态、缓存和浏览器可用 DTO。
- Warn-only 静态页面。

它不得导入 `semantic_calibration`，也不得继续暴露语义时间轴、HDF5 写回或语义编辑 API。

### `semantic_calibration`

语义校准使用独立 Python 包、Service、HTTP server、静态页面和启动入口。它根据持久化报告中的人工复核终态计算 eligibility，不调用 `human_qc` service，也不读取 Warn 页面状态。

### 共享边界

两个模块只通过 `asset_qc_report.v2` 通信。领域无关的原子 JSON 写入、revision 和路径限制可位于 `qc_common`。不得共享业务 Service、HTTP facade、前端 adapter 或 DOM。

## 报告合同

### Issue 级人工判定

`manual_review.issue_reviews` 继续以 issue ID 为键保存已提交的人工判定。每条记录至少包含：

- `verdict`: `pass | fail`
- `effective_verdict`
- `machine_verdict`
- `reviewer`
- `reviewed_at`

机器 issue 的 verdict、指标、阈值、区间和 evidence 保持不可变。完成前重新判定同一 issue 时，旧版本写入 `review_audit`。

未查看的 selected issue 不创建 `issue_reviews`，其 effective 状态仍由机器 Warning 表达；它不得被计入人工检查、人工 Pass 或人工 Fail。

### 完成模式

`manual_review.state=completed` 时必须写入 `completion_mode`：

| 模式 | 门禁 | 结果 |
| --- | --- | --- |
| `all_reviewed` | 所有 selected issue 均已判定，且全部为 Pass | 推进到语义校准 |
| `early_fail` | 至少一个已判定 issue 为 Fail | 允许其他 selected issue 未判定，停止流水线 |

操作员始终显式点击“完成复核”。Pass 最后一个待复核项不会直接替代这个动作。Fail 后可以立即完成，也可以继续查看或修改其他 Warning。

若最后一个 Fail 在完成前被改回 Pass，而仍有未查看的 Warning，则 `early_fail` 门禁立即失效，完成按钮重新禁用。

完成是资产终态。服务端完成操作成功后页面自动进入下一条；底部“下一条”只用于不提交当前草稿的直接切换，二者没有重复提交语义。

### 资产级 Fail 原因

人工补充原因统一存放在：

```json
{
  "manual_review": {
    "failure_reason": {
      "mode": "manual",
      "reason_codes": ["occlusion", "other"],
      "other_text": "操作员填写的原因"
    }
  }
}
```

规则如下：

- 没有选择人工补充原因时，每个 Fail 使用该机器 Warning 的默认原因。
- 选择任意人工原因后，人工原因整体替代所有 Fail 的机器默认原因，并统一绑定到资产。
- 原因支持多选并可随时取消。
- 选择 `other` 后立即显示输入框；提交 Fail 和完成复核时均要求 `other_text` 去空白后非空。
- 原因可以先选、再点 Fail。原因选择本身不改变任何 Warning 状态。
- 选项由任务 DTO 下发；默认配置先由服务端提供，后续可改为读取正式 config。
- 原因选择是页面草稿。点击 Fail 时与目标 issue verdict 原子保存；完成时再次提交最终原因版本。
- 直接上一条/下一条只丢弃未提交原因草稿，不撤销已经保存的 Pass/Fail。

## 流水线状态迁移

### 无人工候选

自动 QC 完成且没有 selected Warn 时，`manual_review.state=not_required`，资产直接进入语义校准。

### 全部 Pass

操作员逐项 Pass，最后显式完成：

- `manual_review.state=completed`
- `manual_review.completion_mode=all_reviewed`
- pipeline `next_module=semantic_consistency`
- 初始化或恢复语义待办

### 任一 Fail

操作员至少提交一个 Fail 后显式完成：

- `manual_review.state=completed`
- `manual_review.completion_mode=early_fail`
- `overall_decision=fail`
- `semantic_calibration.state=skipped_due_to_fail`
- pipeline `status=stopped`、`next_module=null`

未判定 Warning 继续保留机器状态，不补写人工 verdict。

### 自动 Hard Fail

人工复核不得覆盖自动 hard fail。`acceptance` profile 仍可在自动 hard fail 时按既有规则提前停止；`supplier_evaluation` 是否继续采集人工证据由 profile 决定，但人工 Pass 不能把最终机器 hard fail 降级。

## Warn 任务 DTO 与 API

任务 DTO 必须只暴露浏览器需要的安全信息：

- `asset_id`、`report_revision`、`manual_review_state`
- 安全的 `source_video_url`
- 经探测验证的 `fps` 和 `total_frames`
- selected issues：ID、显示名称、规范化半开区间 `[start_frame, end_frame_exclusive)`、机器默认原因、阈值提示、evidence 类型
- `issue_reviews`、审计投影和当前完成门禁
- 人工原因选项与已提交的 `failure_reason`
- SAM3 overlay 的稳定状态：`pending | generating | ready | failed`、区间和安全 URL
- lease 占用状态，不返回内部 token 之外的实现信息

浏览器显示闭区间 `start_frame–(end_frame_exclusive-1)`，业务定位和重叠计算统一使用半开区间。

页面加载资产时自动获取 lease，并在可编辑期间续期。删除显式“获取编辑锁”按钮；423 冲突时进入只读占用态。所有写请求携带 lease token 和 expected revision。

建议 Warn-only API 命名空间为 `/api/warn/...`。原视频和 overlay 媒体路由必须验证资产与 evidence 白名单，并支持原视频 HTTP Range。API 不得返回绝对路径、命令、traceback 或原始异常文本。

## 前端结构

前端继续使用原生 ES modules，拆分为四个职责单一的组件：

- `WarnReviewApp`：任务装载、自动 lease、草稿、写请求、状态迁移和底部资产导航。
- `VideoController`：原视频、当前帧、倍速、键盘逐帧和 overlay 同步。
- `WarningTimeline`：区间投影、重叠合并、颜色块、可进入 popover 和蓝色时间针。
- `ReviewPanel`：当前帧命中的 Warning、状态行、阈值提示、人工原因、Pass/Fail 与完成门禁。

页面标题只显示“Warn 复核”。顶部不再提供上一条、下一条或锁按钮；操作顺序固定为人工原因、Pass/Fail、完成复核、上一条/下一条。

正文不显示机器检测分数和阈值字段。每个 Warning 名称后显示问号，悬停或键盘聚焦时展示该 Warning 的阈值。

## 视频与帧控制

- 主播放器始终加载整段原视频，而不是裁剪的问题片段。
- 只保留播放器区域右下角的“当前帧”；不在视频标题栏重复显示。
- 视频获得点击或键盘焦点后，左/右键才分别移动一帧；页面其他输入控件不会被抢键。
- 移除 `-1 帧`、`+1 帧`按钮。
- 倍速按 `0.25 → 0.5 → 1 → 1.5 → 2 → 3` 逐级减速或加速，并持续显示当前倍速。
- 倍速作为本机偏好保存在 localStorage；它不是业务状态，也不进入 QC 报告。
- 帧号以服务端 DTO 的 FPS 和总帧数为准；媒体 seek 后将目标帧换算为时间，并在 `seeked` 后校准 UI。

## 时间轴投影与交互

### 动态布局

每个 Warning 的位置和宽度按整段视频总帧数计算：

```text
left  = start_frame / total_frames
width = (end_frame_exclusive - start_frame) / total_frames
```

宽度必须反映真实问题时长。色块空间不足以容纳文字时只显示颜色，不缩短真实区间，也不让标签溢出覆盖相邻区域。

### 重叠区间

只有实际相交的 Warning 区间才合并为一个视觉色块；相邻但不相交的区间保持分开。重叠合并只属于视图投影，不改写底层 issue 区间。

例如：曝光异常 `[120,169)` 与画面抖动 `[142,182)` 在时间轴上投影为一个覆盖 `[120,182)` 的色块。播放到：

- 120–141：只点亮曝光异常。
- 142–168：同时点亮曝光异常和画面抖动。
- 169–181：只点亮画面抖动。

悬停或聚焦合并色块后，弹出按起始帧、原始顺序排序的纵向 Warning 列表。弹层与色块之间必须有连续的 hover/focus 可达区域，鼠标可以移入黑色弹层并点击任意 Warning。点击条目只跳到该 Warning 的起始帧。

### 蓝色时间针

蓝色时间针的位置永远由当前帧与总帧数计算。点击 Warning 色块或弹层条目时，视频和时间针都定位到该 Warning 的起始帧；不能使用标签左边缘或合并区间之外的坐标替代帧计算。

操作员仍可点击轨道或拖动时间针到任意帧。拖动、点击轨道、点击色块都只改变视频位置，不改变任何人工判定。

## 当前 Warning 与判定顺序

当前帧命中的所有 Warning 都在 ReviewPanel 的棕色框内显示，框头展示这些 Warning 的联合播放区间；每条只显示 Warning 名称、阈值问号和状态。

判定目标与时间轴定位分离：

- 默认目标是当前帧命中且尚未判定的 Warning 中，起始帧最早者；同起始帧按 selected issue 原始顺序。
- 两个 Warning 同时点亮时，第一次 Pass 仍处理第一个，随后自动选中并定位下一个待复核 Warning。
- Pass 后状态行出现明确的“已通过”标志；Fail 显示“已失败”。
- 只有下方 Warning 状态行能显式选择一个已判定 Warning 进行修改。
- 点击时间轴、重叠弹层或原因按钮不得改变任何 Warning 状态。
- Pass 自动进入下一个待复核 Warning。Fail 保持在当前上下文并启用完成复核。

## SAM3 连续 Overlay

当前 producer 每个窗口只生成少量抽样 overlay PNG。本设计新增区间连续 overlay，但不删除原始抽样 evidence。

### 生成

- 只处理 SAM3 Warning 的问题帧并集；重叠帧只推理和编码一次。
- 由自动 QC/evidence 有界后台 worker 异步生成，复用现有模型锁，禁止在浏览器线程或 HTTP 请求中推理。
- 缓存键至少包含源视频 hash、区间集合、模型/config hash 和 renderer 版本。
- 产物使用临时文件完成后原子发布；失败留下稳定状态和可重试信息，不暴露内部异常。
- 设置并发上限、磁盘缓存上限和淘汰策略，避免多用户重复加载模型或无限增长。

### 播放

原视频是唯一有控制条的主播放器。连续 overlay 是无控制条、静音的同步视频层，只在对应问题区间覆盖主画面，并同步：

- play / pause
- seek / 拖动时间针
- 左右键逐帧
- playbackRate

浏览器只预加载当前和下一个 SAM3 区间的 overlay。离开区间立即隐藏 overlay，继续播放原视频。

### 就绪门禁

SAM3 overlay 为 `pending`、`generating` 或 `failed` 时，只禁止对应 SAM3 Warning 的 Pass/Fail。其他 Warning、视频、时间轴、资产导航和页面操作保持可用。状态轮询或推送变为 `ready` 后自动解锁；失败态显示稳定提示与重试入口。

## 错误处理与恢复

- 409 revision conflict：不覆盖服务端状态，保留当前人工原因草稿，提示刷新最新任务。
- 423 lease conflict：页面进入只读占用态，可播放和查看，不可判定或完成。
- 原视频不可访问：阻断当前资产的人工判定，并显示可操作的稳定错误。
- overlay 不可用：只阻断对应 SAM3 Warning。
- Other 为空、完成模式不合法、没有 Fail 却请求 `early_fail`、或最后一个 Fail 已改回 Pass：服务端拒绝提交。
- 页面刷新从服务端恢复已提交 verdict、原因和 revision；localStorage 只恢复倍速等非业务偏好。
- 完成成功后自动进入下一条。直接切换资产时清空未提交草稿，并从目标资产服务端状态重新开始。

## 实施顺序

1. 扩展报告 schema、聚合和 Warn service，先锁定 `completion_mode`、未判定保留和人工原因合同。
2. 完成语义模块切割与流水线门禁反转，保证 Warn 全 Pass/无候选才进入语义。
3. 增加整段媒体 DTO、HTTP Range、自动 lease 和安全媒体路由。
4. 重构 Warn-only 前端组件，先实现视频、时间轴和判定状态机，再接原因和导航。
5. 扩展 SAM3 evidence worker、连续 overlay 缓存和 readiness 门禁。
6. 做 API、Node、浏览器和完整流水线验收，最后移除旧共用页面入口和遗留文案。

## 测试策略

### Python 与报告合同

- 全部 Pass 后只能以 `all_reviewed` 完成并进入语义。
- 首个 Fail 后可 `early_fail` 完成，未查看 issue 不创建 review。
- 最后一个 Fail 改回 Pass 后重新关闭提前完成门禁。
- 人工原因多选、取消、默认原因替换和 Other 必填。
- 已判定 issue 完成前可修改并产生审计；完成后拒绝修改。
- 409 revision、423 lease、自动 lease 续期和人工 Fail 的语义门禁。
- 聚合只统计实际存在的 issue review，不把未查看项算作人工 Pass/Fail。

### Evidence 与媒体

- 连续 overlay 只覆盖问题区间；多个相交区间对帧去重。
- 缓存命中、model/config/renderer 变更失效、生成失败和原子发布。
- 有界 worker 不在 HTTP 请求中推理；API 错误脱敏。
- 原视频 HTTP Range 与安全路径校验。

### Node/DOM

- 区间动态宽度、相交合并、窄色块隐藏文字和可进入 popover。
- 120–181 重叠示例的逐帧 active issue 集合。
- 色块跳起始帧、时间针拖动、轨道点击、视频聚焦后键盘逐帧。
- 倍速阶梯与本机持久化。
- 同时点亮时第一次 Pass 处理最早 pending Warning；原因选择不改变状态。
- Pass/Fail 修改、完成门禁、上一条/下一条草稿丢弃。
- 原视频与 overlay 的播放、暂停、seek、逐帧和倍速同步。

### 端到端验收

- 完整桌面页面只显示 Warn 复核，操作区顺序与原型一致。
- 三个 Warning 中存在重叠区间时，时间轴、当前帧、当前问题和状态行始终一致。
- SAM3 生成中仅锁定对应 Warning，ready 后自动解锁。
- 全 Pass 完成后自动下一条并进入语义；有 Fail 提前完成后自动下一条且语义被跳过。
- 运行 Python 全量测试、Node 全量测试、OpenSpec strict 验证和 `git diff --check`。
