# Comet Design Handoff

- Change: add-human-semantic-warn-review
- Phase: design
- Mode: compact
- Context hash: 8ae05893aee75bbaf34c255c673e096a054a56f5ebf27a937ea633e5e1154d96

Generated-by: comet-handoff.sh

OpenSpec remains the canonical capability spec. This handoff is a deterministic, source-traceable context pack, not an agent-authored summary.

## openspec/changes/add-human-semantic-warn-review/proposal.md

- Source: openspec/changes/add-human-semantic-warn-review/proposal.md
- Lines: 1-37
- SHA256: 118f10c3656c48194fdd7be26203ae0c2fa9eba0c6f8996904448e85006ae542

```md
## Why

当前静态人工复核页面只围绕独立 review queue、CSV 和浏览器进度文件工作，既没有语义时间轴校准，也没有将人工判定作为 warn 的最终处置写回单资产 QC JSON。人工操作因此无法成为可审计、可统计、可恢复的正式流水线步骤。

需要在 `unify-qc-dataflow` 的统一报告合同之上，提供一个串行但解耦的共用工作台：先完成语义校准，再仅对累计 warn 做人工 Pass/Fail，并将结果写回每条数据的最终质量报告。

## What Changes

- 新增共用人工工作台外壳，并以任务 adapter 隔离语义校准和 warn 人工质检。
- 语义校准模式只显示视频、字幕、subtask 时间轴和文字编辑器，不显示 warn 质检组件。
- warn 人工质检模式只显示问题视频段或骨骼 overlay、warn 原因和 Pass/Fail，不显示语义编辑组件。
- 每次时间轴拖动或单条 subtask 文字修改都必须立即确认或取消；未确认时锁定其他段落和下一阶段操作。
- 确认后的语义修改在样本语义阶段完成时一次性原子替换 HDF5；不保存本地原 HDF5 备份。
- QC JSON 只统计 `timeline_edit_count` 和 `subtask_text_edit_count`，同时允许保留 before/after 审计明细。
- 自动检查全部 Pass 且语义校准完成时，跳过 warn 人工质检并形成最终 Pass。
- 机器 warn 的人工判定是该 warn 的最终处置：人工 Pass 消解 warn，人工 Fail 将其升级为确认失败；机器原始观测仍保留用于审计。
- 准入模式下，自动 hard fail 数据不进入语义校准；供应商测评模式下，自动 fail 不截断，所有数据仍执行语义和适用的 warn 人工质检。
- 本次不实现模型语义检查、语义 Pass/Fail、自动 hard-fail 申诉或正常样本人工抽检。

## Capabilities

### New Capabilities

- `semantic-calibration`: 规定人工 subtask 时间轴/文字校准、逐次确认、HDF5 原子替换和修改次数审计。
- `warn-human-review`: 规定 warn 候选展示、人工 Pass/Fail、effective verdict 和最终资产结论。
- `human-qc-workbench`: 规定两个串行任务共用工作台但组件互斥、状态隔离和任务切换条件。

### Modified Capabilities

- `qc-pipeline-execution`: 在 `unify-qc-dataflow` 的统一编排能力上补充人工语义与 warn 复核阶段的完成条件。
- `qc-json-batch-aggregation`: 补充人工 Pass/Fail、误报率和语义修改次数的统计来源。

## Impact

- 主要影响人工复核服务、HTML/前端工作台、HDF5 文本写回、QC JSON Schema、人工状态机和批次统计字段。
- 依赖 `unify-qc-dataflow` 先提供统一单资产报告、profile 和路由接口。
- 现有 `manual_labels.csv` 和 progress JSON 可提供迁移读取，但不再是最终事实来源。
```

## openspec/changes/add-human-semantic-warn-review/design.md

- Source: openspec/changes/add-human-semantic-warn-review/design.md
- Lines: 1-78
- SHA256: f13f279074fe5877cf9eb5bd6a988896a7b9bd7099bc741a6c79cff086648b6b

```md
## Context

`unify-qc-dataflow` 将提供单资产 QC JSON、执行 profile、稳定 issue/evidence 和人工路由状态。本 change 在该合同上替换现有以静态 HTML、CSV、localStorage 和 progress JSON 为主的人工流程，并实现用户确认的串行关系：自动 QC Gate → 语义校准 → 对累计 warn 做人工 Pass/Fail → 最终二元结论。

## Goals / Non-Goals

**Goals:**

- 为语义校准和 warn 人工质检提供一个共用但解耦的工作台。
- 保证每次语义修改可定位、可确认、可取消且不会误改其他段落。
- 在样本级安全地原子替换 HDF5，不保留本地原文件备份。
- 让人工 Pass/Fail 成为机器 warn 的最终处置，并写回 QC JSON。
- 记录语义修改次数、人工误报和人工确认失败等质量指标。

**Non-Goals:**

- 不实现模型语义判断。
- 不给语义校准设置 Pass/Fail。
- 不提供自动 hard fail 申诉或人工覆盖。
- 不对全 Pass 正常样本进行额外人工抽检。
- 不把原始 HDF5 长期复制到本地作为备份。

## Decisions

### 1. 工作台外壳与任务 adapter 分离

共用层负责资产队列、视频播放、reviewer、报告 revision、保存错误和阶段导航。`SemanticCalibrationAdapter` 只管理 subtask/时间轴编辑；`WarnReviewAdapter` 只管理 issue/evidence 和 Pass/Fail。adapter 通过显式 task state 通信，不互相读取 DOM。

相比制作两个独立页面，该方案复用播放器和会话；相比一个同时展示所有组件的大页面，它能保证后期用模型替换语义模块时不影响人工 warn 质检。

### 2. 每次编辑使用单一 pending transaction

前端任一时刻最多存在一个 pending edit，记录 before/after 和编辑类型。进入 pending 后锁定其他 segment 和阶段切换；确认后调用服务端 API 并增加相应计数，取消则还原。样本完成前服务端再次检查没有 pending edit。

时间轴在统一 contract 中不表示为互相独立的 subtask 区间，而表示为有序共享边界 `b_0...b_n`，其中第 `i` 段为半开区间 `[b_i, b_{i+1})`。只有内部边界 `b_1...b_{n-1}` 显示拖动手柄；首段最左边界和末段最右边界固定，整个 subtask 色块不得拖动。源 HDF5 若使用闭区间，source adapter 在读写时负责将 `[start_frame, end_frame]` 与内部半开区间互转；例如界面闭区间结束帧 410 对应内部下一边界 411。

拖动内部边界 `b_i` 是一个共享边界事务：它同时修改前一段的结束边界和后一段的起始边界。pending edit 必须记录 `boundary_id`、操作者主动拖动的手柄、两个受影响的 subtask ID，以及两段各自的 before/after；界面同时高亮并展示两段变化。确认或取消必须原子作用于两段，一次确认只增加一次 `timeline_edit_count`，不得因为两段记录都变化而重复计数。

共享边界必须始终严格单调并覆盖原时间轴，服务端和前端都要拒绝产生空档、重叠、逆序或零长度区间的拖动。边界不得越过相邻的外侧边界；本次不实现整段平移、级联挤压第三段、拆分、合并或删除 subtask。

### 3. HDF5 使用临时副本校验后原子替换

服务端从当前 HDF5 创建同目录临时文件，将所有已确认语义结果写入目标 dataset，重新打开并验证结构、JSON 和帧/时间轴边界，然后 fsync + `os.replace`。不生成持久 `.bak`。QC JSON 只在替换成功后记录 semantic completed 和最终 hash。

### 4. 机器观测与有效结论并存

issue 保留机器 verdict、指标、阈值和 evidence；人工 review 追加 reviewer、时间、Pass/Fail 和可选原因。`effective_verdict` 对 warn 取人工结论。这样人工判断真实决定最终质量，同时保留误报率和规则改进所需证据。

### 5. Evidence 采用问题窗口和缓存 overlay

视频 warn 使用 issue context 的起止帧转成短片播放范围。21 点骨骼 overlay 优先使用自动模块生成的 evidence；缺失时由服务端按 issue window 生成并缓存，不为整段视频实时叠加，以控制 CPU、解码和网络开销。

### 6. 正式状态全部服务端持久化

浏览器每次确认都带 expected report revision。服务端原子更新工作状态/QC JSON并返回新 revision。localStorage 和 JSON/CSV 导出仅保留为迁移及故障恢复辅助，不参与最终批次统计。

## Risks / Trade-offs

- **无持久 HDF5 备份提高写错风险** → 临时副本全量校验、同目录原子替换、替换前后 hash 和 dataset 结构测试必须全部通过。
- **逐次确认增加操作次数** → 提供明确键盘操作和 pending 高亮，但不合并确认，优先满足防误操作要求。
- **长视频 overlay 成本高** → 只处理 issue window、按需缓存，限制并发生成任务。
- **并发 reviewer 冲突** → 使用 expected revision 和任务 lease；过期提交被拒绝并要求刷新。
- **语义结构因供应商不同而变化** → HDF5 读写由 source adapter 封装，工作台只消费统一 subtask contract。

## Migration Plan

1. 扩展 QC JSON 的 semantic/manual review 状态、review 和审计 Schema。
2. 实现语义工作状态、HDF5 临时写入/校验/替换服务。
3. 重构现有人工复核服务器为 revision-aware API，并提供遗留 CSV/progress 导入。
4. 实现共用工作台及两个 task adapter。
5. 接入 profile 路由：acceptance 自动 fail 跳过；supplier_evaluation 全流程继续。
6. 将人工与修改次数统计接入 QC JSON 聚合器。

回滚时停用新工作台写 API并保留只读 QC JSON；已经原子替换的 HDF5 不自动回滚，因为用户明确要求不保留本地原版，原始数据仍由云端保存。

## Open Questions

无阻塞问题。完整 JSON 字段名将在获取同事最终数据格式后通过 source adapter 映射，不改变工作台和 QC JSON 的稳定合同。
```

## openspec/changes/add-human-semantic-warn-review/tasks.md

- Source: openspec/changes/add-human-semantic-warn-review/tasks.md
- Lines: 1-76
- SHA256: 6be267b90ea24a9ef8440f5ac7745be79093553dde8844a35043c665c58531b9

```md
## 1. 人工阶段报告合同

- [x] 1.1 扩展 QC JSON Schema，加入 semantic calibration 状态、pending edit、审计计数和 HDF5 hash
- [x] 1.2 扩展 manual review Schema，加入 selected issue、逐 issue 人工 verdict、effective verdict 和完成约束
- [x] 1.3 更新批次投影，统计人工检查/消解/确认失败以及两类语义修改次数

## 2. 语义校准服务

- [x] 2.1 为供应商 HDF5 实现统一 subtask 读取和共享边界规范化 adapter，明确闭区间与内部半开区间转换
- [x] 2.2 实现只允许内部边界手柄拖动的双段联动 pending edit 状态机、expected revision 校验以及原子确认/取消 API
- [x] 2.3 实现临时 HDF5 写入、结构与内容校验、fsync 和无备份原子替换
- [x] 2.4 验证共享边界联动、两段 before/after、非法边界拒绝、单事务计数、文字修改、取消操作和写入失败时的原文件安全性

## 3. Warn 人工质检服务

- [x] 3.1 从 QC JSON selected warn issues 创建任务并返回问题区间、理由和 evidence
- [x] 3.2 实现人工 Pass/Fail 写回、机器观测保留和 effective verdict 计算
- [x] 3.3 实现所有候选完成检查、全 Pass 跳过人工质检和最终二元结论
- [x] 3.4 验证 supplier_evaluation 中自动 hard fail 不被人工 warn Pass 覆盖

## 4. 共用工作台

- [x] 4.1 建立共用资产导航、视频播放器、reviewer、revision 和保存错误外壳
- [x] 4.2 实现只显示视频/字幕/共享边界手柄/subtask 编辑器的 SemanticCalibrationAdapter，禁止整段平移并同时展示两段联动差异
- [x] 4.3 实现只显示问题片段/overlay、warn 原因和 Pass/Fail 的 WarnReviewAdapter
- [x] 4.4 在共享边界或文字 pending edit 存在时锁定其他边界、文字、模式切换、完成样本和下一资产操作
- [x] 4.5 实现问题窗口短片播放与 21 点骨骼 overlay 按需生成/缓存

## 5. Profile 路由与迁移

- [x] 5.1 接入 acceptance profile，使自动 hard fail 数据不创建语义或人工任务
- [x] 5.2 接入 supplier_evaluation profile，使自动 fail 数据继续语义和适用的 warn 人工质检
- [x] 5.3 提供遗留 manual CSV/progress JSON 的一次性导入，但禁止其成为最终事实源

## 6. 文档与端到端验证

- [x] 6.1 同步 PRD、JSON 格式文档和人工操作说明中的串行流程及逐次确认规则
- [x] 6.2 测试全 Pass 跳过人工质检、warn 人工 Pass、warn 人工 Fail 和多 warn 未完成四类流程
- [x] 6.3 测试浏览器刷新、并发 reviewer stale revision、overlay 生成失败和 HDF5 原子写入失败
- [x] 6.4 运行全量测试并验证每份最终 QC JSON 可独立生成资产质量报告

## 7. Warn 完成与原因合同重构

- [ ] 7.1 扩展 QC JSON Schema，加入 `completion_mode`、资产级 `failure_reason`、未查看 selected issue 兼容和审计约束
- [ ] 7.2 以测试驱动改造 Warn service，支持 `all_reviewed`、`early_fail`、完成前修改 verdict、人工原因覆盖和 Other 必填
- [ ] 7.3 更新批次投影，仅统计实际 issue review，并单独统计 early-fail 后未查看 Warn
- [ ] 7.4 为旧报告添加兼容读取与显式迁移测试，已完成历史报告保持只读

## 8. 模块切割与流水线门禁反转

- [ ] 8.1 将语义领域代码、API、静态页面和启动入口迁入独立 `semantic_calibration` 包
- [ ] 8.2 将 `human_qc` 收敛为 Warn-only，移除对语义 Service、adapter、DOM 和 HDF5 API 的运行时依赖
- [ ] 8.3 调整 pipeline 为自动 QC → Warn 人工复核 → 语义校准，并覆盖无候选、全 Pass、early Fail 和自动 hard-fail profile 场景
- [ ] 8.4 保留自动 lease/revision 并为两个独立服务验证路由与依赖边界

## 9. 整段视频与 Warning 时间轴

- [ ] 9.1 提供安全原视频 URL、FPS、总帧数、规范化半开区间、阈值提示和原因选项 DTO，并支持 HTTP Range
- [ ] 9.2 以 ES modules 拆分 `WarnReviewApp`、`VideoController`、`WarningTimeline` 和 `ReviewPanel`
- [ ] 9.3 实现真实帧宽度、重叠合并、窄色块、可进入弹层、色块跳起始帧和可拖动蓝色时间针
- [ ] 9.4 实现视频聚焦后的左右键逐帧、六档倍速持久化、唯一当前帧和视频/判定状态分离
- [ ] 9.5 实现同时命中 Warning 的确定 Pass 顺序、已通过标志、已判定项修改和底部完成/资产导航
- [ ] 9.6 实现人工原因先选后 Fail、多选取消、Other 输入框和直接资产切换丢弃未提交草稿

## 10. SAM3 区间连续 Overlay

- [ ] 10.1 扩展 evidence worker，只对 SAM3 问题帧并集生成连续 overlay，并去重重叠帧
- [ ] 10.2 实现基于源视频、区间、模型/config 和 renderer 版本的有界缓存、异步状态和失败恢复
- [ ] 10.3 在浏览器中将无控制条 overlay 层与原视频的播放、暂停、seek、逐帧和倍速同步
- [ ] 10.4 只预加载当前和下一个 overlay，并仅在非 ready 时锁定对应 SAM3 Warning

## 11. 验证与文档同步

- [ ] 11.1 增加 Python、Node、DOM、API 和浏览器测试，覆盖 120–181 重叠区间与 early-fail 完整流程
- [ ] 11.2 同步 PRD、JSON 格式、reviewer guide 和启动说明中的 Warn 前置流转与新操作规则
- [ ] 11.3 运行 Python/Node 全量测试、OpenSpec strict 验证、静态资源验收和 `git diff --check`
```

## openspec/changes/add-human-semantic-warn-review/specs/human-qc-workbench/spec.md

- Source: openspec/changes/add-human-semantic-warn-review/specs/human-qc-workbench/spec.md
- Lines: 1-38
- SHA256: 30323ccb073bd364f773d012d12ef65a587e574736cf1aa6d3a58b332b1ef252

```md
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
```

## openspec/changes/add-human-semantic-warn-review/specs/qc-json-batch-aggregation/spec.md

- Source: openspec/changes/add-human-semantic-warn-review/specs/qc-json-batch-aggregation/spec.md
- Lines: 1-17
- SHA256: c885a64e0ad3a6d49f2f638851a90be99ea8e1d9c4ffb42c8c39169f523c380c

```md
## MODIFIED Requirements

### Requirement: 自动失败与人工失败分别统计
批次统计 MUST 分别计算自动 hard-fail 资产/issue 数、机器 warn 数、人工检查 warn 数、人工消解 warn 数、人工确认 fail 数、未查看 selected warn 数、最终 fail 资产数、最终通过率、时间轴修改次数和 subtask 文字修改次数。人工统计 MUST 只来源于实际存在的 `issue_reviews`；`early_fail` 完成时未查看的 selected issue MUST 保持未查看，不得被计入人工检查、Pass 或 Fail。一个资产内多个 issue MUST NOT 被错误计算为多个资产，取消的语义修改不得计数。

#### Scenario: 人工处理两个 Warn 并修改一次语义
- **WHEN** 一个资产的两个机器 warn 分别被人工判定为 Pass 和 Fail，且确认一次时间轴修改
- **THEN** 人工检查 warn 数增加二
- **THEN** 人工消解 warn 数和人工确认 fail 数各增加一
- **THEN** 时间轴修改次数增加一
- **THEN** 最终 fail 资产数增加一

#### Scenario: 首个 Fail 后提前完成
- **WHEN** 一个资产有三个 selected Warn，仅第一个被人工判定为 Fail 后以 `early_fail` 完成
- **THEN** 人工检查 warn 数和人工确认 fail 数各增加一
- **THEN** 未查看 selected warn 数增加二
- **THEN** 人工消解 warn 数不增加
```

## openspec/changes/add-human-semantic-warn-review/specs/qc-pipeline-execution/spec.md

- Source: openspec/changes/add-human-semantic-warn-review/specs/qc-pipeline-execution/spec.md
- Lines: 1-43
- SHA256: c3d98d5a42565483052880ea650b26374fbdf8f7353d14b83b7806cfcaad5aa7

```md
## MODIFIED Requirements

### Requirement: 准入模式由自动 Hard Fail 截断
`acceptance` profile MUST 在任何自动模块产生 hard fail 时按既有准入规则停止后续正常流转，将资产最终结论设为 fail，并跳过 Warn 人工复核和语义校准。未被自动 hard fail 截断的资产 MUST 先完成或跳过 Warn 人工复核，再决定是否进入语义校准。

#### Scenario: 自动模块 Hard Fail
- **WHEN** `keypoint_presence` 在准入模式产生 fail
- **THEN** 后续自动模块按既有规则停止
- **THEN** Warn 人工复核和语义校准状态为因 fail 跳过
- **THEN** 最终结论为 fail

### Requirement: 人工质检前置于语义校准
未被自动门禁截断的资产 MUST 按 `自动 QC → Warn 人工复核 → 语义校准` 顺序流转。没有人工候选或所有 selected Warn 最终均为 Pass 时 MUST 进入语义校准；任一人工 Fail 以 `early_fail` 完成时 MUST 停止流水线并将语义校准标记为 `skipped_due_to_fail`。浏览器参数不得绕过该服务端门禁。

#### Scenario: 自动检查无 Warn
- **WHEN** 自动阶段完成且没有人工候选
- **THEN** `manual_review.state=not_required`
- **THEN** 资产进入语义校准

#### Scenario: 所有 Warn 人工 Pass
- **WHEN** manual review 以 `completion_mode=all_reviewed` 完成
- **THEN** pipeline `next_module=semantic_consistency`
- **THEN** 语义服务可以创建或恢复该资产任务

#### Scenario: 人工 Fail 后提前完成
- **WHEN** manual review 以 `completion_mode=early_fail` 完成
- **THEN** pipeline 状态为 stopped 且 `next_module=null`
- **THEN** `semantic_calibration.state=skipped_due_to_fail`

### Requirement: 供应商测评模式保留完整自动结果
`supplier_evaluation` profile MUST 保留所有自动 fail 及其证据，并按该 profile 的既有策略决定是否继续采集人工证据；无论是否继续，人工 Pass MUST NOT 覆盖自动 hard fail。只有未被最终失败门禁终止且完成或跳过 Warn 复核的资产才可进入语义校准。

#### Scenario: 自动 Fail 后继续人工取证
- **WHEN** profile 配置为自动 fail 后仍继续采集 Warn 人工证据
- **THEN** 报告保留自动 fail 和后续人工 review
- **THEN** 最终结论仍为 fail

### Requirement: 最终业务结论只有 Pass 和 Fail
资产在必需阶段未完成时 `final_decision` MUST 为 null。最终结论只能为 pass 或 fail：存在任一自动 hard fail 或人工确认 fail 时为 fail；否则在所有必需自动模块、适用的人工复核和语义校准完成后为 pass。Warn、review 和 accept_with_risk MUST NOT 作为最终业务状态。

#### Scenario: 机器 Warn 被人工消解并完成语义
- **WHEN** 资产没有自动 hard fail、所有 selected Warn 均被人工判定为 Pass，且语义校准完成
- **THEN** 最终业务结论为 pass
```

## openspec/changes/add-human-semantic-warn-review/specs/semantic-calibration/spec.md

- Source: openspec/changes/add-human-semantic-warn-review/specs/semantic-calibration/spec.md
- Lines: 1-72
- SHA256: d16415fe4ea1c7c63ae26ca8381a66d1a9e32524f0b4bc99a093441c823078f6

```md
## ADDED Requirements

### Requirement: 语义校准的进入条件由执行 profile 决定
准入模式下，只有未产生自动 hard fail 的资产 SHALL 进入语义校准；供应商测评模式下，只要自动阶段没有运行错误，包含自动 fail 的资产也 SHALL 进入语义校准。

#### Scenario: 准入模式自动 fail
- **WHEN** 资产在 acceptance profile 的自动阶段产生 hard fail
- **THEN** 语义校准状态标记为因自动 fail 跳过
- **THEN** 不创建该资产的语义编辑任务

#### Scenario: 供应商测评模式自动 fail
- **WHEN** 同一资产在 supplier_evaluation profile 产生 hard fail但自动阶段正常结束
- **THEN** 系统仍创建语义校准任务

### Requirement: 每次语义修改必须独立确认
时间轴共享边界拖动或一条 subtask 文字修改 MUST 立即进入 pending confirmation。存在 pending confirmation 时，工作台 MUST 禁止修改其他段落、切换任务或完成样本，直到操作者确认或取消本次修改。

#### Scenario: 拖动时间轴后尝试编辑另一条文字
- **WHEN** 操作者拖动一个 subtask 的边界但尚未确认
- **THEN** 另一条 subtask 的文字编辑和时间轴拖动被禁用
- **THEN** 界面只允许确认或取消当前修改

### Requirement: 时间轴使用共享边界联动相邻 Subtask
语义时间轴 MUST 将相邻 subtask 建模为共享边界序列；只有相邻段之间的内部边界可以拖动，整个 subtask 区间不得平移。拖动一个内部边界 MUST 在同一个 pending transaction 中同时更新前一段的结束边界和后一段的起始边界。首段最左边界和末段最右边界 MUST 固定。

#### Scenario: 拖动中间 Subtask 的右边界
- **WHEN** 操作者将第二段与第三段之间的内部半开边界从 411 拖到 429，对应界面将第二段闭区间结束帧从 410 改为 428
- **THEN** 第二段的界面结束帧从 410 更新为 428，第三段的界面起始帧从 411 更新为 429，并处于同一个 pending transaction
- **THEN** 工作台同时展示并高亮第二段和第三段的 before/after
- **THEN** 第一段和其他边界保持不变

#### Scenario: 确认共享边界修改
- **WHEN** 操作者确认一个影响两段的共享边界 pending transaction
- **THEN** 两段的新边界作为一个不可拆分的操作提交
- **THEN** `timeline_edit_count` 只增加一而不是按受影响段数增加

#### Scenario: 取消共享边界修改
- **WHEN** 操作者取消一个共享边界 pending transaction
- **THEN** 两个受影响 subtask 的边界同时恢复到修改前
- **THEN** `timeline_edit_count` 不增加

### Requirement: 共享边界不得破坏时间轴连续性
内部时间轴 MUST 使用严格递增的半开边界序列 `[b_i, b_{i+1})`，并保持原时间轴连续覆盖。工作台和服务端 MUST 拒绝会产生空档、重叠、逆序或零长度区间的边界值；源格式使用闭区间时，source adapter MUST 负责无歧义的帧边界转换。

#### Scenario: 边界越过相邻外侧边界
- **WHEN** 操作者尝试将内部边界拖过前一段起点或后一段终点
- **THEN** 工作台拒绝该边界值且不得创建可确认的 pending transaction
- **THEN** 服务端即使收到等价无效请求也拒绝写入

### Requirement: 确认后的修改形成轻量审计
每次确认 MUST 保存编辑类型、操作者、时间和 before/after。共享边界编辑还 MUST 保存稳定 `boundary_id`、两个受影响的 subtask 标识和两段各自的 before/after。QC JSON MUST 维护独立的 `timeline_edit_count` 与 `subtask_text_edit_count`；统计次数按已确认的用户编辑事务计算，而不是按受影响记录数计算，取消操作不得计数。

#### Scenario: 修改后取消
- **WHEN** 操作者拖动边界后点击取消
- **THEN** 时间轴恢复确认前值
- **THEN** `timeline_edit_count` 不增加

### Requirement: 样本完成时原子替换 HDF5
语义阶段完成时，系统 MUST 将所有已确认时间轴和 subtask 文本一次性写入临时 HDF5，完整校验后原子替换原 HDF5。系统 MUST NOT 在本地保留原 HDF5 备份；写入失败时原文件必须保持不变。

#### Scenario: 临时 HDF5 校验失败
- **WHEN** 已确认编辑写入临时文件后结构校验失败
- **THEN** 系统删除临时文件并保留原 HDF5
- **THEN** QC JSON 不得将语义阶段标记为完成

### Requirement: 语义校准不产生 Pass/Fail
当前人工语义校准 SHALL 只产生 completed、skipped 或 error 等流程状态以及修改审计，MUST NOT 产生语义质量 pass/fail 或改变资产最终质量结论。

#### Scenario: 操作者修改多个 subtask
- **WHEN** 操作者确认修改并完成语义阶段
- **THEN** 系统记录完成状态和修改次数
- **THEN** 不生成 semantic fail issue
```

## openspec/changes/add-human-semantic-warn-review/specs/warn-human-review/spec.md

- Source: openspec/changes/add-human-semantic-warn-review/specs/warn-human-review/spec.md
- Lines: 1-121
- SHA256: 73b7fe7739c49e1ab3a5777bfc00c8db7a77d9aca7e88c8270725ea36c7be85c

[TRUNCATED]

```md
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

```

Full source: openspec/changes/add-human-semantic-warn-review/specs/warn-human-review/spec.md

