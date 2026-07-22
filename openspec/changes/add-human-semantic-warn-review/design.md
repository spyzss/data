## Context

`unify-qc-dataflow` 提供单资产 QC JSON、执行 profile、稳定 issue/evidence 和人工路由状态。本 change 在该合同上实现用户确认的串行关系：自动 QC Gate → 对累计 warn 做人工 Pass/Fail → 独立语义校准 → 最终二元结论。人工 Fail 终止资产；无候选的 `not_required` 资产直接进入语义。

## Goals / Non-Goals

**Goals:**

- 为语义校准提供独立前端、后端、API 和启动入口，与 warn 人工质检运行时彻底切割。
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

### 1. 两个应用以 QC JSON 为唯一衔接面

`semantic_calibration` 独占语义 domain service、application facade、HTTP server、静态页面、HDF5 adapter 和正式启动入口。`human_qc` 独占 Warn task/evidence/Pass-Fail。两者不得互相导入领域服务，也不得共享 HTTP facade、DOM 或 task adapter；共同依赖只允许放在 `qc_common`。

语义 server 只暴露 `/api/semantic/...`。根地址自动加载第一条符合门禁的任务，支持 `?asset_id=...` 深链；直接请求未完成人工质检或已人工 Fail 的资产必须被服务端拒绝。

### 2. 每次编辑使用单一 pending transaction

前端任一时刻最多存在一个 pending edit，记录 before/after 和编辑类型。进入 pending 后锁定其他 segment 和阶段切换；确认后调用服务端 API 并增加相应计数，取消则还原。样本完成前服务端再次检查没有 pending edit。

时间轴在统一 contract 中不表示为互相独立的 subtask 区间，而表示为有序共享边界 `b_0...b_n`，其中第 `i` 段为半开区间 `[b_i, b_{i+1})`。只有内部边界 `b_1...b_{n-1}` 显示拖动手柄；首段最左边界和末段最右边界固定，整个 subtask 色块不得拖动。源 HDF5 若使用闭区间，source adapter 在读写时负责将 `[start_frame, end_frame]` 与内部半开区间互转；例如界面闭区间结束帧 410 对应内部下一边界 411。

拖动内部边界 `b_i` 是一个共享边界事务：它同时修改前一段的结束边界和后一段的起始边界。pending edit 必须记录 `boundary_id`、操作者主动拖动的手柄、两个受影响的 subtask ID，以及两段各自的 before/after；界面同时高亮并展示两段变化。确认或取消必须原子作用于两段，一次确认只增加一次 `timeline_edit_count`，不得因为两段记录都变化而重复计数。

共享边界必须始终严格单调并覆盖原时间轴，服务端和前端都要拒绝产生空档、重叠、逆序或零长度区间的拖动。边界不得越过相邻的外侧边界；本次不实现整段平移、级联挤压第三段、拆分、合并或删除 subtask。

### 3. HDF5 使用临时副本校验后原子替换

服务端从当前 HDF5 创建同目录临时文件，将所有已确认语义结果写入目标 dataset，重新打开并验证结构、JSON 和帧/时间轴边界，然后 fsync + `os.replace`。不生成持久 `.bak`。QC JSON 只在替换成功后记录 semantic completed 和最终 hash。

### 4. 人工质检是语义校准前置门禁

自动阶段先创建/跳过人工质检。`manual_review.state=completed` 且
`completion_mode=all_reviewed`，或 `state=not_required` 时，pipeline cursor 才推进到
`semantic_consistency`。`completion_mode=early_fail` 时流水线停止并将语义标记为
`skipped_due_to_fail`。资格由持久化报告与 cursor 联合计算，不能由前端布尔值决定。

### 5. 机器观测与有效结论并存

issue 保留机器 verdict、指标、阈值和 evidence；人工 review 追加 reviewer、时间、Pass/Fail 和可选原因。`effective_verdict` 对 warn 取人工结论。这样人工判断真实决定最终质量，同时保留误报率和规则改进所需证据。

### 6. Evidence 采用问题窗口和缓存 overlay

视频 warn 使用 issue context 的起止帧转成短片播放范围。21 点骨骼 overlay 优先使用自动模块生成的 evidence；缺失时由服务端按 issue window 生成并缓存，不为整段视频实时叠加，以控制 CPU、解码和网络开销。

### 7. 正式状态全部服务端持久化

浏览器每次确认都带 expected report revision。服务端原子更新工作状态/QC JSON并返回新 revision。localStorage 和 JSON/CSV 导出仅保留为迁移及故障恢复辅助，不参与最终批次统计。

## Risks / Trade-offs

- **无持久 HDF5 备份提高写错风险** → 临时副本全量校验、同目录原子替换、替换前后 hash 和 dataset 结构测试必须全部通过。
- **逐次确认增加操作次数** → 提供明确键盘操作和 pending 高亮，但不合并确认，优先满足防误操作要求。
- **长视频 overlay 成本高** → 只处理 issue window、按需缓存，限制并发生成任务。
- **跨应用状态竞态** → 人工完成/失败与 cursor 迁移在一次 revision-aware 报告写入中完成；语义服务每次读取都重算 eligibility。
- **并发 reviewer 冲突** → 两个应用各自使用 expected revision 和任务 lease；过期提交被拒绝并要求刷新。
- **语义结构因供应商不同而变化** → HDF5 读写由 source adapter 封装，工作台只消费统一 subtask contract。

## Migration Plan

1. 扩展 QC JSON 的 semantic/manual review 状态、review 和审计 Schema。
2. 实现语义工作状态、HDF5 临时写入/校验/替换服务。
3. 重构现有人工复核服务器为 revision-aware API，并提供遗留 CSV/progress 导入。
4. 将人工阶段放到语义之前：Pass/not_required 推进语义，Fail 终止资产。
5. 将语义代码迁入独立包，建立独立 application、HTTP server、静态页面和启动入口。
6. 接入 profile 路由并将人工与修改次数统计接入 QC JSON 聚合器。

回滚时停用新工作台写 API并保留只读 QC JSON；已经原子替换的 HDF5 不自动回滚，因为用户明确要求不保留本地原版，原始数据仍由云端保存。

## Open Questions

无阻塞问题。完整 JSON 字段名将在获取同事最终数据格式后通过 source adapter 映射，不改变工作台和 QC JSON 的稳定合同。
