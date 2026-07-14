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
