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

## 4. 独立工作台边界

- [x] 4.1 建立独立 `semantic_calibration` package、application facade、HTTP server、静态页面和启动入口
- [x] 4.2 语义 server 只暴露 `/api/semantic/...`，根地址自动加载首条待办并支持 asset 深链
- [x] 4.3 移除 human_qc 对语义 service/adapter/路由的托管，两个领域包只共享 QC JSON 合同
- [x] 4.4 实现只显示视频/字幕/共享边界手柄/subtask 编辑器的语义页面，禁止整段平移并同时展示两段联动差异
- [x] 4.5 在共享边界或文字 pending edit 存在时锁定其他边界、文字、资产切换和完成操作

## 5. Profile 路由与迁移

- [x] 5.1 接入 acceptance profile，使自动 hard fail 数据不创建人工或语义任务
- [x] 5.2 接入 supplier_evaluation profile，使机器自动 fail 数据继续适用的 warn 人工质检，并在人工 Pass/not_required 后继续语义
- [x] 5.3 提供遗留 manual CSV/progress JSON 的一次性导入，但禁止其成为最终事实源
- [x] 5.4 人工全 Pass/not_required 原子推进语义，人工 early-fail 终止并阻止语义读取/写入

## 6. 文档与端到端验证

- [x] 6.1 同步 README、独立语义操作指南、PRD、JSON 格式、迁移、人工操作和 OpenSpec
- [x] 6.2 测试全 Pass 跳过人工质检、warn 人工 Pass、warn 人工 Fail 和多 warn 未完成四类流程
- [x] 6.3 测试浏览器刷新、并发 reviewer stale revision、overlay 生成失败和 HDF5 原子写入失败
- [x] 6.4 运行全量测试并验证每份最终 QC JSON 可独立生成资产质量报告
