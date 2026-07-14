# Brainstorm Summary

- Change: unify-qc-dataflow
- Date: 2026-07-14
- Status: 已确认

## 已确认事实

- 单资产 `quality_archive/<asset_id>.json` 是质量事实源，sidecar 仅作证据。
- 使用统一 Config，并发布新的不可变 schema/config 版本；历史 v1.1.0 不原地修改。
- 单资产报告发布 `asset_qc_report.v2`，统一使用现有 PRD 字段名 `overall_decision`；v1 只读兼容。
- acceptance profile 遇到自动 hard fail 立即停止，失败数据不进入语义或 warn 人工质检。
- supplier_evaluation profile 记录 fail 但继续完整流程，最终结论仍反映 fail。
- 自动检查全部 Pass、语义完成且无 warn 候选时跳过人工质检。
- 最终业务状态只有 Pass/Fail；运行错误保持未完成而不是伪装成质量结论。
- 当前 change 只统一自动模块、编排、Config、QC JSON 和批次统计；人工工作台由依赖 change 实现。

## 确认的技术方案

### 方案 A：适配器 + 报告事务 + 资产编排器（已选定）

保留现有检测器和 sidecar，由模块 adapter 归一为 `ModuleResult`；共享报告事务负责模块所有权、稳定 issue、revision 和原子写回；资产 orchestrator 按配置和 profile 控制流转。

### 未采用方案 B：直接重写现有检测器为统一 ModuleRunner

接口最整齐，但会同时改变算法调用、输出和数据流，回归范围大，不适合本次合并收口。

### 未采用方案 C：离线收集 sidecar 后拼装 QC JSON

修改最少，但无法在运行时执行 Gate、恢复和并发写入控制，继续保留多事实源问题，不满足 PRD。

## 推荐架构

- `qc_common/contracts.py`：`ModuleResult`、`Issue`、`EvidenceRef`、flow 类型和稳定 issue ID。
- `qc_common/report_mutation.py`：模块所有权感知的报告合并、候选重建、revision 校验和原子写回。
- `qc_common/module_registry.py`：统一自动模块注册与 enabled/disabled/not_implemented 区分。
- `qc_pipeline/orchestrator.py`：按资产顺序运行模块、应用 execution profile、恢复 `next_module`。
- `qc_pipeline/adapters/`：Precheck、video、SAM3 遗留结果适配器。
- `qc_reporting/`：只读取 QC JSON 的扁平投影、聚合和可重建缓存。

Config 采用 `qc_acceptance_config_schema.v2` 与 `qc_acceptance_v2.0.0`，包含 execution profiles、模块顺序、模块实现绑定、规则和人工路由策略；每份资产报告锁定 config version/hash。

## 关键取舍与风险

- 适配器优先保证算法行为不变，代价是迁移期仍生成遗留 sidecar。
- 资产内串行写回避免 revision 冲突，批次内仍可并行不同资产。
- enabled 但未注册的模块产生 runtime unavailable，`overall_decision` 保持 null，禁止伪造 Pass。
- supplier_evaluation 成本更高，因此只在显式 profile 下启用。
- 旧报表与新投影在迁移期做 fixture 对账，正式入口最终只保留 QC JSON 聚合。

## 测试策略

- 合同单测：Schema、Config hash、稳定 issue ID、模块所有权、revision 冲突。
- adapter golden tests：同一遗留结果映射出固定 module block/issues/evidence。
- orchestrator 状态机测试：pass、warn、hard fail、runtime error 在两种 profile 下的流转。
- 集成测试：真实小型 HDF5/video/SAM3 fixture 形成完整 revision 轨迹。
- 聚合测试：删除派生缓存后仅凭 QC JSON 重建相同批次统计。
- 回归测试：现有检测器阈值和 sidecar 输出测试继续通过。

## Spec Patch

已回写：统一 Config 固定为 `qc_acceptance_config_schema.v2` / `qc_acceptance_v2.0.0`；单资产报告升级为 `asset_qc_report.v2` 并统一字段名 `overall_decision`；同一资产内模块串行写回，不同资产允许并行。其余 OpenSpec 验收场景已覆盖。
