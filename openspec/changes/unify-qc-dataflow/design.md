## Context

仓库中已经存在两类相互脱节的实现：`video_quality` 使用 `qc_common.report` 原子更新单资产 QC JSON；Precheck、manifest 视频、SAM3、人工复核和多套 ledger/report 则各自读写 JSON、Parquet、CSV 或 HTML 进度文件。统一配置已经声明模块顺序和 `aggregate_from_quality_archive_only`，但没有一个编排器实际执行这些合同，Schema 也只对视频模块建立了专用 block。

本 change 先统一自动模块和数据流，不重写同事已经合入的检测算法。人工语义校准和 warn 复核由依赖 change `add-human-semantic-warn-review` 实现。

## Goals / Non-Goals

**Goals:**

- 将现有模块结果归一为单资产 QC JSON 的 module block、稳定 issue、evidence 和 flow。
- 建立可恢复、可按资产运行的统一编排器。
- 支持 fail 截断的准入模式和 fail 不截断的供应商测评模式。
- 使所有批次统计只依赖单资产 QC JSON。
- 保持现有 sidecar 作为大体量逐帧/逐窗口证据和调试产物。

**Non-Goals:**

- 不重新实现或重新定义各检测算法阈值。
- 不在本 change 实现语义时间轴工作台、HDF5 修改或人工 Pass/Fail UI。
- 不将配置中尚无代码实现的 duplicate/content/effective-duration 模块伪造成已实现。
- 不支持人工推翻自动 hard fail。

## Decisions

### 1. 使用适配层统一结果，不重写检测器

为现有 Precheck aggregate/candidate、video result 和 SAM3 summary 建立模块 adapter。adapter 将遗留输出转换为统一 `ModuleResult`，包含模块名、机器 verdict、metrics、issues、evidence 和执行元数据。检测器仍可生成现有 sidecar，adapter 负责主报告写回。

选择该方案是为了保留已合入算法和测试。替代方案是要求每个检测器直接重写为统一接口，但会扩大回归范围并把数据流改造与算法重构混在一起。

### 2. 共享报告事务统一处理 revision、所有权和派生状态

在 `qc_common` 增加报告 mutation/transaction API：加载当前 revision，移除本模块旧 issue，合并新 module block，重建顶层 issue 索引和人工候选，计算 pipeline state，Schema 校验后原子替换。adapter 不直接手写完整 JSON。

该事务层使用模块所有权映射，确保重跑只替换本模块数据并保留其他模块和未来扩展字段。

人工语义工作台是后续 change 的外部写入者。这里的 v2 Schema 与事务层只提供可扩展、revision-aware 的原子写回边界，不把语义修改约束为单片段事件；因此后续工作台可以把一次共享边界拖动及其影响的相邻两段作为同一个 revision 提交。external-stage payload 原样保留严格递增半开边界 `[b_i, b_{i+1})`，界面闭区间结束帧只作为 `b_{i+1} - 1` 的显示换算，统一数据流不进行隐式端点改写。

### 3. 机器结果与实际流转分离

`result_gate.verdict` 永远反映机器规则结果；`exit_gate` 由 `execution_profile` 决定。acceptance 下 fail 产生 `stop_qc`；supplier_evaluation 下同一个 fail 产生 `continue`，同时在执行轨迹记录 `continued_after_fail=true`。这样测评模式不会通过篡改 verdict 实现全流程。

### 4. 编排器按资产和配置注册表执行

新增资产级 orchestrator，读取版本化 config 的模块顺序，通过 registry 找到实现并传入 manifest/source_files。每完成一个模块立即提交一次报告 revision，使中断后可以从 `pipeline_state.next_module` 恢复。

配置为 disabled 的模块写 skipped；配置为 enabled 但 registry 无实现的模块写运行错误并保持 final decision 为 null。当前只接入仓库已有实现；未来模块通过同一 registry 扩展。

### 5. Sidecar 是引用证据，不是第二份结论

逐帧结果、候选窗口、overlay 和模型明细继续保留。QC JSON 只保存相对路径、内容类型、坐标系、帧区间和可选 checksum。人工队列与批次报表从 QC JSON 的 issue/evidence 索引生成，不再反向解释 sidecar 得出不同结论。

### 6. 批次报表改为投影层

新增统一 QC JSON reader 和扁平化投影，现有 ledger/report 输出逐步迁移到该投影。所有比率明确区分资产、issue、区间和帧四种分母，并按 execution profile 分组。Parquet/CSV 只作为可重建缓存。

## Risks / Trade-offs

- **遗留输出字段不一致** → 为每个 adapter 建立固定映射和 golden fixture，无法映射的值进入结构化 runtime error，禁止静默 pass。
- **多个模块并发更新同一资产** → 默认资产内串行，报告写入继续使用 expected revision 防止丢失更新。
- **Schema 一次扩展过大** → 先增加通用 module block 定义和已接入模块的严格字段，再用模块级测试逐步收紧。
- **旧报表与新投影短期不一致** → 迁移期对相同 fixture 同时运行旧/新报表并记录差异，最终只保留 QC JSON 投影为正式入口。
- **供应商测评执行成本较高** → profile 明确是显式选择，不改变准入模式的 fail-fast 默认行为。

## Migration Plan

1. 扩展配置和 Schema，加入 execution profile、通用 module contract 和运行错误状态。
2. 实现报告事务、稳定 issue ID 和 adapter 基础接口。
3. 依次接入 Precheck 模块、manifest 视频兼容路径和 SAM3。
4. 增加 orchestrator，验证 acceptance 与 supplier_evaluation 的相同输入不同流转。
5. 将 manual queue 输入改为 QC JSON issue/evidence 投影，为后续人工 change 提供接口。
6. 将批次 ledger/report 切换为 QC JSON 聚合，并保留一次性遗留输入迁移/对账工具。

回滚时可保留 sidecar 生成和现有独立 runner，但不得让新旧路径同时写同一资产报告；通过 feature flag 选择旧 runner 或统一 orchestrator。

## Open Questions

无阻塞问题。尚未实现的自动模块只建立注册合同，不在本 change 内虚构算法结果。
