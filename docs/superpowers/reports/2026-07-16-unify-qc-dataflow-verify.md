# OpenSpec Verification Report：unify-qc-dataflow

验证日期：2026-07-16
验证对象：`openspec/changes/unify-qc-dataflow`
工作流：`spec-driven` / `repo-local`
验证基线：包含规格对齐提交 `6fe8784` 的当前分支
结论：**通过，可进入 archive 阶段**

## 1. Summary Scorecard

| 维度 | 状态 | 结果 |
|---|---|---|
| Completeness | PASS | 21/21 tasks 完成；17/17 requirements 有实现证据；24/24 scenarios 有测试或明确的合同测试证据。 |
| Correctness | PASS | 自动模块、双 profile、CAS 报告、JSON-only 聚合、缓存和迁移行为均与 delta specs 一致。 |
| Coherence | PASS | proposal、design、3 份 delta specs、关联 Design Doc 与当前 v2.1.0 active contract 一致；此前 version 与 disabled/skipped drift 已消除。 |
| OpenSpec | PASS | `openspec validate unify-qc-dataflow --strict`：`Change 'unify-qc-dataflow' is valid`。 |
| Test evidence | PASS | 全量 `1071 passed, 1 skipped`；专项 `63 passed`；独立 code review 与 spec review 均 CLEAN。 |
| Workspace hygiene | PASS | `git diff --check` 无输出；本验证未修改代码、spec、design、tasks 或 `.comet`。 |

## 2. 验证范围与方法

完整读取并核对：

- `openspec/changes/unify-qc-dataflow/proposal.md`
- `openspec/changes/unify-qc-dataflow/design.md`
- `openspec/changes/unify-qc-dataflow/tasks.md`
- `openspec/changes/unify-qc-dataflow/specs/asset-qc-module-contract/spec.md`
- `openspec/changes/unify-qc-dataflow/specs/qc-json-batch-aggregation/spec.md`
- `openspec/changes/unify-qc-dataflow/specs/qc-pipeline-execution/spec.md`
- `docs/superpowers/specs/2026-07-14-unify-qc-dataflow-design.md`
- `docs/superpowers/plans/2026-07-14-unify-qc-dataflow.md`

验证同时覆盖当前实现、JSON Schema、配置快照、模块 adapters、资产级 orchestrator、
报告 mutation、批次投影/聚合/缓存、正式报表入口、迁移工具和相应测试。

## 3. Completeness

### 3.1 Task completion

`openspec instructions apply --change unify-qc-dataflow --json` 返回：

- total：21
- complete：21
- remaining：0
- state：`all_done`

`tasks.md` 中实际 checkbox 统计同样为 21 个 `[x]`、0 个 `[ ]`，不存在仅在状态
元数据中完成、但 artifact 未勾选的情况。

### 3.2 Requirement coverage

#### asset-qc-module-contract：6/6

| Requirement | 实现证据 | 场景证据 | 结论 |
|---|---|---|---|
| 单资产 QC JSON 是唯一质量事实源 | `qc_common/report.py:62`；`qc_reporting/projection.py:39` | `tests/test_qc_reporting_entrypoints.py:25`；`tests/test_qc_migration_reconciliation.py:30` | PASS |
| 模块写回边界明确 | `qc_common/report_mutation.py:348`、`:439` | `tests/test_report_mutation.py:412`；共享边界扩展原样保留测试 | PASS |
| Issue 标识与字段稳定 | `qc_common/contracts.py:148` | `tests/test_qc_contracts.py:55`、`:78`；adapter 稳定 ID 回归 | PASS |
| 模块使用统一 Flow 合同 | `qc_common/report_mutation.py:487`、`:498` | `tests/test_report_mutation.py:455`；`tests/test_qc_orchestrator.py:351` | PASS |
| QC JSON 原子且并发安全地更新 | `qc_common/report.py:62-120` | `tests/test_report_mutation.py:113`；`tests/test_asset_qc_schema_v2.py` CAS 并发回归 | PASS |
| 未实现模块不得伪装为通过 | `qc_pipeline/orchestrator.py:155-197` | disabled：`tests/test_qc_orchestrator.py:489`；skipped：`:321`；unavailable：`:742` | PASS |

该 capability 的 11 个 scenarios 全部有覆盖。新增的两个状态场景也已闭环：

- 配置禁用模块写 `state=disabled` 和原因，由 `record_disabled_transition()` 与
  orchestrator disabled 测试覆盖；
- enabled 且明确不适用时才写 `state=skipped`，由 `quality_hand` 缺少可选供应商
  signal、temporal 空输入、morphology 不适用等 adapter 测试以及 module-state 测试覆盖。

#### qc-json-batch-aggregation：4/4

| Requirement | 实现证据 | 场景证据 | 结论 |
|---|---|---|---|
| 批次统计只读取单资产 QC JSON | `qc_reporting/projection.py:39-94` | `tests/test_qc_reporting_entrypoints.py:25` 验证 sidecar 冲突时 QC JSON 胜出 | PASS |
| 自动失败与人工失败分别统计 | `qc_reporting/aggregate.py:130-205` | `tests/test_qc_reporting_aggregate.py:11`、`:32` | PASS |
| 不同执行 profile 可分别聚合 | `qc_reporting/aggregate.py:23-52` | `tests/test_qc_reporting_aggregate.py:51` | PASS |
| 派生缓存可完全重建 | `qc_reporting/cache.py:133`、`:467` | `tests/test_qc_reporting_cache.py:26`、`:46`、`:144` | PASS |

4 个 scenarios 全部覆盖：sidecar 不覆盖主结论、多 issue 不重复资产分母、两种 profile
分组、缓存删除/损坏/过期后仅凭 QC JSON 重建。

#### qc-pipeline-execution：7/7

| Requirement | 实现证据 | 场景证据 | 结论 |
|---|---|---|---|
| 流程顺序来自不可变版本化配置 | `qc_common/config.py:69-108` | `tests/test_qc_config_v2.py:22`、`:65`、`:150` | PASS |
| 并发边界按资产隔离 | `tools/run_qc_pipeline.py:315-385` | `tests/test_qc_orchestrator.py:1345`；`tests/test_qc_pipeline_profiles_e2e.py:310` | PASS |
| 准入模式由 hard fail 截断 | `qc_common/report_mutation.py:487-540` | `tests/test_qc_orchestrator.py:351`；revision trace hard-fail fixture | PASS |
| 供应商测评模式记录 fail 但不截断 | `qc_common/report_mutation.py:487-504` | `tests/test_qc_pipeline_profiles_e2e.py:222`、`:288` | PASS |
| Warn 累积但不立即截断 | `qc_common/report_mutation.py:365-381` | `tests/test_qc_pipeline_revision_trace.py:269`；多模块 warn candidate 回归 | PASS |
| 最终业务结论只有 Pass 和 Fail | `schemas/asset_qc_report.v2.schema.json` 顶层状态条件；report reducer | `tests/test_asset_qc_schema_v2.py` 的 null/pass/fail mutation；最终 fixture | PASS |
| 运行错误不是质量结论 | `qc_pipeline/orchestrator.py:135-153`、`:193-207` | `tests/test_qc_orchestrator.py:664`、`:742`、`:855` | PASS |

9 个 scenarios 全部覆盖，包括 active v2.1.0 与对应 immutable snapshot 字节一致、
v1.1.0/v2.0.0 历史快照 hash 不变、跨资产并行隔离、acceptance 截断、supplier fail
后继续、warn 累积、最终二元结论和 runtime error 保持 `overall_decision=null`。

人工语义与 Warn 工作台本身仍属于依赖 change
`add-human-semantic-warn-review`，不属于本 change 的实现范围；本 change 已提供并测试
`awaiting_external`、候选集合、fail 跳过状态和 revision-aware 外部写入边界，没有用
fixture 冒充 UI 已实现。

## 4. Correctness

### 4.1 单资产报告与原子写回

- `ModuleResult`、`Issue`、`EvidenceRef` 为统一内部合同；稳定 issue ID 仅由规范化
  identity 字段计算，不受自然语言和运行时间影响。
- mutation 在写入前校验 expected revision、asset/config/profile、模块顺序、issue 和
  evidence；仅替换本模块拥有的 block/issue，并从顶层 issues 全量重建人工候选与
  fail 统计引用。
- `write_asset_qc_report()` 在锁内进行 revision 检查、Schema 校验、临时文件写入、
  文件 fsync、`os.replace` 和目录 fsync；stale writer 不覆盖新 revision。
- v1 读取为只读兼容；只有正式写回才迁移并校验 v2。

### 4.2 Profile 与状态机

- `result_gate.verdict` 保留机器结果；`exit_gate` 才应用 execution profile。
- acceptance hard fail 形成 `stopped/fail`，并将剩余自动/外部阶段标记为
  `skipped_due_to_fail`。
- supplier_evaluation 保留同一 fail issue 并记录 `continued_after_fail`，继续后续可用
  模块；最终 reducer 仍保留 fail。
- disabled、skipped、not_implemented/module_unavailable、runtime_error、
  awaiting_external 语义互不混用。

### 4.3 JSON-only aggregation

- 正式投影只遍历 `quality_archive/*.json`，逐份执行 Schema 校验；不会重新解释
  candidate、SAM3、video 或 manual sidecar 得出主结论。
- asset、issue、execution 使用独立稳定分母；同一资产多个 issue 不重复计数资产。
- overall 与 `by_profile` 从同一投影计算；两种 profile 的 coverage 不混合。
- 缓存由 source report revision/hash manifest 驱动；任何缺失、损坏或过期均返回
  cache miss，再从 QC JSON 重建。

## 5. Coherence 与 drift 复核

提交 `6fe8784` 修改的四份 artifact 内部一致：

1. `qc-pipeline-execution` 不再把 v2.0.0 写成永久默认；合同改为 active entry 当前
   `config_version` 必须对应 immutable snapshot。
2. 当前 active 为 `qc_acceptance_v2.1.0`，与
   `configs/qc_acceptance/qc_acceptance_v2.1.0.yaml` 字节一致。
3. v1.1.0 与 v2.0.0 继续作为不可变历史快照保留，未被方案 B 修改。
4. OpenSpec design、asset module delta spec 和关联 Design Doc 均明确：配置禁用写
   `disabled`；只有 enabled、实现可用且明确不适用才写 `skipped`；实现缺失和运行错误
   不得借用 skipped。

未发现新的 proposal/spec/design/code drift。目录结构、命名和依赖边界符合关联 Design
Doc；后续人工模块仍通过 external-stage 合同解耦。

## 6. Test 与门禁证据

| 门禁 | 结果 |
|---|---|
| 全量测试 | `1071 passed, 1 skipped in 322.10s` |
| schema/config、semantic equivalence、Publisher fault/atomicity 专项 | `63 passed in 48.35s` |
| 独立 code review | CLEAN，无 Blocker/Important |
| 独立 spec review | CLEAN，无 Blocker/Important |
| `openspec validate unify-qc-dataflow --strict` | PASS |
| `git diff --check` | PASS，无输出 |
| 历史 Config hash | v1.1.0=`0ef584...04ea41d`；v2.0.0=`747dc6...94c37b`，保持不变 |
| 当前 active snapshot | v2.1.0 active 与 immutable snapshot 字节一致 |

全量与专项证据记录于 `docs/canonical-qc-verification-report.md:113-125`；本次 OpenSpec
复核另外重新执行了 strict validation、artifact/实现映射和 whitespace 检查。

## 7. Issues by Priority

### CRITICAL

无。

### WARNING / IMPORTANT

无。

### SUGGESTION

无阻塞建议。未来 active Config 再升级时，应继续通过新的 OpenSpec delta 修改“当前
active snapshot”场景，同时保持本次版本无关 requirement 和所有历史快照不可变。

## 8. Final Assessment

**All checks passed. Ready for archive.**

`unify-qc-dataflow` 的 21 项任务、17 项 requirement 和 24 个 scenario 已完整覆盖；
实现、测试、OpenSpec artifacts 与关联 Design Doc 一致。没有 CRITICAL、IMPORTANT、
WARNING 或未解释的 spec drift。
