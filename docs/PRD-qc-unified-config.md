# PRD: QC 统一配置系统

## 1. 文档状态

| 项目 | 当前值 |
|---|---|
| 配置 schema | `qc_acceptance_config_schema.v2` |
| 当前配置版本 | `qc_acceptance_v2.1.0` |
| 当前模块版本 | `video_prefilter_v0.3.2` |
| 活动配置 | `configs/qc_acceptance.yaml` |
| 不可变归档 | `configs/qc_acceptance/qc_acceptance_v2.1.0.yaml` |
| 配置 schema 文件 | `schemas/qc_acceptance_config.v2.schema.json` |
| 已接入代码 | 自动 adapters、双 profile orchestrator、projection |
| 外部阶段 | `semantic_consistency`、`manual_review` |

本 PRD 定义全流程模块共用一份版本化配置的规范。v2 Config 是模块顺序、阈值、
rule、profile 和外部阶段边界的唯一来源；每条数据的报告只保存 config 引用和
实际读取字节的 hash。

## 2. 目标

1. 所有 QC 阈值、模块顺序和人工路由策略集中在一份 YAML。
2. 一个 asset 开始 QC 时锁定一份 config，流程中途不得变更。
3. `<asset_id>.json` 顶层只保存 config 引用和 hash，不复制 thresholds。
4. 每个问题通过稳定 `rule_id` 回查触发规则。
5. 修改阈值时升级 `config_version` 并归档完整 YAML。
6. loader 在模块运行前拒绝结构错误、模块缺失或重复 rule ID。

## 3. 非目标

- 本 PRD 不把所有 QC 逻辑合并进一个模块。
- 本次代码不实现其他同事负责的 QC 模块。
- 不允许模块继续维护私有 threshold YAML 作为生产入口。
- 不把 mask、overlay、窗口明细等大对象放入配置。
- 不在单资产 JSON 中保存 threshold snapshot。

## 4. 唯一生产入口

默认配置路径：

```text
configs/qc_acceptance.yaml
```

视频 QC 命令：

```bash
python run_acceptance_video_quality.py \
  --batch <batch_dir> \
  --config configs/qc_acceptance.yaml
```

`--config` 可以指向另一份完整统一配置，但不接受旧的 video-only YAML。v2 统一
配置必须包含 `schema_version`、`config_version`、`execution_profiles`、`pipeline`
和 `modules`。

## 5. 顶层结构

```yaml
schema_version: qc_acceptance_config_schema.v2
config_version: qc_acceptance_v2.1.0
config_name: acceptance_gate
execution_profiles:
  acceptance:
    fail_action: stop
    runtime_error_action: stop_incomplete
  supplier_evaluation:
    fail_action: record_and_continue
    runtime_error_action: stop_incomplete
pipeline: {}
modules: {}
```

| 字段 | 用途 |
|---|---|
| `schema_version` | YAML 结构版本；结构变化时升级。 |
| `config_version` | 阈值与流程策略版本。 |
| `config_name` | 当前固定为 `acceptance_gate`。 |
| `execution_profiles` | acceptance 截断或 supplier_evaluation 记录后继续；runtime error 都停止未完成。 |
| `pipeline` | 模块顺序、默认 profile 和 terminal module。 |
| `modules` | 各模块参数和规则表。 |

## 6. 版本策略

### 6.1 何时升级 `schema_version`

- 新增或删除必填顶层字段；
- 改变配置字段类型或嵌套方式；
- 改变单资产 JSON 所需 config 引用字段。

### 6.2 何时升级 `config_version`

- 任意 pass/warn/fail 阈值变化；
- 规则严重级别或 gate 行为变化；
- pipeline 模块顺序变化；
- 人工路由或批次统计策略变化；
- 模块默认开关变化。

只改注释或说明文字可以不升级。

### 6.3 发布步骤

1. 基于当前活动 YAML 修改。
2. 升级 `config_version`。
3. 通过配置 schema 和模块测试。
4. 将完整文件复制到：

```text
configs/qc_acceptance/<config_version>.yaml
```

5. 归档文件发布后不得原地修改；再次调整必须发布新版本。
6. 同一个 asset pipeline 从初始化到完成一直使用同一版本和 hash。

## 7. Pipeline 配置

```yaml
pipeline:
  default_profile: acceptance
  terminal_module: batch_statistics
  modules:
    - hdf5_text_info
    - quality_hand
    - keypoint_presence
    - keypoint_morphology
    - keypoint_temporal
    - video_quality
    - sam3_containment
    - semantic_consistency
    - manual_review
    - duplicate_check
    - content_validity
    - effective_duration
```

规则：

- `pipeline.modules` 是默认 gate 顺序。
- 模块必须存在于 `modules.<module_name>`。
- `pass` 和 `warn` 继续；`acceptance` 的 `fail` 直接转 `batch_statistics`。
- `supplier_evaluation` 的 `fail` 记录后继续，最终 `overall_decision` 仍为 `fail`。
- warn 只累计人工候选，是否人工由 `manual_review` 模块决定。
- `overall_decision` 不是每个模块都重算的 summary；流程未结束时保持 `null`。

## 8. 模块配置规范

通用结构：

```yaml
modules:
  module_name:
    enabled: true
    module_version: optional_module_version
    parameters: {}
    rules:
      stable_rule_name:
        rule_id: module_name.stable_rule_name
        verdict: warn
```

要求：

- 模块只读取自己的 section 和必要的 pipeline 元数据。
- 参数默认值必须来自统一配置；代码默认只能用于 dataclass 构造和类型校验，不能
  悄悄覆盖生产 YAML。
- `rule_id` 在整个配置内必须唯一。
- `rule_id` 不能包含 config 版本，版本变化不应破坏历史聚合。
- 每个 warn/fail 分支必须能映射到一个 rule ID。

### 8.1 视频模块

当前已实现：

```yaml
modules:
  video_quality:
    enabled: true
    module_version: video_prefilter_v0.3.2
    hard_fail_blocks_next_qc: true
    parameters:
      pipeline: {}
      fps: {}
      resolution: {}
      timeline: {}
      decode: {}
      exposure: {}
      sharpness_global: {}
      freeze: {}
      defects: {}
      hdf5_alignment: {}
    rules: {}
```

视频模块不接受以下旧字段：

```text
threshold_version
threshold_profile
hand_roi
```

清晰度只保留全帧代理；手部 ROI 已从参数、指标、rule 和 JSON 全部移除。

### 8.2 其他模块

统一配置已经给出 HDF5 文本、`quality_hand`、关键点、SAM3、语义、人工和批次
统计的配置合同。`semantic_consistency` 与 `manual_review` 使用
`execution_kind: external`，到达时由外部人工工作台/未来模型 adapter 接续，
不在 Config 中伪造本地 implementation。`duplicate_check`、`content_validity`、
`effective_duration` 未注册时必须显式 disabled/not_implemented，不得写成 pass。

## 9. JSON 配置引用

建档时写入：

```json
{
  "qc_config": {
    "schema_version": "qc_acceptance_config_schema.v2",
    "config_version": "qc_acceptance_v2.1.0",
    "config_name": "acceptance_gate",
    "config_path": "configs/qc_acceptance.yaml",
    "config_hash": "sha256:<actual loaded file bytes>"
  }
}
```

规则：

- hash 基于实际读取字节，不基于重新序列化后的对象。
- 模块更新既有 JSON 时必须核对 version 和 hash。
- 不允许中途用另一份配置继续写同一个 asset。
- `config_path` 用仓库相对路径；外部配置可记录调用时路径。

模块和 issue 禁止重复写：

```text
thresholds
threshold_version
config_ref
issue.config_version
```

## 10. Issue 与 Rule 的关系

一个触发条件对应一个 issue：

```json
{
  "issue_id": "video_quality:fps_below_pass:001",
  "code": "fps_below_pass",
  "severity": "warn",
  "module": "video_quality",
  "issue_type": "low_fps",
  "metric": "video_basic.fps",
  "observed_value": 22.5,
  "operator": "<",
  "boundary_value": 24.0,
  "rule_id": "video_quality.fps_below_pass",
  "needs_manual_review": true,
  "context": {}
}
```

其中：

- `observed_value` 是本次实际值。
- `operator + boundary_value` 是当时触发该问题的可读边界。
- `rule_id` 用于定位统一配置中的规则语义。
- config 版本只从顶层 `qc_config` 读取。
- 多个指标触发时生成多个 issue，而不是给一个 issue 填多个 value。

## 11. Loader 合同

公共 loader 位于 `qc_common/config.py`，必须：

1. 读取调用方指定路径或默认活动配置。
2. 保存实际读取字节并计算 SHA-256。
3. YAML 解析后通过 `schemas/qc_acceptance_config.v2.schema.json`。
4. 校验 pipeline 中每个模块均有配置。
5. 校验所有 rule ID 唯一。
6. 暴露 `module_parameters(module_name)`。
7. 暴露顶层 JSON config reference。
8. 对旧 video-only YAML 直接报错，不做静默迁移。

模块不得各自重新实现 YAML merge、hash 或版本判断。

## 12. Schema 要求

配置 schema 至少校验：

- 顶层必填字段、双 profile 和版本格式；
- `pipeline.modules` 非空且元素唯一；
- `json_report.issue_fields_required` 使用统一 issue 字段；
- 视频参数结构和必要字段；
- 视频配置中不存在手部 ROI；
- 未知视频参数不能静默通过。

活动配置与归档配置都必须通过同一 schema。

## 13. 同事模块接入清单

每个模块负责人需要：

1. 删除生产路径中的私有 threshold YAML/default 覆盖。
2. 通过公共 loader 读取 `modules.<name>`。
3. 为每个 warn/fail 分支登记唯一 `rule_id`。
4. 使用顶层 `qc_config`，不写模块级版本或 thresholds。
5. 生成顶层 issue，并在模块 block 中只引用 `issue_id`。
6. 保留其他模块和未知字段。
7. 遵守 revision、schema 校验和原子写入。
8. 添加 config 缺字段、重复 rule ID、错误类型和 gate 行为测试。

## 14. 验收标准

- `load_qc_acceptance_config()` 能加载活动配置和归档配置。
- 修改实际文件字节会改变 `config_hash`。
- 缺模块、重复 rule ID、旧 video-only YAML、未知视频字段会失败。
- 视频 QC 的全部阈值来自 `modules.video_quality.parameters`。
- 视频 JSON 顶层 config reference 与实际加载配置一致。
- 视频 JSON 不包含 threshold snapshot、per-issue config version 或手部 ROI。
- 同一 asset 若已有另一 config 版本/hash，视频 writer 拒绝覆盖。
- 完整测试通过后才允许发布新 config 版本。

## 15. v2 发布、流转与回滚合同

当前生产值必须保持一致：

```text
config schema: qc_acceptance_config_schema.v2
config version: qc_acceptance_v2.1.0
active: configs/qc_acceptance.yaml
immutable: configs/qc_acceptance/qc_acceptance_v2.1.0.yaml
report: asset_qc_report.v2
report root: <batch>/quality_archive/*.json
```

发布新配置时以活动 YAML 为输入，升级 `config_version`，运行 v2 schema、rule 唯一性
和全量测试，再将同一字节复制到 `configs/qc_acceptance/<version>.yaml`。快照发布后
不可原地修改；同一 asset 的 report 只能使用一个 config hash，发现漂移时拒绝写回。
历史 v1 快照和 hash 保持不变，供只读迁移/对账使用。

### 15.1 双 profile 与 external 模块

```yaml
execution_profiles:
  acceptance:
    fail_action: stop
    runtime_error_action: stop_incomplete
  supplier_evaluation:
    fail_action: record_and_continue
    runtime_error_action: stop_incomplete
```

`acceptance` hard fail 立即 `stopped`/`fail`，不进入
`semantic_consistency` 或 `manual_review`；`supplier_evaluation` 保留机器 fail、
记录 `continued_after_fail` 并继续。两种 profile 的 runtime error 均为
`error`/`overall_decision=null`。`semantic_consistency` 和 `manual_review` 是
`execution_kind: external`，前者完成后才根据 `candidate_issue_ids` 决定后者
`not_required` 或 `queued`。

### 15.2 唯一事实源、证据路径和 CAS

- `quality_archive/*.json` 是唯一 master verdict；sidecar 只作证据和 reconciliation，
  不参加正式 aggregate，也不覆盖 JSON 的 machine/human verdict。
- `EvidenceRef.path` 必须是相对 batch root 的 POSIX 路径；绝对路径和 `..` 越界路径
  直接拒绝。
- 写回流程是读取当前 revision → 校验 asset/config/profile/next_module → 只替换模块
  所有权 → 重建 candidates/failures → revision + 1 → v2 schema → `fsync` + `os.replace`。
  expected revision 不一致是显式 CAS/stale-write 错误，不能静默覆盖。

### 15.3 CLI 与 cache

```bash
python tools/build_qc_json_projection.py \
  --quality-archive sampled/XJGT_20260616/quality_archive \
  --output-dir sampled/XJGT_20260616/qc_projection \
  --cache-dir sampled/XJGT_20260616/qc_cache \
  --formats csv parquet xlsx markdown

python tools/build_manual_review_queue.py \
  --quality-archive sampled/XJGT_20260616/quality_archive \
  --output-dir sampled/XJGT_20260616/manual_review

python tools/build_batch_qc_ledger.py \
  --quality-archive sampled/XJGT_20260616/quality_archive \
  --output-dir sampled/XJGT_20260616/ledger \
  --formats csv parquet xlsx markdown

python tools/build_xjgt_acceptance_report.py \
  --quality-archive sampled/XJGT_20260616/quality_archive \
  --output-dir sampled/XJGT_20260616/xjgt_report
```

cache 的 source manifest 记录 report relative path、revision、SHA-256；任何不一致都
必须删除/重建 cache。旧 sidecar CLI 参数仅生成对账表，不得改变 `quality_archive/*.json`
投影或批次统计。
