# PRD: QC 统一配置系统

## 1. Summary

本 PRD 定义机器人数据验收 QC pipeline 的统一配置系统。目标是把当前分散在视频 QC、precheck、SAM/mask、人工质检、批次统计脚本里的阈值和流程开关收敛到一份版本化中心配置：

```text
configs/qc_acceptance.yaml
```

新规范要求：`<asset_id>.json` 顶层只记录本次使用的 `qc_config` 版本、路径和 hash；各模块 block 不再复制 thresholds。历史结果通过 `qc_config.config_version + rule_id` 回查对应配置版本来解释。

## 2. Contacts

| 角色 | 责任 |
|---|---|
| 验收流程负责人 | 确认 gate 顺序、config 版本、pass/warn/fail 语义。 |
| QC 模块开发同事 | 实现统一配置 loader，并改造各模块从中心配置读取参数。 |
| 视频 QC 开发同事 | 停止向 JSON 写入 thresholds，改为写 rule_id、metric value、qc_config 顶层版本。 |
| 人工质检同事 | 基于 JSON 中的 `manual_review.candidates` 和 `qc_config.config_version` 判断是否人工。 |
| 批次统计同事 | 从 `quality_archive/*.json` 聚合结果，并按配置版本回查阈值。 |

## 3. Background

当前 QC 代码中已有多类配置来源：

- 视频预筛有独立阈值版本，例如 `video_prefilter_v0.3.1`。
- precheck 模块有 dataclass 默认值和 `configs/precheck_example.yaml`。
- `tmp/integrate-colleague-acceptance-20260707` 新增了 `keypoint_morphology` 和对应阈值。
- SAM/mask、语义一致性、人工抽样、批次统计也各自有参数。

旧方案曾建议每个模块把 `thresholds` 写入 JSON。这个方案可复现，但会让 JSON 变重，也会造成多处阈值副本。现在改成：

```text
阈值只存在于版本化 config。
JSON 顶层记录使用的 config 版本和 hash。
模块 issue 记录 rule_id 和实际 value。
需要解释阈值时，用 config_version + rule_id 回查 config。
```

## 4. Objective

### 4.1 目标

1. 提供一份中心配置 `configs/qc_acceptance.yaml`。
2. 中心配置管理 pipeline 顺序、stop-on-fail、模块阈值、人工策略、批次输出策略。
3. 每个 QC 模块运行时从中心配置读取自己的参数。
4. `<asset_id>.json` 顶层写入唯一配置引用：

```json
{
  "qc_config": {
    "schema_version": "qc_acceptance_config_schema.v1",
    "config_version": "qc_acceptance_v1.0.0",
    "config_name": "acceptance_gate",
    "config_path": "configs/qc_acceptance.yaml",
    "config_hash": "sha256:<computed_at_runtime>"
  }
}
```

5. 模块 block 不再写 `thresholds` 或模块级 `config_ref`。
6. 所有 warn/fail reason detail 必须写 `rule_id`、`value`、`comparison`、`config_version`。
7. 批次统计使用 `config_version + rule_id` 回查阈值，不从 JSON 读取阈值副本。

### 4.2 不做的事

- 不把所有模块逻辑合并成一个大模块。
- 不要求每个模块 import 其它模块代码。
- 不把大体积 sidecar、overlay、mask 明细写进中心配置。
- 不在 JSON 的每个模块 block 里复制 thresholds。
- 不在本阶段重新定所有阈值的最优值；先统一管理，再逐步校准。

### 4.3 成功标准

- 任意模块都能通过 `module_name` 获取自己的 resolved config。
- 任意一条 `<asset_id>.json` 都能看出当时使用的 `config_version`。
- 修改某个阈值只需要改中心配置并升级 `config_version`。
- 人工质检和批次统计可以根据 `rule_id` 回查对应阈值。
- 旧 JSON 中未知字段不会被模块写入流程删除。

## 5. Users And Constraints

### 5.1 用户

- QC pipeline：按中心配置决定模块顺序和是否继续运行。
- 模块开发同事：只关心自己模块的配置 schema 和 rule_id。
- 验收负责人：统一 review 所有阈值。
- 人工质检人员：看到需要人工的问题、数值和证据。
- 批次统计人员：用 `config_version + rule_id` 解释每条数据的判定。

### 5.2 约束

- 配置文件使用 YAML，路径固定为 `configs/qc_acceptance.yaml`。
- 配置版本字段使用 `config_version`，不是 `threshold_version`。
- 配置结构版本字段使用 `schema_version`。
- 模块更新 JSON 时必须保留未知字段。
- 配置 hash 必须基于实际读取的配置内容计算。
- JSON 里不写 thresholds；如发现模块继续写入，应视为旧字段，后续读逻辑忽略。

## 6. Value Proposition

统一配置系统解决三个核心问题：

1. **方便调参**：所有模块阈值集中在一处，减少漏改和冲突。
2. **JSON 更干净**：每条数据只记录配置版本，不复制大段阈值。
3. **方便追溯**：`config_version + rule_id` 可以精确回查当时判定规则。

## 7. Solution

## 7.1 配置文件路径

默认路径：

```text
configs/qc_acceptance.yaml
```

命令行允许覆盖：

```bash
--qc-config configs/qc_acceptance.yaml
```

## 7.2 顶层 schema

```yaml
schema_version: qc_acceptance_config_schema.v1
config_version: qc_acceptance_v1.0.0
config_name: acceptance_gate
config_date: "2026-07-09"
description: Versioned QC gate config for one-asset-one-json acceptance reports.
source: {}
version_policy: {}
pipeline: {}
json_report: {}
module_flow_contract: {}
rule_id_prefix: {}
modules: {}
precheck_compatibility: {}
legacy_checks: {}
batch_statistics: {}
```

| 字段 | 含义 |
|---|---|
| `schema_version` | 配置结构版本。字段结构变更时升级。 |
| `config_version` | 阈值和策略版本。阈值、模块顺序、判定语义变化时升级。 |
| `config_name` | 配置名称，当前为 `acceptance_gate`。 |
| `source` | 说明该配置对齐的分支和新增模块。 |
| `pipeline` | 模块顺序和流程开关。 |
| `json_report` | JSON 写入规则。 |
| `modules` | 每个模块的阈值、规则和 rule_id。 |

## 7.3 版本规则

```yaml
version_policy:
  schema_version:
    bump_when:
      - config file structure changes
      - required JSON config reference fields change
  config_version:
    bump_when:
      - any threshold changes
      - pass/warn/fail semantics change
      - module order changes
      - manual review routing policy changes
```

规则：

- 改 YAML 字段结构，升级 `schema_version`。
- 改任何阈值、模块顺序、人工路由、fail/warn 语义，升级 `config_version`。
- 只改注释或描述，不需要升级版本。

## 7.4 Pipeline 配置

```yaml
pipeline:
  stop_on_fail: true
  default_start_module: hdf5_text_info
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

- `modules` 是默认执行顺序。
- 每个模块仍要读上一模块 `flow.exit_gate.continue_to_next_module`。
- fail 后 `pipeline_state.next_module = "batch_statistics"`。
- warn 不阻断，只写入 `manual_review.candidates`。

## 7.5 JSON 写入配置

```yaml
json_report:
  root_dir_name: quality_archive
  config_field_name: qc_config
  write_config_reference_at_top_level: true
  write_config_hash: true
  write_module_threshold_snapshot: false
  preserve_unknown_fields: true
```

JSON 顶层必须写：

```json
{
  "qc_config": {
    "schema_version": "qc_acceptance_config_schema.v1",
    "config_version": "qc_acceptance_v1.0.0",
    "config_name": "acceptance_gate",
    "config_path": "configs/qc_acceptance.yaml",
    "config_hash": "sha256:<computed_at_runtime>"
  }
}
```

模块 block 禁止新写：

```json
{
  "thresholds": {},
  "config_ref": {}
}
```

如果历史 JSON 已经有这些字段，读取端可以兼容，但新版本模块不应继续写入。

## 7.6 Issue 明细标准

所有 warn/fail reason detail 必须包含：

```yaml
issue_fields_required:
  - code
  - severity
  - module
  - issue_type
  - metric
  - value
  - comparison
  - rule_id
  - config_version
  - needs_manual_review
```

示例：

```json
{
  "code": "bone_length_ratio_spread_review",
  "severity": "warn",
  "module": "keypoint_morphology",
  "issue_type": "keypoint_static_morphology_abnormal",
  "metric": "keypoint_morphology.left_bone_length_ratio_spread_p95",
  "value": 3.6,
  "comparison": ">=",
  "rule_id": "keypoint_morphology.bone_length_ratio_spread",
  "config_version": "qc_acceptance_v1.0.0",
  "needs_manual_review": true
}
```

注意：这里不写 threshold。阈值通过 `config_version + rule_id` 回查。

## 7.7 模块配置范围

`configs/qc_acceptance.yaml` 当前覆盖：

- `hdf5_text_info`
- `quality_hand`
- `keypoint_presence`
- `keypoint_morphology`
- `keypoint_temporal`
- `video_quality`
- `sam3_containment`
- `semantic_consistency`
- `manual_review`
- `duplicate_check`
- `content_validity`
- `effective_duration`

其中 `keypoint_morphology` 直接来自 `tmp/integrate-colleague-acceptance-20260707` 新增模块，包含：

- `duplicate_joint_distance_m`
- `min_palm_scale_m`
- `max_bone_length_ratio_spread_review/fail`
- `max_normalized_bone_length_review/fail`
- `max_zero_length_bone_count_review/fail`
- `max_duplicate_joint_pair_count_review/fail`
- `min_joint_angle_deg_review/fail`
- `max_joint_angle_violation_fraction_review/fail`

## 7.8 Rule ID 规范

每个可触发 warn/fail 的规则必须有稳定 `rule_id`。

格式：

```text
<module_name>.<rule_name>
```

示例：

```text
keypoint_morphology.bone_length_ratio_spread
video_quality.confirmed_freeze
semantic_consistency.object_mismatch
duplicate_check.high_similarity_duplicate
```

要求：

- `rule_id` 一旦进入历史 JSON，不要重命名。
- 如果规则语义变化，升级 `config_version`。
- 如果规则废弃，保留在配置的兼容说明里，不要让历史 JSON 无法解释。

## 7.9 模块读取配置方式

建议实现共享工具：

```text
qc_common/qc_config.py
```

建议 API：

```python
load_qc_acceptance_config(path)
resolve_module_config(config, module_name)
compute_config_hash(config_path)
build_top_level_qc_config(config, config_path)
resolve_rule(config, rule_id)
```

行为要求：

- `load_qc_acceptance_config` 读取 YAML，并校验必填字段。
- `resolve_module_config` 返回 `modules.<module_name>`。
- `compute_config_hash` 对实际配置文件内容计算 sha256。
- `build_top_level_qc_config` 返回 JSON 顶层 `qc_config`。
- `resolve_rule` 支持批次统计按 `config_version + rule_id` 回查阈值和判定规则。

## 7.10 视频 QC 接入要求

视频 QC 已经输出过 thresholds。新版本应改为：

- 不再写 `video_quality.thresholds`。
- 不再写模块级 `video_quality.config_ref`。
- 顶层写 `qc_config`。
- 每个 warn/fail detail 写 `rule_id`、`value`、`comparison`、`config_version`。
- 视频模块自己的指标值继续保留，例如冻结区间、bad visual duration、清晰度统计、曝光比例。

视频 QC 可以先保留 `threshold_profile: video_prefilter_v0.3.1` 在中心配置里，用于说明当前阈值口径。

## 7.11 兼容和迁移策略

### 阶段 1：中心配置落地

- 新增 `configs/qc_acceptance.yaml`。
- PRD 明确 JSON 只写顶层 `qc_config`。

### 阶段 2：共享配置 loader

- 新增 `qc_common/qc_config.py`。
- 所有模块 CLI 支持 `--qc-config`。

### 阶段 3：precheck 接入

- `hdf5_text_info`
- `quality_hand`
- `keypoint_presence`
- `keypoint_morphology`
- `keypoint_temporal`

### 阶段 4：视频 QC 接入

- 移除新 JSON 中的视频 thresholds 输出。
- 用 rule_id 和 config_version 解释每个原因。

### 阶段 5：人工质检和批次统计接入

- 人工质检只读 JSON 的 `manual_review.candidates`。
- 批次统计按 `qc_config.config_version` 回查配置。

## 7.12 错误处理

| 情况 | 行为 |
|---|---|
| 找不到中心配置 | pipeline fail fast，不运行 QC。 |
| 配置 YAML 解析失败 | pipeline fail fast，输出配置错误。 |
| 缺少模块配置 | 当前模块 fail fast，并写入配置错误。 |
| 缺少 rule_id | 当前模块 fail fast，这是实现错误。 |
| JSON 顶层缺少 `qc_config` | 新流程视为错误；旧 JSON 可走兼容读取。 |
| 新模块继续写 thresholds | 测试失败，读端忽略该字段。 |

## 7.13 测试要求

必须覆盖：

- 配置文件能被解析。
- `pipeline.modules` 顺序正确。
- 每个模块都能 resolve 自己的 config。
- `config_hash` 稳定可复现。
- 顶层 JSON 写入 `qc_config`。
- 模块 JSON block 不再写 `thresholds`。
- 每个 warn/fail reason detail 都有 `rule_id` 和 `config_version`。
- 修改阈值后，`config_version` 变化能被测试捕获。
- 旧 JSON 中未知字段不会被配置写入流程删除。

## 8. Release

### V1: 文档和中心配置

交付：

- `docs/PRD-qc-unified-config.md`
- `configs/qc_acceptance.yaml`

验收：

- 文档覆盖所有当前 QC 模块。
- YAML 可解析。
- pipeline 顺序和 gate PRD 一致。
- JSON 新规范明确不写模块 thresholds。

### V2: 配置 loader

交付：

- `qc_common/qc_config.py`
- 单元测试

验收：

- 能读取中心配置。
- 能 resolve 模块配置。
- 能计算配置 hash。
- 能生成顶层 `qc_config`。
- 能通过 `rule_id` 回查规则。

### V3: 模块接入

交付：

- precheck 模块接入 `--qc-config`。
- 视频 QC 停止写 thresholds。
- 人工质检读取 `modules.manual_review`。

验收：

- 每条 JSON 顶层都有 `qc_config`。
- 每个 warn/fail 都有 `rule_id`、`value`、`config_version`。
- fail/warn/pass 流程不变。

### V4: 批次统计只读 JSON 和版本化配置

交付：

- 批次统计从 `quality_archive/*.json` 聚合。
- 根据 `qc_config.config_version` 加载对应配置。
- 输出 CSV、XLSX、Markdown。

验收：

- 不依赖散落 sidecar/CSV 也能生成完整批次报告。
- 历史 JSON 不受最新中心配置影响。
