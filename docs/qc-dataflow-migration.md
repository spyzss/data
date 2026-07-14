# QC 数据流 v1 → v2 迁移与回滚指南

本文是 reviewer、批次统计和运维的操作合同。当前 canonical 版本为
`asset_qc_report.v2` + `qc_acceptance_config_schema.v2`，配置版本为
`qc_acceptance_v2.0.0`。

## 1. 目标目录与唯一事实源

每个批次必须有：

```text
<batch>/quality_archive/<asset_id>.json
configs/qc_acceptance.yaml
configs/qc_acceptance/qc_acceptance_v2.0.0.yaml
schemas/asset_qc_report.v2.schema.json
schemas/qc_acceptance_config.v2.schema.json
```

`quality_archive/*.json` 是每个资产的唯一 master verdict 和批次统计事实源。
CSV、XLSX、Markdown、Parquet、cache、ledger event、overlay 和其他 sidecar 都是
可删除的派生物；sidecar 只作证据和 reconciliation，不能覆盖 JSON 的
`overall_decision`、机器 issue 或人工 effective verdict。

## 2. v2 Config 与 profile

活动配置和不可变快照必须字节一致：

```yaml
schema_version: qc_acceptance_config_schema.v2
config_version: qc_acceptance_v2.0.0
config_name: acceptance_gate
execution_profiles:
  acceptance:
    fail_action: stop
    runtime_error_action: stop_incomplete
  supplier_evaluation:
    fail_action: record_and_continue
    runtime_error_action: stop_incomplete
```

模块顺序由 `pipeline.modules` 给出；`semantic_consistency` 和 `manual_review` 使用
`execution_kind: external`。规则阈值、rule ID、顺序、profile 或人工路由变更时，
必须生成新的 `config_version`，复制到不可变快照并更新 hash；不得修改已发布快照。

profile 的流转差异：

| 场景 | acceptance | supplier_evaluation |
|---|---|---|
| 自动 `pass`/`warn` | 记录并进入下一模块 | 记录并进入下一模块 |
| 自动 hard `fail` | `stopped`、`overall_decision=fail`，跳过语义/人工 | 写 fail issue，`continued_after_fail=true`，记录后继续；完成后 `overall_decision=fail` |
| runtime/evidence/config/CAS 错误 | `error`、`overall_decision=null` | 同左；不得把错误当质量 fail |

## 3. v1 只读迁移

1. 停止对 v1 文件的并发写入，保存原文件 hash、`asset_id`、revision 和来源批次。
   对历史配置文件保留原始 hash（例如
   `shasum -a 256 configs/qc_acceptance/qc_acceptance_v1.1.0.yaml`），不可重新序列化
   或覆盖该快照。
2. 用 schema reader 读取 v1；不直接修改 v1，也不把 v1 sidecar 结论拼进新 verdict。
3. 调用纯函数 `migrate_v1_to_v2()`，保留 video block、unknown fields、原
   `report_revision`、source files 和可解释的历史 `qc_config` 引用。
4. 用当前 v2 Config 校验 profile、config version/hash、module cursor 和 v2 Schema。
5. 将迁移结果作为一次 v2 CAS 写回；若预期 revision 不匹配，拒绝写回并重新读取，
   不能静默覆盖。
6. 迁移成功后，v1 原文件仍作为只读历史输入保留；后续正式模块只写 v2 master。

迁移不会自动把一个旧 sidecar 的“建议”变成 machine pass/fail。缺失证据只能形成
结构化 runtime/evidence error 或待人工状态，不能猜测结论。

## 4. v2 正式流转

```text
自动 QC Gate
  acceptance: hard fail -> stopped/fail；不进入语义和人工质检
  supplier_evaluation: hard fail -> 记录并继续
-> semantic_consistency external
-> candidate_issue_ids 为空：manual_review=not_required
-> candidate_issue_ids 非空：manual_review=queued
-> 最终 overall_decision=pass|fail
-> 批次输出只投影 quality_archive/*.json
```

语义校准是人工质检之前的 external 阶段，当前由人工工作台实现，后续可替换为模型
adapter。人工 queue 状态为 `not_evaluated`、`not_required`、`required`、`queued`、
`in_progress`、`completed` 或 `skipped_due_to_fail`。只有累计 warn 候选进入人工队列；
空候选不做正常 Pass 样本抽检。机器 issue 只追加人工 review 记录，不被人工 verdict
覆盖；人工确认 fail 由聚合器单独统计。

## 5. 写回与错误分类

每个模块事务固定为：读取当前 JSON → 校验 asset/config/profile/next_module 和
expected `report_revision` → 只替换本模块 block/issue/evidence → 全量重建候选和
fail 引用 → revision 加 1 → v2 Schema 校验 → 同目录临时文件 `fsync` → 原子
`os.replace`。

`EvidenceRef.path` 必须是相对 batch root 的 POSIX 路径；绝对路径或解析后越出 batch
root 的 `..` 路径直接拒绝。错误分类如下：

| 类型 | 报告状态 | `overall_decision` | 统计含义 |
|---|---|---|---|
| 算法观测 hard fail | acceptance `stopped`；supplier `running`/`completed` | 最终 `fail` | quality fail，保留 issue |
| stale revision/CAS | `error` 或拒绝写回 | `null` | 未完成/需重试，不是质量 fail |
| runtime/evidence/config/schema | `error` | `null` | runtime error，不能计入 pass |
| 人工确认 warn 为 fail | 不改 machine severity | 依流程收口为 `fail` | 单独计入 human confirmed fail |

## 6. 正式 CLI 与可重建 cache

```bash
ARCHIVE=sampled/XJGT_20260616/quality_archive

python tools/build_manual_review_queue.py \
  --quality-archive "$ARCHIVE" \
  --output-dir sampled/XJGT_20260616/manual_review

python tools/build_qc_json_projection.py \
  --quality-archive "$ARCHIVE" \
  --output-dir sampled/XJGT_20260616/qc_projection \
  --cache-dir sampled/XJGT_20260616/qc_cache \
  --formats csv parquet xlsx markdown

python tools/build_batch_qc_ledger.py \
  --quality-archive "$ARCHIVE" \
  --output-dir sampled/XJGT_20260616/ledger \
  --formats csv parquet xlsx markdown

python tools/build_xjgt_acceptance_report.py \
  --quality-archive "$ARCHIVE" \
  --output-dir sampled/XJGT_20260616/xjgt_report
```

正式入口必须传 `--quality-archive`；读取器逐文件校验 v2 JSON，并生成 asset、issue、
execution 三类表。cache 是加速层，不参与事实判定；source manifest 必须逐项匹配
relative report path、asset ID、revision 和 SHA-256，任一变化就删除/重建。旧
candidate-window、SAM3、video 或 manual sidecar 只能通过显式
`--legacy-reconciliation-*` 参数生成对账证据，不能改变 canonical projection。

## 7. 回滚与禁止事项

回滚步骤：

1. 停止新 profile 的写入任务并保留当前 v2 master、hash 和 revision。
2. 删除可重建 projection/cache，保留 sidecar 作为证据。
3. 将消费者切换为 v1 **只读** reader 或上一份已验证的 v2 projection；如需继续
   写入，必须发布兼容的新 Config/Schema，而不是回写旧快照。
4. 对账完成后重新从 `quality_archive/*.json` 构建输出。

禁止：用 sidecar/ledger/旧 CSV 覆盖 master verdict；把 v1 内容直接复制回 v2；
降低 revision；跳过 CAS、evidence 相对路径或 schema 校验；把 runtime error 当作
quality pass/fail；把 `supplier_evaluation` 的机器 fail 改成 warn/pass。

## 8. 三条可执行迁移路径

下面的命令以批次根目录为例。迁移前先冻结写入；所有输出目录都可以删除并重建，
`quality_archive` 中的报告字节和 revision 则必须保留。`shasum` 的结果和配置快照
一起归档，作为迁移前后的审计证据。

### 8.1 正常升级：v1 只读输入，首次 module 写回 v2

```bash
set -eu
BATCH_ROOT=sampled/XJGT_20260616
ARCHIVE="$BATCH_ROOT/quality_archive"
SNAPSHOT="$BATCH_ROOT/migration_snapshot/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$SNAPSHOT"

# 1) 停止旧 runner，并保存配置快照、hash 和 master JSON 备份
cp -p configs/qc_acceptance.yaml "$SNAPSHOT/qc_acceptance.yaml"
cp -p configs/qc_acceptance/qc_acceptance_v2.0.0.yaml "$SNAPSHOT/qc_acceptance_v2.0.0.yaml"
shasum -a 256 "$SNAPSHOT"/*.yaml > "$SNAPSHOT/config.sha256"
find "$ARCHIVE" -type f -name '*.json' -exec shasum -a 256 {} \; \
  | sort > "$SNAPSHOT/quality_archive.before.sha256"
tar -czf "$SNAPSHOT/quality_archive.before.tgz" -C "$BATCH_ROOT" quality_archive

# 2) 由受控 runner 对每个 v1 report 调用 migrate_v1_to_v2()，再用预期
#    report_revision 做 CAS 写回；不得原地重序列化 v1 文件或拼接 sidecar verdict。
# 3) 校验 v2 schema/config 后再运行正式 projection 和统计入口。
python tools/build_qc_json_projection.py \
  --quality-archive "$ARCHIVE" \
  --output-dir "$BATCH_ROOT/qc_projection" \
  --formats csv parquet xlsx markdown
```

迁移 runner 必须记录每个 `asset_id` 的旧 revision、新 revision、配置 hash 和
写回结果；CAS 冲突只记录 error 并重试读取，不覆盖并发更新。迁移完成后再次运行
`find ... | shasum`，确认未迁移资产与 `quality_archive.before.sha256` 一致。

### 8.2 只读验证：只生成 projection/reconciliation

该路径不调用任何 report mutation，也不创建 v2 master。适用于先评估旧 sidecar
与当前 JSON 的差异：

```bash
set -eu
BATCH_ROOT=sampled/XJGT_20260616
ARCHIVE="$BATCH_ROOT/quality_archive"
OUT="$BATCH_ROOT/migration_readonly"
python tools/build_qc_json_projection.py \
  --quality-archive "$ARCHIVE" \
  --output-dir "$OUT/projection" \
  --formats csv markdown \
  --legacy-reconciliation-candidate-windows "$BATCH_ROOT/candidate_windows.json" \
  --legacy-reconciliation-sam3-window-summary "$BATCH_ROOT/sam3_window_summary.json" \
  --legacy-reconciliation-video-quality "$BATCH_ROOT/video_quality_results.json" \
  --legacy-reconciliation-issue-events "$BATCH_ROOT/issue_events.json"
```

或者在 Python 只读工具中调用
`reconcile_legacy_outputs(quality_archive=ARCHIVE, legacy_inputs=[...])`；返回值仅
包含差异，并将 `authoritative_source` 标为 `asset_qc_json`。`reconciliation.csv`、
projection 表和 cache 都不参与 verdict 统计，也不能反向写入报告。

### 8.3 回滚：停 v2 writer，恢复旧 runner 但保留 v2 master

```bash
set -eu
BATCH_ROOT=sampled/XJGT_20260616
ARCHIVE="$BATCH_ROOT/quality_archive"
ROLLBACK="$BATCH_ROOT/migration_snapshot/rollback-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$ROLLBACK"

# 1) 先停 v2 writer；保留当前 v2 master、revision 和 hash
find "$ARCHIVE" -type f -name '*.json' -exec shasum -a 256 {} \; \
  | sort > "$ROLLBACK/quality_archive.v2.sha256"
tar -czf "$ROLLBACK/quality_archive.v2.tgz" -C "$BATCH_ROOT" quality_archive

# 2) 旧 runner 只允许生成 sidecar，禁止写入或覆盖 quality_archive/*.json
rm -rf "$BATCH_ROOT/qc_projection" "$BATCH_ROOT/qc_cache"
python tools/build_qc_json_projection.py \
  --quality-archive "$ARCHIVE" \
  --output-dir "$BATCH_ROOT/qc_projection.rollback" \
  --formats csv markdown

# 3) 恢复验证：master JSON 的 hash 必须仍与回滚前相同
find "$ARCHIVE" -type f -name '*.json' -exec shasum -a 256 {} \; \
  | sort > "$ROLLBACK/quality_archive.after.sha256"
cmp "$ROLLBACK/quality_archive.v2.sha256" "$ROLLBACK/quality_archive.after.sha256"
```

如确需从备份恢复，先将当前目录改名，再解包
`quality_archive.v2.tgz`，最后重新执行上面的 `cmp` 和
`python tools/build_qc_json_projection.py` 验证。旧 runner 产生的旧报表、CSV、ledger
或 sidecar 只能作为 reconciliation evidence；无论回滚还是重试，都不得覆盖已经
存在的 v2 master JSON、降低 `report_revision` 或替换其 `overall_decision`。
