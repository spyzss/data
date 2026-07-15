# Resource-Efficient QC Pipeline Implementation Audit

**Date:** 2026-07-15

**Branch:** `codex/human-qc-impl`

**Reviewed base HEAD:** `bba9353 fix(human-qc): recover downstream handoff`

**Scope:** 方案 A（SAM3 只消费本轮 precheck canonical candidate artifact）、precheck 资源优化、producer cache、状态机和现有 human-QC 架构审查。

**Data/model execution:** 未在本地运行供应商数据、SAM3 模型或云端任务。

## 1. Executive summary

方案 A 已实现，并且没有把 precheck、video quality 和 SAM3 合成一个 Python 大模块。当前仍是一个统一 CLI 编排多个 producer，每个 producer 通过 manifest、JSON/CSV sidecar 和 v2 report 通信。

本次实现带来的主要结果：

- 同一个 asset 的五项 precheck 只加载一次 JDT Parquet 或 DeepReach HDF5；五项检查仍分别计算、分别写 module result。
- SAM3 不再读取 manifest 中可能过期的 `candidate_windows_path`，只读取本轮 precheck 的 `module_outputs/<asset_id>/precheck/candidate_windows.json`。
- 有效空候选以 `[]` 落盘，SAM3 在创建 segmenter、查找模型和读取 manifest 之前直接返回 `skipped/no_candidates`。
- precheck、video quality、SAM3 都有 canonical producer artifact；完整且指纹一致的 artifact 可以复用。
- `supplier_evaluation` 遇到单项 quality fail 或 runtime/integration error 会记录并继续后续自动模块；任何 required incomplete 都使最终状态为 `incomplete`、`overall_decision=null`。
- `acceptance` 仍可 fail-fast；后继模块现在明确写 `not_run + blocked_by_quality_fail:<module>`，不再用容易误解的 pass/skipped 状态。
- DeepReach 非空候选不会套用 JDT adapter，也不会伪造 pass；当前明确返回 `adapter_missing`。这意味着 **DR head SAM3 还不能真实运行**，只是失败语义变安全了。
- 当前分支没有 `tools/build_jdt_video_manifest.py`。JDT/JD 的“一视频一 task”只有在输入 manifest 已经做到“一行一个稳定 asset_id/视频”时才成立，本次没有补 manifest builder。

建议当前临时供应商数据使用 `supplier_evaluation`，但把结果理解为“尽可能完整的诊断覆盖”，不是“所有模块已经有最终验收结论”。外部 semantic/manual 阶段仍需单独完成，三个尾部模块目前仍 disabled。

## 2. Current implemented workflow

```text
canonical manifest
  └─ one manifest row -> one AssetContext -> one asset worker
       ├─ precheck producer session (serial checks, one source load)
       │    1. hdf5_text_info
       │    2. quality_hand
       │    3. keypoint_presence
       │    4. keypoint_morphology
       │    5. keypoint_temporal
       │         └─ canonical candidate_windows.json
       │
       ├─ video_quality producer
       │
       ├─ sam3_containment producer
       │    ├─ empty current candidates -> skipped/no_candidates, no model load
       │    ├─ JDT + valid candidates -> direct Parquet 2D keypoints -> SAM3
       │    └─ DeepReach + valid candidates -> adapter_missing
       │
       ├─ semantic_consistency (external; CLI pauses at awaiting_external)
       ├─ manual_review (external; selected issues only)
       ├─ duplicate_check (disabled)
       ├─ content_validity (disabled)
       └─ effective_duration (disabled)

durable module_outputs + quality_archive/<asset_id>.json
  └─ existing ledger/manual-review/reporting consumers
```

`--max-workers` 并行的是 asset，不是同一 asset 内的 module。同一 asset 内当前仍按 config 顺序串行，因此 video quality 虽然在文件契约上独立，统一 CLI 中没有和 precheck 并行执行。

### 2.1 这不是怎样的“耦合”

统一 CLI 确实提供一个 asset 级状态机和 cursor，但它不把上一个 module 的 Python 对象直接传给下一个 heavy producer：

- precheck 与 SAM3 的交接是 durable `candidate_windows.json + run_config.json`；
- video quality 不读取 precheck 内存状态；
- SAM3 raw frame/window/failure/evidence 输出先发布，再适配进 v2 report；
- report 保留每个 module 的 raw verdict/status，不用 final verdict 覆盖它们。

因此它是“统一外部编排 + 文件级模块边界”，不是一个无法单独运行的 monolithic detector。原来的 standalone manifest runners 仍可单独使用。

## 3. Resource behavior before and after

| 项目 | 修改前 | 修改后 | 仍然存在的成本 |
|---|---|---|---|
| 五项 precheck 源读取 | 每项重新加载 Parquet/HDF5，最多五次 | 每 asset/session 一次 | 五项算法仍各自计算一次 |
| PrecheckRunner | 每项单独构造 | 仍按 module 构造轻量 runner，但复用同一 decoded clip | 不是设计文档最初写的“一个 runner 同时启用五项”；实际节省来自共享 source load |
| Temporal -> SAM3 | 依赖 manifest 旧 candidate sidecar | 读取本轮 canonical candidate artifact | producer 顺序仍串行 |
| 空 candidate | 仍可能查模型/manifest | 模型零加载快速跳过 | 尚未发布独立 SAM3 `no_candidates` artifact，skip 记录在 report |
| Video quality 重跑 | 总是重新 analyzer | 指纹命中时读取 artifact | completed report 本身不会自动重开；强制重跑应使用新 batch root |
| SAM3 重跑 | 总是重跑模型 | candidate SHA、视频/2D/model/config 命中时复用 | 自定义注入的 `segmenter_factory` 身份未进入指纹 |

这个改动主要减少重复 I/O、decode、producer setup 和重复模型运行。它不会把五项 QC 算法压缩成一次计算，所以真实耗时仍需在云端数据上测量。CLI 现在输出每个 asset 总耗时以及 producer 的 `computed/reused/skipped/blocked/failed` 状态，可用第一次与第二次运行的日志做实际估算。

## 4. Two-profile behavior

真实名称和选择方式：

| Profile | CLI | Quality fail | Runtime/input/adapter error | 默认 |
|---|---|---|---|---|
| `acceptance` | `--profile acceptance` | stop | `stop_incomplete` | 是 |
| `supplier_evaluation` | `--profile supplier_evaluation` | record and continue | `record_and_continue` | 否 |

当前临时供应商测评必须显式传 `--profile supplier_evaluation`；不传仍是 acceptance。

### 4.1 Acceptance

- quality fail：pipeline `stopped`，最终 quality decision 可为 fail；后继 required module 写 `not_run` 和 blocker reason。
- runtime error、input missing、adapter missing：pipeline `error`，`overall_decision=null`，停在当前 module。
- 它不会预先计算后继 precheck，因此 fail-fast 能真正节省后续计算。

### 4.2 Supplier evaluation

- quality fail：保留 fail block，继续后续自动模块；如果其余 required module 都完整，最终仍可按 raw fail 得到 fail。
- runtime/input/adapter error：写进 `runtime_errors` 和 `execution.module_states`，继续后续自动模块。
- 全部可调度模块走完但存在 incomplete：pipeline `incomplete`，`overall_decision=null`。
- 到 `semantic_consistency` 或需要人工的 `manual_review` 时仍会 `awaiting_external`；它不是一次 CLI 自动跑完所有人工环节。
- 外部阶段完成后如果前面仍 incomplete，恢复逻辑继续保持 `incomplete + overall_decision=null`，不会补成 pass/fail。

### 4.3 State meanings

| State | 含义 | 是否可当 pass |
|---|---|---|
| `completed` | module 有真实结果 | 由 raw verdict 决定 |
| `not_run` | 被 acceptance fail gate 阻止 | 否 |
| `input_missing` | required source/sidecar 不存在 | 否 |
| `input_invalid` | source/candidate 存在但违反契约 | 否 |
| `adapter_missing` | supplier adapter 未实现或未验证 | 否 |
| `blocked` | 依赖/lineage 阻塞 | 否 |
| `runtime_error` | producer 异常 | 否 |
| `not_implemented` | implementation 未注册 | 否 |
| `skipped/no_candidates` | current temporal artifact 有效且候选为空 | 只表示 SAM3 不需要运行，不覆盖 temporal 状态 |
| `disabled` | config 明确禁用 | 只有 policy 明确不要求时才可排除 |

## 5. Artifact and cache contract

```text
module_outputs/<asset_id>/precheck/
  check_results.json
  clip_aggregates.json
  candidate_windows.json
  run_config.json

module_outputs/<asset_id>/video_quality/
  video_quality_result.json
  run_config.json

module_outputs/<asset_id>/sam3_containment/
  frame_results.json
  window_results.json
  failures.json
  evidence_manifest.json
  evidence/
  producer_run_config.json
  run_config.json
```

每个 `run_config.json` 记录 producer、outcome、config/module fingerprint、source identity、创建时间和耗时。SAM3 还包含 exact candidate SHA、queries 和 runtime settings。

复用条件是 required files 全部存在、JSON 可读、outcome 可复用且完整 fingerprint 一致。发布采用 sibling staging directory 和 atomic rename；video/SAM3 重算失败不会覆盖旧 artifact。

已知限制：

1. 本地 source identity 使用 relative path、size、`mtime_ns`。helper 支持 checksum/ETag/version ID，但 `contexts_from_manifest()` 目前没有把 manifest 这些列传入 `source_files`，普通 CLI 仍主要依赖 size + mtime。
2. `--resume` 同时代表“沿现有 report cursor 恢复”和“允许 artifact reuse”。一个已经 terminal 的 report 不会因为 source fingerprint 改变而自动重开。需要强制重跑时，应使用新的 batch root；当前 `--no-resume` 会拒绝覆盖已有 report。
3. precheck 在 temporal 成功后会发布当前 partial artifact，避免 SAM3 使用旧候选；这会替换旧 full precheck cache，不保留历史 rollback。video/SAM3 的失败重算则保留旧 artifact。
4. SAM3 空候选不创建自己的完整 sidecar；它依赖 durable precheck 空 candidate 和 report skip 状态。成本很小，但三类 producer 的 artifact 完整性还不完全对称。
5. generic report evidence validation 主要校验路径必须留在 batch root；producer 自身会校验/复制 SAM3 evidence，但 report mutation 尚未统一强制所有 evidence path 已存在。

## 6. JDT/JD behavior and missing manifest work

当前统一 CLI 的 task 粒度由 manifest 决定：一行就是一个 `AssetContext`，`asset_id` 必须唯一。代码不会自动扫描 JD/JDT 文件夹、统计视频或把 subtask 重组为 video task。

因此要实现“一个视频一个 task”，manifest 必须满足：

- 每个视频/episode 一行；
- 每行有唯一、稳定、可回连 supplier source 的 `asset_id`；
- `primary_video_path` 和 `parquet_path` 指向该视频对应源；
- 可选 `start_frame/end_frame` 使用 inclusive source coordinates；CLI 只在内部转换成 half-open `AssetContext.source_range`。

当前 branch 和 `origin/refresh` 都没有 `tools/build_jdt_video_manifest.py`。分支关系为：

- merge base: `5da8edf9f423fd403ee2439a80e541af56c3b0eb`
- `origin/refresh...HEAD` 独有提交数：`1 / 68`
- refresh 唯一远端提交 `c7c7620` 只修改 `AGENTS.md` 和 `README.md`
- `origin/refresh` 不是当前 HEAD 的祖先

所以当前不能把“refresh 上已经有 JDT video-level builder”作为云端前提。必须先提供已生成好的 canonical manifest，或后续单独移植/实现 builder。

JDT 非空候选 SAM3 路径保持 direct Parquet 2D keypoints，没有伪造 calibration。manifest 旧 `candidate_windows_path` 仍保留为 metadata，但统一 pipeline 不使用；standalone SAM3 runner 的显式 `--candidate-windows` 不变。

## 7. DeepReach head status

本次没有实现 DeepReach head calibration/projection adapter，这是设计中的明确 non-goal。

当前行为：

- manifest 必须自己把 `primary_video_path` 指向 head 视频；统一 runner 不会自动从多相机源里选择 head。
- temporal candidate 为空：SAM3 `skipped/no_candidates`，不需要 calibration。
- temporal candidate 非空：SAM3 在模型加载前返回 `adapter_missing`，reason 明确说明 head calibration/projection adapter 未验证。
- `supplier_evaluation` 会继续后续可执行模块，但最终不能 pass；terminal readiness 是 incomplete。
- 不会把 blocked 当作骨骼 fail，也不会让一个 asset 的 blocked 中止其他 asset worker。

因此“DR 的内参映射已经传入”尚未接入当前 `AssetContext`/manifest adapter；也没有完成外参、2D keypoint field、frame alignment 和 projection lineage 的端到端验证。现在还不能宣称 DR SAM3 可跑。

## 8. Human QC and “head/tail” review

本次资源优化没有修改 `human_qc/`、`qc_reporting/`、旧 manual-review queue 或 ledger 工具。

同事分支里的“头尾”至少有两种不同语义，不能混在一起：

1. video quality 的 `include_head_tail_frames: 10` 是 decode sampling 策略，表示额外覆盖视频头尾帧。
2. semantic calibration 的 shared-boundary timeline 固定最外层起点/终点，只允许移动内部共享 boundary；一个 boundary edit 同时影响相邻两个 segment。这里的“头尾”不是 DR head camera，也不是 production pipeline 中名为 head/tail 的 module。测试里的 `tail` 多数只是下游 module 占位名。

现有 human workflow：

```text
automatic QC report
  -> semantic_consistency external workbench
       -> shared-boundary edit / subtask text edit
       -> staged, recoverable HDF5 replacement
  -> manual_review external workbench
       -> selected_issue_ids
       -> issue_reviews
  -> orchestrator resume
  -> qc_reporting projection/export
```

风险和重复：

- `human_qc` 以 v2 report 的 `issue_id/selected_issue_ids` 为主；旧 `tools/build_manual_review_queue.py`、manual labels 和 ledger 以 `review_id`/window contract 为主。`human_qc/legacy_import.py` 提供迁移，但两套入口仍并存，属于逻辑重叠，不是完全统一的 review contract。
- warn candidates 只有被 selection policy/外部操作写入 `selected_issue_ids` 才进入人工 review；空 selected list 会变成 `not_required`。必须确保供应商测评所需 issue selection 已执行，不能把空列表误解为“没有问题”。
- semantic workbench 可以经过 recoverable transaction 替换 source HDF5 bytes。这有完整恢复测试，但云端必须把它当写操作，使用备份、lease 和 revision guard；纯审查/测评环境不应默认开放写源文件权限。
- candidate window 仍不能直接当 rejected duration。现有旧 ledger/manual tools保留 `review_id` 优先和 affected intervals 统计；本次没有改这些语义。

## 9. Module boundary audit

| Constraint | Result | Notes |
|---|---|---|
| precheck 不加载 SAM3/DA3/VLM | 通过 | heavy model 只在 SAM3 runner；`precheck/` 未引入模型依赖 |
| producer 通过显式文件契约通信 | 通过 | temporal -> SAM3 使用 canonical JSON + run config |
| 缺失/blocked 不伪装 pass | 通过 | typed states + incomplete/null final decision |
| final verdict 不覆盖 raw status | 通过 | module blocks 和 execution states 保留 |
| candidate duration 不等于 rejected duration | 本次未改，现有工具测试通过 | human/ledger 仍需人工确认 affected intervals |
| 人工匹配优先 review_id | 旧工具通过；新 human stack为不同 issue_id contract | 需要集成时人工选择唯一 canonical review identity |
| SAM3 blocked 不让整批自动 fail | 通过 | record-level state；asset workers 独立继续 |
| supplier-specific schema 经 adapter | 部分通过 | JDT explicit；DeepReach explicit missing；supplier 缺失时 runner 当前默认 `jdt`，manifest 应强制提供 supplier |

### 9.1 未修复的 policy 风险

`quality_hand.low_quality_both_hands` 在 active v2.1 config 中仍是 hard fail。仓库规则明确 `quality_hand` 是 supplier-provided evidence，不能作为跨供应商通用 skeleton hard fail。本次没有擅自改阈值/验收语义；在用于多供应商前必须按 supplier policy 拆分或降级，否则存在 schema 不冲突但行为错误的风险。

另外，active config 的 video-quality pipeline 中 `do_keypoint_quality_check: false` 重复出现两次。两个值相同，所以当前 YAML 解析后的行为不变，但应在下一次 config 版本清理并加入 duplicate-key lint。

## 10. Output/schema changes

本次新增或改变：

- active config 发布为 `qc_acceptance_v2.1.0`；v2.0 snapshot 字节未改并有 SHA 测试。
- config schema 允许 `runtime_error_action=record_and_continue`。
- report pipeline status 增加 `incomplete`；该状态只允许 `overall_decision=null`。
- `execution.module_states` 显式支持 `not_run/input_missing/input_invalid/adapter_missing/blocked/runtime_error/not_implemented`。
- CLI summary 增加 producer state counts 和 asset elapsed time。
- canonical producer artifacts 新增，但没有替换旧 ledger/raw module output schema。

需要注意：`quality_archive` v2 report + `qc_reporting` 与旧 batch ledger/manual-review/weekly workbook 仍是两套相邻的聚合层。Git 不一定冲突，但如果两套同时作为最终事实来源会产生行为/schema 冲突。

## 11. Tests and verification

最终本地验证：

```text
.venv/bin/python -m pytest -q
803 passed, 1 skipped in 20.26s

python3 -m compileall precheck qc_common tools acceptance_pull annotation_verify qc_pipeline
PASS

git diff --check
PASS
```

新增测试覆盖：

- artifact path、fingerprint、损坏/缺文件、原子发布保护；
- 五项 precheck 单次 source load、acceptance early stop、跨 asset 隔离；
- complete/partial precheck artifact、空 candidate、跨 session cache、source invalidation；
- SAM3 current candidate handoff、旧 manifest candidate 忽略、candidate bounds/coordinate validation；
- 空 candidate 不加载模型、DeepReach adapter missing、stale precheck run rejection；
- video/SAM3 cache hit、candidate SHA invalidation、失败重算保留旧 artifact；
- acceptance vs supplier-evaluation runtime/fail behavior；
- incomplete external-resume 不能生成 final pass/fail；
- CLI legacy candidate metadata、producer summary 和 elapsed time。

没有运行：

- 真实 JDT/JD Parquet/video；
- 真实 DeepReach HDF5/head video/calibration；
- 真实 SAM3 checkpoint；
- 云端吞吐、GPU memory 或运行时 benchmark。

## 12. Residual risks by priority

### High

1. **DR head SAM3 adapter 未实现。** 当前只能安全标记缺失，不能交付真实 DR containment 数据。
2. **JDT video-level manifest builder 缺失。** 当前 runner 不会自动把多视频目录变成一视频一 task。
3. **`quality_hand` 跨供应商 hard-fail policy 不符合仓库边界。** 在正式多供应商验收前必须明确 supplier-specific policy。

### Medium

1. 同一 asset 的独立 producer 仍串行，video quality 与 precheck 没有并行调度。
2. terminal report 不会因 source 改动自动重开；强制重跑需新 batch root。
3. `human_qc` issue-id workflow 与旧 review-id queue/ledger 并行存在，最终人工事实来源需统一。
4. precheck 当前 partial artifact 会替换旧 full artifact，防止 stale candidates，但不保留历史 rollback。
5. checksum/ETag/version ID 尚未从普通 manifest CLI 映射到 source identity。

### Low

1. 空 candidate 没有独立 SAM3 no-candidates artifact。
2. 自定义 `segmenter_factory` identity 不在 cache fingerprint。
3. report 层尚未统一强制每种 EvidenceRef 的文件存在性。
4. active v2.1 config 有一个相同值的 duplicate YAML key。

## 13. Cloud validation commands

本地修改目前未 commit、未 push，因此云端现在还 pull 不到本报告中的实现。用户明确要求 commit/push 后，云端同步命令应为：

```bash
git fetch origin --prune
git switch codex/human-qc-impl
git pull --ff-only origin codex/human-qc-impl
```

使用已经准备好的“一视频一行” canonical manifest，在新的 batch root 中运行：

```bash
python tools/run_qc_pipeline.py \
  --batch-root /path/to/qc_run_20260715 \
  --manifest /path/to/qc_run_20260715/manifest.jsonl \
  --profile supplier_evaluation \
  --max-workers 1 \
  --resume
```

GPU/SAM3 真实验证前先用少量 asset，确认：

- JDT 每一行 `asset_id` 唯一且确实对应一个视频；
- source paths 都在 `batch-root` 内；
- temporal artifact 中窗口是 source-inclusive 且位于 clip bounds；
- 第二次同输入运行显示 producer `reused`；
- 空 candidate 显示 `skipped` 且无模型初始化；
- DeepReach 非空 candidate 显示 `blocked/adapter_missing`，不能期待真实 SAM3 结果。

并发数应在单 asset 成功、显存峰值和 I/O 时间确认后再提高；当前 `max-workers` 会让多个 asset 的 SAM3 实例并发，可能按 worker 数放大显存占用。

## 14. Exact files requiring manual review before merge

本次实现与状态合同：

- `configs/qc_acceptance.yaml`
- `configs/qc_acceptance/qc_acceptance_v2.1.0.yaml`
- `schemas/qc_acceptance_config.v2.schema.json`
- `schemas/asset_qc_report.v2.schema.json`
- `qc_common/contracts.py`
- `qc_common/module_registry.py`
- `qc_common/report_mutation.py`
- `qc_pipeline/artifacts.py`
- `qc_pipeline/default_registry.py`
- `qc_pipeline/orchestrator.py`
- `qc_pipeline/runners/precheck.py`
- `qc_pipeline/runners/video_quality.py`
- `qc_pipeline/runners/sam3_containment.py`
- `tools/run_qc_pipeline.py`

人工 QC、source mutation 与重复工具边界：

- `human_qc/semantic_service.py`
- `human_qc/hdf5_commit.py`
- `human_qc/timeline.py`
- `human_qc/warn_service.py`
- `human_qc/legacy_import.py`
- `human_qc/report_updates.py`
- `qc_reporting/projection.py`
- `qc_reporting/aggregate.py`
- `tools/build_manual_review_queue.py`
- `tools/serve_manual_review.py`
- `tools/build_batch_qc_ledger.py`
- `tools/build_acceptance_ledger.py`

供应商/manifest 集成前还必须人工确认：

- JDT/JD video-level manifest 的实际生成脚本和字段契约；
- DeepReach head video 选择、内参/外参映射、2D keypoint 字段和 frame alignment；
- `quality_hand` 的 supplier-specific verdict policy；
- 最终以 v2 report/qc_reporting 还是旧 ledger/workbook 作为交付事实来源。

## 15. Integration recommendation

当前修改应继续留在 `codex/human-qc-impl` 做云端小样本验证，不建议直接整体 merge 到 refresh。原因不是测试不足，而是仍有三个必须先落地的集成条件：JDT video-level manifest、DR head adapter、`quality_hand` supplier policy。

推荐顺序：

1. 先 commit/push 当前资源优化和安全状态语义；
2. 云端用少量 JDT canonical manifest 验证 producer artifact/cache/SAM3；
3. 单独实现或移植 JDT video-level manifest builder；
4. 以独立 adapter change 实现 DR head calibration/projection；
5. 明确人工 QC 和 ledger 的唯一交付 contract；
6. 再决定将同事分支整体 merge，还是只移植已验证的 producer/status commits。

在这些条件完成前，最接近的策略是“同事分支作为实现基础，但按模块验证后选择性集成”，不是无条件整体 merge。
