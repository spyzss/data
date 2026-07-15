---
change: resource-efficient-qc-pipeline
design-doc: docs/superpowers/specs/2026-07-15-resource-efficient-qc-pipeline-design.md
base-ref: bba9353
---

# 资源优先 QC Pipeline 实施计划

**Implementation status:** Completed and audited on 2026-07-15. Final verification:
`803 passed, 1 skipped`; see
`docs/reviews/2026-07-15-resource-efficient-qc-pipeline-audit.md` for residual risks.

> **Execution mode:** 当前会话 inline execution。每个行为先写失败测试，再做最小实现。仓库规则优先：未经用户明确要求，不 stage、commit 或 push。

**Goal:** 在保留单入口和五个独立 QC 结果的前提下，让同一 asset 的 precheck 源只加载一次、让 producer sidecar 可安全复用，并让 SAM3 直接消费本轮 temporal 产生的 canonical candidate artifact。

**Architecture:** `build_default_registry()` 为每个 `AssetContext` 创建一个有状态但不跨 asset 共享的 `PrecheckSession`；session 按 orchestrator 请求逐项执行，因此 acceptance 能在 fail 后真正停止，而 supplier evaluation 能继续完成五项。`ArtifactStore` 只管理显式文件、指纹和原子发布；video quality 与 SAM3 仍是独立 producer。v2 report 只适配已发布 sidecar，不覆盖 raw status。

**Tech Stack:** Python 3.11、dataclasses/typing、hashlib/json、pathlib/tempfile、PyYAML、jsonschema、pytest；现有 pandas/pyarrow、OpenCV、SAM3 接口保持不变。

## Global constraints

- `precheck/` 不得 import SAM3、DA3、Qwen/VLM 或 acceptance ledger。
- producer 之间只通过 manifest/JSON/CSV/Parquet sidecar 通信。
- JDT 继续直接读取 Parquet 2D keypoints；不得伪造 calibration。
- DeepReach head projection adapter 不在本次实现范围；其 SAM3 状态必须为 `adapter_missing`/`blocked`，不得 pass。
- manifest frame bounds 为 inclusive source coordinates；`AssetContext.source_range` 继续使用内部 half-open 表达，offset 只能执行一次。
- candidate 空列表必须落成存在的 JSON 文件；缺少 candidate 文件不是 no-candidates。
- candidate window 不得直接计作 rejected duration。
- v2.0 config 快照不可修改；行为变化发布 v2.1.0。
- 不提交生成物、视频、HDF5、Parquet 数据、模型或 credentials。

---

### Task 1: 发布资源执行 profile 与显式 incomplete 状态合同

**Files:**
- Create: `configs/qc_acceptance/qc_acceptance_v2.1.0.yaml`
- Modify: `configs/qc_acceptance.yaml`
- Modify: `schemas/qc_acceptance_config.v2.schema.json`
- Modify: `schemas/asset_qc_report.v2.schema.json`
- Modify: `qc_common/contracts.py`
- Modify: `qc_common/report_mutation.py`
- Modify: `qc_pipeline/orchestrator.py`
- Modify: `tests/test_qc_config_v2.py`
- Modify: `tests/test_asset_qc_schema_v2.py`
- Modify: `tests/test_qc_orchestrator.py`

**Contract:**

- `acceptance.runtime_error_action = stop_incomplete`。
- `supplier_evaluation.runtime_error_action = record_and_continue`。
- 未执行和不可执行状态使用 `not_run`、`input_missing`、`input_invalid`、`adapter_missing`、`blocked`、`runtime_error`；不得生成 `ModuleResult(verdict="pass")`。
- 自动模块全部遍历完成但存在 required incomplete 时，pipeline 状态为 `incomplete`、`overall_decision = null`。
- acceptance quality fail 仍为 `stopped/fail`，但后继状态改为 `not_run`，reason 为 `blocked_by_quality_fail:<module>`。

- [ ] **Step 1: 为 v2.1 config 快照和 v2.0 不变性添加失败测试**

```python
def test_active_config_publishes_resource_aware_v21() -> None:
    loaded = load_qc_acceptance_config()
    assert loaded.config_version == "qc_acceptance_v2.1.0"
    assert loaded.execution_profile("acceptance")["runtime_error_action"] == "stop_incomplete"
    assert loaded.execution_profile("supplier_evaluation")["runtime_error_action"] == "record_and_continue"

def test_v20_snapshot_hash_is_unchanged() -> None:
    assert sha256(Path("configs/qc_acceptance/qc_acceptance_v2.0.0.yaml").read_bytes()).hexdigest() == EXPECTED_V20_SHA
```

- [ ] **Step 2: 运行 config 测试并确认因 v2.1 缺失而失败**

Run: `.venv/bin/python -m pytest tests/test_qc_config_v2.py -q`

- [ ] **Step 3: 添加 profile/runtime-error 状态机失败测试**

覆盖：

1. acceptance 的 runner runtime error 停在当前 module，`overall_decision is None`；
2. supplier evaluation 的当前 module 记录 `runtime_error` 后继续调用下一 automatic module；
3. supplier evaluation 最终 required incomplete 时是 `pipeline_state.status == "incomplete"` 且 decision 为 null；
4. acceptance quality fail 的后继为 `not_run`，不是 pass 或 `skipped` quality result；
5. `adapter_missing`、`input_missing`、`blocked` 都能通过 schema。

- [ ] **Step 4: 运行状态机测试并确认 enum/schema/继续逻辑缺失**

Run: `.venv/bin/python -m pytest tests/test_asset_qc_schema_v2.py tests/test_qc_orchestrator.py -q`

- [ ] **Step 5: 发布 v2.1 config 并实现最小状态转换**

实现一个 report mutation helper，职责仅为：记录 incomplete 原因、更新 `execution.module_states[module]`、按 profile 决定 advance 或 stop。不要构造 module evaluation/metrics/issue block。

- [ ] **Step 6: 运行 Task 1 回归**

Run: `.venv/bin/python -m pytest tests/test_qc_config.py tests/test_qc_config_v2.py tests/test_asset_qc_schema_v2.py tests/test_report_mutation.py tests/test_qc_orchestrator.py -q`

Expected: PASS；v2.0 hash 不变，active config 与 v2.1 快照字节一致。

---

### Task 2: 建立 sidecar path、指纹与原子发布基础设施

**Files:**
- Create: `qc_pipeline/artifacts.py`
- Create: `tests/test_qc_pipeline_artifacts.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class ProducerArtifact:
    producer: str
    directory: Path
    required_files: tuple[str, ...]

def artifact_for(context: AssetContext, producer: str) -> ProducerArtifact: ...
def file_identity(path: Path, *, declared: Mapping[str, Any] | None = None) -> dict[str, Any]: ...
def config_fingerprint(config: LoadedQcConfig, modules: Sequence[str]) -> dict[str, Any]: ...
def build_run_fingerprint(...) -> dict[str, Any]: ...
def reusable_artifact(artifact, expected_fingerprint) -> bool: ...
@contextmanager
def staged_artifact(artifact) -> Iterator[Path]: ...
def promote_artifact(staging: Path, artifact: ProducerArtifact) -> None: ...
```

All hashes use canonical UTF-8 JSON (`sort_keys=True`, compact separators) and SHA-256. `run_config.json` stores schema version, producer outcome, fingerprint, created time, and elapsed seconds.

- [ ] **Step 1: 添加 artifact layout 与 source identity 失败测试**

验证：per-asset 路径固定为 `module_outputs/<asset>/...`；本地 identity 包含 relative path、size、mtime_ns；declared checksum/ETag 被保留；路径逃逸被拒绝。

- [ ] **Step 2: 添加 fingerprint 精确匹配和 required-file 校验失败测试**

验证任一字段改变、required file 缺失、run outcome 非 reusable、JSON 损坏时返回 false。

- [ ] **Step 3: 添加事务发布失败保护测试**

先发布 valid v1，再模拟 staging 内抛异常；断言已发布 bytes 不变且临时目录被清理。

- [ ] **Step 4: 运行并确认 module 缺失**

Run: `.venv/bin/python -m pytest tests/test_qc_pipeline_artifacts.py -q`

- [ ] **Step 5: 实现最小 artifact store**

只使用标准库；禁止在 helper 内加载 supplier 大文件或模型。发布时同一 filesystem 内 rename，替换前保留旧目录，成功后清理 backup；异常恢复旧目录。

- [ ] **Step 6: 运行 artifact 测试与静态边界检查**

Run: `.venv/bin/python -m pytest tests/test_qc_pipeline_artifacts.py -q`

Run: `python3 -m compileall qc_pipeline/artifacts.py`

---

### Task 3: 用每 asset 的 PrecheckSession 消除五次源加载

**Files:**
- Modify: `qc_pipeline/runners/precheck.py`
- Modify: `qc_pipeline/default_registry.py`
- Modify: `tests/test_qc_orchestrator.py`
- Create: `tests/test_qc_pipeline_precheck_session.py`

**Interfaces:**

```python
class PrecheckSession:
    def __init__(self, context: AssetContext, config: LoadedQcConfig): ...
    def runner_for(self, module: str) -> ModuleRunner: ...
    def run_module(self, module: str) -> ModuleResult: ...
```

Session belongs to one registry/context. It lazily loads the clip on the first requested precheck module, retains it through later precheck modules, accumulates raw `CheckResult` and candidates, and never pre-executes a later module.

- [ ] **Step 1: 添加 supplier-evaluation 单次加载失败测试**

Monkeypatch `_load_clip` 计数，顺序调用 registry 中五个 precheck runners，断言 loader 调用一次、五个 module results 分离且顺序正确。

- [ ] **Step 2: 添加 acceptance early-fail 不预执行失败测试**

第二项返回 fail 后由 orchestrator 停止；断言第三至第五 check factory/run 从未调用，同时 source loader 仍只有一次。

- [ ] **Step 3: 添加 context 隔离失败测试**

两个 asset registry 各自加载一次；对错误 context 调用仍拒绝，session 不共享 mutable results。

- [ ] **Step 4: 运行并确认当前实现加载五次**

Run: `.venv/bin/python -m pytest tests/test_qc_pipeline_precheck_session.py tests/test_qc_orchestrator.py -q`

- [ ] **Step 5: 实现 lazy shared session 并改 registry binding**

复用现有 `_load_clip`、`precheck_config_from_unified`、adapter 和 source-coordinate mapping。不要修改 `precheck/` 的模型边界。

- [ ] **Step 6: 运行 precheck adapter/orchestrator 回归**

Run: `.venv/bin/python -m pytest tests/test_qc_pipeline_precheck_session.py tests/test_precheck_qc_adapter.py tests/test_qc_orchestrator.py -q`

---

### Task 4: 持久化 canonical precheck artifact 并支持跨进程复用

**Files:**
- Modify: `qc_pipeline/runners/precheck.py`
- Modify: `qc_pipeline/adapters/precheck.py`
- Modify: `qc_pipeline/artifacts.py`
- Modify: `tests/test_qc_pipeline_precheck_session.py`
- Modify: `tests/test_precheck_qc_adapter.py`

**Contract:**

- `check_results.json` 和 `clip_aggregates.json` 记录已执行 checks。
- supplier-evaluation 完成五项后 artifact outcome 为 `completed`，可复用。
- acceptance early stop 的部分 artifact outcome 为 `partial`，其已执行结果可留作证据但不能冒充完整 reusable artifact。
- `candidate_windows.json` 在 temporal 成功后始终存在，空集为 `[]`。
- v2 evidence paths 指向真实 `module_outputs/<asset>/precheck/...` 文件。

- [ ] **Step 1: 添加 canonical files 与真实 evidence path 失败测试**

执行 temporal 后断言四个 JSON 文件存在，candidate 为 source coordinates，report evidence 相对 batch root 且文件存在。

- [ ] **Step 2: 添加 empty candidate 文件失败测试**

Temporal 无候选时断言 `candidate_windows.json == []`，不能缺文件。

- [ ] **Step 3: 添加 valid resume 零 producer 调用失败测试**

先完成五项并发布，再创建新 registry/session；断言 source loader 和 check runners 均不调用，五个 `ModuleResult` 可从 artifact 确定性重建。

- [ ] **Step 4: 添加 invalidation/partial artifact 失败测试**

改变 source size/mtime、source range、config hash、module version 时重算；partial artifact 不作为五项全量 cache hit。

- [ ] **Step 5: 运行并确认 canonical/cache 行为缺失**

Run: `.venv/bin/python -m pytest tests/test_qc_pipeline_precheck_session.py tests/test_precheck_qc_adapter.py -q`

- [ ] **Step 6: 实现 raw JSON 序列化/反序列化和 sidecar publish**

使用 `CheckResult.to_record()` 和显式 constructor 重建；所有 NumPy 值先通过现有 JSON-safe converter。candidate mapping 只执行一次。

- [ ] **Step 7: 运行 Task 4 与 standalone precheck 回归**

Run: `.venv/bin/python -m pytest tests/test_qc_pipeline_precheck_session.py tests/test_precheck_qc_adapter.py tests/test_manifest_precheck_runner.py -q`

---

### Task 5: SAM3 改读本轮 candidate artifact，并在空候选时零模型开销

**Files:**
- Modify: `qc_pipeline/runners/sam3_containment.py`
- Modify: `qc_pipeline/adapters/sam3_containment.py`
- Modify: `tools/run_qc_pipeline.py`
- Create: `tests/test_qc_pipeline_sam3_runner.py`
- Modify: `tests/test_qc_orchestrator.py`

**Contract:**

- Full pipeline ignores manifest `candidate_windows_path` and reads `artifact_for(context, "precheck") / candidate_windows.json`.
- Standalone `tools/run_manifest_sam3_containment.py` keeps explicit candidate argument unchanged.
- Empty current artifact returns `ModuleResult(module="sam3_containment", verdict="skipped", evaluation.reason="no_candidates")` before factory/model lookup.
- Unsupported supplier/adapter raises a typed prerequisite condition mapped to `adapter_missing`/`blocked`, not quality fail.

- [ ] **Step 1: 添加 stale manifest path 被忽略的失败测试**

给 manifest legacy file 写不同窗口，给 current artifact 写目标窗口；mock standalone producer，断言接收 current artifact 内容。

- [ ] **Step 2: 添加空候选不加载 segmenter/model 的失败测试**

Factory 若调用即抛错；断言返回 skipped/no_candidates 且 factory 调用为零。

- [ ] **Step 3: 添加候选校验失败测试**

覆盖 wrong asset、start > end、越出 inclusive clip bounds、重复 offset 风险；断言 `input_invalid`，不调用 SAM3。

- [ ] **Step 4: 添加 JDT/DeepReach supplier 边界测试**

JDT 保留 direct Parquet 2D path；DeepReach 在 adapter 未实现时明确 blocked/adapter-missing，不调用通用 JDT reshaper。

- [ ] **Step 5: 运行并确认当前 runner 仍要求 manifest candidate path**

Run: `.venv/bin/python -m pytest tests/test_qc_pipeline_sam3_runner.py -q`

- [ ] **Step 6: 实现 artifact handoff、empty fast path 与 typed prerequisite**

从 `_SOURCE_COLUMNS` 移除 full-pipeline candidate 依赖，但保留 legacy manifest metadata 供告警/审计；不要删除 standalone CLI 参数。

- [ ] **Step 7: 运行 SAM3 runner/adapter/standalone 回归**

Run: `.venv/bin/python -m pytest tests/test_qc_pipeline_sam3_runner.py tests/test_sam3_qc_adapter.py tests/test_manifest_sam3_containment_runner.py -q`

---

### Task 6: 为 video quality 和 SAM3 增加精确 cache/recompute

**Files:**
- Modify: `qc_pipeline/runners/video_quality.py`
- Modify: `qc_pipeline/runners/sam3_containment.py`
- Modify: `qc_pipeline/adapters/video_quality.py`
- Modify: `qc_pipeline/adapters/sam3_containment.py`
- Create: `tests/test_qc_pipeline_producer_cache.py`

**Contract:**

- video fingerprint 独立于 precheck-only threshold，包含 video/HDF5 identity、source range、video config、implementation version。
- SAM3 fingerprint 包含 candidate SHA、video/2D identity、model identity、queries 和 thresholds。
- Matching cache 不调用 analyzer/segmenter；stale cache 只重算该 producer。
- 失败重算不覆盖 prior valid artifact。

- [ ] **Step 1: 添加 video cache hit 与独立 invalidation 失败测试**

第二次运行 analyzer 调用为零；只改 precheck config 不 invalidate video；改 video threshold/range/file identity 会 invalidate。

- [ ] **Step 2: 添加 SAM3 candidate SHA/model/config invalidation 失败测试**

格式化不同但语义/bytes 不同的 candidate 文件按 exact artifact SHA 处理；任何 fingerprint 变化触发重算。

- [ ] **Step 3: 添加 failed recompute preserves old artifact 失败测试**

模拟 analyzer/segmenter 抛错，断言已发布 result/run_config bytes 完全不变。

- [ ] **Step 4: 运行并确认 cache 缺失**

Run: `.venv/bin/python -m pytest tests/test_qc_pipeline_producer_cache.py -q`

- [ ] **Step 5: 实现 producer-specific serialize/adapt/reuse**

Adapter 从 durable raw result 重建 `ModuleResult`；report runtime metadata 记录 `artifact_state = computed|reused`、elapsed seconds、fingerprint digest。

- [ ] **Step 6: 运行 producer 与 manifest 回归**

Run: `.venv/bin/python -m pytest tests/test_qc_pipeline_producer_cache.py tests/test_video_quality_qc_adapter.py tests/test_sam3_qc_adapter.py tests/test_manifest_video_quality_runner.py tests/test_manifest_sam3_containment_runner.py -q`

---

### Task 7: 完成 unified CLI 的 resume/force、计时和批次容错输出

**Files:**
- Modify: `tools/run_qc_pipeline.py`
- Modify: `qc_pipeline/context.py`
- Modify: `qc_pipeline/default_registry.py`
- Modify: `tests/test_qc_orchestrator.py`
- Create: `tests/test_run_qc_pipeline_cli.py`
- Modify: `README.md`

**CLI behavior:**

- `--resume/--no-resume` 控制 producer artifact reuse，不再把 `--no-resume` 解释为“只要 report 已存在就报错”。
- 为安全起见，不覆盖已有 report；force producer recompute 与 report restart 是不同概念。若需要重建 report，使用新的 output root，而不是删除旧 report。
- JSON summary 输出每 asset 的 status/revision/executed modules，以及 producer `computed/reused/skipped/blocked/failed` 计数和 elapsed seconds。
- Full pipeline manifest 不要求 candidate path。

- [ ] **Step 1: 添加 manifest/context output-root 与 legacy candidate 失败测试**

断言 canonical artifact root 可由 context 得到；legacy candidate 字段不会进入 SAM3 required source_files。

- [ ] **Step 2: 添加 CLI resume/force 与 summary 失败测试**

覆盖 resume cache hit、no-resume producer recompute、单 asset error 不阻止其他 asset summary、timing fields 为 JSON-safe 数值。

- [ ] **Step 3: 运行并确认当前 no-resume/summary 语义不符**

Run: `.venv/bin/python -m pytest tests/test_run_qc_pipeline_cli.py tests/test_qc_orchestrator.py -q`

- [ ] **Step 4: 实现最小 CLI plumbing 和 README 云端示例**

README 只描述经过测试的参数和 sidecar layout，不声称本地完成模型验证。

- [ ] **Step 5: 运行 CLI/orchestrator 回归**

Run: `.venv/bin/python -m pytest tests/test_run_qc_pipeline_cli.py tests/test_qc_orchestrator.py -q`

---

### Task 8: Ledger/manual review/schema 回归与完整架构审查

**Files:**
- Modify only if a failing compatibility test proves necessary:
  - `tools/build_batch_qc_ledger.py`
  - `tools/build_manual_review_queue.py`
  - `tools/build_video_review_clips.py`
  - `tools/build_acceptance_ledger.py`
- Create: `docs/reviews/2026-07-15-resource-efficient-qc-pipeline-audit.md`

- [ ] **Step 1: 运行人工 QC 与 ledger targeted suite**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_batch_qc_ledger.py \
  tests/test_manual_review_queue.py \
  tests/test_video_review_clips.py \
  tests/test_acceptance_ledger.py \
  tests/test_human_qc_report_schema.py \
  tests/test_human_qc_profile_routing.py \
  -q
```

- [ ] **Step 2: 对任何真实回归先添加精确测试，再做最小兼容修复**

重点守住 `review_id` 优先、exact fallback、多个 `affected_segments`、合并 overlap 后计时、candidate 不计 rejected duration、raw module status 不被 final verdict 覆盖。

- [ ] **Step 3: 运行标准及 manifest 验证**

Run:

```bash
python3 -m compileall precheck qc_common qc_pipeline tools acceptance_pull annotation_verify
.venv/bin/python -m pytest tests/test_qc_modules_smoke.py -q
.venv/bin/python -m pytest \
  tests/test_manifest_precheck_runner.py \
  tests/test_manifest_video_quality_runner.py \
  tests/test_manifest_sam3_containment_runner.py \
  -q
git diff --check
```

- [ ] **Step 4: 进行完整源码审查并写 audit**

审查报告必须包含：

1. 五项 precheck 的实际调用/加载次数与剩余计算成本；
2. acceptance 与 supplier evaluation 的真实控制流；
3. sidecar fingerprint、复用、失效与失败保护；
4. SAM3 current-candidate handoff 和空候选 fast path；
5. raw statuses、overall decision、blocked/not-run 语义；
6. human QC/head-tail/review-id/duration 行为是否变化；
7. 与现有 ledger/manual 工具的重叠或 schema 风险；
8. JDT/JD video-level manifest 的兼容点；
9. DeepReach head SAM3 仍需实现的 adapter/lineage；
10. 云端验证命令、计时字段、未在本地运行的模型测试。

- [ ] **Step 5: 最终工作区审计**

Run: `git status --short --branch`

Run: `git diff --stat`

Run: `git diff --check`

只报告 intentional files；不 stage、commit 或 push。

## Definition of done

- Supplier evaluation 的五个 precheck 模块对每个 asset 只调用一次 raw source loader。
- Acceptance 在 early fail 后不执行后续 checks，并把它们标为 `not_run`。
- Temporal 产出 durable current candidate file，SAM3 不再依赖 manifest legacy path。
- Empty candidates 不调用 segmenter factory、不要求 model path。
- Precheck/video/SAM3 matching sidecars 可跨进程复用，stale inputs 精确重算。
- Required incomplete 状态不能得到 pass；SAM3 blocked 不让整批自动 fail。
- Standalone manifest runners、manual review、ledger 和 schema 回归通过。
- 审查文档明确剩余 DR head adapter 风险和云端模型验证边界。
