---
design-doc: docs/superpowers/specs/2026-07-15-canonical-qc-ingest-publisher-design.md
field-contract: docs/canonical-qc-required-fields-v1.md
base-ref: a77a8a4
---

# Canonical QC 接入与 LeRobot v3 发布实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Every production change follows superpowers:test-driven-development: observe the intended test fail before writing implementation. Do not run implementation tasks in parallel because all tasks share the Canonical contract.

**Goal:** 用两个严格 Adapter 把标准 HDF5/LeRobot 归一为同一 `CanonicalQcEpisode.v1`，接入现有 QC JSON 数据流，并在最终 Pass 后由我方 Publisher 原子发布经过独立验证的 Curated LeRobot v3。

**Architecture:** `canonical_qc` 是只读输入边界和不变量所有者；`CanonicalQcBridge` 只向现有 runner 投影视图；`qc_common.report_mutation` 继续独占 QC JSON；`lerobot_v3_publisher` 在独立 staging 中写产物、重新打开验证，最后才更新不可变 release 与 `CURRENT.json`。

**Tech Stack:** Python 3.11、dataclasses、NumPy、h5py、PyArrow、FFmpeg/ffprobe、jsonschema、pytest，以及在 Publisher 任务开始前冻结的官方 `lerobot` 版本。官方版本和 lock 必须进入依赖文件，不能仅依赖开发机临时安装。

## 全局约束

- 只在当前 `codex/human-qc-semantic-review` 分支工作，不切换或直接修改 main。
- 保留 `.codex/`、`.superpowers/` 和现有未跟踪 human plan，不纳入本计划提交。
- 源 HDF5、LeRobot、MP4 始终只读；测试必须比较修改前后 hash。
- Canonical 区间只有严格半开 `[start,end_exclusive)`；旧闭区间只在 Bridge 边界转换。
- `timestamps_ns` 是权威时间轴；不得从 FPS 或 float timestamp 重建。
- `quality_hand` 是可选 Evidence；缺失 clean skip，禁止把 validity 当 quality。
- Adapter 失败、QC quality fail、runtime error 和 Publisher error 必须区分。
- 每个任务先运行新增测试并记录目标失败，再实现最小代码，再运行相关回归，再提交。
- 每个实现任务完成后进行独立 spec review 与 code-quality review；有问题先修复再进入下一任务。

## Task 1: 定义不可变 Canonical 合同与严格 Validator

**Files:**
- Create: `canonical_qc/__init__.py`
- Create: `canonical_qc/contracts.py`
- Create: `canonical_qc/errors.py`
- Create: `canonical_qc/validation.py`
- Create: `canonical_qc/provenance.py`
- Create: `tests/test_canonical_qc_contracts.py`

**Interfaces:**

```python
def validate_episode(episode: CanonicalQcEpisode) -> None: ...
def semantic_fingerprint(episode: CanonicalQcEpisode) -> str: ...
def source_fingerprint(source_files, *, source_schema_version, adapter_id, adapter_version) -> str: ...
```

- [x] Step 1: 写合法最小 episode fixture、数组只读、shape/dtype、valid/NaN、严格时间戳、标定、subtask 连续性和 optional hand quality 测试。
- [x] Step 2: 运行 `.venv/bin/python -m pytest tests/test_canonical_qc_contracts.py -q`，确认因 `canonical_qc` 缺失而失败。
- [x] Step 3: 实现 frozen dataclasses、稳定诊断异常、共享 Validator 和两种 fingerprint。
- [x] Step 4: 重跑测试，并运行 `.venv/bin/python -m pytest tests/test_qc_contracts.py tests/test_qc_config_v2.py -q`。
- [x] Step 5: `git diff --check`，提交 `feat(canonical): define strict QC episode contract`。

## Task 2: 实现视频探测和权威时间轴对齐

**Files:**
- Create: `canonical_qc/video_probe.py`
- Create: `tests/test_canonical_video_probe.py`
- Modify: `canonical_qc/contracts.py`
- Modify: `canonical_qc/validation.py`

**Interfaces:**

```python
def probe_video(path: Path) -> ProbedVideo: ...
def validate_video_alignment(time_axis: TimeAxis, video: ProbedVideo, *, max_delta_ns: int) -> None: ...
```

- [x] Step 1: 用 tiny MP4/monkeypatched ffprobe 添加 frame count、尺寸、codec、逐帧 PTS、PTS 归零、数量不足、偏差超限和 ffprobe 缺失测试。
- [x] Step 2: 运行新增测试，确认函数缺失导致失败。
- [x] Step 3: 实现无 shell 插值的 `subprocess.run([...])`，解析 rational FPS 和 frame PTS；不使用 FPS 合成 PTS。
- [x] Step 4: 运行新增测试和 `tests/test_acceptance_video_quality.py` 的时间轴用例。
- [x] Step 5: 提交 `feat(canonical): validate video timestamps against canonical time`。

## Task 3: 实现 StandardHdf5Adapter

**Files:**
- Create: `canonical_qc/adapters/__init__.py`
- Create: `canonical_qc/adapters/base.py`
- Create: `canonical_qc/adapters/standard_hdf5.py`
- Create: `tests/test_standard_hdf5_adapter.py`
- Modify: `tests/fixtures.py`

**Interfaces:**

```python
class StandardHdf5Adapter:
    adapter_id = "standard_hdf5"
    adapter_version = "1.0.0"
    def inspect(self, source: Path) -> SourceInspection: ...
    def load(self, source: Path) -> CanonicalQcEpisode: ...
```

- [x] Step 1: 新建标准 HDF5 fixture，覆盖所有固定 paths、语义 JSON、可选/缺失 hand quality、外置 MP4 和源 hash 不变测试。
- [x] Step 2: 运行测试，确认 Adapter 缺失失败。
- [x] Step 3: 严格读取固定字段；禁止别名、默认 FPS、静默 shape 修正或 validity→quality。
- [x] Step 4: 增加缺 dataset、错误 dtype/shape、非法 status code、视频不匹配的诊断测试并通过。
- [x] Step 5: 运行 legacy HDF5 adapter 回归，提交 `feat(canonical): add standard HDF5 source adapter`。

## Task 4: 实现 StandardLeRobotAdapter 与跨格式等价性

**Files:**
- Create: `canonical_qc/adapters/standard_lerobot.py`
- Create: `tests/test_standard_lerobot_adapter.py`
- Create: `tests/test_canonical_adapter_equivalence.py`
- Modify: `canonical_qc/adapters/__init__.py`
- Modify: `tests/fixtures.py`

**Interfaces:**

```python
class StandardLeRobotAdapter:
    adapter_id = "standard_lerobot"
    adapter_version = "1.0.0"
    def load(self, source: Path, *, episode_index: int | None = None) -> CanonicalQcEpisode: ...
```

- [x] Step 1: 写固定 LeRobot fixture 和目标失败测试，要求 `meta/episodes`、data Parquet、`episode_semantics.jsonl`、外置 main MP4 与精确 `timestamp_ns`。
- [x] Step 2: 运行新增测试，确认 Adapter 缺失失败。
- [x] Step 3: 实现受支持布局识别、唯一 episode 选择、固定 feature 解码和共享 Validator。
- [x] Step 4: 写同一逻辑 episode 的 HDF5/LeRobot 对照，断言除 provenance/source_format 外字段相同且 semantic fingerprint 相同。
- [x] Step 5: 增加缺 `timestamp_ns`、float timestamp 漂移、episode selector 歧义和目录逃逸测试；运行 annotation reader 回归。
- [x] Step 6: 提交 `feat(canonical): normalize standard LeRobot inputs`。

## Task 5: 实现 CanonicalQcBridge 并接入现有 runner

**Files:**
- Create: `canonical_qc/bridge.py`
- Create: `tests/test_canonical_qc_bridge.py`
- Modify: `qc_pipeline/runners/precheck.py`
- Modify: `qc_pipeline/runners/video_quality.py`
- Modify: `qc_pipeline/runners/sam3_containment.py`
- Modify: `tools/run_qc_pipeline.py`
- Modify: `tests/test_qc_pipeline_profiles_e2e.py`

**Interfaces:**

```python
class CanonicalQcBridge:
    def asset_context(...) -> AssetContext: ...
    def clip_inputs(...) -> ClipInputs: ...
    def video_path(...) -> Path: ...
    def semantic_payload(...) -> Mapping[str, Any]: ...
```

- [x] Step 1: 添加 bridge 映射测试：21 点顺序、数值不变、半开 slice、invalid NaN、video path 和 semantics。
- [x] Step 2: 确认测试因 Bridge 缺失失败。
- [x] Step 3: 实现只读视图；`quality_hand` 不进入 legacy machine validity，新增 `supplier_hand_quality_status` 只读属性。
- [x] Step 4: 让三个 runner 在存在 `canonical_episode` 时优先使用 Bridge；无 Canonical 时 legacy 路径行为不变。
- [x] Step 5: 添加标准 HDF5 和 LeRobot 各自进入现有 orchestrator 的 E2E，断言同一机器结果、相同 Gate 和 QC JSON CAS revision。
- [x] Step 6: 运行 QC orchestrator、precheck、video、SAM3 回归；提交 `feat(qc): bridge canonical episodes into existing runners`。

## Task 6: 标准化 optional hand quality Evidence 与分歧统计

**Files:**
- Create: `canonical_qc/hand_quality.py`
- Create: `configs/qc_acceptance/qc_acceptance_v2.1.0.yaml`
- Create: `tests/test_supplier_hand_quality_evidence.py`
- Modify: `canonical_qc/bridge.py`
- Modify: `qc_common/types.py`
- Modify: `qc_pipeline/adapters/precheck.py`
- Modify: `qc_pipeline/adapters/sam3_containment.py`
- Modify: `qc_pipeline/runners/precheck.py`
- Modify: `qc_pipeline/runners/sam3_containment.py`
- Modify: `precheck/checks/quality_score.py`
- Modify: `precheck/checks/keypoint_missing.py`
- Modify: `configs/qc_acceptance.yaml`
- Modify: `ACCEPTANCE.md`
- Modify: `WORKFLOW_INTERFACE.md`
- Modify: `docs/PRD-qc-gated-json.md`
- Modify: `docs/PRD-qc-unified-config.md`
- Modify: `docs/asset-qc-json-format.md`
- Modify: `docs/canonical-qc-required-fields-v1.md`
- Modify: `docs/superpowers/specs/2026-07-15-canonical-qc-ingest-publisher-design.md`
- Modify: `tests/test_acceptance_video_quality.py`
- Modify: `tests/test_qc_config.py`
- Modify: `tests/test_qc_config_v2.py`
- Modify: `tests/test_qc_docs_contract.py`

**Interfaces:**

```python
def compare_supplier_and_machine(status, machine_status) -> SupplierAgreementResult: ...
```

- [x] Step 1: 写 `unknown` 不 fail、不计一致率；good+machine fail 生成 `supplier_mask_disagreement` warn；bad+machine pass 记 false positive 的测试。
- [x] Step 2: 运行测试并确认旧裸数值逻辑不符合预期。
- [x] Step 3: 删除 Canonical 路径中的 `==0`/`<0.5` 推断，保留 legacy 行为所需兼容适配；统一消费 enum status。
- [x] Step 4: 配置不再把 `[0,1]` 声明成全局供应商语义；保留 `v2.0.0` 不变，发布 `qc_acceptance_v2.1.0` immutable snapshot 并让 active 与新快照逐字节一致。
- [x] Step 5: 运行所有 precheck/QC config/report 测试，提交 `feat(qc): treat supplier hand quality as optional evidence`。

## Task 7: 实现 Publisher 合同、布局和前置条件

**Files:**
- Create: `lerobot_v3_publisher/__init__.py`
- Create: `lerobot_v3_publisher/contracts.py`
- Create: `lerobot_v3_publisher/layout.py`
- Create: `lerobot_v3_publisher/prerequisites.py`
- Create: `tests/test_lerobot_v3_publish_prerequisites.py`

**Interfaces:**

```python
def validate_publish_request(request: PublishRequest) -> PublishPlan: ...
def release_id_for(request: PublishRequest) -> str: ...
```

- [ ] Step 1: 写 pass/completed、人工完成或 not_required、source fingerprint、report revision、路径安全和 deterministic release ID 测试。
- [ ] Step 2: 运行测试并确认 publisher package 缺失失败。
- [ ] Step 3: 实现不可变请求/计划/结果/manifest 合同和前置校验，不修改 QC report。
- [ ] Step 4: 增加 fail/null/error/stale/source drift 拒绝测试并通过。
- [ ] Step 5: 提交 `feat(publisher): define gated LeRobot v3 release contract`。

## Task 8: 写入 Curated LeRobot v3 staging

**Files:**
- Create: `lerobot_v3_publisher/writer.py`
- Create: `tests/test_lerobot_v3_writer.py`
- Modify: `lerobot_v3_publisher/layout.py`

**Interfaces:**

```python
def write_staging(plan: PublishPlan, staging_root: Path) -> StagedRelease: ...
```

- [ ] Step 1: 写目标目录结构、逐帧 Parquet、episode metadata、semantics、video、manifest 和 checksums 的失败测试。
- [ ] Step 2: 实现固定 schema writer；float timestamp 只能从 `timestamps_ns` 派生，subtask index 只能从半开边界展开。
- [ ] Step 3: 证明输入为 LeRobot v3 时仍重写 meta/Parquet，只有 hash 一致的 MP4 可复用。
- [ ] Step 4: 证明 source 文件 hash 全部不变、staging 不含绝对路径或未登记文件。
- [ ] Step 5: 提交 `feat(publisher): write curated LeRobot v3 staging release`。

## Task 9: 独立回读验证与原子发布

**Files:**
- Create: `lerobot_v3_publisher/validation.py`
- Create: `lerobot_v3_publisher/publisher.py`
- Create: `tests/test_lerobot_v3_publisher_atomicity.py`
- Modify: `annotation/lerobot_v3_dataset.py`

**Interfaces:**

```python
def validate_staged_release(staged: StagedRelease, expected: CanonicalQcEpisode) -> ValidationReport: ...
def publish(request: PublishRequest) -> PublishResult: ...
```

- [ ] Step 1: 写独立 reopen 的 row count/index/timestamp_ns/float timestamp/arrays/subtask/video PTS/checksum 测试。
- [ ] Step 2: 运行测试并确认 validator/publish 缺失失败。
- [ ] Step 3: 实现不共享 writer 内存状态的 reader/validator。
- [ ] Step 4: 冻结官方 `lerobot` 版本，使用官方 `LeRobotDataset` 回读 staging；本地 reader 成功但官方 reader 失败时门禁必须失败。
- [ ] Step 5: 实现 `.staging`、fsync、不可变 release rename、最后原子更新 `CURRENT.json` 和同请求幂等返回。
- [ ] Step 6: 在 writer、validator、rename、CURRENT replace 各故障点注入异常，断言旧 CURRENT 和旧 release 字节不变，partial release 不可见。
- [ ] Step 7: 运行官方 reader 与 annotation reader 回归，提交 `feat(publisher): validate and atomically commit LeRobot v3 releases`。

## Task 10: CLI、版本化配置与端到端验收

**Files:**
- Create: `configs/canonical_qc/canonical_qc_v1.0.0.yaml`
- Create: `schemas/canonical_qc_config.v1.schema.json`
- Create: `tools/run_canonical_qc.py`
- Create: `tools/publish_lerobot_v3.py`
- Create: `tests/test_canonical_qc_cli_e2e.py`
- Create: `docs/canonical-qc-ingest-publish-runbook.md`
- Modify: `WORKFLOW_INTERFACE.md`
- Modify: `ACCEPTANCE.md`

**Interfaces:**

```text
python tools/run_canonical_qc.py --source ... --quality-archive ... --profile acceptance
python tools/publish_lerobot_v3.py --source ... --qc-report ... --release-root ...
```

- [ ] Step 1: 写 HDF5 与 LeRobot 两条 CLI E2E，覆盖 ingest、QC resume、人工完成 fixture、publish、独立回读。
- [ ] Step 2: 写 JSON error 输出测试，区分输入 fail、runtime error、publish prerequisite 和 validation failure。
- [ ] Step 3: 实现薄 CLI、配置加载和文档；业务逻辑只调用 package API。
- [ ] Step 4: 运行 `.venv/bin/python -m pytest tests/test_canonical_qc_cli_e2e.py -q`，并执行两种输入的等价发布 smoke test。
- [ ] Step 5: 提交 `feat: run canonical QC and publish curated LeRobot v3`。

## Task 11: Thorough 全量验证与设计对账

**Files:**
- Modify: `docs/canonical-qc-required-fields-v1.md`
- Modify: `docs/superpowers/plans/2026-07-15-canonical-qc-ingest-publisher.md`
- Create: `docs/canonical-qc-verification-report.md`

- [ ] Step 1: 运行 `.venv/bin/python -m pytest -q`，必须全量通过。
- [ ] Step 2: 运行 `.venv/bin/python -m compileall canonical_qc lerobot_v3_publisher qc_pipeline tools`。
- [ ] Step 3: 运行 `git diff --check`、schema/config 校验、HDF5/LeRobot semantic fingerprint 等价测试和 Publisher 故障注入测试。
- [ ] Step 4: 对照字段合同逐项记录 implemented/tested/deferred；首版边界之外不得伪装支持。
- [ ] Step 5: 由独立 reviewer 做跨任务 spec compliance 与代码质量审查；修复所有 Blocker/Important 后复跑全量测试。
- [ ] Step 6: 更新计划 checkbox 和验证报告，提交 `test: complete canonical QC publisher verification`。

## 最终验收命令

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall canonical_qc lerobot_v3_publisher qc_pipeline tools
git diff --check
git status --short --branch
```

期望：全部测试通过；两种标准输入归一语义一致；Source Core/视频/时间戳错误 fail closed；可选 hand quality 缺失 clean skip；Publisher 故障不会改变已发布训练 release；工作区只保留用户原有未跟踪文件和本计划明确改动。
