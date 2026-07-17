# Canonical Data 与 Publisher 定位修正实施计划

> **For agentic workers:** Execute with test-driven development. Preserve public QC contracts and backward compatibility while migrating the data/publisher modules.

**Goal:** 将 Canonical 定位修正为长期 Data Canonical，并将 Publisher 定义为基于 Raw、Canonical metadata、最终 QC 和可选 revision artifact 的 Curated LeRobot v3 发布器。

**Architecture:** Raw 始终不可变；Adapter 提供包含 Core、供应商扩展和批次属性的标准化视图；QC 只把派生结论写入现有 `asset_qc_report.v2`；Publisher 以 Raw/Canonical inventory 为完整 payload 来源并应用可选语义 revision。迁移采用兼容扩展：保留 `CanonicalQcEpisode` 名称别名和既有 CLI 默认行为，新增 `CanonicalDataEpisode` 术语、typed extension/batch contracts 及 optional revision。

**Tech Stack:** Python 3.11、frozen dataclasses、NumPy、h5py、PyArrow、JSON/sha256、pytest、Markdown。

## Global Constraints

- 不修改 `asset_qc_report.v2` schema、`quality_archive` 路径或 QC Gate 语义。
- 不修改配置快照、`asset_qc_report.v2` schema 或已归档 OpenSpec。
- 不 stage、commit 或 push。
- 保留当前实现标识 `CanonicalQcEpisode` / `canonical_qc_episode.v1` 的兼容事实。

---

### Task 1: 建立权威 P0 架构修正规范

**Files:**
- Create: `docs/superpowers/specs/2026-07-17-canonical-data-publisher-positioning-design.md`

- [x] 定义 Canonical Data 三层结构、batch metadata、Raw immutable 和四输入 Publisher。
- [x] 定义 revision artifact 的职责、最小字段和 fail-closed 边界。
- [x] 区分目标架构与当前 `CanonicalQcEpisode` 兼容实现。

### Task 2: 更新活跃合同与操作文档

**Files:**
- Modify: `docs/canonical-qc-required-fields-v1.md`
- Modify: `docs/canonical-qc-ingest-publish-runbook.md`
- Modify: `WORKFLOW_INTERFACE.md`
- Modify: `ACCEPTANCE.md`
- Modify: `docs/asset-qc-json-format.md`

- [x] 将 Canonical 的叙述从 QC 输入 DTO 改为长期标准化数据视图。
- [x] 写清 Core、supplier extensions/evidence、derived/QC outputs 的边界。
- [x] 写清 batch metadata/dataset attributes 与 `quality_archive` 的职责。
- [x] 将 Publisher 逻辑输入改为 Raw + metadata + final QC + optional revision。

### Task 3: 标注历史实现资料与代码 TODO

**Files:**
- Modify: `docs/superpowers/specs/2026-07-15-canonical-qc-ingest-publisher-design.md`
- Modify: `docs/superpowers/plans/2026-07-15-canonical-qc-ingest-publisher.md`
- Modify: `docs/canonical-qc-verification-report.md`

- [x] 标明 2026-07-15 文档是当前实现基线，架构定位已由 2026-07-17 文档修正。
- [x] 保留历史测试结论，同时列出 extension、batch metadata、Publisher full-payload 和 revision artifact 的未实现状态。

### Task 4: 扩展 Data Canonical 合同与 batch metadata

**Files:**
- Create: `canonical_qc/batch_metadata.py`
- Create: `tests/test_canonical_data_extensions.py`
- Modify: `canonical_qc/contracts.py`
- Modify: `canonical_qc/validation.py`
- Modify: `canonical_qc/provenance.py`
- Modify: `canonical_qc/__init__.py`
- Modify: `canonical_qc/workflow.py`

- [x] 先写失败测试：`CanonicalDataEpisode` 兼容别名、immutable `BatchMetadata`、extension field dtype/shape/time alignment、重复 published name 拒绝。
- [x] 实现 `BatchMetadata`、`SupplierExtensionField`、`SupplierExtensions`，并给旧 episode 构造提供空默认值。
- [x] 实现显式 batch manifest 读取、identity/hash 校验和 workflow overlay；不把 attributes 写进 QC verdict。
- [x] 扩展 data fingerprint，使 batch metadata/extension payload 变化可影响发布 identity。

### Task 5: Adapter 建立额外字段 inventory

**Files:**
- Modify: `canonical_qc/adapters/standard_hdf5.py`
- Modify: `canonical_qc/adapters/standard_lerobot.py`
- Modify: `tests/test_standard_hdf5_adapter.py`
- Modify: `tests/test_standard_lerobot_adapter.py`
- Modify: `tests/fixtures.py`

- [x] 先写失败测试：HDF5 未知 frame/episode dataset 与 LeRobot 额外 Parquet column 被读取且 Raw hash 不变。
- [x] HDF5 枚举未被 Core/Evidence 消费的 dataset；支持非 object NumPy dtype，object/vlen 等无损策略未登记时结构化拒绝。
- [x] LeRobot 保留已登记的额外 frame column 及其原始 published name/type/shape。
- [x] 额外字段命名冲突、非法 shape 或未登记 feature 必须 fail closed。

### Task 6: Publisher 全字段 preservation

**Files:**
- Modify: `lerobot_v3_publisher/layout.py`
- Modify: `lerobot_v3_publisher/prerequisites.py`
- Modify: `lerobot_v3_publisher/writer.py`
- Modify: `lerobot_v3_publisher/validation.py`
- Modify: `tests/test_lerobot_v3_writer.py`
- Modify: `tests/test_lerobot_v3_publish_prerequisites.py`

- [x] 先写失败测试：frame-aligned extension 成为 LeRobot Parquet feature；episode extension 与 batch attributes 进入版本化 semantic metadata；独立 validator 对值做 round-trip 比较。
- [x] Publisher 写入所有已登记 extension；unsupported/冲突字段拒绝，不静默 drop。
- [x] Release ID 绑定 source/data fingerprint，避免 Core 相同但额外字段不同的 release 冲突。
- [x] 验证 Raw 文件发布前后 SHA-256 不变。

### Task 7: Canonical revision artifact 与 Publisher 集成

**Files:**
- Create: `canonical_qc/revision.py`
- Create: `tests/test_canonical_revision_artifact.py`
- Modify: `lerobot_v3_publisher/contracts.py`
- Modify: `lerobot_v3_publisher/prerequisites.py`
- Modify: `lerobot_v3_publisher/workflow.py`
- Modify: `lerobot_v3_publisher/writer.py`
- Modify: `lerobot_v3_publisher/validation.py`
- Modify: `tools/publish_lerobot_v3.py`
- Modify: `tests/test_lerobot_v3_publish_prerequisites.py`
- Modify: `tests/test_canonical_qc_cli_e2e.py`

- [x] 先写失败测试：无 artifact 的非零 edit 仍拒绝；合法 task/subtask text 或共享边界 patch 生成新不可变 episode；before/hash/revision 不匹配拒绝。
- [x] 实现 `canonical_revision_artifact.v1` 读取、typed allowlist、CAS/fingerprint 校验和纯函数应用。
- [x] Publisher 接受 optional `--revision-artifact`，以应用后 episode 校验 final binding，并在 release manifest 绑定 artifact SHA-256。
- [x] 非语义字段 patch、Raw 回写、edit count 与 artifact 不一致全部 fail closed。

### Task 8: 全量回归、文档状态和 Git 验证

**Files:**
- Verify: all modified Markdown files

- [x] 将字段标准、runbook 和验证报告中的 Deferred 状态更新为实际实现范围。
- [x] 运行 Canonical/Adapter/Publisher/revision 定向测试，再运行全量 pytest。
- [x] 运行 `rg`，确认所有 `CanonicalQcEpisode` 出现均有兼容实现语境，且没有把 QC report 描述为训练 payload 来源。
- [x] 运行 `git diff --check` 和 Markdown 链接/结构检查。
- [x] 汇总修改文件、剩余 TODO 和 `git diff --stat`。
