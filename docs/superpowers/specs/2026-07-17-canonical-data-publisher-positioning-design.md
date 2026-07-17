---
role: architecture-correction
supersedes-positioning-in: docs/superpowers/specs/2026-07-15-canonical-qc-ingest-publisher-design.md
field-contract: docs/canonical-qc-required-fields-v1.md
---

# Canonical Data 与 Curated Publisher P0 架构修正

日期：2026-07-17
状态：已批准并实现首版迁移；unsupported supplier format/type 继续 fail closed

## 1. 修正范围

本修正只改变 Canonical 与 Publisher 的职责定位，不改变以下现有合同：

- `asset_qc_report.v2` 的字段、Gate、CAS 和最终判定语义；
- `<batch>/quality_archive/<asset_id>.json` 的主报告路径；
- 自动 QC、语义一致性和 Warn 人工复核的先后关系；
- Curated LeRobot v3 的原子 staging、验证、release 和 `CURRENT.json` 机制。

## 2. 数据生命周期

```text
immutable supplier raw source
        |
        +----------------------------+
        |                            |
        v                            v
Source Adapter                source provenance
        |
        v
Canonical Data view
  - standardized core
  - supplier extensions/evidence
  - batch metadata / dataset attributes
        |
        +------------------+
        |                  |
        v                  v
QC projection          publication metadata
        |                  |
        v                  |
asset_qc_report.v2         |
        |                  |
        +--------+---------+
                 |
      optional canonical revision artifact
                 |
                 v
Publisher(raw + canonical metadata + final QC + optional revision)
                 |
                 v
        Curated LeRobot v3 release
```

Raw source 是不可变事实源。Canonical 是 Adapter 提供的标准化、可查询视图，不是
为了 QC 临时裁剪出来的传输对象；QC 只是它的一个消费者。Publisher 不能把 QC JSON
或 QC 所需字段当作训练数据全集。

## 3. Canonical Data 的三层结构

### 3.1 Standardized core

跨供应商语义已经冻结的字段进入 Core，例如 identity/provenance、精确时间轴、视频、
手部 keypoints/validity、标定和任务/subtask。QC 与训练消费者都通过这些字段获得
一致语义。

### 3.2 Supplier extensions/evidence

供应商提供但尚未进入跨供应商 Core 的字段必须保留或显式登记，不能因为当前 QC
不读取而丢弃。典型字段包括 robot state/action、额外相机、depth、force/tactile、
joint rotation、音频、供应商模型输出和私有 metadata。

每个字段至少登记原始名称、类型/shape、单位/坐标系、时间对齐方式、来源路径和
preservation status。能无损写入 Curated LeRobot v3 的字段由 Publisher 保留；暂不
支持的字段必须在发布前形成机器可读的 unsupported-field 诊断，不能静默删除。

Supplier evidence（现有 `quality_hand`）属于这一层。它可以被 QC 引用，但不能覆盖
我方机器观测或成为 Core validity 的替代品。

### 3.3 Derived/QC outputs

freeze、blur、duplicate、semantic verdict、manual verdict、effective duration 等检测
结果只进入 `asset_qc_report.v2` 或可重建 evidence sidecar，不写回 Raw，也不污染
Canonical Data。`quality_archive/*.json` 仍是质量结论唯一事实源，但不是训练 payload
的事实源。

## 4. Batch metadata / dataset attributes

批次 manifest 应提供可索引的 `dataset_attributes`，至少覆盖：

- `sensors`、`cameras`、`camera_roles`；
- `robot_platform`、`control_mode`；
- `annotation_version`、`adapter_version`；
- `languages`、`modalities`；
- 坐标系、时间同步方式和供应商 schema 版本；
- 每类可选字段的 availability/coverage。

这些属性用于批次分类、检索和训练集组合，不参与单资产 QC verdict。单资产可以引用
批次级 immutable manifest，并在必要时声明受控 override；不得从目录名猜测属性。

## 5. Publisher 输入合同

Publisher 的逻辑输入固定为：

```text
raw source
+ Canonical metadata / field mapping / provenance
+ final asset_qc_report.v2
+ optional canonical revision artifact
```

各输入职责如下：

| 输入 | 职责 |
| --- | --- |
| Raw source | 提供完整原始训练 payload；只读，发布过程不得原地修改。 |
| Canonical metadata | 提供标准字段映射、扩展字段 inventory、batch attributes、时间对齐和 provenance。 |
| Final QC report | 只提供发布门禁、审计绑定、revision 和质量追踪，不提供训练数据。 |
| Revision artifact | 只覆盖经确认允许修订的语义/时间轴字段；不携带整份 Raw 副本。 |

Publisher 必须验证最终 QC Pass、source/revision/fingerprint 绑定、字段 preservation
结果和输出可回读性。QC 不检测的字段仍需保留；若无法保留，发布必须 fail closed 或
依据显式版本化策略隔离到 extension sidecar，禁止静默 drop。

## 6. 语义修订

语义一致性是首个允许产生受控修改的阶段。修改不得回写 HDF5、供应商 LeRobot 或
MP4，而应产生 format-neutral Canonical revision artifact。最小合同需要包含：

- `asset_id`、`parent_canonical_revision`、`canonical_revision`；
- source/semantic before fingerprint；
- 允许修改的 typed path 与 before/after 值；
- reviewer、时间、原因和关联 QC report revision；
- artifact 自身 hash、CAS 前置值和应用后 semantic fingerprint。

允许的首版 patch scope 仅包含 task/subtask 文本和已批准的 subtask 共享边界。其他
Raw/Canonical 字段默认不可修改。Publisher 应将 patch 应用到标准化视图后生成最终
训练数据，并在 release manifest 绑定 artifact hash。

## 7. 目录关系

现有兼容目录保持：

```text
<batch>/
  <supplier raw files and directories>   # immutable
  batch_manifest.json                     # batch metadata / dataset attributes
  quality_archive/
    <asset_id>.json                       # asset_qc_report.v2
  canonical_revisions/
    <asset_id>/<revision>.json            # canonical_revision_artifact.v1
```

`quality_archive/` 与 Raw 位于同一批次根下但生命周期独立。运行时不得把 QC JSON
写进源 HDF5/LeRobot 文件，也不得把训练 release 写回供应商 Raw 目录。

## 8. 当前实现兼容与迁移边界

当前代码保留 `CanonicalQcEpisode` / `canonical_qc_episode.v1` 作为兼容标识，并提供
`CanonicalDataEpisode` 架构别名。文档术语统一使用 Canonical Data / Canonical Data
view。本 change 已完成：

1. 引入 supplier extension inventory 和无损 passthrough/sidecar policy；
2. 引入 batch manifest 与 `dataset_attributes` 的 typed contract；
3. 把 `PublishRequest` 扩展为 raw + metadata + final QC + optional revision；
4. 实现 Canonical revision artifact、CAS、patch 应用和 manifest 绑定；
5. 增加“QC 未使用字段仍完整保留”的 HDF5/LeRobot round-trip 测试；
6. 用兼容别名引入 `CanonicalDataEpisode`；真正移除旧名只允许在新 major schema。

实现范围覆盖标准 HDF5 的已支持 dataset、标准 LeRobot 已登记且规则定长的列、typed
batch metadata，以及 task/subtask text/共享边界 revision。object/vlen、ragged/null、
schema 声明与物理 dtype/shape 不一致和尚无 Adapter 的供应商格式继续 fail closed；
无 artifact 的非零语义编辑仍以 `canonical_revision_artifact_required` 拒绝。

## 9. 验收标准

- 所有入口文档都把 Canonical 定义为长期标准化数据视图，而不是 QC 中间格式。
- 所有 Publisher 文档都明确四类逻辑输入，且 QC report 只作为 gate/audit。
- Raw immutable、`quality_archive` 路径和 `asset_qc_report.v2` 合同保持不变。
- Core、supplier extensions/evidence、derived/QC outputs 的边界无歧义。
- Batch metadata 可用于批次分类但不参与单资产质量判定。
- 当前实现限制和后续代码 change 明确列出，不用目标架构措辞替代实现证据。
