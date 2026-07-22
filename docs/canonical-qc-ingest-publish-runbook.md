# Canonical Data Ingest, QC and LeRobot v3 Publish Runbook

本手册覆盖标准 HDF5 / LeRobot 输入、可恢复自动 QC、人工阶段边界，以及 Curated
LeRobot v3 原子发布。顶层 active 配置为 `configs/canonical_qc.yaml`，必须与
`configs/canonical_qc/canonical_qc_v1.1.0.yaml` 逐字节一致；它绑定 QC config
版本/hash、Adapter 版本、时间戳 tolerance、Publisher/toolchain 和官方 reader 合同。
`--config` 只接受仓库中已登记且逐字节匹配的不可变快照；例如
`canonical_qc_v1.1.1.yaml` 是 timestamp tolerance 为 0 的严格诊断版本，不能用临时
YAML 绕过 active 合同。

Canonical 是供应商 Raw 经过 Adapter 标准化后的长期数据视图；QC 只是消费者。
`quality_archive/*.json` 是质量结论事实源，但不是训练 payload。Publisher 的目标
逻辑输入为 Raw source + Canonical metadata + final QC report + optional revision
artifact。当前 CLI 仍通过 `CanonicalQcEpisode` 兼容对象承载 v1 Core，通用额外字段
以 typed inventory 保留，并支持 batch metadata 和受控非零 revision，见第 3.1 节。

## 1. 路径与输入原则

- 必须显式给出 `--source-format hdf5|lerobot`，禁止通过后缀猜格式。
- 必须显式给出 `--source-root` / `--canonical-source-root`；source 必须位于其中。
- QC 的 source root、quality archive 必须位于显式 `--batch-root` 内。
- LeRobot 多 episode 数据集必须给 `--episode-index`；单 episode 可省略。
- HDF5 禁止传 `--episode-index`。
- 内部时间段全部是半开区间 `[start_frame,end_frame_exclusive)`。
- Raw source 必须只读；任何 QC、人工或 Publisher 步骤都不得原地修改 HDF5、
  LeRobot、MP4 或供应商 metadata。
- `quality_archive/` 位于同一批次根、与 Raw 并列保存；它只承载
  `asset_qc_report.v2`，不得嵌入源容器。
- 批次 manifest 应使用 `canonical_batch_metadata.v1`，提供 sensors/cameras/robot/
  annotation version/language/modality 等 `dataset_attributes`；CLI 用
  `--batch-metadata` 显式绑定，跨批次检索索引另行实现。

推荐布局：

```text
<batch>/
  <supplier raw files and directories>   # immutable
  batch_manifest.json                     # canonical_batch_metadata.v1
  quality_archive/
    <asset_id>.json                       # asset_qc_report.v2
  canonical_revisions/
    <asset_id>/<revision>.json            # canonical_revision_artifact.v1
```

## 2. Ingest 与自动 QC

HDF5：

```bash
python tools/run_canonical_qc.py \
  --source /data/batch/asset-001 \
  --source-format hdf5 \
  --source-root /data/batch/asset-001 \
  --batch-root /data/batch \
  --quality-archive /data/batch/quality_archive \
  --profile acceptance \
  --asset-id asset-001 \
  --batch-id batch-20260716 \
  --supplier-id supplier-001 \
  --batch-metadata /data/batch/batch_manifest.json
```

LeRobot：

```bash
python tools/run_canonical_qc.py \
  --source /data/batch/supplier-lerobot \
  --source-format lerobot \
  --source-root /data/batch/supplier-lerobot \
  --episode-index 7 \
  --batch-root /data/batch \
  --quality-archive /data/batch/quality_archive \
  --profile supplier_evaluation \
  --asset-id asset-007 \
  --batch-id batch-20260716 \
  --supplier-id supplier-001 \
  --batch-metadata /data/batch/batch_manifest.json
```

默认 `--resume`。报告已是 terminal 或 `awaiting_external` 时再次执行是字节幂等；
`--no-resume` 只允许全新报告。`--dry-run` 只执行配置、路径、Adapter、Canonical
合同和时间轴验证，不创建 report。

`asset_id/batch_id/supplier_id` 必须来自批次 manifest 或编排调用方，禁止从路径或
损坏的供应商内容猜测。Source Gate 在 Adapter 前先确定报告路径：成功时先原子写
Gate Pass revision，再继续自动 QC；确定性合同失败写 `stopped/fail` 和 fail issue；
临时 I/O 写 `error/null`。临时错误恢复后同一报告 CAS 前进，当前
`runtime_errors` 清空，旧错误保留在 `execution.runtime_error_history`。
`canonical_qc_v1.0.x` 快照缺少 Source Gate rule registry，只作为历史配置保留，
不可执行；运行时必须使用 `v1.1.0` 或更高的已登记快照，不能为旧配置静默补规则。

输出报告固定为：

```text
<quality-archive>/<asset_id>.json
```

自动阶段遇到 `semantic_consistency` 会以 exit 0 返回
`state=awaiting_external, overall_decision=null`。当前生产仓库尚未提供 external stage
完成 mutation API；人工工作台 change 应通过 CAS 接口推进。CLI 不伪造人工完成，
测试中的 completed report 只是明确 fixture。

Publisher 路径入口从显式 Raw source 重新加载 Canonical Data，并可叠加 batch manifest
与 format-neutral revision artifact。QC JSON 只用于 Gate/revision/fingerprint 绑定，
不是 LeRobot 数据来源。如果任一 edit count 大于 0 但没有 `--revision-artifact`，仍以
`canonical_revision_artifact_required` fail closed，绝不发布源文件中的旧文本。

## 3. 发布前置

目标逻辑请求包含：

```text
raw source
+ Canonical metadata / supplier extension inventory / batch attributes
+ final asset_qc_report.v2
+ optional canonical revision artifact
```

Raw 提供完整训练 payload；Canonical metadata 提供标准字段映射；QC report 只做
发布门禁和追踪；revision artifact 只覆盖允许修改的语义/时间轴字段。

发布必须读取最终报告中的 `report_revision` 和
`canonical_binding.canonical_revision`；CLI 不允许调用者覆盖它们。要求包括：

- pipeline completed 且 `overall_decision=pass`；
- 报告含 `source_gate` 时，Gate 必须为 completed/pass/continue；任何 fail 或
  runtime_error 即使被篡改为顶层 pass 也拒绝发布；
- 语义校准 completed；人工 Warn review 为 completed 或 not_required；
- report、source、semantic/source fingerprint 与当前 Canonical episode 一致；
- source 文件未漂移。

先做无写入检查：

```bash
python tools/publish_lerobot_v3.py \
  --source /data/batch/asset-001 \
  --source-format hdf5 \
  --canonical-source-root /data/batch/asset-001 \
  --qc-report /data/batch/quality_archive/asset-001.json \
  --batch-metadata /data/batch/batch_manifest.json \
  --release-root /training/curated-egodata \
  --dry-run
```

零编辑资产不得提供 `--revision-artifact`；非零编辑资产必须提供。Artifact 只允许
task/description、subtask 双语文本和成对共享边界 patch，并绑定 before/after semantic
fingerprint、parent/final canonical revision、QC report revision、edit count 和文件 hash。
共享边界必须同时提交相邻 segment 的 end/start 两条 patch，但只计为一次
`timeline_edit_count`。
非零编辑发布在上述命令中追加：

```text
--revision-artifact /data/batch/canonical_revisions/asset-001/3.json
```

正式发布去掉 `--dry-run`。Publisher 会 staging、独立回读、冻结版
`LeRobotDataset==0.6.0` 首末帧验证、fsync、no-replace rename，最后原子更新
`CURRENT.json`。失败不改变旧 CURRENT 和旧 release；同一请求重试返回
`already_published`。

### 3.1 当前实现边界与后续代码项

当前 `PublishRequest` 绑定 Raw-derived episode、canonical source root、final QC、typed
batch metadata 和 optional revision artifact。已验证 Core、`quality_hand` Evidence、
标准 HDF5/LeRobot 已登记 extensions、主视频及非零语义修订；release identity/manifest
绑定完整 data fingerprint 和 artifact SHA-256。

剩余边界：object/vlen、ragged/null、物理 dtype/shape 与 `info.features` 不一致、多媒体
extension 或尚无 Adapter 的 MCAP/NPZ 等格式会结构化拒绝；需新增显式 supplier
profile/policy，禁止删字段后重试。跨批次检索索引与人工 revision artifact 生成工作台
也不在本 Publisher change 内。

## 4. 训练读取

训练程序不依赖固定数据路径。先读取：

```text
<release-root>/CURRENT.json
```

再按其中 `release_id` 打开：

```text
<release-root>/releases/<release_id>/
```

训练端不得扫描 `.staging`，也不得猜“最新修改时间”。HDF5 与供应商 LeRobot 都由
我方 Publisher 重写成同一 Curated LeRobot v3 schema。目标状态下，标准 Core 语义
一致，供应商扩展按 inventory 保留；不同 source provenance 或 extension payload
必须产生不同 release identity。当前实现只对 v1 Core/`quality_hand` 提供该保证。

## 5. JSON 与退出码

每次命令只输出一个 compact JSON。成功/合法 quality state 写 stdout；参数、输入、
运行和发布异常写 stderr，均无 traceback。

| Exit | category | 含义 |
| ---: | --- | --- |
| 0 | success | pass、awaiting_external、dry-run validated、published/already_published |
| 2 | input_contract / quality_fail | 输入合同失败，或 terminal 机器质量 fail；看 category 区分 |
| 3 | qc_runtime / publish_runtime | QC 运行或 staging 运行错误 |
| 4 | publish_prerequisite | 最终报告或 source 不满足发布条件 |
| 5 | publish_validation | 独立/官方 reader 验证失败 |
| 6 | commit_conflict | 同 release ID 已存在不同不可变内容 |

批处理 Gate 同时检查 exit code、`category`、`state`、`overall_decision` 和
`report_path`；报告仍是质量判定权威。

## 6. 故障处理

- input/timebase fail：修正供应商字段或显式 episode selector，不降低 tolerance 猜测。
- `awaiting_external`：先在人工质检服务完成 Warn review，再在独立语义工作台完成
  semantic calibration；每阶段都从同一 report resume。
- runtime：保留 report/runtime_errors，修复依赖后 `--resume`。
- `source_integrity_error`：按 JSON 的 `retryable` 区分；临时 I/O 可重试，确定性的
  缺失、格式损坏、size/hash 漂移必须修复输入，禁止无限重试。
- prerequisite：核对 final report、revision、binding 和 source hash。
- `canonical_revision_artifact_required`：该资产有人工语义编辑；等待人工模块输出
  format-neutral working revision artifact，不能回写或猜测源 HDF5/LeRobot。
- extension preservation：若 inventory 中字段没有已登记的 LeRobot feature/sidecar
  policy，禁止发布并补齐 Adapter/Publisher mapping；不得删除字段后重试。
- validation：不要更新 CURRENT；检查冻结 validator 环境与 staged artifact。
- commit conflict：保留现有 release，调查 release ID/不可变内容冲突，不覆盖目录。

原始供应商 HDF5/LeRobot 不会被 CLI 原地修改；长期归档与 Curated training release
是不同生命周期。
