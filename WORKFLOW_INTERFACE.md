# Marmalade QC Workflow Interface

本文档是当前仓库对外接口的主入口。`README.md`、`QUICKSTART.md`、`STRUCTURE.md` 和各实现说明都应服从这里的 workflow contract。

## 1. Workflow

外部编排可以按下面顺序执行：

```text
supplier data / raw video
-> precheck
-> optional SAM3 keypoint containment sidecar
-> annotation
-> annotation_verify
-> batch ledger / human review
```

注意：这个顺序是外部 workflow，不得硬编码进任何 root module。每个 root module 都必须能独立运行或被独立测试。

### 1.1 Canonical QC dataflow (v2)

The acceptance workflow that produces formal quality decisions is separate from
the legacy annotation outputs above. It is driven by
`configs/qc_acceptance.yaml` (`qc_acceptance_config_schema.v2`,
`qc_acceptance_v2.1.0`) and writes one
`<batch>/quality_archive/<asset_id>.json` per asset with
`schema_version=asset_qc_report.v2`.

```text
automatic QC Gate
  acceptance: hard fail -> stopped/fail; skip semantic/manual
  supplier_evaluation: hard fail -> record and continue
-> no candidate_issue_ids -> manual_review=not_required
-> candidate_issue_ids -> Warn manual_review=queued/in_progress/completed
-> manual_review=all_reviewed or not_required -> semantic_consistency (external)
-> manual_review=early_fail -> stopped; unviewed Warn stays unchanged
-> overall_decision=pass|fail
-> batch projections read quality_archive/*.json only
```

Warn 人工复核 is before `semantic_consistency`; the latter is an independent external
human calibration stage that a future model may implement through the same
interface. `not_required` and `all_reviewed` may enter semantic; `early_fail`
stops the asset and preserves unviewed Warn state. Runtime errors, evidence
failures, config drift and CAS conflicts set
`pipeline_state.status=error`, append `runtime_errors`, and leave
`overall_decision=null`; they are not quality fails. The only final decision
values are `pass`, `fail`, and `null` while incomplete.

`quality_archive/*.json` is the sole master source. Sidecars, overlays, CSV,
XLSX, Markdown, events and cache are derived evidence/reconciliation only
（sidecar 只作证据）; they cannot replace or overwrite a report verdict.

The operator endpoints are separate: Warn review is the automatic-lease service
on 8897 (`tools/serve_human_qc_workbench.py`, required `--batch-root`,
`--quality-archive`, `--reviewer`, optional `--sam3-model`); semantic calibration
is the independent 8898 service. Lease acquisition is automatic.

### 1.2 Canonical Data ingest and curated publish

标准 HDF5 与 LeRobot 使用同一显式入口：

```text
immutable Raw -> SourceAdapter -> Canonical Data view
                                   ├─ standardized Core -> CanonicalQcBridge -> QC
                                   ├─ supplier extensions/evidence
batch manifest / dataset attributes┘

QC -> final asset_qc_report.v2 -------------------------+
Raw + Canonical metadata -------------------------------+-> LeRobotV3Publisher
optional canonical revision artifact -------------------+        |
                                                                 v
                                                    immutable release/CURRENT
```

Canonical Data 是长期标准化数据视图，不是仅供 QC 使用的中间格式。Core 是跨供应商
统一字段；supplier extensions/evidence 保留当前 QC 不读取但训练、检索或预研可能
需要的字段；Derived/QC outputs 只进入 `asset_qc_report.v2`，不得污染 Canonical。
批次 manifest 的 `dataset_attributes` 用于 sensors/cameras/robot/annotation version/
language/modality 分类，不参与单资产 QC verdict。

`configs/canonical_qc.yaml` 是 ingest/publish 顶层 active config，并绑定 immutable
snapshot、QC config hash、Adapter/Publisher/toolchain/official-reader 版本。CLI 必须
显式接收 source format、source root、路径及 manifest 提供的
`asset_id/batch_id/supplier_id`；不得猜格式、根目录或失败资产 identity。Source Gate
在 Adapter 前建立资产报告，确定性合同失败仍进入唯一 QC JSON；可重试错误恢复后
以 CAS revision 继续并保留历史。自动流程在
`semantic_consistency` 返回 `awaiting_external`，不会伪造人工完成。训练从
`CURRENT.json -> releases/<release_id>` 读取，不扫描 staging 或按 mtime 选数据。

当前 path Publisher 支持 format-neutral `canonical_revision_artifact.v1`：只允许
task/description、subtask 双语文本和成对共享边界 patch，并以 CAS/fingerprint/
revision/edit-count 校验；无 artifact 的非零编辑仍 fail closed。

Publisher 的目标输入是 Raw source + Canonical metadata/field inventory + final QC
report + optional revision artifact。QC report 只作为 Gate 和审计绑定，不能被描述为
训练 payload 来源。当前实现通过 Raw 重建兼容 `CanonicalQcEpisode`，发布 Core、
`quality_hand`、已登记 supplier extensions 和 typed batch attributes，并在 manifest
绑定 data/artifact fingerprint。unsupported 类型、schema 漂移或尚无 Adapter 的格式
会 fail closed，不得静默 drop 后宣称“任意 Raw 全量发布”。

命令和故障恢复见 `docs/canonical-qc-ingest-publish-runbook.md`。

## 2. Module Ownership

| Module | Owner Scope | Loads Heavy Models | Main Inputs | Main Outputs |
| --- | --- | --- | --- | --- |
| `precheck/` | 数据可信度、HDF5 文本、`quality_hand`、骨骼点几何、基础画质、mask containment 消费端 | 否 | HDF5、optional frames、optional masks、camera intrinsics | `check_results.json`、`clip_aggregates.json`、parquet/csv fallback |
| `tools/sam3_keypoint_containment.py` | 云端验收 sidecar：抽样 mp4、调用 SAM3、投影 HDF5 手部骨骼点、计算 inside ratio | 是，SAM3 | HDF5 + mp4 + SAM3 model | `frame_keypoint_containment.json`、`clip_keypoint_containment.json` |
| `annotation/` | 视觉标注：discovery、SAM3 segmentation、DA3 depth、storage、annotation QC | 是，SAM3 / DA3 | LeRobot dataset / RGB frames / instruction | masks parquet、depth PNG+JSON、sampling manifest、QC images |
| `annotation_verify/` | 语义一致性验证契约；当前 VLM 是 stub | 目前否 | `ClipInputs` frames/instruction 或未来外部注入 | `check_results.json`、`clip_aggregates.json`、parquet/csv fallback |
| `qc_common/` | 跨模块共享契约和纯工具 | 否 | 无 runtime workflow input | `ClipInputs`、`CheckResult`、keypoint topology、registry、IO helpers |

## 3. Shared Keys

所有可对齐的结果都必须使用：

```text
(episode_idx, frame_idx)
```

约定：

- frame-level row：真实 `frame_idx`。
- clip-level summary row：`frame_idx = -1`。
- batch-level 统计：不能覆盖 frame/clip row，应另写 batch ledger。

## 4. Precheck Interface

入口：

```bash
python run_precheck.py configs/precheck_example.yaml
```

输入配置：

```yaml
output_dir: outputs/precheck_example
input_paths:
  - /path/to/supplier_hdf5_directory
  # or:
  # - /path/to/episode_000001.hdf5
enabled_checks:
  - text_integrity
  - quality_score
  - keypoint_missing
  - keypoint_temporal
  - skeleton_quality_score
  - composite_frame_verdict
```

`input_paths` 可以是文件或目录。precheck 输入层会自动识别：

- `.h5` / `.hdf5` 文件或目录：使用 supplier HDF5 adapter。
- LeRobot/parquet 风格目录：目前会识别并提示 adapter 未实现。
- CSV/video bundle：目前会识别并提示 adapter 未实现。

输出：

```text
<output_dir>/check_results.json
<output_dir>/clip_aggregates.json
<output_dir>/check_results.parquet or check_results.csv
```

`check_results.json` 记录结构：

```json
{
  "check": "skeleton_quality_score",
  "episode_idx": 0,
  "frame_idx": 123,
  "metrics": {
    "joint_angle_change_deg_max": 3.2,
    "rotation_delta_max": null,
    "joint_acceleration_m_s2_max": 8.4,
    "joint_displacement_m_max": 0.012,
    "skeleton_score": 1.0
  },
  "flag": null,
  "reason": "temporal skeleton geometry within thresholds"
}
```

## 5. SAM3 Keypoint Containment Sidecar

这个脚本是云端验收工具，不是 `precheck/checks`，因为它会加载 SAM3。

入口：

```bash
python tools/sam3_keypoint_containment.py \
  --hdf5-dir /path/to/hdf5 \
  --video-dir /path/to/mp4 \
  --sam3-model /path/to/sam3 \
  --output-dir outputs/sam3_keypoint_containment \
  --sample-fraction 0.10 \
  --projection-mode auto
```

输出：

```text
<output_dir>/frame_keypoint_containment.json
<output_dir>/clip_keypoint_containment.json
<output_dir>/run_manifest.json
```

核心 clip metric：

```text
clip_keypoint_inside_ratio = inside_keypoints / total_expected_keypoints
valid_projected_inside_ratio = inside_keypoints / valid_projected_keypoints
```

该 sidecar 可以用于准入口抽检；后续如果要接入 `precheck/mask_containment.py`，应通过 `ClipInputs.masks` 或外部 adapter 注入 mask，而不是让 precheck 直接 import SAM3。

## 6. Annotation Interface

入口：

```bash
python run_annotate.py configs/anygrasp_full.yaml
python run_annotate.py configs/seg_only.yaml --stage segmentation
python run_annotate.py configs/depth_only.yaml --stage depth
```

输出以 `(episode_idx, frame_idx)` 对齐：

```text
masks.parquet
depth/<camera>/episode_<idx>/frame_<idx>.png
depth/<camera>/episode_<idx>/frame_<idx>.json
sampling_manifest.parquet
qc/*.png
```

Annotation 不读取 precheck verdict，也不假设 precheck 已运行。

## 7. Annotation Verify Interface

当前是 semantic verification stub：

```bash
python run_annotation_verify.py configs/annotation_verify_example.yaml
```

输出与 `precheck` 保持同形：

```text
<output_dir>/check_results.json
<output_dir>/clip_aggregates.json
<output_dir>/check_results.parquet or check_results.csv
```

当前 `instruction_consistency` 是 clip-level row，使用 `frame_idx = -1`。

未来接 VLM 时仍需保持边界：只做 instruction/video semantic consistency，不做 signal-quality、骨骼点、mask containment 或 annotation 修复。

## 8. Batch Ledger Contract

批次台账应在外部聚合以下 JSON/parquet：

- `precheck/check_results.json`
- `precheck/clip_aggregates.json`
- `sam3_keypoint_containment/clip_keypoint_containment.json`
- annotation masks/depth manifests
- `annotation_verify/check_results.json`
- `annotation_verify/clip_aggregates.json`

建议输出字段：

```text
batch_id
episode_idx
asset_id / source_file
precheck_pass
skeleton_quality_pass_ratio
sam3_keypoint_inside_ratio
text_integrity_flag
quality_hand_pass_ratio
annotation_outputs_available
semantic_verify_flag
human_review_status
final_decision
```

## 9. Test Strategy

测试也按模块边界组织：

| Test Type | Allowed Location | Contract |
| --- | --- | --- |
| Module smoke test | `tests/` | 直接构造 `ClipInputs` 或模块 config，只验证单个 module 的 runner/check/contract。 |
| Temporary coupled workflow test | `tests/` or `tools/` | 可以按 `precheck -> sidecar -> annotation -> verify` 串联，但必须通过文件/config/JSON 输出连接，不得在 root modules 之间互相 import runtime internals。 |
| Cloud GPU containment test | `tools/sam3_keypoint_containment.py` | 允许加载 SAM3；输出 JSON 给台账或后续 adapter 消费。 |
| Production runner behavior | root module runners | 保持独立，不假设其他 module 已经运行。 |

临时耦合测试可以存在，但它是测试/验收脚本，不是架构约束。测试通过不代表可以把 workflow 顺序写进 `precheck/`、`annotation/` 或 `annotation_verify/`。

## 10. Boundary Rules

- `precheck/` 不加载 SAM3、DA3、VLM。
- `annotation/` 不 import `precheck/` 或 `annotation_verify/` runtime internals。
- `annotation_verify/` 不做 signal-quality checks。
- `qc_common/` 只放稳定契约和纯工具。
- Heavy model sidecars 可以放在 `tools/`，但必须明确标注为外部 workflow step。

## 11. Canonical QC projection and migration commands

```bash
python tools/build_manual_review_queue.py \
  --quality-archive sampled/XJGT_20260616/quality_archive \
  --output-dir sampled/XJGT_20260616/manual_review

python tools/build_qc_json_projection.py \
  --quality-archive sampled/XJGT_20260616/quality_archive \
  --output-dir sampled/XJGT_20260616/qc_projection \
  --cache-dir sampled/XJGT_20260616/qc_cache \
  --formats csv parquet xlsx markdown

python tools/build_batch_qc_ledger.py \
  --quality-archive sampled/XJGT_20260616/quality_archive \
  --output-dir sampled/XJGT_20260616/ledger \
  --formats csv parquet xlsx markdown

python tools/build_xjgt_acceptance_report.py \
  --quality-archive sampled/XJGT_20260616/quality_archive \
  --output-dir sampled/XJGT_20260616/xjgt_report
```

All formal commands validate each JSON before projection. Cache reuse requires a
source manifest matching relative report path, `report_revision` and SHA-256;
otherwise rebuild it. Legacy sidecar flags are explicit reconciliation inputs
only and cannot modify canonical rows.

Report writers use read → identity/config/profile/next-module check → module-owned
mutation → candidate/fail rebuild → revision + 1 → v2 schema → fsync/atomic
replace. A stale revision is a CAS error and must not silently overwrite a
concurrent writer. `asset_qc_report.v1` is read-only; the first v2 write uses
`migrate_v1_to_v2()` and a new v2 report. Rollback is read-only/sidecar based and
must not overwrite the master verdict.
