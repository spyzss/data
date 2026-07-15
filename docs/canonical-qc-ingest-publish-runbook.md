# Canonical QC Ingest and LeRobot v3 Publish Runbook

本手册覆盖标准 HDF5 / LeRobot 输入、可恢复自动 QC、人工阶段边界，以及 Curated
LeRobot v3 原子发布。顶层 active 配置为 `configs/canonical_qc.yaml`，必须与
`configs/canonical_qc/canonical_qc_v1.0.0.yaml` 逐字节一致；它绑定 QC config
版本/hash、Adapter 版本、时间戳 tolerance、Publisher/toolchain 和官方 reader 合同。
`--config` 只接受仓库中已登记且逐字节匹配的不可变快照；例如
`canonical_qc_v1.0.1.yaml` 是 timestamp tolerance 为 0 的严格诊断版本，不能用临时
YAML 绕过 active 合同。

## 1. 路径与输入原则

- 必须显式给出 `--source-format hdf5|lerobot`，禁止通过后缀猜格式。
- 必须显式给出 `--source-root` / `--canonical-source-root`；source 必须位于其中。
- QC 的 source root、quality archive 必须位于显式 `--batch-root` 内。
- LeRobot 多 episode 数据集必须给 `--episode-index`；单 episode 可省略。
- HDF5 禁止传 `--episode-index`。
- 内部时间段全部是半开区间 `[start_frame,end_frame_exclusive)`。

## 2. Ingest 与自动 QC

HDF5：

```bash
python tools/run_canonical_qc.py \
  --source /data/batch/asset-001 \
  --source-format hdf5 \
  --source-root /data/batch/asset-001 \
  --batch-root /data/batch \
  --quality-archive /data/batch/quality_archive \
  --profile acceptance
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
  --profile supplier_evaluation
```

默认 `--resume`。报告已是 terminal 或 `awaiting_external` 时再次执行是字节幂等；
`--no-resume` 只允许全新报告。`--dry-run` 只执行配置、路径、Adapter、Canonical
合同和时间轴验证，不创建 report。

输出报告固定为：

```text
<quality-archive>/<asset_id>.json
```

自动阶段遇到 `semantic_consistency` 会以 exit 0 返回
`state=awaiting_external, overall_decision=null`。当前生产仓库尚未提供 external stage
完成 mutation API；人工工作台 change 应通过 CAS 接口推进。CLI 不伪造人工完成，
测试中的 completed report 只是明确 fixture。

## 3. 发布前置

发布必须读取最终报告中的 `report_revision` 和
`canonical_binding.canonical_revision`；CLI 不允许调用者覆盖它们。要求包括：

- pipeline completed 且 `overall_decision=pass`；
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
  --release-root /training/curated-egodata \
  --dry-run
```

正式发布去掉 `--dry-run`。Publisher 会 staging、独立回读、冻结版
`LeRobotDataset==0.6.0` 首末帧验证、fsync、no-replace rename，最后原子更新
`CURRENT.json`。失败不改变旧 CURRENT 和旧 release；同一请求重试返回
`already_published`。

## 4. 训练读取

训练程序不依赖固定数据路径。先读取：

```text
<release-root>/CURRENT.json
```

再按其中 `release_id` 打开：

```text
<release-root>/releases/<release_id>/
```

训练端不得扫描 `.staging`，也不得猜“最新修改时间”。HDF5 与供应商 LeRobot 都会
由我方 Publisher 重写成同一 Curated LeRobot v3 schema；不同 source provenance 的
release ID 可以不同，但 normalized Canonical 语义和数组必须等价。

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
- `awaiting_external`：在人工工作台完成语义校准/Warn review，再从同一 report resume。
- runtime：保留 report/runtime_errors，修复依赖后 `--resume`。
- `source_integrity_error`：按 JSON 的 `retryable` 区分；临时 I/O 可重试，确定性的
  缺失、格式损坏、size/hash 漂移必须修复输入，禁止无限重试。
- prerequisite：核对 final report、revision、binding 和 source hash。
- validation：不要更新 CURRENT；检查冻结 validator 环境与 staged artifact。
- commit conflict：保留现有 release，调查 release ID/不可变内容冲突，不覆盖目录。

原始供应商 HDF5/LeRobot 不会被 CLI 原地修改；长期归档与 Curated training release
是不同生命周期。
