# Supplier Acceptance Tools

This repository contains supplier-acceptance workflows beside the SAM3/DA3
annotation pipeline. The canonical QC contract is v2:

```text
asset_qc_report.v2
qc_acceptance_config_schema.v2 / qc_acceptance_v2.1.0
<batch>/quality_archive/*.json
```

Every asset has one master report. Batch statistics, the manual candidate queue,
and compatibility ledgers are projections; sidecar 只作证据 and reconciliation.

## Canonical ingest and Curated LeRobot v3

新标准入口同时支持显式 HDF5 与 LeRobot source：

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
  --supplier-id supplier-001
```

自动 QC 可以安全 resume，并在当前人工语义边界返回 `awaiting_external`。最终人工
状态和 `canonical_binding` 完成后执行：

显式 identity 来自批次 manifest；Source Gate 的确定性失败也会原子写入该资产的
QC JSON，保证后续批次统计不依赖 CLI 日志。存在非零语义编辑时，首版 Publisher
在 format-neutral revision artifact 落地前 fail closed，不会发布旧语义。

```bash
python tools/publish_lerobot_v3.py \
  --source /data/batch/asset-001 \
  --source-format hdf5 \
  --canonical-source-root /data/batch/asset-001 \
  --qc-report /data/batch/quality_archive/asset-001.json \
  --release-root /training/curated-egodata
```

两条命令均支持 `--dry-run`，输出单行机器 JSON。正式发布只在所有门禁和官方
reader 验证通过后原子更新 `CURRENT.json`。完整参数、退出码、训练读取和恢复步骤见
`docs/canonical-qc-ingest-publish-runbook.md`。

The workflows are:

- `acceptance_pull`: validate a supplier manifest, sample IDs by scene/task,
  and pull paired HDF5/video files into a local batch.
- `acceptance_video_quality`: run the low-cost no-reference video prefilter and
  update one QC archive per asset.
- the unified orchestrator: execute the configured modules under `acceptance` or
  `supplier_evaluation`, then pause at external semantic/manual stages.

The durable per-asset report is:

```text
<batch>/quality_archive/<asset_id>.json
```

Other QC modules are described in `docs/PRD-qc-gated-json.md`. Their source
code is owned by the corresponding module developers; this repository's
current implementation change only covers video QC and its shared config/report
contract.

## Batch Sampling And Pull

Example local config:

```yaml
manifest: XJGT_20260616.xlsx
readme: README.txt
output: sampled/XJGT_20260616
workers: 8
seed: 20260701
sample_ratio: 0.01
hdf5:
  kind: local
  root: hdf5
video:
  kind: local
  root: video
```

Run:

```bash
python run_acceptance_pull.py --config pull.yaml
```

The pull workflow writes:

```text
sampled/XJGT_20260616/
  hdf5/
  video/
  reports/
    id_consistency.csv
    sample_manifest.csv
    pull_report.csv
    summary.json
```

Sampling uses the manifest/video/HDF5 ID intersection, keeps full scene
coverage, and fills the remaining quota with task diversity. `sample_ratio`
defaults to `0.01` and uses `ceil(valid_id_count * sample_ratio)` as the minimum
sample size.

## Manual Inputs And Confirmation Points

The pull workflow intentionally keeps these choices human-owned:

- `output`: local destination for the sampled batch.
- `sample_ratio`: acceptance sampling ratio; default `0.01`.
- `workers`: pull concurrency for the current network and disk; default `8`.
- `seed`: explicit value for a reproducible historical sample, or blank to use
  the run date.
- Local mode: `manifest`, `readme`, `hdf5.root`, and `video.root`.
- OSS mode: `batch_uri` and `region`. Do not put credentials or browser login
  state in YAML.
- Acceptance policy: the reviewed, immutable `qc_config.config_version` used
  to initialize every asset JSON in one pipeline run.

After pulling, review:

- `reports/id_consistency.csv` for missing or extra IDs.
- `reports/pull_report.csv` for failed pull operations.
- `reports/summary.json` for actual sample size and coverage.
- `quality_archive/<asset_id>.json` for the asset's accumulated QC evidence.

Do not infer a verdict from a pull report, CSV, overlay or legacy sidecar. They
are evidence only; the canonical source is `quality_archive/*.json`.

## OSS Batch Input

For local OSS testing with `oss-browser2`, keep credentials outside YAML and
logs. Use a batch-level URI:

```yaml
batch_uri: oss://xingjiguitu/egodata/XJGT_20260616
region: beijing
output: sampled/XJGT_20260616
workers: 8
sample_ratio: 0.01
```

The tool derives `<batch prefix>/hdf5` and `<batch prefix>/video`, then downloads
`README.txt` and the first `.xlsx` manifest when they are not provided.

## Unified QC Config

Video QC accepts only the unified versioned config:

```text
configs/qc_acceptance.yaml
```

The immutable copy for the current release is:

```text
configs/qc_acceptance/qc_acceptance_v2.1.0.yaml
```

The video parameters live under:

```yaml
schema_version: qc_acceptance_config_schema.v2
config_version: qc_acceptance_v2.1.0
execution_profiles:
  acceptance:
    fail_action: stop
    runtime_error_action: stop_incomplete
  supplier_evaluation:
    fail_action: record_and_continue
    runtime_error_action: stop_incomplete
modules:
  video_quality:
    module_version: video_prefilter_v0.3.2
    parameters:
      decode: {}
      exposure: {}
      sharpness_global: {}
      freeze: {}
      defects: {}
      hdf5_alignment: {}
```

Do not pass a legacy video-only YAML. To test another reviewed config, copy the
whole unified document, bump `config_version`, edit
`modules.video_quality.parameters`, and run with `--config`.

## Video Quality Check

Run the default reviewed config:

```bash
python run_acceptance_video_quality.py --batch sampled/XJGT_20260616
```

Run a different complete unified config:

```bash
python run_acceptance_video_quality.py \
  --batch sampled/XJGT_20260616 \
  --config configs/qc_acceptance/qc_acceptance_v2.1.0.yaml
```

Output:

```text
sampled/XJGT_20260616/
  hdf5/
  video/
  quality_archive/
    <asset_id>.json
```

The command updates only video-owned content in the existing asset JSON and
preserves unknown fields and blocks written by other modules. It increments
`report_revision`, validates `asset_qc_report.v2`, writes a temporary file, and
atomically replaces the archive with a CAS check.

The top-level `qc_config` records the actual loaded config's schema version,
config version, path, and SHA-256 hash. Thresholds and per-issue config versions
are not copied into the asset JSON. Every warn/fail item is one object in the
top-level `issues` array with a stable `issue_id`, `rule_id`, actual value,
operator, boundary value, and context.

Video flow rules:

- `pass`: continue to the next configured module.
- `warn`: keep issue IDs in `manual_review.candidate_issue_ids` and continue.
- `acceptance` + `fail`: stop automated QC, route to `batch_statistics`, and do
  not run later high-cost, semantic or manual modules.
- `supplier_evaluation` + `fail`: record the machine fail and continue; the
  completed report still has `overall_decision=fail`.
- runtime/evidence/config/CAS errors: write `runtime_errors`, set status `error`,
  and leave `overall_decision=null`; this is not a quality fail.
- `video_quality.flow.exit_gate.continue_to_next_module` is authoritative.
- `video_quality.evaluation.should_run_mask_qc` is a compatibility alias for
  the same video gate and should not become the cross-module orchestrator.

## Unified QC flow and manual routing

The profile-aware flow is:

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

`semantic_consistency` is an external stage before manual warn review. It is
currently human-operated and may later be replaced by a model adapter without
changing the report contract. Only accumulated warn candidates enter manual
review（仅累计 warn 进入人工质检）. An empty candidate list sets
`manual_review.required=false`, `state=not_required`; it does not create a
normal Pass-sample review task. Non-empty candidates use `queued`, then
`in_progress`, and finally `completed`. An acceptance hard fail uses
`skipped_due_to_fail` and never creates a semantic/manual task.

Machine issues remain immutable observations. Human review records verdict,
reviewer, timestamp and evidence references alongside them; human-confirmed
failures are counted separately and never erase the machine fail/warn rate.

### Canonical projection and reconciliation commands

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

The formal commands read only `quality_archive/*.json`. Legacy candidate-window,
SAM3, video and manual sidecars are accepted only as explicit reconciliation
inputs; they cannot change canonical asset/issue/execution rows or verdicts.
Cache files are disposable: their source manifest must match every report's
relative path, revision and SHA-256 before reuse.

## Video Prefilter Scope

`video_prefilter_v0.3.2` is a practical no-reference prefilter for robot
pretraining video. It checks:

- open/stream/codec/metadata health;
- FPS and minimum display resolution;
- PTS monotonicity, estimated missing frames, interval p99, and maximum gap;
- sampled decode completeness;
- black, over-dark, and over-exposed frames;
- normalized global Laplacian and Tenengrad indicators;
- low-motion, freeze candidates, confirmed freezes, and continuous intervals;
- combined defect-duration ratio;
- video/HDF5 frame-count alignment.

Drop detection prefers real per-frame PTS from `ffprobe`, then PyAV. OpenCV
`CAP_PROP_POS_MSEC` is only a fallback and records
`drop_detection_reliable=false`. `drop_frame_ratio` is based on
`estimated_missing_frames`.

`adjacent_near_duplicate_ratio` is a low-motion indicator, not a reject rule.
Frames 0.5 seconds apart must still be near-duplicates to become a freeze
candidate; 1.0 second is required for confirmed freeze. A confirmed visual
freeze with clear HDF5 keypoint/action/camera-pose motion is recorded as
`video_state_conflict` and evaluated more strictly in critical interaction
windows.

Sharpness is calibrated for content that may later be downsampled to roughly
448x256. The objective is to distinguish edges and objects and reject extreme
blur, not to demand high-definition imagery. Hand ROI quality is not computed.

Video QC does not perform keypoint accuracy validation, keypoint-mask matching,
trajectory checks, mask IoU, semantic consistency, or subtask acceptance.

## Contracts

- Asset JSON format: `docs/asset-qc-json-format.md`
- Gate and colleague implementation PRD: `docs/PRD-qc-gated-json.md`
- Unified config PRD: `docs/PRD-qc-unified-config.md`
- Config schema: `schemas/qc_acceptance_config.v2.schema.json`
- Asset report schema: `schemas/asset_qc_report.v2.schema.json`

## v1 read-only / v2 writeback / rollback

An `asset_qc_report.v1` file is a read-only migration input. Before the first
v2 write, run the pure `migrate_v1_to_v2()` path, preserve its video block,
unknown fields and revision, then validate and atomically write v2 under a CAS.
If migration or a write fails, keep the v1 master and write diagnostics or a
sidecar reconciliation artifact only. Rollback means returning to read-only v1
consumption; it must never overwrite the master verdict with an old sidecar or
erase a newer v2 `overall_decision`.
