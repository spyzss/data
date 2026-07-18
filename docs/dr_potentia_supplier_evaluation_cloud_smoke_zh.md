# DR / Potentia supplier_evaluation 云端 Smoke

本文只给出云端操作命令，不在本地执行。所有路径均为占位变量。`BATCH_ROOT` 必须是 manifest 中所有源文件的共同父目录；如果源文件通过 symlink 暴露，则 symlink 本身必须位于 `BATCH_ROOT` 内，并为 pipeline 增加 `--allow-symlinked-sources`。

## 1. DR：3～5 个 task-level asset

```bash
export BATCH_ROOT=/path/to/common_batch_root
export DR_ROOT="$BATCH_ROOT/dr_delivery"
export DR_CALIB_ROOT="$BATCH_ROOT/dr_calibration"
export DR_RUN="$BATCH_ROOT/qc_runs/dr_smoke"
export DR_MANIFEST="$DR_RUN/manifests/supplier_manifest_deepreach.csv"
export DR_QC_CONFIG=/path/to/qc_acceptance_dr_smoke.yaml
export DR_PROJECTION_MAPPING=/path/to/dr_projection_mapping.yaml
export DR_CALIBRATION_MAP=/path/to/task_content_mapping.csv

# 仅在供应商正式 frame contract 已确认 timestamp 为 reference 后使用该值；
# 否则替换为已确认的参与 dataset，禁止省略或猜测。

python -m acceptance_pull.build_supplier_manifest \
  --supplier dr \
  --root "$DR_ROOT" \
  --calib-cache "$DR_CALIB_ROOT" \
  --calibration-map "$DR_CALIBRATION_MAP" \
  --output-dir "$DR_RUN" \
  --primary-camera head \
  --granularity task \
  --hdf5-reference-dataset timestamp \
  --max-assets 5
```

先检查 manifest，不要直接跑全量：

```bash
python - "$DR_MANIFEST" <<'PY'
import csv, json, sys
with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
print(json.dumps(rows, ensure_ascii=False, indent=2))
PY
```

必须确认：每个 `asset_id` 是 task 且不带 camera 后缀；三路 video path 均保留；`primary_camera=head`；显式 mapping 中的 `task_name` 与 `content_id` 正确；`hdf5_reference_dataset=timestamp`；`hdf5_expected_frame_count` 与 `hdf5_dataset_lengths` 一致；`hdf5_mismatch_ranges={}`；`start_frame=0`、`end_frame=N-1`；HDF5、calib 和 trajectory 属于同一 task。若出现 `hdf5_status=inconsistent_frame_count`，不要把共同前缀当完整数据；pipeline 会把 precheck 记为 input-invalid，但 supplier evaluation 的 video/audit 仍继续。

### DR supplier_evaluation

`DR_QC_CONFIG` 应从当前 `configs/qc_acceptance.yaml` 复制，只在 `modules.supplier_data_audit.parameters.suppliers.dr.mapping` 中填写已经从真实文件确认的字段。trajectory mapping 必须显式写 `timestamp_unit: s|ms|us|ns` 或单一 `timestamp_scale_to_seconds`，还要写 transform direction；不得根据示例猜字段。首次 smoke 可以保留 mapping 为空，此时 audit 必须显示 `unverified`。

```bash
python tools/run_qc_pipeline.py \
  --batch-root "$BATCH_ROOT" \
  --manifest "$DR_MANIFEST" \
  --profile supplier_evaluation \
  --config "$DR_QC_CONFIG" \
  --max-workers 2 \
  --no-resume
```

DR 未经 overlay 验证时，SAM3 不会加载模型，`execution.module_states.sam3_containment` 必须是：

```json
{"state": "blocked", "reason": "calibration_unverified"}
```

若 mapping 明确暴露外参方向冲突，则 reason 应为 `transform_ambiguous`。只有 projection 已人工确认、manifest 写成 `validated`、runtime mapping contract 全部通过且存在明确 `sam3_eligible=true` 候选时，才会加载模型并执行 DR head containment。有效 temporal 输出但零候选仍是 `skipped/no_candidates`。

### DR projection overlay 抽样

`DR_PROJECTION_MAPPING` 必须显式定义：trajectory frame 列、translation 列、quaternion 列、`transform_direction`、每个 camera 的内参 JSON path，以及 `transform_chain.calibration_extrinsic`。只有交付约定确认静态外参确为恒等时才可写 `identity`。

```bash
python tools/audit_deepreach_projection.py \
  --manifest "$DR_MANIFEST" \
  --mapping-config "$DR_PROJECTION_MAPPING" \
  --output-dir "$DR_RUN/projection_audit" \
  --camera head \
  --max-assets 5 \
  --max-samples 5
```

人工逐张核对 `projection_audit/overlays/`，同时检查：

- HDF5 keypoints 的坐标系和单位；
- trajectory pose 对应 source frame 还是 timestamp；
- quaternion 顺序和左右乘约定；
- `world_to_camera` / `camera_to_world` 方向；
- calib 的 camera 名称、内参分辨率和实际视频分辨率；
- source frame → local video frame 只减去一次 manifest `start_frame`；
- 左右手没有交换，快速运动段与头尾帧均对齐。

输出 `projected_unverified` 只表示工具完成投影，不表示已通过人工验证。

### 通过 overlay 后的 SAM3 开启条件

当前 runtime 会逐 asset 重新核对，而不是只相信一个 `validated` 字符串：

1. 显式 `task_name -> content_id` mapping 覆盖当前 asset；
2. manifest 记录 `projection_validation_status=validated`；
3. supplier config 的 `mapping_status=verified`，且 projection 明确声明 `camera_name=head`、`joints3d_coordinate_frame=head_camera`、`joints3d_unit=meter`、`projection_direction=direct_camera`、`trajectory_usage=lineage_only` 和 resolution policy；
4. calibration K、标定分辨率、实际 head 视频分辨率、HDF5/video 帧数和 source range 一致；
5. 当前 precheck fingerprint、有效 temporal 输出和 source-inclusive candidate 均通过；
6. hard keypoint presence invalid 会 blocked，不进入 SAM3；零 eligible candidates 会 skipped 且不加载模型。

满足这些条件后，DR adapter 从 HDF5 读取左右手 21x3，投影为 head 21x2，复用统一 containment、overlay、artifact publisher 和共享 SAM3 runtime。云端正式启用前仍必须用真实 overlay 验证 mapping；不能仅手工修改状态字段。

## 2. Potentia：3～5 个 task-level asset

```bash
export BATCH_ROOT=/path/to/common_batch_root
export POTENTIA_ROOT="$BATCH_ROOT/potentia_delivery"
export POTENTIA_RUN="$BATCH_ROOT/qc_runs/potentia_smoke"
export POTENTIA_MANIFEST="$POTENTIA_RUN/manifests/supplier_manifest_potentia.csv"
export POTENTIA_QC_CONFIG=/path/to/qc_acceptance_potentia_smoke.yaml

python -m acceptance_pull.build_supplier_manifest \
  --supplier potentia \
  --root "$POTENTIA_ROOT" \
  --output-dir "$POTENTIA_RUN" \
  --max-assets 5

python tools/run_qc_pipeline.py \
  --batch-root "$BATCH_ROOT" \
  --manifest "$POTENTIA_MANIFEST" \
  --profile supplier_evaluation \
  --config "$POTENTIA_QC_CONFIG" \
  --max-workers 2 \
  --no-resume
```

`POTENTIA_QC_CONFIG` 只填写已从真实 CSV/JSON 确认的 dot path、column name 和 timestamp unit/scale。`timestamp_unit` 与 `timestamp_scale_to_seconds` 只能选一个；fps、Hz、gap、coverage 全部基于归一化秒。`frame_index` 不要求从 0 开始；fps 同时保留实际视频值和 frames timestamp 推导值；标定必须同时保留 raw、scaled 和 actual video resolution。`scaling_mismatch_action` 默认 `review`，只有真实供应商策略明确要求时才改为 `fail`。

### 结果审计

```bash
find "$BATCH_ROOT/module_outputs" -path '*/supplier_data_audit/supplier_data_audit_result.json' -print
find "$BATCH_ROOT/quality_archive" -name 'potentia__*.json' -print

python - "$BATCH_ROOT/quality_archive" <<'PY'
import json, pathlib, sys
for path in sorted(pathlib.Path(sys.argv[1]).glob("potentia__*.json")):
    report = json.loads(path.read_text(encoding="utf-8"))
    states = report.get("execution", {}).get("module_states", {})
    audit = report.get("supplier_data_audit", {})
    print(path.name, {
        "precheck": {name: states.get(name) for name in (
            "hdf5_text_info", "quality_hand", "keypoint_presence",
            "keypoint_morphology", "keypoint_temporal")},
        "supplier_data_audit": audit.get("flow", {}).get("result_gate"),
        "video_quality": report.get("video_quality", {}).get("flow", {}).get("result_gate"),
        "sam3": states.get("sam3_containment"),
    })
PY
```

预期语义：

- 自动顺序为 `precheck -> video_quality -> supplier_data_audit -> sam3_containment`；video 与 supplier audit 独立，顺序不代表依赖；
- package 不出现在 `asset_id` 中，只保留为 `source_partition`；
- 一个 task 目录只产生一个 `potentia__<task_id>`；
- 五个 precheck 模块因无 HDF5/keypoint 为 `input_missing`，不是 pass；
- supplier_data_audit 和 video_quality 仍执行；
- Potentia video quality 的 HDF5 alignment 为 `disabled`，但其他供应商仍为原行为；
- SAM3 为 `blocked/no_keypoint_input`；
- `meta.qc` 只出现在 `supplier_quality_signal`，不产生 skeleton pass/fail。

## 3. 扩展至 100 条前的门槛

- 3～5 条 manifest 没有重复 task、跨 package 覆盖或 source-path 越界；
- DR 的 HDF5 source range 与 task 视频帧数口径已确认；
- DR trajectory/calibration mapping 经人工 overlay 验证；
- Potentia frames/aligned/IMU 的真实列名和时间单位已确认；
- 1-based frame index、约 60fps 样本和 timestamp gap 均能如实保留；
- calibration raw/scaled/video resolution 差异有可解释记录；
- 缺文件、缺 mapping、无 keypoint、SAM3 blocked 都没有被聚合成 pass；
- `precheck` fingerprint 是 `precheck-session-v7-calibrated-temporal-validity` 且含 `keypoint_temporal.output.v2`；supplier audit 是 `supplier-data-audit-producer-v3` 且 raw schema 为 `supplier_data_audit.raw.v2`；
- 先复跑 5 条 `--resume` 验证 artifact fingerprint/reuse，再扩大数量。
