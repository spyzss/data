# DeepReach Supplier Adapter 使用说明

## 1. 目的与边界

DeepReach adapter 将三视角 deliverable 扫描为 acceptance workflow 可消费的 canonical CSV，并可为独立 `video_quality` runner 创建轻量 symlink staging。

当前交付范围：

- canonical supplier manifest
- calibration sidecar manifest
- head/left-wrist/right-wrist 视频发现
- DeepReach HDF5 到 canonical `ClipInputs` 的 precheck schema adapter
- HDF5 dataset frame-contract 审计
- video-quality staging symlinks
- 经显式验证的 head-camera HDF5 3D→2D SAM3 containment adapter

当前不包含：

- DA3 模型执行
- 未经 mapping / overlay 验证的 calibration 自动放行
- 视频、HDF5 或 calib 文件复制

实现位置：

- `acceptance_pull/build_supplier_manifest.py`
- `acceptance_pull/supplier_adapters/deepreach.py`

## 2. 已知目录结构

```text
/mnt/oss/dr-3camera-deliverable/
  README.md
  hdf5/<task>.h5
  lerobot_v2/<task>/
    meta/info.json
    meta/episodes.jsonl
    meta/tasks.jsonl
    meta/wrist_calibration.json
    data/chunk-000/episode_000000.parquet
    videos/chunk-000/observation.images.head/episode_000000.mp4
    videos/chunk-000/observation.images.left_wrist/episode_000000.mp4
    videos/chunk-000/observation.images.right_wrist/episode_000000.mp4
```

外部 calibration cache：

```text
/mnt/workspace/spy/marmalade/data_cache/deepreach_calib/
  mano-mesh-3d/<content_id>/calib.json
```

Adapter 会对 `hdf5/` 文件名和 `lerobot_v2/` 子目录名取并集，因此缺少其中一侧时仍会尽量发出 manifest row。

Calibration 必须另外提供显式 `task_name,content_id` 映射。Adapter 只会在有该映射时读取 `<calib-cache>/<content_id>/calib.json`，不再用 `task_name` 充当 `content_id`。

## 3. Asset 和 Camera 定义

默认每条 manifest row 表示一个 task-level canonical asset：

```text
asset_id = <task_name>
```

同一行保留 `head`、`left_wrist`、`right_wrist` 三路路径，并用 `primary_camera` / `primary_video_path` 选择本轮正式 video quality 视角。主视角缺失时输出 `primary_camera_missing`，不得回退。旧 camera-suffixed 结构只通过显式 `--granularity task_camera` 保留为兼容模式，不得与 task-level manifest 混用。

支持的 `camera_name`：

- `head`
- `left_wrist`
- `right_wrist`

CLI 默认只生成 `head`。重复传入 `--camera` 可生成多视角。

## 4. Manifest Schema

输出：

```text
<output-dir>/manifests/supplier_manifest_deepreach.csv
```

字段：

| 字段 | 含义 |
| --- | --- |
| `supplier_id` | 固定为 `dr` |
| `supplier_name` | 固定为 `DR` |
| `asset_id` | `<task_name>` |
| `task_name` | HDF5 stem / LeRobot task 目录名 |
| `content_id` | 由显式 mapping 绑定的 calibration content ID |
| `calibration_mapping_status` | `mapped` 或 `mapping_missing` |
| `projection_validation_status` | 默认 `calibration_unverified`；完成显式审计后才能为 `validated` |
| `primary_camera` | 正式 video-quality 主视角 |
| `primary_video_path` | 主视角视频；不做静默 fallback |
| `head_video_path` / wrist paths | 同一 task 的三路视频 inventory |
| `hdf5_path` | task-level HDF5 |
| `lerobot_task_dir` | LeRobot task 根目录 |
| `parquet_path` | episode parquet |
| `wrist_calibration_path` | deliverable 内 wrist calibration |
| `calib_path` | 外部 mano-mesh calib 候选路径 |
| `calibration_status` | `present_unverified` 或 `missing` |
| `hdf5_reference_dataset` | frame contract 的权威 dataset，必须在确认供应商契约后显式指定 |
| `hdf5_expected_frame_count` | reference dataset 长度 |
| `hdf5_dataset_lengths` | 五个参与 dataset 的实际长度 |
| `hdf5_mismatch_ranges` | 相对 reference 的缺失/多余 inclusive range |
| `adapter_status` | `ready_unverified`、`input_missing` 或 `input_invalid` |

缺视频、HDF5、Parquet、wrist calibration 或外部 calib 时不会 crash。路径仍保留在 row 中，便于 ledger 显示缺失证据。

Calibration sidecar：

```text
<output-dir>/manifests/deepreach_calibration_sidecar.csv
```

包含：

```text
asset_id,task_name,content_id,primary_camera,calib_path,camera_trajectory_path,
calibration_status,trajectory_status,calibration_mapping_status,
projection_validation_status,projection_validation_reason
```

## 5. Calibration 与 projection 门禁

Calibration mapping CSV 格式：

```text
task_name,content_id
task_001,content_abc
```

`projection_validation_status=validated` 不是单独放行条件。Runtime 还会验证 supplier config 中 `mapping_status=verified`，以及 `projection` 的 head camera、`head_camera`、meter、`direct_camera`、`trajectory_usage=lineage_only` 和分辨率政策。Calibration K、实际 head video 分辨率、HDF5/video 帧数和 source range 也必须一致。任一条不成立时 SAM3 blocked，不加载模型。

## 6. 生成 Head Manifest

```bash
RUN=outputs/acceptance_5x100/deepreach

.venv/bin/python -m acceptance_pull.build_supplier_manifest \
  --supplier dr \
  --root /mnt/oss/dr-3camera-deliverable \
  --output-dir "$RUN" \
  --calib-cache /mnt/workspace/spy/marmalade/data_cache/deepreach_calib/mano-mesh-3d \
  --calibration-map /path/to/task_content_mapping.csv \
  --granularity task \
  --primary-camera head \
  --hdf5-reference-dataset timestamp \
  --stage-video-quality
```

生成：

```text
$RUN/manifests/supplier_manifest_deepreach.csv
$RUN/manifests/deepreach_calibration_sidecar.csv
$RUN/video_quality/hdf5/<asset_id>_hdf5.hdf5
$RUN/video_quality/video/<asset_id>_video.mp4
```

最后两类文件是 symlink，不复制源数据。重复运行会复用指向相同源文件的 symlink；遇到冲突目标会显式报错，不静默覆盖。

旧 task-camera 兼容示例：

```bash
.venv/bin/python -m acceptance_pull.build_supplier_manifest \
  --supplier deepreach \
  --root /mnt/oss/dr-3camera-deliverable \
  --output-dir "$RUN" \
  --camera head \
  --camera left_wrist \
  --camera right_wrist \
  --granularity task_camera \
  --calib-cache /mnt/workspace/spy/marmalade/data_cache/deepreach_calib/mano-mesh-3d
```

## 7. 运行 Head-view Video Quality

```bash
mkdir -p "$RUN/logs"
set -o pipefail

.venv/bin/python run_acceptance_video_quality.py \
  --batch "$RUN/video_quality" \
  2>&1 | tee "$RUN/logs/video_quality.log"
```

Per-asset JSON 写入：

```text
$RUN/video_quality/quality_archive/<asset_id>.json
```

注意：`run_acceptance_video_quality.py`返回 2 表示至少一个 asset QC fail，不一定表示 runner crash。

## 8. Precheck Frame Contract

仓库虽然有 `configs/precheck_dr3camera.yaml`，但当前通用 supplier HDF5 adapter 面向官方 21-point transform schema；DeepReach 已知结构是：

```text
/hand/<side>/valid
/hand/<side>/wrist_trans
/hand/<side>/root_orient
/hand/<side>/pose_body
/hand/<side>/joints3d
/camera/slam_pos
/camera/slam_quat
/camera/slam_valid
```

当前 adapter 必需读取 `timestamp` 和左右手 `joints3d`。左右手 `valid` 为 optional；存在时必须与 reference 等长，并展开为每手 21 点 validity；不存在时只使用 finite 3D values 判定有效性。禁止使用 `min(lengths)` 静默截断。长度不一致时，manifest 保存 reference、expected count、所有已提供 dataset 长度和 mismatch ranges，正式 precheck 输出 `input_invalid/inconsistent_frame_count`。

真实数据扩量前仍需确认：

- `joints3d` 的点数、顺序、单位和坐标系；
- 是否能可靠映射到 supplier official 21-point acceptance topology；
- `valid` 的左右手语义；
- task-level HDF5 与 episode_000000 的帧对齐；
- camera SLAM pose 与三个视频视角的对应关系。
