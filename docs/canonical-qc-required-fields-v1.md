# Canonical QC 首版必需字段标准

文档版本：`canonical_qc_required_fields.v1`  
适用数据 Profile：`human_ego_hand_pose.v1`  
状态：设计确认稿  
日期：2026-07-15

## 1. 目标

首版只解决一件事：让 HDF5 和 LeRobot 两种供应商提交格式经过统一 Adapter 后，得到同一份 `CanonicalQcEpisode.v1`，完整执行自动 QC、人工语义校准和 Warn 人工复核，并由我方统一脚本发布为 Curated LeRobot v3。

```text
标准 HDF5 ──────> StandardHdf5Adapter ───┐
                                          ├─> CanonicalQcEpisode.v1
标准 LeRobot ───> StandardLeRobotAdapter ─┘
                                                    │
                                                    v
自动 QC -> 语义校准 -> Warn 人工复核 -> LeRobotV3Publisher
```

本版不支持供应商自定义字段映射模板，也不执行供应商提供的转换脚本。供应商必须使用本标准规定的字段名、类型、shape、单位和语义。

## 2. 首版边界

### 2.1 包含

- 单路主相机，`camera_id=main`。
- 外置 MP4。
- 双手各 21 点的 3D/2D 骨骼与 validity。
- 视频、时间轴、标定和任务/subtask 语义。
- HDF5 与 LeRobot 两种输入格式。
- 自动 QC Gate、人工语义校准、Warn 人工 Pass/Fail。
- 通过 QC 后由我方统一发布 LeRobot v3。

### 2.2 不包含

- 多目、双目、头部加腕部等多相机配置。
- Robot Teleop `observation.state/action`。
- 供应商 mask、depth、空间重建、音频和触觉。
- joint rotation、confidence 和其他可选特征。
- 供应商自定义字段路径、代码或转换脚本。

以上内容在首版全流程稳定后通过新 Profile 或新 schema 版本扩展，不修改 `CanonicalQcEpisode.v1` 的既有语义。

## 3. 必需字段总表

`R` 表示供应商提交中必须存在；`A` 表示由 Adapter 从真实文件探测或计算，供应商不得伪造；`O` 表示可选 Evidence。

### 3.1 身份与来源

| Canonical 字段 | 类型 | 级别 | 规则与用途 |
| --- | --- | --- | --- |
| `schema_version` | string | A | 固定为 `canonical_qc_episode.v1`。 |
| `asset_id` | string | R | 全局稳定资产 ID；不得包含路径分隔符。 |
| `batch_id` | string | R | 所属交付批次。 |
| `supplier_id` | string | R | 我方分配的稳定供应商 ID。 |
| `source_format` | enum | R | `hdf5` 或 `lerobot`。 |
| `source_schema_version` | string | R | 输入合同版本；未知 major 版本拒绝接入。 |
| `source_files[]` | object[] | A | 相对路径、文件角色、字节数、SHA-256；Adapter 对实际文件计算。 |
| `source_fingerprint` | SHA-256 | A | 对源文件 hash、输入 schema 和 Adapter 版本计算；用于缓存失效和发布前复核。 |

### 3.2 权威时间轴

| Canonical 字段 | 类型 / shape | 级别 | 规则与用途 |
| --- | --- | --- | --- |
| `time_axis.frame_count` | int64 | R | 记为 `T`，必须大于 0。 |
| `time_axis.timestamps_ns` | int64 `[T]` | R | 纳秒时间戳，严格递增。 |
| `time_axis.fps_num` | int64 | R | 名义 FPS 的有理数分子，大于 0。 |
| `time_axis.fps_den` | int64 | R | 名义 FPS 的有理数分母，大于 0。 |
| `time_axis.frame_index_base` | const `0` | A | Canonical 帧号固定从 0 开始。 |
| `time_axis.interval_semantics` | const `half_open` | A | 所有区间统一为 `[start,end_exclusive)`。 |

内部完整 episode 的帧范围固定为 `[0,T)`。供应商输入若使用闭区间，Adapter 只能在输入边界转换一次；QC、语义校准和 Publisher 不再使用闭区间。

### 3.3 单路主视频

| Canonical 字段 | 类型 | 级别 | 规则与用途 |
| --- | --- | --- | --- |
| `main_video.camera_id` | const `main` | A | 首版唯一相机。 |
| `main_video.camera_role` | const `ego` | A | 首版固定为第一视角主视频。 |
| `main_video.path` | relative path | R | 外置 MP4 路径。 |
| `main_video.sha256` | SHA-256 | A | Adapter 对 MP4 实际字节计算。 |
| `main_video.frame_count` | int64 | A | 从视频实际解码/探测；必须与 `T` 对齐。 |
| `main_video.width_px` | int32 | A | 从视频探测。 |
| `main_video.height_px` | int32 | A | 从视频探测。 |
| `main_video.fps_num/fps_den` | int64 pair | A | 从容器和 PTS 探测，不信任仅由供应商声明的 FPS。 |
| `main_video.codec` | string | A | 实际编码器。 |
| `main_video.pixel_format` | string | A | 实际像素格式。 |

逐帧视频 PTS 由 Video QC 直接读取并与 `time_axis.timestamps_ns` 比较，不要求供应商重复提交一份视频时间戳数组。

### 3.4 双手 21 点 Core

| Canonical 字段 | 类型 / shape | 级别 | 规则与用途 |
| --- | --- | --- | --- |
| `observation.hand_keypoints_3d` | float32 `[T,2,21,3]` | R | 双手 3D 点；单位为 meter。 |
| `observation.hand_joint_valid_3d` | bool `[T,2,21]` | R | 逐帧、逐手、逐关节有效性。 |
| `observation.hand_keypoints_2d` | float32 `[T,2,21,2]` | R | 主视频像素坐标。 |
| `observation.hand_joint_valid_2d` | bool `[T,2,21]` | R | 2D 点有效性。 |
| `hand_order` | const `[left,right]` | A | 手维度固定顺序。 |
| `joint_topology` | const `egodata_hand21.v1` | R | 21 点顺序必须匹配该版本。 |
| `coordinate_frame_3d` | const `camera:main` | R | 3D 点已经位于主相机坐标系。 |
| `length_unit` | const `meter` | R | 禁止混用 mm/cm/m。 |
| `coordinate_space_2d` | const `pixel` | R | 2D 点为主视频显示尺寸上的像素坐标。 |

有效性规则：

- validity 为 `true` 时，对应坐标必须全部为有限数值。
- validity 为 `false` 时，对应坐标必须为 NaN。
- 禁止用 `(0,0,0)` 或 `(0,0)` 表示缺失。
- 禁止删除缺失 joint 后缩短数组。
- 3D 与 2D 的 `T`、hand 顺序和 joint 顺序必须一致。

这些字段是 Keypoint Presence、Morphology、Temporal、Overlay 和 SAM3 Containment 的共同输入。

### 3.5 主相机标定

| Canonical 字段 | 类型 / shape | 级别 | 规则与用途 |
| --- | --- | --- | --- |
| `calibration.intrinsic_matrix` | float64 `[3,3]` | R | 主相机内参 K。 |
| `calibration.distortion_model` | enum | R | 首版允许 `none` 或标准中登记的模型。 |
| `calibration.distortion_coefficients` | float64 `[N]` | R | 数量必须与 distortion model 一致；无畸变写空数组。 |
| `calibration.image_width_px` | int32 | R | 必须与视频显示宽度一致。 |
| `calibration.image_height_px` | int32 | R | 必须与视频显示高度一致。 |
| `calibration.camera_axes` | const | R | 首版固定 `x_right_y_down_z_forward`。 |
| `calibration.pixel_origin` | const | R | 首版固定 `top_left`。 |

因为 `hand_keypoints_3d` 已统一到 `camera:main`，首版不要求静态或逐帧外参。Canonical Validator 使用内参、畸变模型、3D/2D 点执行重投影一致性检查。

### 3.6 任务与 Subtask 语义

| Canonical 字段 | 类型 | 级别 | 规则与用途 |
| --- | --- | --- | --- |
| `semantics.scene_id` | string | R | 规范场景 ID。 |
| `semantics.task_id` | string | R | 规范任务 ID。 |
| `semantics.task_category` | string | R | 任务分类。 |
| `semantics.task_cn` | string | R | 简洁中文任务名称。 |
| `semantics.task_en` | string | R | 简洁英文任务名称。 |
| `semantics.description_cn` | string | R | 完整中文 episode 描述。 |
| `semantics.description_en` | string | R | 完整英文 episode 描述。 |
| `semantics.subtask_sequence` | object[] | R | 非空、连续、完整覆盖 `[0,T)`。 |

每个 subtask 必须包含：

```json
{
  "subtask_id": "subtask_001",
  "start_frame": 0,
  "end_frame_exclusive": 411,
  "description_cn": "拿起物体",
  "description_en": "pick up the object"
}
```

Subtask 强约束：

- 第一段 `start_frame=0`。
- 最后一段 `end_frame_exclusive=T`。
- `start_frame < end_frame_exclusive`。
- 相邻段共享同一个边界。
- 不允许空档、重叠、逆序或零长度。
- 人工语义校准拖动内部共享边界时，必须原子修改相邻两段；一次确认只增加一次 `timeline_edit_count`。

## 4. 可选供应商 Evidence：`quality_hand`

`quality_hand` 不是 Core 必填字段。供应商未提供时，不得因此判定资产 fail；Keypoint 和 SAM3 QC 仍然正常执行。

### 4.1 Canonical 结构

```text
supplier.hand_quality
├── provided
├── raw_value
├── normalized_score
├── status
└── mapping_version
```

| 字段 | 类型 / shape | 级别 | 规则 |
| --- | --- | --- | --- |
| `supplier.hand_quality.provided` | bool | O | 是否提供该 Evidence。 |
| `supplier.hand_quality.raw_value` | supplier-defined `[T,2]` | O | 原始值；不得覆盖或二值化。 |
| `supplier.hand_quality.normalized_score` | float32 `[T,2]` | O | 可选；范围 `[0,1]` 且高值更好。 |
| `supplier.hand_quality.status` | enum `[T,2]` | O | `bad/warning/good/unknown`。 |
| `supplier.hand_quality.mapping_version` | string | O | 原始值到 score/status 的映射版本。 |

标准状态语义：

- `bad`：供应商认为该手数据不可用。
- `warning`：供应商认为存在风险或不确定。
- `good`：供应商认为可用。
- `unknown`：供应商未提供、局部缺失或无法判断。

不同原始值必须依据供应商书面定义和冻结的映射版本处理，禁止根据数值大小猜测。例如 `0/0.5/1` 只属于明确声明该语义的供应商，不能成为全局枚举。

当 `provided=false` 时，Canonical 使用 `status=unknown`；不得默认填成 `bad`。

## 5. `quality_hand` 与我方 Mask/骨骼观测

两者必须保持独立：

```text
supplier.hand_quality.status   # 供应商声明
sam3.mask_status               # 我方机器观测
```

SAM3/Mask 模块由我方生成并输出：

- `mask_available`
- `hand_mask_present`
- `keypoint_inside_ratio`
- `valid_projected_point_count`
- `mask_confidence`
- `containment_status`

对齐键固定为：

```text
(frame_idx, hand_side, camera_id)
```

首版 `camera_id` 固定为 `main`。

| 供应商状态 | Mask/骨骼结果 | 处理 |
| --- | --- | --- |
| `good` | pass | 一致。 |
| `good` | fail | 生成 `supplier_mask_disagreement` Warn，进入人工检查。 |
| `bad` | pass | 记录供应商可能误报，用于供应商质量统计。 |
| `bad` | fail | 一致；最终仍按我方机器 QC 结果处理。 |
| `unknown` | 任意 | 不计算一致率，只使用我方机器结果。 |
| 任意 | Mask 无法运行 | 不做一致性判断，不得猜测。 |

最终 Pass/Fail 以我方 Keypoint、SAM3、视频 QC 和人工结果为准。供应商 `quality_hand` 只用于一致率、误报率、漏报率、人工抽查优先级和供应商评估。

## 6. 固定 HDF5 输入合同

HDF5 必须使用固定路径；`StandardHdf5Adapter` 不搜索别名，也不按供应商猜测路径。

### 6.1 Root attributes

```text
schema_version = egodata_hdf5_qc_input.v1
asset_id
batch_id
supplier_id
frame_count
fps_num
fps_den
joint_topology = egodata_hand21.v1
coordinate_frame_3d = camera:main
length_unit = meter
coordinate_space_2d = pixel
```

### 6.2 必需 datasets

```text
/time/timestamps_ns                                      int64   [T]
/observation/hand_keypoints_3d                           float32 [T,2,21,3]
/observation/hand_joint_valid_3d                         bool    [T,2,21]
/observation/hand_keypoints_2d                           float32 [T,2,21,2]
/observation/hand_joint_valid_2d                         bool    [T,2,21]
/camera/main/intrinsic_matrix                            float64 [3,3]
/camera/main/distortion_coefficients                     float64 [N]
/semantics/annotation_json                               UTF-8 scalar JSON
```

`/camera/main` attributes 必须包含：

```text
distortion_model
image_width_px
image_height_px
camera_axes = x_right_y_down_z_forward
pixel_origin = top_left
```

外置视频固定放在：

```text
data/<asset_id>/<asset_id>.h5
data/<asset_id>/main.mp4
```

### 6.3 可选 Evidence datasets

```text
/supplier/hand_quality/raw_value
/supplier/hand_quality/normalized_score                  float32 [T,2]
/supplier/hand_quality/status                            uint8   [T,2]
```

`status` 的存储编码固定为：

```text
0 unknown
1 bad
2 warning
3 good
```

`/supplier/hand_quality` attributes 记录 `provided` 和 `mapping_version`。Adapter 将 uint8 编码转换为 Canonical 枚举。

## 7. 固定 LeRobot 输入合同

`StandardLeRobotAdapter` 可以识别受支持的 LeRobot v2.1/v3 目录差异，但不处理供应商自定义 feature 名。

### 7.1 必需逐帧字段

```text
episode_index
frame_index
timestamp
timestamp_ns
observation.hand_keypoints_3d
observation.hand_joint_valid_3d
observation.hand_keypoints_2d
observation.hand_joint_valid_2d
observation.images.main
task_index
subtask_index
```

Feature metadata 必须声明与 Canonical 相同的 shape、维度顺序、单位、hand order、joint topology 和坐标系。

### 7.2 必需 episode metadata

每个 episode 必须能解析出：

```text
asset_id
batch_id
supplier_id
frame_count
fps_num
fps_den
joint_topology = egodata_hand21.v1
coordinate_frame_3d = camera:main
camera intrinsic/distortion/image size
scene/task/description
subtask_sequence
```

首版使用固定 `meta/episode_semantics.jsonl` 保存每个 episode 的语义对象；每行按 `episode_index` 和 `asset_id` 关联，不允许用 README 自然语言代替机器字段。

### 7.3 可选 Evidence 字段

```text
supplier.hand_quality.raw_value
supplier.hand_quality.normalized_score
supplier.hand_quality.status
supplier.hand_quality.mapping_version
```

已是 LeRobot v3 的供应商数据也不能直接进入训练集。QC 通过后仍由我方 Publisher 重新生成或重打包 Curated LeRobot v3 的 meta、Parquet、索引、统计、checksum 和 ReleaseManifest；合规 MP4 可以按 hash 复用。

## 8. 各 QC 阶段的字段依赖

| 阶段 | 必需输入 |
| --- | --- |
| Source/Schema Gate | identity、source files/hash、全部 Core shape/dtype/unit。 |
| Text/Metadata | task、description、subtask sequence。 |
| Keypoint Presence | 3D/2D keypoints、3D/2D validity。 |
| Keypoint Morphology | 3D keypoints、3D validity、joint topology、meter。 |
| Keypoint Temporal | timestamps、3D keypoints、3D validity。 |
| Video Quality | MP4、Canonical 时间轴、frame count。 |
| SAM3 Containment | MP4、2D keypoints、2D validity；mask 由我方生成。 |
| Reprojection Validation | 3D/2D keypoints、validity、intrinsics、distortion。 |
| 人工语义校准 | MP4、subtask sequence、半开共享边界、双语描述。 |
| Warn 人工质检 | QC JSON issues/evidence；不新增供应商 Core 字段。 |
| Duplicate Check | MP4、asset ID、timestamps。 |
| Content Validity | MP4、task/description/subtasks。 |
| Effective Duration | timestamps 及前序 QC issue 区间。 |
| LeRobot v3 Publisher | 完整 Canonical、最终语义 revision、最终 QC pass。 |

因此本标准列出的 Core 已足够支撑完整 QC；无需把 `quality_hand`、供应商 mask、joint rotation、confidence 或多相机字段强制加入首版。

## 9. Adapter 规则

两个 Adapter 必须遵守同一行为：

1. 只读取固定字段，不搜索别名。
2. 不使用默认 FPS 代替缺失值。
3. 不把 validity 当成 `quality_hand`。
4. 不根据 `quality_hand` 数值大小猜测语义。
5. 不静默跳过非法 shape、dtype 或 joint。
6. 不截断不同长度的数组来伪造对齐。
7. 不在 Adapter 内形成 QC Pass/Fail。
8. 输出不可变 `CanonicalQcEpisode.v1` 或结构化失败诊断。
9. 同一逻辑 episode 的 HDF5 与 LeRobot 输入，除 source provenance 外必须产生相同 Canonical 语义 fingerprint。

## 10. 错误与最终状态

| 情况 | 分类 | 处理 |
| --- | --- | --- |
| Core 字段缺失 | 数据合同缺陷 | 写 Source Gate fail issue，进入供应商失败统计。 |
| shape/dtype/单位错误 | 数据合同缺陷 | fail，不进入依赖该字段的 QC。 |
| 时间戳不递增 | 数据合同缺陷 | fail。 |
| 视频与 `T` 不对齐 | 数据合同缺陷 | fail。 |
| 3D/2D 重投影严重不一致 | 数据质量问题 | 按统一 QC Config 生成 warn/fail。 |
| Subtask 不连续或越界 | 数据合同缺陷 | fail；不得由 UI 自动猜测修复。 |
| `quality_hand` 未提供 | 正常可选缺失 | 不 fail，status 视为 unknown。 |
| `quality_hand` 与机器结果冲突 | 供应商 Evidence 分歧 | 生成 Warn/统计，不直接覆盖机器结论。 |
| 文件暂时无法读取、网络或权限错误 | Runtime Error | `overall_decision=null`，可重试。 |
| QC report revision 冲突 | CAS Error | 拒绝旧写入，重新读取。 |
| Publisher 验证失败 | Publish Error | 不暴露新 release，原 QC JSON 不变。 |

## 11. 人工语义修订与发布

为同时支持 HDF5 和 LeRobot 输入，人工语义修订不依赖某一种源格式：

1. 人工确认一次时间边界或文本修改。
2. Canonical working revision 原子更新。
3. QC JSON 记录 `timeline_edit_count`、`subtask_text_edit_count` 及 before/after 审计。
4. 源 HDF5/LeRobot 保持只读。
5. 最终 LeRobotV3Publisher 只读取最新 Canonical revision，将修订后的文本和时间轴写入 Curated LeRobot v3。

只有 `overall_decision=pass`、语义阶段完成、Warn 复核完成且 source fingerprint 未变化的资产可以发布。

## 12. 验收清单

- [ ] 供应商提交格式只能是标准 HDF5 或受支持的标准 LeRobot。
- [ ] 所有 Core 字段存在且符合固定名称、shape、dtype、单位和坐标系。
- [ ] 主视频可解码且与 `T`、FPS、时间轴一致。
- [ ] 3D/2D 骨骼与 validity 对齐。
- [ ] joint topology 严格等于 `egodata_hand21.v1`。
- [ ] 3D 点位于 `camera:main`，2D 点使用 pixel。
- [ ] 标定可用于重投影验证。
- [ ] Subtask 使用半开共享边界并完整覆盖 `[0,T)`。
- [ ] `quality_hand` 缺失不会触发 fail。
- [ ] `quality_hand` 原值、标准化状态和我方 Mask 结果互不覆盖。
- [ ] HDF5 与 LeRobot 的 Canonical 等价性测试通过。
- [ ] QC 通过后只使用我方 LeRobotV3Publisher 生成训练数据。
- [ ] Publisher 未完成验证前，训练 release 不可见。
