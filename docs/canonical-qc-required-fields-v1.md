# Canonical Data 首版字段与 QC 投影标准

文档版本：`canonical_qc_required_fields.v1`  
适用数据 Profile：`human_ego_hand_pose.v1`  
状态：目标架构 + 当前 v1 兼容实现（Data extension/revision 已实现）
日期：2026-07-17

> 文件名和当前 Python/schema 标识中的 `canonical_qc` 为 v1 兼容名称，不再代表
> Canonical 只服务 QC。架构术语统一使用 **Canonical Data**；当前已实现的
> `CanonicalQcEpisode` / `canonical_qc_episode.v1` 是兼容类型/schema 名称；
> `CanonicalDataEpisode` 是同一实现的架构别名。

## 1. 目标

目标端态是让 HDF5、LeRobot 及后续供应商格式经过统一 Adapter 后形成同一套
Canonical Data view。它是供应商 Raw 的长期标准化访问层，而不是只为 QC 裁剪的
中间格式；QC 与 Publisher 都是消费者。Raw 始终不可变，Canonical 负责统一语义、
保留供应商扩展、绑定 batch metadata 和 provenance，检测派生结果则只进入现有
`asset_qc_report.v2`。

当前实现以 `CanonicalQcEpisode.v1` 兼容对象承载 Core、typed supplier extension
inventory 和 batch metadata。Publisher 将 frame-aligned extension 写成 LeRobot
feature，把 episode/batch extension 写入版本化 semantic sidecar，并支持受控非零
语义 revision artifact。无法无损表示、schema 声明与 Raw dtype/shape 不一致或尚无
供应商 Adapter 的字段会 fail closed；不能把这一范围扩大解释为任意格式 Raw 已全量支持。

```text
immutable supplier Raw ───────────────┐
标准 HDF5 ──────> StandardHdf5Adapter ├─> Canonical Data view
标准 LeRobot ───> StandardLeRobotAdapter ┘    ├─ standardized Core
                                               ├─ supplier extensions/evidence
batch manifest / dataset attributes ──────────┘
                                               |
                          +--------------------+--------------------+
                          |                                         |
                          v                                         v
              已登记自动 QC -> external stages          publication metadata
                          |                                         |
                          v                                         |
              asset_qc_report.v2                                   |
                          +------------------+----------------------+
                                             |
                         optional canonical revision artifact
                                             |
                                             v
                    Publisher(Raw + metadata + QC + revision)
                                             |
                                             v
                                  Curated LeRobot v3
```

当前 v1 Adapter 不支持供应商自定义字段映射模板，也不执行供应商提供的转换脚本。
供应商要进入当前 Core projection，必须使用本标准规定的字段名、类型、shape、单位和
语义；这不授权 Adapter 或 Publisher 丢弃 Raw 中其他字段。未进入 Core 的字段必须在
extension inventory 中保留或以结构化错误标记 unsupported，禁止静默丢失。

## 2. Canonical Data 分层与首版边界

### 2.1 三层职责

| 层 | 内容 | 持久化与消费者 |
| --- | --- | --- |
| Standardized Core | 跨供应商已统一的 identity、provenance、时间轴、视频、keypoints/validity、标定、任务/subtask。 | Canonical Data；QC 与训练共同读取。 |
| Supplier extensions/evidence | 当前尚未统一但可能用于训练、检索或预研的 state/action、额外相机、depth、force/tactile、joint rotation、音频、供应商模型输出、私有 metadata；现有 `quality_hand` 属于 Evidence。 | Canonical extension inventory + Raw 引用；不得因 QC 未使用而丢弃。 |
| Derived/QC outputs | freeze、blur、duplicate、语义判定、人工 verdict、effective duration 等检测结果。 | 只进入 `asset_qc_report.v2` 或可重建 evidence sidecar，不写回 Raw/Canonical。 |

Canonical Data 是 Raw 的标准化只读视图，不是 Raw 的可变副本。标准化只允许单位、
坐标系、字段名、dtype/shape 和时间对齐等确定性转换；任何可能改变数据含义的人工
修订都必须经过独立 revision artifact。

### 2.2 当前 v1 已实现 Core projection

- 单路主相机，`camera_id=main`。
- 外置 MP4。
- 双手各 21 点的 3D/2D 骨骼与 validity。
- 视频、时间轴、标定和任务/subtask 语义。
- HDF5 与 LeRobot 两种输入格式。
- 已登记且启用的自动 QC Gate；未实现模块以 Deferred/disabled 明示。
- 人工语义校准与 Warn 人工 Pass/Fail 的 external stage 合同和暂停点；工作台实现 Deferred。
- 最终 QC Pass 后由我方统一发布 Core、已登记 extensions、batch attributes；非零
  文本/共享边界编辑由 `canonical_revision_artifact.v1` 纯函数应用并绑定 release manifest。

### 2.3 通过 extension inventory 登记、但尚未统一为 Core 的字段

- 多目、双目、头部加腕部等多相机配置；
- Robot Teleop `observation.state/action`；
- 供应商 mask、depth、空间重建、音频和触觉；
- joint rotation、confidence 和其他可选特征；
- 供应商自定义字段路径、代码或转换脚本。

这些字段不需要进入 Core。标准 HDF5 的非 Core dataset 和标准 LeRobot 中已登记、
规则定长的额外列会进入 typed inventory；Publisher 对 frame-aligned 值写 feature，
对 episode/batch 值写 base64 ndarray sidecar。object/vlen、ragged/null、声明与物理
dtype/shape 不一致、多媒体专用编码或尚无 Adapter 的格式会 fail closed，需新增显式
profile/policy 后才能发布。

### 2.4 Batch metadata / dataset attributes

每个批次必须有稳定的批次 manifest，用于数据分类、检索和组合，不参与单资产 QC
verdict。目标字段至少包含：

```json
{
  "schema_version": "canonical_batch_metadata.v1",
  "batch_id": "batch-20260716",
  "supplier_id": "supplier-001",
  "dataset_attributes": {
    "sensors": ["rgb", "force"],
    "cameras": ["head", "left_wrist", "right_wrist"],
    "robot_platform": "franka",
    "annotation_version": "supplier-annotation.v3",
    "languages": ["zh", "en"],
    "modalities": ["video", "hand_keypoints_3d", "force"],
    "coordinate_system": "camera",
    "time_sync": "timestamps_ns"
  }
}
```

属性必须由 `canonical_batch_metadata.v1` manifest 显式提供，禁止从目录名猜测。
CLI 通过 `--batch-metadata` 绑定 manifest、identity 和内容 hash；跨批次检索索引仍为
后续数据目录能力，不影响当前发布完整性。

## 3. Standardized Core 必需字段总表

`R` 表示供应商提交中必须存在；`A` 表示由 Adapter 从真实文件探测或计算，供应商不得伪造；`O` 表示可选 Evidence。

本节冻结的是当前跨供应商 Core，不是 Canonical Data 的字段全集。当前实现
`CanonicalQcEpisode` / `canonical_qc_episode.v1` 继续使用这些兼容标识；新增
extension 不得改变下列字段的既有含义。

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
| `source_fingerprint` | SHA-256 | A | 对源文件 hash、输入 schema、Adapter 版本和必填的 `main_video.source_frame_range` 计算；用于缓存失效和发布前复核。范围参数必须是非负、非空的严格整数半开 tuple，禁止 `None`、list 或 bool。 |
| `batch_metadata_ref` | string | O | CLI 的 `--batch-metadata` 指向 immutable `canonical_batch_metadata.v1`；不能从目录名推导。 |
| `dataset_attributes` | object | O | 传感器、相机、robot、annotation version、language、modality 等可索引属性；不参与 QC verdict。 |

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
| `main_video.frame_count` | int64 | A | 选中逻辑 span 的帧数，必须等于 `T`；共享 MP4 的完整物理帧数可以大于 `T`，但必须完整包含 `source_frame_range`。 |
| `main_video.source_frame_range` | int64 pair | A | MP4 中的物理半开区间 `[start,end)`；必须满足 `start>=0` 且 `end-start=T`。独立视频固定为 `[0,T)`，共享 LeRobot v3 视频必须保留 episode 的真实 offset。 |
| `main_video.width_px` | int32 | A | 从视频探测。 |
| `main_video.height_px` | int32 | A | 从视频探测。 |
| `main_video.fps_num/fps_den` | int64 pair | A | 从容器和 PTS 探测，不信任仅由供应商声明的 FPS。 |
| `main_video.codec` | string | A | 实际编码器。 |
| `main_video.pixel_format` | string | A | 实际像素格式。 |

逐帧视频 PTS 由 Video QC 直接读取并与 `time_axis.timestamps_ns` 比较，不要求供应商重复提交一份视频时间戳数组。

Canonical 内部及 QC JSON 始终使用逻辑帧号 `[0,T)`；`source_frame_range`
只用于读取共享 MP4 的物理帧。Runner 必须在 producer 边界将物理坐标一次性
转换回逻辑坐标，禁止把共享文件 offset 写入 QC metrics、issues 或 evidence。
物理区间参与 source fingerprint 和当前 v1 semantic fingerprint，避免同一共享
MP4 文件中不同 span 发生身份碰撞。
Bridge 在任何 Runner 读取前重新探测物理 MP4；若 `source_frame_range.end`
超过实际物理帧数，必须以输入完整性错误拒绝，禁止依赖解码器静默截断。

运行期信任边界：源 bucket 从 Adapter 建立 Canonical provenance 起，到该样本
全部 QC Runner 完成为止必须保持只读。系统在文件型 producer 前后复核来源的
size/SHA-256，用于发现误写和普通漂移；不会为每个 Runner 复制整份 MP4。
能够精确卡在两次复核之间替换文件并在复核前恢复的主动对抗写入者不属于首版
威胁模型，存储 ACL 必须在部署层阻止此类写入。

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

`egodata_hand21.v1` 的数组索引是冻结的序列化合同，不等同于按手指分组的
展示顺序。左右手都使用同一张表，仅由 hand 维度区分 side：

| index | Acceptance base name |
| ---: | --- |
| 0 | `Hand` |
| 1 | `IndexFingerKnuckle` |
| 2 | `IndexFingerIntermediateBase` |
| 3 | `IndexFingerIntermediateTip` |
| 4 | `MiddleFingerKnuckle` |
| 5 | `MiddleFingerIntermediateBase` |
| 6 | `MiddleFingerIntermediateTip` |
| 7 | `LittleFingerKnuckle` |
| 8 | `LittleFingerIntermediateBase` |
| 9 | `LittleFingerIntermediateTip` |
| 10 | `RingFingerKnuckle` |
| 11 | `RingFingerIntermediateBase` |
| 12 | `RingFingerIntermediateTip` |
| 13 | `ThumbKnuckle` |
| 14 | `ThumbIntermediateBase` |
| 15 | `ThumbIntermediateTip` |
| 16 | `ThumbTip` |
| 17 | `IndexFingerTip` |
| 18 | `MiddleFingerTip` |
| 19 | `RingFingerTip` |
| 20 | `LittleFingerTip` |

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

因为 `hand_keypoints_3d` 已统一到 `camera:main`，首版不要求静态或逐帧外参。
当前 Canonical Validator 只验证标定的 dtype、shape、有限性、图像尺寸与登记模型；
3D/2D 重投影阈值规则尚未实现，明确为 Deferred，不得把“字段已具备”解释为“已检查”。

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

## 4. Supplier extensions / evidence

Canonical Data 必须维护供应商额外字段 inventory。每项至少记录：

| 字段 | 规则 |
| --- | --- |
| `canonical_key` | 稳定、无碰撞的发布键。 |
| `source_path` | Raw 中的原始字段/attribute/column 路径。 |
| `dtype` / `shape` | 原始类型与 shape，不得为方便 QC 擅自缩窄。 |
| `unit` / `coordinate_frame` | 已知时显式登记；未知时写 unknown，不猜测。 |
| `time_alignment` | `frame`、`episode`、`batch` 或明确的外部时间键。 |
| `preservation` | `lerobot_feature`、`versioned_sidecar` 或 `unsupported`。 |

首版代码只实现 `quality_hand`，泛化 inventory、frame-aligned passthrough 和 sidecar
policy 为后续模块改造项。`unsupported` 必须阻止正式发布或依据已审查的版本化策略
隔离，不能静默删除。

### 4.1 可选供应商 Evidence：`quality_hand`

`quality_hand` 不是 Core 必填字段。供应商未提供时，不得因此判定资产 fail；Keypoint 和 SAM3 QC 仍然正常执行。

### 4.2 Canonical 结构

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

首版在单资产 QC JSON 的 `sam3_containment.metrics.supplier_hand_quality`
持久化以下统计。输入必须来自 SAM3 逐帧结果，并按逻辑
`(frame_idx, hand_side, camera_id="main")` 对齐；重叠窗口的相同 key 去重，
冲突结果 fail closed，`both` 不能作为逐帧 hand side：

- `good/pass` 为 TN，`good/fail` 为 supplier FN 并生成
  `supplier_mask_disagreement` Warn；
- `bad/pass` 为 supplier FP，仅记录统计；`bad/fail` 为 TP；
- 分母只包含 supplier `{good,bad}` 与 machine `{pass,fail}`；supplier
  `unknown/warning` 以及 machine `unavailable/review/skipped` 分别记录排除计数；
- `agreement_rate`、`supplier_false_positive_rate`、
  `supplier_false_negative_rate` 在分母为 0 时必须为 `null`，并始终保留对应
  numerator/denominator；
- 字段使用 `supplier_false_positive_*` / `supplier_false_negative_*`，避免与
  我方算法或人工判定的同名概念混淆。

本 Task 只冻结 per-asset QC JSON 合同。批次 projection 与 `by_supplier` rollup
将在统一聚合任务中从这些资产 JSON 计算，不在 SAM3 runner 内维护第二份账本。

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

Canonical 内存接口中的 `status` 是 `unknown/bad/warning/good` 语义枚举；Curated
LeRobot v3 Parquet 为兼容冻结的官方 reader，使用版本化 `uint8 [T,2]` wire 编码：
`0=unknown`、`1=bad`、`2=warning`、`3=good`。编码表写入
`meta/episode_semantics.jsonl`，Adapter 回读后恢复字符串枚举。该编码与供应商
`mapping_version` 是两层合同，不能混用，也不能按供应商裸数值猜测状态。
`raw_value` 的 numeric/bool 类型保留在帧 Parquet；string/bytes 使用版本化 semantic
sidecar 保存 dtype、`[T,2]` shape 与 utf8/base64 payload，由 Adapter 无损恢复，
避免官方 reader 对 fixed-size string list 的解码限制。

已是 LeRobot v3 的供应商数据也不能绕过 Gate 直接进入训练集。QC 通过后仍由我方
Publisher 重新生成 Curated LeRobot v3 的 meta、Parquet、索引、统计、checksum 和
ReleaseManifest；但重写不等于只复制 QC Core。Publisher 的 payload 来源是 Raw，
Canonical metadata 提供标准字段映射和 extension inventory，最终 QC report 只提供
Gate/audit，可选 revision artifact 只覆盖被批准的语义修改。供应商原有且 QC 未使用
的字段仍须保留。

正式输出冻结为官方 `lerobot[dataset]==0.6.0` v3 合同；供应商旧方言作为 Adapter
输入，由标准字段或版本化 extension sidecar 转换，不原样冒充官方字段。当前实现
证明 Core、`quality_hand`、已登记 extensions 和 batch attributes round-trip；
unsupported 字段/格式结构化拒绝。Publisher 保留 `fps_num/fps_den` 精确帧率，并把 Python、
NumPy、PyArrow、Pandas、ffmpeg、libx264 工具链指纹绑定到 manifest 和 release ID。
完整 `[0,T)` 且 frame count/PTS/hash 一致的 MP4 可独立复制；共享 span 必须裁剪，
禁止 hardlink 源文件。

## 8. 各 QC 阶段的字段依赖与实现状态

| 阶段 | 必需输入 | 当前状态 |
| --- | --- | --- |
| Source/Schema Gate | manifest identity、source locator、全部 Core shape/dtype/unit。 | Implemented + tested；Pass/Fail/runtime 均写唯一 QC JSON。 |
| Text/Metadata | task、description、subtask sequence。 | Implemented + tested。 |
| Keypoint Presence | 3D/2D keypoints、3D/2D validity。 | Implemented + tested。 |
| Keypoint Morphology | 3D keypoints、3D validity、joint topology、meter。 | Implemented + tested。 |
| Keypoint Temporal | timestamps、3D keypoints、3D validity。 | Implemented + tested。 |
| Video Quality | MP4、Canonical 时间轴、frame count。 | Implemented + tested。 |
| SAM3 Containment | MP4、2D keypoints、2D validity；mask 由我方生成。 | 接口/结果桥接已实现；模型依赖按运行环境提供。 |
| Reprojection Validation | 3D/2D keypoints、validity、intrinsics、distortion。 | **Deferred**；没有版本化阈值与判定实现。 |
| 人工语义校准 | MP4、subtask sequence、半开共享边界、双语描述。 | **Deferred external change**；当前 runner 只停在 `awaiting_external`。 |
| Warn 人工质检 | QC JSON issues/evidence；不新增供应商 Core 字段。 | **Deferred external change**。 |
| Duplicate Check | MP4、asset ID、timestamps。 | **Deferred**；active config 为 disabled / `no_registered_implementation`。 |
| Content Validity | MP4、task/description/subtasks。 | **Deferred**；active config 为 disabled / `no_registered_implementation`。 |
| Effective Duration | timestamps 及前序 QC issue 区间。 | **Deferred**；active config 为 disabled / `no_registered_implementation`。 |
| LeRobot v3 Publisher | Raw source、Canonical metadata/field inventory、最终 QC pass、optional revision artifact。 | Core、Evidence、已登记 extensions、typed batch metadata、非零文本/共享边界 revision implemented + tested；unsupported Raw 类型/格式 fail closed。 |

因此本标准列出的 Core 已足够作为目标 QC 的输入合同，但不等于训练数据全集；上表
Deferred 阶段不能宣称已有检测或发布覆盖。无需把 `quality_hand`、供应商 mask、
joint rotation、confidence 或多相机强制提升为 Core，但必须通过 supplier extension
机制保留/登记。

## 9. Adapter 规则

两个 Adapter 必须遵守同一行为：

1. 只读取固定字段，不搜索别名。
2. 不使用默认 FPS 代替缺失值。
3. 不把 validity 当成 `quality_hand`。
4. 不根据 `quality_hand` 数值大小猜测语义。
5. 不静默跳过非法 shape、dtype 或 joint。
6. 不截断不同长度的数组来伪造对齐。
7. 不在 Adapter 内形成 QC Pass/Fail。
8. 输出不可变 Canonical Data view；当前兼容对象为 `CanonicalQcEpisode.v1`。
9. 同一逻辑 episode 且视频物理 span 相同的 HDF5 与 LeRobot 输入，除 source provenance 外必须产生相同 Canonical 语义 fingerprint；不同 `source_frame_range` 在当前 v1 中视为不同视频身份。
10. 枚举未进入 Core 的 Raw 字段并形成 extension inventory；标准 HDF5/LeRobot 的
    已支持 dtype/shape 必须 round-trip，unsupported 字段结构化拒绝。
11. 绑定 batch manifest / dataset attributes，不从路径猜测批次能力。

### 9.1 统一 QC CLI 的 Canonical manifest 合同

`tools/run_canonical_qc.py` 必须显式接收 `--asset-id`、`--batch-id`、
`--supplier-id`。三者来自批次 manifest/调用方，不从路径或供应商内容猜测；Adapter
成功后必须与源内 identity 精确相等。这样即使 Core 已损坏，仍能先确定
`quality_archive/<asset_id>.json` 并原子写 Source Gate Fail。

`tools/run_qc_pipeline.py` 通过 manifest 行启用 Canonical 输入，固定字段如下：

| 字段 | 必需 | 规则 |
| --- | --- | --- |
| `asset_id` | 是 | 必须与 Adapter 解析出的 episode `asset_id` 完全一致。 |
| `batch_id` | 是 | 调用方提供的稳定批次 ID；必须与 Adapter 结果一致。 |
| `supplier_id` | 是 | 调用方提供的稳定供应商 ID；必须与 Adapter 结果一致。 |
| `canonical_format` | 是 | 仅允许 `hdf5` 或 `lerobot`。 |
| `canonical_source_path` | 是 | 位于 `batch_root` 内的标准 HDF5 episode 路径或 LeRobot dataset 根目录。 |
| `episode_index` | LeRobot 多 episode 时是 | 必须精确选择一个 episode；不允许默认取第一条。 |
| `start_frame` / `end_frame_exclusive` | 否 | Canonical manifest 直接使用逻辑半开区间 `[start_frame,end_frame_exclusive)`，必须成对出现；禁止使用 legacy 的 inclusive `end_frame`。 |
| `candidate_windows_path` | SAM3 时是 | supplemental source；必须位于 `batch_root` 内。 |
| `sam3_model` / `sam3_model_path` | 无注入模型时是 | supplemental source；必须位于 `batch_root` 内。 |
| `manifest` 及其他既有 runner 输入 | 按模块 | 作为 supplemental source 保留；Canonical 生成的 `video`、`hdf5/parquet` 和 provenance 保留键拥有最终优先级。 |

入口会在 worker 前严格加载 Adapter、验证所有 provenance 文件，并把完整、纯 JSON
的 Canonical provenance 写入 `AssetContext.source_files`。不得在 manifest 中放置
Python 对象或覆盖 Canonical 保留键。

非 Canonical legacy manifest 为兼容既有调用仍使用 `start_frame/end_frame`
闭区间，并仅在 legacy 入口执行一次 `end_frame + 1`。两个合同按
`canonical_format/canonical_source_path` 显式分支，禁止在同一行混用。

## 10. 错误与最终状态

| 情况 | 分类 | 处理 |
| --- | --- | --- |
| Core 字段缺失 | 数据合同缺陷 | 写 Source Gate fail issue，进入供应商失败统计。 |
| shape/dtype/单位错误 | 数据合同缺陷 | fail，不进入依赖该字段的 QC。 |
| 时间戳不递增 | 数据合同缺陷 | fail。 |
| 视频与 `T` 不对齐 | 数据合同缺陷 | fail。 |
| 3D/2D 重投影严重不一致 | Deferred | 当前只校验标定合同；重投影质量规则尚未实现，不生成结论。 |
| Subtask 不连续或越界 | 数据合同缺陷 | fail；不得由 UI 自动猜测修复。 |
| `quality_hand` 未提供 | 正常可选缺失 | 不 fail，status 视为 unknown。 |
| `quality_hand` 与机器结果冲突 | 供应商 Evidence 分歧 | 生成 Warn/统计，不直接覆盖机器结论。 |
| 文件暂时无法读取、网络或权限错误 | Runtime Error | `overall_decision=null`，可重试。 |
| QC report revision 冲突 | CAS Error | 拒绝旧写入，重新读取。 |
| Publisher 验证失败 | Publish Error | 不暴露新 release，原 QC JSON 不变。 |

上述 Source Gate 持久化保证从“显式 identity、路径和配置均验证通过”后开始；CLI
参数缺失、unsafe path 或配置本身不可验证时没有可信报告目标，只输出单行 CLI 错误。

## 11. 人工语义修订与发布

为同时支持 HDF5 和 LeRobot 输入，人工语义修订不依赖某一种源格式：

Publisher 的逻辑输入关系为：

```text
Raw source
+ Canonical metadata / field inventory / batch attributes
+ final asset_qc_report.v2
+ optional Canonical revision artifact
-> Curated LeRobot v3
```

具体规则：

1. 人工确认一次时间边界或文本修改。
2. 目标合同要求 format-neutral Canonical working revision artifact 通过 CAS 原子更新。
3. QC JSON 记录 `timeline_edit_count`、`subtask_text_edit_count` 及 before/after 审计，
   但不承载修订后的训练 payload。
4. 源 HDF5/LeRobot/MP4 保持只读。
5. Publisher 从 Raw 读取完整 payload，通过 Canonical metadata 进行字段映射，并将最新
   artifact 只应用到允许修订的语义/时间轴字段。
6. Release manifest 绑定 Raw fingerprint、field inventory、QC revision、artifact hash
   和应用后的 semantic fingerprint。

第 2、5、6 步已由 `canonical_revision_artifact.v1`、纯函数 patch 应用和
ReleaseManifest hash 绑定实现。现有 path Publisher 从 Raw 重新加载 Canonical Data；
任一 edit count 非零但缺少 artifact 时返回 `canonical_revision_artifact_required`，
禁止把 raw source 的旧语义发布出去。

只有 `overall_decision=pass`、语义阶段完成、Warn 复核完成且 source fingerprint 未变化的资产可以发布。

## 12. 验收清单

- [x] 供应商提交格式只能是标准 HDF5 或受支持的标准 LeRobot。
- [x] 所有 Core 字段存在且符合固定名称、shape、dtype、单位和坐标系。
- [x] 主视频可解码且与 `T`、FPS、时间轴一致。
- [x] 3D/2D 骨骼与 validity 对齐。
- [x] joint topology 严格等于 `egodata_hand21.v1`。
- [x] 3D 点位于 `camera:main`，2D 点使用 pixel。
- [ ] 重投影质量规则（Deferred；当前只验证标定合同）。
- [x] Subtask 使用半开共享边界并完整覆盖 `[0,T)`。
- [x] `quality_hand` 缺失不会触发 fail。
- [x] `quality_hand` 原值、标准化状态和我方 Mask 结果互不覆盖。
- [x] HDF5 与 LeRobot 的 Canonical 等价性测试通过。
- [x] QC 通过后只使用我方 LeRobotV3Publisher 生成训练数据。
- [x] Publisher 未完成验证前，训练 release 不可见。
- [x] Supplier extension inventory 与已支持 Raw 额外字段无损发布；unsupported 类型 fail closed。
- [x] Batch metadata / dataset attributes typed contract 与发布绑定（跨批次检索索引另行实现）。
- [x] 非零人工语义编辑的 format-neutral revision artifact、CAS 与 manifest hash 绑定。
