---
role: technical-design
field-contract: docs/canonical-qc-required-fields-v1.md
base-ref: a77a8a4
---

# Canonical QC 接入、桥接与 LeRobot v3 发布技术设计

## 1. 目标

本设计实现两种固定供应商输入到同一 QC 与训练发布链路：

```text
标准 HDF5 ──> StandardHdf5Adapter ──┐
                                     ├─> CanonicalQcEpisode.v1
标准 LeRobot ─> StandardLeRobotAdapter ┘
                                                │
                                                v
                                       CanonicalQcBridge
                                                │
                                                v
                             现有 QC Orchestrator / ModuleResult
                                                │
                           自动 Gate 通过 + 人工阶段完成
                                                │
                                                v
                                      LeRobotV3Publisher
                                                │
                                                v
                                   Curated LeRobot v3 Release
```

核心目标是消除供应商脚本差异。Adapter 只解析两种标准交付合同；QC 只消费 Canonical；Publisher 只消费最终、经过人工修订的 Canonical revision。源 HDF5、源 LeRobot 和源 MP4 始终只读。

## 2. 首版边界

首版固定为 `human_ego_hand_pose.v1`：单路 `main` 外置 MP4、双手各 21 个 3D/2D 点、逐点 validity、权威纳秒时间戳、主相机内参、双语任务和连续 subtask。

本版不实现多相机、Robot Teleop state/action、depth、音频、供应商 mask 或自定义字段映射。扩展必须增加 profile/schema 版本，不得改变 `canonical_qc_episode.v1` 已发布字段的含义。

## 3. 模块边界与目录

```text
canonical_qc/
  __init__.py
  contracts.py              # 不可变 Canonical 数据合同
  errors.py                 # Adapter/validator 的结构化错误
  validation.py             # shape/dtype/单位/时间轴/语义强校验
  provenance.py             # hash、source manifest、语义 fingerprint
  video_probe.py            # ffprobe 元数据与 PTS
  adapters/
    __init__.py
    base.py                 # SourceAdapter Protocol 与 registry
    standard_hdf5.py        # 固定 HDF5 合同
    standard_lerobot.py     # 固定 LeRobot v2.1/v3 合同
  bridge.py                 # Canonical -> 旧 QC 所需只读视图

lerobot_v3_publisher/
  __init__.py
  contracts.py              # PublishRequest/Plan/Result/Manifest
  layout.py                 # 受控目录与文件命名
  writer.py                 # meta、Parquet、视频物化
  validation.py             # 独立回读与时间轴验证
  publisher.py              # staging -> validate -> atomic commit

tools/
  run_canonical_qc.py       # 标准输入接入现有 QC
  publish_lerobot_v3.py     # 通过报告的受控发布入口
```

`canonical_qc` 可以读取 HDF5、Parquet 和视频，但不得加载 SAM3、DA3 或 VLM。`bridge.py` 不写 QC JSON；报告写入仍由 `qc_common.report_mutation` 独占。Publisher 不解释 QC 算法，只检查发布前置条件和数据完整性。

## 4. CanonicalQcEpisode.v1

### 4.1 不可变对象

```python
@dataclass(frozen=True)
class CanonicalQcEpisode:
    schema_version: Literal["canonical_qc_episode.v1"]
    profile: Literal["human_ego_hand_pose.v1"]
    identity: EpisodeIdentity
    provenance: SourceProvenance
    time_axis: TimeAxis
    main_video: VideoStream
    observation: HandObservation
    calibration: CameraCalibration
    semantics: EpisodeSemantics
    supplier_evidence: SupplierEvidence
```

数组在构造后设置为只读。对象不得用 `setattr` 携带供应商临时字段。所有 Core 校验通过后才允许返回 Canonical；错误通过 `CanonicalInputError` 暴露，不返回部分对象。

### 4.2 校验顺序

1. schema/profile/identity；
2. source 文件存在、位于 source root 内、hash 与 size；
3. `T`、FPS 有理数和严格递增 `timestamps_ns`；
4. 3D/2D/validity shape 和 dtype；
5. valid 点 finite、invalid 点 NaN；
6. topology、单位和坐标系常量；
7. 视频实际 frame count、尺寸、PTS 与 Canonical 时间轴；
8. 标定尺寸与视频一致；
9. subtask 半开共享边界连续覆盖 `[0,T)`；
10. 可选 `quality_hand` 仅校验其自身合同，不参与 Core Gate。

Validator 不截断数组、不补默认 FPS、不重新生成时间戳，也不自动修补 subtask。

### 4.3 两种 fingerprint

- `source_fingerprint`：输入 schema、Adapter 版本、所有源文件相对路径/size/hash，以及必填的物理 `main_video.source_frame_range` 的稳定 JSON SHA-256。范围必须是非负、非空的严格整数半开 tuple，不能省略或传 `None`/list/bool。用于检测源文件漂移并防止共享 MP4 的 span 身份碰撞。
- `semantic_fingerprint`：忽略 `source_format` 与路径，仅对 Canonical Core 值、时间轴、标定和语义计算。用于证明同一 episode 的 HDF5 与 LeRobot 归一结果等价。

浮点数组按固定 dtype、shape、C-order bytes 计算；JSON 字符串使用 UTF-8、排序键和禁止 NaN 的稳定编码。

## 5. SourceAdapter

### 5.1 统一接口

```python
class SourceAdapter(Protocol):
    adapter_id: str
    adapter_version: str

    def inspect(self, source: Path) -> SourceInspection: ...
    def load(self, source: Path) -> CanonicalQcEpisode: ...
```

Registry 只注册：

```text
standard_hdf5 -> StandardHdf5Adapter
standard_lerobot -> StandardLeRobotAdapter
```

自动识别只依据明确标识：HDF5 root `schema_version`，或 LeRobot `meta/info.json` 与固定目录结构。无法唯一识别时返回 `format_unsupported`，不按文件内容猜供应商。

### 5.2 StandardHdf5Adapter

输入为 episode 目录或 `.h5` 路径。`.h5` 必须与同目录 `main.mp4` 关联。Adapter 严格读取字段标准中的固定 attributes/datasets，解析 `/semantics/annotation_json`，探测视频，构造 provenance，然后调用共享 Validator。

不得复用当前 `precheck.adapters.supplier_hdf5` 的宽松字段搜索；该 adapter 继续服务 legacy 输入，新标准入口使用独立实现。

### 5.3 StandardLeRobotAdapter

输入为 LeRobot dataset root 和唯一 episode selector。首版读取：

- `meta/info.json`；
- `meta/episodes/chunk-*/file-*.parquet`；
- `data/chunk-*/file-*.parquet`；
- `meta/episode_semantics.jsonl`；
- `videos/main/chunk-*/file-*.mp4` 或由 episode metadata 明确指向的固定相对路径。

Adapter 可处理已登记的 v2.1/v3 布局差异，但 feature 名、shape 和语义固定。输入中 `timestamp` 只用于交叉检查；Canonical 权威值必须来自精确 `timestamp_ns`。缺少 `timestamp_ns` 直接拒绝，不能从 float 秒恢复纳秒时间轴。

Reader 同时保留供应商已登记的 `egodata_lerobot_qc_input.v1` 旧方言兼容，并读取
Publisher 生成的官方 v3.0 布局。Publisher 不输出二者的 hybrid：正式输出以官方
`lerobot==0.6.0`（tag `v0.6.0`，commit
`30da8e687a6dfc617fcd94afc367ac7071c376ce`）合同为准，使用
`{chunk_index}/{file_index}` path placeholder、逐帧全局 `index`、episode video
`from_timestamp/to_timestamp` 和带 `task` index 的 `meta/tasks.parquet`。项目扩展
`timestamp_ns/subtask_index` 仍登记在 features 并保持严格类型。

## 6. 时间戳与视频对齐

### 6.1 权威来源

`time_axis.timestamps_ns` 是唯一权威时间轴。FPS 只是名义属性，不能用于重建逐帧时间。Publisher 的 float `timestamp` 由下式派生：

```text
timestamp = (timestamps_ns[i] - timestamps_ns[0]) / 1_000_000_000
```

### 6.2 视频 PTS

`ffprobe` 读取每帧 best-effort timestamp；以第一帧 PTS 为零点归一化后与 Canonical 相对时间比较。Validator 至少检查：

- 可读取 PTS 数量与 `T` 相同；
- PTS 严格递增；
- 每帧差值不超过配置的 `max_timestamp_delta_ns`；
- 实际视频 frame count、尺寸与声明一致。

ffprobe 不可用、PTS 数量不足或偏差超限属于 `timebase_invalid`/`source_integrity_error`，首版 fail closed。不能用 OpenCV 的 `frame_idx / fps` 伪装通过。

## 7. CanonicalQcBridge

Bridge 只做确定性视图转换：

```python
class CanonicalQcBridge:
    def asset_context(self, episode, *, batch_root, report_path) -> AssetContext: ...
    def clip_inputs(self, episode, source_range=None) -> ClipInputs: ...
    def video_path(self, episode) -> Path: ...
    def semantic_payload(self, episode) -> Mapping[str, Any]: ...
```

### 7.1 与现有管线的连接

- `AssetContext.metadata["canonical_episode"]` 保存进程内只读引用；序列化身份仍只依赖 source manifest。
- `qc_pipeline.runners.precheck._load_clip()` 优先从 Bridge 获取 `ClipInputs`。
- Video/SAM3 runner 从 Bridge 获取同一 `main.mp4` 和统一 source range。
- 旧检测器输出仍经现有 adapter 转成 `ModuleResult`。
- `apply_module_result()` 继续负责 issue 所有权、revision CAS 和 QC JSON 原子写入。

Bridge 是半开区间和旧闭区间之间唯一转换层。Canonical 内部永远使用 `[start,end_exclusive)`；旧 evidence 若要求 inclusive end，Bridge 显式写 `end_frame=end_exclusive-1` 和 `coordinate_system=source_inclusive`。

统一 QC CLI 的 Canonical manifest 也直接使用 `start_frame` 与
`end_frame_exclusive`；legacy manifest 才允许 `start_frame/end_frame` 并在入口
执行一次 `+1`。Canonical 行出现 inclusive `end_frame` 必须拒绝，禁止按调用方
猜测区间语义。

### 7.2 Keypoint 视图

`ClipInputs.keypoints` 使用仓库现有 `left/right + acceptance joint name` 字典。Bridge 从 `[T,2,21,3]` 按 `egodata_hand21.v1` 的冻结索引映射，不改变数值。Invalid joint 保持 NaN；validity 作为独立 canonical metadata 供新 presence check 使用，不得伪装为 `quality_hand`。

供应商 hand quality 在 legacy `ClipInputs.quality_hand` 中不再作为机器 validity。首版 Bridge 只通过新 `supplier_hand_quality_status` 只读属性暴露，机器 Keypoint/SAM3 Gate 不消费其裸数值。

Canonical `quality_hand` 模块只验证并记录 enum Evidence，不根据 raw/score 推断，
也不形成质量 Gate。Canonical Keypoint Presence 独立消费
`hand_joint_valid_3d` 与 finite `hand_keypoints_3d`；legacy numeric
`quality_hand` 路径保持兼容。

供应商/机器一致性属于后置 SAM3 block。Runner 必须读取逐帧 containment rows，
按逻辑 `(frame_idx, hand_side, camera_id="main")` 对齐并对重叠窗口去重；
`hand_side=both` 或同 key 冲突必须 fail closed。`good/fail` 生成
`supplier_mask_disagreement` Warn，`bad/pass` 只累计 supplier false positive。
supplier `unknown/warning` 和 machine `unavailable/review/skipped` 不进入一致率。
每资产统计写入 QC JSON；批次 `by_supplier` 汇总从资产 JSON 派生，不由 Runner
维护第二份状态。

## 8. QC Gate 与人工阶段

编排调用方必须在 Adapter 前提供稳定 `asset_id/batch_id/supplier_id` 与 source
locator。Source Gate 先通过现有 report CAS 原子建立该资产的
`asset_qc_report.v2`：Pass revision 进入首个自动模块；数据合同/Adapter 失败形成
`stopped/fail` issue；runtime I/O 形成 `overall_decision=null` 的可重试 error。恢复后
同一报告 CAS 前进，旧 runtime 进入 `execution.runtime_error_history`。CLI 参数、unsafe
path 或配置自身不可验证时尚无可信报告目标，只输出机器错误。进入现有自动 QC 后
沿用两种 profile：

- `acceptance`：自动 fail 立即停止，不做语义校准；自动全 pass 且无 warn 时完成语义校准后无需 Warn 人工质检；存在 warn 才进入人工 Pass/Fail。
- `supplier_evaluation`：机器 fail 仍继续执行后续阶段，以便评估全流程，但最终结论保留 fail。

人工语义时间轴使用共享半开边界；一次边界确认原子影响相邻两段并只计一次。目标
合同由 format-neutral Canonical working revision artifact 连接只读 HDF5/LeRobot 与
Publisher。该 artifact 接口由人工 change 后续实现；当前 path Publisher 在任一 edit
count 非零时以 `canonical_revision_artifact_required` fail closed，禁止发布 raw source
中的旧语义。

## 9. LeRobotV3Publisher

Canonical 内存中的 `supplier.hand_quality.status` 保持
`unknown/bad/warning/good` 语义枚举。Curated LeRobot v3 的 Parquet wire contract
使用官方 reader 可解码的 `uint8 [T,2]`：`0/1/2/3` 分别对应上述四种状态；
`meta/episode_semantics.jsonl` 同时保存固定
`supplier_hand_quality_status.v1` 编码表和供应商 `mapping_version`。Adapter 必须按
编码版本还原语义枚举，不能按数值猜测。`normalized_score` 固定为 `float32 [T,2]`，
`raw_value` 保留受支持的原始 primitive dtype；未提供 Evidence 时不得生成这些
Parquet 列、info feature 或 encoding sidecar。

`raw_value` 的 numeric/bool primitive 保持逐帧 Parquet 列；string/bytes 因冻结的
官方 reader 无法可靠解码 fixed-size string list，改写入
`supplier_hand_quality_raw_value.v1` semantic sidecar。Sidecar 绑定 numpy dtype、
`[T,2]` shape 和 `utf8/base64` 编码，Adapter 必须无损恢复；不得因此缩窄
Canonical 已支持的 raw Evidence 类型。

### 9.1 前置条件

```python
@dataclass(frozen=True)
class PublishRequest:
    episode: CanonicalQcEpisode
    canonical_revision: int
    canonical_source_root: Path
    qc_report_path: Path
    expected_report_revision: int
    release_root: Path
```

Publisher 必须验证：

- 报告为 `asset_qc_report.v2`；
- `overall_decision=pass` 且 `pipeline_state.status=completed`；
- 语义阶段完成，Warn 复核已完成或 `not_required`；
- report 的 `canonical_binding.v1` 冻结 canonical revision、semantic fingerprint、
  source fingerprint 和唯一 final QC report revision，并与请求及 episode 三方一致；
- report 的 `canonical_qc_range` 必须是完整 `[0,T)` 半开区间；
- report source fingerprint 与 episode 一致；
- report revision 与请求一致；
- `acceptance` profile、immutable QC config snapshot/hash、所有启用 module state、
  issues、runtime errors、semantic consistency 和 manual review 状态组合一致；每个
  enabled automatic module 都必须有顶层 result block 和完整 flow，result Gate 不得
  fail，且 entry/result/exit Gate、`evaluation.decision`、module state、配置中的下一
  模块和最终 cursor 必须逐项一致；`stop_qc`、缺字段或 evaluation/result 分歧均
  fail closed；
- clean skip 使用显式模块合同：`quality_hand` 只允许未提供 optional Evidence，
  `sam3_containment` 只允许零候选窗口；其他 skipped、`skipped_due_to_fail` 和
  runtime error 均不可发布；
- Warn 人工复核只消费正式 `manual_review.reviews[]` 合同；每条记录包含
  `review_id/issue_id/reviewer/reviewed_at/verdict/asset_action/comment/evidence_paths`，
  候选必须恰好覆盖一次，且只有最终 `asset_action=accept/accept_with_risk` 可发布；
- 通过显式 `canonical_source_root` 重新读取当前源文件，拒绝相对路径、escape、
  symlink 和 hash 漂移；`release_root` 不得与 source root 重叠或与 report/source
  别名；不得从 `qc_report_path.parent` 猜源目录。

`validate_publish_request` 只读报告和源文件，并在 `PublishPlan` 冻结 QC report
SHA-256 与 strictly typed frozen source snapshot。报告和源文件都使用同一个
`O_NOFOLLOW` file descriptor 做 pre/post `fstat` 与 hash，hash 期间发生原地修改或
路径替换必须拒绝。源文件只流式更新 SHA-256，不缓存文件 payload；只有 QC report
允许捕获内容，且上限为 16 MiB。Writer、独立验证器和 commit 前都必须调用 plan
revalidation；同 revision 内容变化或当前源 hash 漂移必须拒绝。

### 9.2 输出布局

首版每次发布一个不可变 release：

```text
<release_root>/releases/<release_id>/
  meta/info.json
  meta/episodes/chunk-000/file-000.parquet
  meta/tasks.parquet
  meta/subtask.parquet
  meta/stats.json
  meta/episode_semantics.jsonl
  data/chunk-000/file-000.parquet
  videos/observation.images.main/chunk-000/file-000.mp4
  release_manifest.json
  checksums.sha256

<release_root>/CURRENT.json
```

`release_id` 由 asset、canonical revision、semantic fingerprint 和 publisher version
四元组稳定派生。canonical revision 不能由调用方任填：它必须绑定到报告中的
唯一 final QC revision；同一 canonical revision 不允许换一个 final QC revision
重新发布，如需重跑必须产生新的 canonical revision。重复发布同一请求返回已有
release；同 ID 内容不同属于 `commit_conflict`。

`publisher_version` 必须包含会影响产物字节的工具链指纹。首版指纹至少覆盖
Python、NumPy、PyArrow、Pandas 版本，完整 `ffmpeg -version/-buildconf` 输出摘要和
libx264 encoder signature；完整工具链身份写入 manifest。不同工具链不得复用同一个
release ID。

目录和基础 metadata 必须兼容官方 LeRobotDataset v3，而不仅是本仓库旧 reader。官方 v3 使用 file-based shards、`meta/episodes` 关系元数据、`meta/tasks.parquet`、`meta/stats.json`、`data/chunk-*/file-*.parquet` 和 `videos/<camera_key>/chunk-*/file-*.mp4`。`meta/episode_semantics.jsonl`、`meta/subtask.parquet` 和 `release_manifest.json` 是本项目在官方可扩展字段之外增加的审计/人工语义 sidecar；它们不得替代官方必需 metadata。

逐帧 Parquet 的 `timestamp` 固定为 float64，且只能由
`(timestamps_ns[i]-timestamps_ns[0])/1e9` 派生；这项项目精度合同高于官方默认
float32 feature，避免长 episode 静默丢失 1 ns 对齐精度。官方 v3 reader 支持该
登记 dtype；`meta/info.json` 额外登记精确的 `fps_num/fps_den`，禁止只从浮点 FPS
反推 NTSC 等有理数帧率。Task 9 仍必须用冻结版本 reader 做最终门禁。

`meta/stats.json` 覆盖逐帧表中的 index、时间、骨骼、validity、task/subtask index
以及实际发布的数值型供应商 hand quality Evidence。骨骼 min/max/mean/std 必须按
对应 validity 排除无效点；bool 另写 true_count。timestamp/timestamp_ns 只登记
count/range，不作为归一化统计。首版不计算视频像素统计，也不得伪造视频统计。

### 9.3 写入策略

1. 在 `<release_root>/.staging/<transaction_id>` 创建全新目录；
2. 从 Canonical 写逐帧 Parquet、episode metadata、semantics 和 manifest；
3. 禁止 hardlink 源 MP4；完整 `[0,T)`、物理文件恰好 T 帧且 PTS 对齐时，才可从
   `O_NOFOLLOW` fd 独立复制。共享/偏移 span 必须裁成恰好 T 帧并把 PTS 归零；
   manifest 必须记录源相对路径、源半开帧区间、source/target hash 和
   `verified_copy` 或 `transcoded_frame_range`；
4. 生成全文件 checksum；
5. 使用独立 reader 从 staging 重新打开全部产物；
6. 验证 schema、行数、frame index、timestamp、数组值、subtask、视频 PTS 和 hash；
7. fsync 文件及目录；
8. 原子 rename staging 为不可变 release；
9. 最后以临时文件 + `os.replace` 原子更新 `CURRENT.json`。

Task 8 只写 `<release_root>/.staging/<unique_tx>` 的全新同文件系统直属子目录，不碰
`releases/` 或 `CURRENT.json`。manifest 登记全部 payload；`checksums.sha256` 覆盖
payload 加 manifest，仅排除自身。失败只清理本 transaction，不得删除其他 staging。
任何产物不得嵌入 wall-clock、绝对 staging path 或 transaction ID；同一 Plan 的两次
staging 必须逐文件字节一致。

staging 根、transaction 目录和其父目录必须通过持有的 directory fd 与
`openat`/`O_NOFOLLOW` 操作；文件以 `O_EXCL` 创建。祖先路径或目标文件被替换为
symlink 时必须失败，且不得写入外部路径。ffmpeg timeout 按片段时长扩展并设硬上限。

任何第 1–8 步失败都不能修改 `CURRENT.json`。已有 release 不覆盖、不原地修改。训练程序读取 `CURRENT.json` 或明确 release ID，不依赖固定 HDF5 路径。

### 9.4 独立回读验证

验证器不能复用 writer 内存对象作为真值。它必须重新读取实际文件并确认：

- frame rows 恰为 `T`，`frame_index == range(T)`；
- `timestamp_ns` 与 Canonical 字节级相同；
- float `timestamp` 误差在 1 ns 等价容差内；
- keypoints/validity shape、dtype、值相同；
- 每帧 `subtask_index` 与半开边界一致；
- episode semantics 与最终人工修订一致；
- MP4 hash、frame count、尺寸、PTS 对齐；
- manifest 列出的文件均存在且 checksum 匹配；
- 不存在未登记文件或绝对路径。

验证最后必须使用固定版本的官方 `lerobot.datasets.LeRobotDataset` 再回读一次。只有本仓库 `annotation.LeRobotV3Dataset` 可读而官方 reader 不可读时，发布必须失败，不能标记为 Curated LeRobot v3。

官方 reader 门禁运行在离线独立子进程中，冻结 LeRobot、datasets、Torch、视频解码
依赖以及 Python/数值栈/平台/backend 身份，并把版本集合及 fingerprint 写入验证报告。
子进程必须实际读取第 0 帧和末帧，返回并由父进程复核 length、index、frame_index、
timestamp_ns 和视频 tensor shape；timeout、signal、错误退出或不完整响应均失败。

原子提交持有 release_root、`.staging`、`releases` 的 nofollow directory fd 和发布锁。
所有文件/子目录 fsync 后必须再做一次完整独立验证和 plan revalidation，随后立即执行
dirfd no-replace rename；rename 后 fsync staging/releases/root，最后用 O_EXCL 临时文件、
fsync、`os.replace` 更新 CURRENT。已有同 ID 只有独立验证完全相同时才幂等；无效或
不同内容为 commit_conflict。rename 后 CURRENT 前中断留下的是完整孤儿 release，重试
必须验证、补齐 durability 和 CURRENT 后返回 already_published。

## 10. 错误模型

```python
@dataclass(frozen=True)
class CanonicalDiagnostic:
    code: str
    stage: str
    field: str | None
    message: str
    retryable: bool
```

稳定错误码：

| 错误码 | 类别 | 结果 |
| --- | --- | --- |
| `format_unsupported` | 输入合同 | Source Gate fail |
| `schema_missing` | 输入合同 | Source Gate fail |
| `field_mapping_error` | 输入合同 | Source Gate fail |
| `timebase_invalid` | 输入合同 | Source Gate fail |
| `source_integrity_error` | 数据/运行 | hash 漂移为 fail；临时 I/O 为 error |
| `canonical_revision_artifact_required` | 发布前置 | 有人工编辑但无 format-neutral revision artifact，不发布 |
| `publish_prerequisite_failed` | 发布前置 | 不发布 |
| `staging_failed` | 发布运行 | 清理 staging，CURRENT 不变 |
| `validation_failed` | 发布完整性 | 隔离 staging，CURRENT 不变 |
| `commit_conflict` | 并发/幂等 | 拒绝覆盖，重新读取 |

异常对象必须携带 diagnostics，CLI 将其写成机器可读 JSON；不得只打印 traceback，也不得把 runtime error 伪装成数据 fail。

## 11. 配置

新增版本化 `configs/canonical_qc/canonical_qc_v1.0.0.yaml`，只包含非语义阈值和实现版本：

```yaml
schema_version: canonical_qc_config.v1
config_version: canonical_qc_v1.0.0
profile: human_ego_hand_pose.v1
adapter_versions:
  standard_hdf5: 1.0.0
  standard_lerobot: 1.0.0
video_alignment:
  max_timestamp_delta_ns: 1000000
publisher:
  format_version: lerobot_v3_curated.v1
  chunk_size_episodes: 1000
```

实现收尾新增 `canonical_qc_v1.1.0`，只扩展 pre-pipeline Source Gate 的版本化
rule registry；旧 v1.0.x snapshot 保持不可变。Source Gate fail issue 的 `rule_id`
必须从该 registry 读取，不能硬编码未登记规则。

供应商 `quality_hand` 映射不写入全局阈值。若提供，输入必须已经同时给出明确 `status` 与 `mapping_version`；首版 Adapter 不猜 raw value 到 status 的映射。
活动 QC 配置从 immutable `qc_acceptance_v2.1.0` 读取此 enum 合同和分歧 Warn
rule；`v2.0.0` 历史快照保持字节不变，numeric `0/1` 规则只保留在明确命名的
legacy compatibility block。

## 12. 验收标准

1. 合法 HDF5 和等价 LeRobot 得到相同 semantic fingerprint。
2. 任一 Core 字段缺失、shape/dtype/单位错误均显式拒绝。
3. `quality_hand` 缺失不会导致 fail，validity 不会被当作 quality。
4. float timestamp 或 FPS 不能替代 `timestamps_ns`。
5. 视频 PTS/帧数不对齐不能进入 QC 或发布。
6. Bridge 可运行现有关键点、视频、SAM3 runner，报告仍由统一 CAS 写回。
7. 只有最终 pass 的 QC revision 可发布。
8. Publisher 输出能由独立 reader 完整回读并与 Canonical 比对。
9. 注入任一步失败时旧 `CURRENT.json` 与旧 release 保持字节不变。
10. 全量现有测试和新增 E2E 测试通过。
