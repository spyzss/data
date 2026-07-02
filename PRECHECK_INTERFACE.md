# Precheck 开发与接口规范

本文档记录当前蓝圈范围的数据准入口检查能力，以及 `precheck/` 与 `qc_common/` 的接口约定。目标是让蓝圈检查、红圈拉取/画质模块、后续人工质检和批次台账可以按统一契约对接，避免把流程顺序硬编码进任一 root module。

## 1. 蓝圈范围当前已完成

| 流程图模块 | 当前实现 | 状态 | 说明 |
| --- | --- | --- | --- |
| HDF5 文本信息检查 | `precheck/checks/text_integrity.py` | 已完成 | 检查 `label/text_label` 中 `scene`、`task`、`text_en` 等必填字段是否存在、非空、JSON 是否可解析。 |
| `quality_hand` 二元数组检查 | `precheck/checks/quality_score.py`、`precheck/checks/keypoint_missing.py` | 已完成 | `quality_score` 按帧打分；任一手为 0 则该帧 0 分，否则 1 分。`keypoint_missing` 实现 10 秒窗口内低质量帧不超过 1 秒的客户规则。 |
| 21 个骨骼点连续性 | `precheck/checks/keypoint_temporal.py`、`precheck/checks/skeleton_quality_score.py` | 已完成 | 只使用供应商官方 hand acceptance topology，输出角度突变、旋转突变、加速度、位移等连续指标；`skeleton_quality_score` 给出 vendor-agnostic 骨架质量分。 |
| jump / 断点检测 | `keypoint_temporal` + `skeleton_quality_score` | 已完成 | 通过 `rotation_delta_max`、`joint_angle_change_deg_max`、`joint_acceleration_m_s2_max`、`joint_displacement_m_max` 捕捉相邻帧突变。测试中有 synthetic rotation/displacement jump。 |
| 自有 / SAM mask 抽检复核 | `precheck/checks/mask_containment.py` | 接口就绪 | precheck 只消费 injected hand masks 做 projected keypoint containment，不加载 SAM、不生成 mask、不修复标签。 |
| 供应商标签与几何结果联动审计 | `precheck/checks/composite_frame_verdict.py` | 已完成 | `quality_hand` 只作为 vendor weight / supplier audit signal；真正通用质量分来自 `skeleton_quality_score`。 |

不属于蓝圈当前实现的内容：拉取模块、快速画质检查的完整规则集、LLM/VLM 语义一致性、人工质检系统、小批准入结论、批次全量拉取、重复检查、有效时长台账。`overexposure` 目前只有基础过曝检查模板，不代表完整快速画质模块。

## 2. 模块边界

`precheck/` 只做数据可信度、信号质量、统计/几何检查。它可以读取 HDF5、供应商 skeleton、`quality_hand`、可选 masks 和相机内参，但不能加载 SAM3、DA3、VLM，也不能做语义推理或修复标签。

`qc_common/` 只放共享稳定契约和纯工具，包括 `ClipInputs`、`CheckResult`、hand keypoint topology、registry、小型 HDF5 scalar parser。它不能 import root modules 的 runtime internals。

外部可以编排：

```text
precheck -> annotation -> annotation_verify
```

但该顺序不得写死进 `precheck/`、`annotation/` 或 `annotation_verify/`。

## 3. 输入契约：`ClipInputs`

所有 precheck check 都消费 `qc_common.types.ClipInputs`。字段是 optional 的；缺失 optional input 时，check 应 clean skip 或输出可追踪的失败行，不能 crash runner。

关键字段：

| 字段 | 类型 | 生产方 | 消费方 |
| --- | --- | --- | --- |
| `episode_idx` | `int` | adapter / runner | 所有 check |
| `frame_indices` | `list[int] \| None` | adapter / runner | 用于真实帧号映射 |
| `frames` | `list[np.ndarray] \| np.ndarray \| None` | raw video loader | `overexposure` 等画质检查 |
| `keypoints` | `dict[str, np.ndarray] \| None` | HDF5 adapter | `keypoint_temporal`、`mask_containment` |
| `rotations` | `dict[str, np.ndarray] \| None` | HDF5 adapter | `keypoint_temporal`、`skeleton_quality_score` |
| `confidences` | `dict[str, np.ndarray] \| None` | HDF5 adapter | 信息性输出，不作为 missing 信号 |
| `quality_hand` | `np.ndarray \| None`，shape `(N, 2)` | HDF5 adapter | `quality_score`、`keypoint_missing`、`composite_frame_verdict` |
| `masks` | mask 容器或 `None` | 外部注入 | `mask_containment` |
| `intrinsics` | `np.ndarray \| None` | HDF5 adapter | `mask_containment` 2D 投影 |
| `instruction` | `str \| None` | text label / adapter | verification stub 或外部消费 |
| `text_label` | `dict[str, Any] \| None` | HDF5 adapter | `text_integrity` |
| `text_label_raw` | `str \| None` | HDF5 adapter | JSON parse fallback |
| `text_label_parse_error` | `str \| None` | HDF5 adapter | `text_integrity` malformed JSON flag |
| `fps` | `float \| None` | adapter / config | temporal window / velocity / acceleration |

### HDF5 adapter contract

默认 adapter 是 `precheck/adapters/supplier_hdf5.py`，读取：

```text
transforms/<joint_name>      -> keypoints[joint] = matrix[:, :3, 3]
                             -> rotations[joint] = matrix[:, :3, :3]
confidences/<joint_name>     -> confidences[joint]
label/quality_hand          -> quality_hand
label/text_label            -> text_label, text_label_raw, text_label_parse_error
camera/intrinsic            -> intrinsics
attrs.fps / attrs.frame_rate -> fps
```

`label/text_label` 用 `qc_common/hdf5_loader.py` 安全解析。JSON 解析失败不在 loader crash，而是记录 `text_label_parse_error`，交给 `text_integrity` 输出 `flag=True`。

## 4. 输出契约：`CheckResult`

所有 check 输出 `qc_common.types.CheckResult`：

```python
CheckResult(
    check: str,
    episode_idx: int,
    frame_idx: int,
    metrics: dict[str, float],
    flag: bool | None,
    reason: str,
)
```

约定：

- 帧级行使用真实 `frame_idx`，通过 `clip.frame_idx_at(offset)` 映射。
- clip summary 或 metadata-only check 使用 `frame_idx = -1`。
- `metrics` 尽量保留 raw continuous values，不要只输出二值化结论。
- `flag=True` 表示 actionable 问题或明确 verdict；`flag=None` 表示未标记/信息性/未校准。
- 若 check 有客户明确规则，可以输出 `flag=True/False`；未校准 raw metrics 保持 `flag=None`。

### JSON 交付文件

precheck runner 会同时写 parquet/csv 和 JSON。对外先统一消费 JSON：

```text
<output_dir>/check_results.json
<output_dir>/clip_aggregates.json
```

`check_results.json` 是 list，每条记录结构为：

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
    "which_thresholds_exceeded": [],
    "skeleton_score": 1.0
  },
  "flag": null,
  "reason": "temporal skeleton geometry within thresholds"
}
```

约定：

- `metrics` 在 JSON 中保持对象，不再是字符串。
- 缺失或不可计算的数值写为 `null`，不写非标准 `NaN`。
- `frame_idx=-1` 表示 clip-level summary 或 metadata-only row。

`clip_aggregates.json` 是按 `(episode_idx, check)` 聚合后的 list：

```json
{
  "episode_idx": 0,
  "check": "skeleton_quality_score",
  "checked_frames": 1001,
  "flagged_frames": 25,
  "uncalibrated_frames": 976,
  "clip_flag": true
}
```

## 5. 当前 checks 接口

### `text_integrity`

目的：检查 HDF5 `label/text_label` 文本字段是否可用。

配置：

```yaml
text_integrity:
  required_fields:
  - scene
  - task
  - text_en
```

输出：单个 clip-level row，`frame_idx=-1`。

关键 metrics：

```text
field_present_<field>
field_nonempty_<field>
missing_field_count
empty_field_count
```

flag 规则：

- 无 `text_label`：`flag=True`，`reason="no text_label"`
- JSON 解析失败：`flag=True`，`reason="text_label not valid JSON"`
- 任一 required field 缺失或为空：`flag=True`
- 全部存在且非空：`flag=None`

### `quality_score`

目的：供应商 `quality_hand` 的 clip acceptance score。

输入：`quality_hand` shape `(N, 2)`，列顺序 `[left, right]`。

逐帧规则：

```text
if left == 0 or right == 0:
    frame_score = 0.0
else:
    frame_score = 1.0
```

逐帧 metrics：

```text
frame_score
quality_left
quality_right
```

summary row：`frame_idx=-1`。

```text
total_score = sum(frame_score)
pass_ratio = total_score / num_frames
flag = pass_ratio >= pass_threshold
```

默认 `pass_threshold=0.90`。

### `keypoint_missing`

目的：按客户规则检测低质量/缺失 hand frame。

输入：`quality_hand`。

规则：

```text
任意 configurable 10-second window 内，
quality_hand < 0.5 的 low-quality/missing frames 总量
不得超过 configurable 1 second。
```

注意：

- `confidence == 0` 不是 missing / absence signal。
- 只根据 `quality_hand < 0.5` 判 low-quality。
- 会输出 `keypoint_missing_repair_candidates.json`，作为人工修复候选，不自动修复。

### `keypoint_temporal`

目的：输出供应商 skeleton 的 raw temporal metrics。

输入：

```text
keypoints: dict[joint_name, (N, 3)]
rotations: dict[joint_name, (N, 3, 3)]
fps
quality_hand optional
confidences optional
intrinsics optional
```

关键 metrics：

```text
joint_angle_change_deg_max
rotation_delta_max
joint_acceleration_m_s2_max
joint_displacement_m_*
joint_velocity_m_s_*
bone_length_m_*
bone_length_change_m_*
confidence_*
quality_hand_*
```

flag：当前为 `None`，因为 raw temporal thresholds 在该 check 内不做校准 verdict。

### `skeleton_quality_score`

目的：独立于供应商自标的通用骨架质量分。

复用 `keypoint_temporal` 输出，不重复实现几何计算。

默认阈值：

```yaml
skeleton_quality_score:
  joint_angle_change_deg_max_threshold: 10.0
  rotation_delta_max_threshold: 0.45
  joint_acceleration_m_s2_max_threshold: 15.0
  joint_displacement_m_max_threshold: 0.05
  pass_threshold: 0.90
```

逐帧 metrics：

```text
joint_angle_change_deg_max
rotation_delta_max
joint_acceleration_m_s2_max
joint_displacement_m_max
joint_angle_change_deg_penalty
rotation_delta_penalty
joint_acceleration_m_s2_penalty
joint_displacement_m_penalty
which_thresholds_exceeded
exceeded_threshold_count
missing_geometry_metric_count
skeleton_score
skeleton_verdict_code
```

flag 规则：

- 任一阈值超出：`flag=True`
- 否则：`flag=None`

summary row：

```text
count_good
count_suspect
mean_skeleton_score
min_skeleton_score
good_ratio / pass_ratio
pass_threshold
```

### `composite_frame_verdict`

目的：供应商标签与通用骨架分的审计，不替代 `skeleton_quality_score`。

语义重点：

- `quality_hand == 1.0` 只表示供应商未 downweight，不表示 frame 一定好。
- 通用质量分必须来自 `skeleton_quality_score`。
- 缺失 `quality_hand` 时，`vendor_quality_weight` 默认 `1.0`，但仍必须独立计算 skeleton score。

关键 metrics：

```text
frame_score
weighted_frame_score
skeleton_score
skeleton_verdict_code
joint_angle_change_deg_max
rotation_delta_max
joint_acceleration_m_s2_max
joint_displacement_m_max
which_thresholds_exceeded
vendor_quality_weight
supplier_quality_left
supplier_quality_right
supplier_label_available
supplier_audit_code
supplier_no_downweight
audit_suspect
```

flag 规则：

- 供应商未 downweight 且 skeleton geometry suspect：`flag=True`
- 其他情况：`flag=None`

### `mask_containment`

目的：使用外部注入 hand masks 对 projected keypoints 做 gross-error gate。

输入必须同时存在：

```text
clip.masks
clip.keypoints
clip.intrinsics
```

缺任一项则 clean skip，不输出 rows。

注意：

- 不加载 SAM。
- 不移动 keypoints。
- 不修复 labels。
- 只输出 containment metrics 和 gross-error signal。

## 6. SAM3 mask 抽样与 containment 对接

mentor 关注的“21 个骨骼点是否在手部区域内”，不能只靠 HDF5 skeleton 完成。HDF5-only 检查只能验证轨迹自洽，例如角度、旋转、加速度、位移是否突变；它不知道图像中真实手部区域在哪里。

推荐链路：

```text
mp4 / RGB frames
-> tools/sam3_keypoint_containment.py 抽样调用 SAM3 生成 hand / robot hand / gripper 2D mask
-> HDF5 3D keypoints 投影到同一张 RGB 图像坐标
-> 输出 frame / clip JSON keypoint containment ratio
```

云端 sidecar 脚本：

```text
tools/sam3_keypoint_containment.py
```

示例命令：

```bash
python tools/sam3_keypoint_containment.py \
  --hdf5-dir /path/to/hdf5 \
  --video-dir /path/to/mp4 \
  --sam3-model /path/to/sam3 \
  --output-dir outputs/sam3_keypoint_containment \
  --sample-fraction 0.10 \
  --projection-mode auto
```

这表示每个 clip 均匀抽 10% 帧跑 SAM3 mask，并输出 `clip_keypoint_inside_ratio = inside_keypoints / total_expected_keypoints`。这是验收 sidecar 脚本，不属于 annotation sampling；precheck check 仍然不加载 SAM3，只消费生成好的 masks 或 JSON 指标。

### 为什么 SAM3 是 2D mask 还需要 intrinsics

SAM3 的输出确实是 2D mask，坐标系是图像像素坐标：

```text
mask[y, x] = hand region
```

但 HDF5 里的 `joints3d` / `transforms` 通常是 3D 点：

```text
keypoint = (X, Y, Z)
```

要判断一个 3D 骨骼点是否落在 2D mask 内，需要先把 3D 点投影到图像像素坐标：

```text
u = fx * X / Z + cx
v = fy * Y / Z + cy
```

这里的 `fx, fy, cx, cy` 就来自 camera intrinsics。因此：

- 如果 keypoints 已经是 camera coordinate，且有 intrinsics，可以严谨投影到 RGB。
- 如果 keypoints 是 world/base coordinate，还需要 extrinsics，把点先变换到 camera coordinate。
- 如果没有 RGB 或没有标定参数，只能做 HDF5-only 几何检查，不能严谨做 mask containment。

## 7. 配置入口

示例配置在 `configs/precheck_example.yaml`。

启用检查：

```yaml
enabled_checks:
- keypoint_temporal
- keypoint_missing
- quality_score
- text_integrity
- skeleton_quality_score
- composite_frame_verdict
```

默认 `PrecheckConfig.enabled_checks` 仍保持保守，不自动启用所有新增检查；生产配置应显式列出要跑的 checks。

## 8. 新增 check 开发规范

新增 check 时遵循：

1. 文件放在 `precheck/checks/<name>.py`。
2. 继承 `precheck.base.BaseCheck`。
3. 设置：

```python
name = "<check_name>"
granularity = "frame"  # or "clip"
```

4. 使用 `@register` 装饰器注册。
5. config 放入 `precheck/config.py`，命名为 `<CamelName>Config`。
6. 在 `precheck/checks/__init__.py` import，保证 registry decorator 生效。
7. 示例配置写入 `configs/precheck_example.yaml`。
8. 测试写入 `tests/test_qc_modules_smoke.py` 或新增 focused test。
9. 缺 optional input 时 clean skip 或输出明确 failure row，不能 crash runner。
10. 不 import `annotation/`、`annotation_verify/` runtime internals。

最小模板：

```python
from precheck.base import BaseCheck
from precheck.registry import register
from qc_common.types import CheckResult, ClipInputs


@register
class ExampleCheck(BaseCheck):
    name = "example"
    granularity = "clip"

    def __init__(self, config: dict) -> None:
        self.threshold = float(config.get("threshold", 1.0))

    def run(self, clip: ClipInputs) -> list[CheckResult]:
        return [
            CheckResult(
                check=self.name,
                episode_idx=clip.episode_idx,
                frame_idx=-1,
                metrics={"example_metric": 0.0},
                flag=None,
                reason="example check",
            )
        ]
```

## 9. 验证命令

轻量验证：

```bash
.venv/bin/python -m compileall qc_common precheck annotation_verify annotation
.venv/bin/python -m pytest tests/test_qc_modules_smoke.py
```

运行 precheck：

```bash
python run_precheck.py configs/precheck_example.yaml
```

不要在 precheck 开发中启动 SAM3 / DA3 / VLM heavy jobs。
