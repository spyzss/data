# Marmalade Annotation Agents

本仓库是 VLA / 机器人数据验收与视觉标注项目，当前重点是 `marmalade_annotation` 下的数据预检查、抽样验收、SAM3/DA3 标注、语义验证和批次台账。

> ⚠️ **核心原则**：Agents 必须遵守模块边界，严禁将所有逻辑耦合成一个单体 pipeline。

---

## Current Priority

当前工作优先级及推荐流转方向：

1. `precheck/` 增强 cheap HDF5 / temporal / projection checks。
2. 输出 candidate windows，供后续 SAM3/DA3 sidecar 或人工复核。
3. SAM3/DA3 **不进入** `precheck/`。
4. 最终 verdict 放到 postcheck / batch ledger，不在 cheap precheck 阶段过早判死。
5. 保留原始连续指标，不要只输出二值结果。

### 推荐工作流
```text
precheck cheap metrics
 └──> candidate_windows / needs_visual_review
       └──> optional SAM3/DA3 visual sidecar
             └──> postcheck / batch ledger final verdict
```

---

## Module Boundaries

### 1. `precheck/`
只做低成本、可大规模运行的数据可信度和信号质量检查。

* **允许**：
    * HDF5 字段检查。
    * NaN / inf / missing / 有效点数量检查。
    * supplier `quality_hand` 记录。
    * hand keypoint temporal metrics。
    * keypoint topology / bone length sanity。
    * 3D keypoints -> 2D projected keypoints。
    * projection out-of-frame / near-border risk。
    * candidate window generation。
    * 消费 optional precomputed / injected masks。
* **禁止**：
    * ❌ 不得加载 SAM3。
    * ❌ 不得加载 DA3。
    * ❌ 不得加载 VLM / Qwen。
    * ❌ 不得 import `annotation/` runtime internals。
    * ❌ 不得移动、snap、repair keypoints。
    * ❌ 不得把 `rotation_delta` 单独作为 hard fail。
    * ❌ 不得把 `quality_hand=1` 解释为 skeleton 一定正确。
    * ❌ 不得把 visible mask containment 当成完整可见性证明。

### 2. `annotation/`
负责正式视觉标注。

* **允许**：
    * Discovery。
    * SAM3 segmentation。
    * DA3 depth。
    * mask/depth/manifest 输出。
    * annotation QC 可视化。
* **禁止**：
    * ❌ 不得依赖 precheck 已经运行。
    * ❌ 不得把 precheck 逻辑塞进 annotation。
    * ❌ 不得把 semantic verification 塞进 annotation。

### 3. `annotation_verify/`
负责语义一致性验证。

* **允许**：
    * instruction / video semantic consistency。
    * Qwen / VLM HTTP endpoint interface。
    * stub verifier。
* **禁止**：
    * ❌ 不得做 HDF5 temporal / missing / projection checks。
    * ❌ 不得修改 annotation outputs。
    * ❌ 不得默认本地 Qwen / VLM 权重存在。

### 4. `qc_common/`
只放共享纯工具和稳定 schema。

* **允许**：
    * `ClipInputs` / `CheckResult`
    * keypoint topology / projection helper
    * candidate window helper
    * registry / base types
* **禁止**：
    * ❌ 不得加载模型。
    * ❌ 不得写 outputs。
    * ❌ 不得读取大 HDF5 / 视频。
    * ❌ 不得 import root module runtime internals。

### 5. `acceptance_pull/` / `video_quality/`
用于快速验收、抽样、视频质量检查和 batch ledger 输入。

* **允许**：
    * 读取 precheck 输出 / annotation 输出 / visual sidecar 输出。
    * 生成 batch summary / ledger。
    * 做视频画质检查：过曝、欠曝、黑屏、模糊、卡顿、fps、duration、帧数对齐。
* **禁止**：
    * ❌ 不得反向修改 precheck 原始结果。
    * ❌ 不得把 heavyweight model loading 塞回 precheck。
    * ❌ 不得提交 generated outputs / videos / HDF5 / model weights。

---

## Keypoint Semantics

使用 supplier hand acceptance topology。每只手 21 个 keypoints，双手共 42 个。

* **数量期望**：单手期望 21 个有效点，双手共 42 个。
* **硬性拦截**：HDF5 缺点 / NaN / inf / 有效点不足是 **hard invalid**。
* **零置信度**：`confidence == 0` 不等于 missing。
* **质量信号**：`quality_hand` 是 supplier 质量/降权信号，`quality_hand=1` 不代表骨骼点一定正确。
* **边界情况**：HDF5 有 42 个 finite 点，但画面里部分不可见，**不算 missing**，应进入 occlusion / out-of-frame / visual review。

---

## Current Precheck Metrics

当前重点保留并输出这些 raw metrics：
* `joint_displacement_m_max`
* `joint_acceleration_m_s2_max`
* `joint_angle_change_deg_max`
* `rotation_delta_max`

### 指标语义
* **displacement / acceleration**：主要抓 temporal jump、single-joint spike、tracking jitter。
* **joint angle change**：辅助判断姿态突变。
* **rotation_delta**：主要作为 visual review trigger，尤其用于手出画面、遮挡、快速翻手候选召回。**（不应单独 hard fail）**
* **聚合规则**：当前指标如果是左右手所有 21 点取 max，需要在输出中明确记录。max 适合召回，不适合单独最终判错。

---

## Projection Review

Projected keypoints 指：把 HDF5 里的 3D keypoints 投影到图像 2D 坐标。

* **若 keypoints 已在相机坐标系**：
    $$u = f_x \cdot \frac{X}{Z} + c_x$$
    $$v = f_y \cdot \frac{Y}{Z} + c_y$$
* **若 keypoints 不在相机坐标系，必须先用外参转换**：
    $$P_{\text{cam}} = R \cdot P_{\text{world}} + t$$
* **异常处理**：没有内参 / 图像尺寸 / 坐标系不确定时，**不要 crash**，应当 clean skip projection，输出 `projection_enabled=false` 或清楚记录原因。

### 建议输出的 Projection Metrics
```text
u_min
u_max
v_min
v_max
hand_bbox_area_2d
hand_bbox_center_u
hand_bbox_center_v
hand_bbox_center_jump_px
num_points_outside_image
num_points_near_border
num_projection_invalid
keypoint_bbox_touches_border
```

### 建议输出的 Projection Flags
```text
needs_projection_review
needs_out_of_frame_review
needs_rotation_mask_review
needs_visual_review
```

> 💡 **核心组合判定规则**：
> `rotation_delta high` + `projection/border risk` -> **visual review candidate**
> `rotation_delta high alone` -> **not hard fail**

---

## Candidate Windows

Candidate windows 用于把 cheap trigger frames 合并成小窗口，后续只对这些窗口跑 SAM3/DA3 sidecar 或人工复核。

### 窗口生成逻辑
1. 找到 trigger frames（来源：temporal jump, projection outside image, near-border risk, topology instability 等）。
2. 每个 trigger frame 扩展前后 $N$ 帧。
3. 相邻/重叠窗口按 gap threshold 进行 merge。
4. 每个窗口选择 `peak_frame`。
5. 后续视觉复核只抽样查看 `start` / `peak` / `end` / `optional middle`。

### 字段 Schema 定义
```json
{
  "asset_id": "string (or episode_idx)",
  "hand_side": "string",
  "start_frame": "int",
  "end_frame": "int",
  "peak_frame": "int",
  "trigger_metrics": {
    "rotation_delta_max": "float",
    "joint_displacement_m_max": "float",
    "joint_acceleration_m_s2_max": "float",
    "joint_angle_change_deg_max": "float",
    "num_points_outside_image": "int",
    "num_points_near_border": "int",
    "num_projection_invalid": "int",
    "hand_bbox_center_jump_px": "float",
    "hand_bbox_area_2d": "float"
  },
  "trigger_reason": "temporal_jump | rotation_edge_risk | projection_outside | bbox_center_jump | ...",
  "review_type": "temporal_skeleton_review | out_of_frame_review | rotation_visual_review | projection_review",
  "priority": "high | medium | low"
}
```

---

## Mask / SAM3 / DA3 Semantics

* **SAM3 mask** 是 visible mask，只代表画面中可见区域，不代表被遮挡或画面外的部分。
* **DA3 depth** 是可见表面深度，不能恢复遮挡物后或画面外的真实手部位置。
* ❌ **错误逻辑**：`outside hand mask -> skeleton fail` 或 `inside hand mask -> good`。
* ✅ **正确区分**：`visible_contained` / `mask_boundary_acceptable` / `contained_but_truncated_review` / `visual_conflict_review` 等。

### wrist/root rule
wrist/root 对 hand mask containment 可以更宽松，但对 image border / out-of-frame 必须更敏感。
> **严格执行顺序**：
> 1. HDF5 missing / NaN / inf -> `hard_invalid`
> 2. projection FOV / border check
> 3. hand mask truncation check
> 4. wrist/root boundary tolerance
> *注：不要让 wrist/root 宽松规则放过出画面情况。*

### hand mask truncation
`keypoints inside hand mask` 不等于整只手完整可见。还需要检查 `hand_mask_touches_border` 与 `mask_bbox_near_border`。
* 若 keypoints 都在 mask 内但 mask 触边：输出 `contained_but_truncated_review`。
* 若验收要求手必须完整可见，可在 postcheck / ledger 判为 `coverage_fail`，但**不要**写成 `skeleton_keypoint_missing`。

---

## Output Contracts

### 1. Precheck outputs
* **保持已有输出**：`check_results.json / parquet`，`clip_aggregates.json / parquet`
* **可新增输出**：`candidate_windows.json / parquet`，`projection_metrics.json / parquet`
* **单样本核心保留字段**：`asset_id`, `frame_idx`, `check`, `flag/risk/review`, `raw metrics`, `reason`, `needs_visual_review`

### 2. Ranking outputs
* **保持已有输出**：`top_skeleton_quality_clips.json / csv`，`skeleton_quality_ranking.json / csv`，`ranking_failures.json`
* ⚠️ **注意**：不要只输出 binary result，必须保留 raw continuous metrics。

### 3. Visual sidecar outputs (后续实现)
* **建议输出**：`keypoint_visual_arbitration.json / parquet`，`window_visual_aggregates.json / parquet`

---

## Manual Labels

人工标注统一归拢，**不要分散成多个小 JSON**。
* **统一存放路径**：`manual_labels_xingjiguitu.json`
* **结构定义**：
    ```json
    {
      "asset_id": "string",
      "frame_start": "int",
      "frame_end": "int",
      "hand_side": "string",
      "manual_verdict": "string",
      "confidence": "float",
      "algorithm_outcome": "positive | acceptable_flagged | review | true_positive | partial | ...",
      "notes": "string",
      "trigger_metrics": {}
    }
    ```
* *注：人工判断用于规则校准，但不要覆盖 raw metrics。*

---

## Common Commands

```bash
# 状态与编译检查
git status --short
git branch --show-current
python3 -m compileall precheck qc_common tools
.venv/bin/python -m pytest tests/test_qc_modules_smoke.py -q

# 运行 Precheck 示例
python run_precheck.py configs/precheck_example.yaml

# Skeleton ranking
python tools/rank_skeleton_quality_clips.py \
  --hdf5-dir /path/to/hdf5 \
  --output-dir outputs/xjgt_skeleton_rank \
  --top-k 10 \
  --fps 29.97 \
  --asset-ids <id1> <id2>
```

### 星际归途云端路径
```text
/mnt/oss/egodata/XJGT_20260629/hdf5
/mnt/oss/egodata/XJGT_20260629/video
```
⚠️ **开发注意**：不要对 15 万 HDF5 直接全量跑重任务。请务必先使用 `--asset-ids`、`--max-clips`、`--sample-step` 或两阶段预筛。

---

## Git / Editing Rules

* **修改前**：
    * 先确认 owning module。
    * 先看 `git status --short`。
    * 当前 QC 工作默认在 `feat/qc-modules` 分支，不要改无关文件。
* **修改时**：
    * 保持改动局部，不删旧逻辑，先新增字段/输出做消融。
    * 不要引入跨模块 runtime coupling。
    * 严禁提交：generated outputs、videos、HDF5、model weights、virtualenv。
    * 新阈值必须由 **config-driven** 控制。Optional inputs 缺失时必须 clean skip。
    * 失败要隔离，单帧失败不能崩整个 job。
* **完成后**：
    * 报告 files changed、新增 config / output fields、验证命令、未验证项和假设。
    * ⚠️ **除非用户明确要求，否则不要 stage / commit / push**。

---

## Codex Task Strategy

大任务必须分批、按 Phase 逐步落地。

* **Phase 1：precheck cheap metrics**
    * 进：HDF5 existence / temporal metrics / projection metrics / candidate windows / synthetic tests.
    * 退：SAM3 / DA3 / annotation runtime import / final verdict replacement.
* **Phase 2：visual sidecar**
    * 进：读取 candidate windows / 读取已有 SAM3 masks 及 DA3 depth / 输出 visual arbitration statuses.
    * 退：把 SAM3/DA3放进 precheck / 直接覆盖 precheck verdict.
* **Phase 3：postcheck / ledger**
    * 进：汇总 precheck、visual sidecar、manual labels / 输出 final batch ledger.
    * 退：ledger 内重跑 heavyweight models / 覆盖 raw module outputs.

---

## Current Rule of Thumb

Agent 决策核心速查：

```text
HDF5 缺点 / NaN / inf / 有效点不足
-> hard_invalid

HDF5 有 finite points，但视觉不可见 / 遮挡 / 出画面
-> visual review / coverage review

rotation_delta 高
-> visual review trigger
-> not hard fail

inside SAM3 mask
-> containment evidence
-> not proof of full hand visibility
```

---
*Last Updated: 2026-07-06*