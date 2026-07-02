# Marmalade Annotation Agents - Project Context

**项目目标**：本仓库是多模块机器人视觉标注、数据预检查与语义一致性验证系统，面向 LeRobot / VLA 数据集，提供 annotation、precheck、verification 能力。

**当前定位**：这不是单一 `annotation/` pipeline。Agents 必须遵守四个 root modules 的边界：

* `annotation/`：视觉标注
* `precheck/`：数据可信度与信号质量检查
* `annotation_verify/`：语义一致性验证
* `qc_common/`：共享契约、schema、keypoint 定义与工具

**核心原则**：模块之间没有代码层面的强制 workflow 顺序。外部可以编排：

```text
precheck -> annotation -> annotation_verify
```

但该顺序不得硬编码进 root modules。

---

## 架构设计

Root modules：

```text
annotation/
precheck/
annotation_verify/
qc_common/
```

### Boundary rules

* 模块边界优先级高于复用便利性。
* 跨模块改动必须先说明 owning module、输入输出契约和 boundary risk。
* 共享代码只能放入 `qc_common/`，且必须是稳定、无模型、无 runner side effect 的契约或工具。
* 不得通过 import runtime internals 来隐式建立 workflow。
* SAM3 / DA3 / VLM loading 只能留在各自 owning module。

---

## 1. `annotation/`（视觉标注）

`annotation/` 只负责视觉标注 pipeline。

### 职责

* Discovery：从 instruction 中提取目标物体 query。
* Segmentation：SAM3 文本提示分割，输出 per-instance mask。
* Depth：Depth Anything 3 深度估计。
* Storage：mask / depth / annotation manifest 落盘。
* Annotation QC：mask overlay、depth 伪彩图等可视化。

### 允许

* 加载 annotation dataset。
* 运行 discovery、segmentation、depth、storage。
* 写出 masks、depth maps、annotation manifests。
* 生成 annotation 自己的 QC visualization。

### 禁止

* 不得 import `precheck/` runtime internals。
* 不得 import `annotation_verify/` runtime internals。
* 不得执行 data-trust checks 或 semantic verification。
* 不得让 annotation runner 假设 precheck 已运行。
* 不得把 `precheck -> annotation -> verify` 写成 annotation 内部流程。

### Discovery

统一接口：

```python
discover_objects(instruction, config) -> list[str]
```

规则：

* `annotation/discovery/factory.py` 创建 extractor、追加 `always_include`、归一化、去重。
* Instruction 来源由 config 控制：`episode_field`、`join_file`、`none`。
* AnyGrasp 默认使用 episode metadata 字段 `expand_task`。
* 无自然语言 instruction 的数据集必须显式使用 `instruction_source: none` 或固定 vocab。

### Segmentation

输入：

* RGB image
* object queries

输出：

* instance mask
* bbox
* score
* category
* RLE encoded mask

规则：

* 同一帧多个 query 应尽量复用 vision encoder 输出。
* SAM3 加载只属于 `annotation/segmentation/`。
* 单帧 segmentation 失败不得 crash 整个 job。

### Depth

输入：

* RGB image

输出：

* depth map
* depth metadata

规则：

* DA3 加载只属于 `annotation/depth/`。
* Metric depth 必须保留 conversion / metadata。
* 单帧 depth 失败不得 crash 整个 job。

### Storage

* Mask：Parquet + COCO compressed RLE。
* Depth：16-bit PNG + JSON metadata。
* Alignment key：`(episode_idx, frame_idx)`。

### Sampling

* `frame_indices` 显式指定时优先。
* 支持 `sampling.mode: uniform`。
* 支持 `sampling.mode: subtask_aware`。
* Subtask-aware 应按 subtask 段采样，并在配置开启时保留 boundary frames。
* 每次运行应写 sampling manifest，把采样帧映射回原视频帧和 subtask 信息。

### Stage decoupling

* `--stage segmentation`：只加载 SAM3。
* `--stage depth`：只加载 DA3。
* `--stage both`：两者都跑。
* 两个 stage 各自独立 checkpoint / idempotency。

---

## 2. `precheck/`（数据可信度与信号质量检查）

`precheck/` 只负责 data trust、signal quality、统计 / 几何检查。

### 职责

* 数据可信度检查。
* 图像 / 视频信号质量检查。
* supplier skeleton HDF5 信号检查。
* hand keypoint missing / temporal / containment checks。
* 通过 adapters 消费 supplier-specific inputs。

### 当前 checks

* `overexposure`
* `keypoint_temporal`
* `keypoint_missing`
* `mask_containment`

### 允许

* 消费 raw frames、supplier skeleton HDF5、camera intrinsics、FPS。
* 消费 optional injected masks。
* 使用 `precheck/adapters/` 做 supplier-specific loading。
* optional input 缺失时 clean skip。

### 禁止

* 不得做 semantic reasoning。
* 不得判断 instruction 是否匹配视频内容。
* 不得加载 SAM3 / DA3。
* 不得依赖 `annotation/` runtime internals。
* 不得修正、snap、repair labels。
* 只能 detection and reporting。

### Runner contract

* 独立可执行。
* 不假设 annotation 或 annotation_verify 会运行。
* optional masks 必须通过 `ClipInputs` 或 config 注入。
* 缺失 optional masks 时，相关 check skip cleanly。
* 单个 check failure 不得阻止其他 checks。

---

## 3. `annotation_verify/`（语义一致性验证）

`annotation_verify/` 只负责 semantic consistency verification。

### 职责

* 比较 supplier instruction / description 与 video content 是否一致。
* 定义 verification contracts、config、registry、runner。
* 输出 semantic verification rows。

### 当前状态

* `instruction_consistency` 是 stub。
* VLM call 是 stub。
* 不实现真实 VLM / model logic，除非用户明确要求。

### 允许

* 通过 `ClipInputs` 接收 frames / instruction。
* 定义 semantic verification contract。
* 定义 stub output。
* 保留未来 VLM 接口。

### 禁止

* 不得执行 signal-quality checks。
* 不得执行 keypoint temporal / missing / containment。
* 不得修改 annotation outputs。
* 不得 import `annotation/` 或 `precheck/` internals。
* 不得提前实现真实 VLM logic。

### Runner contract

* 独立可执行。
* 不假设 annotation 或 precheck 已运行。
* 通过 `ClipInputs` 接收必要输入。
* 当前 VLM stub 不应被替换成真实模型，除非用户明确要求。

---

## 4. `qc_common/`（共享契约与工具）

`qc_common/` 只放跨模块共享、稳定、无模型的契约与工具。

### 职责

* `CheckResult`
* `ClipInputs`
* shared base classes
* keypoint topology
* registry utilities
* small pure helpers

### 允许

* 被 `precheck/` import。
* 被 `annotation_verify/` import。
* 被 annotation 的 QC / storage 辅助逻辑 import，但不能形成 workflow coupling。

### 禁止

* 不得 import root modules 的 runtime internals。
* 不得加载 SAM3 / DA3 / VLM。
* 不得启动 runner。
* 不得包含 supplier-specific business workflow。

---

## Keypoint System

使用供应商官方 hand acceptance topology，不是 MediaPipe、OpenPose 或其他外部标准。

### Topology

每只手 21 个 acceptance keypoints，双手共 42 个：`Hand` plus four points per finger for Thumb, Index, Middle, Ring, Little.

Finger point naming uses the supplier topology:

* Thumb：`Knuckle`, `IntermediateBase`, `IntermediateTip`, `Tip`
* Other fingers：`FingerKnuckle`, `FingerIntermediateBase`, `FingerIntermediateTip`, `FingerTip`

### Keypoint rules

* No Metacarpal joint is an acceptance keypoint.
* No body, torso, leg, or non-hand joint is an acceptance keypoint.
* Metacarpal transforms may exist in supplier data but must be excluded from default keypoint metrics.
* `confidence == 0` is not a missing / absence signal.
* Missing / low-quality judgement is based on `label/quality_hand`。
* `quality_hand < 0.5` 表示该 hand 在该 frame low-quality。

---

## Check Semantics

### `keypoint_temporal`

* Purpose：检测 supplier hand skeleton temporal drift / jitter。
* Primary signal：per-joint rotation-block frame-to-frame delta。
* Secondary signal：joint-angle frame-to-frame change。
* Sanity-only signal：bone length / bone-length change。
* Bone length variance 不是 drift signal，只能作为 sanity check。
* 阈值未校准时输出 raw metrics 且 `flag=None`。
* 不得根据 observed data 静默推阈值。

### `keypoint_missing`

客户规则：

```text
任意 configurable 10-second window 内，
low-quality / missing frames 总量不得超过 configurable 1 second。
```

* Signal：`label/quality_hand`。
* `quality_hand < 0.5` 表示该 hand 在该 frame low-quality。
* `confidence == 0` 不是 missing / absence signal。
* 该 check 可以设置 `flag=True/False`，因为阈值来自客户规则。

### `mask_containment`

* Purpose：使用 injected hand masks 对 projected keypoints 做 gross-error gate。
* 如果没有 injected mask，clean skip。
* 只测量，不移动 keypoints。
* 不修复 labels。
* 不做 snap / repair。
* 输出 containment metrics 和 gross-error signal。

### `overexposure`

* Purpose：检查 raw frames 是否存在明显过曝等基础视觉质量问题。
* 属于 signal-quality check。
* 不得做 semantic reasoning。
* 不得判断任务是否完成。
* 阈值应 config-driven。

### `instruction_consistency`

* Purpose：检查 instruction 与视频内容是否语义一致。
* 当前为 `annotation_verify/` stub。
* 不得在 `precheck/` 中实现。
* 不得提前接真实 VLM，除非用户明确要求。

---

## Failure Policy

所有 independent runners 必须隔离失败。

### Frame-level failure

* 单帧失败时，如果 frame-level isolation 可行，不得 crash 整个 clip / job。
* 失败应记录到结果或日志中。
* 其他帧应继续执行。

### Check-level failure

* 单个 check 失败时，不得阻止其他 checks。
* Failed check 应有可追踪原因。
* Runner 不应因为一个 check failure 丢失整批结果。

### Optional inputs

* optional inputs 缺失时，默认该 check 不输出 result rows。
* 除非 check-specific contract 另有说明。
* 不得因为 optional input 缺失 crash runner。

### Flag policy

* uncalibrated metrics 必须输出 `flag=None`。
* 只有明确客户规则可以设置 `flag=True` 或 `flag=False`。
* 不得根据 observed data 静默反推阈值。
* 不得把 confidence 误当 missing signal。

### Output key

Frame-aligned result rows 必须 keyed by：

```text
(episode_idx, frame_idx)
```

---

## 核心命令

常用 runner / smoke commands：

```bash
git status --short
python -m compileall qc_common precheck annotation_verify annotation

python run_precheck.py --help
python run_precheck.py <config>

python run_dryrun.py configs/anygrasp_dryrun.yaml
python run_annotate.py configs/anygrasp_full.yaml --use-mock
python run_annotate.py configs/anygrasp_full.yaml

python run_annotate.py configs/seg_only.yaml --stage segmentation
python run_annotate.py configs/depth_only.yaml --stage depth
python run_annotate.py configs/anygrasp_full.yaml --stage both

python run_annotation_verify.py --help
python run_annotation_verify.py <config>

python validate_install.py
```

不要运行 heavyweight SAM3 / DA3 jobs，除非用户明确要求。

---

## 关键文件位置

### Agent contracts

* `CLAUDE.md`
* `AGENTS.md`

### Shared common

* `qc_common/types.py`
* `qc_common/base.py`
* `qc_common/keypoints.py`
* `qc_common/registry.py`
* `qc_common/io.py`

### Precheck

* `run_precheck.py`
* `precheck/config.py`
* `precheck/runner.py`
* `precheck/registry.py`
* `precheck/checks/*.py`
* `precheck/adapters/*.py`

### Annotation

* `pipeline.py`
* `run_annotate.py`
* `run_dryrun.py`
* `annotation/config.py`
* `annotation/lerobot_v3_dataset.py`
* `annotation/discovery/`
* `annotation/segmentation/`
* `annotation/depth/`
* `annotation/storage/`
* `annotation/qc/`

### Annotation verification

* `run_annotation_verify.py`
* `annotation_verify/config.py`
* `annotation_verify/runner.py`
* `annotation_verify/registry.py`
* `annotation_verify/checks/*.py`

### Configs

* `configs/anygrasp_dryrun.yaml`
* `configs/anygrasp_full.yaml`
* `configs/seg_only.yaml`
* `configs/depth_only.yaml`
* `configs/dryrun_subtask_sampling.yaml`

---

## Agent Editing Rules

### Before editing

* Identify the owning module.
* State whether there is module boundary risk.
* Keep the change scoped to the user request.
* If the request crosses modules, explain the contract first.

### When editing

* Do not modify code unless the user asks for code changes.
* Keep edits minimal and local.
* Do not change generated outputs.
* Do not introduce hidden workflow coupling.
* Do not import runtime internals across root modules.
* Do not add dependencies unless the owning module clearly needs them.
* Do not move SAM3 / DA3 / VLM loading outside its owning module.
* New checks should be added through config / registry patterns.
* Optional inputs must remain optional.
* Use config for paths, thresholds, windows, and check switches.
* Use `logging`, not ad hoc `print`, for runtime code.
* Prefer type annotations for new functions.

### Before finishing

* Run the narrowest relevant smoke / syntax check when code changed.
* For docs-only changes, validate file shape with `wc -l` and a quick read.
* Report which files changed.
* Report validations run and any skipped validation.

### Git hygiene

* Check `git status --short` before reporting completion.
* Do not stage or commit unless explicitly asked.
* Do not revert unrelated user changes.
* Do not submit model weights, virtualenvs, generated outputs, debug images, or large local data files.

---

**最后更新**：2026-07-01

**当前优先级**：保持四模块架构边界清晰，增量补充 precheck / annotation_verify checks；annotation 的 SAM3 / DA3 / sampling / storage 经验继续保留，但不得重新引入旧的单 pipeline 心智模型。
