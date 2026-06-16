# Marmalade Visual Annotation Pipeline

为 **LeRobot v3.0** 格式的机器人数据集补充视觉标注：用 **SAM3** 产出实例分割 mask、用 **Depth Anything 3 (DA3)** 产出 **metric（米制）深度**，无损存储为可下游复用的标注。当前已在 AnyGrasp 数据集上端到端跑通。

数据流：

```
episode instruction → 物体发现(Discovery) → SAM3 实例 mask(Segmentation)
                                           → DA3 metric 深度(Depth)        → 无损存储(Storage) → QC 可视化
```

下游用途未定（可能用于 vision backbone 预训练或作监督信号），因此存储层**不绑定任何训练表征**，只无损保存原始标注。

---

## 四层架构

层间解耦、config 驱动、可插拔（代码在 `annotation/`）：

1. **Discovery（物体发现）** — 从任务 instruction 提取要分割的物体 query，并自动补 `robot hand` / `gripper`。
   抽取器可配（`discovery.extractor`）：`rule`（规则，临时）、`manual`（人工/oracle 上限对照）、`qwen`（真实 LLM，**待实习生部署**）、`mock`（无模型链路验证）。统一接口 `discover_objects(instruction, config) -> list[str]`。
2. **Segmentation（分割）** — SAM3 文本提示分割，输出 per-instance mask + bbox + score。同一帧多 query 复用 vision encoder 输出加速。
3. **Depth（深度）** — DA3 单目 metric 深度，**与分割层完全独立**。
4. **Storage（落盘）** — mask 存 Parquet（COCO compressed RLE），depth 存 16-bit PNG（毫米）+ JSON metadata。

---

## 环境配置

- **Python 3.12**，建议用独立虚拟环境（与训练环境隔离）。本地开发环境名为 `.venv_annotation`。
- 依赖见 `requirements.txt`（numpy / pandas / pillow / pycocotools / matplotlib / pyyaml）。
  另需 **PyTorch**、**transformers ≥ 5.0**（SAM3）、**depth-anything-3** 包（DA3，独立包，非 transformers）。

```bash
python3.12 -m venv .venv_annotation
source .venv_annotation/bin/activate
pip install -r requirements.txt
# 再按各自 GPU 环境安装 torch / transformers>=5.0 / depth-anything-3
```

### 模型权重（不进 git，需单独下载）

内网环境从镜像下载：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

- **SAM3**：`facebook/sam3`，**gated 模型**，需先在 HuggingFace 申请通过并配置 token（`huggingface-cli login` 或 `HF_TOKEN`）。下载到本地后 config 指向该目录（如 `models/sam3`）。
- **DA3（metric）**：`depth-anything/DA3METRIC-LARGE`。
  - HF 官方列表里**没有** `DA3METRIC-LARGE-1.1`；metric 目标就用无 `-1.1` 的 `DA3METRIC-LARGE`。
  - `DA3-LARGE-1.1` / `DA3-GIANT-1.1` **不是 metric checkpoint**，不能替代。

`models/`、`*.safetensors` 等已在 `.gitignore` 中排除。

---

## 使用

激活环境并设置镜像：

```bash
source .venv_annotation/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
```

```bash
# Dry-run：只跑 Discovery（不加载模型），验证 instruction → query 与帧采样
python run_dryrun.py configs/anygrasp_dryrun.yaml

# 完整标注（SAM3 + DA3）
python run_annotate.py configs/anygrasp_full.yaml

# 只跑分割（只加载 SAM3，绝不加载 DA3）
python run_annotate.py configs/seg_only.yaml --stage segmentation

# 只跑深度（只加载 DA3，绝不加载 SAM3）
python run_annotate.py configs/depth_only.yaml --stage depth

# 无模型环境用 mock 验证编排
python run_annotate.py configs/anygrasp_full.yaml --use-mock
```

`--stage segmentation|depth|both` 覆盖 config 里的 `stage`（不传则用 config，默认 `both`）。

### Subtask-aware 采样

长视频逐帧标注浪费（相邻帧 mask/depth 近乎不变）。`sampling.mode: subtask_aware` 会读取每帧的
`subtask_index`，按 subtask 段分组，段内按固定 stride 采样，并**强制保留每段起止关键帧**：

```yaml
sampling:
  mode: subtask_aware          # uniform | subtask_aware
  stride_frames: 30            # 段内每隔 N 帧采一帧；与 stride_seconds 二选一
  # stride_seconds: 1.0        # 按秒（用 meta/info.json 的 fps 换算）
  keep_subtask_boundaries: true  # 保留每个 subtask 段的首/末帧
```

`uniform` 模式则在全长上均匀取 `sampling.frames_per_episode` 帧。`frame_indices` 显式指定时优先于采样。

每次运行会写 `<output_dir>/sampling_manifest.parquet`（列 `episode_idx, frame_idx, subtask_index, atomic_skill, subtask, timestamp`），把每个采样帧映射回原视频帧号与所属 subtask，供下游对齐。

---

## Config 关键字段

```yaml
dataset_path: /path/to/lerobot_v3/dataset   # 数据集根目录
dataset_type: lerobot_v3                     # auto | lerobot_v3 | mock
camera_names:
  - observation.images.top

stage: both                                  # segmentation | depth | both

discovery:
  extractor: rule                            # rule | manual | qwen | mock
  instruction_source: episode_field          # episode_field | join_file | none
  instruction_field: expand_task             # episode metadata 里的 instruction 字段
  always_include: [robot hand, gripper]      # 强制追加的 query

sampling:
  mode: subtask_aware                         # uniform | subtask_aware
  stride_frames: 30                           # 或 stride_seconds
  keep_subtask_boundaries: true

segmentation:
  model_path: models/sam3                     # 本地 SAM3 权重目录或 HF id

depth:
  model_path: depth-anything/DA3METRIC-LARGE
  output_metric: true                         # true=米制；false=相对深度
  # 相机内参（像素焦距）+ 标定分辨率，用于 metric 换算缩放：
  fx: 369.81                                  # GR3 + OAK-D-W-97 @ 832x480
  fy: 341.14
  calibration_width: 832
  calibration_height: 480

storage:
  output_dir: ./outputs/anygrasp_full
  overwrite: false                            # false=已标注帧跳过（断点续跑）
  # 可选：分 stage 独立目录（不设则自动用 <output_dir>/segmentation 和 /depth_stage）
  # segmentation_output_dir: ...
  # depth_output_dir: ...

episode_indices: [0, 1, 2]                    # 可选：只跑部分 episode
```

换数据集时**不改代码**，只改 config：`dataset_path`、`camera_names`、`discovery.instruction_source` /
`instruction_field`（或 `instruction_join_*`）、`depth` 下的相机内参。若数据集无自然语言 instruction，
设 `instruction_source: none` 或用 vocab 模式，loader 会 warning 后回退。

---

## 关键技术点

**DA3 必须用 `from_pretrained` 加载权重。** 直接构造 `DepthAnything3(model_name="da3metric-large")`
只建网络结构、**不加载 checkpoint**，会导致深度输出**恒为常数**。正确方式：

```python
from depth_anything_3.api import DepthAnything3

model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
result = model.inference([frame], process_res=504, process_res_method="upper_bound_resize")
raw_depth = result.depth[0]   # raw network output，不是米制
```

诊断证据：直接构造时首层参数 `nonzero=0/1024`；`from_pretrained` 后 `nonzero=1024/1024`。代码在加载后会断言首层参数非零。

**metric 换算公式。** `result.depth` 是 raw output，需手动换算：

```
metric_depth_m = focal_px * raw_depth / 300.0
```

`focal_px` 用像素焦距，并**按 DA3 实际输出分辨率从标定分辨率缩放**（代码从 `result.depth.shape` 推导缩放比例，不写死分辨率）。该公式只套**一次**。

实现见 `annotation/depth/depth_anything3.py`。

---

## 输出格式

所有输出以 `(episode_idx, frame_idx)` 为对齐 key。

**Masks** — `<output_dir>/masks.parquet`：
`episode_idx, frame_idx, instance_id, category, score, rle_counts, rle_size, area, bbox`（COCO compressed RLE）。

**Depth** — `<output_dir>/depth/<camera_name>/episode_<idx>/frame_<idx>.png`（16-bit PNG，毫米）
+ 同名 `.json`（`depth_type`、`encoding`、`original_min/max`、metric 换算说明）。

**Sampling manifest** — `<output_dir>/sampling_manifest.parquet`（采样帧 ↔ 原帧号 ↔ subtask）。

**QC** — `<output_dir>/qc/qc_ep<episode>_frame<frame>.png`（RGB / mask overlay / depth 伪彩三联图）。

> `outputs/` 整个目录在 `.gitignore` 中，不进 git。

---

## 两人分工标注的合并约定

SAM3 与 DA3 已解耦，可由两人分别运行 `--stage segmentation` 和 `--stage depth`，互不加载对方模型、
各自独立的输出目录（默认 `<output_dir>/segmentation/` 与 `<output_dir>/depth_stage/`）和独立 checkpoint
（各自独立断点续跑）。

**对齐前提：两个 stage 使用相同的 `dataset_path` + `episode_indices` + `sampling` 配置。**
这样两者采样到的帧集合完全一致（`sampling_manifest.parquet` 相同），mask 与 depth 即可按
`(episode_idx, frame_idx)` join 对齐合并。

---

## 当前验证状态

- ✅ **功能完整、端到端跑通**：Discovery → SAM3 → DA3 metric → 存储 → QC 全链路在 AnyGrasp 上验证。
- ✅ SAM3 用真实权重分割正常；DA3 metric 深度已验证（值域合理，如 0.32–1.74m）。
- ✅ Subtask-aware 采样、stage 解耦均已验证（边界帧保留、单 stage 不加载对方模型、输出按帧对齐）。
- ✅ 断点续跑、单帧失败隔离、mock 路径均验证通过。
- ⚠️ **仅小规模验证**（几条 anygrasp episode、十几帧），**未做全量生产验证**。
- ⚠️ **真实 Qwen 抽取器待部署**：当前用 `rule`（有脏 query 问题）/ `manual`（oracle 对照）。

---

## 实习生扩展全量待办

下一步扩展工作是**抽取器质量与全量规模**，不是 pipeline 接线（接线已完成）：

1. **部署真实 Qwen 抽取器**：在 `annotation/discovery/qwen_extractor.py` 实现真实模型加载与解析
   （文件内有 `TODO` 标注的接入点），保持公共接口 `discover_objects(instruction, config) -> list[str]` 不变，
   config 用 `extractor: qwen` 即可切换。
2. **改 config 指向新数据集**：`dataset_path`、`camera_names`、`discovery.instruction_source` /
   `instruction_field`、`depth` 相机内参。代码无需改动。
3. **注意已知问题——`rule` 抽取脏 query**：`rule` 会产出长短语（如
   `green bottle from table and place it inside blue basket`）和泛化词（如 `toy`）。这是临时限制，
   应由真实 Qwen 解决；用 `configs/delivery_manual_clean_queries.yaml` 的 manual oracle 作质量上限对照。
4. **小优化点：checkpoint-skip 前会先加载模型**。当前 CLI 即使整批 episode 都已完成、会被 checkpoint 跳过，
   仍会先初始化 SAM3/DA3。后续可在确认全部跳过时延迟/跳过模型加载，省下无谓的显存与加载时间。

---

## 目录结构

```
annotation/
  config.py                  # Config schema（dataclass + draccus 风格）
  lerobot_v3_dataset.py      # LeRobot v3 读取 + 帧采样（uniform / subtask_aware）
  discovery/                 # 物体发现层（rule / manual / qwen / mock）
  segmentation/sam3.py       # SAM3 分割
  depth/depth_anything3.py   # DA3 metric 深度
  storage/                   # Parquet mask + PNG16 depth 落盘
  qc/visualize.py            # QC 三联图
pipeline.py                  # 主编排
run_annotate.py              # 完整标注入口（--stage）
run_dryrun.py                # Dry-run 入口（只跑 Discovery）
configs/                     # YAML 配置
CLAUDE.md                    # 完整项目上下文与决策记录
```
