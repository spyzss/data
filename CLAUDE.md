# Marmalade Visual Annotation Pipeline - Project Context

**项目目标**：为机器人数据集（LeRobot v3.0 格式）补充视觉标注——分割 mask 和深度 depth，无损存储。

**下游用途**：开放未定（可能用于 vision backbone 预训练或作监督信号），因此存储层不绑定任何训练表征，只无损保存原始标注。

**当前阶段**：先在 anygrasp 数据集上跑通 pipeline，验证后交给实习生扩展到全部数据集。

---

## 架构设计

四层架构，层间解耦：

### 1. Discovery（物体发现）
从 task instruction 中提取要分割的物体 query，自动补充 `robot hand`。

- **Mode 选项**：
  - `instruction`：从任务描述中抽取物体（LLM-based）
  - `vocab`：用户提供固定物体词表
  - `auto`：开放词表分割
- **Extractor 选项**：
  - `qwen`：使用 Qwen 模型提取
  - `rule`：规则匹配
  - `mock`：轻量 mock，供无模型环境验证链路
  - `manual`：人工/oracle query 映射，用于验证 SAM3/DA3 上限
- **Instruction 来源**：
  - `episode_field`：从 episode metadata 字段读取，anygrasp 使用 `expand_task`
  - `join_file`：从单独 parquet 表按 key join
  - `none`：无 instruction 数据集才回退到默认/vocab

**代码**：`annotation/discovery/`

### 2. Segmentation（分割）
SAM3 文本提示分割，输出 per-instance mask。

- 输入：物体 query 列表 + RGB 图像
- 输出：每个实例的 mask、bbox、score
- Vision 特征复用：同一帧多个 query 共享 vision encoder 输出，大幅加速

**代码**：`annotation/segmentation/sam3.py`

### 3. Depth（深度估计）
Depth Anything 3 单目深度估计，与分割层解耦。

- 输入：RGB 图像
- 输出：depth map（metric 或 relative）
- Metric vs Relative 自动检测：根据值域启发式判断，防止静默错标

**代码**：`annotation/depth/depth_anything3.py`

### 4. Storage（落盘）
无损存储标注结果，作为 LeRobot v3.0 自定义 feature。

- **Mask**：Parquet 格式，COCO compressed RLE 编码
- **Depth**：16-bit PNG（带 scale）+ JSON metadata

**代码**：`annotation/storage/`

---

## 帧采样策略（Sampling）

长视频逐帧标注浪费（相邻帧 mask/depth 近乎不变）。采样策略在 `config.sampling` 配置，
由 `LeRobotV3Dataset` 实现（`annotation/lerobot_v3_dataset.py`）。

- **`sampling.mode: uniform`**：在 episode 全长上 `np.linspace` 均匀取 `frames_per_episode` 帧（向后兼容旧行为）。
- **`sampling.mode: subtask_aware`**：读取每帧的 `subtask_index`（来自 data parquet），按 subtask 段分组，
  **段内按固定 stride 采样**，并**强制保留每个 subtask 段的起止关键帧**（`keep_subtask_boundaries: true`）。
  - stride 二选一：`sampling.stride_frames`（帧数）或 `sampling.stride_seconds`（秒，按 `meta/info.json` 的 fps 换算）。
    两者不可同时设置。
  - subtask 文本来自 `meta/subtask.parquet`（`atomic_skill` / `subtask`），用于日志可读性。
- `frame_indices` 显式指定时优先于任何采样模式（保持现有契约）。

**落盘账本（对齐下游）**：每次运行写 `<stage_output>/sampling_manifest.parquet`，
列 `episode_idx, frame_idx, subtask_index, atomic_skill, subtask, timestamp`，
把每个采样帧映射回原视频帧号与所属 subtask，便于下游按帧/按 subtask 对齐。

示例（anygrasp ep77，1081 帧，stride=30）：subtask-aware 采 39 帧，
段 159 [0,266] 与段 160 [267,1080] 的边界帧 266/267 及末帧 1080 全部保留；
同数量 uniform 采样会漏掉 266/267 边界。验证配置：`configs/dryrun_subtask_sampling.yaml`。

---

## Stage 解耦（SAM3 / DA3 独立运行）

SAM3（分割）与 DA3（深度）可独立运行，互不加载对方模型——两人可分工标注。
由 `config.stage` 控制，`run_annotate.py --stage` 可覆盖：

- **`--stage segmentation`**：只加载 SAM3，产出 mask。**绝不实例化 DA3**（无谓显存/下载）。
- **`--stage depth`**：只加载 DA3，产出 depth。**绝不实例化 SAM3**。
- **`--stage both`**（默认）：两者都跑，写共享 `output_dir`。

**独立目录与断点**（YAML 未显式指定时自动默认）：
- 分割 → `<output_dir>/segmentation/`（`masks.parquet` + manifest + `qc/`），checkpoint 在该目录 `.checkpoints/`。
- 深度 → `<output_dir>/depth_stage/`（`depth/<cam>/episode_/frame_.png` + manifest + `qc/`），独立 checkpoint。
- 可用 `storage.segmentation_output_dir` / `depth_output_dir`、`segmentation_checkpoint_dir` / `depth_checkpoint_dir` 覆盖。
- 两 stage 各自独立断点续跑；idempotency 只检查当前 stage 自己的产物
  （depth-only 不会因为没有 mask 而永不跳过）。

**对齐约定**：两 stage 输出都以 `(episode_idx, frame_idx)` 为 key（mask parquet 列 / depth PNG 路径 / manifest）。
合并 = 按 `(episode_idx, frame_idx)` join。只要两 stage 用相同 `dataset_path` + `episode_indices` + `sampling`，
采样帧集合一致（manifest 完全相同），mask 与 depth 天然对齐。
验证配置：`configs/seg_only.yaml`、`configs/depth_only.yaml`（同 dataset/sampling/output_dir）。

## 工程保证

- **Config 驱动**：所有路径/参数在 YAML 中，零硬编码
- **断点续跑/幂等**：支持 checkpoint，已处理帧跳过
- **Per-frame 失败隔离**：单帧失败不影响整个 episode
- **Dry-run 模式**：只跑 discovery 层，验证物体提取逻辑
- **QC 可视化**：自动生成 mask overlay 和 depth 伪彩图

---

## 环境配置

**服务器**：DSW，A800 80GB，当前无训练任务在跑

**隔离环境**：`.venv_annotation`（独立于主项目 marmalade 的训练环境）
- Python 3.12
- PyTorch 2.5.1+cu121
- Transformers 5.12.0
- depth-anything-3（独立包，非 transformers）

**网络配置**：
```bash
export HF_ENDPOINT=https://hf-mirror.com  # 内网环境，连不上 huggingface 官网
```

**激活环境**：
```bash
cd /mnt/workspace/spy/marmalade/marmalade_annotation
source .venv_annotation/bin/activate
```

---

## 当前状态

### ✅ 整体架构
- [x] 四层架构完成
- [x] Mock 端到端跑通（含断点续跑、QC）
- [x] Config schema 定义
- [x] 失败隔离、幂等性

### ✅ SAM3 分割层
- [x] HuggingFace transformers API 集成（`Sam3Model` + `Sam3Processor`）
- [x] Vision 特征复用优化（多 query 场景）
- [x] 后处理 pipeline（`post_process_instance_segmentation`）
- [x] 真实 anygrasp 数据验证：平均 19.6 实例/帧
- [x] 代码定稿：`annotation/segmentation/sam3.py`

**权重位置**：`models/sam3`（facebook/sam3，gated 已申请通过）

### ✅ Discovery 层

**AnyGrasp instruction 来源**：没有 `tasks.jsonl`。真实 episode 级 instruction 在：
```text
/mnt/oss/Data/anygrasp/task_001155/batch_000000/meta/episodes/chunk-*/file-*.parquet
```

字段名是 `expand_task`。例：`episode_index=0` 的 instruction 为：
```text
Pick the green bottle from the table and place it inside the blue basket.
```

也可以通过 `meta/expand_annotation/expand_task_annotation.parquet` 按 `expand_task_index` join，但当前 pipeline 默认直接读 episode metadata 的 `expand_task`。

**接入方式**：
- `dataset_type: lerobot_v3`
- `discovery.instruction_source: episode_field`
- `discovery.instruction_field: expand_task`
- 如果目标数据集没有自然语言 instruction，才使用 `instruction_source: none` 或固定 vocab；pipeline 会 warning 后回退到 `default_instruction` / vocab。

**Extractor 接口**：`annotation/discovery/factory.py` 统一创建 `rule` / `qwen` / `mock` extractor，接口保持：
```python
discover_objects(instruction, config) -> list[str]
```

factory 负责追加 `always_include`（例如 `robot hand`、`gripper`）、归一化和去重。真实 Qwen 模型加载仍是扩展点：实习生后续在 `QwenExtractor` 内部署和调优，不需要改 pipeline 编排。

**验证结果**：anygrasp `episode_idx=0`, `frame_idx=100` 完整 pipeline 已接到真实 `expand_task`，SAM3 输出类别包含 `blue basket`、`bottle`、`green bottle`、`robot hand`，不再全是 `robot hand`。

**交付前小批量验证**：
- `configs/delivery_rule_small_batch.yaml`：episode 0/1/2，每条均匀采样 5 帧，共 15 帧，真实 SAM3 + DA3 跑通，15/15 成功，0 失败，总耗时 49.5s（不含 QC 约 29s），共 140 个 mask。
- checkpoint 重跑同配置：pipeline 层 0.17s 跳过 3/3 episode；当前 CLI 仍会先初始化模型，再进入 checkpoint skip。
- 故障注入验证：episode 0 采样 3 帧，故意让第 2 帧 segmentation 抛错，结果 2/3 成功、1 帧失败，job 未崩，失败记录为 frame 170。

**干净 query 上限验证**：
- 对比配置：
  - `configs/delivery_rule_clean_compare.yaml`
  - `configs/delivery_manual_clean_queries.yaml`
- episode 0/1/14，每条均匀采样 5 帧，共 15 帧。
- rule baseline：164 个 mask，ep0/ep1/ep14 每帧实例数均值 8.4 / 14.0 / 10.4。
- manual/oracle：91 个 mask，ep0/ep1/ep14 每帧实例数均值 5.8 / 5.8 / 6.6。
- 关键观察：clean query 保留真实物体并去掉 rule 的长短语/泛化词噪声；ep1 的 `red toy drill` 从 rule 长短语只检出 1 个实例，manual clean query 5/5 帧检出，mean score 0.931。
- 对比图：`outputs/delivery_checks/qc_comparisons/compare_ep*.png`。

### ✅ DA3 深度层

**最终结论**：DA3 单图 metric 深度可用。之前“深度几乎是常数 / DA3 单图给不了 metric / 建议换 Depth Anything V2”的判断作废；真因只是没有加载 checkpoint 权重。

**正确模型**：`depth-anything/DA3METRIC-LARGE`
- HF 官方列表里没有 `DA3METRIC-LARGE-1.1`；metric 目标使用无 `-1.1` 的 `DA3METRIC-LARGE`。
- `DA3-LARGE-1.1` / `DA3-GIANT-1.1` 不是 metric checkpoint，不能替代 metric 目标。

**正确加载方式**：
```python
# annotation/depth/depth_anything3.py
from depth_anything_3.api import DepthAnything3

self.model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
result = self.model.inference([frame], process_res=504, process_res_method="upper_bound_resize")
raw_depth = result.depth[0]
```

严禁用 `DepthAnything3(model_name="da3metric-large")` 直接构造模型；这种方式只建网络结构，不加载 checkpoint。诊断证据：旧方式首层参数 `nonzero=0/1024`，`from_pretrained` 后 `nonzero=1024/1024`。

**Metric 换算**：
- `da3metric-large` 的 `result.depth` 是 raw network output，不是米制 depth。
- 必须手动套官方公式：`metric_depth_m = focal_px * raw_depth / 300.0`。
- `focal_px` 使用像素单位焦距，并按 DA3 实际输出分辨率缩放：
  - 标定源：GR3 + OAK-D-W-97，832x480 下 `fx=369.81`, `fy=341.14`。
  - 当前处理输出约 294x504，缩放后 `focal≈216.5px`。
  - 代码从 `result.depth.shape` 推导缩放，不写死 294x504。

**验证结果**：
- 固定帧：anygrasp `episode_idx=0`, `frame_idx=100`, `pixel_sha256=9c33f4f9e5f39c38b8a98509612b66ea6ef54c9059031a3b07134804557e99cf`。
- 真实 pipeline depth 路径输出：
  - raw depth：min 0.4057, max 2.4398, mean 1.2112, std 0.5184。
  - metric depth：min 0.2927m, max 1.7606m, mean 0.8740m, std 0.3741m。
  - 写出 PNG16 后读回：uint16 mm 编码，span 1.468m，JSON `depth_type="metric"`。
- Mock depth 路径仍可用，`--use-mock` 未破坏。

---

## 核心命令

### 基础运行
```bash
# Dry-run：只测试 discovery 层
python run_dryrun.py configs/anygrasp_dryrun.yaml

# 完整 pipeline（使用 Mock）
python run_annotate.py configs/anygrasp_full.yaml --use-mock

# 真实模型（SAM3 + DA3）
python run_annotate.py configs/anygrasp_full.yaml
```

### Stage 解耦运行（两人分工）
```bash
# 只跑分割（只加载 SAM3，输出在 <output_dir>/segmentation/）
python run_annotate.py configs/seg_only.yaml --stage segmentation

# 只跑深度（只加载 DA3，输出在 <output_dir>/depth_stage/）
python run_annotate.py configs/depth_only.yaml --stage depth

# --stage 覆盖 config.stage；不传则用 config 里的 stage（默认 both）
```

### Subtask-aware 采样
```bash
# Dry-run 验证采样（不加载模型，打印每段采了哪些帧、边界保留）
python run_dryrun.py configs/dryrun_subtask_sampling.yaml
```

### 调试 DA3
```bash
# 启用 debug_depth_range 查看深度值域
# 修改 configs/anygrasp_full.yaml:
#   depth.debug_depth_range: true

python run_annotate.py configs/anygrasp_full.yaml

# 检查日志输出：
# DA3 raw depth: shape=(H, W) min=... max=... mean=... std=...
# DA3 output depth: type=metric min=... max=... mean=... std=...
```

### 验证安装
```bash
python validate_install.py
```

---

## 关键文件位置

### 核心实现
- `pipeline.py` - 主编排逻辑
- `annotation/discovery/` - 物体发现层
- `annotation/segmentation/sam3.py` - SAM3 分割（✅ 定稿）
- `annotation/depth/depth_anything3.py` - DA3 深度（✅ metric 已验证）
- `annotation/storage/` - 存储层
- `annotation/qc/` - QC 可视化

### 配置
- `configs/anygrasp_dryrun.yaml` - Dry-run 配置
- `configs/anygrasp_full.yaml` - 完整 pipeline 配置
- `configs/delivery_rule_small_batch.yaml` - 交付前多 episode 小批量验证
- `configs/delivery_rule_clean_compare.yaml` - rule baseline 对比
- `configs/delivery_manual_clean_queries.yaml` - manual/oracle clean-query 对比
- `annotation/config.py` - Config schema 定义

### 诊断归档
- `archive/debug_artifacts_2026-06-15/` - 一次性 SAM3/DA3 诊断脚本、临时图片、npz/txt 输出。默认交付路径不需要阅读这些文件。

### 文档
- `README.md` - 项目总览
- `IMPLEMENTATION_SUMMARY.md` - 实现总结
- `SAM3_IMPLEMENTATION.md` - SAM3 详细文档
- `DA3_IMPLEMENTATION.md` - DA3 详细文档

---

## 输出格式

### Segmentation Masks
**路径**：`<output_dir>/masks.parquet`

**Schema**：
```
episode_idx: int
frame_idx: int
instance_id: int
category: str
score: float
rle_counts: bytes  # COCO compressed RLE
rle_size: (int, int)  # (height, width)
area: int
bbox: (float, float, float, float)  # (x, y, w, h)
```

### Depth Maps
**PNG**：`<output_dir>/depth/<camera_name>/episode_<idx>/frame_<idx>.png`
- 16-bit grayscale PNG
- 值域：`[0, 65535]` 线性映射到 `[original_min, original_max]`

**Metadata JSON**：`<output_dir>/depth/<camera_name>/episode_<idx>/frame_<idx>.json`
```json
{
  "depth_type": "metric",  // or "relative"
  "scale": 1.0,
  "original_min": 0.3,
  "original_max": 2.5,
  "unit": "meters"  // if metric
}
```

---

## 关键设计决策

### 1. Mock vs Real 并存策略
- **问题**：模型部署有先后，不能阻塞 pipeline 验证
- **解决**：`--use-mock` 标志控制，Mock 和 Real 实现并存
- **收益**：短期用 Mock 验证逻辑，模型就绪后平滑切换

### 2. SAM3 Vision 特征复用
- **问题**：一帧 10 个 query，每次重新编码图像太慢
- **解决**：第一个 query 时计算 `vision_features`，后续复用
- **收益**：理论加速 5-8x

### 3. DA3Metric Raw-to-Metric 显式处理
- **问题**：`DA3METRIC-LARGE` 的 API 返回 raw network output，直接当米制 depth 会错。
- **解决**：
  - 固定用 `DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")` 加载权重。
  - `result.depth` 手动转换：`metric_depth_m = focal_px * raw_depth / 300.0`。
  - `focal_px` 根据实际输出深度图尺寸从标定分辨率缩放。
  - `debug_depth_range: true` 同时打印 raw 和 metric 的 min/max/mean/std。
- **代码**：`annotation/depth/depth_anything3.py`

### 4. AnyGrasp Instruction 接入
- **问题**：anygrasp 没有 `tasks.jsonl`；真实任务文本是 episode 级 `expand_task`。
- **解决**：
  - `annotation/lerobot_v3_dataset.py` 读取 `meta/episodes/chunk-*/file-*.parquet`。
  - 通过 config 控制 instruction 来源字段或 join 文件，不写死 anygrasp 专有路径。
  - dry-run 不解码视频，只读取 metadata 以快速验证 instruction -> queries。
- **代码**：`annotation/lerobot_v3_dataset.py`, `annotation/discovery/factory.py`

---

## 故障排查 Checklist

### SAM3 问题
- [ ] `transformers >= 5.0`
- [ ] 模型路径/ID 正确
- [ ] HuggingFace token（如果 gated）
- [ ] GPU 内存足够

### DA3 问题
- [ ] `depth-anything-3` 包已安装
- [ ] `HF_ENDPOINT` 设置（内网）
- [ ] 使用 `DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")`，不要用 `DepthAnything3(model_name=...)`
- [ ] 日志中确认 `first_param_nonzero` 非 0
- [ ] `debug_depth_range: true` 同时查看 raw 和 metric 值域
- [ ] 配置相机 `fx/fy/calibration_width/calibration_height`
- [ ] 确认 `metric = focal * raw / 300` 只套一次

### Discovery 问题
- [ ] anygrasp 应检查 `meta/episodes/...parquet` 是否有 `expand_task`
- [ ] `discovery.instruction_source` / `instruction_field` 是否匹配数据集
- [ ] 无 instruction 数据集是否显式配置 `instruction_source: none` 或 vocab
- [ ] Qwen 模型路径配置；真实 Qwen 抽取器尚待实习生部署
- [ ] `rule` extractor 的长短语/泛化词是已知临时问题；用 `manual` oracle 配置判断 SAM3/DA3 上限

---

## 开发规范

### 代码风格
- 类型注解：所有新函数
- 日志：使用 `logging`，不用 `print`
- 失败隔离：单帧错误不能崩溃整个 job
- 幂等性：重复运行不重复处理

### Git 规范
- 功能分支开发
- Commit message 格式：`feat/fix/docs: <summary>`
- PR 前跑通测试

### 测试策略
- 单元测试：核心逻辑（RLE 编码、depth 缩放）
- 集成测试：端到端 dry-run
- Mock 测试：无模型时验证架构

---

## 参考资料

### 外部依赖
- [SAM 3 (Facebook)](https://huggingface.co/facebook/sam3)
- [Depth Anything 3 (ByteDance)](https://github.com/ByteDance-Seed/Depth-Anything-3)
- [LeRobot v3.0](https://github.com/huggingface/lerobot)

### 内部文档
- 主项目 marmalade 的 `AGENTS.md` - 主训练项目的 agent 操作规范
- 主项目的 `DEVELOPMENT.md` - 训练合约和架构

---

**最后更新**：2026-06-16
**当前优先级**：交付给实习生部署真实 Qwen 抽取器，并扩展 config 到全量数据集；pipeline 接线、SAM3、DA3、存储、QC 已完成小批量验证
