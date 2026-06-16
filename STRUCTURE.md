# marmalade_annotation - Project Structure

```
marmalade_annotation/
│
├── 📋 Documentation
│   ├── README.md              # 项目概览、架构、快速开始
│   ├── STATUS.md              # 完整状态、已完成功能、TODO列表
│   ├── QUICKSTART.md          # 快速上手指南（从安装到使用）
│   └── requirements.txt       # Python依赖列表
│
├── 🔧 Configuration
│   └── configs/
│       ├── anygrasp_dryrun.yaml   # Dry-run示例（仅discovery）
│       └── anygrasp_full.yaml     # 完整pipeline示例
│
├── 🚀 Entry Points
│   ├── run_dryrun.py          # Dry-run入口（验证discovery）
│   ├── run_annotate.py        # 完整标注入口
│   └── validate_install.py    # 安装验证脚本
│
├── 🏗️ Core Pipeline
│   ├── pipeline.py            # 主编排逻辑
│   │   ├── AnnotationPipeline       # 四层串联
│   │   ├── Checkpoint管理           # 断点续跑
│   │   ├── 失败隔离                 # Per-frame错误处理
│   │   └── 进度统计                 # 耗时、成功率追踪
│   │
│   └── annotation/
│       ├── types.py           # 核心数据类型
│       │   ├── InstanceMask          # 分割实例
│       │   └── DepthResult           # 深度结果
│       │
│       ├── config.py          # 配置Schema（dataclass）
│       │
│       ├── mock_dataset.py    # Mock LeRobot dataset（测试用）
│       │
│       ├── 📍 Layer 1: Discovery
│       ├── discovery/
│       │   ├── base.py               # ObjectDiscoverer接口
│       │   ├── rule_extractor.py     # 正则+规则抽取 ✅
│       │   └── qwen_extractor.py     # Qwen模型抽取（mock实现）🟡
│       │
│       ├── 📍 Layer 2: Segmentation
│       ├── segmentation/
│       │   ├── base.py               # Segmenter接口
│       │   ├── mock_segmenter.py     # Mock实现（测试用）✅
│       │   └── sam3.py               # SAM 3实现（骨架+TODO）🟡
│       │
│       ├── 📍 Layer 3: Depth
│       ├── depth/
│       │   ├── base.py               # DepthEstimator接口
│       │   ├── mock_depth_estimator.py  # Mock实现（测试用）✅
│       │   └── depth_anything3.py    # Depth Anything 3（骨架+TODO）🟡
│       │
│       ├── 📍 Layer 4: Storage
│       ├── storage/
│       │   ├── base.py               # MaskWriter/DepthWriter接口
│       │   ├── mask_writer.py        # Parquet + COCO RLE ✅
│       │   └── depth_writer.py       # PNG16 + sidecar JSON ✅
│       │
│       └── 📍 QC Visualization
│           └── qc/
│               └── visualize.py      # RGB+Mask+Depth可视化 ✅
│
└── 📦 Outputs (generated at runtime)
    └── outputs/
        ├── anygrasp_dryrun/
        │   └── discovery_queries.jsonl    # Dry-run结果
        │
        └── anygrasp_full/
            ├── masks.parquet              # 所有分割实例
            ├── depth/
            │   └── observation.images.top/
            │       └── episode_XXXXXX/
            │           ├── frame_XXXXXX.png   # 16-bit depth
            │           └── frame_XXXXXX.json  # metadata
            ├── qc/
            │   └── qc_epXXXXXX_frameXXXXXX.png  # 可视化
            └── .checkpoints/
                └── completed_episodes.json    # 断点信息
```

---

## 图例

- ✅ 完全实现并验证
- 🟡 骨架完成，待真实模型部署后对接API
- 📋 文档
- 🔧 配置
- 🚀 可执行脚本
- 🏗️ 核心代码
- 📍 四层架构标记
- 📦 运行时生成

---

## 代码行数统计

```bash
# Python代码
find annotation -name "*.py" | xargs wc -l | tail -1
# ~2500 lines

# 总计（含文档、配置）
find . -name "*.py" -o -name "*.md" -o -name "*.yaml" | xargs wc -l | tail -1
# ~3000+ lines
```

---

## 关键文件说明

### 核心编排
- **`pipeline.py`** (400+ lines) - 主编排器，负责：
  - 四层串联
  - Checkpoint加载/保存
  - Per-episode/per-frame循环
  - 失败隔离和统计

### Layer实现
- **`discovery/rule_extractor.py`** - 完整的规则抽取器
- **`storage/mask_writer.py`** - Parquet写入，COCO RLE压缩
- **`storage/depth_writer.py`** - PNG16写入，JSON元数据
- **`qc/visualize.py`** - Matplotlib可视化生成

### 待完成
- **`segmentation/sam3.py`** - 第38、80行有TODO标记
- **`depth/depth_anything3.py`** - 第38、75行有TODO标记

---

## 依赖关系

```
run_annotate.py
    ↓
pipeline.py
    ↓
┌───────────────┬───────────────┬───────────────┬───────────────┐
│   Discovery   │  Segmentation │     Depth     │    Storage    │
│               │               │               │               │
│ rule_extractor│ mock_segmenter│mock_depth_est │  mask_writer  │
│ qwen_extractor│     sam3      │depth_anything3│  depth_writer │
└───────────────┴───────────────┴───────────────┴───────────────┘
                              ↓
                         qc/visualize
```

---

## 测试覆盖

✅ **已验证的场景**：
- Dry-run模式（只跑discovery）
- 完整pipeline（mock模型）
- 断点续跑（checkpoint机制）
- 失败隔离（单帧错误不影响全局）
- 幂等写入（重复运行跳过已完成）
- QC可视化生成
- Mask的RLE压缩和解码
- Depth的PNG16编码和元数据

🟡 **待真实模型验证**：
- SAM 3推理
- Depth Anything 3推理
- 实际LeRobot数据集加载

---

## 扩展点

### 添加新的Discovery方法
1. 继承 `annotation/discovery/base.py::ObjectDiscoverer`
2. 实现 `discover_objects(instruction, config)`
3. 在 `run_annotate.py` 中注册

### 添加新的Segmentation后端
1. 继承 `annotation/segmentation/base.py::Segmenter`
2. 实现 `segment_frame(frame, queries, config)`
3. 返回 `list[InstanceMask]`

### 添加新的Depth后端
类似Segmentation，继承 `DepthEstimator`

### 添加新的Storage格式
继承 `MaskWriter` 或 `DepthWriter`，实现写入和幂等检查

---

## Git结构建议

```
marmalade_annotation/
├── .gitignore           # 忽略 outputs/, __pycache__
├── README.md
├── STATUS.md
├── QUICKSTART.md
├── requirements.txt
└── [其余文件]
```

`.gitignore` 建议：
```
outputs/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
.venv/
```

---

## 下一步行动

1. **验证安装**：
   ```bash
   .venv/bin/python marmalade_annotation/validate_install.py
   ```

2. **运行dry-run**：
   ```bash
   cd marmalade_annotation
   ../.venv/bin/python run_dryrun.py configs/anygrasp_dryrun.yaml
   ```

3. **检查discovery结果**：
   ```bash
   cat outputs/anygrasp_dryrun/discovery_queries.jsonl
   ```

4. **运行完整测试（mock）**：
   ```bash
   ../.venv/bin/python run_annotate.py configs/anygrasp_full.yaml --use-mock
   ```

5. **部署SAM3/DA3后**：
   - 更新config中的model_path
   - 编辑对应的py文件，取消TODO注释
   - 去掉`--use-mock`重新运行

---

**当前状态**：🟢 架构完整，mock可运行，待真实模型对接
