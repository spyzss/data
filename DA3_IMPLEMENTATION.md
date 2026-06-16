# Depth Anything 3 Implementation - Complete

## ✅ 实现状态

Depth Anything 3真实模型调用已实现，基于独立的`depth-anything-3`包API（非transformers）。

**重要**：由于DA3完整推理API尚未确认，实现中标注了多处TODO，需要在模型部署后核对。

### 已实现功能

1. **模型加载（已确认）**
   - ✅ 使用`depth_anything_3.cfg`的`load_config()`和`create_object()`
   - ✅ 自动检测GPU/CPU
   - ✅ 模型设置为eval模式

2. **推理流程（待确认）**
   - 🟡 输入预处理（TODO：确认是否需要normalize/resize）
   - 🟡 推理调用方式（TODO：`model()`? `model.infer()`? `model.predict()`?）
   - 🟡 输出格式（TODO：dict? tensor? 字段名?）

3. **Metric vs Relative处理（核心特性）**
   - ✅ **自动检测深度类型**：根据值域启发式判断metric/relative
   - ✅ **显式验证与警告**：期望metric但检测到normalized时记录warning
   - ✅ **Debug模式**：`debug_depth_range=true`时打印min/max/mean
   - ✅ **防止静默错标**：不硬编码depth_type，根据实际值域决定

4. **Config集成**
   - 新增`debug_depth_range`参数（调试开关）
   - `output_metric`：期望输出类型（true=米制，false=相对）

### API验证状态

按照你提供的信息实现：

✅ **已确认部分**：
```python
from depth_anything_3.cfg import create_object, load_config
model = create_object(load_config("path/to/config"))
```

🟡 **待确认部分**（标注TODO）：
```python
# TODO: 确认单图推理接口
depth = model(frame_tensor)  # or model.infer()? model.predict()?

# TODO: 确认输出格式
if isinstance(depth_output, dict):
    depth = depth_output.get("depth", depth_output.get("pred_depth"))
```

---

## 🔍 Metric vs Relative 处理逻辑

### 问题背景

DA3可能默认输出归一化/相对深度（0~1或类似范围），而非metric depth（米）。如果静默当作metric存储，下游使用会出错。

### 解决方案

**启发式检测 + 显式警告**：

```python
# 分析值域
depth_min, depth_max, depth_mean = depth.min(), depth.max(), depth.mean()

# 启发式判断
is_likely_metric = (depth_min > 0.01) and (depth_max < 100.0) and (depth_mean > 0.1)
is_likely_normalized = (depth_min >= -1.1) and (depth_max <= 1.1)

# 如果期望metric但检测到normalized
if output_metric and is_likely_normalized and not is_likely_metric:
    logger.warning(
        f"Config expects metric but values suggest normalized "
        f"(min={depth_min:.3f}, max={depth_max:.3f}). "
        f"Check model output or set output_metric=false."
    )
    # 标记为relative，不硬编码成metric
    depth_type = "relative"
```

**调试模式**：

```yaml
depth:
  debug_depth_range: true  # 打印每帧的min/max/mean
```

输出示例：
```
Depth range: min=0.1234, max=0.9876, mean=0.5432
```

根据这些值判断：
- **Metric (合理)**：桌面场景min~0.3m, max~2m, mean~1m
- **Normalized (需要调整config)**：min~0, max~1

---

## 🧪 测试验证

### 基础测试（已通过）

```bash
cd /mnt/workspace/spy/marmalade
.venv/bin/python marmalade_annotation/test_da3.py
```

**结果**：
```
✓ Import
✓ Init without model
✓ Estimate without model
✓ Config schema
✓ Mock still works
```

✅ 所有测试通过

### Mock路径验证（已通过）

```bash
cd marmalade_annotation
../.venv/bin/python run_annotate.py configs/anygrasp_full.yaml --use-mock
```

**结果**：✅ Mock depth estimator正常工作，未被破坏

---

## 📝 使用方式

### 方式1：使用Mock（当前，模型未部署时）

```bash
cd marmalade_annotation
../.venv/bin/python run_annotate.py configs/anygrasp_full.yaml --use-mock
```

### 方式2：使用真实DA3（模型部署后）

**步骤**：

1. **部署DA3并获得config路径**：
   ```
   /path/to/da3_model/config.yaml
   ```

2. **更新config**（`configs/anygrasp_full.yaml`）：
   ```yaml
   depth:
     model_path: /path/to/da3_model/config.yaml
     model_type: depth_anything_v3
     output_metric: true
     debug_depth_range: true  # 首次运行开启，检查值域
   ```

3. **首次运行（调试模式）**：
   ```bash
   ../.venv/bin/python run_annotate.py configs/anygrasp_full.yaml
   ```

4. **检查日志**：
   ```
   Depth range: min=0.xxxx, max=x.xxxx, mean=x.xxxx
   ```

   - 如果是0~1范围 → 模型输出normalized，设置`output_metric: false`
   - 如果是合理米数 → 保持`output_metric: true`

5. **关闭调试模式**（确认无误后）：
   ```yaml
   debug_depth_range: false
   ```

---

## ⚙️ 配置说明

### DepthConfig新增字段

```yaml
depth:
  model_path: /path/to/da3/config.yaml
  model_type: depth_anything_v3
  
  # 期望的输出类型
  output_metric: true  # true=米制深度, false=相对深度[0,1]
  
  # 调试开关：打印每帧深度值域
  debug_depth_range: false  # 首次部署时设为true
```

**参数说明**：
- `output_metric`：
  - `true`：期望米制深度，但如果检测到归一化会warning并标记为relative
  - `false`：强制归一化到[0,1]，无论模型输出什么
  
- `debug_depth_range`：
  - `true`：每帧打印min/max/mean，用于确认depth类型
  - `false`：静默运行

---

## 🔧 待部署后确认的TODO

### 关键位置

**文件**：`annotation/depth/depth_anything3.py`

**第54-60行** - 输入预处理：
```python
# TODO: Deploy-time verification - confirm preprocessing/inference API
frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()
frame_tensor = frame_tensor.to(self.device)

# TODO: May need preprocessing (normalization, resize)
# If model has .preprocess() method, use it here
```

**需要确认**：
- 模型是否接受`[B, C, H, W]`格式的float tensor？
- 是否需要normalize（如ImageNet mean/std）？
- 是否需要resize到特定分辨率？

---

**第64-72行** - 推理调用：
```python
# TODO: Deploy-time verification - confirm exact inference method
# Possibilities:
# - depth = self.model(frame_tensor)
# - depth = self.model.infer(frame_tensor)
# - depth = self.model.predict(frame_tensor)
depth_output = self.model(frame_tensor)  # Placeholder - adjust after deployment
```

**需要确认**：
- 实际的推理方法名？
- 参数格式？
- 返回值结构？

---

**第74-78行** - 输出解析：
```python
# TODO: Deploy-time verification - confirm output format
# Extract depth map (may need to access specific field)
if isinstance(depth_output, dict):
    depth = depth_output.get("depth", depth_output.get("pred_depth"))
else:
    depth = depth_output
```

**需要确认**：
- 返回dict还是直接tensor？
- 字段名是`"depth"`、`"pred_depth"`还是其他？
- Shape是`[1, H, W]`还是`[H, W]`？

---

## 📊 对比：Mock vs Real

| 特性 | MockDepthEstimator | DepthAnything3Estimator (Real) |
|------|-------------------|-------------------------------|
| 用途 | 测试pipeline逻辑 | 实际生产标注 |
| 输出 | 径向梯度+噪声 | 真实深度估计 |
| Depth类型 | 可配置metric/relative | 自动检测+警告 |
| 性能 | ~0.001s/帧 | ~0.1-0.5s/帧（取决于GPU和分辨率） |
| 依赖 | 仅numpy | depth-anything-3, torch |
| 何时用 | 开发、CI、模型未就绪 | 生产标注、质量验证 |

---

## 🚀 部署后集成checklist

- [ ] 安装`depth-anything-3`包
- [ ] 获取模型config.yaml路径
- [ ] 更新pipeline config的`model_path`
- [ ] **开启`debug_depth_range`运行1帧**
- [ ] **检查日志中的depth值域，确认metric/relative**
- [ ] **调整`output_metric`配置（如果值域不符）**
- [ ] 关闭`debug_depth_range`
- [ ] 运行小规模测试（1个episode）
- [ ] 检查depth PNG和JSON质量
- [ ] 验证QC可视化中的depth伪彩是否合理
- [ ] 大规模运行

---

## 🐛 常见问题

### Q: Warning说期望metric但检测到normalized，怎么办？

A: 模型输出的是归一化depth，不是米制。解决方案二选一：
1. **改config**（推荐）：设置`output_metric: false`
2. **Scale校准**（高级）：找到dataset的depth scale factor手动换算

### Q: debug_depth_range打印的值不合理？

A: 可能的原因：
- 值全是0或全是同一个数 → 模型推理失败
- 值很大（如>1000） → 单位可能不是米
- 负值 → 模型可能输出disparity而非depth

### Q: Depth PNG看起来全黑或全白？

A: 检查：
- JSON中的`original_min/max`是否合理
- 是否误将normalized depth当metric存储（PNG会全白）
- QC可视化是否正常（如果QC正常说明PNG本身没问题）

---

## 📚 相关文件

- **实现**：`annotation/depth/depth_anything3.py` (~150 lines)
- **Config**：`annotation/config.py` (DepthConfig)
- **示例config**：`configs/anygrasp_full.yaml`
- **Mock保留**：`annotation/depth/mock_depth_estimator.py`（未改动）
- **测试**：`test_da3.py`

---

**当前状态**：🟢 DA3骨架完成 | 🟡 多处TODO待真实模型验证 | 🟢 Metric/Relative检测逻辑完整
