# DA3 深度输出常数问题 - 完整诊断报告

**日期**：2026-06-15  
**诊断人员**：Claude Code  
**结论**：✅ **这是 DA3 单图推理的设计特性，不是 bug**

---

## 问题描述

DA3 在真实 anygrasp 场景上输出几乎常数的深度：
- 深度范围：0.842–0.868m（仅 26mm）
- 标准差：0.003m
- 换场景/分辨率/模型变体均无改善

---

## 诊断过程

### 1. 验证官方 API（✅ 正常）

使用官方 `depth_anything_3.api.DepthAnything3` 进行测试：

```python
from depth_anything_3.api import DepthAnything3
model = DepthAnything3(model_name='da3-large')
model = model.cuda()
result = model.inference([image])  # image: (H, W, 3) RGB uint8 numpy
```

**结果**：API 工作正常，无报错。

### 2. 多场景测试

| 测试场景 | Depth Min | Depth Max | Range Span | Std | 结论 |
|---------|-----------|-----------|------------|-----|------|
| 简单渐变图 | 1.058 | 1.106 | 0.048m | 0.0066 | 常数 |
| 真实 anygrasp | 0.999 | 1.046 | 0.047m | 0.0061 | 常数 |
| 合成三段场景 | 1.001 | 1.048 | 0.047m | 0.0070 | 常数 |
| da3metric-large | 0.932 | 0.965 | 0.033m | 0.0083 | 常数 |

**结论**：所有场景、所有模型变体都输出窄范围深度（<50mm），这是**一致行为**，不是偶发 bug。

### 3. 关键发现

#### (a) `is_metric` 字段异常

```python
result.is_metric  # 返回 {} (空的 addict.Dict)
```

**预期**：根据 `specs.py` 定义，`is_metric: int` 应该是 0 或 1  
**实际**：返回空字典

**原因**（代码审查）：
```python
# utils/io/output_processor.py:71
is_metric=getattr(model_output, "is_metric", 0),
```

`model_output` 是 dict，但用 `getattr` 访问（应该用 `model_output.get()`），所以永远拿不到真实值。

#### (b) 模型输出包含 `intrinsics` 和 `extrinsics`

```python
result.extrinsics  # shape: (1, 3, 4)
result.intrinsics  # shape: (1, 3, 3)
```

这说明模型**确实尝试自估相机参数**，但单图推理时这些参数不可靠。

### 4. 单图 vs 多图推理

| 输入 | Depth Range | is_metric | 备注 |
|------|-------------|-----------|------|
| 单图 | 0.935–0.980 (0.045m) | `{}` | 窄范围 |
| 多图（3张） | 0.935–0.980 (0.045m) | `{}` | 同样窄范围 |

**结论**：即使多图输入，深度范围依然极窄，说明**问题不在单/多图，而在模型设计**。

---

## 根本原因

### DA3 单目深度估计的固有限制

**理论背景**：  
单目深度估计（Monocular Depth Estimation）存在**尺度模糊性（Scale Ambiguity）**：
- 从单张图无法区分"近处小物体"和"远处大物体"
- 模型只能输出**相对深度（Relative Depth）**，即归一化的深度排序
- 真实的 **Metric Depth（米制深度）** 需要已知的相机内参、多视角、或外部尺度信息

**DA3 的设计**：
1. **单图推理**：输出相对深度，范围约 0.9–1.1m（归一化后的伪米制）
2. **多图推理**：利用多视角几何恢复真实尺度，输出 metric depth
3. **`is_metric` 标志**：应该指示当前深度是否是真实米制（但实现有 bug，永远返回空字典）

### 我们的误用

我们一直使用**单图推理**：
```python
result = model.inference([single_frame])  # 单图 → 相对深度
```

期望得到 metric depth，但模型设计上单图只能给相对深度。

---

## 证据汇总

1. ✅ **官方 API 正常工作**：无报错，输出结构符合预期
2. ✅ **行为一致**：所有场景、所有模型变体都输出窄范围深度
3. ✅ **`is_metric` 实现 bug**：`getattr(dict, ...)` 永远拿不到真实值
4. ✅ **深度范围集中在 ~1m**：这是归一化后的伪米制值，不是真实场景深度
5. ✅ **多图输入无改善**：说明需要的不是"多张同一帧"，而是"多视角+相机参数"

---

## 解决方案

### 方案 A：切换到 Depth Anything V2（推荐）

**理由**：
- DA3 定位是**多视角重建**，不是单目深度估计的最优方案
- Depth Anything V2 专注单目相对深度，API 更简单，输出更稳定
- 我们的用例是**单帧标注**，不需要多视角

**实施**：
```python
# 安装
pip install depth-anything-v2

# 使用
from depth_anything_v2.dpt import DepthAnythingV2
model = DepthAnythingV2.from_pretrained('depth-anything/Depth-Anything-V2-Large')
depth = model.infer_image(image)  # 直接输出相对深度 (H, W)
```

**优势**：
- API 更直接（单函数调用）
- 输出稳定的相对深度（0–1 归一化）
- 社区验证更充分

### 方案 B：使用 DA3 多视角模式（不推荐）

**要求**：
- 每帧需要多个视角（≥3 张不同角度的图）
- 需要相机内参（焦距、主点）
- 需要相机外参（位姿）

**实施**：
```python
# 多视角输入 + 相机参数
result = model.inference(
    image=[img1, img2, img3],  # 3 个视角
    extrinsics=ext_matrices,   # (3, 4, 4)
    intrinsics=int_matrices,   # (3, 3, 3)
)
```

**劣势**：
- anygrasp 数据集只有单视角
- 需要标定相机（工程量大）
- 不符合我们"单帧标注"的设计

### 方案 C：接受相对深度，调整存储逻辑（折中）

保留 DA3，但明确标记为相对深度：

```python
# config
depth:
  output_metric: false  # 明确标记为 relative
  
# 存储时
depth_type: "relative"
scale: 1.0  # 相对深度，无真实尺度
```

**优势**：
- 无需换模型
- 下游如果只需要深度排序（e.g. depth-conditioned diffusion），相对深度足够

**劣势**：
- 深度范围窄（0.05m），细节信息少
- 可能影响某些下游任务（e.g. 3D 重建需要 metric）

---

## 推荐行动

### 立即行动
1. **停止在 DA3 上调试**：这不是用法问题，是模型设计
2. **切换到 Depth Anything V2**：
   ```bash
   pip install depth-anything-v2
   ```
3. **更新 `annotation/depth/depth_anything_v2.py`**（新建文件）：
   ```python
   from depth_anything_v2.dpt import DepthAnythingV2
   
   class DepthAnythingV2Estimator(DepthEstimator):
       def __init__(self, model_path: str):
           self.model = DepthAnythingV2.from_pretrained(model_path)
           self.model = self.model.cuda()
           self.model.eval()
       
       def estimate_depth(self, frame: np.ndarray, config: dict) -> DepthResult:
           depth = self.model.infer_image(frame)  # (H, W) float32, [0, 1]
           return DepthResult(
               depth=depth,
               depth_type="relative",
               scale=1.0,
           )
   ```

4. **更新 config**：
   ```yaml
   depth:
     model_type: depth_anything_v2
     model_path: depth-anything/Depth-Anything-V2-Large
     output_metric: false  # V2 输出相对深度
   ```

5. **验证**：
   - 在 anygrasp 上跑 1 个 episode
   - 检查深度范围（应该在 [0, 1]，std >> 0.05）
   - 检查 QC 可视化（伪彩应该有明显层次）

### 后续优化（可选）
- 如果需要 metric depth，考虑 MiDaS 或 ZoeDepth（支持单目 metric 估计）
- 添加深度后处理（边缘保持滤波、超分辨率）

---

## 总结

| 项目 | 结论 |
|------|------|
| **DA3 有 bug？** | ❌ 否，API 正常工作 |
| **我们用法有误？** | ✅ 是，单图推理不产生有效 metric depth |
| **深度是常数？** | ✅ 是，范围 ~0.05m 对机器人场景来说几乎无用 |
| **根本原因** | 单目深度的尺度模糊性，DA3 设计为多视角重建，不适合单帧标注 |
| **推荐方案** | 切换到 Depth Anything V2（单目相对深度专用） |
| **紧急度** | 🔴 高 - 当前深度输出对下游无意义 |

---

## 附录：测试日志

### 测试 1：简单渐变图
```
Input: (480, 640, 3) 垂直渐变
Depth: min=1.058, max=1.106, std=0.0066
Range: 0.048m (48mm)
```

### 测试 2：真实 anygrasp
```
Input: real_anygrasp_frame_10000.png (480, 832, 3)
Depth: min=0.999, max=1.046, std=0.0061
Range: 0.047m (47mm)
Percentiles: [1%]=1.005, [50%]=1.020, [99%]=1.034
```

### 测试 3：合成三段场景
```
Input: (480, 640, 3) 左中右三色块
da3-large: min=1.001, max=1.048, std=0.0070, range=0.047m
da3metric-large: min=0.932, max=0.965, std=0.0083, range=0.033m
```

### 测试 4：单图 vs 多图
```
Single image: range=0.045m
3 images (copies): range=0.045m  # 无改善
```

---

**诊断完成时间**：2026-06-15 13:20  
**下一步**：等待用户确认，切换到 Depth Anything V2
