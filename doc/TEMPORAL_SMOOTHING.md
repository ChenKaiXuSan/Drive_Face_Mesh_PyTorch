# 时间序列3D关键点优化模块

## 概述

`temporal_smooth.py` 模块提供了多种方法来优化和平滑按时间序列排列的3D关键点数据。这在处理融合后的3D关键点(3D kpt)时非常有用，可以减少噪声、提高时间连续性和稳定性。

## 数据格式

### 输入数据格式
```
关键点数组: (T, N, 3) np.ndarray
  - T: 帧数 (时间维度)
  - N: 关键点数量
  - 3: x, y, z 坐标

可见性掩码: (T, N) np.ndarray (可选)
  - 布尔值: True 表示该帧的该关键点有效，False 表示无效/遮挡
```

### 示例
```python
import numpy as np

# 100帧，17个关键点（如人体姿态模型），每个关键点3个坐标
keypoints = np.random.randn(100, 17, 3)

# 可见性掩码（某些帧的某些点被遮挡）
visibility = np.ones((100, 17), dtype=bool)
visibility[30:40, 5] = False  # 第30-39帧的第5个关键点被遮挡
```

## 使用方法

### 1. 高斯平滑 (Gaussian Smoothing)

**特点**: 简单快速，对所有方向平等平滑
**适用场景**: 对整体平滑度要求较高的场景

```python
from head3D_fuse.temporal_smooth import smooth_keypoints_sequence

# 方法1: 直接函数调用
smoothed = smooth_keypoints_sequence(
    keypoints, 
    method="gaussian",
    sigma=1.5  # 标准差越大，平滑效果越强
)

# 方法2: 使用类
from head3D_fuse.temporal_smooth import TemporalKeypointOptimizer

optimizer = TemporalKeypointOptimizer(method="gaussian", sigma=1.5)
smoothed = optimizer.optimize(keypoints, visibility=visibility)
```

**参数**:
- `sigma`: 高斯核的标准差 (默认: 1.0)
  - 较小的值 (0.5-1.0): 轻度平滑，保留更多细节
  - 中等值 (1.5-2.0): 平衡平滑和细节
  - 较大的值 (3.0+): 强平滑，可能丢失快速动作

---

### 2. Savitzky-Golay 滤波 (SavGol)

**特点**: 在平滑的同时保留信号的形状和峰值，适合保留快速运动
**适用场景**: 需要保持关键点突变(如击打、接触)的场景

```python
smoothed = smooth_keypoints_sequence(
    keypoints,
    method="savgol",
    window_length=11,  # 过滤窗口大小（必须是奇数）
    polyorder=3        # 多项式拟合阶数
)
```

**参数选择建议**:
- `window_length`: 通常选择 5, 7, 11, 15 等奇数
  - 较小 (5): 保留更多细节，但平滑效果弱
  - 中等 (11): 平衡效果
  - 较大 (21+): 强平滑，可能丢失细节
  
- `polyorder`: 通常 2-4
  - 2 (二次): 适合简单曲线
  - 3 (三次): 推荐，平衡拟合和平滑
  - 4+ (更高阶): 更好地捕捉特征，但计算量大

```python
# 示例：快速动作序列
smoothed = smooth_keypoints_sequence(
    keypoints,
    method="savgol",
    window_length=7,   # 较小窗口保留快速变化
    polyorder=2
)
```

---

### 3. 卡尔曼滤波 (Kalman Filter)

**特点**: 考虑过程噪声和测量噪声，具有预测能力，可处理缺失值
**适用场景**: 有规律运动的序列，或需要填补缺失部分

```python
smoothed = smooth_keypoints_sequence(
    keypoints,
    method="kalman",
    process_variance=1e-5,      # 过程噪声方差（越小越信任预测）
    measurement_variance=1e-2   # 测量噪声方差（越小越信任观测）
)
```

**参数调整**:
```python
# 案例1: 高精度融合结果（信任融合，信任预测）
smoothed = smooth_keypoints_sequence(
    keypoints,
    method="kalman",
    process_variance=1e-6,      # 很小，预测很准确
    measurement_variance=1e-3   # 很小，融合很准确
)

# 案例2: 噪声较大的结果（不太信任融合，但信任运动模型）
smoothed = smooth_keypoints_sequence(
    keypoints,
    method="kalman",
    process_variance=1e-4,      # 较大，允许预测偏差
    measurement_variance=1e-1   # 较大，不完全信任测量
)

# 案例3: 有明显缺失值（使用卡尔曼填补）
smoothed = smooth_keypoints_sequence(
    keypoints,
    method="kalman",
    process_variance=1e-5,
    measurement_variance=1e-2
)
# 通过 visibility 掩码传递缺失信息
```

---

### 4. 双侧滤波 (Bilateral Filter)

**特点**: 边界保持滤波，在平滑的同时保留运动边界（如姿态突变）
**适用场景**: 需要保留关键点运动的跳跃(如帧间运动较大)

```python
smoothed = smooth_keypoints_sequence(
    keypoints,
    method="bilateral",
    sigma_space=1.5,    # 时间维度的宽度
    sigma_range=0.1     # 值域维度的宽度
)
```

**参数调整**:
```python
# 强边界保持
smoothed = smooth_keypoints_sequence(
    keypoints,
    method="bilateral",
    sigma_space=2.0,    # 宽的时间窗口
    sigma_range=0.05    # 窄的值域窗口，更易保留边界
)

# 更强的平滑
smoothed = smooth_keypoints_sequence(
    keypoints,
    method="bilateral",
    sigma_space=1.0,    # 窄的时间窗口
    sigma_range=0.2     # 宽的值域窗口，平滑多变化
)
```

---

### 5. 带约束的优化 (Constrained Optimization)

**功能**: 在时间平滑的同时，保持结构约束（如骨长不变）
**适用场景**: 有已知的结构约束（如人体骨骼）

```python
from head3D_fuse.temporal_smooth import optimize_keypoints_with_constraints

# 定义骨骼结构（关键点对和它们之间的距离）
bone_constraints = {
    (0, 1): 0.50,   # 点0和点1之间的距离约为0.50
    (1, 2): 0.45,   # 点1和点2之间的距离约为0.45
    (2, 3): 0.60,
    # ... 其他骨骼
}

optimized = optimize_keypoints_with_constraints(
    keypoints,
    bone_constraints=bone_constraints,
    smoothness_weight=2.0,      # 时间平滑的权重
    constraint_weight=1.0       # 结构约束的权重
)
```

**权重调整**:
```python
# 优先保持骨长
optimized = optimize_keypoints_with_constraints(
    keypoints,
    bone_constraints=bone_constraints,
    smoothness_weight=0.5,      # 较低的平滑权重
    constraint_weight=2.0       # 较高的约束权重
)

# 优先时间平滑
optimized = optimize_keypoints_with_constraints(
    keypoints,
    bone_constraints=bone_constraints,
    smoothness_weight=3.0,      # 较高的平滑权重
    constraint_weight=0.5       # 较低的约束权重
)
```

---

### 6. 速度和加速度估计

```python
from head3D_fuse.temporal_smooth import estimate_velocity, estimate_acceleration

# 估计速度 (T-1, N, 3)
velocity = estimate_velocity(smoothed_keypoints)

# 估计加速度 (T-2, N, 3)
acceleration = estimate_acceleration(smoothed_keypoints)

# 统计分析
import numpy as np
vel_magnitude = np.linalg.norm(velocity, axis=-1)
print(f"平均速度: {np.mean(vel_magnitude):.4f}")
print(f"最大速度: {np.max(vel_magnitude):.4f}")

accel_magnitude = np.linalg.norm(acceleration, axis=-1)
print(f"平均加速度: {np.mean(accel_magnitude):.4f}")
```

---

## 处理可见性/遮挡

大多数方法都支持可见性掩码，用于标记无效或被遮挡的关键点：

```python
# 创建可见性掩码
visibility = np.ones((num_frames, num_keypoints), dtype=bool)

# 标记某些帧中某些点为无效
visibility[30:50, 5] = False      # 帧30-50的点5被遮挡
visibility[:, 10] = False          # 点10在所有帧中都不可见

# 在平滑时使用
smoothed = smooth_keypoints_sequence(
    keypoints,
    method="gaussian",
    sigma=2.0,
    visibility=visibility
)
```

**行为**:
- 无效的点（visibility=False）会被排除在优化计算外
- 但其值可能会被相邻帧的有效点推断或保持原值
- 这有助于处理部分遮挡或追踪失败的情况

---

## 完整工作流示例

```python
import numpy as np
from head3D_fuse.temporal_smooth import (
    smooth_keypoints_sequence,
    optimize_keypoints_with_constraints,
    estimate_velocity,
)

# 1. 加载融合后的3D关键点
fused_kpts = np.load('fused_keypoints.npy')  # shape: (T, N, 3)
print(f"原始形状: {fused_kpts.shape}")

# 2. 创建可见性掩码（可选）
visibility = np.ones(fused_kpts.shape[:2], dtype=bool)
# ... 根据融合过程中的置信度或追踪失败标记无效点

# 3. 第一步: 基础平滑
smoothed_v1 = smooth_keypoints_sequence(
    fused_kpts,
    method="savgol",
    window_length=11,
    polyorder=3,
    visibility=visibility
)

# 4. 第二步: 应用结构约束（如果有已知的骨骼结构）
bone_constraints = {
    (0, 1): 0.5,
    (1, 2): 0.45,
    # ... 定义骨骼
}

smoothed_final = optimize_keypoints_with_constraints(
    smoothed_v1,
    bone_constraints=bone_constraints,
    smoothness_weight=2.0,
    constraint_weight=1.0
)

# 5. 计算动作特性
velocity = estimate_velocity(smoothed_final)
print(f"速度范围: [{np.min(velocity):.4f}, {np.max(velocity):.4f}]")

# 6. 保存结果
np.save('smoothed_keypoints.npy', smoothed_final)
```

---

## 性能和选择建议

| 方法 | 速度 | 平滑度 | 边界保持 | 推荐场景 |
|------|------|--------|----------|----------|
| Gaussian | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ | 快速处理，一般平滑 |
| SavGol | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | 保留形状的平滑 |
| Kalman | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 规律运动，缺失值填补 |
| Bilateral | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 快速动作，保留运动边界 |
| Constrained | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 有骨骼约束的场景 |

---

## 常见问题和解决方案

### Q1: 平滑后出现延迟或滞后现象
**答**: 大多数因果滤波方法(如卡尔曼)可能有延迟。解决方案:
- 使用非因果方法(高斯、SavGol)
- 调整卡尔曼滤波的方差参数

### Q2: 平滑后丢失快速运动细节
**答**: 
- 减小平滑参数 (sigma, window_length)
- 使用 SavGol 或双侧滤波保留边界
- 检查输入数据的质量

### Q3: 处理有缺失值的序列
**答**:
- 使用 `visibility` 掩码标记缺失值
- 卡尔曼滤波会自动从模型预测填补
- 或预先使用插值填补缺失值

### Q4: 优化计算很慢
**答**:
- 对于带约束的优化，减少关键点数量或帧数
- 降低 `max_nfev` (最大函数评估次数)
- 使用高斯或SavGol等更快的方法

---

## 参考文献

- 高斯滤波: Gaussian filtering concepts
- Savitzky-Golay: Savitzky, A., & Golay, M. J. (1964)
- 卡尔曼滤波: Kalman, R. E. (1960)
- 双侧滤波: Tomasi, C., & Manduchi, R. (1998)

---

## 贡献和改进

如有问题或改进建议，请提交 issue 或 pull request。

