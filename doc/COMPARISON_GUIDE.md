# 融合和平滑关键点比较工具使用指南

## 📋 功能概述

`compare_fused_smoothed.py` 提供了完整的工具来比较融合后的3D关键点和平滑后的3D关键点，帮助评估时间平滑的效果。

## 📊 评估指标

### 1. 差异指标
- **平均差异** (mean_difference): 所有关键点的平均L2距离
- **最大差异** (max_difference): 最大的单点差异
- **标准差** (std_difference): 差异的标准差
- **中位数** (median_difference): 差异的中位数

### 2. 速度指标
- **平均速度**: 关键点的移动速度（一阶导数）
- **速度降低**: 平滑后速度的降低百分比

### 3. 加速度指标（抖动）
- **平均加速度**: 速度的变化率（二阶导数）
- **加速度降低**: 平滑后加速度的降低百分比

### 4. 抖动分数
- **抖动分数**: 加速度标准差的平均值（越小越平滑）
- **抖动降低**: 平滑后抖动的降低百分比

## 🚀 使用方法

### 方法1: Python API

```python
import numpy as np
from pathlib import Path
from head3D_fuse.compare_fused_smoothed import KeypointsComparator

# 加载数据 (T, N, 3)
fused_kpts = np.load('fused_keypoints.npy')
smoothed_kpts = np.load('smoothed_keypoints.npy')

# 创建比较器
comparator = KeypointsComparator(fused_kpts, smoothed_kpts)

# 计算所有指标
metrics = comparator.compute_metrics()
print(f"平均差异: {metrics['mean_difference']:.6f}")
print(f"抖动降低: {metrics['jitter_reduction']:.2f}%")

# 生成报告
report = comparator.generate_report(save_path='report.txt')
print(report)

# 生成可视化
comparator.plot_comparison(
    save_path='trajectory_comparison.png',
    keypoint_indices=[0, 5, 10]  # 选择要可视化的关键点
)

comparator.plot_metrics(save_path='metrics_comparison.png')
```

### 方法2: 从NPZ目录加载

```python
from pathlib import Path
from head3D_fuse.compare_fused_smoothed import compare_fused_and_smoothed

compare_fused_and_smoothed(
    fused_dir=Path('data/person_01/day_high/fused_npz'),
    smoothed_dir=Path('data/person_01/day_high/smoothed_fused_npz'),
    output_dir=Path('comparison_results'),
    keypoint_indices=[0, 5, 10, 15]
)
```

### 方法3: 命令行使用

```bash
python code/head3D_fuse/compare_fused_smoothed.py \
    --fused_dir /path/to/fused_npz \
    --smoothed_dir /path/to/smoothed_fused_npz \
    --output_dir /path/to/output \
    --keypoints 0 5 10
```

**参数说明**:
- `--fused_dir`: 融合关键点的npz文件目录
- `--smoothed_dir`: 平滑关键点的npz文件目录
- `--output_dir`: 结果输出目录
- `--keypoints`: 要可视化的关键点索引（可选，默认前3个）

## 📈 输出结果

### 1. 指标JSON (`metrics.json`)
```json
{
  "mean_difference": 0.292619,
  "max_difference": 0.796787,
  "std_difference": 0.145329,
  "median_difference": 0.270336,
  "fused_mean_velocity": 0.440058,
  "smoothed_mean_velocity": 0.272138,
  "velocity_reduction": 38.16,
  "fused_mean_acceleration": 0.427930,
  "smoothed_mean_acceleration": 0.097098,
  "acceleration_reduction": 77.31,
  "fused_jitter": 0.178466,
  "smoothed_jitter": 0.030695,
  "jitter_reduction": 82.80
}
```

### 2. 文本报告 (`comparison_report.txt`)
```
======================================================================
融合关键点 vs 平滑关键点 - 对比报告
======================================================================

数据概览:
  帧数: 100
  关键点数: 17
  数据形状: (100, 17, 3)

----------------------------------------------------------------------
差异指标:
----------------------------------------------------------------------
  平均差异:   0.292619
  最大差异:   0.796787
  标准差:     0.145329
  中位数:     0.270336

----------------------------------------------------------------------
平滑度指标 - 速度:
----------------------------------------------------------------------
  融合后平均速度:     0.440058
  平滑后平均速度:     0.272138
  速度降低:           38.16%

----------------------------------------------------------------------
平滑度指标 - 加速度（抖动）:
----------------------------------------------------------------------
  融合后平均加速度:   0.427930
  平滑后平均加速度:   0.097098
  加速度降低:         77.31%

----------------------------------------------------------------------
抖动分数:
----------------------------------------------------------------------
  融合后抖动分数:     0.178466
  平滑后抖动分数:     0.030695
  抖动降低:           82.80%

======================================================================
评估结论:
======================================================================
  ✓ 优秀: 加速度显著降低，抖动明显减少
  ✓ 平滑保真度高: 与原始数据差异很小
======================================================================
```

### 3. 可视化图表

#### `trajectory_comparison.png`
显示选定关键点在X、Y、Z三个轴上的轨迹对比：
- 蓝色点线: 融合后的原始轨迹（有噪声）
- 橙色线: 平滑后的轨迹（更光滑）

#### `metrics_comparison.png`
包含4个子图：
1. **Frame-wise Difference**: 每帧的平均差异
2. **Velocity Magnitude**: 速度大小对比
3. **Acceleration Magnitude**: 加速度大小对比
4. **Mean Difference per Keypoint**: 每个关键点的平均差异

## 📐 指标解读

### 差异指标
| 平均差异 | 评价 | 说明 |
|---------|------|------|
| < 0.01 | 优秀 | 平滑后几乎不改变原始数据 |
| 0.01-0.05 | 良好 | 有一定差异但可接受 |
| 0.05-0.1 | 一般 | 差异较大，可能需要调整参数 |
| > 0.1 | 较差 | 过度平滑，丢失细节 |

### 加速度降低
| 降低百分比 | 评价 | 说明 |
|-----------|------|------|
| > 70% | 优秀 | 显著减少抖动 |
| 50-70% | 良好 | 明显改善 |
| 30-50% | 一般 | 有一定效果 |
| < 30% | 较差 | 效果不明显，考虑调整参数 |

### 抖动降低
| 降低百分比 | 评价 |
|-----------|------|
| > 80% | 非常平滑 |
| 60-80% | 平滑效果好 |
| 40-60% | 中等效果 |
| < 40% | 效果有限 |

## 🎯 使用场景

### 场景1: 评估平滑参数
不同的平滑参数会产生不同的效果，使用比较工具可以帮助选择最佳参数：

```python
# 测试不同的 sigma 值
for sigma in [0.5, 1.0, 1.5, 2.0, 2.5]:
    # 应用平滑
    smoothed = smooth_keypoints_sequence(fused_kpts, method="gaussian", sigma=sigma)
    
    # 比较
    comparator = KeypointsComparator(fused_kpts, smoothed)
    metrics = comparator.compute_metrics()
    
    print(f"Sigma={sigma}: 抖动降低={metrics['jitter_reduction']:.2f}%, "
          f"差异={metrics['mean_difference']:.6f}")
```

### 场景2: 比较不同平滑方法
```python
methods = ["gaussian", "savgol", "kalman", "bilateral"]

for method in methods:
    smoothed = smooth_keypoints_sequence(fused_kpts, method=method)
    comparator = KeypointsComparator(fused_kpts, smoothed)
    metrics = comparator.compute_metrics()
    
    print(f"{method}: 抖动降低={metrics['jitter_reduction']:.2f}%")
```

### 场景3: 批量评估
```python
from pathlib import Path

persons = ["01", "02", "03"]
envs = ["day_high", "day_low", "night_high"]

for person in persons:
    for env in envs:
        fused_dir = Path(f"data/person_{person}/{env}/fused_npz")
        smoothed_dir = Path(f"data/person_{person}/{env}/smoothed_fused_npz")
        output_dir = Path(f"comparison/person_{person}/{env}")
        
        if fused_dir.exists() and smoothed_dir.exists():
            compare_fused_and_smoothed(fused_dir, smoothed_dir, output_dir)
```

## 💡 实用建议

### 1. 选择可视化的关键点
选择代表性的关键点进行可视化：
```python
# 头部关键点
keypoint_indices = [0, 1, 2, 3, 4]  # 鼻子、眼睛、耳朵

# 手部关键点
keypoint_indices = [21, 30, 42, 51]  # 左右手腕和中指

# 混合
keypoint_indices = [0, 5, 21, 42]  # 鼻子、肩膀、双手
```

### 2. 解读结果
- **差异小但抖动降低大** → 理想效果
- **差异大且抖动降低大** → 过度平滑，考虑减小平滑参数
- **差异小但抖动降低小** → 平滑不足，考虑增大平滑参数

### 3. 调整平滑参数
根据比较结果调整配置：
```yaml
# 如果差异太大（> 0.05）
temporal_smooth_sigma: 1.0  # 减小 sigma

# 如果抖动降低不够（< 50%）
temporal_smooth_sigma: 2.5  # 增大 sigma
```

## 📚 相关文档

- 时间平滑模块: `code/head3D_fuse/temporal_smooth.py`
- 平滑文档: `code/head3D_fuse/TEMPORAL_SMOOTHING.md`
- 配置示例: `code/configs/temporal_smooth_config_example.yaml`

## 🧪 测试

运行测试脚本查看演示：
```bash
cd /workspace
python3 code/analysis/test_comparison_tool.py
```

查看生成的结果：
```bash
ls -la /workspace/test_comparison_output/
```
