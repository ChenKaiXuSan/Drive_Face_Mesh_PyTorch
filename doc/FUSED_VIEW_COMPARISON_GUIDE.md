# 融合结果与单视角关键点对比评估

## 概述

`FusedViewComparator` 是一个用于评估融合后的3D关键点与各个单视角3D关键点质量差异的工具类。它可以帮助你：

1. 量化融合结果相对于单视角结果的改进程度
2. 了解融合结果是否合理（是否在各视角的"中心"）
3. 评估时间稳定性的提升
4. 分析各视角之间的一致性

## 评估指标

### 1. 与各视角的距离
- **平均欧氏距离**：融合关键点与每个单视角关键点之间的平均距离
- **标准差**：距离的波动程度
- **用途**：评估融合结果是否偏向某个特定视角

### 2. 与质心的距离
- **质心**：所有视角关键点的平均位置
- **距离**：融合结果与质心的距离
- **用途**：越小说明融合越"居中"，越接近理想的融合结果

### 3. 时间稳定性（Jitter）
- **定义**：关键点轨迹的二阶差分（加速度）的范数
- **对比**：融合结果 vs 各单视角的 jitter
- **用途**：量化融合带来的平滑效果

### 4. 视角一致性
- **定义**：两两视角之间的关键点差异
- **用途**：了解原始数据的质量，一致性高说明原始数据准确

### 5. 每个关键点的详细统计
- 每个关键点与各视角的距离
- 关键点级别的质量分析

## 使用方法

### 1. 在配置文件中启用

在你的 YAML 配置文件中添加：

```yaml
infer:
  # 启用融合结果与单视角的对比评估
  enable_fused_view_comparison: true
  
  # 要评估的关键点索引（默认前7个）
  fused_view_comparison_keypoint_indices: [0, 1, 2, 3, 4, 5, 6]
  
  # 是否生成可视化图表
  enable_fused_view_comparison_plots: true
  
  view_list: ["front", "left", "right"]
```

### 2. 在代码中直接使用

```python
from head3D_fuse.fuse.compare_fused import FusedViewComparator
import numpy as np

# 准备数据
fused_keypoints = np.array(...)  # shape: (T, N, 3)
view_keypoints = {
    "front": np.array(...),  # shape: (T, N, 3)
    "left": np.array(...),   # shape: (T, N, 3)
    "right": np.array(...),  # shape: (T, N, 3)
}

# 创建比较器
comparator = FusedViewComparator(fused_keypoints, view_keypoints)

# 计算指标（评估前7个关键点）
metrics = comparator.compute_metrics(keypoint_indices=list(range(7)))

# 生成报告
report = comparator.generate_report(
    save_path="comparison_report.txt",
    keypoint_indices=list(range(7))
)

# 生成可视化
comparator.plot_comparison(
    save_path="comparison_plot.png",
    keypoint_indices=list(range(7))
)

# 导出JSON
comparator.export_metrics_json(
    save_path="metrics.json",
    keypoint_indices=list(range(7))
)
```

## 输出结果

运行后会在以下目录生成结果文件：

```
<out_root>/<person_id>/<env_name>/fused_vs_views_comparison/
├── fused_vs_views_metrics.json          # JSON格式的所有指标
├── fused_vs_views_report.txt            # 详细的文本报告
└── fused_vs_views_comparison.png        # 可视化对比图表（6张子图）
```

### 输出示例

#### 文本报告示例

```
================================================================================
融合关键点与单视角关键点对比评估报告
================================================================================
数据维度: 300 帧, 70 关键点, 3 视角
评估的关键点索引: [0, 1, 2, 3, 4, 5, 6]

--------------------------------------------------------------------------------
1. 融合结果与各视角的平均距离（越小越接近该视角）
--------------------------------------------------------------------------------
       front:   0.0234 ±   0.0156
        left:   0.0287 ±   0.0189
       right:   0.0265 ±   0.0172

--------------------------------------------------------------------------------
2. 融合结果与视角质心的距离（越小说明融合越居中）
--------------------------------------------------------------------------------
  Mean: 0.0089
  Std:  0.0067

--------------------------------------------------------------------------------
3. 时间稳定性对比（Jitter = 加速度，越小越稳定）
--------------------------------------------------------------------------------
  融合结果 Jitter: 0.000456 ± 0.000312
  
  各视角 Jitter 及相对融合结果的改善:
       front: 0.000789 ± 0.000534 (改善: +42.15%)
        left: 0.000812 ± 0.000598 (改善: +43.84%)
       right: 0.000756 ± 0.000487 (改善: +39.68%)

--------------------------------------------------------------------------------
4. 视角间一致性（两两视角的差异，越小说明原始数据越一致）
--------------------------------------------------------------------------------
        front_vs_left: 0.0423 ± 0.0189
       front_vs_right: 0.0398 ± 0.0167
         left_vs_right: 0.0445 ± 0.0201
```

#### 可视化图表说明

生成的 PNG 图表包含 6 个子图：

1. **融合结果与各视角的平均距离**：柱状图，显示融合结果偏向哪个视角
2. **时间稳定性对比（Jitter）**：柱状图，对比融合结果与各视角的抖动程度
3. **各关键点与各视角的距离**：热图，显示每个关键点的质量
4. **关键点时序轨迹对比**：线图，显示融合结果如何平滑单视角轨迹
5. **视角间一致性矩阵**：热图，显示视角之间的差异
6. **每帧的平均距离**：时序图，显示融合质量随时间的变化

## 解读建议

### 理想的融合结果应该：

1. **与质心距离小**：说明融合结果在各视角的"中心"，没有偏向某一个视角
2. **Jitter 小于单视角**：说明融合带来了时间平滑效果
3. **与各视角距离相近**：说明融合考虑了所有视角，没有忽略某个视角
4. **视角一致性高**：说明原始数据质量好（这不是融合的指标，而是输入数据的质量）

### 常见问题诊断：

1. **融合结果与某个视角距离特别小**
   - 可能原因：该视角的置信度明显高于其他视角
   - 建议：检查融合算法是否正确考虑了所有视角

2. **融合结果 Jitter 反而增大**
   - 可能原因：融合算法引入了不稳定性
   - 建议：检查对齐方法和融合方法的配置

3. **视角一致性很差**
   - 可能原因：原始检测质量低或相机标定不准确
   - 建议：改进单视角检测或重新标定相机

## API 参考

### FusedViewComparator 类

#### 初始化
```python
FusedViewComparator(fused_keypoints, view_keypoints)
```
- `fused_keypoints`: 融合后的关键点，形状 (T, N, 3)
- `view_keypoints`: 各视角的关键点字典，{"view_name": (T, N, 3)}

#### 方法

##### compute_metrics(keypoint_indices=None)
计算所有评估指标。
- 返回：包含所有指标的字典

##### generate_report(save_path=None, keypoint_indices=None)
生成详细的文本报告。
- `save_path`: 保存路径（可选）
- `keypoint_indices`: 要评估的关键点索引（可选）
- 返回：报告文本字符串

##### plot_comparison(save_path=None, keypoint_indices=None, figsize=(16, 12))
绘制对比可视化图表。
- `save_path`: 保存路径（可选）
- `keypoint_indices`: 要可视化的关键点索引（可选）
- `figsize`: 图表尺寸

##### export_metrics_json(save_path, keypoint_indices=None)
导出指标到JSON文件。
- `save_path`: 保存路径（必需）
- `keypoint_indices`: 要评估的关键点索引（可选）

## 相关文件

- [compare_fused.py](head3D_fuse/fuse/compare_fused.py) - 主要实现
- [compare_fused_example.py](head3D_fuse/fuse/compare_fused_example.py) - 使用示例
- [infer.py](head3D_fuse/infer.py) - 集成代码
- [fused_view_comparison_example.yaml](configs/fused_view_comparison_example.yaml) - 配置示例

## 注意事项

1. 确保所有视角的关键点数组形状一致
2. 处理缺失帧时要保持帧索引对齐
3. 建议至少评估头部关键点（索引0-6）
4. 大量关键点可能导致可视化图表过于密集，建议分批评估

## 许可证

Copyright (c) 2026 The University of Tsukuba
