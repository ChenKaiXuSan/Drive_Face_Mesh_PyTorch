# Head Movement Analysis - 快速开始

## 📋 功能概述

这个模块用于分析融合后的3D关键点数据，计算头部的三个转动角度：
- **Pitch (俯仰角)**: 上下点头
- **Yaw (偏航角)**: 左右转头
- **Roll (翻滚角)**: 头部左右倾斜

## 🚀 快速开始

### 1. 分析单个文件
```bash
cd /workspace/code
python head_movement_analysis/main.py
```

### 2. 导出为CSV文件
```bash
cd /workspace/code/head_movement_analysis

# 导出整个序列
python export_to_csv.py \
  --input /workspace/data/head3d_fuse_results/01/夜多い/fused_npz \
  --output results.csv

# 指定帧范围
python export_to_csv.py \
  --input /workspace/data/head3d_fuse_results/01/夜多い/fused_npz \
  --output results.csv \
  --start 619 \
  --end 1000
```

### 3. 可视化角度变化
```bash
cd /workspace/code/head_movement_analysis

# 生成时间序列图
python visualize_angles.py \
  --input /workspace/data/head3d_fuse_results/01/夜多い/fused_npz \
  --output angles_plot.png \
  --no-show

# 生成3D轨迹图
python visualize_angles.py \
  --input /workspace/data/head3d_fuse_results/01/夜多い/fused_npz \
  --output angles_3d.png \
  --mode 3d \
  --no-show

# 同时生成两种图
python visualize_angles.py \
  --input /workspace/data/head3d_fuse_results/01/夜多い/fused_npz \
  --output angles.png \
  --mode both \
  --no-show
```

## 💻 Python API 使用

### 基本使用
```python
from pathlib import Path
from head_movement_analysis.main import HeadPoseAnalyzer

# 创建分析器
analyzer = HeadPoseAnalyzer()

# 分析单帧
npy_path = Path("/workspace/data/head3d_fuse_results/01/夜多い/fused_npz/frame_000619_fused.npy")
result = analyzer.analyze_head_pose(npy_path)

print(f"Pitch: {result['pitch']:.2f}°")
print(f"Yaw: {result['yaw']:.2f}°")
print(f"Roll: {result['roll']:.2f}°")
```

### 批量分析
```python
# 分析整个序列
fused_dir = Path("/workspace/data/head3d_fuse_results/01/夜多い/fused_npz")
results = analyzer.analyze_sequence(fused_dir, start_frame=619, end_frame=1000)

# 遍历结果
for frame_idx in sorted(results.keys()):
    angles = results[frame_idx]
    print(f"Frame {frame_idx}: Pitch={angles['pitch']:.2f}°, "
          f"Yaw={angles['yaw']:.2f}°, Roll={angles['roll']:.2f}°")
```

### 统计分析
```python
import numpy as np

# 提取所有角度
pitches = [angles['pitch'] for angles in results.values()]
yaws = [angles['yaw'] for angles in results.values()]
rolls = [angles['roll'] for angles in results.values()]

# 计算统计量
print(f"Pitch - Mean: {np.mean(pitches):.2f}°, Std: {np.std(pitches):.2f}°")
print(f"Yaw   - Mean: {np.mean(yaws):.2f}°, Std: {np.std(yaws):.2f}°")
print(f"Roll  - Mean: {np.mean(rolls):.2f}°, Std: {np.std(rolls):.2f}°")
```

## 📊 输出格式

### CSV格式
```csv
frame_idx,pitch_deg,yaw_deg,roll_deg
619,-49.81,-2.14,177.84
620,-49.81,-2.01,177.05
621,-49.82,-2.62,177.87
```

### Python字典格式
```python
{
    619: {'pitch': -49.81, 'yaw': -2.14, 'roll': 177.84},
    620: {'pitch': -49.81, 'yaw': -2.01, 'roll': 177.05},
    621: {'pitch': -49.82, 'yaw': -2.62, 'roll': 177.87},
}
```

## 📐 角度解释

### Pitch (俯仰角)
- **范围**: -90° 到 +90°
- **正值**: 抬头（向上看）
- **负值**: 低头（向下看）
- **0°**: 平视前方

### Yaw (偏航角)
- **范围**: -180° 到 +180°
- **正值**: 向右转头
- **负值**: 向左转头
- **0°**: 正对前方

### Roll (翻滚角)
- **范围**: -180° 到 +180°
- **正值**: 头部向右倾斜
- **负值**: 头部向左倾斜
- **0°**: 头部竖直

## 🔍 示例输出

```
2026-02-06 15:21:29,699 - __main__ - INFO - 单帧分析结果:
2026-02-06 15:21:29,699 - __main__ - INFO -   俯仰角 (Pitch): -49.81°
2026-02-06 15:21:29,699 - __main__ - INFO -   偏航角 (Yaw): -2.14°
2026-02-06 15:21:29,699 - __main__ - INFO -   翻滚角 (Roll): 177.84°
2026-02-06 15:21:29,791 - __main__ - INFO - Successfully analyzed 11 frames
```

## 📁 文件结构

```
head_movement_analysis/
├── main.py                  # 主分析模块
├── export_to_csv.py         # CSV导出工具
├── visualize_angles.py      # 可视化工具
├── README.md                # 详细文档
└── QUICKSTART.md           # 本文档
```

## 🛠️ 依赖项

- numpy
- matplotlib
- logging
- pathlib

## ⚠️ 注意事项

1. 确保输入的关键点数据已经过融合处理
2. 如果关键点无效（NaN或无穷大），该帧会被跳过
3. Roll角接近±180°时可能有符号翻转（正常现象）
4. 可视化时中文字体警告不影响功能

## 📞 联系方式

- 作者: Kaixu Chen
- 邮箱: chenkaixusan@gmail.com
- 机构: The University of Tsukuba
