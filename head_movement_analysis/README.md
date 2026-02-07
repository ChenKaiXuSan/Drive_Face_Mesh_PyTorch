# 头部姿态分析 (Head Pose Analysis)

## 功能说明

该模块用于从融合后的3D关键点数据中计算头部的三个转动角度：

1. **Pitch（俯仰角）**：上下点头的角度
   - 正值：抬头
   - 负值：低头
   - 范围：约 -90° 到 +90°

2. **Yaw（偏航角）**：左右转头的角度
   - 正值：向右转
   - 负值：向左转
   - 范围：约 -180° 到 +180°

3. **Roll（翻滚角）**：头部左右倾斜的角度
   - 正值：向右倾斜
   - 负值：向左倾斜
   - 范围：约 -180° 到 +180°

## 使用方法

### 1. 基本使用

```python
from pathlib import Path
from head_movement_analysis.main import HeadPoseAnalyzer

# 创建分析器
analyzer = HeadPoseAnalyzer()

# 分析单个帧
npy_path = Path("/path/to/frame_000619_fused.npy")
result = analyzer.analyze_head_pose(npy_path)

if result:
    print(f"俯仰角 (Pitch): {result['pitch']:.2f}°")
    print(f"偏航角 (Yaw): {result['yaw']:.2f}°")
    print(f"翻滚角 (Roll): {result['roll']:.2f}°")
```

### 2. 批量分析序列

```python
from pathlib import Path
from head_movement_analysis.main import HeadPoseAnalyzer

# 创建分析器
analyzer = HeadPoseAnalyzer()

# 分析整个序列
fused_dir = Path("/workspace/data/head3d_fuse_results/01/夜多い/fused_npz")
results = analyzer.analyze_sequence(fused_dir, start_frame=619, end_frame=1000)

# 遍历结果
for frame_idx, angles in sorted(results.items()):
    print(f"Frame {frame_idx}: "
          f"Pitch={angles['pitch']:6.2f}°, "
          f"Yaw={angles['yaw']:6.2f}°, "
          f"Roll={angles['roll']:6.2f}°")
```

### 3. 导出结果到CSV

```python
import csv
from pathlib import Path
from head_movement_analysis.main import HeadPoseAnalyzer

analyzer = HeadPoseAnalyzer()
fused_dir = Path("/workspace/data/head3d_fuse_results/01/夜多い/fused_npz")
results = analyzer.analyze_sequence(fused_dir)

# 保存为CSV
output_csv = Path("head_pose_angles.csv")
with output_csv.open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['frame_idx', 'pitch', 'yaw', 'roll'])
    
    for frame_idx in sorted(results.keys()):
        angles = results[frame_idx]
        writer.writerow([
            frame_idx,
            f"{angles['pitch']:.2f}",
            f"{angles['yaw']:.2f}",
            f"{angles['roll']:.2f}"
        ])

print(f"Results saved to {output_csv}")
```

## 数据格式

### 输入数据

- 文件格式：`.npy` 文件（融合后的3D关键点）
- 文件名格式：`frame_{frame_idx:06d}_fused.npy`
- 数据结构：
  ```python
  {
      'fused_keypoints_3d': np.ndarray,  # shape: (70, 3)
      'fused_mask': np.ndarray,
      'valid_views': int,
      'npz_paths': dict,
  }
  ```

### 输出数据

返回字典包含三个角度（单位：度）：
```python
{
    'pitch': float,  # 俯仰角
    'yaw': float,    # 偏航角
    'roll': float,   # 翻滚角
}
```

## 关键点说明

使用的MHR70关键点索引：
- 0: nose（鼻子）
- 1: left-eye（左眼）
- 2: right-eye（右眼）
- 3: left-ear（左耳）
- 4: right-ear（右耳）
- 5: left-shoulder（左肩）
- 6: right-shoulder（右肩）
- 69: neck（颈部）

## 计算原理

### Pitch（俯仰角）
使用鼻子和颈部连线与水平面的夹角来计算头部的抬头/低头角度。

### Yaw（偏航角）
使用眼睛中心到鼻子的向量在水平面上的投影，计算头部左右转动的角度。

### Roll（翻滚角）
使用左右眼睛的连线与水平面的夹角，计算头部左右倾斜的角度。

## 运行示例

直接运行脚本查看示例：
```bash
cd /workspace/code
python head_movement_analysis/main.py
```

示例输出：
```
2026-02-06 15:21:29,699 - __main__ - INFO - 单帧分析结果:
2026-02-06 15:21:29,699 - __main__ - INFO -   俯仰角 (Pitch): -49.81°
2026-02-06 15:21:29,699 - __main__ - INFO -   偏航角 (Yaw): -2.14°
2026-02-06 15:21:29,699 - __main__ - INFO -   翻滚角 (Roll): 177.84°
2026-02-06 15:21:29,791 - __main__ - INFO - Successfully analyzed 11 frames
```

## API文档

### `HeadPoseAnalyzer`

#### `load_fused_keypoints(npy_path: Path) -> Optional[np.ndarray]`
读取融合后的3D关键点数据。

**参数：**
- `npy_path`: .npy文件路径

**返回：**
- 形状为 (70, 3) 的3D关键点数组，失败返回None

#### `analyze_head_pose(npy_path: Path) -> Optional[Dict[str, float]]`
分析单帧的头部姿态。

**参数：**
- `npy_path`: 融合后的.npy文件路径

**返回：**
- 包含三个角度的字典，失败返回None

#### `analyze_sequence(fused_dir: Path, start_frame: int = None, end_frame: int = None) -> Dict[int, Dict[str, float]]`
分析序列中所有帧的头部姿态。

**参数：**
- `fused_dir`: 包含融合npy文件的目录
- `start_frame`: 起始帧索引（可选）
- `end_frame`: 结束帧索引（可选）

**返回：**
- 字典，键为帧索引，值为包含三个角度的字典

## 依赖项

- numpy
- logging
- pathlib

## 注意事项

1. 确保输入的关键点数据已经经过融合处理
2. 如果关键点无效（NaN或无穷大），该帧会被跳过
3. 角度计算基于标准的欧拉角定义
4. Roll角可能接近±180°时会有符号翻转，这是正常的数学现象

## 作者

Kaixu Chen (chenkaixusan@gmail.com)
The University of Tsukuba
