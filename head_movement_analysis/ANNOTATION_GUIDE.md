# 头部动作标注比较功能使用指南

##  概述

该功能可以将计算出的头部角度与人工标注进行比较，用于验证算法准确性。

## 快速使用

### 1. 加载标注并创建分析器

```python
from pathlib import Path
from head_movement_analysis import (
    HeadPoseAnalyzer,
    load_head_movement_annotations,
)

# 加载标注文件
annotation_path = Path("/workspace/data/annotation/label/full.json")
annotations = load_head_movement_annotations(annotation_path)

# 创建带标注的分析器
analyzer = HeadPoseAnalyzer(annotation_dict=annotations)
```

### 2. 分析并比较

```python
# 分析序列并与标注比较
results = analyzer.analyze_sequence_with_annotations(
    video_id="01_day_high",  # 视频ID
    fused_dir=Path("/workspace/data/head3d_fuse_results/01/昼多い/fused_npz"),
    start_frame=380,
    end_frame=520,
)

# 获取结果
angles = results["angles"]  # 计算的角度
comparisons = results["comparisons"]  # 与标注的比较结果
```

### 3. 查看比较结果

```python
for frame_idx, comparison in comparisons.items():
    print(f"\nFrame {frame_idx}:")
    print(f"  角度: Pitch={comparison['angles']['pitch']:.2f}°, "
          f"Yaw={comparison['angles']['yaw']:.2f}°, "
          f"Roll={comparison['angles']['roll']:.2f}°")
    
    for match_info in comparison["matches"]:
        annotation = match_info["annotation"]
        print(f"  标注: {annotation.label} [{annotation.start_frame}-{annotation.end_frame}]")
        print(f"    角度类型: {match_info['angle_type']}")
        print(f"    角度值: {match_info['angle_value']:.2f}°")
        print(f"    匹配: {'✓' if match_info['is_match'] else '✗'}")
```

## 标注格式

### JSON结构

标注文件是LabelStudio导出的JSON格式，包含：

```json
[
  {
    "id": 1,
    "data": {
      "video": "/data/local-files/?d=mydata/drive_data/person_01_day_high_h265.mp4"
    },
    "annotations": [
      {
        "result": [
          {
            "value": {
              "ranges": [{"start": 382, "end": 445}],
              "timelinelabels": ["right"]
            },
            "type": "timelinelabels"
          }
        ]
      }
    ]
  }
]
```

### 支持的标注标签

| 标签 | 对应角度 | 预期方向 | 说明 |
|------|----------|----------|------|
| `right` | yaw | positive (>0) | 向右转头 |
| `left` | yaw | negative (<0) | 向左转头 |
| `up` | pitch | positive (>0) | 抬头 |
| `down` | pitch | negative (<0) | 低头 |

## 比较逻辑

### 默认阈值

默认使用 **15度** 作为判断阈值：
- 如果标注为 `right`，要求 yaw > 15°
- 如果标注为 `left`，要求 yaw < -15°
- 如果标注为 `up`，要求 pitch > 15°
- 如果标注为 `down`，要求 pitch < -15°

### 自定义阈值

```python
# 单帧比较
comparison = analyzer.compare_with_annotations(
    video_id="01_day_high",
    frame_idx=400,
    angles={'pitch': -42.0, 'yaw': 1.5, 'roll': -165.0},
    threshold_deg=10.0  # 使用10度阈值
)
```

## 测试脚本

### 基本测试

```bash
cd /workspace/code/head_movement_analysis
python test_annotation.py
```

### 分析不同阈值

```bash
python analyze_thresholds.py
```

这会输出：
- 不同标注类型的角度统计（平均值、标准差、范围）
- 不同阈值（5°-30°）下的匹配率

## 实验结果

基于 `01_day_high` 视频（frame 380-520）的测试：

### 角度统计

| 标注类型 | 样本数 | 平均值 | 中位数 | 标准差 | 范围 |
|---------|--------|--------|--------|--------|------|
| down (pitch) | 23 | -42.04° | -41.88° | 0.86° | -43.79° ~ -40.75° |
| right (yaw) | 64 | 0.24° | 0.71° | 2.45° | -8.49° ~ 3.44° |

### 匹配率

使用15°阈值：
- **总匹配: 26.4%** (23/87)
- down标注：✓ **100%匹配** (pitch平均-42°，远超阈值)
- right标注：✗ **0%匹配** (yaw平均0.24°，远低于阈值)

## 当前限制与建议

### 1. Yaw角度较小

**现象**：right/left标注区间的yaw角度很小（平均<5°）

**可能原因**：
- 相机视角问题：可能相机不是完全正对驾驶员
- 坐标系定义：可能需要调整坐标系转换
- 标注粒度：标注的"right"可能指轻微转头，而非大幅度转头

**建议**：
- 对于yaw，使用更小的阈值（如5°）
- 或使用相对变化量而非绝对值
- 检查并调整坐标系定义

### 2. 调整阈值

根据数据分布调整阈值：

```python
# 对不同标注类型使用不同阈值
CUSTOM_THRESHOLDS = {
    'pitch': 20.0,  # down/up使用20度
    'yaw': 5.0,     # right/left使用5度
    'roll': 15.0,   # 倾斜使用15度
}
```

### 3. 使用相对变化

考虑使用角度变化量而非绝对值：

```python
# 比较当前帧与前几帧的角度变化
delta_yaw = current_yaw - baseline_yaw
if delta_yaw > threshold:
    # 检测到向右转头
```

## API参考

### HeadPoseAnalyzer

```python
class HeadPoseAnalyzer:
    def __init__(self, annotation_dict: Optional[Dict[str, List[HeadMovementLabel]]] = None)
```

**参数**：
- `annotation_dict`: 可选的标注字典，加载自 `load_head_movement_annotations()`

**方法**：

#### `compare_with_annotations()`
```python
def compare_with_annotations(
    self,
    video_id: str,
    frame_idx: int,
    angles: Dict[str, float],
    threshold_deg: float = 15.0,
) -> Optional[Dict]
```

比较单帧的角度与标注。

#### `analyze_sequence_with_annotations()`
```python
def analyze_sequence_with_annotations(
    self,
    video_id: str,
    fused_dir: Path,
    start_frame: int = None,
    end_frame: int = None,
) -> Dict
```

分析整个序列并与标注比较。

**返回**：
```python
{
    "angles": {frame_idx: {'pitch': float, 'yaw': float, 'roll': float}},
    "comparisons": {frame_idx: comparison_result}
}
```

### 辅助函数

```python
# 加载标注
def load_head_movement_annotations(json_path: Path) -> Dict[str, List[HeadMovementLabel]]

# 获取单个标注
def get_annotation_for_frame(annotations: List[HeadMovementLabel], frame_idx: int) -> Optional[HeadMovementLabel]

# 获取所有标注（可能重叠）
def get_all_annotations_for_frame(annotations: List[HeadMovementLabel], frame_idx: int) -> List[HeadMovementLabel]
```

## 完整示例

```python
from pathlib import Path
from head_movement_analysis import (
    HeadPoseAnalyzer,
    load_head_movement_annotations,
)

# 1. 加载标注
annotations = load_head_movement_annotations(
    Path("/workspace/data/annotation/label/full.json")
)
print(f"加载了 {len(annotations)} 个视频的标注")

# 2. 创建分析器
analyzer = HeadPoseAnalyzer(annotation_dict=annotations)

# 3. 分析序列
results = analyzer.analyze_sequence_with_annotations(
    video_id="01_day_high",
    fused_dir=Path("/workspace/data/head3d_fuse_results/01/昼多い/fused_npz"),
    start_frame=380,
    end_frame=520,
)

# 4. 统计匹配率
total = 0
matched = 0
for comparison in results["comparisons"].values():
    for match_info in comparison["matches"]:
        total += 1
        if match_info["is_match"]:
            matched += 1

print(f"匹配率: {matched}/{total} = {matched/total*100:.1f}%")

# 5. 分析角度分布
import numpy as np
down_pitches = []
right_yaws = []

for comparison in results["comparisons"].values():
    for match_info in comparison["matches"]:
        label = match_info["annotation"].label
        angle_value = match_info["angle_value"]
        
        if label == "down":
            down_pitches.append(angle_value)
        elif label == "right":
            right_yaws.append(angle_value)

print(f"\ndown标注的pitch: 平均={np.mean(down_pitches):.2f}°")
print(f"right标注的yaw: 平均={np.mean(right_yaws):.2f}°")
```

## 作者

Kaixu Chen (chenkaixusan@gmail.com)  
The University of Tsukuba  
February 7, 2026
