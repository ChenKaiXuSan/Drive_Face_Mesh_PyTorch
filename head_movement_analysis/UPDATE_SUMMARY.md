# Head Movement Analysis - 功能更新总结

## 🎉 新增功能（2026-02-07）

### 1. 模块化重构

#### load.py - 数据加载模块
- ✅ 移动了 `load_fused_keypoints()` 函数到独立模块
- ✅ 新增 `HeadMovementLabel` 数据类，用于存储标注信息
- ✅ 新增 `load_head_movement_annotations()` 函数，加载LabelStudio标注
- ✅ 新增 `get_annotation_for_frame()` 辅助函数
- ✅ 新增 `get_all_annotations_for_frame()` 辅助函数

#### main.py - 核心分析模块增强
- ✅ 集成标注功能：`HeadPoseAnalyzer` 现在可选地接受 `annotation_dict` 参数
- ✅ 新增 `compare_with_annotations()` 方法，比较计算角度与标注
- ✅ 新增 `analyze_sequence_with_annotations()` 方法，批量分析并比较
- ✅ 支持自定义阈值进行匹配判断

### 2. 标注比较功能

#### 支持的标注类型
| 标注标签 | 对应角度 | 判断条件 |
|---------|---------|---------|
| `right` | yaw | yaw > threshold° |
| `left` | yaw | yaw < -threshold° |
| `up` | pitch | pitch > threshold° |
| `down` | pitch | pitch < -threshold° |

#### 比较结果包含
- 标注的时间范围（start_frame, end_frame）
- 标注的动作类型（right/left/up/down）
- 计算出的角度值
- 匹配状态（是否符合预期）

### 3. 新增工具脚本

#### test_annotation.py
- 测试标注加载和比较功能
- 显示匹配率统计
- 输出详细的比较结果

#### analyze_thresholds.py
- 分析不同标注类型的角度分布
- 测试不同阈值（5°-30°）下的匹配率
- 提供数据统计（平均值、中位数、标准差、范围）

### 4. 文档完善

新增文档：
- **ANNOTATION_GUIDE.md**: 标注比较功能详细使用指南
- **本文档**: 功能更新总结

更新文档：
- **__init__.py**: 导出新的类和函数
- **README.md**: 保持完整的API文档

## 📊 测试结果

### 基本功能测试
```bash
✓ HeadPoseAnalyzer 创建成功
✓ 单帧分析成功
✓ 序列分析成功 (5帧)
所有测试通过! ✓
```

### 标注比较测试（01_day_high, frames 380-520）

#### 数据统计
- **加载标注**: 88个视频，成功 ✓
- **分析帧数**: 71帧
- **有标注帧数**: 64帧

#### 角度分布
| 标注类型 | 样本数 | 平均值 | 标准差 | 范围 |
|---------|--------|--------|--------|------|
| down (pitch) | 23 | -42.04° | 0.86° | -43.79° ~ -40.75° |
| right (yaw) | 64 | 0.24° | 2.45° | -8.49° ~ 3.44° |

#### 匹配率（15°阈值）
- **down标注**: ✓ 100% 匹配 (23/23)
- **right标注**: ✗ 0% 匹配 (0/64)
- **总体**: 26.4% (23/87)

#### 结论
1. ✅ **Pitch角度计算准确**：down动作能100%匹配，平均-42°
2. ⚠️ **Yaw角度偏小**：right标注区间的yaw平均仅0.24°
   - 可能原因：相机视角、坐标系定义、标注粒度
   - 建议：降低yaw阈值至5°，或检查坐标系

## 🚀 使用示例

### 基本使用（无标注）
```python
from head_movement_analysis import HeadPoseAnalyzer
from pathlib import Path

# 创建分析器
analyzer = HeadPoseAnalyzer()

# 分析单帧
result = analyzer.analyze_head_pose(
    Path("frame_000619_fused.npy")
)
print(f"Pitch: {result['pitch']:.2f}°")
```

### 带标注比较
```python
from head_movement_analysis import (
    HeadPoseAnalyzer,
    load_head_movement_annotations,
)
from pathlib import Path

# 1. 加载标注
annotations = load_head_movement_annotations(
    Path("/workspace/data/annotation/label/full.json")
)

# 2. 创建分析器
analyzer = HeadPoseAnalyzer(annotation_dict=annotations)

# 3. 分析并比较
results = analyzer.analyze_sequence_with_annotations(
    video_id="01_day_high",
    fused_dir=Path("/workspace/data/head3d_fuse_results/01/昼多い/fused_npz"),
    start_frame=380,
    end_frame=520,
)

# 4. 查看结果
angles = results["angles"]
comparisons = results["comparisons"]

# 5. 统计匹配率
total = sum(len(c["matches"]) for c in comparisons.values())
matched = sum(
    sum(1 for m in c["matches"] if m["is_match"])
    for c in comparisons.values()
)
print(f"匹配率: {matched}/{total} = {matched/total*100:.1f}%")
```

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

## 📁 文件结构

```
head_movement_analysis/
├── __init__.py                 # 包初始化，导出API
├── load.py                     # 【新增】数据加载模块
├── main.py                     # 【更新】核心分析模块
├── export_to_csv.py            # CSV导出工具
├── visualize_angles.py         # 可视化工具
├── test.py                     # 基本功能测试
├── test_annotation.py          # 【新增】标注比较测试
├── analyze_thresholds.py       # 【新增】阈值分析工具
├── requirements.txt            # Python依赖
├── README.md                   # 完整API文档
├── QUICKSTART.md              # 快速开始指南
├── SUMMARY.md                  # 实现总结
├── ANNOTATION_GUIDE.md         # 【新增】标注比较指南
└── UPDATE_SUMMARY.md          # 【本文档】功能更新总结
```

## 🔧 API变更

### 新增导出
```python
from head_movement_analysis import (
    # 核心类
    HeadPoseAnalyzer,        # 头部姿态分析器（支持标注）
    KEYPOINT_INDICES,        # 关键点索引常量
    
    # 数据类
    HeadMovementLabel,       # 【新增】标注数据类
    
    # 加载函数
    load_fused_keypoints,    # 【新增】加载3D关键点
    load_head_movement_annotations,  # 【新增】加载标注
    
    # 辅助函数
    get_annotation_for_frame,        # 【新增】获取单个标注
    get_all_annotations_for_frame,   # 【新增】获取所有标注
)
```

### HeadPoseAnalyzer 变更

#### 构造函数
```python
# 旧版本
analyzer = HeadPoseAnalyzer()

# 新版本（兼容旧版本）
analyzer = HeadPoseAnalyzer(annotation_dict=None)  # 不使用标注
analyzer = HeadPoseAnalyzer(annotation_dict=annotations)  # 使用标注
```

#### 新增方法
```python
# 比较单帧
comparison = analyzer.compare_with_annotations(
    video_id, frame_idx, angles, threshold_deg=15.0
)

# 分析序列并比较
results = analyzer.analyze_sequence_with_annotations(
    video_id, fused_dir, start_frame=None, end_frame=None
)
```

## 🎯 下一步建议

### 1. 调整Yaw计算
- 检查坐标系定义是否正确
- 考虑相机外参的影响
- 可能需要相对基线计算而非绝对值

### 2. 自适应阈值
```python
# 对不同角度类型使用不同阈值
THRESHOLDS = {
    'pitch': 20.0,  # down/up较明显
    'yaw': 5.0,     # right/left较微妙
    'roll': 15.0,
}
```

### 3. 增量检测
```python
# 检测角度变化而非绝对值
delta_yaw = current_yaw - baseline_yaw
if abs(delta_yaw) > threshold:
    # 检测到转头动作
```

### 4. 时序平滑
- 对角度序列进行时序平滑
- 减少抖动，提高稳定性

### 5. 更多可视化
- 可视化标注区间
- 绘制角度曲线与标注区间的对比图
- 生成匹配/不匹配的报告

## ✅ 完成的任务清单

- [x] 将load功能移到load.py模块
- [x] 实现HeadMovementLabel数据类
- [x] 实现load_head_movement_annotations()函数
- [x] 解析LabelStudio JSON格式
- [x] 在HeadPoseAnalyzer中集成标注字典
- [x] 实现compare_with_annotations()方法
- [x] 实现analyze_sequence_with_annotations()方法
- [x] 支持可配置的匹配阈值
- [x] 创建test_annotation.py测试脚本
- [x] 创建analyze_thresholds.py分析工具
- [x] 编写ANNOTATION_GUIDE.md文档
- [x] 更新__init__.py导出新API
- [x] 测试和验证所有功能
- [x] 生成UPDATE_SUMMARY.md总结文档

## 📞 联系方式

- **作者**: Kaixu Chen
- **邮箱**: chenkaixusan@gmail.com
- **机构**: The University of Tsukuba
- **日期**: February 7, 2026

## 📝 版本历史

### v1.1.0 (2026-02-07)
- ✨ 新增标注比较功能
- ✨ 模块化重构（load.py）
- ✨ 新增测试和分析工具
- 📚 完善文档

### v1.0.0 (2026-02-07)
- ✨ 初始版本发布
- ✅ 实现核心角度计算
- ✅ CSV导出和可视化工具
