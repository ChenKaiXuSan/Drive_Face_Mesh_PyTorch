#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
角度计算模块提取总结

=============================================================================

本次重构将所有与头部转角计算相关的代码从 main.py 提取至独立的模块。

=============================================================================

文件结构变化:
  
1. 新增: angle_calculator.py (5.8 KB)
   - 用于处理所有与角度计算相关的逻辑
   
2. 修改: main.py (35.6 KB)
   - 移除了重复的常量和函数定义
   - 现在从 angle_calculator 导入计算相关函数
   - 专注于 CLI 和数据处理逻辑
   
3. 更新: __init__.py (2.1 KB)
   - 添加了新的导出符号

4. 新增: __main__.py (152 bytes)
   - 支持 python -m head_movement_analysis 的入口点

=============================================================================

angle_calculator.py 中的公开接口:

常量:
  - KEYPOINT_INDICES: Dict[str, int]
      关键点名称到MHR70索引的映射
      
  - LABEL_DIRECTION_MAP: Dict[str, Tuple[int, int]]
      标注标签到方向的映射（pitch_dir, yaw_dir）

函数:
  - extract_head_keypoints(keypoints_3d, keypoint_indices)
      从3D关键点数组中提取头部相关关键点
      
  - calculate_head_angles(head_kpts) -> Tuple[float, float, float]
      计算Pitch, Yaw, Roll三个转角（单位：度）
      - Pitch: 俯仰角（上下点头）
      - Yaw: 偏航角（左右转头）
      - Roll: 翻滚角（左右倾斜）
      
  - direction_match(angle_value, expected_dir, threshold) -> bool
      检查角度值是否匹配预期方向
      
  - classify_label(pitch, yaw, threshold) -> str
      根据Pitch和Yaw角度分类标注标签

=============================================================================

转角计算方法（已优化）:

1. Pitch（俯仰角）
   计算鼻子到肩膀中心的向量与水平面的夹角
   pitch = arctan2(nose_shoulder_vec[Y], horizontal_distance)
   正值=抬头，负值=低头

2. Yaw（偏航角）
   计算眼睛中心到鼻子向量在水平面上的投影，相对于前方（-Z轴）
   yaw = arctan2(face_forward[X], -face_forward[Z])
   正值=向右转，负值=向左转

3. Roll（翻滚角）
   计算左右眼睛连线的倾斜角
   roll = arctan2(right_eye_Y - left_eye_Y, right_eye_X - left_eye_X)
   正值=向右倾，负值=向左倾

=============================================================================

使用示例:

from head_movement_analysis import (
    extract_head_keypoints,
    calculate_head_angles,
    classify_label,
    KEYPOINT_INDICES,
)
import numpy as np

# 假设有70个3D关键点
keypoints_3d = np.random.randn(70, 3)

# 提取头部关键点
head_kpts = extract_head_keypoints(keypoints_3d, KEYPOINT_INDICES)

# 计算三个转角
pitch, yaw, roll = calculate_head_angles(head_kpts)

# 分类为标注标签
label = classify_label(pitch, yaw, threshold=15.0)

print(f"Pitch: {pitch:.2f}°, Yaw: {yaw:.2f}°, Roll: {roll:.2f}°")
print(f"Predicted Label: {label}")

=============================================================================

CLI 命令示例（改变不大）:

python -m head_movement_analysis compare \
    --label-dir /workspace/data/multi_view_driver_action/label \
    --base-dir /workspace/data/head3d_fuse_results \
    --output-dir /workspace/data/head_pose_analysis_results \
    --workers 8

=============================================================================

优势:

1. 代码模块化
   - 角度计算逻辑独立于数据处理
   - 便于单元测试和代码维护
   
2. 可复用性
   - 其他项目可以直接导入angle_calculator
   - 不需要依赖整个main.py
   
3. 易于理解
   - 分离的职责使代码更清晰
   - 每个模块有明确的目的
   
4. 便于扩展
   - 可以轻松添加新的角度计算方法
   - 可以独立优化计算性能

=============================================================================

兼容性:

- 所有导出的符号保持相同的函数签名
- 现有代码无需修改即可继续工作
- 完全向后兼容
"""
