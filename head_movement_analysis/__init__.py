#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Head Movement Analysis Module

该模块用于从融合后的3D关键点数据中分析头部的转动角度。

主要功能:
- 读取融合后的3D关键点数据
- 计算头部的Pitch（俯仰角）、Yaw（偏航角）、Roll（翻滚角）
- 加载头部动作标注
- 比较计算角度与标注label
- 导出结果到CSV文件
- 可视化角度变化

使用示例:
    >>> from head_movement_analysis import HeadPoseAnalyzer
    >>> from head_movement_analysis import load_head_movement_annotations
    >>> from pathlib import Path
    >>> 
    >>> # 加载标注
    >>> annotations = load_head_movement_annotations(
    ...     Path("/workspace/data/annotation/label/full.json")
    ... )
    >>> 
    >>> # 创建分析器
    >>> analyzer = HeadPoseAnalyzer(annotation_dict=annotations)
    >>> 
    >>> # 分析并比较
    >>> results = analyzer.analyze_sequence_with_annotations(
    ...     video_id="01_day_high",
    ...     fused_dir=Path("/workspace/data/head3d_fuse_results/01/昼多い/fused_npz")
    ... )

Author: Kaixu Chen (chenkaixusan@gmail.com)
Date: February 7, 2026
"""

from .load import (
    HeadMovementLabel,
    get_all_annotations_for_frame,
    get_annotation_for_frame,
    load_fused_keypoints,
    load_head_movement_annotations,
    load_multi_annotator_annotations,
)
from .angle_calculator import (
    KEYPOINT_INDICES,
    LABEL_DIRECTION_MAP,
    calculate_head_angles,
    classify_label,
    direction_match,
    extract_head_keypoints,
)
from .main import HeadPoseAnalyzer

__version__ = "1.0.0"
__author__ = "Kaixu Chen"
__email__ = "chenkaixusan@gmail.com"

__all__ = [
    "HeadPoseAnalyzer",
    "KEYPOINT_INDICES",
    "LABEL_DIRECTION_MAP",
    "HeadMovementLabel",
    "load_fused_keypoints",
    "load_head_movement_annotations",
    "load_multi_annotator_annotations",
    "get_annotation_for_frame",
    "calculate_head_angles",
    "classify_label",
    "direction_match",
    "extract_head_keypoints",
    "get_all_annotations_for_frame",
]
