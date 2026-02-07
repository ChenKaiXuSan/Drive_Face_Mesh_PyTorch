#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/code/head_movement_analysis/angle_calculator.py
Project: /workspace/code/head_movement_analysis
Created Date: Saturday February 7th 2026
Author: Kaixu Chen
-----
Comment: 头部转角计算模块
Have a good code time :)
-----
Copyright (c) 2026 The University of Tsukuba
-----
'''
import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# MHR70关键点索引定义
KEYPOINT_INDICES = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "neck": 69,
}

# 标注方向到角度方向的映射
LABEL_DIRECTION_MAP = {
    "up": (1, 0),
    "down": (-1, 0),
    "left": (0, -1),
    "right": (0, 1),
    "left_up": (1, -1),
    "left_down": (-1, -1),
    "right_up": (1, 1),
    "right_down": (-1, 1),
    "front": (0, 0),
}


def extract_head_keypoints(
    keypoints_3d: np.ndarray, keypoint_indices: Dict[str, int]
) -> Optional[Dict[str, np.ndarray]]:
    """
    从3D关键点数组中提取头部相关的关键点

    Args:
        keypoints_3d: 形状为 (70, 3) 的3D关键点数组
        keypoint_indices: 关键点名称到索引的映射字典

    Returns:
        包含头部关键点的字典，如果关键点无效则返回None
    """
    head_kpts = {}
    required_kpts = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
    ]

    for name in required_kpts:
        idx = keypoint_indices[name]
        kpt = keypoints_3d[idx]

        # 检查关键点是否有效
        if not np.isfinite(kpt).all():
            logger.warning(f"Invalid keypoint: {name} (index {idx})")
            return None

        head_kpts[name] = kpt

    return head_kpts


def calculate_head_angles(
    head_kpts: Dict[str, np.ndarray],
) -> Tuple[float, float, float]:
    """
    计算头部的三个转动角度

    Args:
        head_kpts: 包含头部关键点的字典

    Returns:
        (pitch, yaw, roll) 三个角度，单位为度
        - pitch: 俯仰角（上下点头），正值表示抬头，负值表示低头
        - yaw: 偏航角（左右转头），正值表示向右转，负值表示向左转
        - roll: 翻滚角（左右倾斜），正值表示向右倾斜，负值表示向左倾斜
    """
    nose = head_kpts["nose"]
    left_eye = head_kpts["left_eye"]
    right_eye = head_kpts["right_eye"]
    left_ear = head_kpts["left_ear"]
    right_ear = head_kpts["right_ear"]
    left_shoulder = head_kpts["left_shoulder"]
    right_shoulder = head_kpts["right_shoulder"]

    # 计算眼睛中心点、耳朵中心点和肩膀中心点
    eye_center = (left_eye + right_eye) / 2
    ear_center = (left_ear + right_ear) / 2
    shoulder_center = (left_shoulder + right_shoulder) / 2

    # ===== 1. 计算Pitch（俯仰角）=====
    # 使用鼻子和肩膀中心的向量在垂直平面上的投影
    nose_shoulder_vec = nose - shoulder_center
    # 计算与水平面的夹角（假设Y轴向上）
    horizontal_dist = np.sqrt(nose_shoulder_vec[0] ** 2 + nose_shoulder_vec[2] ** 2)
    pitch = np.arctan2(nose_shoulder_vec[1], horizontal_dist)
    pitch_deg = np.degrees(pitch)

    # ===== 2. 计算Yaw（偏航角）=====
    # 使用眼睛中心到肩膀中心的向量作为头部朝向的参考
    # 这种方法比单纯用眼睛更稳定，利用整个头-肩膀系统的对齐
    face_forward = nose - eye_center
    shoulder_ref = shoulder_center - eye_center  # 从眼睛到肩膀的参考向量

    # 在水平面上计算头部朝向
    face_forward_horizontal = np.array([face_forward[0], 0, face_forward[2]])

    # 计算与前方（-Z轴）的夹角
    if np.linalg.norm(face_forward_horizontal) > 1e-6:
        face_forward_horizontal = (
            face_forward_horizontal / np.linalg.norm(face_forward_horizontal)
        )
        # 假设相机坐标系中-Z是前方，X是右侧
        yaw = np.arctan2(face_forward_horizontal[0], -face_forward_horizontal[2])
        yaw_deg = np.degrees(yaw)
    else:
        yaw_deg = 0.0

    # ===== 3. 计算Roll（翻滚角）=====
    # 使用左右眼睛连线与水平面的夹角
    eye_vec = right_eye - left_eye
    # 在X-Y平面上计算倾斜角度
    roll = np.arctan2(eye_vec[1], eye_vec[0])
    roll_deg = np.degrees(roll)

    return pitch_deg, yaw_deg, roll_deg


def direction_match(angle_value: float, expected_dir: int, threshold: float) -> bool:
    """
    检查角度值是否匹配预期方向

    Args:
        angle_value: 计算出的角度值（度）
        expected_dir: 期望方向 (1=正向, -1=负向, 0=中立)
        threshold: 阈值（度）

    Returns:
        bool: 是否匹配
    """
    if expected_dir > 0:
        return angle_value > threshold
    if expected_dir < 0:
        return angle_value < -threshold
    return abs(angle_value) <= threshold


def classify_label(pitch: float, yaw: float, threshold: float) -> str:
    """
    根据Pitch和Yaw角度分类标注标签

    Args:
        pitch: 俯仰角（度）
        yaw: 偏航角（度）
        threshold: 分类阈值（度）

    Returns:
        str: 分类后的标签名称
    """
    pitch_dir = 0
    yaw_dir = 0

    if pitch > threshold:
        pitch_dir = 1
    elif pitch < -threshold:
        pitch_dir = -1

    if yaw > threshold:
        yaw_dir = 1
    elif yaw < -threshold:
        yaw_dir = -1

    if pitch_dir == 0 and yaw_dir == 0:
        return "front"

    for label, (expected_pitch, expected_yaw) in LABEL_DIRECTION_MAP.items():
        if expected_pitch == pitch_dir and expected_yaw == yaw_dir:
            return label

    return "front"
