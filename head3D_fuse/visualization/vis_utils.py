#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/head3D_fuse/vis_utils.py
Project: /workspace/code/head3D_fuse
Created Date: Monday February 2nd 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Monday February 2nd 2026 7:08:54 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from omegaconf import DictConfig

from head3D_fuse.visualization.skeleton_visualizer import SkeletonVisualizer

logger = logging.getLogger(__name__)
DUMMY_IMAGE_SIZE = (
    10,
    10,
)  # Placeholder size; visualize_3d_skeleton only uses it as a canvas.


LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

visualizer = SkeletonVisualizer(line_width=2, radius=5)


def visualize_2d_results(
    img_cv2: np.ndarray, outputs: List[Dict[str, Any]], visualizer: SkeletonVisualizer
) -> List[np.ndarray]:
    """Visualize 2D keypoints and bounding boxes"""
    results = []

    for pid, person_output in enumerate(outputs):
        img_vis = img_cv2.copy()

        # Draw keypoints
        # keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d = person_output["filtered_pred_keypoints_2d"]
        keypoints_2d_vis = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )

        # draw skeleton
        for pt in keypoints_2d_vis:
            if pt[2] > 0:
                cv2.circle(
                    img_vis,
                    (int(pt[0]), int(pt[1])),
                    visualizer.radius,
                    (0, 0, 255),  # Red color
                    -1,
                )

        # Draw bounding box
        bbox = person_output["bbox"]
        img_vis = cv2.rectangle(
            img_vis,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 255, 0),  # Green color
            2,
        )

        # Add person ID text
        cv2.putText(
            img_vis,
            f"Person {pid}",
            (int(bbox[0]), int(bbox[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        results.append(img_vis)

    return results


def _save_view_visualizations(
    output: dict,
    save_root: Path,
    view: str,
    frame_idx: int,
    cfg: DictConfig,
    visualizer: SkeletonVisualizer,
) -> None:
    """融合不同视角的三个结果到一张图片上。
    左面：原始frame
    中间：2d kpt结果
    右边：3d kpt结果
    """

    frame = output.get("frame")
    if frame is None:
        logger.warning("Missing frame for view=%s frame=%s", view, frame_idx)
        return

    # Visualization helpers expect a list of outputs, even for a single person.
    outputs_list = [output]

    # 准备三个图像：frame(左), 2d_kpt(中), 3d_kpt(右)
    images_to_combine = []

    # 1. 原始frame (左边)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.shape[2] == 3 else frame
    images_to_combine.append(frame_bgr.copy())

    # 2. 2D keypoints visualization (中间)
    results = visualize_2d_results(frame, outputs_list, visualizer)
    if results and results[0] is not None:
        images_to_combine.append(results[0])
    else:
        logger.warning("2D visualization failed for view=%s frame=%s", view, frame_idx)
        # 使用原始frame作为占位符
        images_to_combine.append(frame_bgr.copy())

    # 3. 3D keypoints visualization (右边)
    kpt3d_img = visualizer.draw_3d_skeleton(
        img_cv2=frame, keypoints_3d=outputs_list[0]["pred_keypoints_3d"]
    )
    if kpt3d_img is None:
        # 使用空白图像作为占位符
        images_to_combine.append(np.zeros_like(frame_bgr))
    else:
        # 如果不需要3D可视化，使用原始frame
        images_to_combine.append(kpt3d_img.copy())

    # 4. 统一高度并水平拼接
    if len(images_to_combine) >= 3:
        # 以第一张图（原始frame）的高度为基准
        target_h = images_to_combine[0].shape[0]

        resized_images = []
        for img in images_to_combine:
            h, w = img.shape[:2]
            # 计算保持宽高比的新宽度
            new_w = int(w * (target_h / h))
            resized = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
            resized_images.append(resized)

        # 水平拼接：frame(左) + 2d_kpt(中) + 3d_kpt(右)
        combined_img = np.hstack(resized_images)

        # 保存组合图像
        save_dir = save_root / view / "combined"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"frame_{frame_idx:06d}_combined.png"
        cv2.imwrite(str(save_path), combined_img)


def _save_fused_visualization(
    save_dir: Path,
    frame_idx: int,
    fused_keypoints: np.ndarray,
) -> np.ndarray:
    save_dir.mkdir(parents=True, exist_ok=True)
    outputs = [{"pred_keypoints_3d": fused_keypoints}]
    dummy_img = np.zeros((*DUMMY_IMAGE_SIZE, 3), dtype=np.uint8)
    kpt3d_img = visualizer.draw_3d_skeleton(
        img_cv2=dummy_img, keypoints_3d=outputs[0]["pred_keypoints_3d"]
    )
    save_path = save_dir / f"frame_{frame_idx:06d}_3d_kpt.png"
    cv2.imwrite(str(save_path), kpt3d_img)
    return dummy_img


def _save_frame_fuse_3dkpt_visualization(
    save_dir: Path,
    frame_idx: int,
    fused_keypoints: np.ndarray,
    outputs: Dict[str, dict],
    visualizer: SkeletonVisualizer,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. 提取并转换颜色 (RGB -> BGR)
    def to_bgr(img):
        # 如果图片颜色看起来发蓝，取消下面这一行的注释
        # return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    view_images = []
    sorted_keys = sorted(outputs.keys())
    for key in sorted_keys[:3]:
        img = outputs[key].get("frame")
        if img is not None:
            view_images.append(to_bgr(img.copy()))

    outputs = [{"pred_keypoints_3d": fused_keypoints}]
    dummy_img = np.zeros((*DUMMY_IMAGE_SIZE, 3), dtype=np.uint8)
    kpt3d_img = visualizer.draw_3d_skeleton(
        img_cv2=dummy_img, keypoints_3d=outputs[0]["pred_keypoints_3d"]
    )

    # 2. 统一左侧尺寸并堆叠
    # 假设以第一张视角图的原始大小为准
    v_h, v_w = view_images[0].shape[:2]
    # bgr to rgb
    view_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in view_images]
    resized_views = [cv2.resize(img, (v_w, v_h)) for img in view_images]
    left_column = np.vstack(resized_views)

    total_h = left_column.shape[0]  # 左侧总高度

    # 3. 重点：确保右侧图不是黑的，并且拉伸到 total_h
    f_h, f_w = kpt3d_img.shape[:2]

    # 计算右侧图为了匹配高度所需的宽度
    new_f_w = int(f_w * (total_h / f_h))

    # 使用 CUBIC 插值放大，确保清晰
    right_column = cv2.resize(
        kpt3d_img, (new_f_w, total_h), interpolation=cv2.INTER_CUBIC
    )

    # 4. 左右拼接
    final_visualization = np.hstack([left_column, right_column])

    # 5. 保存
    save_path = save_dir / f"fused_{frame_idx:06d}.png"
    cv2.imwrite(str(save_path), final_visualization)

    return save_path
