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
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

from head3D_fuse.metadata.mhr70 import pose_info as mhr70_pose_info
from head3D_fuse.visualization.skeleton_visualizer import SkeletonVisualizer

logger = logging.getLogger(__name__)
DUMMY_IMAGE_SIZE = (
    10,
    10,
)  # Placeholder size; visualize_3d_skeleton only uses it as a canvas.
VISUALIZATION_BACKGROUND_COLOR = "white"
# Threshold for near-zero ranges; assumes coordinates are in meter-scale units.
NEAR_ZERO_RANGE_THRESHOLD = 1e-6
DEFAULT_AXIS_RANGE = 1.0
MATPLOTLIB_OUTPUTS_RGB = True
BOX_ASPECT_WARNING_MESSAGE = (
    "Matplotlib does not support set_box_aspect; update matplotlib to enable "
    "aspect locking and avoid distorted 3D skeletons."
)
BOX_ASPECT_WARNING_REGEX = re.compile(rf"^{re.escape(BOX_ASPECT_WARNING_MESSAGE)}$")


LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

visualizer = SkeletonVisualizer(line_width=2, radius=5)
visualizer.set_pose_meta(mhr70_pose_info)


def _get_frame_is_rgb(output: Optional[Dict[str, Any]], default: bool = True) -> bool:
    """Determine whether frame data is in RGB format.

    Args:
        output: Output dictionary that may include frame color metadata.
        default: Fallback when no metadata is available.

    Returns:
        True if the frame should be treated as RGB; otherwise False.
    """
    if not output:
        return default
    frame_is_rgb = output.get("frame_is_rgb")
    if frame_is_rgb is not None:
        return bool(frame_is_rgb)
    frame_color = output.get("frame_color")
    if isinstance(frame_color, str):
        return frame_color.lower() != "bgr"
    return default


def _ensure_bgr(image: Optional[np.ndarray], input_is_rgb: bool) -> Optional[np.ndarray]:
    """Ensure image is in BGR format for OpenCV.

    Args:
        image: Input image array or None.
        input_is_rgb: Whether the input image is RGB.

    Returns:
        The BGR image if input_is_rgb is True, otherwise the original image.
    """
    if image is None:
        return None
    if input_is_rgb:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def visualize_2d_results(
    img_cv2: np.ndarray, outputs: List[Dict[str, Any]], visualizer: SkeletonVisualizer
) -> List[np.ndarray]:
    """Visualize 2D keypoints and bounding boxes"""
    results = []

    for pid, person_output in enumerate(outputs):
        img_vis = img_cv2.copy()

        # Draw keypoints
        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d_vis = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )
        img_vis = visualizer.draw_skeleton(img_vis, keypoints_2d_vis)

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


def visualize_3d_mesh(
    img_cv2: np.ndarray, outputs: List[Dict[str, Any]], faces: np.ndarray
) -> List[np.ndarray]:
    """Visualize 3D mesh overlaid on image and side view"""
    # Local import keeps optional OpenGL dependencies from loading in non-mesh flows.
    from head3D_fuse.visualization.renderer import Renderer

    results = []

    for pid, person_output in enumerate(outputs):
        # Create renderer for this person
        renderer = Renderer(focal_length=person_output["focal_length"], faces=faces)

        # 1. Original image
        img_orig = img_cv2.copy()

        # 2. Mesh overlay on original image
        img_mesh_overlay = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                img_cv2.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        ).astype(np.uint8)

        # 3. Mesh on white background (front view)
        white_img = np.ones_like(img_cv2) * 255
        img_mesh_white = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                white_img,
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        ).astype(np.uint8)

        # 4. Side view
        img_mesh_side = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                white_img.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                side_view=True,
            )
            * 255
        ).astype(np.uint8)

        # Combine all views
        combined = np.concatenate(
            [img_orig, img_mesh_overlay, img_mesh_white, img_mesh_side], axis=1
        )
        results.append(combined)

    return results


def visualize_3d_skeleton(
    img_cv2: np.ndarray,
    outputs: List[Dict[str, Any]],
    visualizer: SkeletonVisualizer,
) -> np.ndarray:
    """
    3D 骨架现场绘制接口。
    """
    # 1. 初始化 Matplotlib 3D 画布
    # 使用 Agg 后端防止在服务器报错（如果在 main 开头设置过则此处不需要）
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    # 设置基础外观
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_facecolor(VISUALIZATION_BACKGROUND_COLOR)
    fig.patch.set_facecolor(VISUALIZATION_BACKGROUND_COLOR)
    ax.grid(False)
    ax.set_axis_off()
    supports_box_aspect = hasattr(ax, "set_box_aspect")

    # 2. Prepare color data (normalize 0-255 to 0-1)
    kpt_colors = (
        np.array(visualizer.kpt_color, dtype=np.float32) / 255.0
        if visualizer.kpt_color is not None
        else None
    )
    link_colors = (
        np.array(visualizer.link_color, dtype=np.float32) / 255.0
        if visualizer.link_color is not None
        else None
    )

    has_data = False

    # Collect all points to unify scale across people.
    all_points = []
    for target in outputs:
        pts = target.get("pred_keypoints_3d")
        if pts is not None:
            all_points.append(pts.reshape(-1, 3))

    if all_points:
        has_data = True
        all_points_np = np.concatenate(all_points, axis=0)

        # Auto-adjust axis scale to avoid distortion.
        max_range = (all_points_np.max(axis=0) - all_points_np.min(axis=0)).max() / 2.0
        if max_range <= NEAR_ZERO_RANGE_THRESHOLD:
            logger.debug(
                "3D skeleton range near zero (%.6f); using default axis range.",
                max_range,
            )
            max_range = DEFAULT_AXIS_RANGE
        mid = (all_points_np.max(axis=0) + all_points_np.min(axis=0)) / 2.0
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        if supports_box_aspect:  # Matplotlib versions that support aspect lock.
            ax.set_box_aspect((1, 1, 1))
        else:
            warnings.warn(
                BOX_ASPECT_WARNING_MESSAGE,
                RuntimeWarning,
                stacklevel=2,
            )
            warnings.filterwarnings(
                "ignore",
                message=BOX_ASPECT_WARNING_REGEX,
                category=RuntimeWarning,
            )

        # 3. Begin rendering
        for i, target in enumerate(outputs):
            pts_3d = target.get("pred_keypoints_3d")
            if pts_3d is None:
                continue
            if pts_3d.ndim == 3:
                pts_3d = pts_3d[0]  # Handle (1, N, 3)

            # Draw keypoints.
            ax.scatter(
                pts_3d[:, 0],
                pts_3d[:, 1],
                pts_3d[:, 2],
                c=kpt_colors if kpt_colors is not None else "r",
                s=visualizer.radius * 5,
                alpha=getattr(visualizer, "alpha", 0.8),
            )

            # Draw skeleton links.
            if visualizer.skeleton is not None:
                for j, (p1_idx, p2_idx) in enumerate(visualizer.skeleton):
                    if p1_idx < len(pts_3d) and p2_idx < len(pts_3d):
                        p1, p2 = pts_3d[p1_idx], pts_3d[p2_idx]
                        color = (
                            link_colors[j % len(link_colors)]
                            if link_colors is not None
                            else "b"
                        )
                        ax.plot(
                            [p1[0], p2[0]],
                            [p1[1], p2[1]],
                            [p1[2], p2[2]],
                            color=color,
                            linewidth=visualizer.line_width,
                            alpha=getattr(visualizer, "alpha", 0.8),
                        )

    if not has_data:
        ax.text(0.5, 0.5, 0.5, "No Data", ha="center")

    # Set initial view angle (top-down).
    ax.view_init(elev=-30, azim=270)

    # 4. Convert to image array.
    fig.tight_layout(pad=0.1)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img_3d = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img_3d = img_3d.reshape((h, w, 4))[:, :, :3]  # 去掉 alpha 通道

    # 4. Close fig to release memory.
    plt.close(fig)
    return img_3d


def _save_view_visualizations(
    output: dict,
    save_root: Path,
    view: str,
    frame_idx: int,
    cfg: DictConfig,
    visualizer,
) -> None:
    logger = logging.getLogger(__name__)
    frame = output.get("frame")
    if frame is None:
        logger.warning("Missing frame for view=%s frame=%s", view, frame_idx)
        return
    frame_is_rgb = _get_frame_is_rgb(output)
    frame_bgr = _ensure_bgr(frame, frame_is_rgb)
    if frame_bgr is None:
        logger.warning("Missing frame data for view=%s frame=%s", view, frame_idx)
        return

    # Visualization helpers expect a list of outputs, even for a single person.
    outputs_list = [output]
    plot_2d = cfg.visualize.get("plot_2d", False)
    if plot_2d:
        save_dir = save_root / view / "2d"
        save_dir.mkdir(parents=True, exist_ok=True)
        # visualize_2d_results returns a list; the first entry corresponds to the single output.
        results = visualize_2d_results(frame_bgr, outputs_list, visualizer)
        if not results or results[0] is None:
            logger.warning(
                "2D visualization failed for view=%s frame=%s", view, frame_idx
            )
        else:
            cv2.imwrite(str(save_dir / f"frame_{frame_idx:06d}_2d.png"), results[0])

    if cfg.visualize.get("save_3d_keypoints", False):
        save_dir = save_root / view
        save_dir.mkdir(parents=True, exist_ok=True)
        kpt3d_img = visualize_3d_skeleton(
            img_cv2=frame, outputs=outputs_list, visualizer=visualizer
        )
        if kpt3d_img is None:
            logger.warning(
                "3D keypoint visualization failed for view=%s frame=%s",
                view,
                frame_idx,
            )
        else:
            kpt3d_bgr = _ensure_bgr(kpt3d_img, MATPLOTLIB_OUTPUTS_RGB)
            cv2.imwrite(
                str(save_dir / f"frame_{frame_idx:06d}_3d_kpt.png"), kpt3d_bgr
            )


def _save_fused_visualization(
    save_dir: Path,
    frame_idx: int,
    fused_keypoints: np.ndarray,
) -> np.ndarray:
    save_dir.mkdir(parents=True, exist_ok=True)
    outputs = [{"pred_keypoints_3d": fused_keypoints}]
    dummy_img = np.zeros((*DUMMY_IMAGE_SIZE, 3), dtype=np.uint8)
    kpt3d_img = visualize_3d_skeleton(
        img_cv2=dummy_img, outputs=outputs, visualizer=visualizer
    )
    save_path = save_dir / f"frame_{frame_idx:06d}_3d_kpt.png"
    kpt3d_bgr = _ensure_bgr(kpt3d_img, MATPLOTLIB_OUTPUTS_RGB)
    cv2.imwrite(str(save_path), kpt3d_bgr)
    return dummy_img


def _save_frame_fuse_3dkpt_visualization(
    save_dir: Path,
    frame_idx: int,
    fused_keypoints: np.ndarray,
    outputs: Dict[str, dict],
    visualizer: SkeletonVisualizer,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)

    view_images = []
    sorted_keys = sorted(outputs.keys())
    for key in sorted_keys[:3]:
        output = outputs[key]
        img = output.get("frame")
        if img is not None:
            frame_is_rgb = _get_frame_is_rgb(output)
            converted = _ensure_bgr(img, frame_is_rgb)
            if converted is not None:
                view_images.append(converted)
        else:
            logger.warning("Missing frame for view=%s frame=%s", key, frame_idx)

    outputs = [{"pred_keypoints_3d": fused_keypoints}]
    dummy_img = np.zeros((*DUMMY_IMAGE_SIZE, 3), dtype=np.uint8)
    kpt3d_img = visualize_3d_skeleton(
        img_cv2=dummy_img, outputs=outputs, visualizer=visualizer
    )
    kpt3d_bgr = _ensure_bgr(kpt3d_img, MATPLOTLIB_OUTPUTS_RGB)

    if not view_images:
        logger.warning(
            "No view images available for frame=%s; saving fused 3D keypoints only.",
            frame_idx,
        )
        save_path = save_dir / f"fused_{frame_idx:06d}_3d_only.png"
        cv2.imwrite(str(save_path), kpt3d_bgr)
        return save_path

    # 2. Unify left column size and stack.
    # Assume the first view image's original size as reference.
    v_h, v_w = view_images[0].shape[:2]
    resized_views = [cv2.resize(img, (v_w, v_h)) for img in view_images]
    left_column = np.vstack(resized_views)

    total_h = left_column.shape[0]  # Left column total height.

    # 3. Ensure right-side image is not black and stretch to total_h.
    f_h, f_w = kpt3d_bgr.shape[:2]

    # Compute right-side width to match height.
    new_f_w = int(f_w * (total_h / f_h))

    # Resize with cubic interpolation for clarity.
    right_column = cv2.resize(
        kpt3d_bgr, (new_f_w, total_h), interpolation=cv2.INTER_CUBIC
    )

    # 4. Concatenate left and right columns.
    final_visualization = np.hstack([left_column, right_column])

    # 5. Save.
    save_path = save_dir / f"fused_{frame_idx:06d}.png"
    cv2.imwrite(str(save_path), final_visualization)

    return save_path
