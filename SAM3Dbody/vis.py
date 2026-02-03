#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Visualization utilities entry for SAM3Dbody.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .sam_3d_body.visualization.renderer import Renderer
from .sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
from .tools.vis_utils import (
    visualize_sample_together,
    visualize_2d_results,
    visualize_3d_mesh,
    visualize_3d_skeleton,
)

logger = logging.getLogger(__name__)

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def vis_results(
    img_cv2: np.ndarray,
    outputs: List[Dict[str, Any]],
    faces: np.ndarray,
    save_dir: str,
    image_name: str,
    visualizer: SkeletonVisualizer,
    cfg: Optional[Dict[str, Any]] = None,
):
    """Save visualization results to disk according to cfg flags."""

    os.makedirs(save_dir, exist_ok=True)
    cfg = cfg or {}

    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

    # Save focal length
    if outputs and cfg.get("save_focal_length", False):
        focal_length_data = {"focal_length": float(outputs[0]["focal_length"])}
        focal_length_path = os.path.join(save_dir, f"{image_name}_focal_length.json")
        with open(focal_length_path, "w") as f:
            json.dump(focal_length_data, f, indent=2)
        logger.info("Saved focal length: %s", focal_length_path)

    for pid, person_output in enumerate(outputs):
        renderer = Renderer(focal_length=person_output["focal_length"], faces=faces)

        if cfg.get("save_mesh_ply", False):
            tmesh = renderer.vertices_to_trimesh(
                person_output["pred_vertices"], person_output["pred_cam_t"], LIGHT_BLUE
            )
            mesh_filename = f"{image_name}_mesh_{pid:03d}.ply"
            mesh_path = os.path.join(save_dir, mesh_filename)
            tmesh.export(mesh_path)
            logger.info("Saved mesh ply file: %s", mesh_path)

        if cfg.get("save_mesh_overlay", False):
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

            overlay_filename = f"{image_name}_overlay_{pid:03d}.png"
            cv2.imwrite(os.path.join(save_dir, overlay_filename), img_mesh_overlay)
            logger.info("Saved overlay: %s", os.path.join(save_dir, overlay_filename))

        if cfg.get("save_bbox_image", False):
            img_bbox = img_cv2.copy()
            bbox = person_output["bbox"]
            img_bbox = cv2.rectangle(
                img_bbox,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (0, 255, 0),
                4,
            )
            bbox_filename = f"{image_name}_bbox_{pid:03d}.png"
            cv2.imwrite(os.path.join(save_dir, bbox_filename), img_bbox)
            logger.info("Saved bbox: %s", os.path.join(save_dir, bbox_filename))

        if cfg.get("plot_2d", False):
            vis_results_2d = visualize_2d_results(img_cv2, outputs, visualizer)
            cv2.imwrite(
                os.path.join(save_dir, f"{image_name}_2d_visualization.png"),
                vis_results_2d[pid],
            )
            logger.info(
                "Saved 2D visualization: %s",
                os.path.join(save_dir, f"{image_name}_2d_visualization.png"),
            )

        if cfg.get("save_3d_mesh", False):
            mesh_results = visualize_3d_mesh(img_cv2, outputs, faces)
            cv2.imwrite(
                os.path.join(save_dir, f"{image_name}_3d_mesh_visualization_{pid}.png"),
                mesh_results[pid],
            )
            logger.info(
                "Saved 3D mesh visualization: %s",
                os.path.join(
                    save_dir, f"{image_name}_3d_mesh_visualization_{pid}.png"
                ),
            )

        if cfg.get("save_3d_keypoints", False):
            kpt3d_img = visualize_3d_skeleton(
                img_cv2=img_cv2.copy(), outputs=outputs, visualizer=visualizer
            )
            cv2.imwrite(
                os.path.join(save_dir, f"{image_name}_3d_kpt_visualization_{pid}.png"),
                kpt3d_img,
            )
            logger.info(
                "Saved 3D keypoint visualization: %s",
                os.path.join(
                    save_dir, f"{image_name}_3d_kpt_visualization_{pid}.png"
                ),
            )

        if cfg.get("save_together", False):
            together_img = visualize_sample_together(
                img_cv2=img_cv2,
                outputs=outputs,
                faces=faces,
                visualizer=visualizer,
            )
            cv2.imwrite(
                os.path.join(save_dir, f"{image_name}_together_visualization.png"),
                together_img,
            )
            logger.info(
                "Saved together visualization: %s",
                os.path.join(save_dir, f"{image_name}_together_visualization.png"),
            )

