#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/head3D_fuse/fuse.py
Project: /workspace/code/head3D_fuse
Created Date: Monday February 2nd 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Monday February 2nd 2026 6:47:37 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from head3D_fuse.load import (
    assemble_view_npz_paths,
    compare_npz_files,
    get_annotation_dict,
    load_npz_output,
)
from head3D_fuse.mesh_3d_eval import (
    evaluate_face3d_pro,
    export_report,
)

# save
from head3D_fuse.save import _save_fused_keypoints
from head3D_fuse.visualization.merge_video import merge_frames_to_video

# vis
from head3D_fuse.visualization.vis_utils import (
    _save_fused_visualization,
    _save_view_visualizations,
    visualize_3d_skeleton,
    visualizer,
)

from head3D_fuse.fuse import fuse_3view_keypoints

logger = logging.getLogger(__name__)
MIN_POINTS_FOR_ALIGNMENT = 3
VALID_ALIGNMENT_METHODS = ("none", "procrustes", "procrustes_trimmed")


def _normalize_keypoints(keypoints: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if keypoints is None:
        return None
    keypoints = np.asarray(keypoints)
    if keypoints.ndim == 3 and keypoints.shape[0] >= 1:
        return keypoints[0]
    return keypoints


# ---------------------------------------------------------------------
# 核心处理逻辑：处理单个人的数据
# ---------------------------------------------------------------------
def process_single_person_env(
    person_env_dir: Path,
    out_root: Path,
    infer_root: Path,
    cfg: DictConfig,
):
    """处理单个人员的所有环境和视角"""

    person_id = person_env_dir.parent.name
    env_name = person_env_dir.name
    view_list = cfg.infer.get("view_list")

    if view_list is None:
        view_list = ["front", "left", "right"]

    annotation_dict = get_annotation_dict(cfg.paths.start_mid_end_path)

    logger.info(f"==== Starting Process for Person: {person_id}, Env: {env_name} ====")

    frame_triplets, report = assemble_view_npz_paths(
        person_env_dir, view_list, annotation_dict
    )
    if not frame_triplets:
        logger.warning(f"No aligned frames found for {person_id}/{env_name}")
        return

    diff_reports = []
    fused_method = cfg.infer.get("fuse_method", "median")

    for triplet in tqdm(frame_triplets, desc=f"Fusing {person_id}/{env_name}"):
        diff = compare_npz_files(triplet.npz_paths)
        if diff:
            diff_reports.append(diff)

        outputs = {
            view: load_npz_output(npz_path)
            for view, npz_path in triplet.npz_paths.items()
        }

        keypoints_by_view = {
            view: _normalize_keypoints(outputs[view].get("pred_keypoints_3d"))
            for view in view_list
        }
        missing_views = [view for view, kpt in keypoints_by_view.items() if kpt is None]
        if missing_views:
            logger.warning(
                "Missing pred_keypoints_3d for frame %s in %s/%s (views: %s)",
                triplet.frame_idx,
                person_id,
                env_name,
                ",".join(missing_views),
            )
            continue

        view_transforms = cfg.infer.get("view_transforms")
        transform_mode = cfg.infer.get("transform_mode", "world_to_camera")
        alignment_method = cfg.infer.get("alignment_method", "none")
        alignment_reference = cfg.infer.get("alignment_reference")
        alignment_scale = cfg.infer.get("alignment_scale", True)
        alignment_trim_ratio = cfg.infer.get("alignment_trim_ratio", 0.2)
        alignment_max_iters = cfg.infer.get("alignment_max_iters", 3)
        fused_kpt, fused_mask, n_valid = fuse_3view_keypoints(
            keypoints_by_view,
            method=fused_method,
            view_transforms=view_transforms,
            transform_mode=transform_mode,
            alignment_method=alignment_method,
            alignment_reference=alignment_reference,
            alignment_scale=alignment_scale,
            alignment_trim_ratio=alignment_trim_ratio,
            alignment_max_iters=alignment_max_iters,
        )

        # 保存融合后的关键点
        save_dir = infer_root / person_id / env_name / "fused_npz"
        _save_fused_keypoints(
            save_dir=save_dir,
            frame_idx=triplet.frame_idx,
            fused_keypoints=fused_kpt,
            fused_mask=fused_mask,
            n_valid=n_valid,
            npz_paths=triplet.npz_paths,
        )

        # 可视化保存各个视角结果
        for view in view_list:
            if view not in outputs:
                logger.warning(
                    f"Missing output for view={view} frame={triplet.frame_idx}"
                )
                continue
            vis_root = out_root / person_id / env_name / "different_vis"
            _save_view_visualizations(
                output=outputs[view],
                save_root=vis_root,
                view=view,
                frame_idx=triplet.frame_idx,
                cfg=cfg,
                visualizer=visualizer,
            )

        if cfg.visualize.get("save_3d_keypoints", False):
            _save_fused_visualization(
                save_dir=out_root / person_id / env_name / "vis" / "fused_3d_keypoints",
                frame_idx=triplet.frame_idx,
                fused_keypoints=fused_kpt,
            )

    if diff_reports:
        diff_path = out_root / person_id / env_name / "npz_diff_report.json"
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        with diff_path.open("w", encoding="utf-8") as f:
            json.dump(diff_reports, f, ensure_ascii=False, indent=2)
        logger.info("Saved npz diff report to %s", diff_path)
    else:
        logger.info("No npz differences found for %s/%s", person_id, env_name)

    # 融合frame到video
    merge_frames_to_video(
        frame_dir=str(out_root / person_id / env_name / "vis" / "fused_3d_keypoints"),
        output_video_path=str(
            out_root / person_id / env_name / "merged_video" / "fused_3d_keypoints.mp4"
        ),
        fps=30,
    )
    # different view
    merge_frames_to_video(
        frame_dir=str(out_root / person_id / env_name / "different_vis" / "front"),
        output_video_path=str(
            out_root / person_id / env_name / "merged_video" / "front.mp4"
        ),
        fps=30,
    )
    merge_frames_to_video(
        frame_dir=str(out_root / person_id / env_name / "different_vis" / "left"),
        output_video_path=str(
            out_root / person_id / env_name / "merged_video" / "left.mp4"
        ),
        fps=30,
    )
    merge_frames_to_video(
        frame_dir=str(out_root / person_id / env_name / "different_vis" / "right"),
        output_video_path=str(
            out_root / person_id / env_name / "merged_video" / "right.mp4"
        ),
        fps=30,
    )
