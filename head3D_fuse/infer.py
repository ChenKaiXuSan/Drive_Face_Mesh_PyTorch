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
from typing import Optional, cast

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from head3D_fuse.fuse import fuse_3view_keypoints
from head3D_fuse.load import (
    assemble_view_npz_paths,
    compare_npz_files,
    get_annotation_dict,
    load_npz_output,
)

# save
from head3D_fuse.save import _save_fused_keypoints
from head3D_fuse.visualization.merge_video import merge_frames_to_video

# vis
from head3D_fuse.visualization.vis_utils import (
    _save_frame_fuse_3dkpt_visualization,
    _save_fused_visualization,
    _save_view_visualizations,
    visualizer,
)

logger = logging.getLogger(__name__)
MIN_POINTS_FOR_ALIGNMENT = 3
VALID_ALIGNMENT_METHODS = ("none", "procrustes", "procrustes_trimmed")

# 定义需要保留的关键点索引：头部 + 肩部/颈部 + 双手
KEEP_KEYPOINT_INDICES = (
    # 头部: 鼻子、眼睛、耳朵
    list(range(0, 5))  # 0-4: nose, left-eye, right-eye, left-ear, right-ear
    # 肩部和颈部
    + [5, 6]  # left-shoulder, right-shoulder
    # 双手（包括手腕）
    + list(range(21, 63))  # 21-62: 右手(21-41) + 左手(42-62)
    # 肩峰和颈部
    + [67, 68, 69]  # left-acromion, right-acromion, neck
)


def _normalize_keypoints(keypoints: Optional[np.ndarray]) -> np.ndarray:
    """归一化关键点并过滤只保留头部、肩部和双手的关键点。

    Args:
        keypoints: 输入的关键点数组，形状可能是 (batch, N, 3) 或 (N, 3)

    Returns:
        过滤后的关键点数组，形状为 (M, 3)，其中M是保留的关键点数量
        如果输入为None，返回填充NaN的数组
    """
    num_keep_points = len(KEEP_KEYPOINT_INDICES)

    if keypoints is None:
        # 当关键点缺失时，创建填充NaN的数组
        return np.full((num_keep_points, 3), np.nan, dtype=np.float32)

    # 明确类型以避免类型检查错误
    kpt_array = cast(np.ndarray, np.asarray(keypoints))
    assert kpt_array is not None  # 帮助类型检查器

    # 处理batch维度
    if kpt_array.ndim == 3 and kpt_array.shape[0] >= 1:
        kpt_array = kpt_array[0]

    # 过滤关键点，只保留头部、肩部和双手
    if kpt_array.shape[0] > max(KEEP_KEYPOINT_INDICES):
        filtered_keypoints = kpt_array[KEEP_KEYPOINT_INDICES]
    else:
        # 如果关键点数量不足，填充NaN
        logger.warning(
            "Keypoints shape %s is smaller than expected, padding with NaN",
            kpt_array.shape,
        )
        filtered_keypoints = np.full((num_keep_points, 3), np.nan, dtype=np.float32)
        # 复制可用的关键点
        available_indices = [i for i in KEEP_KEYPOINT_INDICES if i < kpt_array.shape[0]]
        for new_idx, old_idx in enumerate(available_indices):
            if new_idx < num_keep_points:
                filtered_keypoints[new_idx] = kpt_array[old_idx]

    return filtered_keypoints


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

    for i, triplet in enumerate(
        tqdm(frame_triplets, desc=f"Fusing {person_id}/{env_name}")
    ):
        # if i == 60:
        #     break  # for debug

        diff = compare_npz_files(triplet.npz_paths)
        if diff:
            diff_reports.append(diff)

        outputs = {
            view: load_npz_output(npz_path)
            for view, npz_path in triplet.npz_paths.items()
        }

        keypoints_by_view = {}
        for view in view_list:
            # 这里虽然准备了过滤之后的结果，但是不好用
            filtered_view_3dkpt = _normalize_keypoints(
                outputs[view].get("pred_keypoints_3d")
            )
            filtered_view_2dkpt = _normalize_keypoints(
                outputs[view].get("pred_keypoints_2d")
            )
            # keypoints_by_view[view] = filtered_view_3dkpt
            keypoints_by_view[view] = outputs[view]["pred_keypoints_3d"]
            outputs[view]["filtered_pred_keypoints_3d"] = filtered_view_3dkpt
            outputs[view]["filtered_pred_keypoints_2d"] = filtered_view_2dkpt

        # 检查是否有包含NaN的视角（表示关键点缺失或不完整）
        missing_views = [
            view for view, kpt in keypoints_by_view.items() if np.all(np.isnan(kpt))
        ]
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

        # 保存三个视角的frame和融合结果的可视画
        _save_frame_fuse_3dkpt_visualization(
            save_dir=out_root / person_id / env_name / "vis" / "vis_together",
            frame_idx=triplet.frame_idx,
            fused_keypoints=fused_kpt,
            outputs=outputs,
            visualizer=visualizer,
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
        frame_dir=out_root / person_id / env_name / "vis" / "vis_together",
        output_video_path=out_root
        / person_id
        / env_name
        / "merged_video"
        / "fused_3d_keypoints.mp4",
        fps=30,
    )
    # different view
    merge_frames_to_video(
        frame_dir=out_root / person_id / env_name / "different_vis" / "front",
        output_video_path=out_root
        / person_id
        / env_name
        / "merged_video"
        / "front.mp4",
        fps=30,
    )
    merge_frames_to_video(
        frame_dir=out_root / person_id / env_name / "different_vis" / "left",
        output_video_path=out_root / person_id / env_name / "merged_video" / "left.mp4",
        fps=30,
    )
    merge_frames_to_video(
        frame_dir=out_root / person_id / env_name / "different_vis" / "right",
        output_video_path=out_root
        / person_id
        / env_name
        / "merged_video"
        / "right.mp4",
        fps=30,
    )
