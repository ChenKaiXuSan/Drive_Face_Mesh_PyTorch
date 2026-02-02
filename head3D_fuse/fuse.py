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
from pathlib import Path
from omegaconf import DictConfig
from typing import Dict, List, Optional, Tuple
import json
import logging
import numpy as np
import cv2
from head3D_fuse.load import (
    assemble_view_npz_paths,
    compare_npz_files,
    get_annotation_dict,
    load_npz_output,
)

from head3D_fuse.mesh_3d_eval import (
    evaluate_face3d_pro,
    export_report,
    visualize_metrics,
)

# vis
from head3D_fuse.visualization.vis_utils import (
    visualize_2d_results,
    visualize_sample_together,
    visualize_3d_skeleton,
    visualizer,
)

# save
from mesh_triangulation.save import save_3d_joints

logger = logging.getLogger(__name__)
DUMMY_IMAGE_SIZE = (10, 10)  # Placeholder size; visualize_3d_skeleton only uses it as a canvas.


def _normalize_keypoints(keypoints: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if keypoints is None:
        return None
    keypoints = np.asarray(keypoints)
    if keypoints.ndim == 3 and keypoints.shape[0] >= 1:
        return keypoints[0]
    return keypoints


def fuse_3view_keypoints(
    keypoints_by_view: Dict[str, np.ndarray],
    method: str = "median",
    zero_eps: float = 1e-6,
    fill_value: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse three-view 3D keypoints by ignoring invalid (near-zero) joints."""
    if fill_value is None:
        fill_value = np.nan
    view_list = list(keypoints_by_view.keys())
    stacked = np.stack(
        [keypoints_by_view[view] for view in view_list], axis=0
    ).astype(np.float64)
    finite = np.isfinite(stacked).all(axis=-1)
    nonzero = np.linalg.norm(stacked, axis=-1) >= zero_eps
    valid = finite & nonzero

    fused_mask = valid.any(axis=0)
    n_valid = valid.sum(axis=0).astype(np.int64)
    fused = np.full(stacked.shape[1:], fill_value, dtype=np.float64)

    if method == "first":
        for joint_idx in range(stacked.shape[1]):
            for view_idx in range(stacked.shape[0]):
                if valid[view_idx, joint_idx]:
                    fused[joint_idx] = stacked[view_idx, joint_idx]
                    break
    elif method in ("mean", "median"):
        reducer = np.nanmean if method == "mean" else np.nanmedian
        stacked_slice = stacked[:, fused_mask, :]
        valid_slice = valid[:, fused_mask]
        stacked_slice[~valid_slice] = np.nan
        fused[fused_mask] = reducer(stacked_slice, axis=0)
    else:
        raise ValueError(
            "method must be one of: 'mean', 'median', 'first' "
            f"(default from cfg.infer.fuse_method), got '{method}'"
        )

    return fused, fused_mask, n_valid


def _save_fused_keypoints(
    save_dir: Path,
    frame_idx: int,
    fused_keypoints: np.ndarray,
    fused_mask: np.ndarray,
    n_valid: np.ndarray,
    npz_paths: Dict[str, Path],
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"frame_{frame_idx:06d}_fused.npy"
    payload = {
        "frame_idx": frame_idx,
        "fused_keypoints_3d": fused_keypoints,
        "fused_mask": fused_mask,
        "valid_views": n_valid,
        "npz_paths": {view: str(path) for view, path in npz_paths.items()},
    }
    np.save(save_path, payload)
    return save_path


def _save_fused_visualization(
    save_dir: Path,
    frame_idx: int,
    fused_keypoints: np.ndarray,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    outputs = [{"pred_keypoints_3d": fused_keypoints}]
    dummy_img = np.zeros((*DUMMY_IMAGE_SIZE, 3), dtype=np.uint8)
    kpt3d_img = visualize_3d_skeleton(
        img_cv2=dummy_img, outputs=outputs, visualizer=visualizer
    )
    save_path = save_dir / f"frame_{frame_idx:06d}_3d_kpt.png"
    cv2.imwrite(str(save_path), kpt3d_img)
    return save_path
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
    view_list = cfg.infer.get("view_list", cfg.infer.get("views_list", ["front", "left", "right"]))
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
    for triplet in frame_triplets:
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

        fused_kpt, fused_mask, n_valid = fuse_3view_keypoints(
            keypoints_by_view,
            method=fused_method,
        )

        save_dir = out_root / env_name / "fused"
        _save_fused_keypoints(
            save_dir=save_dir,
            frame_idx=triplet.frame_idx,
            fused_keypoints=fused_kpt,
            fused_mask=fused_mask,
            n_valid=n_valid,
            npz_paths=triplet.npz_paths,
        )

        if cfg.visualize.get("save_3d_keypoints", False):
            _save_fused_visualization(
                save_dir=save_dir / "vis",
                frame_idx=triplet.frame_idx,
                fused_keypoints=fused_kpt,
            )

    if diff_reports:
        diff_path = out_root / env_name / "npz_diff_report.json"
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        with diff_path.open("w", encoding="utf-8") as f:
            json.dump(diff_reports, f, ensure_ascii=False, indent=2)
        logger.info("Saved npz diff report to %s", diff_path)
    else:
        logger.info("No npz differences found for %s/%s", person_id, env_name)


def detail_to_txt(detail: dict, save_path: Path):
    """
    把 triangulate_with_missing 的 detail 结果整理成可写入 txt 的格式
    detail: triangulate_with_missing 返回的第三个字典
    save_path: 输出 txt 路径
    """

    save_path.parent.mkdir(parents=True, exist_ok=True)

    if detail is None:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("No triangulation detail available.\n")
        return

    lines = []
    views = detail["views_order"]
    used_mask = detail["used_mask"]
    used_weights = detail["used_weights"]
    used_pts2d = detail["used_pts2d"]
    per_point = detail["per_point"]

    lines.append(f"# Triangulation detail report")
    lines.append(f"# Total points: {len(per_point)}")
    lines.append(f"# Views: {', '.join(views)}")
    lines.append("-" * 80)

    for j, info in enumerate(per_point):
        if info is None:
            lines.append(f"[{j:04d}] Status: <None>\n")
            continue

        lines.append(f"[{j:04d}] Status: {info.get('status', 'unknown')}")
        lines.append(f"  Views used: {', '.join(info.get('views', []))}")
        lines.append(f"  Num views: {info.get('num_views', 0)}")

        if "reproj_err" in info:
            lines.append(
                f"  Reproj error: {info['reproj_err']:.3f}px (thr={info.get('thr', 'NA')})"
            )
        if "max_sampson" in info:
            lines.append(
                f"  Max Sampson: {info['max_sampson']:.3f} (thr={info.get('thr', 'NA')})"
            )

        # --- 每个视角的坐标与权重 ---
        lines.append(f"  2D points:")
        for v_idx, v in enumerate(views):
            if used_mask[j, v_idx]:
                x, y = used_pts2d[j, v_idx]
                w = used_weights[j, v_idx]
                lines.append(f"    - {v}: ({x:.2f}, {y:.2f}), weight={w:.2f}")
        lines.append("-" * 80)

    txt = "\n".join(lines)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"[INFO] Saved triangulation detail to: {save_path}")


# ---------- 主处理函数 ----------
def process_one_frame_list(
    f_frame,
    f_mesh,
    l_frame,
    l_mesh,
    r_frame,
    r_mesh,
    rt_info,
    K,
    output_path: Path,
    i: int,
    vis,
):
    # * 确保三视点图像尺寸一致
    if (
        f_frame.shape[0:2] != l_frame.shape[0:2]
        or f_frame.shape[0:2] != r_frame.shape[0:2]
        or l_frame.shape[0:2] != r_frame.shape[0:2]
    ):

        # 统一目标尺寸（取最大值）
        target_h = max(f_frame.shape[0], l_frame.shape[0], r_frame.shape[0])
        target_w = max(f_frame.shape[1], l_frame.shape[1], r_frame.shape[1])

        f_frame_resized, f_mesh_unnormalized = resize_frame_and_mesh(
            f_frame, f_mesh, (target_w, target_h)
        )
        l_frame_resized, l_mesh_unnormalized = resize_frame_and_mesh(
            l_frame, l_mesh, (target_w, target_h)
        )
        r_frame_resized, r_mesh_unnormalized = resize_frame_and_mesh(
            r_frame, r_mesh, (target_w, target_h)
        )
    else:
        target_h, target_w = f_frame.shape[0:2]

        f_frame_resized, f_mesh_unnormalized = f_frame, f_mesh
        l_frame_resized, l_mesh_unnormalized = l_frame, l_mesh
        r_frame_resized, r_mesh_unnormalized = r_frame, r_mesh

    observations = {
        "front": f_mesh_unnormalized[:, :2],
        "left": l_mesh_unnormalized[:, :2],
        "right": r_mesh_unnormalized[:, :2],
    }

    mesh_3d, stats, detail = triangulate_with_missing(
        observations=observations,
        extrinsics=rt_info,
        Ks=K,
        max_err_px=800.0,
        sampson_px=200.0,
    )

    if np.isnan(mesh_3d).all():
        logger.warning(f"Triangulation failed for frame {i}")

        # 保存3D关键点构成信息
        detail_path = output_path / "frame_triangulation_detail" / f"frame_{i:04d}.txt"
        detail_to_txt(detail, detail_path)

        # 保存统计信息
        stats_path = output_path / "frame_triangulation_stats.txt"
        with open(stats_path, "a") as f:
            f.write(f"Frame {i}: Triangulation failed.\n")

        return False, (target_h, target_w)

    else:
        logger.info(
            f"Frame {i}: Triangulation stats: {stats['ok']}/{stats['total']} points OK, "
            f"Mean RPE: {stats['mean_rpe']:.2f}px, "
            f"Epi Inlier: {stats['epi_inlier']*100:.2f}%, "
            f"Cheirality: {stats['cheirality']*100:.2f}%"
        )

        # 保存3D关键点构成信息
        detail_path = output_path / "frame_triangulation_detail" / f"frame_{i:04d}.txt"
        detail_to_txt(detail, detail_path)

        # 保存统计信息
        stats_path = output_path / "frame_triangulation_stats.txt"
        with open(stats_path, "a") as f:
            f.write(
                f"Frame {i}: Triangulation stats: {stats['ok']}/{stats['total']} points OK, "
                f"Mean RPE: {stats['mean_rpe']:.2f}px, "
                f"Epi Inlier: {stats['epi_inlier']*100:.2f}%, "
                f"Cheirality: {stats['cheirality']*100:.2f}%\n"
            )

        if vis.save_mesh_frame:
            draw_and_save_mesh_from_frame(
                frame=f_frame_resized,
                mesh=f_mesh_unnormalized,
                save_path=output_path / "vis" / f"mesh_frames/front/frame_{i:04d}.png",
                color=(0, 255, 0),
                radius=2,
                draw_tesselation=True,
                draw_contours=True,
                with_index=False,
            )
            draw_and_save_mesh_from_frame(
                frame=l_frame_resized,
                mesh=l_mesh_unnormalized,
                save_path=output_path / "vis" / f"mesh_frames/left/frame_{i:04d}.png",
                color=(0, 255, 0),
                radius=2,
                draw_tesselation=True,
                draw_contours=True,
                with_index=False,
            )
            draw_and_save_mesh_from_frame(
                frame=r_frame_resized,
                mesh=r_mesh_unnormalized,
                save_path=output_path / "vis" / f"mesh_frames/right/frame_{i:04d}.png",
                color=(0, 255, 0),
                radius=2,
                draw_tesselation=True,
                draw_contours=True,
                with_index=False,
            )

        return mesh_3d, (target_h, target_w)


def process_one_video(
    environment_dir: dict[str, Path], output_path: Path, rt_info, K, vis
):

    output_path.mkdir(parents=True, exist_ok=True)

    front_frames, front_mesh, front_video_info, front_not_mesh_list = (
        load_SAM3D_results_from_npz(environment_dir["front"])
    )
    if front_frames is None:
        logger.warning(f"No front view data found in {environment_dir['front']}")
        return

    left_frames, left_mesh, left_video_info, left_not_mesh_list = (
        load_SAM3D_results_from_npz(environment_dir["left"])
    )
    right_frames, right_mesh, right_video_info, right_not_mesh_list = (
        load_SAM3D_results_from_npz(environment_dir["right"])
    )

    # * 确保三视点帧数一致
    if (
        left_frames.shape[0] != right_frames.shape[0]
        or left_frames.shape[0] != front_frames.shape[0]
        or right_frames.shape[0] != front_frames.shape[0]
    ):
        min_frames = min(
            left_frames.shape[0], right_frames.shape[0], front_frames.shape[0]
        )
        left_frames = left_frames[:min_frames]
        left_mesh = left_mesh[:min_frames]
        right_frames = right_frames[:min_frames]
        right_mesh = right_mesh[:min_frames]
        front_frames = front_frames[:min_frames]
        front_mesh = front_mesh[:min_frames]

        logger.warning(
            f"Aligned all views to {min_frames} frames based on the shortest video."
        )
    else:
        min_frames = left_frames.shape[0]

    all_mesh_3d = []

    # * 逐帧处理三视点数据，三角测量 3D 网格
    for i in range(min_frames):

        # ! debug
        # if i > 50:
        #     break

        f_mesh = front_mesh[i]
        l_mesh = left_mesh[i]
        r_mesh = right_mesh[i]

        f_frame = front_frames[i]
        l_frame = left_frames[i]
        r_frame = right_frames[i]

        mesh_3d, (target_h, target_w) = process_one_frame(
            f_frame,
            f_mesh,
            l_frame,
            l_mesh,
            r_frame,
            r_mesh,
            rt_info,
            K,
            output_path,
            i,
            vis,
        )

        if mesh_3d is False:
            all_mesh_3d.append(np.full_like(f_mesh, np.nan))
        else:
            all_mesh_3d.append(mesh_3d)

    all_mesh_3d = np.array(all_mesh_3d)  # (T, N, 3)

    # * 检查缺帧 & 自动插值
    all_mesh_3d, info = interpolate_missing_frames(
        all_mesh_3d, method="auto", zero_as_missing=True, zero_axes=(0, 1, 2)
    )
    logger.info(f"Missing frame ratio: {info['miss_ratio']:.3f}")
    logger.info(f"Longest missing run: {info['longest_missing_run']} frames")
    logger.info(f"Interpolation method used: {info['method_effective']}")

    # * 评估原始三角测量结果
    metrics_before = evaluate_face3d_pro(all_mesh_3d, fps=30)
    logger.info("====== Evaluation Metrics Before Smoothing ======")
    for k, v in metrics_before.items():
        logger.info(f"{k}: {v:.5f}")
    # * 导出评估报告
    export_report(
        metrics_before,
        outdir=output_path / "evaluation_report_before_smoothing",
    )

    # * 平滑 3D 网格序列
    all_mesh_3d_smooth, report = smooth_combo(all_mesh_3d, alpha=0.3, win=9, poly=2)

    # * 打印平滑报告
    logger.info("====== Smooth Report ======")
    logger.info(f"Frames: {report['frames']}, Joints: {report['joints']}")
    logger.info(f"Motion Energy Before: {report['motion_energy_before']:.5f}")
    logger.info(f"Motion Energy After : {report['motion_energy_after']:.5f}")
    logger.info(f"Reduction Rate      : {report['reduction_rate']*100:.2f}% ↓")

    # * 单点轨迹可视化
    # TODO:

    # * 全剧平均改变量
    diff = np.nanmean(np.linalg.norm(all_mesh_3d_smooth - all_mesh_3d, axis=2))
    logger.info(f"Average 3D point change after smoothing: {diff:.5f}")

    # * 评估模型输出稳定性和多视一致性
    metrics = evaluate_face3d_pro(all_mesh_3d_smooth, fps=30)

    logger.info("====== Evaluation Metrics ======")
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.5f}")

    # * 导出评估报告
    export_report(
        metrics,
        outdir=output_path / "evaluation_report",
    )

    # * 保存 3D 关键点
    for frame_idx in range(all_mesh_3d_smooth.shape[0]):

        if vis.save_mesh_3d and not np.isnan(all_mesh_3d_smooth[frame_idx]).all():

            save_3d_joints(
                mesh_3d=all_mesh_3d_smooth[frame_idx],
                save_dir=output_path / "3d_joints" / "smooth",
                frame_idx=frame_idx,
                rt_info=rt_info,
                k=K,
                video_path={
                    "front": str(environment_dir["front"]["video"]),
                    "left": str(environment_dir["left"]["video"]),
                    "right": str(environment_dir["right"]["video"]),
                },
                npz_path={
                    "front": str(environment_dir["front"]["npz"]),
                    "left": str(environment_dir["left"]["npz"]),
                    "right": str(environment_dir["right"]["npz"]),
                },
            )

            visualize_3d_mesh(
                mesh_3d=all_mesh_3d_smooth[frame_idx],
                save_path=output_path
                / "vis"
                / f"mesh_3D_frames_depth_colored/frame_{frame_idx:04d}.png",
                title=f"Frame {frame_idx} - 3D Mesh (Depth Colored)",
                rt_info=rt_info,
                K=K,
                image_size=(target_w, target_h),
                save_views=["default", "front", "left", "right"],  # 需要哪些就写哪些
                default_view=(20.0, -60.0),
                invert_y=False,
            )

    # * merge 3d mesh visualization frames to video
    if vis.merge_3d_frames_to_video:
        merge_frames_to_video(
            frame_dir=output_path / "vis" / "mesh_3D_frames_depth_colored/top",
            output_video_path=output_path / "vis" / (output_path.stem + "_top.mp4"),
            fps=30,
        )

        merge_frames_to_video(
            frame_dir=output_path / "vis" / "mesh_3D_frames_depth_colored/default",
            output_video_path=output_path / "vis" / (output_path.stem + "_default.mp4"),
            fps=30,
        )
