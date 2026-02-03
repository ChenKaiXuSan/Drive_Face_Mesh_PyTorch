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

import cv2
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


# vis
from head3D_fuse.visualization.vis_utils import (
    visualize_3d_skeleton,
    visualizer,
    _save_view_visualizations,
    _save_fused_visualization,
)

from head3D_fuse.visualization.merge_video import merge_frames_to_video

# save
from head3D_fuse.save import _save_fused_keypoints

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


def _valid_keypoints_mask(keypoints: np.ndarray, zero_eps: float) -> np.ndarray:
    finite = np.isfinite(keypoints).all(axis=-1)
    nonzero = np.linalg.norm(keypoints, axis=-1) >= zero_eps
    return finite & nonzero


def _estimate_similarity_transform(
    source: np.ndarray,
    target: np.ndarray,
    allow_scale: bool,
    eps: float = 1e-8,
) -> Tuple[float, np.ndarray, np.ndarray]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    h_mat = source_centered.T @ target_centered
    u_mat, s_vals, vt_mat = np.linalg.svd(h_mat)
    rot = vt_mat.T @ u_mat.T
    if np.linalg.det(rot) < 0:
        vt_mat[-1, :] *= -1
        rot = vt_mat.T @ u_mat.T
    scale = 1.0
    if allow_scale:
        denom = float(np.sum(source_centered**2))
        scale = float(np.sum(s_vals) / denom) if denom > eps else 1.0
    translation = target_mean - scale * (source_mean @ rot)
    return scale, rot, translation


def _align_keypoints_to_reference(
    reference: np.ndarray,
    source: np.ndarray,
    zero_eps: float,
    allow_scale: bool,
) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float64)
    src = np.asarray(source, dtype=np.float64)
    ref_valid = _valid_keypoints_mask(ref, zero_eps)
    src_valid = _valid_keypoints_mask(src, zero_eps)
    pair_valid = ref_valid & src_valid
    if pair_valid.sum() < MIN_POINTS_FOR_ALIGNMENT:
        logger.warning(
            "Not enough valid joints for alignment (need >=%d, got %d).",
            MIN_POINTS_FOR_ALIGNMENT,
            pair_valid.sum(),
        )
        return source
    scale, rot, translation = _estimate_similarity_transform(
        src[pair_valid], ref[pair_valid], allow_scale
    )
    return _finalize_aligned_keypoints(src, source, src_valid, scale, rot, translation)


def _finalize_aligned_keypoints(
    source_points: np.ndarray,
    original_source: np.ndarray,
    valid_mask: np.ndarray,
    scale: float,
    rot: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    aligned = (scale * source_points @ rot) + translation
    aligned[~valid_mask] = original_source[~valid_mask]
    return aligned


def _select_trimmed_inliers(
    residuals: np.ndarray,
    valid_mask: np.ndarray,
    trim_ratio: float,
) -> np.ndarray:
    if trim_ratio <= 0:
        # trim_ratio <= 0 means no trimming; keep all valid points.
        return valid_mask
    if trim_ratio >= 1.0:
        raise ValueError(
            f"trim_ratio must be less than 1.0 (<=0 disables trimming), got {trim_ratio}"
        )
    valid_idx = np.flatnonzero(valid_mask)
    n_valid = valid_idx.size
    if n_valid == 0:
        return valid_mask
    raw_keep = int(np.ceil((1.0 - trim_ratio) * n_valid))
    if raw_keep < MIN_POINTS_FOR_ALIGNMENT:
        logger.warning(
            "trim_ratio keeps fewer than %d points; clamping to minimum.",
            MIN_POINTS_FOR_ALIGNMENT,
        )
    n_keep = min(n_valid, max(MIN_POINTS_FOR_ALIGNMENT, raw_keep))
    order = np.argsort(residuals[valid_idx])
    keep_idx = valid_idx[order[:n_keep]]
    trimmed = np.zeros_like(valid_mask, dtype=bool)
    trimmed[keep_idx] = True
    return trimmed


def _align_keypoints_trimmed(
    reference: np.ndarray,
    source: np.ndarray,
    zero_eps: float,
    allow_scale: bool,
    trim_ratio: float,
    max_iters: int,
) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float64)
    src = np.asarray(source, dtype=np.float64)
    ref_valid = _valid_keypoints_mask(ref, zero_eps)
    src_valid = _valid_keypoints_mask(src, zero_eps)
    pair_valid = ref_valid & src_valid
    if pair_valid.sum() < MIN_POINTS_FOR_ALIGNMENT:
        logger.warning(
            "Not enough valid joints for robust alignment (need >=%d, got %d).",
            MIN_POINTS_FOR_ALIGNMENT,
            pair_valid.sum(),
        )
        return source
    inlier_mask = pair_valid.copy()
    scale = 1.0
    rot = np.eye(3)
    translation = np.zeros(3)
    for _ in range(max_iters):
        if inlier_mask.sum() < MIN_POINTS_FOR_ALIGNMENT:
            break
        scale, rot, translation = _estimate_similarity_transform(
            src[inlier_mask], ref[inlier_mask], allow_scale
        )
        aligned = (scale * src @ rot) + translation
        residuals = np.linalg.norm(aligned - ref, axis=-1)
        new_inlier_mask = _select_trimmed_inliers(residuals, pair_valid, trim_ratio)
        if np.array_equal(new_inlier_mask, inlier_mask):
            break
        inlier_mask = new_inlier_mask
    if inlier_mask.sum() < MIN_POINTS_FOR_ALIGNMENT:
        logger.warning(
            "Robust alignment fell back to original source (inliers=%d, min=%d).",
            inlier_mask.sum(),
            MIN_POINTS_FOR_ALIGNMENT,
        )
        return source
    return _finalize_aligned_keypoints(src, source, src_valid, scale, rot, translation)


def _apply_view_transform(
    keypoints: Optional[np.ndarray],
    transform: Optional[Dict[str, np.ndarray]],
    mode: str,
) -> Optional[np.ndarray]:
    """
    Align keypoints to a common world coordinate using per-view extrinsics.

    Args:
        keypoints: (N, 3) 3D keypoints in the source coordinate system.
        transform: dict containing:
            - "R": (3, 3) rotation matrix.
            - "t": (3,) camera origin in world coordinates (same as C).
            - "t_wc": (3,) optional world->camera translation (OpenCV style).
              With OpenCV convention X_cam = R_wc @ X_world + t_wc and row-vector
              points, the camera origin is C = -(t_wc @ R_wc).
            - "C": (3,) optional camera origin in world (alias of "t").
        mode:
            - "world_to_camera": uses world->camera extrinsics (R_wc, t_wc) to align
              camera coordinates into world coordinates. With row-vector points,
              X_world = (X_cam - t_wc) @ R_wc. If "C" or "t" is provided instead of
              "t_wc", uses X_world = X_cam @ R_wc + C.
            - "camera_to_world": uses camera->world extrinsics (R_cw, t_cw) via
              X_world = X_cam @ R_cw.T + t_cw (camera origin in world).
    """
    if keypoints is None or transform is None:
        return keypoints
    if transform.get("R") is None:
        raise ValueError("transform['R'] is required for view alignment")
    R = np.asarray(transform.get("R"))
    if R.shape != (3, 3):
        raise ValueError(f"Expected rotation matrix with shape (3, 3), got {R.shape}")
    t = transform.get("t")
    if t is not None:
        t = np.asarray(t, dtype=np.float64).reshape(3)
    t_wc = transform.get("t_wc")
    if t_wc is not None:
        t_wc = np.asarray(t_wc, dtype=np.float64).reshape(3)
    camera_center = transform.get("C")
    if camera_center is not None:
        camera_center = np.asarray(camera_center, dtype=np.float64).reshape(3)
    if camera_center is None and t is not None:
        camera_center = t
    keypoints = np.asarray(keypoints, dtype=np.float64)
    if mode == "world_to_camera":
        if t_wc is None:
            if camera_center is None:
                raise ValueError("world_to_camera mode requires 't', 't_wc', or 'C'")
            return (keypoints @ R) + camera_center
        return (keypoints - t_wc) @ R
    if mode == "camera_to_world":
        if camera_center is None:
            raise ValueError("camera_to_world mode requires 't' or 'C' translation")
        return (keypoints @ R.T) + camera_center
    raise ValueError(
        "transform_mode must be 'world_to_camera' or 'camera_to_world', "
        f"got '{mode}'"
    )


def fuse_3view_keypoints(
    keypoints_by_view: Dict[str, np.ndarray],
    method: str = "median",
    zero_eps: float = 1e-6,
    fill_value: Optional[float] = None,
    view_transforms: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    transform_mode: str = "world_to_camera",
    alignment_method: str = "none",
    alignment_reference: Optional[str] = None,
    alignment_scale: bool = True,
    alignment_trim_ratio: float = 0.2,
    alignment_max_iters: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fuse three-view 3D keypoints by ignoring invalid (near-zero) joints.

    Optionally apply per-view extrinsics to align each view into a common
    world coordinate system before fusing. If no extrinsics are available,
    a Procrustes alignment (rotation/translation and optional scale) can be
    used to align each view to a reference view.

    Args:
        alignment_trim_ratio: fraction of points to trim as outliers (0.0 to <1.0).
            Values <= 0 keep all points. Only used when alignment_method is
            "procrustes_trimmed".
        alignment_max_iters: maximum iterations for trimmed alignment refinement.
            Only used when alignment_method is "procrustes_trimmed".
    """
    if fill_value is None:
        fill_value = np.nan
    if alignment_method not in VALID_ALIGNMENT_METHODS:
        raise ValueError(
            f"alignment_method must be one of {VALID_ALIGNMENT_METHODS}, "
            f"got '{alignment_method}'"
        )
    view_list = list(keypoints_by_view.keys())
    if view_transforms:
        keypoints_by_view = {
            view: _apply_view_transform(
                keypoints_by_view[view],
                view_transforms.get(view),
                transform_mode,
            )
            for view in view_list
        }
    elif alignment_method in ("procrustes", "procrustes_trimmed"):
        reference_view = alignment_reference or view_list[0]
        if reference_view not in keypoints_by_view:
            raise ValueError(f"Reference view '{reference_view}' not found in views.")
        reference = keypoints_by_view[reference_view]
        keypoints_by_view = {
            view: (
                reference
                if view == reference_view
                else (
                    _align_keypoints_to_reference(
                        reference,
                        keypoints_by_view[view],
                        zero_eps=zero_eps,
                        allow_scale=alignment_scale,
                    )
                    if alignment_method == "procrustes"
                    else _align_keypoints_trimmed(
                        reference,
                        keypoints_by_view[view],
                        zero_eps=zero_eps,
                        allow_scale=alignment_scale,
                        trim_ratio=alignment_trim_ratio,
                        max_iters=alignment_max_iters,
                    )
                )
            )
            for view in view_list
        }
    stacked = np.stack([keypoints_by_view[view] for view in view_list], axis=0).astype(
        np.float64
    )
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
    from head3D_fuse.visualization.vis_utils import (
        _save_fused_visualization,
        _save_view_visualizations,
        visualizer,
    )
    person_id = person_env_dir.parent.name
    env_name = person_env_dir.name
    view_list = cfg.infer.get("view_list")
    if view_list is None:
        view_list = cfg.infer.get("views_list")
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
