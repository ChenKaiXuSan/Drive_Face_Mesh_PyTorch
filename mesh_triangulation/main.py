#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Triangulate 3D joints from two-view 2D keypoints using either video frames or pre-extracted keypoints.
Supports modular triangulation, pose estimation, and interactive 3D visualization.

Author: Kaixu Chen
Last Modified: August 4th, 2025
"""

import os
import numpy as np
import cv2
from pathlib import Path

import logging

logger = logging.getLogger(__name__)

import hydra


from mesh_triangulation.camera_position_mapping import prepare_camera_position
from mesh_triangulation.load import load_mesh_from_npz
from mesh_triangulation.multi_triangulation import triangulate_with_missing

# vis
from mesh_triangulation.vis.frame_visualization import draw_and_save_mesh_from_frame
from mesh_triangulation.vis.mesh_visualization import visualize_3d_mesh
from mesh_triangulation.vis.merge_video import merge_frames_to_video

# save
from mesh_triangulation.save import save_3d_joints


def resize_frame_and_mesh(frame: np.ndarray, mesh: np.ndarray, new_size):
    """
    同时缩放图像 frame 和 mesh 坐标
    参数:
      frame: np.ndarray, 形状 (H, W, 3)
      mesh: np.ndarray, 形状 (N, 3)
      new_size: (new_w, new_h)
    返回:
      frame_resized, mesh_rescaled
    """
    new_w, new_h = new_size

    # 图像缩放
    if frame.shape[1] != new_w and frame.shape[0] != new_h:

        frame_resized = cv2.resize(
            frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )
    else:
        frame_resized = frame.copy()

    # normalize mesh 坐标
    if not np.allclose(mesh, 0):
        mesh_normalized = np.asarray(mesh, dtype=np.float32).copy()
        mesh_normalized[:, 0] *= new_w
        mesh_normalized[:, 1] *= new_h  # y 方向缩放
        # z 一般保持不变（除非是相机空间深度）
    else:
        mesh_normalized = mesh.copy()

    return frame_resized, mesh_normalized


# ---------- 主处理函数 ----------
def process_one_video(
    environment_dir: dict[str, Path], output_path: Path, rt_info, K, vis
):

    os.makedirs(output_path, exist_ok=True)

    front_frames, front_mesh, front_video_info = load_mesh_from_npz(
        environment_dir["front"]
    )
    if front_frames is None:
        print(f"[WARN] No front view data found in {environment_dir['front']}")
        return

    left_frames, left_mesh, left_video_info = load_mesh_from_npz(
        environment_dir["left"]
    )
    right_frames, right_mesh, right_video_info = load_mesh_from_npz(
        environment_dir["right"]
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

    for i in range(min_frames):

        # ! debug
        # if i > 30:

        f_mesh = front_mesh[i]
        l_mesh = left_mesh[i]
        r_mesh = right_mesh[i]

        f_frame = front_frames[i]
        l_frame = left_frames[i]
        r_frame = right_frames[i]

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
            f_frame_resized, f_mesh_unnormalized = f_frame, f_mesh
            l_frame_resized, l_mesh_unnormalized = l_frame, l_mesh
            r_frame_resized, r_mesh_unnormalized = r_frame, r_mesh

        observations = {
            "front": f_mesh_unnormalized[:, :2],
            "left": l_mesh_unnormalized[:, :2],
            "right": r_mesh_unnormalized[:, :2],
        }

        mesh_3d = triangulate_with_missing(
            observations=observations,
            extrinsics=rt_info,
            Ks=K,
            max_err_px=800.0,
        )

        if np.isnan(mesh_3d).all():
            logger.warning(f"Triangulation failed for frame {i}")
            continue

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

        if vis.save_mesh_3d and not np.isnan(mesh_3d).all():
            save_3d_joints(
                mesh_3d=mesh_3d,
                save_dir=output_path / "3d_joints",
                frame_idx=i,
                rt_info=rt_info,
                k=K,
                video_path={
                    "front": str(environment_dir["front"]['video']),
                    "left": str(environment_dir["left"]['video']),
                    "right": str(environment_dir["right"]['video']),
                },
                npz_path={
                    "front": str(environment_dir["front"]['npz']),
                    "left": str(environment_dir["left"]['npz']),
                    "right": str(environment_dir["right"]['npz']),
                },
            )

            visualize_3d_mesh(
                mesh_3d,
                output_path / "vis" / f"mesh_3D_frames/frame_{i:04d}.png",
                title=f"Frame {i} - 3D Mesh",
            )

    # * merge 3d mesh visualization frames to video
    if vis.merge_3d_frames_to_video:
        merge_frames_to_video(
            frame_dir=output_path / "vis" / "mesh_3D_frames",
            output_video_path=output_path / "vis" / (output_path.stem + ".mp4"),
            fps=30,
        )


# ---------- 多人批量处理入口 ----------
def process_person_videos(
    mesh_path: Path, video_path: Path, output_path: Path, rt_info, K, vis_flag
):
    subjects = sorted(mesh_path.glob("*/"))
    if not subjects:
        raise FileNotFoundError(f"No folders found in: {mesh_path}")
    print(f"[INFO] Found {len(subjects)} subjects in {mesh_path}")
    for person_dir in subjects:

        person_name = person_dir.name
        print(f"\n[INFO] Processing: {person_name}")

        # 不同的环境
        for environment_dir in person_dir.glob("*/"):
            print(f"[INFO] Found video in {environment_dir.name}")

            npz_file = sorted(environment_dir.glob("*/*.npz"))

            mapped_info = {}

            out_dir = output_path / person_name / environment_dir.name

            # mapping the npz files to left, right, front
            for file in npz_file:
                mapped_info[file.stem] = {
                    "npz": file,
                    "video": video_path
                    / person_name
                    / environment_dir.name
                    / f"{file.stem}.mp4",
                }

            process_one_video(mapped_info, out_dir, rt_info, K, vis_flag)


@hydra.main(
    version_base=None, config_path="../configs", config_name="mesh_triangulation"
)
def main(cfg):

    # 准备相机外部参数
    camera_position_dict = prepare_camera_position(
        K=cfg.camera_K,
        T=cfg.camera_position.T,
        z=cfg.camera_position.z,
        output_path=cfg.paths.log_path,
        img_size=cfg.camera_K.image_size,
        dist_front=cfg.camera_position.dist_front,
        dist_left=cfg.camera_position.dist_left,
        dist_right=cfg.camera_position.dist_right,
        baseline=cfg.camera_position.baseline,
    )

    process_person_videos(
        mesh_path=Path(cfg.paths.mesh_path),
        video_path=Path(cfg.paths.video_path),
        output_path=Path(cfg.paths.log_path),
        rt_info=camera_position_dict["rt_info"],
        K=camera_position_dict["K_map"],
        vis_flag=cfg.vis,
    )


if __name__ == "__main__":

    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
