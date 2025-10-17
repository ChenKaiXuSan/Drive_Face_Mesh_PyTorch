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
import torch

import logging

logger = logging.getLogger(__name__)

import hydra

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


from mesh_triangulation.camera_position_mapping import prepare_camera_position
from mesh_triangulation.load import load_mesh_from_npz
from mesh_triangulation.multi_triangulation import triangulate_with_missing


# ---------- 可视化工具 ----------
def draw_and_save_keypoints_from_frame(
    frame,
    keypoints,
    save_path,
    color=(0, 255, 0),
    radius=4,
    thickness=-1,
    with_index=True,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img = frame.numpy() if isinstance(frame, torch.Tensor) else frame.copy()
    for i, (x, y) in enumerate(keypoints):
        if np.isnan(x) or np.isnan(y):
            continue
        cv2.circle(img, (int(x), int(y)), radius, color, thickness)
        if with_index:
            cv2.putText(
                img,
                str(i),
                (int(x) + 4, int(y) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)
    print(f"[INFO] Saved image with keypoints to: {save_path}")


def draw_camera(ax, R, T, scale=0.1, label="Cam"):
    origin = T.reshape(3)
    x_axis = R @ np.array([1, 0, 0]) * scale + origin
    y_axis = R @ np.array([0, 1, 0]) * scale + origin
    z_axis = R @ np.array([0, 0, 1]) * scale + origin
    view_dir = R @ np.array([0, 0, -1]) * scale * 1.5 + origin  # 摄像头朝向（-Z轴）

    ax.plot(
        [origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], c="r"
    )
    ax.plot(
        [origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], c="g"
    )
    ax.plot(
        [origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], c="b"
    )

    # 视线方向箭头（黑色）
    ax.plot(
        [origin[0], view_dir[0]],
        [origin[1], view_dir[1]],
        [origin[2], view_dir[2]],
        c="k",
        linestyle="--",
    )

    # 相机标签
    ax.text(*origin, label, color="black")


def visualize_3d_joints(joints_3d, R, T, save_path, title="Triangulated 3D Joints"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    draw_camera(ax, np.eye(3), np.zeros(3), label="Cam1")
    draw_camera(ax, R, T, label="Cam2")

    ax.scatter(joints_3d[:, 0], joints_3d[:, 2], joints_3d[:, 1], c="blue", s=30)
    for i, (x, y, z) in enumerate(joints_3d):
        ax.text(x, z, y, str(i), size=8)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved: {save_path}")


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
    h, w = frame.shape[:2]
    new_w, new_h = new_size

    # 图像缩放
    frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 坐标缩放比例
    sx, sy = new_w / w, new_h / h

    # 缩放 mesh 坐标
    mesh_rescaled = np.asarray(mesh, dtype=np.float32).copy()
    mesh_rescaled[:, 0] *= sx  # x 方向缩放
    mesh_rescaled[:, 1] *= sy  # y 方向缩放
    # z 一般保持不变（除非是相机空间深度）
    return frame_resized, mesh_rescaled


# ---------- 主处理函数 ----------
def process_one_video(environment_dir: dict[str, Path], output_path: Path, rt_info, K):

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
            target_h = max(f_mesh.shape[0], l_mesh.shape[0], r_mesh.shape[0])
            target_w = max(f_mesh.shape[1], l_mesh.shape[1], r_mesh.shape[1])

            f_frame_resized, f_mesh_rescaled = resize_frame_and_mesh(
                f_frame, f_mesh, (target_w, target_h)
            )
            l_frame_resized, l_mesh_rescaled = resize_frame_and_mesh(
                l_frame, l_mesh, (target_w, target_h)
            )
            r_frame_resized, r_mesh_rescaled = resize_frame_and_mesh(
                r_frame, r_mesh, (target_w, target_h)
            )

        observations = {
            "front": f_mesh_rescaled[:, :2],
            "left": l_mesh_rescaled[:, :2],
            "right": r_mesh_rescaled[:, :2],
        }

        mesh_3d = triangulate_with_missing(
            observations=observations,
            extrinsics=rt_info,
            Ks=K,
            max_err_px=8.0,
        )

        if np.isnan(mesh_3d).all():
            logger.warning(f"Triangulation failed for frame {i}")
            continue

        visualize_3d_joints(
            mesh_3d,
            R,
            T,
            os.path.join(output_path, f"3d/frame_{i:04d}.png"),
            title=f"Frame {i} - 3D Joints",
        )


# ---------- 多人批量处理入口 ----------
# TODO：这里需要同时加载一个人的四个视频逻辑才行
# TODO: 这里需要使用rt info里面的外部参数才行
def process_person_videos(
    mesh_path: Path, video_path: Path, output_path: Path, rt_info, K
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

            process_one_video(mapped_info, out_dir, rt_info, K)


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
    )


if __name__ == "__main__":

    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
