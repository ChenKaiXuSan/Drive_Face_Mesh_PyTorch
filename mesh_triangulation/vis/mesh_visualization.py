#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/triangulation/vis/pose_visualization.py
Project: /workspace/code/triangulation/vis
Created Date: Tuesday October 14th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Tuesday October 14th 2025 10:55:01 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""
from typing import Iterable, List, Optional, Sequence, Tuple, Union, Dict

from pathlib import Path
import logging

logger = logging.getLogger(__name__)

import numpy as np

import matplotlib.pyplot as plt

import mediapipe as mp

# MediaPipe 的连接表（tesselation + 轮廓）
mp_face_mesh = mp.solutions.face_mesh
TES = mp_face_mesh.FACEMESH_TESSELATION
CON = mp_face_mesh.FACEMESH_CONTOURS

# ---- 新增：骨长计算 ----


def compute_bone_lengths(
    pts: np.ndarray,
    skeleton: Iterable[Tuple[int, int]],
    *,
    ignore_nan: bool = True,
) -> np.ndarray:
    """
    计算一帧 3D 关键点在给定骨架下的骨长。
    pts: (K,3)
    返回: (E,) 对应 skeleton 中每条边的长度；无效边为 np.nan
    """
    P = np.asarray(pts, dtype=float)
    L: List[float] = []
    for i, j in skeleton:
        if i >= len(P) or j >= len(P):
            L.append(np.nan)
            continue
        a, b = P[i], P[j]
        if ignore_nan and (not np.all(np.isfinite(a)) or not np.all(np.isfinite(b))):
            L.append(np.nan)
            continue
        L.append(float(np.linalg.norm(a - b)))
    return np.asarray(L, dtype=float)


def compute_bone_stats(lengths: np.ndarray) -> Dict[str, float]:
    """
    对骨长（含 nan）做统计，返回 mean/median/std/min/max/valid_count。
    """
    x = np.asarray(lengths, dtype=float)
    valid = np.isfinite(x)
    if not np.any(valid):
        return dict(
            mean=np.nan,
            median=np.nan,
            std=np.nan,
            min=np.nan,
            max=np.nan,
            valid_count=0,
        )
    xv = x[valid]
    return dict(
        mean=float(np.nanmean(xv)),
        median=float(np.nanmedian(xv)),
        std=float(np.nanstd(xv)),
        min=float(np.nanmin(xv)),
        max=float(np.nanmax(xv)),
        valid_count=int(valid.sum()),
    )


def draw_camera(
    ax: plt.Axes,
    rt_info: Union[Dict[str, np.ndarray], Tuple[np.ndarray, np.ndarray]],
    K: np.ndarray,  # (3,3)
    image_size: Tuple[int, int],  # (W, H) in px
    axis_len: float = 1.0,
    frustum_depth: float = 1.0,
    colors: Tuple[str, str, str] = ("r", "g", "b"),
    label: Optional[str] = None,
    convention: str = "cam2world",  # or "cam2world"
    ray_scale_mode: str = "depth",  # "depth" or "focal"
    linewidths: Optional[Dict[str, float]] = None,
    frustum_alpha: float = 1.0,
) -> np.ndarray:
    """
    在 Matplotlib 3D 轴上绘制 OpenCV 相机坐标系与视锥体。

    坐标系对应：
      OpenCV: x→右, y→下, z→前
      Matplotlib: x→右, y→前, z→上

    返回:
        C_plt: (3,) 相机中心在 Matplotlib 世界坐标中的位置
    """
    # ---------------- 参数准备 ----------------
    if linewidths is None:
        linewidths = {"axis": 1.0, "frustum": 0.5}

    R, T, C = rt_info["R"], rt_info["t"], rt_info["C"]
    K = np.asarray(K, float).reshape(3, 3)
    W, H = [float(v) for v in image_size]

    # ---------------- 相机中心计算 ----------------
    # Xc = R Xw + T → C = -R^T T
    R_wc = R
    C_world = -R.T @ T

    C_plt = C_world.ravel()

    # ---------------- 绘制相机坐标轴 ----------------
    for axis_vec, color in zip(R_wc, colors):  # R_wc 的行向量即各相机轴方向
        end_cv = axis_vec * axis_len
        end_plt = end_cv
        ax.plot(
            [C_plt[0], C_plt[0] + end_plt[0]],
            [C_plt[1], C_plt[1] + end_plt[1]],
            [C_plt[2], C_plt[2] + end_plt[2]],
            c=color,
            lw=linewidths["axis"],
        )

    # ---------------- 计算视锥体四个角点 ----------------
    corners_px = np.array(
        [
            [0, 0, 1],
            [W - 1, 0, 1],
            [W - 1, H - 1, 1],
            [0, H - 1, 1],
        ],
        dtype=float,
    )
    rays_cam = np.linalg.inv(K) @ corners_px.T  # (3,4)

    if ray_scale_mode == "depth":
        scale = frustum_depth / np.clip(rays_cam[2, :], 1e-9, None)
        rays_cam = rays_cam * scale
    else:
        fx, fy = K[0, 0], K[1, 1]
        s = max(axis_len, frustum_depth) / max((fx + fy) / 2, 1e-6)
        rays_cam *= s

    # cam→world(OpenCV)
    corners_world_cv = (R_wc @ rays_cam).T + C_world.reshape(1, 3)
    # world(OpenCV) → Matplotlib
    corners_world_plt = corners_world_cv

    # ---------------- 绘制视锥体边缘 ----------------
    for p in corners_world_plt:
        ax.plot(
            [C_plt[0], p[0]],
            [C_plt[1], p[1]],
            [C_plt[2], p[2]],
            c="k",
            lw=linewidths["frustum"],
            alpha=frustum_alpha,
        )

    loop = [0, 1, 2, 3, 0]
    ax.plot(
        corners_world_plt[loop, 0],
        corners_world_plt[loop, 1],
        corners_world_plt[loop, 2],
        c="k",
        lw=linewidths["frustum"],
        alpha=frustum_alpha,
    )

    # ---------------- 标签 ----------------
    if label:
        ax.text(C_plt[0], C_plt[1], C_plt[2], label, color="black", fontsize=9)

    return C_plt


def visualize_3d_mesh(
    mesh_3d: np.ndarray,
    save_path: Path,
    title="Triangulated 3D Mesh",
):

    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    xlab, ylab, zlab = "X", "Z", "Y (up)"

    # 点与索引
    mesh_3d[:, 1] = -mesh_3d[:, 1]  # 反转Y轴以符合Y朝上习惯
    ax.scatter(mesh_3d[:, 0], mesh_3d[:, 1], mesh_3d[:, 2], c="blue", s=30)

    for i, (x, y, z) in enumerate(mesh_3d):
        ax.text(x, y, z, str(i), size=8)

    # mesh连接骨架
    for a, b in TES:
        pa, pb = mesh_3d[a], mesh_3d[b]
        ax.plot(
            [pa[0], pb[0]],
            [pa[1], pb[1]],
            [pa[2], pb[2]],
            c="gray",
            linewidth=0.5,
        )

    plt.tight_layout()

    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_zlabel(zlab)

    plt.tight_layout()

    fig.savefig(str(save_path))
    plt.close(fig)
    logger.info(f"Saved 3D mesh visualization to {save_path}")
