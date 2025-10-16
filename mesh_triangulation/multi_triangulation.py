#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/code/mesh_triangulation/multi_triangulation.py
Project: /workspace/code/mesh_triangulation
Created Date: Wednesday October 15th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Wednesday October 15th 2025 8:47:31 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''
import numpy as np
import cv2
from typing import Dict, List, Tuple

# --------- 工具函数 ---------
def build_P(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """K(3x3), R(3x3), t(3,) -> P(3x4)"""
    return K @ np.hstack([R, t.reshape(3,1)])

def triangulate_two_views(P1: np.ndarray, P2: np.ndarray,
                          x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """两视点线性三角测量，x1/x2: (2,) 像素坐标 → 返回 X(3,) 齐次归一化"""
    # OpenCV 需要 (2, N)
    X_h = cv2.triangulatePoints(P1, P2, x1.reshape(2,1), x2.reshape(2,1))
    X = (X_h[:3] / X_h[3]).reshape(3)
    return X

def project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    """3D 点投影到像素 (2,)"""
    X_h = np.hstack([X, 1.0])
    x = P @ X_h
    return (x[:2] / x[2])

def cheirality_ok(R: np.ndarray, t: np.ndarray, X: np.ndarray) -> bool:
    """正深度检查：点在相机前方（z_cam>0）"""
    X_cam = R @ X + t
    return X_cam[2] > 0

def reproj_error(Ps: List[np.ndarray], X: np.ndarray,
                 xs: List[np.ndarray]) -> float:
    """平均重投影误差（像素）"""
    errs = []
    for P, x in zip(Ps, xs):
        x_hat = project(P, X)
        errs.append(np.linalg.norm(x_hat - x))
    return float(np.mean(errs)) if errs else np.inf

# --------- 主函数 ---------
def triangulate_with_missing(
    observations: Dict[str, np.ndarray],
    # observations[view] = (N,2) 或 (N,3) (第三维可做置信度)，缺失用 np.nan
    Ks: Dict[str, np.ndarray],
    extrinsics: Dict[str, Tuple[np.ndarray, np.ndarray]],  # {view: (R_wc, t_wc)} 使得 X_cam = R_wc X_w + t_wc
    max_err_px: float = 5.0
) -> np.ndarray:
    """
    对每个关键点自动选择可用视点对进行三角测量并返回世界坐标 (N,3)，不可测设为 NaN。
    - observations: 每个视点的关键点像素坐标（缺失为 nan）
    - Ks, extrinsics: 每视点的 K, (R_wc, t_wc)
    - 返回: X_world (N,3)
    """
    # 组装各视点的投影矩阵（世界坐标系）
    Ps = {}
    Rcs = {}
    tcs = {}
    
    for v in observations.keys():
        R_wc, t_wc, C_wc = extrinsics[v].values()           # world -> cam
        K = Ks[v]
        P = build_P(K, R_wc, t_wc)            # x = K [R|t] X_world
        Ps[v] = P
        Rcs[v], tcs[v] = R_wc, t_wc.reshape(3)

    # 关键点数量 N（取第一个视点的形状）
    first_view = next(iter(observations))
    N = observations[first_view].shape[0]

    X_world = np.full((N, 3), np.nan, dtype=np.float64)

    views = list(observations.keys())

    for j in range(N):
        # 收集该关键点在各视点的可用观测
        avail = []
        for v in views:
            pt = observations[v][j]
            if np.all(np.isfinite(pt[:2])):           # 只要 x,y 不是 nan 就算可用
                avail.append(v)

        if len(avail) < 2:
            continue  # 不足两视点，无法三角测量

        # 枚举所有视点对，选择最优
        best = {"err": np.inf, "X": None}
        for i in range(len(avail)):
            for k in range(i+1, len(avail)):
                v1, v2 = avail[i], avail[k]
                x1 = observations[v1][j][:2].astype(float)
                x2 = observations[v2][j][:2].astype(float)

                try: # FIXME: 这里有问题
                    X = triangulate_two_views(Ps[v1], Ps[v2], x1, x2)  # 世界坐标
                except cv2.error:
                    continue

                # 正深度检查（两个相机前方）
                if not (cheirality_ok(Rcs[v1], tcs[v1], X) and
                        cheirality_ok(Rcs[v2], tcs[v2], X)):
                    continue

                # 用所有可用视点计算平均重投影误差
                Ps_all = [Ps[v] for v in avail]
                xs_all = [observations[v][j][:2].astype(float) for v in avail]
                err = reproj_error(Ps_all, X, xs_all)

                if err < best["err"]:
                    best = {"err": err, "X": X}

        # 误差阈值过滤
        if best["X"] is not None and best["err"] <= max_err_px:
            X_world[j] = best["X"]
        # 否则保持 NaN（不可用/不可靠）

    return X_world



# ---------- 三角测量 ----------
def triangulate_joints(keypoints1, keypoints2, K, R, T):
    if keypoints1.shape != keypoints2.shape or keypoints1.shape[1] != 2:
        raise ValueError(
            f"Keypoints shape mismatch: {keypoints1.shape} vs {keypoints2.shape}"
        )

    if keypoints1.dtype == object:
        keypoints1 = np.array([kp for kp in keypoints1], dtype=np.float32)
    if keypoints2.dtype == object:
        keypoints2 = np.array([kp for kp in keypoints2], dtype=np.float32)

    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, T.reshape(3, 1)))
    pts_4d = cv2.triangulatePoints(P1, P2, keypoints1.T, keypoints2.T)
    return (pts_4d[:3, :] / pts_4d[3, :]).T
