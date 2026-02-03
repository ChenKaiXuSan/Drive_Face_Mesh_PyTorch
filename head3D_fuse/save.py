#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/triangulation/save.py
Project: /workspace/code/triangulation
Created Date: Friday October 10th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday October 10th 2025 10:58:39 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""
from typing import Dict
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)


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
    logger.info(f"Fused keypoints saved → {save_path}")
