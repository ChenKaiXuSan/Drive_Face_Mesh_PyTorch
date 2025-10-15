#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/mesh_triangulation/load.py
Project: /workspace/code/mesh_triangulation
Created Date: Wednesday October 15th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Wednesday October 15th 2025 2:42:03 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""
import logging

logger = logging.getLogger(__name__)
import numpy as np
from torchvision.io import read_video
from pathlib import Path


def load_mesh_from_npz(file_info: dict[str, Path]):

    raw_frame: dict[int, np.ndarray] = {}
    mesh: dict[int, np.ndarray] = {}

    npz_path = file_info["npz"]
    video_path = file_info["video"]

    data = np.load(npz_path, allow_pickle=True)

    # unpack the npz
    for frame_num, info in data.items():

        _one_frame_info = info.item()

        raw_frame[frame_num] = _one_frame_info["raw_frame"]
        mesh[frame_num] = (
            _one_frame_info["mesh"]
            if _one_frame_info["mesh"] is not None
            else np.zeros((1, 478, 3))
        )
        video_info = _one_frame_info["video_info"]
        video_path = _one_frame_info["video_path"]

    logger.info(f"Loaded npz from {npz_path}, total {len(data)} frames.")

    # convert to np
    raw_frame = np.stack([raw_frame[i] for i in sorted(raw_frame.keys())], axis=0)
    mesh = np.stack([mesh[i] for i in sorted(mesh.keys())], axis=0).squeeze()

    # shape check
    assert raw_frame.ndim == 4 and raw_frame.shape[3] in [3, 4]
    assert mesh.ndim == 3 and mesh.shape[1] == 478 and mesh.shape[2] == 3
    assert raw_frame.shape[0] == mesh.shape[0]
    assert video_info is not None and video_path is not None

    return raw_frame, mesh, video_info
