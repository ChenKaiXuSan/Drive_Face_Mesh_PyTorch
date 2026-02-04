#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/triangulation/vis/merge_video.py
Project: /workspace/code/triangulation/vis
Created Date: Tuesday October 14th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Tuesday October 14th 2025 12:53:44 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import logging
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


def merge_frames_to_video(
    frame_dir: Path, output_video_path: Path, fps: int = 30
) -> None:
    """
    将指定目录下的图像帧合并为视频。
    frame_dir: 图像帧目录，假设命名格式为 frame_0000.jpg, frame_0001.jpg, ...
    output_video_path: 输出视频路径
    fps: 视频帧率
    """

    if frame_dir.exists() is False:
        raise ValueError(f"Frame directory does not exist: {frame_dir}")
    if not output_video_path.parent.exists():
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(frame_dir.glob("*.png"))

    if not frame_files:
        raise ValueError(f"No frames found in directory: {frame_dir}")

    # 读取第一张图片以获取尺寸信息
    first_frame = cv2.imread(str(frame_files[0]))
    height, width, _ = first_frame.shape

    # 定义视频编码器和输出文件
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 使用 mp4v 编码
    video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(str(frame_file))
        if frame is None:
            logger.warning(f"Could not read frame {frame_file}, skipping.")
            continue
        video_writer.write(frame)

    video_writer.release()
    logger.info(f"Video saved to {output_video_path}")
