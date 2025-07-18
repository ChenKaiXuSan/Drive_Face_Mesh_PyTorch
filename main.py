#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/main.py
Project: /workspace/code
Created Date: Thursday July 17th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday July 17th 2025 5:03:00 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from face_mesh import process_video_with_face_mesh, process_image_with_face_mesh
from pathlib import Path


# TODO: 处理视频文件
def process(path, result_path=None):

    if path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        process_image_with_face_mesh(path, show=False, output_path=result_path)
    elif path.suffix.lower() in [".mp4"]:
        process_video_with_face_mesh(path, show=False, output_path=result_path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


if __name__ == "__main__":

    video_input_dir = Path("/workspace/data/videos")
    image_input_dir = Path("/workspace/data/image")
    result_dir = Path("/workspace/data/result")

    for one_person in video_input_dir.iterdir():
        if not one_person.is_dir():
            continue

        for video_dir in one_person.iterdir():
            if not video_dir.is_dir():
                continue

            for video_file in video_dir.glob("*.mp4"):
                _res_path = result_dir / one_person.name / video_dir.name

                print(f"Processing video: {video_file}")
                process(path=video_file, result_path=_res_path / video_file.name)
                print(f"Finished processing video: {video_file}")
