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

import cv2

from face_mesh import load_image, predict_face_mesh, draw_face_mesh
from pathlib import Path

# TODO: 处理视频文件
def process(img_path: str):
    # 加载图像
    image_bgr, image_rgb = load_image(img_path)

    # FaceMesh 预测
    landmarks = predict_face_mesh(image_rgb)

    # 绘制 mesh 并显示
    if landmarks:
        output_img = draw_face_mesh(image_bgr.copy(), landmarks)
        cv2.imshow("Face Mesh", output_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No face detected.")


if __name__ == "__main__":

    video_input_dir = Path("/workspace/data/videos")

    for one_person in video_input_dir.iterdir():
        if not one_person.is_dir():
            continue

        for video_dir in one_person.iterdir():
            if not video_dir.is_dir():
                continue

            for video_file in video_dir.glob("*.mp4"):
                print(f"Processing video: {video_file}")
                process(str(video_file))
                print(f"Finished processing video: {video_file}")
