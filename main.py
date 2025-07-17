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

# face_mesh_utils.py

import cv2
import mediapipe as mp


# 初始化 mediapipe 模块（只初始化一次）
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh_model = mp_face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1, refine_landmarks=True
)


def load_image(img_path: str):
    """读取并转换为 RGB 图像"""
    bgr_img = cv2.imread(img_path)
    if bgr_img is None:
        raise FileNotFoundError(f"Cannot load image: {img_path}")
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return bgr_img, rgb_img


def predict_face_mesh(image_rgb):
    """执行 FaceMesh 预测"""
    results = face_mesh_model.process(image_rgb)
    return results.multi_face_landmarks


def draw_face_mesh(image_bgr, face_landmarks):
    """在图像上绘制 mesh（contours + tessellation）"""
    for landmarks in face_landmarks:
        mp_drawing.draw_landmarks(
            image=image_bgr,
            landmark_list=landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        mp_drawing.draw_landmarks(
            image=image_bgr,
            landmark_list=landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
        )
    return image_bgr


if __name__ == "__main__":
    from pathlib import Path

    input_dir = Path("/workspace/data/splits")

    for one_person in input_dir.iterdir():
        if not one_person.is_dir():
            continue

        for video_dir in one_person.iterdir():
            if not video_dir.is_dir():
                continue

            for img_file in video_dir.glob("*.png"):
                print(f"[INFO] Processing image: {img_file}")
                img_path = str(img_file)

                try:
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
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    

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
