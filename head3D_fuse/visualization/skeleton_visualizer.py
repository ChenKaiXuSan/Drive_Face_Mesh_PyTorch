# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import matplotlib.pyplot as plt


EDGES_FILTERED_IDX = [
    # ====================
    # Head
    # ====================
    (1, 2),  # left_eye - right_eye
    (0, 1),  # nose - left_eye
    (0, 2),  # nose - right_eye
    (1, 3),  # left_eye - left_ear
    (2, 4),  # right_eye - right_ear
    # ====================
    # Neck / Shoulders
    # ====================
    (69, 5),  # neck - left_shoulder
    (69, 6),  # neck - right_shoulder
    (5, 6),  # left_shoulder - right_shoulder
    # ====================
    # Acromion
    # ====================
    (67, 5),  # left_acromion - left_shoulder
    (68, 6),  # right_acromion - right_shoulder
    # ====================
    # Shoulder -> Wrist (bridge)
    # ====================
    (5, 62),  # left_shoulder - left_wrist
    (6, 41),  # right_shoulder - right_wrist
    # ====================
    # Left hand
    # ====================
    (62, 45),
    (45, 44),
    (44, 43),
    (43, 42),  # wrist -> thumb
    (62, 49),
    (49, 48),
    (48, 47),
    (47, 46),  # wrist -> index
    (62, 53),
    (53, 52),
    (52, 51),
    (51, 50),  # wrist -> middle
    (62, 57),
    (57, 56),
    (56, 55),
    (55, 54),  # wrist -> ring
    (62, 61),
    (61, 60),
    (60, 59),
    (59, 58),  # wrist -> pinky
    # ====================
    # Right hand
    # ====================
    (41, 24),
    (24, 23),
    (23, 22),
    (22, 21),  # wrist -> thumb
    (41, 28),
    (28, 27),
    (27, 26),
    (26, 25),  # wrist -> index
    (41, 32),
    (32, 31),
    (31, 30),
    (30, 29),  # wrist -> middle
    (41, 36),
    (36, 35),
    (35, 34),
    (34, 33),  # wrist -> ring
    (41, 40),
    (40, 39),
    (39, 38),
    (38, 37),  # wrist -> pinky
]


class SkeletonVisualizer:
    def __init__(
        self,
        bbox_color: Optional[Union[str, Tuple[int]]] = "green",
        kpt_color: Optional[Union[str, Tuple[Tuple[int]]]] = "red",
        link_color: Optional[Union[str, Tuple[Tuple[int]]]] = None,
        text_color: Optional[Union[str, Tuple[int]]] = (255, 255, 255),
        line_width: Union[int, float] = 1,
        radius: Union[int, float] = 3,
        alpha: float = 1.0,
        show_keypoint_weight: bool = False,
    ):
        self.bbox_color = bbox_color
        self.kpt_color = kpt_color
        self.link_color = link_color
        self.line_width = line_width
        self.text_color = text_color
        self.radius = radius
        self.alpha = alpha
        self.show_keypoint_weight = show_keypoint_weight

    def draw_skeleton(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        kpt_thr: float = 0.3,
        show_kpt_idx: bool = False,
    ):
        """Draw keypoints and skeletons (optional) of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            keypoints (np.ndarray): B x N x 3
            kpt_thr (float, optional): Minimum threshold of keypoints
                to be shown. Default: 0.3.
            show_kpt_idx (bool): Whether to show the index of keypoints.
                Defaults to ``False``

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """

        image = image.copy()
        img_h, img_w, _ = image.shape
        if len(keypoints.shape) == 2:
            keypoints = keypoints[None, :, :]

        # loop for each person
        for cur_keypoints in keypoints:
            kpts = cur_keypoints[:, :-1]
            score = cur_keypoints[:, -1]

            # draw links
            for edge in EDGES_FILTERED_IDX:
                kpt1_idx, kpt2_idx = edge
                if (
                    score[kpt1_idx] < kpt_thr
                    or score[kpt2_idx] < kpt_thr
                    or kpts[kpt1_idx] is None
                    or kpts[kpt2_idx] is None
                ):
                    # skip the edge that should not be drawn
                    continue

                color = self.link_color
                if isinstance(color, (list, tuple)) and len(color) > max(
                    kpt1_idx, kpt2_idx
                ):
                    color = color[kpt1_idx]
                if not isinstance(color, str):
                    color = tuple(int(c) for c in color)

                transparency = self.alpha
                if self.show_keypoint_weight:
                    transparency *= (
                        max(0, min(1, score[kpt1_idx]))
                        + max(0, min(1, score[kpt2_idx]))
                    ) / 2

                if transparency == 1.0:
                    image = cv2.line(
                        image,
                        (int(kpts[kpt1_idx][0]), int(kpts[kpt1_idx][1])),
                        (int(kpts[kpt2_idx][0]), int(kpts[kpt2_idx][1])),
                        color,
                        int(self.line_width),
                    )
                else:
                    temp = image = cv2.line(
                        image.copy(),
                        (int(kpts[kpt1_idx][0]), int(kpts[kpt1_idx][1])),
                        (int(kpts[kpt2_idx][0]), int(kpts[kpt2_idx][1])),
                        color,
                        int(self.line_width),
                    )
                    image = cv2.addWeighted(
                        image, 1 - transparency, temp, transparency, 0
                    )

            # draw each point on image
            for kpt_id, kpt in enumerate(kpts):
                if score[kpt_id] < kpt_thr or kpt is None:
                    continue

                color = self.kpt_color
                if isinstance(color, (list, tuple)) and len(color) > kpt_id:
                    color = color[kpt_id]
                if not isinstance(color, str):
                    color = tuple(int(c) for c in color)

                transparency = self.alpha
                if self.show_keypoint_weight:
                    transparency *= max(0, min(1, score[kpt_id]))

                if transparency == 1.0:
                    image = cv2.circle(
                        image,
                        (int(kpt[0]), int(kpt[1])),
                        int(self.radius),
                        color,
                        -1,
                    )
                else:
                    temp = image.copy()
                    temp = cv2.circle(
                        temp,
                        (int(kpt[0]), int(kpt[1])),
                        int(self.radius),
                        color,
                        -1,
                    )
                    image = cv2.addWeighted(
                        image, 1 - transparency, temp, transparency, 0
                    )

                if show_kpt_idx:
                    cv2.putText(
                        image,
                        str(kpt_id),
                        (int(kpt[0] + 3), int(kpt[1] - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        self.text_color,
                        1,
                    )

        return image

    def draw_3d_skeleton(
        self,
        img_cv2: np.ndarray,
        keypoints_3d: np.ndarray,
    ) -> np.ndarray:
        """
        3D 骨架现场绘制接口（适配 filtered_pred_keypoints_3d + 过滤后 skeleton）
        """
        # 1) 初始化 Matplotlib 3D 画布
        fig = plt.figure(figsize=(10, 10), dpi=100)
        ax = fig.add_subplot(111, projection="3d")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # draw point
        for kpt in keypoints_3d:
            ax.scatter(kpt[0], kpt[1], kpt[2], c="r", marker="o")

        # draw links
        for edge in EDGES_FILTERED_IDX:
            kpt1_idx, kpt2_idx = edge
            kpt1 = keypoints_3d[kpt1_idx]
            kpt2 = keypoints_3d[kpt2_idx]
            xs = [kpt1[0], kpt2[0]]
            ys = [kpt1[1], kpt2[1]]
            zs = [kpt1[2], kpt2[2]]
            ax.plot(xs, ys, zs, c="b")

        # 4) fig -> image
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img_3d = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_3d = img_3d.reshape((h, w, 4))[:, :, :3]

        plt.close(fig)
        return img_3d
