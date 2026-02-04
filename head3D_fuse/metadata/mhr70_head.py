#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/code/head3D_fuse/metadata/mhr70 copy.py
Project: /workspace/code/head3D_fuse/metadata
Created Date: Wednesday February 4th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Wednesday February 4th 2026 2:08:27 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""The first 70 of 308 MHR keypoints, ignoring the rest for face keypoints"""

mhr_names = [
    "nose",
    "left-eye",
    "right-eye",
    "left-ear",
    "right-ear",
    "left-shoulder",
    "right-shoulder",
    "left-elbow",
    "right-elbow",
    "left-hip",
    "right-hip",
    "left-knee",
    "right-knee",
    "left-ankle",
    "right-ankle",
    "left-big-toe-tip",
    "left-small-toe-tip",
    "left-heel",
    "right-big-toe-tip",
    "right-small-toe-tip",
    "right-heel",
    "right-thumb-tip",
    "right-thumb-first-joint",
    "right-thumb-second-joint",
    "right-thumb-third-joint",
    "right-index-tip",
    "right-index-first-joint",
    "right-index-second-joint",
    "right-index-third-joint",
    "right-middle-tip",
    "right-middle-first-joint",
    "right-middle-second-joint",
    "right-middle-third-joint",
    "right-ring-tip",
    "right-ring-first-joint",
    "right-ring-second-joint",
    "right-ring-third-joint",
    "right-pinky-tip",
    "right-pinky-first-joint",
    "right-pinky-second-joint",
    "right-pinky-third-joint",
    "right-wrist",
    "left-thumb-tip",
    "left-thumb-first-joint",
    "left-thumb-second-joint",
    "left-thumb-third-joint",
    "left-index-tip",
    "left-index-first-joint",
    "left-index-second-joint",
    "left-index-third-joint",
    "left-middle-tip",
    "left-middle-first-joint",
    "left-middle-second-joint",
    "left-middle-third-joint",
    "left-ring-tip",
    "left-ring-first-joint",
    "left-ring-second-joint",
    "left-ring-third-joint",
    "left-pinky-tip",
    "left-pinky-first-joint",
    "left-pinky-second-joint",
    "left-pinky-third-joint",
    "left-wrist",
    "left-olecranon",
    "right-olecranon",
    "left-cubital-fossa",
    "right-cubital-fossa",
    "left-acromion",
    "right-acromion",
    "neck",
]

pose_info_head_hands_shoulders = dict(
    pose_format="mhr70",
    paper_info=dict(
        author="",
        year="",
        homepage="",
    ),
    min_visible_keypoints=8,
    image_height=4096,
    image_width=2668,

    # -------------------------
    # Only keep: head + hands + shoulders
    # -------------------------
    original_keypoint_info={
        # head
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        69: "neck",

        # shoulders
        5: "left_shoulder",
        6: "right_shoulder",
        67: "left_acromion",
        68: "right_acromion",

        # right hand (incl. wrist)
        21: "right_thumb_tip",
        22: "right_thumb_first_joint",
        23: "right_thumb_second_joint",
        24: "right_thumb_third_joint",
        25: "right_index_tip",
        26: "right_index_first_joint",
        27: "right_index_second_joint",
        28: "right_index_third_joint",
        29: "right_middle_tip",
        30: "right_middle_first_joint",
        31: "right_middle_second_joint",
        32: "right_middle_third_joint",
        33: "right_ring_tip",
        34: "right_ring_first_joint",
        35: "right_ring_second_joint",
        36: "right_ring_third_joint",
        37: "right_pinky_tip",
        38: "right_pinky_first_joint",
        39: "right_pinky_second_joint",
        40: "right_pinky_third_joint",
        41: "right_wrist",

        # left hand (incl. wrist)
        42: "left_thumb_tip",
        43: "left_thumb_first_joint",
        44: "left_thumb_second_joint",
        45: "left_thumb_third_joint",
        46: "left_index_tip",
        47: "left_index_first_joint",
        48: "left_index_second_joint",
        49: "left_index_third_joint",
        50: "left_middle_tip",
        51: "left_middle_first_joint",
        52: "left_middle_second_joint",
        53: "left_middle_third_joint",
        54: "left_ring_tip",
        55: "left_ring_first_joint",
        56: "left_ring_second_joint",
        57: "left_ring_third_joint",
        58: "left_pinky_tip",
        59: "left_pinky_first_joint",
        60: "left_pinky_second_joint",
        61: "left_pinky_third_joint",
        62: "left_wrist",
    },

    keypoint_info={
        # head
        0: dict(name="nose", id=0, color=[51, 153, 255], type="upper", swap=""),
        1: dict(name="left_eye", id=1, color=[51, 153, 255], type="upper", swap="right_eye"),
        2: dict(name="right_eye", id=2, color=[51, 153, 255], type="upper", swap="left_eye"),
        3: dict(name="left_ear", id=3, color=[51, 153, 255], type="upper", swap="right_ear"),
        4: dict(name="right_ear", id=4, color=[51, 153, 255], type="upper", swap="left_ear"),
        69: dict(name="neck", id=69, color=[51, 153, 255], type="", swap=""),

        # shoulders
        5: dict(name="left_shoulder", id=5, color=[51, 153, 255], type="upper", swap="right_shoulder"),
        6: dict(name="right_shoulder", id=6, color=[51, 153, 255], type="upper", swap="left_shoulder"),
        67: dict(name="left_acromion", id=67, color=[51, 153, 255], type="", swap="right_acromion"),
        68: dict(name="right_acromion", id=68, color=[51, 153, 255], type="", swap="left_acromion"),

        # right hand + wrist
        21: dict(name="right_thumb4", id=21, color=[51, 153, 255], type="upper", swap="left_thumb4"),
        22: dict(name="right_thumb3", id=22, color=[51, 153, 255], type="upper", swap="left_thumb3"),
        23: dict(name="right_thumb2", id=23, color=[51, 153, 255], type="upper", swap="left_thumb2"),
        24: dict(name="right_thumb_third_joint", id=24, color=[51, 153, 255], type="upper", swap="left_thumb_third_joint"),
        25: dict(name="right_forefinger4", id=25, color=[51, 153, 255], type="upper", swap="left_forefinger4"),
        26: dict(name="right_forefinger3", id=26, color=[51, 153, 255], type="upper", swap="left_forefinger3"),
        27: dict(name="right_forefinger2", id=27, color=[51, 153, 255], type="upper", swap="left_forefinger2"),
        28: dict(name="right_forefinger_third_joint", id=28, color=[51, 153, 255], type="upper", swap="left_forefinger_third_joint"),
        29: dict(name="right_middle_finger4", id=29, color=[51, 153, 255], type="upper", swap="left_middle_finger4"),
        30: dict(name="right_middle_finger3", id=30, color=[51, 153, 255], type="upper", swap="left_middle_finger3"),
        31: dict(name="right_middle_finger2", id=31, color=[51, 153, 255], type="upper", swap="left_middle_finger2"),
        32: dict(name="right_middle_finger_third_joint", id=32, color=[51, 153, 255], type="upper", swap="left_middle_finger_third_joint"),
        33: dict(name="right_ring_finger4", id=33, color=[51, 153, 255], type="upper", swap="left_ring_finger4"),
        34: dict(name="right_ring_finger3", id=34, color=[51, 153, 255], type="upper", swap="left_ring_finger3"),
        35: dict(name="right_ring_finger2", id=35, color=[51, 153, 255], type="upper", swap="left_ring_finger2"),
        36: dict(name="right_ring_finger_third_joint", id=36, color=[51, 153, 255], type="upper", swap="left_ring_finger_third_joint"),
        37: dict(name="right_pinky_finger4", id=37, color=[51, 153, 255], type="upper", swap="left_pinky_finger4"),
        38: dict(name="right_pinky_finger3", id=38, color=[51, 153, 255], type="upper", swap="left_pinky_finger3"),
        39: dict(name="right_pinky_finger2", id=39, color=[51, 153, 255], type="upper", swap="left_pinky_finger2"),
        40: dict(name="right_pinky_finger_third_joint", id=40, color=[51, 153, 255], type="upper", swap="left_pinky_finger_third_joint"),
        41: dict(name="right_wrist", id=41, color=[51, 153, 255], type="upper", swap="left_wrist"),

        # left hand + wrist
        42: dict(name="left_thumb4", id=42, color=[51, 153, 255], type="upper", swap="right_thumb4"),
        43: dict(name="left_thumb3", id=43, color=[51, 153, 255], type="upper", swap="right_thumb3"),
        44: dict(name="left_thumb2", id=44, color=[51, 153, 255], type="upper", swap="right_thumb2"),
        45: dict(name="left_thumb_third_joint", id=45, color=[51, 153, 255], type="upper", swap="right_thumb_third_joint"),
        46: dict(name="left_forefinger4", id=46, color=[51, 153, 255], type="upper", swap="right_forefinger4"),
        47: dict(name="left_forefinger3", id=47, color=[51, 153, 255], type="upper", swap="right_forefinger3"),
        48: dict(name="left_forefinger2", id=48, color=[51, 153, 255], type="upper", swap="right_forefinger2"),
        49: dict(name="left_forefinger_third_joint", id=49, color=[51, 153, 255], type="upper", swap="right_forefinger_third_joint"),
        50: dict(name="left_middle_finger4", id=50, color=[51, 153, 255], type="upper", swap="right_middle_finger4"),
        51: dict(name="left_middle_finger3", id=51, color=[51, 153, 255], type="upper", swap="right_middle_finger3"),
        52: dict(name="left_middle_finger2", id=52, color=[51, 153, 255], type="upper", swap="right_middle_finger2"),
        53: dict(name="left_middle_finger_third_joint", id=53, color=[51, 153, 255], type="upper", swap="right_middle_finger_third_joint"),
        54: dict(name="left_ring_finger4", id=54, color=[51, 153, 255], type="upper", swap="right_ring_finger4"),
        55: dict(name="left_ring_finger3", id=55, color=[51, 153, 255], type="upper", swap="right_ring_finger3"),
        56: dict(name="left_ring_finger2", id=56, color=[51, 153, 255], type="upper", swap="right_ring_finger2"),
        57: dict(name="left_ring_finger_third_joint", id=57, color=[51, 153, 255], type="upper", swap="right_ring_finger_third_joint"),
        58: dict(name="left_pinky_finger4", id=58, color=[51, 153, 255], type="upper", swap="right_pinky_finger4"),
        59: dict(name="left_pinky_finger3", id=59, color=[51, 153, 255], type="upper", swap="right_pinky_finger3"),
        60: dict(name="left_pinky_finger2", id=60, color=[51, 153, 255], type="upper", swap="right_pinky_finger2"),
        61: dict(name="left_pinky_finger_third_joint", id=61, color=[51, 153, 255], type="upper", swap="right_pinky_finger_third_joint"),
        62: dict(name="left_wrist", id=62, color=[51, 153, 255], type="upper", swap="right_wrist"),
    },

    skeleton_info={
    # head
    0: dict(link=("left_eye", "right_eye"), id=0, color=[51, 153, 255]),
    1: dict(link=("nose", "left_eye"), id=1, color=[51, 153, 255]),
    2: dict(link=("nose", "right_eye"), id=2, color=[51, 153, 255]),
    3: dict(link=("left_eye", "left_ear"), id=3, color=[51, 153, 255]),
    4: dict(link=("right_eye", "right_ear"), id=4, color=[51, 153, 255]),

    # shoulders + head->shoulder
    5: dict(link=("left_shoulder", "right_shoulder"), id=5, color=[51, 153, 255]),
    6: dict(link=("left_ear", "left_shoulder"), id=6, color=[51, 153, 255]),
    7: dict(link=("right_ear", "right_shoulder"), id=7, color=[51, 153, 255]),

    # neck (recommended)
    8: dict(link=("neck", "left_shoulder"), id=8, color=[51, 153, 255]),
    9: dict(link=("neck", "right_shoulder"), id=9, color=[51, 153, 255]),

    # acromion (recommended)
    10: dict(link=("left_acromion", "left_shoulder"), id=10, color=[51, 153, 255]),
    11: dict(link=("right_acromion", "right_shoulder"), id=11, color=[51, 153, 255]),

    # simplified shoulder->wrist bridge (optional)
    12: dict(link=("left_shoulder", "left_wrist"), id=12, color=[51, 153, 255]),
    13: dict(link=("right_shoulder", "right_wrist"), id=13, color=[51, 153, 255]),

    # left hand
    14: dict(link=("left_wrist", "left_thumb_third_joint"), id=14, color=[255, 128, 0]),
    15: dict(link=("left_thumb_third_joint", "left_thumb2"), id=15, color=[255, 128, 0]),
    16: dict(link=("left_thumb2", "left_thumb3"), id=16, color=[255, 128, 0]),
    17: dict(link=("left_thumb3", "left_thumb4"), id=17, color=[255, 128, 0]),

    18: dict(link=("left_wrist", "left_forefinger_third_joint"), id=18, color=[255, 153, 255]),
    19: dict(link=("left_forefinger_third_joint", "left_forefinger2"), id=19, color=[255, 153, 255]),
    20: dict(link=("left_forefinger2", "left_forefinger3"), id=20, color=[255, 153, 255]),
    21: dict(link=("left_forefinger3", "left_forefinger4"), id=21, color=[255, 153, 255]),

    22: dict(link=("left_wrist", "left_middle_finger_third_joint"), id=22, color=[102, 178, 255]),
    23: dict(link=("left_middle_finger_third_joint", "left_middle_finger2"), id=23, color=[102, 178, 255]),
    24: dict(link=("left_middle_finger2", "left_middle_finger3"), id=24, color=[102, 178, 255]),
    25: dict(link=("left_middle_finger3", "left_middle_finger4"), id=25, color=[102, 178, 255]),

    26: dict(link=("left_wrist", "left_ring_finger_third_joint"), id=26, color=[255, 51, 51]),
    27: dict(link=("left_ring_finger_third_joint", "left_ring_finger2"), id=27, color=[255, 51, 51]),
    28: dict(link=("left_ring_finger2", "left_ring_finger3"), id=28, color=[255, 51, 51]),
    29: dict(link=("left_ring_finger3", "left_ring_finger4"), id=29, color=[255, 51, 51]),

    30: dict(link=("left_wrist", "left_pinky_finger_third_joint"), id=30, color=[0, 255, 0]),
    31: dict(link=("left_pinky_finger_third_joint", "left_pinky_finger2"), id=31, color=[0, 255, 0]),
    32: dict(link=("left_pinky_finger2", "left_pinky_finger3"), id=32, color=[0, 255, 0]),
    33: dict(link=("left_pinky_finger3", "left_pinky_finger4"), id=33, color=[0, 255, 0]),

    # right hand
    34: dict(link=("right_wrist", "right_thumb_third_joint"), id=34, color=[255, 128, 0]),
    35: dict(link=("right_thumb_third_joint", "right_thumb2"), id=35, color=[255, 128, 0]),
    36: dict(link=("right_thumb2", "right_thumb3"), id=36, color=[255, 128, 0]),
    37: dict(link=("right_thumb3", "right_thumb4"), id=37, color=[255, 128, 0]),

    38: dict(link=("right_wrist", "right_forefinger_third_joint"), id=38, color=[255, 153, 255]),
    39: dict(link=("right_forefinger_third_joint", "right_forefinger2"), id=39, color=[255, 153, 255]),
    40: dict(link=("right_forefinger2", "right_forefinger3"), id=40, color=[255, 153, 255]),
    41: dict(link=("right_forefinger3", "right_forefinger4"), id=41, color=[255, 153, 255]),

    42: dict(link=("right_wrist", "right_middle_finger_third_joint"), id=42, color=[102, 178, 255]),
    43: dict(link=("right_middle_finger_third_joint", "right_middle_finger2"), id=43, color=[102, 178, 255]),
    44: dict(link=("right_middle_finger2", "right_middle_finger3"), id=44, color=[102, 178, 255]),
    45: dict(link=("right_middle_finger3", "right_middle_finger4"), id=45, color=[102, 178, 255]),

    46: dict(link=("right_wrist", "right_ring_finger_third_joint"), id=46, color=[255, 51, 51]),
    47: dict(link=("right_ring_finger_third_joint", "right_ring_finger2"), id=47, color=[255, 51, 51]),
    48: dict(link=("right_ring_finger2", "right_ring_finger3"), id=48, color=[255, 51, 51]),
    49: dict(link=("right_ring_finger3", "right_ring_finger4"), id=49, color=[255, 51, 51]),

    50: dict(link=("right_wrist", "right_pinky_finger_third_joint"), id=50, color=[0, 255, 0]),
    51: dict(link=("right_pinky_finger_third_joint", "right_pinky_finger2"), id=51, color=[0, 255, 0]),
    52: dict(link=("right_pinky_finger2", "right_pinky_finger3"), id=52, color=[0, 255, 0]),
    53: dict(link=("right_pinky_finger3", "right_pinky_finger4"), id=53, color=[0, 255, 0]),
    }.


    # 仍然给 70 长度，保持外部代码兼容（如 loss 里直接按 70 维索引）
    joint_weights=[1.0] * 70,

    # 只保留 head + shoulders + wrists（你说要肩膀&head，这里给一个“主体点”列表）
    body_keypoint_names=[
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "neck",
        "left_shoulder",
        "right_shoulder",
        "left_wrist",
        "right_wrist",
    ],

    # 去掉脚
    foot_keypoint_names=[],

    # hands（与你原来的命名一致）
    left_hand_keypoint_names=[
        "left_thumb4",
        "left_thumb3",
        "left_thumb2",
        "left_thumb_third_joint",
        "left_forefinger4",
        "left_forefinger3",
        "left_forefinger2",
        "left_forefinger_third_joint",
        "left_middle_finger4",
        "left_middle_finger3",
        "left_middle_finger2",
        "left_middle_finger_third_joint",
        "left_ring_finger4",
        "left_ring_finger3",
        "left_ring_finger2",
        "left_ring_finger_third_joint",
        "left_pinky_finger4",
        "left_pinky_finger3",
        "left_pinky_finger2",
        "left_pinky_finger_third_joint",
    ],
    right_hand_keypoint_names=[
        "right_thumb4",
        "right_thumb3",
        "right_thumb2",
        "right_thumb_third_joint",
        "right_forefinger4",
        "right_forefinger3",
        "right_forefinger2",
        "right_forefinger_third_joint",
        "right_middle_finger4",
        "right_middle_finger3",
        "right_middle_finger2",
        "right_middle_finger_third_joint",
        "right_ring_finger4",
        "right_ring_finger3",
        "right_ring_finger2",
        "right_ring_finger_third_joint",
        "right_pinky_finger4",
        "right_pinky_finger3",
        "right_pinky_finger2",
        "right_pinky_finger_third_joint",
    ],

    # 额外点：只留 acromion（你想保留肩膀相关）
    extra_keypoint_names=[
        "left_acromion",
        "right_acromion",
    ],

    sigmas=[],
)
