#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Split all video frames into four quadrants and save them in separate folders.
"""

from pathlib import Path
from tqdm import tqdm
import cv2


def split_and_save_frame(img, save_root: Path, frame_idx: int):
    height, width = img.shape[:2]
    mid_x, mid_y = width // 2, height // 2

    # 划分四个区域
    regions = {
        "left": img[0:mid_y, 0:mid_x],
        "right": img[0:mid_y, mid_x:width],
        "dive_view": img[mid_y:height, 0:mid_x],
        "front": img[mid_y:height, mid_x:width],
    }

    for name, crop in regions.items():
        region_dir = save_root / name
        region_dir.mkdir(parents=True, exist_ok=True)
        save_path = region_dir / f"{frame_idx:04d}.png"
        cv2.imwrite(str(save_path), crop)
        print(f"[INFO] Saved: {save_path}")


def process_video_all_frames(video_path: Path, save_root: Path):
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    success, frame = cap.read()

    while success:

        # if frame_idx % 30 != 0:
        #     break

        # crop frame
        frame = frame[:-30, 47:-10]
        split_and_save_frame(frame, save_root, frame_idx)
        frame_idx += 1
        success, frame = cap.read()

    cap.release()


def save_frame_to_video(image_path: Path, output_path: Path):

    for view in image_path.iterdir():
        if not view.is_dir():
            continue

        frames = []
        sorted_frames = sorted(view.glob("*.png"), key=lambda x: int(x.stem))

        for img_file in sorted_frames:
            img = cv2.imread(str(img_file))
            if img is not None:
                frames.append(img)

        # 获取第一帧的尺寸
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        _output_path = output_path / f"{view.name}.mp4"
        if not _output_path.parent.exists():
            _output_path.parent.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(str(_output_path), fourcc, 30.0, (width, height))

        for frame in frames:
            out.write(frame)

        out.release()
        print(f"[INFO] Video saved to: {_output_path}")


def process_videos(data_root: Path, image_output_path: Path, video_output_path: Path):

    for person_dir in tqdm(
        data_root.iterdir(), desc="Processing person directories", ncols=100
    ):
        if person_dir.is_dir():
            for video_file in tqdm(person_dir.iterdir(), desc="Processing video files", ncols=100):

                if video_file.stem.startswith("._"):
                    continue

                if video_file.is_file() and video_file.suffix == ".mpg":
                    print(f"[INFO] Processing: {video_file}")
                    # 输出根目录按视频名命名
                    image_output_root = (
                        image_output_path / person_dir.stem / video_file.stem
                    )
                    process_video_all_frames(video_file, image_output_root)

                    # save frame to video
                    print(f"[INFO] Finished processing: {video_file}")
                    video_output_root = (
                        video_output_path / person_dir.stem / video_file.stem
                    )
                    save_frame_to_video(image_output_root, video_output_root)


if __name__ == "__main__":
    # TODO: 不通过图片中转，直接从视频划分并保存视频，这样节省存储空间
    root_path = Path("/workspace/data/raw")
    img_output_path = Path("/workspace/data/image")
    video_output_path = Path("/workspace/data/videos")

    img_output_path.mkdir(parents=True, exist_ok=True)
    video_output_path.mkdir(parents=True, exist_ok=True)

    process_videos(root_path, img_output_path, video_output_path)
