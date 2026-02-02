import dataclasses
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PersonInfo:
    person_id: str  # idは '01' などの文字列である可能性が高いためstrに変更
    env_id: str
    front_npz_paths: List[Path] = dataclasses.field(default_factory=list)
    left_npz_paths: List[Path] = dataclasses.field(default_factory=list)
    right_npz_paths: List[Path] = dataclasses.field(default_factory=list)


# TODO: 这里先整理三个视角的npz文件
# TODO： 后续根据annotation_dict进行裁剪，并确认
def load_SAM3D_results_from_npz(
    person_env_dir: Path, view_list: List[str], annotation_dict: Optional[dict] = None
) -> Dict[str, List[np.ndarray]]:
    """
    各視点のディレクトリからnpzファイルを検索し、データをリストとして読み込みます。
    """
    view_frames: Dict[str, List[np.ndarray]] = {}

    for view in view_list:
        view_path = person_env_dir / view

        # globを直接リスト化し、ファイルが存在するか確認
        npz_files = list(view_path.glob("*.npz"))

        if not npz_files:
            logger.warning(f"No npz files found in: {view_path}")
            view_frames[view] = []
            continue

        # 読み込み処理の実行（最新のファイルを1つ読み込む、または全読み込み）
        frames_in_view = []
        for npz_file in npz_files:
            frames_in_view.append(npz_file)

        view_frames[view] = sorted(frames_in_view)

    #TODO：
    # 根据annotation_dict进行裁剪（如果提供了的话）
    if annotation_dict:
        person_id = person_env_dir.parent.name
        env_id = person_env_dir.name

        if (
            person_id in annotation_dict
            and env_id in annotation_dict[person_id]
        ):
            frames_info = annotation_dict[person_id][env_id]
            start_frame = frames_info.get("start")
            end_frame = frames_info.get("end")

            for view in view_list:
                if view in view_frames:
                    # npzファイル名からフレーム番号を抽出してフィルタリング
                    filtered_files = []
                    for npz_path in view_frames[view]:
                        frame_num_str = npz_path.stem.split("_")[-1]
                        try:
                            frame_num = int(frame_num_str)
                            if (
                                (start_frame is None or frame_num >= start_frame)
                                and (end_frame is None or frame_num <= end_frame)
                            ):
                                filtered_files.append(npz_path)
                        except ValueError:
                            logger.warning(f"Invalid frame number in file: {npz_path}")
                    view_frames[view] = filtered_files

    return PersonInfo(
        person_id=person_env_dir.parent.name,
        env_id=person_env_dir.name,
        front_npz_paths=view_frames.get("front", []),
        left_npz_paths=view_frames.get("left", []),
        right_npz_paths=view_frames.get("right", []),
    )


MAPPING = {
    "night_high": "夜多い",
    "night_low": "夜少ない",
    "day_high": "昼多い",
    "day_low": "昼少ない",
}


def get_annotation_dict(file_path: str) -> dict:
    """
    JSONファイルからアノテーション情報を抽出し、
    { person_id: { env_name: { label: frame_num } } } の形式で返します。
    """
    path = Path(file_path)
    if not path.exists():
        logger.error(f"Annotation file not found: {file_path}")
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    master_dict = {}

    for item in data:
        video_path = item.get("video", "")
        file_name = os.path.basename(video_path)
        parts = file_name.split("_")

        # インデックスエラーを防ぐためのガード
        if len(parts) < 4:
            continue

        person = f"{parts[0]}_{parts[1]}"
        env_key = f"{parts[2]}_{parts[3]}"
        env_name = MAPPING.get(env_key, env_key)

        # 辞書の初期化を setdefault で簡潔に
        person_entry = master_dict.setdefault(person, {})

        # 必要なラベル情報を抽出
        frames = {"start": None, "mid": None, "end": None}
        for label_obj in item.get("videoLabels", []):
            labels = label_obj.get("timelinelabels", [])
            if not labels:
                continue

            label_name = labels[0]
            if label_name in frames:
                # ranges[0]['start'] が存在するか安全に取得
                ranges = label_obj.get("ranges", [{}])
                frames[label_name] = ranges[0].get("start")

        person_entry[env_name] = frames

    return master_dict
