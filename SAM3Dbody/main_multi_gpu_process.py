import logging
import os
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List
import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf

# 假设这些是从你的其他模块导入的
from .infer import process_frame_list
from .load import load_data

logger = logging.getLogger(__name__)

# --- 常量定义 ---
VALID_VIEWS = {"front", "left", "right"}
REQUIRED_VIEWS = {"front", "left", "right"}


# ---------------------------------------------------------------------
# 核心处理逻辑：处理单个人的数据
# ---------------------------------------------------------------------
def process_single_person(
    person_dir: Path,
    source_root: Path,
    out_root: Path,
    infer_root: Path,
    cfg: DictConfig,
    infer_type: str,
):
    """处理单个人员的所有环境和视角"""
    person_id = person_dir.name
    vid_patterns = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.MP4", "*.MOV"]
    img_patterns = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]

    env_dirs = sorted([x for x in person_dir.iterdir() if x.is_dir()])
    if not env_dirs:
        logger.warning(f"跳过：{person_dir} 下没有环境目录")
        return

    for env_dir in env_dirs:
        env_name = env_dir.name
        rel_env = env_dir.relative_to(source_root)

        # --- 视频处理逻辑 ---
        if infer_type == "video":
            view_map = {}
            for pat in vid_patterns:
                for f in env_dir.glob(pat):
                    stem = f.stem.lower()
                    if stem in VALID_VIEWS:
                        view_map[stem] = f.resolve()

            if not all(v in view_map for v in REQUIRED_VIEWS):
                logger.warning(f"[Skip] {rel_env}: 视角不全 {list(view_map.keys())}")
                continue

            file_list = [view_map[v] for v in ("front", "left", "right")]
            frame_list = load_data(file_list)

            _execute_inference(
                frame_list, out_root / rel_env, infer_root / rel_env, cfg
            )

        # --- 图像处理逻辑 ---
        else:
            for view_name in REQUIRED_VIEWS:
                view_dir = env_dir / view_name
                if not view_dir.exists():
                    continue

                files = []
                for pat in img_patterns:
                    files.extend(view_dir.glob(pat))
                files = sorted(list(set(files)))

                if not files:
                    continue

                rel_view = view_dir.relative_to(source_root)
                frame_list = load_data(files)

                _execute_inference(
                    frame_list, out_root / rel_view, infer_root / rel_view, cfg
                )


def _execute_inference(frame_list, out_dir, infer_dir, cfg):
    """执行推断的辅助函数"""
    out_dir.mkdir(parents=True, exist_ok=True)
    infer_dir.mkdir(parents=True, exist_ok=True)
    process_frame_list(
        frame_list=frame_list,
        out_dir=out_dir,
        inference_output_path=infer_dir,
        cfg=cfg,
    )


# ---------------------------------------------------------------------
# GPU Worker：进程执行函数
# ---------------------------------------------------------------------
def gpu_worker(
    gpu_id: int,
    person_dirs: List[Path],
    source_root: Path,
    out_root: Path,
    infer_root: Path,
    cfg_dict: dict,
    infer_type: str,
):
    """
    每个进程的入口：设置环境变量，并处理分配的任务列表
    """
    # 1. 隔离 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 2. 将字典转回 Hydra 配置（多进程传递对象时，转为字典更安全）
    cfg = OmegaConf.create(cfg_dict)

    logger.info(f"🟢 GPU {gpu_id} 进程启动，待处理人数: {len(person_dirs)}")

    for p_dir in person_dirs:
        try:
            process_single_person(
                p_dir, source_root, out_root, infer_root, cfg, infer_type
            )
        except Exception as e:
            logger.error(f"❌ GPU {gpu_id} 处理 {p_dir.name} 时出错: {e}")

    logger.info(f"🏁 GPU {gpu_id} 所有任务处理完毕")


# ---------------------------------------------------------------------
# Main 入口
# ---------------------------------------------------------------------
@hydra.main(config_path="../configs", config_name="sam3d_body", version_base=None)
def main(cfg: DictConfig) -> None:
    # 1. 路径准备
    infer_type = cfg.infer.get("type", "video")
    out_root = Path(cfg.paths.log_path).resolve()
    infer_root = Path(cfg.paths.result_output_path).resolve()
    source_root = Path(
        cfg.paths.video_path if infer_type == "video" else cfg.paths.image_path
    ).resolve()

    gpu_ids = cfg.infer.get("gpu_ids", [0, 1])  # 从配置文件读取 GPU 列表，默认 [0, 1]

    all_person_dirs = sorted([x for x in source_root.iterdir() if x.is_dir()])
    if not all_person_dirs:
        logger.error(f"未找到数据目录: {source_root}")
        return

    # 2. 自动分组逻辑 (Task Chunking)
    # 将所有目录分成 N 份，N 等于 GPU 的数量
    num_gpus = len(gpu_ids)
    # 使用 np.array_split 可以确保即使除不尽，分配也尽可能均匀
    chunks = np.array_split(all_person_dirs, num_gpus)

    logger.info(f"检测到 {num_gpus} 个 GPU: {gpu_ids}")
    for i, gpu_id in enumerate(gpu_ids):
        logger.info(f"  - GPU {gpu_id} 分配任务数: {len(chunks[i])}")

    # 3. 启动并行进程
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    mp.set_start_method("spawn", force=True)

    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        person_list = chunks[i].tolist()  # 转回普通列表
        if not person_list:
            continue

        p = mp.Process(
            target=gpu_worker,
            args=(
                gpu_id,
                person_list,
                source_root,
                out_root,
                infer_root,
                cfg_dict,
                infer_type,
            ),
        )
        p.start()
        processes.append(p)

    # 4. 等待所有进程完成
    for p in processes:
        p.join()

    logger.info("🎉 [SUCCESS] 所有 GPU 任务已圆满完成！")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
