#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/SAM3Dbody/main_multi_gpu_process.py
Project: /workspace/code/SAM3Dbody
Created Date: Monday January 26th 2026
Author: Kaixu Chen
-----
Comment:
根据多GPU并行处理SAM-3D-Body推理任务。

Have a good code time :)
-----
Last Modified: Monday January 26th 2026 5:12:10 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

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

# --- 常量定义 ---
REQUIRED_VIEWS = {"front", "left", "right"}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# 核心处理逻辑：处理单个人的数据
# ---------------------------------------------------------------------
def process_single_person_env(
    person_env_dir: Path,
    source_root: Path,
    out_root: Path,
    infer_root: Path,
    cfg: DictConfig,
):
    """处理单个人员的所有环境和视角"""
    person_id = person_env_dir.parent.name
    env_name = person_env_dir.name
    vid_patterns = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.MP4", "*.MOV"]

    # --- 1. Person専用のログ設定 ---
    log_dir = out_root / "person_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    person_log_file = log_dir / f"{person_id}_{env_name}.log"

    # 新しいハンドラを作成
    handler = logging.FileHandler(person_log_file, mode="a", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    logger = logging.getLogger(f"{person_id}_{env_name}")  # このPerson専用のロガーを取得
    logger.addHandler(handler)
    # logger.propagate = False  # 親（Root）ロガーにログを流さない（混ざるのを防ぐ）

    logger.info(f"==== Starting Process for Person: {person_id}, Env: {env_name} ====")
    rel_env = person_env_dir.relative_to(source_root)

    # --- 视频处理逻辑 ---
    view_map: Dict[str, Path] = {}
    for pat in vid_patterns:
        for f in person_env_dir.glob(pat):
            stem = f.stem.lower()
            if stem in REQUIRED_VIEWS:
                view_map[stem] = f.resolve()

    if not all(v in view_map for v in REQUIRED_VIEWS):
        logger.warning(f"[Skip] {rel_env}: 视角不全 {list(view_map.keys())}")

    view_frames: Dict[str, List[np.ndarray]] = load_data(view_map)

    for view, frames in view_frames.items():
        logger.info(f"  视角 {view} 处理了 {len(frames)} 帧数据。")
        _out_root = out_root / rel_env / view
        _out_root.mkdir(parents=True, exist_ok=True)
        _infer_root = infer_root / rel_env / view
        _infer_root.mkdir(parents=True, exist_ok=True)

        process_frame_list(
            frame_list=frames,
            out_dir=_out_root,
            inference_output_path=_infer_root,
            cfg=cfg,
        )


# ---------------------------------------------------------------------
# GPU Worker：进程执行函数
# ---------------------------------------------------------------------
def gpu_worker(
    gpu_id: int,
    env_dirs: List[Path],
    source_root: Path,
    out_root: Path,
    infer_root: Path,
    cfg_dict: dict,
):
    """
    每个进程的入口：设置环境变量，并处理分配的任务列表
    """
    # 1. 隔离 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cfg_dict["infer"]["gpu"] = 0  # 因为上面已经隔离了 GPU，所以这里设为 0

    # 2. 将字典转回 Hydra 配置（多进程传递对象时，转为字典更安全）
    cfg = OmegaConf.create(cfg_dict)

    logger.info(f"🟢 GPU {gpu_id} 进程启动，待处理任务数: {len(env_dirs)}")

    for env_dir in env_dirs:
        try:
            process_single_person_env(env_dir, source_root, out_root, infer_root, cfg)
        except Exception as e:
            logger.error(f"❌ GPU {gpu_id} 处理 {env_dir.name} 时出错: {e}")

    logger.info(f"🏁 GPU {gpu_id} 所有任务处理完毕")


# ---------------------------------------------------------------------
# Main 入口
# ---------------------------------------------------------------------
@hydra.main(config_path="../configs", config_name="sam3d_body", version_base=None)
def main(cfg: DictConfig) -> None:
    # 1. 経路準備
    out_root = Path(cfg.paths.log_path).resolve()
    infer_root = Path(cfg.paths.result_output_path).resolve()
    source_root = Path(cfg.paths.video_path).resolve()

    # --- 1. 制御パラメータの取得 ---
    # cfg.infer.person_list: [1, 2, 3] または [-1]
    # cfg.infer.env_list: ["room1", "outdoor"] または ["all"]
    target_person_ids = [int(pid) for pid in cfg.infer.get("person_list", [-1])]
    target_envs = cfg.infer.get("env_list", ["all"])

    # --- 2. 条件に合致する Person/Env タスクを収集 ---
    all_env_tasks = []

    # Person フォルダをループ
    for p_dir in sorted(source_root.iterdir()):
        if not p_dir.is_dir():
            continue

        # Person ID によるフィルタリング
        current_p_id = int(p_dir.name)
        if current_p_id in target_person_ids or -1 in target_person_ids:

            # Env フォルダをループ
            for env_dir in sorted(p_dir.iterdir()):
                if not env_dir.is_dir():
                    continue

                # Env 名によるフィルタリング
                current_env_name = env_dir.name
                if current_env_name in target_envs or "all" in target_envs:
                    all_env_tasks.append(env_dir)

    if not all_env_tasks:
        logger.error(
            f"条件に合うタスクが見つかりませんでした。 (Person: {target_person_ids}, Env: {target_envs})"
        )
        return

    # --- 3. 並列処理の実行 ---
    gpu_ids = cfg.infer.get("gpu", [0, 1])
    workers_per_gpu = cfg.infer.get("workers_per_gpu", 2)

    total_workers = len(gpu_ids) * workers_per_gpu
    chunks = np.array_split(all_env_tasks, total_workers)

    logger.info(f"使用 GPU: {gpu_ids} (各 {workers_per_gpu} ワーカー)")
    logger.info(f"総プロセス数: {total_workers}")
    logger.info(f"総処理人数: {len(target_person_ids)}")

    # 3. 启动并行进程
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    mp.set_start_method("spawn", force=True)

    processes = []
    for i, gpu_id in enumerate(np.repeat(gpu_ids, workers_per_gpu)):
        env_list = chunks[i].tolist()
        if not env_list:
            continue

        logger.info(f"  - Worker {i} (GPU {gpu_id}) 分配任务数: {len(env_list)}")

        p = mp.Process(
            target=gpu_worker,
            args=(
                gpu_id,
                env_list,
                source_root,
                out_root,
                infer_root,
                cfg_dict,
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
