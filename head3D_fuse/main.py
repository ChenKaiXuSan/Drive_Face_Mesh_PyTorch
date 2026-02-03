#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Triangulate 3D joints from two-view 2D keypoints using either video frames or pre-extracted keypoints.
Supports modular triangulation, pose estimation, and interactive 3D visualization.

Author: Kaixu Chen
Last Modified: August 4th, 2025
"""

import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import List

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from head3D_fuse.fuse import process_single_person_env

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Worker：进程执行函数
# ---------------------------------------------------------------------
def _worker(
    env_dirs: List[Path],
    out_root: Path,
    infer_root: Path,
    cfg_dict: dict,
):
    """
    每个进程的入口：设置环境变量，并处理分配的任务列表
    """

    # 1. 将字典转回 Hydra 配置（多进程传递对象时，转为字典更安全）
    cfg = OmegaConf.create(cfg_dict)

    logger.info(f"🏃‍♂️ {_worker.__name__} 启动，分配任务数: {len(env_dirs)}")

    for env_dir in env_dirs:
        try:
            process_single_person_env(env_dir, out_root, infer_root, cfg)
        except Exception as e:
            logger.error(f"❌ {_worker.__name__} 处理 {env_dir.name} 时出错: {e}")

    logger.info(f"🏁 {_worker.__name__} 所有任务处理完毕")


@hydra.main(version_base=None, config_path="../configs", config_name="head3d_fuse")
def main(cfg: DictConfig) -> None:
    # 1. 経路準備
    log_root = Path(cfg.log_path).resolve()
    infer_root = Path(cfg.paths.result_output_path).resolve()
    video_root = Path(cfg.paths.video_path).resolve()
    sam_3d_root = Path(cfg.paths.sam3d_results_path).resolve()

    # --- 1. 制御パラメータの取得 ---
    # cfg.infer.person_list: [1, 2, 3] または [-1]
    # cfg.infer.env_list: ["room1", "outdoor"] または ["all"]
    target_person_ids = [int(pid) for pid in cfg.infer.get("person_list", [-1])]
    target_envs = cfg.infer.get("env_list", ["all"])

    # --- 2. 条件に合致する Person/Env タスクを収集 ---
    all_env_tasks = []

    # Person フォルダをループ
    for p_dir in sorted(sam_3d_root.iterdir()):
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
    total_workers = cfg.infer.get("workers", 2)

    chunks = np.array_split(all_env_tasks, total_workers)

    logger.info(f"総プロセス数: {total_workers}")
    logger.info(f"総処理人数: {len(target_person_ids)}")

    # 3. 启动并行进程
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    mp.set_start_method("spawn", force=True)

    processes = []
    for i, worker_id in enumerate(range(total_workers)):
        env_list = chunks[i].tolist()
        if not env_list:
            continue

        logger.info(f"  - Worker {i} 分配任务数: {len(env_list)}")

        p = mp.Process(
            target=_worker,
            args=(
                env_list,
                log_root,
                infer_root,
                cfg_dict,
            ),
        )
        p.start()
        processes.append(p)

    # 4. 等待所有进程完成
    for p in processes:
        p.join()

    logger.info("🎉 [SUCCESS] 所有任务已圆满完成！")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
