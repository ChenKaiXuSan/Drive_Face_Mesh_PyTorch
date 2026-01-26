#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
from pathlib import Path
from typing import Dict, List

import hydra
from omegaconf import DictConfig, OmegaConf

from .infer import process_frame_list
from .load import load_data

logger = logging.getLogger(__name__)

VALID_VIEWS = {"front", "left", "right"}
REQUIRED_VIEWS = {"front", "left", "right"}  # VIDEO 是否强制三视角齐全


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def list_dirs(p: Path) -> List[Path]:
    return sorted([x for x in p.iterdir() if x.is_dir()])


def rel_to_root(root: Path, p: Path) -> Path:
    return p.relative_to(root)


def collect_video_views(env_dir: Path, patterns: List[str]) -> Dict[str, Path]:
    """Collect front/left/right videos under env_dir (ignore dive_view)."""
    view_map: Dict[str, Path] = {}
    for pat in patterns:
        for f in env_dir.glob(pat):
            stem = f.stem.lower()  # front/left/right/dive_view
            if stem in VALID_VIEWS:
                view_map[stem] = f.resolve()
    return view_map


def collect_images_in_view_dir(view_dir: Path, patterns: List[str]) -> List[Path]:
    files: List[Path] = []
    for pat in patterns:
        files.extend(view_dir.glob(pat))
    return sorted({f.resolve() for f in files if f.is_file()})


# ---------------------------------------------------------------------
# VIDEO: videos/{person}/{env}/{front,left,right}.mp4
# Process order: person -> env
# ---------------------------------------------------------------------
def run_video_person_env(
    video_root: Path,
    out_root: Path,
    infer_root: Path,
    cfg: DictConfig,
    require_all_views: bool = True,
) -> int:
    vid_patterns = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.MP4", "*.MOV"]

    done = 0
    person_dirs = list_dirs(video_root)
    if not person_dirs:
        raise RuntimeError(f"No person dirs under: {video_root}")

    for person_dir in person_dirs:
        person_id = person_dir.name
        logger.info(f"==== PERSON START (VIDEO): {person_id} ====")

        env_dirs = list_dirs(person_dir)
        if not env_dirs:
            logger.warning(f"[No env dirs] {person_dir}")
            continue

        for env_dir in env_dirs:
            env_name = env_dir.name
            rel_env = rel_to_root(video_root, env_dir)  # e.g. 01/夜多い

            view_map = collect_video_views(env_dir, vid_patterns)

            if require_all_views and REQUIRED_VIEWS and not REQUIRED_VIEWS.issubset(view_map.keys()):
                logger.warning(
                    f"[Skip] VIDEO {rel_env}: missing views, found={sorted(view_map.keys())}"
                )
                continue

            if not view_map:
                logger.warning(f"[No valid video views] {rel_env}")
                continue

            file_list = [view_map[v] for v in ("front", "left", "right") if v in view_map]

            logger.info(f"Processing VIDEO: person={person_id} env={env_name} | files={[p.name for p in file_list]}")

            frame_list = load_data(file_list)

            current_out_dir = out_root / rel_env
            current_infer_dir = infer_root / rel_env
            current_out_dir.mkdir(parents=True, exist_ok=True)
            current_infer_dir.mkdir(parents=True, exist_ok=True)

            process_frame_list(
                frame_list=frame_list,
                out_dir=current_out_dir,
                inference_output_path=current_infer_dir,
                cfg=cfg,
            )
            done += 1

        logger.info(f"==== PERSON DONE (VIDEO): {person_id} ====")

    return done


# ---------------------------------------------------------------------
# IMAGE: image/{person}/{env}/{front,left,right}/*.png
# Process order: person -> env -> view
# ---------------------------------------------------------------------
def run_image_person_env(
    image_root: Path,
    out_root: Path,
    infer_root: Path,
    cfg: DictConfig,
) -> int:
    img_patterns = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]

    done = 0
    person_dirs = list_dirs(image_root)
    if not person_dirs:
        raise RuntimeError(f"No person dirs under: {image_root}")

    for person_dir in person_dirs:
        person_id = person_dir.name
        logger.info(f"==== PERSON START (IMAGE): {person_id} ====")

        env_dirs = list_dirs(person_dir)
        if not env_dirs:
            logger.warning(f"[No env dirs] {person_dir}")
            continue

        for env_dir in env_dirs:
            env_name = env_dir.name
            logger.info(f"---- ENV START (IMAGE): person={person_id} env={env_name} ----")

            # env_dir 下按 view 目录组织：front/left/right/dive_view
            for view_name in REQUIRED_VIEWS:
                view_dir = env_dir / view_name
                if not view_dir.exists():
                    logger.warning(f"[Skip] IMAGE missing view dir: {rel_to_root(image_root, view_dir)}")
                    continue

                files = collect_images_in_view_dir(view_dir, img_patterns)
                if not files:
                    logger.warning(f"[Skip] IMAGE empty view dir: {rel_to_root(image_root, view_dir)}")
                    continue

                rel_view = rel_to_root(image_root, view_dir)  # e.g. 01/夜多い/front
                logger.info(f"Processing IMAGE: {rel_view} | n_files={len(files)}")

                frame_list = load_data(files)

                current_out_dir = out_root / rel_view
                current_infer_dir = infer_root / rel_view
                current_out_dir.mkdir(parents=True, exist_ok=True)
                current_infer_dir.mkdir(parents=True, exist_ok=True)

                process_frame_list(
                    frame_list=frame_list,
                    out_dir=current_out_dir,
                    inference_output_path=current_infer_dir,
                    cfg=cfg,
                )
                done += 1

            logger.info(f"---- ENV DONE (IMAGE): person={person_id} env={env_name} ----")

        logger.info(f"==== PERSON DONE (IMAGE): {person_id} ====")

    return done


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
@hydra.main(config_path="../configs", config_name="sam3d_body", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("==== Config ====\n" + OmegaConf.to_yaml(cfg))

    infer_type = cfg.infer.get("type", "video")  # video | image

    out_root = Path(cfg.paths.log_path).resolve()
    infer_root = Path(cfg.paths.result_output_path).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    infer_root.mkdir(parents=True, exist_ok=True)

    if infer_type == "video":
        video_root = Path(cfg.paths.video_path).resolve()
        if not video_root.exists():
            raise FileNotFoundError(f"video_path not found: {video_root}")

        done = run_video_person_env(
            video_root=video_root,
            out_root=out_root,
            infer_root=infer_root,
            cfg=cfg,
            require_all_views=True,  # 改 False 则缺视角也跑
        )

    elif infer_type == "image":
        image_root = Path(cfg.paths.image_path).resolve()
        if not image_root.exists():
            raise FileNotFoundError(f"image_path not found: {image_root}")

        done = run_image_person_env(
            image_root=image_root,
            out_root=out_root,
            infer_root=infer_root,
            cfg=cfg,
        )

    else:
        raise ValueError(f"Unknown infer.type: {infer_type} (expected 'video' or 'image')")

    logger.info(f"==== ALL DONE ==== processed_tasks={done}")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
