#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/head_movement_analysis/visualize_angles.py
Project: /workspace/code/head_movement_analysis
Created Date: Saturday February 7th 2026
Author: Kaixu Chen

可视化头部姿态角度变化的工具脚本
"""
import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from main import HeadPoseAnalyzer

logger = logging.getLogger(__name__)


def plot_head_angles(
    results: dict,
    output_path: Path = None,
    show: bool = True,
):
    """
    绘制头部姿态角度随时间的变化曲线

    Args:
        results: HeadPoseAnalyzer.analyze_sequence返回的结果字典
        output_path: 图片保存路径（可选）
        show: 是否显示图片
    """
    # 提取数据
    frame_indices = sorted(results.keys())
    pitches = [results[idx]["pitch"] for idx in frame_indices]
    yaws = [results[idx]["yaw"] for idx in frame_indices]
    rolls = [results[idx]["roll"] for idx in frame_indices]

    # 创建图表
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle("Head Pose Angles Over Time", fontsize=16, fontweight="bold")

    # Pitch（俯仰角）
    axes[0].plot(frame_indices, pitches, "b-", linewidth=2, label="Pitch")
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Pitch Angle (degrees)", fontsize=12)
    axes[0].set_title("Pitch (上下点头): 正值=抬头, 负值=低头", fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Yaw（偏航角）
    axes[1].plot(frame_indices, yaws, "g-", linewidth=2, label="Yaw")
    axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_ylabel("Yaw Angle (degrees)", fontsize=12)
    axes[1].set_title("Yaw (左右转头): 正值=向右, 负值=向左", fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Roll（翻滚角）
    axes[2].plot(frame_indices, rolls, "r-", linewidth=2, label="Roll")
    axes[2].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[2].set_xlabel("Frame Index", fontsize=12)
    axes[2].set_ylabel("Roll Angle (degrees)", fontsize=12)
    axes[2].set_title("Roll (头部倾斜): 正值=向右倾, 负值=向左倾", fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()

    # 保存图片
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved to {output_path}")

    # 显示图片
    if show:
        plt.show()

    plt.close()


def plot_3d_trajectory(
    results: dict,
    output_path: Path = None,
    show: bool = True,
):
    """
    在3D空间中绘制头部姿态的轨迹

    Args:
        results: HeadPoseAnalyzer.analyze_sequence返回的结果字典
        output_path: 图片保存路径（可选）
        show: 是否显示图片
    """
    # 提取数据
    frame_indices = sorted(results.keys())
    pitches = np.array([results[idx]["pitch"] for idx in frame_indices])
    yaws = np.array([results[idx]["yaw"] for idx in frame_indices])
    rolls = np.array([results[idx]["roll"] for idx in frame_indices])

    # 创建3D图表
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # 绘制轨迹
    scatter = ax.scatter(
        pitches,
        yaws,
        rolls,
        c=frame_indices,
        cmap="viridis",
        s=50,
        alpha=0.6,
    )

    # 连接轨迹线
    ax.plot(pitches, yaws, rolls, "gray", alpha=0.3, linewidth=1)

    # 标记起点和终点
    ax.scatter(
        pitches[0], yaws[0], rolls[0], c="green", s=200, marker="o", label="Start"
    )
    ax.scatter(
        pitches[-1], yaws[-1], rolls[-1], c="red", s=200, marker="X", label="End"
    )

    # 设置标签
    ax.set_xlabel("Pitch (degrees)", fontsize=12)
    ax.set_ylabel("Yaw (degrees)", fontsize=12)
    ax.set_zlabel("Roll (degrees)", fontsize=12)
    ax.set_title("3D Head Pose Trajectory", fontsize=14, fontweight="bold")

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label("Frame Index", fontsize=11)

    ax.legend()

    # 保存图片
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"3D trajectory plot saved to {output_path}")

    # 显示图片
    if show:
        plt.show()

    plt.close()


def main():
    """命令行工具入口"""
    parser = argparse.ArgumentParser(
        description="可视化头部姿态角度变化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 绘制时间序列图
  python visualize_angles.py \\
    --input /workspace/data/head3d_fuse_results/01/夜多い/fused_npz \\
    --output head_angles_plot.png

  # 绘制3D轨迹图
  python visualize_angles.py \\
    --input /workspace/data/head3d_fuse_results/01/夜多い/fused_npz \\
    --output head_angles_3d.png \\
    --mode 3d

  # 指定帧范围
  python visualize_angles.py \\
    --input /workspace/data/head3d_fuse_results/01/夜多い/fused_npz \\
    --output head_angles_plot.png \\
    --start 619 \\
    --end 1000
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="融合后的npy文件所在目录",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="输出图片路径（可选）",
    )

    parser.add_argument(
        "--start",
        "-s",
        type=int,
        default=None,
        help="起始帧索引（可选）",
    )

    parser.add_argument(
        "--end",
        "-e",
        type=int,
        default=None,
        help="结束帧索引（可选）",
    )

    parser.add_argument(
        "--mode",
        "-m",
        choices=["time", "3d", "both"],
        default="time",
        help="绘图模式: time=时间序列图, 3d=3D轨迹图, both=两者都绘制",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="不显示图片，仅保存",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # 检查输入目录
    if not args.input.exists():
        logger.error(f"Input directory does not exist: {args.input}")
        return

    # 分析序列
    logger.info(f"Analyzing sequences from {args.input}")
    analyzer = HeadPoseAnalyzer()
    results = analyzer.analyze_sequence(args.input, args.start, args.end)

    if not results:
        logger.error("No valid results to visualize")
        return

    # 绘制图表
    show = not args.no_show

    if args.mode == "time" or args.mode == "both":
        output_path = args.output
        if args.mode == "both" and output_path:
            output_path = output_path.parent / f"{output_path.stem}_time{output_path.suffix}"
        plot_head_angles(results, output_path, show)

    if args.mode == "3d" or args.mode == "both":
        output_path = args.output
        if args.mode == "both" and output_path:
            output_path = output_path.parent / f"{output_path.stem}_3d{output_path.suffix}"
        plot_3d_trajectory(results, output_path, show)


if __name__ == "__main__":
    main()
