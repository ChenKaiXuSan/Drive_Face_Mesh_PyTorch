#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/head_movement_analysis/export_to_csv.py
Project: /workspace/code/head_movement_analysis
Created Date: Saturday February 7th 2026
Author: Kaixu Chen

导出头部姿态角度到CSV文件的工具脚本
"""
import argparse
import csv
import logging
from pathlib import Path

from main import HeadPoseAnalyzer

logger = logging.getLogger(__name__)


def export_head_angles_to_csv(
    fused_dir: Path,
    output_csv: Path,
    start_frame: int = None,
    end_frame: int = None,
):
    """
    分析头部姿态并导出到CSV文件

    Args:
        fused_dir: 包含融合npy文件的目录
        output_csv: 输出CSV文件路径
        start_frame: 起始帧索引（可选）
        end_frame: 结束帧索引（可选）
    """
    # 创建分析器
    analyzer = HeadPoseAnalyzer()

    # 分析序列
    logger.info(f"Analyzing sequences from {fused_dir}")
    results = analyzer.analyze_sequence(fused_dir, start_frame, end_frame)

    if not results:
        logger.error("No valid results to export")
        return

    # 保存为CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "pitch_deg", "yaw_deg", "roll_deg"])

        for frame_idx in sorted(results.keys()):
            angles = results[frame_idx]
            writer.writerow(
                [
                    frame_idx,
                    f"{angles['pitch']:.2f}",
                    f"{angles['yaw']:.2f}",
                    f"{angles['roll']:.2f}",
                ]
            )

    logger.info(f"Successfully exported {len(results)} frames to {output_csv}")


def main():
    """命令行工具入口"""
    parser = argparse.ArgumentParser(
        description="导出头部姿态角度到CSV文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析整个序列
  python export_to_csv.py \\
    --input /workspace/data/head3d_fuse_results/01/夜多い/fused_npz \\
    --output head_pose_angles.csv

  # 指定帧范围
  python export_to_csv.py \\
    --input /workspace/data/head3d_fuse_results/01/夜多い/fused_npz \\
    --output head_pose_angles.csv \\
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
        required=True,
        help="输出CSV文件路径",
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

    # 导出
    export_head_angles_to_csv(
        fused_dir=args.input,
        output_csv=args.output,
        start_frame=args.start,
        end_frame=args.end,
    )


if __name__ == "__main__":
    main()
