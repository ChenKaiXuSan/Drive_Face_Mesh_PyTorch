#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
测试 infer.py 中的比较功能集成
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from head3D_fuse.smooth.compare_fused_smoothed import KeypointsComparator
from head3D_fuse.smooth.temporal_smooth import smooth_keypoints_sequence


def test_infer_comparison_flow():
    """模拟 infer.py 中的比较流程"""
    
    print("\n" + "=" * 70)
    print("测试 infer.py 中的比较功能集成")
    print("=" * 70)
    
    # 1. 模拟融合后的关键点数据
    print("\n[1] 生成模拟数据...")
    np.random.seed(42)
    num_frames = 100
    num_keypoints = 70  # SAM3D body
    
    # 生成带噪声的融合关键点
    t = np.linspace(0, 4 * np.pi, num_frames)
    all_fused_kpts = {}
    
    for frame_idx in range(num_frames):
        kpt = np.zeros((num_keypoints, 3))
        for n in range(num_keypoints):
            freq = 1 + n * 0.05
            kpt[n, 0] = np.sin(freq * t[frame_idx]) + np.random.randn() * 0.1
            kpt[n, 1] = np.cos(freq * t[frame_idx]) + np.random.randn() * 0.1
            kpt[n, 2] = np.sin(2 * freq * t[frame_idx]) + np.random.randn() * 0.1
        all_fused_kpts[frame_idx] = kpt
    
    print(f"  ✓ 生成了 {len(all_fused_kpts)} 帧融合关键点")
    
    # 2. 转换为数组并平滑（模拟 infer.py 的流程）
    print("\n[2] 执行时间平滑...")
    sorted_frames = sorted(all_fused_kpts.keys())
    keypoints_array = np.stack([all_fused_kpts[idx] for idx in sorted_frames], axis=0)
    print(f"  ✓ 关键点数组形状: {keypoints_array.shape}")
    
    smoothed_array = smooth_keypoints_sequence(
        keypoints=keypoints_array,
        method="gaussian",
        sigma=1.5
    )
    print(f"  ✓ 平滑后形状: {smoothed_array.shape}")
    
    # 3. 执行比较（这是新加的部分）
    print("\n[3] 创建比较器...")
    comparator = KeypointsComparator(keypoints_array, smoothed_array)
    print("  ✓ 比较器创建成功")
    
    # 4. 计算指标
    print("\n[4] 计算评估指标...")
    metrics = comparator.compute_metrics()
    print(f"  ✓ 计算了 {len(metrics)} 个指标")
    
    # 5. 生成报告
    print("\n[5] 生成报告...")
    output_dir = Path("/workspace/test_infer_comparison")
    output_dir.mkdir(exist_ok=True)
    
    # JSON 指标
    import json
    metrics_path = output_dir / "smoothing_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 保存指标到: {metrics_path}")
    
    # 文本报告
    report_path = output_dir / "smoothing_comparison_report.txt"
    report = comparator.generate_report(save_path=report_path)
    print(f"  ✓ 保存报告到: {report_path}")
    
    # 6. 生成图表
    print("\n[6] 生成可视化图表...")
    trajectory_path = output_dir / "trajectory_comparison.png"
    comparator.plot_comparison(
        save_path=trajectory_path,
        keypoint_indices=[0, 5, 10]
    )
    print(f"  ✓ 轨迹对比图: {trajectory_path}")
    
    metrics_plot_path = output_dir / "metrics_comparison.png"
    comparator.plot_metrics(save_path=metrics_plot_path)
    print(f"  ✓ 指标对比图: {metrics_plot_path}")
    
    # 7. 打印关键指标（模拟日志输出）
    print("\n" + "=" * 70)
    print("关键指标摘要:")
    print("=" * 70)
    print(f"  平均差异:           {metrics['mean_difference']:.6f}")
    print(f"  抖动降低:           {metrics['jitter_reduction']:.2f}%")
    print(f"  加速度降低:         {metrics['acceleration_reduction']:.2f}%")
    print(f"  速度降低:           {metrics['velocity_reduction']:.2f}%")
    print("=" * 70)
    
    # 8. 显示报告预览
    print("\n报告预览:")
    print("-" * 70)
    lines = report.split("\n")
    # 显示前25行
    for line in lines[:25]:
        print(line)
    print("...")
    print(f"(完整报告已保存到 {report_path})")
    print("-" * 70)
    
    print("\n" + "=" * 70)
    print("✅ 所有流程测试通过！")
    print("=" * 70)
    print(f"\n输出文件位置: {output_dir}/")
    print("  - smoothing_metrics.json")
    print("  - smoothing_comparison_report.txt")
    print("  - trajectory_comparison.png")
    print("  - metrics_comparison.png")
    
    return True


def show_config_example():
    """展示配置示例"""
    print("\n" + "=" * 70)
    print("配置文件示例")
    print("=" * 70)
    print("""
在 configs/head3d_fuse.yaml 中添加:

infer:
  # 启用时间平滑
  enable_temporal_smooth: true
  temporal_smooth_method: "gaussian"
  temporal_smooth_sigma: 1.5
  
  # 启用比较报告
  enable_comparison: true
  enable_comparison_plots: true
  comparison_keypoint_indices: [0, 5, 10, 21, 42]
""")
    print("=" * 70)


if __name__ == "__main__":
    success = test_infer_comparison_flow()
    show_config_example()
    
    if success:
        print("\n✅ 集成测试完成！可以在 infer.py 中使用。")
    else:
        print("\n❌ 测试失败")
        sys.exit(1)
