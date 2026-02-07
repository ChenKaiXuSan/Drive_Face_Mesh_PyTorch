#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
简单的测试脚本，用于验证head_movement_analysis模块的功能
"""
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from head_movement_analysis import HeadPoseAnalyzer

def test_basic_functionality():
    """测试基本功能"""
    print("=" * 50)
    print("Head Movement Analysis - 功能测试")
    print("=" * 50)
    
    # 创建分析器
    analyzer = HeadPoseAnalyzer()
    print("\n✓ HeadPoseAnalyzer 创建成功")
    
    # 测试单帧分析
    test_file = Path("/workspace/data/head3d_fuse_results/01/夜多い/fused_npz/frame_000619_fused.npy")
    
    if not test_file.exists():
        print(f"\n✗ 测试文件不存在: {test_file}")
        return False
    
    print(f"\n测试文件: {test_file.name}")
    result = analyzer.analyze_head_pose(test_file)
    
    if result is None:
        print("\n✗ 分析失败")
        return False
    
    print("\n✓ 单帧分析成功:")
    print(f"  - Pitch (俯仰角): {result['pitch']:7.2f}°")
    print(f"  - Yaw   (偏航角): {result['yaw']:7.2f}°")
    print(f"  - Roll  (翻滚角): {result['roll']:7.2f}°")
    
    # 测试序列分析
    fused_dir = test_file.parent
    print(f"\n测试序列分析...")
    results = analyzer.analyze_sequence(fused_dir, start_frame=619, end_frame=623)
    
    if not results:
        print("\n✗ 序列分析失败")
        return False
    
    print(f"\n✓ 序列分析成功 (分析了 {len(results)} 帧)")
    print("\n前5帧结果:")
    for frame_idx in sorted(results.keys())[:5]:
        angles = results[frame_idx]
        print(f"  Frame {frame_idx}: "
              f"Pitch={angles['pitch']:7.2f}°, "
              f"Yaw={angles['yaw']:7.2f}°, "
              f"Roll={angles['roll']:7.2f}°")
    
    print("\n" + "=" * 50)
    print("所有测试通过! ✓")
    print("=" * 50)
    return True

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
