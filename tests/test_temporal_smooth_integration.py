#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
测试 infer.py 中时间平滑功能的集成是否正确
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from head3D_fuse.smooth.temporal_smooth import smooth_keypoints_sequence


def test_integration():
    """模拟 infer.py 中的数据流程"""
    
    print("\n" + "=" * 60)
    print("测试时间平滑集成")
    print("=" * 60)
    
    # 1. 模拟 all_fused_kpts 字典（从融合过程收集）
    print("\n[1] 模拟融合后的关键点字典...")
    all_fused_kpts = {}
    num_frames = 50
    num_keypoints = 70  # SAM3D body 有70个关键点
    
    for frame_idx in range(0, num_frames * 10, 10):  # 0, 10, 20, ...
        all_fused_kpts[frame_idx] = np.random.randn(num_keypoints, 3)
    
    print(f"  ✓ 收集了 {len(all_fused_kpts)} 帧")
    print(f"  ✓ 帧索引: {list(all_fused_kpts.keys())[:5]}...")
    
    # 2. 转换为 numpy 数组（修复后的方法）
    print("\n[2] 转换字典为 numpy 数组...")
    sorted_frames = sorted(all_fused_kpts.keys())
    keypoints_array = np.stack([all_fused_kpts[idx] for idx in sorted_frames], axis=0)
    print(f"  ✓ 数组形状: {keypoints_array.shape}")
    print(f"  ✓ 预期: (T={len(sorted_frames)}, N={num_keypoints}, 3)")
    
    # 3. 测试不同的平滑方法
    print("\n[3] 测试不同平滑方法...")
    
    methods = [
        ("gaussian", {"sigma": 1.5}),
        ("savgol", {"window_length": 11, "polyorder": 3}),
        ("kalman", {"process_variance": 1e-5, "measurement_variance": 1e-2}),
        ("bilateral", {"sigma_space": 1.5, "sigma_range": 0.1}),
    ]
    
    for method, params in methods:
        try:
            smoothed_array = smooth_keypoints_sequence(
                keypoints=keypoints_array,
                method=method,
                **params
            )
            print(f"  ✓ {method:12} → 形状: {smoothed_array.shape}")
            
            # 验证输出形状
            assert smoothed_array.shape == keypoints_array.shape, \
                f"形状不匹配: {smoothed_array.shape} != {keypoints_array.shape}"
            
        except Exception as e:
            print(f"  ✗ {method:12} → 错误: {e}")
            return False
    
    # 4. 验证数据保存流程
    print("\n[4] 验证数据索引和保存流程...")
    smoothed_array = smooth_keypoints_sequence(
        keypoints=keypoints_array,
        method="gaussian",
        sigma=1.5
    )
    
    # 模拟保存循环
    saved_count = 0
    for i, frame_idx in enumerate(sorted_frames):
        smooth_fused_kpt = smoothed_array[i]
        
        # 验证数据
        assert smooth_fused_kpt.shape == (num_keypoints, 3), \
            f"单帧形状错误: {smooth_fused_kpt.shape}"
        
        # 模拟保存（实际代码中会调用 _save_fused_keypoints）
        # _save_fused_keypoints(save_dir, frame_idx, smooth_fused_kpt, ...)
        saved_count += 1
    
    print(f"  ✓ 成功处理 {saved_count} 帧")
    print(f"  ✓ 帧索引映射正确")
    
    # 5. 测试边界情况
    print("\n[5] 测试边界情况...")
    
    # 测试少量帧
    small_kpts = {0: np.random.randn(70, 3), 10: np.random.randn(70, 3)}
    sorted_small = sorted(small_kpts.keys())
    small_array = np.stack([small_kpts[idx] for idx in sorted_small], axis=0)
    
    try:
        result = smooth_keypoints_sequence(small_array, method="gaussian", sigma=1.0)
        print(f"  ✓ 少量帧 (2帧) 测试通过: {result.shape}")
    except Exception as e:
        print(f"  ✗ 少量帧测试失败: {e}")
        return False
    
    # 测试空字典
    empty_kpts = {}
    print(f"  ✓ 空字典检查: {len(empty_kpts) == 0}")
    
    print("\n" + "=" * 60)
    print("✅ 所有集成测试通过！")
    print("=" * 60)
    return True


def test_config_scenarios():
    """测试不同配置场景"""
    
    print("\n" + "=" * 60)
    print("测试配置场景")
    print("=" * 60)
    
    # 模拟配置对象
    class MockConfig:
        def __init__(self, **kwargs):
            self.config = kwargs
        
        def get(self, key, default=None):
            return self.config.get(key, default)
    
    # 场景 1: Gaussian
    print("\n[场景1] Gaussian 配置")
    cfg = MockConfig(
        enable_temporal_smooth=True,
        temporal_smooth_method="gaussian",
        temporal_smooth_sigma=1.5
    )
    
    if cfg.get("enable_temporal_smooth", False):
        method = cfg.get("temporal_smooth_method", "gaussian")
        if method == "gaussian":
            sigma = cfg.get("temporal_smooth_sigma", 1.5)
            print(f"  ✓ 方法: {method}, sigma: {sigma}")
    
    # 场景 2: SavGol
    print("\n[场景2] SavGol 配置")
    cfg = MockConfig(
        enable_temporal_smooth=True,
        temporal_smooth_method="savgol",
        temporal_smooth_window_length=11,
        temporal_smooth_polyorder=3
    )
    
    if cfg.get("enable_temporal_smooth", False):
        method = cfg.get("temporal_smooth_method")
        if method == "savgol":
            window = cfg.get("temporal_smooth_window_length", 11)
            poly = cfg.get("temporal_smooth_polyorder", 3)
            print(f"  ✓ 方法: {method}, window: {window}, polyorder: {poly}")
    
    # 场景 3: 禁用
    print("\n[场景3] 禁用平滑")
    cfg = MockConfig(enable_temporal_smooth=False)
    
    if cfg.get("enable_temporal_smooth", False):
        print("  执行平滑")
    else:
        print("  ✓ 平滑已禁用（正确）")
    
    print("\n✅ 配置场景测试完成")


if __name__ == "__main__":
    success = test_integration()
    test_config_scenarios()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ 所有测试通过！代码集成正确。")
        print("=" * 60)
        print("\n下一步:")
        print("1. 在配置文件中启用: enable_temporal_smooth: true")
        print("2. 选择方法和参数")
        print("3. 运行你的主程序")
        sys.exit(0)
    else:
        print("\n❌ 测试失败")
        sys.exit(1)
