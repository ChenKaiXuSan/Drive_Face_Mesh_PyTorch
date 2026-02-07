#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
测试带annotation比较的头部姿态分析功能
"""
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from head_movement_analysis import (
    HeadPoseAnalyzer,
    load_head_movement_annotations,
)


def test_with_annotations():
    """测试带annotation比较的功能"""
    print("=" * 60)
    print("Head Movement Analysis - Annotation比较测试")
    print("=" * 60)
    
    # 1. 加载标注
    annotation_path = Path("/workspace/data/annotation/label/full.json")
    if not annotation_path.exists():
        print(f"\n✗ 标注文件不存在: {annotation_path}")
        return False
    
    print(f"\n加载标注文件: {annotation_path}")
    annotations = load_head_movement_annotations(annotation_path)
    
    if not annotations:
        print("\n✗ 标注加载失败")
        return False
    
    print(f"✓ 成功加载 {len(annotations)} 个视频的标注")
    
    # 显示前几个视频的标注信息
    print("\n标注信息示例:")
    for i, (video_id, labels) in enumerate(list(annotations.items())[:3]):
        print(f"  {video_id}: {len(labels)} 个标注")
        for label in labels[:2]:  # 显示前2个标注
            print(f"    - {label.label}: frames {label.start_frame}-{label.end_frame}")
    
    # 2. 创建带标注的分析器
    analyzer = HeadPoseAnalyzer(annotation_dict=annotations)
    print("\n✓ 创建带标注的HeadPoseAnalyzer成功")
    
    # 3. 选择一个测试视频
    video_id = "01_day_high"
    fused_dir = Path("/workspace/data/head3d_fuse_results/01/昼多い/fused_npz")
    
    if not fused_dir.exists():
        print(f"\n✗ 测试目录不存在: {fused_dir}")
        return False
    
    print(f"\n测试视频: {video_id}")
    print(f"数据目录: {fused_dir}")
    
    # 4. 分析序列并比较
    print("\n开始分析...")
    results = analyzer.analyze_sequence_with_annotations(
        video_id=video_id,
        fused_dir=fused_dir,
        start_frame=380,
        end_frame=450,
    )
    
    angles = results["angles"]
    comparisons = results["comparisons"]
    
    print(f"\n✓ 分析完成")
    print(f"  - 分析了 {len(angles)} 帧")
    print(f"  - 其中 {len(comparisons)} 帧有标注可比较")
    
    # 5. 显示比较结果
    if comparisons:
        print("\n比较结果示例:")
        for frame_idx in sorted(comparisons.keys())[:5]:
            comparison = comparisons[frame_idx]
            print(f"\nFrame {frame_idx}:")
            print(f"  计算角度: Pitch={comparison['angles']['pitch']:7.2f}°, "
                  f"Yaw={comparison['angles']['yaw']:7.2f}°, "
                  f"Roll={comparison['angles']['roll']:7.2f}°")
            
            for match_info in comparison["matches"]:
                annotation = match_info["annotation"]
                print(f"  标注: {annotation.label} "
                      f"[{annotation.start_frame}-{annotation.end_frame}]")
                print(f"    检测类型: {match_info['angle_type']}")
                print(f"    角度值: {match_info['angle_value']:.2f}°")
                print(f"    预期方向: {match_info['expected_direction']}")
                print(f"    匹配结果: {'✓ 匹配' if match_info['is_match'] else '✗ 不匹配'}")
        
        # 统计匹配率
        total_matches = 0
        successful_matches = 0
        for comparison in comparisons.values():
            for match_info in comparison["matches"]:
                total_matches += 1
                if match_info["is_match"]:
                    successful_matches += 1
        
        if total_matches > 0:
            match_rate = (successful_matches / total_matches) * 100
            print(f"\n统计:")
            print(f"  总比较次数: {total_matches}")
            print(f"  成功匹配: {successful_matches}")
            print(f"  匹配率: {match_rate:.1f}%")
    
    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_with_annotations()
    sys.exit(0 if success else 1)
