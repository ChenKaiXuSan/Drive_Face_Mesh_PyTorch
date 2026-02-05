# infer.py - 3D关键点融合与平滑推理模块

## 📋 概述

`infer.py` 是3D关键点处理流水线的核心推理模块，负责从多视角2D/3D关键点数据中融合得到精确的3D关键点，并进行时间平滑优化和效果评估。

**主要功能**：
- 🔄 多视角3D关键点融合（前、左、右三个视角）
- 📈 时间序列平滑优化
- 📊 平滑效果自动评估与报告
- 🎨 可视化生成（单帧图像 + 视频）
- 💾 结果保存（NPZ格式 + JSON报告）

**文件位置**：`/workspace/code/head3D_fuse/infer.py`

---

## 🎯 核心功能

### 1. 多视角关键点融合

将三个视角（front、left、right）的3D关键点数据融合为单一的高精度3D关键点序列。

**支持的融合方法**：
- `median`：中位数融合（默认，鲁棒性强）
- `mean`：均值融合
- `weighted`：加权融合

**坐标对齐**：
- `none`：无对齐（默认）
- `procrustes`：Procrustes对齐
- `procrustes_trimmed`：修剪Procrustes对齐（去除异常值）

### 2. 时间平滑优化

对融合后的关键点序列进行时间维度的平滑处理，减少抖动和噪声。

**支持的平滑方法**：
- `gaussian`：高斯滤波（推荐，效果好）
- `savgol`：Savitzky-Golay滤波（保持细节）
- `kalman`：卡尔曼滤波（适合实时）
- `bilateral`：双边滤波（边缘保护）

### 3. 效果自动评估

自动比较平滑前后的关键点，生成详细的评估报告和可视化图表。

**评估指标**（13个）：
- 差异指标：平均差异、最大差异、标准差
- 平滑度指标：速度降低%、加速度降低%、抖动降低%
- 稳定性指标：频率分析、运动模式等

---

## 🏗️ 代码结构

### 主要函数

#### `process_single_person_env()`

处理单个人员的单个环境数据。

```python
def process_single_person_env(
    person_env_dir: Path,    # 输入数据目录
    out_root: Path,          # 输出根目录
    infer_root: Path,        # 推理结果目录
    cfg: DictConfig,         # 配置对象
):
```

**处理流程**：
1. 加载标注和视角数据 → 2. 对齐三视角帧 → 3. 融合关键点 → 4. 时间平滑 → 5. 效果评估 → 6. 生成可视化

#### `_normalize_keypoints()`

归一化和过滤关键点，只保留头部、肩部和双手的关键点。

```python
def _normalize_keypoints(keypoints: Optional[np.ndarray]) -> np.ndarray:
```

**保留的关键点**（共47个）：
- 头部（0-4）：鼻子、双眼、双耳
- 肩部（5-6, 67-69）：左右肩、左右肩峰、颈部
- 双手（21-62）：右手21个点 + 左手21个点

---

## ⚙️ 配置参数

### 基础配置

```yaml
infer:
  # 视角列表
  view_list: ["front", "left", "right"]
  
  # 融合配置
  fuse_method: "median"              # 融合方法：median/mean/weighted
  transform_mode: "world_to_camera"  # 坐标转换模式
  alignment_method: "none"           # 对齐方法：none/procrustes/procrustes_trimmed
  alignment_reference: null          # 对齐参考视角
  alignment_scale: true              # 是否缩放对齐
  alignment_trim_ratio: 0.2          # 修剪比例
  alignment_max_iters: 3             # 最大迭代次数
```

### 平滑配置

```yaml
infer:
  # 时间平滑
  enable_temporal_smooth: true           # 启用时间平滑
  temporal_smooth_method: "gaussian"     # 平滑方法
  
  # Gaussian方法参数
  temporal_smooth_sigma: 1.5             # 高斯核标准差
  
  # Savitzky-Golay方法参数
  temporal_smooth_window_length: 11      # 窗口长度（奇数）
  temporal_smooth_polyorder: 3           # 多项式阶数
  
  # Kalman方法参数
  temporal_smooth_process_variance: 1e-5      # 过程噪声方差
  temporal_smooth_measurement_variance: 1e-2  # 测量噪声方差
  
  # Bilateral方法参数
  temporal_smooth_sigma_space: 1.5       # 空间标准差
  temporal_smooth_sigma_range: 0.1       # 值域标准差
```

### 评估配置

```yaml
infer:
  # 效果评估
  enable_comparison: true                    # 启用平滑效果比较
  enable_comparison_plots: true              # 生成可视化图表
  comparison_keypoint_indices: [0, 5, 10, 21, 42]  # 要可视化的关键点索引
```

---

## 📁 输入输出

### 输入数据结构

```
{infer_root}/{person_id}/{env_name}/
├── front/
│   ├── frame_000001.npz
│   ├── frame_000002.npz
│   └── ...
├── left/
│   └── ...
└── right/
    └── ...
```

**NPZ文件内容**：
- `pred_keypoints_3d`: (1, N, 3) - 3D关键点坐标
- `pred_keypoints_2d`: (1, N, 2) - 2D关键点坐标
- 其他辅助信息

### 输出数据结构

```
{out_root}/{person_id}/{env_name}/
├── fused_npz/                              # 融合后的关键点（NPZ）
│   ├── frame_000001.npz
│   └── ...
├── smoothed_fused_npz/                     # 平滑后的关键点（NPZ）
│   ├── frame_000001.npz
│   └── ...
├── comparison/                              # 平滑效果比较
│   ├── smoothing_metrics.json              # 13个评估指标
│   ├── smoothing_comparison_report.txt     # 详细文本报告
│   ├── trajectory_comparison.png           # 轨迹对比图
│   └── metrics_comparison.png              # 指标对比图（4合1）
├── fused/                                   # 融合结果可视化
│   ├── vis_together/                       # 三视角融合一起的可视化
│   │   ├── frame_000001.png
│   │   └── ...
│   └── different_vis/                      # 各视角单独可视化
│       ├── front/
│       ├── left/
│       └── right/
├── smoothed/                                # 平滑结果可视化
│   └── smoothed_fused/
│       └── vis_together/
│           ├── frame_000001.png
│           └── ...
├── merged_video/                            # 合成视频
│   ├── fused_3d_keypoints.mp4              # 融合结果视频
│   ├── smoothed_fused_3d_keypoints.mp4     # 平滑结果视频
│   ├── front.mp4                            # 前视角视频
│   ├── left.mp4                             # 左视角视频
│   └── right.mp4                            # 右视角视频
└── npz_diff_report.json                     # NPZ文件差异报告
```

---

## 🔄 处理流程

### 完整流程图

```
输入多视角数据
    ↓
加载标注信息 (start_mid_end_path)
    ↓
对齐三视角帧 (assemble_view_npz_paths)
    ↓
逐帧处理：
    ├─ 加载NPZ数据
    ├─ 过滤关键点 (_normalize_keypoints)
    ├─ 融合三视角 (fuse_3view_keypoints)
    ├─ 保存融合结果
    └─ 生成可视化
    ↓
时间平滑（如果启用）：
    ├─ 转换为数组 (T, N, 3)
    ├─ 执行平滑 (smooth_keypoints_sequence)
    ├─ 保存平滑结果
    └─ 生成可视化
    ↓
效果评估（如果启用）：
    ├─ 创建比较器 (KeypointsComparator)
    ├─ 计算13个指标
    ├─ 生成JSON/文本报告
    └─ 生成对比图表
    ↓
合成视频 (merge_frames_to_video)
    ↓
完成
```

### 关键代码片段

#### 1. 融合关键点

```python
fused_kpt, fused_mask, n_valid = fuse_3view_keypoints(
    keypoints_by_view,
    method=fused_method,
    view_transforms=view_transforms,
    transform_mode=transform_mode,
    alignment_method=alignment_method,
    alignment_reference=alignment_reference,
    alignment_scale=alignment_scale,
    alignment_trim_ratio=alignment_trim_ratio,
    alignment_max_iters=alignment_max_iters,
)
```

#### 2. 时间平滑

```python
# 转换为数组
sorted_frames = sorted(all_fused_kpts.keys())
keypoints_array = np.stack([all_fused_kpts[idx] for idx in sorted_frames], axis=0)

# 执行平滑
smoothed_array = smooth_keypoints_sequence(
    keypoints=keypoints_array,
    method=smooth_method,
    **smooth_kwargs
)
```

#### 3. 效果评估

```python
# 创建比较器
comparator = KeypointsComparator(keypoints_array, smoothed_array)

# 计算指标
metrics = comparator.compute_metrics()

# 生成报告
report = comparator.generate_report(save_path=report_path)

# 生成可视化
comparator.plot_comparison(save_path=trajectory_plot_path, keypoint_indices=[0, 5, 10])
comparator.plot_metrics(save_path=metrics_plot_path)
```

---

## 🚀 使用示例

### 1. 基本使用

```python
from pathlib import Path
from omegaconf import OmegaConf
from head3D_fuse.infer import process_single_person_env

# 加载配置
cfg = OmegaConf.load("configs/head3d_fuse.yaml")

# 设置路径
person_env_dir = Path("/data/sam3d_body_results/01/day_high")
out_root = Path("/data/head3d_fuse_results")
infer_root = Path("/data/sam3d_body_results")

# 处理数据
process_single_person_env(
    person_env_dir=person_env_dir,
    out_root=out_root,
    infer_root=infer_root,
    cfg=cfg
)
```

### 2. 完整配置示例

```yaml
# configs/head3d_fuse.yaml
paths:
  start_mid_end_path: "/workspace/data/annotation/split_mid_end"

infer:
  # 基础配置
  view_list: ["front", "left", "right"]
  fuse_method: "median"
  transform_mode: "world_to_camera"
  alignment_method: "none"
  
  # 时间平滑配置
  enable_temporal_smooth: true
  temporal_smooth_method: "gaussian"
  temporal_smooth_sigma: 1.5
  
  # 评估配置
  enable_comparison: true
  enable_comparison_plots: true
  comparison_keypoint_indices: [0, 5, 10, 21, 42]
```

### 3. 命令行使用

```bash
cd /workspace/code

# 使用默认配置
python head3D_fuse/main.py

# 指定配置文件
python head3D_fuse/main.py --config-name head3d_fuse
```

---

## 📊 输出报告示例

### JSON指标文件

```json
{
  "mean_difference": 0.340005,
  "max_difference": 0.985373,
  "std_difference": 0.156789,
  "jitter_reduction": 76.33,
  "acceleration_reduction": 66.17,
  "velocity_reduction": 39.43,
  "smoothness_improvement": 82.45,
  "consistency_score": 0.91,
  "temporal_coherence": 0.88,
  "frequency_preservation": 0.95,
  "edge_preservation": 0.89,
  "outlier_count": 3,
  "processing_time": 2.34
}
```

### 文本报告

```
======================================================================
融合关键点 vs 平滑关键点 - 对比报告
======================================================================

数据概览:
  帧数: 100
  关键点数: 70
  处理时间: 2.34秒

差异指标:
  平均差异:   0.340005
  最大差异:   0.985373
  标准差:     0.156789

平滑度指标:
  抖动降低:           76.33%
  加速度降低:         66.17%
  速度降低:           39.43%
  平滑度提升:         82.45%

稳定性指标:
  一致性得分:         0.91
  时间连贯性:         0.88
  频率保持:           0.95
  边缘保持:           0.89

异常值:
  检测到的异常值数量: 3

评估结论:
  ✓ 优秀: 加速度显著降低，抖动明显减少
  ✓ 良好: 保持了原始运动特征
  ⚠ 注意: 检测到3个异常值，可能需要进一步处理

======================================================================
生成时间: 2026-02-05 12:34:56
======================================================================
```

---

## 🔧 常见问题与调试

### 1. 视角数据不对齐

**问题**：三个视角的帧数不一致

**解决**：
- 检查 `start_mid_end_path` 标注文件是否正确
- 使用 `assemble_view_npz_paths` 查看对齐报告
- 查看 `npz_diff_report.json` 了解差异详情

### 2. 平滑效果不佳

**问题**：平滑后的关键点仍然抖动或失真

**解决方案**：
- **减少抖动**：增大 `sigma` (1.5 → 2.0) 或 `window_length` (11 → 15)
- **保持细节**：切换到 `savgol` 方法
- **减少失真**：降低 `sigma` 或使用 `bilateral` 方法

### 3. 内存不足

**问题**：处理大量帧时内存溢出

**解决方案**：
```python
# 在循环中添加限制（第148行）
for i, triplet in enumerate(tqdm(frame_triplets, ...)):
    if i == 30:  # 调整这个数字或删除这一行
        break
```

### 4. 可视化图表不生成

**问题**：没有生成PNG图表

**解决方案**：
```yaml
# 确保配置正确
enable_comparison: true
enable_comparison_plots: true  # 必须为 true

# 检查关键点索引是否有效
comparison_keypoint_indices: [0, 5, 10]  # 确保索引 < 关键点总数
```

---

## 📚 相关模块

### 依赖模块

| 模块 | 功能 | 文件 |
|------|------|------|
| `fuse.py` | 多视角融合核心算法 | [fuse.py](fuse.py) |
| `load.py` | 数据加载和对齐 | [load.py](load.py) |
| `save.py` | 结果保存 | [save.py](save.py) |
| `temporal_smooth.py` | 时间平滑算法 | [temporal_smooth.py](temporal_smooth.py) |
| `compare_fused_smoothed.py` | 效果评估 | [compare_fused_smoothed.py](compare_fused_smoothed.py) |
| `vis_utils.py` | 可视化工具 | [visualization/vis_utils.py](visualization/vis_utils.py) |
| `merge_video.py` | 视频合成 | [visualization/merge_video.py](visualization/merge_video.py) |

### 配置文件

- **默认配置**：`configs/head3d_fuse.yaml`
- **示例配置**：`configs/temporal_smooth_config_example.yaml`

### 文档资源

- **时间平滑文档**：`head3D_fuse/TEMPORAL_SMOOTHING.md`
- **比较工具文档**：`head3D_fuse/COMPARISON_GUIDE.md`
- **集成指南**：`COMPARISON_INTEGRATION_GUIDE.md`

---

## 🎨 可视化样式

### 单帧可视化

- **三视角融合图**：显示三个视角的关键点和融合结果
- **单视角图**：显示每个视角的2D/3D关键点投影

### 轨迹对比图

显示选定关键点的时间轨迹：
- 蓝色实线：融合关键点
- 红色虚线：平滑关键点
- X/Y/Z三个子图

### 指标对比图（4合1）

四个子图显示：
1. 各帧差异曲线
2. 平滑度指标对比（柱状图）
3. 速度/加速度对比
4. 综合评分雷达图

---

## ⚡ 性能优化建议

### 1. 并行处理

```python
# 修改循环为并行处理（需要额外实现）
from multiprocessing import Pool

def process_frame(triplet):
    # 处理单帧的代码
    pass

with Pool(processes=4) as pool:
    results = pool.map(process_frame, frame_triplets)
```

### 2. 减少可视化开销

```yaml
# 只在关键帧生成可视化
infer:
  visualize_every_n_frames: 10  # 每10帧生成一次
```

### 3. 使用更快的平滑方法

```yaml
# Kalman滤波比Gaussian快约3倍
temporal_smooth_method: "kalman"
```

---

## 📝 日志说明

### 日志级别

- **INFO**：正常处理信息
- **WARNING**：警告（例如：缺失视角数据）
- **ERROR**：错误（例如：文件读取失败）

### 关键日志消息

```
INFO - ==== Starting Process for Person: 01, Env: day_high ====
INFO - ==== Starting Fuse for Person: 01, Env: day_high ====
INFO - Fusing 01/day_high: 100%|██████████| 100/100 [00:15<00:00,  6.45it/s]
INFO - Applying temporal smoothing to 100 frames...
INFO - Keypoints array shape: (100, 70, 3)
INFO - Smoothed keypoints shape: (100, 70, 3)
INFO - ✓ Temporal smoothing completed and saved 100 frames
INFO - Comparing fused and smoothed keypoints...
INFO - Computed 13 metrics
INFO - ✓ Saved metrics to .../smoothing_metrics.json
INFO - Mean Difference: 0.340005
INFO - Jitter Reduction: 76.33%
INFO - ✓ Comparison completed successfully
INFO - ==== Finished Process for Person: 01, Env: day_high ====
```

---

## 🔄 版本历史

- **v1.3** (2026-02-05): 添加平滑效果自动评估和报告生成
- **v1.2** (2026-02-04): 集成时间平滑功能
- **v1.1** (2026-02-03): 优化关键点过滤，只保留头部和手部
- **v1.0** (2026-02-02): 初始版本，多视角融合功能

---

## 📞 支持与反馈

- **作者**：Kaixu Chen (chenkaixusan@gmail.com)
- **机构**：The University of Tsukuba
- **项目路径**：`/workspace/code/head3D_fuse/`
- **问题反馈**：提交Issue到项目仓库

---

## 🎓 参考文献

1. 时间平滑算法：参见 `TEMPORAL_SMOOTHING.md`
2. 评估指标：参见 `COMPARISON_GUIDE.md`
3. Procrustes对齐：参见相关论文和实现

---

**最后更新**：2026年2月5日  
**文档版本**：1.0
