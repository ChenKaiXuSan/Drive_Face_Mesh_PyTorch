# 日志实现 (Logging Implementation)

## 概述

在多进程 3D 头部融合处理流程中实现了分层级的日志管理系统，确保每个任务的日志输出清晰可追踪。

## 日志架构

### 三层日志结构

```
logs/
├── worker_0.log          # Worker 0 的汇总日志
├── worker_1.log          # Worker 1 的汇总日志
├── person_01_room1.log   # Person 01, Env: room1 的专用日志
├── person_01_outdoor.log # Person 01, Env: outdoor 的专用日志
├── person_02_room1.log   # Person 02, Env: room1 的专用日志
└── ...
```

### 日志文件命名规则

1. **Worker 汇总日志**: `worker_{worker_id}.log`
   - 记录该 Worker 进程的整体执行流程
   - 显示分配的任务数量和完成状态
   - 每个 Worker 有一个独立的文件

2. **任务专用日志**: `{person_id}_{env_name}.log`
   - 为每个具体的 Person 和 Environment 组合创建
   - 记录该任务的详细执行过程
   - Person ID 和 Environment 名称从目录结构自动提取

## 实现细节

### _configure_worker_logging()

```python
def _configure_worker_logging(log_root: Path, worker_id: int, env_dirs: List[Path]) -> None:
    """为 Worker 进程配置日志输出"""
```

**职责**:
- 创建日志根目录
- 配置 Worker 进程级别的日志处理器
- 设置日志格式: `%(asctime)s | %(processName)s | %(levelname)s | %(name)s | %(message)s`
- 同时输出到文件和控制台

**参数**:
- `log_root`: 日志文件的存储根目录
- `worker_id`: Worker 进程的 ID 编号
- `env_dirs`: 该 Worker 要处理的所有环境目录列表

### _worker() 中的任务级日志

在处理每个 `env_dir` 时:

```python
for env_dir in env_dirs:
    # 提取 person_id 和 env_name
    person_id = env_dir.parent.name
    env_name = env_dir.name
    
    # 创建专用 logger
    env_logger = logging.getLogger(f"process_{worker_id}_{person_id}_{env_name}")
    
    # 清除旧的处理器
    for handler in list(env_logger.handlers):
        env_logger.removeHandler(handler)
    
    # 创建专用日志文件
    log_filename = f"{person_id}_{env_name}.log"
    log_path = out_root / log_filename
    
    # 配置文件处理器
    env_file_handler = logging.FileHandler(log_path, encoding="utf-8")
    env_file_handler.setFormatter(formatter)
    
    env_logger.addHandler(env_file_handler)
    env_logger.setLevel(logging.INFO)
    
    # 记录任务开始和完成
    env_logger.info(f"开始处理 Person: {person_id}, Env: {env_name}")
    process_single_person_env(env_dir, out_root, infer_root, cfg)
    env_logger.info(f"完成处理 Person: {person_id}, Env: {env_name}")
```

## 日志输出示例

### worker_0.log

```
2025-01-15 10:23:45,123 | MainProcess | INFO | __main__ | 🏃‍♂️ _worker 启动，Worker 0 任务数: 5
2025-01-15 10:23:46,000 | MainProcess | INFO | __main__ | 开始处理 Person: person_01, Env: room1
2025-01-15 10:24:12,456 | MainProcess | INFO | __main__ | 完成处理 Person: person_01, Env: room1
...
2025-01-15 10:25:30,789 | MainProcess | INFO | __main__ | 🏁 _worker Worker 0 所有任务处理完毕
```

### person_01_room1.log

```
2025-01-15 10:23:46,000 | INFO | process_0_person_01_room1 | 开始处理 Person: person_01, Env: room1
2025-01-15 10:23:47,123 | INFO | head3D_fuse.infer | 开始融合 3D 关键点...
2025-01-15 10:24:05,456 | INFO | head3D_fuse.infer | 融合完毕，处理后续步骤...
2025-01-15 10:24:12,456 | INFO | process_0_person_01_room1 | 完成处理 Person: person_01, Env: room1
```

## 使用优势

1. **清晰的任务追踪**: 每个任务有专用日志文件，易于定位问题
2. **分工明确**: Worker 汇总日志 + 任务详细日志相辅相成
3. **自动命名**: 日志文件名从目录结构自动生成，无需手动配置
4. **文件整理**: 所有日志集中在 `log_root` 目录，便于归档和分析
5. **并发安全**: 多个 Worker 和任务不会相互干扰

## 配置说明

日志根目录由配置文件指定:

```yaml
# configs/head3d_fuse.yaml
log_path: logs/head3d_fuse
```

可以根据需要修改 `log_path` 来改变日志存储位置。

## 调试建议

1. **快速查看任务状态**: `tail worker_*.log` 查看 Worker 汇总
2. **深度调试特定任务**: `cat {person_id}_{env_name}.log` 查看任务详情
3. **比较多个任务**: `diff {person_01_room1.log} {person_01_outdoor.log}`
4. **批量搜索错误**: `grep ERROR *.log` 找出所有错误

## 相关代码位置

- **主要实现**: [head3D_fuse/main.py](../head3D_fuse/main.py)
- **配置文件**: [configs/head3d_fuse.yaml](../configs/head3d_fuse.yaml)
- **日志输出目录**: `logs/head3d_fuse/`

---

**Last Updated**: 2025-01-15
**Implementation Status**: ✅ Complete
