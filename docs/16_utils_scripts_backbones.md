# 16 Utils、Scripts 与 Vision Backbones

## utils/

### `utils/device.py`

负责解析：

- `--device`
- `--gpu-ids`

行为：

- 设置 CUDA 可见设备。
- CUDA 不可用时 fallback CPU。
- 返回 device selection 信息，用于日志和 summary。

### `utils/diagnostics.py`

提供统一诊断工具：

- `print_event`
- `append_jsonl`
- `write_json`
- `save_bar_png`
- `save_line_png`
- `ensure_dir`

所有 preprocess、training、inference 脚本都使用它输出结构化日志。

### `utils/env_utils.py`

AirVLN 环境底层工具：

- `SimState`
- teacher action 获取。
- graph progress sensor。
- action 到 pose 的转换。

V0 直接依赖其中的 action physics 逻辑间接保持和 AirSim 一致。

### `utils/env_vector.py`

多进程/线程 vector env 封装：

- 给每个 env worker 发送 set_batch、get_obs 等命令。
- 格式化 obs/reward/done/info。

### `utils/shorest_path_sensor.py`

路径图和 shortest path 相关逻辑，主要服务 dagger/SF 模式。

## scripts/

### `scripts/check_v0_setup.py`

检查：

- 端口是否可连接。
- GPU 参数是否能解析。
- 本地运行环境是否基本可用。

### dataset download scripts

- `scripts/download_dataset_aerialvln.sh`
- `scripts/download_dataset_aerialvln-s.sh`

用于下载或提示数据集放置方式。Windows 下可能需要 Git Bash 或 WSL。

## vision_backbones/

目录用于本地放置大模型资源：

```text
vision_backbones/
  dinov2/
  resnet/
  torch_hub/
```

仓库只提交 `.gitkeep` 和 README，不提交大模型权重。

## DATA/

`DATA/` 是运行产物目录，不是源码模块。

典型内容：

- checkpoints。
- diagnostics。
- eval result。
- 训练中间产物。

建议：

- 不要把大 checkpoint 和完整数据集提交到 Git。
- 对重要实验保留 `summary.json`、配置文件和 commit hash。

## logger

`utils/logger.py` 为环境模块提供日志输出。V0 自身更多使用 `utils/diagnostics.py`。

## 维护建议

- 新增脚本时，统一写入 `DATA/v0/diagnostics/<script_name>/summary.json`。
- 新增 GPU 脚本时，统一使用 `select_torch_device()`。
- 新增可视化时，优先复用 `save_bar_png` 和 `save_line_png`。
