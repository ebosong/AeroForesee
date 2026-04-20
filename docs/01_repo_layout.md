# 01 仓库结构与模块边界

## 顶层目录

```text
airsim_plugin/       AirSim simulator server/client 和动作定义
configs/             模型、融合器、路径配置
DATA/                V0 输出产物、诊断、checkpoint 默认目录
inference/           在线 planner 和 eval 入口
models/              V0 的核心模型与决策模块
preprocess/          离线数据构造流程
prompts/             milestone parser 和 action prior prompt
schemas/             milestone JSON schema 校验
scripts/             数据下载、环境检查脚本
src/                 AirVLN 环境封装
training/            数据集封装、evaluator 训练、fuser 调参
utils/               设备选择、诊断日志、环境工具、路径传感器
vision_backbones/    DINOv2 / ResNet / torch hub cache 占位目录
```

## 代码边界

### V0 自研核心

- `models/`
- `preprocess/`
- `training/`
- `inference/`
- `configs/`
- `prompts/`
- `schemas/`

这些目录构成 AeroForesee V0 的主要逻辑。

### AirVLN 兼容层

- `src/vlnce_src/env.py`
- `utils/env_utils.py`
- `utils/env_vector.py`
- `airsim_plugin/`

这些文件负责和原 AirVLN/AirSim 体系对齐，包括：

- episode batch 管理。
- simulator scene 打开和 pose 设置。
- RGB/depth 获取。
- action 执行。
- success、nDTW、sDTW、path_length 等指标。

### 产物目录

`DATA/` 是默认输出目录，典型结构如下：

```text
DATA/v0/
  checkpoints/
  diagnostics/
  eval/
```

根 README 中部分命令会输出到小写 `data/`，例如：

```text
data/instruction_plan.jsonl
data/step_windows/train.jsonl
data/action_prior_cache/train.jsonl
data/rollout_labels/train.jsonl
data/latent_targets/train_index.jsonl
```

如果本地没有 `data/`，脚本会在写入时创建父目录。

## 主要入口

### 离线入口

- `preprocess/parse_instruction.py`
- `preprocess/build_step_windows.py`
- `preprocess/build_action_prior_cache.py`
- `preprocess/build_rollout_labels.py`
- `preprocess/build_latent_targets.py`

### 训练入口

- `training/train_action_evaluator.py`
- `training/tune_fuser.py`

### 在线入口

- `inference/run_eval_aerialvln.py`
- `inference/planner_loop.py`

## 依赖方向

推荐理解方式：

```text
configs/prompts/schemas
  -> preprocess
  -> training
  -> models
  -> inference
  -> src/airsim_plugin/utils
```

其中 `models/` 同时服务于离线训练和在线推理。

## 维护建议

- 修改数据格式时，优先同步更新 `build_step_windows.py`、`v0_dataset.py`、`planner_loop.py`。
- 修改 action id 或动作名称时，必须检查 `airsim_settings.py`、`action_space.py`、`build_rollout_labels.py`。
- 修改 hidden_dim、视觉骨干时，必须检查 `configs/model.yaml`、`build_latent_targets.py`、`train_action_evaluator.py` 和 checkpoint 兼容性。
