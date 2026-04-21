# AeroForesee: AirVLN V0 Latent World Model 闭环

AeroForesee V0 是一个面向 **single-view / egocentric-view 长程 UAV Vision-Language Navigation** 的 AirVLN/AerialVLN 闭环系统。

它保留 AirVLN 的仿真通信和环境指标接口，但不使用原 AirVLN baseline 训练链路。V0 的核心是：

```text
显式 milestone
  + Qwen/VLM action prior
  + causal latent action evaluator
  + 手工融合决策/fallback
  + AirSim 单步执行
  + 指标统计与 PNG 可视化
```

运行时主输入：

- 自然语言长指令
- UAV 当前前视 RGB
- 历史关键帧
- 当前及历史动作/轨迹
- 当前 milestone/progress 状态

V0 headline setting 是 **front-view RGB only**。`inference/run_eval_aerialvln.py` 会强制传入 `--run_type eval --ablate_depth`，depth 不作为主输入。

## 1. 目录结构

请按你的工作区结构放置：

```text
AirVLN_ws/
  .vscode/
  AeroForesee/          # 本仓库，所有命令默认从这里运行
  AirVLN/               # 原 AirVLN 仓库，可保留作参考；V0 不跑 baseline
  DATA/
    data/
      aerialvln/
        train.json
        val_seen.json
        val_unseen.json
        train_vocab.txt
      aerialvln-s/
        train.json
        val_seen.json
        val_unseen.json
    models/
      ddppo-models/     # 如果原 AirVLN 环境/工具需要
  ENVs/
    env_1/
    env_2/
    ...
```

关键原则：

- 从 `AirVLN_ws/AeroForesee` 运行命令。
- 数据集路径使用 `../DATA/...`。
- 仿真场景路径使用 `../ENVs`。
- 不需要运行原 AirVLN baseline，也不需要把 baseline 训练代码接回来。
- 原始 `AirVLN/` 文件夹只作为参考或备份，不参与 AeroForesee V0 主流程。

## 2. 代码结构

```text
configs/                 V0 模型、路径、融合配置
prompts/                 milestone/action-prior prompt
schemas/                 milestone schema 校验
models/                  state builder / action prior / latent evaluator / fuser / fallback
preprocess/              离线 milestone、step-window、label、latent target 构造
training/                latent action evaluator 训练与 fuser 调参
inference/               planner loop 与 AirVLN eval 入口
airsim_plugin/           AirVLN/AirSim 仿真通信
src/vlnce_src/env.py     AirVLN 环境封装和指标计算
utils/                   日志、诊断图、JSON 工具
vision_backbones/        DINOv2 / ResNet / torch hub cache 的本地存放目录
```

闭环入口和核心链路：

```text
inference/run_eval_aerialvln.py
  -> AirVLNENV.reset/get_obs
  -> V0PlannerLoop.act
  -> MilestoneAwareStateBuilder
  -> ActionPriorModule
  -> CausalLatentActionEvaluator
  -> DecisionFuser + FallbackPolicy
  -> AirVLNENV.makeActions
  -> AirVLNENV.update_measurements
  -> success / nDTW / sDTW / path_length / oracle_success / steps_taken
```

## 3. 环境安装

建议环境：

```bash
conda create -n AeroForesee python=3.8
conda activate AeroForesee
cd AirVLN_ws/AeroForesee
pip install pip==24.0 setuptools==63.2.0
pip install -r requirements.txt
pip install airsim==1.7.0
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cuXXX
pip install pytorch-transformers==1.2.0
```

说明：

- `cuXXX` 换成你的 CUDA 版本，例如 `cu118`。
- 如果只做 CPU smoke test，可安装 CPU 版 PyTorch，但完整 AirSim eval 建议使用 Nvidia GPU。
- Windows/PowerShell 生成的 JSONL 可能带 BOM，代码已兼容 `utf-8-sig`。

### 3.1 本地部署 Checklist

本地部署建议按“代码、Python 环境、数据、场景、模型资源、服务检查”的顺序处理：

```bash
cd AirVLN_ws/AeroForesee
git pull --ff-only
conda activate AeroForesee
python -m pip install -r requirements.txt
python -m pip install airsim==1.7.0
```

如果你的 Git 全局配置把 `https://github.com/` 重写到了镜像站，且镜像证书过期，可以在 Linux shell 中临时忽略全局配置拉取：

```bash
GIT_CONFIG_GLOBAL=/dev/null git pull --ff-only
```

数据集放置：

```text
AirVLN_ws/
  DATA/
    data/
      aerialvln/
        train.json
        val_seen.json
        val_unseen.json
        test.json
        train_vocab.txt
      aerialvln-s/
        train.json
        val_seen.json
        val_unseen.json
        test.json
```

仓库提供了两个下载脚本：

```bash
bash scripts/download_dataset_aerialvln.sh
bash scripts/download_dataset_aerialvln-s.sh
```

脚本只是下载 JSON 到当前目录，实际使用前需要把文件移动到上面的 `../DATA/data/aerialvln/` 和 `../DATA/data/aerialvln-s/`。Windows 下可用 Git Bash/WSL，或直接按脚本里的 S3 URL 下载。

AirSim 场景放置：

```text
AirVLN_ws/
  ENVs/
    env_1/
    env_2/
    ...
```

视觉骨干资源：

- 有网环境可以保持 `configs/model.yaml` 默认配置，首次运行时由 `torch.hub` 拉取 DINOv2。
- 离线环境把 DINOv2 仓库放到 `vision_backbones/dinov2/`，把权重或 Torch Hub cache 放到 `vision_backbones/torch_hub/`。
- 如果只做 smoke test，可把 `configs/model.yaml` 中 `vision.backbone` 改为 `resnet18` 且 `vision.pretrained` 改为 `false`，减少外部下载依赖。

Qwen 配置：

- 仅跑工程 smoke test 时，`parse_instruction.py --client rule`、`build_action_prior_cache.py --client uniform`、`run_eval_aerialvln.py --vlm-client uniform` 不需要 API key。
- 正式使用 `qwen_api` 时，需要在 `models/qwen_config.py` 写入 `QWEN_API_KEY`、模型名和 API base URL。

部署后先做检查：

```bash
python scripts/check_v0_setup.py --eval-split val_unseen --port 30000 --gpu-ids 0
```

完整在线 eval 还需要启动 simulator server：

```bash
python airsim_plugin/AirVLNSimulatorServerTool.py --gpus 0 --port 30000
```

再开另一个终端运行 `run_eval_aerialvln.py`，并保证 `--simulator_tool_port` 与 server 的 `--port` 一致。

### 3.2 GPU/CUDA 设备选择

所有会使用 PyTorch CUDA 的 V0 脚本都支持：

```text
--device cuda 或 --device cuda:0 或 --device cpu
--gpu-ids 0 或 --gpu-ids 0,1
```

含义：

- `--gpu-ids` 是物理显卡编号，代码会在进程内设置 `CUDA_VISIBLE_DEVICES`，限制这个脚本只能看到指定显卡。
- `--device cuda` 会使用可见 GPU 列表中的第一张卡；如果同时传 `--gpu-ids 2,3`，实际 PyTorch 设备会显示为 `cuda:0`，对应物理 GPU 2。
- `--device cuda:1` 表示使用当前可见 GPU 列表中的第二张卡；如果没有传 `--gpu-ids`，它对应系统物理 GPU 1。
- 如果机器没有可用 CUDA，脚本会自动 fallback 到 CPU，并在日志和 `summary.json` 里写出 `device_note=cuda_unavailable_fallback_cpu`。
- AirSim/ENVs 渲染进程使用 `airsim_plugin/AirVLNSimulatorServerTool.py --gpus 0,1` 或等价的 `--gpu-ids 0,1` 指定渲染显卡。

检查显卡编号：

```bash
nvidia-smi
python scripts/check_v0_setup.py --gpu-ids 0,1 --port 30000
```

## 4. 视觉骨干放置

`configs/model.yaml` 默认使用 DINOv2-S：

```yaml
vision:
  backbone: dinov2_s
  pretrained: true
  freeze: true
  assets_dir: vision_backbones
  dinov2_repo: vision_backbones/dinov2
  torch_hub_dir: vision_backbones/torch_hub
  resnet_weights: ""
```

模型别名：

```text
dinov2_s -> torch.hub model: dinov2_vits14, output dim 384
dinov2_b -> torch.hub model: dinov2_vitb14, output dim 768
resnet50 / resnet18 -> torchvision ResNet
```

本仓库已创建单独存放目录：

```text
AeroForesee/
  vision_backbones/
    dinov2/       # 可放 DINOv2 本地仓库
    resnet/       # 可放 ResNet 权重，例如 resnet50.pth
    torch_hub/    # torch hub cache 和 DINOv2 checkpoints
```

有网复现：

- 可以保持默认配置。
- 首次运行时会通过 `torch.hub` 拉取 `facebookresearch/dinov2`。
- cache/checkpoint 写入 `vision_backbones/torch_hub/`。

离线复现：

```bash
cd AirVLN_ws/AeroForesee
git clone https://github.com/facebookresearch/dinov2.git vision_backbones/dinov2
```

然后把 DINOv2 权重放到：

```text
vision_backbones/torch_hub/checkpoints/
```

如果使用 ResNet 本地权重：

```yaml
vision:
  backbone: resnet50
  resnet_weights: vision_backbones/resnet/resnet50.pth
```

`.gitignore` 已忽略 `vision_backbones/` 下的大模型文件，只提交 `.gitkeep` 和说明文档。

## 5. Qwen 配置

所有 LLM/VLM 调用统一通过 `models/qwen_config.py` 配置：

```python
QWEN_API_KEY = "PASTE_YOUR_QWEN_API_KEY_HERE"
QWEN_API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_TEXT_MODEL = "qwen-plus"
QWEN_VLM_MODEL = "qwen-vl-plus"

QWEN_LOCAL_LLM_COMMAND = "python tools/qwen_local_llm.py"
QWEN_LOCAL_VLM_COMMAND = "python tools/qwen_local_vlm.py"
```

客户端选择：

```text
rule       只用于 milestone parser smoke test
uniform    只用于 action prior smoke test
qwen_api   使用 Qwen / DashScope API
qwen_local 使用本地命令
```

正式实验建议：

- `parse_instruction.py --client qwen_api`
- `build_action_prior_cache.py --client qwen_api`
- `run_eval_aerialvln.py --vlm-client qwen_api`

没有 Qwen key 时可以先用 `rule/uniform` 跑通 V0 工程闭环。

## 6. 运行前检查

放好 `DATA/` 和 `ENVs/` 后，先运行：

```bash
cd AirVLN_ws/AeroForesee
python scripts/check_v0_setup.py --eval-split val_unseen --port 30000
```

检查内容：

- `../DATA` 是否存在
- `../DATA/data/aerialvln/*.json` 是否存在并能解析
- `../DATA/data/aerialvln-s/*.json` 是否存在并能解析
- `../ENVs/env_*` 是否存在
- `configs/model.yaml` / `configs/fuser.yaml` 是否存在
- DINOv2 / Torch Hub / ResNet 路径是否配置
- Qwen key 是否还是 placeholder
- AirSim simulator server 端口是否打开

严格模式：

```bash
python scripts/check_v0_setup.py --strict
```

如果 `--strict` 因 Qwen key 或端口未开而失败，先判断你是否只是跑 smoke test。`rule/uniform` smoke test 不需要 Qwen key；完整 eval 需要先启动仿真服务。

## 7. V0 全流程

下面命令都从 `AirVLN_ws/AeroForesee` 执行。

### 7.1 Milestone 解析

Smoke test：

```bash
python preprocess/parse_instruction.py \
  --dataset-json ../DATA/data/aerialvln-s/train.json \
  --output data/instruction_plan.jsonl \
  --bad-output data/bad_cases.jsonl \
  --client rule
```

正式 Qwen：

```bash
python preprocess/parse_instruction.py \
  --dataset-json ../DATA/data/aerialvln-s/train.json \
  --output data/instruction_plan.jsonl \
  --bad-output data/bad_cases.jsonl \
  --client qwen_api \
  --max-retries 1
```

结果：

```text
data/instruction_plan.jsonl
data/bad_cases.jsonl
DATA/v0/diagnostics/parse_instruction/summary.json
DATA/v0/diagnostics/parse_instruction/events.jsonl
DATA/v0/diagnostics/parse_instruction/milestone_count_distribution.png
```

schema 对纯动作 milestone 做了宽容处理：如果 Qwen 返回的 milestone 是 `turn_left`、`go_up`、`land`、`stop` 等动作语义，且没有显式 `landmarks` 和 `spatial_relation`，`schemas/milestone_schema.py` 会自动补一个弱 `spatial_relation="toward"`。这样可以减少因为原始指令缺少地标而被丢弃的样本，同时仍然保留 3-8 个 milestone、连续 `mid` 和至少 1 个 `verification_cues` 的校验。

### 7.2 构造 step-window

```bash
python preprocess/build_step_windows.py \
  --dataset-json ../DATA/data/aerialvln-s/train.json \
  --instruction-plan data/instruction_plan.jsonl \
  --output data/step_windows/train.jsonl \
  --history-len 16 \
  --max-keyframes 8 \
  --keyframe-interval 4
```

结果：

```text
data/step_windows/train.jsonl
DATA/v0/diagnostics/build_step_windows/summary.json
DATA/v0/diagnostics/build_step_windows/events.jsonl
DATA/v0/diagnostics/build_step_windows/episode_step_count_distribution.png
```

说明：

- 若原始 dataset JSON 含 `rgb_paths/image_paths/images/frames` 字段，step-window 会记录 `rgb_path`、`next_rgb_path`、`keyframe_rgb_paths`。
- 若没有这些字段，训练图像和 latent target 会使用可控 fallback，并在 summary 中记录。

### 7.3 构造 action prior cache

Smoke test：

```bash
python preprocess/build_action_prior_cache.py \
  --step-windows data/step_windows/train.jsonl \
  --output data/action_prior_cache/train.jsonl \
  --client uniform
```

正式 Qwen：

```bash
python preprocess/build_action_prior_cache.py \
  --step-windows data/step_windows/train.jsonl \
  --output data/action_prior_cache/train.jsonl \
  --image-root ../DATA/data/aerialvln-s \
  --client qwen_api \
  --preview-count 20
```

`build_action_prior_cache.py` 会读取 `rgb_path` 和 `keyframe_rgb_paths`，把“当前 RGB + 最近关键帧 + milestone/progress 摘要 + 官方动作表”一起送给 VLM；`summary.json` 里的 `current_rgb_loaded` 和 `keyframes_loaded` 用来检查视觉证据是否真的接入。

结果：

```text
data/action_prior_cache/train.jsonl
DATA/v0/diagnostics/build_action_prior_cache/summary.json
DATA/v0/diagnostics/build_action_prior_cache/prior_preview.jsonl
DATA/v0/diagnostics/build_action_prior_cache/top_prior_action_distribution.png
```

### 7.4 构造 rollout labels

```bash
python preprocess/build_rollout_labels.py \
  --step-windows data/step_windows/train.jsonl \
  --output data/rollout_labels/train.jsonl
```

如果 step-window 里有 `reference_pose/next_reference_pose`，label 构造会用 AirSim 官方动作尺度做一阶 local world-model：对每个候选动作模拟下一 pose，再按 reference segment 的 completion 增量、到下一参考点的偏离、反向/绕路代价生成 progress/cost。缺少 reference pose 时才回退到启发式 consequence label。`summary.json` 里的 `geometric_action_labels` 和 `heuristic_action_labels` 用来检查实际覆盖率。

结果：

```text
data/rollout_labels/train.jsonl
DATA/v0/diagnostics/build_rollout_labels/summary.json
DATA/v0/diagnostics/build_rollout_labels/positive_progress_by_action.png
DATA/v0/diagnostics/build_rollout_labels/average_cost_by_action.png
```

### 7.5 构造 latent targets

```bash
python preprocess/build_latent_targets.py \
  --step-windows data/step_windows/train.jsonl \
  --output-dir data/latent_targets/train \
  --index-output data/latent_targets/train_index.jsonl \
  --model-config configs/model.yaml \
  --image-root ../DATA/data/aerialvln-s \
  --device cuda \
  --gpu-ids 0
```

结果：

```text
data/latent_targets/train/*.pt
data/latent_targets/train_index.jsonl
DATA/v0/diagnostics/build_latent_targets/summary.json
```

`summary.json` 重点看：

```text
encoded_from_images
missing_images_zero_fallback
```

正式实验希望 `encoded_from_images` 尽量大；如果全部 fallback，说明 dataset JSON 没提供离线图像路径，或 `--image-root` 不对。

### 7.6 训练 evaluator

```bash
python training/train_action_evaluator.py \
  --step-windows data/step_windows/train.jsonl \
  --rollout-labels data/rollout_labels/train.jsonl \
  --action-prior-cache data/action_prior_cache/train.jsonl \
  --latent-index data/latent_targets/train_index.jsonl \
  --image-root ../DATA/data/aerialvln-s \
  --model-config configs/model.yaml \
  --output-dir DATA/v0/checkpoints/action_evaluator \
  --device cuda \
  --gpu-ids 0 \
  --batch-size 16 \
  --epochs 10 \
  --lr 0.00025 \
  --prior-loss-weight 0.5
```

训练时 `prev_latent` 会从上一时刻 latent target 递推读取；`--prior-loss-weight` 会把离线 action prior 作为 progress/cost supervision 的样本权重使用。

结果：

```text
DATA/v0/checkpoints/action_evaluator/ckpt_epoch_*.pth
DATA/v0/checkpoints/action_evaluator/ckpt_last.pth
DATA/v0/diagnostics/train_action_evaluator/training_log.jsonl
DATA/v0/diagnostics/train_action_evaluator/loss_curve.png
DATA/v0/diagnostics/train_action_evaluator/summary.json
```

### 7.7 启动仿真服务

完整 eval 前需要 AirSim/ENVs 服务。

```bash
cd AirVLN_ws/AeroForesee
python airsim_plugin/AirVLNSimulatorServerTool.py --gpus 0 --port 30000
```

多张渲染卡可以写成 `--gpus 0,1`，场景会按顺序轮询分配到这些显卡；`--gpu-ids 0,1` 是同义参数。

检查：

```bash
python scripts/check_v0_setup.py --port 30000
```

如果提示端口 `30000` 已打开，说明服务在监听。若端口冲突，换一个端口，并在 eval 时传相同的 `--simulator_tool_port`。

### 7.8 Eval 闭环

```bash
python inference/run_eval_aerialvln.py \
  --v0-checkpoint DATA/v0/checkpoints/action_evaluator/ckpt_last.pth \
  --model-config configs/model.yaml \
  --fuser-config configs/fuser.yaml \
  --eval-output DATA/v0/eval/aerialvln_val_unseen.json \
  --diagnostics-dir DATA/v0/diagnostics/eval_aerialvln \
  --instruction-plan data/instruction_plan.jsonl \
  --vlm-client qwen_api \
  --device cuda \
  --gpu-ids 0 \
  --score-preview-steps 5 \
  --stop-completion-threshold 0.35 \
  --batchSize 1 \
  --EVAL_DATASET val_unseen \
  --EVAL_NUM -1 \
  --maxAction 500 \
  --collect_type TF \
  --simulator_tool_port 30000
```

结果：

```text
DATA/v0/eval/aerialvln_val_unseen.json
DATA/v0/diagnostics/eval_aerialvln/summary.json
DATA/v0/diagnostics/eval_aerialvln/eval_steps.jsonl
DATA/v0/diagnostics/eval_aerialvln/eval_metrics.png
DATA/v0/diagnostics/eval_aerialvln/planner_loop/step_decisions.jsonl
DATA/v0/diagnostics/eval_aerialvln/planner_loop/episode_*_step_*_scores.png
```

`run_eval_aerialvln.py` 会强制启用：

```text
--run_type eval --ablate_depth
```

在线 planner 里 milestone 不再只由 env progress 映射。`MilestoneProgressController` 会维护每个 episode 的当前 milestone 和 completion，并用 VLM action prior 的 STOP 证据、已选动作、evaluator progress gain 以及可用的全局 progress 来推进 milestone；`step_decisions.jsonl` 会同时记录当前和更新后的 milestone 状态。

## 8. 参数说明

### 8.1 `configs/base.yaml`

| 参数 | 默认值 | 作用 | 常见修改 |
| --- | --- | --- | --- |
| `workspace_root` | `..` | 工作区根目录，对你的结构就是 `AirVLN_ws` | 目录变化时改 |
| `dataset_root` | `../DATA/data` | 数据集根目录 | 数据不在相邻 `DATA` 时改 |
| `envs_root` | `../ENVs` | AirSim 环境根目录 | 环境不在相邻 `ENVs` 时改 |
| `output_root` | `DATA/v0` | V0 日志、结果和 checkpoint 输出根目录 | 多实验可改成 `DATA/v0_exp_x` |
| `device` | `cuda` | 默认 PyTorch 设备 | CPU smoke test 改成 `cpu` |
| `gpu_ids` | `null` | 默认 GPU 可见列表；命令行 `--gpu-ids` 优先生效 | 固定机器可写 `0` 或 `0,1` |
| `num_workers` | `0` | DataLoader worker 数 | Linux 可增大，Windows 建议从 0 开始 |

### 8.2 `configs/model.yaml`

| 参数 | 默认值 | 作用 | 常见修改 |
| --- | --- | --- | --- |
| `vision.backbone` | `dinov2_s` | 视觉骨干 | `dinov2_b/resnet50/resnet18` |
| `vision.pretrained` | `true` | 加载预训练权重 | 无网 smoke test 可设 `false` |
| `vision.freeze` | `true` | 冻结视觉骨干 | 端到端微调设 `false` |
| `vision.assets_dir` | `vision_backbones` | 视觉骨干统一存放目录 | 通常不改 |
| `vision.dinov2_repo` | `vision_backbones/dinov2` | DINOv2 本地仓库 | 离线时 clone 到这里 |
| `vision.torch_hub_dir` | `vision_backbones/torch_hub` | Torch Hub cache/checkpoints | 多机器复现时固定 |
| `vision.resnet_weights` | 空 | ResNet 本地权重 | `vision_backbones/resnet/resnet50.pth` |
| `vision.image_height/width` | `224/224` | RGB 输入尺寸 | 与 AirVLN 图像保持一致 |
| `history.max_keyframes` | `8` | 历史关键帧数量 | 显存不足时降低 |
| `history.interval` | `4` | 固定关键帧间隔 | 历史太稀可降低 |
| `trajectory.history_len` | `16` | 动作/pose/fallback 历史长度 | 振荡多时增大 |
| `model.hidden_dim` | `512` | latent token 维度 | 显存不足时降低 |
| `model.num_heads` | `8` | Transformer heads | 需整除 hidden_dim |
| `model.num_layers` | `2` | causal evaluator 层数 | 欠拟合可增大 |
| `model.dropout` | `0.1` | dropout | 过拟合可增大 |
| `model.max_milestones` | `8` | milestone id 上限 | 与 parser 3-8 对齐 |
| `training.batch_size` | `16` | 默认 batch size | 显存不足时降低 |
| `training.epochs` | `10` | 默认 epoch | 按 loss 曲线调整 |
| `training.lr` | `0.00025` | 学习率 | loss 震荡时降低 |
| `training.*_loss_weight` | `1.0` | 三类 loss 权重 | 按训练曲线调整 |

### 8.3 `configs/fuser.yaml`

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `w_progress` | `1.0` | progress gain 权重 |
| `w_cost` | `0.7` | cost 惩罚权重 |
| `w_prior` | `0.6` | VLM action prior 权重 |
| `fallback.low_progress_threshold` | `0.05` | 低进展阈值 |
| `fallback.low_score_threshold` | `0.05` | 低综合分阈值 |
| `fallback.progress_patience` | `3` | 连续低进展步数 |
| `fallback.repeated_turn_patience` | `2` | 连续转向步数 |
| `fallback.conservative_actions` | `MOVE_FORWARD, STOP` | fallback 优先动作 |

### 8.4 命令行参数总览

下面只列仓库当前实际解析的命令行参数。没有出现在表里的配置项通常来自 `configs/*.yaml`，或是 `run_eval_aerialvln.py` 透传给原 AirVLN 参数解析器的 legacy 参数。

#### `scripts/check_v0_setup.py`

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--workspace-root` | `None` | 工作区根目录，里面应包含 `AeroForesee/`、`DATA/`、`ENVs/`，不传时取项目目录的父级 |
| `--project-dir` | `None` | AeroForesee 项目目录，不传时自动取脚本所在仓库根目录 |
| `--model-config` | `configs/model.yaml` | 要检查的模型配置文件 |
| `--fuser-config` | `configs/fuser.yaml` | 要检查的融合器配置文件 |
| `--eval-split` | `val_unseen` | 检查 `../DATA/data/aerialvln/{split}.json` 是否存在 |
| `--port` | `30000` | 检查本地 simulator server 端口是否已监听 |
| `--gpu-ids` | `None` | 逗号分隔的物理 GPU 编号，用 `nvidia-smi` 验证是否存在 |
| `--strict` | `False` | 有 warning 也返回非零退出码，适合 CI 或严格部署检查 |

#### `preprocess/parse_instruction.py`

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--dataset-json` | 必填 | 输入 dataset JSON，例如 `../DATA/data/aerialvln-s/train.json` |
| `--output` | `data/instruction_plan.jsonl` | 成功解析出的 instruction-to-milestone 结果 |
| `--bad-output` | `data/bad_cases.jsonl` | 多次解析或 schema 校验失败的样本 |
| `--prompt` | `prompts/milestone_prompt.txt` | milestone parser 的 system prompt |
| `--client` | `rule` | LLM 客户端，`rule` 用于 smoke test，`qwen_api` 用 DashScope API，`qwen_local` 调本地命令 |
| `--max-retries` | `1` | 单条指令解析失败后的重试次数，总尝试次数是 `max_retries + 1` |
| `--diagnostics-dir` | `DATA/v0/diagnostics/parse_instruction` | summary、events 和 milestone 分布图输出目录 |

#### `preprocess/build_step_windows.py`

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--dataset-json` | 必填 | 输入 dataset JSON，读取 actions、reference_path、图像路径等字段 |
| `--instruction-plan` | `data/instruction_plan.jsonl` | milestone 解析结果；缺失时用 `default_plan()` 生成 3 段 follow plan |
| `--output` | `data/step_windows/train.jsonl` | 输出逐步训练样本窗口 |
| `--history-len` | `16` | 每个样本保留的历史 action 和 pose delta 长度，不足时前置补零 |
| `--max-keyframes` | `8` | 每个样本最多保留的历史关键帧数量 |
| `--keyframe-interval` | `4` | 固定间隔选择关键帧时的步长 |
| `--diagnostics-dir` | `DATA/v0/diagnostics/build_step_windows` | summary、events 和 episode step 分布图输出目录 |

#### `preprocess/build_action_prior_cache.py`

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--step-windows` | 必填 | `build_step_windows.py` 生成的 JSONL |
| `--output` | `data/action_prior_cache/train.jsonl` | 每个 sample 的 action prior 分数缓存 |
| `--prompt` | `prompts/action_prior_prompt.txt` | VLM action prior 的 prompt |
| `--client` | `uniform` | VLM 客户端，`uniform` 输出均匀先验，`qwen_api` 用 DashScope VLM，`qwen_local` 调本地命令 |
| `--image-root` | `None` | 相对图像路径的根目录，例如 `../DATA/data/aerialvln-s` |
| `--max-keyframes` | `2` | 读取 step-window 中最近多少张关键帧；当前送入 prior 时还会截取最后 2 张 |
| `--diagnostics-dir` | `DATA/v0/diagnostics/build_action_prior_cache` | summary、prior preview 和 top action 分布图输出目录 |
| `--preview-count` | `20` | 写入 `prior_preview.jsonl` 并打印 preview 的前 N 个样本数 |

#### `preprocess/build_rollout_labels.py`

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--step-windows` | 必填 | 输入 step-window JSONL |
| `--output` | `data/rollout_labels/train.jsonl` | 输出每个 sample、每个候选动作的 progress/cost label |
| `--diagnostics-dir` | `DATA/v0/diagnostics/build_rollout_labels` | summary、positive progress 和 average cost 图输出目录 |

#### `preprocess/build_latent_targets.py`

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--step-windows` | 必填 | 输入 step-window JSONL，使用其中的 `next_rgb_path` 构造 future latent |
| `--output-dir` | `data/latent_targets/train` | 每个 sample 的 `.pt` latent target 输出目录 |
| `--index-output` | `data/latent_targets/train_index.jsonl` | sample id 到 latent target 文件的索引 |
| `--model-config` | `configs/model.yaml` | 读取视觉骨干、输入尺寸和 hidden_dim |
| `--image-root` | `None` | 相对图像路径根目录 |
| `--device` | `cuda` | PyTorch 设备，支持 `cuda`、`cuda:0`、`cpu` |
| `--gpu-ids` | `None` | 限制本进程可见的物理 GPU，例如 `0` 或 `0,1` |
| `--token-dim` | `None` | 覆盖 latent token 维度；不传时使用 `model.hidden_dim` |
| `--diagnostics-dir` | `DATA/v0/diagnostics/build_latent_targets` | summary 输出目录 |
| `--preview-count` | `10` | 打印前 N 个 latent target 构造 preview |

#### `training/train_state_encoder.py`

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--model-config` | `configs/model.yaml` | 读取 state builder 初始化所需的模型结构 |
| `--output` | `DATA/v0/checkpoints/state_builder_init.pth` | 仅保存初始化 state builder checkpoint 的路径 |

#### `training/train_action_evaluator.py`

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--step-windows` | 必填 | 输入 step-window JSONL |
| `--rollout-labels` | 必填 | 输入 rollout progress/cost label JSONL |
| `--action-prior-cache` | `None` | 可选 action prior cache；传入后 prior 会参与样本权重 |
| `--latent-index` | `None` | 可选 future latent target 索引；缺失时 dataset 使用 fallback |
| `--image-root` | `None` | 相对 RGB 路径根目录 |
| `--model-config` | `configs/model.yaml` | 模型、视觉骨干、history 和训练默认配置 |
| `--output-dir` | `DATA/v0/checkpoints/action_evaluator` | checkpoint 输出目录 |
| `--device` | `cuda` | PyTorch 训练设备 |
| `--gpu-ids` | `None` | 限制本训练进程可见的物理 GPU |
| `--batch-size` | `16` | DataLoader batch size |
| `--epochs` | `10` | 训练轮数 |
| `--lr` | `0.00025` | AdamW 学习率 |
| `--num-workers` | `0` | DataLoader worker 数，Windows 建议从 0 开始 |
| `--progress-loss-weight` | `1.0` | progress gain MSE loss 权重 |
| `--cost-loss-weight` | `1.0` | cost MSE loss 权重 |
| `--latent-loss-weight` | `1.0` | next latent MSE loss 权重 |
| `--prior-loss-weight` | `0.5` | 用 cached action prior 放大 progress/cost supervision 权重的系数 |
| `--diagnostics-dir` | `DATA/v0/diagnostics/train_action_evaluator` | training log、loss curve 和 summary 输出目录 |
| `--preview-batches` | `2` | 每个 epoch 打印前 N 个 batch 的 loss 明细 |

#### `training/tune_fuser.py`

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--output` | `configs/fuser.yaml` | 写出的融合配置路径 |
| `--w-progress` | `0.5 1.0 1.5` | progress gain 权重候选列表 |
| `--w-cost` | `0.3 0.7 1.0` | cost 惩罚权重候选列表 |
| `--w-prior` | `0.0 0.3 0.6` | VLM prior 权重候选列表 |

当前 `tune_fuser.py` 是占位校准脚本，会写入网格中的第一个组合；真正按验证集指标搜索时需要替换内部 score function。

#### `airsim_plugin/AirVLNSimulatorServerTool.py`

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--gpus` | `0` | AirSim 场景渲染使用的物理 GPU 编号，逗号分隔 |
| `--gpu-ids` | `None` | `--gpus` 的同义参数；如果传入则优先生效 |
| `--port` | `30000` | simulator RPC server 监听端口 |

#### `inference/run_eval_aerialvln.py`

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--v0-checkpoint` | `None` | evaluator checkpoint；不传则使用随机初始化模型，只适合 smoke test |
| `--model-config` | `configs/model.yaml` | 模型结构和视觉骨干配置 |
| `--fuser-config` | `configs/fuser.yaml` | 决策融合和 fallback 配置 |
| `--vlm-client` | `uniform` | 在线 action prior 客户端，支持 `uniform`、`qwen_api`、`qwen_local` |
| `--device` | `cuda` | planner/evaluator 使用的 PyTorch 设备 |
| `--gpu-ids` | `None` | 限制 V0 planner 进程可见的物理 GPU |
| `--eval-output` | `DATA/v0/eval/results.json` | eval 汇总 JSON 输出路径 |
| `--diagnostics-dir` | `DATA/v0/diagnostics/eval_aerialvln` | eval summary、step log、metric 图和 planner 诊断输出目录 |
| `--instruction-plan` | `data/instruction_plan.jsonl` | 在线 eval 时合并到 episode 的 milestone plan |
| `--stop-completion-threshold` | `0.35` | milestone completion 低于该值时抑制 STOP prior |
| `--score-preview-steps` | `5` | 每个 episode 前 N 步保存候选动作分数 PNG |

`run_eval_aerialvln.py` 使用 `parse_known_args()`，不认识的参数会继续交给 `src/common/param.py`。它还会强制追加 `--run_type eval --ablate_depth`，保证 V0 使用 eval 模式和 front-view RGB-only 设置。

#### `inference/run_eval_openfly.py`

当前无可配置参数。该入口在 V0 中会直接抛出 `NotImplementedError`，OpenFly transfer 计划留到 V2。

### 8.5 `run_eval_aerialvln.py` 可透传的 AirVLN 参数

这些参数来自 `src/common/param.py`。V0 eval 常用的是 `--batchSize`、`--EVAL_DATASET`、`--EVAL_NUM`、`--maxAction`、`--simulator_tool_port`；其余多为保留的 AirVLN baseline/训练参数。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--project_prefix` | 当前目录父级 | 工作区根目录，默认从 `AirVLN_ws/AeroForesee` 推到 `AirVLN_ws` |
| `--run_type` | `train`，V0 eval 强制为 `eval` | 原 AirVLN 运行模式，限定为 `collect/train/eval` |
| `--policy_type` | `v0` | 策略类型；AeroForesee 固定为 `v0` |
| `--collect_type` | `TF` | 环境监督模式，支持 `TF` 或 `dagger` |
| `--name` | `default` | 实验名，用于 legacy 日志路径 |
| `--maxInput` | `300` | 最大指令 token/长度相关 legacy 参数 |
| `--maxAction` | `500` | 单个 episode 最大动作步数 |
| `--dagger_it` | `1` | DAgger 迭代编号，V0 eval 通常不使用 |
| `--epochs` | `10` | legacy 训练 epoch，V0 eval 通常不使用 |
| `--lr` | `0.00025` | legacy 学习率，V0 eval 通常不使用 |
| `--batchSize` | `8` | AirSim 并行 episode 数；调试建议设为 `1` |
| `--trainer_gpu_device` | `0` | legacy trainer GPU 编号，V0 PyTorch 设备使用 `--gpu-ids/--device` |
| `--Image_Height_RGB` | `224` | 环境 RGB 图像高度 |
| `--Image_Width_RGB` | `224` | 环境 RGB 图像宽度 |
| `--Image_Height_DEPTH` | `256` | 环境 depth 图像高度；V0 eval 强制 ablate depth |
| `--Image_Width_DEPTH` | `256` | 环境 depth 图像宽度；V0 eval 强制 ablate depth |
| `--inflection_weight_coef` | `1.9` | legacy imitation learning inflection 权重 |
| `--nav_graph_path` | `../DATA/data/disceret/processed/nav_graph_10` | graph progress/shortest path 相关文件路径 |
| `--token_dict_path` | `../DATA/data/disceret/processed/token_dict_10` | legacy tokenizer 字典路径 |
| `--vertices_path` | `../DATA/data/disceret/scene_meshes` | legacy scene mesh vertices 路径 |
| `--dagger_mode_load_scene` | `[]` | DAgger 模式指定加载的场景列表 |
| `--dagger_update_size` | `8000` | DAgger 单轮更新样本量 |
| `--dagger_mode` | `end` | DAgger 模式，支持 `end/middle/nearest` |
| `--dagger_p` | `1.0` | DAgger teacher forcing 概率 |
| `--TF_mode_load_scene` | `[]` | TF 模式指定加载的场景列表 |
| `--ablate_instruction` | `False` | 是否屏蔽语言指令 |
| `--ablate_rgb` | `False` | 是否屏蔽 RGB |
| `--ablate_depth` | `False`，V0 eval 强制为 `True` | 是否屏蔽 depth |
| `--SEQ2SEQ_use_prev_action` | `False` | legacy seq2seq 是否使用上一动作 |
| `--PROGRESS_MONITOR_use` | `False` | legacy progress monitor 开关 |
| `--PROGRESS_MONITOR_alpha` | `1.0` | legacy progress monitor loss 权重 |
| `--EVAL_CKPT_PATH_DIR` | `None` | legacy eval checkpoint 目录，V0 checkpoint 用 `--v0-checkpoint` |
| `--EVAL_DATASET` | `val_unseen` | eval split，对应 `../DATA/data/aerialvln/{split}.json` |
| `--EVAL_NUM` | `-1` | eval episode 数，`-1` 表示全量 |
| `--EVAL_GENERATE_VIDEO` | `False` | legacy 视频输出开关 |
| `--rgb_encoder_use_place365` | `False` | legacy RGB encoder 选项 |
| `--tokenizer_use_bert` | `False` | legacy tokenizer 选项 |
| `--simulator_tool_port` | `30000` | 连接 simulator server 的端口，必须与 server `--port` 一致 |
| `--DDP_MASTER_PORT` | `20000` | legacy DDP master port |
| `--continue_start_from_dagger_it` | `None` | legacy 断点续训的 DAgger 轮次 |
| `--continue_start_from_checkpoint_path` | `None` | legacy 断点续训 checkpoint 路径 |
| `--vlnbert` | `prevalent` | legacy VLN-BERT 选项，当前 parser 用法保留了历史默认值 |
| `--featdropout` | `0.4` | legacy feature dropout 选项，当前 parser 用法保留了历史默认值 |
| `--action_feature` | `32` | legacy action feature 维度，当前 parser 用法保留了历史默认值 |

## 9. 调参指南

优先顺序：

1. 先用 `rule/uniform` 跑通工程和日志。
2. 再切到 `qwen_api`，检查 action prior 是否合理。
3. 固定模型，调 `configs/fuser.yaml`。
4. 再调 fallback。
5. 最后调模型容量和历史长度。

常见现象：

| 现象 | 优先检查 | 调参建议 |
| --- | --- | --- |
| 过早 STOP | `step_decisions.jsonl`, `episode_*_scores.png` | 降低 `w_prior` 或降低 `--stop-completion-threshold` |
| 一直前进但不转向 | action prior 是否塌缩 | 提高 `w_prior`，检查 Qwen prompt 和 `prior_preview.jsonl` |
| 来回转向 | `fallback.repeated_turn_patience` | 降低 patience，提高 `w_cost` |
| fallback 太频繁 | progress/cost 分布 | 降低 `low_progress_threshold` 或 `low_score_threshold` |
| 路径很长 | `average_cost_by_action.png` | 提高 `w_cost` |
| loss 不降 | `loss_curve.png`, label 分布 | 降低 `lr`，检查 `rollout_labels` 和 latent target |
| `encoded_from_images=0` | `build_latent_targets/summary.json` | 检查 step-window 是否有图像路径和 `--image-root` |
| CUDA OOM | 终端日志、`summary.json` 里的 `device/gpu_ids` | 降低 `--batch-size`、`history.max_keyframes` 或 `model.hidden_dim`，也可换 `--gpu-ids` 到空闲显卡 |
| 用错显卡 | `summary.json` 里的 `cuda_visible_devices` | 显式传 `--gpu-ids 0` 或 `--gpu-ids 0,1`；AirSim server 也同步改 `--gpus` |

## 10. 日志、结果和可视化

所有阶段都会打印阶段日志，例如：

```text
[parse_instruction] parsed | instruction_id=... milestones=...
[build_step_windows] episode_done | instruction_id=... steps=...
[build_latent_targets] done | latent_targets=... encoded=... missing_images=...
[train_action_evaluator] epoch_done | epoch=... loss=...
[planner_loop] step_decision | episode_id=... step=... action=... fallback=...
[run_eval_aerialvln] episode_done | episode_id=... success=... ndtw=...
```

所有图像可视化只保存 PNG：

```text
DATA/v0/diagnostics/parse_instruction/milestone_count_distribution.png
DATA/v0/diagnostics/build_step_windows/episode_step_count_distribution.png
DATA/v0/diagnostics/build_action_prior_cache/top_prior_action_distribution.png
DATA/v0/diagnostics/build_rollout_labels/positive_progress_by_action.png
DATA/v0/diagnostics/build_rollout_labels/average_cost_by_action.png
DATA/v0/diagnostics/train_action_evaluator/loss_curve.png
DATA/v0/diagnostics/eval_aerialvln/eval_metrics.png
DATA/v0/diagnostics/eval_aerialvln/planner_loop/episode_*_step_*_scores.png
```

关键 JSON/JSONL：

```text
DATA/v0/diagnostics/*/summary.json
DATA/v0/diagnostics/*/events.jsonl
DATA/v0/diagnostics/build_action_prior_cache/prior_preview.jsonl
DATA/v0/diagnostics/train_action_evaluator/training_log.jsonl
DATA/v0/diagnostics/eval_aerialvln/eval_steps.jsonl
DATA/v0/diagnostics/eval_aerialvln/planner_loop/step_decisions.jsonl
DATA/v0/eval/*.json
```

其中 `build_latent_targets`、`train_action_evaluator` 和 `eval_aerialvln` 的 `summary.json` 会记录：

```text
device
gpu_ids
cuda_visible_devices
device_note
```

最终 eval output 格式：

```json
{
  "summary": {
    "distance_to_goal": 0.0,
    "success": 0.0,
    "ndtw": 0.0,
    "sdtw": 0.0,
    "path_length": 0.0,
    "oracle_success": 0.0,
    "steps_taken": 0.0,
    "num_episodes": 0
  },
  "episodes": {
    "episode_id": {
      "success": 0.0,
      "ndtw": 0.0,
      "trajectory": []
    }
  }
}
```

## 11. 项目文件实现对照

| 项目文件要求 | 代码落点 | 当前实现 |
| --- | --- | --- |
| Instruction-to-Milestone Parser | `prompts/milestone_prompt.txt`, `schemas/milestone_schema.py`, `preprocess/parse_instruction.py` | 离线解析，3-8 milestone，字段校验，失败重试，bad case 输出 |
| Milestone-Aware State Builder | `models/vision_backbone.py`, `models/history_encoder.py`, `models/trajectory_encoder.py`, `models/state_builder.py` | 当前 RGB、关键帧、动作/位姿/fallback 历史、milestone/progress、prev latent |
| 历史关键帧规则 | `models/history_encoder.py`, `preprocess/build_step_windows.py` | 固定间隔、milestone 切换、连续转向、连续无进展 |
| VLM-Based Action Prior | `models/action_space.py`, `models/action_prior.py`, `prompts/action_prior_prompt.txt`, `preprocess/build_action_prior_cache.py` | 官方动作集合打分，非法动作删除，归一化，低 completion 抑制 STOP |
| Causal Latent Action Evaluator | `models/action_encoder.py`, `models/causal_latent_action_evaluator.py` | action token + causal Transformer，输出 progress/cost/next_latent |
| future latent supervision | `preprocess/build_latent_targets.py`, `training/v0_dataset.py` | 有下一帧图像时编码真实 latent；缺图像时记录 fallback |
| Step-wise Decision and Execution | `models/action_mask.py`, `models/decision_fuser.py`, `models/fallback.py`, `inference/planner_loop.py` | 合法动作过滤，手工融合，fallback，单步执行并更新 latent |
| AirVLN 仿真闭环与指标 | `inference/run_eval_aerialvln.py`, `src/vlnce_src/env.py` | `reset -> act -> makeActions -> get_obs -> update_measurements` |
| 日志和 PNG 可视化 | `utils/diagnostics.py` 和各阶段脚本 | print log, JSON/JSONL, PNG-only |

V0 按项目文件要求不实现：

- uncertainty
- alignment head
- OpenFly transfer
- learned decision head
- 复杂 recovery policy
- 原 AirVLN baseline 训练

## 12. 注意事项和检查方案

### DATA 检查

```bash
python scripts/check_v0_setup.py
```

如果缺 `../DATA/data/aerialvln/{split}.json`，完整 eval 不能跑。`run_eval_aerialvln.py` 的环境默认读取 full AerialVLN：

```text
../DATA/data/aerialvln/{EVAL_DATASET}.json
```

如果只做预处理 smoke test，可以先使用：

```text
../DATA/data/aerialvln-s/train.json
```

### ENVs 检查

```bash
ls ../ENVs
```

应该能看到 `env_*`。启动 server：

```bash
python airsim_plugin/AirVLNSimulatorServerTool.py --gpus 0 --port 30000
```

再次检查：

```bash
python scripts/check_v0_setup.py --port 30000
```

### GPU 检查

```bash
nvidia-smi
python scripts/check_v0_setup.py --gpu-ids 0,1 --port 30000
```

PyTorch 阶段和 AirSim 阶段需要分别指定 GPU：

```bash
python preprocess/build_latent_targets.py ... --device cuda --gpu-ids 0
python training/train_action_evaluator.py ... --device cuda --gpu-ids 0
python inference/run_eval_aerialvln.py ... --device cuda --gpu-ids 0 --simulator_tool_port 30000
python airsim_plugin/AirVLNSimulatorServerTool.py --gpus 0,1 --port 30000
```

如果 `summary.json` 里 `device` 变成 `cpu`，说明当前 Python 环境没有可用 CUDA 或 PyTorch CUDA 版本不匹配。先用 `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"` 检查。

### 端口冲突

若 `30000` 被占用：

```bash
python airsim_plugin/AirVLNSimulatorServerTool.py --gpus 0 --port 31000
python inference/run_eval_aerialvln.py ... --simulator_tool_port 31000
```

### Qwen 检查

如果使用 `qwen_api`，确认 `models/qwen_config.py` 不是 placeholder。否则先用：

```text
parse_instruction.py --client rule
build_action_prior_cache.py --client uniform
run_eval_aerialvln.py --vlm-client uniform
```

### 图像路径检查

看：

```text
DATA/v0/diagnostics/build_latent_targets/summary.json
```

如果 `missing_images_zero_fallback` 很高，说明离线 JSON 中没有图像路径或 `--image-root` 不对。这不影响工程跑通，但会影响训练质量。

### DINOv2 检查

本地仓库：

```text
vision_backbones/dinov2/hubconf.py
```

Torch Hub cache：

```text
vision_backbones/torch_hub/
```

如果无网且没有 DINOv2，可临时：

```yaml
vision:
  backbone: resnet18
  pretrained: false
```

### OpenMP / matplotlib 检查

Windows 下 torch 和 matplotlib 可能出现 OpenMP runtime 冲突。`utils/diagnostics.py` 已在绘图入口设置：

```text
KMP_DUPLICATE_LIB_OK=TRUE
MPLBACKEND=Agg
```

若仍有冲突，先单独运行：

```bash
python -c "from utils.diagnostics import save_bar_png; save_bar_png('DATA/v0/diagnostics/check.png','check',{'ok':1})"
```

## 13. 验收 Checklist

满足下面条件即可认为 V0 闭环可执行：

- `python scripts/check_v0_setup.py` 必要项通过。
- `parse_instruction` 生成 `instruction_plan.jsonl` 和 milestone PNG。
- `build_step_windows` 生成 step-window 和 episode step PNG。
- `build_action_prior_cache` 生成 action prior cache 和 top action PNG。
- `build_rollout_labels` 生成 progress/cost labels 和 PNG。
- `build_latent_targets` 生成 `.pt` latent targets 和 summary。
- `train_action_evaluator` 生成 `ckpt_last.pth` 和 `loss_curve.png`。
- simulator server 正常监听端口。
- `run_eval_aerialvln.py` 生成 eval JSON、`eval_metrics.png` 和 `step_decisions.jsonl`。
- `step_decisions.jsonl` 每步包含 action、fallback、fused scores。
- eval JSON 的 `summary` 包含 success、nDTW、sDTW、path length、oracle success、steps taken。
