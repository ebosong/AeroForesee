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
python preprocess/parse_instruction.py ^
  --dataset-json ../DATA/data/aerialvln-s/train.json ^
  --output data/instruction_plan.jsonl ^
  --bad-output data/bad_cases.jsonl ^
  --client rule
```

正式 Qwen：

```bash
python preprocess/parse_instruction.py ^
  --dataset-json ../DATA/data/aerialvln-s/train.json ^
  --output data/instruction_plan.jsonl ^
  --bad-output data/bad_cases.jsonl ^
  --client qwen_api ^
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

### 7.2 构造 step-window

```bash
python preprocess/build_step_windows.py ^
  --dataset-json ../DATA/data/aerialvln-s/train.json ^
  --instruction-plan data/instruction_plan.jsonl ^
  --output data/step_windows/train.jsonl ^
  --history-len 16 ^
  --max-keyframes 8 ^
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
python preprocess/build_action_prior_cache.py ^
  --step-windows data/step_windows/train.jsonl ^
  --output data/action_prior_cache/train.jsonl ^
  --client uniform
```

正式 Qwen：

```bash
python preprocess/build_action_prior_cache.py ^
  --step-windows data/step_windows/train.jsonl ^
  --output data/action_prior_cache/train.jsonl ^
  --client qwen_api ^
  --preview-count 20
```

结果：

```text
data/action_prior_cache/train.jsonl
DATA/v0/diagnostics/build_action_prior_cache/summary.json
DATA/v0/diagnostics/build_action_prior_cache/prior_preview.jsonl
DATA/v0/diagnostics/build_action_prior_cache/top_prior_action_distribution.png
```

### 7.4 构造 rollout labels

```bash
python preprocess/build_rollout_labels.py ^
  --step-windows data/step_windows/train.jsonl ^
  --output data/rollout_labels/train.jsonl
```

结果：

```text
data/rollout_labels/train.jsonl
DATA/v0/diagnostics/build_rollout_labels/summary.json
DATA/v0/diagnostics/build_rollout_labels/positive_progress_by_action.png
DATA/v0/diagnostics/build_rollout_labels/average_cost_by_action.png
```

### 7.5 构造 latent targets

```bash
python preprocess/build_latent_targets.py ^
  --step-windows data/step_windows/train.jsonl ^
  --output-dir data/latent_targets/train ^
  --index-output data/latent_targets/train_index.jsonl ^
  --model-config configs/model.yaml ^
  --image-root ../DATA/data/aerialvln-s ^
  --device cuda
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
python training/train_action_evaluator.py ^
  --step-windows data/step_windows/train.jsonl ^
  --rollout-labels data/rollout_labels/train.jsonl ^
  --action-prior-cache data/action_prior_cache/train.jsonl ^
  --latent-index data/latent_targets/train_index.jsonl ^
  --image-root ../DATA/data/aerialvln-s ^
  --model-config configs/model.yaml ^
  --output-dir DATA/v0/checkpoints/action_evaluator ^
  --device cuda ^
  --batch-size 16 ^
  --epochs 10 ^
  --lr 0.00025
```

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

检查：

```bash
python scripts/check_v0_setup.py --port 30000
```

如果提示端口 `30000` 已打开，说明服务在监听。若端口冲突，换一个端口，并在 eval 时传相同的 `--simulator_tool_port`。

### 7.8 Eval 闭环

```bash
python inference/run_eval_aerialvln.py ^
  --v0-checkpoint DATA/v0/checkpoints/action_evaluator/ckpt_last.pth ^
  --model-config configs/model.yaml ^
  --fuser-config configs/fuser.yaml ^
  --eval-output DATA/v0/eval/aerialvln_val_unseen.json ^
  --diagnostics-dir DATA/v0/diagnostics/eval_aerialvln ^
  --vlm-client qwen_api ^
  --device cuda ^
  --score-preview-steps 5 ^
  --stop-completion-threshold 0.35 ^
  --batchSize 1 ^
  --EVAL_DATASET val_unseen ^
  --EVAL_NUM -1 ^
  --maxAction 500 ^
  --collect_type TF ^
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

## 8. 参数说明

### 8.1 `configs/model.yaml`

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

### 8.2 `configs/fuser.yaml`

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

### 8.3 常用脚本参数

| 脚本 | 关键参数 |
| --- | --- |
| `parse_instruction.py` | `--dataset-json`, `--output`, `--bad-output`, `--client`, `--max-retries`, `--diagnostics-dir` |
| `build_step_windows.py` | `--dataset-json`, `--instruction-plan`, `--history-len`, `--max-keyframes`, `--keyframe-interval` |
| `build_action_prior_cache.py` | `--step-windows`, `--output`, `--client`, `--preview-count` |
| `build_rollout_labels.py` | `--step-windows`, `--output` |
| `build_latent_targets.py` | `--step-windows`, `--model-config`, `--image-root`, `--device`, `--token-dim` |
| `train_action_evaluator.py` | `--step-windows`, `--rollout-labels`, `--action-prior-cache`, `--latent-index`, `--image-root`, `--batch-size`, `--epochs`, `--lr` |
| `run_eval_aerialvln.py` | `--v0-checkpoint`, `--vlm-client`, `--score-preview-steps`, `--stop-completion-threshold`, `--batchSize`, `--EVAL_DATASET`, `--EVAL_NUM`, `--maxAction`, `--simulator_tool_port` |

### 8.4 原 AirVLN 透传参数

`run_eval_aerialvln.py` 会把未知参数透传给 `src/common/param.py`。常用：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--project_prefix` | 当前目录父级 | 对你的结构就是 `AirVLN_ws` |
| `--batchSize` | `8` | AirSim 并行 episode 数，调试建议 `1` |
| `--EVAL_DATASET` | `val_unseen` | split 名称，对应 `../DATA/data/aerialvln/{split}.json` |
| `--EVAL_NUM` | `-1` | eval episode 数，`-1` 表示全量 |
| `--maxAction` | `500` | 单 episode 最大动作数 |
| `--simulator_tool_port` | `30000` | simulator server 端口 |
| `--collect_type` | `TF` | 保留 AirVLN 环境模式 |

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
Get-ChildItem ../ENVs
```

应该能看到 `env_*`。启动 server：

```bash
python airsim_plugin/AirVLNSimulatorServerTool.py --gpus 0 --port 30000
```

再次检查：

```bash
python scripts/check_v0_setup.py --port 30000
```

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
