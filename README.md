# AeroForesee: AirVLN V0 Latent World Model 闭环

本仓库保留原 AirVLN/AerialVLN 的仿真环境接口，移除了原始 Seq2Seq/CMA baseline 训练代码，新增一套面向 UAV-VLN 的 V0 闭环系统：

**显式 milestone + Qwen action prior + causal latent action evaluator + 决策融合/fallback + AirSim 单步执行 + 指标统计。**

从任务形态上看，当前代码已经可以构成一个 latent world model 式 AirVLN 闭环：

```text
自然语言指令/RGB/轨迹历史
        |
        v
milestone-aware state builder
        |
        v
causal latent action evaluator: (state, action) -> progress_gain/cost/next_latent
        |
        v
Qwen action prior + DecisionFuser + FallbackPolicy
        |
        v
AirVLNENV.makeActions -> AirSim 仿真推进 -> AirVLNENV.get_obs
        |
        v
success / nDTW / sDTW / path_length / oracle_success / steps_taken
```

运行时主输入只使用自然语言长指令、当前前视 RGB、历史关键帧、当前及历史轨迹、当前 milestone/progress 状态。`inference/run_eval_aerialvln.py` 会强制启用 `--ablate_depth`，depth 不作为 V0 主输入。

## 1. 代码结构

```text
configs/                 V0 模型与融合配置
prompts/                 Qwen milestone/action-prior prompt
schemas/                 milestone schema 校验
models/                  state builder / action prior / latent evaluator / fuser / fallback
preprocess/              离线 milestone、step-window、label、latent target 构造
training/                latent action evaluator 训练与 fuser 调参
inference/               planner loop 与 AirVLN eval 入口
airsim_plugin/           原 AirVLN/AirSim 仿真通信
src/vlnce_src/env.py     原 AirVLN 环境封装和指标计算
utils/                   环境、日志、诊断图和 JSON 工具
```

核心闭环文件：

- `inference/run_eval_aerialvln.py`：启动 AirVLN 环境，加载 checkpoint，逐步调用 planner，执行动作并汇总指标。
- `inference/planner_loop.py`：维护 episode memory、构造 latent state、枚举动作、融合分数、写出每步诊断。
- `models/causal_latent_action_evaluator.py`：因果 Transformer 评估 `(state, action)` 的 progress/cost/next_latent。
- `src/vlnce_src/env.py`：保留原 AirVLN 的 `reset/get_obs/makeActions/update_measurements`，完成仿真推进和指标更新。

## 2. 环境与原 AirVLN 注意事项

原 AirVLN 仓库要求 Ubuntu、Nvidia GPU、Python 3.8+ 和 Conda；仿真器约 35GB，标注数据包括 AerialVLN 与 AerialVLN-S，项目目录建议保持为：

```text
Project workspace/
  AirVLN_ws/
  DATA/
    data/
      aerialvln/
      aerialvln-s/
    models/
      ddppo-models/
  ENVs/
    env_1/
    env_2/
    ...
```

本仓库默认相对路径仍按这个布局读取：

```text
../DATA/data/aerialvln
../DATA/data/aerialvln-s
../ENVs
```

依赖安装建议：

```bash
conda create -n AirVLN python=3.8
conda activate AirVLN
pip install pip==24.0 setuptools==63.2.0
pip install -r requirements.txt
pip install airsim==1.7.0
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cuXXX
pip install pytorch-transformers==1.2.0
```

从原 AirVLN README 继承的重点注意事项：

- 首次运行前确认 `./ENVs` 里的 AirSim 场景可以单独打开；无 GUI 服务器需要 headless mode 或 virtual display。
- 默认通信端口是 `30000`；若遇到 `Address already in use`，需要释放端口或改端口。
- 若出现 `Failed to open scenes` 或 request timeout，优先检查场景路径、GPU 可见性、AirSim 进程状态，并把 `--batchSize` 降到 `1`。
- 原仓库提示首次使用时检查 `airsim_plugin/AirVLNSimulatorClientTool.py` 中 `_getImages` 的图像通道顺序，确保可视化颜色正常。
- 仿真、数据、场景组织沿用 AirVLN；本仓库只替换导航决策链路，不重新定义环境协议。

原始项目参考：`https://github.com/AirVLN/AirVLN`

### DINOv2 放置方式

`configs/model.yaml` 里的 `vision.backbone: dinov2_s` 不是一个本地文件名，而是代码中的模型别名：

```text
dinov2_s -> torch.hub 模型 dinov2_vits14，输出维度 384
dinov2_b -> torch.hub 模型 dinov2_vitb14，输出维度 768
```

默认配置如下：

```yaml
vision:
  backbone: dinov2_s
  pretrained: true
  freeze: true
  dinov2_repo: ""
  torch_hub_dir: ""
```

有网环境下不需要手工放文件。首次运行时 `torch.hub` 会从 `facebookresearch/dinov2` 拉取代码；`pretrained: true` 时会同时下载权重。默认缓存位置由 PyTorch 决定：

```text
Windows: C:\Users\<username>\.cache\torch\hub
Linux:   ~/.cache/torch/hub
权重:    <torch hub cache>/checkpoints
```

推荐的项目内离线放置方式是：

```text
Project workspace/
  AirVLN_ws/
  DATA/
    models/
      dinov2/          # git clone https://github.com/facebookresearch/dinov2.git
      torch_hub/       # torch hub cache，可放 DINOv2 权重 checkpoint
```

然后把 `configs/model.yaml` 改成：

```yaml
vision:
  backbone: dinov2_s
  pretrained: true
  freeze: true
  dinov2_repo: ../DATA/models/dinov2
  torch_hub_dir: ../DATA/models/torch_hub
```

`dinov2_repo` 是 DINOv2 代码仓库路径，`torch_hub_dir` 是 hub 缓存路径。相对路径按你启动脚本时的当前目录解析；建议所有命令都从 `AirVLN_ws/` 下运行。如果只是做无网 smoke test，可以临时把 `pretrained` 改成 `false`，代码会在 DINOv2 不可用时退回轻量 CNN，但正式实验应使用 `pretrained: true` 的 DINOv2。

## 3. Qwen 配置

所有 LLM/VLM 调用统一限定为 Qwen。API 版和本地版都读取：

```text
models/qwen_config.py
```

按你的环境修改：

```python
QWEN_API_KEY = "PASTE_YOUR_QWEN_API_KEY_HERE"
QWEN_API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_TEXT_MODEL = "qwen-plus"
QWEN_VLM_MODEL = "qwen-vl-plus"

QWEN_LOCAL_LLM_COMMAND = "python tools/qwen_local_llm.py"
QWEN_LOCAL_VLM_COMMAND = "python tools/qwen_local_vlm.py"
```

`qwen_api` 通过 DashScope 兼容接口调用；`qwen_local` 通过本地命令调用；`rule` 和 `uniform` 只用于 smoke test。不要把真实 API key 提交到仓库。

## 4. 离线数据构造

先解析长指令为 milestone：

```bash
python preprocess/parse_instruction.py ^
  --dataset-json ../DATA/data/aerialvln-s/train.json ^
  --output data/instruction_plan.jsonl ^
  --client rule
```

使用 Qwen API 时把 `--client rule` 改成 `--client qwen_api`。

构造训练所需的 step-window、action prior、rollout label 和 latent target：

```bash
python preprocess/build_step_windows.py ^
  --dataset-json ../DATA/data/aerialvln-s/train.json ^
  --instruction-plan data/instruction_plan.jsonl ^
  --output data/step_windows/train.jsonl

python preprocess/build_action_prior_cache.py ^
  --step-windows data/step_windows/train.jsonl ^
  --output data/action_prior_cache/train.jsonl ^
  --client uniform

python preprocess/build_rollout_labels.py ^
  --step-windows data/step_windows/train.jsonl ^
  --output data/rollout_labels/train.jsonl

python preprocess/build_latent_targets.py ^
  --step-windows data/step_windows/train.jsonl ^
  --output-dir data/latent_targets/train ^
  --index-output data/latent_targets/train_index.jsonl
```

没有 Qwen VLM 时，可以先用 `uniform` 跑通；正式实验再切到 `qwen_api` 或 `qwen_local`。

## 5. 训练 Latent Action Evaluator

```bash
python training/train_action_evaluator.py ^
  --step-windows data/step_windows/train.jsonl ^
  --rollout-labels data/rollout_labels/train.jsonl ^
  --action-prior-cache data/action_prior_cache/train.jsonl ^
  --latent-index data/latent_targets/train_index.jsonl ^
  --output-dir DATA/v0/checkpoints/action_evaluator ^
  --epochs 10
```

训练完成后会保存：

```text
DATA/v0/checkpoints/action_evaluator/ckpt_last.pth
DATA/v0/diagnostics/train_action_evaluator/training_log.jsonl
DATA/v0/diagnostics/train_action_evaluator/loss_curve.png
```

## 6. Eval 闭环运行

```bash
python inference/run_eval_aerialvln.py ^
  --v0-checkpoint DATA/v0/checkpoints/action_evaluator/ckpt_last.pth ^
  --eval-output DATA/v0/eval/aerialvln_s_val_unseen.json ^
  --vlm-client qwen_api ^
  --score-preview-steps 5 ^
  --stop-completion-threshold 0.35 ^
  --batchSize 1 ^
  --EVAL_DATASET val_unseen ^
  --collect_type TF
```

该入口会强制补上 `--run_type eval --ablate_depth`，然后执行：

```text
AirVLNENV.reset
  -> planner.act
  -> AirVLNENV.makeActions
  -> AirVLNENV.get_obs
  -> AirVLNENV.update_measurements
  -> aggregate metrics
```

`--eval-output` 写出结构为：

```json
{
  "summary": {
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
      "ndtw": 0.0
    }
  }
}
```

## 7. 诊断与 PNG 可视化

所有诊断图只保存 PNG，默认目录是 `DATA/v0/diagnostics/`。

重点文件：

- `parse_instruction/summary.json`：合法解析数量和 bad case 数量。
- `parse_instruction/milestone_count_distribution.png`：milestone 数量分布。
- `build_step_windows/summary.json`：step-window 数量。
- `build_step_windows/episode_step_count_distribution.png`：每条轨迹步数分布。
- `build_action_prior_cache/prior_preview.jsonl`：前若干样本 action prior。
- `build_action_prior_cache/top_prior_action_distribution.png`：VLM top action 分布。
- `build_rollout_labels/positive_progress_by_action.png`：正 progress 标签分布。
- `build_rollout_labels/average_cost_by_action.png`：各动作平均 cost。
- `build_latent_targets/summary.json`：latent target 数量和维度。
- `train_action_evaluator/loss_curve.png`：训练 loss 曲线。
- `eval_aerialvln/eval_steps.jsonl`：每一步动作和 fallback。
- `eval_aerialvln/eval_metrics.png`：最终平均指标图。
- `eval_aerialvln/planner_loop/step_decisions.jsonl`：每一步 fused score。
- `eval_aerialvln/planner_loop/episode_*_step_*_scores.png`：前几步动作分数条形图。

建议检查顺序：

1. `parse_instruction`：bad case 少，milestone 有地标和空间关系。
2. `build_action_prior_cache`：top action 不应长期塌缩到同一个动作。
3. `build_rollout_labels`：progress/cost 不应是单一常数。
4. `train_action_evaluator`：loss 应下降或稳定。
5. `eval_aerialvln/planner_loop`：不要频繁 fallback、连续转向或过早 STOP。
6. `eval_metrics.png` 与 `eval-output`：确认 success、nDTW、sDTW、steps 等指标被正常统计。

## 8. 调参指南

优先从 `configs/fuser.yaml` 调闭环行为：

```yaml
w_progress: 1.0
w_cost: 0.7
w_prior: 0.6
fallback:
  low_progress_threshold: 0.05
  low_score_threshold: 0.05
  progress_patience: 3
  repeated_turn_patience: 2
  conservative_actions:
    - MOVE_FORWARD
    - STOP
```

常见现象和调法：

- 过早 STOP：降低 `w_prior`，降低 `ActionPriorModule.stop_completion_threshold`，或提高 STOP 相关训练样本质量。
- 来回转向：降低 `repeated_turn_patience`，提高 `w_cost`，检查 `episode_*_scores.png` 中 TURN_LEFT/RIGHT 是否异常偏高。
- 一直前进但不对齐 milestone：提高 `w_prior`，检查 Qwen prompt 与 `prior_preview.jsonl`。
- fallback 过多：降低 `low_progress_threshold` 或 `low_score_threshold`，同时检查 evaluator 是否欠训练。
- 路径过长：提高 `w_cost`，检查 rollout label 的 `average_cost_by_action.png`。
- 指标有 nDTW 但 success 低：检查 STOP 时机、`SUCCESS_DISTANCE` 和最终位置是否接近目标。

`configs/model.yaml` 负责模型容量与历史长度：

- `model.hidden_dim`：latent token 维度，增大后表达更强但显存更高。
- `vision.backbone`：默认 `dinov2_s`，也支持 `dinov2_b/resnet50/resnet18`；正式实验建议用 DINOv2。
- `vision.pretrained/freeze`：默认 `true/true`，表示加载预训练权重并冻结视觉骨干；无网 smoke test 才建议临时设 `pretrained: false`。
- `vision.dinov2_repo/torch_hub_dir`：控制 DINOv2 代码仓库和 Torch Hub 缓存放置位置，见上面的 DINOv2 放置方式。
- `model.num_layers/num_heads/dropout`：evaluator 容量和正则。
- `history.max_keyframes`：视觉历史帧数，过大增加显存和延迟。
- `trajectory.history_len`：动作/pose/fallback 历史长度。
- `training.lr` 与 `--lr`：loss 震荡时先降学习率。
- `training.*_loss_weight` 与训练参数：平衡 latent、progress、cost 三类监督。
- `--score-preview-steps`：控制 eval 时每个 episode 前多少步输出动作分数 PNG。
- `--stop-completion-threshold`：控制 action prior 对早停动作的保守程度。

推荐调参顺序：

1. 先用 `uniform` prior 跑 smoke test，确认数据构造和 AirSim 闭环可用。
2. 固定模型，调 `w_progress/w_cost/w_prior`，观察 `eval_metrics.png` 和 step score PNG。
3. 再调 fallback 阈值和 patience，处理卡住、连续转向、过早 STOP。
4. 最后扩大模型容量或历史长度，避免先用大模型掩盖数据或仿真问题。

## 9. 验收 Checklist

满足下面条件即可认为当前代码完成“仿真-导航-指标统计”的 latent world model 式 AirVLN 闭环：

- `inference/run_eval_aerialvln.py` 能创建 `AirVLNENV` 并完成 `reset/get_obs/makeActions` 循环。
- `V0PlannerLoop` 每步输出 action、fallback、fused score 和 next latent。
- `src/vlnce_src/env.py` 正常更新 `success/ndtw/sdtw/path_length/oracle_success/steps_taken`。
- `--eval-output` 包含 `summary` 和 `episodes`。
- `DATA/v0/diagnostics/eval_aerialvln/eval_metrics.png` 存在。
- `DATA/v0/diagnostics/eval_aerialvln/planner_loop/episode_*_step_*_scores.png` 存在。
- README 中的调参项与实际配置文件、代码行为一致。
