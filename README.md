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

如果把 `dinov2_repo` 和 `torch_hub_dir` 留空，`torch.hub` 会使用 PyTorch 默认缓存位置：

```yaml
vision:
  backbone: dinov2_s
  pretrained: true
  freeze: true
  dinov2_repo: ""
  torch_hub_dir: ""
```

首次运行时 `torch.hub` 会从 `facebookresearch/dinov2` 拉取代码；`pretrained: true` 时会同时下载权重。PyTorch 默认缓存位置通常是：

```text
Windows: C:\Users\<username>\.cache\torch\hub
Linux:   ~/.cache/torch/hub
权重:    <torch hub cache>/checkpoints
```

本仓库已经创建了单独的视觉骨干目录：

```text
AirVLN_ws/
  vision_backbones/
    dinov2/       # 可放 DINOv2 本地仓库：git clone https://github.com/facebookresearch/dinov2.git
    resnet/       # 可放 ResNet 本地权重，例如 resnet50.pth
    torch_hub/    # Torch Hub 缓存和 DINOv2 checkpoints
```

默认 `configs/model.yaml` 已经指向这个目录：

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

如果 `vision_backbones/dinov2/hubconf.py` 存在，代码会优先从本地 DINOv2 仓库加载；否则会走 `torch.hub` 在线/缓存加载，并把 hub 缓存写到 `vision_backbones/torch_hub/`。如果想用 ResNet 本地权重，把 `backbone` 改成 `resnet50` 或 `resnet18`，并设置：

```yaml
vision:
  resnet_weights: vision_backbones/resnet/resnet50.pth
```

`vision_backbones/` 下的大模型文件默认被 `.gitignore` 忽略，不会误提交到 GitHub。相对路径按你启动脚本时的当前目录解析；建议所有命令都从 `AirVLN_ws/` 下运行。如果只是做无网 smoke test，可以临时把 `pretrained` 改成 `false`，代码会在 DINOv2 不可用时退回轻量 CNN，但正式实验应使用 `pretrained: true` 的 DINOv2。

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
  --index-output data/latent_targets/train_index.jsonl ^
  --model-config configs/model.yaml ^
  --image-root ../DATA/data/aerialvln-s
```

没有 Qwen VLM 时，可以先用 `uniform` 跑通；正式实验再切到 `qwen_api` 或 `qwen_local`。

## 5. 训练 Latent Action Evaluator

```bash
python training/train_action_evaluator.py ^
  --step-windows data/step_windows/train.jsonl ^
  --rollout-labels data/rollout_labels/train.jsonl ^
  --action-prior-cache data/action_prior_cache/train.jsonl ^
  --latent-index data/latent_targets/train_index.jsonl ^
  --image-root ../DATA/data/aerialvln-s ^
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
- `vision.dinov2_repo/torch_hub_dir/resnet_weights`：控制 DINOv2 代码仓库、Torch Hub 缓存和 ResNet 本地权重放置位置，见上面的 DINOv2 放置方式。
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

## 9. 项目文件实现对照

`细节.md` 中 V0 要求的模块已经在代码中对应落点如下：

| 项目文件要求 | 代码落点 | 当前实现 |
| --- | --- | --- |
| Instruction-to-Milestone Parser | `prompts/milestone_prompt.txt`, `schemas/milestone_schema.py`, `preprocess/parse_instruction.py` | 离线解析，3-8 个 milestone，字段校验，失败重试，bad case 输出 |
| Milestone-Aware State Builder | `models/vision_backbone.py`, `models/history_encoder.py`, `models/trajectory_encoder.py`, `models/state_builder.py` | 当前 RGB、关键帧历史、动作/位姿/fallback 历史、milestone/progress、prev latent 共同构成 state dict |
| 历史关键帧规则 | `models/history_encoder.py`, `preprocess/build_step_windows.py` | 固定间隔、milestone 切换、连续转向、连续无进展规则均落成 `select_keyframe_indices` |
| VLM-Based Action Prior | `models/action_space.py`, `models/action_prior.py`, `prompts/action_prior_prompt.txt`, `preprocess/build_action_prior_cache.py` | 官方动作集合打分、非法动作过滤、分数归一化、低 completion 抑制 STOP、训练缓存 |
| Causal Latent Action Evaluator | `models/action_encoder.py`, `models/causal_latent_action_evaluator.py`, `preprocess/build_rollout_labels.py`, `preprocess/build_latent_targets.py`, `training/train_action_evaluator.py` | action token + causal Transformer，输出 progress/cost/next_latent，三项 loss 训练 |
| 一步 future latent 监督 | `preprocess/build_latent_targets.py`, `training/v0_dataset.py` | step-window 存 `next_rgb_path` 时用视觉骨干编码下一帧；缺失离线图像时用零向量 fallback 并在 summary 记录 |
| Step-wise Decision and Execution | `models/action_mask.py`, `models/decision_fuser.py`, `models/fallback.py`, `inference/planner_loop.py` | 合法动作过滤，手工权重融合，低进展/重复转向/低分 fallback，单步执行并更新 latent |
| AirVLN 仿真闭环与指标 | `inference/run_eval_aerialvln.py`, `src/vlnce_src/env.py` | `reset -> act -> makeActions -> get_obs -> update_measurements`，输出 success/nDTW/sDTW/path_length/oracle_success/steps |
| 日志和可视化 | `utils/diagnostics.py` 和各阶段脚本 | 阶段日志、JSON/JSONL 诊断、PNG-only 可视化 |

V0 按项目文件要求不实现 uncertainty、alignment head、OpenFly transfer、learned decision head、复杂 recovery policy；`inference/run_eval_openfly.py` 保持为 V2 占位。

## 10. 复现参数总表

### 配置文件

`configs/model.yaml`

| 参数 | 默认值 | 作用 | 常见修改 |
| --- | --- | --- | --- |
| `vision.backbone` | `dinov2_s` | 视觉骨干，支持 `dinov2_s/dinov2_b/resnet50/resnet18` | 正式实验用 DINOv2；快速 smoke 可用 ResNet |
| `vision.pretrained` | `true` | 是否加载预训练权重 | 无网 smoke test 可临时设 `false` |
| `vision.freeze` | `true` | 是否冻结视觉骨干 | 若要端到端微调设 `false` |
| `vision.assets_dir` | `vision_backbones` | 视觉骨干统一存放目录 | 通常不改 |
| `vision.dinov2_repo` | `vision_backbones/dinov2` | DINOv2 本地代码仓库路径 | 离线复现时把 DINOv2 clone 到这里 |
| `vision.torch_hub_dir` | `vision_backbones/torch_hub` | Torch Hub 缓存与 DINOv2 权重目录 | 多机器复现时固定到项目目录 |
| `vision.resnet_weights` | 空 | 本地 ResNet checkpoint 路径 | 例如 `vision_backbones/resnet/resnet50.pth` |
| `vision.image_height/width` | `224/224` | RGB 输入尺寸 | 与数据预处理保持一致 |
| `history.max_keyframes` | `8` | 最多保留历史关键帧数 | 显存不足时降低 |
| `history.interval` | `4` | 固定关键帧间隔 | 历史过稀可降低 |
| `trajectory.history_len` | `16` | 动作/位姿/fallback 历史长度 | 长程振荡多可增大 |
| `model.hidden_dim` | `512` | latent token 维度 | 显存不足时降低 |
| `model.num_heads/num_layers/dropout` | `8/2/0.1` | evaluator Transformer 容量 | 欠拟合增大，过拟合增大 dropout |
| `model.max_milestones` | `8` | milestone id embedding 上限 | 与 parser 3-8 保持一致 |
| `training.batch_size` | `16` | 训练 batch size | 显存不足时降低 |
| `training.epochs` | `10` | 默认训练轮数 | 正式实验按 val 曲线调整 |
| `training.lr` | `0.00025` | 学习率 | loss 震荡时降低 |
| `training.*_loss_weight` | `1.0` | latent/progress/cost loss 权重 | 根据曲线和行为调平衡 |

`configs/fuser.yaml`

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `w_progress` | `1.0` | progress gain 分数权重 |
| `w_cost` | `0.7` | cost 惩罚权重 |
| `w_prior` | `0.6` | Qwen/VLM action prior 权重 |
| `fallback.low_progress_threshold` | `0.05` | 连续低进展判定阈值 |
| `fallback.low_score_threshold` | `0.05` | 所有动作分数过低时触发 fallback |
| `fallback.progress_patience` | `3` | 连续多少步低进展触发 fallback |
| `fallback.repeated_turn_patience` | `2` | 连续多少步转向触发 fallback |
| `fallback.conservative_actions` | `MOVE_FORWARD, STOP` | fallback 时优先考虑的保守动作 |

`models/qwen_config.py`

| 参数 | 作用 |
| --- | --- |
| `QWEN_API_KEY` | DashScope/Qwen API key，正式 API 调用前必须设置 |
| `QWEN_API_BASE_URL` | Qwen 兼容 OpenAI API endpoint |
| `QWEN_TEXT_MODEL` | milestone parser 用文本模型 |
| `QWEN_VLM_MODEL` | action prior 用 VLM 模型 |
| `QWEN_LOCAL_LLM_COMMAND` | 本地 LLM 命令，stdin 读 JSON，stdout 输出 plan JSON |
| `QWEN_LOCAL_VLM_COMMAND` | 本地 VLM 命令，stdin 读 JSON，stdout 输出 action score JSON |

### 脚本参数

`preprocess/parse_instruction.py`

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--dataset-json` | 必填 | AerialVLN/AerialVLN-S split JSON |
| `--output` | `data/instruction_plan.jsonl` | milestone 解析结果 |
| `--bad-output` | `data/bad_cases.jsonl` | 校验失败样本 |
| `--prompt` | `prompts/milestone_prompt.txt` | parser prompt |
| `--client` | `rule` | `rule/qwen_api/qwen_local` |
| `--max-retries` | `1` | LLM 输出失败后的重试次数 |
| `--diagnostics-dir` | `DATA/v0/diagnostics/parse_instruction` | 日志和 PNG 输出目录 |

`preprocess/build_step_windows.py`

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--dataset-json` | 必填 | split JSON |
| `--instruction-plan` | `data/instruction_plan.jsonl` | milestone 输入 |
| `--output` | `data/step_windows/train.jsonl` | step-window 输出 |
| `--history-len` | `16` | 动作/位姿历史长度 |
| `--max-keyframes` | `8` | 最多关键帧数 |
| `--keyframe-interval` | `4` | 固定关键帧间隔 |
| `--diagnostics-dir` | `DATA/v0/diagnostics/build_step_windows` | 诊断目录 |

输出的每条 step-window 会尽量包含 `rgb_path`、`next_rgb_path`、`keyframe_rgb_paths`。如果原始 dataset JSON 没有 `rgb_paths/image_paths/images/frames` 这类字段，后续训练和 latent target 构造会自动使用零图像/零 latent fallback，并在诊断 summary 里体现。

`preprocess/build_action_prior_cache.py`

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--step-windows` | 必填 | step-window 输入 |
| `--output` | `data/action_prior_cache/train.jsonl` | prior 缓存 |
| `--prompt` | `prompts/action_prior_prompt.txt` | VLM prior prompt |
| `--client` | `uniform` | `uniform/qwen_api/qwen_local` |
| `--preview-count` | `20` | 写入预览 JSONL 的样本数 |
| `--diagnostics-dir` | `DATA/v0/diagnostics/build_action_prior_cache` | 诊断目录 |

`preprocess/build_rollout_labels.py`

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--step-windows` | 必填 | step-window 输入 |
| `--output` | `data/rollout_labels/train.jsonl` | progress/cost label 输出 |
| `--diagnostics-dir` | `DATA/v0/diagnostics/build_rollout_labels` | 诊断目录 |

`preprocess/build_latent_targets.py`

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--step-windows` | 必填 | step-window 输入 |
| `--output-dir` | `data/latent_targets/train` | latent target `.pt` 输出目录 |
| `--index-output` | `data/latent_targets/train_index.jsonl` | latent target 索引 |
| `--model-config` | `configs/model.yaml` | 读取视觉骨干、图像尺寸和 hidden dim |
| `--image-root` | 空 | 解析相对图像路径的根目录 |
| `--device` | `cuda` | 编码下一帧图像的设备，CUDA 不可用会回退 CPU |
| `--token-dim` | 空 | 覆盖 latent target 维度；默认使用 `model.hidden_dim` |
| `--preview-count` | `10` | 预览日志数量 |
| `--diagnostics-dir` | `DATA/v0/diagnostics/build_latent_targets` | 诊断目录 |

`training/train_action_evaluator.py`

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--step-windows` | 必填 | step-window 输入 |
| `--rollout-labels` | 必填 | progress/cost labels |
| `--action-prior-cache` | 空 | action prior 缓存 |
| `--latent-index` | 空 | latent target 索引 |
| `--image-root` | 空 | 解析 `rgb_path/keyframe_rgb_paths` 的根目录 |
| `--model-config` | `configs/model.yaml` | 模型配置 |
| `--output-dir` | `DATA/v0/checkpoints/action_evaluator` | checkpoint 输出目录 |
| `--device` | `cuda` | 训练设备，CUDA 不可用会回退 CPU |
| `--batch-size` | `16` | batch size |
| `--epochs` | `10` | 训练轮数 |
| `--lr` | `0.00025` | 学习率 |
| `--num-workers` | `0` | DataLoader workers |
| `--progress-loss-weight` | `1.0` | progress loss 权重 |
| `--cost-loss-weight` | `1.0` | cost loss 权重 |
| `--latent-loss-weight` | `1.0` | latent loss 权重 |
| `--preview-batches` | `2` | 每轮打印多少个 batch 预览 |
| `--diagnostics-dir` | `DATA/v0/diagnostics/train_action_evaluator` | 训练日志和 loss PNG 输出目录 |

`training/train_state_encoder.py`

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--model-config` | `configs/model.yaml` | 初始化 state builder 的配置 |
| `--output` | `DATA/v0/checkpoints/state_builder_init.pth` | 初始化 checkpoint 输出 |

`training/tune_fuser.py`

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--output` | `configs/fuser.yaml` | 输出 fuser 配置 |
| `--w-progress` | `0.5 1.0 1.5` | progress 权重候选 |
| `--w-cost` | `0.3 0.7 1.0` | cost 权重候选 |
| `--w-prior` | `0.0 0.3 0.6` | prior 权重候选 |

`inference/run_eval_aerialvln.py`

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--v0-checkpoint` | 空 | evaluator checkpoint，正式 eval 应指定 `ckpt_last.pth` |
| `--model-config` | `configs/model.yaml` | 模型配置 |
| `--fuser-config` | `configs/fuser.yaml` | 融合/fallback 配置 |
| `--vlm-client` | `uniform` | `uniform/qwen_api/qwen_local` |
| `--device` | `cuda` | 推理设备，CUDA 不可用会回退 CPU |
| `--eval-output` | `DATA/v0/eval/results.json` | 指标 JSON 输出 |
| `--diagnostics-dir` | `DATA/v0/diagnostics/eval_aerialvln` | eval 诊断目录 |
| `--stop-completion-threshold` | `0.35` | milestone completion 低于该值时抑制 STOP prior |
| `--score-preview-steps` | `5` | 每个 episode 前多少步保存 action score PNG |

`run_eval_aerialvln.py` 还会透传原 AirVLN 参数；复现时最常改的是：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--batchSize` | `8` | AirSim 并行 batch，调试建议 `1` |
| `--EVAL_DATASET` | `val_unseen` | 评估 split |
| `--EVAL_NUM` | `-1` | 限制评估 episode 数，`-1` 表示全量 |
| `--maxAction` | `500` | 单 episode 最大步数 |
| `--collect_type` | `TF` | 保留 AirVLN 数据/环境模式 |
| `--simulator_tool_port` | `30000` | AirSim 通信端口 |
| `--project_prefix` | 仓库父级 | 数据和 ENVs 的基准路径 |

### 复现顺序

1. 准备 `../DATA/data/aerialvln-s`、`../DATA/data/aerialvln`、`../ENVs`。
2. 准备 `vision_backbones/`：有网可自动下载；离线则把 DINOv2 仓库放到 `vision_backbones/dinov2`，权重/cache 放到 `vision_backbones/torch_hub`。
3. 修改 `models/qwen_config.py`，或用 `rule/uniform` 跑 smoke test。
4. 依次跑 `parse_instruction.py -> build_step_windows.py -> build_action_prior_cache.py -> build_rollout_labels.py -> build_latent_targets.py`。
5. 跑 `training/train_action_evaluator.py` 生成 `ckpt_last.pth`。
6. 启动 AirSim/ENVs 后跑 `inference/run_eval_aerialvln.py`，查看 `eval-output` 和 `DATA/v0/diagnostics/**/*.png`。

## 11. 验收 Checklist

满足下面条件即可认为当前代码完成“仿真-导航-指标统计”的 latent world model 式 AirVLN 闭环：

- `inference/run_eval_aerialvln.py` 能创建 `AirVLNENV` 并完成 `reset/get_obs/makeActions` 循环。
- `V0PlannerLoop` 每步输出 action、fallback、fused score 和 next latent。
- `src/vlnce_src/env.py` 正常更新 `success/ndtw/sdtw/path_length/oracle_success/steps_taken`。
- `--eval-output` 包含 `summary` 和 `episodes`。
- `DATA/v0/diagnostics/eval_aerialvln/eval_metrics.png` 存在。
- `DATA/v0/diagnostics/eval_aerialvln/planner_loop/episode_*_step_*_scores.png` 存在。
- README 中的调参项与实际配置文件、代码行为一致。
