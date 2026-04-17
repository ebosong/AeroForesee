# AeroForesee：AirVLN V0 里程碑因果规划系统

本仓库保留 AirVLN 的仿真环境接口，移除了原始 Seq2Seq/CMA baseline 训练代码，新增一套面向 UAV-VLN 的 V0 系统：

**显式 milestone + Qwen action prior + 轻量 causal latent action evaluator + 手工决策融合 + 单步闭环执行。**

目标任务是 single-view / egocentric-view 长程 UAV Vision-Language Navigation。运行时主输入只使用：

- 自然语言长指令
- UAV 当前前视 RGB
- 历史关键帧
- 当前及历史轨迹
- 当前 milestone/progress 状态

运行时不会把 depth 作为主输入；`inference/run_eval_aerialvln.py` 会强制启用 `--ablate_depth`。

## 1. 代码结构

```text
configs/                 V0 配置
prompts/                 Qwen milestone/action-prior prompt
schemas/                 milestone schema 校验
models/                  V0 模型与 Qwen client
preprocess/              离线数据构造
training/                evaluator 训练
inference/               planner loop 与 eval 入口
airsim_plugin/           AirVLN/AirSim 仿真通信
src/vlnce_src/env.py     AirVLN 环境封装
utils/                   仿真、环境、日志、诊断工具
```

原 AirVLN baseline 的 `Model/`、`src/vlnce_src/train.py`、`src/vlnce_src/dagger_train.py` 等训练策略代码已移除；当前仓库以 V0 planner 为主。

## 2. Qwen 配置

所有 LLM/VLM 调用统一限定为 Qwen。API 版和本地版都读取硬编码配置：

```text
models/qwen_config.py
```

需要按你的环境修改：

```python
QWEN_API_KEY = "PASTE_YOUR_QWEN_API_KEY_HERE"
QWEN_API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_TEXT_MODEL = "qwen-plus"
QWEN_VLM_MODEL = "qwen-vl-plus"

QWEN_LOCAL_LLM_COMMAND = "python tools/qwen_local_llm.py"
QWEN_LOCAL_VLM_COMMAND = "python tools/qwen_local_vlm.py"
```

说明：

- `qwen_api`：通过 Qwen / DashScope 兼容接口调用。
- `qwen_local`：通过本地命令调用 Qwen 模型服务或脚本。
- `rule` 和 `uniform` 只用于 smoke test。
- 当前提交没有写入真实 API key，避免密钥进入 GitHub。

## 3. 离线 Milestone 解析

无模型 smoke test：

```bash
python preprocess/parse_instruction.py ^
  --dataset-json ../DATA/data/aerialvln-s/train.json ^
  --output data/instruction_plan.jsonl ^
  --client rule
```

Qwen API：

```bash
python preprocess/parse_instruction.py ^
  --dataset-json ../DATA/data/aerialvln-s/train.json ^
  --output data/instruction_plan.jsonl ^
  --client qwen_api
```

本地 Qwen：

```bash
python preprocess/parse_instruction.py ^
  --dataset-json ../DATA/data/aerialvln-s/train.json ^
  --output data/instruction_plan.jsonl ^
  --client qwen_local
```

本地 LLM 命令从 stdin 接收：

```json
{"system": "...", "user": "..."}
```

并向 stdout 输出合法 JSON instruction plan。

## 4. 构造训练数据

```bash
python preprocess/build_step_windows.py ^
  --dataset-json ../DATA/data/aerialvln-s/train.json ^
  --instruction-plan data/instruction_plan.jsonl ^
  --output data/step_windows/train.jsonl

python preprocess/build_action_prior_cache.py ^
  --step-windows data/step_windows/train.jsonl ^
  --output data/action_prior_cache/train.jsonl ^
  --client qwen_api

python preprocess/build_rollout_labels.py ^
  --step-windows data/step_windows/train.jsonl ^
  --output data/rollout_labels/train.jsonl

python preprocess/build_latent_targets.py ^
  --step-windows data/step_windows/train.jsonl ^
  --output-dir data/latent_targets/train ^
  --index-output data/latent_targets/train_index.jsonl
```

如果还没有 Qwen VLM，可先把 `build_action_prior_cache.py` 的 client 改成 `uniform` 跑通流程。

## 5. 训练 Evaluator

```bash
python training/train_action_evaluator.py ^
  --step-windows data/step_windows/train.jsonl ^
  --rollout-labels data/rollout_labels/train.jsonl ^
  --action-prior-cache data/action_prior_cache/train.jsonl ^
  --latent-index data/latent_targets/train_index.jsonl ^
  --output-dir DATA/v0/checkpoints/action_evaluator ^
  --epochs 10
```

V0 先复用当前 ResNet 风格视觉骨干，后续可在 `models/vision_backbone.py` 中替换 DINOv2。

## 6. Eval

```bash
python inference/run_eval_aerialvln.py ^
  --v0-checkpoint DATA/v0/checkpoints/action_evaluator/ckpt_last.pth ^
  --eval-output DATA/v0/eval/aerialvln_s_val_unseen.json ^
  --vlm-client qwen_api ^
  --batchSize 1 ^
  --EVAL_DATASET val_unseen ^
  --collect_type TF
```

本地 Qwen VLM：

```bash
python inference/run_eval_aerialvln.py ^
  --v0-checkpoint DATA/v0/checkpoints/action_evaluator/ckpt_last.pth ^
  --eval-output DATA/v0/eval/aerialvln_s_val_unseen.json ^
  --vlm-client qwen_local ^
  --batchSize 1 ^
  --EVAL_DATASET val_unseen ^
  --collect_type TF
```

## 7. 日志和可视化怎么看

每个阶段都会打印英文阶段日志，例如：

```text
[parse_instruction] parsed | instruction_id=... milestones=...
[build_step_windows] episode_done | instruction_id=... steps=...
[train_action_evaluator] epoch_done | epoch=... loss=...
[planner_loop] step_decision | episode_id=... action=... fallback=...
```

同时会在 `DATA/v0/diagnostics/` 下输出 JSON / JSONL / SVG：

- `parse_instruction/summary.json`：合法解析数量和 bad case 数量。`bad` 越少越好。
- `parse_instruction/milestone_count_distribution.svg`：milestone 数量分布。大多数样本应落在 3 到 8 内，且不要全部集中在同一个数量。
- `build_step_windows/summary.json`：生成的 step-window 数量。应明显大于 episode 数量。
- `build_step_windows/episode_step_count_distribution.svg`：每条轨迹步数分布。异常的 0 步或极端短轨迹需要检查数据。
- `build_action_prior_cache/prior_preview.jsonl`：前若干样本的动作 prior。检查 top action 是否和 milestone 语义大致一致。
- `build_action_prior_cache/top_prior_action_distribution.svg`：VLM 最偏好的动作分布。如果几乎全是同一个动作，说明 prompt、图像输入或 Qwen 输出可能有问题。
- `build_rollout_labels/positive_progress_by_action.svg`：不同动作的正 progress 标签分布。若只有 STOP 或单一动作占满，需要检查标签生成逻辑。
- `build_rollout_labels/average_cost_by_action.svg`：各动作平均 cost。GT 动作相关 cost 应整体更低。
- `build_latent_targets/summary.json`：latent target 数量和维度。数量应与 step-window 对齐。
- `train_action_evaluator/training_log.jsonl`：每个 epoch loss。
- `train_action_evaluator/loss_curve.svg`：训练 loss 曲线。正常情况下应整体下降或至少稳定；持续上升说明学习率、标签或输入维度需要检查。
- `eval_aerialvln/eval_steps.jsonl`：eval 每一步动作和 fallback 情况。
- `eval_aerialvln/planner_loop/step_decisions.jsonl`：每一步 fused score、动作和 fallback。
- `eval_aerialvln/planner_loop/episode_*_step_*_scores.svg`：前几步动作分数条形图。合理结果应体现 milestone 相关动作分数更高，fallback 不应频繁触发。

判断模块效果的顺序建议：

1. 先看 `parse_instruction`：bad case 是否少，milestone 是否有地标和空间关系。
2. 再看 `action_prior_cache`：Qwen prior 是否能根据 milestone 改变 top action。
3. 再看 `rollout_labels`：progress/cost 是否不是单一常数。
4. 然后看 `train_action_evaluator`：loss 是否下降。
5. 最后看 `planner_loop`：是否出现连续重复转向、频繁 fallback 或 STOP 过早。

## 8. 仿真和数据

仿真器、场景和数据集仍沿用 AirVLN/AerialVLN 的组织方式。请将数据放到工作区的：

```text
../DATA/data/aerialvln
../DATA/data/aerialvln-s
../ENVs
```

AirSim 通信相关代码保留在 `airsim_plugin/`，环境封装保留在 `src/vlnce_src/env.py`。

