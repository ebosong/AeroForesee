# 14 步骤：在线 Planner Loop

## 入口

核心文件：

- `inference/run_eval_aerialvln.py`
- `inference/planner_loop.py`

命令：

```bash
python inference/run_eval_aerialvln.py ^
  --v0-checkpoint DATA/v0/checkpoints/action_evaluator/ckpt_last.pth ^
  --model-config configs/model.yaml ^
  --fuser-config configs/fuser.yaml ^
  --eval-output DATA/v0/eval/aerialvln_val_unseen.json ^
  --diagnostics-dir DATA/v0/diagnostics/eval_aerialvln ^
  --instruction-plan data/instruction_plan.jsonl ^
  --vlm-client qwen_api ^
  --device cuda ^
  --gpu-ids 0 ^
  --batchSize 1 ^
  --EVAL_DATASET val_unseen ^
  --EVAL_NUM -1 ^
  --maxAction 500 ^
  --collect_type TF ^
  --simulator_tool_port 30000
```

## run_eval_aerialvln.py

职责：

1. 解析 V0 参数和 AirVLN 透传参数。
2. 强制设置 `--run_type eval --ablate_depth`。
3. 初始化 state builder、evaluator、VLM client、fuser。
4. 加载 checkpoint。
5. 初始化 AirVLNENV。
6. 将 `instruction_plan.jsonl` merge 回 env.data。
7. 按 batch 循环 reset、act、makeActions、get_obs。
8. 汇总指标并写出 eval JSON。

## V0PlannerLoop

`V0PlannerLoop.act()` 是在线单步决策核心。

每步流程：

1. 读取当前 obs。
2. 从 memory 读取历史 RGB、动作、pose、latent。
3. `MilestoneProgressController.current()` 给出当前 milestone。
4. 调 `ActionPriorModule.score()`。
5. 构造 state。
6. 对所有官方动作跑 evaluator。
7. 用 fuser 融合 progress/cost/prior。
8. fallback 修正动作。
9. 更新 latent。
10. `MilestoneProgressController.update()` 更新 milestone completion。
11. 写 step decision 日志。

## EpisodeMemory

每个 episode 单独维护：

- `rgb_history`
- `action_history`
- `pose_history`
- `progress_history`
- `global_progress_history`
- `fallback_history`
- `latent`
- `fallback`
- `progress_controller`

分析：

- 这让 batch 中不同 episode 的历史互不污染。
- 如果 episode_id 重复复用，memory 需要正确初始化。

## 输出

主输出：

- `DATA/v0/eval/*.json`

诊断：

- `DATA/v0/diagnostics/eval_aerialvln/eval_steps.jsonl`
- `DATA/v0/diagnostics/eval_aerialvln/eval_metrics.png`
- `DATA/v0/diagnostics/eval_aerialvln/planner_loop/step_decisions.jsonl`
- `episode_*_step_*_scores.png`

## step_decisions 关键字段

- `episode_id`
- `step`
- `action_id`
- `action_name`
- `fallback`
- `milestone_id`
- `milestone_text`
- `milestone_completion`
- `updated_milestone_id`
- `updated_milestone_completion`
- `scores`

## 分析

在线 planner 的强弱主要取决于：

- checkpoint 质量。
- VLM prior 质量。
- milestone progress controller 是否合理推进。
- fuser 权重是否平衡。
- AirSim obs 是否稳定返回 RGB。

## 局限

- VLM prior 在线调用有延迟。
- controller 是规则 evidence-driven，不是 learned verifier。
- fallback 是保守策略，不是完整 recovery policy。
- eval TF 模式下 env progress 通常很弱，因此不要只看 obs progress。
