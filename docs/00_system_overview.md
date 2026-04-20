# 00 系统总览

## 目标

AeroForesee V0 面向 single-view / egocentric-view 的长程 UAV Vision-Language Navigation。它保留 AirVLN 的仿真、批处理和指标计算接口，但将决策核心替换为一套 milestone-aware 的闭环规划器。

V0 的主张不是直接训练一个端到端 action predictor，而是把长指令拆成阶段，再在每一步评估候选动作的短期后果：

```text
显式 milestone
  + 当前 RGB / 历史关键帧
  + VLM action prior
  + causal latent action evaluator
  + 手工融合和 fallback
  + AirSim 单步执行
```

## 核心输入

- 自然语言长指令。
- 当前前视 RGB。
- 最近关键帧。
- 历史动作和 pose delta。
- 当前 milestone 文本、id、completion。
- 上一步 latent state。
- fallback 历史。

## 核心输出

- 当前步动作 id。
- 每个候选动作的融合分数。
- 是否触发 fallback。
- 更新后的 recurrent latent。
- 更新后的 milestone completion。

## 离线流程

离线流程负责把原始 episode 转成 evaluator 可训练的数据：

1. `preprocess/parse_instruction.py`  
   把 instruction 解析成 3 到 8 个结构化 milestone。
2. `preprocess/build_step_windows.py`  
   根据 episode actions、reference path、milestone plan 构造逐步窗口。
3. `preprocess/build_action_prior_cache.py`  
   对每个 step-window 调 VLM 或 uniform client，缓存动作先验。
4. `preprocess/build_rollout_labels.py`  
   用 reference-geometry local world-model 或启发式 fallback 构造 progress/cost 标签。
5. `preprocess/build_latent_targets.py`  
   用下一帧 RGB 的视觉 token 作为 latent future target。
6. `training/train_action_evaluator.py`  
   训练 state builder + causal latent action evaluator。

## 在线闭环

在线 eval 入口是 `inference/run_eval_aerialvln.py`。它做三件事：

- 初始化 AirVLNENV。
- 加载 V0 模型和 VLM/fuser 配置。
- 循环调用 `V0PlannerLoop.act()`，再把动作交给 `env.makeActions()`。

`V0PlannerLoop` 每一步会：

1. 用 `MilestoneProgressController` 读出当前 milestone 状态。
2. 构造 milestone-aware state。
3. 让 VLM 对官方动作集合打 prior。
4. 让 evaluator 对所有动作预测 progress_gain、cost、next_latent。
5. 用 `DecisionFuser` 融合 evaluator 和 prior。
6. 用 `FallbackPolicy` 做安全兜底。
7. 更新 latent、动作历史和 milestone completion。

## 设计取舍

当前 V0 是工程可跑版本，不是最终论文版本：

- 优点：模块边界清楚，离线/在线链路可调试，有 JSONL 和 PNG 诊断。
- 优点：训练阶段已经接入 prior cache、prev_latent、reference-geometry labels。
- 优点：在线 milestone 不再只由 env progress 映射，而是有独立 controller。
- 局限：rollout label 仍是一阶 local world-model，不是真实 AirSim 多步 rollout。
- 局限：text encoder 是 HashTextEncoder，语义能力有限。
- 局限：VLM prior 的质量高度依赖图像路径、prompt 和 Qwen 配置。

## 最重要的检查点

- `DATA/v0/diagnostics/parse_instruction/summary.json`
- `DATA/v0/diagnostics/build_action_prior_cache/summary.json`
- `DATA/v0/diagnostics/build_rollout_labels/summary.json`
- `DATA/v0/diagnostics/build_latent_targets/summary.json`
- `DATA/v0/diagnostics/train_action_evaluator/training_log.jsonl`
- `DATA/v0/diagnostics/eval_aerialvln/planner_loop/step_decisions.jsonl`
