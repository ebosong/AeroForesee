# 06 Latent Evaluator、Fuser 与 Fallback

## CausalLatentActionEvaluator

文件：`models/causal_latent_action_evaluator.py`

输入：

- state token 字典。
- 候选 action id。

内部 sequence：

```text
current_visual_token
history_token
traj_token
milestone_token
progress_token
action_token
prev_latent
```

通过 causal transformer 后，最后一个 token 作为 `next_latent`。

输出：

- `progress_gain`
- `cost`
- `next_latent`

分析：

- `progress_gain` 表示执行该动作后预期能带来的局部推进。
- `cost` 表示偏离、绕路、重复、错误动作等后果代价。
- `next_latent` 用于在线 recurrent 更新，也用于训练 latent future supervision。

## 训练监督

训练时三类 loss：

- progress MSE。
- cost MSE。
- next_latent MSE。

此外：

- `prior` 会作为 progress/cost loss 的样本权重。
- `prev_latent` 会从上一时刻 latent target 读取，不再固定为零。

## DecisionFuser

文件：`models/decision_fuser.py`

融合公式：

```text
score = w_progress * progress_gain
      - w_cost * cost
      + w_prior * prior
```

输出是每个 action id 的融合分数。

分析：

- fuser 是手工权重，不是 learned policy head。
- V0 选择手工融合是为了快速消融每一项贡献。
- `training/tune_fuser.py` 当前是占位网格配置写入，不是真实验证集搜索。

## FallbackPolicy

文件：`models/fallback.py`

触发条件：

- 连续若干步 progress_gain 很低。
- 连续若干步重复 TURN_LEFT/TURN_RIGHT。
- 所有动作最高分低于阈值。

触发后：

- 优先从 `conservative_actions` 中选一个分数较高动作。
- 默认候选是 `MOVE_FORWARD` 和 `STOP`。

分析：

- fallback 不等于 recovery policy，只是 V0 安全阀。
- 它能避免模型在低置信度时继续执行极端动作。
- 如果任务需要复杂绕障，fallback 需要升级成 learned recovery 或 replan。

## 三者关系

```text
Evaluator: 预测动作后果
Prior:     给语义/视觉动作偏好
Fuser:     合并后果和先验
Fallback:  在低置信度时修正动作
```

最终动作由：

```text
FallbackPolicy.select(DecisionFuser.score(...))
```

得到。

## 常见调参

- 一直不动或过早 STOP：降低 `w_cost` 或检查 STOP prior。
- 一直前进：提高 `w_cost`，检查 rollout label 和 prior 分布。
- 连续左右转：降低 turn prior，或调小 repeated_turn_patience。
- progress_gain 全低：检查 evaluator 训练 label 和 latent target 是否有效。
