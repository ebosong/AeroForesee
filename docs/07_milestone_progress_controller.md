# 07 Milestone Progress Controller

## 模块位置

文件：`models/milestone_progress.py`

在线使用位置：

- `inference/planner_loop.py`
- `EpisodeMemory.progress_controller`

## 设计动机

早期版本中，当前 milestone 主要由 env 的 global progress 映射：

```text
idx = int(progress * len(milestones))
```

这个做法在训练数据中还可用，因为 step-window 有 reference action progress；但在线 eval 的 AirVLN TF 模式中，done 之前 progress 常常是 0，done 后才是 1。这样会导致 milestone 长时间停在第一个阶段。

`MilestoneProgressController` 的目标是让 milestone 推进不只依赖 env progress，而是结合在线证据。

## 状态

每个 episode 持有一个 controller，内部状态包括：

- `active_index`
- `completion`
- `steps_in_milestone`
- `advance_completion`
- `stop_prior_threshold`
- `min_steps_per_milestone`

## current()

`current(item, global_progress)` 返回当前 milestone 状态：

```python
MilestoneStatus(
    text=...,
    milestone_id=...,
    completion=...,
    index=...
)
```

如果 env global progress 可用，并且显示已经进入更后 milestone，controller 会用它做校正。

## update()

每步动作选择后调用：

输入：

- 当前 episode item。
- chosen action。
- VLM prior。
- evaluator progress_gain。
- env global progress。

更新逻辑：

- motion action 给 completion 增量。
- turn action 给较小 completion 增量。
- evaluator progress_gain 给额外增量。
- STOP prior 高时，将 completion 抬高到接近完成。
- completion 超过阈值且满足最少步数后，允许进入下一 milestone。

## 证据来源

当前 controller 使用：

- VLM prior 中 STOP 分数。
- 已选动作类型。
- evaluator progress_gain。
- 可用 global progress。

它还没有直接使用图像检测器判断 verification cue 是否出现。V0 里的视觉证据通过 VLM prior 间接进入 controller。

## 与 planner_loop 的关系

每一步开始：

```text
status = controller.current(...)
```

用于构建：

- milestone_text
- milestone_id
- completion

每一步结束：

```text
controller.update(...)
```

用于推进下一步状态。

诊断日志会写：

- `milestone_id`
- `milestone_text`
- `milestone_completion`
- `updated_milestone_id`
- `updated_milestone_completion`

## 当前局限

- completion 增量仍是规则化 controller，不是单独训练的 progress head。
- 没有显式解析 `verification_cues` 并让 VLM 判断 cue 是否满足。
- STOP prior 高时可能推进过快，需要结合 score logs 检查。

## 后续升级

- 增加 `MilestoneVerifier`，让 VLM 对 verification_cues 输出 completed / not completed。
- 将 controller 参数暴露到 YAML。
- 用验证集 tune `advance_completion` 和 `stop_prior_threshold`。
- 将 controller 输出作为训练数据，蒸馏成 learned progress controller。
