# 11 步骤：build_rollout_labels

## 入口

文件：`preprocess/build_rollout_labels.py`

命令：

```bash
python preprocess/build_rollout_labels.py ^
  --step-windows data/step_windows/train.jsonl ^
  --output data/rollout_labels/train.jsonl
```

## 输入

- step-window JSONL。
- `reference_pose`
- `next_reference_pose`
- `completion`
- `next_milestone_id`
- `gt_action`
- `next_action`
- `action_history`

## 输出

- `data/rollout_labels/train.jsonl`
- `DATA/v0/diagnostics/build_rollout_labels/summary.json`
- `positive_progress_by_action.png`
- `average_cost_by_action.png`

单行结构：

```json
{
  "sample_id": "...",
  "labels": {
    "0": {"progress": 0.0, "cost": 1.0},
    "1": {"progress": 0.8, "cost": 0.1}
  }
}
```

## 当前标签策略

优先使用 reference-geometry local world-model：

1. 读取当前 reference pose。
2. 读取下一 reference pose。
3. 对每个候选动作模拟一步 pose 后果。
4. 计算候选 pose 相对 reference segment 的推进量。
5. 计算到下一参考点的偏离。
6. 计算 heading alignment。
7. 得到 progress/cost。

如果缺少 reference pose，则回退到启发式 consequence label。

## 几何标签的含义

progress 由三项组成：

- 沿 reference segment 的投影推进。
- 到下一 reference point 的距离改善。
- heading alignment。

cost 由三项组成：

- off-route distance。
- 反向移动惩罚。
- heading mismatch。

STOP 特殊处理：

- milestone 接近完成或即将切换时 cost 低。
- milestone 未完成时 cost 高。

## 启发式 fallback

缺 reference pose 时，规则考虑：

- 是否等于 gt_action。
- 是否等于 next_action。
- STOP 时机。
- 反向动作。
- 重复无效动作。
- milestone 是否切换。

## 诊断字段

`summary.json` 中新增：

- `geometric_action_labels`
- `heuristic_action_labels`

如果 `heuristic_action_labels` 很高，说明 step-window 缺 reference pose，训练标签会弱化。

## 分析

这一步决定 evaluator 学到的是 action imitation 还是 consequence prediction。

当前版本已经不再只是 action match：

- 它会比较候选动作的一步几何后果。
- 它会给非 gt_action 但几何合理的动作一定 progress。
- 它会惩罚偏离 reference segment 的动作。

## 局限

- 仍是一阶 local world-model，不是真实 AirSim 多步 rollout。
- 没有检查碰撞、可见性变化、动态障碍。
- 对转向动作的好坏主要依赖 heading alignment。

## 后续升级

- 调 AirSim 对候选动作 rollout N 步，真实测 collision、distance、nDTW delta。
- 加入 milestone-specific target point，而不只是下一 reference point。
- 加入视觉变化指标，例如 landmark visibility。
