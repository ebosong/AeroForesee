# 04 State Builder 与历史编码

## 模块位置

核心文件：

- `models/state_builder.py`
- `models/vision_backbone.py`
- `models/history_encoder.py`
- `models/trajectory_encoder.py`

## MilestoneAwareStateBuilder

`MilestoneAwareStateBuilder` 把当前观测和任务状态编码成 evaluator 使用的 token 字典。

输入：

- `current_rgb`
- `history_rgbs`
- `action_history`
- `pose_deltas`
- `milestone_ids`
- `milestone_texts`
- `completion`
- `recent_progress_flag`
- `prev_latent`
- `fallback_flags`

输出：

```text
current_visual_token
history_token
traj_token
milestone_token
progress_token
prev_latent
```

这些 token 会被 `CausalLatentActionEvaluator` 和候选 action token 组成 causal sequence。

## 视觉骨干

`VisionBackbone` 支持：

- DINOv2-S
- DINOv2-B
- ResNet50
- ResNet18

职责：

- 输入 RGB tensor。
- 输出全局 visual token。
- 对 DINOv2 还会保留 patch token。

分析：

- V0 默认 `dinov2_s`，平衡速度和表达能力。
- `vision.freeze=true` 时，训练只更新 V0 轻量模块，不更新大视觉骨干。
- 如果本地没有 DINOv2 代码或 checkpoint，首次运行可能需要联网或改用 ResNet。

## HistoryEncoder

`HistoryEncoder` 接收当前 visual token 和历史帧 token，输出 `history_token`。

历史关键帧来自：

- 离线：`build_step_windows.py` 的 `select_keyframe_indices()`。
- 在线：`EpisodeMemory.rgb_history[-max_keyframes:]`。

分析：

- 历史帧帮助判断是否重复、是否转过弯、是否在同一区域徘徊。
- 当前实现是轻量聚合，不是复杂 temporal transformer。

## TrajectoryEncoder

`TrajectoryEncoder` 编码：

- 历史动作 embedding。
- pose delta。
- fallback flag。

输出 `traj_token`。

分析：

- action history 对识别重复转向和无效动作很关键。
- pose delta 在离线训练中来自 reference path，在在线 eval 中来自真实 pose history。
- fallback flag 可以让模型知道最近是否处于保守恢复模式。

## HashTextEncoder

`state_builder.py` 内部的 `HashTextEncoder` 将 milestone_text 分词后 hash 到 embedding，再平均。

优点：

- 无外部文本模型依赖。
- 快速、稳定、易复现。

局限：

- 语义能力弱。
- hash collision 不可避免。
- 对同义词、复杂空间关系理解有限。

后续升级方向：

- 用轻量 sentence encoder。
- 离线缓存 milestone text embedding。
- 让 `verification_cues` 单独编码，而不是拼接进一段文本。

## Progress Token

`progress_mlp` 编码两个标量：

- `completion`
- `recent_progress_flag`

训练时 completion 来自 step-window 的 milestone-local completion。在线时来自 `MilestoneProgressController`。

分析：

- progress token 是 milestone-aware state 的关键。
- 如果 completion 一直不动，evaluator 会倾向认为任务没有推进。
- 在线 controller 的质量会直接影响 state builder 输入。
