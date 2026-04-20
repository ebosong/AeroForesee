# 05 VLM Action Prior

## 模块位置

核心文件：

- `models/action_prior.py`
- `models/vlm_clients.py`
- `models/action_space.py`
- `prompts/action_prior_prompt.txt`
- `preprocess/build_action_prior_cache.py`

## ActionPriorModule 的职责

`ActionPriorModule.score()` 给官方动作集合打语义先验分。

输入：

- 当前 RGB。
- 最近 1 到 2 张关键帧。
- 当前 milestone 文本。
- progress summary。
- legal action ids。
- milestone completion。

输出：

```python
Dict[int, float]
```

其中 key 是 action id，value 是归一化后的 prior score。

## Prompt 内容

`ActionPriorModule._build_prompt()` 会把三类信息拼进 prompt：

```text
Current milestone: ...
Current progress summary: ...
Official action list: [...]
```

图像不写进文本，而是通过 VLM client 作为 image input 传入。

## 后处理逻辑

`postprocess()` 做三件事：

1. 只保留 legal action ids。
2. 将负分裁剪到 0。
3. 对所有合法动作归一化。

额外规则：

- 如果 milestone completion 小于 `stop_completion_threshold`，STOP 分数会被压到最多 `0.05`。

分析：

- 这能减少早停，但也可能在 milestone 实际完成但 controller completion 偏低时压制 STOP。
- `stop_completion_threshold` 默认 0.35，由 eval 参数传入。

## 离线 prior cache

`preprocess/build_action_prior_cache.py` 对所有 step-window 构造 prior：

- 读取 `rgb_path`。
- 读取最近 keyframe paths。
- 调 `ActionPriorModule.score()`。
- 写出 `data/action_prior_cache/train.jsonl`。

诊断：

- `current_rgb_loaded`
- `keyframes_loaded`
- `prior_preview.jsonl`
- `top_prior_action_distribution.png`

分析：

- 如果 `current_rgb_loaded=0`，prior cache 实际退化成 text-only。
- 如果使用 `uniform` client，所有动作近似均匀，只能测试流程。
- 如果使用 `qwen_api`，需要确保 API key、网络和图像路径都可用。

## 在线 prior

在线 `V0PlannerLoop` 每一步直接调用 action prior：

```text
current_rgb = 当前 AirSim RGB
keyframes = memory.rgb_history[-2:]
milestone_text = controller.current().text
progress_summary = completion/global_progress/recent_actions
```

prior 会参与：

- `DecisionFuser.score()`
- `MilestoneProgressController.update()` 的 STOP 证据

## 典型问题

- VLM prior 偏向 MOVE_FORWARD：可能导致一直前进不转向。
- VLM prior 偏向 STOP：可能导致早停，需要检查 completion 和 STOP 抑制。
- prior cache 和在线 prior 分布不一致：通常是离线图像路径缺失或在线 prompt 信息不同。
