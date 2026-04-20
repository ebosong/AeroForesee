# 10 步骤：build_action_prior_cache

## 入口

文件：`preprocess/build_action_prior_cache.py`

命令：

```bash
python preprocess/build_action_prior_cache.py ^
  --step-windows data/step_windows/train.jsonl ^
  --output data/action_prior_cache/train.jsonl ^
  --image-root ../DATA/data/aerialvln-s ^
  --client qwen_api
```

Smoke test：

```bash
python preprocess/build_action_prior_cache.py ^
  --step-windows data/step_windows/train.jsonl ^
  --output data/action_prior_cache/train.jsonl ^
  --client uniform
```

## 输入

- step-window JSONL。
- 当前 RGB path。
- keyframe RGB paths。
- milestone_text。
- completion/global_completion。
- action history。
- VLM client。

## 输出

- `data/action_prior_cache/train.jsonl`
- `DATA/v0/diagnostics/build_action_prior_cache/summary.json`
- `prior_preview.jsonl`
- `top_prior_action_distribution.png`

单行结构：

```json
{
  "sample_id": "...",
  "prior": {
    "0": 0.01,
    "1": 0.45
  }
}
```

## 逻辑

1. 读取 step-windows。
2. 尝试从 `rgb_path` 加载当前 RGB。
3. 尝试从 `keyframe_rgb_paths` 加载最近关键帧。
4. 构造 progress summary。
5. 调 `ActionPriorModule.score()`。
6. 写出每个 sample 的 prior。
7. 统计 top action 分布。

## 诊断字段

`summary.json` 中重点看：

- `step_windows`
- `cached_priors`
- `client`
- `current_rgb_loaded`
- `keyframes_loaded`

如果 `current_rgb_loaded=0`，离线 prior cache 没有真正吃视觉输入。

## 分析

prior cache 进入训练：

- `V0ActionDataset` 会读取 prior。
- `train_action_evaluator.py` 用 prior 做 progress/cost loss 的样本权重。

prior cache 也可以用于诊断 VLM 直觉：

- 是否总是偏向 MOVE_FORWARD。
- 是否在 completion 很低时仍偏向 STOP。
- 是否对 turning milestone 给出 turn action。

## 局限

- 离线 prior 是单步打分，不会考虑未来多步。
- VLM 分数依赖 prompt 和图像质量。
- 如果 RGB 路径不在 dataset 中，需要补 image_root 或重新导出图像路径。
