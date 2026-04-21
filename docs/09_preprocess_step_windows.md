# 09 步骤：build_step_windows

## 入口

文件：`preprocess/build_step_windows.py`

命令：

```bash
python preprocess/build_step_windows.py ^
  --dataset-json ../DATA/data/aerialvln-s/train.json ^
  --instruction-plan data/instruction_plan.jsonl ^
  --rgb-index data/runtime_rgb/train_index.jsonl ^
  --output data/step_windows/train.jsonl
```

## 输入

- 原始 dataset JSON。
- `instruction_plan.jsonl`。
- runtime RGB index，通常由 `preprocess/export_lmdb_rgb.py` 生成。
- 历史长度参数。
- keyframe 参数。

episode 中最好包含：

- `actions`
- `reference_path`
- RGB 路径字段，例如 `rgb_paths`、`image_paths`、`images`、`frames`

如果 baseline annotation 没有 RGB 路径，则应先运行：

```text
inference/collect_tf_rgb.py -> preprocess/export_lmdb_rgb.py
```

然后通过 `--rgb-index` 回填 `rgb_path`、`next_rgb_path` 和 `keyframe_rgb_paths`。

## 输出

- `data/step_windows/train.jsonl`
- `DATA/v0/diagnostics/build_step_windows/summary.json`
- `DATA/v0/diagnostics/build_step_windows/events.jsonl`
- `episode_step_count_distribution.png`

## 单条 step-window 主要字段

- `sample_id`
- `prev_sample_id`
- `next_sample_id`
- `instruction_id`
- `milestone_id`
- `next_milestone_id`
- `milestone`
- `milestone_text`
- `completion`
- `milestone_completion`
- `global_completion`
- `next_global_completion`
- `recent_progress_flag`
- `gt_action`
- `next_action`
- `action_history`
- `pose_deltas`
- `keyframe_indices`
- `rgb_path`
- `next_rgb_path`
- `keyframe_rgb_paths`
- `legal_action_ids`
- `reference_pose`
- `next_reference_pose`
- `prev_reference_pose`

## 逻辑

1. 加载 episode。
2. 根据 `instruction_id` 找 milestone plan。
3. 如果找不到，生成 default plan。
4. 根据 action 长度和 reference path 长度确定 steps。
5. 构造 global progress。
6. 映射 milestone id。
7. 计算 milestone-local completion。
8. 构造历史动作和 pose delta。
9. 选择关键帧 index。
10. 优先读取 episode 内置 RGB 路径，缺失时按 `trajectory_id/episode_id + step` 查 runtime RGB index。
11. 保存每一步的训练窗口。

## 分析

这是训练数据的骨架。后续 prior cache、rollout labels、latent targets 都依赖它。

关键点：

- `completion` 当前是 milestone-local completion。
- `global_completion` 单独保留，避免丢失全局进度。
- `reference_pose` 系列字段用于几何 consequence label。
- `prev_sample_id` 用于训练时读取上一时刻 latent。
- `rgb_index` 是 AirVLN baseline annotation 与 V0 视觉训练链路之间的关键接线。

## 局限

- milestone id 仍由 reference progress 初始化，不是由视觉证据得到。
- 如果 dataset 没有 RGB 路径且没有提供 `--rgb-index`，视觉相关训练会 fallback 到零图。
- 如果 reference_path 缺失，几何 consequence label 会回退成启发式。

## 排查

优先检查：

- `step_windows` 数量是否等于预期 episode steps 总和。
- `rgb_path` 是否非空。
- `summary.json` 里的 `rgb_from_index` 是否大于 0。
- `summary.json` 里的 `missing_rgb_path` 是否接近 0。
- `reference_pose` 是否非空。
- `milestone_id` 是否在 1 到 8。
