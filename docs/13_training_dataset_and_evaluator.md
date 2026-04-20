# 13 步骤：训练 Dataset 与 Evaluator

## 入口

核心文件：

- `training/v0_dataset.py`
- `training/train_action_evaluator.py`
- `training/tune_fuser.py`

训练命令：

```bash
python training/train_action_evaluator.py ^
  --step-windows data/step_windows/train.jsonl ^
  --rollout-labels data/rollout_labels/train.jsonl ^
  --action-prior-cache data/action_prior_cache/train.jsonl ^
  --latent-index data/latent_targets/train_index.jsonl ^
  --image-root ../DATA/data/aerialvln-s ^
  --model-config configs/model.yaml ^
  --output-dir DATA/v0/checkpoints/action_evaluator ^
  --device cuda ^
  --gpu-ids 0 ^
  --batch-size 16 ^
  --epochs 10 ^
  --lr 0.00025 ^
  --prior-loss-weight 0.5
```

## V0ActionDataset

读取：

- step-window。
- rollout label。
- prior cache。
- latent target index。
- 当前 RGB 和历史 RGB。

每个训练 item 对应一个：

```text
(sample_id, action_id)
```

也就是说同一个 step-window 会展开成多个候选动作样本。

## Batch 字段

- `current_rgb`
- `history_rgbs`
- `action_history`
- `pose_deltas`
- `fallback_flags`
- `milestone_id`
- `milestone_text`
- `completion`
- `recent_progress_flag`
- `prev_latent`
- `action_id`
- `progress_label`
- `cost_label`
- `latent_target`
- `prior`

## prev_latent

dataset 会通过 `prev_sample_id` 找上一时刻 latent target：

```text
sample_t.prev_latent = latent_target(sample_{t-1})
```

如果找不到，则使用和 latent target 同维度的零向量。

分析：

- 这让训练阶段和推理阶段都具有 recurrent latent 条件。
- 第一帧仍然使用零 latent 是合理的。

## 训练模型

训练同时包含：

- `MilestoneAwareStateBuilder` 的轻量模块。
- `CausalLatentActionEvaluator`。

默认视觉骨干可能 freeze，具体由 `configs/model.yaml` 决定。

## Loss

```text
loss = progress_loss_weight * loss_progress
     + cost_loss_weight     * loss_cost
     + latent_loss_weight   * loss_latent
```

其中：

- `loss_progress` 是 weighted MSE。
- `loss_cost` 是 weighted MSE。
- `loss_latent` 是 next_latent MSE。
- `prior` 通过 `prior_loss_weight` 作为 progress/cost loss 的样本权重。

## 输出

- `DATA/v0/checkpoints/action_evaluator/ckpt_epoch_*.pth`
- `DATA/v0/checkpoints/action_evaluator/ckpt_last.pth`
- `DATA/v0/diagnostics/train_action_evaluator/training_log.jsonl`
- `loss_curve.png`
- `summary.json`

## tune_fuser

`training/tune_fuser.py` 当前是占位工具：

- 接收权重网格参数。
- 写出第一个组合。
- 没有真实跑验证集 eval。

如果要做正式 tune，需要接入 validation score function。

## 排查

- loss 全部不降：检查标签和图像是否有效。
- latent loss 极大：检查 hidden_dim 和 latent target 维度。
- progress/cost loss 异常：检查 rollout label 分布图。
- GPU 不生效：检查 `device_note` 和 `CUDA_VISIBLE_DEVICES`。
