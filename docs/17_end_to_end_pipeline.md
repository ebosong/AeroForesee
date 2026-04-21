# 17 端到端流程

本文件给出从原始数据到在线 eval 的完整步骤。每一步都可以单独运行和排查。

## 0. 环境检查

```bash
python scripts/check_v0_setup.py --port 30000
```

如果要跑完整 eval，需要先启动 AirSim simulator server。

## 1. 解析 instruction

```bash
python preprocess/parse_instruction.py ^
  --dataset-json ../DATA/data/aerialvln-s/train.json ^
  --output data/instruction_plan.jsonl ^
  --bad-output data/bad_cases.jsonl ^
  --client qwen_api
```

产物：

- `data/instruction_plan.jsonl`
- `data/bad_cases.jsonl`

检查：

- `DATA/v0/diagnostics/parse_instruction/summary.json`

## 2. 构造 step windows

AirVLN baseline annotation 通常不带逐步 RGB 文件路径。先沿 teacher-forcing 轨迹采集 simulator RGB：

```bash
python airsim_plugin/AirVLNSimulatorServerTool.py --gpus 0 --port 30000
```

```bash
python inference/collect_tf_rgb.py ^
  --split train ^
  --batchSize 1 ^
  --name v0_rgb_train ^
  --maxAction 500 ^
  --simulator_tool_port 30000
```

再把 LMDB 导出成 PNG 和索引：

```bash
python preprocess/export_lmdb_rgb.py ^
  --lmdb-dir ../DATA/img_features/collect/v0_rgb_train/train_rgb ^
  --output-root data/runtime_rgb/train ^
  --index-output data/runtime_rgb/train_index.jsonl ^
  --dataset-json ../DATA/data/aerialvln-s/train.json
```

## 3. 构造 step windows

```bash
python preprocess/build_step_windows.py ^
  --dataset-json ../DATA/data/aerialvln-s/train.json ^
  --instruction-plan data/instruction_plan.jsonl ^
  --rgb-index data/runtime_rgb/train_index.jsonl ^
  --output data/step_windows/train.jsonl
```

产物：

- `data/step_windows/train.jsonl`

检查：

- step-window 数量。
- RGB path。
- reference pose。

## 4. 构造 action prior cache

```bash
python preprocess/build_action_prior_cache.py ^
  --step-windows data/step_windows/train.jsonl ^
  --output data/action_prior_cache/train.jsonl ^
  --client qwen_api
```

检查：

- `current_rgb_loaded`
- `keyframes_loaded`
- top action distribution。

## 5. 构造 rollout labels

```bash
python preprocess/build_rollout_labels.py ^
  --step-windows data/step_windows/train.jsonl ^
  --output data/rollout_labels/train.jsonl
```

检查：

- `geometric_action_labels`
- `heuristic_action_labels`
- average cost by action。

## 6. 构造 latent targets

```bash
python preprocess/build_latent_targets.py ^
  --step-windows data/step_windows/train.jsonl ^
  --output-dir data/latent_targets/train ^
  --index-output data/latent_targets/train_index.jsonl ^
  --model-config configs/model.yaml ^
  --device cuda ^
  --gpu-ids 0
```

检查：

- `encoded_from_images`
- `missing_images_zero_fallback`

## 7. 训练 evaluator

```bash
python training/train_action_evaluator.py ^
  --step-windows data/step_windows/train.jsonl ^
  --rollout-labels data/rollout_labels/train.jsonl ^
  --action-prior-cache data/action_prior_cache/train.jsonl ^
  --latent-index data/latent_targets/train_index.jsonl ^
  --model-config configs/model.yaml ^
  --output-dir DATA/v0/checkpoints/action_evaluator ^
  --device cuda ^
  --gpu-ids 0 ^
  --batch-size 16 ^
  --epochs 10 ^
  --lr 0.00025 ^
  --prior-loss-weight 0.5
```

检查：

- `training_log.jsonl`
- `loss_curve.png`
- `ckpt_last.pth`

## 8. 启动 simulator server

```bash
python airsim_plugin/AirVLNSimulatorServerTool.py --gpus 0 --port 30000
```

## 9. 在线 eval

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
  --score-preview-steps 5 ^
  --stop-completion-threshold 0.35 ^
  --batchSize 1 ^
  --EVAL_DATASET val_unseen ^
  --EVAL_NUM -1 ^
  --maxAction 500 ^
  --collect_type TF ^
  --simulator_tool_port 30000
```

检查：

- `eval_metrics.png`
- `planner_loop/step_decisions.jsonl`
- 每步 action、fallback、milestone completion。

## 推荐调试顺序

1. 用 `rule` 和 `uniform` 跑通所有文件产物。
2. 切换 parser 到 `qwen_api`。
3. 切换 prior 到 `qwen_api`。
4. 确保 geometric labels 覆盖率高。
5. 训练 evaluator。
6. 用 `--EVAL_NUM 1 --batchSize 1` 跑短 eval。
7. 再跑完整 split。
