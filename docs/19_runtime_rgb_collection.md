# 19 Runtime RGB 采集与导出

## 为什么需要这一步

AirVLN baseline 的标注 JSON 通常只有：

- instruction
- actions
- reference_path
- scene/start/goal 信息

它一般没有 V0 需要的逐步图片路径：

- `rgb_path`
- `next_rgb_path`
- `keyframe_rgb_paths`

如果不先采集 runtime RGB：

- `build_action_prior_cache.py` 会退化成 text-only prior。
- `build_latent_targets.py` 会大量生成零向量 latent。
- `train_action_evaluator.py` 看到的 current/history RGB 会是零图。

## 正确路线

```text
AirVLN simulator teacher-forcing collect
  -> RGB LMDB
  -> PNG image files + JSONL index
  -> build_step_windows --rgb-index
```

## 1. 启动 simulator server

```bash
python airsim_plugin/AirVLNSimulatorServerTool.py --gpus 0 --port 30000
```

## 2. 采集 teacher-forcing RGB

```bash
python inference/collect_tf_rgb.py ^
  --split train ^
  --batchSize 1 ^
  --name v0_rgb_train ^
  --maxAction 500 ^
  --simulator_tool_port 30000
```

该脚本会强制：

```text
--run_type collect --collect_type TF --ablate_depth
```

它会沿 dataset 中的 teacher action 执行，并在每个 `get_obs()` 时把 RGB 写入 LMDB。

默认输出：

```text
../DATA/img_features/collect/v0_rgb_train/train_rgb
```

LMDB key 格式：

```text
{trajectory_id}_{step}_rgb
```

## 3. 导出 PNG 和索引

```bash
python preprocess/export_lmdb_rgb.py ^
  --lmdb-dir ../DATA/img_features/collect/v0_rgb_train/train_rgb ^
  --output-root data/runtime_rgb/train ^
  --index-output data/runtime_rgb/train_index.jsonl ^
  --dataset-json ../DATA/data/aerialvln-s/train.json
```

索引字段：

- `trajectory_id`
- `episode_id`
- `instruction_id`
- `scene_id`
- `step`
- `rgb_path`
- `lmdb_key`

## 4. 生成 step windows

```bash
python preprocess/build_step_windows.py ^
  --dataset-json ../DATA/data/aerialvln-s/train.json ^
  --instruction-plan data/instruction_plan.jsonl ^
  --rgb-index data/runtime_rgb/train_index.jsonl ^
  --output data/step_windows/train.jsonl
```

`build_step_windows.py` 会优先使用 episode 自带图像路径；如果没有，就按 `trajectory_id/episode_id + step` 查 `--rgb-index`。

## 诊断

检查：

- `DATA/v0/diagnostics/collect_tf_rgb/<split>/summary.json`
- `DATA/v0/diagnostics/export_lmdb_rgb/summary.json`
- `DATA/v0/diagnostics/build_step_windows/summary.json`

关键字段：

- `episodes_collected`
- `exported`
- `rgb_from_index`
- `missing_rgb_path`

理想情况：

```text
exported > 0
rgb_from_index > 0
missing_rgb_path 接近 0
```

## 常见问题

### LMDB 是空的

可能原因：

- simulator server 没启动。
- `--simulator_tool_port` 不一致。
- split 路径不对。
- collect 过程中 RGB 被 ablate。

### step-window 仍然没有 rgb_path

可能原因：

- `--rgb-index` 没传。
- index 中的 `trajectory_id` 与 annotation 不匹配。
- collect 的 split 和 build_step_windows 的 dataset JSON 不是同一个。

### image-root 误用

如果 `rgb_path` 是 `data/runtime_rgb/train/...` 这种相对仓库根目录的路径，后续 `build_action_prior_cache.py`、`build_latent_targets.py`、`train_action_evaluator.py` 可以不传 `--image-root`。

只有当路径是相对某个外部数据根目录时，才需要传 `--image-root`。
