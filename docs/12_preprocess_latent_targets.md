# 12 步骤：build_latent_targets

## 入口

文件：`preprocess/build_latent_targets.py`

命令：

```bash
python preprocess/build_latent_targets.py ^
  --step-windows data/step_windows/train.jsonl ^
  --output-dir data/latent_targets/train ^
  --index-output data/latent_targets/train_index.jsonl ^
  --model-config configs/model.yaml ^
  --image-root ../DATA/data/aerialvln-s ^
  --device cuda ^
  --gpu-ids 0
```

## 输入

- step-window JSONL。
- `next_rgb_path`。
- model config。
- image root。
- device/gpu settings。

## 输出

- `data/latent_targets/train/*.pt`
- `data/latent_targets/train_index.jsonl`
- `DATA/v0/diagnostics/build_latent_targets/summary.json`

index 行结构：

```json
{
  "sample_id": "...",
  "latent_target": "data/latent_targets/train/xxx.pt"
}
```

## 逻辑

1. 读取 `configs/model.yaml`。
2. 初始化 `VisionBackbone`。
3. 对每个 step-window 读取 `next_rgb_path`。
4. 用视觉骨干编码下一帧 RGB。
5. 保存视觉 token 为 latent target。
6. 写出 index JSONL。

如果下一帧图像缺失：

- 保存全零 latent。
- `missing_images_zero_fallback` 加一。

## 诊断字段

重点看：

- `encoded_from_images`
- `missing_images_zero_fallback`
- `token_dim`
- `device`
- `device_note`

## 分析

latent target 是 evaluator 的 future supervision。

训练目标：

```text
current state + action + prev_latent -> next_latent
```

其中 `next_latent` 被监督为下一帧视觉 token。

当前 dataset 会把上一时刻 latent target 作为 `prev_latent`，让训练和推理都更接近 recurrent latent 更新。

## 局限

- latent target 是视觉 token，不是完整 world state。
- 一步 supervision 不包含多步 rollout。
- 图像缺失时零 latent 会削弱训练。

## 排查

如果训练 latent loss 异常：

1. 检查 `encoded_from_images` 是否太低。
2. 检查 `token_dim` 是否与 `model.hidden_dim` 一致。
3. 检查 image root 是否正确。
4. 检查 DINOv2/ResNet 权重是否成功加载。
