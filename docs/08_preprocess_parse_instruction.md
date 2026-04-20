# 08 步骤：parse_instruction

## 入口

文件：`preprocess/parse_instruction.py`

典型命令：

```bash
python preprocess/parse_instruction.py ^
  --dataset-json ../DATA/data/aerialvln-s/train.json ^
  --output data/instruction_plan.jsonl ^
  --client qwen_api
```

Smoke test：

```bash
python preprocess/parse_instruction.py ^
  --dataset-json ../DATA/data/aerialvln-s/train.json ^
  --output data/instruction_plan.jsonl ^
  --client rule
```

## 输入

- dataset JSON。
- milestone prompt。
- LLM client 类型。

dataset episode 需要包含 instruction 文本。读取逻辑在 `preprocess/common.py`：

- `load_episodes()`
- `instruction_id()`
- `instruction_text()`

## 输出

- `data/instruction_plan.jsonl`
- `data/bad_cases.jsonl`
- `DATA/v0/diagnostics/parse_instruction/summary.json`
- `DATA/v0/diagnostics/parse_instruction/events.jsonl`
- `DATA/v0/diagnostics/parse_instruction/milestone_count_distribution.png`

每行 plan 结构：

```json
{
  "instruction_id": "...",
  "milestones": [
    {
      "mid": 1,
      "action_type": "...",
      "landmarks": ["..."],
      "spatial_relation": "...",
      "verification_cues": ["..."]
    }
  ]
}
```

## 逻辑

1. 读取 dataset episodes。
2. 用 prompt 和 instruction 构造 LLM 输入。
3. 调用 LLM client 输出 JSON。
4. 写入 `instruction_id`。
5. 用 `validate_instruction_plan()` 校验。
6. 成功写入 good rows。
7. 失败写入 bad cases。

## 分析

这是整个 V0 的语义入口。后续模块不会再重新理解整条长指令，而是依赖 milestone_text。

质量影响：

- milestone 太粗：planner 难以分阶段。
- milestone 太细：completion controller 易频繁切换。
- 缺 verification_cues：未来做证据驱动 completion 会缺依据。

## 排查

看 `summary.json`：

- `valid` 是否接近 `total`。
- `bad` 是否异常高。

看 `milestone_count_distribution.png`：

- 是否集中在 3 到 8。
- 是否大量只有 3 个 milestone，说明 parser 过粗。

## 局限

- 当前 parser 是离线一次性解析。
- 在线不会 reparse 或动态 replan milestone。
- bad case 只输出，不自动修复。
