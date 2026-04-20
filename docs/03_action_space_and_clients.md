# 03 动作空间与 LLM/VLM Client

## 官方动作空间

动作定义来自 `airsim_plugin/airsim_settings.py`：

```text
0 STOP
1 MOVE_FORWARD
2 TURN_LEFT
3 TURN_RIGHT
4 GO_UP
5 GO_DOWN
6 MOVE_LEFT
7 MOVE_RIGHT
```

`models/action_space.py` 将 AirSim 动作包装成 V0 统一接口：

- `action_ids`
- `official_name(action_id)`
- `prompt_name(action_id)`
- `id_from_official_name(name)`
- `prompt_action_list(ids)`
- `valid_ids(ids)`

## Prompt 动作别名

`PROMPT_ALIASES` 把官方动作名映射成适合 VLM prompt 的短词：

```text
MOVE_FORWARD -> forward
GO_UP        -> ascend
GO_DOWN      -> descend
```

分析：

- official name 用于代码和 AirSim。
- prompt name 用于 VLM 打分。
- 二者不能混用，否则 prior score 可能无法映射回 action id。

## ActionEncoder

`models/action_encoder.py` 是一个简单 embedding：

- 输入：action id。
- 输出：`token_dim` 维 action token。
- 用途：给 `CausalLatentActionEvaluator` 作为候选动作条件。

## ActionMask

`models/action_mask.py` 负责将合法动作列表转成 mask 或合法 id 列表。

当前 V0 的 step-window 默认 `legal_action_ids=list(range(8))`，也就是所有官方动作都合法。后续如果加入碰撞、禁飞高度或任务阶段限制，可以从这里接入。

## LLM Client

`models/llm_clients.py` 提供 instruction parser 使用的 client：

- `rule`：离线规则 fallback，不调用外部模型。
- `qwen_api`：通过 DashScope/OpenAI compatible API 调 Qwen。
- `qwen_local`：通过本地命令行调用。

用途：

```text
parse_instruction.py -> build_llm_client(args.client)
```

分析：

- `rule` 适合 smoke test，但 milestone 质量弱。
- `qwen_api` 适合正式构造 `instruction_plan.jsonl`。
- `qwen_local` 适合私有化部署或离线模型。

## VLM Client

`models/vlm_clients.py` 提供 action prior 使用的 client：

- `uniform`：所有动作均匀分布，用于工程测试。
- `qwen_api`：把 prompt 和图像发到 Qwen VLM。
- `qwen_local`：把 prompt、动作表、图像 base64 交给本地命令。

用途：

```text
build_action_prior_cache.py -> build_vlm_client(args.client)
run_eval_aerialvln.py       -> build_vlm_client(v0_args.vlm_client)
```

## Qwen 配置

`models/qwen_config.py` 存储：

- `QWEN_API_KEY`
- `QWEN_API_BASE_URL`
- `QWEN_TEXT_MODEL`
- `QWEN_VLM_MODEL`
- 本地 LLM/VLM command

注意：

- 不要把真实 API key 提交到仓库。
- 如果 `QWEN_API_KEY` 仍是占位符，`qwen_api` 会报错。

## 风险点

- VLM 输出必须是 JSON object，否则解析会失败。
- action_names 必须用 prompt aliases，postprocess 才能按 action id 取回分数。
- uniform client 只能验证工程链路，不能说明 VLM prior 是否有效。
