# 02 配置、Prompt 与 Schema

## configs/

### `configs/model.yaml`

控制视觉骨干、历史长度、轨迹长度、隐藏维度和训练默认值。

关键字段：

- `vision.backbone`：`dinov2_s`、`dinov2_b`、`resnet50`、`resnet18`。
- `vision.pretrained`：是否加载预训练权重。
- `vision.freeze`：是否冻结视觉骨干。
- `vision.image_height / image_width`：训练和推理统一 resize 尺寸。
- `history.max_keyframes`：历史关键帧数量。
- `trajectory.history_len`：历史动作和 pose delta 长度。
- `model.hidden_dim`：state token 和 latent token 维度。
- `model.max_milestones`：milestone id embedding 上限。

分析：

- `hidden_dim` 会影响 state builder、evaluator、latent target 三处，不能只改一个地方。
- `vision.freeze=true` 更适合 V0 小规模验证，降低训练成本。
- `history.max_keyframes` 越大，显存压力越高，但 prior 和 state builder 都能看到更长历史。

### `configs/fuser.yaml`

控制手工融合器和 fallback。

关键字段：

- `w_progress`：evaluator progress_gain 权重。
- `w_cost`：evaluator cost 惩罚权重。
- `w_prior`：VLM action prior 权重。
- `fallback.low_progress_threshold`：低进展判定。
- `fallback.low_score_threshold`：低融合分判定。
- `fallback.progress_patience`：连续低进展触发步数。
- `fallback.repeated_turn_patience`：连续转向触发步数。
- `fallback.conservative_actions`：fallback 候选动作。

分析：

- `w_prior` 太高时容易被 VLM 文本/视觉误判带偏。
- `w_cost` 太高时动作会保守，可能过早 STOP 或少转向。
- fallback 是在线安全阀，不参与训练 loss。

### `configs/base.yaml`

用于记录工作区、数据、环境和输出目录的默认约定。当前主脚本更多依赖命令行参数和 README 中的路径约定。

## prompts/

### `prompts/milestone_prompt.txt`

给 instruction parser 使用。要求 LLM 输出 JSON，并将长指令拆成 3 到 8 个 milestone。

每个 milestone 应包含：

- `mid`
- `action_type`
- `landmarks`
- `spatial_relation`
- `verification_cues`

分析：

- `verification_cues` 对在线 milestone progress controller 很重要，因为它会进入 milestone_text。
- 如果 prompt 只输出抽象 action，VLM prior 和 state builder 都会变弱。

### `prompts/action_prior_prompt.txt`

给 VLM action prior 使用。输入会追加：

- 当前 milestone 文本。
- progress summary。
- 官方动作表。
- 当前 RGB 和最近关键帧。

分析：

- VLM 必须对官方动作集合打分，不能自由生成动作。
- `ActionPriorModule.postprocess()` 会过滤非法动作并归一化。
- milestone completion 低时会抑制 STOP，避免过早结束。

## schemas/

### `schemas/milestone_schema.py`

负责校验 LLM 输出的 milestone plan。

主要约束：

- milestone 数量在 3 到 8。
- `mid` 递增并从 1 开始。
- 每个 milestone 字段类型正确。
- 输出再转回普通 dict，供 JSONL 保存。

分析：

- schema 是离线 parser 的第一道防线。
- 不合法样本会进入 bad cases，避免污染 step-window。
- schema 保证了在线 `MilestoneProgressController` 能稳定读取 `mid` 和文本字段。

## 修改顺序建议

如果要改变 milestone 表达：

1. 改 `prompts/milestone_prompt.txt`。
2. 改 `schemas/milestone_schema.py`。
3. 改 `preprocess/common.py::milestone_to_text`。
4. 改 `models/milestone_progress.py::_milestone_text`。
5. 重新生成 `instruction_plan.jsonl` 和 `step_windows`。
