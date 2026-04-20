# 18 风险、局限与扩展点

## 当前主要风险

### 1. 图像路径缺失

影响：

- action prior cache 退化成 text-only。
- latent targets 退化成零向量。
- evaluator 训练质量下降。

检查：

- `current_rgb_loaded`
- `keyframes_loaded`
- `encoded_from_images`
- `missing_images_zero_fallback`

### 2. Milestone 质量不稳定

影响：

- milestone_text 弱，VLM prior 变弱。
- completion controller 推进语义不清。
- state builder 的 milestone token 噪声大。

检查：

- `instruction_plan.jsonl`
- milestone count distribution。
- bad cases。

### 3. Progress controller 仍是规则化

当前 controller 已经不只依赖 env progress，但仍不是 learned verifier。

风险：

- STOP prior 高时过快推进。
- progress_gain 偏高时 completion 过快累积。
- 对视觉 cue 的判断是间接的。

扩展：

- 增加 VLM milestone verifier。
- 将 verification cues 单独传给 VLM 判断。
- 用真实 trajectory 成功/失败蒸馏 controller。

### 4. Rollout label 仍是一阶 local world-model

当前 label 比纯 action supervision 更强，但仍不是真实 AirSim rollout。

缺少：

- 多步后果。
- 碰撞检测。
- landmark visibility。
- detour 的全局路径代价。

扩展：

- 对每个候选动作在 simulator 中 rollout N 步。
- 记录 nDTW delta、distance-to-goal delta、collision。
- 将 label 从一阶几何改成真实局部后果。

### 5. HashTextEncoder 表达能力有限

风险：

- 复杂空间关系表达弱。
- 同义词不稳定。
- milestone text 的语义泛化差。

扩展：

- 接入 sentence transformer。
- 离线缓存 text embedding。
- 对 action_type、landmarks、spatial_relation、verification_cues 分字段编码。

### 6. Fuser 是手工权重

风险：

- 不同 split 最优权重不同。
- prior/evaluator 分布变化时需要重调。

扩展：

- 用 validation eval 做真实 grid search。
- 训练 learned decision head。
- 加 uncertainty 或 confidence calibration。

## 最值得优先做的改进

1. 真实 AirSim candidate rollout label。  
   这是当前 supervision 和理想 V0 差距最大的部分。
2. Milestone verification VLM。  
   让 controller 直接判断 verification cue 是否满足。
3. 更稳定 text encoder。  
   提升 milestone-aware state 的语义能力。
4. fuser validation tuning。  
   避免手工权重只适合当前小样本。
5. failure analysis 工具。  
   自动聚合早停、反复转向、一直前进、prior 崩坏等 case。

## Debug 优先级

如果效果异常，按这个顺序查：

1. 数据路径和图像加载。
2. milestone plan 质量。
3. rollout label 覆盖率和分布。
4. latent target 是否真实编码。
5. evaluator loss 曲线。
6. prior preview 是否合理。
7. step_decisions 中 milestone completion 是否推进。
8. fallback 是否频繁触发。

## 提交实验结果时建议记录

- Git commit hash。
- `configs/model.yaml`。
- `configs/fuser.yaml`。
- parser client 和 prior client。
- 所有 summary.json。
- `step_decisions.jsonl` 的失败样本片段。
- eval output JSON。
