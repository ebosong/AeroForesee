# AeroForesee 文档索引

本目录是对当前 repo 的模块级说明和 V0 流程级分析。每个步骤都有独立的 Markdown 文件，便于单独阅读、排查和扩展。

## 阅读顺序

1. [00_system_overview.md](00_system_overview.md)  
   系统目标、闭环链路、核心设计取舍。
2. [01_repo_layout.md](01_repo_layout.md)  
   仓库目录、代码边界、数据与产物位置。
3. [02_configs_prompts_schemas.md](02_configs_prompts_schemas.md)  
   配置、prompt、schema 的职责和联动关系。
4. [03_action_space_and_clients.md](03_action_space_and_clients.md)  
   官方动作空间、LLM/VLM client、Qwen 配置。
5. [04_state_builder_and_history.md](04_state_builder_and_history.md)  
   视觉、历史、轨迹、milestone state 表征。
6. [05_action_prior.md](05_action_prior.md)  
   VLM action prior 的输入、归一化和 STOP 抑制。
7. [06_latent_evaluator_fuser_fallback.md](06_latent_evaluator_fuser_fallback.md)  
   causal latent evaluator、融合器、fallback。
8. [07_milestone_progress_controller.md](07_milestone_progress_controller.md)  
   evidence-driven milestone completion controller。
9. [08_preprocess_parse_instruction.md](08_preprocess_parse_instruction.md)  
   离线 instruction-to-milestone parsing。
10. [09_preprocess_step_windows.md](09_preprocess_step_windows.md)  
    step-window 构造和训练样本骨架。
11. [10_preprocess_action_prior_cache.md](10_preprocess_action_prior_cache.md)  
    离线 VLM prior cache 构造。
12. [11_preprocess_rollout_labels.md](11_preprocess_rollout_labels.md)  
    reference-geometry consequence label。
13. [12_preprocess_latent_targets.md](12_preprocess_latent_targets.md)  
    latent target 构造。
14. [13_training_dataset_and_evaluator.md](13_training_dataset_and_evaluator.md)  
    dataset、loss、训练流程。
15. [14_inference_planner_loop.md](14_inference_planner_loop.md)  
    在线 planner loop、单步决策和日志。
16. [15_airvln_env_and_airsim.md](15_airvln_env_and_airsim.md)  
    AirVLN 环境封装、AirSim 通信、指标。
17. [16_utils_scripts_backbones.md](16_utils_scripts_backbones.md)  
    工具、脚本、视觉骨干目录。
18. [17_end_to_end_pipeline.md](17_end_to_end_pipeline.md)  
    从数据到 eval 的完整命令链。
19. [18_risks_and_extension_points.md](18_risks_and_extension_points.md)  
    当前局限、排查优先级和后续扩展建议。
20. [19_runtime_rgb_collection.md](19_runtime_rgb_collection.md)
    AirVLN runtime RGB 采集、LMDB 导出和 step-window 图像路径接线。

## 文档覆盖的 repo 模块

- `configs/`
- `prompts/`
- `schemas/`
- `models/`
- `preprocess/`
- `training/`
- `inference/`
- `airsim_plugin/`
- `src/vlnce_src/`
- `utils/`
- `scripts/`
- `vision_backbones/`
- `DATA/` 输出产物约定

## 当前 V0 主链路

```text
parse_instruction
  -> build_step_windows
  -> build_action_prior_cache
  -> build_rollout_labels
  -> build_latent_targets
  -> train_action_evaluator
  -> run_eval_aerialvln
```

在线闭环：

```text
AirVLNENV.get_obs
  -> V0PlannerLoop.act
  -> MilestoneProgressController.current
  -> MilestoneAwareStateBuilder
  -> ActionPriorModule
  -> CausalLatentActionEvaluator
  -> DecisionFuser
  -> FallbackPolicy
  -> MilestoneProgressController.update
  -> AirVLNENV.makeActions
  -> AirVLNENV.update_measurements
```
