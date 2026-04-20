# 15 AirVLN 环境与 AirSim 通信

## 模块位置

- `src/vlnce_src/env.py`
- `utils/env_utils.py`
- `utils/env_vector.py`
- `airsim_plugin/AirVLNSimulatorServerTool.py`
- `airsim_plugin/AirVLNSimulatorClientTool.py`
- `airsim_plugin/airsim_settings.py`

## AirVLNENV

`src/vlnce_src/env.py` 中的 `AirVLNENV` 负责：

- 加载 dataset split。
- 维护 batch。
- 打开 simulator scenes。
- reset episode 到起点。
- 获取 RGB/depth obs。
- 执行动作。
- 更新指标。

## 观测格式

`utils/env_vector.py::_format_obs_at()` 返回：

- `instruction`
- `progress`
- `teacher_action`
- `pose`
- `rgb`
- `depth`，如果未 ablate

V0 eval 会强制 `--ablate_depth`，所以 planner 主要使用 RGB。

## 动作执行

`AirVLNENV.makeActions(action_list)`：

1. 对每个 sim state 复制当前 pose。
2. 用 `getPoseAfterMakeAction()` 计算新 pose。
3. 调 simulator client 设置 poses。
4. 更新 state.step、state.pose、trajectory、pre_action。
5. 调 `update_measurements()`。

动作物理尺度在 `airsim_plugin/airsim_settings.py`：

- forward step: 5
- up/down step: 2
- left/right step: 5
- turn angle: 15 度

## 指标

`update_measurements()` 更新：

- DistanceToGoal
- Success
- nDTW
- sDTW
- PathLength
- OracleSuccess
- StepsTaken

这些最终进入 eval result JSON。

## Env progress 的注意点

在 `utils/env_utils.py` 中：

- train/collect TF 模式：progress 约等于 `step / len(actions)`。
- eval TF 模式：done 前 progress 为 0，done 后 progress 为 1。
- dagger/SF 模式：可以使用 graph-based progress sensor。

因此，V0 在线 eval 不能依赖 env progress 推进 milestone，这也是 `MilestoneProgressController` 存在的原因。

## AirSim Server

启动 simulator server：

```bash
python airsim_plugin/AirVLNSimulatorServerTool.py --gpus 0 --port 30000
```

检查：

```bash
python scripts/check_v0_setup.py --port 30000
```

## 常见问题

- 端口不通：检查 simulator server 是否启动，端口是否一致。
- batchSize 大于可用场景：降低 `--batchSize`。
- RGB 为 None：检查 AirSim scene 和 ablate 参数。
- 指标全零：检查 episode 是否真正执行动作，是否过早 STOP。
