# 阶段 4A 验收包：CNN Actor/Critic 与 Affordance Query Head

日期：2026-09-07。状态：实现、完整工程回归与公共 CLI 验证完成，待用户验收；阶段 4B/5 未开始。

## 1. 交付范围

新增两个独立任务，均使用既有 Interaction 环境实现：

| Task | Tracker | Head | Algorithm | 有效辅助权重 |
|---|---|---|---|---:|
| `ISAL-Humanoid-Rough-CNN-v0` | 关闭 | 不创建 | 普通 PPO | 0 |
| `ISAL-Humanoid-Rough-CNN-Aux-v0` | 开启 | 创建 | 普通 PPO | 0 |

Aux-only 在本阶段仅有采样接口和可调用预测器。普通 PPO 的 `enable_aux_loss=False`，
没有辅助 buffer、loss 接入、schedule 或额外 optimizer，也不把预测输入 Actor。
实际标签公式、reward、terrain、curriculum、动作控制、随机化和旧任务行为保持不变。

本机没有执行策略训练。仿真测试对 `OnPolicyRunner.learn`、`PPO.update`、
Adam/AdamW/SGD `step` 设置了禁止调用断言；所有控制动作均为固定零动作。
纯 PyTorch 测试仅计算前向和梯度，没有执行 optimizer 更新。

### 新增文件

| 文件（相对仓库根目录） | 用途 |
|---|---|
| `isal/isal/learning/__init__.py` | 无仿真依赖的学习包 |
| `isal/isal/learning/config.py` | 网络参数和实际环境布局绑定 |
| `isal/isal/learning/models/__init__.py` | 公共模型导出 |
| `isal/isal/learning/models/affordance_actor_critic.py` | CNN、Actor/Critic、query head、normalization、checkpoint 元数据、独立导出 |
| `isal/isal/tasks/direct/humanoid_rough/agents/affordance_agent_cfg.py` | 两个变体的独立 policy/agent 配置 |
| `isal/scripts/export_actor.py` | 不启动 Isaac Sim 的 checkpoint 导出 CLI |
| `isal/tests/stage4a/conftest.py` | 纯张量测试输入及 CPU/CUDA fixture |
| `isal/tests/stage4a/test_model.py` | 形状、坐标、梯度、初始化、归一化、checkpoint 和 tracker→head |
| `isal/tests/stage4a/test_export.py` | TorchScript、ONNX Runtime、动态 batch 和无 head 图检查 |
| `isal/tests/stage4a/test_runtime.py` | 配置公平性与独立进程真实仿真 |
| `isal/STAGE4A_ACCEPTANCE.md` | 本验收记录 |

### 修改文件

- `isal/isal/tasks/direct/humanoid_rough/isal_env_cfg.py`：新增两个配置子类，仅切换 tracker。
- `isal/isal/tasks/direct/humanoid_rough/__init__.py`：新增两个 Gym task 注册。
- `isal/scripts/rsl_rl/train.py`：创建实际环境后、runner 前，仅为新模型绑定布局。
- `isal/README.md`：参数、验证、checkpoint、导出命令及阶段边界。

根目录 `ISAL_SPEC_AND_PLAN_REVISION_REVIEW.md` 是上一轮已存在的未提交审查草稿，不属于本阶段新增实现。

## 2. 关键设计决策

### 网络与历史

Actor 继续使用独立的 `policy` 和 `height_scan` observation groups；未改变环境 observation schema。
默认 proprio 为 `[B,780]`，扫描为 `[B,187]`，在模型内部变为 `[B,1,17,11]`。

- Actor CNN：channels `[16,32,32]`，kernel 3、stride `[1,2,1]`、padding 1、ELU。
- 实际 dummy forward 推导 `flat_dim`；默认为 1728，随后 `128→64` projection。
- Proprio：`780→256→128`；Actor body：`192→256→128→23`。
- Critic 保留全部历史，默认 `[B,10,326]`；每帧分为 139 维状态与 187 点 clean scan。
- Critic 状态历史 `1390→256→128`；每帧扫描通过同一 Critic 专用 CNN，按历史顺序拼接 10 个 64 维 latent，再与状态 latent 融合，经 `256→128→1` 输出 value。
- Actor/Critic 参数独立，action distribution 沿用 scalar std、初值 1.0，无新增 action clip。
- Head：`77→128→64→1→Sigmoid`，输入采用当前阶段 3 packet 的连续 query 与物理尺度 context。

Head 在公共网络构建完成后初始化，保存并恢复 RNG 状态。
相同 seed 的两个变体公共参数逐 tensor 相等；开启 head 不改变后续 CPU/CUDA RNG 状态。

### 动态布局与归一化

`bind_perceptive_model_config(env, agent_cfg)` 从实际环境获取 H/W、ordering、状态维度和历史长度，
同时保存最终 perception 参数。构建在所有 CLI/Hydra 覆盖之后执行。旧任务调用该函数会直接返回，配置不变。

模型使用 FQN：

```text
isal.learning.models.affordance_actor_critic:AffordanceActorCritic
```

不修改上游 RSL-RL 注册表、runner 或 PPO 源码。模型导入测试确认无需加载 `isaaclab.app` 或 `isal.tasks`。

扫描保持固定 `clamp(terrain_z-root_z,-1.5,0.4)/0.5`。
Actor 本体历史、Critic 非扫描状态历史分别启用独立运行均值方差归一化，且只有
`update_normalization(obs)` 可以更新统计。辅助预测、普通前向和导出不会更新统计。
新 agent 的 deprecated `empirical_normalization` 设为 None，实际开关显式放在 policy 中。

### 梯度和行为隔离

| 梯度来源 | 允许更新的可训练模块 |
|---|---|
| auxiliary SmoothL1 | Actor terrain encoder、head |
| Actor likelihood / entropy | Actor proprio encoder、Actor terrain encoder、Actor body、std |
| Actor symmetry 路径 | Actor proprio encoder、Actor terrain encoder、Actor body |
| value | Critic state encoder、Critic terrain encoder、Critic body |

逐参数检查不允许的梯度为 None；每个允许模块均存在非零梯度。
辅助预测不读取 target、不替换动作分布。将 target 改为 NaN 不改变预测；修改或移除 head 不改变 Actor action mean。

### Checkpoint 与导出

模型 state dict 保存布局、网络、variant、预处理和 context 定义元数据，以及 normalizer buffers。
加载前先检查元数据，即使 `strict=False` 也拒绝旧 MLP 或不匹配的输入定义。
模型 `load_state_dict` 返回当前 RSL-RL runner 所要求的 resume boolean。

真实 runner 测试保存未训练 checkpoint，再临时改动 Actor 参数、normalizer、optimizer 学习率和 iteration，
随后加载，逐项确认恢复。iteration=7 是明确标注的合成 checkpoint 元数据，不代表执行过 7 次训练。
Optimizer 没有做过 step，因此没有已训练的 Adam moment；本阶段验证其参数组保存恢复和标准 runner 接口。

独立导出 wrapper 复制 Actor normalizer、proprio encoder、terrain encoder 和 action mean body。
它不包含 Critic、head、std 或仿真状态，不改变原模型的 train/eval 状态。
导出 TorchScript 和 ONNX opset 17；固定 feature dimensions，支持动态 batch。

默认 `17×11 / 10+10 history` Baseline 参数量：

| 部分 | 参数量 |
|---|---:|
| Actor 部署网络 | 561,719 |
| Critic | 862,497 |
| Action std | 23 |
| Baseline 总计 | 1,424,239 |
| Aux-only 额外 head | 18,305 |

同布局 Aux-only 的总量为公共网络加 head；部署 Actor 结构与参数量相同。

## 3. 测试命令和实际结果

工作目录：`C:\Users\Admin\Documents\ISAL`。所有 Python 均通过 `env_isaaclab`。

### 完整阶段 1～4A 回归

```powershell
conda run --no-capture-output -n env_isaaclab python -u -m pytest `
  isal/tests -q --tb=short -p no:cacheprovider `
  > outputs/stage4a_full_acceptance_tests.log 2>&1
```

**121 passed, 4 warnings in 180.94s**。既有 83 项全部通过，新增 38 项通过，无 skip。

4 个 pytest warning 为一个旧任务 `empirical_normalization` 弃用提示，以及三个 PyTorch legacy ONNX exporter 弃用提示。
Isaac Sim 退出时仍打印已有的 USD detach / recursive unload 信息；测试报告与导出正常完成。

覆盖：

- CPU/CUDA，`17×11`、`17×9`、`19×11`，`xy/yx`，默认与 `3/4` 历史组合。
- 带帧标记的解析高度 `x+10y`，逐帧验证空间顺序和历史保留，不仅验证 reshape 尺寸。
- head 开关初始化/RNG/action 完全一致，head 调用、修改和移除不影响 policy。
- auxiliary、Actor likelihood/entropy、symmetry、value 四条梯度路由。
- Normalizer 不统计 scan，辅助调用不改变统计。
- 模型 checkpoint 等价恢复；相同点数但不同 H/W、ordering 或 clip 参数时拒绝加载。
- TorchScript / ONNX Runtime 在 batch=1 和 batch=4 下与原模型对齐，`rtol=1e-4, atol=1e-5`。
- 导出 state/ONNX initializer 中不包含 Critic/head。
- 使用真实 `FootInteractionTracker`、合成接触反馈产生 2 环境×双脚的 4 条已结算样本，
  按 `[N,2,P]` valid mask 展平后接入 head 并完成辅助 backward；这不是自然步态数据。

### 独立真实仿真与 runner

```powershell
conda run --no-capture-output -n env_isaaclab python -u -m pytest `
  isal/tests/stage4a/test_runtime.py -q --tb=short -p no:cacheprovider `
  > outputs/stage4a_runtime_tests.log 2>&1
```

**4 passed in 73.29s**。包含 1 项完整配置公平性测试和 3 个独立进程场景；完整回归中再次通过。

| 场景 | 环境数 | Grid / ordering | Actor/Critic history | 固定动作步数 | 自然结算样本 |
|---|---:|---|---|---:|---:|
| baseline_default | 1 | 17×11 / xy | 10 / 10 | 40 | 0；tracker 关闭 |
| aux_small_xy | 2 | 17×9 / xy | 3 / 4 | 40 | 0 |
| aux_large_yx | 2 | 19×11 / yx | 3 / 4 | 40 | 0 |

每个场景验证真实注册、布局绑定、标准 runner 加载、reset/step、镜像后的 Actor/Critic 前向、
checkpoint、TorchScript 和 ONNX 对齐。权重在测试前后逐参数相等，普通 PPO auxiliary loss 始终关闭。

40 步仅为 0.8 秒固定零动作检查，未产生自然结算标签；不能据此评价标签分布、步态或 head 预测质量。
Tracker→head 的接口覆盖由上述明确标记的合成生命周期测试提供。

### 公共 CLI

```powershell
conda run --no-capture-output -n env_isaaclab python -u `
  isal/scripts/rsl_rl/train.py --task ISAL-Humanoid-Rough-CNN-v0 `
  --headless --num_envs 1 --validate-only `
  > outputs/stage4a_validate_baseline.log 2>&1

conda run --no-capture-output -n env_isaaclab python -u `
  isal/scripts/rsl_rl/train.py --task ISAL-Humanoid-Rough-CNN-Aux-v0 `
  --headless --num_envs 1 --validate-only `
  'env.terrain_perception.size=[1.8,1.0]' `
  env.robot.actor_obs_history_length=3 env.robot.critic_obs_history_length=4 `
  env.scene.height_scanner.pattern_cfg.ordering=yx `
  > outputs/stage4a_validate_aux_dynamic.log 2>&1
```

两条命令均完成全部 validation 输出，无 Python 异常：

| 检查 | Baseline | Aux-only 动态配置 |
|---|---|---|
| policy | (1,780) | (1,234) |
| height_scan | (1,187) | (1,209) |
| critic | (1,3260) | (1,1392) |
| CNN grid | (1,1,17,11) | (1,1,19,11) |
| action | (1,23) | (1,23) |
| dynamic_augmentation | True | True |
| reward_finite | True | True |
| stepped_with_zero_actions | True | True |
| runner.learn_called | False | False |

Baseline extras 无 auxiliary；Aux-only extras 包含 auxiliary 和 interaction_stats。
CLI 中的 `actor_input_shape` 表示模型拆分后的 proprio 分支，不包含独立 scan。

两条 CLI 日志仍有本机已有的 `DriverShaderCacheManager` 图形接口错误信息及
`Aftermath Error 0xbad00009` 提示。已与先前 `outputs/perception_acceptance/validate_dynamic.log`
核对，为此前存在的相同启动诊断；本次两个环境均成功构建、执行并完成验证。

### 无仿真导出 CLI

```powershell
conda run -n env_isaaclab python isal/scripts/export_actor.py `
  --checkpoint outputs/stage4a_acceptance/baseline_default/untrained_checkpoint.pt `
  --output outputs/stage4a_acceptance/cli_export
```

执行成功，输出 `actor.pt`、`actor.onnx`、`metadata.json`。
日志：`outputs/stage4a_export_cli.log`。导出使用 CPU，不创建环境、不调用 optimizer。
另外加载 CLI 与真实 runner 测试导出的两个 TorchScript 文件，逐项比较全部参数和 buffers，
结果 `CLI_EXPORT_STATE_MATCH=PASS`。

## 4. 产物与限制

`outputs/stage4a_acceptance/` 下每个场景包含：

- `<scenario>.log`：独立仿真 pytest 输出。
- `<scenario>/summary.json`：布局、参数量、控制步数、无训练标志和验收结果。
- `<scenario>/untrained_checkpoint.pt`：未训练 runner checkpoint。
- `<scenario>/export/{actor.pt,actor.onnx,metadata.json}`：对应 Actor-only 导出。
- `cli_export/`：独立 CLI 重建模型后导出的 Baseline Actor。

首轮受限沙箱执行得到 27 passed、5 errors，错误均在 pytest 临时目录 fixture 创建阶段。
指定工作区 basetemp 后仍遭遇 Windows 临时目录权限问题。获准常规缓存访问后，纯模型/导出测试
得到 32 passed；随后补充两项 CPU/CUDA tracker→head 测试，最终完整回归得到上述 121 passed。
没有为权限问题修改源码、删除缓存目录或绕开失败用例。

最终 `git diff --check` 无 whitespace error；`git -C robolab status --short` 和
`git -C rsl_rl status --short` 均为空输出。未改动上游 submodule。

当前限制：

- head 尚未训练；所有 checkpoint 都不能作为稳定行走策略使用。
- CNN 与旧 MLP 容量/结构不同；这里只保证新 E1/E2 之间的公共结构与初始化一致。
- 不验证策略学习收益、训练后 Adam 状态、多 GPU 训练或实机部署性能；推理延迟留待实际 checkpoint 评估。
- 没有阶段 4B 的预测输入、gate 或 query lattice，没有阶段 5 的采样 buffer/loss/schedule。
- Aux-only 的完整 checkpoint 需要同 variant 恢复；部署删 head 通过专用导出实现，不通过忽略缺失键加载。

下一阶段必须等待本阶段用户验收通过。
