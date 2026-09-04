# ISAL-Humanoid 阶段 0：仓库审计与执行契约

状态：待用户验收  
审计日期：2026-09-03  
规格来源：`humanoid_self_supervised_affordance_codex_spec.md`

## 1. 阶段 0 结论

ISAL-Humanoid 的 MVP 应基于现有 `RPO-Rough` 任务扩展，不从零实现基础人形行走，也不以 `RPO-Parkour` 为基线。

选择 `RPO-Rough` 的原因：

- 它是现有的 23-DoF 人形速度跟踪 PPO 任务；
- 已有 joint-position action、粗糙地形、terrain curriculum、接触传感器和 RayCaster；
- Actor 本体观测与规格中的 `9 + 3N` 完全吻合；
- 它使用普通 PPO，研究变量比 AMP/Parkour 更干净；
- `RPO-Parkour` 同时引入 AMP、depth encoder 和 MoE，不适合作为本项目的最小可消融基线。

实现将放在 `robolab` 自己的任务和 learning 包中。现有 RSL-RL 3.3.0 支持使用完全限定名称加载自定义 policy 和 algorithm，因此目前没有修改上游 Isaac Lab 或 RSL-RL 源码的必要。

## 2. 不可变执行约束

### 2.1 Conda 环境

所有 Python 命令必须显式使用：

```powershell
conda run -n env_isaaclab python ...
```

需要实时显示 Isaac Sim 输出时使用：

```powershell
conda run --no-capture-output -n env_isaaclab python -u ...
```

不允许依赖当前 shell 已激活的环境，也不允许直接调用裸 `python`、`pip` 或 `pytest`。

### 2.2 本机禁止训练

本机禁止：

- 调用 `robolab/scripts/rsl_rl/train.py`；
- 调用 `OnPolicyRunner.learn()`；
- 运行任何 iteration 形式的策略学习，包括所谓的短训练；
- 生成用于结论的 learning curve。

本机允许：

- import、配置解析和 Gym 注册测试；
- 1～4 个环境的 reset 和有限步固定动作 smoke test；
- 纯 PyTorch forward/backward、梯度路由和单次 optimizer 单元测试；
- 使用已有 checkpoint 的 play/evaluation；
- checkpoint、ONNX/export 和可视化一致性测试。

外部训练机负责 200～500 iteration 短验证和正式多 seed 训练。本仓库只准备可复现命令、配置、评估脚本，并分析用户带回的日志与 checkpoint。

### 2.3 阶段门禁

每一阶段结束时提交：

1. 文件变更清单；
2. 设计决策；
3. 使用 `env_isaaclab` 执行的测试命令与结果；
4. 已知风险和未完成项；
5. 下一阶段拟进行的工作。

用户明确回复验收通过前，不进入下一阶段。

## 3. 仓库与依赖快照

| 项目 | 审计值 |
|---|---|
| 根仓库分支 | `main` |
| 根仓库 commit | `92008a6317d6d0efe8b58abc2fb3c630c75992c0` |
| `robolab` submodule commit | `6b1c3d9988497c8961dcba77892de32edc1770e1` |
| `rsl_rl` submodule commit | `6986d4d1b9fbab96fb61d51f07de419bc95432f1` |
| Python | 3.11.14 |
| Isaac Sim | 5.1.0.0 |
| Isaac Lab | 0.54.0 |
| RSL-RL | 3.3.0 |
| RoboLab | 1.0.0 |
| GPU | NVIDIA GeForce RTX 3060, 12 GB |
| OS | Windows 11 Pro |

`robolab` 和 `rsl_rl` 均以 editable install 指向当前工作区：

- `C:\Users\Admin\Documents\ISAL\robolab`
- `C:\Users\Admin\Documents\ISAL\rsl_rl`

审计开始时根工作区仅有一个既有未跟踪文件：

```text
?? humanoid_self_supervised_affordance_codex_spec.md
```

该文件属于用户输入，不修改、不移动。

## 4. 现有任务和训练入口

### 4.1 Gym 注册

`RPO-Rough` 注册为：

```text
task id:        RPO-Rough
entry point:    robolab.tasks.direct.base.base_env:BaseEnv
env config:     robolab.tasks.direct.base.rpo_env_cfg:RPORoughEnvCfg
agent config:   robolab.tasks.direct.base.agents.rpo_agent_cfg:RPORoughAgentCfg
```

`robolab.tasks` 使用 Isaac Lab 的 `import_packages()` 递归导入任务包；阶段 1 新建的任务只需提供 package `__init__.py` 注册即可被现有入口发现。

### 4.2 训练入口

现有入口：

```text
robolab/scripts/rsl_rl/train.py
```

流程为：

```text
Gym/Hydra config
  -> gym.make()
  -> RslRlVecEnvWrapper
  -> OnPolicyRunner
  -> policy + algorithm resolved from config
```

训练脚本已经支持 Hydra override、RSL-RL 配置和 DirectRLEnv。阶段 1～5 优先不修改它。

## 5. `RPO-Rough` 基线契约

### 5.1 环境类型与时间尺度

| 项目 | 当前值 |
|---|---:|
| 环境类型 | `DirectRLEnv` |
| physics dt | 0.005 s / 200 Hz |
| decimation | 4 |
| control dt | 0.02 s / 50 Hz |
| episode length | 20 s |
| 默认并行环境数 | 4096 |
| 动作维度 | 23 |
| action 类型 | default pose 上的 joint-position target |
| action scale | 0.25 |

规格允许沿用现有任务类型，因此不把稳定的 DirectRLEnv 强制改写为 ManagerBasedRLEnv。

### 5.2 Actor observation

每个当前帧为 78 维：

```text
base angular velocity       3
projected gravity           3
velocity command            3
joint position offset      23
joint velocity             23
previous action            23
                         -----
                            78
```

这正好等于规格中的 `D_p = 9 + 3N`，其中 `N=23`。

当前 actor history length 为 10，因此运行时 policy observation shape 为：

```text
[num_envs, 780]
```

### 5.3 Critic observation

不含 height scan 的 critic 当前帧为 139 维，包括 actor 当前帧、本体线速度、双脚接触、接触力、air time、脚高、关节加速度和关节力矩。

现有 rough task 另外加入 187 个 height points：

```text
139 + 187 = 326
```

critic history length 为 10，因此运行时 critic observation shape 为：

```text
[num_envs, 3260]
```

现有 height scan 为 11×17 个点。阶段 2 仍须从实际 sensor pattern 动态推导 H/W 并 assert，不能依赖这里的静态推算。

### 5.4 当前 height scan 行为

现有配置：

- RayCaster 绑定 `base_link`；
- yaw alignment；
- size 为 `(1.6, 1.0)` m；
- resolution 为 0.1 m；
- 只扫描 `/World/ground` 静态 mesh；
- 默认只进入 critic，`enable_height_scan_actor=False`；
- 当前预处理为 sensor-z 减 hit-z、再减 0.5、clip 到 `[-1,1]`；
- 当前噪声配置不符合 ISAL MVP 的零噪声要求。

阶段 2 必须为新任务实现规格定义的 root-relative、clip/scale 预处理，并将二维 scan 作为 Actor 的独立 observation group。原任务行为保持不变。

### 5.5 接触信息

现有环境提供：

- 覆盖所有机器人 link 的 `ContactSensorCfg`；
- contact history length 3；
- air-time tracking；
- 左右 ankle-roll link 的世界位置和速度；
- 左右脚局部 RayCaster。

这些数据足够实现 liftoff、touchdown、slip、tilt、contact persistence 和 survival 标签，无需 terrain metadata。

现有 locomotion 的 contact 判定使用历史接触力范数大于 1 N；这不能直接复用于辅助标签。ISAL tracker 将使用独立配置的法向力阈值，默认 20 N。

### 5.6 Reward、termination 与 curriculum

当前 rough task 有 29 个 reward terms，包括：

- linear/yaw velocity tracking；
- orientation、vertical velocity、angular velocity；
- torque、energy、joint velocity/acceleration；
- action rate/smoothness；
- undesired contacts 和 termination；
- feet air time、slide、force、stumble、orientation 和 height。

`feet_slide` 已经存在于原 reward，因此 Baseline 与 Proposed 都会保留相同 slip reward，避免 Proposed 获得额外 reward 信息。

当前 terrain 包含 flat、random rough、slopes、stairs 和 random grid，并已有 terrain-level curriculum。stepping stones 只存在于 hard terrain 配置中。

阶段 1 保留基线 reward、termination、commands、randomization 和 terrain 行为用于复制验证。研究规格要求的 command range、MVP terrain mixture 和完整 curriculum 只在后续阶段对 Baseline/Proposed 同步启用。

## 6. RSL-RL 对接审计

### 6.1 自定义类加载

RSL-RL 的 `resolve_callable()` 支持：

```text
module.path:ClassName
```

`OnPolicyRunner` 分别对 `policy.class_name` 和 `algorithm.class_name` 调用该解析器。因此计划使用：

```text
robolab...learning.models:AffordanceActorCritic
robolab...learning.algorithms:PPOWithAffordance
```

无需将新类加入 `rsl_rl.modules.__init__` 或 `rsl_rl.algorithms.__init__`。

### 6.2 Env extras 数据流

`RslRlVecEnvWrapper.step()` 会保留环境返回的 `extras`，只额外写入 `time_outs`。随后 `OnPolicyRunner.learn()` 把同一 `extras` 传给：

```text
PPO.process_env_step()
Logger.process_env_step()
```

因此阶段 3 可以使用 `extras["auxiliary"]` 传递 GPU tensor，不需要全局变量、CPU list 或 wrapper 补丁。

### 6.3 当前 PPO auxiliary hook

本地 PPO 已有通用字段：

```text
enable_aux_loss
aux_loss_coef
policy.get_aux_loss()
```

但它不能满足本规格，因为它没有：

- 从 `extras` 收集 interaction samples；
- current-rollout-only buffer；
- auxiliary batch sampling；
- warmup/ramp schedule；
- 所需 contact/target/prediction 日志。

阶段 5 将在 `robolab` 中继承 PPO，实现上述功能，不直接扩写这个通用 hook。

### 6.4 Checkpoint 与 schedule 风险

现有 runner checkpoint 包含：

- policy state dict；
- PPO optimizer state dict；
- runner iteration；
- RND 状态（若启用）。

Actor/Critic normalizer 作为 policy 子模块时会包含在 policy state dict 中，auxiliary rollout buffer 不需要保存。

风险：runner 不会把当前 iteration 传给 algorithm，也不会保存自定义 algorithm state。阶段 5 必须采用可恢复的 auxiliary schedule 状态，或实现最小项目本地 runner 扩展；不能在 resume 后重新开始 warmup。

### 6.5 Logging 风险

现有 Logger 会把 algorithm 返回的 loss key 统一写成 `Loss/<key>`，而规格要求稳定的 `aux/*` 等标签。阶段 5 先验证现有接口能否满足分析脚本；若不能，使用项目本地 runner/logger 扩展。只有证明无法规避时，才考虑集中且最小的 RSL-RL 补丁。

## 7. Symmetry 与公平性决策

`RPORoughAgentCfg` 当前启用左右镜像 data augmentation 和 mirror loss。现有镜像索引假定：

- policy 只有 78 维本体帧历史；
- height scan 只位于 critic；
- history length 固定为 10。

Actor 加入二维 height scan 后，现有索引不能直接复用。

阶段 4 的默认决策是保留基线已有 symmetry，并为分组 observation 实现正确的左右镜像：

- proprio 对应关节和符号映射；
- height scan 沿左右轴翻转；
- 若将来增强 auxiliary sample，则 `query_y` 取负且 foot-side 对换。

Baseline 与 Proposed 使用相同的 symmetry 配置。若实际接口使正确映射代价过高，允许把 symmetry 在两个变体中同时关闭，但该变更必须在阶段 4 单独报告并获得用户验收。

## 8. 计划中的代码归属

阶段 1 起建议新增：

```text
robolab/robolab/tasks/direct/isal_humanoid/
├── __init__.py
├── isal_env.py
├── isal_env_cfg.py
├── terrain_cfg.py
├── self_supervised_cfg.py
├── agents/
└── mdp/

robolab/robolab/learning/isal/
├── models/
├── algorithms/
└── storage/

robolab/tests/isal/
```

目录名称可以在阶段 1 根据现有 package 风格微调，但职责必须分离。禁止把 optimizer 写进 interaction tracker，也禁止把 label 阈值散落在模型代码中。

## 9. 本机 smoke test 记录

执行类型：单环境、固定零动作一步；未创建 runner，未调用训练。

关键结果：

```text
registered=True
entry_point=robolab.tasks.direct.base.base_env:BaseEnv
made_env=True
obs_keys=['critic', 'policy']
policy_shape=(1, 780)
critic_shape=(1, 3260)
num_actions=23
step_tuple_len=5
post_policy_shape=(1, 780)
reward_finite=True
terminated_shape=(1,)
truncated_shape=(1,)
extras_keys=['log', 'time_outs']
```

RSL-RL callable resolver 的现有测试也在指定环境中通过：

```powershell
conda run --no-capture-output -n env_isaaclab python -m pytest `
  rsl_rl\tests\utils\test_resolve_callable.py -q -p no:cacheprovider
```

本次结果为 `18 passed`。实际审计首次执行时未关闭 pytest cache provider，因 submodule 目录的缓存写权限产生一条 `PytestCacheWarning`，不影响测试结果；后续测试固定使用 `-p no:cacheprovider`。

测试发现的宿主机问题：

1. Isaac Lab 默认地形缓存 `/tmp/isaaclab/terrains` 在受限沙箱中不可写，仿真 smoke test 需要允许 Isaac Lab 写自己的缓存目录；
2. 使用普通 `conda run` 时 Isaac Sim/pytest 输出缓冲明显，而且 Windows GBK 可能使 conda 在转发输出时抛出编码错误并错误返回 exit code 0；验收测试必须使用 `--no-capture-output`，仿真脚本同时使用 `python -u`；
3. Isaac Sim 报告 D3D12/Vulkan shader cache 与 Aftermath 警告，但本次环境创建、reset 和 step 成功；
4. 直接在 AppLauncher 之前 import `isaaclab_rl` 会因 `pxr` 未初始化失败，所有 Isaac Lab 集成测试必须先启动 AppLauncher；
5. `conda run` 输出 `Did not find path entry ...\miniconda3\bin` 警告，但实际解释器和全部包版本均来自 `env_isaaclab`，不影响本次测试。

## 10. 阶段 1 进入条件

阶段 0 验收后，阶段 1 仅进行隔离 Baseline 复制与注册，不加入 Actor height scan、不加入 tracker、不加入新网络或 auxiliary loss。

阶段 1 必须做到：

- 原 `RPO-Rough` 源文件和注册保持不变；
- 新 task 能注册、创建、reset、固定动作 step；
- action、reward、termination、commands、randomization 和 PPO 参数与基线差异可审计；
- 不在本机调用训练入口；
- 没有用户验收不得进入阶段 2。

## 11. 阶段 0 验收清单

- [x] 完整读取项目规格。
- [x] 确认 Isaac Lab、Isaac Sim、RSL-RL、RoboLab 和 Python 版本。
- [x] 记录根仓库及两个 submodule commit。
- [x] 选择并说明稳定基线 `RPO-Rough`。
- [x] 确认训练入口和 Gym 注册机制。
- [x] 确认 custom model/algorithm 注册机制。
- [x] 确认 `extras` 到 PPO 的传输路径。
- [x] 记录 observation/action/reward/terrain/sensor 现状。
- [x] 执行无训练的单环境固定动作 smoke test。
- [x] 固化 `env_isaaclab` 和本机禁止训练约束。
- [x] 列出后续已知接口风险与解决顺序。
