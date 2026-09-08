# 阶段 5 验收包：Current-Rollout Buffer、PPO 辅助学习与双调度

日期：2026-09-08。状态：阶段 5 工程实现、回归及公共入口验证完成，待用户验收。
支持单设备 CPU/CUDA；阶段 6 未开始。

## 1. 交付内容

| 任务 | Tracker/head | 辅助学习 | Actor 预测输入 |
|---|---|---|---|
| `ISAL-Humanoid-Rough-CNN-Train-v0` | 关闭 | 关闭 | 无 |
| `ISAL-Humanoid-Rough-CNN-Aux-Train-v0` | 开启 | 开启 | 无 |
| `ISAL-Humanoid-Rough-AffordanceObs-Train-v0` | 开启 | 开启 | 完整网格，gate 调度 |
| `ISAL-Humanoid-Rough-AffordanceZero-Train-v0` | 开启 | 开启 | 永久零输入 |

四个新增任务共用项目本地 PPO/runner，旧任务继续使用原配置。
没有修改 reward、terrain、curriculum、动作控制、标签公式和阶段 4A/4B 模型。
未实现跨 iteration replay、新的 planner、target network、辅助样本增强或分布式辅助更新。

### 新增文件（相对仓库根目录）

| 文件 | 用途 |
|---|---|
| `isal/isal/learning/training_config.py` | 两套 schedule、buffer/loss 和 probe 默认参数 |
| `isal/isal/learning/auxiliary_buffer.py` | 当前 rollout buffer、独立采样 RNG、均匀容量限制 |
| `isal/isal/learning/ppo_affordance.py` | PPO loss helper、联合 update、边界管理与诊断接入 |
| `isal/isal/learning/affordance_runner.py` | 本地 runner、计数、日志、严格训练 checkpoint |
| `isal/isal/learning/training_diagnostics.py` | 标签分布、相关性、梯度以外的只读 probe 统计 |
| `isal/isal/learning/LICENSE_RSL_RL` | 改编上游 PPO/runner 的 BSD-3-Clause 许可 |
| `isal/isal/tasks/direct/humanoid_rough/agents/training_agent_cfg.py` | 四个独立训练配置 |
| `isal/tests/stage5/__init__.py` | 测试包 |
| `isal/tests/stage5/conftest.py` | CPU/CUDA fixture、禁止训练/update/optimizer 的断言 |
| `isal/tests/stage5/helpers.py` | 合成观测、packet、minibatch、checkpoint fixture |
| `isal/tests/stage5/test_learning.py` | buffer、损失、梯度、schedule、恢复和指标测试 |
| `isal/tests/stage5/test_timing.py` | 跨 rollout 结算和标签不变测试 |
| `isal/tests/stage5/test_runtime.py` | 配置公平性、四个独立进程真实场景 |
| `isal/STAGE5_ACCEPTANCE.md` | 本验收记录 |

### 修改文件

- `isal/isal/interaction/config.py`：新增默认关闭的 `emit_sample_timing`。
- `isal/isal/interaction/tracker.py`：可选 liftoff/finalization 步数诊断及 reset 清理。
- `isal/isal/tasks/direct/humanoid_rough/isal_env_cfg.py`：阶段 5 辅助任务启用 timing。
- `isal/isal/tasks/direct/humanoid_rough/__init__.py`：四个新 task 注册。
- `isal/scripts/rsl_rl/train.py`：为新配置分派 runner；validate-only 检查 packet→buffer。
- `isal/README.md`：任务、参数位置、命令、日志及恢复语义。

## 2. 数据链路与实现决策

### 样本生命周期

`process_env_step()` 保留原 PPO transition/normalization 行为，并将本次物理步的新结算
packet 送入 buffer。`valid[N,2,P]` 同时保留左右脚及所有有效 pending slot。
样本以独立 tensor 存储；显式退出 inference mode 后 clone，保证可用于之后的辅助 backward。

只存当前 rollout 新结算的样本。Tracker pending 可以跨 rollout，update 只清空学习 buffer。
默认容量 16,384，通过独立随机优先级 top-k 保留均匀子集，记录收到/保留/丢弃数。
辅助 batch=2,048，有放回均匀采样，至少 256 个样本才启用 loss。
随机优先级与 minibatch 使用独立 generator，不推进 PPO 的全局 RNG。

仅新辅助任务输出 `snapshot_step` 和 `finalized_step`。样本 age 是两者之差；
结合当前 rollout 内的步号判断 liftoff 是否早于 rollout 起点。
没有 age 过滤，没有改动 outcome/survival 窗口、label 权重或 contact event 状态转换。

### 联合 PPO

FQN：`isal.learning.ppo_affordance:PPOWithAffordance`。
依据当前 RSL-RL commit `6986d4d1b9fbab96fb61d51f07de419bc95432f1` 的 PPO 实现改编，
在项目内提取 `compute_minibatch_loss()`；上游源码不变。

生产 update 循环调用该 helper，合并辅助 loss 后执行一次 backward、一次全局梯度裁剪和
一次现有 Adam step。无第二个 optimizer，不启用上游 `get_aux_loss()` 通道。
PPO surrogate/value clipping、entropy、symmetry 和 adaptive learning rate 沿用原公式与阈值。

辅助 loss 为 `SmoothL1(beta=0.1)`，仅更新 Actor terrain encoder/head。
Actor PPO/symmetry 通过直接 latent 路径更新 encoder/body；4B projection 可获得 PPO 梯度，
head 不获得 PPO 梯度。Critic 保持独立。每个 minibatch 使用当前参数重新计算完整策略。

关闭辅助学习时不创建 buffer、不采样、不建立辅助反向图。
仅 coefficient=0 时仍可采样和诊断，但跳过辅助反向图；样本不足也跳过，并记录原因。

### 调度、恢复和日志

FQN：`isal.learning.affordance_runner:AffordanceRunner`。
`k` 明确定义为已经完成的 update 次数：

```text
lambda(k) = 0.05 * clamp((k - 100) / 200, 0, 1)
gate(k)   = 1.00 * clamp((k - 300) / 100, 0, 1)
```

两个 schedule 独立配置。ramp=0 时在 start 直接切换。Rollout 开始设置一次，整轮采样和
update epochs 不再变化；下一轮采样前推进。零输入任务始终 gate=0。

Checkpoint 只在完整 update 边界保存；顶层新增 `stage5` schema 1，模型 metadata 仍保持
4A/4B，因此现有导出 CLI 可继续工作。保存 completed updates、实际 environment steps、
累计计时、当前 coefficient/gate、schedule/算法/环境定义、模型/normalizer/optimizer、
adaptive learning rate、Python/NumPy/Torch/CUDA RNG 和独立辅助 generator。

恢复要求同配置，不做旧阶段训练状态迁移。恢复后 reset 环境并清除 tracker pending 和
aux buffer，再恢复训练 RNG；保留 checkpoint 中的 gate，直到下一次 begin_rollout。
不保存物理仿真状态，因此不承诺连续轨迹或逐步完全复现。

每次 update 写入控制台、`stage5_metrics.jsonl` 和 scalar writer。JSONL 同时包含
completed updates/environment steps；Stage5 scalar tags 以 environment steps 为横轴。
所有保留 target 的分布和标签分量单独统计；预测与相关性使用至多 256 个固定 probe。
常量 target 的相关性为 `null`，`correlation_valid=false`。
同一次 update 前后使用相同 probe，记录预测/动作漂移和 KL，不修改 normalizer/gate/distribution。
辅助梯度范数在第一个 active minibatch 上测量、未乘 coefficient；联合梯度范数在裁剪前测量。

## 3. 本机验证

所有 Python 命令使用 `env_isaaclab`。未执行 `runner.learn()`、算法 `update()` 或 optimizer
`step()`；测试 fixture 对这些入口设置了禁止调用断言。仿真执行动作始终为固定零动作。
生产 loss helper、辅助预测、backward、采样和 checkpoint 在本机实际执行。

阶段 5 定向命令：

```powershell
conda run --no-capture-output -n env_isaaclab python -u -m pytest `
  isal/tests/stage5 -q --tb=short -p no:cacheprovider `
  > outputs/stage5_tests.log 2>&1
```

实际结果：**29 passed in 100.00s**。

完整回归命令：

```powershell
conda run --no-capture-output -n env_isaaclab python -u -m pytest `
  isal/tests -q --tb=short -p no:cacheprovider `
  > outputs/stage5_full_acceptance_tests.log 2>&1
```

实际结果：**187 passed, 7 warnings in 355.69s**，无失败或 skip，覆盖阶段 1～4B 的
158 项和当时阶段 5 的 29 项。7 条警告为旧 normalizer 字段弃用（1 条）与 ONNX legacy
logging 弃用（6 条），并非训练失败。

收尾审查后新增了 CPU/CUDA 两项生产 `process_env_step()` 链路测试，并重跑全部 loss tests：

```powershell
conda run --no-capture-output -n env_isaaclab python -u -m pytest `
  isal/tests/stage5/test_learning.py -q --tb=short -p no:cacheprovider `
  > outputs/stage5_loss_chain_tests.log 2>&1
```

实际结果：**24 passed in 1.69s**。其中 22 项与完整回归重叠，2 项为新增检查；
最终 **189 个不同用例均已通过**，不是单次运行产生的 189 项统计。新增测试直接经过
`act → process_env_step → PPO storage/aux buffer → compute_returns → minibatch loss → backward`，
全程只有合成观测，无 optimizer update。最终新增用例后没有改动生产实现。

### 公共 CLI

所有命令都只执行 1 个环境、1 个固定零动作 step，不调用训练和算法更新。

```powershell
conda run --no-capture-output -n env_isaaclab python -u `
  isal/scripts/rsl_rl/train.py --task ISAL-Humanoid-Rough-CNN-Train-v0 `
  --headless --num_envs 1 --validate-only `
  > outputs/stage5_validate_baseline.log 2>&1

conda run --no-capture-output -n env_isaaclab python -u `
  isal/scripts/rsl_rl/train.py --task ISAL-Humanoid-Rough-CNN-Aux-Train-v0 `
  --headless --num_envs 1 --validate-only `
  > outputs/stage5_validate_aux.log 2>&1

conda run --no-capture-output -n env_isaaclab python -u `
  isal/scripts/rsl_rl/train.py --task ISAL-Humanoid-Rough-AffordanceObs-Train-v0 `
  --headless --num_envs 1 --validate-only `
  'env.terrain_perception.size=[1.8,1.0]' `
  env.scene.height_scanner.pattern_cfg.ordering=yx `
  env.robot.actor_obs_history_length=3 env.robot.critic_obs_history_length=4 `
  agent.algorithm.auxiliary_learning.loss_coef=0.01 `
  agent.algorithm.auxiliary_learning.start_iteration=0 `
  agent.algorithm.auxiliary_learning.ramp_iterations=0 `
  agent.algorithm.affordance_gate_schedule.start_iteration=0 `
  agent.algorithm.affordance_gate_schedule.ramp_iterations=0 `
  > outputs/stage5_validate_predicted_dynamic.log 2>&1

conda run --no-capture-output -n env_isaaclab python -u `
  isal/scripts/rsl_rl/train.py --task ISAL-Humanoid-Rough-AffordanceZero-Train-v0 `
  --headless --num_envs 1 --validate-only `
  > outputs/stage5_validate_zero.log 2>&1
```

四个命令均已通过。Baseline、Aux-only 和 Zero 默认 coefficient=0；Zero gate=0。
默认 shape 为 policy `(1,780)`、scan `(1,187)`、critic `(1,3260)`。
动态命令实测为 policy `(1,234)`、scan `(1,209)`、critic `(1,1392)`，
预测网格 `(1,2,19,11)`、coefficient=0.01、gate=1.0，environment_steps=1。
日志同时确认 reward finite、动态镜像成功、执行零动作、`runner.learn_called=False`。

### 独立导出和工作区检查

```powershell
conda run --no-capture-output -n env_isaaclab python -u isal/scripts/export_actor.py `
  --checkpoint outputs/stage5_acceptance/AffordanceObs/untrained_checkpoint.pt `
  --output outputs/stage5_acceptance/cli_export `
  > outputs/stage5_export_cli.log 2>&1

git diff --check
git -C robolab status --short
git -C rsl_rl status --short
```

独立导出 CLI 成功，输出 `cli_export/actor.pt`、`actor.onnx`、`metadata.json`。
Git whitespace 检查通过；两个上游工作区均无修改。
Isaac 仍有已知图形栈/MaterialX/URDF 和退出时 USD detach/unload 提示，场景检查均完成。
Conda 的 `miniconda3/bin` 路径提示未阻止执行。

### 已验证的主要场景

- CPU/CUDA：buffer ownership、inference tensor 转换、双脚多 slot、容量限制、RNG 隔离及恢复。
- PPO clipped/unclipped value 与 surrogate 参考公式；辅助启用/禁用/warmup/样本不足。
- 4A Aux-only、4B predicted/zero 的实际 loss helper 和梯度路由。
- schedule 边界、零长度 ramp、跨 rollout 固定 gate、update 完成后的 buffer 清空。
- 常量 target、空样本和有 head 扰动时的 drift/KL；诊断不改变分布及状态。
- 真实 tracker 第 2 步 liftoff、第 28 步结算，跨 24 步 rollout 边界，四个样本只结算一次；
  timing 开/关下所有原 packet 字段逐 tensor 一致。
- 同配置 checkpoint 恢复、不同配置拒绝；随机数恢复、物理 pending 清理和不保存旧样本。

### 四个真实场景

| 变体 | 环境数 | 固定零动作步 | 自然样本 | 合成补充样本 |
|---|---:|---:|---:|---:|
| CNN | 1 | 40 | 0 | 0 |
| CNN-Aux | 1 | 40 | 0 | 1 |
| AffordanceObs | 2 | 40 | 0 | 1 |
| AffordanceZero | 1 | 40 | 0 | 1 |

各场景独立进程，完成 runner 构建、reset、网络前向、实际 symmetry loss backward、checkpoint、
TorchScript/ONNX 对齐和参数未更新检查。预测任务额外覆盖 19×11、yx、历史 3/4、非单位角速度 scale。
三个辅助任务用显式标识的合成 packet 补足接口检查，不把它们算作自然接触样本或额外环境步。

## 4. 未训练产物

目录：`outputs/stage5_acceptance/{CNN,CNN-Aux,AffordanceObs,AffordanceZero}/`。
每个包含 `untrained_checkpoint.pt`、`export/actor.pt`、`export/actor.onnx`、
`export/metadata.json` 和 `summary.json`；父目录还有独立进程日志。

Checkpoint 中 completed_updates=351、environment_steps=123 是**人为设置的恢复测试字段**，
不是训练结果。随后真实场景累计 40 或 80 个环境步，因此 summary 总数为 163 或 203。
预测版 checkpoint gate=0.5；恢复后下一次 begin_rollout 为约 0.51；zero 始终为 0。
这些是未训练工程产物，不是行走策略。Optimizer 没有执行过 step，没有学习产生的 Adam moments。

## 5. 验证边界与后续工作

没有未通过的本阶段工程检查。所有源码与说明保留在当前工作区，未自动提交或启动下一阶段。

生产联合优化循环已实现，但本机没有运行它；实际多轮 optimizer 行为、行走学习、PPO 稳定性、
辅助样本数量和预测质量必须在外部短训练验收。纯 loss/backward 检查不能替代这项证据。

当前帧推理与 liftoff-only 监督、noisy Actor context 与 clean tracker context 的差异仍然存在。
辅助梯度和 head 数值变化不受 PPO clipping 的硬限制；本阶段增加诊断，没有宣称解决该稳定性问题。
本机零动作未产生自然有效样本；真实样本量和 target 分布也需要外部训练验证。

本阶段完成后停止等待验收，不开始阶段 6 外部训练或阶段 7 地形/curriculum 实现。
