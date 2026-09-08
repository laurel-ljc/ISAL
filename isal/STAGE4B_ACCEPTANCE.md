# 阶段 4B 验收包：完整 Affordance 网格输入 Actor

日期：2026-09-07。状态：阶段 4B 实现、回归与公共入口验证完成，待用户验收。
普通 PPO，有效辅助权重为 0；阶段 5 未开始。

## 1. 交付范围与文件

新增两个独立任务，均继承现有 CNN-Aux 环境、tracker/head、普通 PPO 和镜像配置。

| 任务 | 控制输入 | 默认 gate | 辅助更新 |
|---|---|---:|---|
| `ISAL-Humanoid-Rough-AffordanceObs-v0` | 当前完整网格预测 | 0 | 关闭 |
| `ISAL-Humanoid-Rough-AffordanceZero-v0` | 永久零向量 | 0 | 关闭 |

没有新增 observation group，没有修改标签、reward、terrain、curriculum、随机化或动作控制。
没有辅助 buffer、联合 loss、schedule 或 Stage 5 实现。本机没有执行策略训练。

新增文件（路径相对仓库根目录）：

| 文件 | 作用 |
|---|---|
| `isal/isal/learning/models/affordance_observation_actor_critic.py` | 新模型、gate、严格 checkpoint 校验、完整预测 Actor 导出 |
| `isal/isal/learning/models/dense_query.py` | 实际网格坐标、物理 context、分块 dense head |
| `isal/scripts/benchmark_affordance.py` | gate=1、batch=1 的完整推理开销测量 |
| `isal/tests/stage4b/__init__.py` | 隔离测试 helper 导入，避免与 4A conftest 冲突 |
| `isal/tests/stage4b/conftest.py` | CPU/CUDA fixture、线程数设置 |
| `isal/tests/stage4b/helpers.py` | 合成布局、观测及有效辅助样本 |
| `isal/tests/stage4b/test_model.py` | 网格、context、RNG、gate、梯度、checkpoint |
| `isal/tests/stage4b/test_export.py` | 实际 checkpoint 导出器、TorchScript/ONNX、动态 batch |
| `isal/tests/stage4b/test_runtime.py` | 配置公平性、真实镜像函数、独立进程场景和 runner |
| `isal/STAGE4B_ACCEPTANCE.md` | 本验收记录 |

修改文件：

- `isal/isal/learning/config.py`：预测输入配置；仅对 4B 追加实际 query/scales 绑定。
- `isal/isal/learning/models/__init__.py`：导出新模型，保持无 Isaac Sim 导入依赖。
- `isal/isal/tasks/direct/humanoid_rough/__init__.py`：注册两个新任务。
- `isal/isal/tasks/direct/humanoid_rough/agents/affordance_agent_cfg.py`：两个独立 agent/policy 配置。
- `isal/scripts/export_actor.py`：根据 metadata 分派 4A/4B；提供纯模型加载函数。
- `isal/scripts/rsl_rl/train.py`：仅在 validate-only 中追加预测网格检查与输出。
- `isal/README.md`：使用说明、参数位置、验证/导出/基准命令和阶段边界。

## 2. 结构与实现决策

模型 FQN：
`isal.learning.models.affordance_observation_actor_critic:AffordanceObservationActorCritic`。
复用 4A Actor/Critic/normalizer/标量 Gaussian std 实现；未修改上游模块注册表。

实际 `RayCaster.ray_starts` 已包含 sensor offset，绑定时提取其中 x/y，检查实际 H/W 和
Cartesian 网格，再按 canonical x/y 排列，不额外添加前移量。两个脚共享该坐标表；
分数输出 `[B,2,H,W]`，先左脚后右脚。坐标单位为米，参考当前 root yaw frame。

当前 `policy` 历史最后一帧的前 9 维依次为 angular velocity、gravity、command。
控制 context 在统计归一化之前提取，除以环境绑定的相应 scales，按
`command, base_ang_vel, projected_gravity` 送入 head，保留现有噪声和 clipping。
不读取 Critic、tracker pending 或 future target；原 liftoff 辅助接口继续使用物理尺度样本。

Actor 先计算一次 terrain latent。dense head 复用该 latent，在 `no_grad` 下分块计算，
输出显式 detach。默认每块 64 个 query，不逐环境循环，不重复 CNN。
`gate * (2*scores-1)` 经无偏置投影后，加到原 Actor 第一层线性输出、ELU 之前。
零输入版跳过控制用 head，但保留相同网络参数及独立诊断接口。

投影在公共网络和 head 创建完成后隔离 RNG 初始化。相同 seed 的 4A 公共参数、
预测版和零输入版公共参数逐 tensor 一致，CPU/CUDA RNG 状态一致。
`gate=0` 和永久零输入模式均与对应 4A Actor 输出完全一致。

Gate 是 checkpoint buffer，默认 0，通过 `set_affordance_input_gate(value)` 显式修改。
诊断预测不会改变 gate、normalizer 或动作分布。预测模式即使 gate=0 也保留完整 head 路径。

### 默认网络参数量

| 项目 | 预测版 | 零输入版 |
|---|---:|---:|
| 完整 Actor/Critic/head/projection/std | 1,538,288 | 1,538,288 |
| 新增投影 `374→256,bias=False` | 95,744 | 95,744 |
| Head | 18,305 | 18,305 |
| 实际 Actor 部署导出 | 675,768 | 561,719 |

零输入导出省略永久无效的 head/projection。两个变体参数形状一致，但不据此声称排除了
有效网络容量差异。Critic 完整沿用 4A，不消费预测。

### Checkpoint 与导出

4B 使用 schema 2，记录 layout、scan preprocessing、网络、输入模式、query coordinates、
context scales、投影结构。拒绝 4A checkpoint、不同模式、不同输入定义；没有迁移逻辑。
固定 query/scales/foot-side buffer 与 metadata 交叉核对，gate 值作为可变状态恢复。

Runner save/load 实测恢复模型、normalizer、gate、optimizer 配置和 iteration。
验收 checkpoint 的 iteration=7 是人为写入的测试字段，**不是训练迭代结果**；
未执行 optimizer step，因此不包含经过学习的 Adam moments。

预测导出包含 Actor、head、context adapter、query buffers、projection、实际 gate；
gate=0 时也检查 head/projection 在 TorchScript 和 ONNX 中存在。
零输入导出等价于公共 Actor。4A 导出路径保留。
输入仍是 proprio history 与已完成物理 clip/scale 的 canonical scan，输出 action mean；
TorchScript 和 ONNX opset 17 固定特征尺寸、动态 batch。

## 3. 验证命令与结果

工作目录为 `C:\Users\Admin\Documents\ISAL`；全部 Python 命令通过 `env_isaaclab`。
Isaac/pytest 使用正常 Windows 临时目录和缓存权限。仿真测试禁止调用
`OnPolicyRunner.learn()`、`PPO.update()`、Adam/AdamW/SGD `step()`；只执行固定零动作。
CPU/CUDA 合成测试只做前向及 backward，没有优化更新。

### 完整回归

```powershell
conda run --no-capture-output -n env_isaaclab python -u -m pytest `
  isal/tests -q --tb=short -p no:cacheprovider `
  > outputs/stage4b_full_acceptance_tests.log 2>&1
```

实际结果：**158 passed, 7 warnings in 239.39s**。阶段 1～4A 的 121 项回归与
本阶段 37 项全部通过，无 skip 或失败。7 条 pytest 警告分别为旧 runner normalization
字段弃用（1 条）和 ONNX legacy logging 弃用（6 条）。

分组检查也已实际运行：

```powershell
conda run --no-capture-output -n env_isaaclab python -u -m pytest `
  isal/tests/stage4b/test_model.py isal/tests/stage4b/test_export.py `
  -q --tb=short -p no:cacheprovider

conda run --no-capture-output -n env_isaaclab python -u -m pytest `
  isal/tests/stage4b/test_runtime.py -q --tb=short -p no:cacheprovider `
  > outputs/stage4b_runtime_tests.log 2>&1
```

分别为 **31 passed, 3 warnings in 4.85s** 和 **6 passed in 68.47s**。
后续完整回归覆盖了导出加载函数的最终重构。

| 验收项 | 实际结果 |
|---|---|
| 17×11、17×9、19×11；xy/yx；历史 10/10 和 3/4 | CPU/CUDA shape、canonical query、左右脚顺序通过 |
| 最后历史帧、非单位 scales、噪声、非零 yaw | 物理 context 与 yaw-frame 坐标语义通过 |
| 每次 Actor 前向仅一次 terrain encoder | hook 计数通过；chunk=1、64、全量预测一致 |
| 相同 seed 的参数/RNG；gate=0/零输入 | 与对应 4A 公共 Actor 逐 tensor/输出完全一致 |
| gate=1 扰动 head；future/pending 污染 | head 扰动改变 action；无关未来字段不改变 action |
| Auxiliary backward | 只进入 Actor terrain encoder/head，无 projection/body/Critic 梯度 |
| Actor likelihood、entropy、symmetry、value | PPO 路径不进入 head；value 只进入 Critic；无 optimizer step |
| Frozen normalization/gate 的重复 log-prob | ratio≈1；手动修改 head 后重新计算分布 |
| 镜像 | 现有 command/gravity/angular velocity 符号、scan y 反射通过；解析 head 验证左右脚对应 |
| Checkpoint | 同配置恢复参数/normalizers/query/gate；拒绝 4A、模式及输入定义不匹配 |
| TorchScript/ONNX | batch=1、4 对齐，rtol=1e-4、atol=1e-5；预测版 gate=0 仍有 head |
| 真实任务与普通 PPO | 两任务和动态布局均通过；tracker packet 保留、enable_aux_loss=False |

### 独立进程仿真

| 场景 | 环境数 | 零动作控制步 | 网格/order | Actor/Critic 历史 | 恢复与导出 |
|---|---:|---:|---|---|---|
| predicted_default | 1 | 40 | 17×11 / xy | 10 / 10 | 通过 |
| zero_default | 1 | 40 | 17×11 / xy | 10 / 10 | 通过 |
| predicted_dynamic_yx | 2 | 40 | 19×11 / yx | 3 / 4 | 通过 |

各场景在独立进程创建/关闭环境。每步推理只用于数值检查，仿真执行的动作始终为零。
参数在场景结束时与初始值逐 tensor 一致。三个场景的自然有效样本数均为 0，
不将固定动作测试冒充足够的训练样本收集验证。

默认两个变体的随机 head 镜像 MAE 均为约 **0.003041**，动态场景约 **0.005518**。
只记录、不设置随机权重等变性阈值。

### 公共 validate-only 入口

以下三条命令均实际执行成功。每条仅 1 个环境、1 个固定零动作 step，日志均确认
`reward_finite=True`、`dynamic_augmentation=True`、`runner.learn_called=False`。

```powershell
conda run --no-capture-output -n env_isaaclab python -u `
  isal/scripts/rsl_rl/train.py --task ISAL-Humanoid-Rough-AffordanceObs-v0 `
  --headless --num_envs 1 --validate-only `
  agent.policy.affordance_observation.input_gate=1.0 `
  > outputs/stage4b_validate_predicted.log 2>&1

conda run --no-capture-output -n env_isaaclab python -u `
  isal/scripts/rsl_rl/train.py --task ISAL-Humanoid-Rough-AffordanceZero-v0 `
  --headless --num_envs 1 --validate-only `
  > outputs/stage4b_validate_zero.log 2>&1

conda run --no-capture-output -n env_isaaclab python -u `
  isal/scripts/rsl_rl/train.py --task ISAL-Humanoid-Rough-AffordanceObs-v0 `
  --headless --num_envs 1 --validate-only `
  'env.terrain_perception.size=[1.8,1.0]' `
  env.robot.actor_obs_history_length=3 env.robot.critic_obs_history_length=4 `
  env.scene.height_scanner.pattern_cfg.ordering=yx `
  env.normalization.obs_scales.ang_vel=2.0 `
  env.normalization.obs_scales.projected_gravity=3.0 `
  env.normalization.obs_scales.commands=4.0 `
  agent.policy.affordance_observation.input_gate=1.0 `
  > outputs/stage4b_validate_dynamic.log 2>&1
```

默认 policy/scan/critic shape 为 `(1,780)/(1,187)/(1,3260)`，grid 为 `(1,2,17,11)`；
动态覆盖为 `(1,234)/(1,209)/(1,1392)`，grid 为 `(1,2,19,11)`。
零输入命令保留默认 gate=0，其余两条显式 gate=1。

### 导出与上游检查

```powershell
conda run --no-capture-output -n env_isaaclab python -u isal/scripts/export_actor.py `
  --checkpoint outputs/stage4b_acceptance/predicted_default/untrained_checkpoint.pt `
  --output outputs/stage4b_acceptance/cli_export `
  > outputs/stage4b_export_cli.log 2>&1

conda run --no-capture-output -n env_isaaclab python -u isal/scripts/export_actor.py `
  --checkpoint outputs/stage4a_acceptance/baseline_default/untrained_checkpoint.pt `
  --output outputs/stage4b_acceptance/cli_export_4a_regression `
  > outputs/stage4b_export_4a_regression.log 2>&1

git diff --check
git -C robolab status --short
git -C rsl_rl status --short
```

两个独立导出 CLI 均成功；diff 无 whitespace 错误，两个上游工作区 status 均为空。
RoboLab HEAD 为 `6b1c3d9988497c8961dcba77892de32edc1770e1`，
RSL-RL HEAD 为 `6986d4d1b9fbab96fb61d51f07de419bc95432f1`。

Isaac 日志仍有图形栈消息，例如 `DriverShaderCacheManager::init()` graphics interface
错误、Aftermath/MaterialX/URDF 提示和退出时的 USD detach/unload 警告；本阶段未修改这些上游组件。
实际场景和 CLI 均完成且数值断言通过。Conda 的缺失 `miniconda3/bin` 路径提示未阻止执行。

## 4. 未训练产物与工程开销

产物根目录：`outputs/stage4b_acceptance/`。
三个场景为 `predicted_default`、`zero_default`、`predicted_dynamic_yx`；每个包含：

- `untrained_checkpoint.pt`：实际 runner 生成的未训练 checkpoint。
- `export/actor.pt`、`export/actor.onnx`、`export/metadata.json`。
- `summary.json`：布局、参数量、gate、镜像误差、固定动作步数和无训练状态。

这些测试 checkpoint 的 gate=1，用于真实路径验证；任务默认仍为 0。
不是可用于行走的策略。导出工具不自动修改 checkpoint 中的 gate。

独立导出额外位于 `cli_export/`；4A 导出回归位于 `cli_export_4a_regression/`。
`outputs/` 由仓库忽略规则排除，产物保存在本机，不作为已提交的训练权重。

完整预测路径基准命令：

```powershell
conda run --no-capture-output -n env_isaaclab python -u isal/scripts/benchmark_affordance.py `
  --checkpoint outputs/stage4b_acceptance/predicted_default/untrained_checkpoint.pt `
  --output outputs/stage4b_acceptance/benchmark.json
```

实测条件：batch=1、17×11、两脚 374 queries、chunk=64、gate=1；warmup=20、
重复 100 次、Torch CPU threads=2、PyTorch 2.7.0+cu128。运行完整 `act_inference`，
测量前仿真回归已退出。GPU 每次同步，报告主机计时的端到端调用耗时。

| 设备 | 中位延迟 ms | 平均 ms | P95 ms |
|---|---:|---:|---:|
| Intel Core i5-13400F CPU | 1.7885 | 1.8242 | 2.2657 |
| NVIDIA GeForce RTX 3060 | 2.4947 | 2.6058 | 3.4720 |

CUDA baseline allocated 为 **14,717,952 bytes**，峰值 allocated 为
**15,842,816 bytes（15.109 MiB）**，相对 baseline 增量为
**1,124,864 bytes（1.073 MiB）**。这是完整模型驻留后的 PyTorch 分配统计，
不等于进程显存/驱动占用，也不是单独导出模型的显存测量。参数量与上表一致。
初次基准试运行修正了脚本调用 gate setter 的名称，以上为修正后的完整成功结果。

## 5. 限制及待外部训练验证的事项

没有未通过的本阶段功能验收项；以下研究问题继续留待外部训练验证。

- 训练样本仍来自 liftoff snapshot，控制预测覆盖当前每个控制时刻；两者分布差异尚未验证。
- 控制 context 保留 Actor 噪声，tracker context 使用既有干净物理量；未修改样本协议解决差异。
- Head 未训练；普通随机 CNN/MLP 不保证镜像等变。仅解析假 head 被要求满足已知的空间对称关系，
  实际模型记录镜像 MAE，不把它当作失败阈值或学习质量证据。
- 固定零动作场景不保证产生有效 liftoff 样本；样本协议由阶段 3 回归和合成梯度测试验证。
- 工程开销使用固定合成输入，不构成闭环性能、成功率、样本效率或泛化结果。
- 本阶段完成后等待验收；不实现阶段 5 的 buffer、辅助损失、双 schedule 或阶段 6 外部训练。
