# 感知裁剪、动态镜像与采样诊断验收

日期：2026-09-07。范围：HeightScan / Interaction 任务的两项修复及诊断扩展。

## 实现结果

### 高度与诊断

- 默认预处理改为 `clamp(terrain_z-root_z, -1.5, 0.4)/0.5`；Actor、Critic、liftoff snapshot 保持同一定义。
- root_z=0.75 m 时，地形 `0、−0.05、−0.10、−0.20、−0.30 m` 的输出分别为 `−1.5、−1.6、−1.7、−1.9、−2.1`，不再提前压平下降高度差。
- 原始射线生成 finite/lower/upper 三个 bool mask，非有限射线与有限下限饱和分开统计。等于边界计为饱和。
- 全图与 query 邻域计数随 liftoff 输入保存；邻域使用旧坐标系中最近网格点及一圈邻居，边界裁掉，不重复计点。等距取较小坐标。
- touchdown 时将旧 mask 缩减为计数，pending 不持有整张诊断图。finalize、自动 reset 保留本步输出、显式 reset 清理旧输出的规则与原 tracker 一致。
- `extras['auxiliary']` 添加诊断可用标志、两类区域的点计数及比例。原字段、target 公式、样本有效性规则保持不变。
- recorder 的 JSON 使用累计点数计算比例；CSV/PT 保留逐样本信息。旧包没有诊断时显式标记不可用；空输出没有 NaN。

### 动态布局与镜像

- 新任务公开 `perceptive_observation_layout`，包含实际网格、native ordering、Actor/Critic history length 和单帧维度。
- 初始化重新解析最终 perception 参数，更新已有 scanner 配置，不重建或覆盖其他 scene 设置。调用 Isaac Lab 实际 pattern 得到射线数，并与运行时 sensor 网格核对。
- Actor 状态固定为当前机器人 78 维；Critic 非扫描状态固定为 139 维，并用实际当前观测校验。Critic 单帧总维度动态为 `139+射线数`。
- 新任务镜像按历史帧拆分状态和扫描，保留旧状态左右交换/符号规则，扫描转规范 `(x,y)` 后翻转 y，再还原 native ordering。Actor 当前扫描仍为规范展平顺序。
- 默认 187 射线、10 帧历史的状态镜像结果与旧实现一致；非默认维度不再使用 326 的历史步长。
- 保留 `data_augmentation_func(env, obs, actions)` 接口、仅观测/仅动作调用，以及 Stage 1 旧路径。错误维度抛出明确异常。
- `--validate-only` 使用运行时布局，并实际执行一次 augmentation 形状检查；不调用学习或 optimizer update。

## 测试

### 全套 Stage 1–3

```powershell
conda run --no-capture-output -n env_isaaclab python -u -m pytest `
  isal/tests -q --tb=short -p no:cacheprovider
```

最终结果：**83 passed, 1 warning in 109.96s**。

覆盖：

- 既有 Stage 1 配置/源边界与 Stage 2/3 回归；tracker 合成测试在 CPU/CUDA 上执行。
- 高度边界、下降高度差、NaN/Inf 独立计数、无有限点和空诊断。
- `17×11`、`17×9`、`19×11`，`xy/yx` ordering，`10/10` 和 `3/4` 历史组合。
- 非对称地形 `h=x+10y` 的空间语义、各帧独立标记、双镜像还原及默认旧实现一致性。
- 配置构造后的尺寸、分辨率、偏移和历史覆盖；输入布局不匹配拒绝。
- 离地 mask 复制所有权、旋转坐标系、中心/边界邻域、pending 与 reset、诊断缺失及全部无效的情况下 target 不变。
- recorder 的点数加权汇总、旧/新包混合、不可用 CSV 字段、空数据 JSON/PT 导出。
- 默认真实场景与非默认真实场景（每个新动态场景使用独立进程）：较小 HeightScan 为 153 射线，较大 Interaction 为 209 射线；两个环境、40 个固定零动作步、wrapper、网络前向、augmentation 和导出。
- 真实场景搭配明确标记的合成接触/终止反馈，验证自动 reset 前后诊断 packet 的存续。

首轮受限沙箱运行结果为 74 passed、5 failed、4 errors：5 个真实场景因 USD/Kit 缓存权限未完成，4 个 recorder 用例因 pytest 临时目录权限在 fixture 阶段失败。获准常规缓存访问后完整重跑得到上述 83 passed，最终没有遗留测试失败或权限阻塞。

唯一 pytest warning 是原有 RSL-RL `empirical_normalization` 弃用提示；Isaac Sim 退出时仍有原有 USD detach / recursive unload 信息。

### 公共 CLI 动态配置

```powershell
conda run --no-capture-output -n env_isaaclab python -u `
  isal/scripts/rsl_rl/train.py --task ISAL-Humanoid-Rough-HeightScan-v0 `
  --headless --num_envs 1 --validate-only `
  'env.terrain_perception.size=[1.8,1.0]' `
  env.robot.actor_obs_history_length=3 env.robot.critic_obs_history_length=4 `
  env.scene.height_scanner.pattern_cfg.ordering=yx
```

验证结果：

```text
policy_shape=(1,234)
height_scan_shape=(1,209)
critic_shape=(1,1392)
actor_input_shape=(1,443)
height_scan_grid_shape=(1,1,19,11)
action_shape=(1,23)
dynamic_augmentation=True
reward_finite=True
runner.learn_called=False
```

该控制步的扫描总点数/有限点数均为 209，lower/upper/invalid 均为 0。此单步结果仅用于链路检查，不代表全部地形或正常步态的饱和率。

日志：`outputs/perception_acceptance/validate_dynamic.log`。

### 公共 CLI 真实标签导出

```powershell
conda run --no-capture-output -n env_isaaclab python -u `
  isal/scripts/debug_interactions.py --headless --num_envs 1 --steps 200 `
  --output outputs/perception_acceptance/zero_action_200
```

真实固定零动作运行得到 2 条有效样本，其中 1 条 partial。两条 target 分别为
`0.3302446604`、`0.1284778863`，与修复前同配置、同种子的验收 CSV 一致。
两条记录均携带诊断；全图点数累计 374，query 邻域分别为 9 和 6 点（累计 15），
此次记录均为有限点，lower/upper/invalid 计数均为 0。

导出的 JSON、CSV 包含诊断及新的预处理元数据，PT 包含原始样本 tensor。
该运行没有 runner；这些结果证明真实采样与诊断导出链路通过，不代表正常步态质量。
非默认大网格的 40 步真实测试未产生有效落脚样本，正确导出空数据与不可用诊断。

## 兼容与后续工作

- 标签公式、奖励、课程、动作控制和 Stage 1 基线行为保持原样。`robolab`、`rsl_rl` 工作区没有修改。
- 默认网络形状未变，但输入数值分布变化。旧实验需显式恢复 `min_height=-0.8`；不自动转换 checkpoint。射线数变化需要匹配的新模型维度。
- 正常步态数据分布、真实 query 饱和比例和训练收益未验证；本次没有启动训练，也没有调用 optimizer update。
- 原 `STAGE2_ACCEPTANCE.md`、`STAGE3_ACCEPTANCE.md` 保留历史结论及旧参数。本记录说明此次增量修复。
