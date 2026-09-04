# 人形机器人自监督地形可供性辅助学习项目设计规格（面向 Codex 实现）

> **建议项目名**：Interaction-Supervised Affordance Learning for Sample-Efficient Perceptive Humanoid Locomotion  
> **简称**：ISAL-Humanoid  
> **目标框架**：Isaac Lab + RSL-RL（PPO）  
> **文档用途**：直接交给 Codex 作为工程实现规格。  
> **核心要求**：先做一个简单、稳定、可消融的研究原型，不把创新押在 Transformer、Cross-Attention、World Model 或复杂视觉网络上。

---

# 0. 给 Codex 的总指令

请先完整阅读本规格，再修改代码。

实现时遵守以下原则：

1. **先审计现有仓库，再写代码。**
   - 找到当前 Isaac Lab 版本、RSL-RL 版本、训练入口、任务注册方式、已有 humanoid locomotion task、已有 terrain/reward/observation 写法。
   - 优先复制并继承一个已经能稳定走路的 humanoid velocity locomotion task，不要从零重写基础 locomotion。
   - 不要假设仓库使用最新 RSL-RL API。当前上游 RSL-RL 已支持通过配置传入自定义 model/algorithm，但本项目本地版本可能不同；请以仓库实际代码为准适配。

2. **尽量不修改上游 Isaac Lab 和 RSL-RL 源码。**
   - 优先在项目自己的 `learning/`、`tasks/`、`mdp/` 等目录新增类。
   - 如果本地 RSL-RL 版本确实无法注册自定义 algorithm/model，再做最小补丁。
   - 所有补丁必须集中、注释清晰，避免散落修改。

3. **第一版只使用 RayCaster height scan，不使用真实渲染 Depth Camera。**
   - 研究问题是“能否利用 foot-terrain interaction 自监督信号改善 perceptive RL 的样本利用效率”。
   - 不要同时引入 depth sim2real、camera noise、视觉域随机化等额外难题。
   - 第二阶段再将 height scan 替换为 depth image。

4. **Self-supervised label 只能由机器人交互后果构造。**
   - 可使用：foot contact、foot velocity、IMU/base orientation、joint state、contact force、是否摔倒等。
   - 不允许用 terrain type、真实 friction、真实 slope class、手工 terrain label 作为 self-supervised target。
   - Critic 可以使用 privileged observation，但不能拿 privileged terrain metadata 构造所谓“self-supervised”标签。

5. **主方法中，affordance prediction 只作为 auxiliary task，不直接作为 action planner。**
   - Actor 使用 terrain encoder 的 latent。
   - Auxiliary head 使用相同 terrain encoder，预测“这一步真实落脚后的 contact quality”。
   - 训练时 auxiliary loss 反向传播到 terrain encoder。
   - 部署时 affordance head 可以删除，policy 结构不依赖它。
   - 这样可以干净地验证：额外的 interaction-grounded learning signal 是否能提高 perceptive PPO 的 sample efficiency。

6. **第一版不要跨 PPO iteration 使用旧 replay 数据更新共享 terrain encoder。**
   - PPO 是 on-policy。
   - 主方案只使用“当前 rollout 中新产生的 self-supervised samples”，在 PPO update 的多个 epoch 内复用。
   - 跨迭代 replay 仅作为后续可选实验，不作为 MVP。

7. **所有重要超参数必须集中到 config。**
   - 禁止在算法、label 计算和网络内部散落 magic number。
   - 下面文档给出的默认值可以调整，但必须配置化。

---

# 1. 项目到底要做什么

## 1.1 研究问题

普通 perceptive locomotion PPO 会从：

- 本体感知；
- 局部地形高度扫描；
- 最终 RL reward；

中学习 terrain representation。

问题在于，每一次 foot-terrain interaction 实际包含很多额外信息，例如：

- 这一脚有没有滑；
- 这一脚接触后身体有没有明显倾倒；
- 支撑是否稳定持续；
- 是否很快发生 recovery 或摔倒。

传统 PPO 最终只通过 reward / return 间接利用这些信息，存在信息利用不足。

本项目提出：

> **同一批 locomotion rollout 同时用于 PPO 更新和 interaction-grounded self-supervised affordance prediction，使 terrain encoder 更快学到“对控制真正有用的地形表征”，从而提高 perceptive locomotion 的环境样本利用效率。**

核心不是构建一个高层 foothold planner，而是增加一个训练期辅助任务。

---

## 1.2 核心假设

### H1：辅助 affordance 学习能提高样本效率

在相同的 environment steps 下：

- Baseline：PPO + height scan；
- Proposed：PPO + height scan + interaction-supervised auxiliary loss；

Proposed 应更快达到相同 success rate / tracking performance。

### H2：辅助任务学到的是控制相关 terrain feature，而不是单纯几何重建

通过 contact outcome 监督，terrain encoder 应更关注：

- 台阶边缘；
- 半脚支撑区域；
- 高低突变；
- 落脚后容易产生 body disturbance 的区域；
- 与当前运动状态相关的局部地形。

### H3：最终策略不依赖 auxiliary head

部署时删除 affordance head，Actor 仍正常工作。

如果 Proposed 仍优于 Baseline，则说明收益来自训练期 representation shaping，而不是额外 planner。

---

# 2. 第一版明确不做什么

为了控制工程难度，MVP 不包含：

- raw RGB；
- rendered depth camera；
- cross attention；
- Transformer；
- recurrent world model；
- teacher-student；
- diffusion；
- deformable terrain；
- 沙地、泥地等复杂物理材料；
- running；
- 高层路径规划；
- 显式 foothold planner；
- gait generator；
- 跨 iteration 大规模 experience replay；
- perception uncertainty。

第一版只解决：

> **Walking humanoid + static rough terrain + local height scan + PPO + self-supervised contact-quality auxiliary task。**

---

# 3. 软件和框架要求

## 3.1 Isaac Lab

使用 Manager-Based RL Environment（如果现有任务已经是 ManagerBasedRLEnv，则沿用）。

主要使用：

- `TerrainGeneratorCfg` / `TerrainImporterCfg`；
- `RayCasterCfg` + `GridPatternCfg`；
- `ContactSensorCfg`；
- Observation / Reward / Curriculum / Event managers；
- 现有 joint-position action term；
- RSL-RL wrapper。

RayCaster 第一版只扫描静态 terrain mesh，不要设计动态地形。

## 3.2 RSL-RL

使用 PPO。

实现一个自定义 algorithm/model 扩展，使每个 PPO minibatch 的总损失变为：

\[
L_{total}=L_{PPO}+\lambda_{aux}L_{aff}
\]

其中：

\[
L_{PPO}=L_{policy}+c_vL_{value}-c_eH
\]

\[
L_{aff}=\operatorname{SmoothL1}(\hat y_{contact},y_{contact})
\]

**注意：**

- auxiliary gradient 只应流经 `terrain_encoder` 和 `affordance_head`；
- 不要让 auxiliary loss 更新 critic；
- 不要直接对 actor 输出层施加 auxiliary target；
- terrain encoder 是 Actor 与 auxiliary head 的共享模块。

---

# 4. 整体数据流

```text
                      ┌──────────────────────┐
                      │   Local Height Scan  │
                      │     17 x 11          │
                      └──────────┬───────────┘
                                 │
                                 ▼
                         Terrain Encoder
                                 │
                            z_terrain (64)
                           /             \
                          /               \
                         ▼                 ▼
              Actor Policy Body      Affordance Head
                    ▲                     ▲
                    │                     │
               proprioception       query position
                    │               foot side
                    │               motion context
                    │                     │
                    ▼                     ▼
                  action        predicted contact quality
                                          │
                                          │ SmoothL1
                                          ▼
                               self-supervised target
                                          ▲
                                          │
                                  future foot contact
                                  slip / stability /
                                  persistence / fall
```

训练期间：

```text
rollout
  ├── PPO transition --------------------> PPO loss
  │
  └── foot interaction event -----------> auxiliary sample
                                           |
                                           v
                                      affordance loss
```

部署期间：

```text
height scan -> terrain encoder -> actor -> action
```

Auxiliary head 不需要运行。

---

# 5. Environment 基线

## 5.1 优先继承已有稳定 humanoid task

Codex 首先搜索仓库中：

- flat humanoid locomotion；
- rough terrain humanoid locomotion；
- velocity tracking task；
- 当前机器人 asset 的稳定 reward 配置。

如果已有可稳定行走的 task：

1. 完整 clone 一份新的 task；
2. 保持原 action、PD、基础 reward、termination 基本不变；
3. 新增 height scan；
4. 新增 auxiliary interaction sampler；
5. 新增 custom policy/algorithm。

**不要为了新项目重新调一套基础走路 reward。**

只有在仓库完全没有可用 task 时，才采用本文档第 10 节提供的 fallback reward。

---

# 6. 动作设计

第一版采用最普通的 joint position action：

\[
q_{target}=q_{default}+s_a a_t
\]

其中：

- `a_t ∈ [-1, 1]^N`；
- `N = actuated_dof`；
- `s_a` 沿用现有稳定 humanoid task 的 action scale。

建议：

```yaml
action_type: joint_position
action_scale: use_existing_task_value
control_frequency: 50 Hz preferred
physics_frequency: 200 Hz or existing value
decimation: keep existing task
```

不要增加：

- foothold action；
- phase action；
- residual action；
- torque policy。

---

# 7. Actor Observation 设计

设机器人可驱动关节数为 `N`。

## 7.1 Proprioception observation

Actor 使用：

| Observation | Dim | 说明 |
|---|---:|---|
| base angular velocity | 3 | body frame |
| projected gravity | 3 | body frame |
| velocity command | 3 | vx, vy, yaw rate |
| joint position relative to default | N | normalized |
| joint velocity | N | scaled |
| previous action | N | 上一控制周期 action |

总维度：

\[
D_p = 9 + 3N
\]

第一版**不要求 Actor 使用真实 base linear velocity**，除非现有任务本来就使用且实机也有对应估计器。

如果现有 task 有必须保留的 observation，可以保留，但请记录在 config 和 README 中。

---

## 7.2 External terrain observation：局部 height scan

使用 Isaac Lab `RayCasterCfg + GridPatternCfg`。

推荐默认参数：

```yaml
height_scan:
  size_x: 1.6       # m
  size_y: 1.0       # m
  resolution: 0.1   # m
  offset_x: 0.4     # scan center 向机器人前方偏移
  ray_alignment: yaw
```

目标网格约为：

```text
H = 17
W = 11
```

具体点数以本地 Isaac Lab `GridPatternCfg` 实际生成结果为准，不要硬编码 reshape，初始化时从 sensor pattern 计算并 assert。

扫描区域约覆盖：

```text
x: [-0.4 m, +1.2 m]
y: [-0.5 m, +0.5 m]
```

相对 root/pelvis 前向偏置，重点覆盖未来 1~2 步落脚区域。

### Height preprocessing

从 ray hit world z 构造：

\[
h_{ij}=z_{hit,ij}-z_{root}
\]

然后：

```python
h = clip(h, min_height, max_height)
h = h / height_scale
```

建议默认：

```yaml
min_height: -0.8
max_height: 0.4
height_scale: 0.5
```

目标是数值大致落在 `[-2, 1]`，再由 observation normalizer / encoder 处理。

不要直接把 world z 输入网络。

### Actor height scan noise

MVP：

```yaml
noise_std: 0.0
random_dropout: 0.0
```

待核心方法验证后，再做 perception robustness 扩展。

---

# 8. Critic Observation

使用 asymmetric actor-critic，但不要过度 privileged。

Critic 可以使用：

- Actor 全部 observation；
- base linear velocity（3）；
- base height（1）；
- left/right foot contact state（2）；
- left/right foot normal contact force（2）；
- clean height scan（如果后续 Actor scan 加 noise）。

第一版不建议把：

- terrain type id；
- terrain difficulty ground truth；
- terrain friction ground truth；

塞给 critic，除非现有 baseline 本来就这么做。

原因：保持实验简单，避免 critic 过度强大影响不同方法对比。

---

# 9. Neural Network 设计

创新不在网络，所以结构要简单。

## 9.1 Terrain Encoder

输入：

```text
[B, 1, H, W]
```

推荐：小型 CNN。

```text
Conv2d(1, 16, kernel=3, stride=1, padding=1)
ELU
Conv2d(16, 32, kernel=3, stride=2, padding=1)
ELU
Conv2d(32, 32, kernel=3, stride=1, padding=1)
ELU
Flatten
Linear(flat_dim, 128)
ELU
Linear(128, 64)
```

输出：

```text
z_terrain: 64
```

必须在初始化时根据实际 H/W 动态推断 `flat_dim`，不要手工计算写死。

---

## 9.2 Proprio Encoder

```text
proprio D_p
 -> Linear(D_p, 256)
 -> ELU
 -> Linear(256, 128)
 -> ELU
```

输出：

```text
z_proprio: 128
```

---

## 9.3 Actor

```text
concat(z_proprio=128, z_terrain=64)
 -> 192
 -> Linear(192, 256)
 -> ELU
 -> Linear(256, 128)
 -> ELU
 -> Linear(128, N_actions)
```

Action distribution 使用现有 RSL-RL Gaussian action distribution。

初始 action std 沿用现有稳定 PPO task；若无参考，默认 `1.0`，之后调试。

---

## 9.4 Critic

Critic 使用独立 encoder，不与 Actor terrain encoder 共享权重，避免 value loss 干扰辅助学习分析。

可使用简单 MLP，或：

```text
critic_proprio + privileged 1D -> MLP
clean height scan -> independent terrain CNN
concat -> MLP -> V(s)
```

如果当前 RSL-RL 版本实现独立 Actor/Critic model 更方便，则按本地结构实现。

---

## 9.5 Affordance Query Head

第一版**不直接输出整张 dense affordance map**。

原因：每一步真实 interaction 只给一个落脚位置提供可靠监督，dense supervision 很稀疏。

使用 query-based prediction：

输入：

1. `z_terrain`: 64；
2. touchdown query position `(x, y)`：2；
3. foot side one-hot：2；
4. velocity command `(vx, vy, yaw)`：3；
5. base angular velocity：3；
6. projected gravity：3。

总输入：

```text
64 + 2 + 2 + 3 + 3 + 3 = 77
```

Head：

```text
Linear(77, 128)
ELU
Linear(128, 64)
ELU
Linear(64, 1)
Sigmoid
```

输出：

\[
\hat y \in [0,1]
\]

解释：

> 在当前 terrain observation 和运动状态下，如果对应脚落在 query `(x,y)`，预计这次支撑质量有多好。

### 为什么不用 dense map

query head 更适合真实稀疏 interaction supervision，并且之后可以通过遍历网格 `(x,y)` 来生成 dense 可视化 affordance map，而无需改变训练目标。

---

# 10. Reward Function

## 10.1 首选原则

如果现有 humanoid task 已经稳定：

> **直接复制原 reward。**

本项目研究 auxiliary learning，不要一边改 reward 一边比较 sample efficiency。

Baseline 和 Proposed 必须使用完全相同 reward。

---

## 10.2 Fallback reward（仅当没有稳定 baseline 时）

建议如下：

### 正奖励

#### 线速度跟踪

\[
r_{lin}=\exp(-\|v_{xy}-v^{cmd}_{xy}\|^2/\sigma_v)
\]

```yaml
weight: +1.5
sigma: 0.25
```

#### yaw 角速度跟踪

\[
r_{yaw}=\exp(-(\omega_z-\omega_z^{cmd})^2/\sigma_w)
\]

```yaml
weight: +0.75
sigma: 0.25
```

#### alive / upright

```yaml
weight: +0.1
```

#### feet air time / gait encouragement

如果现有 task 有成熟实现则沿用。

```yaml
weight: +0.2
```

### 惩罚

| Reward term | 建议 weight |
|---|---:|
| vertical base velocity | -2.0 |
| roll/pitch angular velocity | -0.05 |
| orientation deviation | -1.5 |
| joint torque squared | -2e-6 |
| joint acceleration squared | -2.5e-7 |
| action rate | -0.01 |
| joint limit | -1.0 |
| undesired contact | -1.0 |
| foot slip | -0.2 |

这些数值只是 fallback，优先使用已有 task。

### 重要实验规则

如果 foot slip 同时用于 reward 和 self-supervised label：

- Baseline 也必须拥有同样的 foot slip reward；
- Proposed 的优势只能来自 auxiliary representation learning，而不是额外 reward 信息。

---

# 11. Episode Termination

沿用现有 humanoid termination。

典型包括：

- torso/pelvis 非预期接触地面；
- base height 低于阈值；
- roll/pitch 超过安全范围；
- episode timeout。

推荐 episode length：

```yaml
20 s
```

第一版 walking task：

```yaml
vx: [0.2, 1.0] m/s
vy: [-0.25, 0.25] m/s
yaw_rate: [-0.6, 0.6] rad/s
```

先不训练 running。

---

# 12. Self-Supervised Interaction Sample：核心设计

这是整个项目最重要的部分。

## 12.1 什么算一个 training sample

每次某只脚完成：

```text
liftoff -> swing -> touchdown -> short support outcome window
```

就产生一个候选 auxiliary sample。

样本包括：

```python
AuxSample = {
    "height_scan":       scan_at_liftoff,
    "query_xy":          actual_touchdown_xy_in_liftoff_frame,
    "foot_side":         left_or_right,
    "command":           command_at_liftoff,
    "base_ang_vel":      base_ang_vel_at_liftoff,
    "projected_gravity": projected_gravity_at_liftoff,
    "quality_target":    y_contact,
}
```

---

# 13. 为什么使用 liftoff snapshot

在 liftoff 时保存 terrain scan，代表：

> 机器人在这一步真正发生之前，对前方地形拥有的信息。

touchdown 后再观察实际结果。

这形成天然的：

```text
before-contact exteroception
        ↓
actual interaction
        ↓
after-contact pseudo label
```

与 self-supervised traversability 的思想一致。

如果实际机器人步态存在 liftoff detection 不稳定，则保留第二模式：

```yaml
snapshot_mode: pre_touchdown_lag
pre_touchdown_lag_steps: 5
```

但默认使用 `liftoff`。

---

# 14. Foot Event Detection

对左右脚分别维护状态。

定义接触：

```python
contact_now = normal_contact_force > contact_force_threshold
```

推荐：

```yaml
contact_force_threshold: 20 N
```

实际需要根据机器人质量和 sensor noise 调整。

### Liftoff

```python
liftoff = previous_contact and (not contact_now)
```

发生 liftoff 时保存：

- height scan；
- root world xy；
- root yaw；
- command；
- base angular velocity；
- projected gravity；
- foot side。

### Touchdown

```python
touchdown = (not previous_contact) and contact_now
```

touchdown 后：

1. 获取 foot world position；
2. 用 liftoff 时 root yaw + root xy 转到 snapshot local frame；
3. 得到 `query_xy`；
4. 如果 query 不在 height scan 覆盖范围，丢弃此 sample；
5. 创建 pending outcome record。

坐标转换：

\[
p_{local}=R_z(-\psi_{liftoff})(p_{foot,w}-p_{root,w}^{liftoff})
\]

只取 xy。

---

# 15. Outcome Window

Touchdown 后继续观察短时间窗口：

```yaml
outcome_window: 0.25 s
survival_window: 0.5 s
```

如果控制频率 50 Hz：

```text
outcome_window_steps ≈ 12~13
```

不要在代码中硬编码 steps，应由：

```python
round(outcome_window_s / control_dt)
```

计算。

---

# 16. Self-Supervised Quality Label

第一版 label 不追求物理完美，要求：

- 简单；
- 连续；
- 可从机器人自身反馈获得；
- 对“这次支撑是否可靠”有意义。

建议使用四个量。

## 16.1 Slip score

只在该脚接触期间统计水平脚速：

\[
v_{slip}=\operatorname{mean}(\|v_{foot,xy}\|)
\]

转换：

\[
s_{slip}=\exp(-v_{slip}/\tau_{slip})
\]

默认：

```yaml
slip_scale: 0.15 m/s
```

---

## 16.2 Body disturbance score

记录 touchdown 时 base roll/pitch，之后统计 outcome window 内最大变化：

\[
d_{tilt}=\max_t \sqrt{(\phi_t-\phi_0)^2+(\theta_t-\theta_0)^2}
\]

转换：

\[
s_{tilt}=\exp(-d_{tilt}/\tau_{tilt})
\]

默认：

```yaml
tilt_scale: 0.20 rad
```

如果现有代码更方便使用 projected gravity，则可用 projected gravity change 等价实现。

---

## 16.3 Contact persistence

\[
s_{contact}=\frac{\text{该脚在 outcome window 中保持 contact 的步数}}{K}
\]

对于 walking 任务，好的 foothold 通常能形成稳定支撑。

注意：未来如果训练 running，此指标需要重新设计，因为跑步的接触时间本来就短。

---

## 16.4 Survival score

如果 touchdown 后 `survival_window` 内发生 fall termination：

```text
s_survival = 0
```

否则：

```text
s_survival = 1
```

---

# 17. 最终 quality target

默认：

\[
y=0.40s_{slip}+0.25s_{tilt}+0.20s_{contact}+0.15s_{survival}
\]

然后：

```python
y = clamp(y, 0.0, 1.0)
```

配置：

```yaml
label_weights:
  slip: 0.40
  tilt: 0.25
  contact_persistence: 0.20
  survival: 0.15
```

第一版**不把 peak impact force 放进 target**，只记录到日志。

理由：impact 与 walking speed、controller stiffness 等高度相关，容易把 label 搞得复杂。

后续可以作为 ablation。

---

# 18. 没有 touchdown 怎么处理

例如：

- 踩空；
- 直接掉进 gap；
- swing 过程中机器人先摔倒。

第一版：

> **不生成该 auxiliary sample。**

原因：没有可靠 actual contact query。

因此 MVP terrain 不应以大 gap 为主，否则 bad sample 大量缺失。

后续可以设计 “failed-step negative sample”，但不要放在第一版。

---

# 19. Auxiliary Sample 生命周期

每个 env、每只脚维护：

```text
FootInteractionTracker
```

状态：

```text
STANCE
SWING_WITH_SNAPSHOT
PENDING_OUTCOME
```

### STANCE -> SWING_WITH_SNAPSHOT

liftoff 时保存 snapshot。

### SWING_WITH_SNAPSHOT -> PENDING_OUTCOME

touchdown 时计算 query，开始 outcome accumulation。

### PENDING_OUTCOME -> STANCE

outcome window 完成，输出一个 finalized sample。

如果 episode reset：

- 清空所有 pending tracker；
- 如果 survival label 已满足判定条件，可以在 reset 前 finalize；
- 否则丢弃 incomplete sample。

---

# 20. Auxiliary 数据如何从 Env 送给 RSL-RL

目标：不要建立 Python 全局变量或 CPU list，保持 GPU parallel。

每个 control step，每个 env 最多允许输出 1 个 finalized sample。

建议输出固定 tensor：

```python
auxiliary = {
    "valid":             Tensor[num_envs] bool,
    "height_scan":       Tensor[num_envs, 1, H, W],
    "query_xy":          Tensor[num_envs, 2],
    "foot_side":         Tensor[num_envs, 2],
    "command":           Tensor[num_envs, 3],
    "base_ang_vel":      Tensor[num_envs, 3],
    "projected_gravity": Tensor[num_envs, 3],
    "target":            Tensor[num_envs, 1],
}
```

如果同一个 env 理论上同一步左右脚同时 finalize：

- 第一版任选一个；或
- 扩展为 `[num_envs, 2, ...]`。

正常 walking 几乎不会成为主要问题。

使用现有 `env.step()` 返回的 `extras / infos` 通道传入 RSL-RL。

Codex 必须先查看本地：

- `RslRlVecEnvWrapper.step()`；
- `PPO.process_env_step()`；

确定当前版本 extras 的具体键结构。

不要凭最新 upstream API 猜。

---

# 21. Current-Rollout Aux Buffer

在 custom PPO algorithm 内创建：

```text
AuxiliaryRolloutBuffer
```

它只存**当前 PPO rollout**中新产生的 valid auxiliary samples。

每次 `process_env_step()`：

```python
valid = infos["auxiliary"]["valid"]
aux_buffer.append(infos["auxiliary"][valid])
```

每次 PPO `update()` 完成后：

```python
aux_buffer.clear()
```

不跨 iteration 保留。

建议 buffer 最大容量：

```yaml
max_samples_per_rollout: 16384
```

如果超过，随机 reservoir 或直接截断。

4096 envs、24-step rollout 下通常几千个 contact sample 即可。

---

# 22. Self-Supervised Learning 何时开始

一开始随机 policy 的 interaction 非常差，直接给较大 auxiliary weight 容易干扰 locomotion。

推荐：

```yaml
aux_start_iteration: 100
aux_ramp_iterations: 200
aux_loss_coef_final: 0.05
```

即：

```text
iter < 100:        lambda_aux = 0
100 ~ 300:         0 -> 0.05 linear ramp
iter >= 300:       lambda_aux = 0.05
```

公式：

\[
\lambda_{aux}(k)=\lambda_{max}\cdot
\operatorname{clip}\left(\frac{k-k_0}{K_{ramp}},0,1\right)
\]

需要做 coefficient ablation：

```text
0.00 / 0.01 / 0.05 / 0.10
```

---

# 23. PPO Update 中怎么加入 Auxiliary Loss

主方案：**同一个 PPO minibatch optimization step 中联合更新。**

伪代码：

```python
for epoch in range(num_learning_epochs):
    for rl_batch in rollout_storage.mini_batch_generator(...):

        policy_loss, value_loss, entropy_loss = compute_ppo_losses(rl_batch)

        if aux_enabled and len(aux_buffer) >= min_aux_samples:
            aux_batch = aux_buffer.sample(aux_batch_size)
            pred = model.predict_affordance(aux_batch)
            aux_loss = smooth_l1(pred, aux_batch.target)
        else:
            aux_loss = 0.0

        lambda_aux = aux_schedule(current_iteration)

        total_loss = (
            policy_loss
            + value_loss_coef * value_loss
            - entropy_coef * entropy
            + lambda_aux * aux_loss
        )

        optimizer.zero_grad()
        total_loss.backward()
        clip_grad_norm_(...)
        optimizer.step()
```

### Gradient 要求

`predict_affordance()`：

```text
height_scan -> shared actor terrain_encoder -> z_terrain
z_terrain + query/context -> affordance_head
```

因此 auxiliary gradient 更新：

```text
terrain_encoder
+ affordance_head
```

但不能更新：

```text
critic encoder
actor output head directly
```

Actor 受影响的渠道仅为共享 terrain representation。

---

# 24. Auxiliary Loss

第一版：

```python
SmoothL1Loss(beta=0.1)
```

若本地 PyTorch API不同则用普通 SmoothL1。

不建议第一版使用 binary classification，因为 contact quality 本质连续。

日志：

```text
aux/loss
a ux/num_samples  # 实现时不要真的带空格
a ux/target_mean
a ux/target_std
a ux/pred_mean
a ux/mae
a ux/correlation
```

实际 tag 请写成：

```text
aux/loss
aux/num_samples
aux/target_mean
aux/target_std
aux/pred_mean
aux/mae
aux/correlation
```

---

# 25. 样本分布不均问题

训练早期可能：

```text
大量 bad contacts
```

训练后期可能：

```text
大量 good contacts
```

第一版先记录 histogram，不立即复杂化。

建议日志 bins：

```text
bad:    y < 0.35
middle: 0.35 <= y < 0.70
good:   y >= 0.70
```

如果发现严重塌缩，再加入：

```yaml
balanced_aux_sampling: true
```

按三类近似均匀采样。

这应作为调试选项，不默认开启。

---

# 26. 为什么第一版不跨 iteration replay

不要直接做：

```text
old contact replay
 -> update shared terrain encoder many times
 -> current PPO actor representation suddenly changes
```

这样容易绕过 PPO 的 trust-region/clipped update 约束。

主方法已经能做到样本复用：

- 一个 environment interaction 同时产生 RL signal 和 auxiliary signal；
- 当前 rollout 的 auxiliary samples 会在多个 PPO epochs 内复用。

论文中应称：

> **multi-objective reuse of on-policy interaction experience**

而不是强行宣传大规模 off-policy replay。

后续如要研究 replay，可以单独设计 target encoder / frozen encoder / stage-wise pretraining，不放进 MVP。

---

# 27. Terrain 设计

第一版 terrain 必须满足：

1. 静态 mesh；
2. 几何因素真实影响 foothold quality；
3. 不依赖复杂材料模拟；
4. 大多数 step 最终能发生 touchdown，方便产生 label。

推荐 terrain mixture：

| Terrain | 比例 | 用途 |
|---|---:|---|
| flat | 10% | 保持基础 locomotion |
| random rough | 25% | 高频局部高度变化 |
| slopes up/down | 20% | 连续坡度 |
| stairs up/down | 25% | 明确结构化接触 |
| boxes/discrete obstacles | 10% | 台阶边缘/局部平台 |
| stepping stones | 10% | 支撑面积与边缘 |

第一版不把 large gaps 放入核心训练集合。

---

# 28. Terrain 参数建议

## 28.1 Difficulty levels

使用 Isaac Lab TerrainGenerator 的：

```text
difficulty ∈ [0,1]
```

建议：

```yaml
num_rows: 10   # 10 个 difficulty level
num_cols: according_to_terrain_types
curriculum: true
```

## 28.2 Rough terrain

```yaml
height_range:
  easiest: 0.00 ~ 0.02 m
  hardest: 0.06 ~ 0.08 m
horizontal_scale: existing/default
```

## 28.3 Slope

```yaml
slope_angle:
  easiest: 0 ~ 5 deg
  hardest: 15 ~ 20 deg
```

## 28.4 Stairs

```yaml
step_height:
  easiest: 0.02 ~ 0.05 m
  hardest: 0.16 ~ 0.18 m
step_width:
  0.25 ~ 0.35 m
```

## 28.5 Boxes

```yaml
box_height:
  easiest: 0.02 m
  hardest: 0.16 m
```

## 28.6 Stepping stones

如果项目已有对应 terrain，优先复用。

建议：

```yaml
stone_size:
  easy: 0.50 m
  hard: 0.25 ~ 0.30 m
stone_gap:
  easy: 0.02 m
  hard: 0.12 ~ 0.15 m
```

不要一开始做很窄 balance beam。

---

# 29. 不要在 MVP 使用空间随机 friction patch

这是一个重要设计约束。

如果 Actor 只看到 height scan，而 friction patch 在几何上完全不可见：

```text
相同 height observation
 -> 一个区域 friction=1.0
 -> 一个区域 friction=0.2
```

则 auxiliary 网络理论上无法从 height map 预测哪个更滑。

这会让 target 产生不可解释噪声。

因此：

- MVP 不使用 spatially varying invisible friction；
- 可以做轻微 episode-level global friction domain randomization，但先关闭；
- 未来加入 RGB/semantic/material perception 后，再研究 material-dependent traversability。

---

# 30. Terrain Curriculum

使用 adaptive difficulty curriculum。

每个 env 记录：

```text
start_xy
current_xy
episode_time
commanded_speed
terminated_early
```

估计：

\[
d_{expected}=\|v^{cmd}_{xy}\|T
\]

\[
d_{actual}=\|p_{xy}(T)-p_{xy}(0)\|
\]

对于 command norm 太小的 episode 不更新 terrain level。

推荐规则：

### Promote

```text
no fall
AND
d_actual > 0.8 * d_expected
```

### Demote

```text
early fall
OR
d_actual < 0.4 * d_expected
```

否则保持。

如果现有 Isaac Lab task 已有 `terrain_levels_vel` 一类 curriculum，优先继承现有实现，只调整 terrain set。

---

# 31. Training Curriculum 总体阶段

## Stage 0：基础 locomotion sanity

目标：确认新 task 没有被 sensor / custom model 改坏。

Terrain：

```text
flat 50%
very mild rough 50%
```

Auxiliary：

```text
off
```

只需要跑到能稳定基本行走。

如果已有 pretrained baseline，可直接跳过或仅做短 sanity run。

---

## Stage 1：核心 geometry curriculum

Terrain：

```text
flat
rough
slope
low stairs
low boxes
```

Auxiliary：

```text
100 iter 后开启
线性 ramp 到 lambda=0.05
```

目标：验证 proposed loss 不破坏 PPO。

---

## Stage 2：完整训练 terrain

加入：

```text
higher stairs
harder rough
stepping stones
```

Terrain level 自适应升降。

Auxiliary 保持稳定。

---

## Stage 3：Generalization evaluation

训练不再改变。

在未见参数上测试：

- 更高/更低 stair height；
- 新 rough seed；
- 新 box arrangement；
- 新 stepping stone layout；
- 轻微 height scan noise（可选）。

不要在 test 时继续更新权重。

---

# 32. Domain Randomization

MVP 先少量使用，不要影响主要研究变量。

可以沿用现有 task：

- base mass；
- link mass；
- motor strength；
- joint damping；
- external push。

但建议先：

```yaml
domain_randomization_level: mild
```

保证 Baseline 和 Proposed 完全一致。

若主要目标是论文 sample efficiency，前期实验甚至可以先关闭大部分 DR，验证方法后再加入。

---

# 33. 推荐项目目录结构

请根据实际 package 名调整，但保持职责分离。

```text
<repo>/
├── source/<project_pkg>/<project_pkg>/
│   ├── tasks/
│   │   └── locomotion/
│   │       └── humanoid_affordance/
│   │           ├── __init__.py
│   │           ├── humanoid_affordance_env_cfg.py
│   │           ├── terrain_cfg.py
│   │           ├── self_supervised_cfg.py
│   │           ├── agents/
│   │           │   ├── __init__.py
│   │           │   └── rsl_rl_ppo_cfg.py
│   │           └── mdp/
│   │               ├── __init__.py
│   │               ├── observations.py
│   │               ├── rewards.py
│   │               ├── curriculums.py
│   │               ├── events.py
│   │               └── interaction_labels.py
│   │
│   └── learning/
│       ├── __init__.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── affordance_actor_critic.py
│       ├── algorithms/
│       │   ├── __init__.py
│       │   └── ppo_with_affordance.py
│       └── storage/
│           ├── __init__.py
│           └── auxiliary_rollout_buffer.py
│
├── scripts/
│   └── rsl_rl/
│       ├── train.py       # 尽量不改
│       └── play.py        # 尽量不改
│
└── tools/
    └── visualize_affordance.py
```

如果现有项目已有类似结构，优先融入现有结构，不要机械复制目录。

---

# 34. 各文件职责

## `humanoid_affordance_env_cfg.py`

负责：

- robot；
- scene；
- sensors；
- actions；
- observations；
- rewards；
- terminations；
- commands；
- curriculum；
- event randomization。

## `terrain_cfg.py`

只负责：

- terrain generator；
- terrain proportions；
- difficulty ranges。

## `self_supervised_cfg.py`

集中所有辅助学习参数。

## `mdp/interaction_labels.py`

负责：

- foot event tracking；
- liftoff snapshot；
- touchdown query；
- outcome accumulation；
- target 计算；
- 输出 fixed tensor auxiliary data。

不要在这里写 optimizer。

## `learning/models/affordance_actor_critic.py`

负责：

- terrain encoder；
- proprio encoder；
- actor；
- critic；
- affordance head；
- `act()` / `evaluate()` / `predict_affordance()` 等与本地 RSL-RL 对接接口。

## `learning/algorithms/ppo_with_affordance.py`

负责：

- 继承或包装现有 PPO；
- 收集 aux samples；
- PPO + auxiliary joint update；
- auxiliary logging；
- coefficient schedule。

## `storage/auxiliary_rollout_buffer.py`

只存当前 rollout auxiliary samples。

---

# 35. 配置文件：必须集中参数

推荐 dataclass：

```python
@configclass
class SelfSupervisedAffordanceCfg:
    enabled: bool = True

    snapshot_mode: str = "liftoff"

    contact_force_threshold: float = 20.0
    outcome_window_s: float = 0.25
    survival_window_s: float = 0.50

    slip_scale: float = 0.15
    tilt_scale: float = 0.20

    slip_weight: float = 0.40
    tilt_weight: float = 0.25
    contact_weight: float = 0.20
    survival_weight: float = 0.15

    start_iteration: int = 100
    ramp_iterations: int = 200
    loss_coef: float = 0.05

    aux_batch_size: int = 2048
    min_samples_per_update: int = 256
    max_samples_per_rollout: int = 16384

    loss_type: str = "smooth_l1"
    smooth_l1_beta: float = 0.1

    balanced_sampling: bool = False
```

---

# 36. Network 参数配置

```python
@configclass
class AffordanceNetworkCfg:
    terrain_channels: list[int] = [16, 32, 32]
    terrain_latent_dim: int = 64

    proprio_hidden_dims: list[int] = [256, 128]
    actor_hidden_dims: list[int] = [256, 128]

    affordance_hidden_dims: list[int] = [128, 64]

    activation: str = "elu"
```

不要把网络尺寸写死在 module 中。

---

# 37. Terrain perception 参数配置

```python
@configclass
class TerrainPerceptionCfg:
    size_x: float = 1.6
    size_y: float = 1.0
    resolution: float = 0.1
    offset_x: float = 0.4

    min_height: float = -0.8
    max_height: float = 0.4
    height_scale: float = 0.5

    noise_std: float = 0.0
    dropout_prob: float = 0.0
```

---

# 38. PPO 默认参数

优先继承现有稳定 agent cfg。

如果没有，可从：

```yaml
num_steps_per_env: 24
max_iterations: 5000
save_interval: 100

ppo:
  num_learning_epochs: 5
  num_mini_batches: 4
  clip_param: 0.2
  gamma: 0.99
  lam: 0.95
  value_loss_coef: 1.0
  entropy_coef: 0.01
  learning_rate: 1.0e-3
  max_grad_norm: 1.0
  desired_kl: 0.01
```

开始。

但不要为了符合本文档覆盖项目里已经验证过的 PPO 参数。

---

# 39. Experiment Variant 配置

为了做公平实验，必须能用一个 config switch 开关方法。

至少支持：

```text
Variant A: ppo_heightmap
Variant B: ppo_heightmap_aux
```

建议：

```python
experiment_variant: str = "ppo_heightmap_aux"
```

内部对应：

```text
ppo_heightmap:
    self_supervised.enabled = false

ppo_heightmap_aux:
    self_supervised.enabled = true
```

除此之外所有设置一致。

---

# 40. Task 注册

建议注册两个 Gym task ID，或一个 task + 两个 agent config。

例如：

```text
Humanoid-Affordance-Rough-v0
```

Baseline 与 Proposed 最好共享同一个 environment，只切换 algorithm/agent cfg。

这样可以确保 terrain/reward 不被误改。

---

# 41. 训练命令

具体命令必须适配当前仓库现有脚本。

目标形式类似：

```bash
python scripts/rsl_rl/train.py \
  --task Humanoid-Affordance-Rough-v0 \
  --headless \
  --num_envs 4096
```

Baseline：

```text
self_supervised.enabled=false
```

Proposed：

```text
self_supervised.enabled=true
```

如果仓库使用 Hydra，请让所有上述参数可通过 Hydra override 修改。

不要额外造一套 argparse 配置系统。

---

# 42. Checkpoint

必须确保 checkpoint 包含：

- actor；
- critic；
- terrain encoder；
- affordance head；
- optimizer state；
- observation normalizer state（若有）。

Auxiliary rollout buffer 不需要保存。

Play / export 时允许：

```text
disable_affordance_head=true
```

并验证 action output 与包含 head 时完全一致。

---

# 43. Logging

除原 PPO 日志外增加：

```text
aux/loss
aux/coef
aux/num_samples
aux/target_mean
aux/target_std
aux/pred_mean
aux/mae
aux/bad_fraction
aux/mid_fraction
aux/good_fraction

contact/slip_mean
contact/tilt_change_mean
contact/persistence_mean
contact/survival_fraction

terrain/mean_level
terrain/promotion_rate
terrain/demotion_rate
```

Sample-efficiency 相关：

```text
train/environment_steps
train/success_rate
train/fall_rate
train/mean_episode_length
```

必须能够使用 environment steps 作为 TensorBoard 横轴或导出到 CSV。

---

# 44. Evaluation Metrics

## 44.1 主指标：Environment sample efficiency

必须比较：

```text
Success Rate vs Environment Steps
```

以及：

```text
Mean Reward vs Environment Steps
```

不要只画 iteration。

环境步数：

\[
N_{steps}=N_{env}\times N_{rollout}\times N_{iterations}
\]

如果使用多 GPU，按真实总 env interaction 统计。

---

## 44.2 Steps-to-threshold

例如：

```text
达到 70% success 需要多少 environment steps
达到 80% success 需要多少 environment steps
```

最终报告：

\[
\text{sample reduction}
=1-\frac{N_{ours}}{N_{baseline}}
\]

---

## 44.3 Final locomotion metrics

- velocity tracking error；
- fall rate；
- terrain completion rate；
- episode length；
- energy / torque（可选）；
- foot slip。

---

## 44.4 Auxiliary prediction metrics

单独验证辅助任务不是瞎学：

- MAE；
- Pearson correlation；
- good/bad binary ROC-AUC（仅作为额外分析）；
- calibration plot 可选。

---

# 45. 最少需要的实验组

## E0：Proprio-only（可选）

没有 height scan。

用来说明 perceptive terrain observation 本身有价值。

## E1：PPO Heightmap Baseline（必须）

```text
height scan
PPO
no auxiliary loss
```

## E2：PPO + Interaction Auxiliary（必须）

```text
height scan
PPO
self-supervised contact quality
```

## E3：Shuffled Auxiliary Label（推荐）

把当前 rollout 的 target 随机打乱。

如果 E2 > E3，说明收益不是“随便加个 auxiliary gradient”。

## E4：Aux coef ablation（推荐）

```text
0.01 / 0.05 / 0.10
```

---

# 46. Seeds

至少：

```text
3 seeds
```

如果计算资源允许：

```text
5 seeds
```

不要用单条 learning curve 宣称 sample efficiency。

---

# 47. Generalization Set

训练集和测试集必须使用不同随机 seed。

另外设计参数外推：

### Train

```text
stairs height <= 0.16 m
rough <= 0.06 m
stone gap <= 0.10 m
```

### Test

```text
stairs 0.16 ~ 0.20 m
rough 0.06 ~ 0.08 m
stone gap 0.10 ~ 0.15 m
```

注意不要设置成机器人机械上必然无法完成。

---

# 48. Affordance 可视化工具

增加：

```text
tools/visualize_affordance.py
```

对一个当前 observation：

1. encode height scan；
2. 枚举 grid 中每个 `(x,y)`；
3. 分别假设 left/right foot；
4. 调用 query head；
5. 得到：

```text
left_affordance[H,W]
right_affordance[H,W]
```

保存为 numpy / image，或在 debug viewer 中显示。

这个 dense map **仅用于分析和论文图，不进入 Actor**。

---

# 49. 单元测试 / 工程测试

至少实现以下 tests。

## Test 1：Height scan shape

```text
expected [num_envs, 1, H, W]
finite
no NaN
```

## Test 2：Contact event

人工让机器人静止站立：

- 不应重复产生 touchdown event。

脚离地再落地：

- 产生一次 sample。

## Test 3：Coordinate transform

已知 root yaw 和 foot world position，检查 query local xy 正确。

## Test 4：Quality target range

```text
0 <= y <= 1
```

## Test 5：Bad contact sanity

人工制造高 slip：

```text
y_bad < y_stable
```

## Test 6：Gradient routing

一次 auxiliary backward 后：

```text
terrain_encoder grad != 0
affordance_head grad != 0
critic-only params grad from aux == 0
```

## Test 7：Aux off reproduces baseline

```text
self_supervised.enabled=false
```

时 custom algorithm 应退化为普通 PPO，不能仍然产生 auxiliary gradient。

## Test 8：Export

删除/忽略 affordance head 后，Actor action 与训练模型 Actor action 数值一致。

---

# 50. 调试顺序

严格按以下顺序，不要同时开全部功能。

## M0：Repository audit

输出一份简短记录：

```text
Isaac Lab version
RSL-RL version
existing task cloned from
train entrypoint
custom model registration mechanism
```

## M1：复制 baseline task

要求：

- 不加 aux；
- 训练/播放正常；
- performance 与原 task 接近。

## M2：只加 height scan

使用普通 PPO。

检查：

- tensor shape；
- debug vis；
- Actor 能训练。

## M3：只实现 interaction tracker

先不训练 auxiliary network。

记录：

```text
liftoff count
touchdown count
sample count
query distribution
target histogram
```

并可保存 1000 个 sample 分析。

## M4：加入 affordance head

先：

```text
lambda_aux = 0
```

确保网络接入不改变 baseline。

## M5：开启 auxiliary loss

```text
lambda_aux = 0.01
```

短跑 200~500 iter，检查稳定性。

再使用：

```text
0.05
```

## M6：完整 curriculum

最后再加入全部 terrain difficulty。

---

# 51. Debug 必看统计

如果训练效果差，先看这些，不要立刻改网络：

1. `aux/num_samples` 是否足够；
2. query_xy 是否大量落在 scan 外；
3. target 是否几乎全为 0 或全为 1；
4. `aux/loss` 是否下降；
5. affordance prediction 与 target 是否有相关性；
6. terrain encoder gradient norm 是否异常大；
7. 加 aux 后 PPO KL 是否突然升高；
8. action std 是否塌缩；
9. terrain curriculum 是否过快升难度；
10. baseline 在相同 terrain 上是否本来就不稳定。

---

# 52. Auxiliary Gradient 稳定性保护

建议配置：

```yaml
aux_loss_coef: 0.05
max_grad_norm: use_existing_ppo_value
```

额外记录：

```text
grad/terrain_encoder_from_total
grad/affordance_head
```

如果 auxiliary 干扰 PPO：

按顺序尝试：

1. `lambda_aux 0.05 -> 0.01`；
2. 延长 warmup；
3. auxiliary head 学习率缩放 0.5；
4. 每两个 PPO minibatch 做一次 aux loss；
5. 最后才考虑 gradient surgery / separate optimizer。

第一版不要上复杂 PCGrad。

---

# 53. 关于 Separate Optimizer

MVP 建议一个联合 optimizer，最简单。

如果本地 RSL-RL 结构非常不方便，可以使用：

```text
ppo_optimizer
aux_optimizer
```

但必须保证：

- auxiliary optimizer 只包含 terrain encoder + affordance head；
- auxiliary step 发生在 PPO update 内；
- 每个 iteration 只使用 current rollout aux samples；
- 日志记录两者学习率。

联合 optimizer 优先。

---

# 54. 关于 Observation Normalization

Proprioception 沿用 RSL-RL normalizer。

Height scan 建议在 env observation term 中先做物理尺度 clip/scale，再使用 model normalization（若现有框架支持）。

Auxiliary sample 中保存的是：

> 与 Actor 同定义、同 preprocessing 的 height scan。

不要让 Actor 和 auxiliary head 看到两套尺度不同的数据。

---

# 55. 关于左右镜像

第一版不要专门做 mirror augmentation，除非现有 stable task 已经使用。

原因：

- 先验证 auxiliary signal；
- 避免 foot-side label 和镜像 transform 引入额外 bug。

未来若开启 symmetry augmentation，必须同步变换：

- height scan 左右翻转；
- query y 取负；
- left/right foot one-hot 对换；
- proprio 对应关节映射。

---

# 56. 论文层面的公平性要求

Baseline 与 Proposed 必须保持：

- 同 robot；
- 同 terrain；
- 同 terrain curriculum；
- 同 reward；
- 同 PPO hyperparameters；
- 同 network actor/critic capacity；
- 同 randomization；
- 同 num_envs；
- 同 rollout length。

Proposed 唯一主要新增：

```text
affordance_head + auxiliary loss
```

否则无法证明 sample efficiency 来自 self-supervised interaction learning。

---

# 57. 计算开销报告

论文中除 environment steps 外，同时报告：

```text
wall-clock training time
GPU memory
policy inference latency
```

因为 Proposed 增加 auxiliary compute。

理想结果：

- environment steps 明显下降；
- wall-clock 也下降或增幅较小；
- deployment inference 无额外 head，因此运行时几乎无额外开销。

---

# 58. 第二阶段：从 Height Scan 扩展到 Depth

只有当 height-scan 版本证明 hypothesis 成立后才做。

第二阶段：

```text
Depth Image
 -> Depth Encoder
 -> z_terrain
 -> Actor
```

Auxiliary supervision 完全不改：

```text
foot interaction
 -> quality label
```

这正是方法可扩展性的关键。

未来可以比较：

```text
Depth PPO
vs
Depth PPO + Interaction Auxiliary
```

此时 self-supervision 的意义更强，因为真实视觉 feature 更难仅靠 RL reward 学好。

---

# 59. Depth 扩展时不要立刻使用 Cross-Attention

第二阶段第一版仍建议：

```text
Depth CNN -> latent
Proprio MLP -> latent
concat -> Actor
```

研究点仍然是：

> interaction supervision 是否改善 terrain representation 与 sample efficiency。

如果普通 concat 已经有效，就没有必要引入复杂 fusion。

---

# 60. 第三阶段可研究的方向（不进入当前实现）

以下只记录，不要现在实现：

1. uncertainty-aware affordance；
2. material / RGB conditioned affordance；
3. spatial friction self-supervision；
4. cross-iteration auxiliary replay；
5. dense affordance control input；
6. foothold planner；
7. active perception；
8. real robot online adaptation；
9. failed-step negative mining；
10. running-aware contact label。

---

# 61. 论文可能的核心方法描述

可以用以下逻辑写论文：

### 问题

Perceptive humanoid RL relies primarily on sparse task rewards to learn control-relevant terrain representations, under-utilizing rich interaction feedback generated by each footstep.

### 方法

Use future foot-terrain interaction outcomes as self-supervised targets for an auxiliary foothold-quality prediction task. The auxiliary head shares the terrain encoder with the PPO actor.

### 关键点

```text
same rollout
 -> RL return
 -> contact-derived pseudo label
```

让每一次环境 interaction 同时产生：

- policy learning signal；
- terrain representation learning signal。

### 目标

Improve sample efficiency without increasing deployment-time policy complexity.

---

# 62. 与无人车 Self-Supervised Traversability 的联系

参考思想：

**How Does It Feel? Self-Supervised Costmap Learning for Off-Road Vehicle Traversability, ICRA 2023**。

其核心思想是：

```text
exteroceptive terrain observation
        ↓
vehicle drives through terrain
        ↓
proprioceptive / IMU interaction feedback
        ↓
self-supervised traversability target
```

本项目将这个思想从：

```text
vehicle-terrain traversability
```

转换到：

```text
foot-terrain interaction quality
```

并进一步将其从高层 costmap learning 引入：

```text
low-level perceptive PPO representation learning
```

这是项目的主要研究定位之一。

---

# 63. 与现有人形 terrain affordance 工作的区分

当前已有工作会使用：

- flatness；
- steepness；
- height feasibility；
- imagined foothold；
- geometric terrain cost；

来帮助 foothold selection 或 reward shaping。

本项目第一版刻意不做手工 terrain affordance cost。

区别：

> **affordance target 来自真实 foot contact 后果，而不是地形几何规则。**

同时第一版不直接用 affordance map 控制机器人，而是把它作为 terrain encoder 的辅助监督。

---

# 64. 验收标准（Codex 完成后必须满足）

## 工程验收

- [ ] 新 task 能注册、训练、play。
- [ ] `self_supervised.enabled=false` 时等价普通 PPO。
- [ ] height scan shape 正确。
- [ ] 左右脚 interaction tracker 正常。
- [ ] auxiliary samples 能从 env 进入 PPO algorithm。
- [ ] target 全部在 `[0,1]`。
- [ ] auxiliary loss 能下降。
- [ ] auxiliary gradient 到 terrain encoder。
- [ ] Critic 不被 auxiliary loss 直接更新。
- [ ] checkpoint/load 正常。
- [ ] export policy 不需要 auxiliary head。

## 训练验收

先做短训练，不要求论文级结果：

- [ ] Baseline 在 easy terrain 能稳定学习。
- [ ] Proposed 不出现明显 PPO collapse。
- [ ] 每 iteration 有足量 aux samples。
- [ ] target histogram 非完全塌缩。
- [ ] prediction 与 target 出现正相关。

## 研究验收

最后进行：

- [ ] 3 个以上 seed。
- [ ] Baseline / Proposed 相同 setup。
- [ ] 画 success vs environment steps。
- [ ] 计算 steps-to-70% / steps-to-80%。
- [ ] 测试 unseen terrain parameters。

---

# 65. Codex 实现完成后需要输出的说明

完成代码后，请生成一个：

```text
IMPLEMENTATION_NOTES.md
```

内容包括：

1. 修改/新增了哪些文件；
2. 本地 Isaac Lab / RSL-RL 版本；
3. 新 task ID；
4. baseline 训练命令；
5. proposed 训练命令；
6. 所有 self-supervised 参数位置；
7. terrain 参数位置；
8. network 参数位置；
9. reward 参数位置；
10. 如何关闭 auxiliary；
11. 如何画 affordance map；
12. 目前尚未实现的可选功能。

---

# 66. 建议 Codex 的最终执行顺序

不要一次性写完整系统。严格按：

```text
1. Audit repo
2. Clone stable humanoid rough locomotion task
3. Verify baseline
4. Add height scanner
5. Verify height scanner
6. Implement foot interaction tracker
7. Log labels only
8. Inspect label quality
9. Implement terrain encoder + affordance head
10. Set lambda_aux=0, verify baseline equivalence
11. Implement custom PPO auxiliary loss
12. Train easy terrain with small lambda
13. Add aux coefficient ramp
14. Add full terrain curriculum
15. Add evaluation scripts
16. Add dense affordance visualization
17. Run baseline/proposed multi-seed experiments
```

如果某一步失败，不要继续叠加下一步。

---

# 67. 最终推荐的 MVP 超参数汇总

```yaml
simulation:
  control_frequency: 50Hz   # 若现有 task 不同则沿用现有值
  episode_length: 20.0

height_scan:
  size_x: 1.6
  size_y: 1.0
  resolution: 0.1
  offset_x: 0.4
  noise_std: 0.0
  dropout_prob: 0.0

network:
  terrain_latent_dim: 64
  terrain_channels: [16, 32, 32]
  proprio_hidden_dims: [256, 128]
  actor_hidden_dims: [256, 128]
  affordance_hidden_dims: [128, 64]

self_supervised:
  enabled: true
  snapshot_mode: liftoff
  contact_force_threshold: 20.0
  outcome_window_s: 0.25
  survival_window_s: 0.50

  slip_scale: 0.15
  tilt_scale: 0.20

  label_weights:
    slip: 0.40
    tilt: 0.25
    contact: 0.20
    survival: 0.15

  start_iteration: 100
  ramp_iterations: 200
  loss_coef: 0.05
  aux_batch_size: 2048
  min_samples_per_update: 256
  max_samples_per_rollout: 16384
  smooth_l1_beta: 0.1

commands:
  vx: [0.2, 1.0]
  vy: [-0.25, 0.25]
  yaw_rate: [-0.6, 0.6]

terrain:
  curriculum: true
  num_levels: 10
  proportions:
    flat: 0.10
    rough: 0.25
    slopes: 0.20
    stairs: 0.25
    boxes: 0.10
    stepping_stones: 0.10
```

---

# 68. 最后提醒

这个项目最重要的不是把网络做复杂，而是保证实验逻辑干净：

```text
同样的 PPO
同样的 reward
同样的 terrain
同样的 actor capacity
同样的 environment steps

唯一核心差别：
是否使用 foot-interaction-derived auxiliary supervision
```

如果它能显著让 learning curve 左移，那么这就是最直接、最有说服力的结果。

第一版成功以后，再把 `height scan -> depth image`，研究价值会进一步提高。

---

# 69. 参考实现与背景资料（供 Codex/研究阅读）

1. Isaac Lab 官方文档：RayCaster / GridPattern / ContactSensor / TerrainGenerator / RSL-RL wrapper。
2. RSL-RL 官方仓库与文档：PPO runner、algorithm、model、storage、自定义 model/algorithm 扩展方式。
3. Guaman Castro et al., **How Does It Feel? Self-Supervised Costmap Learning for Off-Road Vehicle Traversability**, ICRA 2023。
4. Huang et al., **BarlowWalk: Self-supervised Representation Learning for Legged Robot Terrain-adaptive Locomotion**, 2025。
5. 近期 humanoid terrain affordance / perceptive locomotion 工作只作为相关工作参考；当前工程不要复制其复杂 planner 或 fusion architecture。

---

**END OF SPEC**
