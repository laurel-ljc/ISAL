# ISAL 规格与阶段计划修订审查

日期：2026-09-07。状态：修订建议，尚未替代原规格，也不代表阶段 4 已启动。

已完整阅读 `humanoid_self_supervised_affordance_codex_spec.md`，读取引用任务“整理人形机器人论文”的最近两轮完整问答，并核对阶段 0/3、感知修复验收记录和当前环境、采样器、网络配置及本地 PPO 源码。

## 1. 结论与研究边界

需要修订，但不需要推倒阶段 0～3。最新讨论增加了第二个研究问题：除了辅助监督能否改善样本效率，还要验证把预测接触质量显式提供给 Actor，能否进一步改善落脚行为。

建议保留三个主实验：

| ID | 变体 | 辅助监督 | 预测作为 Actor 输入 | 奖励 | 部署运行 head |
|---|---|---|---|---|---|
| E1 | PPO Heightmap Baseline | 无 | 无 | 原 locomotion reward | 否 |
| E2 | Aux-only | 有 | 无 | 与 E1 相同 | 否 |
| E5 | Affordance-as-Observation | 有 | 有，输出 detach | 与 E1 相同 | 是 |

沿用原 E3（shuffled labels）和 E4（aux coefficient）编号，避免已有实验名被重新解释。E5 是新增必做组，前提是项目最终采用新讨论中的显式输入方向。

- E2 vs E1 检验辅助表征学习。
- E5 vs E2 检验显式预测输入，必须控制 Actor 容量差异。
- E6 可留给真实 outcome reward 的后续消融，不进入当前主方法；不默认实现 predicted-affordance reward。

两项研究假设都待实验验证。Aux-only 可能通过已有稳定性、滑移和终止奖励改善落脚；不能因为没有显式 affordance 输入，就断言它无法改善落脚。同样，把预测提供给 Actor 也不保证 Actor 会使用它，更不保证性能提高。引用对话中的结果表属于预期，不能当作项目结论。

标签仍称为 **interaction-derived contact-quality score**，而非安全落脚概率、可达性判定或对所有候选落点的因果保证。

## 2. 原规格需要修改的位置

| 原规格章节 | 修订内容 |
|---|---|
| §0.5、§1、§2、§4 | 将“只做训练期辅助任务”限定为 E2；增加 E5 的第二条研究假设和控制数据流。仍不引入 planner。 |
| §7、§9.3、§9.5、§36 | 添加预测输入分支、query 网格、context 契约与配置；继续复用同一个 query head，不要求另建 dense prediction 网络。 |
| §10、§17 | 主实验保持原奖励和现有标签；明确 target 语义、失败样本和无 touchdown 的局限。 |
| §20、§21、§23、§26 | 同步实际 packet；定义延迟结算与 rollout 边界；增加 E5 的 PPO 重算、梯度和预测漂移约定。 |
| §22、§35、§39 | 分离采样开关、辅助学习开关和预测输入开关；辅助权重与预测输入强度分别调度。 |
| §42、§49 Test 8、§57、§64 | E1/E2 可以删除 head；E5 必须保留 head 和输入分支。导出测试按变体执行。 |
| §43～§46、§56、§68 | 三组主实验、容量控制、预测使用诊断、推理开销和多 seed 对比。 |
| §48、§60.5、§63 | “预测图只可视化、不进入 Actor”仅适用于 E1/E2；E5 不再属于遥远的未来工作。 |
| §50、§66 | 与用户的阶段 0～8 门禁统一；工程阶段与 terrain training stage 分开命名。 |
| §61 | 不把所有 PPO 奖励描述为 sparse reward；当前任务已有 dense locomotion rewards。无额外部署复杂度的主张仅适用于 E2。 |

此外，§41 的命令必须使用本项目实际入口及 `conda run -n env_isaaclab python ...`，明确训练只在外部训练机执行。

## 3. 不依赖新研究方向、也应同步的现状

### 3.1 高度预处理与 observation

当前已采用：

```text
clamp(terrain_z - root_z, -1.5, 0.4) / 0.5
```

因此 §7.2、§37 的 `min_height=-0.8` 应更新为 `-1.5`；尺度范围应改成 `[-3.0, 0.8]`。参数实现在 `isal/isal/tasks/direct/humanoid_rough/terrain_perception_cfg.py`。旧实验兼容必须显式记录旧参数，不能悄悄改变 checkpoint 的输入定义。

当前实际接口：

- Actor `policy`：78 维每帧，默认 10 帧历史，合计 780 维。
- `height_scan`：独立 observation group，但对外是 `[N,H*W]`，内部保留 `[N,1,H,W]`；默认 187 个点。
- Critic：默认 10 帧，每帧 `139 + H*W`，默认 3260 维。
- 网格、ordering、历史长度已通过 `perceptive_observation_layout` 动态解析；Actor canonical ordering 与 Critic native ordering 的区别已有镜像实现。

工作计划阶段 2 的“独立二维 group”应改成上述真实契约。阶段 4 在模型边界还原 CNN 网格即可，不应为满足旧文案重写已通过的环境接口。保留 Actor 历史；Critic 独立 encoder 的历史处理也必须明确，不能不记录就把 10 帧改成单帧。

### 3.2 已完成的 interaction 契约

§19～§20 仍描述单条 pending、每环境每步最多一条结果，以及双脚事件任选其一。这已落后于阶段 3：

- 每脚独立 swing snapshot，加多个 pending outcome 槽。
- `extras['auxiliary']['valid']` 为 `[N,2,P]`，其他字段共享此前缀。
- 同步结算的双脚及多条 pending 全部保留，禁止任选一条丢弃。
- 有效样本应逐字段用 `packet[key][packet['valid']]` 展平。
- 早期 fall 按已观察窗口结算，并标记 partial；timeout 不当成 fall。
- 自动 reset 后仍发布本步终止前结果；显式 reset 清理旧结果。

这些应作为正式契约写回规格，历史验收文件保留原日期和结论。

### 3.3 框架、目录与 terrain

当前是项目独立包 `isal/` 下的 DirectRLEnv 继承链，不是 ManagerBasedRLEnv，也不需要改成后者。阶段 0 中“新增实现放入 robolab”的历史规划已被独立 `isal/` 包取代。

阶段 1 计划写着 flat + very mild rough，但当前 Rough 配置实际包含 flat 40%、stairs 20%、slopes 20%、grid 10%、rough 10%。不能在计划中宣称默认已是 easy terrain。

阶段 6 应显式增加共享的 `easy` 实验预设用于最初稳定行走验证，再用受控的低难度几何混合验证辅助信号。阶段 7 才扩充完整难度与 stepping stones。所有组使用同一预设；沿用课程规则不等于各组实际经历的 terrain level 分布一定相同，需同时记录。

## 4. E5 网络的建议实现契约

### 4.1 数据流

```text
current height scan -> actor terrain encoder -> z_t -------> Actor
                                               |
                                               v
                         query head(z_t, q, side, context_t)
                                               |
                                           A_left/right
                                               |
                                             detach
                                               |
                                  fixed centering + input gate
                                               |
                                      Actor input branch

proprio history -----------------------------------------> Actor

clean critic observations -> independent encoders --------> Value

liftoff scan + actual touchdown query + liftoff context
                         -> same terrain encoder/query head
                         -> SmoothL1 with observed outcome
```

预测属于 policy 内部计算出的特征，环境不调用学习模型。环境只提供因果可用的传感器输入和训练期 extras。Actor 不读取 future outcome、actual future touchdown、pending target 或地形 metadata。

E1/E2 继续使用原 `concat(z_proprio,z_terrain)` 接口。E5 增加固定维度预测分支。可以为三组预留同样尺寸的输入槽，E1/E2 恒填零，以保持 Actor 参数形状一致；必须报告 E5 多了有效输入，不能仅凭参数数目相同就声称消除了全部容量影响。建议额外加入参数量相近的普通 latent 分支对照。

### 4.2 Query 的空间与时序

第一版 E5 建议先用左右脚各 K 个固定 query，复用 §9.5 的 MLP；不引入 argmax 落点选择、Top-K planner 或 swing-foot 专用 action。

- 可先测试每脚 5×5 的规则 query lattice，K=25，仅是工程起点。
- 坐标从实际 scan 覆盖范围构造，单位为米，以 root yaw frame 表示；不重复添加 `offset_x`。
- 左右脚使用同一空间网格加不同 side one-hot，便于动态镜像；query 数与范围配置化并随 checkpoint 保存。
- 此网格只是采样位置，不宣称每个点对对应脚均可达，也不把 scan 有效性 mask 当成安全标签。
- 训练时 actual touchdown query 可以是连续坐标，不必投到最近网格点。监督仍只作用于真实交互位置；其余位置只是预测，不能自举成真标签。
- 全分辨率 `[2,H,W]` 用于后续可视化；是否给 Actor 用完整网格由开销和性能测试决定。

必须区分两个时间原点：训练样本 query 位于 liftoff frame；E5 当前帧预测 query 位于当前 root frame。各自的 scan、context、query 必须来自同一时刻和坐标约定，不能把旧 query 直接配给当前 scan。

目前 head 的训练输入只采自 liftoff，E5 若每个控制步都查询，就新增了状态分布迁移。建议先明确 E5 每步预测的含义为“假设从当前状态开始后续落脚的质量估计”，并将 liftoff-only 监督到全控制周期的泛化列为待验证项，不能宣称已经对所有 gait phase 有监督。

如果 phase 分层诊断表明不成立，应在独立的阶段 3 增量包中增加有限个因果的 pre-contact snapshots，并为每条快照独立重算 query；E2/E5 必须使用同样样本协议。另一路是使用 liftoff 缓存预测，但那需要部署态缓存、reset、age 和坐标变换，不能无说明地切换成有状态 policy。当前审查不要求立刻改 tracker。

### 4.3 Context 与 normalization

head 使用 command、base angular velocity、projected gravity。当前 tracker 保存物理量，Actor 本体 group 已经过 scale，并可能加噪声；因此不能直接从经过 normalizer 的 780 维向量切片后送入 head。

阶段 4/5 要定义统一的 context adapter：明确 frame、物理尺度、当前历史帧索引、噪声处理和可部署来源。若增加独立原始 context group，必须在各实验共享环境中一致提供，且不能让 E5 独享更干净的传感器信息而不做说明。

建议 CNN scan 仅使用固定物理 clip/scale，不额外进行 running normalization；proprio 和 critic state 的 normalizer 保持配置化。Actor 与 auxiliary 的 scan preprocessing 完全一致。辅助 minibatch 不单独更新 running statistics。归一化策略变更须对各组一致应用并记录，不能称作旧 MLP 的数值等价复制。

## 5. 梯度与 PPO 更新必须补充的约定

### 5.1 detach 的准确含义

```python
z = actor_terrain_encoder(scan)
a_pred = query_head(z, queries, side, context)
a_input = input_gate * (2.0 * a_pred.detach() - 1.0)
action_mean = actor(z_proprio, z, a_input)
```

仅为结构伪代码。query expansion、batch layout 等由阶段 4 实现。

| 损失来源 | Actor terrain encoder | Query head | Actor body/输出层 | Critic-only |
|---|---|---|---|---|
| policy/entropy/actor symmetry | 经直接 z 路径有梯度 | 无直接梯度 | 有 | 无 |
| value loss | 无 | 无 | 无 | 有 |
| auxiliary loss | 有 | 有 | 无直接梯度 | 无 |

detach 必须位于 head 输出。仅写 `head(z.detach())` 仍会让 PPO 更新 head；只在采样时 no_grad、更新时不 detach 也不满足此契约。

**detach 不冻结数值。** PPO 改变共享 encoder，auxiliary 改变 encoder/head，下一次预测都会变化。辅助梯度也不受 PPO clipped surrogate 的硬约束，即使只使用当前 rollout 也不能保证总策略变化被 clip 限制。

### 5.2 存储与重算

推荐 E5 作为完整 policy 内部计算图：

1. rollout 存原始 observation、action、old log_prob、old mean/std；head 不在环境中更新。
2. PPO minibatch 使用当前参数，从存下来的原始输入重新计算 z 和预测输入，再计算新 log_prob。
3. 辅助 loss 另用本次 buffer 中的 sample 计算，联合 optimizer step。
4. 不用缓存旧预测替代当前完整 policy 的新预测来计算 ratio；若未来要采用冻结预测器，需单独定义 encoder/head 的冻结范围和更新时机。

这与本地 `rsl_rl/rsl_rl/algorithms/ppo.py` 中采样保存旧分布、update 重算新分布的流程一致。缓存预测可用于诊断，但不应悄悄改变待优化 policy 的定义。

增加固定 probe batch 的预测漂移、action-mean 漂移及更新后的 KL 诊断。仅看更新前 KL 不足以发现最后一个 auxiliary step 的影响。无需第一版增加 target network；先用小辅助权重、输入 gate 和已有稳定性监测。

### 5.3 两个独立 schedule 与开关

建议新增配置（拟定字段，尚非当前可执行 CLI）：

```yaml
experiment_variant: ppo_heightmap | ppo_heightmap_aux | ppo_heightmap_affobs
auxiliary_learning:
  enabled: true
  start_iteration: 100
  ramp_iterations: 200
  loss_coef: 0.05
affordance_observation:
  enabled: false
  detach: true
  query_shape: [5, 5]
  centering: fixed_2a_minus_1
  gate_start_iteration: 300
  gate_ramp_iterations: 100
  gate_final: 1.0
contact_reward:
  enabled: false
```

上述 gate 数值只是默认候选，应在外部短训练后确认。训练从第一个 minibatch 就输入随机预测并非必要。辅助权重 warmup 与预测输入 gate 不应共用一个变量。

- `self_supervised.enabled` 当前控制 tracker；保留其现有语义。
- E1：tracker/aux learning/预测输入全部关闭。
- E2：tracker 打开；辅助权重可暂时为零；预测输入关闭。
- E5：tracker 与辅助学习打开；预测分支存在，gate 按调度开启。
- `lambda_aux=0` 只表示不施加辅助损失，不代表 tracker 必须关闭，也不代表 E5 自动等价 E1。

schedule 在 rollout/update 边界推进，不能在同一 minibatch 内意外改变门控。保存恢复 iteration、环境总步数、两套 schedule 状态和输入 gate；部署使用 checkpoint 实际 gate，不自动跳到最终值。

外部验证必须覆盖 gate 开启并稳定一段时间：若采用 300/100 的 gate 默认值，200 iteration 运行无法验证 E5，500 iteration 也仅有约 100 次完全开启后的更新。

### 5.4 延迟标签与 rollout 边界

当前 rollout 为 24 控制步，survival 为 25 步，另有 swing 延迟。若强制 liftoff 和 finalize 都落在同一 rollout，会丢失大量完整样本。

建议正式定义：aux buffer 只存本次 rollout **新结算**的样本；tracker 中未完成的物理事件允许跨 rollout 保留，同一样本只结算、消费一次。update 后清空学习 buffer，不清掉仍在发生的事件。

这不等于跨 iteration replay，但样本快照可能由较早策略产生。记录 snapshot/sample age 和跨边界计数，不宣传所有辅助输入都来自当前参数策略。恢复 checkpoint 时不恢复旧 pending 物理事件和 aux buffer；训练状态与新环境 reset 一致。

### 5.5 Symmetry

已有 height-scan 和历史帧动态镜像应复用。

E5 在镜像的原始 scan/context 上重新运行 query head；空间映射为 y 反号、左右脚交换、command 中 vy/yaw-rate 反号、角速度按轴向量规则转换。镜像不能只交换两个预测向量而遗漏内部 query 的 y 顺序。

普通 CNN/MLP 不保证 `head(mirror(input)) == mirror(head(input))`；不能给任意随机权重写这种必然相等的测试。应测试坐标/side 映射与模型消费路径，用可控的解析假 head 检验空间语义，并记录训练后的预测镜像误差。若该误差影响 E5，再单独评估 auxiliary sample 镜像或一致性约束，并同步相关对照组。

## 6. 标签与 reward 的处理

阶段 3 标签公式先保留，以免同时更换监督目标和控制接口。

原审查已发现：若 slip/tilt 很小，touchdown 同帧 fall 的 target 仍可能约为 0.667；完整 outcome 后在 survival 边界 fall 可能得到 0.85。因此它不能解释成“高分必然不会摔倒”。这些是标签定义的性质，不是新模型能自动消除的问题。

至少分层记录：partial/full、survival=0/1、左右脚、query 空间覆盖、missed touchdown，以及各评分分量。未接触区域没有可靠标签，dense 查询不是完整可通行性地图；人体晃动和摔倒也不一定能归因于某一只脚。

如果未来需要“失败必低分”，应单独提交 label-v2 规格与回归包，E2/E5 同时使用新版。不能在 E5 中悄悄改成 edge/support-area 标签；当前 tracker 并没有测量真实接触面积，不能从地形几何规则生成后再称其为交互自监督。

真实 outcome reward 也不是无成本扩展：target 要等窗口结算，无法在 touchdown 时立即获得；可能与原 slip/orientation/termination 奖励重复，还需要明确延迟信用分配、早期终止和每次接触奖励导致的步频偏好。故先保持 E1/E2/E5 的原 reward 完全一致。

## 7. 对阶段计划的建议改版

沿用“阶段完成 → 验收包 → 用户确认 → 下一阶段”。本机禁止策略训练，所有 Python/pytest/Isaac Lab 命令使用 env_isaaclab。阶段编号保留，只拆分新增分支的验收。

| 阶段 | 建议工作与新增验收 |
|---|---|
| 0～3 | 保留已完成实现；同步第 3 节现状。新 head 使用当前 packet，不要求返工。标签、context 或多快照若确需改变，另交阶段 3 增量包。 |
| 4A | CNN、proprio encoder、独立 Critic、query head，首先 E1/E2、lambda=0；确定历史和 normalization；做原有 shape、参数独立、FQN 加载、checkpoint 与可删 head 导出测试。 |
| 4B | E5 query 输入分支、detach、固定 query 和容量控制；补充 context/frame、镜像、无未来信息依赖及完整 head 导出；仅本地合成前后向和允许的少量仿真。4A 通过后才进入。 |
| 5 | Current-rollout buffer、joint loss、双 schedule、变体一致性、sample age、预测漂移及日志；必要时添加项目本地 runner。验证 E1 无采样/无辅助梯度，E2/E5 梯度路由及 resume。 |
| 6A | 外部训练机：共享 easy terrain 的 E1、E2；先 0.01，再 0.05。基础稳定后在共享低难度几何混合上检查标签和预测，不以纯平地高分塌缩否定方法。 |
| 6B | 6A 通过后，外部机器验证 E5，覆盖 gate 全开后的稳定区间，检查预测/动作漂移、KL、跟踪和落脚指标。明确记录是从头训练还是从 E2 初始化，后者计入之前全部训练成本。 |
| 7 | 仅在 6A/6B 通过后完成全面 terrain curriculum、泛化、固定权重评估、CSV 和 dense 可视化；增加预测输入使用诊断及真实 query 覆盖分析。 |
| 8 | E1/E2/E5 至少 3 seeds；E3/E4 推荐；容量对照建议加入，E6 reward 可选。报告失败 seed、均值方差及样本/时间/部署成本。 |

阶段 4 的“lambda=0 时 action 一致”须限定为相同 Actor 权重、输入、normalizer、预测输入关闭或 gate=0，比较 action mean 或受控随机采样。训练过的 E2 与 E1 不要求 action 相同；E5 gate>0 时即使 lambda=0，也不能要求 action 不变。

## 8. 新增测试和研究验收

本轮只审查文档与代码，没有执行以下测试；这些是阶段 4/5 的待实现验收。

1. E1/E2 相同权重、head 开关下 deterministic action mean 一致；随机数消耗不能污染初始化公平性。
2. E5 gate=0 的容量匹配配置能复现对应 E2；gate>0 时，受控预测扰动可沿预定分支影响 action。
3. 分别对 auxiliary、policy/entropy/symmetry、value 做 backward，核对第 5.1 节路由。
4. 动态 H/W、query 网格、ordering、非默认历史长度与镜像坐标语义测试。
5. raw context 与辅助 context 经统一适配后尺度一致；aux batch 不更新 normalizer。
6. 未更新权重且冻结 normalization/schedule 时，新旧 log_prob 的 ratio≈1；改变 head 后 E5 的完整 policy 重算能反映变化。
7. 延迟结算跨 rollout、双脚多 pending、update 清空和 resume 无旧样本复用。
8. lambda warmup 与输入 gate 分别保存恢复，恢复前后同 observation 的完整策略输出一致。
9. E1/E2 导出去掉 head；E5 导出保留 head、query 参数、输入分支、normalizer/gate，分别对原模型验证输出一致。
10. 单次合成优化路径只用于单元测试，不调用 runner.learn，不开展本机策略学习。

研究阶段还应增加：

- E5 固定 checkpoint 下，预测输入置零/空间打乱/左右交换的动作敏感性及完整 episode 性能诊断；这些是干预实验，会改变输入分布，不能单独当成因果证明。
- 真正的落脚指标：slip、后续稳定性、query 覆盖、有效接触质量；若要报告精确落脚或边缘距离，先定义独立评估测量，不能拿模型自己的预测当成功率。
- E1/E2/E5 在同一固定测试 terrain 集上评估 success，事先定义 success 和阈值；训练课程中的 reward 曲线不能直接替代固定条件的成功率。
- 足够数量的已实现接触用于 held-out prediction 评估，并分层分析 partial/full。没有接触的候选点不能计算有真值的 MAE。
- 环境步数、完整训练时间、峰值显存、E5 完整部署延迟；迁移初始化计入来源训练成本。

## 9. 文件位置和后续交付

已有文件继续复用：

- 感知参数：`isal/isal/tasks/direct/humanoid_rough/terrain_perception_cfg.py`。
- 事件与标签参数：`isal/isal/interaction/config.py`。
- Tracker：`isal/isal/interaction/tracker.py`。
- 环境与 observation：`isal/isal/tasks/direct/humanoid_rough/`。
- Agent/PPO/symmetry 配置：`isal/isal/tasks/direct/humanoid_rough/agents/isal_agent_cfg.py`。

建议新增（尚未实现）：

- `isal/isal/learning/models/affordance_actor_critic.py`：三变体模型及完整导出 wrapper。
- `isal/isal/learning/models/affordance_query.py`：共享 query 构造、context 适配和批量预测。
- `isal/isal/learning/algorithms/ppo_with_affordance.py`：辅助更新和完整策略诊断。
- `isal/isal/learning/storage/auxiliary_rollout_buffer.py`：本次新结算样本。
- `isal/isal/learning/config.py`：网络、辅助优化和预测输入参数；与事件标签参数区分。
- 项目本地 runner 扩展：仅在现有接口不足时增加，负责日志和可恢复 schedule。

本次交付仅本审查文件。原规格与阶段计划应按本文确定的取舍统一修订后，再进入阶段 4。没有修改源码、没有调用训练、没有重跑历史验收；此前 83 项通过等数字属于既有验收记录，不是本轮结果。
