# ISAL 感知任务与交互标签采集审查

审查日期：2026-09-07。范围：当前工作区（包括尚未提交的 Stage 3 文件）、本地工程规格、Stage 2/3 验收记录，以及实际导出的零动作样本。本次没有修改任务实现，没有启动训练。

后续实施记录：本审查中的高度裁剪与动态 Critic 镜像问题已在用户授权的后续实施中修复，并增加了采样诊断。详见 `isal/PERCEPTION_IMPROVEMENTS_ACCEPTANCE.md`。下文保留审查时的代码状态、复现结果及标签语义风险，不将其作为修复后代码的当前状态。

## 结论

当前完成的是：普通带 height scan 的 PPO 环境，以及独立的足地交互采样器。采样器已经能够生成带输入快照、落脚 query、运动上下文和连续质量 target 的样本；尚无项目所需的 terrain encoder、query affordance head、auxiliary rollout buffer 或联合优化。

默认配置下，状态机、坐标转换与 reset 数据保护总体合理。但应先解决扫描范围与高度裁剪不匹配的问题；修改扫描尺寸时还会遇到 Critic 镜像增强的硬编码错误。标签公式基本遵照原规格，然而早期失败的评分语义仍需要调整或通过数据验证。

## 1. 审查发现

### 1.1 [P1] root-relative 高度的裁剪下限会抹去下台阶的深度差别

位置：`isal/isal/tasks/direct/humanoid_rough/height_scan.py:42`、`terrain_perception_cfg.py:11`。

当前计算是 `clip(hit_z - root_z, -0.8, 0.4) / 0.5`。机器人资产初始 root 高度是 0.75 m。因此，在 root 尚未下降时，低于当前地面 5 cm 的区域已经达到裁剪下限。以 root_z=0.75 为例，本次直接调用函数得到：

| 地形世界高度 | 输出 |
|---|---:|
| 0.00 m | -1.5 |
| -0.05 m | -1.6 |
| -0.10 m | -1.6 |
| -0.20 m | -1.6 |
| -0.30 m | -1.6 |

这不表示所有地形都不可见，而是多个下降高度在观察中不可区分。当前 rough 地形确实包含下台阶与下坡，后续 affordance head 也会收到同一份被裁剪的快照，无法通过网络恢复丢失的信息。

建议：按照实际 root 离地高度和预期最大落差扩大 root-relative 区间；或者先加入固定的名义身高偏移，再按地形相对高度裁剪。修改时同步 Actor、Critic、snapshot 的定义，并记录下限/上限饱和比例。原规格本身给出了这组参数，因此这是需要修正规格的设计问题，不是 Stage 3 私自偏离方案。

### 1.2 [P2] Critic 镜像仍固定为 187 条射线与 326 维单帧观测

位置：`isal/isal/tasks/direct/humanoid_rough/agents/isal_agent_cfg.py:86`、`:105`。

Actor scan 的整理和镜像使用实际网格尺寸，Critic 却固定 `generate_height_scan_mirror(139, 11, 17)`、10 帧、每帧步长 326。默认配置的 symmetry augmentation 开启，因此不能仅修改 `TerrainPerceptionCfg.size/resolution` 后直接训练。

本次执行源文件中相关纯函数和索引构建语句，结果：

| 射线数 | Critic 输入 | 镜像结果 |
|---:|---:|---|
| 187 | 3260 | 正常，3260 |
| 153 | 2920 | IndexError |
| 209 | 3480 | 输出仍为 3260，后续无法与原观测拼接 |

即便射线总数不变，改变 ordering 或网格形状也可能产生错误的空间镜像。

建议：根据实际 scan shape、native ordering、Critic 单帧布局和 history length 生成镜像；至少在初始化时拒绝不支持的配置。增加非默认网格的 augmentation 测试，不能仅测试“镜像两次恢复原状”。

### 1.3 标签语义风险：早期摔倒不一定得到低分

位置：`isal/isal/interaction/tracker.py:182`。

摔倒会立即 finalize；slip/tilt 使用已观察片段，persistence 使用完整窗口作分母。于是刚 touchdown 就终止时，若脚水平速度为零，tilt 相对 touchdown 的变化也为零，两个分量直接贡献 0.65。

本次默认配置复现：

- touchdown 同帧终止：`0.40 + 0.25 + 0.20/12 = 0.6667`；不是 bad（bad 阈值 0.35）。
- 前 12 帧稳定，但 survival 边界时发生终止：`0.40 + 0.25 + 0.20 = 0.85`；会进入 good（阈值 0.70）。

第二种情况即使窗口完整也存在，原因是 survival 只占 0.15。因此 target 是加权的局部交互质量指标，不能解释成“安全落脚概率”。早期窗口则额外存在观测时间短、tilt 极值偏小的偏差。

验收文档已明确选择保留早期失败，原规格也使用这组加权系数；这里不是状态机实现错误。建议保留原始分量与 partial 标志，首先把 partial/full 和 survival=0/1 分开统计。若研究目标要求失败落脚低分，可采用失败上限或 survival gating；若更关注局部交互质量，可使用分量预测并对未充分观测的分量设置有效性掩码。不要不加区分地把当前标量解释为二分类标签。

### 1.4 完成范围：环境输出不会自动保存进训练日志或 PPO buffer

位置：`interaction_env.py:49`、`rsl_rl/rsl_rl/algorithms/ppo.py:144`、`rsl_rl/rsl_rl/utils/logger.py:70`。

环境输出 `extras['auxiliary']` 和 `extras['interaction_stats']`，但当前普通 PPO 没有消费它们；标准 logger 读取 episode/log 字段，不会自动记录这两个新字段。训练时产生的标签不会自动成为可检查的数据集。

目前真正持久化它们的是 `isal/scripts/debug_interactions.py`，仅执行零动作，并使用 InteractionRecorder 导出。因此“采样接口已实现”成立；“正常步态的数据质量、每 rollout 样本量及训练中日志已验证”尚不成立。无辅助训练本身符合本阶段范围。

## 2. 目前的感知任务

继承链：`ISALHumanoidEnv → ISALHumanoidHeightScanEnv → ISALHumanoidInteractionEnv`。

- 物理步长 0.005 s，decimation=4，控制频率 50 Hz。
- RayCaster 挂载 base_link，yaw 对齐，1.6×1.0 m，间距 0.1 m，前移 0.4 m。
- 默认本体 yaw 坐标范围 x∈[-0.4,1.2]、y∈[-0.5,0.5]，17×11=187 条射线。
- 网格规范形状 `(N,1,17,11)`，轴顺序 x、y；Actor 接收其展平结果。

| 观测 | 内容 | 默认形状 |
|---|---|---|
| policy | 角速度、重力投影、命令、23 维关节位置差、23 维关节速度差、23 维上一动作；每帧 78 维、10 帧历史 | `(N,780)` |
| height_scan | 当前帧扫描 | `(N,187)` |
| critic | 本体观测加 root 线速度、双脚接触/力/腾空时间/高度、关节加速度/力矩及 scan；每帧 326 维、10 帧历史 | `(N,3260)` |

Actor 目前是输入 967 维的普通 MLP，隐藏层 `[512,256,128]`，输出 23 维动作；Critic 同样使用 MLP。动作变为默认关节位置加 `0.25*action` 的位置目标。

Stage 3 继承原 rough 奖励、课程与随机化。地形比例为平地 40%、上下楼梯合计 20%、上下坡合计 20%、随机网格 10%、粗糙地面 10%。课程在 reset 时根据走过距离及命令速度调整 terrain level，目前还没有 auxiliary 样本量或标签质量课程。

## 3. 一条 label 如何生成

### 3.1 每个控制步，先读取物理反馈，再自动 reset

父类 step 完成四个物理子步后调用 `_get_dones()`。Stage 3 在这里读取当前 scan、root 状态、左右脚 link 的位置/速度，以及 ContactSensor 的力，更新 tracker，随后才让父环境执行自动 reset。

父类 step 返回后，Stage 3 将本步 packet 发布到 extras。这样摔倒前的标签不会误用 reset 后的新机器人状态。

机器人 articulation 和 contact sensor 的 body 索引分别按脚名解析。已有真实导出分别为 robot `[20,21]`、sensor `[6,12]`，二者不能混用。位置和速度均取 ankle-roll link，避免位置/COM 速度混用。

### 3.2 STANCE → SWING：离地时保存输入

每只脚独立判断 `max(F_world_z,0)>20 N`。上一帧接触、这一帧不接触即 liftoff。

保存当时的 height scan、root xy/yaw、command、body-frame base angular velocity、projected gravity。数据 detach/copy，后续观察变化不会修改旧快照。初始化只建立接触基线，首次出现接触不会伪造一个有快照的落脚。

这是未来模型的预测输入。Stage 3 要求 scan noise/dropout 为零，从而保持 Actor scan 与 snapshot 一致；本体观测仍按原基线加噪声。

### 3.3 SWING → PENDING：落地时确定 query

上一帧不接触、当前帧接触即 touchdown。读取真实脚 link 的世界 xy，并使用离地时保存的 root 位姿：

`query_xy = Rz(-yaw_liftoff) * (foot_xy_touchdown - root_xy_liftoff)`。

query 的原点是 liftoff root，不是前移后的 scan 中心。边界直接从 RayCaster 的实际 ray_starts 求得；检查了本地 RayCaster 实现，ray_starts 已包含 offset，因此无需再次减 0.4 m。

越界、没有 liftoff 快照、非有限数据或没有空槽时丢弃并计数。没有 touchdown 的踩空也不构造标签，这是 MVP 的既定限制。

### 3.4 落地后统计交互结果

每个环境、每只脚拥有一条 live swing 和多个独立 pending 槽。默认 14 槽允许旧记录等待 survival 时，新一轮离地/落地继续采样。

- outcome：`round(0.25/0.02)=12` 帧，touchdown 是第 1 帧。
- survival：`round(0.5/0.02)=25` 步，touchdown age=0，到 age=25 才判为完成。
- 只在前 12 帧累积交互统计；之后等待 survival 时，slip/tilt 不再改变。

| 分量 | 计算 |
|---|---|
| slip_mean | 仅接触帧的脚 link 水平速度模长平均值 |
| tilt_change | 相对 touchdown 的 roll/pitch 变化模长最大值，角度差经过 wrap |
| persistence | 前 12 帧接触帧数 / 12 |
| survival | 等待期内终止为 0，完整存活为 1 |
| peak_force | outcome 内力范数峰值，仅诊断 |

`target = 0.40*exp(-slip_mean/0.15) + 0.25*exp(-tilt_change/0.20) + 0.20*persistence + 0.15*survival`，最后限制到 `[0,1]`。

这里不使用 terrain type、坡度类别或真实摩擦系数来生成 target。标签是未来足地交互给过去观测提供的监督，而不是人工地形类别。

### 3.5 结算、输出与 reset

正常等待期结束或发生 terminated 时 finalize。早期终止附带 `partial_window=True` 与实际观察帧数。timeout 不当作摔倒；未成熟记录丢弃。自动 reset 清掉内部活跃状态，但保留本步刚完成的 packet；显式 reset 清理旧输出。已经交给调用方的 packet 不会被下一步修改。

输出前缀 `(N,2,P)`，分别代表环境、左右脚、pending 槽：

| 字段 | 默认形状 |
|---|---|
| valid | `(N,2,14)` |
| height_scan | `(N,2,14,1,17,11)` |
| query_xy / foot_side | `(N,2,14,2)` |
| command / base_ang_vel / projected_gravity | `(N,2,14,3)` |
| target | `(N,2,14,1)` |

另有各评分分量、partial 标志、观察帧数、peak_force。foot_side 为左右脚 one-hot，无效槽清零。

消费方式为 `batch[key] = packet[key][packet['valid']]`。不能只取每环境一条记录：同帧可能多脚、多条记录同时 finalize。

## 4. 实际数据与验证范围

本次运行 tracker CPU/CUDA 测试和 height-scan math 测试：40 passed。两个 recorder pytest 用例因本机临时目录权限在 fixture 阶段失败，换到 workspace basetemp 仍遇到权限错误；随后在普通工作区目录直接调用这两个原测试函数，均通过。不能将此写成完整 pytest 42 passed。

本次没有重跑完整物理场景集成测试，也没有训练或评估步态。对坐标/生命周期的判断结合了代码、合成测试和仓库已有验收产物。

现有 `outputs/stage3_acceptance_zero_action` 是 1 个环境、200 个控制步的真实零动作运行：liftoff=8、touchdown=10、最终样本=2，其中 partial=1，target 约 0.3302 与 0.1285。另有越界=3、无快照 touchdown=5，overflow=0。它证明采样链路曾跑通，不能证明正常步态下的标签质量。

本次新增复现脚本位于 `outputs/review_20260907/probe.py`，复现评分边界、动态网格镜像、下台阶裁剪，并直接执行 recorder 测试。没有修改原有源代码与测试。

## 5. 后续实现前应明确的事项

1. 先处理高度裁剪和动态 Critic 镜像，再收集正常步态样本。调试脚本目前没有 checkpoint 推理路径。
2. 对完整/部分窗口、survival=0/1 分组统计 target；检查相同几何位置的质量是否具有可学习结构。raw 20 N 单阈值没有迟滞/最小腾空时间，应检查阈值抖动导致的重复落脚比例。
3. query 和 slip 都是 ankle link 的代理量，不是压力中心或实际接触点；脚滚动时 link 水平速度不等于接触点滑动速度。先根据真实步态验证这一代理是否适用。
4. 原基线还有摩擦随机化与随机 push。这不是空间摩擦 patch，也没有把 privileged 信息当标签，但可能使反馈包含当前 head 输入无法解释的因素。应保留公平 baseline，并在标签验证中控制这些变量。
5. 每次 PPO rollout 是 24 步，survival 要 25 步，正常成功样本天然会跨 rollout 边界。未来不能在每次 PPO update 清空未完成 pending；应明确按 finalize 时刻收集样本、保留原始 liftoff 输入，并与主方法的数据归属约定一致。这是下一阶段要处理的接口问题，不是当前 tracker 的 replay buffer。

参数入口：采样阈值/窗口/权重在 `isal/isal/interaction/config.py`；扫描参数在 `terrain_perception_cfg.py`；reward 和任务派生配置在 `isal_env_cfg.py`；地形在 `terrain_generator_cfg.py`；PPO/观测组/镜像在 `agents/isal_agent_cfg.py`，共用网络和历史长度默认值在 `base_config.py`。
