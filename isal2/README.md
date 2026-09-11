# ISAL2：独立 RPO 基础 locomotion

当前实现第一至四阶段：RPO 机器人资源、完整 base 环境、MLP/PPO、AME 高程感知 Actor、Affordance 双路感知与交互监督训练，以及任务列表和训练脚本。ONNX 导出和 MuJoCo 手柄控制尚未实现。

运行时不依赖 `isal`、`robolab`。机器人资源包含在本目录；标准强化学习组件使用外部 `rsl_rl`，只有定制网络、runner 和算法扩展放在 `modified_rsl`。外部依赖包括 Isaac Sim、Isaac Lab、RSL-RL、PyTorch。本机验证环境为 `env_isaaclab`，Python 3.11、Isaac Sim 5.1、PyTorch 2.7.0+cu128、RSL-RL 3.3.0。

## 启动

以下命令从本目录执行，先激活已经配置 Isaac Lab 的 Python 环境：

```powershell
conda activate env_isaaclab
python scripts/list_envs.py
python scripts/train.py --task ISAL2-RPO-Base-v0 --terrain rough --headless --num_envs 4096
python scripts/train.py --task ISAL2-RPO-AME-v0 --terrain rough --headless --num_envs 32
```

脚本通过本地包引导直接运行，不要求 editable 安装，不添加相邻工程到 Python 搜索路径。也可以安装为包：

```powershell
python -m pip install -e . --no-deps --no-build-isolation
python -m isal2.scripts.list_envs
```

依赖应安装到已有的 Isaac Lab 环境。项目声明 `rsl-rl-lib==3.3.0`；本机已将工作区的 `rsl_rl` 作为外部 editable 库安装，代码直接 `import rsl_rl`，不在 `isal2` 中复制其实现，也不硬编码外部源码路径。新环境可安装相同版本，或从已有 RSL-RL 源码目录运行 `python -m pip install -e <RSL_RL_SOURCE> --no-deps`。构建 wheel 时会包含机器人资源，资源哈希保存在 `assets/manifest.json`。

## 参数入口

| 内容 | 修改位置 |
|---|---|
| 机器人初始姿态、PD、延迟、关节限制 | `assets/robots/rpo.py` |
| 基础时间步、命令、噪声、随机化、历史长度 | `tasks/base/base_config.py` |
| RPO 环境、地形选择、观测维度 | `tasks/base/base_env_cfg.py` |
| 地形比例、难度范围、地图尺寸 | `tasks/base/terrain_generator_cfg.py` |
| 场景和传感器 | `tasks/base/scene_cfg.py` |
| 奖励权重 | `tasks/base/rpo_env_cfg.py` |
| 奖励函数、curriculum 决策 | `tasks/base/mdp/` |
| 网络尺寸、PPO 参数、日志配置 | `tasks/base/agents/ppo_cfg.py` |
| AME 高程偏置、噪声覆盖 | `tasks/ame/ame_env_cfg.py` |
| AME CNN、注意力、网络与训练参数 | `tasks/ame/agents/ppo_cfg.py` |

配置的 `configure()` 先应用地形和环境数量，并重建场景、计算观测维度；额外的程序化奖励覆盖放在此调用之后。

常用 CLI 参数：`--terrain {flat,rough,rough_hard}`、`--num_envs`、`--seed`、`--device`、`--headless`、`--max_iterations`、`--run_name`、`--resume CHECKPOINT`。`--max_iterations` 始终表示本次新增更新次数；恢复时不会重复已经完成的 iteration。

`--terrain_rows` / `--terrain_cols` 用于小规模验证；默认保持 10 行、20 列。`--num_envs` 不自动减少地形地图。小显存机器应先用 32 或 128 个环境测试，再逐步提高数量。

## 环境与观测

- 控制周期 20 ms（物理 5 ms、decimation 4），episode 20 s，23 维位置动作，目标为默认关节位置加 `0.25 × action`。
- Actor 单帧为角速度 3、重力投影 3、command 3、关节位置偏差 23、关节速度 23、上一动作 23，共 78 维，5 帧历史合计 390 维。
- Critic 单帧额外包含真实线速度 3、双脚接触 2、接触力 6、腾空时间 2、足部高度 2、关节加速度 23、力矩 23，共 139 维；flat 的 5 帧输入为 695 维。
- rough/rough_hard 的 Critic 每帧再加入 187 点干净高程，总输入 1630 维。Actor 不读取高程；双脚扫描器始终保留用于奖励与特权观测。
- 高程为 yaw 对齐的 1.6 × 1.0 m 网格，0.1 m 分辨率。base 保持 RoboLab base 的高度偏置 0.5；后续 AME 配置可覆盖为其对应设置。
- Critic 的本体观测在 Actor 加噪前构造。首次观测填满历史，reset 后仅重置对应环境；历史顺序为从旧到新。

`flat` 是本地生成的大平面 mesh，不依赖远程地面 USD。`rough` 包含上下楼梯、坡面、随机方块、粗糙地面和平地；`rough_hard` 保留高平台、star、gap、stepping stones 等完整复杂地形。

Curriculum 保持按位移升降级：超过地形长度一半升级，未达到命令速度乘 episode 时长一半则降级；升级优先。首次/手动 reset 不更新等级，站立命令不参与升降级。等级最低为 0，超过最高等级后按 Isaac Lab importer 的实现随机分配有效等级。

基础奖励、随机化和机器人参数从 RoboLab 迁移，复制文件保留版权声明。场景使用本地材质和无纹理光照，避免外部素材下载。原始资源内容保持一致，未复制子模块 `.git` 指针。

## 训练与恢复

`train.py` 按任务注册加载环境和训练配置。未传 `--terrain` 时使用该任务的默认地形；Base 与 AME 默认均为 `rough`。

base 的 Actor/Critic 直接使用 `rsl_rl.modules.ActorCritic`，默认均为 `[512, 256, 128]`、ELU。PPO 默认 24 步 rollout、5 epochs、4 minibatches、初始学习率 `1e-4`、clip `0.2`、gamma `0.994`、lambda `0.9`。辅助损失和对称增强关闭。

时间截断使用 reset 前最终观测的 Critic 值 bootstrap，真实终止不 bootstrap。归一化统计在收集/PPO 更新期间冻结，每轮 PPO 完成后更新一次，避免同一轮 log-probability 比较使用不同归一化状态。

输出位于 `outputs/rpo_base/<run_name>/`，包括完整环境配置、训练参数、命令行、关节顺序、TensorBoard、checkpoint 和 `result.json`。缓存位于 `.cache`；Isaac Sim 某些第三方底层缓存仍可能使用其默认全局位置。

```powershell
python scripts/train.py --headless --num_envs 32 --terrain rough --max_iterations 5 --run_name first
python scripts/train.py --headless --num_envs 32 --terrain rough --max_iterations 2 --resume outputs/rpo_base/first/model_5.pt --run_name resumed
tensorboard --logdir outputs/rpo_base
```

恢复需要使用相同观测布局的配置。checkpoint 保存模型、归一化统计、优化器、当前学习率和已完成 iteration 数；环境会新建 episode，不恢复物理仿真状态，也不承诺逐样本完全重现此前轨迹。

## 验证与扩展

```powershell
python -m unittest discover -s tests -p test_core.py
python scripts/train.py --headless --num_envs 4 --terrain rough_hard --terrain_rows 3 --terrain_cols 20 --smoke_steps 200 --check_reset
```

`modified_rsl` 中的 PPO 和 runner 继承外部对应类，storage、MLP、归一化基础组件与日志工具直接复用外部库。AME 定制网络位于 `modified_rsl/modules`；后续 affordance 网络也放在这里。

环境提供 `_after_physics_step()` 和 `_before_reset(env_ids)` hook。前者在传感器更新后调用；后者保留最终接触、位姿和 done 标记，可供后续 affordance 数据收集使用。`terminal_observation` 和 `time_outs` 通过 adapter 交给 PPO；普通 `get_observations()` 读取缓存，不推进历史。

当前验收结果见 `VALIDATION_EXTERNAL_RSL.md`；`VALIDATION.md` 保留初版历史记录。短程测试验证工程与训练闭环，不代表机器人已经学会复杂地形行走。

## AME 高程感知任务

`ISAL2-RPO-AME-v0` 继承 base 的奖励、命令、随机化、终止条件和 curriculum。flat、rough、rough_hard 均启用扫描器，Actor 读取 `policy`（390 维本体历史）和 `height_scan`（当前 187 点带噪高程），Critic 读取 `critic`（1630 维干净特权历史）。扫描偏置为 0.75，噪声为 ±0.025；其余扫描参数沿用 base 的 `HeightScannerCfg`，修改尺寸/分辨率会同步网络地图形状。

高程以 x 快变、y 慢变的 `xy` 顺序还原成 `[B,1,11,17]`。CNN 三层通道为 `[16,32,32]`，3×3 卷积、步长 1、复制填充和 ELU。特征与米制 `(x,y)` 坐标拼接，再由 1×1 卷积投影成 32 维 token。390 维归一化本体经 `[128]` 隐藏层生成 32 维 Q，对地形 K/V 执行 4 头 cross-attention；输出与本体直连特征拼接，送入 `[512,256,128]` Actor。Critic 使用独立 MLP，不共享 CNN。

没有速度估计器或辅助损失，所有 Actor 模块由 PPO 更新。高程不做在线统计归一化；本体与 Critic 沿用 base 的统计更新时机。局部 reset 仅替换对应环境的高程缓存，终止观测保存独立快照。`TerrainAttention.encode_features()` 和 `attend()` 分开提供，后续可在位置编码前接入 affordance 特征。

```powershell
python scripts/train.py --task ISAL2-RPO-AME-v0 --headless --num_envs 32 --max_iterations 5 --run_name ame_first --terrain_rows 3 --terrain_cols 20
python scripts/train.py --task ISAL2-RPO-AME-v0 --headless --num_envs 32 --max_iterations 2 --resume outputs/rpo_ame/ame_first/model_5.pt --run_name ame_resumed --terrain_rows 3 --terrain_cols 20
python -m unittest discover -s tests -p "test_*.py"
```

AME 输出在 `outputs/rpo_ame/`。恢复要求相同网络、观测布局和地图几何；不支持将 Base checkpoint 直接加载为 AME。第三阶段实际验证见 `VALIDATION_AME.md`。

## Affordance 双路感知任务

`ISAL2-RPO-Affordance-v0` 继承 AME 的观测、Critic、奖励、地形及 curriculum。蓝色 U-Net 仅输入高程，输出单张接触质量图；左右脚样本共同训练。图经 sigmoid、gate 和 resize 后，与 policy CNN 特征及 XY 坐标一起投影成 token。它是当前交互数据下的接触质量估计，并不是与机器人状态无关的确定性通行属性；预测值不进入 reward。

蓝色网络通道为 `[16,32,64]`，两次下采样和带 skip connection 的上采样，无 BatchNorm/dropout。PPO 优化器只包含橙色网络和 Critic，监督优化器只包含 U-Net。顺序严格为 `rollout → PPO → SL → next rollout`；Actor 路径对蓝色输出使用 `no_grad`。前 500 轮输入常数 0.5，随后 1000 轮逐渐启用预测；累计有效样本不足 256 时 gate 保持为 0。蓝色网络不必等 gate 开启才学习。

每个控制步在 done 判断后、reset 前处理接触。接触力进入/离开阈值为 5/2 N，连续 2 步确认；保存首次候选离地帧的高程和 root 位姿，以及首次候选接触帧的脚掌中心。脚掌中心使用 ankle-roll link 局部 `(0.025,0,-0.04)`，按完整姿态变换；再用抬脚时的 root 位置与 yaw 投回旧地图。越界 query 不裁边，不给初始无配对接触生成标签。

观察窗为 `ceil(0.25/0.02)=13` 步；支持跨 rollout 和每脚 4 个重叠 pending。标签为 `0.4*exp(-s/0.2)+0.3*p+0.3*q`，其中 s 是接触帧足部刚体水平速度的均值（滑动代理），p 是窗内接触保持率，q 是足部扫描点位于名义脚底高度 ±0.02 m 内的比例均值（几何支撑代理）。无效射线不算支撑。真实终止覆盖本步标签为 0；时间截断和手动 reset 丢弃未完整观察的记录，已完成样本保留。

近期样本缓冲区默认容量 65,536，保留最近 32 轮，存储原始高程、落脚 query、软标签、左右脚、指标和抬脚采集轮数。达到 64 条样本后，每轮 PPO 后进行 8 次监督更新，batch 256、Adam `1e-4`、梯度裁剪 1；在落脚中心用双线性 `grid_sample(align_corners=True)` 采样概率，计算软标签 BCE，不为未踩区域补标签。

| 内容 | 参数入口 |
|---|---|
| 接触阈值、观察窗、脚掌偏移、标签权重、pending 数量 | `tasks/affordance/collection.py` 的 `ContactCollectionCfg`，由 `affordance_env_cfg.py` 持有 |
| U-Net 通道、监督更新、replay 容量/时限、gate | `tasks/affordance/agents/ppo_cfg.py` |
| 两阶段 runner 与恢复 | `modified_rsl/runners/affordance_runner.py` |

```powershell
python scripts/train.py --task ISAL2-RPO-Affordance-v0 --headless --num_envs 32 --max_iterations 5 --run_name aff_first --terrain_rows 3 --terrain_cols 20
python scripts/train.py --task ISAL2-RPO-Affordance-v0 --headless --num_envs 32 --max_iterations 2 --resume outputs/rpo_affordance/aff_first/model_5.pt --run_name aff_resume --terrain_rows 3 --terrain_cols 20
```

短程采集不一定立即达到 64 条，需依据 `affordance_updates` 判断是否实际训练了蓝色网络；可继续恢复训练以累积真实样本。验收时可以用 `--affordance_warmup 0 --affordance_ramp 2` 缩短 gate 时长，正式默认不变。不同 gate/replay/监督配置的 checkpoint 不允许直接恢复。

输出位于 `outputs/rpo_affordance/`，包含配置、TensorBoard、结果和 checkpoint。checkpoint 保存两个网络、两个优化器、归一化、iteration、gate 计数及成熟样本缓冲区。恢复后重新开始物理 episode，pending 清空；不支持 AME/Base 权重迁移。记录监督前后固定观测的动作 KL、质量图变化、gate、有效样本数量及各类接触/丢弃计数。当前仅支持单进程训练。

第四阶段验收记录见 `VALIDATION_AFFORDANCE.md`；长程收敛、最终地形通过率和部署仍需后续评估。
