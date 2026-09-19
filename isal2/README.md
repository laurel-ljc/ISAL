# ISAL2：RPO AME / Affordance 两阶段地形任务

当前默认任务为 `ISAL2-RPO-AME-Stage1-v0`。AME 和 Affordance 各有两个阶段，共四个 active 任务。旧任务已完整迁移到 `deprecated_tasks`，仍可用原来的任务 ID 显式启动；`scripts/list_envs.py` 会标出 active/deprecated。

`tasks` 顶层为 `ame_stage1`、`ame_stage2`、`affordance_stage1`、`affordance_stage2` 和 `common`。`common/base` 保存公共环境和 MDP，`common/ame` 保存 AME 感知与策略配置，`common/affordance` 保存接触采集和 Affordance 配置，`common/course` 保存地形、命令和课程；公共模块不注册任务。旧任务完整实现只保留在 `deprecated_tasks`。

## 新任务训练与恢复

以下命令在 `isal2` 目录、已配置的 `env_isaaclab` 环境中执行：

```powershell
python scripts/list_envs.py
python scripts/train.py --task ISAL2-RPO-AME-Stage1-v0 --headless --num_envs 4096 --run_name stage1
python scripts/train.py --task ISAL2-RPO-AME-Stage2-v0 --headless --num_envs 4096 --warm-start outputs/rpo_ame_stage1/stage1/model_12001.pt --run_name stage2
python scripts/train.py --task ISAL2-RPO-AME-Stage2-v0 --headless --num_envs 4096 --resume outputs/rpo_ame_stage2/stage2/model_12001.pt --run_name stage2_resume
```

Stage2 允许不传 `--warm-start` 从零训练。warm-start 沿用既有语义：严格加载模型和观测归一化统计，动作标准差重置为 0.30，优化器、迭代数及课程重新开始。跨阶段使用 warm-start；resume 只用于相同阶段，要求环境数量、地形列数、seed 与课程配置一致。恢复课程等级和随机数状态后重新开启回合，不恢复物理现场。

AME 阶段任务保留旧 AME 的网络结构和 PPO 参数；四个阶段任务均保留 23 维动作、390 维本体历史、1630 维 Critic、187 点高程，奖励权重与旧 AME 默认 rough 一致。两个阶段 actor/critic 均读取干净的当前高程图：yaw 对齐、1.6×1.0 m、分辨率 0.1 m、11×17；本体观测噪声和动力学随机化仍保留。

## 地形与十级课程

| 阶段 | 地形 |
|---|---|
| Stage1 | stairs、pits、rough、pallets、gaps、grid stones、beams |
| Stage2 | pentagon stones、single-column stones、narrow pallets、consecutive gaps、narrow stairs |

地块为 8×8 m，障碍区约 4 m；起点 x=-2.7 m，终点 x=2.7 m，两端都有平坦安全平台。pits 为可走入走出的浅坑；narrow pallets 缩窄横向宽度，narrow stairs 缩短前后踏面。五边形为真实正五边形支撑。所有悬空间隙下方为 -1 m 坑底，不是隐藏的平路；浅坑的坑底则是合法落脚区域。

每个环境分别记录每种地形的等级，初始全为第1级（内部 0）。每回合等概率随机地形类型，再从该类型的对应等级地块出生。自动结束时，成功升一级，失败或超时降一级，范围固定为1–10级；手动 reset 不计成绩。没有验证解锁、连续成功门槛或课程回放。

命令每步指向终点，巡航速度每回合采样 0.3–0.6 m/s，接近终点减速。根位置进入终点 0.3 m 范围、至少一脚支撑在终点平台并持续 0.2 s 时成功。失败条件包括旧终止条件、无效支撑、跌落和离开路线；成功与失败同帧时失败优先。成功是无 bootstrap 的真实终止，但不施加 -200 失败惩罚。20 s 超时降级，同时保留终止前 Critic bootstrap。

Stage1 无 base 推扰；Stage2 每5–8秒给根水平速度叠加各轴 ±0.10 m/s 的增量，不修改竖直或角速度。两阶段保持其他随机化一致。

默认 Stage1 为10行×14列，Stage2 为10行×10列，每种类型各两列。行数固定为10；`--terrain_cols` 必须分别为7或5的正整数倍。任务限定自己的 stage 地形，不接受 `--terrain rough` 等旧预设。几何参数位于 `tasks/common/course/geometry.py`，任务配置、命令和生命周期逻辑位于同目录，其余 MDP 沿用 `tasks/common/base/mdp`。任务入口分别为 `tasks/ame_stage1`、`tasks/ame_stage2`。

踏石最小有效宽度0.30 m。间隙设计依据0.8 m前视范围，预留0.2 m接近边缘距离及0.2 m对岸落脚可见区域；几何测试还覆盖实际路径间隙和多个扫描网格相位。参数是保守的初始设计，测试不代表已经训练收敛。

## 新任务验证与地形预览

```powershell
python -m unittest discover -s tests -p "test_*.py"
python scripts/train.py --task ISAL2-RPO-AME-Stage1-v0 --headless --num_envs 8 --terrain_cols 7 --smoke_steps 80 --check_reset
python scripts/train.py --task ISAL2-RPO-AME-Stage2-v0 --headless --num_envs 8 --terrain_cols 5 --smoke_steps 450 --check_reset
python scripts/render_course_catalog.py --output outputs/course_catalog
```

目录图包含全部120个地形的俯视图、代表等级侧面高度图和 `terrain_parameters.json`。每次训练同时保存实际地图的 `terrain_atlas.json`，包含随机种子、支撑多边形、起终点和几何参数。TensorBoard 的 `Course/<type>/` 记录等级、成功/失败/超时、失败原因和回合/成功完成时间。

验收结果见 [VALIDATION_AME_STAGES.md](VALIDATION_AME_STAGES.md)：85项测试通过，两个阶段均完成短程PPO、恢复训练及仿真生命周期检查。

## Affordance 两阶段训练

`ISAL2-RPO-Affordance-Stage1-v0` 和 `ISAL2-RPO-Affordance-Stage2-v0` 复用对应 AME 阶段的地形、十级课程、机器人、奖励和干净高程图。Stage1 无推扰，Stage2 每5–8秒叠加各水平轴 ±0.10 m/s 的根速度扰动。策略保持旧 Affordance 的 U-Net 与注意力融合结构。

| 设置 | Affordance Stage1 | Affordance Stage2 |
|---|---|---|
| 预测接入策略 | 等待500轮，再用1000轮逐渐启用 | 首次推理即完整启用 |
| 预测接入样本门槛 | 累计有效样本至少256条 | 无 |
| 初始 alpha | 0，策略接收常数质量图0.5 | 1，策略接收完整预测 |
| 监督训练门槛 | 回放中至少64条有效样本 | 同左 |

这里的“轮”是 PPO iteration，不是 episode 或控制步。Stage1 的 `alpha=clamp((iteration-500)/1000,0,1)`，累计有效样本不足256条时保持0。**warm-up 不阻止 U-Net 学习**：两阶段都在有效样本达到64条后，每轮PPO结束执行8次监督更新，batch为256、Adam学习率1e-4。回放容量65,536，保留最近32轮的样本。PPO与U-Net优化器隔离，更新顺序为 rollout → PPO → 监督训练。

成功到达和超时都保留已完成0.25秒评价窗的样本、丢弃未完成样本，不把成功标成失败；真实失败仍强制将对应待评价接触标为0。手动和局部reset清理相应环境未完成的接触记录。

```powershell
python scripts/train.py --task ISAL2-RPO-Affordance-Stage1-v0 --headless --num_envs 4096 --run_name aff_stage1
python scripts/train.py --task ISAL2-RPO-Affordance-Stage2-v0 --headless --num_envs 4096 --warm-start outputs/rpo_affordance_stage1/aff_stage1/model_12001.pt --run_name aff_stage2
python scripts/train.py --task ISAL2-RPO-Affordance-Stage2-v0 --headless --num_envs 4096 --resume outputs/rpo_affordance_stage2/aff_stage2/model_12001.pt --run_name aff_stage2_resume
python scripts/train.py --task ISAL2-RPO-Affordance-Stage1-v0 --headless --num_envs 8 --terrain_cols 7 --smoke_steps 150 --check_reset
python scripts/train.py --task ISAL2-RPO-Affordance-Stage2-v0 --headless --num_envs 8 --terrain_cols 5 --smoke_steps 150 --check_reset
python scripts/export_onnx.py --checkpoint outputs/rpo_affordance_stage2/aff_stage2/model_12001.pt
```

Stage2也允许从零训练。warm-start必须加载Affordance模型：保留模型权重与归一化统计，清空两组优化器、回放、课程及迭代数，将动作标准差设为0.30；目标Stage1重新预热，目标Stage2的alpha立即设为1。AME→Affordance权重转换不受支持。

同阶段resume要求环境数、seed、地形列数、课程及辅助配置一致，恢复两组优化器、回放、采集统计、课程和随机数状态；Stage1继续原调度，Stage2维持alpha=1。Stage1可用 `--affordance_warmup`（非负）、`--affordance_ramp`（正数）覆盖调度；Stage2只接受这两个参数为0。导出保留实际alpha，包含Stage1预热或渐进状态，以及Stage2完整启用状态。

输出分别在 `outputs/rpo_affordance_stage1` 和 `outputs/rpo_affordance_stage2`。验收记录见 [VALIDATION_AFFORDANCE_STAGES.md](VALIDATION_AFFORDANCE_STAGES.md)。

## Deprecated 任务历史说明

以下内容保留旧任务说明；其中 `tasks/` 路径现在均对应 `deprecated_tasks/`，旧 ID 行为保持不变。

当前实现第一至五阶段：RPO 机器人资源、完整 base 环境、MLP/PPO、AME 高程感知 Actor、Affordance 双路感知与交互监督训练，以及任务列表、训练、ONNX 导出和 MuJoCo 手柄控制脚本。

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
| 基础时间步、命令、噪声、随机化、历史长度 | `deprecated_tasks/base/base_config.py` |
| RPO 环境、地形选择、观测维度 | `deprecated_tasks/base/base_env_cfg.py` |
| 地形比例、难度范围、地图尺寸 | `deprecated_tasks/base/terrain_generator_cfg.py` |
| 场景和传感器 | `deprecated_tasks/base/scene_cfg.py` |
| 奖励权重 | `deprecated_tasks/base/rpo_env_cfg.py` |
| 奖励函数、curriculum 决策 | `deprecated_tasks/base/mdp/` |
| 网络尺寸、PPO 参数、日志配置 | `deprecated_tasks/base/agents/ppo_cfg.py` |
| AME 高程偏置、噪声覆盖 | `deprecated_tasks/ame/ame_env_cfg.py` |
| AME CNN、注意力、网络与训练参数 | `deprecated_tasks/ame/agents/ppo_cfg.py` |

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
| 接触阈值、观察窗、脚掌偏移、标签权重、pending 数量 | `deprecated_tasks/affordance/collection.py` 的 `ContactCollectionCfg`，由 `affordance_env_cfg.py` 持有 |
| U-Net 通道、监督更新、replay 容量/时限、gate | `deprecated_tasks/affordance/agents/ppo_cfg.py` |
| 两阶段 runner 与恢复 | `modified_rsl/runners/affordance_runner.py` |

```powershell
python scripts/train.py --task ISAL2-RPO-Affordance-v0 --headless --num_envs 32 --max_iterations 5 --run_name aff_first --terrain_rows 3 --terrain_cols 20
python scripts/train.py --task ISAL2-RPO-Affordance-v0 --headless --num_envs 32 --max_iterations 2 --resume outputs/rpo_affordance/aff_first/model_5.pt --run_name aff_resume --terrain_rows 3 --terrain_cols 20
```

短程采集不一定立即达到 64 条，需依据 `affordance_updates` 判断是否实际训练了蓝色网络；可继续恢复训练以累积真实样本。验收时可以用 `--affordance_warmup 0 --affordance_ramp 2` 缩短 gate 时长，正式默认不变。不同 gate/replay/监督配置的 checkpoint 不允许直接恢复。

输出位于 `outputs/rpo_affordance/`，包含配置、TensorBoard、结果和 checkpoint。checkpoint 保存两个网络、两个优化器、归一化、iteration、gate 计数及成熟样本缓冲区。恢复后重新开始物理 episode，pending 清空；不支持 AME/Base 权重迁移。记录监督前后固定观测的动作 KL、质量图变化、gate、有效样本数量及各类接触/丢弃计数。当前仅支持单进程训练。

第四阶段验收记录见 `VALIDATION_AFFORDANCE.md`；长程收敛和最终地形通过率仍需后续评估。

## 稀疏地形任务

新增 `ISAL2-RPO-AME-Sparse-v0` / `ISAL2-RPO-Affordance-Sparse-v0`。
二者共享 `deprecated_tasks/sparse/` 的几何、命令、结果和课程，分别使用原 AME / Affordance 网络及训练器。
23 维动作、390 维本体历史、187 点当前扫描、1630 维 critic 保持不变。
Affordance 继续接触自监督和双优化器训练；不能加载 AME checkpoint。

| 参数/入口 | 含义 |
|---|---|
| `--terrain sparse` | 默认 10 档、20 列稀疏混合；列数须为 20 的倍数 |
| `--terrain sparse_rescue` | 固定 80 cm 梁宽，1/2/3 m 长度补救课程，默认 3 行 |
| `--phase acquire` | 默认理想感知、名义动力学、无推扰、执行器延迟固定 0 |
| `--phase robust --robust-step STEP` | 按固定阶段逐项增加噪声/随机化 |
| `--command-stage C0\|C1\|C2` | 对齐通过、较大初始偏差、侧身/平台转向；后两者保留 20% C0 |
| `--warm-start FILE` | 严格加载同类网络与归一化，重建优化器、计数和课程，std=0.30 |
| `--resume FILE` | 恢复同阶段训练状态；环境数量、seed、地形行列等须保持一致 |
| `--advance-from FILE --validation-report FILE` | 校验匹配模型的验证报告，再迁入下一阶段 |
| `--skip-evaluation` | 仅供短程调试；不评估、不解锁几何难度 |

Affordance 热启动保留 U-Net 权重，清空旧 replay 和辅助计数，门控重新执行 500 轮预热＋1000 轮渐增。
阶段迁移保留两个优化器、归一化、探索与累计门控进度，清空跨阶段 replay/pending。
resume 恢复课程、验证历史及 RNG，重新开始 episode，不恢复瞬时物理状态。

几何由 mesh 和路线元数据共同生成。连续两次成功/失败升降一级，能力上限由各类型固定验证独立解锁；
默认梁类课程上限为等级 6（40 cm）；35/30/25 cm 仍进入固定评估。完成可达性验证后可提高 `SparseCfg.target_level`，进入后续扩展课程。
20% 容易档回放不修改能力档连续计数。成功需穿过出口并有效支撑 0.3 s；摆动脚不按无支撑失败处理。
停滞仅记录；成功/超时截断使用 terminal observation bootstrap，身体碰撞、坑底承载和绕路为真实终止。

训练在初始更新与每 250 次阶段更新时，在独立进程评估固定场景并冻结归一化。
核心 384 场景覆盖单梁/新放射梁/原星形梁，另加课程档位与复习场景。
每组至少 32 次尝试，完整记录失败样本，并输出 Wilson 95% 区间。
robust 的 clean 报告表示**关闭感知噪声/漂移的对照，动力学保留阶段设置**。

进入 robust/clean 前，40 cm 单梁、新放射梁、原星形梁 40 cm 档需连续两次通过率 ≥85%、跌落率 <10%；
平地速度跟踪误差相对热启动基线恶化不超过 10%，跌落率 <10%。
后续 robust 顺序为 `clean → height_005 → height_010 → height_025 → proprio_25 → proprio_50 → proprio_100
→ mass → com → material → gains → delay → push_25 → push_50 → push_100 → drift_01 → drift_02 → topology`。
噪声逐档累计。地图漂移每回合固定，仅影响 actor，Affordance 的 query 使用实际地图采样原点；critic 与足部扫描保持干净。
相对阶段入口下降超过 10 个百分点时停止阶段晋级，将容易档回放提高至 40%。

```powershell
# 32 环境短程工程检查，不用于能力结论
python scripts/train.py --task ISAL2-RPO-AME-Sparse-v0 --headless --num_envs 32 --terrain_rows 3 --smoke_steps 200 --check_reset
python scripts/train.py --task ISAL2-RPO-Affordance-Sparse-v0 --headless --num_envs 32 --terrain_rows 3 --max_iterations 10 --skip-evaluation --warm-start outputs_download/rpo_affordance/ISAL2-RPO-Affordance-v0-Rough/model_9001.pt

# 正式 A/B 初筛：相同 seed、更新数与采样 std；以下命令不由文档自动执行
python scripts/train.py --task ISAL2-RPO-AME-Sparse-v0 --headless --num_envs 1024 --seed 42 --max_iterations 500 --warm-start outputs_download/rpo_ame/ISAL2-RPO-AME-v0-Rough/model_9001.pt --run_name sparse_A
python scripts/train.py --task ISAL2-RPO-AME-Sparse-v0 --headless --num_envs 1024 --seed 42 --max_iterations 500 --warm-start outputs_download/rpo_ame/ISAL2-RPO-AME-v0-Rough-Hard/model_18002.pt --run_name sparse_B

# 独立评估；输出 stage/episodes.jsonl、groups.csv、summary.json 和场景清单
python scripts/evaluate_sparse.py --task ISAL2-RPO-AME-Sparse-v0 --checkpoint outputs/rpo_ame_sparse/sparse_A/model_500.pt --output outputs/evaluation/sparse_A_500 --headless

# 仅在已达标时接受；checkpoint 与报告须匹配
python scripts/train.py --task ISAL2-RPO-AME-Sparse-v0 --phase robust --robust-step clean --advance-from outputs/rpo_ame_sparse/sparse_A/model_500.pt --validation-report outputs/evaluation/sparse_A_500/stage/summary.json --headless --num_envs 1024

python scripts/render_terrain_catalog.py --sparse --output outputs/sparse_catalog --headless --enable_cameras
# RTX 无法输出画面时，渲染同一份碰撞三角网格（无需启动仿真）
python scripts/render_terrain_catalog.py --sparse --software --output outputs/sparse_catalog_software
python -m unittest discover -s tests -p 'test_sparse*.py'
```

输出分别位于 `outputs/rpo_ame_sparse/` 与 `outputs/rpo_affordance_sparse/`。
CPU 射线验证和软件图册使用 `rtree`、`matplotlib`（`validation` 可选依赖组）；已有 Isaac Lab 验证环境已具备。
结构和最初设计见 `AME_SPARSE_CURRICULUM_PLAN.md`，实际验收见 `VALIDATION_SPARSE.md`。

## ONNX 导出

导出支持三个任务，读取可信的本地训练 checkpoint；不会启动 Isaac Sim。默认读取 checkpoint 同目录的 `env.yaml`、`joint_names.json`，网络配置来自 checkpoint 的 `train_cfg`。移动文件后可用 `--env-config`、`--joint-names`、`--agent-config` 显式指定。缺失配置或关节映射不完整会报错。

```powershell
# 以下命令均在 isal2 目录下执行；换成自己的训练 checkpoint。
python scripts/export_onnx.py --checkpoint outputs/rpo_ame/ame_train_acceptance/model_5.pt
python scripts/export_onnx.py --checkpoint outputs/rpo_affordance/aff_gate_acceptance/model_10.pt
python scripts/export_onnx.py --checkpoint outputs/rpo_base/external_rsl_train/model_5.pt
```

默认输出到 checkpoint 同目录的 `export/`，可用 `--output <目录>` 修改。`model.onnx` 与 `deployment.json` 必须一起保留，运行端检查 ONNX 哈希；JSON 包含观测定义、关节顺序、默认姿态、逐关节 PD/力矩限制、动作缩放、控制周期和高程几何。历史在运行端维护，Actor 归一化包含在 ONNX 内，不重复归一化。

| 模型 | 输入 | 输出 |
|---|---|---|
| Base | `policy [B,390]` | `actions [B,23]` |
| AME / Affordance | `policy [B,390]`、`height_scan [B,187]` | `actions [B,23]` |

导出使用 float32、opset 17 和动态 batch，输出确定性动作均值。Critic、优化器、replay 不进入图。Affordance 保留 checkpoint 的 U-Net、归一化及 α；早期 checkpoint 的 α=0 是正常结果，不会在导出时自动开启预测图。每次导出自动比较原 Actor、标准运算包装器与 ONNX Runtime 的 batch 1/4/32 输出，误差记录在 JSON 中。

导出需要 PyTorch、RSL-RL、PyYAML、onnx、ONNX Runtime；MuJoCo 运行只加载 NumPy、MuJoCo、ONNX Runtime 及本项目部署模块。本机已有 `onnx==1.20.1`、`onnxruntime-gpu==1.27.0`、`mujoco==3.3.3`，无需安装 pygame，也无需替换已有 ONNX Runtime。新环境可用 `pip install -e ".[deployment]"` 安装项目附加依赖，再根据环境选择安装 `onnxruntime` 或 `onnxruntime-gpu`，两者只装一个。独立运行机器也可直接保留项目目录并仅安装 `numpy mujoco onnxruntime`，通过脚本启动，无需安装训练依赖。

## MuJoCo 与 Windows 手柄

```powershell
# 可视化混合地形；初始暂停，连接 Xbox 兼容手柄后按 Start。
python scripts/sim2sim.py --model outputs/rpo_affordance/aff_gate_acceptance/export/model.onnx

# 无界面验收，实际出生在 rough 的粗糙地面区域。
python scripts/sim2sim.py --model outputs/rpo_ame/ame_train_acceptance/export/model.onnx --terrain rough --spawn-tile 1 --headless --duration 4 --command 0.3 0 0

# 难地形踏石区域；命令范围来自模型的部署配置。
python scripts/sim2sim.py --model outputs/rpo_affordance/aff_gate_acceptance/export/model.onnx --terrain rough_hard --spawn-tile 7 --difficulty 0.7 --seed 42 --headless --duration 4 --command 0.3 0 0
```

默认 CPUExecutionProvider，物理步长 1 ms，策略周期 20 ms，PD 在每个物理步执行。关节按名称映射，目标为 `default_pos + action_scale * clipped_action`，力矩按训练配置逐关节限制。默认无观测噪声、零执行器延迟；保留资源中的名义接触参数。JSON 中保存速度上限供检查，当前不额外硬裁剪 MuJoCo 关节速度，也不模拟 Isaac Lab 随机化。MuJoCo 与 PhysX 的接触动力学并不相同。

本体输入与训练一致，5 帧按旧到新排列，初始帧填满历史。当前高程不堆叠历史、不做在线归一化，187 点按 x 列/y 行恢复为 11×17；扫描随 base yaw 旋转，排除所有机器人几何体。重置会清空上一动作和历史。交互模式下检测到倾覆（机体竖直轴与世界竖直轴点积 <0.35）后暂停；按 Y 重置后再按 Start。无界面模式会自动重置并继续计数。这个恢复判据独立于训练终止条件。

| 输入 | 功能 |
|---|---|
| 左摇杆前后 / 左右 | 前进速度 / 横移速度 |
| LT / RT 扳机 | 左转 / 右转；按压深度控制角速度，两侧输入相减 |
| 右摇杆左右 / 上下 | 镜头左右环绕 / 调整俯仰；松开后保持当前视角偏移 |
| RB | 平滑回到机器人后方的默认视角；拨动右摇杆可中断回正 |
| A | 当前命令清零；之后仍由摇杆决定命令 |
| Y / 键盘 R | 重置机器人与镜头并暂停 |
| 键盘 N / P | 跳到下一处 / 上一处地形入口，重置机器人与镜头并暂停；Start 继续 |
| Start / 空格 | 暂停或继续；继续需要手柄处于连接状态 |
| Back / Esc | 退出 |

`--controller-index` 选择 XInput 0–3，默认 0。摇杆死区默认 0.15，扳机死区默认 0.05，断连会清零并暂停，重连后需重新按 Start。`--controller-config` 接受 JSON，例如 `{"deadzone":0.2,"trigger_deadzone":0.08,"sensitivity":[0.7,0.7,0.5]}`。默认 axes 为 `["ly","lx","triggers"]`，signs 为 `[1,-1,1]`；triggers 表示 LT 减 RT。完整默认值位于 `deployment/controller.py` 的 `DEFAULT_CONTROLLER`；可覆盖 axes、signs、sensitivity 和按钮位掩码。旧配置若显式将第三个 axes 设为 rx，需删除该覆盖或改成 triggers 并将第三个 signs 改成 1。只实现 Windows XInput，尚未完成实物手柄验收；不支持 Linux 手柄。

默认镜头距机器人 4 m，俯视角 -20°，位于机器人后方，yaw 跟随机器人前方。右摇杆控制相对于该朝向的环绕偏移，松开后保持该偏移；按一下 RB 后以 0.8 s 时间常数平滑回正（约 2.4 s 恢复 95%），无需一直按住。回正期间拨动右摇杆会中断回正，松开后保持新角度。暂停时也可调整镜头。`--camera-config` 接受 JSON，例如 `{"distance":3.5,"elevation":-25,"yaw_speed":100,"pitch_speed":50,"return_time":1.2}`。角度与角速度单位为度、度/秒，return_time 单位为秒。完整默认值在 `deployment/camera.py`；可配置镜头死区、左右环绕范围和俯仰范围。

地形参数集中在 `deployment/terrain.py` 的 `DEFAULT_TERRAIN`，`--terrain-config` 接受 JSON 覆盖，例如 `{"step_width":0.35,"pit_depth":1.5}`。可选 flat、rough、rough_hard、mixed；默认 mixed，seed=42、difficulty=0.5。difficulty 对参数范围进行线性插值。

`mixed` 是默认综合障碍场，现为 **4×4 个 8×8 米区域**，总面积 32×32 m；各区域颜色不同，可沿平坦边界通行。编号按 `x` 方向递增，每四块沿 `+y` 换行：

| 第一行（y=0） | 第二行（y=8） | 第三行（y=16） | 第四行（y=24） |
|---|---|---|---|
| 0 平地热身区 | 4 金字塔台阶 | 8 梅花桩 | 12 连续沟壑 |
| 1 随机粗糙地面 | 5 下凹台阶 | 9 独木桥 | 13 交错踏台 |
| 2 金字塔斜坡 | 6 随机方块 | 10 折线桥 | 14 连续波浪坡 |
| 3 下凹斜坡 | 7 踏石与沟槽 | 11 连续矮栏 | 15 上坡—平台—下坡 |

梅花桩使用静态圆柱，桥面、踏台与矮栏使用静态 box 碰撞体。桥和桩下方是默认 1 m 深的真实坑底，间隙没有同高度隐藏地板；高程扫描能同时看到顶面和坑底。波浪坡、上坡平台等使用 heightfield，原有 0–7 区的形状与编号保留。布局参考本仓库 `robolab/scripts/mujoco/sim2sim_rpo_parkour.py` 所加载的 MJCF 障碍场，部署不需要导入 robolab，也没有新增依赖。

默认出生在 0；`--spawn-tile 8` 可从梅花桩入口开始，`--spawn-tile 9` 可从独木桥入口开始。8–15 区从朝向 +X 的平坦入口出生，避免直接生成在狭窄桥面或坑中；0–7 区沿用原来的中心出生点。交互模式按 N/P 切换区域，控制台显示区域编号和名称；切换后暂停，按 Start 继续。Y/R 会回到当前区域的出生点。

```powershell
# 综合场，梅花桩入口（保留原来的手柄和镜头控制）
python scripts/sim2sim.py --model outputs_download/rpo_ame/ISAL2-RPO-AME-v0-Rough/export/model.onnx --terrain mixed --spawn-tile 8
```

`difficulty` 越高，桩径和桥宽越小、间隙和障碍高度越大。例如默认独木桥宽由 0.55 m 缩至 0.25 m，桩半径由 0.27 m 缩至 0.19 m；`difficulty=0.5` 时桥宽 0.40 m、桩半径 0.23 m。新增参数包括 `pillar_radius_range`、`pillar_gap_range`、`pillar_height_range`、`beam_width_range`、`zigzag_width_range`、`hurdle_height_range`、`block_height_range`、`wave_height_range`、`ramp_height_range`，通过 `--terrain-config` JSON 覆盖。所有长度单位为米。

`flat`、`rough`、`rough_hard` 保留用于旧实验复现。原来后三个预设看起来相近，是因为它们共用八宫格，仅最后区域和部分难度参数不同；现在选 `mixed` 即可在一个场景里测试全部 16 类障碍。场景可运行不代表当前策略已经能够通过全部障碍。

输出默认在 `outputs/sim2sim/<时间>/`：`result.json` 保存状态、当前出生区域 active_tile、跌倒/reset、推理耗时、依赖隔离检查及实际参数；`telemetry.json` 保存每步命令、动作、力矩、饱和计数和位置。`scene/terrain.json` 保存地形配置、布局、标签、实际出生地面位置 spawn_position，以及实体障碍的几何描述；`scene/scene.mjb` 包含实际高度数据。`scene.xml` 是用于构建的模板，单独加载它不包含运行时填入的 heightfield 数据。原始机器人 MJCF、mesh 和资源 manifest 不改动。

第五阶段实际命令与验收结果见 `VALIDATION_DEPLOYMENT.md`。短程验收 checkpoint 只适合检查部署链路；实际 locomotion 效果需使用完成训练的模型评估。
