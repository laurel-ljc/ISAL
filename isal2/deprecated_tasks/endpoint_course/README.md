# 已归档的终点课程

此目录使用 `ISAL2-RPO-{AME|Affordance}-Endpoint-Stage{1|2}-v0`；以下为归档前的说明，原 active ID 已由新的参考课程接管。

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

