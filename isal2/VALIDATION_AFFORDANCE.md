# 第四阶段 Affordance 验证（2026-09-11）

实现 `ISAL2-RPO-Affordance-v0`：独立 U-Net、AME 特征融合、每步接触配对、近期样本 replay、PPO 后监督更新和完整恢复。沿用 AME 奖励、Critic、地形及 curriculum。没有加入预测值奖励、AME 权重迁移、ONNX 或 MuJoCo。

## 验证环境和命令

本机 `env_isaaclab`，Python 3.11、Isaac Sim 5.1、PyTorch 2.7、外部 RSL-RL 3.3.0，RTX 3060，`cuda:0`。从 `C:/Users/Admin/Documents/ISAL` 执行，以下 `python` 均指 `C:/Users/Admin/miniconda3/envs/env_isaaclab/python.exe`。

```powershell
python -m unittest discover -s isal2/tests -p "test_*.py"
python -I isal2/scripts/list_envs.py
python -I -u isal2/scripts/train.py --task ISAL2-RPO-Affordance-v0 --terrain flat --headless --num_envs 4 --smoke_steps 200 --check_reset --run_name aff_flat_acceptance
python -I -u isal2/scripts/train.py --task ISAL2-RPO-Affordance-v0 --terrain rough --headless --num_envs 4 --smoke_steps 200 --check_reset --run_name aff_rough_acceptance --terrain_rows 3 --terrain_cols 20
python -I -u isal2/scripts/train.py --task ISAL2-RPO-Affordance-v0 --terrain rough_hard --headless --num_envs 4 --smoke_steps 200 --check_reset --run_name aff_hard_acceptance --terrain_rows 3 --terrain_cols 20
python -I -u isal2/scripts/train.py --task ISAL2-RPO-Affordance-v0 --headless --num_envs 32 --max_iterations 5 --run_name aff_train_acceptance --terrain_rows 3 --terrain_cols 20
python -I -u isal2/scripts/train.py --task ISAL2-RPO-Affordance-v0 --headless --num_envs 32 --max_iterations 2 --resume isal2/outputs/rpo_affordance/aff_train_acceptance/model_5.pt --run_name aff_resume_acceptance --terrain_rows 3 --terrain_cols 20
python -I -u isal2/scripts/train.py --task ISAL2-RPO-Affordance-v0 --headless --num_envs 32 --max_iterations 10 --affordance_warmup 0 --affordance_ramp 2 --run_name aff_gate_acceptance --terrain_rows 3 --terrain_cols 20
```

## CPU 与环境结果

27 项 CPU 测试通过，包含此前 Base/AME 的 14 项回归。新增测试覆盖接触抖动、独立足部配对、首次候选帧保存、旋转/平移与边界、跨 rollout、重叠和溢出、摆动超时、跌倒覆盖完成窗、时间截断与手动 reset、标签公式、replay 容量/过期/恢复、奇数尺寸 U-Net 和双线性采样、PPO/监督参数隔离、α=1 时动作概率重算一致、空监督更新、默认 gate 时间表及双阶段 checkpoint 恢复。补充修复了采集时替换计数张量可能导致 inference-mode 张量在手动 reset 中无法修改的问题，测试验证跨模式 reset 正常。

三类地形均完成 4 环境 × 200 控制步。观测维度为 policy 390、height_scan 187、critic 1630；观测、动作、奖励、标签均有限。地图噪声隔离、射线 XY 顺序、局部 reset、终止快照和 pending 清理检查通过。

| 地形 | Reset 次数 | 抬脚 | 落脚 | 成熟样本 | 越界丢弃 | 无抬脚配对 | 摆动超时 | 失败标签 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flat | 12 | 19 | 40 | 5 | 2 | 33 | 1 | 2 |
| rough | 12 | 18 | 40 | 2 | 4 | 34 | 1 | 0 |
| rough_hard | 12 | 18 | 40 | 4 | 2 | 34 | 0 | 2 |

三次 smoke 中 overflow、未完成 pending 的 reset 丢弃和无效样本均为 0；这些分支通过可控序列单测覆盖。无配对统计包含初始接触，不能当作丢失了有效训练样本。完整统计保留在各 `result.json`。

## 真实训练和恢复

| 运行 | iteration | 真实样本累计 | 监督梯度步累计 | 最终 BCE | 最终 α |
|---|---|---:|---:|---:|---:|
| 默认配置首次训练 | 0 → 5 | 188 | 24 | 0.598864 | 0 |
| 新进程恢复 | 5 → 7 | 263 | 40 | 0.624704 | 0 |
| 独立缩短 warm-up 验收 | 0 → 10 | 413 | 64 | 0.610108 | 1 |

三次训练均检测到 policy CNN、位置投影、Query、attention、Actor、Critic 和 U-Net 参数更新；两类优化器实际执行更新。监督使用仿真真实接触样本，没有以合成样本代替训练验收。保存 `model_5.pt`、`model_7.pt`、`model_10.pt`。

默认训练及恢复的 gate 保持 0，监督前后动作 KL 为 0。缩短 warm-up 的验收最后 α=1，监督引起的质量图平均变化为 `0.0313947`，动作 KL 为 `7.14278e-6`，PPO/监督损失均有限。该验收保留 256 条 gate 最低样本数要求；由于前期尚未达到样本门槛，启用发生在名义 ramp 时段之后。默认 500/1000 轮的 0、0.5、1 插值逻辑由单测检查，没有运行 1500 轮长程训练。

日志位于 `outputs/affordance_unit_tests.log`、`outputs/aff_*_acceptance.log`；结构化结果、完整配置、关节映射、checkpoint 与 TensorBoard 位于 `outputs/rpo_affordance/<run_name>/`。checkpoint 包含双路模型、两个优化器、归一化、成熟 replay、训练轮数与 gate 样本计数。新进程恢复不延续旧 episode 或 pending。

## 独立性与限制

任务列表在 `python -I` 下显示 Base、AME、Affordance；所有仿真使用隔离 Python 启动，并检查未加载 `isal` 或 `robolab`。CPU 回归扫描源码导入与本地机器人资源。标准 MLP/PPO 基础组件仍来自外部 `rsl_rl`，本次没有修改旧工程或外部库。

在 `isal2` 下执行 `python setup.py bdist_wheel --dist-dir outputs/wheels` 成功。wheel 包含采集器、环境、配置、U-Net、定制算法/runner 及验证文档，31 个资源哈希一致，元数据保留外部 RSL 依赖。解包后在隔离 Python 进程中验证 Affordance 注册、α=1 的 Actor 前向和 U-Net 图输出；54 个项目 Python 文件语法检查及 `git diff --check` 通过。结果见 `outputs/aff_package_check.json`。

flat smoke 正常退出（exit code 0）。其余仿真完成结果保存和环境关闭后，在 Windows Isaac Sim 的 simulator 关闭阶段停留；核对命令行后结束了本次测试进程，未将自动退出报告为通过。

本阶段验证工程、接触监督和训练链路。滑动和支撑指标是代理量，尚未验证质量图校准、长程收敛、最终通过率或部署效果。当前 Affordance runner 仅支持单进程训练。
