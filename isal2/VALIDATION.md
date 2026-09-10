# 第一、二阶段验收记录

> 以下为初版历史记录；其中“不加载外部 rsl_rl”的限制来自误解，已按用户澄清取消。当前代码使用外部 RSL-RL，最新重构验证记录位于 `VALIDATION_EXTERNAL_RSL.md`。初版输出可能已清理。

验证日期：2026-09-10。范围为独立工程和完整 base 任务，不包含 AME、affordance、ONNX 导出或 MuJoCo 控制。

## 环境

- Windows 11，NVIDIA RTX 3060 12 GB。
- Python：`C:/Users/Admin/miniconda3/envs/env_isaaclab/python.exe`，3.11.14。
- Isaac Sim 5.1.0.0，PyTorch 2.7.0+cu128。
- Isaac Lab 使用本机源码，核心扩展声明版本 0.54.0；定制 RSL 核心基于工作区 3.3.0 源码完整迁入本包。
- 所有项目修改均在 `isal2`，原有已跟踪文件无 diff。未提交 Git。

## 已通过的检查

| 检查 | 结果 |
|---|---|
| 核心 CPU 回归 | 7 项通过 |
| flat 仿真 | 4 环境 × 200 步，12 次 reset，观测/奖励/动作有限 |
| rough 仿真 | 4 环境 × 200 步，12 次 reset，观测/奖励/动作有限 |
| rough_hard 仿真 | 4 环境 × 200 步，13 次 reset，观测/奖励/动作有限 |
| 初次和显式 reset | 不改变 terrain level |
| Curriculum | 升/降/保持、站立、未开始 episode；最低等级及最高等级回流；环境原点同步 |
| 局部 reset | 重置环境的历史重新填充、动作历史归零，其他环境历史不变 |
| PPO 实际训练 | 32 环境，5 次更新，参数发生变化，保存 `model_5.pt` |
| 独立进程恢复 | 从 iteration 5 恢复，再训练 2 次，保存 `model_7.pt` |
| 资源一致性 | 31 个资源文件 SHA-256 与源文件一致，URDF/MJCF 引用完整 |
| 机器人配置一致性 | 仅替换内部资源路径后的 AST 与源配置相同 |
| 独立启动 | Python `-I` 下 flat 4 环境 × 200 步；未加载 `isal`、`robolab`、外部 `rsl_rl` |
| 退出行为 | 独立启动测试写入结果后正常退出，exit code 0，无残留测试进程 |

Actor 输入维度为 390；Critic 为 flat 695、rough/rough_hard 1630。

PPO 第 5 次更新的最终 loss：value `1.376538`、surrogate `-0.045677`、entropy `32.621198`。
恢复至第 7 次更新的最终 loss：value `1.339249`、surrogate `-0.036416`、entropy `32.602311`。均为有限值。

7 项 CPU 回归覆盖：独立环境历史、curriculum 决策、使用终止帧值的 timeout bootstrap、rollout 内归一化冻结、旧包导入/URDF 检查、资源哈希/MJCF 引用、checkpoint 的 iteration/优化器/学习率/归一化恢复。

## 实际命令与结果位置

以下命令以工作区根目录为当前目录，并使用上述 `env_isaaclab` Python；为简洁使用 `python` 表示该解释器。

```powershell
python -m unittest discover -s isal2/tests -p test_core.py
python isal2/scripts/list_envs.py

python isal2/scripts/train.py --headless --num_envs 4 --terrain flat --smoke_steps 200 --check_reset --run_name acceptance_flat
python isal2/scripts/train.py --headless --num_envs 4 --terrain rough --smoke_steps 200 --check_reset --run_name acceptance_rough --terrain_rows 3 --terrain_cols 20
python isal2/scripts/train.py --headless --num_envs 4 --terrain rough_hard --smoke_steps 200 --check_reset --run_name acceptance_rough_hard --terrain_rows 3 --terrain_cols 20

python isal2/scripts/train.py --headless --num_envs 32 --terrain rough --max_iterations 5 --run_name acceptance_train --terrain_rows 3 --terrain_cols 20
python isal2/scripts/train.py --headless --num_envs 32 --terrain rough --max_iterations 2 --resume isal2/outputs/rpo_base/acceptance_train/model_5.pt --run_name acceptance_resume --terrain_rows 3 --terrain_cols 20

python -I -u isal2/scripts/train.py --headless --num_envs 4 --terrain flat --smoke_steps 200 --check_reset --run_name acceptance_isolated
```

每次运行的配置和 `result.json` 位于 `outputs/rpo_base/<run_name>/`；控制台日志保存在 `outputs/acceptance_*.log`，CPU 测试日志为 `outputs/core_tests.log`。训练恢复未重复已经完成的 iteration。

打包在 `isal2` 目录执行 `python setup.py bdist_wheel --dist-dir outputs/wheels`；此方式在本次受限环境中避开 pip 临时目录权限问题。源码支持常规 editable 安装，但本次未修改现有 conda 环境的安装状态。

## 实现中的必要修正与验证边界

- 迁移时修复初次 reset 升降级、局部历史更新和 timeout 使用上一帧值的问题；终止帧观测在 reset 前保存。
- Policy 归一化统计在 rollout/PPO 中保持固定，PPO 完成后更新，checkpoint 包含这些统计。
- 平地和光照使用本地生成几何、材质，避免标准地面 USD 及纹理的远程下载依赖。
- 官方 `isaaclab_rl` 顶层扩展由 Isaac Sim 自动加载，属于允许的第三方依赖；本项目不使用它的外部 RSL wrapper。
- 本机第三方 Kit 日志曾报告全局缓存写入警告、SharedMutex 诊断；训练和结果检查通过。这些不作为无效 loss 或成功率证据。项目日志、URDF 转换和 Warp 缓存均指向本目录；部分 Kit 底层缓存仍沿用其全局默认路径。
- 默认地形地图是 10 行 × 20 列；本次 rough 系列运行验证使用 3 行 × 20 列以降低资源占用，地形类型保留完整。未执行默认 4096 环境的长程训练。
- 短程 PPO 验证工程闭环，不代表已训练出可靠 locomotion 策略。恢复会重新创建物理 episode，不恢复完整仿真状态。
