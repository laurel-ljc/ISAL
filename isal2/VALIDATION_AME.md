# 第三阶段 AME 验证（2026-09-11）

本阶段实现 `ISAL2-RPO-AME-v0`：图一橙色 Actor 控制链路、独立 MLP Critic、AME 观测和任务配置加载。标准组件仍来自外部 `rsl_rl`；未实现 affordance、速度估计辅助任务、ONNX 或 MuJoCo。

## 环境与命令

本机 `env_isaaclab`：Python 3.11、PyTorch 2.7、Isaac Sim 5.1、RSL-RL 3.3.0，RTX 3060，`cuda:0`。下列命令从工作区 `C:/Users/Admin/Documents/ISAL` 执行，其中 `python` 为 `C:/Users/Admin/miniconda3/envs/env_isaaclab/python.exe`。

```powershell
python -m unittest discover -s isal2/tests -p "test_*.py"
python -I isal2/scripts/list_envs.py
python -I -u isal2/scripts/train.py --task ISAL2-RPO-AME-v0 --terrain flat --headless --num_envs 4 --smoke_steps 200 --check_reset --run_name ame_flat_acceptance
python -I -u isal2/scripts/train.py --task ISAL2-RPO-AME-v0 --terrain rough --headless --num_envs 4 --smoke_steps 200 --check_reset --run_name ame_rough_acceptance --terrain_rows 3 --terrain_cols 20
python -I -u isal2/scripts/train.py --task ISAL2-RPO-AME-v0 --terrain rough_hard --headless --num_envs 4 --smoke_steps 200 --check_reset --run_name ame_rough_hard_acceptance --terrain_rows 3 --terrain_cols 20
python -I -u isal2/scripts/train.py --task ISAL2-RPO-AME-v0 --headless --num_envs 32 --max_iterations 5 --run_name ame_train_acceptance --terrain_rows 3 --terrain_cols 20
python -I -u isal2/scripts/train.py --task ISAL2-RPO-AME-v0 --headless --num_envs 32 --max_iterations 2 --resume isal2/outputs/rpo_ame/ame_train_acceptance/model_5.pt --run_name ame_resume_acceptance --terrain_rows 3 --terrain_cols 20
```

## 通过结果

- 14 项 CPU 测试通过，包含原有 8 项 base 回归测试。AME 测试覆盖非对称地图 XY 排序、形状校验、动作对高程的依赖、Actor/Critic 信息与梯度隔离、rollout 保存高程、归一化冻结及动作概率一致性、各模块 PPO 梯度与参数更新、完整 checkpoint 恢复、拒绝 Base/错误地图几何 checkpoint。
- `list_envs.py` 在 `-I` 模式下列出 Base 与 AME，配置入口均正确；训练通过注册入口实际加载 AME。未指定地形的两次训练使用默认 `rough`。
- 三类地形均为 4 环境、200 控制步；policy/height_scan/critic 维度始终为 390/187/1630。flat、rough、rough_hard 分别触发 11、12、11 次 reset，所有观测、奖励和动作有限。
- 仿真检查实际 Isaac Lab 射线顺序与神经网络米制 XY 坐标一致；Actor 高程噪声在 ±0.025 内，Critic 高程保持干净。重复读取观测稳定、局部 reset 不改变其他环境历史或高程，终止高程快照独立保存。初次 reset 与 curriculum 等级边界检查通过。
- 全部仿真均在隔离 Python 模式下运行，检查未导入 `isal` 或 `robolab`；CPU 测试额外扫描源码旧工程导入和资源引用。
- 在 `isal2` 下执行 `python setup.py bdist_wheel --dist-dir outputs/wheels` 成功。wheel 包含 AME 配置、网络和文档，31 个资源哈希一致，元数据保留外部 RSL 依赖；解包后在 `python -I` 中完成任务注册、AME 实例化和前向检查。记录见 `outputs/ame_package_check.json`。

| 训练 | 开始 iteration | 完成 iteration | Value loss | Surrogate loss | Entropy |
|---|---:|---:|---:|---:|---:|
| 32 环境首次训练 | 0 | 5 | 1.206739 | -0.048062 | 32.642563 |
| 新进程 checkpoint 恢复 | 5 | 7 | 1.509372 | -0.031417 | 32.637530 |

两次训练中 CNN、位置投影、Query、注意力、Actor MLP 和 Critic MLP 均检测到参数更新，loss 均有限。保存 `model_5.pt` 和 `model_7.pt`。CPU 恢复测试逐项比较模型及归一化 state dict，并检查优化器状态与 iteration 恢复。

日志位于 `outputs/ame_unit_tests.log` 和 `outputs/ame_*_acceptance.log`。仿真结构化结果位于 `outputs/rpo_ame/<run_name>/result.json`；同目录包含环境、agent、CLI、关节配置，以及训练的 checkpoint 和 TensorBoard 日志。

## 限制

flat 仿真正常退出，exit code 为 0。其他运行在保存结果、关闭环境后停留于 Windows Isaac Sim 的 simulator 关闭阶段，复现前阶段问题；已核对各测试进程命令行后清理，未修改 simulator 生命周期来绕过退出。训练和环境断言通过，进程自动退出不计为通过。

本阶段没有长程收敛、最终地形通过率或部署验证。短程训练仅验证学习链路，不表示模型已经学会可靠行走。
