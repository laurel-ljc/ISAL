# 外部 RSL-RL 复用验证（2026-09-10）

按用户纠正，独立性仅要求运行时不依赖 `isal`、`robolab`；允许使用外部 `rsl_rl`。本次删除 `modified_rsl` 内复制的标准网络、storage、工具和环境接口，base MLP 直接使用外部 `ActorCritic`。定制 PPO、runner 继承外部实现，后续定制网络仍放在 `modified_rsl/modules`。环境 adapter 移至 `utils/rsl_env.py`。

验证使用 `C:/Users/Admin/miniconda3/envs/env_isaaclab/python.exe`，Python 3.11、PyTorch 2.7、Isaac Sim 5.1、外部 `rsl-rl-lib==3.3.0`。本机外部库已通过 editable 安装；本次没有修改外部源码或安装状态。

以下命令从 `C:/Users/Admin/Documents/ISAL` 执行，`python` 指上述解释器：

```powershell
python -m unittest discover -s isal2/tests -p test_core.py
python -I -u isal2/scripts/train.py --headless --num_envs 32 --terrain rough --max_iterations 5 --run_name external_rsl_train --terrain_rows 3 --terrain_cols 20
python -I -u isal2/scripts/train.py --headless --num_envs 32 --terrain rough --max_iterations 2 --resume isal2/outputs/rpo_base/external_rsl_train/model_5.pt --run_name external_rsl_resume --terrain_rows 3 --terrain_cols 20
```

- 8 项回归测试通过：资源与引用、旧工程导入隔离、外部 RSL 组件来源、历史缓存、curriculum、timeout bootstrap、归一化和 runner/checkpoint。
- 32 环境实际训练完成 5 次 PPO 更新，参数发生更新，保存 `model_5.pt`。最终 value/surrogate/entropy 分别为 `1.3765381 / -0.0456773 / 32.6211983`，均有限。
- 新进程恢复从 iteration 5 继续到 7，参数继续更新，保存 `model_7.pt`。最终 loss 为 `1.3392492 / -0.0364156 / 32.6023106`，均有限。
- 日志位于 `outputs/external_rsl_tests.log`、`outputs/external_rsl_train.log`、`outputs/external_rsl_resume.log`；结构化结果和 checkpoint 位于 `outputs/rpo_base/external_rsl_{train,resume}/`。
- 在 `isal2` 下执行 `python setup.py bdist_wheel --dist-dir outputs/wheels` 成功。检查 wheel 元数据包含外部 RSL 版本依赖，不含已删除的通用 RSL 目录；31 个打包资源的 SHA256 均与 manifest 一致。解包后用 `python -I` 加载包及定制 PPO/runner 成功，标准网络和 storage 来源为外部 RSL，未加载旧工程包。

两次仿真均已保存结果并完成环境关闭，随后停留在 Windows Isaac Sim 的原生 simulator 关闭阶段，未正常退出。已核对命令行并结束这两个测试进程。因此训练与恢复断言通过，但不将进程自动退出报告为通过。

本次未改地形、机器人、奖励或环境生命周期，也未重复三类地形的 200 步验收。其前期记录保留在 `VALIDATION.md`；这些属于历史验证，不算作本次重跑。AME、affordance、ONNX 和 MuJoCo 控制仍在后续阶段。
