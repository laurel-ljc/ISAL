# RSL-RL 扩展

此目录仅存放 ISAL2 的定制实现。标准 `ActorCritic`、`RolloutStorage`、网络基础组件和日志工具均从安装的 `rsl_rl` 导入。

- `modules/`：AME Actor、地形注意力、Affordance U-Net 和融合模型；复用外部 MLP、归一化和动作分布接口。
- `runners/on_policy_runner.py`：继承外部 `OnPolicyRunner`，定制更新次数、有限值检查、checkpoint 语义及后续监督更新位置。
- `algorithms/ppo.py`：继承外部 `PPO`，仅定制终止帧 bootstrap 与归一化更新时机，PPO 优化主体调用 `super().update()`。
- `algorithms/affordance_ppo.py`：排除蓝色参数的 PPO 优化器；`affordance_replay.py` 保存近期成熟接触样本。
- `runners/affordance_runner.py`：每轮 PPO 之后执行监督更新，管理 gate、漂移诊断和双优化器/replay checkpoint。

不要将整套 RSL-RL 源码复制到这里。任务环境的 TensorDict 适配器位于 `isal2/utils/rsl_env.py`。
