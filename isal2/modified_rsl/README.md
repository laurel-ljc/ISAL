# RSL-RL 扩展

此目录仅存放 ISAL2 的定制实现。标准 `ActorCritic`、`RolloutStorage`、网络基础组件和日志工具均从安装的 `rsl_rl` 导入。

- `modules/`：留给后续 AME、affordance 定制网络；base 直接使用外部 MLP。
- `runners/on_policy_runner.py`：继承外部 `OnPolicyRunner`，定制更新次数、有限值检查、checkpoint 语义及后续监督更新位置。
- `algorithms/ppo.py`：继承外部 `PPO`，仅定制终止帧 bootstrap 与归一化更新时机，PPO 优化主体调用 `super().update()`。

不要将整套 RSL-RL 源码复制到这里。任务环境的 TensorDict 适配器位于 `isal2/utils/rsl_env.py`。
